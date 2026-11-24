"""Inkling2 with EXL3 weights (see __init__.py). Single GPU: no tensor or expert parallelism.

What changes when the pack is EXL3 (vllm_config.quant_config is an Exl3Config); nothing changes otherwise:
  * the model is built with quant_config hidden (vLLM asserts Inkling2QuantConfig and would derive an NVFP4 expert
    format from it) while the EXL3 config is supplied to the layers that need it: the attention qkvr / wo_ud (built
    without a quant config upstream), lm_head and embed_tokens (layers.py);
  * attention `_project` / `forward` call the EXL3 layers instead of reading `.weight` (qkvr mixes five EXL3 shards,
    the BF16 attn_sink_proj and an alignment pad: linear.py "mixed" layers);
  * the routed experts are `Exl3Inkling2Experts`: vLLM's own router (select_experts: gate bias, top-k, route scale,
    padding tokens), then per-expert EXL3 GEMMs with Inkling's softcapped SwiGLU. exllamav3's fused exl3_moe kernel has
    no softcap activation, so this path syncs with the CPU once per layer: run with --enforce-eager for now;
  * load_weights routes the EXL3 tensors itself (vLLM's loader rejects their dtypes and counts shards per parameter)
    and gives the rest to vLLM's loader.
"""
import contextlib
import contextvars
import re

import torch
from torch import nn

from vllm.model_executor.layers import linear as _lin
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.models.inkling2.nvidia import attention as _att
from vllm.models.inkling2.nvidia import model as _model
from vllm.models.inkling2.nvidia import moe as _moe
from vllm.models.inkling2.nvidia.loop import Inkling2LoopForConditionalGeneration as _Base

from ...exl3.layers import exl3_config_override
from ...exl3.linear import Exl3Config, Exl3LinearMethod, Exl3ShardParam
from ...exl3.moe import _Proj

_EXL3 = contextvars.ContextVar("quantpack_inkling2_exl3", default=None)
_EXPERT = re.compile(r"^(?P<mod>.*\.mlp\.experts)\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
                     r"(?P<suf>trellis|suh|svh|mcg|mul1)$")
_LAYERS: list = []            # index -> Exl3Inkling2Experts (the custom op only receives the index)


def _is_exl3(layer) -> bool:
    return isinstance(getattr(layer, "quant_method", None), Exl3LinearMethod)


# --- attention linears: built without a quant config upstream ---------------------------------------------------------

class _Merged(_lin.MergedColumnParallelLinear):
    def __init__(self, *args, **kwargs):
        if _EXL3.get() is not None and kwargs.get("quant_config") is None:
            kwargs["quant_config"] = _EXL3.get()
        super().__init__(*args, **kwargs)


class _Row(_lin.RowParallelLinear):
    def __init__(self, *args, **kwargs):
        if _EXL3.get() is not None and kwargs.get("quant_config") is None:
            kwargs["quant_config"] = _EXL3.get()
        super().__init__(*args, **kwargs)


_att.MergedColumnParallelLinear = _Merged
_att.RowParallelLinear = _Row

_orig_project = _att.Inkling2Attention._project
_orig_forward = _att.Inkling2Attention.forward


def _project(self, x_q, x_k, x_v):
    if not _is_exl3(self.qkvr):
        return _orig_project(self, x_q, x_k, x_v)
    if x_k is x_q and x_v is x_q:
        return Exl3LinearMethod.partitions(self.qkvr, x_q, range(6))
    q_side = Exl3LinearMethod.partitions(self.qkvr, x_q, range(4))
    (k,) = Exl3LinearMethod.partitions(self.qkvr, x_k, (4,))
    (v,) = Exl3LinearMethod.partitions(self.qkvr, x_v, (5,))
    return [*q_side, k, v]


def _forward(self, positions, x_q, x_k=None, x_v=None, tau=None, out=None, pass_idx=0):
    if not _is_exl3(self.wo_ud):
        return _orig_forward(self, positions, x_q, x_k, x_v, tau, out, pass_idx)
    # Upstream forward with the output projection through the EXL3 layer instead of linear(output, wo_ud.weight)
    if self.is_local and x_k is None:
        x_k = x_v = x_q
    assert x_k is not None and x_v is not None
    q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
    output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
    self._attention(positions, q, r, gate, sinks, k, v, self.k_sconv.dynamic_taps(x_k),
                    self.v_sconv.dynamic_taps(x_v), tau, output, pass_idx)
    y = self.wo_ud.quant_method.apply(self.wo_ud, output)
    if out is None:
        return y
    out.copy_(y)
    return out


_att.Inkling2Attention._project = _project
_att.Inkling2Attention.forward = _forward


# --- lm_head ---------------------------------------------------------------------------------------------------------
# Inkling2 computes logits in fp32 (head_dtype) and vLLM's LogitsProcessor only allows that for an unquantized head.
# The EXL3 GEMM accumulates in fp32 and returns the model dtype: run it and widen the result.
_orig_apply_head = _model.Inkling2LogitsProcessor._apply_head


def _apply_head(self, lm_head, hidden_states, embedding_bias):
    qm = lm_head.quant_method
    if (self.head_dtype is None or self.head_dtype == hidden_states.dtype
            or not type(qm).__module__.startswith("vllm_quantpack")):
        return _orig_apply_head(self, lm_head, hidden_states, embedding_bias)
    return qm.apply(lm_head, hidden_states, bias=embedding_bias).to(self.head_dtype)


_model.Inkling2LogitsProcessor._apply_head = _apply_head


# --- routed experts ------------------------------------------------------------------------------------------------------

def softcapped_swiglu(g, u, gate_cap, up_cap):
    """Inkling expert activation: softcap(g) * sigmoid(g) * softcap(u), in fp32 (as the exllamav3 port)."""
    g, u = g.float(), u.float()
    return torch.tanh(g / gate_cap) * gate_cap * torch.sigmoid(g) * torch.tanh(u / up_cap) * up_cap


class _ExpertsPostLoad(QuantizeMethodBase):
    """vLLM runs process_weights_after_loading only for modules with a quant_method: stack the experts there."""

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        layer.build()


class Exl3Inkling2Experts(nn.Module):
    """Drop-in for Inkling2Experts (same forward contract, single GPU)."""

    sm100 = False      # Inkling2MoE.forward then routes with select_experts (ids, gammas, src2dst)
    sm120 = False

    def __init__(self, config):
        super().__init__()
        self.ep_rank, self.ep_size = 0, 1
        self.num_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.gate_cap, self.up_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        self.hidden_size = config.hidden_size
        self.register_buffer("expert_bounds", torch.tensor([0, self.num_experts], dtype=torch.int32),
                             persistent=False)
        self.exl3_store = {}           # (expert, projection, suffix) -> tensor, filled by load_weights
        self.quant_method = _ExpertsPostLoad()
        self.exl3_idx = None
        self.prefix = ""

    def build(self):
        device = torch.device("cuda", torch.cuda.current_device())
        projs = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            per = []
            for e in range(self.num_experts):
                d = {s: self.exl3_store[(e, proj, s)] for s in ("trellis", "suh", "svh", "mcg", "mul1")
                     if (e, proj, s) in self.exl3_store}
                if "trellis" not in d:
                    raise RuntimeError(f"vllm-quantpack: {self.prefix}: expert {e} {proj} missing from the EXL3 pack")
                per.append(d)
            projs[proj] = _Proj(f"{self.prefix}.{proj}", per, device)
        self.gate, self.up, self.down = projs["gate_proj"], projs["up_proj"], projs["down_proj"]
        self.exl3_store.clear()
        self.exl3_idx = len(_LAYERS)
        _LAYERS.append(self)

    def compute(self, x, ids, gammas):
        T, k = ids.shape
        y = x.half().contiguous()
        flat_e = ids.reshape(-1).long()
        w = gammas.reshape(-1).float()
        tok = torch.arange(T, device=x.device).repeat_interleave(k)
        keep = flat_e >= 0                     # -1: padding token or not a local expert
        flat_e, w, tok = flat_e[keep], w[keep], tok[keep]
        order = flat_e.argsort()
        flat_e, w, tok = flat_e[order], w[order], tok[order]
        counts = torch.bincount(flat_e, minlength=self.num_experts).tolist()
        out = torch.zeros((T, self.hidden_size), dtype=torch.float, device=x.device)
        start = 0
        for e, c in enumerate(counts):
            if not c:
                continue
            rows = tok[start:start + c]
            xs = y.index_select(0, rows)
            a = softcapped_swiglu(self.gate.gemm(xs, e), self.up.gemm(xs, e), self.gate_cap, self.up_cap)
            # The routing weight goes in before the down projection, as in vLLM's experts
            o = self.down.gemm((a * w[start:start + c, None]).half(), e)
            out.index_add_(0, rows, o.float())
            start += c
        return out.to(x.dtype)

    def forward(self, x, routing, quantized, plan, reduce: bool = True, push=None):
        if push is not None or not reduce:
            raise NotImplementedError("vllm-quantpack: EXL3 Inkling2 experts run on one GPU (no TP / EP / SP)")
        ids, gammas, _ = routing
        return torch.ops.quantpack.exl3_inkling2_experts(x, ids, gammas, self.exl3_idx)


@torch.library.custom_op("quantpack::exl3_inkling2_experts", mutates_args=())
def _experts_op(x: torch.Tensor, ids: torch.Tensor, gammas: torch.Tensor, idx: int) -> torch.Tensor:
    return _LAYERS[idx].compute(x, ids, gammas)


@_experts_op.register_fake
def _experts_fake(x, ids, gammas, idx):
    return torch.empty_like(x)


_OrigMoE = _model.Inkling2MoE


class _MoE(_OrigMoE):
    def __init__(self, config, quant_format, visit_ids=(0,)):
        if _EXL3.get() is None:
            return super().__init__(config, quant_format, visit_ids)
        # Inkling2MoE.__init__ with the EXL3 experts (upstream would allocate NVFP4 experts here)
        nn.Module.__init__(self)
        assert config.n_shared_experts == 0
        self.gate = _moe.Inkling2Gate(config)
        self.experts = Exl3Inkling2Experts(config)
        self.routing_captures = nn.ModuleList(_moe.Inkling2RoutingCapture(v) for v in visit_ids)
        self.register_buffer("all_experts", torch.tensor([0, config.n_routed_experts], dtype=torch.int32),
                             persistent=False)


_model.Inkling2MoE = _MoE


# --- the model ------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _hidden_params(module, names):
    """vLLM's load_weights builds dict(self.named_parameters()) and requires every one of them: hide ours."""
    orig = module.named_parameters
    module.named_parameters = lambda *a, **k: ((n, p) for n, p in orig(*a, **k) if n not in names)
    try:
        yield
    finally:
        del module.named_parameters


class Inkling2LoopForConditionalGeneration(_Base):
    def __init__(self, *, vllm_config, prefix: str = ""):
        qc = vllm_config.quant_config
        self_exl3 = qc if isinstance(qc, Exl3Config) else None
        if self_exl3 is None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._exl3 = None
            return
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError("vllm-quantpack: EXL3 Inkling2 runs on one GPU")
        token = _EXL3.set(qc)
        vllm_config.quant_config = None
        try:
            with exl3_config_override(qc):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_config.quant_config = qc
            _EXL3.reset(token)
        self._exl3 = qc
        for name, m in self.named_modules():
            if isinstance(m, Exl3Inkling2Experts):
                m.prefix = name

    def _rename(self, name):
        for old, new in self._renames:
            name = name.replace(old, new)
        for old, (new, shard) in self._shards.items():
            if old in name:
                return name.replace(old, new), shard
        return name, None

    def load_weights(self, weights):
        if self._exl3 is None:
            return super().load_weights(weights)
        params = dict(self.named_parameters())
        experts = {n: m for n, m in self.named_modules() if isinstance(m, Exl3Inkling2Experts)}
        ours = {n for n, p in params.items() if isinstance(p, Exl3ShardParam)}

        def rest():
            for name, w in weights:
                n, shard = self._rename(name)
                m = _EXPERT.match(n)
                if m is not None and m.group("mod") in experts:
                    experts[m.group("mod")].exl3_store[(int(m.group("e")), m.group("proj"), m.group("suf"))] = w
                    continue
                if n in ours:
                    p = params[n]
                    p.weight_loader(p, w, shard)
                    continue
                # exllamav3 stores unquantized side tensors (e.g. the router's gate.bias) in fp16; Inkling2's
                # load_weights rejects any dtype mismatch, so cast floating tensors to the parameter's dtype
                p = params.get(n)
                if p is not None and w.dtype != p.dtype and w.is_floating_point() and p.is_floating_point():
                    w = w.to(p.dtype)
                yield name, w

        with _hidden_params(self, ours):
            loaded = super().load_weights(rest())
        for n, m in experts.items():
            if not m.exl3_store:
                raise ValueError(f"vllm-quantpack: no EXL3 expert tensors for {n}")
        return loaded | ours
