"""EXL3 MoE experts (routed experts of Qwen3.5/3.6/3.8-MoE, Flint and other FusedMoE models).

The routed experts of a MoE go through vLLM's RoutedExperts/FusedMoE and need their own quant method. exllamav3 stores
every expert separately (`mlp.experts.E.{gate,up,down}_proj.{trellis,suh,svh,mcg|mul1}`) and gives the same bitrate to
all experts of a layer, so they are stacked into one tensor per projection and passed to the fused exllamav3 kernel
through the pointer tables it expects:

  * few assignments (decode, MTP verification; T <= kernel rows): everything runs in `ext.exl3_moe`, with no CPU
    synchronisation -> capturable in CUDA graphs;
  * prefill (default, VLLM_QUANTPACK_MOE_VIRTUAL=1): every expert is cut on the GPU into "virtual experts" of at most
    `fused_rows` rows that point to the same weights, and the fused kernel runs once per row tier (16-row tiles below 64
    rows, 64-row tiles above with the mul1 codebook). The kernel finds each expert's rows by summing the counts and
    indexes the pointer tables by expert, so repeated pointers are fine, and it adds its outputs atomically. Nothing is
    read on the CPU: before, `count.tolist()` stopped the engine loop at every MoE layer of every prefill chunk and a
    Python loop launched ~6 kernels per heavily loaded expert;
  * prefill with VLLM_QUANTPACK_MOE_VIRTUAL=0: experts with few rows use the fused kernel; heavily loaded experts are
    computed by reconstructing their weights (`quantpack::exl3_gemm`, which above 144 rows reconstructs and uses a
    dense GEMM).

Compute is fp16 (as in exllamav3) and the expert sum is fp32. Tensor parallelism needs expert parallelism
(`-tp N --enable-expert-parallel`): each GPU keeps whole experts and the kernel is unchanged.

NOTE: exl3_moe synchronises its blocks by spinning on global counters and needs the whole GPU. vLLM runs the shared
expert on an auxiliary stream IN PARALLEL (batches <= 256 tokens), and then the kernel waits forever. register() sets
VLLM_DISABLE_SHARED_EXPERTS_STREAM=1; also export it in the shell that starts vLLM.

Origin: qwen38_exl3_vision 0.4.1 moe.py (ours).
"""

import os
import re

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

from . import hooks
from .names import EXPERT_SUFFIXES, expert_store_key, gateless_up_key

logger = init_logger("vllm.quantpack.exl3.moe")

_SUFFIXES = EXPERT_SUFFIXES
_NAME = re.compile(r"(?:^|\.)(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg|mul1)$")
_LAYERS: list = []          # index -> _Experts (the custom op only receives the index)
_BUFFERS: dict = {}         # (device, name) -> kernel scratch tensor, shared by all layers


def _buffer(device, name, shape, dtype):
    key = (str(device), name)
    t = _BUFFERS.get(key)
    if t is None or t.shape != torch.Size(shape):
        t = torch.empty(shape, dtype=dtype, device=device)
        _BUFFERS[key] = t
    return t


class _Proj:
    """One projection (gate, up or down) of all experts of a layer, stacked."""

    def __init__(self, name, per_expert: list[dict], device):
        def stack(key, dtype=None):
            # 0.3.0: stack on the CPU and copy once (512 experts x 48 layers = ~100k tiny H2D copies took ~15 min with Flash-Next)
            ts = [d[key] for d in per_expert]
            if all(t.device.type == "cpu" for t in ts):
                t = torch.stack(ts)
                if dtype is not None:
                    t = t.to(dtype)
                return t.to(device).contiguous()
            ts = [t.to(device) for t in ts]
            if dtype is not None:
                ts = [t.to(dtype) for t in ts]
            return torch.stack(ts).contiguous()
        self.trellis = stack("trellis")                       # (E, in/16, out/16, 16*K) int16
        self.suh = stack("suh", torch.half)                   # (E, in)
        self.svh = stack("svh", torch.half)                   # (E, out)
        self.mcg = "mcg" in per_expert[0]
        self.mul1 = "mul1" in per_expert[0]
        if any(("mcg" in d) != self.mcg or ("mul1" in d) != self.mul1 for d in per_expert):
            raise RuntimeError(f"{name}: experts do not share the same codebook")
        k = self.trellis.shape[-1] / 16
        self.K = int(k) if float(k).is_integer() else k
        self.ic, self.oc = self.suh.shape[1], self.svh.shape[1]
        E = self.trellis.shape[0]
        def ptrs(t):
            return torch.tensor([t.data_ptr() + e * t.stride(0) * t.element_size() for e in range(E)],
                                dtype=torch.long, device=device)
        self.p_trellis, self.p_suh, self.p_svh = ptrs(self.trellis), ptrs(self.suh), ptrs(self.svh)

    def gemm(self, x, e):
        return torch.ops.quantpack.exl3_gemm(x, self.trellis[e], self.suh[e], self.svh[e], self.oc,
                                             float(self.K), self.mcg, self.mul1)


class _Experts:
    def __init__(self, prefix, gate, up, down, device, gateless=False):
        from exllamav3.ext import exllamav3_ext as ext
        self.prefix, self.gate, self.up, self.down = prefix, gate, up, down
        self.gateless = gateless       # ReLU² without gate (NemotronH): gate is up, and the kernel ignores it
        self.out_scale = 1.0           # set by a family hook (see hooks.py)
        self.E = gate.trellis.shape[0]
        self.H, self.I = gate.ic, gate.oc
        cbs = [(p.mcg, p.mul1) for p in (gate, up, down)]
        if not (cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True))):
            raise RuntimeError(f"{prefix}: codebook {cbs} not supported by the fused kernel")
        # mul1 uses 32/64-row tiles: the kernel accepts up to 256 rows per expert (128 with mcg)
        self.fused_rows = 256 if gate.mul1 else 128
        C = ext.exl3_moe_max_concurrency(device.index)
        R = self.fused_rows
        self.tsg = _buffer(device, "tsg", (C, R, self.H), torch.half)
        self.tsu = _buffer(device, "tsu", (C, R, self.H), torch.half)
        self.tig = _buffer(device, "tig", (C, R, self.I), torch.half)
        self.tiu = _buffer(device, "tiu", (C, R, self.I), torch.half)

    def fused(self, y, out, count, tok_sorted, w_sorted, num_active, lo, hi):
        from exllamav3.ext import exllamav3_ext as ext
        g, u, d = self.gate, self.up, self.down
        ext.exl3_moe(y, out, count, tok_sorted, w_sorted, self.tsg, self.tsu, self.tig, self.tiu,
                     2 if self.gateless else 0,   # relu² gateless / silu
                     g.K, u.K, d.K,
                     g.p_trellis, g.p_suh, g.p_svh, u.p_trellis, u.p_suh, u.p_svh,
                     d.p_trellis, d.p_suh, d.p_svh,
                     g.mcg, g.mul1, u.mcg, u.mul1, d.mcg, d.mul1,
                     0.0, num_active, None, None, lo, hi, 16)

    def fused_virtual(self, y, out, count, tok_sorted, w_sorted, T, k):
        """Prefill without CPU sync: cut each expert into chunks of <= fused_rows rows (virtual experts with the
        same weight pointers) and run the fused kernel over them. Every size is fixed by T, k and E."""
        from exllamav3.ext import exllamav3_ext as ext
        R, E, dev = self.fused_rows, self.E, y.device
        V = E + (T * k + R - 1) // R                     # upper bound of sum(ceil(count_e / R)) over experts
        cnt = count[:E]
        chunks = (cnt + R - 1) // R
        ends = chunks.cumsum(0)
        v = torch.arange(V, device=dev)
        owner = torch.searchsorted(ends, v, right=True)  # E for the unused tail
        valid = owner < E
        e = owner.clamp(max=E - 1)
        j = v - (ends[e] - chunks[e])                    # chunk index inside its expert
        vcount = torch.where(valid, (cnt[e] - j * R).clamp(0, R), torch.zeros_like(v))
        vcount = torch.cat((vcount, count[E:]))          # the kernel ignores the last bin (sentinel bucket)
        g, u, d = self.gate, self.up, self.down
        tabs = [t.index_select(0, e) for t in (g.p_trellis, g.p_suh, g.p_svh, u.p_trellis, u.p_suh, u.p_svh,
                                               d.p_trellis, d.p_suh, d.p_svh)]
        tiers = ((1, 63, 16), (64, R, 64)) if g.mul1 else ((1, R, 16),)   # wide row tiles exist for mul1 only
        for lo, hi, m_tile in tiers:
            ext.exl3_moe(y, out, vcount, tok_sorted, w_sorted, self.tsg, self.tsu, self.tig, self.tiu,
                         2 if self.gateless else 0, g.K, u.K, d.K, *tabs,
                         g.mcg, g.mul1, u.mcg, u.mul1, d.mcg, d.mul1,
                         0.0, -1, None, None, lo, hi, m_tile)

    def forward(self, x, topk_ids, topk_weights):
        out = self._sum(x, topk_ids, topk_weights)
        return out if self.out_scale == 1.0 else out * self.out_scale

    def _sum(self, x, topk_ids, topk_weights):
        T, k = topk_ids.shape
        y = x.half()
        if not y.is_contiguous():
            y = y.contiguous()
        out = torch.zeros((T, self.H), dtype=torch.float, device=x.device)
        flat_e = topk_ids.reshape(-1).long()
        flat_e = torch.where(flat_e < 0, torch.full_like(flat_e, self.E), flat_e)   # padding -> sentinel bucket E
        flat_w = topk_weights.reshape(-1).half()
        flat_tok = torch.arange(T, device=x.device).repeat_interleave(k)
        order = flat_e.argsort()
        tok_sorted, w_sorted = flat_tok[order], flat_w[order]
        # bincount synchronises with the CPU (it computes the max) and cannot be captured in a CUDA graph
        count = torch.zeros(self.E + 1, dtype=torch.long, device=x.device).scatter_add_(0, flat_e, torch.ones_like(flat_e))
        if T <= self.fused_rows:
            # 0.3.1: a token never picks the same expert twice, so no expert can get more than T rows. T <= fused_rows is enough
            # (0.3.0 checked T * k, which kept CUDA graphs at <= 32 tokens per step with mul1 and <= 16 with mcg). Everything in the kernel,
            # nothing read on the CPU, so this path can be captured in a CUDA graph.
            self.fused(y, out, count, tok_sorted, w_sorted, -1, 1, self.fused_rows)
            return out
        if os.environ.get("VLLM_QUANTPACK_MOE_VIRTUAL", "1") != "0":
            self.fused_virtual(y, out, count, tok_sorted, w_sorted, T, k)
            return out
        counts = count.tolist()
        light = sum(1 for c in counts[:self.E] if 0 < c <= self.fused_rows)
        if light:
            self.fused(y, out, count, tok_sorted, w_sorted, light, 1, self.fused_rows)
        start = 0
        for e, c in enumerate(counts[:self.E]):
            end = start + c
            if c > self.fused_rows:
                rows = tok_sorted[start:end]
                xs = y.index_select(0, rows)
                if self.gateless:
                    a = F.relu(self.up.gemm(xs, e)).square()
                else:
                    a = F.silu(self.gate.gemm(xs, e)) * self.up.gemm(xs, e)
                o = self.down.gemm(a.half(), e)
                out.index_add_(0, rows, o.float() * w_sorted[start:end].float().unsqueeze(1))
            start = end
        return out


@torch.library.custom_op("quantpack::exl3_moe", mutates_args=())
def _moe_op(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, idx: int) -> torch.Tensor:
    return _LAYERS[idx].forward(x, topk_ids, topk_weights).to(x.dtype)


@_moe_op.register_fake
def _moe_fake(x, topk_ids, topk_weights, idx):
    return torch.empty_like(x)


class _Holder(torch.nn.Parameter):
    """Empty parameter: each expert's tensors are kept aside (their shapes are unknown until they are read)."""

    exl3_store = None

    def __new__(cls):
        return super().__new__(cls, torch.empty(0), requires_grad=False)

    def weight_loader(self, param, loaded_weight, name_mapped=None, shard_id=None, expert_id=None, **kwargs):
        # The EAGLE/MTP drafter loads experts through param.weight_loader (fused name + expert and shard ids) instead of
        # the layer's load_weights: put them in the store too, or its MoE ends up empty ("gate_proj missing").
        if self.exl3_store is None or name_mapped is None or expert_id is None:
            return
        key = expert_store_key(name_mapped, shard_id, expert_id)
        if key is not None:
            self.exl3_store[key] = loaded_weight


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, moe, prefix: str = ""):
        super().__init__(moe)
        self.prefix = prefix
        # exl3_moe needs the whole GPU: the shared expert must not run beside it on vLLM's aux stream. vLLM reads this
        # when it builds the layer's SharedExperts, which comes after the routed experts and their quant method. Set
        # only here, for EXL3 MoE models: for other models it would only change the compiled graph.
        os.environ.setdefault("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                       params_dtype, **extra_weight_attrs):
        if self.moe.moe_parallel_config.tp_size > 1 and not self.moe.moe_parallel_config.use_ep:
            # 0.4.0: expert parallelism only. Cutting every expert across GPUs would need a TP-aware exl3_moe; with EP each GPU keeps
            # whole experts and the kernel is unchanged. With -tp N, add --enable-expert-parallel.
            raise NotImplementedError("vllm-quantpack: EXL3 MoE with tensor parallelism needs --enable-expert-parallel")
        layer.exl3_store = {}          # (expert, projection, suffix) -> tensor
        layer.exl3_num_experts = num_experts
        for w in ("w13", "w2"):
            for s in _SUFFIXES:
                h = _Holder()
                h.exl3_store = layer.exl3_store
                layer.register_parameter(f"{w}_{s}", h)
        store = layer.exl3_store
        names = [f"{w}_{s}" for w in ("w13", "w2") for s in _SUFFIXES]

        def load_weights(weights):
            # Replaces RoutedExperts.load_weights: its path treats any 3D tensor as fused experts, and the EXL3 trellis is 3D.
            for name, t in weights:
                if name == "e_score_correction_bias":
                    # Router bias of aux-loss-free balancing (NemotronH): it belongs to the layer, not to the experts.
                    layer.e_score_correction_bias.data.copy_(t)
                    continue
                m = _NAME.search(name)
                if m is None:
                    raise KeyError(f"vllm-quantpack: unexpected expert tensor {name!r}")
                store[(int(m.group(1)), m.group(2), m.group(3))] = t
            yield from names
        layer.load_weights = load_weights

    def process_weights_after_loading(self, layer):
        store = layer.exl3_store
        device = torch.device("cuda", torch.cuda.current_device())
        # 0.4.0 (EP): the checkpoint names carry the GLOBAL expert id; keep this rank's experts, in local order.
        em = getattr(layer, "expert_map", None)
        if em is None:
            experts = list(range(layer.exl3_num_experts))
            layer.exl3_emap = None
        else:
            em_cpu = em.cpu()
            local = {int(em_cpu[g]): g for g in range(em_cpu.numel()) if int(em_cpu[g]) >= 0}
            experts = [local[i] for i in range(len(local))]
            layer.exl3_emap = em.to(device=device, dtype=torch.long)
        up_key = gateless_up_key(store, experts)
        projs = {}
        for proj in ("up_proj", "down_proj") if up_key else ("gate_proj", "up_proj", "down_proj"):
            key = up_key if proj == "up_proj" and up_key else proj
            per = []
            for e in experts:
                d = {s: store[(e, key, s)] for s in _SUFFIXES if (e, key, s) in store}
                if "trellis" not in d:
                    raise RuntimeError(f"{self.prefix}: expert {e} {proj} missing (unquantized experts?)")
                per.append(d)
            projs[proj] = _Proj(f"{self.prefix}.{proj}", per, device)
        if up_key:
            projs["gate_proj"] = projs["up_proj"]   # the kernel does not read gate in relu² mode; it only needs its shapes
        layer.exl3_store = None
        store.clear()
        for w in ("w13", "w2"):
            for s in _SUFFIXES:
                delattr(layer, f"{w}_{s}")
        ex = _Experts(self.prefix, projs["gate_proj"], projs["up_proj"], projs["down_proj"], device,
                      gateless=up_key is not None)
        layer.exl3_idx = len(_LAYERS)
        _LAYERS.append(ex)
        for hook in hooks.MOE:
            hook(self.prefix, layer, ex)
        torch.cuda.empty_cache()

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        em = layer.exl3_emap
        if em is not None:
            # EP: global -> local id; experts of other ranks become -1 (the sentinel bucket). vLLM all-reduces the partial sums.
            ids = topk_ids.long()
            topk_ids = torch.where(ids < 0, ids, em[ids.clamp(min=0)])
        return torch.ops.quantpack.exl3_moe(x, topk_ids, topk_weights, layer.exl3_idx)

