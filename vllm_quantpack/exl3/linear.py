"""Quantization config and linear method for EXL3 checkpoints.

An EXL3 checkpoint stores each projection self-contained: a packed tail-biting trellis plus its own incoherence vectors,
and the bitrate is implied by the trellis width. Two consequences shape this module:

  * Shards of a fused vLLM layer (q/k/v inside qkv_proj, gate/up inside gate_up_proj) are separate EXL3 tensors with
    different shapes and, under a mixed-precision allocation, different K. Each shard is kept as its own buffer and
    `apply` runs one GEMM per shard and concatenates.
  * Nothing about the shapes is known until the weights arrive, so create_weights registers placeholders and a loader
    that captures each shard as it is loaded.

Origin: OrcaSAQ2 config.py (Apache-2.0) with the published patch (prefix normalisation, empty draft lm_head), merged
with what qwen38_exl3_vision 0.4.1 used to add on top by monkeypatching:
  * padding: EXL3 pads some unfused layers to a multiple of 128; inputs are zero-padded to the stored width and the extra
    output columns are trimmed;
  * dense layers stored in fp16 are converted once to the model dtype (OrcaSAQ2 converted them on every call: several ms
    per token with a 1 GB lm_head);
  * host memory: the captured tensors are views of the safetensors mmap; the shard dicts are emptied once each layer is
    on the GPU, otherwise the whole checkpoint stayed resident in host RAM;
  * tensor parallelism: the placeholders collect the WHOLE checkpoint tensor whatever the rank, and each rank keeps its
    slice before the kernels are built (tp_ranges.py). Slices off the 128-column block keep the whole tensor and select
    this rank's columns (or zero-pad this rank's input for row-parallel layers).
"""
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.parameter import BasevLLMParameter

from . import hooks
from . import ops  # noqa: F401  (registers quantpack::exl3_gemm)
from .names import expand, manifest_prefixes, norm_prefix, slot, trellis_k
from .tp_ranges import col_ranges, logical_parts, on_block, row_range

logger = init_logger("vllm.quantpack.exl3")

LINEAR_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1", "weight")

EXLLAMAV3_MISSING = (
    "vllm-quantpack needs the exllamav3 CUDA extension to serve EXL3 checkpoints, and importing it failed: {err}\n"
    "It is not on PyPI: the wheel is ABI-linked to both torch and the CUDA major version, so pick the release asset "
    "matching this environment:\n"
    "  python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
    "  https://github.com/turboderp-org/exllamav3/releases  (cu128.* for CUDA 12.x, cu132.* for CUDA 13.x)")


class Exl3ShardParam(BasevLLMParameter):
    """One placeholder parameter standing in for a list of per-shard EXL3 tensors.

    vLLM wants a Parameter of known shape per weight name; EXL3 shard shapes (and K) are only known once the file is
    read, and differ between shards of a fused layer. So the Parameter is an empty dummy and the real tensors land in
    `shards`, keyed by slot (0 for an unfused layer). `parts` remembers the loader's shard id of each slot, which the
    TP cut needs to know which logical partitions a tensor carries.

    It carries its OWN weight_loader: vLLM's generic path is `param.weight_loader(param, loaded_weight, shard_id)`, and
    the layer's legacy loader would assert the placeholder's shape against the incoming tensor.
    """

    def __new__(cls, **kwargs):
        return super().__new__(cls, data=torch.empty(0))

    def __init__(self, suffix: str):
        super().__init__(data=torch.empty(0), weight_loader=_capture)
        # A dict rather than a fixed list: the number of EXL3 tensors behind a fused layer is not its output-partition
        # count (vLLM builds in_proj_qkvz with four partitions while the pack has two tensors).
        self.shards: dict[int, torch.Tensor] = {}
        self.parts: dict[int, Any] = {}
        self.suffix = suffix

    def slot(self, shard_id):
        return slot(shard_id)

    # The BasevLLMParameter hooks, for whichever layers take vLLM's newer path.
    def load_column_parallel_weight(self, loaded_weight, **kw):
        _capture(self, loaded_weight, kw.get("shard_id", kw.get("loaded_shard_id")))

    load_row_parallel_weight = load_column_parallel_weight
    load_merged_column_weight = load_column_parallel_weight
    load_qkv_weight = load_column_parallel_weight


def _capture(param, loaded_weight, shard_id=None, **kw):
    """Stash a shard; shapes are whatever the checkpoint says."""
    s = slot(shard_id)
    param.shards[s] = loaded_weight
    param.parts[s] = shard_id


def release(layer, names):
    """Empty and drop the placeholders: their tensors are views of the safetensors mmap."""
    for p in names:
        if hasattr(layer, p):
            sh = getattr(getattr(layer, p), "shards", None)
            if sh is not None:
                sh.clear()
            delattr(layer, p)


class Exl3Config(QuantizationConfig):
    """`quant_method: "exl3"`: exllamav3 trellis quantization, any per-tensor bitrate."""

    def __init__(self, bits: float, codebook: str, head_bits: float, full_config: dict):
        super().__init__()
        self.bits = bits
        self.codebook = codebook
        self.head_bits = head_bits
        self.full_config = full_config
        # which module prefixes actually carry a trellis, from the pack's own manifest (normalised)
        self.quantized = manifest_prefixes(full_config.get("tensor_storage") or {})

    def __repr__(self):
        return (f"Exl3Config(bits={self.bits}, codebook={self.codebook!r}, "
                f"head_bits={self.head_bits}, quantized={len(self.quantized)})")

    @classmethod
    def get_name(cls):
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        # Fail here, not in the middle of a forward pass: every GEMM reaches for the extension lazily.
        try:
            from exllamav3.ext import exllamav3_ext  # noqa: F401
        except Exception as e:
            raise RuntimeError(EXLLAMAV3_MISSING.format(err=f"{type(e).__name__}: {e}")) from e
        return cls(bits=float(config.get("bits", 0.0)),
                   codebook=config.get("codebook", "mul1"),
                   head_bits=float(config.get("head_bits", 16)),
                   full_config=config)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

        if isinstance(layer, RoutedExperts):
            from .moe import Exl3MoEMethod
            return Exl3MoEMethod(layer.moe_config, prefix)
        if isinstance(layer, ParallelLMHead):
            # A recipe with head_bits below 16 makes lm_head an ordinary EXL3 tensor; ParallelLMHead is exempt from
            # embedding(), so the linear method is correct. It falls back to a dense weight when the pack has one.
            return Exl3LinearMethod(self, prefix)
        if isinstance(layer, VocabParallelEmbedding):
            from .embedding import Exl3EmbeddingMethod
            return Exl3EmbeddingMethod()
        if not isinstance(layer, LinearBase):
            return None
        # A fused layer is quantized iff its constituents are; the manifest is authoritative when present, otherwise
        # every LinearBase is assumed quantized.
        if self.quantized and not any(norm_prefix(c) in self.quantized for c in expand(prefix)):
            from vllm.model_executor.layers.quantization.unquant import UnquantizedLinearMethod
            return UnquantizedLinearMethod()
        return Exl3LinearMethod(self, prefix)

    def get_scaled_act_names(self):
        return []


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config, prefix: str = ""):
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        # An EXL3 pack is not uniformly quantized: exllamav3 leaves some projections in bf16 (linear_attn.in_proj_a/b,
        # anything given 16 bits) and they arrive as a plain `weight`. Register placeholders for both, decide later.
        for suffix in LINEAR_SUFFIXES:
            layer.register_parameter(suffix, Exl3ShardParam(suffix=suffix))
        layer.exl3_out_total = sum(output_partition_sizes)
        layer.exl3_tp = None
        layer.exl3_row_pad = None
        layer.exl3_col_sel = None
        layer.exl3_out_trim = None
        layer.exl3_out_mul = None                   # set by a family hook (see hooks.py)
        if input_size != input_size_per_partition or sum(output_partition_sizes) != output_size:
            layer.exl3_tp = dict(rank=getattr(layer, "tp_rank", 0), tp=getattr(layer, "tp_size", 1),
                                 row=input_size != input_size_per_partition,
                                 in_local=input_size_per_partition, out_local=list(output_partition_sizes),
                                 vocab=getattr(layer, "shard_indices", None))

    def process_weights_after_loading(self, layer) -> None:
        if layer.exl3_tp is not None:
            sel = self._cut_tp(layer, layer.exl3_tp)
        else:
            sel = None
        tr, suh, svh = layer.trellis.shards, layer.suh.shards, layer.svh.shards
        mcg, mul1 = layer.mcg.shards, layer.mul1.shards
        dense = layer.weight.shards
        # padding of a single unfused layer: the pack stores up to 127 extra output columns
        got = sum(v.numel() for v in svh.values()) if tr else layer.exl3_out_total
        if got != layer.exl3_out_total and len(tr) == 1 and 0 < got - layer.exl3_out_total < 128:
            layer.exl3_out_trim, layer.exl3_out_total = layer.exl3_out_total, got
        slots = sorted(tr) or sorted(dense)          # shard order == vLLM's concat order
        # The loader may hand these over on CPU; pin everything to this worker's device.
        dev = next((t.device for t in list(tr.values()) + list(dense.values()) if t.device.type == "cuda"),
                   torch.device("cuda", torch.cuda.current_device()))
        layer.exl3_shards = None
        layer.exl3_dense = None
        if not tr:
            if not dense:
                if self.prefix.endswith("lm_head"):
                    # A DFlash/EAGLE draft ships no lm_head; vLLM replaces this module with the target's after loading
                    # (_maybe_share_lm_head). Leave it empty instead of failing.
                    release(layer, LINEAR_SUFFIXES)
                    return
                raise RuntimeError(f"{self.prefix}: no trellis and no dense weight loaded")
            w = torch.cat([dense[i] for i in slots], 0) if len(slots) > 1 else dense[slots[0]]
            from vllm.config import get_current_vllm_config
            dt = get_current_vllm_config().model_config.dtype
            layer.exl3_dense = torch.nn.Parameter(w.contiguous().to(dev, dt), requires_grad=False)
        else:
            if dense:
                raise RuntimeError(f"{self.prefix}: {len(tr)} trellis and {len(dense)} dense shards; "
                                   "a fused layer must be all one kind or all the other")
            layer.exl3_shards = [
                dict(trellis=tr[i].contiguous().to(dev), suh=suh[i].to(dev, torch.half).contiguous(),
                     svh=svh[i].to(dev, torch.half).contiguous(),
                     mcg=mcg.get(i) is not None, mul1=mul1.get(i) is not None,
                     K=trellis_k(tr[i].shape[-1]), ic=suh[i].numel(), oc=svh[i].numel())
                for i in slots
            ]
            got = sum(s["oc"] for s in layer.exl3_shards)
            if got != layer.exl3_out_total:
                raise RuntimeError(f"{self.prefix}: shards give {got} outputs, layer wants {layer.exl3_out_total}")
        release(layer, LINEAR_SUFFIXES)
        if sel is not None:
            layer.exl3_col_sel = torch.tensor(sel, dtype=torch.long, device=dev)
        for hook in hooks.LINEAR:
            hook(self.prefix, layer)

    def _cut_tp(self, layer, t):
        """Keep this rank's slice of the captured tensors. Returns output columns to select, or None."""
        rank, tp = t["rank"], t["tp"]
        tr, suh, svh = layer.trellis.shards, layer.suh.shards, layer.svh.shards
        dense = layer.weight.shards
        if t["row"]:
            a, b = row_range(rank, t["in_local"])
            if tr:
                if not on_block([(a, b)]):
                    # e.g. Flash-Next shared_expert.down_proj (640 -> 320 per GPU): the Hadamard block 256:384 straddles
                    # the ranks. Keep the whole matrix and zero-pad this rank's input to full width; the all-reduce of
                    # W·x_r sums to W·x exactly.
                    layer.exl3_row_pad = (a, t["in_local"] * tp - b)
                    return None
                for s in list(tr):
                    tr[s] = tr[s][a // 16:b // 16].contiguous()
                    suh[s] = suh[s][a:b].contiguous()
            for s in list(dense):
                dense[s] = dense[s][:, a:b].contiguous()
            return None
        parts = (layer.trellis if tr else layer.weight).parts
        vocab = t["vocab"]
        sel, off, cut = [], 0, False
        for s in list(tr or dense):
            total = svh[s].numel() if tr else dense[s].shape[0]
            if vocab is not None:                    # lm_head: this rank's vocabulary rows
                a, b = vocab.org_vocab_start_index, vocab.org_vocab_end_index
                if b - a != t["out_local"][0]:
                    raise NotImplementedError(f"{self.prefix}: padded vocabulary partition "
                                              f"({b - a} rows for {t['out_local'][0]})")
                ranges = [(a, b)]
            else:
                ranges = col_ranges(total, logical_parts(parts.get(s), len(t["out_local"])), t["out_local"], rank, tp)
            if tr and not on_block(ranges):
                # Not on a 128-column block (exllamav3's TP granularity; the reconstruct kernel rejects an output width
                # that is not a multiple of 128, e.g. Flash-Next shared_expert gate_up: 320 per GPU): keep the whole
                # tensor, compute every output and keep this rank's columns.
                sel.extend(i for a, b in ranges for i in range(off + a, off + b))
                off += total
                continue
            off += total
            cut = True
            if tr:
                tr[s] = torch.cat([tr[s][:, a // 16:b // 16] for a, b in ranges], 1).contiguous()
                svh[s] = torch.cat([svh[s][a:b] for a, b in ranges], 0).contiguous()
            else:
                dense[s] = torch.cat([dense[s][a:b] for a, b in ranges], 0).contiguous()
        if not sel:
            return None
        if cut:
            raise NotImplementedError(f"{self.prefix}: fused layer with some shards off the 128-column block")
        layer.exl3_out_total = sum(v.numel() for v in svh.values())   # the kernels produce every output
        return sel

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.exl3_row_pad is not None:
            x = F.pad(x, layer.exl3_row_pad)
        if layer.exl3_col_sel is not None:
            out = self._forward(layer, x).index_select(-1, layer.exl3_col_sel)
        elif layer.exl3_out_trim is not None:
            out = self._forward(layer, x)[..., :layer.exl3_out_trim]
        else:
            out = self._forward(layer, x)
        if layer.exl3_out_mul is not None:
            out = out * layer.exl3_out_mul
        return out + bias if bias is not None else out

    @staticmethod
    def _forward(layer, x):
        if layer.exl3_shards is None:
            return F.linear(x, layer.exl3_dense.to(x.dtype))   # a no-op: converted to the model dtype at load
        sh = layer.exl3_shards
        if x.shape[-1] < sh[0]["ic"]:
            x = F.pad(x, (0, sh[0]["ic"] - x.shape[-1]))
        rows = x.numel() // x.shape[-1]
        xf = x.reshape(rows, x.shape[-1]).half()
        if not xf.is_contiguous():
            xf = xf.contiguous()          # the kernels read A as contiguous rows
        # One custom op per shard: the extension calls are invisible to Dynamo (see ops.py).
        outs = [torch.ops.quantpack.exl3_gemm(xf, s["trellis"], s["suh"], s["svh"], s["oc"], float(s["K"]),
                                              s["mcg"], s["mul1"]) for s in sh]
        out = (outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)).to(x.dtype)
        return out.reshape(*x.shape[:-1], out.shape[-1])
