"""The embedding table for vLLM's VocabParallelEmbedding.

exllamav3 refuses to quantize embed_tokens and keeps it in system RAM, so its native engine never pays VRAM for it. vLLM
does: VocabParallelEmbedding is a GPU tensor (2.54 GB in bf16 for a 248,320-token vocabulary). Our packs store it as
int8 rows with a per-row scale (scripts/embed_int8.py in each repo; the ablation measured that as free, KLD 0.0338
against 0.0341 for bf16) or as int4 with one scale per 128 columns (tools/embed_quant.py --bits 4), dequantized on the
GATHERED rows so the table itself stays quantized (formats: embed_codec.py).

Origin: OrcaSAQ2 heads.py (Apache-2.0) with the published patch (empty draft table), plus the TP cut and host release
qwen38_exl3_vision 0.4.1 added: each rank keeps its own vocabulary rows (VocabParallelEmbedding masks the ids and
all-reduces).
"""
import torch
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from .embed_codec import dequantize, gather
from .linear import Exl3ShardParam, release

EMBED_SUFFIXES = ("qweight", "scales", "weight")


class Exl3EmbeddingMethod(QuantizeMethodBase):
    """int8 rows with a per-row scale, int4 with per-group scales, or a dense table, whichever the pack carries."""

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        # `extra_weight_attrs` is deliberately not applied: it carries vLLM's vocab-sharding weight_loader, which would
        # assert against our empty placeholders.
        for suffix in EMBED_SUFFIXES:
            layer.register_parameter(suffix, Exl3ShardParam(suffix=suffix))
        layer.exl3_params_dtype = params_dtype

    def process_weights_after_loading(self, layer):
        if getattr(layer, "tp_size", 1) > 1:
            _cut_tp(layer)
        q = layer.qweight.shards.get(0)
        s = layer.scales.shards.get(0)
        w = layer.weight.shards.get(0)
        dev = next((t.device for t in (q, w) if t is not None and t.device.type == "cuda"),
                   torch.device("cuda", torch.cuda.current_device()))
        release(layer, EMBED_SUFFIXES)
        # vLLM's speculative proposer reads `<embed>.weight` (EAGLE/DFlash width check), so keep the name, as None.
        layer.weight = None
        layer.exl3_embed_q = None
        layer.exl3_embed = None
        if q is None:
            # No table at all: a DFlash/EAGLE draft, whose module vLLM replaces with the target's after loading
            # (_maybe_share_embeddings). embedding() complains if it is ever used.
            if w is not None:
                layer.exl3_embed = torch.nn.Parameter(w.to(dev), requires_grad=False)
            return
        if s is None:
            raise RuntimeError("embed_tokens: qweight without scales")
        layer.exl3_embed_q = torch.nn.Parameter(q.to(dev), requires_grad=False)
        layer.exl3_embed_s = torch.nn.Parameter(s.to(dev, layer.exl3_params_dtype).reshape(s.shape[0], -1),
                                                requires_grad=False)

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        if layer.exl3_embed_q is None and layer.exl3_embed is None:
            raise RuntimeError("embed_tokens: empty table (a draft whose embeddings were not shared with the target)")
        if layer.exl3_embed_q is None:
            return layer.exl3_embed[input_]
        # Index directly rather than through F.embedding: the table is int8/int4, and the scales are applied to the
        # gathered rows so the table itself is never materialised in bf16.
        return gather(layer.exl3_embed_q, layer.exl3_embed_s, input_, layer.exl3_embed_s.dtype)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer.exl3_embed_q is None:
            w = layer.exl3_embed
        else:
            w = dequantize(layer.exl3_embed_q, layer.exl3_embed_s, x.dtype)
        out = torch.nn.functional.linear(x, w.to(x.dtype))
        return out + bias if bias is not None else out


def _cut_tp(layer):
    si = layer.shard_indices
    a, b = si.org_vocab_start_index, si.org_vocab_end_index
    n = layer.num_embeddings_per_partition
    for name in EMBED_SUFFIXES:
        d = getattr(layer, name).shards
        if 0 not in d:
            continue
        full = d[0]
        local = full.new_zeros((n,) + tuple(full.shape[1:]))
        local[: b - a] = full[a:b]
        d[0] = local
