"""Qwen-family fixes (Qwen3.5/3.6/3.8, their finetunes, Qwen3.8-Flash-Next / Qwen4Exp).

1. Vision attention: exllamav3 stores each vision block's attention twice, as `attn.qkv.weight` (FP16) and split into
   quantized `attn.q_proj` / `k_proj` / `v_proj` tensors. vLLM only knows the fused `attn.qkv`, so the split copies are
   dropped at load time and the vision attention stays dense (~215 MB); the rest of the vision tower stays EXL3.
   Checkpoints without the split copies (GPTQ, BF16) are unaffected: the rule never matches.
2. Flash-Next PLE: its FP8 n-gram table, kept in pinned host memory, is looked up by a Triton kernel typed float8e4nv,
   which Triton rejects below sm_89 (Ampere). There the lookup copies the rows as bytes (uint8 views) and the
   dequantization stays in torch, which works on every GPU.
3. Quantized embedding (GPTQ packs with an int4/int8 `embed_tokens`): Qwen3_5Model and the MTP predictor build
   `embed_tokens` without quant_config or prefix, so compressed-tensors never sees it and the load fails on
   `embed_tokens.weight_packed`. Their constructors announce the prefix and exl3/layers.py passes both. Replaces
   qwen3_5-embed-quant.patch (syv-ai/qwen38-27b-rtx3090, Apache-2.0).
4. service-ple (optional): when the vllm-service-ple package is installed, its patch is applied too
   (`--additional-config '{"service_ple": "host:port"}'`); either load order works.

Origin: qwen38_exl3_vision 0.4.1 __init__.py (ours).
"""
import re


def patch_vision_qkv():
    try:
        from vllm.model_executor.models import qwen3_5
    except ImportError:
        return
    from vllm.model_executor.models.utils import WeightsMapper

    cls = qwen3_5.Qwen3_5ForConditionalGeneration
    if getattr(cls, "_quantpack_vision", False):
        return
    drop = WeightsMapper(orig_to_new_regex={re.compile(r"^model\.visual\.blocks\.\d+\.attn\.[qkv]_proj\."): None})
    cls.hf_to_vllm_mapper = drop | cls.hf_to_vllm_mapper
    cls._quantpack_vision = True


def patch_ple_fp8_lookup():
    import torch
    try:
        from vllm.models.qwen4_exp.common import ngram_embedding as ne
    except Exception:
        return
    cls = ne.Qwen4ExpPLEPinnedHostEmbedding
    if getattr(cls, "_quantpack_ple", False):
        return
    orig = cls._lookup

    def _lookup(self, input_ids, output=None):
        if getattr(self, "_sple_remoto", False):  # table served by service-ple: its own _lookup does it
            return orig(self, input_ids, output)
        w = self._uva_weight
        if w.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or torch.cuda.get_device_capability(w.device) >= (8, 9):
            return orig(self, input_ids, output)
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(expected_shape, dtype=w.dtype, device=input_ids.device)
        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            ne._lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                w.view(torch.uint8), flat_ids, output.view(torch.uint8), self.embedding_dim,
                self.shard_indices.org_vocab_start_index, self.shard_indices.org_vocab_end_index, BLOCK_D=self._block_d)
        return output

    cls._lookup = _lookup
    cls._quantpack_ple = True


def patch_embed_prefix():
    try:
        from vllm.model_executor.models import qwen3_5, qwen3_5_mtp
    except ImportError:
        return
    from vllm.model_executor.models.utils import maybe_prefix

    from ..exl3.layers import embedding_prefix

    for cls in (qwen3_5.Qwen3_5Model, qwen3_5_mtp.Qwen3_5MultiTokenPredictor):
        if getattr(cls, "_quantpack_embed_prefix", False):
            continue

        def make(orig_init):
            def __init__(self, *, vllm_config, prefix: str = ""):
                with embedding_prefix(maybe_prefix(prefix, "embed_tokens")):
                    orig_init(self, vllm_config=vllm_config, prefix=prefix)
            return __init__

        cls.__init__ = make(cls.__init__)
        cls._quantpack_embed_prefix = True


def patch_service_ple():
    try:
        from vllm_service_ple.service_ple import patch
    except ImportError:
        return False
    patch()
    return True


def register(logger):
    patch_vision_qkv()
    patch_ple_fp8_lookup()
    patch_embed_prefix()
    sple = patch_service_ple()
    logger.info("vllm-quantpack: Qwen family (vision qkv, Flash-Next PLE lookup, quantized embedding%s)", ", service-ple" if sple else "")
