"""Qwen-family load fixes against the installed vLLM (CPU only; skipped without vLLM).

vLLM 0.34 loads Qwen3.5/3.6 on CUDA from vllm/models/qwen3_5/nvidia instead of vllm/model_executor/models/qwen3_5:
the vision split-copy drop must reach the class the registry actually returns, or the vision tower fails to load
with "no module or parameter named 'blocks.0.attn.k_proj' in Qwen3_VisionTransformer".
"""
import importlib

import pytest

pytest.importorskip("vllm")

from vllm_quantpack.families import qwen  # noqa: E402

SPLIT = "model.visual.blocks.0.attn.k_proj.trellis"
FUSED = "model.visual.blocks.0.attn.qkv.weight"


def registry_class(arch):
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
    mod, name = _MULTIMODAL_MODELS[arch]
    full = mod if mod.startswith("vllm.") else f"vllm.model_executor.models.{mod}"
    return getattr(importlib.import_module(full), name)


@pytest.mark.parametrize("arch", ["Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"])
def test_vision_split_copies_dropped_in_the_loaded_class(arch):
    qwen.patch_vision_qkv()
    m = registry_class(arch).hf_to_vllm_mapper
    assert m.map_name(SPLIT) is None
    assert m.map_name(FUSED) == "visual.blocks.0.attn.qkv.weight"


def test_language_weights_unchanged():
    qwen.patch_vision_qkv()
    m = registry_class("Qwen3_5MoeForConditionalGeneration").hf_to_vllm_mapper
    assert m.map_name("model.language_model.layers.0.self_attn.k_proj.trellis") == \
        "language_model.model.layers.0.self_attn.k_proj.trellis"


def test_embed_prefix_patched_in_every_implementation():
    qwen.patch_embed_prefix()
    classes = qwen._qwen3_5_modules("Qwen3_5Model", "Qwen3_5MultiTokenPredictor")
    assert classes and all(c.__dict__.get("_quantpack_embed_prefix") for c in classes)
