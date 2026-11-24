"""thinkingmachines Inkling 2 (Inkling-2-Mini, `Inkling2LoopForConditionalGeneration`) from an EXL3 pack.

vLLM's Inkling2 (vllm/models/inkling2, in vLLM builds after 0.34.0) serves only its own NVFP4 / MXFP8 checkpoints: it
asserts its quantization config class, builds the attention projections without a quant config and runs them on the
raw weights, and its routed experts are custom kernels, not a FusedMoE. model.py subclasses it so an EXL3 pack loads;
any other checkpoint goes through vLLM's code unchanged. Registered only when vLLM ships Inkling2.
"""
import importlib.util


def register(logger):
    try:
        if importlib.util.find_spec("vllm.models.inkling2") is None:
            return False
    except ModuleNotFoundError:
        return False
    from . import ampere
    ampere.install(logger)        # before any vllm.models.inkling2 module is imported
    from vllm import ModelRegistry
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig

    # vLLM claims every Inkling2-architecture checkpoint for its own quantization ("inkling2", which then wants
    # hf_quant_config.json), whatever config.json declares. Leave EXL3 packs to quant_method "exl3".
    if not getattr(Inkling2QuantConfig, "_quantpack_patched", False):
        orig = Inkling2QuantConfig.override_quantization_method.__func__

        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            if (hf_quant_cfg or {}).get("quant_method") == "exl3":
                return None
            return orig(cls, hf_quant_cfg, user_quant, hf_config)

        Inkling2QuantConfig.override_quantization_method = classmethod(override_quantization_method)
        Inkling2QuantConfig._quantpack_patched = True

    ModelRegistry.register_model("Inkling2LoopForConditionalGeneration",
                                 "vllm_quantpack.families.inkling2.model:Inkling2LoopForConditionalGeneration")
    return True
