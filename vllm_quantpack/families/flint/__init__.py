"""RecursiveSI Flint (`FlintForCausalLM`, model_type `flint`), which vLLM does not ship.

Registers:
  * the `flint` config class for transformers (serve without --trust-remote-code),
  * a native vLLM implementation of the model (paged attention with sinks, sliding window, sigmoid MoE router),
  * for EXL3 checkpoints, the `interm_div` compensation of the exllamav3 port (see interm_div.py).
"""


def register(logger):
    try:
        from transformers import AutoConfig

        from .config import FlintConfig
    except ImportError as e:      # FlintConfig needs a transformers release with dataclass configs (5.x)
        logger.warning("vllm-quantpack: Flint not registered (%s)", e)
        return False
    from vllm import ModelRegistry

    try:
        AutoConfig.register("flint", FlintConfig, exist_ok=True)
    except TypeError:             # older transformers without exist_ok
        try:
            AutoConfig.register("flint", FlintConfig)
        except ValueError:
            pass
    if "FlintForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("FlintForCausalLM", "vllm_quantpack.families.flint.model:FlintForCausalLM")
    from . import interm_div
    interm_div.register(logger)
    return True
