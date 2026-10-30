"""vllm-quantpack: serve EXL3 (exllamav3 trellis) checkpoints, and the model-family fixes they need, in stock vLLM.

Installed as a vLLM general plugin (entry point `vllm.general_plugins`), so `vllm serve <repo>` picks up any checkpoint
whose config says `quant_method: "exl3"` with no flag and no patch to vLLM.

Layout:
  exl3/      quantization config, linear / embedding / MoE methods, the custom ops, tensor-parallel cuts
  families/  fixes for specific model families (Qwen vision and Flash-Next PLE) and models vLLM lacks or
             maps incompletely (Flint, Spark, Nemotron-3.5-Super-VL)

It replaces OrcaSAQ2 (whose code it includes, Apache-2.0), qwen38_exl3_vision, flint_vllm and spark_vllm, which must
not be installed alongside.
"""
__version__ = "0.1.4"

# Entry-point names of the plugins this package replaces.
_CONFLICTS = ("orcasaq2", "qwen38_exl3_vision", "tarninfer_exl3", "flint_vllm", "spark_vllm")


def _conflicts():
    """Replaced plugins that vLLM will also load: installed, and not left out by VLLM_PLUGINS."""
    import os
    from importlib.metadata import entry_points

    allowed = os.environ.get("VLLM_PLUGINS")
    allowed = None if allowed is None else {p.strip() for p in allowed.split(",")}
    return sorted({ep.name for ep in entry_points(group="vllm.general_plugins")
                   if ep.name in _CONFLICTS and (allowed is None or ep.name in allowed)})


def register():
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization import register_quantization_config

    logger = init_logger("vllm.quantpack")
    found = _conflicts()
    if found:
        # They register the same quant_method and patch the same classes; which one wins would depend on plugin order.
        raise RuntimeError(f"vllm-quantpack replaces the vLLM plugin(s) {', '.join(found)}: uninstall them from this "
                           "environment, or leave them out with VLLM_PLUGINS=quantpack")

    from .exl3.layers import patch_embedding_layers
    from .exl3.linear import Exl3Config

    try:
        register_quantization_config("exl3")(Exl3Config)
    except ValueError:
        return  # already registered in this process
    patch_embedding_layers(logger)

    from .attention.int4_kv import patch_int4_kv
    from .exl3.spec_decode import patch_should_share
    from .families import flint, nemotron, qwen, spark
    patch_should_share(logger)
    patch_int4_kv(logger)
    qwen.register(logger)
    if flint.register(logger):
        logger.info("vllm-quantpack: Flint model registered")
    if spark.register(logger):
        logger.info("vllm-quantpack: Spark model registered")
    if nemotron.register(logger):
        logger.info("vllm-quantpack: Nemotron-3.5-Super-VL model registered")
    logger.info("vllm-quantpack %s: quant_method 'exl3' is loadable (linear, int8 embedding, MoE, TP)", __version__)
