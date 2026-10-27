"""XHToken Spark-X2.5 (`Spark2_5ForCausalLM`, model_type `spark2_5`), which vLLM does not ship.

The checkpoint's remote code targets transformers 4.57 and calls its own eager attention, so it cannot run through
vLLM's Transformers backend. Registers:
  * a transformers-5 compatible config class for `spark2_5` (serve without --trust-remote-code),
  * a native vLLM implementation of the model (paged attention, sliding window, BF16 or EXL3).
"""


def register(logger):
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .config import Spark2_5Config

    try:
        AutoConfig.register("spark2_5", Spark2_5Config, exist_ok=True)
    except TypeError:             # older transformers without exist_ok
        try:
            AutoConfig.register("spark2_5", Spark2_5Config)
        except ValueError:
            pass
    if "Spark2_5ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Spark2_5ForCausalLM", "vllm_quantpack.families.spark.model:Spark2_5ForCausalLM")
    return True
