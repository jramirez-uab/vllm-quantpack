"""SenseNova Loop (`SenseNovaLoopForConditionalGeneration`, model_type `sensenova_loop`), e.g. SenseNova-Loop-9B.

vLLM 0.34.0 does not ship it; vLLM PR #64832 adds a text-only, BF16/AWQ, no-speculation version. This family carries
that implementation plus what an EXL3 pack needs: EXL3 accepted, execution-layer prefixes routed to the physical layer
so the looped layers share one set of EXL3 tensors, the vision tower (Qwen3.5 VL wrapper) and the model's own MTP head.

Idempotent: every piece is registered only when vLLM lacks it, so a vLLM that already ships SenseNova Loop (the PR
merged, or a patched build) keeps its own code and is not broken by this plugin:
  * config class `sensenova_loop`: skipped when vLLM's config registry or transformers already knows it;
  * model class: skipped when vLLM already registers the architecture (set VLLM_QUANTPACK_SENSENOVA=force to use
    this one anyway, e.g. for an EXL3 pack on a vLLM whose version rejects EXL3);
  * VerifyAndUpdateConfig entry (no compile, FULL_DECODE_ONLY graphs): added only when missing;
  * MTP: `sensenova_loop` is mapped to the Qwen3.5 MTP draft only when vLLM's hf_config_override does not
    handle it already.
"""
import inspect
import os

ARCH = "SenseNovaLoopForConditionalGeneration"
MODEL_TYPE = "sensenova_loop"


def _register_config(logger):
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        if MODEL_TYPE in _CONFIG_REGISTRY:
            return "vllm"
    except ImportError:
        pass
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if MODEL_TYPE in CONFIG_MAPPING:
        return "transformers"
    from .config import SenseNovaLoopConfig
    try:
        AutoConfig.register(MODEL_TYPE, SenseNovaLoopConfig, exist_ok=True)
    except TypeError:             # older transformers without exist_ok
        AutoConfig.register(MODEL_TYPE, SenseNovaLoopConfig)
    return "quantpack"


def _register_verify(logger):
    from vllm.model_executor.models import config as model_config

    if ARCH in model_config.MODELS_CONFIG_MAP:
        return False
    from .verify import SenseNovaLoopVerifyConfig
    model_config.MODELS_CONFIG_MAP[ARCH] = SenseNovaLoopVerifyConfig
    return True


def _register_mtp(logger):
    from vllm.config.speculative import SpeculativeConfig

    fn = SpeculativeConfig.hf_config_override
    if getattr(fn, "_quantpack_sensenova", False):
        return False
    try:
        if MODEL_TYPE in inspect.getsource(fn):
            return False          # vLLM maps it itself
    except (OSError, TypeError):
        pass

    def hf_config_override(hf_config, *args, **kwargs):
        # The SenseNova Loop MTP head is a Qwen3.5 MTP layer (mtp.* weights, `mtp_num_hidden_layers`).
        if getattr(hf_config, "model_type", None) == MODEL_TYPE:
            hf_config.model_type = "qwen3_5"
        return fn(hf_config, *args, **kwargs)

    hf_config_override._quantpack_sensenova = True
    SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)
    return True


def register(logger):
    try:
        from transformers import Qwen3_5Config  # noqa: F401  (the config subclasses it)
        from vllm.model_executor.models import qwen3_5  # noqa: F401
    except ImportError as e:
        logger.warning("vllm-quantpack: SenseNova Loop not registered (%s)", e)
        return False
    from vllm import ModelRegistry

    cfg = _register_config(logger)
    force = os.environ.get("VLLM_QUANTPACK_SENSENOVA", "").lower() == "force"
    if ARCH in ModelRegistry.get_supported_archs() and not force:
        logger.info("vllm-quantpack: vLLM ships SenseNova Loop; its implementation is kept "
                    "(VLLM_QUANTPACK_SENSENOVA=force selects the quantpack one, needed for EXL3 packs)")
        return False
    ModelRegistry.register_model(ARCH, "vllm_quantpack.families.sensenova.model:" + ARCH)
    verify = _register_verify(logger)
    mtp = _register_mtp(logger)
    logger.info("vllm-quantpack: SenseNova Loop: config from %s, verify %s, MTP mapping %s", cfg,
                "added" if verify else "from vLLM", "added" if mtp else "from vLLM")
    return True
