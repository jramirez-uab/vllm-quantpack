"""Speculative decoding (MTP / EAGLE drafters) with EXL3 heads.

vLLM decides whether a drafter shares the target's embedding or lm_head by comparing `.weight` tensors
(`_should_share` in v1/worker/gpu/spec_decode/eagle/utils.py). An EXL3 lm_head or embedding has no `.weight` (its
data live in the EXL3 store), so that comparison raised AttributeError and the drafter failed to load (seen with
Nemotron-3.5-Super-VL's MTP head). When either side has no `.weight`, keep the drafter's own copy.
"""
import importlib


def patch_should_share(logger):
    try:
        utils = importlib.import_module("vllm.v1.worker.gpu.spec_decode.eagle.utils")
    except Exception as e:        # another vLLM layout: nothing to patch
        logger.debug("vllm-quantpack: spec-decode utils not patched (%s)", e)
        return
    orig = getattr(utils, "_should_share", None)
    if orig is None or getattr(orig, "_quantpack", False):
        return

    def _should_share(eagle, flag, draft, target):
        if (getattr(eagle, flag, False) and draft is not None and target is not None
                and not (hasattr(draft, "weight") and hasattr(target, "weight"))):
            return False
        return orig(eagle, flag, draft, target)

    _should_share._quantpack = True
    utils._should_share = _should_share
    # Drafters that imported the function by name keep their own reference.
    for mod in ("vllm.v1.worker.gpu.spec_decode.dflash.utils", "vllm.v1.worker.gpu.spec_decode.dspark.utils"):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        if getattr(m, "_should_share", None) is orig:
            m._should_share = _should_share
    logger.debug("vllm-quantpack: drafters keep their own head when the target's is EXL3")
