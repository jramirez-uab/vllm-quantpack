"""EXL3 `interm_div` for Flint layers 46-48.

Layers 46-48 of Flint carry massive activations (layer 46 MLP output ~25k): the gate*up intermediate overflows the fp16
input of down_proj. The exllamav3 port (architecture/flint.py) loads the up projection of those layers with
weight_scale = 1/interm_div and folds interm_div back into the routing weights. The same has to happen here:

  * routed experts: the up trellis on disk is already /= interm_div (the converter applies weight_scale before
    quantizing), so only the expert sum is multiplied by interm_div;
  * shared expert: stored unscaled, so the up shard svh of gate_up_proj /= interm_div and the down_proj output
    *= interm_div.

Only EXL3 layers of a `flint` model are touched; BF16 checkpoints do not go through these quant methods.
"""
import re

INTERM_DIV = 128.0
FIRST_LAYER = 46        # must match exllamav3/architecture/flint.py

# The routed experts may sit one module below `mlp.experts` (FusedMoEFactory), so that one is matched as a prefix
_PREFIX = re.compile(r"layers\.(\d+)\.mlp\.(shared_expert\.gate_up_proj$|shared_expert\.down_proj$|experts(?:\.|$))")


def kind(prefix: str, model_type: str):
    """Which interm_div adjustment a layer needs: 'shared_expert.gate_up_proj', 'shared_expert.down_proj',
    'experts' or None."""
    if model_type != "flint":
        return None
    m = _PREFIX.search(prefix or "")
    if m is None or int(m.group(1)) < FIRST_LAYER:
        return None
    return m.group(2).rstrip("$.")


def _model_type():
    from vllm.config import get_current_vllm_config
    try:
        return get_current_vllm_config().model_config.hf_config.model_type
    except Exception:
        return None


def register(logger):
    from ...exl3 import hooks

    def linear(prefix, layer):
        k = kind(prefix, _model_type())
        if k == "shared_expert.gate_up_proj":
            sh = layer.exl3_shards
            if sh is None or len(sh) != 2:
                raise RuntimeError(f"vllm-quantpack: {prefix}: expected 2 EXL3 shards (gate, up)")
            sh[1]["svh"].mul_(1.0 / INTERM_DIV)
            logger.info("vllm-quantpack: Flint %s up /= %g", prefix, INTERM_DIV)
        elif k == "shared_expert.down_proj":
            layer.exl3_out_mul = INTERM_DIV

    def moe(prefix, layer, ex):
        if kind(prefix, _model_type()) == "experts":
            # The routed up trellis on disk is already divided by interm_div (exllamav3 Linear.load_fp16 applies
            # weight_scale before quantizing, load_exl3 does not apply it again): only the x interm_div compensation is
            # missing (exllamav3 folds it into routed_scaling_factor).
            ex.out_scale = INTERM_DIV
            logger.info("vllm-quantpack: Flint %s routed out *= %g (up already scaled on disk)", prefix, INTERM_DIV)

    global _registered
    if not _registered:
        hooks.LINEAR.append(linear)
        hooks.MOE.append(moe)
        _registered = True


_registered = False
