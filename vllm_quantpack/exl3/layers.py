"""Give VocabParallelEmbedding and ParallelLMHead the quant config the model withholds.

vLLM consults quant_config only for layers that are constructed with one, and several models (qwen3_next.py among them)
build `embed_tokens` and `lm_head` without passing it. A pack whose lm_head is an EXL3 tensor, or whose embedding is int8
rows, then cannot load: the loader looks for `lm_head.weight` / `embed_tokens.weight` and fails on the names it finds.
The two constructors are wrapped to supply the current config when the caller omitted one:

  * EXL3: always (layers of other quant methods are untouched);
  * compressed-tensors (GPTQ packs with an int4/int8 embedding, `re:.*embed_tokens$`): only for the embedding a family
    module announced with `embedding_prefix(...)`, because compressed-tensors picks the scheme by layer name and the
    constructor alone does not know it (see families/qwen.py).

Origin: OrcaSAQ2 __init__.py (Apache-2.0); the compressed-tensors case replaces qwen3_5-embed-quant.patch from
syv-ai/qwen38-27b-rtx3090 (Apache-2.0).
"""
import contextlib
import contextvars
import inspect

_pending_embed = contextvars.ContextVar("quantpack_pending_embed", default=None)


@contextlib.contextmanager
def embedding_prefix(prefix: str):
    """While a model builds its layers: the next embedding built without a quant config is `prefix`."""
    token = _pending_embed.set(prefix)
    try:
        yield
    finally:
        _pending_embed.reset(token)


def take_embedding_prefix():
    """The announced embedding prefix, once; None if there is none or it was already used."""
    prefix = _pending_embed.get()
    if prefix is not None:
        _pending_embed.set(None)
    return prefix


def _current_quant_config():
    from vllm.config import get_current_vllm_config
    try:
        cfg = get_current_vllm_config()
    except Exception:
        return None
    return getattr(cfg, "quant_config", None) or getattr(getattr(cfg, "model_config", None), "quant_config", None)


def _current_exl3_config():
    from .linear import Exl3Config
    qc = _current_quant_config()
    return qc if isinstance(qc, Exl3Config) else None


def patch_embedding_layers(logger):
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe

    if getattr(vpe.VocabParallelEmbedding, "_quantpack_patched", False):
        return
    for cls in (vpe.VocabParallelEmbedding, vpe.ParallelLMHead):
        def make(orig_init, is_head):
            sig = inspect.signature(orig_init)

            def __init__(self, *args, **kwargs):
                # ParallelLMHead forwards quant_config to its parent POSITIONALLY, and the parent is wrapped too, so
                # injecting a keyword blindly collides. Bind against the real signature and only fill an empty slot.
                try:
                    bound = sig.bind_partial(self, *args, **kwargs)
                except TypeError:
                    return orig_init(self, *args, **kwargs)
                if bound.arguments.get("quant_config") is None:
                    qc = _current_exl3_config()
                    if qc is not None:
                        bound.arguments["quant_config"] = qc
                        return orig_init(*bound.args, **bound.kwargs)
                    if not is_head and "prefix" in sig.parameters:
                        prefix = take_embedding_prefix()
                        qc = _current_quant_config() if prefix is not None else None
                        if type(qc).__name__ == "CompressedTensorsConfig":
                            bound.arguments["quant_config"] = qc
                            bound.arguments["prefix"] = prefix
                            return orig_init(*bound.args, **bound.kwargs)
                return orig_init(self, *args, **kwargs)
            return __init__

        cls.__init__ = make(cls.__init__, cls is vpe.ParallelLMHead)
    vpe.VocabParallelEmbedding._quantpack_patched = True
    logger.info("vllm-quantpack: embed_tokens / lm_head receive the EXL3 (or compressed-tensors embedding) quant config")
