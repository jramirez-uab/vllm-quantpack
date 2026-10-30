"""int4_per_token_head KV cache: exact and faster attention on vLLM's Triton backend.

vLLM 0.31's packed INT4 read kernel (`_attn_packed` in v1/attention/ops/int4_per_token_head.py) has three problems,
measured on an A40 and an RTX PRO 6000 Blackwell with Qwen3.8-27B shapes (24 query / 4 KV heads, head 256) against
an fp64 reference computed from the same cache (python -m vllm_quantpack.tools.int4_kv_bench):

1. Wrong output at long context. Its dots run on fp32 operands, which Triton rounds to TF32, but the value
   zero-point correction `sum(P * v_scale * v_zp)` uses the unrounded P. The rounding error times the zero point
   (~8) does not cancel: relative error 11 % at 32k and 28 % at 229k.
2. Slow prefill. fp32 operands keep the tensor cores below bf16 rate and double the shared memory per tile, so it
   runs with BLOCK_M 16: 2 query tokens per program, every program re-reads and unpacks the whole cache.
3. Slow decode with speculative decoding. The split-KV (3D) path is used only when every request has one query
   token; MTP verification has 2, so decode falls back to one program per KV head scanning the whole context.
4. Slow long-context decode. The split-KV path reads 16 tokens per tile in 16 segments: 4 KV heads x 16 segments
   leave most SMs idle and every program unpacks the cache in small steps.

The patch generates a copy of the kernel and its launcher with: tensor-core operands in the query dtype (the nibbles
0..15 are exact in bf16) and the zero-point sum taken on the same rounded P, fp32 accumulators unchanged; prefill
tiles BLOCK_M 64 / TILE 64 / 8 warps / 2 stages; the split-KV path for up to MAX_Q_SPLIT_KV query tokens per request,
with 64-token tiles and a segment count that follows the GPU's SM count (the Triton metadata builder of an int4
cache allocates its segment buffers for that count: +38 MiB at 64 segments and 24 query heads). Result: relative error 0.4 %, prefill attention 1.9x faster,
MTP decode attention 36x faster at 259k (RTX PRO 6000 Blackwell: 436 -> 12 ms per step over 16 layers).

The copy is produced by exact text edits of the installed vLLM source. If any anchor is missing (another vLLM
version) nothing is patched and the stock kernel stays. VLLM_QUANTPACK_INT4_KV=0 disables the patch.
"""
import hashlib
import importlib.util
import inspect
import itertools
import os
import tempfile
import textwrap

MODULE = "vllm.v1.attention.ops.int4_per_token_head"
MAX_Q_SPLIT_KV = 4               # MTP / EAGLE verification with up to 3 draft tokens
# Triton prefill tiles (BLOCK_M, TILE_SIZE, num_warps, num_stages) by compute-capability major, from
# int4_kv_bench. Ampere/Ada (8.x): A40. Consumer Blackwell (12.x): RTX PRO 6000 Server, to be confirmed on an
# RTX 5060 Ti (36 SMs instead of 188). VLLM_QUANTPACK_INT4_TILES="64,64,8,2" overrides.
PREFILL_TILES_BY_MAJOR = {8: (64, 64, 8, 2), 12: (64, 64, 8, 2)}
PREFILL_TILES_DEFAULT = (64, 64, 8, 2)
# Split-KV decode (TILE_SIZE, num_warps), the same on both GPUs measured. MTP step over 16 layers at 259k:
#   RTX PRO 6000 (188 SMs): stock (16, 4) x 16 segments 44 ms; (64, 4) x 16 25 ms, x 32 18 ms, x 64 12 ms, x 128 11.5 ms
#   A40 (84 SMs):           stock 60 ms; (64, 4) x 16 42 ms, x 32 24 ms, x 64 24 ms, x 128 22 ms
# The gain stops once the grid (2 query blocks x 4 KV heads x segments) covers about two waves of SMs, so the segment
# count follows the SM count: next power of two of SMs / 4, between 16 (vLLM's value) and 128.
# VLLM_QUANTPACK_INT4_DECODE="64,4,auto" overrides (tile, warps, segments or auto).
DECODE_TILES_DEFAULT = (64, 4, "auto")
MIN_SEGMENTS, MAX_SEGMENTS = 16, 128


def prefill_tiles(major):
    env = os.environ.get("VLLM_QUANTPACK_INT4_TILES")
    if env:
        return tuple(int(x) for x in env.split(","))
    return PREFILL_TILES_BY_MAJOR.get(major, PREFILL_TILES_DEFAULT)


def segments_for(sms):
    """Split-KV segments per sequence for a GPU with `sms` multiprocessors."""
    n = 1 << max(0, -(-sms // 4) - 1).bit_length()
    return min(MAX_SEGMENTS, max(MIN_SEGMENTS, n))


def decode_tiles(sms):
    """(tile, warps, segments) for the split-KV decode on a GPU with `sms` multiprocessors."""
    env = os.environ.get("VLLM_QUANTPACK_INT4_DECODE")
    tile, warps, segments = env.split(",") if env else DECODE_TILES_DEFAULT
    segments = segments_for(sms) if str(segments).strip() == "auto" else int(segments)
    return int(tile), int(warps), segments

# (old, new): each `old` must occur exactly once in the stock kernel source.
KERNEL_EDITS = (
    ("def _attn_packed(", "def _attn_packed_qp("),
    ("Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)",
     "Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)\n"
     "    Q_s0 = Q_s0.to(query_ptr.dtype.element_ty)\n"
     "    Q_s1 = Q_s1.to(query_ptr.dtype.element_ty)"),
    ("K_s0 = K_s0_u.to(tl.float32)", "K_s0 = K_s0_u.to(query_ptr.dtype.element_ty)"),
    ("K_s1 = K_s1_u.to(tl.float32)", "K_s1 = K_s1_u.to(query_ptr.dtype.element_ty)"),
    ("V_s0 = V_s0_u.to(tl.float32)", "V_s0 = V_s0_u.to(query_ptr.dtype.element_ty)"),
    ("V_s1 = V_s1_u.to(tl.float32)", "V_s1 = V_s1_u.to(query_ptr.dtype.element_ty)"),
    ("Pv_zp_sum = tl.sum(P_v * v_zp[None, :], axis=1)",
     "P_v16 = P_v.to(query_ptr.dtype.element_ty)\n"
     "        Pv_zp_sum = tl.sum(P_v16.to(tl.float32) * v_zp[None, :], axis=1)"),
    ("acc_s0 += tl.dot(P_v, V_s0)", "acc_s0 += tl.dot(P_v16, V_s0)"),
    ("acc_s1 += tl.dot(P_v, V_s1)", "acc_s1 += tl.dot(P_v16, V_s1)"),
)

# (old, new): each `old` must occur exactly once in the stock launcher source.
LAUNCH_EDITS = (
    ("def _launch_packed_attn(", "def _launch_packed_attn_qp("),
    # The segment buffers have one row per query token (seq_threshold_3D rows): split-KV only when they fit.
    ("    use_3d = not (\n",
     "    TILE_SIZE_DECODE = max(TILE_SIZE_DECODE, _QP_DECODE_TILES[0])\n    use_3d = not (\n"),
    ("or max_seqlen_q > 1\n",
     "or max_seqlen_q > _QP_MAX_Q_SPLIT_KV\n        or q.shape[0] > softmax_segm_output.shape[0]\n"),
    ("    grid: tuple[Any, ...]\n",
     "    launch_kw = {'num_warps': _QP_DECODE_TILES[1]} if use_3d else {}\n"
     "    if not use_3d and max_seqlen_q > 1:\n"
     "        BLOCK_M = max(BLOCK_M, _QP_PREFILL_TILES[0])\n"
     "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
     "        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs\n"
     "        TILE_SIZE_PREFILL = _QP_PREFILL_TILES[1]\n"
     "        launch_kw = {'num_warps': _QP_PREFILL_TILES[2], 'num_stages': _QP_PREFILL_TILES[3]}\n"
     "    grid: tuple[Any, ...]\n"),
    ("_attn_packed[grid](", "_attn_packed_qp[grid]("),
    ("PACKING_FACTOR=packing_factor,\n    )", "PACKING_FACTOR=packing_factor,\n        **launch_kw,\n    )"),
)


def _edit(src, edits, what):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise LookupError(f"{what}: anchor found {n} times: {old.splitlines()[0]!r}")
        src = src.replace(old, new)
    return src


def build_source(stock, tiles, decode=(64, 4, 16)):
    """Source of the replacement module for the stock module object `stock`."""
    kernel = _edit(textwrap.dedent(inspect.getsource(stock._attn_packed.fn)), KERNEL_EDITS, "int4 kernel")
    launch = _edit(textwrap.dedent(inspect.getsource(stock._launch_packed_attn)), LAUNCH_EDITS, "int4 launcher")
    return ("# Generated by vllm_quantpack.attention.int4_kv from the installed vLLM source: do not edit.\n"
            f"import {MODULE} as _stock\n"
            "globals().update({k: v for k, v in vars(_stock).items() if not k.startswith('__')})\n"
            f"_QP_PREFILL_TILES = {tuple(tiles)!r}\n"
            f"_QP_DECODE_TILES = {tuple(decode)!r}\n"
            f"_QP_MAX_Q_SPLIT_KV = {MAX_Q_SPLIT_KV}\n\n\n"
            + kernel + "\n\n" + launch)   # the kernel source keeps its @triton.jit decorator


def _load(src):
    # Triton reads kernel source from the defining file, so the module must live on disk.
    base = os.environ.get("VLLM_QUANTPACK_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "vllm-quantpack")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"int4_kv_{hashlib.sha256(src.encode()).hexdigest()[:16]}.py")
    if not os.path.exists(path):
        fd, tmp = tempfile.mkstemp(dir=base, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            fh.write(src)
        os.replace(tmp, path)   # atomic: tensor-parallel workers may write the same file at once
    spec = importlib.util.spec_from_file_location("vllm_quantpack_int4_kv_generated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_int4_kv(logger):
    if os.environ.get("VLLM_QUANTPACK_INT4_KV", "1") == "0":
        return False
    try:
        stock = importlib.import_module(MODULE)
    except Exception as e:      # vLLM without the int4 KV mode
        logger.debug("vllm-quantpack: int4 KV attention not patched (%s)", e)
        return False
    if getattr(stock._launch_packed_attn, "_quantpack", False):
        return True
    try:
        import torch
        cuda = torch.cuda.is_available()
        major = torch.cuda.get_device_capability()[0] if cuda else 0
        sms = torch.cuda.get_device_properties(0).multi_processor_count if cuda else 0
        tiles, decode = prefill_tiles(major), decode_tiles(sms)
        gen = _load(build_source(stock, tiles, decode))
    except Exception as e:
        logger.warning("vllm-quantpack: int4_per_token_head attention left stock (%s)", e)
        return False
    launch = gen._launch_packed_attn_qp
    launch._quantpack = True
    stock._launch_packed_attn = launch   # unified_attention_int4 looks it up in the module globals at call time
    segments = _patch_decode_segments(decode[2], logger)
    fa_prefill = _patch_long_prefill(stock, logger)
    logger.info("vllm-quantpack: int4_per_token_head KV attention fixed (exact zero point, prefill tiles %s, split-KV "
                "decode with MTP, tile %d x %s segments%s)", tiles, decode[0], segments or "stock",
                ", FlashAttention for long prefills" if fa_prefill else "")
    return True


def _patch_decode_segments(segments, logger):
    """Allocate the split-KV segment buffers of int4 caches for `segments` segments instead of vLLM's 16."""
    try:
        from vllm.v1.attention.backends import triton_attn
        builder = triton_attn.TritonAttentionMetadataBuilder
        stock_init = builder.__init__
        triton_attn.NUM_PAR_SOFTMAX_SEGMENTS   # the anchor this relies on
    except Exception as e:
        logger.info("vllm-quantpack: int4 split-KV segments left stock (%s)", e)
        return 0
    if getattr(stock_init, "_quantpack", False):
        return segments
    if segments < 1 or segments & (segments - 1):
        logger.warning("vllm-quantpack: int4 split-KV segments must be a power of two, got %d: left stock", segments)
        return 0

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.cache_config.cache_dtype != "int4_per_token_head":
            return stock_init(self, kv_cache_spec, layer_names, vllm_config, device)
        saved = triton_attn.NUM_PAR_SOFTMAX_SEGMENTS
        triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = segments   # read once, in __init__, to size the buffers
        try:
            stock_init(self, kv_cache_spec, layer_names, vllm_config, device)
        finally:
            triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = saved

    __init__._quantpack = True
    builder.__init__ = __init__
    return segments


def _step_lengths(cu_seqlens_q, seqused_k):
    """cu_seqlens_q and seqused_k as host lists, copied once per forward step (prefill steps run outside CUDA graphs).
    Every attention layer of a step gets the same tensors and each copy makes the CPU wait for the GPU, so the lists
    are kept on the step's forward context, keyed by the tensors themselves (other KV-cache groups get other ones)."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        step = get_forward_context() if is_forward_context_available() else None
    except Exception:
        step = None
    hit = getattr(step, "_quantpack_int4_lengths", None)
    if hit is not None and hit[0] is cu_seqlens_q and hit[1] is seqused_k:
        return hit[2], hit[3]
    cu, used = cu_seqlens_q.tolist(), seqused_k.tolist()
    if step is not None:
        try:
            step._quantpack_int4_lengths = (cu_seqlens_q, seqused_k, cu, used)
        except Exception:
            pass
    return cu, used


def _patch_long_prefill(stock, logger):
    """Route long-context prefill chunks to int4_prefill (FlashAttention over unpacked segments)."""
    if os.environ.get("VLLM_QUANTPACK_INT4_FA_PREFILL", "1") == "0":
        return False
    try:
        import torch
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        from . import int4_prefill
    except Exception as e:      # no FlashAttention build for this GPU / vLLM
        logger.info("vllm-quantpack: long int4 prefills stay on Triton (%s)", e)
        return False
    triton_path = stock.unified_attention_int4

    def unified_attention_int4(q, k_cache, v_cache, out, *, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                               softmax_scale, window_size, block_table, softcap, sinks, alibi_slopes, use_alibi_sqrt,
                               qq_bias, output_scale, mm_prefix_range, k_scale_cache, v_scale_cache, **kw):
        args = dict(cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q, seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale, window_size=window_size,
                    block_table=block_table, softcap=softcap, sinks=sinks, alibi_slopes=alibi_slopes,
                    use_alibi_sqrt=use_alibi_sqrt, qq_bias=qq_bias, output_scale=output_scale,
                    mm_prefix_range=mm_prefix_range, k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache, **kw)
        plain = (window_size[0] < 0 and not softcap and sinks is None and alibi_slopes is None and qq_bias is None
                 and output_scale is None and mm_prefix_range is None)
        if not plain or max_seqlen_q < int4_prefill.MIN_QUERY or max_seqlen_k < int4_prefill.MIN_CONTEXT:
            return triton_path(q, k_cache, v_cache, out, **args)
        cu, used = _step_lengths(cu_seqlens_q, seqused_k)
        long = [i for i in range(len(used)) if cu[i + 1] - cu[i] >= int4_prefill.MIN_QUERY
                and used[i] >= int4_prefill.MIN_CONTEXT]
        if not long:
            return triton_path(q, k_cache, v_cache, out, **args)
        rest = [i for i in range(len(used)) if i not in long and cu[i + 1] > cu[i]]
        if rest:                       # decode tokens and short prefills of the same step: Triton on a sub-batch
            rows = torch.cat([torch.arange(cu[i], cu[i + 1], device=q.device) for i in rest])
            lens = [cu[i + 1] - cu[i] for i in rest]
            sub_cu = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32, device=q.device)
            sel = torch.tensor(rest, device=q.device)
            sub_out = torch.empty((len(rows),) + tuple(out.shape[1:]), dtype=out.dtype, device=out.device)
            triton_path(q[rows], k_cache, v_cache, sub_out, **{**args, "cu_seqlens_q": sub_cu,
                        "max_seqlen_q": max(lens), "seqused_k": seqused_k[sel], "block_table": block_table[sel],
                        "max_seqlen_k": max(used[i] for i in rest)})
            out[rows] = sub_out
        head = q.shape[2]
        for i in long:
            q_rot = stock.single_rht(q[cu[i]:cu[i + 1]].float()).to(q.dtype)
            o = int4_prefill.prefill_sequence(q_rot, k_cache, v_cache, k_scale_cache, v_scale_cache, block_table[i],
                                              used[i], softmax_scale / head, flash_attn_varlen_func)
            out[cu[i]:cu[i + 1]] = (stock.single_rht(o.float(), inverse=True) / head).to(out.dtype)

    unified_attention_int4._quantpack = True
    stock.unified_attention_int4 = unified_attention_int4
    return True
