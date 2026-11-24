"""Inkling2 KV cache at 4 bits on the portable (non-SM100) Triton attention: k8v4_per_token_head and int4_per_token_head.

vLLM's Inkling2 attention has its own kernels: the prologue (``ops/sconv.attention_prologue``) writes K and V into the
cache and ``ops/attention_sm90.rel_attention`` reads them for prefill, MTP verification and decode alike. They know
bf16 and unit-scale e4m3 only. This module builds a copy of those two modules from the installed (or overlaid) vLLM
source with checked text edits, so that a cache row per (token, KV head) can hold:

* ``k8v4_per_token_head``: key int8 (symmetric, one fp16 scale), value int4 (asymmetric, fp16 scale and minimum).
  Row ``[K D int8 | V D/2 bytes | k_scale, pad, v_scale, v_min (fp16)]`` = 1.5 D + 8 bytes (200 at D 128; fp8 256).
* ``int4_per_token_head``: key and value int4 (asymmetric each). Row ``[K D/2 | V D/2 | k_s, k_min, v_s, v_min]`` =
  D + 8 bytes (136 at D 128). The keys are stored rotated by R = diag(signs) H / sqrt(D) (randomized Hadamard,
  ``_kvq_rotate``) and the reader rotates the queries alike: q.k is unchanged and outlier channels are spread over
  the row before the 4-bit rounding. VLLM_QUANTPACK_INKLING2_KV_HADAMARD=0 stores them plain (same row).

Keys are L2-clipped (norm <= qk_l2_clip_threshold) and values RMS-normalized before the write, so per-row scales
fit them well. Each row is quantized from the fp32 values the prologue already holds; the reader dequantizes a key
or value tile to the query dtype before the MMA. The format is chosen from the cache row width the kernels receive
(``kvq_mode``), so every other caller (the DSpark draft's own bf16 cache, fp8 caches) runs the original code path.

Python-side patches of ``nvidia/attention.py``: the layer spec gets ``head_size = row / 2`` uint8 (vLLM sizes pages
as ``(head_size + head_size_v) * dtype``), the layers accept the two dtypes, ``_split_kv_cache`` returns the raw rows,
and the module names that call the kernels point at the copies. If an anchor is missing nothing is patched and
choosing these dtypes fails at model build with the reason. VLLM_QUANTPACK_INKLING2_KVQ=0 disables it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
import threading

import torch

log = logging.getLogger(__name__)

DTYPES = {"k8v4_per_token_head": 1, "int4_per_token_head": 2}
_OPS = "vllm.models.inkling2.nvidia.ops"
_lock = threading.Lock()
_state: dict = {}


def row_bytes(mode: int, d: int) -> int:
    return d + d // 2 + 8 if mode == 1 else d + 8


# int4 keys rotated by the randomized Hadamard (1, default) or stored plain (0); both use the same row
HADAMARD = os.environ.get("VLLM_QUANTPACK_INKLING2_KV_HADAMARD", "1") != "0"


def native_e4m3() -> bool:
    """Whether Triton converts e4m3 in hardware here (SM89+); below, the copies' fast software reads apply."""
    try:
        from vllm.platforms import current_platform
        return current_platform.has_device_capability(89)
    except Exception:
        return True


def kvq_mode(row: int, d: int) -> int:
    """Kernel format of a cache whose (token, head) row has `row` elements, for head size `d`: 0 bf16 / fp8,
    1 k8v4, 2 int4, 3 int4 with rotated keys."""
    if row == d:
        return 0
    if row == row_bytes(1, d):
        return 1
    if row == row_bytes(2, d):
        return 3 if HADAMARD else 2
    raise ValueError(f"Inkling2 KV cache row of {row} elements does not match head size {d}")


# --- attention_sm90 edits ---------------------------------------------------------------------------------------------

_READ_HELPERS = r'''
from vllm_quantpack.families.inkling2.kvq import kvq_mode, native_e4m3

@triton.jit
def _kvq_fwht_stage(x, ROWS: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    y = tl.permute(tl.reshape(x, [ROWS, D // (2 * S), 2, S]), (0, 1, 3, 2))
    a, b = tl.split(y)
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), [ROWS, D])


@triton.jit
def _kvq_rotate(x, ROWS: tl.constexpr, D: tl.constexpr):
    """x [ROWS, D] fp32 times the orthogonal R = diag(signs) H / sqrt(D) (H Sylvester Hadamard, signs a fixed hash
    of the channel): keys are stored rotated and queries rotated alike, so q.k is unchanged."""
    d = tl.arange(0, D).to(tl.uint32)
    sign = 1.0 - 2.0 * (((d * 2654435761) >> 15) & 1).to(tl.float32)
    x = x * sign[None, :]
    for i in tl.static_range(10):
        if (1 << i) < D:
            x = _kvq_fwht_stage(x, ROWS, D, 1 << i)
    return x * (1.0 / tl.sqrt(D * 1.0))


@triton.jit
def _kvq_int8(ptr, offsets, mask, D: tl.constexpr, S_OFF: tl.constexpr):
    """[N, D] fp32 keys of int8 rows at byte `offsets` (fp16 scale at S_OFF)."""
    d = tl.arange(0, D)
    x = tl.load(ptr + offsets[:, None] + d[None, :], mask=mask[:, None], other=0)
    s = tl.load((ptr + offsets + S_OFF).to(tl.pointer_type(tl.float16)), mask=mask, other=0.0)
    return x.to(tl.int8, bitcast=True).to(tl.float32) * s.to(tl.float32)[:, None]


@triton.jit
def _kvq_int4(ptr, offsets, mask, D: tl.constexpr, S_OFF: tl.constexpr):
    """[N, D] fp32 values of nibble rows at byte `offsets` (fp16 scale and minimum at S_OFF)."""
    half = tl.arange(0, D // 2)
    b = tl.load(ptr + offsets[:, None] + half[None, :], mask=mask[:, None], other=0)
    x = tl.interleave((b & 15).to(tl.float32), (b >> 4).to(tl.float32))
    sp = (ptr + offsets + S_OFF).to(tl.pointer_type(tl.float16))
    s = tl.load(sp, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(sp + 1, mask=mask, other=0.0).to(tl.float32)
    return x * s[:, None] + z[:, None]


'''

_ATT_EDITS = [
    # Below SM89 the e4m3 bytes were converted by a chain of selects per element (ops/e4m3._e4m3_f16), which made
    # the prefill attention ~28x slower than bf16 on an RTX 3090. Placing the e4m3 exponent and mantissa bits in an
    # fp16 and multiplying by 2**8 (the bias difference) gives the same value for every finite code, subnormals
    # included (the writer saturates, so the NaN code 0x7F is never stored).
    ("fast e4m3", "        lo = _e4m3_f16(pair & 0xFF).to(tl.float16, bitcast=True)\n"
                  "        hi = _e4m3_f16(pair >> 8).to(tl.float16, bitcast=True)\n",
     "        lo = ((((pair & 0x7F) << 7) | ((pair & 0x80) << 8)).to(tl.float16, bitcast=True) * 256.0)\n"
     "        hi = (((((pair >> 8) & 0x7F) << 7) | (((pair >> 8) & 0x80) << 8)).to(tl.float16, bitcast=True)\n"
     "              * 256.0)\n"),
    ("helpers", "@triton.jit\ndef _rel_attention_kernel(\n", _READ_HELPERS + "@triton.jit\ndef _rel_attention_kernel(\n"),
    ("kernel KVQ", "    BLOCK_N: tl.constexpr,\n):\n    TOKENS: tl.constexpr = BLOCK_M // G\n",
     "    BLOCK_N: tl.constexpr,\n    KVQ: tl.constexpr,\n):\n    TOKENS: tl.constexpr = BLOCK_M // G\n"),
    ("key read", "        if K.dtype.element_ty.primitive_bitwidth == 8:\n            k = _load_fp8_kv(",
     "        if KVQ == 1:\n"
     "            k = _kvq_int8(K, kv_ptrs, key_valid, D, D + D // 2).to(q.dtype).T\n"
     "        elif KVQ >= 2:\n"
     "            k = _kvq_int4(K, kv_ptrs, key_valid, D, D).to(q.dtype).T\n"
     "        elif K.dtype.element_ty.primitive_bitwidth == 8:\n            k = _load_fp8_kv("),
    ("value read", "        if V.dtype.element_ty.primitive_bitwidth == 8:\n            v = _load_fp8_kv(",
     "        if KVQ == 1:\n"
     "            v = _kvq_int4(K + D, kv_ptrs, key_valid, D, D // 2 + 4).to(q.dtype)\n"
     "        elif KVQ >= 2:\n"
     "            v = _kvq_int4(K + D // 2, kv_ptrs, key_valid, D, D // 2 + 4).to(q.dtype)\n"
     "        elif V.dtype.element_ty.primitive_bitwidth == 8:\n            v = _load_fp8_kv("),
    ("query rotation", "    if not DRAFT:\n        tile_end",
     "    if KVQ == 3:\n"
     "        q = _kvq_rotate(q.to(tl.float32), BLOCK_M, D).to(Q.dtype.element_ty)\n"
     "    if not DRAFT:\n        tile_end"),
    ("launcher mode", "    page, num_kv_heads = key_cache.shape[1:3]\n",
     "    page, num_kv_heads = key_cache.shape[1:3]\n    kvq = kvq_mode(key_cache.shape[3], head_dim)\n"),
    # Prefill tiles (block_m 128) per cache format, measured on an RTX 3090 (port/bench_rel_attn.py, 8192 queries,
    # 16k context): fp8 with software reads 64x32/4 warps (29.6 ms vs 32.5), k8v4 64x64/4 (34.8 vs 65.8); bf16 and
    # int4 keep 128x64/8. Decode tiles are unchanged.
    ("prefill tiles", "    block_m = _block_m(max_query_len, group)\n",
     "    block_m = _block_m(max_query_len, group)\n"
     "    block_n = 64 if block_m >= 128 else 128 if block_m <= 32 else 64\n"
     "    warps = 8 if block_m >= 128 else 4\n"
     "    if block_m == 128 and kvq == 1:\n"
     "        block_m, block_n, warps = 64, 64, 4\n"
     "    elif block_m == 128 and kvq == 0 and key_cache.element_size() == 1 and not native_e4m3():\n"
     "        block_m, block_n, warps = 64, 32, 4\n"),
    ("launch KVQ", "        BLOCK_N=64 if block_m >= 128 else 128 if block_m <= 32 else 64,\n"
                   "        num_warps=8 if block_m >= 128 else 4,\n        num_stages=2,\n",
     "        BLOCK_N=block_n,\n        KVQ=kvq,\n        num_warps=warps,\n        num_stages=2,\n"),
]

# --- sconv edits ------------------------------------------------------------------------------------------------------

_STORE_HELPERS = r'''
from vllm_quantpack.families.inkling2.kvq import kvq_mode

@triton.jit
def _kvq_fwht_stage(x, ROWS: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    y = tl.permute(tl.reshape(x, [ROWS, D // (2 * S), 2, S]), (0, 1, 3, 2))
    a, b = tl.split(y)
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), [ROWS, D])


@triton.jit
def _kvq_rotate(x, ROWS: tl.constexpr, D: tl.constexpr):
    """x [ROWS, D] fp32 times the orthogonal R = diag(signs) H / sqrt(D) (H Sylvester Hadamard, signs a fixed hash
    of the channel): keys are stored rotated and queries rotated alike, so q.k is unchanged."""
    d = tl.arange(0, D).to(tl.uint32)
    sign = 1.0 - 2.0 * (((d * 2654435761) >> 15) & 1).to(tl.float32)
    x = x * sign[None, :]
    for i in tl.static_range(10):
        if (1 << i) < D:
            x = _kvq_fwht_stage(x, ROWS, D, 1 << i)
    return x * (1.0 / tl.sqrt(D * 1.0))


@triton.jit
def _kvq_store4(base, y, m, S_OFF: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    """Rows of y [BT, D] as nibbles at `base` [BT] with fp16 scale and minimum at S_OFF."""
    lo = tl.min(y, 1)
    hi = tl.max(y, 1)
    s = tl.where(hi > lo, (hi - lo) / 15.0, 1.0).to(tl.float16).to(tl.float32)
    z = lo.to(tl.float16).to(tl.float32)
    n = tl.clamp(tl.floor((y - z[:, None]) / s[:, None] + 0.5), 0.0, 15.0).to(tl.uint8)
    a, b = tl.split(tl.reshape(n, [BT, D // 2, 2]))
    half = tl.arange(0, D // 2)
    tl.store(base[:, None] + half[None, :], a | (b << 4), mask=m[:, None])
    sp = (base + S_OFF).to(tl.pointer_type(tl.float16))
    tl.store(sp, s.to(tl.float16), mask=m)
    tl.store(sp + 1, z.to(tl.float16), mask=m)


@triton.jit
def _kvq_store(row, y, m, is_v, KVQ: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    """Quantize the K (is_v False) or V rows y [BT, D] of one head into packed cache rows at `row` [BT]."""
    if KVQ == 1:
        if is_v:
            _kvq_store4(row + D, y, m, D // 2 + 4, D, BT)
        else:
            amax = tl.max(tl.abs(y), 1)
            s = tl.where(amax > 0, amax / 127.0, 1.0).to(tl.float16).to(tl.float32)
            qv = tl.clamp(tl.floor(y / s[:, None] + 0.5), -127.0, 127.0).to(tl.int8)
            d = tl.arange(0, D)
            tl.store(row[:, None] + d[None, :], qv.to(tl.uint8, bitcast=True), mask=m[:, None])
            tl.store((row + D + D // 2).to(tl.pointer_type(tl.float16)), s.to(tl.float16), mask=m)
    else:
        if is_v:
            _kvq_store4(row + D // 2, y, m, D // 2 + 4, D, BT)
        else:
            if KVQ == 3:
                y = _kvq_rotate(y, BT, D)
            _kvq_store4(row, y, m, D, D, BT)


'''

_CONV_EDITS = [
    ("helpers", "\n@triton.jit\ndef _attn_prologue_kernel(\n", "\n" + _STORE_HELPERS + "@triton.jit\ndef _attn_prologue_kernel(\n"),
    ("prologue kernel KVQ", "    FUSE_QR: tl.constexpr,\n):\n", "    FUSE_QR: tl.constexpr,\n    KVQ: tl.constexpr,\n):\n"),
    ("tile call KVQ", "            W,\n            D,\n            DYNAMIC,\n            BT,\n        )\n",
     "            W,\n            D,\n            DYNAMIC,\n            BT,\n            KVQ,\n        )\n"),
    ("tile KVQ", "    DYNAMIC: tl.constexpr,\n    BT: tl.constexpr,\n):\n    toff = tl.program_id(0) * BT",
     "    DYNAMIC: tl.constexpr,\n    BT: tl.constexpr,\n    KVQ: tl.constexpr,\n):\n    toff = tl.program_id(0) * BT"),
    ("tile store", "    tl.store(dst, cast_store(y, cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])\n",
     "    if KVQ == 0:\n"
     "        tl.store(dst, cast_store(y, cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])\n"
     "    else:\n"
     "        row = (key_cache_ptr + (safe // kv_block_size) * stride_kv_blk\n"
     "               + (safe % kv_block_size) * stride_kv_n + head * stride_kv_h)\n"
     "        _kvq_store(row, y, kv_slot >= 0, is_v, KVQ, D, BT)\n"),
    ("launcher mode", "    num_heads, head_dim = key_cache.shape[2:]\n",
     "    num_heads, head_dim = key_cache.shape[2], q.shape[2]\n    kvq = kvq_mode(key_cache.shape[3], head_dim)\n"),
    ("launch KVQ", "        FUSE_QR=not qr_tokens,\n", "        FUSE_QR=not qr_tokens,\n        KVQ=kvq,\n"),
]


def _edit(src: str, edits, what: str) -> str:
    for name, old, new in edits:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"{what}: anchor '{name}' found {n} times")
        src = src.replace(old, new)
    return src


def _load_copy(mod_name: str, edits) -> object:
    orig = sys.modules.get(f"{_OPS}.{mod_name}") or importlib.import_module(f"{_OPS}.{mod_name}")
    src = _edit(open(orig.__file__).read(), edits, mod_name)
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    root = os.path.join(os.environ.get("VLLM_QUANTPACK_CACHE", os.path.expanduser("~/.cache/vllm_quantpack")),
                        "inkling2_kvq")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{mod_name}_{tag}.py")
    if not os.path.exists(path):          # Triton needs the kernels' source in a real file
        tmp = f"{path}.{os.getpid()}"
        open(tmp, "w").write(src)
        os.replace(tmp, path)
    name = f"{_OPS}._qp_kvq_{mod_name}"   # inside the ops package, so its relative imports resolve
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install() -> str | None:
    """Patch Inkling2's attention for the packed caches; None on success, else the reason it is unavailable."""
    with _lock:
        if "result" in _state:
            return _state["result"]
        reason = None
        try:
            if os.environ.get("VLLM_QUANTPACK_INKLING2_KVQ", "1") == "0":
                raise RuntimeError("disabled by VLLM_QUANTPACK_INKLING2_KVQ=0")
            _install()
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            log.warning("vllm-quantpack: Inkling2 4-bit KV cache unavailable (%s)", reason)
        _state["result"] = reason
        return reason


def _install() -> None:
    import importlib as _il
    att = _il.import_module("vllm.models.inkling2.nvidia.attention")
    ops_att = _il.import_module(f"{_OPS}.attention")
    from vllm.v1.kv_cache_interface import AttentionSpec

    sm90 = _load_copy("attention_sm90", _ATT_EDITS)
    sconv = _load_copy("sconv", _CONV_EDITS)

    stock_layer_config = att._kv_cache_layer_config
    stock_init = att.Inkling2Attention.__init__
    stock_split = att.Inkling2Attention._split_kv_cache

    def _kv_cache_layer_config(vllm_config, num_kv_heads, head_dim, sliding_window):
        mode = DTYPES.get(vllm_config.cache_config.cache_dtype)
        if not mode:
            return stock_layer_config(vllm_config, num_kv_heads, head_dim, sliding_window)
        spec = AttentionSpec(block_size=vllm_config.cache_config.block_size, num_kv_heads=num_kv_heads,
                             head_size=row_bytes(mode, head_dim) // 2, dtype=torch.uint8)
        if sliding_window is not None:
            return att.SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
        return att.FullAttentionManager.layer_config(spec)

    def __init__(self, *args, **kwargs):
        from vllm.config import get_current_vllm_config
        cc = get_current_vllm_config().cache_config
        mode = DTYPES.get(cc.cache_dtype)
        if not mode:
            stock_init(self, *args, **kwargs)
            self._qp_kvq = 0
            return
        if att.use_sm100_kernels():
            raise ValueError(f"vllm-quantpack: Inkling2 {cc.cache_dtype} KV runs on the Triton attention only "
                             "(not the SM100 FA4 kernels)")
        dtype = cc.cache_dtype
        cc.cache_dtype = "bfloat16"      # the stock dtype check; the spec and kernels follow the real dtype
        try:
            stock_init(self, *args, **kwargs)
        finally:
            cc.cache_dtype = dtype
        self.fp8_kv = False
        self._qp_kvq = mode

    def _split_kv_cache(self, kv_cache):
        if not getattr(self, "_qp_kvq", 0):
            return stock_split(self, kv_cache)
        rows = att.canonicalize_singleton_dim_strides(kv_cache.view(torch.uint8).transpose(1, 2))
        assert rows.shape[-1] == row_bytes(self._qp_kvq, self.head_dim), rows.shape
        return rows, rows

    att._kv_cache_layer_config = _kv_cache_layer_config
    att.Inkling2Attention.__init__ = __init__
    att.Inkling2Attention._split_kv_cache = _split_kv_cache
    att.rel_attention = sm90.rel_attention
    att.attention_prologue = sconv.attention_prologue
    ops_att.rel_attention = sm90.rel_attention
    log.info("vllm-quantpack: Inkling2 KV caches %s enabled (Triton attention; int4 keys %s)", ", ".join(DTYPES),
             "Hadamard-rotated" if HADAMARD else "plain")
