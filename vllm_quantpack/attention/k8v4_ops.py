"""Triton store kernel of the k8v4_per_token_head KV cache (see k8v4_kv.py). Imported on first use: needs Triton."""
from vllm.triton_utils import tl, triton


@triton.jit
def _round(x):
    """Round half away from zero, as vLLM's int4 store."""
    return tl.where(x >= 0, (x + 0.5).to(tl.int32), (x - 0.5).to(tl.int32)).to(tl.float32)


@triton.jit
def _store_k8v4(key_ptr, value_ptr, key_cache_ptr, value_cache_ptr, k_scale_cache_ptr, v_scale_cache_ptr,
                slot_mapping_ptr,
                stride_key_tok: tl.int64, stride_key_head: tl.int64,
                stride_val_tok: tl.int64, stride_val_head: tl.int64,
                stride_kc_blk: tl.int64, stride_kc_slot: tl.int64, stride_kc_head: tl.int64,
                stride_vc_blk: tl.int64, stride_vc_slot: tl.int64, stride_vc_head: tl.int64,
                stride_ks_blk: tl.int64, stride_ks_slot: tl.int64, stride_ks_head: tl.int64,
                stride_vs_blk: tl.int64, stride_vs_slot: tl.int64, stride_vs_head: tl.int64,
                block_size: tl.constexpr, head_size: tl.constexpr, head_size_v: tl.constexpr,
                HEAD_PADDED: tl.constexpr, HALF_PADDED: tl.constexpr):
    """One (token, KV head): key -> int8 with scale absmax / 127; value -> nibbles as vLLM's int4 store."""
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    blk = slot // block_size
    slot_in_blk = slot % block_size

    offs = tl.arange(0, HEAD_PADDED)
    k_mask = offs < head_size
    k = tl.load(key_ptr + tok * stride_key_tok + head * stride_key_head + offs, mask=k_mask,
                other=0.0).to(tl.float32)
    k_scale = tl.maximum(tl.max(tl.abs(k)) / 127.0, 1e-6)
    k_q = tl.clamp(_round(k / k_scale), -127.0, 127.0)
    tl.store(k_scale_cache_ptr + blk * stride_ks_blk + slot_in_blk * stride_ks_slot + head * stride_ks_head,
             k_scale)
    tl.store(key_cache_ptr + blk * stride_kc_blk + slot_in_blk * stride_kc_slot + head * stride_kc_head + offs,
             k_q.to(tl.int8), mask=k_mask)

    half = tl.arange(0, HALF_PADDED)
    even_mask = half * 2 < head_size_v
    odd_mask = half * 2 + 1 < head_size_v
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head
    v_even = tl.load(val_base + half * 2, mask=even_mask, other=0.0).to(tl.float32)
    v_odd = tl.load(val_base + half * 2 + 1, mask=odd_mask, other=0.0).to(tl.float32)
    v_min = tl.minimum(tl.min(tl.where(even_mask, v_even, float("inf"))),
                       tl.min(tl.where(odd_mask, v_odd, float("inf"))))
    v_max = tl.maximum(tl.max(tl.where(even_mask, v_even, float("-inf"))),
                       tl.max(tl.where(odd_mask, v_odd, float("-inf"))))
    v_scale = tl.maximum((v_max - v_min) / 15.0, 1e-6)
    v_zp = tl.clamp(_round(-v_min / v_scale), 0.0, 15.0)
    inv_v = 1.0 / v_scale
    even_q = tl.clamp(_round(v_even * inv_v + v_zp), 0.0, 15.0).to(tl.uint8)
    odd_q = tl.clamp(_round(v_odd * inv_v + v_zp), 0.0, 15.0).to(tl.uint8)
    # the 4-bit zero point rides in the low mantissa bits of the fp32 scale, as in the int4 cache
    v_scale_packed = ((v_scale.to(tl.int32, bitcast=True) & -16) | (v_zp.to(tl.int32) & 0xF)).to(
        tl.float32, bitcast=True)
    tl.store(v_scale_cache_ptr + blk * stride_vs_blk + slot_in_blk * stride_vs_slot + head * stride_vs_head,
             v_scale_packed)
    tl.store(value_cache_ptr + blk * stride_vc_blk + slot_in_blk * stride_vc_slot + head * stride_vc_head + half,
             (even_q & 0xF) | ((odd_q & 0xF) << 4), mask=half < head_size_v // 2)
