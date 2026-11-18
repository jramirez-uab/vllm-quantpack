"""Long-context prefill over the int4_per_token_head KV cache with FlashAttention.

vLLM serves the int4 KV cache only on its Triton backend, whose prefill kernel runs well below FlashAttention even
after the fixes in int4_kv.py (A40, 229k context: 2.56 s per 1024-token chunk against ~1 s for FlashInfer on an fp8
cache). For a request with a long context this module walks the cache in segments of SEGMENT tokens: a small Triton
kernel unpacks one segment to bf16 (in the RHT domain the cache is stored in), FlashAttention 2 attends the chunk's
queries to it and returns the log-sum-exp, and the segments are folded together by log-sum-exp in fp32 (plain torch:
vLLM's merge_attn_states compiles for ~22 s on the first long request of every start). Only the newest segment is
causal; it always holds at least the chunk's own keys. One segment of scratch (~64 MiB at 16k tokens,
4 KV heads, head 256) is reused by every layer.

Decode steps, short contexts and features this path does not cover (sliding window, ALiBi, sinks, softcap, FP8
output...) stay on the Triton kernel.
"""
import torch

from vllm.triton_utils import tl, triton

MAX_HEAD = 256       # largest head size FlashAttention's varlen forward accepts
SEGMENT = 16384      # tokens per segment: 16k x 4 KV heads x 256 x (K, V) x bf16 = 64 MiB of scratch
MIN_QUERY = 64       # a chunk shorter than this is cheap on the Triton kernel
MIN_CONTEXT = 8192   # below this the Triton kernel is already fast


@triton.jit
def _unpack_segment(cache_ptr, scale_ptr, block_table_ptr, out_ptr, start, length,
                    stride_c0, stride_c1, stride_c2, stride_s0, stride_s1, stride_s2,
                    stride_o0, stride_o1,
                    BLOCK_SIZE: tl.constexpr, HALF: tl.constexpr, TOKENS: tl.constexpr):
    """out[t, h, :] = (nibbles - zero_point) * scale for cache positions start + t; low nibble = even channel."""
    t = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    h = tl.program_id(1)
    ok = t < length
    pos = start + t
    blk = tl.load(block_table_ptr + pos // BLOCK_SIZE, mask=ok, other=0).to(tl.int64)
    slot = pos % BLOCK_SIZE
    c = tl.arange(0, HALF)
    packed = tl.load(cache_ptr + blk[:, None] * stride_c0 + slot[:, None] * stride_c1 + h * stride_c2 + c[None, :],
                     mask=ok[:, None], other=0)
    bits = tl.load(scale_ptr + blk * stride_s0 + slot * stride_s1 + h * stride_s2, mask=ok, other=0).to(tl.int32,
                                                                                                       bitcast=True)
    zp = (bits & 0xF).to(tl.float32)                   # the zero point hides in the scale's low mantissa bits
    scale = (bits & -16).to(tl.float32, bitcast=True)
    lo = ((packed & 0xF).to(tl.float32) - zp[:, None]) * scale[:, None]
    hi = (((packed >> 4) & 0xF).to(tl.float32) - zp[:, None]) * scale[:, None]
    vals = tl.interleave(lo, hi).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t[:, None].to(tl.int64) * stride_o0 + h * stride_o1 + tl.arange(0, 2 * HALF)[None, :], vals,
             mask=ok[:, None])


@triton.jit
def _unpack_segment_i8(cache_ptr, scale_ptr, block_table_ptr, out_ptr, start, length,
                       stride_c0, stride_c1, stride_c2, stride_s0, stride_s1, stride_s2,
                       stride_o0, stride_o1,
                       BLOCK_SIZE: tl.constexpr, HEAD: tl.constexpr, TOKENS: tl.constexpr):
    """out[t, h, :] = int8 * scale for cache positions start + t (the keys of a k8v4_per_token_head cache)."""
    t = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    h = tl.program_id(1)
    ok = t < length
    pos = start + t
    blk = tl.load(block_table_ptr + pos // BLOCK_SIZE, mask=ok, other=0).to(tl.int64)
    slot = pos % BLOCK_SIZE
    c = tl.arange(0, HEAD)
    codes = tl.load(cache_ptr + blk[:, None] * stride_c0 + slot[:, None] * stride_c1 + h * stride_c2 + c[None, :],
                    mask=ok[:, None], other=0)
    scale = tl.load(scale_ptr + blk * stride_s0 + slot * stride_s1 + h * stride_s2, mask=ok, other=0.0)
    vals = (codes.to(tl.float32) * scale[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + t[:, None].to(tl.int64) * stride_o0 + h * stride_o1 + c[None, :], vals, mask=ok[:, None])


def unpack_segment(cache, scales, block_row, start, length, out):
    """Unpack cache positions [start, start + length) of one sequence into out[:length] (L, KV heads, head size).
    An int8 cache view holds one code per channel (k8v4 keys); a uint8 one, two nibbles per byte."""
    nkv, head = cache.shape[2], out.shape[2]
    tokens = 16
    strides = (cache.stride(0), cache.stride(1), cache.stride(2), scales.stride(0), scales.stride(1),
               scales.stride(2), out.stride(0), out.stride(1))
    grid = (triton.cdiv(length, tokens), nkv)
    if cache.dtype == torch.int8:
        _unpack_segment_i8[grid](cache, scales, block_row, out, start, length, *strides,
                                 BLOCK_SIZE=cache.shape[1], HEAD=head, TOKENS=tokens)
    else:
        _unpack_segment[grid](cache, scales, block_row, out, start, length, *strides,
                              BLOCK_SIZE=cache.shape[1], HALF=head // 2, TOKENS=tokens)


def _cu_seqlens(length, device):
    """[0, length] as int32 on the GPU, written by kernels: torch.tensor([0, length], device=...) copies from host
    memory and makes the CPU wait for every queued kernel, once per segment of every layer."""
    cu = torch.full((2,), length, dtype=torch.int32, device=device)     # fill kernels: no copy from the host
    cu[0].zero_()
    return cu


@triton.jit
def _attn_i8(q_even, q_odd, q_even_lo, q_odd_lo, q_scale, q_sum, k_cache, v_cache, k_scales, v_scales, block_row, out,
             qlen, n, heads_per_kv,
             stride_q0, stride_q1, stride_qs0,
             stride_kc0, stride_kc1, stride_kc2, stride_vc0, stride_vc1, stride_vc2,
             stride_ks0, stride_ks1, stride_ks2, stride_vs0, stride_vs1, stride_vs2,
             stride_o0, stride_o1,
             BLOCK_SIZE: tl.constexpr, HALF: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             Q_LO: tl.constexpr):
    """One query tile of one head over the whole cached context, reading the int4 cache as stored.

    Q arrives pre-scaled to log2 units, split into even/odd channels and quantized per (token, head) as two int8
    parts, q / q_scale ~ hi + lo / Q_LO (one int8 alone costs ~1.3e-2 relative error in the output), so Q.K^T runs
    as INT8 dots on the raw nibbles (low nibble = even channel) combined exactly in int32; the K zero point comes out
    as q_sum * zero_point and the K scale multiplies the int32 result. P.V runs in FP16 with FP16 accumulation inside a
    tile (twice the FP32-accumulate rate on GeForce Ampere): the V scale is folded into P and the V nibbles minus
    their zero point are small integers, exact in FP16. Tiles are added into FP32 accumulators."""
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // heads_per_kv
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < qlen
    c = tl.arange(0, HALF)
    q_off = rows[:, None].to(tl.int64) * stride_q0 + h * stride_q1 + c[None, :]
    qe = tl.load(q_even + q_off, mask=row_ok[:, None], other=0)
    qo = tl.load(q_odd + q_off, mask=row_ok[:, None], other=0)
    qe_lo = tl.load(q_even_lo + q_off, mask=row_ok[:, None], other=0)
    qo_lo = tl.load(q_odd_lo + q_off, mask=row_ok[:, None], other=0)
    qs = tl.load(q_scale + rows * stride_qs0 + h, mask=row_ok, other=0.0)
    qsum = tl.load(q_sum + rows * stride_qs0 + h, mask=row_ok, other=0)
    pos = n - qlen + rows                                    # absolute position of each query row
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc_e = tl.zeros((BLOCK_M, HALF), tl.float32)
    acc_o = tl.zeros((BLOCK_M, HALF), tl.float32)
    first = n - qlen + pid_m * BLOCK_M                      # keys below this need no causal mask for this tile
    end = tl.minimum(n, n - qlen + (pid_m + 1) * BLOCK_M)
    for start in range(0, end, BLOCK_N):
        keys = start + tl.arange(0, BLOCK_N)
        key_ok = keys < n
        blk = tl.load(block_row + keys // BLOCK_SIZE, mask=key_ok, other=0).to(tl.int64)
        slot = keys % BLOCK_SIZE
        kp = tl.load(k_cache + blk[:, None] * stride_kc0 + slot[:, None] * stride_kc1 + kvh * stride_kc2 + c[None, :],
                     mask=key_ok[:, None], other=0)
        kb = tl.load(k_scales + blk * stride_ks0 + slot * stride_ks1 + kvh * stride_ks2, mask=key_ok,
                     other=0).to(tl.int32, bitcast=True)
        k_lo = tl.trans((kp & 0xF).to(tl.int8))
        k_hi = tl.trans((kp >> 4).to(tl.int8))
        s = (tl.dot(qe, k_lo) + tl.dot(qo, k_hi)) * Q_LO + tl.dot(qe_lo, k_lo) + tl.dot(qo_lo, k_hi)
        s = (s - qsum[:, None] * (kb & 0xF)[None, :]).to(tl.float32)
        s = s * qs[:, None] * (kb & -16).to(tl.float32, bitcast=True)[None, :]
        allowed = key_ok[None, :] & (pos[:, None] >= 0)  # full (BLOCK_M, BLOCK_N): Triton keeps if-branch shapes
        if start + BLOCK_N > first:
            allowed = allowed & (keys[None, :] <= pos[:, None])
        s = tl.where(allowed, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
        vp = tl.load(v_cache + blk[:, None] * stride_vc0 + slot[:, None] * stride_vc1 + kvh * stride_vc2 + c[None, :],
                     mask=key_ok[:, None], other=0)
        vb = tl.load(v_scales + blk * stride_vs0 + slot * stride_vs1 + kvh * stride_vs2, mask=key_ok,
                     other=0).to(tl.int32, bitcast=True)
        vz = (vb & 0xF).to(tl.float16)
        pv = (p * (vb & -16).to(tl.float32, bitcast=True)[None, :]).to(tl.float16)
        v_lo = (vp & 0xF).to(tl.float16) - vz[:, None]
        v_hi = (vp >> 4).to(tl.float16) - vz[:, None]
        acc_e = acc_e * alpha[:, None] + tl.dot(pv, v_lo, out_dtype=tl.float16).to(tl.float32)
        acc_o = acc_o * alpha[:, None] + tl.dot(pv, v_hi, out_dtype=tl.float16).to(tl.float32)
    inv = 1.0 / tl.where(l_i == 0.0, 1.0, l_i)
    o_off = rows[:, None].to(tl.int64) * stride_o0 + h * stride_o1 + 2 * c[None, :]
    tl.store(out + o_off, (acc_e * inv[:, None]).to(out.dtype.element_ty), mask=row_ok[:, None])
    tl.store(out + o_off + 1, (acc_o * inv[:, None]).to(out.dtype.element_ty), mask=row_ok[:, None])


LOG2E = 1.4426950408889634
Q_LO = 254           # weight of the high int8 part of Q: hi * 254 + lo fits int32 (|dot| <= 128 * 127 * 15)


def prefill_sequence_i8(q_rot, k_cache, v_cache, k_scales, v_scales, block_row, n, softmax_scale,
                        block_m=64, block_n=64, num_warps=8, num_stages=2):
    """Same contract as prefill_sequence, in one kernel over the int4 cache (see _attn_i8)."""
    qlen, nq, hs = q_rot.shape
    nkv, half = k_cache.shape[2], hs // 2
    q = q_rot.float() * (softmax_scale * LOG2E)
    q_scale = (q.abs().amax(-1) / 127.0).clamp_min(1e-20)                   # per (token, head)
    qn = q / q_scale[..., None]
    hi = torch.round(qn)
    lo = torch.round((qn - hi) * Q_LO).clamp(-127, 127)                     # |qn - hi| <= 0.5: |lo| <= 127
    q_sum = (hi.sum(-1) * Q_LO + lo.sum(-1)).to(torch.int32)
    hi, lo = hi.to(torch.int8), lo.to(torch.int8)
    q_even, q_odd = hi[..., 0::2].contiguous(), hi[..., 1::2].contiguous()
    q_even_lo, q_odd_lo = lo[..., 0::2].contiguous(), lo[..., 1::2].contiguous()
    out = torch.empty_like(q_rot)
    _attn_i8[(triton.cdiv(qlen, block_m), nq)](
        q_even, q_odd, q_even_lo, q_odd_lo, q_scale / Q_LO, q_sum, k_cache, v_cache, k_scales, v_scales, block_row, out,
        qlen, n, nq // nkv,
        q_even.stride(0), q_even.stride(1), q_scale.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), v_cache.stride(0), v_cache.stride(1),
        v_cache.stride(2), k_scales.stride(0), k_scales.stride(1), k_scales.stride(2), v_scales.stride(0),
        v_scales.stride(1), v_scales.stride(2), out.stride(0), out.stride(1),
        BLOCK_SIZE=k_cache.shape[1], HALF=half, BLOCK_M=block_m, BLOCK_N=block_n, Q_LO=Q_LO,
        num_warps=num_warps, num_stages=num_stages)
    return out


def prefill_sequence(q_rot, k_cache, v_cache, k_scales, v_scales, block_row, n, softmax_scale, fa):
    """Attention of q_rot (the chunk's queries, already RHT-rotated, the last len(q_rot) positions of a context of
    n tokens) over the whole cached context. Returns the output in the RHT domain, in q_rot's dtype."""
    qlen, nq, hs = q_rot.shape
    nkv = k_cache.shape[2]
    seg = min(SEGMENT, n)
    kbuf = torch.empty((seg, nkv, hs), dtype=q_rot.dtype, device=q_rot.device)
    vbuf = torch.empty_like(kbuf)
    cu_q = _cu_seqlens(qlen, q_rot.device)
    cu_full = _cu_seqlens(seg, q_rot.device)      # every segment but the oldest is SEGMENT tokens long
    acc = acc_lse = None               # fp32 output (tokens, heads, head size) and log-sum-exp (heads, tokens)
    end = n
    while end > 0:                     # newest segment first: it is the causal one and holds the chunk's own keys
        start = max(0, end - SEGMENT)
        length = end - start
        unpack_segment(k_cache, k_scales, block_row, start, length, kbuf)
        unpack_segment(v_cache, v_scales, block_row, start, length, vbuf)
        o, lse = fa(q_rot, kbuf[:length], vbuf[:length], max_seqlen_q=qlen, cu_seqlens_q=cu_q, max_seqlen_k=length,
                    cu_seqlens_k=cu_full if length == seg else _cu_seqlens(length, q_rot.device),
                    softmax_scale=softmax_scale, causal=end == n, return_softmax_lse=True, fa_version=2)
        lse = lse.float()
        if acc is None:
            acc, acc_lse = o.float(), lse
        else:
            new_lse = torch.logaddexp(acc_lse, lse)
            acc.mul_(torch.exp(acc_lse - new_lse).T[..., None])   # in place: no extra fp32 copy of the output
            acc.addcmul_(o, torch.exp(lse - new_lse).T[..., None])
            acc_lse = new_lse
        end = start
    return acc.to(q_rot.dtype)

