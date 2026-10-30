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


def unpack_segment(cache, scales, block_row, start, length, out):
    """Unpack cache positions [start, start + length) of one sequence into out[:length] (L, KV heads, head size)."""
    nkv, half = cache.shape[2], cache.shape[3] - 4 // cache.element_size()
    tokens = 16
    _unpack_segment[(triton.cdiv(length, tokens), nkv)](
        cache, scales, block_row, out, start, length,
        cache.stride(0), cache.stride(1), cache.stride(2), scales.stride(0), scales.stride(1), scales.stride(2),
        out.stride(0), out.stride(1), BLOCK_SIZE=cache.shape[1], HALF=half, TOKENS=tokens)


def _cu_seqlens(length, device):
    """[0, length] as int32 on the GPU, written by kernels: torch.tensor([0, length], device=...) copies from host
    memory and makes the CPU wait for every queued kernel, once per segment of every layer."""
    cu = torch.full((2,), length, dtype=torch.int32, device=device)     # fill kernels: no copy from the host
    cu[0].zero_()
    return cu


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
