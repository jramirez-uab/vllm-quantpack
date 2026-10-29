"""Time the int4_per_token_head KV attention on this GPU, to pick VLLM_QUANTPACK_INT4_DECODE / _TILES.

Fills a random int4 cache through vLLM's own write kernel and times, per attention layer:
  decode   one request with 1 + MTP query tokens: vLLM's stock kernel and each split-KV decode config;
  prefill  one 1024-query chunk at the end of the context: the Triton kernel with the prefill tiles and the
           FlashAttention segment path (int4_prefill).
Each line also prints the relative error against an fp64 reference computed from the same cache, so a config that
changes the result shows up. The model shape defaults to Qwen3.8-27B's full-attention layers.

usage: python -m vllm_quantpack.tools.int4_kv_bench [--contexts 32768,131072,262144] [--decode 64,4,16:64,4,32:...]
           [--prefill-tiles 64,64,8,2] [--q-heads 24] [--kv-heads 4] [--head-size 256] [--block-size 3072] [--mtp 1]
"""
import argparse
import os
import time

import torch

from ..attention import int4_kv


def _cache(stock, n, nkv, hs, bs):
    nb = (n + bs - 1) // bs
    phs = hs // 2 + 4                        # packed half head, then one float32 scale per (slot, head)
    kv = torch.zeros((nb, nkv, bs, 2 * phs), dtype=torch.uint8, device="cuda")
    base = torch.tensor([], dtype=torch.float32, device="cuda").set_(kv.untyped_storage())
    st = [s // 4 for s in kv.stride()]
    ks = torch.as_strided(base, (nb, bs, nkv), (st[0], st[2], st[1]), hs // 2 // 4)
    vs = torch.as_strided(base, (nb, bs, nkv), (st[0], st[2], st[1]), (phs + hs // 2) // 4)
    kc, vc = kv.transpose(1, 2).split(phs, dim=-1)
    g = torch.Generator(device="cuda").manual_seed(0)
    for s in range(0, n, 16384):
        t = min(16384, n - s)
        k = torch.randn((t, nkv, hs), generator=g, device="cuda", dtype=torch.bfloat16)
        v = torch.randn((t, nkv, hs), generator=g, device="cuda", dtype=torch.bfloat16)
        stock.reshape_and_cache_int4(k, v, kc, vc, torch.arange(s, s + t, device="cuda"), k_scale_cache=ks,
                                     v_scale_cache=vs)
    return (kc, vc, ks, vs), nb


def _reference(stock, q, cache, n, rows):
    """fp64 attention for query rows `rows` of q (the last len(q) positions of a context of n tokens)."""
    kc, vc, ks, vs = cache
    nq, nkv, hs = q.shape[1], kc.shape[2], q.shape[2]

    def deq(c, sc):
        b = c.reshape(-1, nkv, c.shape[-1])[:n, :, : hs // 2].to(torch.int16)
        u = torch.stack((b & 0xF, b >> 4), -1).reshape(n, nkv, hs).double()
        bits = sc.reshape(-1, nkv)[:n].contiguous().view(torch.int32)
        return (u - (bits & 0xF).double()[..., None]) * (bits & -16).view(torch.float32).double()[..., None]

    K, V = deq(kc, ks), deq(vc, vs)
    qr = stock.single_rht(q[rows].float()).double()
    pos = n - len(q) + torch.tensor(rows, device="cuda")
    out = torch.empty((len(rows), nq, hs), dtype=torch.float64, device="cuda")
    for h in range(nq):
        s = qr[:, h] @ K[:, h // (nq // nkv)].T * (hs ** -0.5 / hs)
        s = s.masked_fill(torch.arange(n, device="cuda")[None] > pos[:, None], float("-inf"))
        out[:, h] = torch.softmax(s, -1) @ V[:, h // (nq // nkv)]
    return stock.single_rht(out.float(), inverse=True).double() / hs


def _attend(stock, q, cache, nb, n, segments, out):
    kc, vc, ks, vs = cache
    rows, nq, hs = 32, q.shape[1], q.shape[2]
    segm = (torch.empty((rows, nq, segments, hs), dtype=torch.float32, device="cuda"),
            torch.empty((rows, nq, segments), dtype=torch.float32, device="cuda"),
            torch.empty((rows, nq, segments), dtype=torch.float32, device="cuda"))
    stock.unified_attention_int4(
        q, kc, vc, out, cu_seqlens_q=torch.tensor([0, len(q)], dtype=torch.int32, device="cuda"),
        max_seqlen_q=len(q), seqused_k=torch.tensor([n], dtype=torch.int32, device="cuda"), max_seqlen_k=n,
        softmax_scale=hs ** -0.5, window_size=(-1, -1),
        block_table=torch.arange(nb, dtype=torch.int32, device="cuda")[None], softcap=0.0, sinks=None,
        alibi_slopes=None, use_alibi_sqrt=False, qq_bias=None, output_scale=None, mm_prefix_range=None,
        k_scale_cache=ks, v_scale_cache=vs, seq_threshold_3D=rows, num_par_softmax_segments=segments,
        softmax_segm_output=segm[0], softmax_segm_max=segm[1], softmax_segm_expsum=segm[2])


def _timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def _relerr(o, ref):
    return ((o.double() - ref).norm() / ref.norm()).item()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contexts", default="32768,131072,262144")
    ap.add_argument("--decode", default=None, help="tile,warps,segments configs separated by ':' "
                    "(default: 64,4 with 16, 32, 64 and 128 segments)")
    ap.add_argument("--prefill-tiles", default=None, help="BLOCK_M,TILE,warps,stages (default: the built-in table)")
    ap.add_argument("--q-heads", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-size", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=3072)
    ap.add_argument("--mtp", type=int, default=1, help="draft tokens: decode steps have 1 + MTP query tokens")
    a = ap.parse_args(argv)
    import importlib
    stock = importlib.import_module(int4_kv.MODULE)
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    from ..attention import int4_prefill
    stock_launch = stock._launch_packed_attn
    props = torch.cuda.get_device_properties(0)
    major = torch.cuda.get_device_capability()[0]
    tiles = tuple(int(x) for x in a.prefill_tiles.split(",")) if a.prefill_tiles else int4_kv.prefill_tiles(major)
    configs = ([tuple(int(x) for x in c.split(",")) for c in a.decode.split(":")] if a.decode
               else [(64, 4, s) for s in (16, 32, 64, 128)])
    auto = int4_kv.segments_for(props.multi_processor_count)
    print(f"{props.name}, {props.multi_processor_count} SMs (auto: {auto} segments) | {a.q_heads} q / {a.kv_heads} "
          f"KV heads, head {a.head_size} | times per attention layer", flush=True)
    gens = {c: int4_kv._load(int4_kv.build_source(stock, tiles, c))._launch_packed_attn_qp for c in configs}
    nq, nkv, hs = a.q_heads, a.kv_heads, a.head_size
    try:
        for n in (int(x) for x in a.contexts.split(",")):
            cache, nb = _cache(stock, n, nkv, hs, a.block_size)
            q = torch.randn((1 + a.mtp, nq, hs), device="cuda", dtype=torch.bfloat16)
            ref = _reference(stock, q, cache, n, list(range(len(q))))
            out = torch.empty_like(q)
            stock._launch_packed_attn = stock_launch
            ms = _timed(lambda: _attend(stock, q, cache, nb, n, 16, out))
            print(f"ctx {n:>7} decode  vLLM stock (16 segments)     {ms:8.3f} ms  relerr {_relerr(out, ref):.1e}")
            for c in configs:
                stock._launch_packed_attn = gens[c]
                ms = _timed(lambda: _attend(stock, q, cache, nb, n, c[2], out))
                print(f"ctx {n:>7} decode  tile {c[0]:>3} warps {c[1]} segments {c[2]:>3}  {ms:8.3f} ms  "
                      f"relerr {_relerr(out, ref):.1e}", flush=True)
            chunk = min(1024, n)
            q = torch.randn((chunk, nq, hs), device="cuda", dtype=torch.bfloat16)
            rows = sorted({0, chunk // 2, chunk - 1})
            ref = _reference(stock, q, cache, n, rows)
            out = torch.empty_like(q)
            stock._launch_packed_attn = gens[configs[0]]
            ms = _timed(lambda: _attend(stock, q, cache, nb, n, 16, out), reps=2)
            print(f"ctx {n:>7} prefill Triton tiles {tiles}  {ms:8.2f} ms  relerr {_relerr(out[rows], ref):.1e}")

            def fa():
                qr = stock.single_rht(q.float()).to(q.dtype)
                o = int4_prefill.prefill_sequence(qr, cache[0], cache[1], cache[2], cache[3],
                                                  torch.arange(nb, dtype=torch.int32, device="cuda"), n,
                                                  hs ** -0.5 / hs, flash_attn_varlen_func)
                out.copy_((stock.single_rht(o.float(), inverse=True) / hs).to(q.dtype))
            ms = _timed(fa, reps=2)
            print(f"ctx {n:>7} prefill FlashAttention segments {int4_prefill.SEGMENT}  {ms:8.2f} ms  "
                  f"relerr {_relerr(out[rows], ref):.1e}", flush=True)
            del cache
            torch.cuda.empty_cache()
    finally:
        stock._launch_packed_attn = stock_launch
    print("Set the fastest decode config with VLLM_QUANTPACK_INT4_DECODE=tile,warps,segments (default 64,4,auto).")


if __name__ == "__main__":
    os.environ.setdefault("VLLM_QUANTPACK_INT4_KV", "0")   # the tool installs each config itself
    main()
