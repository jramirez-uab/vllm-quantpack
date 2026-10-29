import logging
import types

import pytest
import torch

from vllm_quantpack.attention import int4_kv

LOG = logging.getLogger("test")


def test_missing_anchor_is_reported_not_patched():
    with pytest.raises(LookupError):
        int4_kv._edit("def f():\n    return 1\n", (("return 2", "return 3"),), "kernel")


def test_duplicated_anchor_is_reported():
    with pytest.raises(LookupError):
        int4_kv._edit("a\na\n", (("a", "b"),), "kernel")


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("VLLM_QUANTPACK_INT4_KV", "0")
    assert int4_kv.patch_int4_kv(LOG) is False


def test_vllm_without_int4_mode_leaves_everything(monkeypatch):
    monkeypatch.setattr(int4_kv, "MODULE", "vllm_quantpack_no_such_module")
    assert int4_kv.patch_int4_kv(LOG) is False


# ---------------------------------------------------------------- GPU: exactness against fp64, stock vs patched
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NQ, NKV, HS, BS = 24, 4, 256, 3072      # Qwen3.8-27B full-attention layers, hybrid-model attention block


def _cache(m, n):
    nb = (n + BS - 1) // BS
    phs = HS // 2 + 4
    kv = torch.zeros((nb, NKV, BS, 2 * phs), dtype=torch.uint8, device="cuda")
    base = torch.tensor([], dtype=torch.float32, device="cuda").set_(kv.untyped_storage())
    st = [s // 4 for s in kv.stride()]
    ks = torch.as_strided(base, (nb, BS, NKV), (st[0], st[2], st[1]), HS // 2 // 4)
    vs = torch.as_strided(base, (nb, BS, NKV), (st[0], st[2], st[1]), (phs + HS // 2) // 4)
    kc, vc = kv.transpose(1, 2).split(phs, dim=-1)
    g = torch.Generator(device="cuda").manual_seed(0)
    k = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    m.reshape_and_cache_int4(k, v, kc, vc, torch.arange(n, device="cuda"), k_scale_cache=ks, v_scale_cache=vs)
    return kc, vc, ks, vs, nb


def _reference(m, q, kc, vc, ks, vs, n):
    nb = kc.shape[0]

    def deq(c, sc):
        b = c[..., : HS // 2].reshape(nb * BS, NKV, HS // 2)[:n].to(torch.int16)
        u = torch.stack((b & 0xF, b >> 4), -1).reshape(n, NKV, HS).double()
        bits = sc.reshape(nb * BS, NKV)[:n].contiguous().view(torch.int32)
        return (u - (bits & 0xF).double()[..., None]) * (bits & -16).view(torch.float32).double()[..., None]

    K, V = deq(kc, ks), deq(vc, vs)
    qr = m.single_rht(q.float()).double()
    pos = n - len(q) + torch.arange(len(q), device="cuda")
    out = torch.empty((len(q), NQ, HS), dtype=torch.float64, device="cuda")
    for h in range(NQ):
        s = qr[:, h] @ K[:, h // (NQ // NKV)].T * (HS ** -0.5 / HS)
        s = s.masked_fill(torch.arange(n, device="cuda")[None] > pos[:, None], float("-inf"))
        out[:, h] = torch.softmax(s, -1) @ V[:, h // (NQ // NKV)]
    return m.single_rht(out.float(), inverse=True).double() / HS


def _attend(m, q, kc, vc, ks, vs, nb, n):
    out = torch.empty_like(q)
    rows = 32
    segm = (torch.empty((rows, NQ, 16, HS), dtype=torch.float32, device="cuda"),
            torch.empty((rows, NQ, 16), dtype=torch.float32, device="cuda"),
            torch.empty((rows, NQ, 16), dtype=torch.float32, device="cuda"))
    m.unified_attention_int4(q, kc, vc, out, cu_seqlens_q=torch.tensor([0, len(q)], dtype=torch.int32, device="cuda"),
                             max_seqlen_q=len(q), seqused_k=torch.tensor([n], dtype=torch.int32, device="cuda"),
                             max_seqlen_k=n, softmax_scale=HS ** -0.5, window_size=(-1, -1),
                             block_table=torch.arange(nb, dtype=torch.int32, device="cuda")[None], softcap=0.0,
                             sinks=None, alibi_slopes=None, use_alibi_sqrt=False, qq_bias=None, output_scale=None,
                             mm_prefix_range=None, k_scale_cache=ks, v_scale_cache=vs, seq_threshold_3D=rows,
                             num_par_softmax_segments=16, softmax_segm_output=segm[0], softmax_segm_max=segm[1],
                             softmax_segm_expsum=segm[2])
    return out


@gpu
@pytest.mark.parametrize("qlen", [256, 2, 1], ids=["prefill-chunk", "mtp-decode", "decode"])
def test_patched_int4_attention_matches_fp64(qlen):
    m = pytest.importorskip("vllm.v1.attention.ops.int4_per_token_head")
    stock = m._launch_packed_attn
    n = 16384
    kc, vc, ks, vs, nb = _cache(m, n)
    q = torch.randn((qlen, NQ, HS), device="cuda", dtype=torch.bfloat16)
    ref = _reference(m, q, kc, vc, ks, vs, n)
    err = lambda o: ((o.double() - ref).norm() / ref.norm()).item()
    try:
        before = err(_attend(m, q, kc, vc, ks, vs, nb, n))
        assert int4_kv.patch_int4_kv(LOG)
        assert getattr(m._launch_packed_attn, "_quantpack", False)
        after = err(_attend(m, q, kc, vc, ks, vs, nb, n))
    finally:
        m._launch_packed_attn = stock
    assert after < 1e-2, f"patched relative error {after:.3e}"
    assert after < before, f"stock {before:.3e} vs patched {after:.3e}"
