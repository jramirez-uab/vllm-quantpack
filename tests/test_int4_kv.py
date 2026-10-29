import logging
import sys
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


def _fake_triton_attn(monkeypatch):
    """A stand-in for vllm.v1.attention.backends.triton_attn whose builder records the segment count it saw."""
    mod = types.ModuleType("vllm.v1.attention.backends.triton_attn")
    mod.NUM_PAR_SOFTMAX_SEGMENTS = 16

    class Builder:
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            self.num_par_softmax_segments = mod.NUM_PAR_SOFTMAX_SEGMENTS

    mod.TritonAttentionMetadataBuilder = Builder
    names = ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends")
    for name in names:
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(sys.modules["vllm.v1.attention.backends"], "triton_attn", mod, raising=False)
    return mod


def _config(dtype):
    return types.SimpleNamespace(cache_config=types.SimpleNamespace(cache_dtype=dtype))


def test_decode_segments_only_for_int4_caches(monkeypatch):
    mod = _fake_triton_attn(monkeypatch)
    assert int4_kv._patch_decode_segments(64, LOG) == 64
    assert mod.TritonAttentionMetadataBuilder(None, [], _config("int4_per_token_head"), None).num_par_softmax_segments == 64
    assert mod.TritonAttentionMetadataBuilder(None, [], _config("auto"), None).num_par_softmax_segments == 16
    assert mod.NUM_PAR_SOFTMAX_SEGMENTS == 16


def test_decode_segments_not_power_of_two_left_stock(monkeypatch):
    mod = _fake_triton_attn(monkeypatch)
    assert int4_kv._patch_decode_segments(48, LOG) == 0
    assert mod.TritonAttentionMetadataBuilder(None, [], _config("int4_per_token_head"), None).num_par_softmax_segments == 16


def test_decode_override_from_environment(monkeypatch):
    monkeypatch.setenv("VLLM_QUANTPACK_INT4_DECODE", "32,8,32")
    assert int4_kv.decode_tiles(188) == (32, 8, 32)
    monkeypatch.setenv("VLLM_QUANTPACK_INT4_DECODE", "64,4,auto")
    assert int4_kv.decode_tiles(84) == (64, 4, 32)


@pytest.mark.parametrize("sms,segments", [(188, 64), (84, 32), (36, 16), (8, 16), (132, 64), (512, 128)],
                         ids=["rtx-pro-6000", "a40", "rtx-5060-ti", "tiny", "h100", "cap"])
def test_segments_follow_sm_count(monkeypatch, sms, segments):
    monkeypatch.delenv("VLLM_QUANTPACK_INT4_DECODE", raising=False)
    assert int4_kv.decode_tiles(sms) == (64, 4, segments)


# ---------------------------------------------------------------- GPU: exactness against fp64, stock vs patched
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NQ, NKV, HS, BS = 24, 4, 256, 3072      # Qwen3.8-27B full-attention layers, hybrid-model attention block


def _cache(m, nblocks):
    phs = HS // 2 + 4
    kv = torch.zeros((nblocks, NKV, BS, 2 * phs), dtype=torch.uint8, device="cuda")
    base = torch.tensor([], dtype=torch.float32, device="cuda").set_(kv.untyped_storage())
    st = [s // 4 for s in kv.stride()]
    ks = torch.as_strided(base, (nblocks, BS, NKV), (st[0], st[2], st[1]), HS // 2 // 4)
    vs = torch.as_strided(base, (nblocks, BS, NKV), (st[0], st[2], st[1]), (phs + HS // 2) // 4)
    kc, vc = kv.transpose(1, 2).split(phs, dim=-1)
    return kc, vc, ks, vs


def _fill(m, cache, first_block, n, seed):
    kc, vc, ks, vs = cache
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    slots = first_block * BS + torch.arange(n, device="cuda")
    m.reshape_and_cache_int4(k, v, kc, vc, slots, k_scale_cache=ks, v_scale_cache=vs)


def _reference(m, q, cache, first_block, n):
    """fp64 attention of q (the last len(q) positions of a context of n tokens) from the cache."""
    kc, vc, ks, vs = cache

    def deq(c, sc):
        b = c.reshape(-1, NKV, c.shape[-1])[first_block * BS:first_block * BS + n, :, : HS // 2].to(torch.int16)
        u = torch.stack((b & 0xF, b >> 4), -1).reshape(n, NKV, HS).double()
        bits = sc.reshape(-1, NKV)[first_block * BS:first_block * BS + n].contiguous().view(torch.int32)
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


def _attend(m, qs, cache, seqs, segments=16):
    """One attention call for a batch: seqs = [(first_block, n_blocks, context_len), ...], qs = per-seq queries."""
    kc, vc, ks, vs = cache
    q = torch.cat(qs)
    out = torch.empty_like(q)
    rows = 32
    segm = (torch.empty((rows, NQ, segments, HS), dtype=torch.float32, device="cuda"),
            torch.empty((rows, NQ, segments), dtype=torch.float32, device="cuda"),
            torch.empty((rows, NQ, segments), dtype=torch.float32, device="cuda"))
    width = max(nb for _, nb, _ in seqs)
    table = torch.zeros((len(seqs), width), dtype=torch.int32, device="cuda")
    for i, (fb, nb, _) in enumerate(seqs):
        table[i, :nb] = torch.arange(fb, fb + nb, dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, *torch.tensor([len(x) for x in qs]).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    m.unified_attention_int4(q, kc, vc, out, cu_seqlens_q=cu, max_seqlen_q=max(len(x) for x in qs),
                             seqused_k=torch.tensor([n for _, _, n in seqs], dtype=torch.int32, device="cuda"),
                             max_seqlen_k=max(n for _, _, n in seqs), softmax_scale=HS ** -0.5, window_size=(-1, -1),
                             block_table=table, softcap=0.0, sinks=None, alibi_slopes=None, use_alibi_sqrt=False,
                             qq_bias=None, output_scale=None, mm_prefix_range=None, k_scale_cache=ks,
                             v_scale_cache=vs, seq_threshold_3D=rows, num_par_softmax_segments=segments,
                             softmax_segm_output=segm[0], softmax_segm_max=segm[1], softmax_segm_expsum=segm[2])
    return out


def _relerr(o, ref):
    return ((o.double() - ref).norm() / ref.norm()).item()


@pytest.fixture
def int4_module(monkeypatch):
    m = pytest.importorskip("vllm.v1.attention.ops.int4_per_token_head")
    saved = m._launch_packed_attn, m.unified_attention_int4
    yield m
    m._launch_packed_attn, m.unified_attention_int4 = saved


@gpu
@pytest.mark.parametrize("qlen,n,fa", [(256, 16384, "1"), (256, 16384, "0"), (256, 40000, "1"), (2, 16384, "1"),
                                       (1, 16384, "1")],
                         ids=["prefill-flashattn", "prefill-triton", "prefill-3-segments", "mtp-decode", "decode"])
def test_patched_int4_attention_matches_fp64(int4_module, monkeypatch, qlen, n, fa):
    m = int4_module
    nb = (n + BS - 1) // BS
    cache = _cache(m, nb)
    _fill(m, cache, 0, n, 0)
    q = torch.randn((qlen, NQ, HS), device="cuda", dtype=torch.bfloat16)
    ref = _reference(m, q, cache, 0, n)
    before = _relerr(_attend(m, [q], cache, [(0, nb, n)]), ref)
    monkeypatch.setenv("VLLM_QUANTPACK_INT4_FA_PREFILL", fa)
    assert int4_kv.patch_int4_kv(LOG)
    assert getattr(m._launch_packed_attn, "_quantpack", False)
    # decode runs with the segment count the patched metadata builder allocates for int4 caches
    after = _relerr(_attend(m, [q], cache, [(0, nb, n)], int4_kv.decode_tiles(torch.cuda.get_device_properties(0).multi_processor_count)[2]), ref)
    assert after < 1e-2, f"patched relative error {after:.3e}"
    assert after < before, f"stock {before:.3e} vs patched {after:.3e}"


@gpu
def test_mixed_batch_long_prefill_and_mtp_decode(int4_module):
    """One step with a long prefill chunk (FlashAttention path) and an MTP decode (Triton sub-batch)."""
    m = int4_module
    n0, n1 = 40000, 20000
    b0, b1 = (n0 + BS - 1) // BS, (n1 + BS - 1) // BS
    cache = _cache(m, b0 + b1)
    _fill(m, cache, 0, n0, 0)
    _fill(m, cache, b0, n1, 1)
    q0 = torch.randn((256, NQ, HS), device="cuda", dtype=torch.bfloat16)
    q1 = torch.randn((2, NQ, HS), device="cuda", dtype=torch.bfloat16)
    assert int4_kv.patch_int4_kv(LOG)
    out = _attend(m, [q0, q1], cache, [(0, b0, n0), (b0, b1, n1)], int4_kv.decode_tiles(torch.cuda.get_device_properties(0).multi_processor_count)[2])
    assert _relerr(out[:256], _reference(m, q0, cache, 0, n0)) < 1e-2
    assert _relerr(out[256:], _reference(m, q1, cache, b0, n1)) < 1e-2
