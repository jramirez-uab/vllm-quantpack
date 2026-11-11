import logging

import pytest
import torch

from vllm_quantpack.attention import int4_kv, k8v4_kv

import test_int4_kv as t4

LOG = logging.getLogger("test")
gpu = t4.gpu
NQ, NKV, HS, BS = t4.NQ, t4.NKV, t4.HS, t4.BS


# ---------------------------------------------------------------- page cell and views (CPU)
def test_cell_is_key_bytes_plus_packed_value_plus_two_scales():
    assert k8v4_kv.cell_bytes(256, 256) == 392
    assert k8v4_kv.cell_bytes(128, 64) == 168


@pytest.mark.parametrize("layout", ["hnd", "nhd"])
def test_views_cover_the_cell_without_overlap(layout):
    """Each view writes its own bytes of the cell: [K | K scale | V | V scale], for both physical layouts."""
    nb, nkv, bs, hs = 3, 2, 4, 16
    cell = k8v4_kv.cell_bytes(hs, hs)
    if layout == "hnd":
        kv = torch.zeros((nb, nkv, bs, cell), dtype=torch.int8)
    else:
        kv = torch.zeros((nb, bs, nkv, cell), dtype=torch.int8).transpose(1, 2)
    key, value, ks, vs = k8v4_kv.views(kv, hs)
    assert key.shape == (nb, bs, nkv, hs) and key.dtype == torch.int8
    assert value.shape == (nb, bs, nkv, hs // 2) and value.dtype == torch.uint8
    assert ks.shape == vs.shape == (nb, bs, nkv) and ks.dtype == torch.float32
    key.fill_(1)
    ks.view(torch.int32).fill_(0x02020202)
    value.fill_(3)
    vs.view(torch.int32).fill_(0x04040404)
    want = torch.tensor([1] * hs + [2] * 4 + [3] * (hs // 2) + [4] * 4, dtype=torch.int8)
    assert (kv == want).all()
    ks[1, 2, 1] = 7.0                      # (block, slot, head) of the scale = (block, head, slot) of the cache
    assert kv[1, 1, 2, hs:hs + 4].view(torch.float32).item() == 7.0


def test_views_reject_another_cell_size():
    with pytest.raises(ValueError, match="expected 392"):
        k8v4_kv.views(torch.zeros((1, 4, 16, 264), dtype=torch.int8), 256)


def test_int4_cache_views_are_not_taken_for_k8v4():
    assert not k8v4_kv.is_k8v4(torch.zeros(1, dtype=torch.uint8))
    assert k8v4_kv.is_k8v4(torch.zeros(1, dtype=torch.int8))


def test_generated_source_has_both_kernels():
    m = pytest.importorskip("vllm.v1.attention.ops.int4_per_token_head")
    src = int4_kv.build_source(m, (64, 64, 8, 2))
    for name in ("def _attn_packed_qp(", "def _launch_packed_attn_qp(", "def _attn_k8v4_qp(", "def _launch_k8v4_qp("):
        assert src.count(name) == 1, name
    k8 = src[src.index("def _attn_k8v4_qp("):src.index("def _launch_k8v4_qp(")]
    assert "raw_dot = tl.dot(Q, K)" in k8 and "unpack_int4_nibbles(K_packed)" not in k8
    assert "V_s0_u, V_s1_u = unpack_int4_nibbles(V_packed)" in k8      # the value half is the int4 one


# ---------------------------------------------------------------- registration in vLLM
@pytest.fixture
def vllm_int4(monkeypatch):
    m = pytest.importorskip("vllm.v1.attention.ops.int4_per_token_head")
    saved = m._launch_packed_attn, m.unified_attention_int4
    yield m
    m._launch_packed_attn, m.unified_attention_int4 = saved


def test_dtype_is_registered_and_sized(vllm_int4):
    cache_config = pytest.importorskip("vllm.config.cache")
    if not hasattr(cache_config, "register_kv_cache_dtype"):
        pytest.skip("vLLM without the KV cache dtype registry")
    from dataclasses import replace
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode, get_kv_quant_mode

    assert int4_kv.patch_int4_kv(LOG)
    assert cache_config.is_known_kv_cache_dtype(k8v4_kv.NAME)
    assert STR_DTYPE_TO_TORCH_DTYPE[k8v4_kv.NAME] == torch.int8
    assert get_kv_quant_mode(k8v4_kv.NAME) == KVQuantMode.INT4_PER_TOKEN_HEAD
    assert TritonAttentionBackend.supports_kv_cache_dtype(k8v4_kv.NAME)
    int4 = FullAttentionSpec(block_size=16, num_kv_heads=NKV, head_size=HS, dtype=torch.uint8,
                             kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD)
    assert TritonAttentionBackend.customize_spec(int4).state_content_size_bytes == 264      # stock int4 unchanged
    k8v4 = replace(int4, dtype=torch.int8)
    assert TritonAttentionBackend.customize_spec(k8v4).state_content_size_bytes == 392
    fp8 = replace(int4, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    assert TritonAttentionBackend.customize_spec(fp8).state_content_size_bytes == 520


# ---------------------------------------------------------------- GPU: store and attention against fp64
def _cache(nblocks):
    kv = torch.zeros((nblocks, NKV, BS, k8v4_kv.cell_bytes(HS, HS)), dtype=torch.int8, device="cuda")
    return k8v4_kv.views(kv, HS)


def _fill(m, cache, first_block, n, seed):
    kc, vc, ks, vs = cache
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    slots = first_block * BS + torch.arange(n, device="cuda")
    m.reshape_and_cache_int4(k, v, kc, vc, slots, k_scale_cache=ks, v_scale_cache=vs)
    return k, v


def _stored(cache, first_block, n):
    """(K, V) the cache holds for n tokens from first_block, in fp64 (RHT domain)."""
    kc, vc, ks, vs = cache
    rows = slice(first_block * BS, first_block * BS + n)
    K = kc.reshape(-1, NKV, HS)[rows].double() * ks.reshape(-1, NKV)[rows].double()[..., None]
    b = vc.reshape(-1, NKV, HS // 2)[rows].to(torch.int16)
    u = torch.stack((b & 0xF, b >> 4), -1).reshape(n, NKV, HS).double()
    bits = vs.reshape(-1, NKV)[rows].contiguous().view(torch.int32)
    V = (u - (bits & 0xF).double()[..., None]) * (bits & -16).view(torch.float32).double()[..., None]
    return K, V


def _reference(m, q, cache, first_block, n):
    K, V = _stored(cache, first_block, n)
    qr = m.single_rht(q.float()).double()
    pos = n - len(q) + torch.arange(len(q), device="cuda")
    out = torch.empty((len(q), NQ, HS), dtype=torch.float64, device="cuda")
    for h in range(NQ):
        s = qr[:, h] @ K[:, h // (NQ // NKV)].T * (HS ** -0.5 / HS)
        s = s.masked_fill(torch.arange(n, device="cuda")[None] > pos[:, None], float("-inf"))
        out[:, h] = torch.softmax(s, -1) @ V[:, h // (NQ // NKV)]
    return m.single_rht(out.float(), inverse=True).double() / HS


def _segments():
    return int4_kv.decode_tiles(torch.cuda.get_device_properties(0).multi_processor_count)[2]


@gpu
def test_store_keeps_keys_at_8_bits_and_values_at_4(vllm_int4):
    m = vllm_int4
    assert int4_kv.patch_int4_kv(LOG)
    n = 4096
    cache = _cache(2)
    k, v = _fill(m, cache, 0, n, 0)
    K, V = _stored(cache, 0, n)
    kr, vr = m.single_rht(k.float()).double(), m.single_rht(v.float()).double()
    k_err, v_err = t4._relerr(K, kr), t4._relerr(V, vr)
    assert k_err < 0.02, f"keys {k_err:.3e}"                    # int8 over a Gaussian row: ~1 %
    assert 0.05 < v_err < 0.2, f"values {v_err:.3e}"            # 4 bits: ~10 %
    assert (cache[0].reshape(-1, NKV, HS)[n:] == 0).all()       # nothing written past the tokens
    slots = torch.full((8,), -1, device="cuda")                 # padding slots are skipped
    before = cache[0].clone()
    m.reshape_and_cache_int4(k[:8], v[:8], cache[0], cache[1], slots, k_scale_cache=cache[2], v_scale_cache=cache[3])
    assert (cache[0] == before).all()


@gpu
@pytest.mark.parametrize("qlen,n,fa", [(256, 16384, "1"), (256, 16384, "0"), (256, 40000, "1"), (2, 16384, "1"),
                                       (4, 16384, "1"), (3, 16384, "1"), (1, 16384, "1")],
                         ids=["prefill-flashattn", "prefill-triton", "prefill-3-segments", "mtp-decode", "mtp3-decode",
                              "mtp2-decode", "decode"])
def test_k8v4_attention_matches_fp64(vllm_int4, monkeypatch, qlen, n, fa):
    m = vllm_int4
    monkeypatch.setenv("VLLM_QUANTPACK_INT4_FA_PREFILL", fa)
    assert int4_kv.patch_int4_kv(LOG)
    nb = (n + BS - 1) // BS
    cache = _cache(nb)
    _fill(m, cache, 0, n, 0)
    q = torch.randn((qlen, NQ, HS), device="cuda", dtype=torch.bfloat16)
    err = t4._relerr(t4._attend(m, [q], cache, [(0, nb, n)], _segments()), _reference(m, q, cache, 0, n))
    assert err < 1e-2, f"relative error {err:.3e}"


@gpu
def test_k8v4_mixed_batch_long_prefill_and_mtp_decodes(vllm_int4):
    """A long prefill chunk (FlashAttention path) with MTP decodes of 4, 3 and 1 query tokens in one step."""
    m = vllm_int4
    assert int4_kv.patch_int4_kv(LOG)
    lens = [(256, 40000), (4, 20000), (3, 9000), (1, 16384)]
    blocks = [(n + BS - 1) // BS for _, n in lens]
    first = [sum(blocks[:i]) for i in range(len(lens))]
    cache = _cache(sum(blocks))
    for i, (_, n) in enumerate(lens):
        _fill(m, cache, first[i], n, i)
    qs = [torch.randn((ql, NQ, HS), device="cuda", dtype=torch.bfloat16) for ql, _ in lens]
    out = t4._attend(m, qs, cache, [(first[i], blocks[i], n) for i, (_, n) in enumerate(lens)], _segments())
    at = 0
    for i, (ql, n) in enumerate(lens):
        assert t4._relerr(out[at:at + ql], _reference(m, qs[i], cache, first[i], n)) < 1e-2, f"sequence {i}"
        at += ql


@gpu
def test_int4_cache_still_goes_through_the_int4_kernels(vllm_int4):
    """With the k8v4 dispatch installed, an int4 cache is stored and read as before."""
    m = vllm_int4
    assert int4_kv.patch_int4_kv(LOG)
    n = 16384
    nb = (n + BS - 1) // BS
    cache = t4._cache(m, nb)
    t4._fill(m, cache, 0, n, 0)
    q = torch.randn((4, NQ, HS), device="cuda", dtype=torch.bfloat16)
    assert t4._relerr(t4._attend(m, [q], cache, [(0, nb, n)], _segments()), t4._reference(m, q, cache, 0, n)) < 1e-2
