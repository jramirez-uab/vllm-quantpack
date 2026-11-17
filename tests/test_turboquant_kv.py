import logging

import pytest
import torch

from vllm_quantpack.attention import turboquant_kv

LOG = logging.getLogger("test")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NQ, NKV, HS, BS = 24, 4, 256, 3072      # Qwen3.8-27B full-attention layers, hybrid-model attention block


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("VLLM_QUANTPACK_TURBOQUANT", "0")
    assert turboquant_kv.patch_turboquant_kv(LOG) is False


def test_vllm_without_turboquant_leaves_everything(monkeypatch):
    monkeypatch.setattr(turboquant_kv, "DECODE_MODULE", "vllm_quantpack_no_such_module")
    assert turboquant_kv.patch_turboquant_kv(LOG) is False


def test_missing_anchor_is_reported_not_patched():
    class Kernel:
        @staticmethod
        def fn():
            return 1

    modules = {turboquant_kv.DECODE_MODULE: type("M", (), {"_tq_decode_stage1": Kernel, "_tq_full_dequant_kv": Kernel}),
               turboquant_kv.STORE_MODULE: type("M", (), {"_tq_fused_store_fp8": Kernel})}
    with pytest.raises(LookupError):
        turboquant_kv.build_source(modules)


def test_token_rows_give_each_query_token_its_own_context():
    """Requests with 4, 1 and 3 query tokens; two padding tokens after them repeat the last request."""
    qsl = torch.tensor([0, 4, 5, 8], dtype=torch.int32)
    seq = torch.tensor([1000, 20, 7], dtype=torch.int32)
    table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    lens, rows = turboquant_kv.token_rows(qsl, seq, table, 10)
    assert lens.tolist() == [997, 998, 999, 1000, 20, 5, 6, 7, 7, 7]
    assert rows[:, 0].tolist() == [1, 1, 1, 1, 3, 5, 5, 5, 5, 5]
    assert lens.dtype == seq.dtype


def test_token_rows_during_graph_capture_stay_positive():
    """CUDA graph capture fills seq_lens with 1: no token gets an empty or negative context."""
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    lens, _ = turboquant_kv.token_rows(qsl, torch.ones(2, dtype=torch.int32), torch.zeros((2, 1), dtype=torch.int32), 8)
    assert lens.tolist() == [1] * 8


def test_speculative_limit_comes_from_the_builder_config():
    """The forward has no current config: before the builder runs, steps stay stock; after it, 4-token steps use decode."""
    pytest.importorskip("vllm.v1.attention.backend")
    from types import SimpleNamespace as NS
    calls = []

    class Impl:
        def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT=None, layer=None):
            calls.append("stock")

        def _max_capture_batch_size(self):
            return 2

        def _decode_attention(self, query, kv_cache, meta, Pi, centroids, PiT, layer):
            calls.append(("decode", meta.seq_lens.tolist(), meta.is_prefill, meta.max_query_len))

    class Builder:
        def _reserve_workspace(self):
            calls.append("reserve")

    backend = NS(TurboQuantAttentionImpl=Impl, TurboQuantMetadataBuilder=Builder, TurboQuantMetadata=NS)
    assert turboquant_kv._patch_spec_decode(backend, LOG)
    meta = NS(max_query_len=4, query_start_loc=torch.tensor([0, 4]), seq_lens=torch.tensor([100]),
              block_table=torch.tensor([[7]]), slot_mapping=None, max_seq_len=100)
    query = torch.zeros(4, 1, 1)
    Impl()._prefill_attention(query, None, None, None, meta, None, None)
    assert calls == ["stock"] and Impl()._max_capture_batch_size() == 2
    builder = Builder()
    builder.vllm_config = NS(speculative_config=NS(num_speculative_tokens=3, parallel_drafting=False,
                                                        max_num_new_slots_for_drafting=0),
                             use_v2_model_runner=True, scheduler_config=NS(max_num_seqs=2))
    builder._reserve_workspace()
    Impl()._prefill_attention(query, None, None, None, meta, None, None)
    assert calls[1:] == ["reserve", ("decode", [97, 98, 99, 100], False, 1)]
    assert Impl()._max_capture_batch_size() == 8
    meta.max_query_len = 5                       # a real prefill chunk stays on the stock path
    Impl()._prefill_attention(query, None, None, None, meta, None, None)
    assert calls[-1] == "stock"


@pytest.fixture
def tq():
    """(store module, decode module, backend module, config) with the stock kernels restored afterwards."""
    store = pytest.importorskip(turboquant_kv.STORE_MODULE)
    decode = pytest.importorskip(turboquant_kv.DECODE_MODULE)
    backend = pytest.importorskip(turboquant_kv.BACKEND_MODULE)
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    impl, builder = backend.TurboQuantAttentionImpl, backend.TurboQuantMetadataBuilder
    saved = (store._tq_fused_store_fp8, decode._tq_decode_stage1, decode._tq_full_dequant_kv,
             backend._tq_full_dequant_kv, impl._prefill_attention, impl._max_capture_batch_size,
             builder._reserve_workspace)
    yield store, decode, backend, TurboQuantConfig.from_cache_dtype("turboquant_k8v4", HS)
    (store._tq_fused_store_fp8, decode._tq_decode_stage1, decode._tq_full_dequant_kv, backend._tq_full_dequant_kv,
     impl._prefill_attention, impl._max_capture_batch_size, builder._reserve_workspace) = saved


def _store_and_attend(tq, k, v, q):
    """Store n tokens in a fresh cache and attend with one query over all of them. Returns (cache, output)."""
    store, decode, backend, cfg = tq
    n = k.shape[0]
    nb = (n + BS - 1) // BS
    cache = torch.zeros((nb, BS, NKV, cfg.slot_size_aligned), dtype=torch.uint8, device="cuda")
    H = backend._build_hadamard(HS, "cuda")
    mid = torch.zeros(1, device="cuda")
    store.triton_turboquant_store(k, v, cache, torch.arange(n, dtype=torch.int32, device="cuda"), H, mid,
                                  mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
                                  value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8)
    out = decode.triton_turboquant_decode_attention(
        query=q, kv_cache=cache, block_table=torch.arange(nb, dtype=torch.int32, device="cuda")[None],
        seq_lens=torch.tensor([n], dtype=torch.int32, device="cuda"), Pi=H, centroids=mid, scale=HS ** -0.5,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction, PiT=H)
    return cache, out


def _exact(q, k, v):
    out = torch.empty((1, NQ, HS), dtype=torch.float64, device="cuda")
    for h in range(NQ):
        s = q[0, h].double() @ k[:, h // (NQ // NKV)].double().T * HS ** -0.5
        out[0, h] = torch.softmax(s, -1) @ v[:, h // (NQ // NKV)].double()
    return out


def _relerr(o, ref):
    return ((o.double() - ref).norm() / ref.norm()).item()


def _data(n, key_std):
    g = torch.Generator(device="cuda").manual_seed(0)
    k = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16) * key_std
    v = torch.randn((n, NKV, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    q = torch.randn((1, NQ, HS), generator=g, device="cuda", dtype=torch.bfloat16)
    return k, v, q


@gpu
def test_keys_beyond_the_e4b15_range_survive_the_cache(tq):
    """Keys with |x| up to ~10: stored as E4M3 bytes and attended like the exact keys on every GPU."""
    k, v, q = _data(5000, 3.0)
    ref = _exact(q, k, v)
    emulated = turboquant_kv.needs_e4m3_emulation()
    before = _relerr(_store_and_attend(tq, k, v, q)[1], ref)
    assert turboquant_kv.patch_turboquant_kv(LOG)
    assert getattr(tq[0]._tq_fused_store_fp8, "_quantpack", False) is emulated
    assert getattr(tq[2].TurboQuantAttentionImpl._prefill_attention, "_quantpack", False)
    cache, out = _store_and_attend(tq, k, v, q)
    after = _relerr(out, ref)
    want = k.float().cpu().to(torch.float8_e4m3fn).view(torch.uint8)
    got = cache.reshape(-1, NKV, cache.shape[-1])[:5000, :, :HS].cpu()
    assert torch.equal(got, want)
    assert after < 0.2, f"relative error {after:.3e}"
    if emulated:
        assert after < before / 2, f"stock {before:.3e} vs patched {after:.3e}"
    print(f"\nturboquant k8v4 decode relative error: stock {before:.3e}, patched {after:.3e}")


@gpu
def test_small_keys_unchanged_by_the_patch(tq):
    """Keys inside +-1.875 (exact in both FP8 formats' range): same attention output before and after."""
    k, v, q = _data(5000, 0.25)
    before = _store_and_attend(tq, k, v, q)[1]
    turboquant_kv.patch_turboquant_kv(LOG)
    after = _store_and_attend(tq, k, v, q)[1]
    assert _relerr(after, _exact(q, k, v)) < 0.2
    assert _relerr(after, before.double()) < 2e-2
