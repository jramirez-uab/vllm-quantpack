import json
import os

import pytest
import torch

from vllm_quantpack.exl3.embed_codec import dequantize, gather, is_int4, quantize_int4, quantize_int8


def _table(v=6, h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(v, h, generator=g)


def test_int4_packs_two_signed_nibbles_per_byte_low_first():
    w = torch.tensor([[-8.0, 7.0, 0.0, 1.0]])
    q, s = quantize_int4(w, group=4)
    assert q.dtype == torch.uint8 and q.shape == (1, 2)
    assert s.shape == (1, 1) and s.item() == pytest.approx(8.0 / 7.0)
    # scale = absmax / 7; stored nibble = q + 8: -8 -> -7 -> 1, 7 -> 6.125 -> 6 -> 14, 0 -> 0 -> 8, 1 -> 0.875 -> 1 -> 9
    assert q.tolist() == [[1 | (14 << 4), 8 | (9 << 4)]]


def test_int4_dequantize_is_nibble_times_group_scale():
    w = _table(h=8)
    q, s = quantize_int4(w, group=4)
    assert s.shape == (6, 2)
    lo, hi = (q & 0xF).to(torch.int16) - 8, (q >> 4).to(torch.int16) - 8
    ints = torch.stack((lo, hi), -1).reshape(6, 8).float()
    ref = (ints.reshape(6, 2, 4) * s.float()[:, :, None]).reshape(6, 8)
    assert torch.equal(dequantize(q, s, torch.float32), ref)
    assert ints.min() >= -8 and ints.max() <= 7
    assert ((ref - w).norm() / w.norm()).item() < 0.2


def test_int4_gather_equals_rows_of_the_dequantized_table():
    w = _table(v=10, h=16)
    q, s = quantize_int4(w, group=8)
    ids = torch.tensor([[3, 0, 9], [9, 9, 1]])
    assert torch.equal(gather(q, s, ids, torch.bfloat16), dequantize(q, s, torch.bfloat16)[ids])


def test_int8_gather_keeps_the_per_row_formula():
    w = _table(v=10, h=16)
    q, s = quantize_int8(w)
    assert q.dtype == torch.int8 and s.shape == (10,)
    ids = torch.tensor([4, 4, 7])
    old = q[ids].to(torch.bfloat16) * s.to(torch.bfloat16).reshape(-1, 1)[ids]
    assert torch.equal(gather(q, s.to(torch.bfloat16), ids, torch.bfloat16), old)
    assert torch.equal(gather(q, s.to(torch.bfloat16).reshape(-1, 1), ids, torch.bfloat16), old)


def test_format_is_told_apart_by_the_qweight_dtype():
    w = _table(h=8)
    assert is_int4(quantize_int4(w, group=4)[0])
    assert not is_int4(quantize_int8(w)[0])


def test_int4_rejects_a_width_the_group_does_not_divide():
    with pytest.raises(ValueError):
        quantize_int4(_table(h=12), group=8)


def _checkpoint(tmp_path, w):
    from safetensors.torch import save_file
    src = tmp_path / "src"
    src.mkdir()
    k = "model.language_model.embed_tokens.weight"
    save_file({k: w.to(torch.bfloat16), "model.norm.weight": torch.ones(4)}, src / "model-00001.safetensors")
    save_file({"lm_head.weight": torch.zeros(2, 4)}, src / "model-00002.safetensors")
    idx = {"metadata": {"total_size": 0}, "weight_map": {k: "model-00001.safetensors",
                                                         "model.norm.weight": "model-00001.safetensors",
                                                         "lm_head.weight": "model-00002.safetensors"}}
    (src / "model.safetensors.index.json").write_text(json.dumps(idx))
    (src / "config.json").write_text("{}")
    return src


def test_tool_writes_int4_table_and_links_the_rest(tmp_path):
    from safetensors import safe_open
    from vllm_quantpack.tools.embed_quant import main
    w = _table(v=6, h=256)
    src = _checkpoint(tmp_path, w)
    dst = tmp_path / "dst"
    main([str(src), str(dst), "--bits", "4"])
    k = "model.language_model.embed_tokens"
    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    assert idx["weight_map"][f"{k}.qweight"] == idx["weight_map"][f"{k}.scales"] == "model-00001.safetensors"
    assert f"{k}.weight" not in idx["weight_map"]
    with safe_open(dst / "model-00001.safetensors", "pt") as f:
        q, s = f.get_tensor(f"{k}.qweight"), f.get_tensor(f"{k}.scales")
        assert torch.equal(f.get_tensor("model.norm.weight"), torch.ones(4))
    q_ref, s_ref = quantize_int4(w.to(torch.bfloat16).float())
    assert torch.equal(q, q_ref) and torch.equal(s, s_ref.to(torch.bfloat16))
    assert os.stat(dst / "model-00002.safetensors").st_ino == os.stat(src / "model-00002.safetensors").st_ino
    assert os.stat(dst / "config.json").st_ino == os.stat(src / "config.json").st_ino
    sizes = sum(os.path.getsize(dst / n) for n in ("model-00001.safetensors", "model-00002.safetensors"))
    assert 0 < idx["metadata"]["total_size"] < sizes


def test_tool_sim_stores_the_round_trip_as_bf16_weight(tmp_path):
    from safetensors import safe_open
    from vllm_quantpack.tools.embed_quant import main
    w = _table(v=6, h=256)
    src = _checkpoint(tmp_path, w)
    dst = tmp_path / "sim"
    main([str(src), str(dst), "--bits", "4", "--sim"])
    k = "model.language_model.embed_tokens.weight"
    with safe_open(dst / "model-00001.safetensors", "pt") as f:
        got = f.get_tensor(k)
    q, s = quantize_int4(w.to(torch.bfloat16).float())
    assert got.dtype == torch.bfloat16 and torch.equal(got, dequantize(q, s.to(torch.bfloat16), torch.bfloat16))
    assert json.loads((dst / "model.safetensors.index.json").read_text())["weight_map"][k] == "model-00001.safetensors"


def test_tool_refuses_an_existing_destination(tmp_path):
    from vllm_quantpack.tools.embed_quant import main
    src = _checkpoint(tmp_path, _table(v=6, h=256))
    (tmp_path / "dst").mkdir()
    with pytest.raises(FileExistsError):
        main([str(src), str(tmp_path / "dst"), "--bits", "4"])
