"""Quantize embed_tokens of an EXL3 checkpoint for vLLM: int8 per row or int4 per group of 128 (exl3/embed_codec.py).

exllamav3 leaves embed_tokens in bf16 (2.37 GiB for Qwen3.8-27B); vllm-quantpack serves it as int8 (1.18 GiB) or int4
(0.59 GiB + scales). Writes a new directory: every other file is a hard link to the source (no extra space), except the
shard holding the embedding and the index.
--sim writes the int4/int8 round trip back as a bf16 `weight` instead, for tools that only read bf16 tables (qbench KLD).

usage: python -m vllm_quantpack.tools.embed_quant SOURCE_DIR DEST_DIR --bits {4,8} [--sim]
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..exl3.embed_codec import dequantize, quantize_int4, quantize_int8

KEY = "model.language_model.embed_tokens.weight"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bits", type=int, choices=(4, 8), required=True)
    ap.add_argument("--sim", action="store_true", help="store the round trip as a bf16 weight (for qbench)")
    a = ap.parse_args(argv)
    if os.path.exists(a.dst):
        raise FileExistsError(a.dst)
    if os.path.exists(f"{a.src}/model.safetensors.index.json"):
        idx = json.load(open(f"{a.src}/model.safetensors.index.json"))
    else:  # small model in a single model.safetensors: build the index
        with safe_open(f"{a.src}/model.safetensors", "pt") as f:
            idx = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in f.keys()}}
    shard = idx["weight_map"][KEY]
    os.makedirs(a.dst)
    for f in os.listdir(a.src):
        if f not in (shard, "model.safetensors.index.json") and os.path.isfile(f"{a.src}/{f}"):
            os.link(f"{a.src}/{f}", f"{a.dst}/{f}")
    out = {}
    with safe_open(f"{a.src}/{shard}", "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            out[k] = f.get_tensor(k)
    w = out.pop(KEY).float()
    q, s = quantize_int4(w) if a.bits == 4 else quantize_int8(w)
    s = s.to(torch.bfloat16)
    deq = dequantize(q, s, torch.bfloat16)
    print(f"{KEY}: {tuple(w.shape)} -> int{a.bits} | relative error {((deq.float() - w).norm() / w.norm()).item():.4f}")
    names = [KEY] if a.sim else [KEY.replace(".weight", ".qweight"), KEY.replace(".weight", ".scales")]
    if a.sim:
        out[KEY] = deq.contiguous()
    else:
        out[names[0]], out[names[1]] = q.contiguous(), s.contiguous()
    save_file(out, f"{a.dst}/{shard}", metadata=meta)
    wm = {k: v for k, v in idx["weight_map"].items() if k != KEY}
    wm.update({n: shard for n in names})
    idx["weight_map"] = dict(sorted(wm.items()))
    total = 0  # recompute total_size from the safetensors headers (the quantized table is smaller)
    for f in sorted(set(wm.values())):
        with open(f"{a.dst}/{f}", "rb") as fh:
            h = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in h.items() if k != "__metadata__")
    idx.setdefault("metadata", {})["total_size"] = total
    with open(f"{a.dst}/model.safetensors.index.json", "w") as fh:
        json.dump(idx, fh, indent=2)
    print("done:", a.dst)


if __name__ == "__main__":
    main()
