"""EXL3 MoE prefill path on real expert weights (GPU).

Set VLLM_QUANTPACK_TEST_MOE to one or more EXL3 MoE checkpoints (comma separated). One layer of experts is loaded
from each and the virtual-expert path (one fused launch per row tier, no CPU sync) is compared with the per-expert
path it replaces.
"""
import json
import os
import re

import pytest

torch = pytest.importorskip("torch")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
CKPTS = [p for p in os.environ.get("VLLM_QUANTPACK_TEST_MOE", "").split(",") if p]
_NAME = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mcg|mul1)$")


def _load_layer(ckpt):
    """_Experts of the first MoE layer of an EXL3 checkpoint."""
    from safetensors import safe_open
    from vllm_quantpack.exl3 import moe, ops  # noqa: F401  (ops registers quantpack::exl3_gemm)
    wmap = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    names = [n for n in wmap if _NAME.match(n)]
    prefix = min({_NAME.match(n).group(1) for n in names}, key=lambda p: int(re.search(r"layers\.(\d+)", p).group(1)))
    store = {}
    for n in names:
        m = _NAME.match(n)
        if m.group(1) != prefix:
            continue
        with safe_open(os.path.join(ckpt, wmap[n]), "pt") as f:
            store[(int(m.group(2)), m.group(3), m.group(4))] = f.get_tensor(n)
    E = 1 + max(e for e, _, _ in store)
    gateless = (0, "gate_proj", "trellis") not in store
    projs = {}
    for proj in ("up_proj", "down_proj") if gateless else ("gate_proj", "up_proj", "down_proj"):
        per = [{s: store[(e, proj, s)] for s in ("trellis", "suh", "svh", "mcg", "mul1") if (e, proj, s) in store}
               for e in range(E)]
        projs[proj] = moe._Proj(proj, per, torch.device("cuda", torch.cuda.current_device()))
    if gateless:
        projs["gate_proj"] = projs["up_proj"]
    return moe._Experts(prefix, projs["gate_proj"], projs["up_proj"], projs["down_proj"], torch.device("cuda", torch.cuda.current_device()),
                        gateless=gateless)


def _routing(ex, T, k, seed):
    """Skewed top-k routing (a few hot experts, as in real prefills) with distinct experts per token."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((T, ex.E), generator=g, device="cuda") + torch.linspace(3, 0, ex.E, device="cuda")
    w, ids = logits.softmax(-1).topk(k, -1)
    return ids.to(torch.int32), (w / w.sum(-1, keepdim=True)).to(torch.float32)


@pytest.fixture(scope="module", params=CKPTS or [None], ids=lambda p: os.path.basename(p) if p else "none")
def experts(request):
    if not request.param or not torch.cuda.is_available():
        pytest.skip("set VLLM_QUANTPACK_TEST_MOE to EXL3 MoE checkpoints")
    pytest.importorskip("exllamav3")
    return _load_layer(request.param)


@gpu
@pytest.mark.parametrize("T", [300, 2048, 8192])
def test_virtual_experts_match_per_expert_path(experts, T, monkeypatch):
    ex = experts
    x = torch.randn((T, ex.H), device="cuda", dtype=torch.bfloat16) * 0.5
    ids, w = _routing(ex, T, 8, T)
    ids[0, -1] = -1                                       # an EP padding slot goes to the sentinel bucket
    monkeypatch.setenv("VLLM_QUANTPACK_MOE_VIRTUAL", "0")
    ref = ex._sum(x, ids, w)
    monkeypatch.setenv("VLLM_QUANTPACK_MOE_VIRTUAL", "1")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = ex._sum(x, ids, w)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    err = ((out - ref).norm() / ref.norm()).item()
    # Both paths are deterministic (repeat runs differ by ~1e-8). With no expert above fused_rows (T=300 here) they run
    # the same fused kernel and must agree exactly; hot experts went through a different kernel in the old path and
    # differ by 1.1-1.7e-3 measured on both checkpoints (summation order of the bf16 products), so 5e-3 there.
    tol = 1e-5 if T == 300 else 5e-3
    assert err < tol, f"relative error {err:.2e}"
