"""Fused EXL3 decode MoE for Inkling2: exllamav3's exl3_moe_coop with Inkling's softcapped SwiGLU.

exllamav3's fused decode kernel (``ext.exl3_moe_coop``, up to 8 tokens, two launches per layer) is about twice as
fast as running the gate/up/down projections through ``exl3_mgemm``, but it only knows the SiLU/GELU/ReLU2/OAI
activations. Inkling's experts use ``softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap)``.

This module builds (once, cached by torch's extension loader) a private copy of that kernel from the sources the
installed exllamav3 ships, with three checked text patches:

* a new activation ``MOE_COOP_ACT_SOFTCAP_SWIGLU`` (gate cap in ``act_limit``, up cap in a new ``act_limit_u``);
* the activation range check accepting it;
* an entry point that sets the up cap.

Everything else (trellis decode, Hadamard rotations, split-k, autotuning) is exllamav3's own code. If the patches
do not apply (a newer exllamav3 changed the sources) or the build fails, ``available()`` is False and the caller
keeps the exl3_mgemm path.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading

import torch

log = logging.getLogger(__name__)

MAX_ROWS = 8                       # MAX_BSZN of the kernel (rows per launch)
ACT_SOFTCAP_SWIGLU = 4

_lock = threading.Lock()
_mods: dict = {}                   # K set -> module or None (failed)

_ENTRY = r'''
#include <torch/extension.h>
#include "exl3_moe_coop.cuh"

void inkling_moe_coop
(
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    int min_expert, int max_expert, int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    float Kg, float Ku, float Kd, bool mcg, bool mul1,
    float gate_cap, float up_cap,
    at::Tensor& had_g, at::Tensor& had_u, at::Tensor& gu_g, at::Tensor& gu_u,
    at::Tensor& act_out, at::Tensor& d_out, at::Tensor& ctr, at::Tensor& out
)
{
    float K_gu, K_d; int cb;
    MoeCoopParams p = exl3_moe_coop_prepare(Hi, g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh,
                                            d_trellis, d_suh, d_svh, {}, {}, {}, Kg, Ku, Kd, mcg, mul1,
                                            MOE_COOP_ACT_SOFTCAP_SWIGLU, gate_cap, true,
                                            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out, {}, K_gu, K_d, cb);
    p.act_limit_u = up_cap;
    p.min_expert = min_expert;
    p.max_expert = max_expert;
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, {});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("inkling_moe_coop", &inkling_moe_coop, "exl3_moe_coop with Inkling's softcapped SwiGLU");
}
'''

_SOFTCAP = ("    if (act == MOE_COOP_ACT_SOFTCAP_SWIGLU)\n"
            "        return tanhf(g / limit) * limit * (1.0f / (1.0f + __expf(-g))) * tanhf(u / limit_u) * limit_u;\n")


def _sub_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"exllamav3 source changed ({what}: {text.count(old)} matches)")
    return text.replace(old, new)


def _patch(src_quant: str, dst: str, ks: set[int]) -> list[str]:
    """Copy and patch the coop sources from exllamav3's quant/ into dst; return the .cu files to compile."""
    os.makedirs(os.path.join(dst, "comp_units"), exist_ok=True)
    rd = lambda n: open(os.path.join(src_quant, n)).read()

    h = rd("exl3_moe_coop.cuh")
    h = _sub_once(h, "#define MOE_COOP_ACT_SILU_OAI 3\n",
                  f"#define MOE_COOP_ACT_SILU_OAI 3\n#define MOE_COOP_ACT_SOFTCAP_SWIGLU {ACT_SOFTCAP_SWIGLU}\n",
                  "activation ids")
    h = _sub_once(h, "    float act_limit;\n", "    float act_limit;\n    float act_limit_u;\n", "params struct")
    open(os.path.join(dst, "exl3_moe_coop.cuh"), "w").write(h)

    kh = rd("exl3_moe_coop_kernel.cuh")
    kh = _sub_once(kh, "float act_gate(int act, bool gated, float g, float u, float limit)\n{\n",
                   "float act_gate(int act, bool gated, float g, float u, float limit, float limit_u)\n{\n" + _SOFTCAP,
                   "act_gate")
    n = len(re.findall(r"act_gate\(p\.act, p\.gated, (\w+), (\w+), p\.act_limit\)", kh))
    if n != 4:
        raise RuntimeError(f"exllamav3 source changed (act_gate calls: {n})")
    kh = re.sub(r"act_gate\(p\.act, p\.gated, (\w+), (\w+), p\.act_limit\)",
                r"act_gate(p.act, p.gated, \1, \2, p.act_limit, p.act_limit_u)", kh)
    open(os.path.join(dst, "exl3_moe_coop_kernel.cuh"), "w").write(kh)

    cu = rd("exl3_moe_coop.cu")
    cu = _sub_once(cu, "act >= 0 && act <= 3", f"act >= 0 && act <= {ACT_SOFTCAP_SWIGLU}", "activation check")
    cu = _sub_once(cu, "    p.act_limit = act_limit;\n", "    p.act_limit = act_limit;\n    p.act_limit_u = 0.0f;\n",
                   "prepare")
    open(os.path.join(dst, "exl3_moe_coop.cu"), "w").write(cu)

    inst = rd(os.path.join("comp_units", "exl3_moe_coop_instances.cuh"))
    open(os.path.join(dst, "comp_units", "exl3_moe_coop_instances.cuh"), "w").write(inst)
    files = [os.path.join(dst, "exl3_moe_coop.cu")]
    # Integer bit widths the model uses get the real instances; the others (and the half-integer rates) are stubs,
    # so the build compiles one kernel set per K actually present instead of eleven
    stub_k = ("MoeCoopKernel exl3_moe_coop_kernel_a_k{K}(int, int, bool) {{ return {{ nullptr, 0 }}; }}\n"
              "MoeCoopKernel exl3_moe_coop_kernel_b_k{K}(int, bool) {{ return {{ nullptr, 0 }}; }}\n")
    stub_h = ("MoeCoopKernel exl3_moe_coop_kernel_a_h{K}(int, bool) {{ return {{ nullptr, 0 }}; }}\n"
              "MoeCoopKernel exl3_moe_coop_kernel_b_h{K}(bool) {{ return {{ nullptr, 0 }}; }}\n")
    for k in range(1, 9):
        f = os.path.join(dst, "comp_units", f"inst_k{k}.cu")
        body = (f'#include "exl3_moe_coop_instances.cuh"\n#include "../exl3_moe_coop_kernel.cuh"\n'
                f"EXL3_MOE_COOP_INST_DEF({k})\n") if k in ks else \
            f'#include "exl3_moe_coop_instances.cuh"\n' + stub_k.format(K=k)
        open(f, "w").write(body)
        files.append(f)
    f = os.path.join(dst, "comp_units", "inst_h.cu")
    open(f, "w").write('#include "exl3_moe_coop_instances.cuh"\n' + "".join(stub_h.format(K=k) for k in (1, 2, 3)))
    files.append(f)
    f = os.path.join(dst, "entry.cu")
    open(f, "w").write(_ENTRY)
    files.append(f)
    # exl3_devctx.cu (device properties singleton) is needed by the launcher; compile a private copy
    shutil.copy(os.path.join(src_quant, "exl3_devctx.cu"), os.path.join(dst, "exl3_devctx.cu"))
    files.append(os.path.join(dst, "exl3_devctx.cu"))
    return files


def load(ks: set[int]):
    """The extension for these integer bit widths, or None (disabled, failed build or sources changed)."""
    if os.environ.get("VLLM_QUANTPACK_INKLING2_COOP", "1") == "0":
        return None
    key = tuple(sorted(ks))
    with _lock:
        if key in _mods:
            return _mods[key]
        mod = None
        try:
            import exllamav3
            from torch.utils.cpp_extension import load as _load
            from exllamav3.util import cuda_flags
            ext_dir = os.path.join(os.path.dirname(exllamav3.__file__), "exllamav3_ext")
            quant = os.path.join(ext_dir, "quant")
            ver = getattr(exllamav3, "__version__", "")
            tag = hashlib.sha1(f"{ver}{key}{_ENTRY}{_SOFTCAP}{open(__file__).read()}".encode()).hexdigest()[:12]
            root = os.path.join(os.environ.get("VLLM_QUANTPACK_CACHE", os.path.expanduser("~/.cache/vllm_quantpack")),
                                f"inkling2_coop_{tag}")
            src = os.path.join(root, "src", "quant")
            files = _patch(quant, src, set(key))
            build = os.path.join(root, "build")
            os.makedirs(build, exist_ok=True)
            log.info("vllm-quantpack: building the Inkling2 fused EXL3 MoE kernel (K %s) in %s", key, root)
            mod = _load(name=f"qp_inkling2_coop_{tag}", sources=files, build_directory=build,
                        extra_include_paths=[quant, ext_dir],
                        extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags.cuda_cflags(),
                        verbose=os.environ.get("VLLM_QUANTPACK_VERBOSE_BUILD") == "1")
        except Exception as e:      # keep serving with the exl3_mgemm path
            log.warning("vllm-quantpack: Inkling2 fused EXL3 MoE kernel unavailable (%s: %s); using exl3_mgemm",
                        type(e).__name__, e)
            mod = None
        _mods[key] = mod
        return mod


class Scratch:
    """Shared per-device buffers of the fused kernel (layers run one after another on the stream)."""

    _by_dev: dict = {}

    def __init__(self, device, hi: int, inter: int, ho: int, topk: int):
        slots = MAX_ROWS * topk
        smax = 4 * slots                       # room for split-k partial rows, as exllamav3's module allocates
        kw = dict(device=device)
        self.had_g = torch.empty((slots, hi), dtype=torch.half, **kw)
        self.had_u = torch.empty_like(self.had_g)
        self.gu_g = torch.empty((smax, 1, inter), dtype=torch.float, **kw)    # fp32 gate/up: the caps act on them
        self.gu_u = torch.empty_like(self.gu_g)
        self.act_out = torch.empty((smax, 1, inter), dtype=torch.half, **kw)
        self.d_out = torch.empty((smax, 1, ho), dtype=torch.float, **kw)
        self.out = torch.zeros((MAX_ROWS, ho), dtype=torch.float, **kw)
        self.ctr = torch.zeros((smax * (inter // 128) + MAX_ROWS * (ho // 128) + 2 * smax + 3,),
                               dtype=torch.int, **kw)

    @classmethod
    def get(cls, device, hi, inter, ho, topk):
        key = (str(device), hi, inter, ho, topk)
        if key not in cls._by_dev:
            cls._by_dev[key] = cls(device, hi, inter, ho, topk)
        return cls._by_dev[key]
