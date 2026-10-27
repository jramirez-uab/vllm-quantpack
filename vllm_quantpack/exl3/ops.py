"""The EXL3 shard GEMM, wrapped as a PyTorch custom op.

Origin: OrcaSAQ2 ops.py (Apache-2.0). vLLM compiles the model with torch.compile and Dynamo cannot trace a raw pybind11
function (it has no source file to inline), so the engine died at startup with "Attempted to call function marked as
skipped ... exllamav3_ext". A custom op tells Dynamo "this is opaque, here is the shape it returns": it stays one node in
the graph and everything around it still compiles, CUDA graphs included.

The whole shard is one op rather than one op per extension call, so the row-count dispatch below stays ordinary eager
Python at runtime instead of becoming a graph-level branch.
"""
import os

import torch
from torch import Tensor

# exllamav3's own dispatch point (exl3.py: AUTO_RECONSTRUCT_THRESHOLD). At or below it the fused kernel reads the packed
# trellis directly; above it, rebuilding the weight once and handing it to cuBLAS is cheaper than one fused GEMM per row.
GEMM_MAX_ROWS = int(os.environ.get("QUANTPACK_EXL3_GEMM_MAX_ROWS", os.environ.get("ORCA_EXL3_GEMM_MAX_ROWS", "144")))


@torch.library.custom_op("quantpack::exl3_gemm", mutates_args=())
def exl3_gemm(x: Tensor, trellis: Tensor, suh: Tensor, svh: Tensor,
              oc: int, k: float, mcg: bool, mul1: bool) -> Tensor:
    # k is declared float because the mul1 half rates are 1.5/2.5/3.5; the extension takes either, but a custom-op schema
    # has to pick one type, so narrow it back here.
    from exllamav3.ext import exllamav3_ext as ext
    if float(k).is_integer():
        k = int(k)
    rows, ic = x.shape
    y = torch.empty((rows, oc), dtype=torch.half, device=x.device)
    if rows <= GEMM_MAX_ROWS:
        # (A, trellis, C, suh, A_had_scratch, svh, kernel, mcg, mul1, acc_mode); kernel -1 lets the extension choose.
        ext.exl3_gemm(x, trellis, y, suh, torch.empty_like(x), svh, -1, mcg, mul1, 0)
    else:
        # Reconstruct into a SCRATCH buffer, freed on exit: keeping it resident would undo the compression.
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        w = torch.empty((ic, oc), dtype=torch.half, device=x.device)
        ext.reconstruct(w, trellis, k, mcg, mul1)
        ext.hgemm_recon(xh, w, y)
        del w, xh
        ext.had_r_128(y, y, None, svh, 1.0)
    return y


@exl3_gemm.register_fake
def _exl3_gemm_fake(x: Tensor, trellis: Tensor, suh: Tensor, svh: Tensor,
                    oc: int, k: float, mcg: bool, mul1: bool) -> Tensor:
    return x.new_empty((x.shape[0], oc), dtype=torch.half)
