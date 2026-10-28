"""Quantized embedding tables: per-row int8 and grouped int4, pure torch.

Shared by the loader (exl3/embedding.py) and the converter (tools/embed_quant.py), so the table written is the table read.
- int8: `qweight` int8 [V, H], `scales` [V] (one per row): the format of every published pack.
- int4: `qweight` uint8 [V, H/2], two signed nibbles per byte stored as q + 8 (low nibble = even column), `scales`
  [V, H/group] (one per group of `group` columns); symmetric, scale = absmax / 7.
The qweight dtype tells them apart. Both dequantize on the gathered rows only: the table never exists in bf16 on the GPU.
"""
import torch

INT4_GROUP = 128


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.float()
    s = (w.abs().amax(1) / 127).clamp_min(1e-12)
    return torch.round(w / s[:, None]).clamp_(-127, 127).to(torch.int8), s


def quantize_int4(w: torch.Tensor, group: int = INT4_GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    v, h = w.shape
    if h % group or group % 2:
        raise ValueError(f"int4 embedding: width {h} is not a multiple of the group {group} (an even number)")
    g = w.float().reshape(v, h // group, group)
    s = (g.abs().amax(2) / 7).clamp_min(1e-12)
    q = (torch.round(g / s[:, :, None]).clamp_(-8, 7) + 8).to(torch.uint8).reshape(v, h)
    return q[:, 0::2] | (q[:, 1::2] << 4), s


def is_int4(qweight: torch.Tensor) -> bool:
    return qweight.dtype == torch.uint8


def gather(qweight: torch.Tensor, scales: torch.Tensor, ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantized rows `ids` of the table, in `dtype`."""
    rows = qweight[ids]
    if is_int4(rows):
        rows = torch.stack((rows & 0xF, rows >> 4), -1).flatten(-2).to(dtype) - 8
    else:
        rows = rows.to(dtype)
    s = scales.reshape(scales.shape[0], -1)[ids].to(dtype)
    if s.shape[-1] == 1:
        return rows * s
    return (rows.unflatten(-1, (s.shape[-1], -1)) * s.unsqueeze(-1)).flatten(-2)


def dequantize(qweight: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The whole table in `dtype` (a tied lm_head, or a bf16 copy for tools that cannot read the quantized table)."""
    return gather(qweight, scales, torch.arange(qweight.shape[0], device=qweight.device), dtype)
