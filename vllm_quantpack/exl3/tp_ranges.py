"""Tensor-parallel cut ranges for EXL3 tensors (pure, unit-tested in tests/).

The cuts are exllamav3's own (modules/quant/exl3.py, tp_import_split / tp_import_split_n):

* column parallel (qkv, gate_up, in_proj_qkvz, lm_head): suh whole; svh[a:b]; trellis[:, a//16:b//16]. A checkpoint
  tensor that holds several of vLLM's logical partitions (in_proj_qkv = q|k|v, a fused vision qkv) is cut partition by
  partition and the pieces concatenated, so rank r holds [q_r, k_r, v_r] in vLLM's order.
* row parallel (o_proj, down_proj, out_proj): suh[a:b]; svh whole; trellis[a//16:b//16]; vLLM does the all-reduce.
  The input Hadamard works on blocks of 128, so a slice that starts and ends on a multiple of 128 is exact.
* replicated KV heads (tp > num_kv_heads) fall out of the same rule: a partition with n slices goes to rank r // (tp / n).
"""


def logical_parts(shard_id, nparts: int) -> list[int]:
    """Which of vLLM's logical output partitions a checkpoint tensor carries."""
    if shard_id is None:
        return list(range(nparts))              # unfused name, or a pre-fused tensor (vision attn.qkv)
    if isinstance(shard_id, int):
        return [shard_id]
    if isinstance(shard_id, tuple):
        return list(shard_id)                   # in_proj_qkv -> (0, 1, 2)
    return [{"q": 0, "k": 1, "v": 2}[shard_id]]


def col_ranges(full_total: int, parts: list[int], local_sizes: list[int], rank: int, tp: int) -> list[tuple[int, int]]:
    """(start, end) output ranges of this rank inside one checkpoint tensor."""
    if len(parts) == 1:
        full = [full_total]                     # its own tensor: may hold fewer slices than tp (replicated KV)
        logical = local_sizes[parts[0]] * tp
        if 0 < full_total - logical < 128:
            full = [logical]                    # EXL3 padding to a multiple of 128 (vision fc1 4304 -> 4352):
                                                # the extra columns at the end belong to no rank
    else:
        full = [local_sizes[i] * tp for i in parts]
        if sum(full) != full_total:
            raise NotImplementedError(f"tensor of {full_total} outputs does not split as {full} across tp={tp}")
    out, base = [], 0
    for i, f in zip(parts, full):
        p = local_sizes[i]
        n = f // p                              # slices of this partition; < tp when KV heads are replicated
        if f % p or tp % n:
            raise NotImplementedError(f"partition of {f} outputs does not split into slices of {p} for tp={tp}")
        r = rank // (tp // n)
        out.append((base + r * p, base + (r + 1) * p))
        base += f
    return out


def row_range(rank: int, in_local: int) -> tuple[int, int]:
    return rank * in_local, (rank + 1) * in_local


def on_block(ranges, block: int = 128) -> bool:
    return all(a % block == 0 and b % block == 0 for a, b in ranges)
