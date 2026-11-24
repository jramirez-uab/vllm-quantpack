"""Pure helpers for EXL3 tensor names and shapes (no torch or vLLM imports, unit-tested in tests/).

Origin: OrcaSAQ2 (github.com/Continuum-AI-Corp/OrcaSAQ2-kernel, Apache-2.0) config.py / ops.py, plus the prefix
normalisation of the patch shipped with our published checkpoints (2026-10-18).
"""

# vLLM fuses these; each constituent is its own EXL3 tensor, so a fused layer carries a list of shards. Order matters:
# it must match the order vLLM concatenates outputs in.
SHARD_IDS = {
    "qkv_proj": ["q", "k", "v"],
    "gate_up_proj": [0, 1],
    "in_proj_qkvz": [0, 1],
    "qkvr": [0, 1, 2, 3, 4, 5],
}

# Fused vLLM name -> the HF names its shards came from.
FUSED_TO_HF = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    # Inkling2 attention (vLLM models/inkling2, model.py _shards); attn_sink_proj stays BF16 in an EXL3 pack
    "qkvr": ["wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv"],
}

# Multimodal models add tree segments ("language_model.model.") that the pack manifest does not have ("backbone.",
# "model."); both sides are compared without them. Inkling2 packs keep the checkpoint tree "model.llm." that vLLM
# renames to "model.", so it is stripped before "model.".
_LEADS = ("language_model.model.", "language_model.backbone.", "model.llm.", "model.", "backbone.")


def slot(shard_id) -> int:
    """Dict slot of a loader shard id: 0 for an unfused layer, the index inside its fused layer otherwise."""
    if shard_id is None:
        return 0
    if isinstance(shard_id, int):
        return shard_id
    if isinstance(shard_id, tuple):           # consecutive-range form
        return shard_id[0]
    for ids in SHARD_IDS.values():
        if shard_id in ids:
            return ids.index(shard_id)
    raise ValueError(f"unknown shard id {shard_id!r}")


def expand(prefix: str) -> list[str]:
    """Fused vLLM prefix -> the HF prefixes of its shards (the prefix itself when it is not fused)."""
    leaf = prefix.rsplit(".", 1)[-1]
    if leaf in FUSED_TO_HF:
        return [prefix[: -len(leaf)] + m for m in FUSED_TO_HF[leaf]]
    return [prefix]


def norm_prefix(prefix: str) -> str:
    for lead in _LEADS:
        if prefix.startswith(lead):
            return prefix[len(lead):]
    return prefix


def mixed_slots(trellis_oc: dict, dense_oc: dict, out_sizes: list) -> list:
    """Plan of a fused layer whose shards are not all of one kind, one entry per output partition, in order:
    ("t", slot) an EXL3 shard, ("d", slot) a dense weight, ("z", width) zero columns. Only the LAST partition may be
    missing (the alignment pad of Inkling2's qkvr, which no checkpoint tensor fills); every shard must have its
    partition's width. trellis_oc / dense_oc map slot -> output columns."""
    plan = []
    for s, width in enumerate(out_sizes):
        if s in trellis_oc or s in dense_oc:
            kind, got = ("t", trellis_oc[s]) if s in trellis_oc else ("d", dense_oc[s])
            if got != width:
                raise ValueError(f"shard {s} has {got} output columns, its partition has {width}")
            plan.append((kind, s))
        elif s == len(out_sizes) - 1:
            plan.append(("z", width))
        else:
            raise ValueError(f"partition {s} of {len(out_sizes)} has no shard (only the last one may be a pad)")
    extra = set(trellis_oc) | set(dense_oc)
    if extra - set(range(len(out_sizes))):
        raise ValueError(f"shards {sorted(extra - set(range(len(out_sizes))))} beyond the {len(out_sizes)} partitions")
    return plan


def manifest_prefixes(tensor_storage: dict) -> set[str]:
    """Normalised prefixes of the modules that carry a trellis, from quantization_config.json's tensor_storage."""
    out = set()
    for k, v in (tensor_storage or {}).items():
        if any(n.endswith(".trellis") for n in (v.get("stored_tensors") or {})):
            out.add(norm_prefix(k))
    return out


def trellis_k(last_dim: int):
    """Steps per 16-weight tile. The mul1 half rates (1.5/2.5/3.5) are NOT integers: a 3.5-bit tensor packs 56 int16 per
    tile, and 56 // 16 silently gives 3, which the kernel rejects. exllamav3 keeps the float and the kernel accepts it."""
    k = last_dim / 16
    return int(k) if float(k).is_integer() else k


# Tensor suffixes of one EXL3 expert projection (moe.py).
EXPERT_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")


def gateless_up_key(store, experts):
    """ReLU² MoE without gate (NemotronH, mlp_hidden_act relu2): the checkpoint has only up_proj and down_proj.
    Returns the key the up projection is stored under, or None for a gated (SiLU) MoE. The MTP drafter's
    expert_params_mapping loads up_proj into the gate slot (w1), so its store can hold gate_proj without up_proj."""
    has = {p: all((e, p, "trellis") in store for e in experts) for p in ("gate_proj", "up_proj")}
    if has["up_proj"] != has["gate_proj"]:
        return "up_proj" if has["up_proj"] else "gate_proj"
    return None


def expert_store_key(name_mapped, shard_id, expert_id):
    """Store key for a tensor loaded through param.weight_loader: experts.w13_<suffix> / experts.w2_<suffix>."""
    w, _, suffix = name_mapped.rsplit(".", 1)[-1].partition("_")
    if suffix not in EXPERT_SUFFIXES:
        return None
    proj = {"w13": "gate_proj" if shard_id == "w1" else "up_proj", "w2": "down_proj"}.get(w)
    return None if proj is None else (int(expert_id), proj, suffix)
