"""Pure helpers: shard slots, prefix handling, K, tensor-parallel ranges. No GPU, no vLLM."""
import pytest

from vllm_quantpack.exl3.names import expand, manifest_prefixes, norm_prefix, slot, trellis_k
from vllm_quantpack.exl3.tp_ranges import col_ranges, logical_parts, on_block, row_range


def test_slot_maps_every_loader_shard_id_form():
    assert slot(None) == 0
    assert slot(1) == 1
    assert slot((0, 1, 2)) == 0
    assert [slot(s) for s in ("q", "k", "v")] == [0, 1, 2]
    with pytest.raises(ValueError):
        slot("x")


def test_expand_returns_hf_names_of_fused_layers():
    assert expand("model.layers.3.self_attn.qkv_proj") == [
        "model.layers.3.self_attn.q_proj", "model.layers.3.self_attn.k_proj", "model.layers.3.self_attn.v_proj"]
    assert expand("model.layers.3.linear_attn.in_proj_qkvz") == [
        "model.layers.3.linear_attn.in_proj_qkv", "model.layers.3.linear_attn.in_proj_z"]
    assert expand("model.layers.3.mlp.down_proj") == ["model.layers.3.mlp.down_proj"]


def test_norm_prefix_strips_one_tree_segment():
    assert norm_prefix("language_model.model.layers.0.mlp.down_proj") == "layers.0.mlp.down_proj"
    assert norm_prefix("model.layers.0.mlp.down_proj") == "layers.0.mlp.down_proj"
    assert norm_prefix("backbone.layers.0.mixer.in_proj") == "layers.0.mixer.in_proj"
    assert norm_prefix("visual.blocks.0.mlp.fc1") == "visual.blocks.0.mlp.fc1"


def test_manifest_keeps_only_modules_with_a_trellis():
    ts = {
        "model.layers.0.mlp.down_proj": {"stored_tensors": {"model.layers.0.mlp.down_proj.trellis": {}}},
        "model.layers.0.linear_attn.in_proj_a": {"stored_tensors": {"model.layers.0.linear_attn.in_proj_a.weight": {}}},
        "backbone.layers.1.mixer.in_proj": {"stored_tensors": {"backbone.layers.1.mixer.in_proj.trellis": {}}},
    }
    assert manifest_prefixes(ts) == {"layers.0.mlp.down_proj", "layers.1.mixer.in_proj"}
    assert manifest_prefixes({}) == set()


def test_trellis_k_keeps_mul1_half_rates():
    assert trellis_k(64) == 4 and isinstance(trellis_k(64), int)
    assert trellis_k(56) == 3.5


def test_logical_parts():
    assert logical_parts(None, 3) == [0, 1, 2]
    assert logical_parts(1, 2) == [1]
    assert logical_parts((0, 1, 2), 4) == [0, 1, 2]
    assert logical_parts("k", 3) == [1]


def test_col_ranges_single_partition():
    # gate_proj of 17408 outputs, tp=2: each rank keeps half
    assert col_ranges(17408, [0], [8704, 8704], 0, 2) == [(0, 8704)]
    assert col_ranges(17408, [0], [8704, 8704], 1, 2) == [(8704, 17408)]


def test_col_ranges_tensor_holding_several_partitions():
    # in_proj_qkv = q|k|v of 2048 each, tp=2: rank 1 keeps [q_1, k_1, v_1]
    assert col_ranges(6144, [0, 1, 2], [1024, 1024, 1024, 4096], 1, 2) == [(1024, 2048), (3072, 4096), (5120, 6144)]


def test_col_ranges_replicated_kv_heads():
    # k_proj with 2 KV heads of 256 on tp=4: ranks 0,1 share head 0 and ranks 2,3 share head 1
    local = [1536, 256, 256]
    assert [col_ranges(512, [1], local, r, 4) for r in range(4)] == [[(0, 256)], [(0, 256)], [(256, 512)], [(256, 512)]]


def test_col_ranges_padded_single_layer():
    # Groot vision mlp.linear_fc1: 4304 outputs stored padded to 4352, tp=2; the padding columns belong to no rank
    assert [col_ranges(4352, [0], [2152], r, 2) for r in range(2)] == [[(0, 2152)], [(2152, 4304)]]
    with pytest.raises(NotImplementedError):     # 128 extra columns or more is not padding
        col_ranges(4432, [0], [2152], 0, 2)


def test_col_ranges_rejects_a_split_that_does_not_add_up():
    with pytest.raises(NotImplementedError):
        col_ranges(6000, [0, 1, 2], [1024, 1024, 1024], 0, 2)


def test_row_range_and_block_alignment():
    assert row_range(1, 2560) == (2560, 5120)
    assert on_block([(0, 2560), (2560, 5120)])
    assert not on_block([(320, 640)])


def test_flint_interm_div_targets_only_late_layers_of_flint():
    from vllm_quantpack.families.flint.interm_div import kind
    assert kind("model.layers.46.mlp.shared_expert.gate_up_proj", "flint") == "shared_expert.gate_up_proj"
    assert kind("model.layers.48.mlp.shared_expert.down_proj", "flint") == "shared_expert.down_proj"
    assert kind("model.layers.47.mlp.experts", "flint") == "experts"
    assert kind("model.layers.47.mlp.experts.routed", "flint") == "experts"     # FusedMoEFactory child
    assert kind("model.layers.45.mlp.experts", "flint") is None
    assert kind("model.layers.46.self_attn.qkv_proj", "flint") is None
    assert kind("model.layers.46.mlp.experts", "qwen3_5_moe") is None


def test_embedding_prefix_is_claimed_once_inside_its_context():
    from vllm_quantpack.exl3.layers import embedding_prefix, take_embedding_prefix
    assert take_embedding_prefix() is None
    with embedding_prefix("model.language_model.embed_tokens"):
        assert take_embedding_prefix() == "model.language_model.embed_tokens"
        assert take_embedding_prefix() is None          # only the model's own embed_tokens, not a later embedding
    assert take_embedding_prefix() is None


def test_gateless_moe_is_detected_from_the_stored_projections():
    from vllm_quantpack.exl3.names import gateless_up_key
    gated = {(e, p, "trellis"): 0 for e in range(2) for p in ("gate_proj", "up_proj", "down_proj")}
    relu2 = {(e, p, "trellis"): 0 for e in range(2) for p in ("up_proj", "down_proj")}
    drafter = {(e, p, "trellis"): 0 for e in range(2) for p in ("gate_proj", "down_proj")}   # up loaded into w1
    assert gateless_up_key(gated, range(2)) is None
    assert gateless_up_key(relu2, range(2)) == "up_proj"
    assert gateless_up_key(drafter, range(2)) == "gate_proj"


def test_expert_store_key_maps_fused_parameter_names():
    from vllm_quantpack.exl3.names import expert_store_key
    assert expert_store_key("mtp.layers.0.mixer.experts.w13_trellis", "w1", 5) == (5, "gate_proj", "trellis")
    assert expert_store_key("experts.w13_suh", "w3", "7") == (7, "up_proj", "suh")
    assert expert_store_key("experts.w2_svh", "w2", 0) == (0, "down_proj", "svh")
    assert expert_store_key("experts.w2_weight", "w2", 0) is None
    assert expert_store_key("experts.e_score_correction_bias", None, 0) is None
