"""Native vLLM (0.31) implementation of XHToken Spark-X2.5 (`Spark2_5ForCausalLM`).

Architecture (from the checkpoint's modeling_spark.py):
  * decoder-only, pre-norm RMSNorm, GELU-gated MLP (gate_proj / up_proj / down_proj, exact GELU);
  * attention with a fused q_k_v_proj (BF16 checkpoint) or q/k/v_proj (EXL3 pack), GQA, head_dim 256,
    a head-wise sigmoid output gate (g_proj: hidden -> num_heads) and out_proj;
  * hybrid layers (`layer_types`): sliding-window attention (window `sliding_window`) with full RoPE, and full attention
    with partial RoPE (`partial_rotary_factor`), each type with its own rope_theta; neox-style rotation;
  * tied embeddings (`model.embedding` in BF16, `model.embed_tokens` in the EXL3 pack, which also carries a quantized lm_head).
The reference keeps the residual stream in float32; here it follows vLLM's usual dtype (bf16/fp16), a small numeric difference.
"""
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors


class Spark2_5MLP(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=config.mlp_bias,
            quant_config=quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias,
            quant_config=quant_config, prefix=f"{prefix}.down_proj")
        if config.hidden_act != "gelu":
            raise ValueError(f"Spark-X2.5 uses hidden_act='gelu', got {config.hidden_act!r}")
        self.act_fn = get_act_and_mul_fn("gelu")  # exact (erf) GELU, as ACT2FN["gelu"] in the reference

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class Spark2_5Attention(nn.Module):
    def __init__(self, config, cache_config: CacheConfig | None, quant_config: QuantizationConfig | None, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp == 0
        self.num_heads = self.total_num_heads // tp
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads,
            bias=config.attention_bias, quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.out_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias,
            quant_config=quant_config, prefix=f"{prefix}.out_proj")
        # Head-wise output gate. Small and left unquantized by exllamav3 (fp16 in the pack): keep it dense.
        self.g_proj = ReplicatedLinear(
            config.hidden_size, self.total_num_heads, bias=config.attention_bias,
            quant_config=None, prefix=f"{prefix}.g_proj") if config.headwise_attn_output_gate else None
        if config.gate_attn_act_mode not in ("sigmoid", "silu"):
            raise ValueError(f"unsupported gate_attn_act_mode {config.gate_attn_act_mode!r}")
        self.gate_act = torch.sigmoid if config.gate_attn_act_mode == "sigmoid" else nn.functional.silu
        if tp != 1:
            raise NotImplementedError("vllm-quantpack (Spark): tensor parallelism not implemented (the model fits on one GPU)")

        layer_type = config.layer_types[extract_layer_index(prefix)]
        self.sliding = layer_type == "sliding_attention"
        rope = config.rope_for(layer_type) if hasattr(config, "rope_for") else dict(config.rope_parameters[layer_type])
        rope.setdefault("rope_type", "default")
        self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings,
                                   is_neox_style=True, rope_parameters=rope)
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=cache_config, quant_config=quant_config,
            per_layer_sliding_window=config.sliding_window if self.sliding else None,
            prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        if self.g_proj is not None:
            g, _ = self.g_proj(hidden_states)
            gate = self.gate_act(g.float()).to(out.dtype)
            out = (out.view(-1, self.num_heads, self.head_dim) * gate.unsqueeze(-1)).view(-1, self.q_size)
        out, _ = self.out_proj(out)
        return out


class Spark2_5DecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str):
        super().__init__()
        self.self_attn = Spark2_5Attention(config, cache_config, quant_config, prefix=f"{prefix}.self_attn")
        self.mlp = Spark2_5MLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


_STACKED = {
    ".q_proj": (".qkv_proj", "q"),
    ".k_proj": (".qkv_proj", "k"),
    ".v_proj": (".qkv_proj", "v"),
    ".gate_proj": (".gate_up_proj", 0),
    ".up_proj": (".gate_up_proj", 1),
}


@support_torch_compile
class Spark2_5Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.embedding.": "model.embed_tokens."},
                                      orig_to_new_stacked=_STACKED)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                   quant_config=quant_config, prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Spark2_5DecoderLayer(config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _split_fused_qkv(weights, config):
    """BF16 checkpoint: split self_attn.q_k_v_proj.* into q/k/v_proj.* so the stacked mapping can load it."""
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    for name, w in weights:
        if ".q_k_v_proj." in name:
            for part, t in zip(("q_proj", "k_proj", "v_proj"), w.split([q, kv, kv], dim=0)):
                yield name.replace("q_k_v_proj", part), t
        else:
            yield name, w


class Spark2_5ForCausalLM(nn.Module, SupportsPP):
    hf_to_vllm_mapper = Spark2_5Model.hf_to_vllm_mapper
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = Spark2_5Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        # An EXL3 pack with head_bits < 16 ships its own quantized lm_head (what exllamav3 runs and what the KLD measures):
        # use it. Otherwise (BF16 checkpoint) tie to the embeddings as the reference does.
        exl3_head = (quant_config is not None and quant_config.get_name() == "exl3"
                     and float(getattr(quant_config, "head_bits", 16)) < 16)
        self.tied = bool(config.tie_word_embeddings) and not exl3_head
        if self.tied:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."] if self.tied else None)
        return loader.load_weights(_split_fused_qkv(weights, self.config), mapper=self.hf_to_vllm_mapper)
