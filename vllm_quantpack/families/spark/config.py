"""Spark-X2.5 config for transformers 5 (the checkpoint's configuration_spark.py is written for 4.57)."""
from transformers import PretrainedConfig


class Spark2_5Config(PretrainedConfig):
    model_type = "spark2_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        hidden_act="gelu",
        max_position_embeddings=1048576,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        headwise_attn_output_gate=True,
        gate_attn_act_mode="sigmoid",
        sliding_window=512,
        layer_types=None,
        rope_parameters=None,
        tie_word_embeddings=True,
        pad_token_id=2,
        bos_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        # transformers 5 validates RoPE inside PretrainedConfig.__init__ (and resets layer_types there): call it first,
        # then set the model attributes, so the per-layer-type rope_parameters are never validated as a flat dict.
        kwargs.pop("use_cache", None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.gate_attn_act_mode = gate_attn_act_mode
        self.sliding_window = sliding_window
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        # Per-layer-type RoPE, each entry made explicit ("rope_type": "default") so transformers 5 validates it.
        rope = rope_parameters or {"full_attention": {"rope_theta": 10000.0}}
        self.rope_parameters = {lt: {"rope_type": "default", **dict(p)} for lt, p in rope.items()}

    def rope_for(self, layer_type):
        p = self.rope_parameters.get(layer_type) or self.rope_parameters.get("full_attention") or {}
        return dict(p)
