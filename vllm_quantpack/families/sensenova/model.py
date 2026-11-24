# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text inference for fixed-count SenseNova Loop checkpoints with optional mHC."""

import re
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm_quantpack.families.sensenova.mhc import MHCRouter
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ProcessingInfo,
    Qwen3_5RMSNorm,
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    build_mm_input_norm,
    cached_tokenizer_from_config,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix


class SenseNovaLoopAWQEmbeddingMethod(QuantizeMethodBase):
    """Decode the row-major INT4 embedding for the requested tokens."""

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if input_size % (8 * self.group_size):
            raise ValueError("SenseNova Loop AWQ embedding dimensions are misaligned")
        num_rows = sum(output_partition_sizes)
        num_groups = input_size // self.group_size
        for name, width, dtype in (
            ("qweight", input_size // 8, torch.int32),
            ("qzeros", num_groups // 8, torch.int32),
            ("scales", num_groups, torch.float32),
        ):
            param = nn.Parameter(
                torch.empty(num_rows, width, dtype=dtype), requires_grad=False
            )
            set_weight_attrs(param, {"output_dim": 0, **extra_weight_attrs})
            layer.register_parameter(name, param)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        shifts = torch.arange(0, 32, 4, device=input_.device, dtype=torch.int32)

        def unpack(packed: torch.Tensor, width: int) -> torch.Tensor:
            words = F.embedding(input_, packed)
            return ((words.unsqueeze(-1) >> shifts) & 15).reshape(*input_.shape, width)

        groups = layer.embedding_dim // self.group_size
        weights = unpack(layer.qweight, layer.embedding_dim).unflatten(
            -1, (groups, self.group_size)
        )
        zeros = unpack(layer.qzeros, groups)
        scales = F.embedding(input_, layer.scales)
        return (
            ((weights - zeros.unsqueeze(-1)) * scales.unsqueeze(-1))
            .flatten(-2)
            .to(layer.params_dtype)
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SenseNova Loop AWQ embeddings are lookup-only")


class SenseNovaLoopDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str) -> None:
        super().__init__(vllm_config, layer_type, prefix)
        config = vllm_config.model_config.hf_text_config
        self.physical_index = config.loop_physical_indices[self.layer_idx]
        self.mhc = config.mhc
        self.mhc_active = self.mhc is not None and (
            self.mhc["start"] <= self.physical_index < self.mhc["end"]
        )
        if self.mhc_active:
            self.mhc_attn = MHCRouter(config.hidden_size, self.mhc)
            self.mhc_ffn = MHCRouter(config.hidden_size, self.mhc)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.mhc_active:
            return super().forward(hidden_states, residual, positions, **kwargs)
        if residual is not None:
            hidden_states = hidden_states + residual
        streams = self.mhc["streams"]
        if self.physical_index == self.mhc["start"]:
            hidden_states = hidden_states.unsqueeze(-2).expand(-1, streams, -1)
        else:
            hidden_states = hidden_states.unflatten(-1, (streams, -1))
        branch, post, mix = self.mhc_attn(hidden_states)
        branch = self.input_layernorm(branch.contiguous())
        if self.layer_type == "linear_attention":
            branch = self.linear_attn(hidden_states=branch)
        else:
            branch = self.self_attn(hidden_states=branch, positions=positions)
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        branch, post, mix = self.mhc_ffn(hidden_states)
        branch = self.mlp(self.post_attention_layernorm(branch.contiguous()))
        hidden_states = MHCRouter.merge(hidden_states, branch, post, mix)
        if self.physical_index == self.mhc["end"] - 1:
            hidden_states = hidden_states.mean(-2)
        else:
            hidden_states = hidden_states.flatten(-2)
        return hidden_states, None


_LAYER_PREFIX = re.compile(
    r"^((?:language_model\.)?model\.(?:language_model\.)?layers\.)(\d+)(\..*)$"
)


def _is_exl3(quant_config) -> bool:
    return quant_config is not None and quant_config.get_name() == "exl3"


def _route_exl3_prefixes(quant_config, physical: list[int]) -> None:
    """EXL3 picks each layer's method from the pack's manifest by prefix. The manifest only knows physical layers,
    so execution layer i (second loop pass included) is looked up as its physical layer."""
    if getattr(quant_config, "_sensenova_loop_routed", False):
        return
    orig = quant_config.get_quant_method

    def get_quant_method(layer, prefix):
        m = _LAYER_PREFIX.match(prefix)
        if m and int(m[2]) < len(physical):
            prefix = f"{m[1]}{physical[int(m[2])]}{m[3]}"
        return orig(layer, prefix)

    quant_config.get_quant_method = get_quant_method
    quant_config._sensenova_loop_routed = True


class SenseNovaLoopModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        embedding_method = None
        if isinstance(self.quant_config, AutoAWQConfig):
            embedding = self.quant_config.full_config.get("embedding")
            if embedding is not None:
                group_size = embedding.get("group_size")
                if (
                    embedding.get("layout") != "row_major"
                    or embedding.get("bits") != 4
                    or embedding.get("zero_point") is not True
                    or type(group_size) is not int
                    or group_size <= 0
                    or self.config.tie_word_embeddings
                ):
                    raise ValueError("Unsupported SenseNova Loop AWQ embedding config")
                embedding_method = SenseNovaLoopAWQEmbeddingMethod(group_size)
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.config.hidden_size,
            quant_method=embedding_method,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            SenseNovaLoopDecoderLayer(vllm_config, layer_type, f"{prefix}.layers.{i}")
            for i, layer_type in enumerate(self.config.layer_types)
        )
        self.norm = Qwen3_5RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        if _is_exl3(self.quant_config):
            self.share_loop_linears()

    def share_loop_linears(self) -> None:
        """EXL3 builds its kernels' tensors in process_weights_after_loading as plain attributes, which
        share_loop_parameters cannot reach. Give every later visit of a physical layer the first visit's linear
        modules instead, so each EXL3 tensor exists once; attention and recurrent state stay per visit."""
        first: dict[int, nn.Module] = {}
        for physical, layer in zip(self.config.loop_physical_indices, self.layers):
            if physical not in first:
                first[physical] = layer
                continue
            for name, module in list(first[physical].named_modules()):
                if isinstance(module, LinearBase):
                    parent, _, leaf = name.rpartition(".")
                    setattr(layer.get_submodule(parent), leaf, module)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, positions)
        if residual is None:
            return self.norm(hidden_states)
        return self.norm(hidden_states, residual)[0]

    def share_loop_parameters(self) -> None:
        first: dict[int, nn.Module] = {}
        for physical, layer in zip(self.config.loop_physical_indices, self.layers):
            if physical not in first:
                first[physical] = layer
                continue
            for name, param in first[physical].named_parameters():
                parent, _, leaf = name.rpartition(".")
                setattr(layer.get_submodule(parent), leaf, param)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=Qwen3_5Model.hf_to_vllm_mapper
        )


class SenseNovaLoopForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE):
    packed_modules_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_5ForCausalLMBase.get_mamba_state_copy_func()

    get_mrope_input_positions = Qwen3_5ForCausalLMBase.get_mrope_input_positions

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = self.model_config.hf_text_config
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.lora_config is not None
            or getattr(self.config, "layer_scale", False)
        ):
            raise ValueError(
                "SenseNova Loop currently requires inference "
                "without pipeline parallelism, LoRA or layer_scale"
            )
        self.quant_config = vllm_config.quant_config
        if self.quant_config is not None and not (
            isinstance(self.quant_config, AutoAWQConfig) or _is_exl3(self.quant_config)
        ):
            raise ValueError(
                "SenseNova Loop supports only unquantized, AWQ and EXL3 weights"
            )
        if _is_exl3(self.quant_config):
            _route_exl3_prefixes(self.quant_config, self.config.loop_physical_indices)
        self.scheduler_config = vllm_config.scheduler_config
        self.model = SenseNovaLoopModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = None  # no pipeline parallelism

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        destinations: dict[int, list[int]] = {}
        for execution, physical in enumerate(self.config.loop_physical_indices):
            destinations.setdefault(physical, []).append(execution)

        def expanded_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                name = name.replace("model.language_model.", "model.", 1)
                if name == "model.loop_gate.logits":
                    continue
                if name.startswith(("model.visual.", "visual.", "mtp.")):
                    continue
                if name.startswith("model.layers."):
                    _, _, index, suffix = name.split(".", 3)
                    for execution in destinations[int(index)]:
                        yield f"model.layers.{execution}.{suffix}", weight
                else:
                    yield name, weight

        return AutoWeightsLoader(self).load_weights(expanded_weights())

    def process_weights_after_loading(self) -> None:
        # Parameters share storage; attention modules and their caches stay distinct.
        self.model.share_loop_parameters()


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class SenseNovaLoopForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """SenseNova Loop with its Qwen3.5-VL vision tower: the Qwen3.5 multimodal wrapper (vision encoder, processor,
    M-RoPE positions, embedding merge) around the loop text model. MTP is served by vLLM's Qwen3_5MTP draft
    (speculative method "mtp"), which reads the checkpoint's mtp.* weights on its own."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.get_multimodal_config()

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                input_norm=build_mm_input_norm(self.model_config),
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = SenseNovaLoopForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = None

    get_mamba_state_dtype_from_config = SenseNovaLoopForCausalLM.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = SenseNovaLoopForCausalLM.get_mamba_state_shape_from_config
    get_mamba_state_copy_func = SenseNovaLoopForCausalLM.get_mamba_state_copy_func

    def process_weights_after_loading(self) -> None:
        self.language_model.process_weights_after_loading()
