"""NVIDIA Nemotron-3.5-Super-VL (`NemotronH_Omni_Reasoning_V3`, model_type `nemotron_h_omni`) for EXL3 checkpoints.

Stock vLLM maps this architecture to NemotronH_Nano_VL_V2 (same composition: RADIO ViT + mlp1 + NemotronH LLM).
Differences handled here:
  * the config carries `llm_config` instead of `text_config`: aliased before the base __init__;
  * the EXL3 LLM weights are named backbone.* / lm_head.*, while the base class routes LLM weights by the
    "language_model." prefix: load_weights rewrites backbone.* -> language_model.backbone.* (the text model's own
    mapper then does backbone. -> model.) and drops mtp.* (the MTP drafter loads them separately);
  * a vision_final_layernorm (LayerNorm on the ViT output, before the pixel shuffle) whose weights are in the
    checkpoint (vision_projector.vision_final_layernorm.*): applied in extract_feature / extract_feature_dynamic
    without wrapping the ViT, so vLLM's weight-name matching is unchanged;
  * no audio in this checkpoint.
"""
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLDummyInputsBuilder,
    NanoNemotronVLMultiModalProcessor,
    NanoNemotronVLProcessingInfo,
    NemotronH_Nano_VL_V2,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

logger = init_logger("vllm.quantpack.nemotron")


@MULTIMODAL_REGISTRY.register_processor(
    NanoNemotronVLMultiModalProcessor,
    info=NanoNemotronVLProcessingInfo,
    dummy_inputs=NanoNemotronVLDummyInputsBuilder,
)
class NemotronH_Omni_Reasoning_V3(NemotronH_Nano_VL_V2):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"language_model": "language_model.model"},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        cfg = vllm_config.model_config.hf_config
        if getattr(cfg, "text_config", None) is None and getattr(cfg, "llm_config", None) is not None:
            cfg.text_config = cfg.llm_config
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        vc = self.config.vision_config
        self.vision_final_layernorm = nn.LayerNorm(
            self.config.vit_hidden_size, eps=vc.layer_norm_eps
        ).to(self.llm_dtype)

    def extract_feature_dynamic(self, pixel_values, imgs_sizes):
        _, vit_embeds = self.vision_model(pixel_values, imgs_sizes=imgs_sizes)
        vit_embeds = self.vision_final_layernorm(vit_embeds)
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
        vit_embeds = self.pixel_shuffle_dynamic_res(vit_embeds, imgs_sizes=imgs_sizes)
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def extract_feature(self, pixel_values, num_frames=None):
        N, _C, H, W = pixel_values.shape
        T = self.video_temporal_patch_size if num_frames is not None else 1
        micro_batch_size = 128 - (128 % T)
        patch_size = self.patch_size
        H_patches = H // patch_size
        W_patches = W // patch_size

        vit_embeds_list = []
        for i in range(0, N, micro_batch_size):
            chunk = pixel_values[i : i + micro_batch_size]
            if num_frames is not None and T > 1:
                _, vit_embeds = self.vision_model(chunk, num_frames=chunk.shape[0])
            else:
                _, vit_embeds = self.vision_model(chunk)
            vit_embeds = self.vision_final_layernorm(vit_embeds)
            vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], H_patches, W_patches, -1
            )
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.downsample_ratio
            )
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], -1, vit_embeds.shape[-1]
            )
            vit_embeds = self.mlp1(vit_embeds)
            vit_embeds_list.append(vit_embeds)

        vit_embeds = torch.cat(vit_embeds_list, dim=0)
        return vit_embeds

    def load_weights(self, weights):
        ln_w = ln_b = None
        rest = []
        for name, w in weights:
            if name.startswith("vision_projector.vision_final_layernorm."):
                if name.endswith(".weight"):
                    ln_w = w
                elif name.endswith(".bias"):
                    ln_b = w
            elif name.startswith("mtp."):
                continue
            elif name.startswith("backbone.") or name.startswith("lm_head."):
                rest.append(("language_model." + name, w))
            else:
                rest.append((name, w))
        super().load_weights(iter(rest))
        if ln_w is None:
            raise KeyError("vllm-quantpack (Nemotron): vision_projector.vision_final_layernorm.weight missing")
        with torch.no_grad():
            self.vision_final_layernorm.weight.copy_(ln_w)
            if ln_b is not None:
                self.vision_final_layernorm.bias.copy_(ln_b)
