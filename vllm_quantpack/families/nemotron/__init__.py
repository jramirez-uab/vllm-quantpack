"""NVIDIA Nemotron-3.5-Super-VL (`NemotronH_Omni_Reasoning_V3`).

Stock vLLM 0.31 maps this architecture to NemotronH_Nano_VL_V2, which misses the vision_final_layernorm and the
backbone.* names of an EXL3 checkpoint. Registers the class in model.py in its place. The gateless ReLU² MoE of its
LLM is handled by the EXL3 MoE method (exl3/moe.py), and its MTP head by exl3/spec_decode.py.
"""


def register(logger):
    from vllm import ModelRegistry

    ModelRegistry.register_model("NemotronH_Omni_Reasoning_V3",
                                 "vllm_quantpack.families.nemotron.model:NemotronH_Omni_Reasoning_V3")
    return True
