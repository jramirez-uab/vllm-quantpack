# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# From vLLM PR #64832 (vllm/model_executor/models/config.py), registered by the plugin when vLLM lacks it.
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig, VerifyAndUpdateConfig


class SenseNovaLoopVerifyConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        from vllm.config.compilation import CompilationMode, CUDAGraphMode
        from vllm.platforms import current_platform

        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        if vllm_config.load_config.load_format in (
            "sharded_state",
            "runai_streamer_sharded",
        ):
            raise NotImplementedError(
                "SenseNova Loop does not support loading sharded state "
                "with shared loop weights; load the original checkpoint instead"
            )
        offload = vllm_config.offload_config
        if offload.uva.cpu_offload_gb > 0 or offload.prefetch.offload_group_size > 0:
            raise NotImplementedError(
                "SenseNova Loop does not support CPU offload with shared loop weights"
            )
        if current_platform.is_cuda():
            compilation_config = vllm_config.compilation_config
            if compilation_config.mode is None:
                compilation_config.mode = CompilationMode.NONE
            if compilation_config.cudagraph_mode is None:
                compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

