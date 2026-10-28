# vllm-quantpack

**Experimental, early stage.** A vLLM general plugin that serves EXL3 (exllamav3 trellis) checkpoints, and the
model-family fixes they need, in stock vLLM: no patch to vLLM, no per-model plugin.

## What it does

- Registers `quant_method: "exl3"`, so `vllm serve <repo>` loads EXL3 packs with no extra flag:
  - linear layers at any per-tensor bitrate, including fused layers whose shards have different bitrates;
  - `lm_head` as an EXL3 tensor (`head_bits`) or dense;
  - `embed_tokens` as int8 rows with a per-row scale, dequantized on the gathered rows;
  - routed MoE experts on exllamav3's fused kernel, capturable in CUDA graphs;
  - tensor parallelism for dense layers (`-tp N`) and expert parallelism for MoE (`-tp N --enable-expert-parallel`).
- Model-family fixes:
  - **Qwen 3.5 / 3.6 / 3.8**: the vision tower's duplicated attention tensors;
  - **Qwen3.8-Flash-Next**: the FP8 PLE lookup on GPUs below sm_89.

## Requirements

- vLLM 0.31 (0.30 also works).
- exllamav3 1.6.5, as the prebuilt wheel that matches your torch and CUDA. The prebuilt wheels are on the
  [exllamav3 releases](https://github.com/turboderp-org/exllamav3/releases) page; PyPI only has a generic wheel that
  compiles the CUDA kernels on first use (it needs a CUDA toolkit and takes a while). For vLLM 0.31 (torch 2.13.0, CUDA 13):
  ```
  W=exllamav3-1.6.5+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
  curl -L -o /tmp/$W "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.5/${W/+/%2B}"
  pip install /tmp/$W
  ```
  For other versions: `python -c "import torch; print(torch.__version__, torch.version.cuda)"` and pick the matching asset.
- An NVIDIA GPU of compute capability 8.0 or newer (Ampere, Ada, Hopper, Blackwell).
- FlashInfer's prebuilt kernels in the same environment, unless you have a CUDA toolkit: without them vLLM looks for
  `nvcc` at start and fails with `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`. For vLLM 0.31:
  ```
  pip install flashinfer-cubin==0.7.0.post1 flashinfer-jit-cache==0.7.0.post1+cu130 \
    --extra-index-url https://flashinfer.ai/whl/ --extra-index-url https://flashinfer.ai/whl/cu130/
  ```
- The Python headers and a C compiler (`sudo apt-get install python3.12-dev build-essential` on Debian/Ubuntu). Triton
  compiles a small C helper every time vLLM starts; with a system Python and no headers it stops with
  `fatal error: Python.h: No such file or directory`.

Each model card on Hugging Face has the full install block that was tested for that checkpoint.

## Install

```
pip install vllm-quantpack
```

It replaces OrcaSAQ2, `qwen38_exl3_vision`, `flint_vllm` and `spark_vllm`. If any of them is installed in the same
environment, uninstall it, or load only this plugin with `VLLM_PLUGINS=quantpack`.

## Status

| Part | State |
|---|---|
| EXL3 linear, int8 embedding, lm_head | validated: same scores as the previous stack (dense Qwen 27B) |
| EXL3 MoE | validated on one GPU (Groot 19B-A3B): same as the previous stack within run-to-run noise |
| Tensor / expert parallelism | validated with MoE TP2 + EP (Groot 19B-A3B); Flash-Next TP not tested yet |
| Qwen vision, Flash-Next PLE | validated (Kestrel, Groot, Flash-Next 2.25) |
| Flint architecture | validated (Flint 3.88): same as flint_vllm within run-to-run noise |
| GPTQ with int4/int8 embeddings (Qwen3.5 family, vLLM <= 0.31) | validated: loads on unpatched vLLM 0.31, bit-identical to the patched vLLM in eager mode |
| Spark architecture | served (Spark-X2.5 4B) |
| Nemotron-3.5-Super-VL (gateless ReLU² MoE, MTP, vision) | served (2.25 bpw, one 96 GB GPU) |
| Every published JRamirez-UAB checkpoint | each served with its card's Quick start command on a clean vLLM 0.31 + exllamav3 1.5.1 install: text, image and tool call answered |

## Credits

Includes code from [OrcaSAQ2](https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel) (Apache-2.0). The EXL3 format and
kernels are [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT), a streamlined variant of
[QTIP](https://github.com/Cornell-RelaxML/qtip). See NOTICE.

## License

Apache-2.0.
