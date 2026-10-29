# vllm-quantpack

**Experimental, early stage.** A vLLM general plugin that serves EXL3 (exllamav3 trellis) checkpoints, and the
model-family fixes they need, in stock vLLM: no patch to vLLM, no per-model plugin.

## What it does

- Registers `quant_method: "exl3"`, so `vllm serve <repo>` loads EXL3 packs with no extra flag:
  - linear layers at any per-tensor bitrate, including fused layers whose shards have different bitrates;
  - `lm_head` as an EXL3 tensor (`head_bits`) or dense;
  - `embed_tokens` as int8 rows with a per-row scale, or int4 with a scale per group of 128 columns, dequantized on the gathered rows
    (convert a pack's bf16 table with `python -m vllm_quantpack.tools.embed_quant SRC DST --bits {4,8}`);
  - routed MoE experts on exllamav3's fused kernel, capturable in CUDA graphs;
  - tensor parallelism for dense layers (`-tp N`) and expert parallelism for MoE (`-tp N --enable-expert-parallel`).
- Model-family fixes:
  - **Qwen 3.5 / 3.6 / 3.8**: the vision tower's duplicated attention tensors;
  - **Qwen3.8-Flash-Next**: the FP8 PLE lookup on GPUs below sm_89.
- vLLM's int4 KV cache (`--kv-cache-dtype int4_per_token_head`, TRITON_ATTN backend), made exact and fast enough to
  serve 262k-token sessions: see [int4 KV cache](#int4-kv-cache).

## int4 KV cache

`int4_per_token_head` halves the KV cache of fp8 (a 27B model fits 262,144 tokens in 16 GB), but in vLLM 0.31 its
attention is inaccurate at long context and slow. The plugin replaces it when the int4 mode is present:

- **Exact read kernel.** Stock tensor-core dots round to TF32 while the value zero-point correction uses the unrounded
  probabilities: 11 % relative error at 32k context, 28 % at 229k. The plugin uses bf16 operands (the nibbles are
  exact) and the same rounded values in the correction: 0.4 %.
- **Long prefills on FlashAttention.** For a chunk of 64+ queries over 8k+ cached tokens, the cache is unpacked to bf16
  in segments of 16k tokens (~124 MiB of scratch, reused by every layer) and attended with FlashAttention 2, merging
  the segments by log-sum-exp. Shorter chunks run on the Triton kernel with larger tiles.
- **Split-KV decode with speculative decoding.** Stock vLLM uses the split-KV path only for one query token per
  request; MTP verification has two, so decode scanned the whole context in 4 programs. The plugin uses it for up to 4
  query tokens, with 64-token tiles and a segment count sized to the GPU (next power of two of SMs / 4, from 16 to
  128; each extra 16 segments cost ~12.6 MiB at 24 query heads, head 256).

Measured with Swift-1.5 27B (EXL3 2.25 bpw), MTP 1, chunked prefill 1024, one request, tokens per second:

| RTX PRO 6000 Server, context | prefill: exact kernel only / plugin / fp8 KV | decode: exact kernel only / plugin / fp8 KV |
|---|---|---|
| 128k | 1431 / 2220 / 1887 | 48.6 / 95 / 99 |
| 259k | 901 / 1737 / 1397 | 30.2 / 75 / 82 |

Stock vLLM int4 for comparison: 651 prefill and 3.8 decode at 201k on the same GPU; 216 and 3.2 at 261k on an A40.

Options (environment variables of the vLLM process):

| Variable | Default | Effect |
|---|---|---|
| `VLLM_QUANTPACK_INT4_KV` | `1` | `0` leaves vLLM's int4 attention untouched |
| `VLLM_QUANTPACK_INT4_FA_PREFILL` | `1` | `0` keeps long prefills on the Triton kernel (no FlashAttention, no scratch) |
| `VLLM_QUANTPACK_INT4_DECODE` | `64,4,auto` | split-KV decode `tile,warps,segments`; `segments` is a power of two or `auto` (from the SM count) |
| `VLLM_QUANTPACK_INT4_TILES` | `64,64,8,2` | Triton prefill `BLOCK_M,TILE,warps,stages` (short chunks, or with FlashAttention prefill off) |
| `VLLM_QUANTPACK_CACHE` | `~/.cache/vllm-quantpack` | where the generated kernel module is written (Triton needs it on disk) |

The defaults come from an A40 (84 SMs) and an RTX PRO 6000 (188 SMs). On another GPU, measure with
`bench_int4_attn.py` from the validation suite (`MODE=decode` for decode tiles and segments, `MODE=fa` for prefill) and
set the variables. The first long request after a fresh install compiles the new kernels (tens of seconds); Triton
caches them on disk for later starts.

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
| EXL3 int4 embedding | validated (dense Qwen 27B): KLD +0.16 % vs bf16 table, tool calls and agent tasks unchanged |
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
