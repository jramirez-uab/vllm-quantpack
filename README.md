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
- vLLM's TurboQuant KV cache (`--kv-cache-dtype turboquant_k8v4` and the other presets), made correct with
  speculative decoding and on GPUs below sm_89: see [TurboQuant KV cache](#turboquant-kv-cache).
- A KV cache with int8 keys and int4 values (`--kv-cache-dtype k8v4_per_token_head`): closer to fp8 in quality than
  int4, 1.3 times the tokens of fp8 in the same memory, at int4 speed: see [k8v4 KV cache](#k8v4-kv-cache).

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
  Each program holds every query token of the step (up to 64 query rows; vLLM's block is 16 rows = 2 tokens at 6
  query heads per KV head), so a step with 3 draft tokens reads the cache once instead of twice.

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
| `VLLM_QUANTPACK_INT4_DECODE` | `64,4,auto` | split-KV decode `tile,warps,segments[,rows]`; `segments` is a power of two or `auto` (from the SM count); `rows` caps the query rows per program (default 64; 16 is vLLM's block and reads the cache once per 2 query tokens) |
| `VLLM_QUANTPACK_INT4_TILES` | `64,64,8,2` | Triton prefill `BLOCK_M,TILE,warps,stages` (short chunks, or with FlashAttention prefill off) |
| `VLLM_QUANTPACK_CACHE` | `~/.cache/vllm-quantpack` | where the generated kernel module is written (Triton needs it on disk) |

The defaults come from an A40 (84 SMs) and an RTX PRO 6000 (188 SMs). On another GPU, time the options on it:

```
python -m vllm_quantpack.tools.int4_kv_bench --contexts 32768,131072,262144
```

It fills a random int4 cache and prints, per attention layer, vLLM's stock decode and each split-KV decode config
(`--decode 64,4,16:64,4,32:...`), and a prefill chunk on Triton and on FlashAttention, each with its relative error
against an fp64 reference. The model shape defaults to Qwen3.8-27B (`--q-heads`, `--kv-heads`, `--head-size`,
`--block-size`, `--mtp` change it). Then set `VLLM_QUANTPACK_INT4_DECODE` to the fastest config.

On Blackwell (RTX 50xx, RTX PRO 6000), vLLM 0.31's FlashAttention 2 build carries sm_80 code only, so the driver
compiles it for the GPU on its first use: the first long prefill takes ~30 s more. The driver keeps the result in
`~/.nv/ComputeCache`; a service without `HOME` (a systemd unit, for example) loses it and pays it at every start, so
give it `HOME` or `CUDA_CACHE_PATH`. Ampere and Ada run the sm_80 code directly.


## k8v4 KV cache

`--kv-cache-dtype k8v4_per_token_head` is a cache format the plugin adds to vLLM 0.33 (it needs vLLM's KV cache
dtype registry; on a vLLM without it the option does not exist). Per token and KV head it stores the key as int8 with one fp32 scale and the value as
`int4_per_token_head` does, both in the randomized-Hadamard domain: 1.5 bytes per element pair instead of 1 (int4)
or 2 (fp8). The attention scores are where the int4 cache loses quality, so the keys get the 8 bits.

It reuses the int4 attention of this plugin (same prefill and split-KV decode, with the key half read as int8), so
it needs the int4 patch: with `VLLM_QUANTPACK_INT4_KV=0` the dtype is not registered and vLLM rejects the option.
The `VLLM_QUANTPACK_INT4_*` variables apply to it as well.

KL divergence of the next-token distribution against a bf16 cache (Swift-1.5 27B BF16, last 2,048 positions of a
250k-token context; lower is better): int8 0.0011, fp8 0.0023, **k8v4_per_token_head 0.0032**, vLLM's
`turboquant_k8v4` 0.0065, int4 0.0093.

Served with Swift-1.5 27B (EXL3 3.85 bpw), 262,144-token sessions, MTP 3, one request, vLLM 0.33, 20 GB of KV cache:

| | RTX PRO 6000 | A40 |
|---|---|---|
| KV cache capacity, tokens (fp8 517k, int4 912k) | 650,777 | 663,726 |
| Prefill, tokens/s at 30k / 128k / 260k | 2843 / 2363 / 1845 | 1314 / 965 / 787 |
| Decode, tokens/s at 30k / 128k / 260k | 97 / 86 / 73 | 62 / 57 / 37 |
| 8-fact needle at 62k, 123k, 236k | 8 of 8 each | 8 of 8 each |
| Tool calls (real agent request) | 15 of 15 | 15 of 15 |

Speeds are those of the int4 cache on the same setup (260k: 1895 / 74 and 824 / 39); fp8 gives 1727 / 88 and
724 / 42. Decode figures with MTP move by some 20 % between runs with the acceptance rate. The first request after a
start compiles the kernels (about a minute on a cold Triton cache).

## TurboQuant KV cache

vLLM 0.33's TurboQuant backend (`turboquant_k8v4`: FP8 keys, 4-bit values) has two defects that the plugin fixes when
the backend is present:

- **Speculative decoding under CUDA graphs.** A verification step carries 1 + MTP query tokens per request, so vLLM
  sends it down the prefill path, which reads request lengths as Python numbers; a CUDA graph freezes the lengths
  seen at capture and every later step attends to the wrong context (Swift-1.5 27B, MTP 3, 62k-token prompt: 0 of 8
  needles, repeated syllables; 8 of 8 without MTP or with `--enforce-eager`). The plugin gives each query token of
  such a step its own decode row, computed with tensors only, and sizes the graph batch and the workspace for it.
- **Keys on GPUs below sm_89** (Ampere: A40, RTX 30xx). There the stock kernels store keys as fp8e4b15, which
  saturates far below the keys' range (relative error 0.93 on Qwen3.8-27B keys; 0 of 8 needles on an A40 even
  without MTP). The plugin encodes and decodes E4M3 with integer operations in the same kernels: relative error 0.125
  and the same cache bytes as on newer GPUs.

Measured with Swift-1.5 27B (EXL3 3.85 bpw), MTP 3, one request, vLLM 0.33: needles 8 of 8 at 62k, 123k and 236k
tokens and tool calls 15 of 15 on both an RTX PRO 6000 and an A40.

Decode stays slow at long context, because the stock decode kernel scans the cache once per query head and per query
token (tokens per second at 30k / 128k / 260k: 69 / 30 / 18 on the RTX PRO 6000, 32 / 12 / 6.5 on the A40; fp8 KV
gives 88 at 260k on the RTX PRO 6000). Prefill is at fp8 speed. For long sessions prefer the
[int4 KV cache](#int4-kv-cache), which holds more tokens and decodes four times faster.

| Variable | Default | Effect |
|---|---|---|
| `VLLM_QUANTPACK_TURBOQUANT` | `1` | `0` leaves vLLM's TurboQuant attention untouched |

## Requirements

- vLLM 0.34 (0.31 and 0.33 also work; `k8v4_per_token_head` needs 0.33 or newer). vLLM 0.34 moved the Qwen 3.5 / 3.6
  model to a new module: use 0.1.6 or newer of this plugin there, or the vision tower fails to load with
  `There is no module or parameter named 'blocks.0.attn.k_proj' in Qwen3_VisionTransformer`.
- exllamav3 1.6.5, as the prebuilt wheel that matches your torch and CUDA. The prebuilt wheels are on the
  [exllamav3 releases](https://github.com/turboderp-org/exllamav3/releases) page; PyPI only has a generic wheel that
  compiles the CUDA kernels on first use (it needs a CUDA toolkit and takes a while). For vLLM 0.31 (torch 2.13.0, CUDA 13):
  ```
  W=exllamav3-1.6.5+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
  curl -L -o /tmp/$W "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.5/${W/+/%2B}"
  pip install /tmp/$W
  ```
  For other versions: `python -c "import torch; print(torch.__version__, torch.version.cuda)"` and pick the matching asset.
  vLLM 0.34 uses torch 2.14, which has no prebuilt exllamav3 wheel yet: install it from PyPI
  (`pip install exllamav3`, the kernels compile on first use) or build it from source.
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
| int4 KV cache attention (`int4_per_token_head`) | validated (Swift-1.5 27B 2.25 bpw on an A40 limited to 16 GB, 262,144 tokens, MTP 1): tool calls 30/30, agent tasks 10/10, 9/10, 10/10; speed above, A40 and RTX PRO 6000 |
| int8-key / int4-value KV cache (`k8v4_per_token_head`) | measured on Swift-1.5 27B (numbers above), Nemotron 3.5 Lightning and Qwen3-4B-Instruct-2507 (vLLM 0.34): KLD vs a bf16 cache below fp8's at 32k, 128k and 250k on all three; Gemma 4 26B-A4B (head 512) served on an A40, where vLLM's Triton backend has no fp8 cache |
| Gemma 4 EXL3 (padded dense MLP) | served (turboderp 4.10 bpw, A40, vLLM 0.34); long int4/k8v4 prefills with head 512 run on Triton (FlashAttention stops at 256; FlashInfer measured 2.7-2.9x slower) |
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
