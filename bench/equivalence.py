#!/usr/bin/env python3
"""Run one checkpoint offline with greedy decoding and dump tokens, logprobs and throughput as JSON.

Run it twice, once with the current plugin stack and once with vllm-quantpack (choose with VLLM_PLUGINS), then compare
the two dumps with `equivalence.py --compare a.json b.json`. Same GPU and same flags for both runs.

  VLLM_PLUGINS=quantpack python equivalence.py --model /path/to/model --out qp.json [--image]

Greedy generation is not reproducible run to run, even with the same code (the EXL3 kernels accumulate in a varying
order), so token-by-token equality cannot be the test. `--score REF.json` scores the text answers of a previous run
with teacher forcing (log-probability of each token given the ones before), and `--compare-scores` reports the mean
NLL and the per-token difference between two scorings: run it for base vs base and for base vs quantpack.
"""
import argparse
import base64
import io
import json
import math
import os
import sys
import time

PROMPTS = [
    "How do I find which process is listening on port 8443 on a Linux server?",
    "Explain the difference between a hard link and a symbolic link, with commands.",
    "Write a systemd unit that runs /usr/local/bin/backup.sh every night at 02:30.",
    "A disk shows 100% used but du reports much less. What are the likely causes and how do I check them?",
    "Write a bash function that retries a command up to 5 times with exponential backoff.",
    "How do I see why a systemd service failed to start? List the commands and what to look for.",
    "Explain what the load average numbers in uptime mean on a 16-core machine.",
    "Write an nftables rule set that allows SSH from 10.0.0.0/8 and drops everything else inbound.",
    "How do I safely grow an ext4 filesystem on an LVM logical volume online?",
    "What does `set -euo pipefail` do in a bash script? Give an example where each part matters.",
    "Write a Python script that tails a log file and prints lines containing ERROR with a timestamp.",
    "How can I check which NVIDIA driver and CUDA version a machine has, and which processes use the GPU?",
]


def image_message():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 240), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((20, 20, 620, 220), outline="black", width=4)
    d.text((60, 90), "QUANTPACK 4821 - port 8443 OPEN", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": "Transcribe exactly the text in the image."}]}]


def engine_args(a):
    """Engine settings shared by run and score; --tp/--ep select tensor or expert parallelism."""
    args = dict(model=a.model, max_model_len=8192, kv_cache_dtype="fp8_e4m3", max_num_seqs=16,
                kv_cache_memory_bytes=a.kv_bytes, seed=0, tensor_parallel_size=a.tp,
                enable_expert_parallel=a.ep)
    args.update(json.loads(a.engine_json))        # model-specific settings from its card, e.g. the attention backend
    return args


def run(a):
    from vllm import LLM, SamplingParams
    mm = dict(limit_mm_per_prompt={"image": 1, "video": 0}) if a.image else {}   # text-only models reject it
    llm = LLM(**engine_args(a), **mm)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=1, seed=0)
    convs = [[{"role": "user", "content": p}] for p in PROMPTS]
    if a.image:
        convs.append(image_message())
    llm.chat(convs[:2], sp, use_tqdm=False)                     # warm-up
    t = time.time()
    if a.serial:
        # one request at a time: no batch-composition effects, so two runs of the same code must match exactly
        outs = [llm.chat([c], sp, use_tqdm=False)[0] for c in convs]
    else:
        outs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t
    res = []
    for o in outs:
        c = o.outputs[0]
        lp = [next(iter(d.values())).logprob if d else None for d in (c.logprobs or [])]
        res.append(dict(tokens=list(c.token_ids), logprobs=lp, text=c.text))
    ntok = sum(len(r["tokens"]) for r in res)
    json.dump(dict(model=a.model, seconds=dt, tokens=ntok, tok_s=ntok / dt, outputs=res), open(a.out, "w"))
    print(f"{a.out}: {len(res)} outputs, {ntok} tokens, {ntok / dt:.1f} tok/s")


def score(a):
    """Teacher-forced log-probabilities of the text answers stored in a.score (the image answer is skipped)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    ref = json.load(open(a.score))
    llm = LLM(**engine_args(a), enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    prompts, spans = [], []
    # Only the text answers: the image answer (stored after them) needs the image tokens, which are not in the dump.
    for p, o in zip(PROMPTS, ref["outputs"][:len(PROMPTS)]):
        # tokenize=True returns a BatchEncoding with some tokenizers, so render the text and encode it explicitly
        text = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        head = tok.encode(text, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=head + o["tokens"]))
        spans.append((len(head), len(head) + len(o["tokens"])))
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
    res = []
    for o, (s, e) in zip(outs, spans):
        ids = o.prompt_token_ids
        res.append([o.prompt_logprobs[j][ids[j]].logprob for j in range(s, e)])
    json.dump(dict(model=a.model, ref=a.score, logprobs=res), open(a.out, "w"))
    n = sum(map(len, res))
    print(f"{a.out}: {n} tokens scored, mean NLL {-sum(map(sum, res)) / n:.5f}")


def compare_scores(fa, fb):
    a, b = json.load(open(fa)), json.load(open(fb))
    if os.path.basename(a["ref"]) != os.path.basename(b["ref"]):
        sys.exit(f"different references: {a['ref']} vs {b['ref']}")
    A, B = a["logprobs"], b["logprobs"]
    if [len(r) for r in A] != [len(r) for r in B]:
        sys.exit("the two scorings do not cover the same tokens")
    d = [abs(x - y) for ra, rb in zip(A, B) for x, y in zip(ra, rb)]
    n = len(d)
    if not n:
        sys.exit("no tokens scored")
    nll_a = -sum(map(sum, A)) / n
    nll_b = -sum(map(sum, B)) / n
    print(f"tokens {n}; mean NLL {nll_a:.5f} vs {nll_b:.5f} ({nll_b - nll_a:+.5f}); "
          f"per-token |diff| mean {sum(d) / n:.6f}, max {max(d):.6f}, exact {sum(x == 0 for x in d)}/{n}")


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    same, first_diff, maxd = 0, [], 0.0
    for i, (x, y) in enumerate(zip(A["outputs"], B["outputs"])):
        n = min(len(x["tokens"]), len(y["tokens"]))
        k = next((j for j in range(n) if x["tokens"][j] != y["tokens"][j]), None)
        if k is None and len(x["tokens"]) == len(y["tokens"]):
            same += 1
            k = n
        else:
            first_diff.append((i, k))
        for j in range(k):
            p, q = x["logprobs"][j], y["logprobs"][j]
            if p is not None and q is not None and math.isfinite(p) and math.isfinite(q):
                maxd = max(maxd, abs(p - q))
    print(f"identical outputs: {same}/{len(A['outputs'])}; first divergence (output, token): {first_diff}")
    print(f"max |logprob diff| on the common prefix: {maxd:.6f}")
    print(f"throughput: {A['tok_s']:.1f} vs {B['tok_s']:.1f} tok/s ({100 * (B['tok_s'] / A['tok_s'] - 1):+.1f} %)")
    if A["outputs"] and len(A["outputs"]) > len(PROMPTS):
        print("image answer A:", A["outputs"][-1]["text"][-200:].replace("\n", " "))
        print("image answer B:", B["outputs"][-1]["text"][-200:].replace("\n", " "))
    return 0 if same == len(A["outputs"]) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--kv-bytes", type=int, default=4_000_000_000)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--ep", action="store_true", help="expert parallelism for MoE models (needs --tp > 1)")
    ap.add_argument("--engine-json", default="{}", help="extra vllm.LLM arguments as a JSON object")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--score")
    ap.add_argument("--compare-scores", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    if a.compare_scores:
        sys.exit(compare_scores(*a.compare_scores))
    sys.exit(score(a) if a.score else run(a))
