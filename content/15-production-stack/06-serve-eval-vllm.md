# 15.6 Serving, Quantizing and Evaluating: vLLM, lm-eval-harness, llama.cpp

[Chapter 15.5](05-posttrain-trl.html) handed off an aligned checkpoint — `ckpts/stack-100m-grpo`, the output of SFT → DPO → narrow GRPO run through `TRL` — sitting on disk in Hugging Face format. [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html) already did the *conceptual* work this chapter is about: it built a token-level `generate()` loop, hand-wrote round-to-nearest int8/int4 quantization with a bespoke `QuantizedLinear`, and assembled a small, honest capability battery from scratch. Every mechanism in that chapter — the KV cache byte-counting, the group-wise quantization math, the "re-run the eval after every edit" discipline — still holds. What changes here is the *plumbing*: instead of a Python `for` loop calling `model.generate()` one request at a time, you get continuous batching serving hundreds of concurrent requests; instead of a hand-rolled `QuantizedLinear` only `stacklm`'s own loader understands, you get a calibration-driven quantizer that writes a format three different serving engines already read; instead of five bespoke probes in a Python list, you get a standardized harness whose numbers are comparable to every other model's numbers.

This is the last chapter of Part XV, and it closes the loop the part opened in [15.1](01-toolchain-map.html): the same Stack-100M, the same architecture, the same aligned checkpoint — now shipped with **vLLM** for GPU serving, **`llm-compressor`**/**AutoAWQ** for production post-training quantization, **GGUF**/**llama.cpp** for the CPU/laptop path, and **`lm-evaluation-harness`** for evaluation you can defend to someone who wasn't in the room when you wrote the probes. Three separate, substantial open-source systems own three rows of the crosswalk table from [15.1](01-toolchain-map.html); this chapter gives each the room it needs.

## 1. Where We Pick Up, and What This Chapter Adds

Recall the state of the checkpoint. [Ch. 15.5](05-posttrain-trl.html) exported `stacklm`'s custom `nn.Module` — RoPE with every-4th-layer NoPE, GQA, QK-norm, SwiGLU, tied embeddings ([Ch. 14.4](../14-capstone/04-architecture.html)) — into a `LlamaConfig`/`LlamaForCausalLM`-shaped checkpoint via a `remap_state_dict` that renames parameters, gated by a fixed-batch assertion that the exported model's logits match the original's to under 1e-4. Read that gate carefully: only an export that preserves the architecture can pass it, which means path (1) below. Two of Stack-100M's architectural choices, NoPE and QK-norm, do not fit in a stock `LlamaConfig`; 15.5 flagged this explicitly and offered two honest paths: (1) ship a `modeling_stacklm.py` alongside the checkpoint and load it with `trust_remote_code=True`, preserving bit-exact fidelity, or (2) accept the faithful-enough Llama export, which drops those two knobs but is readable by *every* tool in this chapter without a line of custom code.

This chapter takes path (2) as the default, and says exactly why: `llm-compressor`, `AutoAWQ`, and llama.cpp's GGUF converter are all written against well-known HF architectures (Llama, Qwen2, Mistral, …). Feeding them a `trust_remote_code=True` custom model either fails outright or forces you to also write custom kernels for whichever tool you're using — a real cost, not a hypothetical one. If your production model leans harder on exotic architecture choices than Stack-100M's two small ones, path (1) is still there, and vLLM in particular *can* run a `trust_remote_code` HF model through its **Transformers backend** — a fallback execution path, distinct from vLLM's natively optimized model classes, that trades some throughput for "it just runs your `AutoModelForCausalLM`." That escape hatch is not free, though, and it is not universal: the modeling file has to satisfy vLLM's compatibility contract — attention dispatched through Transformers' `ALL_ATTENTION_FUNCTIONS` registry (so vLLM can substitute its own paged-attention implementation), the class advertising `_supports_attention_backend = True`, and `**kwargs` threaded through the forward chain. A `modeling_stacklm.py` lifted straight from [Ch. 14.4](../14-capstone/04-architecture.html), whose `Attention.forward` calls `F.scaled_dot_product_attention` directly, is rejected rather than silently run slower until that one path is rewritten. Which flag enables it (`--model-impl transformers` as of some 2025 releases) is exactly the kind of surface that moves between vLLM versions; treat the name below as illustrative and check `vllm serve --help` against your pin.

!!! warning "Verify the export before you trust anything downstream"

    Every tool in this chapter — vLLM, `llm-compressor`, `AutoAWQ`, llama.cpp — takes the HF checkpoint on faith. If the Llama export silently dropped QK-norm's effect on the logits (it does, by construction) and nobody checked how much that matters *for this checkpoint*, every number in this chapter inherits an unverified assumption. Ch. 15.5's <1e-4 logit-matching assertion certifies the `trust_remote_code` export, not this one — the stock-`LlamaConfig` path cannot pass it, and its fidelity has to be established empirically instead. So establish it: take Ch. 14.11's `compute_perplexity`, re-run on the exported HF model, next to the number `stacklm`'s own model reports on the same held-out shard. A gap bigger than run-to-run noise means the approximation cost more than expected, and you should reach for the `trust_remote_code` path instead of shipping a quietly worse model.

Here is the piece of the crosswalk table this chapter expands, first sketched in [15.1](01-toolchain-map.html):

| Stage | `stacklm` module (Ch. 14.11) | Production library (2026) | What the library adds |
|---|---|---|---|
| Serving | `stacklm.serve.generate` — token loop, one request at a time | **vLLM** (`vllm serve`) | Continuous batching, PagedAttention, OpenAI-compatible API, Prometheus metrics |
| Quantization (GPU) | `stacklm.serve.quantize` — hand-rolled RTN `QuantizedLinear` | **`llm-compressor`** (GPTQ), **AutoAWQ** | Calibration-driven accuracy at 4-bit, a standard on-disk format vLLM reads natively |
| Quantization (CPU/edge) | the same `QuantizedLinear` int4 checkpoint, run on CPU (Ch. 14.11 §9) — dequantize-then-fp32-matmul, the laptop demo | **GGUF** via **llama.cpp** | K-quant super-block packing, a mature CPU inference engine, laptop-class deployment |
| Evaluation | `stacklm.eval.probes` — five bespoke functions | **`lm-evaluation-harness`** | Standardized, comparable tasks; `stderr` on every metric; a `vllm` backend for speed |

Every row below follows the same shape: name the library, show it running against the real checkpoint, and say plainly what it buys you that the from-scratch version did not.

## 2. Serving with vLLM: Continuous Batching, PagedAttention, and the OpenAI API

**vLLM** is the 2026 default for GPU-backed open-weight serving, and the reason is architectural, not fashion. [Ch. 14.11](../14-capstone/11-evaluation-and-serving.html)'s `generate_text` processes one request fully — prefill, then decode token by token — before starting the next. That is fine for an eval probe; it is disastrous for a server, because requests arrive at different times, finish at different lengths, and a naive batcher either waits to fill a static batch (adding latency) or runs GPUs at low utilization (wasting money). [Continuous batching](../07-inference-serving/02-continuous-batching.html) fixes this at the *scheduler* level: at every decode step, the server admits newly arrived requests and evicts newly finished ones from the same running batch, so the GPU is never idle waiting for the slowest request in a static group. **PagedAttention** ([Ch. 4.6](../04-kernels-efficiency/06-paged-attention-kv.html)) is the memory mechanism that makes this practical: it manages the KV cache in fixed-size blocks (like OS virtual memory pages) instead of one contiguous per-request allocation, so blocks can be allocated, freed, and — crucially for [prefix caching](../07-inference-serving/07-prefix-caching.html) — shared, on demand as the batch composition churns every step. [Chapter 7.3](../07-inference-serving/03-vllm-internals.html) derives both mechanisms in full; here we *use* them.

### Install and an offline batch run

```bash
# CUDA wheel; vLLM releases roughly biweekly and the exact pin below will be
# stale by the time you read this — check `pip index versions vllm` and pin
# whatever you tested against, together with the transformers/torch versions
# vLLM's own compatibility matrix names for that release.
pip install "vllm==0.8.5"   # illustrative pin — verify against vllm's release notes
```

```python
"""
offline_batch.py -- vLLM's non-server API. Good for eval sweeps (Section 5) and
for sanity-checking a checkpoint before you stand up a server.
"""
from vllm import LLM, SamplingParams

# gpu_memory_utilization is the fraction of the GPU's TOTAL memory vLLM budgets
# for weights + KV cache + activation scratch. Treat it as memory the process
# really does take, not a ceiling it might stay under: vLLM profiles the
# activation peak, then EAGERLY allocates the KV block pool to fill whatever is
# left of the budget, and holds it for the engine's lifetime. At 101M params
# there is no reason to hand vLLM the whole card; 0.3 leaves headroom for
# anything else running on the box (Section 2.3 works out why this barely
# matters for a model this small).
llm = LLM(
    model="ckpts/stack-100m-grpo",
    dtype="bfloat16",
    max_model_len=2048,           # matches Stack-100M's pretrain context (Ch. 14.4)
    gpu_memory_utilization=0.3,
)

params = SamplingParams(temperature=0.0, max_tokens=64)
prompts = [
    "Compute 37 + 8. Give the final integer after '####'.",
    "Compute 91 - 46. Give the final integer after '####'.",
]
outputs = llm.generate(prompts, params)   # continuous batching applies even to
                                            # a Python list -- vLLM schedules the
                                            # whole list as it would concurrent
                                            # HTTP requests
for o in outputs:
    print(o.prompt, "->", o.outputs[0].text)
```

This one call replaces `stacklm`'s entire `generate_text` + manual batching loop. There is no code here implementing continuous batching, prefix caching, or memory management — that is exactly the point: those are the twelve person-years the library absorbed.

### `vllm serve`: the OpenAI-compatible endpoint

```bash
vllm serve ckpts/stack-100m-grpo \
    --served-model-name stack100m \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.3 \
    --port 8000
```

The chat template you registered on the tokenizer in [Ch. 15.5](05-posttrain-trl.html) — the Jinja `STACK_CHAT_TEMPLATE` with its `{% generation %}` markers — ships inside `tokenizer_config.json` and travels with the checkpoint, so `vllm serve` applies it automatically to every `/v1/chat/completions` request. No extra flag is needed for the common case; you only reach for `--chat-template` if you want to override what the checkpoint carries.

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "stack100m",
      "messages": [
        {"role": "user", "content": "Compute 37 + 8. Give the final integer after #### ."}
      ],
      "temperature": 0,
      "max_tokens": 64
    }'
```

The response is a standard OpenAI chat-completion object — `choices[0].message.content`, `usage.prompt_tokens`/`usage.completion_tokens` — the same shape a client written against a hosted frontier API expects. This is the practical payoff of the OpenAI-compatible surface: any tool built to talk to a commercial API (a LangChain/LlamaIndex client, an evaluation harness, your own agent loop from [Ch. 14.10](../14-capstone/10-agentic-narrow.html)) talks to `ckpts/stack-100m-grpo` with a one-line base-URL change.

vLLM also exposes Prometheus-format metrics:

```bash
curl -s localhost:8000/metrics | grep '^vllm:'
```

The metric *names* are one of vLLM's faster-moving surfaces — check your pinned version's actual output rather than trusting a list in a book — but the categories are stable across releases: running/waiting request counts (`num_requests_running`, `num_requests_waiting`), KV-cache occupancy (`gpu_cache_usage_perc`), and the two latency histograms that matter for user-perceived quality, time-to-first-token and time-per-output-token. Wiring these into Grafana/Prometheus is the standard way to watch a serving fleet — see [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html) for the dashboard side of this, and [Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html) for how these numbers feed capacity planning.

### A tiny model on a big GPU: why you are never memory-bound here

!!! example "Worked example: KV-cache concurrency at 100M parameters"

    [Ch. 14.11](../14-capstone/11-evaluation-and-serving.html) computed that Stack-100M's `KVCache` at `batch_size=1, max_seq=2048, dtype=bf16` is exactly **31,457,280 bytes ≈ 31.5 MB per full-context sequence**. Put that next to a `vllm serve` run at `--gpu-memory-utilization 0.3` on an A100 80GB: vLLM's memory budget is $0.3 \times 80\text{GB} = 24\text{GB}$. The bf16 weights cost ≈203MB (Ch. 14.11's own count), and CUDA-graph/activation scratch is on the order of a few hundred MB to a couple GB depending on `--max-num-seqs` — call the total non-KV overhead ≈1GB, generously. That leaves roughly $24 - 1 = 23\text{GB}$ for the KV-cache block pool:

    $$
    \frac{23{,}000\text{ MB}}{31.5\text{ MB/sequence}} \approx 730 \text{ concurrent full-context sequences}
    $$

    Little's Law relates that concurrency figure to sustainable throughput: for a queueing system in steady state, mean concurrency equals arrival rate times mean time-in-system,
    $$
    L \approx \lambda W
    $$
    If a typical completion takes on the order of $W \approx 2\text{s}$ end to end (prefill plus a modest decode budget — illustrative, measure your own), the KV cache alone could in principle sustain $\lambda \approx 730 / 2 \approx 365$ requests/second before running out of cache room. In practice you will hit `--max-num-seqs`'s default cap, Python-side scheduling overhead, or network I/O long before that — the honest conclusion is not "this server does 365 req/s" but **"KV-cache memory is essentially never the bottleneck for a 100M model,"** which is the opposite of the situation a 70B model is normally in, where PagedAttention's memory efficiency is the entire reason vLLM exists. This is the same "small model, relatively KV-bound at the *cache-vs-weights* ratio, but not at the *cache-vs-GPU* ratio" observation Ch. 14.11 made about the laptop path, now seen from the server side. For the general economics of where a serving system actually spends its dollar, see [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

!!! tip "Practitioner tip: `--max-num-seqs` and `--max-num-batched-tokens` are the knobs that actually matter at this scale"

    Since KV-cache memory is not your constraint, tune `--max-num-seqs` (the cap on requests admitted to a running batch) and `--max-num-batched-tokens` (the cap on tokens processed per scheduler step, which trades prefill latency for decode throughput) directly against your target concurrency and latency SLO, rather than against `--gpu-memory-utilization`. Turning `--gpu-memory-utilization` up for a model this small mostly just reserves memory nothing will ever use.

## 3. Production Quantization: `llm-compressor` and `AutoAWQ`

[Chapter 14.11 §6–7](../14-capstone/11-evaluation-and-serving.html) built round-to-nearest (RTN) quantization from scratch and sketched GPTQ's column-by-column Hessian-guided reconstruction and AWQ's activation-aware channel rescaling by hand — read those sections for the derivations; we do not repeat them here. What changes in production is threefold: (1) a maintained implementation that drives the whole error-propagation path from a single Cholesky factor of $H^{-1}$ instead of the pedagogical `inv(H)` sketch (the same $O(d_{in}^3)$ order, but numerically stable in fp32 and free of the per-column $H^{-1}$ updates a literal Optimal-Brain-Surgeon transcription would redo $d_{in}$ times), handles every layer of a real transformer (not five hand-picked ones) with the correct cross-layer sharing constraints AWQ needs, and is tested against dozens of real architectures; (2) calibration-set plumbing — tokenizing, batching, capturing activations — that you no longer write yourself; and (3) a **standard on-disk format**, `compressed-tensors`, that vLLM (and increasingly other engines) load *directly*, with no custom loader — unlike `stacklm`'s bespoke int4 safetensors format from Ch. 14.11 §7.4, which only `stacklm.serve.quantize.load_quantized` knows how to read.

### GPTQ via `llm-compressor`

`llm-compressor` is vLLM's own project for producing quantized checkpoints; it is the maintained successor to the earlier standalone GPTQ tooling and shares the `compressed-tensors` format with vLLM's native loader.

```python
"""
gptq_quantize.py -- production INT4 GPTQ, operating on the HF checkpoint from
Ch. 15.5. Replaces stacklm.serve.quantize's gptq_quantize_column_by_column
sketch (Ch. 14.11 Section 6.2) with the maintained implementation.

pip install "llm-compressor==0.3.1"   # young, fast-moving package -- check the
                                        # repo's examples/ directory against
                                        # whatever you pin; the oneshot() API
                                        # shape below has already changed once.
"""
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_DIR = "ckpts/stack-100m-grpo"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# GPTQ's Hessian is built from REAL activations run through the layer -- the
# X_calib argument in Ch. 14.11's from-scratch gptq_quantize_column_by_column().
# llm-compressor handles tokenize-and-forward bookkeeping; the DATA CHOICE is
# still yours. Calibrate on a held-out slice of the SAME mix the model trained
# on (Ch. 14.2's FineWeb-Edu/Cosmopedia/code/math blend) -- calibrating on
# out-of-distribution text (e.g. pure code) measurably skews which channels
# GPTQ decides are salient, the same failure mode Ch. 14.11 warns about for the
# hand-rolled version.
calib = load_dataset(
    "json", data_files="data/holdout_calib.jsonl", split="train"
).select(range(512))

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",          # 4-bit weights, activations stay bf16/fp16
    ignore=["lm_head"],      # leave the tied embedding/head unquantized (bf16) --
                              # the same instinct as Ch. 14.11's embedding_bits=8,
                              # though strictly higher precision than it, and for
                              # the same reason: the 32768x512 table is 17% of the
                              # params and its rare-token rows are thinly trained.
                              # For int8 parity instead, target lm_head with its
                              # own W8A16 config_groups entry rather than ignoring it
    dampening_frac=0.01,     # the same Hessian damping (`damp`) as Ch. 14.11's
                              # `H += damp * I`
)

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=calib,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

model.save_pretrained("ckpts/stack-100m-gptq-w4a16", save_compressed=True)
tokenizer.save_pretrained("ckpts/stack-100m-gptq-w4a16")
```

Serving it needs no export step, unlike the hand-rolled path:

```bash
vllm serve ckpts/stack-100m-gptq-w4a16 --served-model-name stack100m-int4
```

vLLM detects the `compressed-tensors` quantization config in the checkpoint's `config.json` and dispatches to the matching int4 kernel automatically. Compare this to Ch. 14.11's §7, where `QuantizedLinear.forward` *dequantizes to fp32 and does a standard matmul* — a memory win with no matmul speedup, explicitly flagged there as "not a fast path." vLLM's GPTQ/AWQ kernels (Marlin and friends) instead **fuse the dequantization into the matmul**: the int4 weights are unpacked and scaled in registers/shared memory, tile by tile, and the full-precision weight tensor is never materialized in HBM. The MMA math is still bf16/fp16 — `W4A16` means exactly what it says, and the speedup comes from moving 4× fewer weight bytes per decode step in a regime that is memory-bandwidth-bound, not from cheaper arithmetic. Integer-native *arithmetic* would require quantizing the activations too (`W8A8-INT8`, or the newer W4A8/FP8 schemes `llm-compressor` also emits), which is a different recipe with a different accuracy budget.

### AWQ via `AutoAWQ`

One caveat before the code: `AutoAWQ` is the *original* standalone AWQ implementation, and it was retired by its author in 2025 — AWQ production has moved into `llm-compressor`'s `AWQModifier` (and `GPTQModel`), which write the same `compressed-tensors`/`awq` checkpoints vLLM already loads ([Ch. 4.7](../04-kernels-efficiency/07-quantization-ptq.html)). The recipe below is still the clearest statement of AWQ's *interface* — a quant config, one `quantize()` call, one `save_quantized()` — and pinned releases still run, but for new work prefer the `AWQModifier` path through the same `oneshot()` call the GPTQ recipe above uses, which keeps one tool and one format for both methods.

```python
"""
pip install "autoawq==0.2.7"   # illustrative pin of an archived project; see the
                               # note above before adopting this for new work
"""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_DIR = "ckpts/stack-100m-grpo"
model = AutoAWQForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

quant_config = {
    "zero_point": True,
    "q_group_size": 64,   # matches Ch. 14.11's group_size=64, for an apples-to-
                           # apples comparison in the sweep below
    "w_bit": 4,
    "version": "GEMM",
}
# AWQ's search needs real activations too, and `calib_data` DEFAULTS to a
# `pileval` slice AutoAWQ downloads for you -- 128 samples of generic web text
# truncated to 512 tokens, i.e. exactly the out-of-distribution calibration the
# GPTQ recipe above takes pains to avoid, plus a silent network dependency.
# Hand it the SAME held-out slice, at the same length, or the two checkpoints
# you are about to A/B differ in their calibration set as much as in their
# method.
calib_texts = load_dataset(
    "json", data_files="data/holdout_calib.jsonl", split="train"
).select(range(512))["text"]

# Runs the same salient-channel grid search as Ch. 14.11's
# awq_search_channel_scales() -- find a per-channel scale s that minimizes
# post-quantization OUTPUT error, then fold s into the preceding RMSNorm --
# batched across every linear layer in the model, with the joint-scale
# constraint Ch. 14.11 flagged (layers reading the SAME normalized hidden
# state, like wq/wk/wv, must share one s).
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_texts,   # a list[str]; AutoAWQ also accepts a Dataset or a
                               # Hub dataset name (check your pin's signature)
    max_calib_samples=512,
    max_calib_seq_len=2048,   # match the GPTQ recipe's max_seq_length=2048;
                               # the default here is 512
)
model.save_quantized("ckpts/stack-100m-awq-w4g64")
tokenizer.save_pretrained("ckpts/stack-100m-awq-w4g64")
```

```bash
vllm serve ckpts/stack-100m-awq-w4g64 --served-model-name stack100m-awq
```

!!! warning "Re-measure — production tools don't remove the obligation, they make it cheaper"

    [Ch. 14.11 §8](../14-capstone/11-evaluation-and-serving.html) built an explicit quantization sweep that re-ran the whole probe battery on every quantized variant, because "int8 is free, int4 costs something on the hard tail" is an empirical claim about *your* checkpoint, not a law of nature. That discipline does not go away because the quantizer is now maintained. Run [Section 5](#5-honest-evaluation-with-lm-evaluation-harness)'s `lm_eval` battery against the fp16 base, the GPTQ checkpoint, and the AWQ checkpoint before choosing one. Be honest about scale, too: the "salient channel" phenomenon both GPTQ and AWQ exploit is far more pronounced in larger, more overparameterized models with heavier activation outliers; at 101M parameters the gap between either method and plain RTN can be small enough that the extra calibration complexity is not worth it. Measure — do not assume the production tool always wins just because it is more sophisticated.

## 4. GGUF and llama.cpp: CPU and Laptop Serving

vLLM assumes a GPU. The other half of [Ch. 14.11](../14-capstone/11-evaluation-and-serving.html)'s payoff — generating text on the CPU of the laptop you're reading this on, with zero cloud dependency — has its own standard 2026 tool: **llama.cpp** (Georgi Gerganov and contributors), a C/C++ inference engine with no PyTorch dependency at runtime, and its companion checkpoint format **GGUF**. This is the production version of the exact demo Ch. 14.11 closed with, and it maps directly onto [Ch. 14.11's laptop path](../14-capstone/11-evaluation-and-serving.html) the way vLLM maps onto Ch. 14.11's server-side generation loop.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt   # deps for the Python converter scripts

# Convert the HF checkpoint (Ch. 15.5's Llama-shaped export) to GGUF at f16.
# The converter script's NAME has moved across llama.cpp's history --
# convert.py -> convert-hf-to-gguf.py -> convert_hf_to_gguf.py -- run
# `ls convert*` in your checkout and match whatever is actually there.
python convert_hf_to_gguf.py ../ckpts/stack-100m-grpo \
    --outfile stack100m-f16.gguf \
    --outtype f16

# Build (the current CMake path; the older Makefile build is deprecated).
cmake -B build && cmake --build build --config Release -j

# Quantize the f16 GGUF down to a K-quant. The Q4_K block layout packs 256
# weights per super-block as eight 32-weight sub-blocks with 6-bit scales AND
# 6-bit mins -- exactly 4.5 bits/weight, the scheme Ch. 14.11's memory-budget
# example gestured at. Q4_K_M is the standard "good default": Q4_K almost
# everywhere, with the sensitive tensors promoted to Q6_K, so the file averages
# a few tenths of a bit MORE than 4.5 (worked example below).
./build/bin/llama-quantize stack100m-f16.gguf stack100m-Q4_K_M.gguf Q4_K_M

# An OpenAI-compatible endpoint, CPU-only, on your own machine.
./build/bin/llama-server -m stack100m-Q4_K_M.gguf --port 8080 -c 2048

# Or drive it directly, no server process:
./build/bin/llama-cli -m stack100m-Q4_K_M.gguf \
    -p "Compute 37 + 8. Give the final integer after '####'." -n 64
```

For embedding in a Python application rather than shelling out, **`llama-cpp-python`** wraps the same engine:

```python
"""pip install llama-cpp-python"""
from llama_cpp import Llama

llm = Llama(model_path="stack100m-Q4_K_M.gguf", n_ctx=2048)
out = llm(
    "Compute 37 + 8. Give the final integer after '####'.",
    max_tokens=64, temperature=0.0,
)
print(out["choices"][0]["text"])
```

!!! example "Worked example: GGUF's K-quant vs. our hand-rolled int4"

    Ch. 14.11 computed the hand-rolled int4 checkpoint (per-group, `group_size=64`, fp32 scale + fp32 zero-point per group) at **≈63.3 MB** for Stack-100M's 101,318,656 quantizable parameters, and flagged that 20% of that total is scale/zero-point overhead alone.

    `Q4_K` packs 256 weights per super-block as eight 32-weight sub-blocks, each carrying its own 6-bit scale *and* 6-bit minimum (Q4_K is asymmetric; the sixteen 6-bit values pack into a 12-byte array), plus two fp16 super-scales `d`/`dmin` that rescale those sub-block scales and mins. That is ggml's `block_q4_K`: $128 + 12 + 4 = 144$ bytes per 256 weights, i.e. exactly **4.5 bits/weight** ([Ch. 4.8](../04-kernels-efficiency/08-quantization-formats-qat.html) dissects the struct), vs. our scheme's $4 + 2 \times 32/64 = 5\text{ bits/weight}$ effective rate (4-bit codes plus two fp32 numbers per 64-weight group, amortized). At Stack-100M's parameter count:

    $$
    \frac{101{,}318{,}656 \times 4.5\text{ bits}}{8\text{ bits/byte}} \approx 57.0\text{ MB}
    $$

    plus a small amount for norms and metadata GGUF stores at higher precision — call it **on the order of 57–60 MB** for a file that is `Q4_K` throughout, which is what `Q4_K_S` gives you.

    The `Q4_K_M` we actually produced above is not that: the `_S`/`_M`/`_L` suffix is a **tensor-level mix**, not a different block layout. `llama-quantize` applies `Q4_K` to most tensors but promotes the empirically most sensitive ones to `Q6_K` (6.5625 bits/weight), which pushes the *whole-file* average a few tenths of a bit above 4.5 (see [Ch. 4.8](../04-kernels-efficiency/08-quantization-formats-qat.html)). Which tensors, exactly, is worth checking rather than assuming — the promotion set is model-dependent, and for Stack-100M it is smaller than the usual description suggests:

    - `attn_v` and `ffn_down`, but only in the layers llama.cpp's `use_more_bits` heuristic selects (the first eighth, the last eighth, and every third layer in between) — 14 of our 30 blocks, at $65{,}536 + 720{,}896 = 786{,}432$ parameters per block, so ≈11.0M parameters promoted.
    - The separate output matrix — which **Stack-100M does not have**. Tied embeddings ([Ch. 14.4](../14-capstone/04-architecture.html)) mean the HF checkpoint stores no `lm_head.weight`, the converter emits no `output.weight`, and llama.cpp reuses `token_embd` for the output projection. That single largest tensor (16.8M parameters, 17% of the model) therefore stays at the base `Q4_K` type under the `_M` mix.

    So the promotion costs $11.0\text{M} \times (6.5625 - 4.5)\text{ bits} \approx 2.8$ MB on top of the 57.0 MB `Q4_K` floor: **≈60 MB at ≈4.7 bits/weight**. That is a few MB *below* our hand-rolled 63.3 MB — a real but unspectacular size win, and the size is not the interesting part: **at a comparable bit budget, purpose-built packing spends the bits far better** — finer-grained scales (a 6-bit scale *and* min per 32 weights instead of two fp32 scalars per 64) and extra precision aimed at exactly the tensors that need it, instead of a flat 20% metadata tax spread uniformly. If you want the full size win, `Q4_K_S` buys it at ≈57 MB. Either way this is what Ch. 14.11 previewed when it noted production formats "attack precisely that 20%" of overhead. And do not take the ≈60 MB on faith: `ls -l` the file, or run `llama-quantize`'s own per-tensor log, which prints the type it chose for every tensor.

!!! warning "Common pitfall: comparing tokens/sec across engines without matching conditions"

    A llama.cpp CPU tokens/sec number and a vLLM GPU tokens/sec number are not comparable unless batch size, sequence length, and — critically — whether you are measuring single-stream latency or aggregate server throughput under concurrent load are all held fixed. llama.cpp's single-request CPU decode speed is a *latency* number; vLLM's headline throughput is almost always an *aggregate, continuous-batched* number across many concurrent requests. Report both, labeled, or you will "prove" one engine is faster than the other by comparing apples to a fruit basket. See [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html) for how decode-step cost composes into either metric.

## 5. Honest Evaluation with `lm-evaluation-harness`

Everything above produced artifacts — a served endpoint, a quantized checkpoint, a GGUF file. None of them is worth anything without the discipline [Ch. 14.11 §3–5](../14-capstone/11-evaluation-and-serving.html) built by hand: measure honestly, disclose what the model cannot do, and re-measure after every edit. **EleutherAI's `lm-evaluation-harness`** (`lm-eval`) is the 2026 standard implementation of that discipline — the same harness behind the Hugging Face Open LLM Leaderboard and most published small-model papers, which matters because it makes your numbers *comparable to someone else's*, not just internally consistent.

```bash
pip install "lm-eval==0.4.7"   # pin, and re-check task YAMLs after upgrading --
                                 # metric names occasionally change between releases

# Point it at the HF checkpoint directly.
lm_eval --model hf \
    --model_args pretrained=ckpts/stack-100m-grpo,dtype=bfloat16 \
    --tasks arc_easy,piqa \
    --device cpu \
    --batch_size 8 \
    --output_path results/stack100m_base

# Or route through the vLLM backend -- the harness's `vllm` model type wraps
# exactly the offline-batch LLM() call from Section 2, so continuous batching
# speeds up EVALUATION, not just serving. This is the same call Ch. 15.5's
# post-training gate uses, forward-referenced there as "the same engine you
# serve with, Ch. 15.6."
lm_eval --model vllm \
    --model_args pretrained=ckpts/stack-100m-grpo,dtype=bfloat16,gpu_memory_utilization=0.3 \
    --tasks arc_easy,piqa \
    --batch_size auto
```

Be honest, as [Ch. 14.11's capability report](../14-capstone/11-evaluation-and-serving.html) insists: `arc_easy` and `piqa` are real standardized tasks, but a 100M-parameter model trained on ~20B tokens will land close to chance on most broad-knowledge benchmarks in the harness's catalog. The value of running them is not a leaderboard number — it is **methodology**: the same battery every paper reports, with the exact prompt formatting and scoring convention (cloze scoring for multiple choice, the same BPE-boundary discipline Ch. 14.11's `sequence_logprob` implements by hand) applied consistently, so a future rerun on a better checkpoint is *actually comparable*, which the bespoke probes in Ch. 14.11 were explicitly not designed to be against anyone else's model.

### Porting the hand-rolled arithmetic probe into a custom task

[Ch. 14.11 §4.1](../14-capstone/11-evaluation-and-serving.html) built `make_arithmetic_probe`/`eval_arithmetic`, graded with the *exact* `#### <int>` format the RLVR stage trained. The harness supports exactly this kind of bespoke task via a YAML file — porting it is mechanical, and it is worth doing precisely because it turns a script only you can run into an artifact anyone with the harness installed can rerun, diff against a baseline, and wire into CI.

```yaml
# tasks/stack_arithmetic/arithmetic_2digit.yaml
task: stack_arithmetic_2digit
dataset_path: json
dataset_kwargs:
  data_files: {test: "data/arith_probe_2digit.jsonl"}   # same seed=0, n=200 set
                                                          # Ch. 14.11 generates
test_split: test
output_type: generate_until
doc_to_text: "{{question}}"      # Ch. 14.9's exact prompt: "Compute {a} {op} {b}.
                                  # Give the final integer after '####'."
doc_to_target: "{{answer}}"
generation_kwargs:
  until: ["<|end|>"]
  max_gen_toks: 64               # matches Ch. 14.9's RLVR rollout budget --
                                  # truncate this and you cut off the '####'
                                  # line and measure your own budget, not the
                                  # model, exactly the pitfall Ch. 14.11 flags
  do_sample: false
filter_list:
  - name: "extract_answer"
    filter:
      - function: "regex"
        regex_pattern: "#### (-?\\d+)"   # the SAME parse Ch. 14.9's
                                          # exact_match_reward uses
      - function: "take_first"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
```

```bash
lm_eval --model hf \
    --model_args pretrained=ckpts/stack-100m-grpo,dtype=bfloat16 \
    --tasks stack_arithmetic_2digit \
    --include_path tasks/ \
    --limit 200 \
    --log_samples --output_path results/arith
```

`--log_samples` writes every prompt, generation, and parsed prediction to disk — inspect it, especially the first time, to confirm the harness's `regex` filter is matching the same `#### <int>` your RLVR verifier matches. `--limit` caps the number of examples for a fast smoke test before a full run. `--num_fewshot` controls in-context examples for tasks that use them (`0` for the arithmetic task above, since Ch. 14.9 trained zero-shot).

!!! tip "Match the eval's chat template to training, every time"

    The single most common way to make a correctly post-trained model look broken: score it with `lm_eval` using no chat template, or the wrong one. Pass `--apply_chat_template` so the harness renders prompts through the tokenizer's registered Jinja template (the one [Ch. 15.5](05-posttrain-trl.html) wrote), matching the exact token stream the model was trained on. This is the library-side echo of a from-scratch lesson Ch. 14.11's `chat_prompt` helper enforces by construction — the wire format *is* part of the model's interface, and an eval that improvises its own spacing is measuring a distribution shift you introduced yourself.

For agentic evaluation — the ReAct loop from [Ch. 14.10](../14-capstone/10-agentic-narrow.html), which `lm_eval`'s log-likelihood/generate-until primitives do not naturally decompose — `lighteval` (Hugging Face) and especially **`inspect_ai`** (UK AI Safety Institute) are the right tools, with solvers, tool sandboxes, and per-step trace logging built for exactly the format-validity/tool-choice/turns/EM decomposition Ch. 14.11 §4.4 implemented by hand. See [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html) and [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html).

## 6. The Full Pipeline, and Living With Version Drift

Stitching everything in this chapter into one script makes the shape of the production pipeline concrete: one checkpoint in, three shipped artifacts out, each independently evaluated.

```bash
#!/usr/bin/env bash
# ship_stack100m.sh -- the production pipeline for Part XV, end to end.
set -euo pipefail

CKPT=ckpts/stack-100m-grpo     # Ch. 15.5's final, aligned HF checkpoint

# 1. GPU path: quantize with AWQ, serve with vLLM.
#    NOTE the explicit budget on the backgrounded server. Without an explicit
#    --gpu-memory-utilization it defaults to 0.9 and eagerly reserves ~72GB of
#    an 80GB card, and every lm_eval below (each of which spins up its OWN
#    in-process vLLM engine on the same GPU) then fails at init with "free
#    memory is less than desired GPU memory utilization" -- which, under
#    `set -e`, kills the script before `kill $VLLM_PID` ever runs and leaves an
#    orphaned server holding the card. Budget the whole GPU explicitly
#    (0.3 server + 0.3 for one eval at a time) and clean up via trap.
python awq_quantize.py --model $CKPT --out ckpts/stack-100m-awq-w4g64
vllm serve ckpts/stack-100m-awq-w4g64 --served-model-name stack100m \
    --gpu-memory-utilization 0.3 --port 8000 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

# 2. CPU/laptop path: convert to GGUF, quantize with a K-quant, serve with
#    llama.cpp -- independent of the GPU path above, same source checkpoint.
python llama.cpp/convert_hf_to_gguf.py $CKPT --outfile stack100m-f16.gguf
./llama.cpp/build/bin/llama-quantize stack100m-f16.gguf stack100m-Q4_K_M.gguf Q4_K_M

# 3. Evaluate BOTH shipped artifacts against the same battery -- fp16 baseline,
#    AWQ, and (via llama-cpp-python's HF-compatible wrapper or a small custom
#    lm_eval model class) the GGUF checkpoint. Never ship without this step.
lm_eval --model vllm \
    --model_args pretrained=$CKPT,dtype=bfloat16,gpu_memory_utilization=0.3 \
    --tasks arc_easy,piqa,stack_arithmetic_2digit --include_path tasks/ \
    --output_path results/base.json

lm_eval --model vllm \
    --model_args pretrained=ckpts/stack-100m-awq-w4g64,dtype=bfloat16,gpu_memory_utilization=0.3 \
    --tasks arc_easy,piqa,stack_arithmetic_2digit --include_path tasks/ \
    --output_path results/awq.json

# 4. Smoke-test the endpoint that will actually take production traffic --
#    the served artifact, not just the offline engine the evals used.
for _ in $(seq 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
curl -sf http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"stack100m","messages":[{"role":"user","content":"Compute 37 + 8. Give the final integer after #### ."}],"temperature":0,"max_tokens":64}' \
    > results/serve_smoke.json

# The EXIT trap above stops the server, including on failure.
```

Notice what is *not* in this script: any code implementing continuous batching, GPTQ's Hessian, K-quant packing, or cloze scoring. All of that lives in [Ch. 14.11](../14-capstone/11-evaluation-and-serving.html), where you built it once, by hand, to understand it — and none of it belongs in a script you run every release.

That gap is also exactly where version drift lives. [Chapter 15.1](01-toolchain-map.html) named the general discipline (pin, freeze, verify fields against your actual installed version before trusting a snippet); this chapter's specific hazards are: `vllm serve`'s CLI flags and its Transformers-backend fallback flag, which move roughly on vLLM's release cadence; `llm-compressor`'s `oneshot()` signature, which has already changed shape once since the project's early releases; llama.cpp's converter script name, which has been renamed twice; and `lm-evaluation-harness`'s task-YAML schema, where filter/metric keys occasionally shift between major versions. None of this is a reason to avoid these tools — it is a reason to pin every one of them together in one lockfile, record that lockfile next to the shipped checkpoint (the same reproducibility discipline [Ch. 14.12](../14-capstone/12-retrospective-and-scaleup.html) demands for the training side), and treat each library's own `examples/` directory, not this chapter, as the ground truth the day a flag stops working.

!!! interview "Interview Corner"

    **Q:** You need to ship a fine-tuned checkpoint behind an OpenAI-compatible endpoint on a single GPU. A teammate suggests GPTQ; another suggests AWQ. How do you decide, and does llama.cpp/GGUF have any role once you're already serving from a GPU with vLLM?

    **A:** Start from the mechanism difference. GPTQ quantizes column-by-column and *corrects* each column's error into the columns not yet quantized using a calibration-set Hessian — it can push to very low bit-widths but needs a representative calibration set and more compute to fit well; get the calibration data wrong (out-of-distribution text, too few samples) and the correction can overfit to noise. AWQ instead *protects* a small set of salient input channels by rescaling before quantizing, found via a cheap one-parameter grid search rather than a Hessian inversion — generally more robust to calibration-set choice and simpler to get right, at a slightly higher headline bit-error on some benchmarks. In practice: start with AWQ as the default for its lower calibration sensitivity and simpler failure mode; reach for GPTQ if your target serving kernel specifically expects GPTQ's format, or you need a bit-width AWQ's kernel doesn't cover well. Either way, the decision is not a benchmark-paper fact you import — it's an empirical question about *your* model and *your* task mix, so run both through the same eval battery (`lm-evaluation-harness`, ideally including a task that mirrors your production traffic) before choosing, exactly the "measure the damage" discipline from quantizing by hand. On llama.cpp/GGUF: it is not a competitor to vLLM on the GPU path at all — it is the answer to a different deployment question, CPU-only or edge/laptop nodes with no GPU available. A real production topology often runs both: vLLM+AWQ/GPTQ on GPU nodes for high-throughput serving, and a GGUF build for offline, on-device, or cost-sensitive tiers — same source checkpoint, two different engines for two different constraints.

!!! key "Key Takeaways"

    - **The interfaces are standard formats, not monoliths.** A HF-format checkpoint, verified against the from-scratch model (logit-match on the `trust_remote_code` path, held-out perplexity on the stock-`LlamaConfig` path that drops NoPE/QK-norm), is the one bridge every downstream tool needs; once you have it, vLLM, `llm-compressor`, `AutoAWQ`, and llama.cpp's converter all just work.
    - **vLLM replaces the token loop, not the mechanism.** Continuous batching (iteration-level scheduling) plus PagedAttention (block-based KV memory) is what turns Ch. 14.11's one-request-at-a-time `generate()` into a server; `vllm serve` gives you that plus an OpenAI-compatible API and Prometheus metrics for free.
    - **A 100M model on a data-center GPU is essentially never KV-cache-bound.** The worked example's ~730-sequence concurrency headroom is the opposite regime from a 70B model, where PagedAttention's memory efficiency is the whole point — tune `--max-num-seqs`/`--max-num-batched-tokens` against your latency SLO instead.
    - **`llm-compressor` (GPTQ) and `AutoAWQ` replace the from-scratch RTN sketch with calibration-driven quantizers and a standard `compressed-tensors` format vLLM loads with dequant-fused int4 kernels** — a real speedup, because the weights are unpacked inside the matmul rather than written back to HBM first, unlike the hand-rolled `QuantizedLinear`'s dequantize-then-fp32-matmul. (The math still runs in bf16 at `W4A16`; the win is 4× less weight traffic, not lower-precision arithmetic.)
    - **GGUF's K-quant super-blocks beat a naive per-group scheme at a similar bit budget** by right-sizing scale precision to sub-block granularity — the production answer to Ch. 14.11's laptop demo.
    - **`lm-evaluation-harness` trades bespoke-but-uncomparable for standardized-and-comparable**, and a custom task YAML can port your own probe (the arithmetic exact-match) directly, keeping the exact prompt format and verifier your RLVR stage trained against.
    - **Re-run the eval after every edit** — export, quantize, convert — because each one is a hypothesis about quality, not a free lunch; this discipline does not go away because the tools got better.
    - **Pin every library in one lockfile and record it with the checkpoint.** Flags and APIs across this chapter's four tools move on a timescale of months, not years; the upstream `examples/` directory is the ground truth the day a snippet here stops working.

## Further Reading

- Kwon, Li, Zhuang, Sheng, et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, SOSP 2023) — the memory mechanism behind vLLM's serving.
- Yu, Jeong, Kim, Kim, Chun, *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — the origin of iteration-level continuous batching.
- Frantar, Ashkboos, Hoefler, Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022).
- Lin, Tang, Tang, Yang, Dang, Han, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (MLSys 2024 Best Paper).
- Gao, Tow, Abbasi, et al., *The Language Model Evaluation Harness* (EleutherAI, `lm-evaluation-harness`) — the harness and its task registry.
- Gerganov et al., **`llama.cpp`** / **`ggml`** — the CPU inference engine and tensor library behind GGUF.
- **GGUF format specification** (`ggml-org/ggml`) — the on-disk layout, including K-quant super-block packing.
- **`llm-compressor`** (vLLM project / Neural Magic) and **`compressed-tensors`** — the maintained quantization toolchain and on-disk format vLLM reads natively.
- **`AutoAWQ`** (Casper Hansen) — the original standalone AWQ implementation, used above; retired/archived in 2025, with AWQ production now flowing through `llm-compressor`'s `AWQModifier` (and `GPTQModel`), per [Ch. 4.7](../04-kernels-efficiency/07-quantization-ptq.html).
- **`inspect_ai`** (UK AI Safety Institute) — the agentic-evaluation framework referenced for the ReAct-loop grading Ch. 14.11 built by hand.
