# %% [markdown]
# # INT8 & NF4 Quantization: Memory and Latency on GPU
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will quantize a small causal LM to INT8 (`LLM.int8()`) and 4-bit NF4 (QLoRA's data type) with bitsandbytes, then measure bytes/parameter, peak GPU memory, decode latency, and a correctness sanity check (logit MSE + perplexity) against the BF16 baseline.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/08-quantization-formats-qat.html) for the full explanation.

# %%
# bitsandbytes ships prebuilt CUDA kernels for LLM.int8() and NF4; transformers/accelerate
# wire it into `from_pretrained(..., quantization_config=...)`.
# Alternative worth knowing about: `torchao` (PyTorch's own quantization library) implements
# int8 weight-only, int4 (tinygemm/groupwise), and NF4 quantization with torch.compile-friendly
# kernels — often faster than bitsandbytes' generic dequant-then-fp16-GEMM path. We use
# bitsandbytes here because it is the format described in the chapter (QLoRA's NF4).
%pip install -q "transformers>=4.41" "accelerate>=0.30" bitsandbytes
# %pip install -q torchao  # alternative library; not used in this notebook

# %%
import gc

import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "This notebook needs a CUDA GPU (targets 1x H100 80GB)."
device = torch.device("cuda")
torch.manual_seed(0)

# bf16 is Hopper's native training/inference dtype; it is also the "compute dtype" bitsandbytes
# dequantizes into before running the actual GEMM in the NF4 path below.
compute_dtype = torch.bfloat16

try:
    import bitsandbytes  # noqa: F401  (imported for its side effect of registering CUDA ops)
except ImportError as e:
    raise SystemExit(
        "bitsandbytes is required for this notebook. Run the %pip install cell above first."
    ) from e

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# A small, ungated, Llama-architecture chat model — big enough that quantization has a
# real, measurable effect, small enough to load three separate copies quickly for comparison.
# Swap in a larger model (e.g. an 8B Llama-3.1 checkpoint) if you have the download bandwidth;
# on an 80GB H100 you have plenty of headroom to go much bigger than this.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Llama tokenizers have no pad token by default

print(f"Model: {MODEL_ID}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Three configurations, one model
#
# We load the *same* checkpoint three times — once per quantization scheme — and measure each
# in isolation (peak memory reset, model deleted and cache emptied between loads) so the numbers
# are comparable:
#
# 1. **bf16** — the unquantized baseline (2 bytes/param).
# 2. **int8** — `load_in_8bit=True`, bitsandbytes' `LLM.int8()` mixed-precision decomposition
#    (outlier feature dimensions kept in fp16, the rest quantized per-column to INT8; ~1 byte/param).
# 3. **nf4** — `load_in_4bit=True` with `bnb_4bit_quant_type="nf4"` and double quantization,
#    QLoRA's 4-bit NormalFloat codebook (~0.5-0.6 bytes/param once you include scale overhead).
#
# Expected result: bytes/param should step down roughly 16 &rarr; 8 &rarr; ~4-5 (in bits), i.e.
# close to a 2x memory reduction from bf16&rarr;int8 and roughly 3-4x from bf16&rarr;nf4 — not
# exactly 4x, because scales and zero-points (plus the small fraction of fp16 outlier weights in
# the int8 path) add metadata overhead on top of the raw quantized weights.

# %%
def make_quant_config(kind):
    """Return a BitsAndBytesConfig for 'int8' / 'nf4', or None for the bf16 baseline."""
    if kind == "bf16":
        return None
    if kind == "int8":
        # LLM.int8(): per-column INT8 with runtime outlier-channel decomposition into fp16.
        return BitsAndBytesConfig(load_in_8bit=True)
    if kind == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",              # vs "fp4"; nf4 is QLoRA's default
            bnb_4bit_compute_dtype=compute_dtype,    # dequantize into bf16 for the actual GEMM
            bnb_4bit_use_double_quant=True,          # quantize the group scales themselves
        )
    raise ValueError(f"unknown kind: {kind}")


def load_model(kind):
    """Load MODEL_ID under the given quantization scheme, pinned to a single GPU."""
    quant_config = make_quant_config(kind)
    kwargs = dict(device_map={"": 0})  # pin everything to cuda:0 so memory numbers are comparable
    if quant_config is None:
        kwargs["torch_dtype"] = compute_dtype
    else:
        kwargs["quantization_config"] = quant_config
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def measure_load_footprint(kind):
    """Load a model and report (model, resident-memory delta, peak memory during load), in bytes."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    model = load_model(kind)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    peak_during_load = torch.cuda.max_memory_allocated()
    footprint = mem_after - mem_before  # net resident bytes attributable to the loaded weights
    return model, footprint, peak_during_load

# %% [markdown]
# ## Decode latency: measure with CUDA events, not the wall clock
#
# GPU kernels launch asynchronously, so `time.time()` around a `.generate()` call mostly measures
# Python/dispatch overhead unless you synchronize. We use `torch.cuda.Event` timers with a warmup
# phase (to pay for CUDA graph/kernel-selection caching once, outside the timed region) and a
# final `torch.cuda.synchronize()` so the elapsed time reflects actual GPU work.
#
# Expected result at this small model size: weight-only quantization (int8/nf4) reduces *memory*
# reliably, but decode *latency* on a single small request is not guaranteed to improve — and can
# even regress slightly — because bitsandbytes must dequantize each weight block back to bf16
# before the GEMM, adding a kernel on the critical path. The latency win from low-bit weights
# shows up most clearly at larger models / larger batch sizes, where HBM bandwidth (not the
# dequant kernel) is the bottleneck, or with fused low-bit GEMM kernels (e.g. AWQ/GPTQ + Marlin,
# or torchao's int4 kernels) that skip a separate dequant step entirely.

# %%
@torch.no_grad()
def benchmark_decode(model, prompt, new_tokens=64, warmup_iters=2, timed_iters=5):
    """Time `timed_iters` full generate() calls (prefill + new_tokens decode steps) with CUDA events."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,  # force a fixed decode length so latency is comparable
        do_sample=False,            # greedy decoding: deterministic, no sampling-kernel noise
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    for _ in range(warmup_iters):
        model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(timed_iters):
        model.generate(**inputs, **gen_kwargs)
    end_evt.record()
    torch.cuda.synchronize()

    total_ms = start_evt.elapsed_time(end_evt)
    ms_per_call = total_ms / timed_iters
    ms_per_token = ms_per_call / new_tokens
    tokens_per_sec = 1000.0 / ms_per_token
    return ms_per_token, tokens_per_sec

# %% [markdown]
# ## Correctness sanity check: logit MSE and perplexity
#
# Memory savings are worthless if the model's outputs drift too far from the original. We run a
# single teacher-forced forward pass over a short fixed passage and compare:
#
# - **Logit MSE** against the bf16 baseline's logits (cast to fp32 for a fair comparison).
# - **Perplexity** of each model on the same passage (next-token cross-entropy, exponentiated).
#
# Expected result: for INT8 (`LLM.int8()`'s outlier decomposition) the logit MSE should be tiny
# and perplexity should match bf16 almost exactly. For NF4, expect a small but non-zero perplexity
# increase — bitsandbytes documents NF4 as very close to bf16 for most models, but "very close"
# is not "identical", and round-to-nearest 4-bit quantization has a real, if usually small,
# accuracy cost. This one-passage check is illustrative only — a real accuracy audit needs a
# held-out corpus (e.g. WikiText-2, as in the PTQ chapter's calibration examples), not one sentence.

# %%
@torch.no_grad()
def logits_and_perplexity(model, text):
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # Pass the attention mask explicitly so the forward pass is unambiguous (and warning-free).
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = input_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels)
    perplexity = torch.exp(loss).item()
    return logits.cpu(), perplexity

# %% [markdown]
# ## Run all three configurations
#
# For each of `bf16`, `int8`, `nf4` we: load the model, measure its resident memory footprint,
# run the correctness check against the bf16 baseline, benchmark decode latency, then delete the
# model and clear the CUDA cache before moving to the next configuration. `NUM_PARAMS` is counted
# once from the *unquantized* bf16 model and reused for every bytes/param calculation, because a
# packed 4-bit tensor reports half as many elements as the logical number of weights it encodes —
# counting params from the quantized model itself would silently corrupt the bytes/param ratio.

# %%
EVAL_TEXT = (
    "The transformer architecture replaced recurrence with self-attention as the core "
    "building block of large language models. Its scalability properties are the reason "
    "modern LLMs can be trained on trillions of tokens across thousands of accelerators, "
    "and they are also why serving these models cheaply depends so heavily on quantization."
)
PROMPT = "The key idea behind quantization is"

results = {}
baseline_logits = None
NUM_PARAMS = None

for kind in ["bf16", "int8", "nf4"]:
    print(f"\n=== {kind} ===")
    model, footprint_bytes, peak_bytes = measure_load_footprint(kind)

    if NUM_PARAMS is None:
        # Only valid here because this is the bf16 (unquantized) pass — see markdown above.
        NUM_PARAMS = sum(p.numel() for p in model.parameters())

    logits, ppl = logits_and_perplexity(model, EVAL_TEXT)
    if kind == "bf16":
        baseline_logits = logits
        logit_mse = 0.0
    else:
        logit_mse = F.mse_loss(logits, baseline_logits).item()

    ms_per_token, tokens_per_sec = benchmark_decode(model, PROMPT)

    results[kind] = dict(
        footprint_gb=footprint_bytes / 1e9,
        peak_load_gb=peak_bytes / 1e9,
        bytes_per_param=footprint_bytes / NUM_PARAMS,
        perplexity=ppl,
        logit_mse=logit_mse,
        ms_per_token=ms_per_token,
        tokens_per_sec=tokens_per_sec,
    )
    for k, v in results[kind].items():
        print(f"  {k:16s}: {v:.4g}")

    free_model(model)

# %% [markdown]
# ## A note on round-to-nearest vs. calibrated quantization
#
# Everything measured above is essentially **round-to-nearest (RTN)** quantization: bitsandbytes
# picks each group's scale from that group's own min/max (or max-abs, for NF4) statistics, with no
# reference to how the weights are actually used downstream. `LLM.int8()` adds one refinement —
# runtime outlier-channel detection — but the non-outlier majority is still plain per-column RTN.
#
# Calibration-based post-training quantization goes further: it runs a small calibration dataset
# through the model and *solves* for scales (and sometimes updated weights) that minimize the
# actual output error, rather than just the per-group rounding error. The two dominant algorithms,
# covered in depth in the [PTQ chapter](../04-kernels-efficiency/07-quantization-ptq.html), are:
#
# - **GPTQ** (Frantar et al.) — greedy layer-by-layer weight updates using second-order (Hessian)
#   information from the calibration data, quantizing one column at a time and compensating the
#   remaining columns for the error just introduced.
# - **AWQ** (Lin et al.) — activation-aware per-channel scaling: it scales up the small fraction of
#   weight channels that correspond to large-magnitude activations before quantizing, so RTN error
#   concentrates where it matters least.
#
# Both are implemented in dedicated serving-side libraries (AutoGPTQ, AutoAWQ, and inference
# engines like vLLM/TensorRT-LLM that consume their checkpoints) rather than reimplemented here.
# At 4 bits and below, calibrated GPTQ/AWQ checkpoints typically hold perplexity closer to bf16
# than plain RTN/NF4 does, especially on more aggressive (e.g. per-tensor or very low bit-width)
# configurations — the gap tends to narrow as group size shrinks and widen as it grows.

# %% [markdown]
# ## What you should see
#
# Order-of-magnitude expectations for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on an H100 (your exact
# numbers will vary with driver/library versions — treat all of these as rough guides, not targets
# to match exactly):
#
# - **Memory**: bf16 footprint on the order of ~2.0-2.5 GB; int8 roughly half that (~1.0-1.5 GB,
#   i.e. bytes/param close to but a bit above 1.0 due to the fp16 outlier columns); nf4 the
#   smallest, on the order of ~0.6-0.9 GB (bytes/param roughly 0.5-0.7 once group scales and
#   double-quantization overhead are included) — a ~3-4x reduction from bf16, not a clean 4x.
# - **Latency**: don't expect a clean speedup ladder here. At this small model size, batch size 1,
#   int8/nf4 decode latency on an H100 may be comparable to, or even slower than, bf16 — the
#   dequantize-then-GEMM path adds kernel overhead that a 1.1B model's tiny GEMMs don't amortize
#   well. The latency case for weight-only quantization gets much stronger at larger model sizes
#   or larger batches, where HBM bandwidth dominates.
# - **Accuracy**: int8 logit MSE and perplexity should sit essentially on top of the bf16 baseline;
#   nf4 should show a small, non-zero perplexity increase and a noticeably larger (but still
#   small) logit MSE than int8 — consistent with 4-bit RTN paying a real, if modest, accuracy cost.
#
# **Key takeaways**
#
# 1. Bytes/param step down roughly 2 &rarr; 1 &rarr; ~0.5-0.6 across bf16 &rarr; int8 &rarr; nf4;
#    the "~4x" figure for 4-bit is an approximation once scale/zero-point metadata is counted.
# 2. Weight-only quantization is primarily a **memory** win; it is not automatically a **latency**
#    win at small scale, because the runtime dequant kernel sits on the critical path.
# 3. bitsandbytes' int8/nf4 are RTN-family methods; GPTQ/AWQ calibration (previous chapter, and
#    AutoGPTQ/AutoAWQ in practice) buys back accuracy at the same bit-width, at the cost of an
#    offline calibration pass.
#
# **Next step**: re-run this notebook with a larger model (7-8B+) and a larger generated length to
# see the latency picture shift in quantization's favor, or swap bitsandbytes for `torchao`'s
# int4 weight-only kernels / a Marlin-backed GPTQ checkpoint to see what a fused (non-bitsandbytes)
# low-bit GEMM kernel does for decode throughput on the same hardware.
