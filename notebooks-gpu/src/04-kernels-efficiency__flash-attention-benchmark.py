# %% [markdown]
# # FlashAttention vs Naive Attention: A GPU Benchmark
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will benchmark a naive, materialized-softmax attention implementation against
# `torch.nn.functional.scaled_dot_product_attention` (SDPA) under its `MATH`, `EFFICIENT_ATTENTION`,
# and `FLASH_ATTENTION` backends, in bf16, causal, across a sweep of sequence lengths — and watch the
# naive path's memory blow up quadratically while FlashAttention-style kernels stay linear.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/02-flash-attention-1.html) for the full explanation.

# %%
# Dependencies: only torch (ships with SDPA and its FLASH/EFFICIENT/MATH backends built in) and,
# optionally, matplotlib for the plot at the end (the notebook falls back to a printed table if it's
# missing). Nothing else is required.
# %pip install -q matplotlib
# (Optional) the standalone flash-attn package — only if you want to try its kernel directly in the
# commented section near the end. Building it from source can take several minutes:
# %pip install -q flash-attn --no-build-isolation

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

assert torch.cuda.is_available(), "This notebook requires a CUDA GPU (targets 1x H100 80GB)."
device = torch.device("cuda")
dtype = torch.bfloat16

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Device:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("torch:", torch.__version__)

# %% [markdown]
# ## Step 1 — a naive, materialized-softmax attention
#
# This is the textbook formula computed literally: form the full `(seq, seq)` score matrix,
# apply the causal mask, run `softmax`, then multiply by `V`. Every intermediate — scores, the
# masked scores, the softmax probabilities — is a full `(batch, heads, seq, seq)` tensor that
# PyTorch materializes in HBM. That's the O(seq^2) memory FlashAttention (Dao et al., "FlashAttention:
# Fast and Memory-Efficient Exact Attention with IO-Awareness") avoids by fusing the whole
# score -> softmax -> weighted-sum pipeline into one kernel that never writes the full matrix to HBM.
#
# Expected result: for short sequences this is fine; past a few thousand tokens the `(seq, seq)`
# score matrix (and the fp32 copy softmax needs for numerical stability) starts to dominate memory,
# and at reasonable batch/head counts it is expected to OOM on an 80GB GPU somewhere in the
# long-context range.


# %%
def naive_causal_attention(q, k, v):
    """Naive scaled-dot-product causal attention with an explicitly materialized (seq, seq) score
    matrix. q, k, v: (batch, heads, seq, head_dim), same dtype (bf16 here).
    Softmax is computed in fp32 for numerical stability, matching what SDPA's math backend does
    internally, then cast back to bf16 before the final matmul.
    """
    batch, heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    # (batch, heads, seq, seq) -- this is the tensor that makes memory quadratic in seq_len.
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask: position i can only attend to positions <= i.
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))

    # Softmax in fp32 for stability, then back to the compute dtype.
    probs = F.softmax(scores.float(), dim=-1).to(v.dtype)

    out = torch.matmul(probs, v)  # (batch, heads, seq, head_dim)
    return out


# %% [markdown]
# ## Step 2 — the SDPA backends we'll compare
#
# `torch.nn.functional.scaled_dot_product_attention` dispatches to one of several fused kernels.
# We force each explicitly with `torch.nn.attention.sdpa_kernel(...)` so the comparison is apples
# to apples:
#   - `MATH`: a reference/fallback implementation (similar spirit to our naive version, but PyTorch's
#     own kernel rather than hand-rolled ops). It materializes the full score matrix, so its memory
#     also grows quadratically.
#   - `EFFICIENT_ATTENTION`: the memory-efficient attention kernel (in the spirit of Rabe & Staats,
#     "Self-attention Does Not Need O(n^2) Memory") — never materializes the full score matrix.
#   - `FLASH_ATTENTION`: PyTorch's built-in FlashAttention-2-style kernel (Dao, "FlashAttention-2:
#     Faster Attention with Better Parallelism and Work Partitioning") — tiled, IO-aware, and on
#     Hopper (H100) typically the fastest of the three for this shape range.
#
# Expected result: EFFICIENT and FLASH should show roughly linear-in-seq_len memory growth, while
# MATH tracks the naive path since it materializes the score matrix too.

# %%
def run_sdpa(backend: SDPBackend, q, k, v):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


BACKENDS = {
    "math": SDPBackend.MATH,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
}

# %% [markdown]
# ## Step 3 — a GPU-correct timing + peak-memory harness
#
# GPU kernels launch asynchronously, so `time.time()` around a call mostly measures Python/launch
# overhead, not actual device work. We use `torch.cuda.Event(enable_timing=True)` pairs plus an
# explicit `torch.cuda.synchronize()`, and we always run a few warmup iterations first (the very
# first call to a kernel/backend pays one-time allocator and autotuning costs).
# We measure peak memory with `torch.cuda.reset_peak_memory_stats()` +
# `torch.cuda.max_memory_allocated()`, reset immediately before each measured run. Note the reported
# peak includes the shared q/k/v inputs (allocated once outside), so the *difference* between methods
# is exactly the attention intermediates — which is what we want to compare.

# %%
def bench(fn, *args, warmup=5, iters=20):
    """Return (median_latency_ms, peak_memory_bytes) for calling fn(*args) repeatedly on GPU.
    Returns (None, None) if fn OOMs, after cleaning up so subsequent benchmarks can still run.
    """
    try:
        # Warmup: let cuBLAS/cuDNN autotune and the caching allocator settle before we measure.
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()

        # Reset the peak counter right before the measured loop so it reflects this run only.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn(*args)
            ends[i].record()
        torch.cuda.synchronize()

        times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        median_ms = times_ms[len(times_ms) // 2]
        peak_bytes = torch.cuda.max_memory_allocated()
        return median_ms, peak_bytes
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return None, None


# %% [markdown]
# ## Step 4 — sweep sequence length, fixed batch/heads/head_dim
#
# Fixed shape knobs chosen to look like a real decoder-only transformer block on one H100:
# batch=4, heads=32, head_dim=128 (so model width = 32*128 = 4096, roughly GPT-3-scale width per
# layer). We sweep `seq_len` from short (512) up to long-context (16384) — the naive path (and the
# `math` backend) are expected to OOM somewhere in that range at these batch/head counts; when they
# do we record it as OOM and move on rather than crashing the whole sweep.

# %%
BATCH = 4
HEADS = 32
HEAD_DIM = 128
SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]

results = []  # list of dicts: seq_len, method, latency_ms (or None), peak_gb (or None)

for seq_len in SEQ_LENS:
    q = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(BATCH, HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)

    # --- naive materialized-softmax attention ---
    ms, peak = bench(naive_causal_attention, q, k, v)
    results.append({"seq_len": seq_len, "method": "naive", "latency_ms": ms,
                    "peak_gb": None if peak is None else peak / 1e9})

    # --- SDPA under each backend ---
    for name, backend in BACKENDS.items():
        ms, peak = bench(lambda q=q, k=k, v=v, b=backend: run_sdpa(b, q, k, v))
        results.append({"seq_len": seq_len, "method": name, "latency_ms": ms,
                        "peak_gb": None if peak is None else peak / 1e9})

    del q, k, v
    torch.cuda.empty_cache()

    print(f"seq_len={seq_len:>6} done")

# %% [markdown]
# ## Step 5 — correctness sanity check
#
# Before trusting the speed numbers, confirm the fused kernels agree with the naive reference on a
# small, cheap shape. bf16 has ~8 mantissa bits (~2-3 decimal digits), and different kernels sum in
# different orders, so we expect small (1e-2-scale) disagreements — not zero. We print the actual max
# abs diff for each backend and assert only a loose bound that a genuinely broken kernel (which would
# differ by order-1) could never pass.

# %%
q_small = torch.randn(2, 8, 256, HEAD_DIM, device=device, dtype=dtype)
k_small = torch.randn(2, 8, 256, HEAD_DIM, device=device, dtype=dtype)
v_small = torch.randn(2, 8, 256, HEAD_DIM, device=device, dtype=dtype)

ref = naive_causal_attention(q_small, k_small, v_small)
for name, backend in BACKENDS.items():
    out = run_sdpa(backend, q_small, k_small, v_small)
    max_abs_diff = (out.float() - ref.float()).abs().max().item()
    print(f"{name:>14}: max_abs_diff vs naive = {max_abs_diff:.4f}")
    # Loose bound: catches a broken kernel (order-1 error) without flaking on bf16 reduction-order
    # noise, which is typically well under 0.05 here.
    assert max_abs_diff < 0.3, f"{name} diverged from naive reference (max_abs_diff={max_abs_diff:.4f})"

print("\nAll backends match the naive reference within bf16 tolerance.")

# %% [markdown]
# ## Step 6 — tabulate and plot: latency and memory vs sequence length
#
# We print a table and (if matplotlib is available) plot both curves on log-log axes. Watch for:
#   - `naive` and `math` peak memory growing roughly quadratically with `seq_len` (a steeper line
#     on a log-log memory plot), until one of them hits "OOM" and drops out of the sweep.
#   - `mem_efficient` and `flash` peak memory growing roughly linearly with `seq_len`,
#     staying well under the naive curve at long context.
#   - `flash` latency pulling ahead of `math`/naive as `seq_len` grows, since it also cuts redundant
#     HBM reads/writes, not just peak memory.

# %%
print(f"{'seq_len':>8} {'method':>14} {'latency_ms':>12} {'peak_gb':>10}")
for r in results:
    lat = f"{r['latency_ms']:.2f}" if r["latency_ms"] is not None else "OOM"
    mem = f"{r['peak_gb']:.2f}" if r["peak_gb"] is not None else "OOM"
    print(f"{r['seq_len']:>8} {r['method']:>14} {lat:>12} {mem:>10}")

try:
    import matplotlib.pyplot as plt

    methods = ["naive", "math", "mem_efficient", "flash"]
    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(12, 5))

    for method in methods:
        xs = [r["seq_len"] for r in results if r["method"] == method and r["latency_ms"] is not None]
        ys = [r["latency_ms"] for r in results if r["method"] == method and r["latency_ms"] is not None]
        if xs:
            ax_lat.plot(xs, ys, marker="o", label=method)

    ax_lat.set_xscale("log", base=2)
    ax_lat.set_yscale("log")
    ax_lat.set_xlabel("sequence length")
    ax_lat.set_ylabel("median latency (ms)")
    ax_lat.set_title("Latency vs sequence length")
    ax_lat.legend()
    ax_lat.grid(True, which="both", alpha=0.3)

    for method in methods:
        xs = [r["seq_len"] for r in results if r["method"] == method and r["peak_gb"] is not None]
        ys = [r["peak_gb"] for r in results if r["method"] == method and r["peak_gb"] is not None]
        if xs:
            ax_mem.plot(xs, ys, marker="o", label=method)

    ax_mem.set_xscale("log", base=2)
    ax_mem.set_yscale("log")
    ax_mem.set_xlabel("sequence length")
    ax_mem.set_ylabel("peak memory (GB)")
    ax_mem.set_title("Peak memory vs sequence length")
    ax_mem.legend()
    ax_mem.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()
except ImportError:
    print("(matplotlib not installed; skipping plot -- the printed table above has the same data.)")

# %% [markdown]
# ## Optional: the standalone `flash-attn` package
#
# PyTorch's built-in `FLASH_ATTENTION` SDPA backend already implements a FlashAttention-2-style
# kernel, so for most code you don't need the separate package. If you want to compare directly
# against the reference `flash-attn` implementation (Dao et al.), here is the equivalent call —
# note its API takes `(batch, seq, heads, head_dim)` (heads and seq swapped relative to SDPA's
# `(batch, heads, seq, head_dim)`), so the inputs need a transpose:
#
# ```python
# # %pip install flash-attn --no-build-isolation
# from flash_attn import flash_attn_func
#
# q_fa = q.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
# k_fa = k.transpose(1, 2).contiguous()
# v_fa = v.transpose(1, 2).contiguous()
# out = flash_attn_func(q_fa, k_fa, v_fa, causal=True)  # -> (batch, seq, heads, head_dim)
# ```
#
# You'd benchmark it with the same `bench()` harness above. Building `flash-attn` from source can
# take several minutes on first install; that's why it's commented out rather than installed by
# default in this notebook.

# %% [markdown]
# ## What you should see
#
# - At short sequence lengths (512-1024) all four methods should be reasonably close in latency —
#   the fused kernels' advantage grows with sequence length, and at these sizes launch overhead and
#   compute (not memory traffic) dominate.
# - As `seq_len` grows past a few thousand, `naive` and `math` peak memory should grow roughly
#   *quadratically* (visibly steeper on the log-log plot) while `mem_efficient` and `flash` grow
#   roughly *linearly*. On an 80GB H100 at batch=4, heads=32, head_dim=128, expect `naive` (and
#   likely `math`) to OOM somewhere in the ~8k-16k range, while `flash` and `mem_efficient` should
#   comfortably reach 16384 and beyond.
# - At long sequence lengths, expect `flash` to be on the order of a few times faster than
#   `naive`/`math`, with `mem_efficient` usually in between. Exact ratios depend on shapes,
#   PyTorch/driver version, and clocks — treat these as orders of magnitude, not guarantees, and read
#   off your own printed table above.
#
# **Key takeaways:**
# 1. The naive formula and FlashAttention compute the *exact same* attention output (see the Step 5
#    correctness check) — FlashAttention is a faster, more memory-efficient *algorithm* for the same
#    math, via tiling and online softmax, not an approximation.
# 2. Peak memory, not just FLOPs, is what breaks the naive path: the `(seq, seq)` score matrix is
#    the O(seq^2) term that FlashAttention-style kernels never materialize in HBM.
# 3. In practice you rarely call a raw kernel directly — `F.scaled_dot_product_attention` picks a
#    fast backend automatically, and `sdpa_kernel(...)` (as used here) is mainly a debugging/
#    benchmarking tool to force a specific one.
#
# **Next step:** see the chapter on Triton kernels
# (`notebooks-gpu/src/04-kernels-efficiency__triton-fused-kernel.py`) to write and benchmark a fused
# kernel yourself, and the [FlashAttention chapter](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/02-flash-attention-1.html)
# for the tiling/online-softmax derivation behind why this works.
