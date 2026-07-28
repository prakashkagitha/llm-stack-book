# %% [markdown]
# # Writing a Fused Triton Kernel (Fused Softmax) and Benchmarking It
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will write a real, numerically-stable row-softmax kernel in Triton (`@triton.jit`, a 1-D grid over rows, `tl.program_id`, masked `tl.load`/`tl.store`), verify it against `torch.softmax`, and benchmark it against eager PyTorch and `torch.compile` across row widths to see the memory-bound, single-HBM-round-trip argument play out in wall-clock numbers.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/04-triton-kernels.html) for the full explanation.

# %%
# Triton ships as a standard dependency of the CUDA torch wheel, but pin/upgrade
# it explicitly in case the environment has an older version. matplotlib is for
# the benchmark plot at the end.
%pip install -q -U triton matplotlib

# %%
import math

import matplotlib.pyplot as plt
import torch

import triton
import triton.language as tl

assert torch.cuda.is_available(), "This notebook needs a CUDA GPU (targets 1x H100 80GB)."
device = torch.device("cuda")
dtype = torch.bfloat16  # bf16 activations, as in a real transformer's softmax (attention scores, output logits)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

props = torch.cuda.get_device_properties(device)
print(f"Device: {props.name}, {props.total_memory / 2**30:.1f} GiB total, "
      f"{props.multi_processor_count} SMs")

# %% [markdown]
# ## Why softmax is a fusion opportunity
#
# A row-softmax written the "obvious" way in eager PyTorch —
# `x.max(-1)`, `x - m`, `x.exp()`, `.sum(-1)`, `numerator / denominator` — is five
# separate elementwise/reduction ops. Each one is its own CUDA kernel launch, and
# each one reads its input from HBM and writes its output back to HBM. For an
# `(n_rows, n_cols)` tensor that is on the order of 5 reads + 4 writes of the row
# data (some ops read/write the full row, the reductions read the full row and
# write a single scalar per row) before you get a final answer — several HBM
# round trips for an operation that is entirely memory-bound (softmax does almost
# no arithmetic per byte moved).
#
# A **fused** kernel does the whole row in one program: load the row into
# on-chip registers/shared memory once, compute the row max, the shifted
# exponentials, and the sum entirely on-chip, then write the normalized row back
# once. That is the theoretical floor for any op that must read `x` and produce
# an output the same shape: **one read + one write of HBM**, nothing more. This
# is exactly the "one kernel launch, one HBM round trip" argument from the
# chapter, and it is why a naive multi-kernel softmax is commonly a couple of
# times slower than a fused one at bandwidth-bound sizes — the ceiling is set by
# HBM bandwidth, not by how clever the kernel is. Your exact ratio will depend
# on GPU, PyTorch version, and row width; run the sweep below for your own.

# %%
def naive_softmax_eager(x: torch.Tensor) -> torch.Tensor:
    """A numerically-stable softmax written as separate eager ops.

    Each line below is its own CUDA kernel launch and its own HBM round trip:
    this is the "unfused" baseline the chapter argument is about, NOT a strawman
    -- it's literally what `x - x.max(); .exp(); .sum(); /` compiles to eagerly.
    """
    row_max = x.max(dim=-1, keepdim=True).values      # kernel 1: read x, write (n_rows, 1) maxes
    z = x - row_max                                    # kernel 2: read x + maxes, write z
    numerator = torch.exp(z)                            # kernel 3: read z, write numerator
    denominator = numerator.sum(dim=-1, keepdim=True)   # kernel 4: read numerator, write (n_rows, 1) sums
    return numerator / denominator                       # kernel 5: read numerator + sums, write output


# %% [markdown]
# ## The Triton kernel
#
# One program (`tl.program_id(0)`) handles exactly one row. `BLOCK_SIZE` is the
# compile-time (`tl.constexpr`) width of the tile each program loads — it must be
# a power of two and at least `n_cols`, so we round `n_cols` up with
# `triton.next_power_of_2`. Because `BLOCK_SIZE` is usually larger than `n_cols`
# (rows are rarely exact powers of two), every load/store is masked: out-of-range
# lanes read `-inf` (so they never win the max, and `exp(-inf) = 0` so they never
# contribute to the sum) and are simply not written back.
#
# `BLOCK_SIZE` (and the paired `num_warps`) is also the occupancy knob: it sets
# how many registers and how much shared memory one program needs, which sets
# how many programs (rows) an SM can run concurrently. Too small and you
# under-fill a warp's work; too large (e.g. a 32K-wide row) and a single
# program's working set can spill registers or limit how many programs fit on
# an SM at once, hurting occupancy even though each program does more useful
# work per launch. Production Triton kernels usually search this trade-off with
# `@triton.autotune` rather than hand-picking one point; we hand-pick a simple
# heuristic here for clarity and sweep it explicitly later.

# %%
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Which row this program instance is responsible for.
    row_idx = tl.program_id(0)

    # Column offsets within the row's BLOCK_SIZE-wide tile, and the mask for
    # the (likely) case that BLOCK_SIZE > n_cols.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pointer arithmetic: base address of this row, plus the per-column offsets.
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets

    # Single masked load of the whole row into registers. Out-of-bounds lanes
    # get -inf so they are inert in the max and in exp() below. Cast to fp32 for
    # the reduction/exp (the standard Triton softmax pattern): the input is
    # bf16, but the max-subtraction and exp are done in fp32 for accuracy, then
    # the result is cast back to the output dtype on store.
    row = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Numerically-stable softmax, entirely on-chip: no intermediate touches HBM.
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Single masked store of the whole row back to HBM.
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Launch one Triton program per row; grid = (n_rows,)."""
    assert x.ndim == 2 and x.is_cuda
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Occupancy heuristic: wider rows need more warps to keep the load/reduce
    # on-chip work parallel across the block. This mirrors the standard
    # Triton fused-softmax tutorial's rule of thumb.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = torch.empty_like(x)
    grid = (n_rows,)  # one program per row -> exactly n_rows kernel instances
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


# %% [markdown]
# ## Correctness: `torch.allclose` against `torch.softmax`
#
# Check both power-of-2 and deliberately awkward, non-power-of-2 column counts
# (so the masking path in the kernel is actually exercised, not just the case
# where `BLOCK_SIZE == n_cols`), and check that a row of large values doesn't
# overflow (the numerically-stable max-subtraction should handle this fine).

# %%
test_shapes = [(4, 128), (17, 1000), (256, 781), (8, 8192), (3, 50000)]
for n_rows, n_cols in test_shapes:
    x = torch.randn(n_rows, n_cols, device=device, dtype=dtype) * 5.0  # scale up to stress stability
    ref = torch.softmax(x, dim=-1)
    out = triton_softmax(x)
    ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)  # bf16 tolerance
    max_abs_diff = (out - ref).abs().max().item()
    print(f"shape={n_rows:>4}x{n_cols:<6} allclose={ok!s:<5} max_abs_diff={max_abs_diff:.4f}")
    assert ok, f"mismatch at shape {(n_rows, n_cols)}"

# Also sanity-check the eager "naive" baseline matches, and rows sum to ~1.
x = torch.randn(32, 4096, device=device, dtype=dtype)
assert torch.allclose(naive_softmax_eager(x), torch.softmax(x, dim=-1), atol=1e-2, rtol=1e-2)
row_sums = triton_softmax(x).float().sum(dim=-1)
print("row sums (should be ~1.0):", row_sums[:5].tolist())

# %% [markdown]
# ## Benchmark harness: `torch.cuda.Event`, warmup, and synchronization
#
# GPU kernels launch asynchronously, so `time.time()` around a single call
# mostly measures Python/launch overhead, not device time. We use paired
# `torch.cuda.Event(enable_timing=True)` markers, run several warmup
# iterations first (to get past lazy `torch.compile` tracing and any one-time
# allocator/cache effects), then time a batch of iterations and synchronize
# once at the end before reading elapsed time.

# %%
def bench_ms(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per call


# torch.compile the naive eager version. Inductor should fuse most of the
# elementwise+reduction chain into far fewer kernel launches than the five
# separate ops eager executes -- this is the "does the compiler already do
# this for you" comparison point.
compiled_softmax = torch.compile(naive_softmax_eager)

# Reset the peak-memory counter so the number printed after the sweep reflects
# this benchmark's allocations, not any earlier cells.
torch.cuda.reset_peak_memory_stats()

# %% [markdown]
# ## Sweep row width, compare four implementations
#
# Fixed `n_rows`, sweeping `n_cols` from short attention-score-like rows up to
# very wide vocabulary-logit-like rows. For each width we measure:
#   1. `naive_softmax_eager` — the unfused, 5-kernel-launch baseline.
#   2. `torch.softmax` — PyTorch's own built-in fused CUDA kernel (the ATen op
#      already does what we're about to hand-write; it's the "production
#      baseline" our kernel is trying to match, not beat).
#   3. `compiled_softmax` — the same naive eager code, but through
#      `torch.compile` (Inductor fusion).
#   4. `triton_softmax` — our hand-written fused kernel.

# %%
n_rows = 4096
col_widths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

results = {"naive_eager": [], "torch.softmax": [], "torch.compile": [], "triton": []}

for n_cols in col_widths:
    x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)

    ms_naive = bench_ms(naive_softmax_eager, x)
    ms_torch = bench_ms(torch.softmax, x, -1)
    ms_compiled = bench_ms(compiled_softmax, x)
    ms_triton = bench_ms(triton_softmax, x)

    results["naive_eager"].append(ms_naive)
    results["torch.softmax"].append(ms_torch)
    results["torch.compile"].append(ms_compiled)
    results["triton"].append(ms_triton)

    # Effective bandwidth if we assume the ideal 1 read + 1 write of the row
    # data (only truly achieved by the fused kernels; shown for scale).
    bytes_moved = 2 * n_rows * n_cols * x.element_size()
    gbps_triton = bytes_moved / (ms_triton * 1e-3) / 1e9
    print(f"n_cols={n_cols:>6}  naive={ms_naive:7.4f}ms  torch.softmax={ms_torch:7.4f}ms  "
          f"compile={ms_compiled:7.4f}ms  triton={ms_triton:7.4f}ms  "
          f"(triton ~{gbps_triton:6.0f} GB/s effective)")

print("\nmax_memory_allocated:", torch.cuda.max_memory_allocated() / 2**20, "MiB")

# %% [markdown]
# ## Plot: latency vs. row width
#
# A log-x plot makes the whole sweep (128 to 32768 columns) readable on one
# axis. Expect the gap between `naive_eager` and the fused implementations
# (`torch.softmax`, `torch.compile`, `triton`) to be largest at moderate widths
# where the kernel-launch/memory-round-trip overhead dominates, and for all
# methods to converge somewhat at very large widths where a single kernel's
# raw bandwidth-bound time dominates whatever else is going on.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
ax.set_facecolor("white")

colors = {
    "naive_eager": "#eb6834",     # orange
    "torch.softmax": "#2a78d6",   # blue
    "torch.compile": "#1baf7a",   # aqua
    "triton": "#eda100",          # yellow/gold
}
markers = {"naive_eager": "o", "torch.softmax": "s", "torch.compile": "^", "triton": "D"}

for name, ms_values in results.items():
    ax.plot(col_widths, ms_values, marker=markers[name], markersize=5, linewidth=2,
             color=colors[name], label=name)

ax.set_xscale("log", base=2)
ax.set_xlabel("row width (n_cols)")
ax.set_ylabel("latency (ms / call)")
ax.set_title(f"Row-softmax latency vs. width, n_rows={n_rows}, bf16, H100")
ax.grid(True, which="both", linestyle="-", linewidth=0.5, color="#e1e0d9")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Occupancy in practice: sweeping `num_warps` at fixed `BLOCK_SIZE`
#
# To see the occupancy trade-off directly (rather than just asserting it),
# fix a wide row (`n_cols=8192`, so `BLOCK_SIZE=8192`) and re-launch the same
# kernel by hand with different `num_warps` values. Too few warps under-uses
# the parallelism available to reduce over the row; very high warp counts
# don't help once the row's on-chip reduction is already warp-parallel enough,
# and can even regress if register/shared-memory pressure limits how many
# programs co-reside on an SM. This is exactly the search space
# `@triton.autotune` explores automatically in production kernels.

# %%
def triton_softmax_with_warps(x: torch.Tensor, num_warps: int) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x, x.stride(0), y.stride(0), n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return y


x_wide = torch.randn(n_rows, 8192, device=device, dtype=dtype)
print(f"{'num_warps':>10} | {'latency (ms)':>12}")
for nw in [1, 2, 4, 8, 16, 32]:
    ms = bench_ms(triton_softmax_with_warps, x_wide, nw, warmup=10, iters=30)
    print(f"{nw:>10} | {ms:>12.4f}")

# %% [markdown]
# ## What you should see
#
# - The naive 5-kernel-launch eager softmax should be the slowest at every
#   width, and the gap to the fused implementations should be on the order of a
#   **couple of times** at bandwidth-bound sizes (exact ratio varies by GPU,
#   PyTorch version, and width) — the memory-bound argument from the chapter
#   (one read + one write vs. several).
# - `torch.softmax` (PyTorch's built-in fused CUDA kernel) and our hand-written
#   Triton kernel should land in roughly the same ballpark, since both are
#   doing the same one-read/one-write-per-row work; don't expect Triton to
#   dramatically beat a mature, hand-tuned production kernel here — the point
#   is that a straightforward Triton kernel gets *close* with far less code
#   than a hand-written CUDA kernel would need.
# - `torch.compile` on the naive eager code should close a large fraction of
#   the gap to the fused kernels (Inductor fuses much of the elementwise +
#   reduction chain), though results can vary by row width and PyTorch
#   version.
# - The `num_warps` sweep should show a "too low" region (latency drops as
#   `num_warps` increases from 1) and then a plateau or mild uptick at very
#   high warp counts — the occupancy/BLOCK_SIZE trade-off made concrete.
#
# **Key takeaways:**
# 1. Softmax (and RMSNorm, LayerNorm, and similar row-wise ops) are
#    memory-bound: the win from fusing is capped by bandwidth, not by writing
#    cleverer arithmetic.
# 2. `BLOCK_SIZE` and `num_warps` are occupancy knobs, not just tuning
#    trivia — they trade off per-program working set against how many
#    programs run concurrently per SM.
# 3. Always validate a hand-written kernel against the framework's own op
#    (`torch.allclose`) across power-of-2 *and* awkward shapes before trusting
#    any benchmark number from it.
# 4. A hand-fused Triton kernel is competing with an already-fused ATen op
#    here; Triton's real leverage shows up on ops PyTorch has *not* fused for
#    you (e.g. a custom RMSNorm+residual, or FlashAttention's online-softmax
#    recurrence) — see the FlashAttention benchmarking notebook next.
