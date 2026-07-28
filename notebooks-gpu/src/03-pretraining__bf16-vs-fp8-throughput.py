# %% [markdown]
# # bf16 vs FP8 Matmul Throughput on Hopper
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will measure the achieved TFLOP/s of large GEMMs in bf16 versus FP8 (E4M3) on your own H100, using `torch._scaled_mm` directly and (if installed) the production `transformer_engine.pytorch` path, and see why FP8's 3-bit mantissa forces per-tensor dynamic scaling.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/08-mixed-precision-fp8.html) for the full explanation.

# %%
# torch ships with CUDA/cuDNN support and already includes `torch._scaled_mm`
# for FP8 GEMMs on Hopper/Ada (compute capability >= 8.9), so no install is
# needed for the core experiment.
#
# Transformer Engine (TE) is the *recommended production route* for FP8 in
# real training loops (it manages delayed scaling, amax history, and the
# E4M3/E5M2 split for you). It ships prebuilt in NVIDIA's NGC PyTorch
# containers. If you are not in an NGC container, uncomment the line below —
# it needs a matching CUDA toolkit installed and can take several minutes to
# build from source, so this notebook treats it as optional and guards the
# import.
# %pip install -q transformer_engine[pytorch]

# %%
import torch

assert torch.cuda.is_available(), "This notebook requires a CUDA GPU (targets 1x H100 80GB)."

device = torch.device("cuda")
torch.manual_seed(0)

# Report what we're actually running on. FP8 tensor-core GEMMs via
# `torch._scaled_mm` require compute capability >= 8.9 (Ada) with full
# performance on Hopper (9.0); we warn rather than hard-fail so the notebook
# still runs (in bf16-only mode) on older GPUs.
cap_major, cap_minor = torch.cuda.get_device_capability(device)
print(f"Device: {torch.cuda.get_device_name(device)}  (compute capability {cap_major}.{cap_minor})")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

HAS_SCALED_MM = hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")
FP8_HW_OK = (cap_major, cap_minor) >= (8, 9)
if not (HAS_SCALED_MM and FP8_HW_OK):
    print(
        "WARNING: this GPU/torch build cannot run the FP8 sections "
        f"(torch._scaled_mm present: {HAS_SCALED_MM}, capability>=8.9: {FP8_HW_OK}). "
        "The bf16 benchmarks will still run."
    )

# %% [markdown]
# ## Why FP8 needs dynamic scaling
#
# FP8 E4M3 has 4 exponent bits and only **3 mantissa bits**: adjacent
# representable values are ~12.5% apart (one part in eight). A tensor whose
# values span more than one or two orders of magnitude will not survive a
# naive cast to E4M3 — most of the tensor collapses onto a handful of codes.
# The fix (see the chapter's "Per-tensor scaling and the delayed-scaling
# recipe") is to track each tensor's **amax** (max absolute value) and choose
# a scale factor that maps that amax near the top of the FP8 range, so the
# whole distribution lands inside the narrow representable window before the
# cast. We reproduce that per-tensor dynamic-scaling recipe below, then time
# the resulting FP8 GEMM against a bf16 GEMM of identical shape.
#
# We lay matrices out exactly as `nn.Linear` does — activations `X: [M, K]`
# and weights `W: [N, K]` — and compute `X @ W.T`. This is not just
# convention: `torch._scaled_mm` requires its first operand row-major and its
# second operand **column-major**, and `W.T` of a row-major `[N, K]` weight
# is naturally column-major `[K, N]` with no extra transpose/copy needed.

# %%
FP8_E4M3_MAX = 448.0  # largest finite magnitude representable in E4M3 (matches the chapter's table)


def quantize_e4m3(x: torch.Tensor, margin: float = 0.9):
    """Per-tensor dynamic quantization to FP8 E4M3.

    Chooses scale s = (fp8_max / amax(x)) * margin so the tensor's largest
    value lands just under the FP8 ceiling (margin < 1 leaves headroom for
    the next step's amax to grow a little, as in delayed scaling).
    Returns (fp8_tensor, inv_scale) where inv_scale is the DE-scale factor
    `torch._scaled_mm`'s `scale_a`/`scale_b` arguments expect: the GEMM runs
    on the scaled fp8 values, and its output must be multiplied by
    inv_scale_a * inv_scale_b to recover true units.
    """
    amax = x.abs().amax().clamp_min(1e-12).float()
    quant_scale = (FP8_E4M3_MAX / amax) * margin
    x_fp8 = (x.float() * quant_scale).to(torch.float8_e4m3fn)
    inv_scale = (1.0 / quant_scale).reshape(1).to(device=x.device, dtype=torch.float32)
    return x_fp8, inv_scale


def scaled_mm(a_fp8, b_fp8_t, scale_a, scale_b, out_dtype=torch.bfloat16):
    """Thin wrapper around torch._scaled_mm that tolerates API drift across
    torch versions: some versions accept `use_fast_accum`, some return a
    tuple (output, amax) instead of a bare Tensor when out_dtype is fp8."""
    try:
        result = torch._scaled_mm(
            a_fp8, b_fp8_t, scale_a=scale_a, scale_b=scale_b,
            out_dtype=out_dtype, use_fast_accum=True,
        )
    except TypeError:
        result = torch._scaled_mm(
            a_fp8, b_fp8_t, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype,
        )
    return result[0] if isinstance(result, tuple) else result


# Quick sanity check: a random [512, 512] GEMM should match a bf16 reference
# to within the coarse-but-bounded error the 3-bit mantissa implies.
if HAS_SCALED_MM and FP8_HW_OK:
    Xs = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
    Ws = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
    Xs_fp8, sx = quantize_e4m3(Xs)
    Ws_fp8, sw = quantize_e4m3(Ws)
    out_fp8 = scaled_mm(Xs_fp8, Ws_fp8.t(), sx, sw)
    ref = (Xs.float() @ Ws.float().t())
    rel_err = (out_fp8.float() - ref).norm() / ref.norm()
    print(f"sanity check relative error (fp8 vs fp32 reference): {rel_err.item():.4f}")

# %% [markdown]
# ## Benchmark harness: CUDA events, not `time.time()`
#
# GPU kernels launch asynchronously, so wall-clock timing around a kernel
# call mostly measures Python/launch overhead, not the kernel itself. We use
# `torch.cuda.Event(enable_timing=True)` bracketing many iterations (after a
# warmup that lets clocks boost and caches/autotuning settle), with an
# explicit `torch.cuda.synchronize()` before reading the elapsed time. We
# also track peak memory via `torch.cuda.max_memory_allocated()`.

# %%
def benchmark(fn, warmup: int = 10, iters: int = 50):
    """Time `fn()` (a zero-arg callable issuing one GEMM) on the GPU.
    Returns (avg_ms_per_call, peak_memory_gb)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()  # required before reading elapsed_time

    avg_ms = start_evt.elapsed_time(end_evt) / iters
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return avg_ms, peak_gb


def matmul_tflops(m: int, k: int, n: int, avg_ms: float) -> float:
    """X[M,K] @ W.T[K,N] -> [M,N] is 2*M*N*K FLOPs (one multiply + one add
    per MAC). Convert avg ms/iter to achieved TFLOP/s."""
    flops = 2.0 * m * n * k
    seconds = avg_ms / 1000.0
    return flops / seconds / 1e12


# %% [markdown]
# ## Shapes: attention/FFN-sized GEMMs, plus one large square GEMM
#
# We use a handful of `(M, K, N)` shapes representative of the big matmuls in
# a dense transformer's attention projections and MLP block — a batch*seq
# dimension `M` in the thousands, and `K`/`N` in the low thousands to ~11k
# (typical hidden/intermediate sizes) — plus one large square GEMM that is
# closer to a pure roofline test. Memory for these fits comfortably in 80GB:
# the largest single activation/weight tensor here is at most a few hundred
# megabytes.
#
# **Expected result (hedged):** on an H100, on these large compute-bound
# shapes FP8 typically lands somewhere in the **~1.4-1.9x** range of the bf16
# TFLOP/s on the same shape, and can approach the raw ~2x the tensor-core spec
# sheets suggest when the GEMM is big enough to be firmly compute-bound. Treat
# this as an order-of-magnitude expectation, not a guarantee: smaller or
# memory-bound shapes, plus the fp8 cast, memory traffic, and kernel/launch
# overhead, all pull the achieved ratio back below the theoretical doubling.
# Both dtypes should sit well
# under 100% of each format's dense (no-sparsity) peak; on the order of
# ~700-1,000 TFLOP/s dense bf16 and roughly double that dense FP8 are the
# right ballpark for H100 SXM's published tensor-core numbers — see NVIDIA's
# H100 datasheet for the exact figures, since we deliberately do not hardcode
# them here.

# %%
SHAPES = [
    # (M, K, N)  -- M = batch*seq (rows of X), K = in_features, N = out_features
    (16384, 4096, 4096),    # attention QKV/output-projection-sized GEMM
    (16384, 4096, 11008),   # FFN up-projection (d_model -> d_ff)
    (16384, 11008, 4096),   # FFN down-projection (d_ff -> d_model)
    (8192, 8192, 8192),     # large square GEMM, closer to a pure roofline test
]

results = []  # list of dicts, one per shape

for (M, K, N) in SHAPES:
    torch.cuda.empty_cache()
    X = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=device, dtype=torch.bfloat16)  # nn.Linear layout: [out, in]

    # --- bf16 baseline: plain tensor-core GEMM, no scaling needed --------
    bf16_ms, bf16_mem = benchmark(lambda: X @ W.t())
    bf16_tflops = matmul_tflops(M, K, N, bf16_ms)

    row = {"shape": (M, K, N), "bf16_ms": bf16_ms, "bf16_tflops": bf16_tflops, "bf16_mem_gb": bf16_mem}

    # --- FP8 via torch._scaled_mm -----------------------------------------
    # Quantize ONCE outside the timed loop: in steady-state "delayed scaling"
    # (the chapter's recipe), the scale used this step comes from a rolling
    # history of PAST amax values, so the cast is off the critical path and
    # the timed region is the GEMM itself, not the quantization.
    if HAS_SCALED_MM and FP8_HW_OK:
        X_fp8, x_inv_scale = quantize_e4m3(X)
        W_fp8, w_inv_scale = quantize_e4m3(W)
        Wt_fp8 = W_fp8.t()  # column-major view, required by torch._scaled_mm's 2nd operand

        fp8_ms, fp8_mem = benchmark(
            lambda: scaled_mm(X_fp8, Wt_fp8, x_inv_scale, w_inv_scale)
        )
        fp8_tflops = matmul_tflops(M, K, N, fp8_ms)
        row.update({"fp8_ms": fp8_ms, "fp8_tflops": fp8_tflops, "fp8_mem_gb": fp8_mem})

    results.append(row)
    print(f"shape {(M, K, N)}: bf16 {bf16_ms:.3f} ms ({bf16_tflops:.0f} TFLOP/s)", end="")
    if "fp8_tflops" in row:
        print(f"  |  fp8 {row['fp8_ms']:.3f} ms ({row['fp8_tflops']:.0f} TFLOP/s, "
              f"{row['fp8_tflops'] / bf16_tflops:.2f}x bf16)")
    else:
        print("  |  fp8 skipped (unsupported on this GPU/torch build)")

# %% [markdown]
# ## Reading the results table
#
# For each shape, `bf16_tflops` and `fp8_tflops` are the *achieved* TFLOP/s —
# real work done divided by wall time, not a theoretical figure. The ratio
# `fp8_tflops / bf16_tflops` is the speedup you should compare against the
# ~1.4-1.9x hedge above. If it's close to 1.0x (or FP8 is *slower*), the GEMM
# is likely too small to be compute-bound, or the cast/de-scale overhead
# dominates — try larger `M` or check that memory bandwidth isn't the
# bottleneck (see `bf16_mem_gb` / `fp8_mem_gb` in the `results` list).

# %%
print(f"{'shape (M,K,N)':>24} | {'bf16 TFLOP/s':>12} | {'fp8 TFLOP/s':>12} | {'speedup':>8}")
print("-" * 68)
for row in results:
    shape_str = str(row["shape"])
    fp8_str = f"{row['fp8_tflops']:.0f}" if "fp8_tflops" in row else "  n/a"
    speedup_str = f"{row['fp8_tflops'] / row['bf16_tflops']:.2f}x" if "fp8_tflops" in row else "  n/a"
    print(f"{shape_str:>24} | {row['bf16_tflops']:>12.0f} | {fp8_str:>12} | {speedup_str:>8}")

# %% [markdown]
# ## The recommended production route: Transformer Engine
#
# The `torch._scaled_mm` calls above hand-roll exactly one static per-tensor
# scale per GEMM call — fine for a throughput microbenchmark, but a real
# training loop needs the **delayed-scaling** machinery from the chapter: a
# rolling amax history per tensor, automatic E4M3 (forward) / E5M2 (backward
# gradient) format selection, and a context manager so ordinary `nn.Linear`-
# shaped code "just becomes FP8" without hand-written casts at every call
# site. NVIDIA's **Transformer Engine** (`transformer_engine.pytorch`)
# implements this: `te.Linear` behaves like `nn.Linear`, and `fp8_autocast`
# swaps its GEMM to FP8 using a `DelayedScaling` recipe. We guard the import
# since TE is a heavy optional build (see the pip cell above).

# %%
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("transformer_engine not installed — skipping the TE benchmark. "
          "Install with `pip install transformer_engine[pytorch]` or use an "
          "NGC PyTorch container to run this section.")

if TE_AVAILABLE and FP8_HW_OK:
    M, K, N = 16384, 4096, 11008  # reuse the FFN up-projection shape from above

    # HYBRID = E4M3 for forward-pass tensors (weights/activations), E5M2 for
    # the backward-pass gradient tensors — exactly the chapter's split.
    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,       # rolling window TE uses to pick this step's scale
        amax_compute_algo="max",   # use the max amax over that window
    )

    te_layer = te.Linear(K, N, bias=False, params_dtype=torch.bfloat16).to(device)
    x_te = torch.randn(M, K, device=device, dtype=torch.bfloat16)

    def te_forward():
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return te_layer(x_te)

    te_ms, te_mem = benchmark(te_forward)
    te_tflops = matmul_tflops(M, K, N, te_ms)
    print(f"Transformer Engine FP8 te.Linear({K}->{N}), M={M}: "
          f"{te_ms:.3f} ms/iter, {te_tflops:.0f} TFLOP/s, peak mem {te_mem:.2f} GB")
    print("Compare this TFLOP/s to the raw torch._scaled_mm row for the same "
          "shape above: TE adds a little autocast/bookkeeping overhead but "
          "removes the need to hand-manage scales across training steps.")
elif not FP8_HW_OK:
    print("Skipping TE benchmark: this GPU's compute capability does not support FP8 tensor cores.")

# %% [markdown]
# ## Numerical accuracy: how much does the 3-bit mantissa cost?
#
# E4M3's per-element quantization step is coarse (~12.5% relative spacing),
# but the *matmul's reduction* is where the accuracy story gets better: the
# tensor core accumulates the K-dimension sum in fp32 regardless of the input
# precision (see the chapter's "narrow inputs, wide accumulator" idea), so
# independent per-element rounding errors partially cancel across the sum
# instead of compounding. Expect the whole-matrix relative error to be
# noticeably smaller than the raw ~12.5% per-element step, though still much
# larger than bf16's ~0.4% (7-bit mantissa) rounding.

# %%
if HAS_SCALED_MM and FP8_HW_OK:
    M, K, N = 4096, 4096, 4096
    X = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    ref_fp32 = X.float() @ W.float().t()  # fp32 reference (most accurate we compute here)

    X_fp8, sx = quantize_e4m3(X)
    W_fp8, sw = quantize_e4m3(W)
    out_fp8 = scaled_mm(X_fp8, W_fp8.t(), sx, sw)

    out_bf16 = X @ W.t()

    rel_err_fp8 = (out_fp8.float() - ref_fp32).norm() / ref_fp32.norm()
    rel_err_bf16 = (out_bf16.float() - ref_fp32).norm() / ref_fp32.norm()
    print(f"[{M}x{K}x{N}] relative error vs fp32 reference:")
    print(f"  bf16 GEMM: {rel_err_bf16.item():.4f}")
    print(f"  fp8  GEMM: {rel_err_fp8.item():.4f}   (expect this to be larger, but well under 0.125)")

# %% [markdown]
# ## What you should see
#
# - **Speedup:** on an H100, expect FP8 (`torch._scaled_mm` or `te.Linear`)
#   at roughly **1.4-1.9x** the bf16 TFLOP/s on these large GEMM shapes (and
#   sometimes near the theoretical ~2x when the GEMM is firmly compute-bound)
#   — an order-of-magnitude expectation, not a guarantee; smaller GEMMs and
#   memory-bound shapes will show less, or even no speedup.
# - **Absolute throughput:** both dtypes should land well under their
#   respective dense (no-sparsity) tensor-core peaks; treat "~700-1,000
#   TFLOP/s bf16, roughly double that FP8" as the right order of magnitude,
#   and consult NVIDIA's H100 datasheet for the precise published figures.
# - **Accuracy:** the FP8 GEMM's whole-matrix relative error should be
#   clearly larger than bf16's but should NOT be anywhere near the raw ~12.5%
#   per-element E4M3 quantization step — the fp32 accumulation over the `K`
#   dimension is doing real work.
#
# **Key takeaways**
#
# 1. FP8's 3-bit mantissa makes **per-tensor (or finer, blockwise) dynamic
#    scaling mandatory**, not optional — this is the same idea as fp16 loss
#    scaling, applied per-tensor and continuously.
# 2. `torch._scaled_mm` gives you the raw GEMM primitive (useful for
#    understanding and microbenchmarking); **Transformer Engine's
#    `fp8_autocast` + `DelayedScaling`** is the recommended route for real
#    training loops because it manages the amax history and E4M3/E5M2 split
#    for you.
# 3. Always measure GPU kernels with `torch.cuda.Event` + `synchronize()`,
#    never `time.time()` around an async launch — and always warm up first.
# 4. The theoretical 2x FP8-over-bf16 tensor-core throughput rarely shows up
#    whole in end-to-end numbers; casting, memory traffic, and non-GEMM ops
#    (softmax, norms, residuals — kept in bf16/fp32 per the chapter) eat into
#    it.
#
# **Next step:** see [Quantization I: Post-Training Quantization (GPTQ, AWQ,
# SmoothQuant)](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/07-quantization-ptq.html)
# for how blockwise/outlier-aware scaling generalizes to *inference*-time
# quantization below FP8.
