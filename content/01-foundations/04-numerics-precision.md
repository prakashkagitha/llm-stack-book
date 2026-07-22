# 1.4 Numerical Computing, Floating Point & Precision

Numbers in a computer are not mathematical reals — they are finite approximations governed by precise rules laid out in the IEEE 754 standard. For most software this distinction barely matters. For deep learning it is existential: the choice of floating-point format is a first-class design decision that affects training stability, convergence speed, hardware utilization, and memory budget all at once. This chapter gives you a complete mental model of how floating-point works, where it breaks, and why the community settled on **bf16** as the workhorse of LLM training — along with the algorithms that keep numerics stable when the format alone is not enough.

Familiarity with the concepts here is assumed throughout the rest of the book. [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) applies these ideas to the full training loop, and [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) builds numerically stable attention kernels on top of the logsumexp identity we derive here.

---

## IEEE 754: The Foundation

The **IEEE 754** standard (1985, revised 2008) defines how a real number is represented, rounded, and how arithmetic operations behave. Every mainstream ML accelerator — NVIDIA GPUs, Google TPUs, AMD MI-series — implements IEEE 754 (or a deliberate approximation of it, as we will see with TF32 and FP8).

### The bit layout

Every IEEE 754 number is laid out as three fields:

$$
(-1)^{s} \times 1.\text{mantissa} \times 2^{\text{exponent} - \text{bias}}
$$

| Field | Meaning |
|-------|---------|
| **Sign** $s$ | 1 bit — 0 for positive, 1 for negative |
| **Exponent** $e$ | Biased integer — stored as unsigned, interpreted as $e - \text{bias}$ |
| **Mantissa** (significand) $m$ | Fractional bits after an implicit leading 1 |

The leading `1.` is implicit for normal numbers, buying a free bit of precision. Special bit patterns encode $\pm\infty$ (all exponent bits set, mantissa zero) and NaN (all exponent bits set, non-zero mantissa).

The critical intuition: **exponent bits control dynamic range, mantissa bits control precision.** More exponent bits → you can represent very large and very small numbers without overflow/underflow. More mantissa bits → you can represent nearby numbers that differ by a tiny amount.

---

## The Floating-Point Zoo: fp32, tf32, fp16, bf16, fp8

The last decade has produced a proliferation of numeric formats as hardware designers chase the sweet spot between precision (which costs mantissa bits) and performance (which rewards narrower types). Here is the full comparison:

{{fig:float-formats}}

| Format | Sign | Exponent | Mantissa | Total bits | Max value | Min normal | Notes |
|--------|------|----------|----------|------------|-----------|------------|-------|
| fp64 | 1 | 11 | 52 | 64 | ~1.8×10¹⁰⁸ | ~2.2×10⁻³⁰⁸ | "double" |
| **fp32** | 1 | 8 | 23 | 32 | ~3.4×10³⁸ | ~1.2×10⁻³⁸ | Standard training historically |
| **tf32** | 1 | 8 | 10 | 19\* | ~3.4×10³⁸ | ~1.2×10⁻³⁸ | NVIDIA Ampere+; accumulates in fp32 |
| **bf16** | 1 | 8 | 7 | 16 | ~3.4×10³⁸ | ~1.2×10⁻³⁸ | Training workhorse |
| **fp16** | 1 | 5 | 10 | 16 | ~65504 | ~6.1×10⁻⁵ | Inference; narrow range |
| fp8 E4M3 | 1 | 4 | 3 | 8 | 448 | ~1.95×10⁻³ | FP8 fine-scale |
| fp8 E5M2 | 1 | 5 | 2 | 8 | ~57344 | ~1.5×10⁻⁶ | FP8 gradients |

\* TF32 packs into a 32-bit register; only 19 bits carry information.

```text
FP32  [s|eeeeeeee|mmmmmmmmmmmmmmmmmmmmmmm]   32 bits
TF32  [s|eeeeeeee|mmmmmmmmmm         ---]   19 effective bits
BF16  [s|eeeeeeee|mmmmmmm]                  16 bits
FP16  [s|eeeee|mmmmmmmmmm]                  16 bits
FP8   [s|eeee|mmm]  (E4M3)                   8 bits
      [s|eeeee|mm] (E5M2)                    8 bits
```

The key visual: **bf16 is the top 16 bits of fp32**. They share the same 8-bit exponent, so the dynamic range is identical. You can truncate fp32 to bf16 by simply dropping the lower 16 bits of the mantissa — no reformatting, just a shift. This hardware-friendliness is why bf16 won.

### Machine epsilon and ULP

**Machine epsilon** ($\varepsilon_{\text{mach}}$) is the smallest value such that $1 + \varepsilon_{\text{mach}} \neq 1$ in the given format. It equals $2^{-p}$ where $p$ is the number of mantissa bits:

| Format | $\varepsilon_{\text{mach}}$ |
|--------|-----------------------------|
| fp64 | $\approx 2.2 \times 10^{-16}$ |
| fp32 | $\approx 1.2 \times 10^{-7}$ |
| bf16 | $\approx 7.8 \times 10^{-3}$ |
| fp16 | $\approx 9.8 \times 10^{-4}$ |

**Unit in the last place** (ULP) is the spacing between adjacent representable numbers at any given magnitude. For a normal fp32 number with magnitude around 1.0, 1 ULP ≈ $1.2 \times 10^{-7}$. For a number around $10^4$, 1 ULP ≈ $1.2 \times 10^{-3}$ — precision scales with magnitude.

{{fig:float-number-line}}

---

## Dynamic Range vs. Precision: The Core Tradeoff

The two properties that determine a format's fitness are independent:

- **Dynamic range**: the ratio of the largest to the smallest representable normal number, set by the exponent field.
- **Relative precision**: how finely you can resolve nearby values, set by the mantissa field.

fp16 has a 5-bit exponent, giving a maximum representable value of roughly **65504**. That sounds large for activations, but gradients and weight updates during training can transiently exceed this — and once you hit overflow you get `inf`, which poisons the entire forward/backward pass.

bf16 keeps fp32's 8-bit exponent (max value ~$3.4 \times 10^{38}$) but sacrifices mantissa bits. Its relative precision ($\varepsilon_{\text{mach}} \approx 7.8 \times 10^{-3}$) is roughly 10× worse than fp16, but that turns out to be acceptable for gradient noise in SGD. The dynamic range is what matters for training stability, and bf16 has it.

### Overflow, underflow, and subnormals

**Overflow** occurs when a computed value exceeds the format's maximum. In IEEE 754, the result is `+inf` or `-inf`. Arithmetic involving `inf` often produces NaN, which then propagates and kills training.

**Underflow** occurs when a value is too small to represent as a normal number. IEEE 754 defines **subnormal** (denormal) numbers that fill the gap between zero and the minimum normal value by using a leading `0.` instead of `1.`. Subnormals sacrifice range for a graceful flush to zero, but on hardware they are typically much slower to process — or flushed to zero (FTZ mode) entirely. On most GPU training setups, FTZ is enabled, so very small activations silently become zero.

!!! warning "fp16 overflow in practice"
    During early LLM training, gradient norms can spike to values well above 65504. With fp16 these spikes produce `inf` gradients, which then corrupt the parameter update. The standard remedy is **loss scaling** (multiply the loss by a large scalar before backward, divide gradients after) — but the approach is fragile. bf16 makes the problem largely disappear because the dynamic range matches fp32.

{{fig:format-range-ruler}}

---

## Catastrophic Cancellation

Catastrophic cancellation is the most insidious floating-point pitfall. It occurs when you subtract two nearly-equal numbers, and the relative error in the result explodes.

Consider computing $a - b$ where $a = 1.0000001$ and $b = 1.0000000$ in fp32. The true result is $10^{-7}$, right at the edge of fp32's precision. If both $a$ and $b$ carry even tiny rounding errors (say, each off by $\pm 10^{-7}$), the error in the difference is $\pm 2 \times 10^{-7}$ — the same order of magnitude as the result itself. You have lost all significant digits.

!!! example "Worked Example: Cancellation in softmax"
    Suppose we compute softmax naively for logits $z = [1000.0,\; 1001.0]$.

    Step 1 — compute $e^{z_i}$:

    $$e^{1000} \approx 5.07 \times 10^{434}, \quad e^{1001} \approx 1.38 \times 10^{435}$$

    Both values overflow fp32 (max ~$3.4 \times 10^{38}$) and fp16 long before this. We get `inf / inf = NaN`.

    Step 2 — in bf16 the exponent range is the same as fp32 (max ~$3.4 \times 10^{38}$), so fp32 overflow at $e^{88}$ still applies. Even in fp32, $e^{1000}$ overflows.

    The stable fix shifts by the maximum logit: subtract $m = \max_i z_i$ before exponentiation. Since $z = [1000, 1001]$, we subtract $m = 1001$:

    $$e^{1000 - 1001} = e^{-1} \approx 0.368, \quad e^{1001 - 1001} = e^0 = 1.0$$

    Sum $= 1.368$, softmax $= [0.268, 0.732]$. Perfectly stable, and mathematically identical because the $e^m$ terms cancel.

---

## Stable Softmax and the LogSumExp Identity

The softmax function is defined as:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The naive implementation overflows for large logits and underflows for very negative logits. The standard stabilization uses the identity:

$$
\text{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \quad m = \max_j z_j
$$

This is valid because $e^{z_i - m} / \sum_j e^{z_j - m} = (e^{-m} e^{z_i}) / (e^{-m} \sum_j e^{z_j})$ — the $e^{-m}$ cancels.

The related operation **logsumexp** appears constantly in log-probabilities, cross-entropy, and attention:

$$
\operatorname{logsumexp}(z) = \log \sum_j e^{z_j}
$$

The numerically stable version is:

$$
\operatorname{logsumexp}(z) = m + \log \sum_j e^{z_j - m}, \quad m = \max_j z_j
$$

Because $z_j - m \leq 0$ for all $j$, the exponentials are in $[0, 1]$ — no overflow possible, and underflow is graceful (a very negative $z_j - m$ contributes essentially zero to the sum, which is correct behavior).

{{fig:stable-softmax-two-lanes}}

```python
import torch
import numpy as np

# ------------------------------------------------------------------
# Numerically stable softmax — implemented from scratch in PyTorch.
# Matches torch.nn.functional.softmax to machine precision.
# ------------------------------------------------------------------

def naive_softmax(z: torch.Tensor) -> torch.Tensor:
    """Naive implementation — overflows for large inputs."""
    e = torch.exp(z)              # Can produce inf for large z
    return e / e.sum(dim=-1, keepdim=True)


def stable_softmax(z: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable softmax via the max-subtraction trick.

    The key identity:
        softmax(z)_i = softmax(z - max(z))_i
    because the max subtraction cancels in numerator and denominator.
    """
    # Subtract maximum along last dimension; keepdim for broadcasting
    m = z.max(dim=-1, keepdim=True).values   # shape (..., 1)
    shifted = z - m                           # all values <= 0
    e = torch.exp(shifted)                    # all values in (0, 1]
    return e / e.sum(dim=-1, keepdim=True)


def stable_log_softmax(z: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log-softmax.

    log softmax(z)_i = z_i - logsumexp(z)
    logsumexp(z) = m + log(sum(exp(z - m)))

    Computing log after softmax loses precision; compute log-softmax directly.
    """
    m = z.max(dim=-1, keepdim=True).values
    shifted = z - m
    log_sum_exp = m.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))
    # Broadcast log_sum_exp back for subtraction
    return z - log_sum_exp.unsqueeze(-1)


def stable_logsumexp(z: torch.Tensor) -> torch.Tensor:
    """Standalone logsumexp with the shift trick."""
    m = z.max(dim=-1).values              # shape (...)
    shifted = z - m.unsqueeze(-1)         # values in (-inf, 0]
    return m + torch.log(torch.exp(shifted).sum(dim=-1))


# ---------------------------------------------------------------
# Demonstrate fp16 overflow and how stable_softmax avoids it
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    # Large logits — pathological case
    z_large = torch.tensor([1000.0, 1001.0, 999.0])

    print("=== fp32 naive softmax (large logits) ===")
    z_fp32 = z_large.to(torch.float32)
    result_naive = naive_softmax(z_fp32)
    print(f"  Result: {result_naive}")          # [nan, nan, nan] — overflow!

    print("\n=== fp32 stable softmax (large logits) ===")
    result_stable = stable_softmax(z_fp32)
    print(f"  Result: {result_stable}")         # [0.2447, 0.6652, 0.0900] — correct

    # Show fp16 overflow with much smaller values
    print("\n=== fp16 naive softmax (overflow at ~88) ===")
    z_fp16 = torch.tensor([100.0, 101.0, 99.0], dtype=torch.float16)
    result_fp16_naive = naive_softmax(z_fp16)
    print(f"  Result: {result_fp16_naive}")     # [nan, nan, nan] — fp16 overflows at exp(88)

    print("\n=== fp16 stable softmax ===")
    result_fp16_stable = stable_softmax(z_fp16)
    print(f"  Result: {result_fp16_stable}")    # Correct softmax

    # Verify logsumexp
    print("\n=== logsumexp verification ===")
    z_test = torch.tensor([1.0, 2.0, 3.0])
    lse = stable_logsumexp(z_test)
    expected = torch.log(torch.exp(z_test).sum())
    print(f"  stable_logsumexp: {lse:.6f}")
    print(f"  torch reference:  {expected:.6f}")
    print(f"  match: {torch.allclose(lse, expected)}")
```

```text
=== fp32 naive softmax (large logits) ===
  Result: tensor([nan, nan, nan])

=== fp32 stable softmax (large logits) ===
  Result: tensor([0.244728, 0.665241, 0.090031])

=== fp16 naive softmax (overflow at ~88) ===
  Result: tensor([nan, nan, nan], dtype=torch.float16)

=== fp16 stable softmax ===
  Result: tensor([0.244751, 0.665527, 0.090088], dtype=torch.float16)

=== logsumexp verification ===
  stable_logsumexp: 3.407606
  torch reference:  3.407606
  match: True
```

FlashAttention extends exactly this idea to the attention operation, fusing the online softmax into a single GPU pass without materializing the full $N \times N$ attention matrix — see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) for the full derivation.

---

## Kahan Summation and Compensated Arithmetic

Summing $N$ floating-point numbers naively accumulates $O(N)$ rounding errors. For $N = 10^9$ (a common size in LLM weight norms), this can become significant.

**Kahan summation** (Kahan, 1965) maintains a running "compensation" variable that tracks the error lost to rounding at each step, restoring it to the next addition:

$$
\text{sum} \leftarrow \text{sum} + y,\quad \text{but tracking } c = (y - \text{sum}_{\text{prev}}) - \text{sum}
$$

Formally:

```python
def kahan_sum(values):
    """
    Kahan compensated summation — reduces floating-point error from O(N*eps)
    to O(eps) regardless of N, at the cost of ~2x FLOPs per element.

    Reference: Kahan, W. (1965). "Further remarks on reducing truncation errors."
    Communications of the ACM, 8(1), 40.
    """
    total = 0.0
    compensation = 0.0   # tracks the "lost" low-order bits

    for v in values:
        # The compensation corrects for the rounding error from the previous step.
        # y is the value adjusted for previous lost bits.
        y = v - compensation
        # t is the running total after adding y; but it loses the low bits of y
        t = total + y
        # (t - total) recovers what was actually added — which differs from y
        # due to rounding. The difference is the new compensation.
        compensation = (t - total) - y
        total = t

    return total


# ---------------------------------------------------------------
# Demonstrate the error reduction on a large sum
# ---------------------------------------------------------------
import numpy as np

N = 1_000_000
# Create values that sum to exactly N (each value is 1.0)
# Then add a tiny perturbation to expose rounding errors
values_f32 = np.ones(N, dtype=np.float32)
values_f32[0] = 1.0 + 1e-7   # tiny perturbation

true_sum = float(N) + 1e-7    # exact answer

# Naive float32 accumulation
naive_result = float(np.sum(values_f32))

# Kahan in Python (pedagogical — slow)
kahan_result = kahan_sum(values_f32.tolist())

# float64 reference
ref_result = float(np.sum(values_f32.astype(np.float64)))

print(f"True sum:          {true_sum:.10f}")
print(f"Naive float32:     {naive_result:.10f}  error={abs(naive_result - true_sum):.2e}")
print(f"Kahan float32:     {kahan_result:.10f}  error={abs(kahan_result - true_sum):.2e}")
print(f"Float64 reference: {ref_result:.10f}  error={abs(ref_result - true_sum):.2e}")
```

Kahan summation matters most in **gradient accumulation** (summing many micro-batch gradients before an optimizer step) and in **weight update** computations. PyTorch's optimizer implementations typically accumulate in fp32 even when parameters are stored in fp16/bf16; this is the "master weights" pattern described in the mixed-precision training chapter.

---

## Why bf16 Won Training

The story of how bf16 displaced fp32 (and then fp16) as the default training format is worth understanding in detail, because it illustrates every tradeoff discussed above.

### The fp16 era and its problems

Around 2017–2018, the community began training in fp16 with the **mixed-precision training** recipe (Micikevicius et al., ICLR 2018). The idea: store weights and activations in fp16, accumulate gradient updates in fp32 ("master weights"), and use loss scaling to prevent gradient underflow. Hardware reasons: fp16 FLOP/s is 2× higher than fp32 on Pascal/Volta GPUs, and memory bandwidth doubles.

The problems:
1. **Overflow**: weight norms, gradient norms, and optimizer states can transiently exceed 65504. Loss scaling helps but requires dynamic tuning.
2. **Precision**: with only 10 mantissa bits ($\varepsilon_{\text{mach}} \approx 10^{-3}$), small weight updates are rounded to zero when the weight magnitude exceeds roughly $10^3 \times \varepsilon_{\text{mach}} = 1$. At large learning rates late in training, this is fine; at small learning rates fine-tuning, it is not.
3. **Engineering complexity**: loss scaling, gradient unscaling, and `inf`/`NaN` checks add fragile infrastructure.

### The bf16 solution

Google introduced bf16 for TPU training and NVIDIA added native bf16 support in Ampere (A100, 2020). The format keeps fp32's 8-bit exponent, sacrificing mantissa bits:

- **No overflow**: the dynamic range matches fp32. A value that fits in fp32 fits in bf16. Loss scaling becomes unnecessary.
- **Simpler hardware path**: truncate the lower 16 bits of a fp32 value to get bf16. No reformatting, no bias adjustment.
- **Acceptable precision loss**: the 7-bit mantissa ($\varepsilon_{\text{mach}} \approx 7.8 \times 10^{-3}$) is coarser than fp16's 10-bit mantissa, but empirically the optimization noise from SGD/Adam dominates this rounding error for large models. The signal-to-noise ratio of gradients at LLM scale is low enough that you cannot tell the difference.

The practical result: training LLMs in pure bf16 (no loss scaling, no master weights in many implementations) "just works." This simplification dramatically reduced training infrastructure complexity.

### TF32: the tensor-core compromise

TF32 (TensorFloat-32) is an NVIDIA-specific format introduced with Ampere. It is not a storage format — it is a compute format used internally by tensor cores. When you do a matrix multiply in fp32, Ampere automatically:
1. Rounds each fp32 input to tf32 (8 exponent bits, 10 mantissa bits — same range as fp32, precision of fp16).
2. Performs the multiply-accumulate in the tensor core.
3. Accumulates the result in fp32.

The effect: 8× higher FLOP/s for matmul vs. fp32, with minimal precision loss because the accumulation is fp32. You get this for free without changing your training code — `torch.backends.cuda.matmul.allow_tf32 = True` (default on Ampere+).

```python
import torch

# Check TF32 settings
print(f"TF32 for matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 for cudnn:  {torch.backends.cudnn.allow_tf32}")

# Demonstrate the precision difference between fp32, tf32, and bf16 matmul
# (requires CUDA GPU)
if torch.cuda.is_available():
    torch.manual_seed(42)
    A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)

    # fp64 reference
    ref = (A.double() @ B.double()).float()

    # fp32 with TF32 disabled (pure fp32 matmul)
    torch.backends.cuda.matmul.allow_tf32 = False
    out_fp32 = A @ B
    err_fp32 = (out_fp32 - ref).abs().max().item()

    # fp32 with TF32 enabled (Ampere behavior)
    torch.backends.cuda.matmul.allow_tf32 = True
    out_tf32 = A @ B
    err_tf32 = (out_tf32 - ref).abs().max().item()

    # bf16 matmul (explicit cast)
    out_bf16 = (A.bfloat16() @ B.bfloat16()).float()
    err_bf16 = (out_bf16 - ref).abs().max().item()

    print(f"\nMax absolute error vs fp64 reference (1024x1024 matmul):")
    print(f"  fp32 (no TF32): {err_fp32:.2e}")
    print(f"  TF32:           {err_tf32:.2e}")
    print(f"  bf16:           {err_bf16:.2e}")
```

### FP8: the frontier

FP8 (introduced in the H100 Transformer Engine) pushes further. Two variants:
- **E4M3** (4 exponent, 3 mantissa): higher precision, for forward activations and weights.
- **E5M2** (5 exponent, 2 mantissa): larger range, for gradients.

Because FP8 has only 3 or 2 mantissa bits, it requires **dynamic quantization** — per-tensor or per-block scaling factors to keep values in the representable range. NVIDIA's Transformer Engine handles this automatically. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) for the full workflow.

---

## Demonstrating fp16 Overflow and Corrective Techniques

Let us make the overflow story concrete with runnable code:

```python
import torch
import struct

def fp32_to_bits(x: float) -> str:
    """Return the IEEE 754 binary representation of a float32."""
    [bits] = struct.unpack('>I', struct.pack('>f', x))
    s = f"{bits:032b}"
    return f"[{s[0]}|{s[1:9]}|{s[9:]}]  (sign|exp|mantissa)"


def show_format_limits():
    """
    Print the representable range of fp32, fp16, and bf16.
    Useful for understanding where overflow occurs.
    """
    formats = [
        ("fp32",   torch.float32),
        ("fp16",   torch.float16),
        ("bf16",   torch.bfloat16),
    ]
    print(f"{'Format':<8} {'Max value':<16} {'Min normal':<16} {'Machine eps':<14}")
    print("-" * 56)
    for name, dtype in formats:
        info = torch.finfo(dtype)
        print(f"{name:<8} {info.max:<16.4e} {info.tiny:<16.4e} {info.eps:<14.4e}")


def demonstrate_fp16_overflow():
    """
    Show fp16 overflow and the bf16/loss-scaling remedies.
    fp16 max is ~65504; anything larger becomes inf.
    """
    # Values near and beyond fp16 max
    test_vals = [1e3, 1e4, 6.5e4, 6.55e4, 1e5, 1e38]

    print(f"\n{'Value':<12} {'fp16 result':<14} {'bf16 result':<14} {'fp32 result'}")
    print("-" * 56)
    for v in test_vals:
        fp16 = torch.tensor(v, dtype=torch.float16).item()
        bf16 = torch.tensor(v, dtype=torch.bfloat16).item()
        fp32 = torch.tensor(v, dtype=torch.float32).item()
        fp16_str = "inf" if fp16 == float('inf') else f"{fp16:.4e}"
        bf16_str = "inf" if bf16 == float('inf') else f"{bf16:.4e}"
        print(f"{v:<12.2e} {fp16_str:<14} {bf16_str:<14} {fp32:.4e}")


def demonstrate_loss_scaling():
    """
    Simulate fp16 loss scaling: scale up before backward,
    scale down gradient after, then check for inf/nan.
    This is the mechanism behind torch.cuda.amp.GradScaler.
    """
    # Simulate a tiny gradient that would underflow in fp16
    true_gradient = torch.tensor(1e-8)   # below fp16 min normal ~6e-5

    scale_factor = 2.0 ** 15            # = 32768, typical initial scale

    # Scale up: small gradient becomes representable
    scaled_grad_fp16 = (true_gradient * scale_factor).to(torch.float16)
    print(f"\nLoss scaling demo:")
    print(f"  True gradient:      {true_gradient.item():.2e}  (underflows fp16)")
    print(f"  Scaled (x{scale_factor:.0f}):    {(true_gradient * scale_factor).item():.2e}")
    print(f"  In fp16:            {scaled_grad_fp16.item():.2e}  (representable!)")

    # Unscale: divide out the scale factor in fp32
    unscaled = scaled_grad_fp16.float() / scale_factor
    print(f"  Unscaled fp32:      {unscaled.item():.2e}  (close to true)")

    # Check for inf (skip update if gradient overflowed)
    has_inf = not torch.isfinite(scaled_grad_fp16).all()
    print(f"  Has inf/nan:        {has_inf}  → {'skip update' if has_inf else 'apply update'}")


if __name__ == "__main__":
    show_format_limits()
    demonstrate_fp16_overflow()
    demonstrate_loss_scaling()
    print("\nIEEE 754 bit layout of 1.0 in fp32:")
    print(f"  1.0 = {fp32_to_bits(1.0)}")
    print("IEEE 754 bit layout of inf in fp32:")
    print(f"  inf = {fp32_to_bits(float('inf'))}")
```

```text
Format   Max value        Min normal       Machine eps
--------------------------------------------------------
fp32     3.4028e+38       1.1755e-38       1.1921e-07
fp16     6.5504e+04       6.1035e-05       9.7656e-04
bf16     3.3895e+38       1.1755e-38       7.8125e-03

Value        fp16 result    bf16 result    fp32 result
--------------------------------------------------------
1.00e+03     1.0000e+03     1.0000e+03     1.0000e+03
1.00e+04     1.0000e+04     9.9840e+03     1.0000e+04
6.50e+04     6.4992e+04     6.5024e+04     6.5000e+04
6.55e+04     6.5504e+04     6.5536e+04     6.5500e+04
1.00e+05     inf            9.9840e+04     1.0000e+05
1.00e+38     inf            9.9692e+37     1.0000e+38

Loss scaling demo:
  True gradient:      1.00e-08  (underflows fp16)
  Scaled (x32768):    3.28e-04
  In fp16:            3.28e-04  (representable!)
  Unscaled fp32:      1.00e-08  (close to true)
  Has inf/nan:        False  → apply update
```

---

## Numerical Stability in Practice: Patterns and Anti-Patterns

The section above covered two named algorithms (stable softmax, Kahan summation). Here we survey the broader landscape of numerical stability patterns that appear throughout the LLM stack.

### Log-probability arithmetic

Language models assign probabilities to sequences. These probabilities are products of many per-token probabilities, each less than 1. For a 1000-token sequence with average per-token probability $10^{-2}$, the sequence probability is $10^{-2000}$ — far below any representable float. Always work in **log space**:

$$
\log P(x_1, \ldots, x_T) = \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

Cross-entropy loss is the negative log-probability, computed by `torch.nn.CrossEntropyLoss`, which internally uses `log_softmax` for numerical stability. Never compute `log(softmax(logits))` — always call `log_softmax` directly.

### Layer normalization and numerical stability

Layer normalization (Ba et al., 2016) computes:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}
$$

The $\varepsilon$ (typically $10^{-5}$ or $10^{-6}$) prevents division by zero when the variance is tiny. In bf16, the computed variance can be zero even for non-constant inputs because the precision is $7.8 \times 10^{-3}$. PyTorch's `LayerNorm` implementation always computes the mean and variance in fp32 internally, regardless of the input dtype — a critical numerical safeguard. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html) for the full normalization story.

### Attention score overflow

Raw attention scores are $q_i \cdot k_j / \sqrt{d_k}$. For large $d_k$ (e.g., 128 in LLaMA-style heads), the dot product magnitudes grow as $O(\sqrt{d_k})$ before scaling. After scaling they are $O(1)$, which is stable. But in implementations that forget the $1/\sqrt{d_k}$ denominator, or in attention variants with non-standard scaling, scores can be large enough to overflow fp16 or saturate softmax.

### Gradient norms and clipping

The gradient of a large model is a concatenated vector of millions to billions of parameters. Its $\ell_2$ norm,

$$
g = \left\| \nabla_\theta \mathcal{L} \right\|_2
$$

is computed as a sum of squares across all layers, then square-rooted. Both operations are numerically sensitive: accumulating squares of large values can overflow, and taking the square root of a very large or very small value loses precision. Gradient clipping, `torch.nn.utils.clip_grad_norm_`, computes this norm in fp32 regardless of parameter dtype.

```python
import torch

def stable_gradient_norm_clip(parameters, max_norm: float, norm_type: float = 2.0):
    """
    Numerically stable gradient norm clipping.
    Always computes in fp32, avoids overflow during squared sum accumulation.

    This is what torch.nn.utils.clip_grad_norm_ does internally.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return torch.tensor(0.0)

    # Cast all gradient norms to fp32 before summing to avoid overflow
    # Each element's squared norm computed in fp32 even if grad is bf16
    total_norm_sq = sum(
        p.grad.detach().float().norm(norm_type) ** norm_type
        for p in params_with_grad
    )

    total_norm = total_norm_sq ** (1.0 / norm_type)   # fp32 result

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)

    return total_norm


# Illustrate: simulate a gradient spike that would overflow fp16
model = torch.nn.Linear(512, 512)
# Inject a large gradient to simulate a loss spike
with torch.no_grad():
    model.weight.grad = torch.randn_like(model.weight) * 1000.0
    model.bias.grad   = torch.randn_like(model.bias)   * 1000.0

norm_before = sum(p.grad.float().norm(2).item() ** 2
                  for p in model.parameters() if p.grad is not None) ** 0.5
print(f"Gradient norm before clipping: {norm_before:.2f}")

norm = stable_gradient_norm_clip(model.parameters(), max_norm=1.0)
print(f"Gradient norm reported:        {norm.item():.2f}")

norm_after = sum(p.grad.float().norm(2).item() ** 2
                 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"Gradient norm after clipping:  {norm_after:.4f}  (target: ≤ 1.0)")
```

---

## Interview Corner

!!! interview "Interview Corner"
    **Q:** You are training a large language model in bf16. You notice that your cross-entropy loss diverges to NaN after 5000 steps. Walk through the numerical causes you would investigate and how you would fix each one.

    **A:** Start by instrumenting the training loop to log gradient norms, loss values, and activation statistics every N steps. The key suspects, in order of likelihood:

    1. **Overflow in activations or logits**: check whether any tensor value becomes `inf` before the NaN. Large logit values going into softmax produce `inf / inf = NaN`. Fix: use numerically stable `log_softmax` and ensure attention scores are properly scaled by $1/\sqrt{d_k}$.

    2. **Gradient explosion**: the gradient norm spikes. Fix: gradient clipping (clip to max norm 1.0 is standard) and verify the learning rate hasn't been set too high.

    3. **Loss scaling residue**: if the code was ported from fp16 and retained loss scaling, the scale factor might be too large, producing overflowed scaled gradients that survive the inf-check (e.g., due to a bug). Fix: remove loss scaling (bf16 doesn't need it) or verify GradScaler logic.

    4. **Numerical instability in LayerNorm**: ensure `LayerNorm` computes statistics in fp32 internally. In PyTorch this is the default; check that no custom norm layer accumulates in bf16.

    5. **Weight update precision**: if optimizer states are in bf16 (not the standard setup — Adam momentum/variance should be fp32 master copies), small weight updates round to zero and the model stops learning. Fix: keep optimizer states in fp32.

    6. **Embedding table values**: embedding lookup can produce large values early in training if weight initialization is too large. Check embedding weight norms and apply weight decay or initialization clipping.

    In practice, most NaN events in bf16 training stem from (1) or (2), and are resolved by gradient clipping + checking the softmax/attention implementation.

---

## Key Takeaways

!!! key "Key Takeaways"
    - IEEE 754 floating-point encodes numbers as $(-1)^s \times 1.m \times 2^{e-\text{bias}}$; exponent bits determine dynamic range, mantissa bits determine relative precision.
    - bf16 keeps fp32's 8-bit exponent (max ~$3.4 \times 10^{38}$) with only 7 mantissa bits. Its identical dynamic range makes loss scaling unnecessary, which is why it displaced fp16 as the LLM training standard.
    - fp16 overflows at ~65504; this is a hard wall that causes NaN in training unless carefully managed with loss scaling. bf16 essentially eliminates this failure mode.
    - Catastrophic cancellation destroys precision when subtracting nearly equal numbers. Stable softmax avoids it by subtracting the maximum logit before exponentiation — a mathematically equivalent but numerically safe rewrite.
    - The logsumexp identity $m + \log \sum_j e^{z_j - m}$ is the foundation of stable softmax, stable log-softmax, numerically correct cross-entropy, and the online attention algorithm in FlashAttention.
    - Kahan summation reduces floating-point summation error from $O(N \varepsilon)$ to $O(\varepsilon)$; the pattern of accumulating in higher precision is used throughout PyTorch (LayerNorm statistics, gradient norm computation, optimizer states).
    - TF32 is a hardware-only intermediate format in NVIDIA Ampere tensor cores: fp32 inputs are rounded to 10-bit mantissa internally, with fp32 accumulation, delivering ~8× higher throughput than fp32 matmul with negligible accuracy loss.
    - FP8 (E4M3 for weights/activations, E5M2 for gradients) requires per-tensor scaling factors because its 3/2 mantissa bits offer almost no dynamic range on their own.
    - When debugging NaN/inf in training: check in order — activation magnitudes, gradient norms, loss scaling logic, LayerNorm implementation, and optimizer state dtypes.

---

!!! sota "State of the Art & Resources (2026)"
    Floating-point numerics for deep learning has matured around **bf16** as the default training format and **FP8** (E4M3/E5M2) as the emerging frontier for H100/Blackwell hardware, with the foundational algorithms (stable softmax, Kahan summation, log-space arithmetic) now standard in every major framework. Current research focuses on sub-8-bit training and mixed-precision scheduling.

    **Textbooks & foundational papers**

    - [Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — the canonical reference on IEEE 754 representation, rounding, and error analysis.
    - [Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM, 2002)](https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/) — graduate-level treatment of backward error analysis, condition numbers, and algorithm stability.
    - [Micikevicius et al., *Mixed Precision Training* (ICLR 2018)](https://arxiv.org/abs/1710.03740) — introduced loss scaling and master-weight fp32 copies to make fp16 training viable at scale.
    - [Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* (2019)](https://arxiv.org/abs/1905.12322) — the empirical case that bf16's fp32-equivalent dynamic range makes loss scaling unnecessary.

    **Recent advances (2022–2026)**

    - [Micikevicius et al., *FP8 Formats for Deep Learning* (2022)](https://arxiv.org/abs/2209.05433) — defines the E4M3 / E5M2 split and demonstrates FP8 training quality on 175B-parameter models.
    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — shows how the logsumexp identity enables numerically stable attention in a single fused GPU kernel.
    - [Hao et al., *Low-Precision Training of Large Language Models: Methods, Challenges, and Opportunities* (2025)](https://arxiv.org/abs/2505.01043) — comprehensive 2025 survey covering integer, floating-point, and custom-format low-precision training methods.

    **Open-source & tools**

    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference implementation of FlashAttention 1–4, supporting CUDA and ROCm, used in virtually every major LLM training stack.
    - [NVIDIA Transformer Engine — FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) — official guide to FP8 / MXFP8 / NVFP4 training with delayed and block-wise scaling on H100/Blackwell.

    **Go deeper**

    - [PyTorch Automatic Mixed Precision tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — hands-on guide to `torch.autocast` and `GradScaler` for bf16/fp16 training in PyTorch.
    - [NVIDIA Technical Blog: *Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training* (2025)](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) — accessible overview of FP8 benefits, hardware support, and real-world training results.
    - [Julia Evans, *Examples of floating point problems* (2023)](https://jvns.ca/blog/2023/01/13/examples-of-floating-point-problems/) — concise, practical examples of cancellation, precision loss, and rounding surprises in real code.

## Further Reading

- **Goldberg, David. "What every computer scientist should know about floating-point arithmetic." ACM Computing Surveys, 1991.** The canonical reference — dense but complete.
- **Kahan, W. "Further remarks on reducing truncation errors." Communications of the ACM, 1965.** The original compensated summation paper.
- **Micikevicius et al. "Mixed Precision Training." ICLR, 2018.** The foundational paper that introduced the loss-scaling recipe for fp16 training.
- **Kalamkar et al. "A Study of BFLOAT16 for Deep Learning Training." arXiv:1905.12322, 2019.** Intel and Google's analysis of bf16 on large-scale training — the paper that anchored bf16 adoption.
- **NVIDIA Transformer Engine documentation.** Covers the FP8 training workflow with per-tensor scaling for H100 and later GPUs.
- **Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022.** Applies stable logsumexp to fused attention in a single GPU kernel pass.
- **Higham, Nicholas J. "Accuracy and Stability of Numerical Algorithms." SIAM, 2002.** The graduate-level reference for backward error analysis, condition numbers, and algorithm stability.
