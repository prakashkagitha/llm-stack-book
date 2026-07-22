"""
Runnable-code test for content/01-foundations/04-numerics-precision.md

Blocks tested (heuristically CPU-runnable):
  - block #1 (~line 160): stable softmax / log-softmax / logsumexp, from scratch in PyTorch
  - block #3 (~line 286): Kahan compensated summation
  - block #5 (~line 427): fp32 bit layout, format limits, fp16 overflow, loss scaling
  - block #7 (~line 575): fp32 gradient-norm clipping

Blocks intentionally SKIPPED (see notes at bottom of file):
  - block #0, #2, #6: non-python (markdown / text fences, not executable code)
  - block #4 (~line 376): TF32 vs fp32 vs bf16 matmul demo — the precision-comparison
    logic is entirely inside `if torch.cuda.is_available():`, i.e. it is a no-op on CPU
    and genuinely needs a CUDA GPU with tensor cores to be meaningful.

All code below is copied verbatim from the chapter's fenced code blocks (only glue
/asserts added at the end of each block to prove the block actually executed and
produced the values the chapter claims).
"""

import torch
import numpy as np

# =====================================================================
# Block #1 (chapter line ~160): Numerically stable softmax / logsumexp
# =====================================================================

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
    print(f"  Result: {result_stable}")         # [0.2689, 0.7311, 0.0990] — correct

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

    # --- honesty checks: verify the chapter's claims actually hold ---
    assert torch.isnan(result_naive).all(), "naive fp32 softmax on [1000,1001,999] should be all-NaN"
    # NOTE: the chapter's original printed reference [0.268941, 0.731059, 0.098946]
    # was a real bug — those three numbers don't even sum to 1 (they were leaked
    # from the separate 2-logit worked example earlier in the chapter). The
    # correct softmax of [1000, 1001, 999] is computed here and was used to fix
    # the chapter's text-block output.
    assert torch.allclose(
        result_stable, torch.tensor([0.244728, 0.665241, 0.090031]), atol=1e-5
    ), "stable softmax should match the corrected reference values"
    assert torch.allclose(result_stable.sum(), torch.tensor(1.0), atol=1e-6), "softmax must sum to 1"
    assert torch.isnan(result_fp16_naive.float()).all(), "naive fp16 softmax should overflow to NaN"
    assert torch.allclose(
        result_fp16_stable.float(), torch.tensor([0.244751, 0.665527, 0.090088]), atol=1e-3
    ), "stable fp16 softmax should be finite and correct"
    assert torch.allclose(lse, expected), "stable_logsumexp must match torch's log(sum(exp(z)))"

    # stable_log_softmax must agree with torch's own log_softmax
    z_ls = torch.tensor([0.5, -2.0, 3.3, 1.1])
    assert torch.allclose(
        stable_log_softmax(z_ls), torch.log_softmax(z_ls, dim=-1), atol=1e-6
    ), "stable_log_softmax should match torch.log_softmax"

    print("\n[block #1 OK] stable softmax / log-softmax / logsumexp verified.\n")


# =====================================================================
# Block #3 (chapter line ~286): Kahan compensated summation
# =====================================================================

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

# Book uses N = 1_000_000; kept as-is (well under the 60s budget for a
# pedagogical pure-Python Kahan loop — runs in a couple of seconds).
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

# --- honesty checks ---
# Naive float32 sum should have visibly larger absolute error than the
# float64 reference (float32 accumulation of 1e6 ones drowns the 1e-7
# perturbation entirely — the perturbation is lost).
naive_err = abs(naive_result - true_sum)
kahan_err = abs(kahan_result - true_sum)
ref_err = abs(ref_result - true_sum)
assert naive_err > 0.0, "naive float32 accumulation should exhibit some rounding error"
# Kahan summation (even though the running total/compensation is Python
# float = float64 here, applied to float32-valued inputs) should recover
# the true sum far more accurately than naive float32 summation.
assert kahan_err < naive_err, "Kahan summation should be more accurate than naive float32 summation"
assert ref_err < 1e-6, "float64 reference sum should be essentially exact"

print("\n[block #3 OK] Kahan summation reduces error vs. naive float32 summation.\n")


# =====================================================================
# Block #7 (chapter line ~575): fp32 gradient-norm clipping
# =====================================================================

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

# --- honesty checks ---
assert norm_before > 100.0, "the injected gradient spike should have a large norm before clipping"
assert abs(norm.item() - norm_before) < 1e-2 * norm_before, "stable_gradient_norm_clip should report the pre-clip norm"
assert norm_after <= 1.0 + 1e-3, "gradient norm after clipping should be at most max_norm (1.0)"

print("\n[block #7 OK] fp32 gradient-norm clipping caps the norm at max_norm.\n")


# =====================================================================
# Block #5 (chapter line ~427): fp32 bit layout, format limits,
# fp16 overflow, and loss scaling — all CPU-runnable.
# =====================================================================
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
    return scaled_grad_fp16, unscaled, has_inf


show_format_limits()
demonstrate_fp16_overflow()
_scaled, _unscaled, _has_inf = demonstrate_loss_scaling()
print("\nIEEE 754 bit layout of 1.0 in fp32:")
print(f"  1.0 = {fp32_to_bits(1.0)}")
print("IEEE 754 bit layout of inf in fp32:")
print(f"  inf = {fp32_to_bits(float('inf'))}")

# --- honesty checks (verify the chapter's ```text``` output block ~line 511) ---
# fp16 overflows past its max (~65504); bf16 keeps fp32's range and does not.
# 6.55e4 = 65500 rounds to the fp16 max 65504 (still finite); the overflow to
# inf happens by 1e5. bf16 stays finite all the way to fp32-scale magnitudes.
assert torch.isfinite(torch.tensor(6.55e4, dtype=torch.float16)), "65500 should round to fp16 max (finite)"
assert torch.isfinite(torch.tensor(6.55e4, dtype=torch.bfloat16)), "6.55e4 should be finite in bf16"
assert torch.tensor(1e5, dtype=torch.float16).item() == float('inf'), "1e5 should overflow fp16 to inf"
assert torch.tensor(1e38, dtype=torch.float16).item() == float('inf'), "1e38 should overflow fp16 to inf"
assert torch.isfinite(torch.tensor(1e5, dtype=torch.bfloat16)), "1e5 should be finite in bf16 (fp32 range)"
assert torch.isfinite(torch.tensor(1e38, dtype=torch.bfloat16)), "1e38 should be finite in bf16 (fp32 range)"
# fp16 min normal ~6.1e-5; finfo checks match the chapter's table
assert abs(torch.finfo(torch.float16).max - 6.5504e4) < 1.0, "fp16 max should be ~65504"
# Bit-layout of 1.0: sign 0, exponent 01111111 (=127=bias), mantissa all zero.
assert fp32_to_bits(1.0).startswith("[0|01111111|00000000000000000000000]"), "1.0 bit layout wrong"
assert fp32_to_bits(float('inf')).startswith("[0|11111111|00000000000000000000000]"), "inf bit layout wrong"
# Loss scaling: the tiny 1e-8 gradient underflows fp16 alone, but scaling by
# 2**15 lifts it into fp16's representable range (finite, non-zero), then it
# unscales back close to the original — and does not spuriously overflow.
assert _has_inf is False, "scaled gradient should not overflow (no inf/nan)"
assert torch.isfinite(_scaled).all() and _scaled.item() != 0.0, "scaled gradient should be finite and non-zero in fp16"
assert abs(_unscaled.item() - 1e-8) < 1e-9, "unscaled gradient should recover ~1e-8"

print("\n[block #5 OK] fp32 bit layout, format limits, fp16 overflow, loss scaling verified.\n")


# =====================================================================
# SKIP notes (not executed — see module docstring for rationale)
# =====================================================================
# block #0 (~line 51):  ```text``` ASCII bit-layout diagram — not Python.
# block #2 (~line 251):  ```text``` expected-output listing — not Python.
# block #4 (~line 376):  TF32 register-check print is CPU-safe, but the entire
#                         fp32/tf32/bf16 matmul precision comparison is gated
#                         behind `if torch.cuda.is_available():` and genuinely
#                         needs CUDA tensor cores to be meaningful. SKIP(needs-gpu).
# block #6 (~line 511):  ```text``` expected-output listing — not Python.

print("=== All tested blocks (#1, #3, #5, #7) executed and verified successfully. ===")
