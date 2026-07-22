"""
Runnable-code test for content/03-pretraining/08-mixed-precision-fp8.md

Blocks tested (CPU-runnable, executed in chapter order so later blocks can
reuse names defined by earlier ones):
  - block #0 (line ~30):  fp32_to_bf16_bits() — bit-level bf16 truncation demo.
  - block #2 (line ~107): DynamicLossScaler — the AIMD loss-scale controller.
  - block #4 (line ~237): DelayedScale — FP8 per-tensor delayed-scaling cast.

Blocks intentionally SKIPPED (see reasons inline where they would appear):
  - block #1 (line ~96):  ```text``` ASCII diagram of the AMP data flow — not
    Python.
  - block #3 (line ~145): the full AMP train_step() — calls build_transformer()
    (undefined in the chapter, needs a real model), device="cuda", and
    torch.autocast(device_type="cuda", ...). Needs a GPU. SKIP(needs-gpu).
  - block #5 (line ~281): imports `transformer_engine.pytorch` and runs
    `te.Linear(...)` / `te.fp8_autocast(...)` on device="cuda". Requires the
    Transformer Engine package and an H100-class GPU. SKIP(needs-gpu).
  - block #6 (line ~324): ```text``` debugging-checklist table — not Python.
"""

import struct
import sys

import torch

# ---------------------------------------------------------------------------
# Block #0 (line ~30): fp32 -> bf16 bit truncation, verified against PyTorch's
# own bf16 cast.
# ---------------------------------------------------------------------------


def fp32_to_bf16_bits(x: float) -> int:
    """Show that bf16 is literally the top 16 bits of fp32.
    Round-to-nearest-even on the 16 discarded mantissa bits."""
    [bits] = struct.unpack("<I", struct.pack("<f", x))  # 32-bit pattern
    # round to nearest even: add the rounding bias before truncating
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    bits = (bits + rounding_bias) >> 16
    return bits & 0xFFFF


x = 3.1415927
print(f"fp32 {x}  ->  bf16 bits 0x{fp32_to_bf16_bits(x):04x}")
# Compare against PyTorch's own conversion:
t = torch.tensor(x, dtype=torch.float32)
print("torch bf16:", t.to(torch.bfloat16).item())  # ~3.140625  (7-bit mantissa)

# --- glue: actually exercise the function and cross-check the two paths ----
manual_bits = fp32_to_bf16_bits(x)
torch_bf16 = t.to(torch.bfloat16)
# Reinterpret our manually-truncated 16 bits (placed in the high half of a
# 32-bit word, low half zero) as an fp32 float and compare to torch's bf16
# value promoted back to fp32 -- they should match exactly, since both are
# "round bf16 mantissa, keep bf16 exponent" of the same source float.
reconstructed_fp32_bits = manual_bits << 16
[reconstructed] = struct.unpack("<f", struct.pack("<I", reconstructed_fp32_bits))
assert reconstructed == torch_bf16.item(), (
    f"manual bf16 truncation {reconstructed} != torch bf16 {torch_bf16.item()}"
)
assert abs(reconstructed - 3.140625) < 1e-6
print("[block #0] fp32->bf16 truncation matches torch.to(torch.bfloat16): OK")


# ---------------------------------------------------------------------------
# Block #2 (line ~107): the conceptual core of a dynamic GradScaler (AIMD
# controller for the fp16 loss scale).
# ---------------------------------------------------------------------------


class DynamicLossScaler:
    def __init__(self, init_scale=2.0**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor      # multiply by this on success
        self.backoff_factor = backoff_factor    # multiply by this on overflow
        self.growth_interval = growth_interval  # steps of success before growing
        self._good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def update(self, found_inf: bool):
        """Call after inspecting unscaled grads for inf/nan."""
        if found_inf:
            self.scale *= self.backoff_factor   # too big -> back off, skip step
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale *= self.growth_factor  # been safe a while -> grow
                self._good_steps = 0


# --- glue: drive the controller through a small, deterministic step schedule
scaler = DynamicLossScaler(init_scale=2.0**16, growth_factor=2.0,
                            backoff_factor=0.5, growth_interval=4)

loss = torch.tensor(2.3)
scaled_loss = scaler.scale_loss(loss)
assert scaled_loss.item() == loss.item() * 2.0**16
print(f"[block #2] scaled loss = {scaled_loss.item():.1f} (scale={scaler.scale})")

# 1) Simulate an overflow -> scale should halve and good-step counter reset.
pre_scale = scaler.scale
scaler.update(found_inf=True)
assert scaler.scale == pre_scale * 0.5
assert scaler._good_steps == 0

# 2) Simulate `growth_interval` clean steps -> scale should double exactly
#    once, on the step that reaches the threshold.
pre_scale = scaler.scale
for step in range(scaler.growth_interval):
    scaler.update(found_inf=False)
assert scaler.scale == pre_scale * 2.0
assert scaler._good_steps == 0
print(f"[block #2] AIMD controller behaves correctly, final scale={scaler.scale}")


# ---------------------------------------------------------------------------
# Block #4 (line ~237): sketch of an FP8 per-tensor cast with delayed scaling
# (the Transformer Engine idea).
# ---------------------------------------------------------------------------

FP8_E4M3_MAX = 448.0


class DelayedScale:
    def __init__(self, history_len=16, margin=1.0):
        self.amax_history = torch.zeros(history_len)
        self.ptr = 0
        self.margin = margin

    def compute_scale(self):
        amax = self.amax_history.max().clamp_min(1e-12)   # max over recent history
        # scale maps amax -> FP8 max, with a safety margin < 1
        return (FP8_E4M3_MAX / amax) * self.margin

    def cast_to_fp8(self, x: torch.Tensor):
        scale = self.compute_scale()                 # uses PAST amax (delayed)
        x_scaled = x * scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)     # native FP8 dtype
        # record THIS tensor's amax for future steps (off the critical path)
        self.amax_history[self.ptr] = x.abs().amax()
        self.ptr = (self.ptr + 1) % self.amax_history.numel()
        # return both: the matmul de-scales its output by 1/scale
        return x_fp8, scale


# --- glue: instantiate, run it on a couple of toy tensors, and check the
# round-trip (cast to fp8, de-scale, compare to the original) is accurate to
# within the ~12.5% relative spacing that E4M3's 3-bit mantissa allows.
torch.manual_seed(0)
ds = DelayedScale(history_len=4, margin=1.0)

x0 = torch.randn(8, 16)
# Step 0: the amax history is still all-zero (cold start), so this call only
# *seeds* the history for next time -- exactly the "delayed" part of delayed
# scaling (this step's cast uses a placeholder scale, same as real TE on its
# very first step before any history exists).
ds.cast_to_fp8(x0)

# Step 1: now compute_scale() uses a real amax from step 0's tensor.
x1 = torch.randn(8, 16)
x1_fp8, scale1 = ds.cast_to_fp8(x1)
assert x1_fp8.dtype == torch.float8_e4m3fn
assert ds.ptr == 2  # two casts recorded into the ring buffer

x1_recovered = x1_fp8.float() / scale1
max_rel_err = ((x1_recovered - x1).abs() / x1.abs().clamp_min(1e-6)).max().item()
print(f"[block #4] FP8 cast scale={scale1.item():.3f}  max relative error={max_rel_err:.3f}")
# E4M3 has ~12.5% relative spacing; allow generous slack for the coarsest bin.
assert max_rel_err < 0.30, f"FP8 round-trip error too large: {max_rel_err}"

print("\nAll CPU-runnable blocks executed successfully.")
