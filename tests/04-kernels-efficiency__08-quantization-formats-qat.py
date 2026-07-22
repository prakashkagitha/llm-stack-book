"""
Runs the CPU-runnable Python blocks from
content/04-kernels-efficiency/08-quantization-formats-qat.md, concatenated in
order so later blocks can rely on names defined by earlier ones. Each block is
copied verbatim from the chapter; minimal glue needed to make a block that
only *defines* something actually execute is added and clearly marked "GLUE".

Blocks covered (as named by the task):
  #4 (line ~237) - GGUF conversion comments (no executable statements; the
                    actual `python convert_hf_to_gguf.py ...` / `llama-quantize`
                    / `llama-cli` commands live in the adjacent ```bash``` block,
                    #5, which is shell, not Python -- SKIP(shell) for that part)
  #6 (line ~287) - Minimal NF4 quantize/dequantize implementation + the
                    chapter's own demo (already exercises the functions)
  #7 (line ~379) - STEQuantize (straight-through estimator) autograd.Function,
                    FakeQuantLinear, and qat_finetune_step -- GLUE below
                    instantiates FakeQuantLinear inside a tiny model and calls
                    qat_finetune_step() so the STE backward path actually runs.
  #9 (line ~511) - quantize_kv_int8 / dequantize_kv + the chapter's own memory
                    comparison demo, run at a much smaller shape (GLUE: the
                    book's literal demo shape is B=1,H=1024,T=8192,D=128 --
                    over 1e9 fp32 elements / ~4.3GB -- far too large for a CPU
                    CI box; shrunk to B=1,H=8,T=64,D=32 here, purely a fixture
                    -size change, the quantize/dequantize logic is untouched).

SKIP(needs-gpu): #0, #1, #2 -- bitsandbytes INT8 (`load_in_8bit`), the FP8
  micro-benchmark, and NF4 loading all require a CUDA GPU (and #0/#2 also
  download a gated Llama checkpoint over the network).
SKIP(shell): #3, #5 -- ```bash``` blocks (llama.cpp build/convert/quantize/run
  commands), not Python.
SKIP(needs-gpu / network): #8 -- QLoRA fine-tuning block imports `peft` and
  `bitsandbytes`, calls `AutoModelForCausalLM.from_pretrained(...)` on a gated
  HF repo with `device_map="auto"` under a 4-bit CUDA quant config -- requires
  both network access and a GPU. `peft`/`bitsandbytes` are third-party and not
  in the guaranteed-available list, so their imports are guarded; the block is
  left defined-not-called (as inert reference text) rather than executed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

# Third-party libs used only by the SKIPPED GPU/network blocks (#0/#2/#8).
# Guarded per the hard rules so this file still loads without them.
try:
    import bitsandbytes as bnb
except Exception:
    bnb = None
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = get_peft_model = TaskType = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None


PASS = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    PASS.append(name)
    print(f"  ok: {name}")


# ---------------------------------------------------------------------------
# Block #4 (line ~237, 8 lines): GGUF conversion intro -- comments only, no
# executable Python statements in the chapter's own fence. Reproduced
# verbatim; it is a legal (empty) statement and "runs" trivially.
# ---------------------------------------------------------------------------
print("Block #4: GGUF conversion header (comment-only block)")

# Convert a Hugging Face model to GGUF Q4_K_M using llama.cpp's converter
# First clone llama.cpp and install dependencies:
# git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
# pip install -r requirements.txt && make -j

# Step 1: Convert HF model to GGUF F16 (lossless intermediate)
# (Run from the llama.cpp directory)

check("block #4 executed (no-op comments)", True)


# ---------------------------------------------------------------------------
# Block #6 (line ~287, 72 lines): Minimal NF4 quantize/dequantize from scratch
# Copied verbatim, including the chapter's own demo at the bottom.
# ---------------------------------------------------------------------------
print("\nBlock #6: Minimal NF4 layer from scratch")

# The 16 NF4 code points (from QLoRA paper, normalized to [-1, 1])
NF4_CODES = torch.tensor([
    -1.0,       -0.6961928,  -0.5250730,  -0.3954816,
    -0.2849375, -0.1832600,  -0.0911578,  0.0,
     0.0795761,  0.1609030,   0.2461331,   0.3379990,
     0.4407979,  0.5626170,   0.7229568,   1.0,
], dtype=torch.float32)

def quantize_nf4(weight: torch.Tensor, group_size: int = 64):
    """
    Quantize a 1-D weight tensor to NF4 per-group.
    Returns: (packed_indices, scales) where packed_indices is uint8
    with two 4-bit indices per byte.
    """
    weight = weight.float()
    n = weight.numel()
    assert n % group_size == 0
    n_groups = n // group_size
    w_groups = weight.view(n_groups, group_size)

    # Scale each group so its max absolute value maps to 1.0
    scales = w_groups.abs().max(dim=1).values  # (n_groups,)
    scales = scales.clamp(min=1e-8)
    w_norm = w_groups / scales.unsqueeze(1)    # (n_groups, group_size) in [-1, 1]

    # Find nearest NF4 code point for each weight
    # Broadcast: (n_groups, group_size, 1) vs (16,)
    codes = NF4_CODES.to(weight.device)
    dists = (w_norm.unsqueeze(-1) - codes).abs()  # (n_groups, group_size, 16)
    indices = dists.argmin(dim=-1).byte()           # (n_groups, group_size), dtype=uint8

    # Pack two 4-bit indices into one byte
    indices_flat = indices.view(-1)  # (n,)
    packed = (indices_flat[0::2] << 4) | indices_flat[1::2]  # (n//2,) uint8

    return packed, scales

def dequantize_nf4(packed: torch.Tensor, scales: torch.Tensor, group_size: int = 64):
    """Unpack NF4 indices and reconstruct FP32 weights."""
    # Unpack nibbles
    hi = (packed >> 4).byte()
    lo = (packed & 0xF).byte()
    indices_flat = torch.stack([hi, lo], dim=1).view(-1)  # interleaved back

    n = indices_flat.numel()
    n_groups = n // group_size
    codes = NF4_CODES.to(packed.device)
    w_norm = codes[indices_flat.long()].view(n_groups, group_size)

    # Re-apply group scales
    w_reconstructed = w_norm * scales.unsqueeze(1)
    return w_reconstructed.view(-1)

# --- Demo ---
torch.manual_seed(42)
w = torch.randn(256)           # simulate a weight vector (one row of a linear layer)
packed, scales = quantize_nf4(w, group_size=64)

print(f"Original size:    {w.numel() * 4} bytes (FP32)")
print(f"Quantized size:   {packed.numel()} bytes (NF4 packed)")
print(f"Scales overhead:  {scales.numel() * 4} bytes (FP32 scales)")

w_hat = dequantize_nf4(packed, scales, group_size=64)
mse = ((w - w_hat) ** 2).mean().item()
snr = (w.var() / mse).item()
print(f"MSE:  {mse:.6f}")
print(f"SNR:  {snr:.1f}  (higher is better; >100 is practically lossless)")

check("NF4 packed size == numel/2", packed.numel() == w.numel() // 2)
check("NF4 dequantized shape matches original", w_hat.shape == w.shape)
check("NF4 round-trip is low-error (SNR > 20 dB)", snr > 20)


# ---------------------------------------------------------------------------
# Block #7 (line ~379, 56 lines): Straight-through estimator (STE) QAT
# Copied verbatim. GLUE: the chapter defines but never calls these; we
# instantiate FakeQuantLinear inside a tiny model and drive
# qat_finetune_step() so the STE forward+backward path actually executes.
# ---------------------------------------------------------------------------
print("\nBlock #7: Straight-through estimator (QAT)")

class STEQuantize(torch.autograd.Function):
    """
    Quantize to b bits with straight-through estimator in backward.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, bits: int):
        # Quantize: clamp to representable range, then round
        qmin = -(2 ** (bits - 1))
        qmax =  (2 ** (bits - 1)) - 1
        x_scaled = x / scale
        x_clipped = x_scaled.clamp(qmin, qmax)
        x_quant = x_clipped.round()
        # Store nothing for backward — STE passes gradient directly
        return x_quant * scale  # dequantized immediately (fake-quant)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        return grad_output, None, None

class FakeQuantLinear(nn.Linear):
    """
    A drop-in replacement for nn.Linear that applies fake-quantization
    to weights during the forward pass (simulates INT4 weight quantization).
    """
    def __init__(self, *args, bits=4, group_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = bits
        self.group_size = group_size

    def get_scale(self, w: torch.Tensor) -> torch.Tensor:
        """Per-group symmetric scale: s = max(|w|) / (2^(b-1) - 1)"""
        w_groups = w.view(-1, self.group_size)
        s = w_groups.abs().max(dim=1).values / (2 ** (self.bits - 1) - 1)
        return s.unsqueeze(1).expand_as(w_groups).reshape_as(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply fake-quantization to weights
        scale = self.get_scale(self.weight)
        w_fq = STEQuantize.apply(self.weight, 1.0, self.bits)  # simplified
        return F.linear(x, w_fq, self.bias)

# Minimal QAT training loop sketch
def qat_finetune_step(model, batch, optimizer):
    """Replace all Linear layers with FakeQuantLinear, then fine-tune."""
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()   # gradients flow through STE
    optimizer.step()
    return loss.item()

# --- GLUE: exercise FakeQuantLinear + qat_finetune_step end to end ---
torch.manual_seed(0)

class TinyQATModel(nn.Module):
    """Toy stand-in for an HF model: forward(**batch) -> object with .loss,
    matching what qat_finetune_step() expects (outputs.loss)."""
    def __init__(self):
        super().__init__()
        self.fq = FakeQuantLinear(128, 4, bits=4, group_size=128)  # 512 weights, 4 groups

    def forward(self, x, target):
        pred = self.fq(x)
        loss = F.mse_loss(pred, target)
        return SimpleNamespace(loss=loss)

tiny_model = TinyQATModel()
weight_before = tiny_model.fq.weight.detach().clone()

x = torch.randn(8, 128)
target = torch.randn(8, 4)
batch = {"x": x, "target": target}
optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.1)

loss1 = qat_finetune_step(tiny_model, batch, optimizer)
loss2 = qat_finetune_step(tiny_model, batch, optimizer)
print(f"QAT step 1 loss: {loss1:.4f}, step 2 loss: {loss2:.4f}")

weight_after = tiny_model.fq.weight.detach().clone()

check("qat_finetune_step returns a finite scalar loss", np.isfinite(loss1))
check("STE backward updated the underlying weights", not torch.allclose(weight_before, weight_after))
check("fake-quantized forward output is finite", torch.isfinite(tiny_model.fq(x)).all().item())

# Directly sanity-check the STE autograd.Function in isolation too: forward
# rounds to an integer multiple of `scale`, backward passes gradient through.
z = torch.randn(16, requires_grad=True)
q = STEQuantize.apply(z, 0.5, 4)  # bits=4 -> qmin=-8, qmax=7
q.sum().backward()
check("STE forward output is quantized to multiples of scale", torch.allclose(q / 0.5, (q / 0.5).round()))
check("STE backward is identity (grad_input == grad_output)", torch.allclose(z.grad, torch.ones_like(z)))


# ---------------------------------------------------------------------------
# Block #9 (line ~511, 30 lines): Simplified KV cache INT8 quantization
# Copied verbatim, except the demo's shape (GLUE, see module docstring: the
# book's literal B=1,H=1024,T=8192,D=128 is ~4.3GB of fp32 and far too large
# for CI; shrunk here to keep the same quantize/dequantize *logic* under a
# small, fast fixture).
# ---------------------------------------------------------------------------
print("\nBlock #9: Simplified KV cache INT8 quantization")

def quantize_kv_int8(kv: torch.Tensor):
    """
    Quantize a KV tensor of shape (batch, heads, seq_len, head_dim) to INT8.
    Per-token (per-position) symmetric quantization.
    Returns int8 tensor + FP16 per-token scale tensor.
    """
    # kv: (B, H, T, D)
    # Compute per-position max-abs across head_dim
    scale = kv.abs().max(dim=-1, keepdim=True).values / 127.0  # (B, H, T, 1)
    scale = scale.clamp(min=1e-8)
    kv_int8 = (kv / scale).round().clamp(-128, 127).to(torch.int8)
    return kv_int8, scale.to(torch.float16)

def dequantize_kv(kv_int8: torch.Tensor, scale: torch.Tensor):
    """Recover approximate FP16 KV from INT8 + scale."""
    return kv_int8.to(torch.float16) * scale

# Memory comparison for a 32-layer, 32-head, 128-dim model at 8K context
# GLUE: book uses B, H, T, D = 1, 32 * 32, 8192, 128 (~4.3GB fp32) -- shrunk
# here to a tiny fixture with the same shape *semantics* (B, flattened-H, T, D).
B, H, T, D = 1, 2 * 4, 64, 32  # flattened heads, shrunk for CPU/CI
kv = torch.randn(B, H, T, D)
kv_int8, scale = quantize_kv_int8(kv.view(B, 2, 4, T, D).view(B, H, T, D))

bf16_size = kv.numel() * 2  # bytes
int8_size  = kv_int8.numel() * 1 + scale.numel() * 2
print(f"BF16 KV size: {bf16_size / 1e9:.6f} GB")
print(f"INT8 KV size: {int8_size  / 1e9:.6f} GB  ({100*int8_size/bf16_size:.0f}% of BF16)")

kv_hat = dequantize_kv(kv_int8, scale)
kv_mse = (kv.to(torch.float16) - kv_hat).float().pow(2).mean().item()

check("KV int8 tensor dtype is torch.int8", kv_int8.dtype == torch.int8)
check("KV int8 values within [-128, 127]", kv_int8.min().item() >= -128 and kv_int8.max().item() <= 127)
check("INT8 KV cache is smaller than BF16 KV cache", int8_size < bf16_size)
check("KV dequantization round-trips with small error", kv_mse < 0.01)


# ---------------------------------------------------------------------------
print(f"\nAll {len(PASS)} checks passed.")
