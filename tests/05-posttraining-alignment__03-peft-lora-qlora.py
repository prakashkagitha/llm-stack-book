"""
Runnability test for content/05-posttraining-alignment/03-peft-lora-qlora.md

Blocks tested (assembled in the order they appear in the chapter, later blocks
depend on names defined earlier -- exactly like a reader executing the chapter
top to bottom):

  - block #0 (line ~105, 89 lines): LoRALinear / inject_lora / mark_only_lora_trainable
  - block #1 (line ~198, 40 lines): toy end-to-end sanity check (frozen base + merge)
  - block #2 (line ~266, ~38 lines): NF4 codebook quantize/dequantize demo
        This block was heuristically flagged "needs-gpu" but on inspection it is
        pure torch tensor math with no CUDA/device calls at all -- trivially
        CPU-safe -- so it is included as a bonus beyond the required 3 blocks.
  - block #4 (line ~390, 26 lines): DoRALinear core forward

Skipped:
  - block #3 (line ~314, "QLoRA in practice with HuggingFace + bitsandbytes + PEFT"):
    SKIP(network+gpu): this block calls
    `AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", ...)`
    which downloads a gated multi-GB checkpoint from the HF Hub (real network
    + auth, forbidden in CI), uses `device_map="auto"`/4-bit bitsandbytes
    kernels that require a CUDA GPU, and imports `transformers`/`peft`, which
    are not in the guaranteed CI import list. It is illustrative "real path"
    code, not something that can run offline on CPU. Left untested here.

No real bugs were found in this chapter's code; all tested blocks ran as
written (only trivial glue -- calling the functions/classes on tiny CPU
tensors -- was added).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Block #0 (line ~105): from-scratch LoRA implementation
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """A frozen base Linear with a trainable low-rank adapter in parallel.

    Forward computes:  h = W0 @ x  +  (alpha / r) * B @ (A @ x)
    Only A and B are trainable; W0 (and bias) are frozen.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        # The effective scale alpha/r decouples magnitude from rank.
        self.scaling = alpha / r

        # --- Frozen base weight (this is W0). Keep the original tensor. ---
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)        # freeze: no grad, no optimizer state

        # --- Trainable low-rank factors. ---
        # A: (r, in)  initialized random (Kaiming);  B: (out, r) initialized ZERO.
        # So B @ A = 0 at init  ->  the adapter starts as a no-op.
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False  # tracks whether the adapter is folded into base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)                      # frozen W0 @ x  (+ bias)
        if self.merged:
            return base_out                          # adapter already folded in
        # Low-rank path: down-project with A, up-project with B, scale.
        lora_out = F.linear(self.dropout(x), self.lora_A)   # (..., r)
        lora_out = F.linear(lora_out, self.lora_B)          # (..., out)
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self):
        """Fold (alpha/r) * B @ A into the base weight for zero-overhead inference."""
        if self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)  # (out, in)
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """Reverse merge() -- useful for hot-swapping adapters on a shared base."""
        if not self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.sub_(delta.to(self.base.weight.dtype))
        self.merged = False


def inject_lora(model: nn.Module, target_names=("q_proj", "k_proj", "v_proj",
                                                 "o_proj", "gate_proj",
                                                 "up_proj", "down_proj"),
                r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Walk the module tree and replace matching nn.Linear layers with LoRALinear."""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
    return model


def mark_only_lora_trainable(model: nn.Module):
    """Freeze everything, then unfreeze only the LoRA factors. Returns trainable count."""
    n_trainable = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            n_trainable += p.numel()
        else:
            p.requires_grad_(False)
    return n_trainable


# ---------------------------------------------------------------------------
# Block #1 (line ~198): tiny end-to-end sanity check -- overfit a single batch
# and confirm only the adapter moves.
# ---------------------------------------------------------------------------

torch.manual_seed(0)

# A toy "model": two linears we will adapt.
model = nn.Sequential(
    nn.Linear(64, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 8, bias=False),
)
# Name the children so inject_lora can find them.
model[0].__class__.__name__  # still nn.Linear; we target by attribute name instead:

# Manually wrap (the helper targets by child attribute name; here we wrap directly):
model[0] = LoRALinear(model[0], r=4, alpha=8)
model[2] = LoRALinear(model[2], r=4, alpha=8)
n = mark_only_lora_trainable(model)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {n} / {total}  ({100*n/total:.2f}%)")   # ~ a few % on this toy

# Snapshot a frozen base weight to prove it does NOT change.
W0_before = model[0].base.weight.detach().clone()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
x = torch.randn(32, 64)
y = torch.randn(32, 8)
for step in range(200):
    opt.zero_grad()
    loss = F.mse_loss(model(x), y)
    loss.backward()
    opt.step()
print(f"final loss {loss.item():.4f}")
assert torch.equal(W0_before, model[0].base.weight), "base weight must stay frozen!"

# Merging must not change the output (within float tolerance).
model.eval()
with torch.no_grad():
    out_unmerged = model(x)
    model[0].merge(); model[2].merge()
    out_merged = model(x)
print("merge max-diff:", (out_unmerged - out_merged).abs().max().item())  # ~1e-6
assert (out_unmerged - out_merged).abs().max().item() < 1e-4, \
    "merging must be output-preserving"


# ---------------------------------------------------------------------------
# Block #2 (line ~266): NF4 codebook quantize/dequantize demo.
# Heuristically flagged "needs-gpu" but this block is pure CPU torch tensor
# math (no .cuda(), no device_map, no external model download) -- trivially
# CPU-safe, so it is exercised here as a bonus beyond the 3 required blocks.
# ---------------------------------------------------------------------------

# The 16 NF4 codebook values (bin midpoints at the quantiles of N(0,1),
# normalized to [-1, 1] with an exact zero). These are constants in bitsandbytes.
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
     0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
     0.7229568362236023, 1.0,
])

def quantize_nf4_block(w_block: torch.Tensor):
    """Quantize one block of weights to NF4. Returns 4-bit codes + the fp scale."""
    absmax = w_block.abs().max()                  # block-wise scale (one fp number)
    w_norm = w_block / (absmax + 1e-8)            # normalize to ~[-1, 1]
    # Nearest codebook entry for each weight -> 4-bit index in [0, 15].
    dist = (w_norm.unsqueeze(-1) - NF4_CODEBOOK).abs()
    codes = dist.argmin(dim=-1).to(torch.uint8)   # store these 4-bit codes
    return codes, absmax

def dequantize_nf4_block(codes: torch.Tensor, absmax: torch.Tensor):
    """Reconstruct bf16 weights from 4-bit codes and the block scale (a lookup)."""
    return NF4_CODEBOOK[codes.long()] * absmax

# Demo: NF4 beats INT4 on Gaussian data.
torch.manual_seed(0)
w = torch.randn(4096)                              # one block of Gaussian weights
codes, scale = quantize_nf4_block(w)
w_hat = dequantize_nf4_block(codes, scale)
nf4_err = (w - w_hat).pow(2).mean().sqrt()

# Plain symmetric INT4 for comparison.
s = w.abs().max() / 7
w_int4 = (w / s).round().clamp(-7, 7) * s
int4_err = (w - w_int4).pow(2).mean().sqrt()
print(f"NF4 RMSE {nf4_err:.4f}   INT4 RMSE {int4_err:.4f}")  # NF4 noticeably lower
assert nf4_err < int4_err, "NF4 should reproduce Gaussian data with lower RMSE than INT4"


# ---------------------------------------------------------------------------
# Block #4 (line ~390): DoRA core forward
# ---------------------------------------------------------------------------

class DoRALinear(nn.Module):
    """DoRA: train direction via low-rank A,B and magnitude m directly."""
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        out_f, in_f = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # Magnitude m = the column-wise norm of the pretrained weight (init so W'=W0).
        with torch.no_grad():
            self.m = nn.Parameter(base.weight.norm(dim=0, keepdim=True))  # (1, in)

    def forward(self, x):
        # Effective directional weight = W0 + scaled low-rank update.
        delta = self.scaling * (self.lora_B @ self.lora_A)          # (out, in)
        V = self.base.weight + delta                                 # direction (unnormalized)
        V_norm = V.norm(dim=0, keepdim=True) + 1e-8                  # column norms (1, in)
        W_eff = self.m * (V / V_norm)                                # rescale to magnitude m
        return F.linear(x, W_eff)                                    # (no separate base path)


# Minimal glue: instantiate on a small Linear and actually call it, verifying
# the chapter's stated invariant -- "at init the adapter is a perfect no-op".
torch.manual_seed(0)
base_linear = nn.Linear(32, 16, bias=False)
dora = DoRALinear(base_linear, r=4, alpha=8)

x_dora = torch.randn(5, 32)
with torch.no_grad():
    out_dora = dora(x_dora)
    out_base = base_linear(x_dora)
dora_diff = (out_dora - out_base).abs().max().item()
print("DoRA init no-op max-diff:", dora_diff)
assert dora_diff < 1e-4, "DoRA must equal the base linear exactly at init"

# Confirm it actually trains (m and the low-rank factors receive gradients).
opt_dora = torch.optim.AdamW(
    [dora.lora_A, dora.lora_B, dora.m], lr=1e-2
)
y_dora = torch.randn(5, 16)
m_before = dora.m.detach().clone()
for _ in range(20):
    opt_dora.zero_grad()
    loss_dora = F.mse_loss(dora(x_dora), y_dora)
    loss_dora.backward()
    opt_dora.step()
print(f"DoRA final loss {loss_dora.item():.4f}")
assert not torch.equal(m_before, dora.m), "magnitude m should have been updated by training"


print("\nAll tested blocks (#0, #1, #2-bonus, #4) executed successfully.")
