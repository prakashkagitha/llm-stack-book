"""
Runs the CPU-runnable code from content/02-transformer/07-build-gpt-from-scratch.md
to prove it actually executes.

Tested blocks (verbatim from the chapter, concatenated in chapter order):
  - block #0  (line ~32):  GPTConfig dataclass
  - block #1  (line ~75):  LayerNorm with optional bias
  - block #6  (line ~312): character-level tokenizing + get_batch
  - block #11 (line ~604): RoPE cache / rotate_half / apply_rope + RMSNorm + SwiGLUFFN
  - block #17 (line ~879): `python -c "import transformers, os; print(...)"` one-liner

Skipped blocks (not exercised here), with reasons:
  - block #2  (CausalSelfAttention class):        fragment; exercised only as part of
                                                     the full GPT forward pass, which the
                                                     harness does not build (kept small/fast).
  - block #3  (MLP class):                         fragment, same reason as #2.
  - block #4  (Block class):                       fragment, same reason as #2.
  - block #5  (GPT module, incl. weight tying):    fragment; instantiating + training the
                                                     full ~10M-param model is the subject of
                                                     block #8 below (needs-gpu / too slow for
                                                     a CI-safe test budget).
  - block #7  (configure_optimizer):               fragment; depends on a live GPT instance
                                                     from block #5.
  - block #8  (the 5000-iteration training loop):  needs-gpu / too slow for CI (~minutes even
                                                     on GPU per the chapter's own timing note).
  - block #9  (save/load checkpoint + loop):        needs-gpu, same training loop as #8.
  - block #10 (generate() + sampling call):         fragment; requires a trained `model` and
                                                     `stoi` from the (skipped) training loop.
  - block #12 (ModernCausalSelfAttention/ModernBlock): fragment; needs a live cos/sin cache
                                                     from a ModernGPT instance to call forward.
  - block #13 (ModernGPT module):                   fragment; same training-loop dependency
                                                     as block #5/#8 (heavy to instantiate+train
                                                     within a CPU-safe budget here); its pieces
                                                     (build_rope_cache/RMSNorm/SwiGLUFFN) are
                                                     already exercised directly via block #11.
  - block #14 (train ModernGPT):                    needs-gpu, same reason as #8.
  - block #15 (`attn_implementation="sdpa"` snippet): fragment; not a standalone statement
                                                     (undefined `name`), and would download a
                                                     checkpoint -- needs-net.
  - block #16 (`AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")`): needs-net,
                                                     downloads a real HF Hub checkpoint.

No network calls are made anywhere in this file.
"""

import os
import shutil
import sys
import tempfile

import torch

# =============================================================================
# Block #0 (line ~32, 17 lines): GPTConfig dataclass
# =============================================================================
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- Sequence / vocabulary ---
    block_size: int = 256     # max context length T (positions the model can see)
    vocab_size: int = 65      # size of the token vocabulary V

    # --- Model shape ---
    n_layer: int = 6          # number of Transformer blocks stacked
    n_head:  int = 6          # attention heads per block (must divide n_embd)
    n_embd:  int = 384        # residual-stream width (a.k.a. d_model)

    # --- Regularization / numerics ---
    dropout: float = 0.0      # dropout prob (0.0 for small/clean data; 0.1+ to regularize)
    bias:    bool  = False    # use bias terms in Linear / LayerNorm? (False = modern default)


# Exercise it: instantiate with the book's defaults, and with a tiny CPU-safe config
# that later blocks in this file will reuse.
default_cfg = GPTConfig()
assert default_cfg.n_embd % default_cfg.n_head == 0
tiny_cfg = GPTConfig(block_size=16, vocab_size=13, n_layer=2, n_head=2, n_embd=8,
                      dropout=0.0, bias=False)
assert tiny_cfg.n_embd % tiny_cfg.n_head == 0
print(f"[block 0] GPTConfig OK: default={default_cfg}\n           tiny={tiny_cfg}")

# =============================================================================
# Block #1 (line ~75, 15 lines): LayerNorm with an optional bias
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch's built-in does not allow bias=None."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))               # gamma, scale
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None  # beta, shift

    def forward(self, x):
        # Normalize over the last dim only. eps keeps us safe when variance ~ 0.
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


# Exercise it: both the bias=False and bias=True paths.
ln_nobias = LayerNorm(tiny_cfg.n_embd, bias=False)
ln_bias = LayerNorm(tiny_cfg.n_embd, bias=True)
x_ln = torch.randn(2, 5, tiny_cfg.n_embd)
y_nobias = ln_nobias(x_ln)
y_bias = ln_bias(x_ln)
assert y_nobias.shape == x_ln.shape == y_bias.shape
assert ln_nobias.bias is None and ln_bias.bias is not None
print(f"[block 1] LayerNorm OK: output shape {tuple(y_nobias.shape)}")

# =============================================================================
# Block #6 (line ~312, 27 lines): tokenizing and batching
# =============================================================================
import numpy as np

# The book reads "input.txt" from disk. Materialize a tiny fixture, run the block's
# code verbatim in a scratch directory, then clean up.
_scratch_dir = tempfile.mkdtemp(prefix="build_gpt_scratch_")
_orig_cwd = os.getcwd()
os.chdir(_scratch_dir)
try:
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write("To be, or not to be: that is the question.\n" * 30)

    # --- Build a character-level vocabulary from the raw text ---
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))            # e.g. 65 unique characters
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
    itos = {i: ch for i, ch in enumerate(chars)}   # int    -> string
    encode = lambda s: [stoi[c] for c in s]        # text -> list[int]
    decode = lambda l: "".join(itos[i] for i in l) # list[int] -> text

    # Encode the entire corpus once, then split 90/10 into train/val.
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split, block_size, batch_size, device):
        """Sample `batch_size` random contiguous windows of length block_size.
        x is tokens [i : i+T]; y is the SAME window shifted by one: [i+1 : i+1+T].
        So y[t] is the next-token target for x[t] — the whole autoregressive labelling."""
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))   # random start offsets
        x = torch.stack([d[i     : i + block_size]     for i in ix])  # (B, T)
        y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])  # (B, T), shifted
        return x.to(device), y.to(device)

    # Exercise it: call get_batch with tiny CPU-safe shapes.
    xb, yb = get_batch("train", block_size=8, batch_size=4, device="cpu")
    assert xb.shape == (4, 8) and yb.shape == (4, 8)
    # Shift-by-one sanity check: y[t] should equal x[t+1] within each window.
    assert torch.equal(yb[:, :-1], xb[:, 1:])
    assert decode(encode("to be")) == "to be"
    print(f"[block 6] tokenize/batch OK: vocab_size={vocab_size}, "
          f"xb={tuple(xb.shape)}, yb={tuple(yb.shape)}")
finally:
    os.chdir(_orig_cwd)
    shutil.rmtree(_scratch_dir, ignore_errors=True)

# =============================================================================
# Block #11 (line ~604, 64 lines): RoPE cache/rotation + RMSNorm + SwiGLUFFN
# (reused verbatim from Chapters 2.5/2.6 inside this chapter's "Modern GPT" section)
# =============================================================================
from typing import Optional


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables for RoPE (Llama / rotate_half convention).
    Returns cos, sin of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE needs an even head dimension."
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                 # (seq_len, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, d)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) -> (-x2, x1) where x1,x2 are the two halves of the last dim."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, n_head, T, head_dim); cos, sin: (T, head_dim)."""
    cos = cos[None, None, :, :]   # (1, 1, T, head_dim) — broadcasts over batch & heads
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019). No bias, no mean subtraction."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SwiGLUFFN(nn.Module):
    """FFN(x) = W_down( Swish(W_gate x) * W_up x ). hidden_dim defaults to
    8/3 * dim rounded up to a multiple of 64, matching Llama's convention."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up   = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# Exercise all four pieces with tiny CPU-safe shapes.
_head_dim = 4
cos_cache, sin_cache = build_rope_cache(seq_len=6, head_dim=_head_dim)
assert cos_cache.shape == (6, _head_dim) and sin_cache.shape == (6, _head_dim)

dummy_qk = torch.randn(2, 3, 6, _head_dim)  # (B, n_head, T, head_dim)
rotated = apply_rope(dummy_qk, cos_cache, sin_cache)
assert rotated.shape == dummy_qk.shape
# RoPE must preserve vector norm per (batch, head, position) — it is a rotation.
assert torch.allclose(rotated.norm(dim=-1), dummy_qk.norm(dim=-1), atol=1e-4)

rms = RMSNorm(tiny_cfg.n_embd)
x_rms = torch.randn(2, 5, tiny_cfg.n_embd)
y_rms = rms(x_rms)
assert y_rms.shape == x_rms.shape

ffn = SwiGLUFFN(dim=tiny_cfg.n_embd)
y_ffn = ffn(x_rms)
assert y_ffn.shape == x_rms.shape
print(f"[block 11] RoPE/RMSNorm/SwiGLUFFN OK: cos/sin={tuple(cos_cache.shape)}, "
      f"rotated={tuple(rotated.shape)}, rms_out={tuple(y_rms.shape)}, "
      f"swiglu_out={tuple(y_ffn.shape)}")

# =============================================================================
# Block #17 (line ~879, 2 lines):
#   python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"
# `transformers` is not a guaranteed CI dependency, so it is imported defensively;
# this reproduces the one-liner's own logic rather than shelling out to `python -c`.
# =============================================================================
try:
    import transformers
except Exception:
    transformers = None

if transformers is not None:
    print(f"[block 17] transformers install path: {os.path.dirname(transformers.__file__)}")
else:
    print("[block 17] SKIP(optional dep not installed): transformers not available in this env")

print("\nAll tested blocks executed successfully.")
