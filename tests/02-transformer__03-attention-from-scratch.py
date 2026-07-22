"""
Executable test for content/02-transformer/03-attention-from-scratch.md

Assembles the chapter's CPU-runnable Python blocks, in the order they appear
in the chapter, into one runnable module. Later blocks depend on names
defined by earlier blocks (e.g. the tiny worked example in block #2 uses the
`scaled_dot_product_attention` / `softmax` defined in block #1; block #5's
SelfAttention module uses the `scaled_dot_product_attention` defined in
block #4). This mirrors what happens if a reader runs the chapter top to
bottom in one Python session — including the fact that the chapter
deliberately defines a NumPy `scaled_dot_product_attention` first (block #1)
and later a PyTorch version with the SAME NAME (block #4), which shadows the
first. That shadowing is intentional in the chapter's narrative ("The Same
Thing in PyTorch") and is preserved here rather than worked around.

Blocks tested (chapter order):
    #0 (line ~21)  - hard dict lookup, motivating example
    #1 (line ~151) - NumPy softmax() + scaled_dot_product_attention()
    #2 (line ~205) - tiny hand-checkable NumPy attention example
    #3 (line ~257) - causal_mask numpy demo (exercises the mask branch of #1)
    #4 (line ~281) - PyTorch scaled_dot_product_attention() (shadows #1's name)
    #5 (line ~331) - SelfAttention nn.Module + smoke test
    #6 (line ~386) - validate hand-rolled SDPA against torch's built-in kernel

Every CPU-runnable code block in the chapter is executed here; nothing is
skipped.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Block #0 (line ~21): hard key-value lookup via a Python dict.
# ---------------------------------------------------------------------------
store = {"paris": "france", "tokyo": "japan", "lima": "peru"}
result = store["tokyo"]  # -> "japan"   (exact match on the key "tokyo")
assert result == "japan"
print("block #0 hard lookup ->", result)


# ---------------------------------------------------------------------------
# Block #1 (line ~151): NumPy softmax + scaled_dot_product_attention.
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    """Numerically stable softmax along `axis`.

    The math identity softmax(x) = softmax(x - c) for any constant c lets us
    subtract the per-row max BEFORE exponentiating. This guarantees the largest
    exponent is exp(0) = 1, so we never overflow to +inf. Without this, a logit
    of, say, 1000 would make exp(1000) = inf and the whole row becomes NaN.
    """
    x_max = np.max(x, axis=axis, keepdims=True)      # largest logit per row
    e = np.exp(x - x_max)                             # in (0, 1], no overflow
    return e / np.sum(e, axis=axis, keepdims=True)    # normalize -> sums to 1


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention, fully from scratch.

    Shapes (single, unbatched):
        Q : (n_q, d_k)   one query per output position
        K : (n_k, d_k)   one key per source position
        V : (n_k, d_v)   one value per source position
        mask : (n_q, n_k) or None. Entries that are True (or 1) are positions
               we are ALLOWED to attend to; False/0 positions are forbidden
               and get score -inf before softmax (so weight -> exactly 0).

    Returns:
        out     : (n_q, d_v)  the attended output, one vector per query
        weights : (n_q, n_k)  the attention distribution (rows sum to 1)
    """
    d_k = Q.shape[-1]

    # Step 1: raw similarity scores. scores[i, j] = q_i · k_j
    scores = Q @ K.T                                  # (n_q, n_k)

    # Step 2: scale so the variance of scores is ~1 regardless of d_k
    scores = scores / np.sqrt(d_k)

    # Step 3 (optional): masking. Set forbidden scores to -inf so that
    # exp(-inf) = 0 and those positions receive exactly zero attention weight.
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    # Step 4: softmax over the KEY axis (last axis) -> attention weights
    weights = softmax(scores, axis=-1)                # (n_q, n_k)

    # Step 5: weighted sum of values
    out = weights @ V                                 # (n_q, d_v)
    return out, weights


# ---------------------------------------------------------------------------
# Block #2 (line ~205): tiny hand-checkable NumPy attention example.
# ---------------------------------------------------------------------------
np.random.seed(0)

# Three keys, deliberately pointing along distinct axes, in d_k = 4.
K = np.array([
    [1.0, 0.0, 0.0, 0.0],   # key 0
    [0.0, 1.0, 0.0, 0.0],   # key 1
    [0.0, 0.0, 1.0, 0.0],   # key 2
])
# Values are easy to recognize: value j is the number (j+1) repeated.
V = np.array([
    [10.0, 10.0],           # value 0
    [20.0, 20.0],           # value 1
    [30.0, 30.0],           # value 2
])
# One query that aligns strongly with key 1 (the second axis).
Q = np.array([[0.1, 3.0, 0.1, 0.0]])

out, w = scaled_dot_product_attention(Q, K, V)
print("attention weights:", np.round(w, 3))   # ~ [[0.16 0.681 0.16]]
print("output:", np.round(out, 2))            # ~ [[20. 20.]] -> mostly value 1
print("rows sum to 1:", np.allclose(w.sum(axis=-1), 1.0))  # True

# Honest checks mirroring the chapter's stated expectations.
assert np.allclose(w.sum(axis=-1), 1.0)
assert w[0, 1] > w[0, 0] and w[0, 1] > w[0, 2]  # weight concentrates on key 1
assert np.allclose(out[0], [20.0, 20.0], atol=1.0)  # output pulled toward value 1


# ---------------------------------------------------------------------------
# Block #3 (line ~257): causal_mask numpy demo. This is the ONLY block that
# exercises the `mask` branch of the NumPy scaled_dot_product_attention above,
# so we execute it (it is fully CPU-runnable and deterministic) rather than
# skip it. It must run BEFORE block #4 redefines `scaled_dot_product_attention`
# to the PyTorch version.
# ---------------------------------------------------------------------------
def causal_mask(n):
    """Lower-triangular boolean mask: entry (i, j) is True iff j <= i.
    True = allowed to attend. Shape (n, n)."""
    return np.tril(np.ones((n, n), dtype=bool))       # tril keeps j <= i

n, d_k, d_v = 4, 8, 8
np.random.seed(1)
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
V = np.random.randn(n, d_v)

out, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
print(np.round(w, 3))
print("no future leakage:", np.allclose(np.triu(w, k=1), 0.0))  # True
print("row 0 weight:", np.round(w[0], 3))   # [1. 0. 0. 0.]

# Honest checks mirroring the chapter's stated expectations.
assert np.allclose(np.triu(w, k=1), 0.0)          # no attending to the future
assert np.allclose(w[0], [1.0, 0.0, 0.0, 0.0])    # row 0 sees only position 0
assert np.allclose(w.sum(axis=-1), 1.0)           # each row is a distribution


# ---------------------------------------------------------------------------
# Block #4 (line ~281): PyTorch scaled_dot_product_attention.
# This intentionally SHADOWS the NumPy function of the same name defined in
# block #1, exactly as happens if you run the chapter's code top to bottom.
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                 is_causal=False):
    """From-scratch SDPA matching torch.nn.functional.scaled_dot_product_attention.

    Tensor shapes (the canonical Transformer layout):
        q : (B, H, Lq, d_k)   B=batch, H=heads, Lq=query length
        k : (B, H, Lk, d_k)
        v : (B, H, Lk, d_v)
        attn_mask : broadcastable to (B, H, Lq, Lk). Either a boolean mask
                    (True = keep) or an additive float mask (-inf = forbid).
    Returns:
        out : (B, H, Lq, d_v)
    """
    d_k = q.size(-1)

    # (B,H,Lq,d_k) @ (B,H,d_k,Lk) -> (B,H,Lq,Lk). transpose(-2,-1) makes Kᵀ.
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

    # Build / apply the causal mask if requested.
    if is_causal:
        Lq, Lk = q.size(-2), k.size(-2)
        # Lower-triangular: position i may see j <= i. Forbidden -> -inf.
        causal = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

    # Apply an explicit mask (padding, or arbitrary attention bias).
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask          # additive bias (e.g. ALiBi)

    # Softmax over the KEY axis (last dim). Cast to fp32 for a stable softmax
    # even when q/k/v are bf16 — a standard trick in production kernels.
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)

    return torch.matmul(weights, v)              # (B,H,Lq,d_v)


# ---------------------------------------------------------------------------
# Block #5 (line ~331): a complete, trainable self-attention module.
# ---------------------------------------------------------------------------
import torch.nn as nn

class SelfAttention(nn.Module):
    """Single-head self-attention: x -> (Q,K,V) -> attention -> output projection."""

    def __init__(self, d_model, d_k=None, causal=False, dropout=0.0):
        super().__init__()
        d_k = d_k or d_model
        self.causal = causal
        self.dropout_p = dropout
        # One learned linear map per role. No bias is the modern default.
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.W_o = nn.Linear(d_k, d_model, bias=False)   # mix attended info back

    def forward(self, x, attn_mask=None):
        # x : (B, L, d_model). Project into the three roles.
        q = self.W_q(x)                          # (B, L, d_k): "what I seek"
        k = self.W_k(x)                          # (B, L, d_k): "how to address me"
        v = self.W_v(x)                          # (B, L, d_k): "what I return"

        # Add a singleton head dim so we can reuse the (B,H,L,d) kernel with H=1.
        q, k, v = (t.unsqueeze(1) for t in (q, k, v))    # (B, 1, L, d_k)

        attended = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal,
        )                                         # (B, 1, L, d_k)

        attended = attended.squeeze(1)            # (B, L, d_k)
        return self.W_o(attended)                 # (B, L, d_model)


# Smoke test: shapes, gradient flow, and causality.
torch.manual_seed(0)
B, L, d_model = 2, 5, 16
x = torch.randn(B, L, d_model, requires_grad=True)
attn = SelfAttention(d_model, causal=True)

y = attn(x)
print("output shape:", y.shape)                  # (2, 5, 16)
assert y.shape == (B, L, d_model)

y.sum().backward()                               # autograd through the whole op
print("grad flows to input:", x.grad is not None and x.grad.abs().sum() > 0)
assert x.grad is not None and x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Block #6 (line ~386): validate hand-rolled SDPA against PyTorch's built-in.
# ---------------------------------------------------------------------------
B, H, L, d = 2, 4, 7, 32
q = torch.randn(B, H, L, d)
k = torch.randn(B, H, L, d)
v = torch.randn(B, H, L, d)

ours = scaled_dot_product_attention(q, k, v, is_causal=True)
ref  = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # PyTorch builtin

print("max abs diff:", (ours - ref).abs().max().item())   # ~1e-6, fp32 noise
assert torch.allclose(ours, ref, atol=1e-5)


print("\nAll blocks executed successfully.")
