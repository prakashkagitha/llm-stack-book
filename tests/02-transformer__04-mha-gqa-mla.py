"""
Executable test for content/02-transformer/04-mha-gqa-mla.md

Assembles the chapter's CPU-runnable Python blocks, in the order they appear
in the chapter, into one runnable module. Each block is copied verbatim from
the book (including its own smoke test / demo code at the bottom of the
block), so this file exercises the book's actual code rather than a
rewritten stand-in.

Blocks tested (chapter order):
    #0 (line ~42)  - MultiHeadAttention (MHA) module + smoke test
    #1 (line ~202) - repeat_kv() + GroupedQueryAttention (unifies MHA/GQA/MQA)
                      + spectrum-verification loop (n_kv in 8, 2, 1)
    #2 (line ~331) - MultiHeadLatentAttentionCore (simplified MLA) + smoke test

All three blocks are self-contained nn.Module definitions with tiny CPU
tensors (batch<=2, seq_len<=16, d_model<=512) and fixed seeds, so nothing
needs mocking and no block needs to be skipped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Block #0 (line ~42): Multi-Head Attention (MHA) from scratch.
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention (MHA).

    d_model : model/embedding width
    n_heads : number of attention heads; head_dim = d_model // n_heads
    All of Q, K, V have n_heads heads — this is the "vanilla" MHA that every
    KV-cache-reduction variant later in this chapter tries to slim down.
    """

    def __init__(self, d_model, n_heads, causal=True, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout

        # One fused projection per role. Output width d_model = n_heads * head_dim.
        # Using three separate Linears (rather than one packed QKV) for clarity;
        # production code often fuses them into a single (d_model -> 3*d_model) GEMM.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # cross-head mixing

    def _split_heads(self, x):
        # x: (B, L, d_model) -> (B, n_heads, L, head_dim)
        B, L, _ = x.shape
        # Split the last axis into (n_heads, head_dim), then move heads up front
        # so each head is a contiguous (L, head_dim) matrix the kernel can batch.
        x = x.view(B, L, self.n_heads, self.head_dim)   # (B, L, h, d_h)
        return x.transpose(1, 2)                          # (B, h, L, d_h)

    def _merge_heads(self, x):
        # x: (B, n_heads, L, head_dim) -> (B, L, d_model)
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous()                # (B, L, h, d_h)
        return x.view(B, L, H * Dh)                        # (B, L, d_model)

    def forward(self, x, attn_mask=None):
        # x: (B, L, d_model)
        q = self._split_heads(self.W_q(x))                # (B, h, L, d_h)
        k = self._split_heads(self.W_k(x))                # (B, h, L, d_h)
        v = self._split_heads(self.W_v(x))                # (B, h, L, d_h)

        # Fused SDPA: softmax(QKᵀ/√d_h + mask) V, computed without materializing
        # the L×L score matrix. is_causal builds the triangular mask internally.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_mask is None,
        )                                                  # (B, h, L, d_h)

        out = self._merge_heads(out)                       # (B, L, d_model)
        return self.W_o(out)                               # (B, L, d_model)


# Smoke test: shapes, causality, gradient flow.
torch.manual_seed(0)
B, L, d_model, n_heads = 2, 6, 64, 8
x = torch.randn(B, L, d_model, requires_grad=True)
mha = MultiHeadAttention(d_model, n_heads, causal=True)
y = mha(x)
print("output:", y.shape)                                  # (2, 6, 64)
assert tuple(y.shape) == (B, L, d_model)
y.sum().backward()
print("grad flows:", x.grad.abs().sum().item() > 0)        # True
assert x.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Block #1 (line ~202): repeat_kv() + GroupedQueryAttention (MHA/GQA/MQA).
# ---------------------------------------------------------------------------

def repeat_kv(x, n_rep):
    """Expand KV heads to match query heads (the GQA broadcast).

    x: (B, n_kv, L, d_h). Repeat each KV head n_rep times along the head axis,
    producing (B, n_kv * n_rep, L, d_h) = (B, n_heads, L, d_h).

    This is a memory-cheap expand (no data copy until the kernel reads it):
    head order becomes [kv0, kv0, ..., kv1, kv1, ...], so query head j uses
    KV group j // n_rep. Mirrors Llama's reference implementation.
    """
    B, n_kv, L, d_h = x.shape
    if n_rep == 1:
        return x
    # Insert a length-1 axis after the head axis, expand it, then flatten.
    x = x[:, :, None, :, :]                       # (B, n_kv, 1,     L, d_h)
    x = x.expand(B, n_kv, n_rep, L, d_h)          # (B, n_kv, n_rep, L, d_h)
    return x.reshape(B, n_kv * n_rep, L, d_h)     # (B, n_heads, L, d_h)


class GroupedQueryAttention(nn.Module):
    """GQA that unifies MHA / GQA / MQA via a single n_kv_heads knob.

        n_kv_heads == n_heads  ->  MHA  (one KV head per query head)
        1 < n_kv_heads < n_heads -> GQA (KV heads shared within groups)
        n_kv_heads == 1        ->  MQA  (one KV head for all query heads)
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None, causal=True, dropout=0.0):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert d_model % n_heads == 0

        self.n_heads = n_heads            # number of QUERY heads
        self.n_kv_heads = n_kv_heads      # number of KEY/VALUE heads (= groups g)
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # query heads per KV head
        self.causal = causal
        self.dropout = dropout

        # KEY INSIGHT: Q projects to the FULL d_model (n_heads * head_dim),
        # but K and V project only to n_kv_heads * head_dim — a SMALLER matrix.
        # That smaller K/V projection is exactly what shrinks the cache.
        self.W_q = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape

        # Project. Q has n_heads heads; K, V have only n_kv_heads heads.
        q = self.W_q(x).view(B, L, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, L, d_h)   k,v: (B, n_kv_heads, L, d_h)

        # Broadcast each KV head across its group so shapes align with q.
        # (At inference, you cache the UNREPEATED k, v — size ∝ n_kv_heads —
        #  and repeat on the fly; that is where the memory saving lives.)
        k = repeat_kv(k, self.n_rep)             # (B, n_heads, L, d_h)
        v = repeat_kv(v, self.n_rep)             # (B, n_heads, L, d_h)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_mask is None,
        )                                         # (B, n_heads, L, d_h)

        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.head_dim)
        return self.W_o(out)


# Verify the spectrum: MHA, GQA, and MQA all run and produce correct shapes.
torch.manual_seed(0)
B, L, d_model, n_heads = 2, 7, 64, 8
x = torch.randn(B, L, d_model)

for n_kv in (8, 2, 1):                            # MHA, GQA(g=2), MQA
    attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads=n_kv)
    y = attn(x)
    assert tuple(y.shape) == (B, L, d_model)
    # Cache width per token (one layer, one sequence), bf16 = 2 bytes, ×2 for K&V:
    kv_bytes = 2 * n_kv * (d_model // n_heads) * 2
    print(f"n_kv={n_kv}: out {tuple(y.shape)}, KV bytes/token/layer = {kv_bytes}")
# n_kv=8: 256 B   n_kv=2: 64 B   n_kv=1: 32 B  -> 8× and 16× smaller than MHA
assert 2 * 8 * (d_model // n_heads) * 2 == 256
assert 2 * 2 * (d_model // n_heads) * 2 == 64
assert 2 * 1 * (d_model // n_heads) * 2 == 32


# ---------------------------------------------------------------------------
# Block #2 (line ~331): Multi-head Latent Attention (MLA), simplified core.
# ---------------------------------------------------------------------------

class MultiHeadLatentAttentionCore(nn.Module):
    """Simplified MLA: cache a low-rank latent c^{KV}, reconstruct per-head K,V.

    Omits decoupled-RoPE and weight-absorption for clarity. The point to see:
    the CACHED object is c_kv of width d_c (small), not the full per-head K,V.
    """

    def __init__(self, d_model, n_heads, d_c, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_c = d_c                                   # latent dim, d_c << d_model
        self.causal = causal

        # Down-projection: hidden -> compressed latent (this is what gets cached).
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        # Up-projections: latent -> per-head keys / values.
        self.W_uk = nn.Linear(d_c, n_heads * self.head_dim, bias=False)
        self.W_uv = nn.Linear(d_c, n_heads * self.head_dim, bias=False)
        # Queries: a normal (optionally low-rank) projection; not cached.
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        H, Dh = self.n_heads, self.head_dim

        c_kv = self.W_dkv(x)                              # (B, L, d_c)  <-- CACHED
        k = self.W_uk(c_kv).view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        v = self.W_uv(c_kv).view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        q = self.W_q(x).view(B, L, H, Dh).transpose(1, 2)       # (B, H, L, Dh)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return self.W_o(out)


torch.manual_seed(0)
m = MultiHeadLatentAttentionCore(d_model=512, n_heads=8, d_c=128)
y = m(torch.randn(2, 16, 512))
print("MLA out:", y.shape)                               # (2, 16, 512)
assert tuple(y.shape) == (2, 16, 512)
# Cache per token/layer = d_c numbers (+ a small decoupled-RoPE key in full MLA),
# vs MHA's 2 * d_model. Here 128 vs 1024 -> ~8× smaller, full head diversity kept.


print("\nAll blocks executed successfully.")
