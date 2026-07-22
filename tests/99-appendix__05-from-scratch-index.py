"""
CI-tested extracts of runnable code blocks from
content/99-appendix/05-from-scratch-index.md

This appendix chapter indexes the from-scratch implementations used
throughout the book, so its blocks intentionally depend on each other
(e.g. the GPT block reuses the MultiHeadAttention class from the
attention block). We therefore concatenate the blocks in the chapter's
own order at module scope -- exactly like running the chapter's code
as a single script -- and then exercise each one (instantiate classes,
call functions) inside a `test_blockN` function.

Blocks tested (numbering matches the orchestrator's block index):
  0  (~line 18)  Scalar autograd `Value` engine
  1  (~line 119) BPE tokenizer training
  2  (~line 219) Scaled dot-product attention + MultiHeadAttention
  3  (~line 312) RoPE (precompute_freqs_cis / apply_rotary_emb)
  4  (~line 386) GPT (TransformerBlock, GPT model)
  5  (~line 539) LoRALinear
  6  (~line 640) DPO (compute_log_probs, dpo_loss)
  7  (~line 728) GRPO (grpo_loss)
  8  (~line 783) FlashAttention CPU reference (online softmax)
  9  (~line 856) Speculative decoding
  10 (~line 947) RAG pipeline (RAGIndex, rag_generate)
  11 (~line 1036) Decoding samplers + beam search

Blocks skipped: none. Every code block in this chapter is pure-CPU
PyTorch/Python and is exercised here. (Blocks #7 grpo_loss and #8
flash_attention_cpu_reference are self-contained functions over plain
tensors -- grpo_loss backprops on CPU, and the FlashAttention block is
an explicitly-named *CPU reference* whose output is checked against
naive full-matrix attention -- so neither warrants a skip.)
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Block 0 (lines ~18-107): Scalar autograd engine
# =============================================================================
class Value:
    """Scalar-valued node in a dynamic computation graph.

    Tracks: data (forward value), grad (accumulated gradient),
    _backward (closure that pushes grad to parents), _prev (parent nodes).
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0                     # dL/d(self), accumulated
        self._backward = lambda: None       # filled in by ops below
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ---- forward ops -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  →  chain rule
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad   # d(ab)/da = b
            other.grad += self.data  * out.grad   # d(ab)/db = a
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad    # d(tanh(x))/dx = 1 - tanh²(x)
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad              # d(e^x)/dx = e^x
        out._backward = _backward
        return out

    def __pow__(self, k):                          # k must be a constant
        out = Value(self.data**k, (self,), f'**{k}')

        def _backward():
            self.grad += k * self.data**(k-1) * out.grad
        out._backward = _backward
        return out

    # Convenience wrappers
    def __neg__(self):  return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    # ---- backward pass (topological sort) ----------------------------------
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def test_block0():
    """L = tanh(a*b + c); check forward value and grads against finite differences."""
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b
    d = e + c
    L = d.tanh()
    L.backward()

    assert abs(L.data - math.tanh(2.0 * -3.0 + 10.0)) < 1e-9

    h = 1e-4
    def f(av, bv, cv):
        return math.tanh(av * bv + cv)
    base = f(2.0, -3.0, 10.0)
    da = (f(2.0 + h, -3.0, 10.0) - base) / h
    db = (f(2.0, -3.0 + h, 10.0) - base) / h
    dc = (f(2.0, -3.0, 10.0 + h) - base) / h
    assert abs(a.grad - da) < 1e-2
    assert abs(b.grad - db) < 1e-2
    assert abs(c.grad - dc) < 1e-2


# =============================================================================
# Block 1 (lines ~119-176): BPE tokenizer training
# =============================================================================
def get_vocab(corpus: str) -> dict[str, int]:
    """Split corpus into whitespace-separated words, append </w> sentinel."""
    vocab: dict[str, int] = defaultdict(int)
    for word in corpus.split():
        # Each word becomes a tuple of characters + end-of-word marker
        vocab[' '.join(list(word) + ['</w>'])] += 1
    return vocab

def get_stats(vocab: dict[str, int]) -> dict[tuple, int]:
    """Count all adjacent symbol pairs across all words."""
    pairs: dict[tuple, int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Replace every occurrence of `pair` with its concatenation."""
    new_vocab: dict[str, int] = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def train_bpe(corpus: str, num_merges: int) -> list[tuple[str, str]]:
    """
    Run BPE training for num_merges steps.
    Returns the ordered list of merge rules.
    """
    vocab = get_vocab(corpus)
    merges: list[tuple[str, str]] = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        # Greedy: pick the most frequent pair. Tie-break on the pair itself so the
        # output is *deterministic* across runs/machines (reproducibility) — this
        # matches the char-level trainer in the Tokenization chapter.
        best_pair = max(pairs, key=lambda p: (pairs[p], p))
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)
        print(f"Merge {i+1:3d}: {best_pair}  (freq={pairs[best_pair]})")
    return merges


def test_block1():
    """Reproduce the chapter's worked example: 'low lower newest widest', 10 merges."""
    corpus = "low lower newest widest"
    merges = train_bpe(corpus, num_merges=10)
    assert len(merges) == 10
    # Worked example in the chapter text: first merge is the lexicographically
    # largest of the 6 tied pairs, ('w', 'e').
    assert merges[0] == ('w', 'e'), f"expected first merge ('w','e'), got {merges[0]}"
    assert merges[1] == ('t', '</w>')


# =============================================================================
# Block 2 (lines ~219-294): Scaled dot-product attention + MultiHeadAttention
# =============================================================================
def scaled_dot_product_attention(
    q: torch.Tensor,   # (B, H, T, d_k)
    k: torch.Tensor,   # (B, H, S, d_k)
    v: torch.Tensor,   # (B, H, S, d_v)
    mask: torch.Tensor | None = None,  # (B, 1, T, S) or (T, S), True = attend
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of scaled dot-product attention.
    Returns (output, attention_weights).
    """
    d_k = q.size(-1)
    # (B, H, T, S) — raw attention logits
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Where mask is False (0), fill with -inf so softmax → 0
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, S)
    # Weighted sum of values
    output = torch.matmul(attn_weights, v)    # (B, H, T, d_v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention from Vaswani et al. 2017.
    Splits d_model into H heads of size d_k = d_model // H.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.H = num_heads
        self.d_k = d_model // num_heads

        # Single fused projection: [Q, K, V] packed for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, d_model) — query source
        context: torch.Tensor | None = None,   # (B, S, d_model) — key/val source (cross-attn)
        mask: torch.Tensor | None = None,      # causal or padding mask
    ) -> torch.Tensor:
        B, T, _ = x.shape
        kv_src = context if context is not None else x
        S = kv_src.size(1)

        # Project and split heads
        qkv_q = self.qkv_proj(x)[:, :, :self.d_model]
        qkv_k = self.qkv_proj(kv_src)[:, :, self.d_model:2*self.d_model]
        qkv_v = self.qkv_proj(kv_src)[:, :, 2*self.d_model:]

        def split_heads(t, length):
            return t.view(B, length, self.H, self.d_k).transpose(1, 2)  # (B,H,L,d_k)

        q = split_heads(qkv_q, T)
        k = split_heads(qkv_k, S)
        v = split_heads(qkv_v, S)

        # Attention + dropout on weights
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
        # Merge heads: (B, H, T, d_k) → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(attn_out)


def test_block2():
    torch.manual_seed(0)
    B, H, T, S, d_k, d_v = 2, 4, 5, 5, 8, 8
    q = torch.randn(B, H, T, d_k)
    k = torch.randn(B, H, S, d_k)
    v = torch.randn(B, H, S, d_v)
    causal_mask = torch.tril(torch.ones(T, S, dtype=torch.bool))
    out, attn = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    assert out.shape == (B, H, T, d_v)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, H, T), atol=1e-5)

    mha = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 6, 32)
    y = mha(x)
    assert y.shape == (2, 6, 32)

    # cross-attention path (context != x)
    ctx = torch.randn(2, 9, 32)
    y_cross = mha(x, context=ctx)
    assert y_cross.shape == (2, 6, 32)


# =============================================================================
# Block 3 (lines ~312-363): RoPE
# =============================================================================
def precompute_freqs_cis(d: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    Pre-compute complex exponentials e^{i m theta_j} for each position m
    and each frequency pair j = 0..d//2-1.

    Returns: (max_seq_len, d//2) complex64 tensor.
    """
    # theta_j = 1 / base^(2j/d)
    freqs = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))  # (d//2,)
    t = torch.arange(max_seq_len)                                  # (max_seq_len,)
    freqs = torch.outer(t, freqs)                                  # (max_seq_len, d//2)
    # Represent rotation as complex number: cos(m*theta) + i*sin(m*theta)
    return torch.polar(torch.ones_like(freqs), freqs)              # complex64


def apply_rotary_emb(
    xq: torch.Tensor,            # (B, T, H, d_head)   — queries
    xk: torch.Tensor,            # (B, T, H, d_head)   — keys
    freqs_cis: torch.Tensor,     # (T, d_head//2)      — precomputed rotations
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    Reshape to (B, T, H, d//2, 2), cast to complex, multiply by freq, cast back.
    """
    # View last dim as pairs: (B, T, H, d//2, 2)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    # Cast to complex: real + i*imag
    xq_c = torch.view_as_complex(xq_)          # (B, T, H, d//2)
    xk_c = torch.view_as_complex(xk_)
    # freqs_cis: (T, d//2) → broadcast over (B, T, H, d//2)
    freqs = freqs_cis[:xq.size(1), :]          # slice to actual seq len
    freqs = freqs.unsqueeze(0).unsqueeze(2)    # (1, T, 1, d//2)

    xq_rot = torch.view_as_real(xq_c * freqs).flatten(3)   # back to (B,T,H,d)
    xk_rot = torch.view_as_real(xk_c * freqs).flatten(3)
    return xq_rot.type_as(xq), xk_rot.type_as(xk)


def test_block3():
    d_head, T, B, H = 64, 16, 2, 8
    freqs = precompute_freqs_cis(d_head, max_seq_len=2048)
    q = torch.randn(B, T, H, d_head)
    k = torch.randn(B, T, H, d_head)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    assert q_rot.shape == (2, 16, 8, 64)
    assert k_rot.shape == (2, 16, 8, 64)
    # rotation must preserve vector norm (it's an orthogonal transform)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)

    # Relative-position property: dot(q_rot[m], k_rot[n]) depends only on m-n.
    d = 8
    freqs2 = precompute_freqs_cis(d, max_seq_len=32)
    base_q = torch.randn(1, 1, 1, d)
    base_k = torch.randn(1, 1, 1, d)
    qq = base_q.expand(1, 32, 1, d).clone()
    kk = base_k.expand(1, 32, 1, d).clone()
    qr, kr = apply_rotary_emb(qq, kk, freqs2)
    dot_5_2 = (qr[0, 5, 0] * kr[0, 2, 0]).sum().item()
    dot_15_12 = (qr[0, 15, 0] * kr[0, 12, 0]).sum().item()
    assert abs(dot_5_2 - dot_15_12) < 1e-4, (
        f"RoPE relative-position invariance violated: {dot_5_2} vs {dot_15_12}"
    )


# =============================================================================
# Block 4 (lines ~386-516): GPT
# =============================================================================
# --- Causal mask (lower triangular) ---
def make_causal_mask(T: int, device) -> torch.Tensor:
    """Returns (T, T) bool tensor: True where attention is allowed."""
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Linear, expansion factor 4×."""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm residual block: LN → Attention → residual → LN → FFN → residual.
    Pre-norm (used by GPT-2+) is more stable than post-norm at large scale.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer
        x = x + self.drop(self.attn(self.ln1(x), mask=mask))
        # FFN sub-layer
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class GPT(nn.Module):
    """
    Decoder-only GPT: token embedding + position embedding + N transformer blocks
    + final LN + unembedding head (weight-tied with token embedding).
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)   # learned positions
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f  = nn.LayerNorm(d_model)
        # Unembedding: project back to vocab logits; weight-tied with tok_emb
        self.head  = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

        # Initialize weights (GPT-2 style: scale residual branches by 1/sqrt(N))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,                 # (B, T) — token indices
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        device = idx.device

        # Token + position embeddings
        tok = self.tok_emb(idx)                             # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T, device=device)) # (T, d_model) broadcast
        x = tok + pos

        # Causal mask: (T, T)
        mask = make_causal_mask(T, device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.head(x)    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, V) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,     # (B, T_prompt)
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation: greedy / top-k sampling."""
        for _ in range(max_new_tokens):
            # Truncate to context window if needed
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature      # (B, V)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


def test_block4():
    torch.manual_seed(0)
    vocab_size, d_model, num_heads, num_layers, max_seq_len = 50, 32, 4, 2, 16
    model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len, dropout=0.0)

    B, T = 2, 8
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, vocab_size)
    assert loss is not None and loss.item() > 0

    # confirm gradients actually flow through the whole stack
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    opt.step()

    gen = model.generate(idx[:, :4], max_new_tokens=3, temperature=1.0, top_k=5)
    assert gen.shape == (B, 4 + 3)


# =============================================================================
# Block 5 (lines ~539-624): LoRALinear
# =============================================================================
class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a LoRA adapter.
    The original weight W0 is frozen; only A and B are trained.

    Usage:
        layer = LoRALinear(in_features=4096, out_features=4096, rank=16, alpha=32)
        # Merge for inference (no runtime overhead):
        layer.merge_weights()
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank  = rank
        self.alpha = alpha
        # Scaling factor: multiply ΔW by alpha/rank so that changing rank
        # doesn't require re-tuning the learning rate.
        self.scaling = alpha / rank

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.bias = None

        # Trainable LoRA matrices
        # A: initialized with Kaiming uniform (non-zero → non-zero grad from step 0)
        # B: initialized to zero so ΔW = BA = 0 at init (preserve pretrained behavior)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(p=dropout)
        self.merged = False

        # Copy pretrained weight in practice; here we just init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Merged: W0 already contains ΔW, single matmul
            return nn.functional.linear(x, self.weight, self.bias)
        # Base path (frozen) + LoRA path (trained)
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = nn.functional.linear(
            nn.functional.linear(self.lora_dropout(x), self.lora_A),   # (*, rank)
            self.lora_B                                                   # (*, out)
        ) * self.scaling
        return base + lora

    @torch.no_grad()
    def merge_weights(self):
        """Fuse ΔW into W0 for zero-overhead inference."""
        if not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    @torch.no_grad()
    def unmerge_weights(self):
        """Un-fuse for continued fine-tuning."""
        if self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}")


def test_block5():
    torch.manual_seed(0)
    layer = LoRALinear(in_features=16, out_features=16, rank=4, alpha=8.0, dropout=0.0)
    # Give the adapter nonzero values (book inits B=0) so merge/unmerge is a
    # nontrivial numerical check rather than a 0==0 tautology.
    with torch.no_grad():
        layer.lora_B.copy_(torch.randn_like(layer.lora_B))

    x = torch.randn(3, 16)
    y_unmerged = layer(x)
    assert not layer.merged
    layer.merge_weights()
    assert layer.merged
    y_merged = layer(x)
    assert torch.allclose(y_unmerged, y_merged, atol=1e-4), "merge_weights changed forward output"

    layer.unmerge_weights()
    assert not layer.merged
    y_unmerged2 = layer(x)
    assert torch.allclose(y_unmerged, y_unmerged2, atol=1e-4)


# =============================================================================
# Block 6 (lines ~640-712): DPO
# =============================================================================
def compute_log_probs(
    model,
    input_ids: torch.Tensor,      # (B, T)
    attention_mask: torch.Tensor, # (B, T)
    response_start: int,          # index where response begins (after prompt)
) -> torch.Tensor:
    """
    Compute per-sequence log-probabilities of the response tokens only.
    Returns: (B,) tensor of sum(log p(y_t | y_<t, x)).
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift: logits[t] predicts token[t+1]
    logits = logits[:, :-1, :]         # (B, T-1, V)
    labels = input_ids[:, 1:]          # (B, T-1)
    mask   = attention_mask[:, 1:]     # (B, T-1)

    # Log-prob of each token
    log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
    ).view(logits.size(0), logits.size(1))  # (B, T-1)

    # Only sum over response tokens (after the prompt)
    response_mask = mask.clone()
    response_mask[:, :response_start] = 0
    return (log_probs * response_mask).sum(dim=1)   # (B,)


def dpo_loss(
    policy_model,
    reference_model,
    batch: dict,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute the DPO loss for a batch of (prompt, winning, losing) triples.
    Reference model runs under no_grad to save memory.
    """
    prompt_len = batch['prompt_len']   # scalar: length of prompt

    # Policy log-probs
    pi_logp_w = compute_log_probs(policy_model,
                                   batch['input_ids_w'],
                                   batch['mask_w'],
                                   prompt_len)
    pi_logp_l = compute_log_probs(policy_model,
                                   batch['input_ids_l'],
                                   batch['mask_l'],
                                   prompt_len)
    # Reference log-probs (frozen)
    with torch.no_grad():
        ref_logp_w = compute_log_probs(reference_model,
                                        batch['input_ids_w'],
                                        batch['mask_w'],
                                        prompt_len)
        ref_logp_l = compute_log_probs(reference_model,
                                        batch['input_ids_l'],
                                        batch['mask_l'],
                                        prompt_len)

    # DPO implicit reward margins
    margin = beta * ((pi_logp_w - ref_logp_w) - (pi_logp_l - ref_logp_l))
    loss = -F.logsigmoid(margin).mean()

    # Diagnostic: fraction of pairs where policy already prefers winner
    accuracy = (margin > 0).float().mean()
    return loss, accuracy


# ---- glue: minimal HF-style model wrapper, not part of the book's code -----
class _DummyLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _DummyLM(nn.Module):
    """Tiny LM exposing the `model(input_ids=..., attention_mask=...).logits`
    / `model(input_ids)` interfaces that compute_log_probs, speculative_decode
    and beam_search assume. Fixture only -- not from the book."""
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        return _DummyLMOutput(self.head(self.emb(input_ids)))


def test_block6():
    torch.manual_seed(0)
    vocab_size = 20
    policy_model = _DummyLM(vocab_size)
    reference_model = _DummyLM(vocab_size)
    B, T = 3, 6
    prompt_len = 2
    batch = {
        'prompt_len': prompt_len,
        'input_ids_w': torch.randint(0, vocab_size, (B, T)),
        'mask_w': torch.ones(B, T, dtype=torch.long),
        'input_ids_l': torch.randint(0, vocab_size, (B, T)),
        'mask_l': torch.ones(B, T, dtype=torch.long),
    }
    loss, accuracy = dpo_loss(policy_model, reference_model, batch, beta=0.1)
    assert loss.dim() == 0
    assert 0.0 <= accuracy.item() <= 1.0

    loss.backward()
    assert policy_model.head.weight.grad is not None
    assert reference_model.head.weight.grad is None  # ref model ran under no_grad


# =============================================================================
# Block 7 (lines ~728-758): GRPO
# =============================================================================
def grpo_loss(
    policy_log_probs: torch.Tensor,   # (G,) current policy log p(o_i|q)
    old_log_probs: torch.Tensor,      # (G,) old policy log p at sampling time
    ref_log_probs: torch.Tensor,      # (G,) reference policy log p
    rewards: torch.Tensor,            # (G,) scalar reward per response
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> torch.Tensor:
    """
    GRPO loss for one question with G sampled responses.
    """
    G = rewards.size(0)

    # Group-normalized advantages
    r_mean = rewards.mean()
    r_std  = rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - r_mean) / r_std   # (G,)

    # Importance sampling ratio
    log_ratio = policy_log_probs - old_log_probs   # (G,)
    ratio     = log_ratio.exp()                    # (G,)

    # Clipped surrogate (PPO-style)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL penalty: KL(pi_theta || pi_ref) ≈ log(pi_theta/pi_ref)
    kl_penalty = (policy_log_probs - ref_log_probs).mean()

    return policy_loss + beta * kl_penalty


def test_block7():
    """grpo_loss is a fully self-contained CPU function over plain tensors."""
    torch.manual_seed(0)
    G = 6
    policy_log_probs = torch.randn(G, requires_grad=True)
    old_log_probs = torch.randn(G)
    ref_log_probs = torch.randn(G)
    rewards = torch.randn(G)

    loss = grpo_loss(policy_log_probs, old_log_probs, ref_log_probs, rewards)
    assert loss.dim() == 0            # scalar
    loss.backward()                   # gradients flow to the policy log-probs
    assert policy_log_probs.grad is not None

    # When policy == old, all importance ratios are exactly 1, so the clipped
    # surrogate reduces to -mean(advantages) == 0 (advantages are zero-mean),
    # leaving only the KL penalty beta * mean(policy - ref).
    same = torch.randn(G)
    loss2 = grpo_loss(same, same, ref_log_probs, rewards, beta=0.01)
    expected_kl = 0.01 * (same - ref_log_probs).mean()
    assert torch.allclose(loss2, expected_kl, atol=1e-6), (
        f"ratio==1 case should equal beta*KL: {loss2.item()} vs {expected_kl.item()}"
    )

    # Group advantages are normalized to zero mean, unit (biased) std.
    r_mean = rewards.mean()
    r_std = rewards.std(unbiased=False) + 1e-8
    adv = (rewards - r_mean) / r_std
    assert abs(adv.mean().item()) < 1e-5


# =============================================================================
# Block 8 (lines ~783-838): FlashAttention CPU reference (online softmax)
# =============================================================================
def flash_attention_cpu_reference(
    Q: torch.Tensor,   # (T, d)
    K: torch.Tensor,   # (T, d)
    V: torch.Tensor,   # (T, d)
    block_size: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    """
    CPU reference implementation of FlashAttention tiling.
    Demonstrates the online softmax algorithm; NOT optimized for speed.

    The algorithm avoids materializing the full (T, T) attention score matrix.
    Instead it processes K,V in blocks and maintains running (m, ell, O) statistics.
    """
    T, d = Q.shape
    scale = d ** -0.5

    # Accumulators
    O     = torch.zeros(T, d)           # output (running weighted sum of V)
    m_row = torch.full((T,), float('-inf'))  # running row-wise max of scores
    ell   = torch.zeros(T)             # running normalization denominator

    for j_start in range(0, T, block_size):
        j_end = min(j_start + block_size, T)
        Kj = K[j_start:j_end]   # (Bc, d)
        Vj = V[j_start:j_end]   # (Bc, d)

        # Scores for all query rows against this K-block: (T, Bc)
        scores = (Q @ Kj.T) * scale

        if causal:
            # Mask: query position i should not attend to key position j > i
            i_idx = torch.arange(T).unsqueeze(1)
            j_idx = torch.arange(j_start, j_end).unsqueeze(0)
            causal_mask = (j_idx > i_idx)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Local row max for this block
        m_block = scores.max(dim=1).values   # (T,)

        # Update running max
        m_new = torch.maximum(m_row, m_block)

        # Rescale previous accumulator by exp(m_prev - m_new)
        alpha = (m_row - m_new).exp()
        # New block softmax (un-normalized)
        beta  = (scores - m_new.unsqueeze(1)).exp()  # (T, Bc)

        # Update denominator and output
        ell_new = alpha * ell + beta.sum(dim=1)
        O = (O * (alpha * ell).unsqueeze(1) + (beta @ Vj)) / ell_new.unsqueeze(1)

        m_row = m_new
        ell   = ell_new

    return O   # (T, d)  — same as softmax(QK^T/sqrt(d)) @ V


def _naive_attention(Q, K, V, causal):
    T, d = Q.shape
    scores = (Q @ K.T) * (d ** -0.5)
    if causal:
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    return F.softmax(scores, dim=-1) @ V


def test_block8():
    """Tiled online-softmax output must match naive full-matrix attention.

    Despite living in the FlashAttention chapter, this is an explicitly-named
    CPU reference in pure PyTorch -- fully CPU-runnable, no GPU required.
    """
    torch.manual_seed(0)
    T, d = 20, 8
    Q = torch.randn(T, d)
    K = torch.randn(T, d)
    V = torch.randn(T, d)

    # Causal, and with a block_size that does NOT divide T (exercises the tail block).
    O = flash_attention_cpu_reference(Q, K, V, block_size=6, causal=True)
    assert O.shape == (T, d)
    ref = _naive_attention(Q, K, V, causal=True)
    assert torch.allclose(O, ref, atol=1e-5), (
        f"flash causal mismatch: max diff {(O - ref).abs().max().item()}"
    )

    # Non-causal path too.
    O2 = flash_attention_cpu_reference(Q, K, V, block_size=8, causal=False)
    ref2 = _naive_attention(Q, K, V, causal=False)
    assert torch.allclose(O2, ref2, atol=1e-5)

    # Single block (block_size >= T) must also match.
    O3 = flash_attention_cpu_reference(Q, K, V, block_size=T, causal=True)
    assert torch.allclose(O3, ref, atol=1e-5)


# =============================================================================
# Block 9 (lines ~856-936): Speculative decoding
# =============================================================================
def speculative_decode(
    target_model,
    draft_model,
    prompt: torch.Tensor,          # (1, T_prompt)
    max_new_tokens: int = 100,
    num_speculative: int = 5,      # K: draft tokens per verification step
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Speculative decoding: draft K tokens with a small model, verify with target.
    Returns the generated sequence (1, T_prompt + max_new_tokens).

    Lossless: the output distribution is identical to sampling from target alone.
    """
    generated = prompt.clone()
    target_tokens_generated = 0

    while target_tokens_generated < max_new_tokens:
        # ---- DRAFT PHASE: generate K tokens with the small model ----
        draft_seq = generated.clone()
        draft_log_probs = []   # store log P_draft(x_t) for each drafted token

        for k in range(num_speculative):
            with torch.no_grad():
                logits_d = draft_model(draft_seq).logits[:, -1, :]   # (1, V)
                logits_d = logits_d / temperature
                p_draft  = F.softmax(logits_d, dim=-1)
                tok      = torch.multinomial(p_draft, 1)
                draft_log_probs.append(p_draft[0, tok.item()].log())
                draft_seq = torch.cat([draft_seq, tok], dim=1)

        # Draft tokens are draft_seq[:, len(generated):]
        draft_tokens = draft_seq[:, generated.size(1):]   # (1, K)

        # ---- VERIFY PHASE: one target model forward pass over all K tokens ----
        # Feed the original context + all K draft tokens in one shot
        with torch.no_grad():
            logits_t = target_model(draft_seq).logits   # (1, T+K, V)

        # For position i in [len(generated)-1, len(generated)-1+K]:
        # logits_t[:, i, :] gives target's distribution for draft_tokens[:, i-len(generated)]
        accepted_count = 0
        for k in range(num_speculative):
            pos = generated.size(1) - 1 + k          # index into logits_t
            logits_target_k = logits_t[:, pos, :] / temperature
            p_target = F.softmax(logits_target_k, dim=-1)
            x_k = draft_tokens[0, k].item()

            # Acceptance criterion: accept with prob min(1, p_target(x)/p_draft(x))
            p_t_xk = p_target[0, x_k].item()
            p_d_xk = draft_log_probs[k].exp().item()
            accept_prob = min(1.0, p_t_xk / (p_d_xk + 1e-10))

            if torch.rand(1).item() < accept_prob:
                # Accept draft token
                generated = torch.cat([generated, draft_tokens[:, k:k+1]], dim=1)
                accepted_count += 1
                target_tokens_generated += 1
            else:
                # Reject: sample a corrected token from (p_target - p_draft)_+
                correction = (p_target[0] - torch.tensor(p_d_xk)).clamp(min=0.0)
                correction /= correction.sum() + 1e-10
                corrected = torch.multinomial(correction, 1)
                generated = torch.cat([generated, corrected.unsqueeze(0)], dim=1)
                target_tokens_generated += 1
                break   # restart draft from corrected token

        if accepted_count == num_speculative:
            # All K accepted — also greedily take the bonus target token
            bonus_logits = logits_t[:, generated.size(1)-1, :] / temperature
            bonus_tok = torch.multinomial(F.softmax(bonus_logits, dim=-1), 1)
            generated = torch.cat([generated, bonus_tok], dim=1)
            target_tokens_generated += 1

    return generated


def test_block9():
    torch.manual_seed(0)
    vocab_size = 30
    target_model = _DummyLM(vocab_size)
    draft_model = _DummyLM(vocab_size)
    target_model.eval()
    draft_model.eval()

    prompt = torch.randint(0, vocab_size, (1, 4))
    out = speculative_decode(
        target_model, draft_model, prompt,
        max_new_tokens=4, num_speculative=2, temperature=1.0,
    )
    assert out.shape[0] == 1
    assert out.shape[1] >= prompt.shape[1] + 4  # at least max_new_tokens appended
    assert torch.equal(out[:, :4], prompt)      # original prompt preserved


# =============================================================================
# Block 10 (lines ~947-1026): RAG pipeline
# =============================================================================
@dataclass
class RAGIndex:
    """
    Minimal in-memory dense retrieval index.
    In production, replace with FAISS, Chroma, Weaviate, etc.
    """
    chunks: list[str] = field(default_factory=list)
    embeddings: torch.Tensor | None = None   # (N, d_embed)

    def add(self, texts: list[str], encoder):
        """Encode and store a batch of text chunks."""
        self.chunks.extend(texts)
        with torch.no_grad():
            vecs = encoder(texts)              # (len(texts), d_embed)
        vecs = F.normalize(vecs, dim=-1)       # unit-normalize for cosine sim
        self.embeddings = (
            torch.cat([self.embeddings, vecs]) if self.embeddings is not None
            else vecs
        )

    def retrieve(
        self,
        query: str,
        encoder,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return top-k (chunk, cosine_similarity) pairs for a query."""
        with torch.no_grad():
            q_vec = encoder([query])           # (1, d_embed)
        q_vec = F.normalize(q_vec, dim=-1)
        # Cosine similarity: dot product because both are unit-normalized
        sims  = (self.embeddings @ q_vec.T).squeeze(-1)   # (N,)
        topk  = sims.topk(min(top_k, len(self.chunks)))
        return [(self.chunks[i], sims[i].item()) for i in topk.indices]


def rag_generate(
    question: str,
    index: RAGIndex,
    encoder,
    generator,
    tokenizer,
    top_k: int = 5,
    max_new_tokens: int = 256,
) -> str:
    """
    Full RAG loop: retrieve → format context → generate answer.
    """
    # 1. Retrieve relevant chunks
    retrieved = index.retrieve(question, encoder, top_k=top_k)

    # 2. Build context string (simple concatenation; production uses reranker)
    context = "\n---\n".join(
        f"[Source {i+1}] (similarity={sim:.3f}):\n{chunk}"
        for i, (chunk, sim) in enumerate(retrieved)
    )

    # 3. Format prompt with context
    prompt = (
        f"Use the following passages to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    # 4. Tokenize + generate
    inputs = tokenizer(prompt, return_tensors='pt')
    output_ids = generator.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,     # greedy for factual tasks
    )
    # Decode only the newly generated tokens
    answer_ids = output_ids[0, inputs['input_ids'].size(1):]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)


# ---- glue: toy encoder/tokenizer/generator, not part of the book's code ----
def _dummy_encoder(texts: list[str]) -> torch.Tensor:
    """Deterministic toy sentence encoder (hash -> fixed random vector)."""
    d = 8
    vecs = []
    for t in texts:
        g = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
        vecs.append(torch.randn(d, generator=g))
    return torch.stack(vecs)


class _DummyTokenizer:
    def __call__(self, text, return_tensors='pt'):
        ids = [(ord(c) % 20) + 1 for c in text[:16]]
        return {'input_ids': torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return f"dummy answer with {len(ids)} tokens"


class _DummyGenerator:
    def generate(self, input_ids, max_new_tokens=256, do_sample=False, **kwargs):
        B, T = input_ids.shape
        new_tokens = torch.randint(1, 20, (B, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)


def test_block10():
    torch.manual_seed(0)
    index = RAGIndex()
    chunks = [
        "The mitochondria is the powerhouse of the cell.",
        "Paris is the capital of France.",
        "Transformers use self-attention.",
    ]
    index.add(chunks, _dummy_encoder)
    assert index.embeddings.shape == (3, 8)

    results = index.retrieve("What is the capital of France?", _dummy_encoder, top_k=2)
    assert len(results) == 2
    assert all(isinstance(c, str) and isinstance(s, float) for c, s in results)

    answer = rag_generate(
        "What is the capital of France?",
        index, _dummy_encoder, _DummyGenerator(), _DummyTokenizer(),
        top_k=2, max_new_tokens=5,
    )
    assert isinstance(answer, str) and len(answer) > 0


# =============================================================================
# Block 11 (lines ~1036-1153): Decoding samplers + beam search
# =============================================================================
def greedy_decode(logits: torch.Tensor) -> int:
    """Argmax of logits. Deterministic; maximizes single-step likelihood."""
    return logits.argmax(dim=-1).item()


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature before softmax.
    T→0: peaked/greedy.  T=1: unmodified.  T→∞: uniform.
    """
    assert temperature > 0
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out all logits except the top-k.
    Returns modified logits (not probabilities).
    """
    if k == 0:
        return logits
    # kth-largest value
    threshold = logits.topk(k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float('-inf'))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling (Holtzman et al., 2019): keep the smallest set of tokens
    whose cumulative probability exceeds p.
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumprobs = sorted_probs.cumsum(dim=-1)
    # Remove tokens after the cumulative probability passes p
    # Shift right by 1 so we always keep at least 1 token
    remove = (cumprobs - sorted_probs) > p
    sorted_probs[remove] = 0.0
    # Scatter back to original order
    probs = probs.scatter(-1, sorted_idx, sorted_probs)
    # Re-convert to logits (for consistent interface)
    return probs.log().clamp(min=-1e9)


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    Min-P sampling: remove tokens whose probability is less than
    min_p * max_probability. Adapts the cutoff to the shape of the distribution.
    """
    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob
    return logits.masked_fill(probs < threshold, float('-inf'))


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> int:
    """
    Composable sampler pipeline:
      temperature → top_k → top_p → min_p → softmax → multinomial
    """
    logits = temperature_scale(logits, temperature)
    if top_k > 0:
        logits = top_k_filter(logits, top_k)
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    if min_p > 0.0:
        logits = min_p_filter(logits, min_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# --- Beam search (minimal, width B) ---
def beam_search(
    model,
    input_ids: torch.Tensor,     # (1, T_prompt)
    beam_width: int = 4,
    max_new_tokens: int = 50,
    length_penalty: float = 1.0,
) -> list[tuple[float, list[int]]]:
    """
    Minimal beam search. Returns list of (score, token_ids) sorted by score.
    Score = sum(log P) / length^alpha.
    """
    # Each beam: (log_prob, token_id_list)
    beams = [(0.0, input_ids[0].tolist())]

    for _ in range(max_new_tokens):
        candidates = []
        for log_prob, seq in beams:
            inp = torch.tensor([seq], dtype=torch.long)
            with torch.no_grad():
                logits = model(inp).logits[0, -1, :]
            log_probs_next = F.log_softmax(logits, dim=-1)
            # Take top beam_width expansions
            topk = log_probs_next.topk(beam_width)
            for tok_lp, tok_id in zip(topk.values, topk.indices):
                candidates.append((
                    log_prob + tok_lp.item(),
                    seq + [tok_id.item()]
                ))
        # Keep top beam_width by length-normalized score
        candidates.sort(
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True
        )
        beams = candidates[:beam_width]

    return beams


def test_block11():
    torch.manual_seed(0)
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 5.0])
    assert greedy_decode(logits) == 4

    scaled = temperature_scale(logits, 2.0)
    assert torch.allclose(scaled, logits / 2.0)

    filtered_k = top_k_filter(logits.clone(), k=2)
    kept = (filtered_k != float('-inf')).sum().item()
    assert kept == 2

    filtered_p = top_p_filter(logits.clone(), p=0.5)
    probs_p = F.softmax(filtered_p, dim=-1)
    assert probs_p[4] > 0  # highest-probability token must always survive

    filtered_minp = min_p_filter(logits.clone(), min_p=0.5)
    assert filtered_minp[3] == float('-inf')  # low-prob token pruned
    assert filtered_minp[4] != float('-inf')  # max-prob token kept

    tok = sample_token(logits.clone(), temperature=1.0, top_k=3, top_p=0.9, min_p=0.0)
    assert 0 <= tok < logits.numel()

    # beam search over a tiny dummy model (reuses _DummyLM from block 6/9 glue)
    vocab_size = 15
    dummy_model = _DummyLM(vocab_size)
    dummy_model.eval()
    prompt = torch.randint(0, vocab_size, (1, 3))
    beams = beam_search(dummy_model, prompt, beam_width=2, max_new_tokens=3, length_penalty=1.0)
    assert len(beams) == 2
    assert all(len(seq) == 3 + 3 for _, seq in beams)
    # sorted descending by length-normalized score
    scores = [lp / (len(seq) ** 1.0) for lp, seq in beams]
    assert scores[0] >= scores[1]


if __name__ == "__main__":
    # Run in chapter order (NOT alphabetical -- block10/block11 would sort
    # before block2..block9 and the dependency comment above would be a lie).
    ordered = [
        ("block0", test_block0),
        ("block1", test_block1),
        ("block2", test_block2),
        ("block3", test_block3),
        ("block4", test_block4),
        ("block5", test_block5),
        ("block6", test_block6),
        ("block7", test_block7),
        ("block8", test_block8),
        ("block9", test_block9),
        ("block10", test_block10),
        ("block11", test_block11),
    ]
    for name, fn in ordered:
        fn()
        print(f"PASS {name} ({fn.__name__})")
    print(f"\nAll {len(ordered)} blocks passed. (No blocks skipped.)")
