#  From-Scratch Code Index

This appendix is a master index of every from-scratch implementation woven throughout the book. For each implementation we give: the chapter where it lives, the key signatures, a brief description of what the code demonstrates, and how it connects to the implementations that follow it. Think of this as the dependency graph of the entire textbook's code.

The implementations form a deliberate learning arc: we start with the mathematical primitives (autograd, numerics), build the transformer piece by piece (BPE, embeddings, attention, RoPE, GPT), layer on post-training techniques (LoRA, DPO, GRPO), add hardware-aware kernels (FlashAttention pseudocode), then move to inference-time algorithms (speculative decoding, RAG pipeline, samplers). Reading the implementations in order is a complete bottom-up construction of an LLM system.

---

## 1. Mathematical Foundations: Autograd & Numerics

### 1.1  Scalar Autograd Engine

**Chapter:** [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html)
**Also relevant:** [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html)

The entry point for the whole book. We build a `Value` class that wraps a scalar, records the operation that produced it, and knows how to propagate gradients backward through a computation graph. This is the kernel of PyTorch's autograd, rendered in ~80 lines of Python.

```python
import math

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
```

**How it connects:** Every subsequent implementation in this book relies on the same principle: forward pass builds a graph, backward pass traverses it in reverse applying the chain rule. When we switch to tensor-valued PyTorch operations, the mechanism is identical — just parallelized across elements.

---

## 2. Tokenization: BPE From Scratch

**Chapter:** [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html)

Byte-Pair Encoding (BPE) starts from individual characters (or bytes) and iteratively merges the most frequent adjacent pair into a new symbol. The training loop is $O(V \cdot T)$ where $V$ is the target vocabulary size and $T$ is the training corpus length; the encode step is $O(N \log N)$ using a priority queue.

```python
from collections import defaultdict
import re

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

# --- demo ---
if __name__ == "__main__":
    corpus = "low lower newest widest"
    merges = train_bpe(corpus, num_merges=10)
    # Six pairs tie at freq 2 on this corpus; the deterministic tie-break picks the
    # lexicographically largest, ('w', 'e'), as the first merge.
```

!!! example "Worked example: BPE on a toy corpus"
    Starting corpus: `"low lower newest widest"` (4 words, 20 characters).

    Initial vocabulary (character level + `</w>`):
    `l o w </w>`, `l o w e r </w>`, `n e w e s t </w>`, `w i d e s t </w>`

    Step 1 — count all adjacent pairs. **Six** pairs tie for most frequent, each at freq 2:
    `('l','o')`, `('o','w')`, `('w','e')`, `('e','s')`, `('s','t')`, `('t','</w>')`. The
    deterministic tie-break (`key=lambda p: (count, p)`) picks the lexicographically largest
    pair, `('w','e')`, and merges it into the single symbol `we` (e.g. `n e w e s t </w>`
    becomes `n e we s t </w>`). Note what does — and does not — change: the word-frequency
    dict still holds the same 4 word entries. A merge never adds or removes words; it only
    shortens each affected word's symbol sequence by one and adds one new symbol (`we`) to
    the token inventory.

    Step 2 — with `we` merged, three pairs now tie at freq 2: `('l','o')`, `('s','t')`,
    `('t','</w>')`. The tie-break again takes the lexicographically largest, `('t','</w>')`,
    merging it to `t</w>`.

    Running all 10 steps yields the ordered merge list `('w','e'), ('t','</w>'), ('s','t</w>'),
    ('l','o'), ('we','st</w>'), ('we','r'), ('wer','</w>'), ('w','i'), ('wi','d'), ('wid','e')`,
    leaving segmentations such as `lo w </w>`, `lo wer</w>`, `n e west</w>`, `wide st</w>`.
    Even on this tiny corpus the merges begin recovering subword units (`lo`, `west</w>`,
    `wide`); on a real corpus of millions of words the same greedy procedure recovers
    morphologically meaningful pieces, and a 50k-merge run produces the GPT-2 tokenizer.

---

## 3. Attention Mechanism From Scratch

**Chapter:** [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html)
**See also:** [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)

The canonical scaled dot-product attention:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

where $Q \in \mathbb{R}^{T \times d_k}$, $K \in \mathbb{R}^{S \times d_k}$, $V \in \mathbb{R}^{S \times d_v}$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
```

---

## 4. Positional Encodings: RoPE From Scratch

**Chapter:** [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)

Rotary Position Embedding (RoPE) encodes position by rotating the query and key vectors in 2D planes. For a token at position $m$ in dimension pair $(2i, 2i+1)$:

$$
\begin{bmatrix} q_{2i}' \\ q_{2i+1}' \end{bmatrix}
= \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
\begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
$$

where $\theta_i = 10000^{-2i/d}$.

```python
import torch

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


# --- quick sanity check ---
if __name__ == "__main__":
    d_head, T, B, H = 64, 16, 2, 8
    freqs = precompute_freqs_cis(d_head, max_seq_len=2048)
    q = torch.randn(B, T, H, d_head)
    k = torch.randn(B, T, H, d_head)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    print(q_rot.shape)   # (2, 16, 8, 64)
    # Relative position: dot(q[m], k[n]) depends only on (m-n), not absolute positions
```

!!! example "Worked numerical example: RoPE rotation"
    Take $d=4$ (2 frequency pairs), position $m=3$, base $=10000$.

    Frequency pair 0: $\theta_0 = 10000^{0/4} = 1.0$, so angle $= 3 \times 1.0 = 3.0$ rad.
    Frequency pair 1: $\theta_1 = 10000^{-2/4} = 0.01$, so angle $= 3 \times 0.01 = 0.03$ rad.

    For query $q = [1, 0, 1, 0]^\top$:
    - Pair (0,1): rotate by 3.0 rad → $[\cos 3, -\sin 3, \sin 3, \cos 3] \cdot [1, 0] = [-0.99, 0.14]$
    - Pair (2,3): rotate by 0.03 rad → $[\cos 0.03, -\sin 0.03, \sin 0.03, \cos 0.03] \cdot [1, 0] = [1.00, 0.030]$

    Result: $q' \approx [-0.99,\ 0.14,\ 1.00,\ 0.030]$. The first pair has rotated far (encodes large relative position well for nearby tokens), the last pair barely rotated (encodes long-range relationships).

---

## 5. Building a GPT From Scratch

**Chapter:** [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html)
**See also:** [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html)

The complete autoregressive language model. Every component above is assembled here. The full code is ~300 lines; the skeleton below highlights the wiring.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
```

**Parameter count:** a model with $d=768$, 12 heads, 12 layers, vocab 50257 has:
$$
12 \times (4 \times 768^2 + 2 \times 768^2 \times 4 + 2 \times 768) + 50257 \times 768 \approx 117\text{M params}
$$
— the GPT-2 small configuration.

---

## 6. Parameter-Efficient Fine-Tuning: LoRA From Scratch

**Chapter:** [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)
**See also:** [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html)

LoRA (Hu et al., 2022) freezes the pretrained weight matrix $W_0 \in \mathbb{R}^{m \times n}$ and adds a low-rank decomposition $\Delta W = BA$ where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m, n)$.

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

Trainable parameters drop from $mn$ to $r(m+n)$. For $m=n=4096$, $r=16$: from 16.8M to 131k — a 128x reduction.

```python
import torch
import torch.nn as nn
import math

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
```

---

## 7. Alignment Algorithms: DPO and GRPO From Scratch

### 7.1  Direct Preference Optimization (DPO)

**Chapter:** [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)

DPO (Rafailov et al., 2023) eliminates the explicit reward model. The loss directly optimizes the log-ratio of winning vs. losing response probabilities, implicitly using the policy itself as the reward model:

$$
\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

```python
import torch
import torch.nn.functional as F

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
```

### 7.2  Group Relative Policy Optimization (GRPO)

**Chapter:** [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)
**See also:** [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)

GRPO (Shao et al., 2024) samples $G$ responses per prompt, computes rewards, normalizes within the group to get advantages, and optimizes with a clipped surrogate objective — no value network needed.

$$
\mathcal{L}_\text{GRPO} = -\frac{1}{G}\sum_{i=1}^{G} \min\!\left(\rho_i \hat{A}_i,\ \operatorname{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) + \beta D_\text{KL}(\pi_\theta \| \pi_\text{ref})
$$

where $\rho_i = \pi_\theta(o_i|q)/\pi_{\theta_\text{old}}(o_i|q)$ and $\hat{A}_i = (r_i - \bar{r})/\sigma_r$.

```python
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
```

---

## 8. FlashAttention: IO-Aware Kernel Pseudocode

**Chapter:** [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)
**See also:** [FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8](../04-kernels-efficiency/03-flash-attention-2-3.html)

Standard attention materializes the full $T \times T$ attention matrix in HBM. For $T=8192$ in FP16, that is $8192^2 \times 2\ \text{bytes} \approx 134\ \text{MB}$ per head per layer — a severe bottleneck. FlashAttention (Dao et al., 2022) tiles the computation into SRAM blocks, computing an online softmax to avoid ever writing the full matrix to HBM.

The key identity: given partial max $m_\text{prev}$ and partial sum $\ell_\text{prev}$, after processing a new block with local max $m_\text{new}$:

$$
m = \max(m_\text{prev},\ m_\text{new})
$$

$$
\ell = e^{m_\text{prev} - m}\,\ell_\text{prev} + e^{m_\text{new} - m}\,\ell_\text{new}
$$

```python
import torch

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
```

!!! interview "Interview Corner"
    **Q:** Why does FlashAttention use tiling and an online softmax rather than materializing the full attention matrix? What is the memory complexity of each approach, and what is the main constraint?

    **A:** Standard attention writes the $(T \times T)$ score matrix to HBM (GPU high-bandwidth memory), costing $O(T^2)$ memory per head — on the order of 100s of MB for long sequences. The bandwidth cost of HBM reads/writes dominates runtime because attention is memory-bandwidth-bound, not compute-bound. FlashAttention tiles $Q$, $K$, and $V$ into SRAM-sized blocks (on the order of a few hundred KiB) and maintains a running (max, sum, weighted-output) triple to compute the softmax in a numerically stable online fashion. The $O(T^2)$ arithmetic is unchanged, but HBM reads/writes drop to $O(T)$ in the output. This makes attention IO-complexity $O(T^2 / M)$ where $M$ is SRAM size, vs $O(T^2)$ for standard attention — delivering 2–4x wall-clock speedup and enabling context lengths that would OOM with naive attention.

---

## 9. Speculative Decoding From Scratch

**Chapter:** [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) uses a small draft model to propose $K$ candidate tokens in one forward pass of the target model, then verifies them all in parallel with a single target model pass. If the draft is accepted the effective tokens/step is $K$; rejected drafts fall back to one token, but this is rare for a good draft model.

The acceptance probability for draft token $x$ is $\min(1,\ p_\text{target}(x) / p_\text{draft}(x))$.

```python
import torch
import torch.nn.functional as F
from typing import Callable

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
```

---

## 10. RAG Pipeline From Scratch

**Chapter:** [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html)
**See also:** [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html), [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)

RAG (Lewis et al., 2020) augments a generation model with retrieved passages. The minimal from-scratch loop below builds a dense retrieval index using a sentence encoder, retrieves top-k chunks, and feeds them as context.

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

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
```

---

## 11. Decoding Samplers From Scratch

**Chapter:** [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html)

All sampler variants manipulate the logit vector before drawing a token. Here we implement greedy, top-k, top-p (nucleus), min-p, and temperature as composable transforms.

```python
import torch
import torch.nn.functional as F

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
```

---

## 12. How the Implementations Fit Together

The figure below traces the data and gradient flow through the complete stack:


{{fig:fsindex-stack-dataflow}}


Each implementation is designed to be *swapped in or out* independently. You can replace the BPE tokenizer with a unigram model, the sinusoidal positions with RoPE, or the standard attention kernel with FlashAttention, and everything else stays the same. This modularity is intentional: the book teaches each piece in isolation precisely so you understand what it costs and what it buys.

### Dependency Graph Summary

| Implementation | Depends On | Enables |
|---|---|---|
| Scalar Autograd | Pure Python | All training loops |
| BPE | None (string ops) | Token IDs for embedding |
| Attention | Linear algebra | TransformerBlock |
| RoPE | Attention | Long-context models |
| GPT | Autograd, Attention, RoPE | All fine-tuning |
| LoRA | GPT (frozen base) | Efficient SFT |
| DPO | LoRA / SFT model | Alignment |
| GRPO | SFT model + verifier | RL reasoning |
| FlashAttention | Attention (replaces) | Memory-efficient long-context |
| Speculative Decoding | Draft + Target model | 2–3x inference speedup |
| RAG | Embeddings + Generator | Grounded generation |
| Samplers | GPT logits | Controllable generation |

---

!!! key "Key Takeaways"
    - The autograd engine is the conceptual foundation: every training algorithm in the book is a specialization of "forward pass, then backward pass."
    - BPE tokenization is a frequency-based greedy merge; understanding it explains vocabulary construction choices (byte-fallback, special tokens, length-efficiency trade-offs).
    - Scaled dot-product attention is $O(T^2 d)$ in FLOPs and $O(T^2)$ in memory; FlashAttention preserves the FLOPs but cuts HBM memory to $O(T)$ via tiled online softmax.
    - RoPE encodes relative position as a rotation in complex space; the long-wavelength frequency pairs capture distant relationships while short-wavelength pairs capture local ones.
    - LoRA reduces trainable parameters by a factor of $\min(m,n)/r$ while preserving the pretrained representation; weight tying and zero-initialization of B ensure the adapter starts as an identity mapping.
    - DPO replaces the reward model with a closed-form reparameterization; the implicit reward is exactly $\beta \log(\pi_\theta / \pi_\text{ref})$ — no RL loop needed.
    - GRPO replaces the value network with group-relative advantage normalization; this halves GPU memory per step vs. PPO in the actor-critic formulation.
    - Speculative decoding is provably distribution-preserving: the acceptance-rejection scheme guarantees the output distribution matches exact sampling from the target.
    - Samplers (temperature, top-k, top-p, min-p) are composable logit transforms; understanding their interaction is essential for production quality tuning.

---

## Further Reading

- **Vaswani et al., "Attention Is All You Need," NeurIPS 2017** — the original Transformer paper; defines scaled dot-product attention and multi-head attention.
- **Karpathy, nanoGPT (GitHub)** — the cleanest reference GPT implementation; the GPT section above follows its conventions.
- **Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021** — the RoPE paper with full derivations.
- **Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022** — the original FlashAttention paper; introduces the tiled online softmax algorithm.
- **Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022** — the LoRA paper; Section 4 contains the rank-one analysis justifying initialization choices.
- **Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS 2023** — derivation of the DPO objective from the Bradley-Terry preference model.
- **Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023** — formal proof of distribution preservation and expected acceptance rate analysis.
- **Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020** — the original RAG paper.
- **Holtzman et al., "The Curious Case of Neural Text Degeneration," ICLR 2020** — introduces nucleus (top-p) sampling and documents repetition collapse under greedy/beam decoding.
- **Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units," ACL 2016** — the BPE-for-NLP paper that made the tokenization method mainstream.
