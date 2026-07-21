# 2.2 Embeddings & The Input Pipeline

Every neural network speaks floating-point. Language is discrete — a sequence of symbols selected from a finite vocabulary. The **embedding layer** is the bridge: it translates a token ID (an integer) into a dense vector that the rest of the model can reason about geometrically. Getting this bridge right matters enormously: the embedding matrix is often the single largest parameter tensor in the model, and the way information enters the model shapes everything downstream.

This chapter takes you from raw bytes to the first hidden state. We cover the mathematical structure of the embedding operation, how weight tying connects the input and output ends of the model, what the embedding dimension buys you, and how the entire input pipeline — tokenizer → IDs → embeddings → positional signal → Transformer block — fits together as a tensor-processing graph.

For the upstream step of turning raw text into token IDs, see [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html). For what happens immediately after the embedding layer — adding positional encodings — see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html). The full model that uses these building blocks lives in [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html).

---

## Why Not One-Hot Vectors?

Before we justify what embeddings are, let us feel why the obvious alternative fails. The simplest way to represent a symbol from a vocabulary of size $V$ is a **one-hot vector** $\mathbf{o}_i \in \{0,1\}^V$, which is all zeros except for a $1$ at position $i$.

This representation has two fatal flaws for a neural network:

1. **Dimensionality.** Modern vocabularies have $V \approx 32{,}000$–$256{,}000$ tokens. A batch of $B = 32$ sequences each of length $T = 2048$ would produce a tensor of shape $[32, 2048, 256000]$, a float32 representation occupying $32 \times 2048 \times 256000 \times 4 \approx 67$ GB. That exceeds the HBM of most GPU clusters, just for the input.

2. **No geometry.** One-hot vectors are orthogonal by construction: every pair of tokens has Euclidean distance $\sqrt{2}$ and cosine similarity $0$, regardless of semantic relationship. The model would need to learn from scratch that "king" and "queen" are related, with no inductive bias whatsoever.

The standard solution is to project the one-hot vector through a learned weight matrix:

$$
\mathbf{e}_i = \mathbf{W}_E \,\mathbf{o}_i = \mathbf{W}_E[:, i]
$$

where $\mathbf{W}_E \in \mathbb{R}^{d \times V}$ is the **embedding matrix**. The expression $\mathbf{W}_E[:, i]$ simply extracts column $i$ — it is a cheap table lookup, not a matrix multiply. The result is a dense vector $\mathbf{e}_i \in \mathbb{R}^d$ where $d \ll V$ (typically $d \in \{512, 768, 1024, 2048, 4096, 8192\}$).

We have replaced a $V$-dimensional sparse vector with a $d$-dimensional dense one. The ratio $V/d$ is often $10\times$–$100\times$, so the representation is radically more compact.

{{fig:one-hot-vs-embedding-geometry}}

---

## The Embedding Matrix in Detail

### Shape, Initialization, and Data Type

PyTorch exposes this via `torch.nn.Embedding(num_embeddings, embedding_dim)`. Internally it is just a `Parameter` of shape `[V, d]` — note the transposition convention relative to the formula above; PyTorch stores rows indexed by token ID.

```python
import torch
import torch.nn as nn

# Typical GPT-2 small vocabulary and model dimension
V = 50_257   # GPT-2 vocab size (BPE)
d_model = 768  # embedding dimension

embedding = nn.Embedding(V, d_model)

# The raw weight tensor: shape [V, d_model]
print(embedding.weight.shape)   # torch.Size([50257, 768])

# Memory: float32
bytes_f32 = V * d_model * 4
print(f"Embedding table (fp32): {bytes_f32 / 1e6:.1f} MB")  # ~154 MB

# Memory: bfloat16 (typical training)
bytes_bf16 = V * d_model * 2
print(f"Embedding table (bf16): {bytes_bf16 / 1e6:.1f} MB")  # ~77 MB
```

Initialization matters. `nn.Embedding` defaults to $\mathcal{N}(0, 1)$. In practice, you often want a tighter distribution — GPT-2 uses $\mathcal{N}(0, 0.02^2)$, and Llama uses a similarly small standard deviation. If the embedding vectors start large, they dominate the residual stream and can destabilize layer normalization early in training.

```python
# Recommended initialization used by GPT-2 / nanoGPT
nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
```

### The Lookup as a Matrix Multiply

The lookup operation `embedding(ids)` where `ids` is a `LongTensor` is *mathematically* equivalent to:

$$
E = \mathbf{O} \, \mathbf{W}_E^\top, \quad \mathbf{O} \in \{0,1\}^{T \times V},\; \mathbf{W}_E^\top \in \mathbb{R}^{V \times d}
$$

but it is implemented as an **index select**, not an actual matmul. This is important: doing a real GEMM would be $O(TV d)$ arithmetic, while the lookup is $O(Td)$ — just copying $T$ rows out of a big table. On GPU, this is done with `torch.index_select` or equivalently `embedding.weight[ids]`.

```python
# Directly equivalent to nn.Embedding forward pass
ids = torch.randint(0, V, (4, 16))   # batch=4, seq_len=16

# Method 1: nn.Embedding (recommended; handles padding_idx, sparse gradients)
out1 = embedding(ids)                 # shape: [4, 16, 768]

# Method 2: Direct indexing (identical for inference, slightly faster in some cases)
out2 = embedding.weight[ids]          # shape: [4, 16, 768]

assert torch.allclose(out1, out2)
```

---

## The Unembedding Matrix and Logits

At the other end of the Transformer, after $L$ layers of self-attention and feed-forward computation, we have a sequence of hidden states $\mathbf{H} \in \mathbb{R}^{T \times d}$. To produce a probability distribution over the vocabulary, we need to **project back** to $\mathbb{R}^V$:

$$
\text{logits} = \mathbf{H} \, \mathbf{W}_U^\top, \quad \mathbf{W}_U \in \mathbb{R}^{V \times d}
$$

The resulting tensor has shape $[T, V]$ (or $[B, T, V]$ batched). Passing it through softmax gives a distribution:

$$
p(x_{t+1} = k \mid x_{\le t}) = \frac{\exp(\text{logits}_{t,k})}{\sum_{j=1}^{V} \exp(\text{logits}_{t,j})}
$$

The unembedding step is a **genuine matrix multiply** — $O(TVd)$ — and it is often the most expensive single operation in the model when $V$ is large. For a sequence of $T = 2048$ tokens in a model with $d = 4096$ and $V = 128{,}000$ (Llama 3), the logit projection is a matmul of shape $[2048, 4096] \times [4096, 128000]$, producing $2048 \times 128000 \approx 262$ million values.

### Weight Tying

A key observation: $\mathbf{W}_E$ (the input embedding) and $\mathbf{W}_U$ (the unembedding) both have shape $[V, d]$ and both need to learn "what each token means." It is natural to **share them** — setting $\mathbf{W}_U = \mathbf{W}_E$.

This idea, called **weight tying** (or **tied embeddings**), was popularized by Press & Wolf (2017) in the paper *Using the Output Embedding to Improve Language Models*. It was adopted by GPT-2, BERT, and most subsequent models.

**Why it works:**

- It halves the parameter count in the vocabulary-dependent tensors (saving $V \times d$ parameters — in GPT-2 small, that is about 38M parameters saved, ~30% of the total).
- It couples the representation space of tokens-as-inputs with tokens-as-outputs, creating a consistency pressure: a token's input embedding must be compatible with its appearance as a prediction target.
- Empirically it improves perplexity at equal parameter counts, especially for smaller models.

**Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedTransformerHead(nn.Module):
    """
    Minimal example of weight tying between token embedding and unembedding.
    In a full model, the Transformer layers would sit between embed() and unembed().
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # The single shared weight matrix: shape [vocab_size, d_model]
        self.embed = nn.Embedding(vocab_size, d_model)
        # No separate nn.Linear for the output: we share weights explicitly.

    def token_embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] LongTensor  ->  [B, T, d_model] float"""
        return self.embed(ids)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, d_model]
        returns logits: [B, T, vocab_size]

        F.linear computes  hidden @ weight.T + bias.
        We pass the embedding weight directly, with no bias.
        """
        return F.linear(hidden, self.embed.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """End-to-end: token IDs -> logits (no Transformer layers for brevity)."""
        x = self.token_embed(ids)          # [B, T, d_model]
        # ... Transformer layers would go here ...
        return self.logits(x)              # [B, T, vocab_size]

# Verify parameter count
model = TiedTransformerHead(vocab_size=32_000, d_model=4096)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")   # 32000 * 4096 = 131,072,000

# Verify the weights are truly shared (same object, not a copy)
assert model.embed.weight.data_ptr() == model.embed.weight.data_ptr()

# Gradient flows through both paths during training:
# dL/d(W_E) accumulates gradients from both token_embed() and logits().
```

!!! warning "Common pitfall: copying instead of sharing"
    A frequent mistake is doing `self.unembed = nn.Linear(d_model, vocab_size); self.unembed.weight = self.embed.weight`. This looks right but `nn.Linear` initializes its own weight before you reassign it — you waste memory for a moment and, more dangerously, the `bias` is still separate. Use `F.linear(hidden, self.embed.weight)` (no `nn.Linear` at all) to avoid any ambiguity. Also be careful: if you serialize the model and reload it, ensure the saved checkpoint does not store two copies of the weight.

{{fig:weight-tying-shared-matrix}}

---

## Choosing the Embedding Dimension

The embedding dimension $d$ (also called `d_model`, `hidden_size`, or `n_embd` in various codebases) is the central architectural hyperparameter. It determines:

- The **width** of the residual stream throughout the entire Transformer.
- The number of parameters in every linear layer (attention projections, MLP layers).
- Memory bandwidth costs at inference.

Typical values across model families:

| Model family | $d$ | $V$ | Embed params ($V \times d$) |
|---|---|---|---|
| GPT-2 small | 768 | 50,257 | 38.6 M |
| GPT-2 XL | 1,600 | 50,257 | 80.4 M |
| Llama 2 7B | 4,096 | 32,000 | 131 M |
| Llama 3 8B | 4,096 | 128,000 | 524 M |
| GPT-3 175B | 12,288 | 50,257 | 617 M |
| Llama 3 70B | 8,192 | 128,000 | 1,048 M |

With weight tying, the embed/unembed table is counted once. Without weight tying (some encoder-decoder models like T5 do not tie weights), double those numbers.

**Scaling law intuition.** The Kaplan et al. (2020) and Chinchilla (Hoffmann et al., 2022) scaling laws treat $d$ as one axis of model capacity. In practice, the ratio of $d$ to the number of layers and the ratio of $d$ to the number of attention heads are constrained (typically $d_{\text{head}} = d / n_{\text{heads}} = 64$ or $128$), so $d$ scales roughly as the square root of the parameter count. See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for the full picture.

---

## End-to-End: Text Becomes Tensors

Let us trace a concrete example from raw string to the first Transformer block input.


{{fig:embed-text-to-tensors-pipeline}}


The entire pipeline is differentiable end-to-end. The only non-differentiable step is the tokenizer itself — it is a deterministic rule-based lookup, not a learned function (though soft approaches like SentencePiece with straight-through gradients have been explored).

Here is the corresponding runnable code:

```python
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Minimal end-to-end input pipeline for a decoder-only LM
# ------------------------------------------------------------------

class InputPipeline(nn.Module):
    """
    Converts a batch of token ID sequences into the first hidden state
    that will be fed to a Transformer block.

    Args:
        vocab_size:  Size of the token vocabulary (V).
        d_model:     Embedding dimension (d).
        max_seq_len: Maximum sequence length supported.
        dropout:     Dropout applied to the summed embedding (regularization).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding table: shape [V, d_model]
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding: shape [max_seq_len, d_model]
        # (Many modern models use RoPE instead; see Chapter 2.5)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.drop = nn.Dropout(dropout)

        # Initialize weights following GPT-2 conventions
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        """
        ids: [B, T] integer tensor of token IDs (0 <= ids[i,j] < vocab_size)
        returns: [B, T, d_model] float tensor — the first hidden state
        """
        B, T = ids.shape
        device = ids.device

        # --- Token embeddings ---
        # Each integer is replaced by its row in the embedding table.
        tok_emb = self.token_embed(ids)           # [B, T, d_model]

        # --- Positional embeddings ---
        # Build position indices [0, 1, ..., T-1] and look them up.
        pos = torch.arange(T, device=device)      # [T]
        pos_emb = self.pos_embed(pos)             # [T, d_model]
        # Broadcast: [B, T, d_model] + [T, d_model] -> [B, T, d_model]

        # --- Sum and apply dropout ---
        x = self.drop(tok_emb + pos_emb)          # [B, T, d_model]
        return x


# Quick sanity check
pipeline = InputPipeline(vocab_size=50_257, d_model=768, max_seq_len=1024)
ids = torch.randint(0, 50_257, (4, 512))          # batch=4, seq_len=512
x = pipeline(ids)
print(x.shape)   # torch.Size([4, 512, 768])
print(f"Output norm (mean token): {x.norm(dim=-1).mean().item():.3f}")
```

---

## Numerical Worked Example: Memory and FLOPs

!!! example "Worked example: Embedding layer sizing for Llama 3 8B"
    Llama 3 uses $V = 128{,}000$, $d = 4{,}096$, and weight tying.

    **Parameter count (embedding table):**
    $$
    V \times d = 128{,}000 \times 4{,}096 = 524{,}288{,}000 \approx 524\text{ M parameters}
    $$
    At bfloat16 (2 bytes/param), this is $524 \times 10^6 \times 2 = 1{,}048$ MB $\approx 1.02$ GB.

    **Logit projection FLOPs (per token, per forward pass):**
    The unembedding step computes $\mathbf{h} \cdot \mathbf{W}_U^\top$ where $\mathbf{h} \in \mathbb{R}^{4096}$ and $\mathbf{W}_U \in \mathbb{R}^{128000 \times 4096}$.
    Floating-point multiply-adds: $4{,}096 \times 128{,}000 = 524{,}288{,}000 \approx 0.5$ GFLOPs per token.

    For a batch of $B = 1$ with $T = 2048$ tokens:
    $$
    0.5 \times 2048 \approx 1\,024 \text{ GFLOPs}
    $$
    for the logit projection alone. This is non-trivial — it is comparable to multiple attention layers.

    **Memory bandwidth cost during decode (single token):**
    During autoregressive decode, we generate one token at a time. The logit projection reads all $524$ M parameters from HBM to compute a single $128{,}000$-dimensional vector. At 900 GB/s (an A100), reading $\approx 1$ GB takes $\approx 1.1$ ms — a significant fraction of per-token latency for smaller models. This is why some inference systems use lower-precision or approximate logit computation.

---

## Gradient Flow Through the Embedding Layer

The embedding operation is differentiable in the sense that autograd knows how to compute $\partial \mathcal{L} / \partial \mathbf{W}_E$. However, because the forward pass is an index select (not a dense matmul), the gradient of $\mathbf{W}_E$ is **sparse**: only the rows corresponding to tokens that appeared in the current batch receive nonzero gradients.

```python
import torch
import torch.nn as nn

embed = nn.Embedding(50_257, 768)
ids = torch.tensor([[464, 3797, 3332, 319, 262, 2603, 13]])  # one sentence

x = embed(ids)             # [1, 7, 768]
loss = x.sum()             # toy loss
loss.backward()

grad = embed.weight.grad   # shape: [50257, 768]
# Most rows are exactly zero — only rows [464, 3797, 3332, 319, 262, 2603, 13]
# have nonzero gradients.
nonzero_rows = (grad.abs().sum(dim=1) > 0).sum().item()
print(f"Rows with nonzero gradient: {nonzero_rows} / 50257")
# Output: 7 / 50257   (one per unique token in the batch)
```

This sparsity has practical consequences:

- **Adam/AdaGrad/Adam-W** maintain per-parameter second-moment accumulators. For the embedding table, most accumulators are never updated, which is wasteful in memory and can skew the effective learning rate.
- **Sparse gradient optimizers** (e.g., `torch.optim.SparseAdam`) only update the rows that received gradients, reducing memory writes. In practice, modern codebases simply use dense AdamW and accept the overhead, because sparse updates complicate distributed training and sharding.
- During **gradient accumulation** with many micro-batches, the probability that a row is "hot" increases, amortizing this effect.

### Gradient Scaling and the Embedding Norm

With weight tying, gradients accumulate from *both* the forward token lookup and the backward logit projection:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_E} = \underbrace{\frac{\partial \mathcal{L}}{\partial \mathbf{e}_i}}_{\text{from embed}} + \underbrace{\mathbf{H}^\top \frac{\partial \mathcal{L}}{\partial \text{logits}}}_{\text{from unembed}}
$$

The unembed gradient is dense (it involves all $T$ hidden states hitting all $V$ rows), while the embed gradient is sparse. This means the gradient magnitude for frequently used tokens will be dominated by the unembed path, while rare tokens only receive signal through the embed path — a subtle asymmetry that the optimizer must handle.

---

## Special Tokens and Padding

Real models reserve slots in the vocabulary for special-purpose tokens. For example:

| Token | GPT-2 ID | Llama 3 ID | Purpose |
|---|---|---|---|
| `<\|endoftext\|>` | 50256 | — | end-of-sequence / pad |
| `<s>` (BOS) | — | 128000 | begin-of-sequence |
| `</s>` (EOS) | — | 128001 | end-of-sequence |
| `<\|pad\|>` | — | 128004 | padding (variable-length batches) |

These tokens have embedding vectors like any other token. The critical implementation detail is the `padding_idx` argument to `nn.Embedding`:

```python
# padding_idx ensures the pad token's embedding stays zero and receives
# no gradient — crucial for variable-length batched sequences.
embed = nn.Embedding(
    num_embeddings=128_256,
    embedding_dim=4096,
    padding_idx=128004,  # <|pad|> token ID
)

# The pad embedding is initialized to zero and frozen there.
print(embed.weight[128004].norm())   # tensor(0.)

# During backward, gradient for padding_idx is zeroed out automatically.
```

In practice, most modern LLM training uses **sequence packing** to avoid padding entirely — multiple documents are concatenated into a single sequence, separated by EOS tokens, and an attention mask prevents cross-document attention. See [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) for how this is implemented in training pipelines.

---

## The Full Embedding Stack: Layer Norm, Scale, and RoPE

In a modern production model, the embedding layer is usually followed by additional transformations before the first attention layer:


{{fig:embed-full-stack-architecture}}


Different architecture families make different choices:

- **GPT-2** (Radford et al., 2019): learned positional embeddings, added to token embeddings, then dropout. Pre-layer-norm.
- **BERT** (Devlin et al., 2018): token + positional + segment type embeddings, then layer norm, then dropout. Post-layer-norm.
- **Llama / Mistral / Gemma**: no positional addition at input; RoPE is applied inside each attention layer. The embedding is just `W_E[ids]` followed directly by the first RMSNorm inside the block.

Here is the Llama-style input pipeline:

```python
import torch
import torch.nn as nn

class LlamaInputPipeline(nn.Module):
    """
    Llama-3 style input pipeline:
      - token_embed only (no positional embedding added here)
      - RoPE is applied inside attention layers (not shown here)
      - RMSNorm is the first op inside each Transformer block (not here)
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Simple lookup; no padding_idx by default in Llama
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        # Initialization: std scales with d_model
        nn.init.normal_(self.embed_tokens.weight, std=d_model ** -0.5)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """input_ids: [B, T]  ->  hidden_states: [B, T, d_model]"""
        return self.embed_tokens(input_ids)
```

For details on the RoPE step that follows, see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).

---

## Residual Stream Perspective

A key conceptual framing (popularized by Anthropic's mechanistic interpretability work) is to think of the embedding layer as **writing the initial state** of the *residual stream*. Every subsequent layer reads from this stream and adds its contribution. The final hidden state that is fed to the unembedding is the accumulated sum:

$$
\mathbf{h}_T = \mathbf{e} + \sum_{\ell=1}^{L} \Delta_\ell
$$

where $\mathbf{e}$ is the initial embedding and each $\Delta_\ell$ is the output of block $\ell$. The unembedding then projects this summed representation.

This perspective has practical implications:

- The initial embedding magnitude sets the scale of the residual stream. If $\|\mathbf{e}\|$ is too large relative to $\|\Delta_\ell\|$, early layers will struggle to modify the signal.
- Weight tying creates a direct algebraic link: a token's embedding vector is the same direction the logit head uses to "vote" for that token. This means tokens that appear in similar contexts (and thus have similar embeddings) will also be predicted with similar probabilities — a desirable regularizer.
- Tokens that are close in embedding space (low cosine distance) will interfere with each other's logit scores. This is why **tokenizer vocabulary size matters**: too small a vocabulary means too many distinct meanings packed into the same token embedding; too large means rare tokens get almost no gradient signal.

!!! note "The rank of the embedding matrix"
    For very large vocabularies ($V \gg d$), the full-rank embedding $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ can only represent $V$ points in a $d$-dimensional space. The geometry is constrained: with $V = 128{,}000$ and $d = 4{,}096$, we have $128{,}000$ points in a $4{,}096$-dimensional space — sparse, but not degenerate. Models with very small $d$ relative to $V$ (early word2vec-style models used $d = 300$ with $V = 10^6$) are more severely constrained.

---

## Interview Corner

!!! interview "Interview Corner"
    **Q:** Why do we tie the input embedding and the output unembedding weights in language models? What are the tradeoffs?

    **A:** Weight tying (proposed by Press & Wolf, 2017) shares the token embedding matrix $\mathbf{W}_E$ between the lookup at the input and the logit projection at the output. The motivations are: (1) it cuts the vocabulary-dependent parameter count roughly in half — for a model with $V = 128{,}000$ and $d = 4{,}096$ that saves about 524 M parameters; (2) it regularizes training by enforcing that a token's input representation is geometrically aligned with the direction the output head uses to predict that same token; (3) it generally improves perplexity at equal model size, especially for smaller models.

    The main tradeoff is that it couples two computations that might benefit from different representations — the input embedding is the start of computation, while the output head reads the fully processed hidden state. For very large models, this coupling becomes less harmful because the model has enough capacity to compensate. Some encoder-decoder models (e.g., early T5 variants) do not tie weights, giving the encoder embedding and decoder LM head full independence.

    An interviewer follow-up: "Where does the gradient go?" — During backprop, $\mathbf{W}_E$ receives a sparse gradient from the input lookup (only the rows corresponding to tokens in the current batch) plus a dense gradient from the logit projection (all rows, weighted by the hidden states). The dense unembed gradient dominates for frequent tokens.

---

## Key Takeaways

!!! key "Key Takeaways"
    - The embedding layer is a learned lookup table $\mathbf{W}_E \in \mathbb{R}^{V \times d}$. Mathematically it is $\mathbf{W}_E \mathbf{o}_i$, but implemented as an index select for $O(Td)$ cost rather than an $O(TVd)$ matmul.
    - The unembedding (logit projection) is a genuine dense matmul $\mathbf{H} \mathbf{W}_U^\top$ of shape $[T, d] \times [d, V]$ and is often one of the most expensive operations at inference time.
    - Weight tying ($\mathbf{W}_U = \mathbf{W}_E$) halves the vocabulary parameter count, regularizes the model, and typically improves perplexity — it is the default in most modern decoder-only LLMs.
    - Embedding gradients are sparse (only touched tokens get nonzero gradient); the unembed gradient is dense. This asymmetry matters for optimizer state memory and effective learning rate.
    - The embedding dimension $d$ is the width of the entire residual stream and must be chosen jointly with the number of layers, heads, and MLP expansion factor.
    - Modern models (Llama, Mistral) do not add positional encodings at the embedding stage — RoPE is applied inside each attention layer, making the input pipeline a single table lookup.
    - Special tokens (`<pad>`, `<bos>`, `<eos>`) occupy regular vocabulary slots; using `padding_idx` in `nn.Embedding` ensures the pad token embedding stays at zero with no gradient.
    - Sequence packing eliminates padding entirely in training by concatenating multiple documents with EOS separators and using attention masks to prevent cross-document attention leakage.

---

!!! sota "State of the Art & Resources (2026)"
    Token embeddings remain the gateway from discrete symbols to the continuous geometry of neural computation. Modern research is extending the embedding layer itself as a scaling axis — growing vocabularies, adding n-gram embeddings, and distributing embedding parameters across layers — while decoder-only LLMs have emerged as dominant embedding models for retrieval and similarity tasks.

    **Foundational work**

    - [Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)](https://arxiv.org/abs/1301.3781) — the original word2vec paper that established dense learned embeddings as the standard input representation.
    - [Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013)](https://arxiv.org/abs/1310.4546) — Skip-gram with negative sampling; the formulation still underpins embedding intuitions taught today.
    - [Press & Wolf, *Using the Output Embedding to Improve Language Models* (2017)](https://arxiv.org/abs/1608.05859) — the paper that popularized weight tying between input embedding and output unembedding, now the default in most LLMs.
    - [Kaplan et al., *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361) — establishes how embedding dimension $d$ scales with total parameter count and compute budget.

    **Recent advances (2023–2026)**

    - [Yu et al., *Scaling Embedding Layers in Language Models* (2025)](https://arxiv.org/abs/2502.01637) — SCONE adds frequent n-gram embeddings off-accelerator so a 1B model matches a 1.9B baseline at half the compute.
    - [Tao et al., *LLMs are Also Effective Embedding Models: An In-depth Overview* (2024)](https://arxiv.org/abs/2412.12591) — surveys how decoder-only LLMs (GPT, LLaMA) now outperform BERT-style encoders for retrieval and semantic similarity.

    **Open-source & tools**

    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — ~300-line reference for weight tying, embedding initialization, and the full token-ID-to-logit pipeline.
    - [huggingface/tokenizers](https://huggingface.co/docs/tokenizers/index) — official docs for the Rust-backed tokenizer library that feeds token IDs into embedding layers in production systems.

    **Go deeper**

    - [Elhage et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021)](https://transformer-circuits.pub/2021/framework/index.html) — introduces the residual-stream framing and analyzes how embedding and unembedding matrices interact via linear decomposition.
    - [Jay Alammar, *The Illustrated Word2vec* (2019)](https://jalammar.github.io/illustrated-word2vec/) — the best visual explainer for why dense embeddings encode semantic geometry and how training shapes the embedding space.

## Further Reading

- **Press & Wolf** — *Using the Output Embedding to Improve Language Models* (2017). The paper that popularized weight tying and provided the theoretical and empirical justification.
- **Mikolov et al.** — *Distributed Representations of Words and Phrases and their Compositionality* (2013). Word2Vec; foundational work on learning dense word representations.
- **Devlin et al.** — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018). Segment and positional embedding design for encoder models.
- **Radford et al.** — *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019). Source of the GPT-2 learned positional embedding and initialization conventions widely copied since.
- **Touvron et al.** — *Llama 2: Open Foundation and Fine-Tuned Chat Models* (2023). Shows the minimal embedding design (no positional addition; RoPE in attention) used by the modern Llama family.
- **Elhage et al.** — *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021). Introduces the residual stream framing and analyzes how embedding and unembedding matrices interact.
- **nanoGPT** — Andrej Karpathy's minimal GPT implementation on GitHub. The `model.py` file is an excellent reference for weight tying, embedding initialization, and the full input-to-logit pipeline in under 300 lines of PyTorch.
