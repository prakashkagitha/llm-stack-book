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

This idea, called **weight tying** (or **tied embeddings**), was popularized by Press & Wolf (2017) in the paper *Using the Output Embedding to Improve Language Models*. It was adopted by GPT-2 and BERT and is standard in small models today.

**Why it works:**

- It halves the parameter count in the vocabulary-dependent tensors (saving $V \times d$ parameters — in GPT-2 small, that is about 38M parameters saved, ~30% of the total).
- It couples the representation space of tokens-as-inputs with tokens-as-outputs, creating a consistency pressure: a token's input embedding must be compatible with its appearance as a prediction target.
- Empirically it improves perplexity at equal parameter counts, especially for smaller models.

**Who actually ties, as of 2026.** Tying is *not* universal — it is a scale-dependent decision, and the deciding quantity is the fraction $Vd/N$ of total parameters $N$ that the table consumes. Small models tie: Gemma 2 and Gemma 3 tie at every size, and the sub-4B members of the Qwen 2.5 / Qwen 3, Llama 3.2 (1B, 3B), and SmolLM families ship with `tie_word_embeddings: true`. Larger models generally untie: Llama 2, Llama 3 8B/70B, and the 7B-and-up Qwen checkpoints all set `tie_word_embeddings: false`. The logic is straightforward — at 1B parameters with $V = 128{,}000$ the table is a third of the model and tying is nearly free capacity, while at 70B it is under 2% and the extra freedom for the output head to specialize is worth the parameters. Because Stack-100M spends ~17% of its budget on the table, our capstone ties; see [The Stack-100M Architecture](../14-capstone/04-architecture.html) for the full parameter accounting and [A Byte-Level BPE Tokenizer From Scratch](../14-capstone/03-tokenizer.html) for why we pick $V = 32{,}768$ in the first place.

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

# Verify there is exactly ONE [V, d] tensor: the parameter list has a single
# entry, and a second (untied) head would have doubled `total` above.
assert total == 32_000 * 4096
assert len(list(model.parameters())) == 1

# Gradient flows through both paths during training:
# dL/d(W_E) accumulates gradients from both token_embed() and logits().
```

In HuggingFace `transformers` you never write this by hand: every `*ForCausalLM` reads the `tie_word_embeddings` boolean from its config and calls `model.tie_weights()` at the end of `from_pretrained`, which rebinds `lm_head.weight` to `model.embed_tokens.weight`. Two consequences bite people in practice. First, a tied checkpoint saved with **safetensors** contains only `model.embed_tokens.weight` — the format refuses to serialize two names pointing at the same storage, so `lm_head.weight` is simply absent from the file and is re-created by `tie_weights()` on load. Loading such a state dict with `strict=True` into a hand-rolled model will fail on a "missing key" that is not actually missing. Second, if you mutate the config to `tie_word_embeddings=False` after loading, you must explicitly clone the tensor (`model.lm_head.weight = nn.Parameter(model.get_input_embeddings().weight.clone())`); otherwise the two remain aliased and "untying" silently does nothing.

!!! warning "Common pitfall: copying instead of sharing"
    A frequent mistake is doing `self.unembed = nn.Linear(d_model, vocab_size); self.unembed.weight = self.embed.weight`. This looks right but `nn.Linear` initializes its own weight before you reassign it — you waste memory for a moment and, more dangerously, the `bias` is still separate. Use `F.linear(hidden, self.embed.weight)` (no `nn.Linear` at all) to avoid any ambiguity. Also be careful: if you serialize the model and reload it, ensure the saved checkpoint does not store two copies of the weight.

{{fig:weight-tying-shared-matrix}}

### The Logit Tensor Is the Training Memory Bottleneck

Here is a fact that surprises almost everyone the first time they train a small model: the largest single tensor in a training step is usually not a weight, an activation, or the KV cache — it is the logits. The unembedding turns a `[N, d]` hidden state into `[N, V]`, and with $V \gg d$ that is an expansion by a factor of $V/d$.

Make it concrete for our capstone model, `Stack-100M` ($d = 512$, $V = 32{,}768$), training with a micro-batch of 16 sequences of 2048 tokens, i.e. $N = 32{,}768$ tokens:

| Tensor | Shape | dtype | Bytes |
|---|---|---|---|
| Final hidden state | $[32768, 512]$ | bf16 | 34 MB |
| Logits (bf16 matmul output) | $[32768, 32768]$ | bf16 | 2.15 GB |
| Logits upcast to fp32 for the softmax | $[32768, 32768]$ | fp32 | 4.29 GB |
| Gradient w.r.t. logits | $[32768, 32768]$ | fp32 | 4.29 GB |

That is roughly **10.7 GB** for the loss computation alone — about eight times the model's entire trainable state (101M params in bf16 plus AdamW moments). The fp32 upcast is not optional: computing `log_softmax` over 32k classes in bf16 loses several bits of the loss signal, so every serious implementation upcasts.

The fix is to never materialize the full `[N, V]` tensor. Split the tokens into chunks, compute the projection *and* the cross-entropy per chunk, and discard each chunk's logits before moving on — recomputing them in the backward pass. This is a plain application of activation checkpointing to the loss head:

```python
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def chunked_linear_ce(hidden, weight, targets, chunk=4096, ignore_index=-100):
    """
    Memory-efficient (unembedding + cross-entropy) fused into one op.

    hidden:  [N, d]  flattened final hidden states (N = B*T)
    weight:  [V, d]  the (possibly tied) unembedding matrix
    targets: [N]     int64 next-token IDs, `ignore_index` where masked

    Peak logit memory is [chunk, V] instead of [N, V]: the chunk's logits are
    freed right after its scalar loss is computed, then recomputed in backward.
    Mathematically identical to F.cross_entropy on the full logit tensor.
    """
    n_valid = (targets != ignore_index).sum().clamp(min=1)

    def chunk_loss(h, t):
        # The [chunk, V] tensor exists only inside this function.
        logits = F.linear(h, weight).float()      # fp32 softmax for stability
        return F.cross_entropy(logits, t, ignore_index=ignore_index,
                               reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.shape[0], chunk):
        total = total + checkpoint(
            chunk_loss, hidden[i:i + chunk], targets[i:i + chunk],
            use_reentrant=False,
        )
    return total / n_valid


# --- equivalence check against the naive full-logit path ---
torch.manual_seed(0)
N, d, V = 4096, 128, 2048
W = (torch.randn(V, d) * 0.02).requires_grad_(True)
h = torch.randn(N, d, requires_grad=True)
tgt = torch.randint(0, V, (N,))

loss_chunked = chunked_linear_ce(h, W, tgt, chunk=512)
loss_naive = F.cross_entropy(F.linear(h, W).float(), tgt)
assert torch.allclose(loss_chunked, loss_naive, atol=1e-5)

loss_chunked.backward()
g_chunked = h.grad.clone(); h.grad = None
loss_naive.backward()
assert torch.allclose(g_chunked, h.grad, atol=1e-5)
print("chunked linear-CE matches the naive path in value and gradient")
```

In production you do not hand-roll this. Two open-source implementations push the same idea into a single fused kernel that never writes the logits to HBM at all:

- **[linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)** — a library of Triton kernels for LLM training whose flagship op is `LigerFusedLinearCrossEntropyLoss`, which fuses the `lm_head` matmul, the log-softmax, and the loss into one kernel with chunked recomputation. It is wired into the HuggingFace `Trainer` behind a single flag: `TrainingArguments(..., use_liger_kernel=True)`, and TRL, Axolotl, and Unsloth all expose it.
- **[apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy)** — "Cut Cross-Entropy" (Wijmans et al., 2024), which goes further: it computes only the logit of the *correct* token densely and evaluates the log-sum-exp denominator in on-chip SRAM, reducing logit memory for the loss to a small constant.

Both give bit-comparable losses and are among the highest-leverage single-line changes available when a training run OOMs. See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html) for the general technique and [The Pretraining Run: A Complete Single-GPU Training Loop](../14-capstone/07-pretraining-run.html) for how this choice sets the micro-batch size of the capstone run.

### Sharding the Vocabulary Across GPUs

When $V \times d$ no longer fits comfortably on one device — or, more often, when the `[N, V]` logits do not — the vocabulary axis is sharded. Megatron-LM's `VocabParallelEmbedding` splits the table row-wise across the tensor-parallel group: rank $r$ owns token IDs $[r \cdot V/P, (r+1) \cdot V/P)$, masks out IDs it does not own, looks up the rest, and an `all-reduce` over the group sums the partial results into the full embedding. The output side mirrors this: `ColumnParallelLinear` for the LM head gives each rank a $[V/P, d]$ slice of the logits, and `vocab_parallel_cross_entropy` computes the loss without ever gathering the full vocabulary — each rank contributes its local max and local sum-of-exponentials to two small all-reduces. This is why Megatron requires `vocab_size` to be padded to a multiple of the tensor-parallel size times 128, and it is the reason real vocab sizes are numbers like 32,768, 128,256, or 50,304 rather than round decimals. See [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) and [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

---

## Choosing the Embedding Dimension

The embedding dimension $d$ (also called `d_model`, `hidden_size`, or `n_embd` in various codebases) is the central architectural hyperparameter. It determines:

- The **width** of the residual stream throughout the entire Transformer.
- The number of parameters in every linear layer (attention projections, MLP layers).
- Memory bandwidth costs at inference.

Typical values across model families:

| Model family | $d$ | $V$ | Table params ($V \times d$) | Tied? |
|---|---|---|---|---|
| GPT-2 small | 768 | 50,257 | 38.6 M | yes |
| GPT-2 XL | 1,600 | 50,257 | 80.4 M | yes |
| Llama 2 7B | 4,096 | 32,000 | 131 M | no |
| Llama 3.2 1B | 2,048 | 128,256 | 263 M | yes |
| Llama 3 8B | 4,096 | 128,256 | 525 M | no |
| GPT-3 175B | 12,288 | 50,257 | 618 M | — |
| Llama 3 70B | 8,192 | 128,256 | 1,051 M | no |

Read the last column carefully: a *tied* model pays for this table once, an *untied* model pays twice. Llama 3 8B therefore spends about 1.05 B of its ~8 B parameters — roughly 13% — on the input embedding plus a separate LM head, while Llama 3.2 1B spends 263 M of ~1.24 B (21%) on a single shared table. Encoder-decoder models such as T5 also keep the decoder LM head separate from the shared token embedding.

**Scaling law intuition.** The Kaplan et al. (2020) and Chinchilla (Hoffmann et al., 2022) scaling laws treat $d$ as one axis of model capacity. In practice, the ratio of $d$ to the number of layers and the ratio of $d$ to the number of attention heads are constrained (typically $d_{\text{head}} = d / n_{\text{heads}} = 64$ or $128$), so $d$ scales roughly as the square root of the parameter count. See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for the full picture.

---

## End-to-End: Text Becomes Tensors

Let us trace a concrete example from raw string to the first Transformer block input.


{{fig:embed-text-to-tensors-pipeline}}


The entire pipeline is differentiable end-to-end. The only non-differentiable step is the tokenizer itself — it is a deterministic rule-based lookup, not a learned function (though soft approaches like SentencePiece with straight-through gradients have been explored).

The first hop — string to integers — is handled by the `tokenizers` / `transformers` stack, and it is worth seeing the seam explicitly rather than starting from a `randint`:

```python
import torch
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("gpt2")     # Rust-backed fast BPE tokenizer
model = AutoModel.from_pretrained("gpt2")

# GPT-2 ships no pad token -- a classic first-run crash. Reuse EOS as pad and
# rely on attention_mask (and label masking) to exclude those positions.
tok.pad_token = tok.eos_token

batch = tok(["The cat sat on the mat.", "Hello world"],
            return_tensors="pt", padding=True)
print(batch["input_ids"])        # int64 [2, T]; the tokenizer's only output
print(batch["attention_mask"])   # 1 for real tokens, 0 for padding

# `wte` ("word token embedding") IS the nn.Embedding of shape [50257, 768].
wte = model.get_input_embeddings()
assert isinstance(wte, torch.nn.Embedding)
assert wte.weight.shape == (50257, 768)

x = wte(batch["input_ids"])      # [2, T, 768] -- the first hidden state
print(x.shape, x.dtype)
```

Note what the tokenizer does *not* produce: no floats, no positions, no mask over the vocabulary. It emits `input_ids` (and, when padding, an `attention_mask`), and every remaining transformation is the model's. `get_input_embeddings()` is the portable accessor across all HF architectures — `wte` in GPT-2, `embed_tokens` in Llama/Qwen/Gemma — and it is the handle you use for resizing, freezing, or inspecting the table. For how `input_ids` are produced in the first place, see [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html).

Here is the corresponding runnable code for the from-scratch path:

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
    Llama 3 8B uses $V = 128{,}256$ (we round to $128{,}000$ below to keep the arithmetic clean) and $d = 4{,}096$. Note that Llama 3 8B does **not** tie: it carries a separate LM head, so the model pays the figures below *twice* — once for `embed_tokens`, once for `lm_head`.

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

{{fig:embed-vs-unembed-asymmetry}}

---

## Special Tokens and Padding

Real models reserve slots in the vocabulary for special-purpose tokens. For example:

| Token | Model | ID | Purpose |
|---|---|---|---|
| `<\|endoftext\|>` | GPT-2 | 50256 | document separator / EOS / pad |
| `<s>`, `</s>` | Llama 2 (SentencePiece) | 1, 2 | BOS, EOS |
| `<\|begin_of_text\|>` | Llama 3 | 128000 | begin-of-sequence |
| `<\|end_of_text\|>` | Llama 3 | 128001 | end of a pretraining document |
| `<\|finetune_right_pad_id\|>` | Llama 3.1+ | 128004 | right padding for fine-tuning |
| `<\|eot_id\|>` | Llama 3 Instruct | 128009 | end of a chat turn |

Note the pattern: Llama 3's vocabulary is 128,000 learned BPE merges plus 256 reserved special-token slots, giving `vocab_size = 128256`. Reserving spare slots up front is deliberate — it lets a fine-tuner introduce new control tokens later without resizing the table at all. These tokens have embedding vectors like any other token. The critical implementation detail is the `padding_idx` argument to `nn.Embedding`:

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

One subtlety with tying: `padding_idx` pins that row to zero, and because the row is shared with the LM head, the pad token's logit becomes exactly $0$ at every position — a constant, not a learned score. That is harmless (the softmax simply treats it as a fixed-scoring class) but it means you cannot rely on the model learning to *avoid* emitting pad. The real mechanism for excluding padding from the objective is label masking: set the label to `-100` at pad positions so `F.cross_entropy(..., ignore_index=-100)` drops them. Masking labels is mandatory; `padding_idx` is a convenience.

In practice, most modern LLM training uses **sequence packing** to avoid padding entirely — multiple documents are concatenated into a single sequence, separated by EOS tokens, and an attention mask prevents cross-document attention. See [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) for how this is implemented in training pipelines.

### Adding Tokens After Pretraining: Resizing the Table

You will hit this the first time you take a base model to SFT. Your chat template needs control tokens (`<|im_start|>`, `<|im_end|>`, a tool-call delimiter) that the pretrained tokenizer does not have. Adding them to the tokenizer grows $V$, so both the embedding table and the LM head must grow to match — and the new rows need sensible values.

The naive move is to let HuggingFace default-initialize the new rows from $\mathcal{N}(0, \sigma^2)$ with the config's `initializer_range`. That is a real footgun: a randomly initialized row sits at an arbitrary direction in a residual stream whose learned embeddings have long since settled into a particular norm and mean. Because the head is usually tied, that random row is also a random *logit direction*, which can produce large spurious logits for the new token and a loss spike in the first few hundred steps. The standard fix is to initialize each new row to the **mean of the existing rows** (equivalently, a small perturbation around it), which places the new token at the center of the learned distribution where it is maximally neutral:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-135M"          # any small base model
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

new_tokens = ["<|im_start|>", "<|im_end|>"]
n_added = tok.add_special_tokens({"additional_special_tokens": new_tokens})

old_V = model.get_input_embeddings().weight.shape[0]

# pad_to_multiple_of keeps the vocab dimension tensor-core / TP friendly.
model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)

with torch.no_grad():
    inp = model.get_input_embeddings().weight
    mean_row = inp[:old_V].mean(dim=0, keepdim=True)
    inp[old_V:] = mean_row                        # neutral init for new rows

    # HF returns the lm_head module even when tied, so compare storage:
    # if it aliases the input table, the assignment above already covered it.
    out = model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != inp.data_ptr():
        out.weight[old_V:] = out.weight[:old_V].mean(dim=0, keepdim=True)

print(f"added {n_added} tokens; vocab {old_V} -> {inp.shape[0]}")
```

Three things to check every time. (1) `resize_token_embeddings` handles the tied head automatically *only* if `config.tie_word_embeddings` is true — otherwise you must initialize the head's new rows yourself, as above. (2) Pass `pad_to_multiple_of` (64 or 128) so the new $V$ stays a friendly multiple; an odd vocabulary size costs measurable throughput in the `lm_head` GEMM and breaks Megatron's tensor-parallel divisibility requirement. (3) The tokenizer and the model must be saved and reloaded *together* — a resized model with the original tokenizer will silently emit garbage. The capstone applies exactly this procedure when it introduces chat and tool-call tokens before SFT; see [Post-Training: SFT, DPO, and Narrow RLVR (GRPO)](../14-capstone/09-post-training.html).

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
        # Llama's HF configs use a flat `initializer_range = 0.02`, inherited
        # from GPT-2 -- NOT a width-dependent rule. (Some research codebases
        # prefer std = d_model ** -0.5; the two happen to agree near d ~ 2500,
        # and the muP-style width scaling is discussed in Chapter 3.10.)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)

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

    The main tradeoff is that it couples two computations that might benefit from different representations — the input embedding is the start of computation, while the output head reads the fully processed hidden state. That is why tying is scale-dependent in practice: below ~4B parameters the table is 15–30% of the model and tying is nearly free capacity (Gemma, Llama 3.2 1B/3B, small Qwen, SmolLM all tie), while at 8B and above the saving is a small fraction of the model and the field generally unties (Llama 3 8B/70B, large Qwen). Encoder-decoder models such as T5 likewise keep the decoder LM head independent of the shared token embedding.

    An interviewer follow-up: "Where does the gradient go?" — During backprop, $\mathbf{W}_E$ receives a sparse gradient from the input lookup (only the rows corresponding to tokens in the current batch) plus a dense gradient from the logit projection (all rows, weighted by the hidden states). The dense unembed gradient dominates for frequent tokens.

    A second follow-up that separates candidates: "Tying halves the *parameters* — does it halve the *memory* of a training step?" — No, and not even close. The dominant term is not the weights but the `[N, V]` logit tensor and its fp32 gradient, which tying does not touch at all. For a 100M-parameter model with $V = 32{,}768$ and a 32k-token micro-batch that is ~10 GB, several times the model plus optimizer state. The lever there is a fused/chunked linear-cross-entropy (Liger-Kernel, Cut Cross-Entropy), not tying.

---

## Key Takeaways

!!! key "Key Takeaways"
    - The embedding layer is a learned lookup table $\mathbf{W}_E \in \mathbb{R}^{V \times d}$. Mathematically it is $\mathbf{W}_E \mathbf{o}_i$, but implemented as an index select for $O(Td)$ cost rather than an $O(TVd)$ matmul.
    - The unembedding (logit projection) is a genuine dense matmul $\mathbf{H} \mathbf{W}_U^\top$ of shape $[T, d] \times [d, V]$ and is often one of the most expensive operations at inference time.
    - Weight tying ($\mathbf{W}_U = \mathbf{W}_E$) halves the vocabulary parameter count and typically improves perplexity at small scale. As of 2026 it is standard *below* roughly 4B parameters (Gemma, Llama 3.2 1B/3B, small Qwen, SmolLM) and usually dropped above it (Llama 3 8B/70B, large Qwen).
    - The `[N, V]` logit tensor — not any weight — is usually the largest single tensor in a training step. Fusing the LM head with the cross-entropy and recomputing logits chunk-by-chunk (hand-rolled with `torch.utils.checkpoint`, or via Liger-Kernel / Cut Cross-Entropy) removes it almost entirely.
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

    - [Yu et al., *Scaling Embedding Layers in Language Models* (2025)](https://arxiv.org/abs/2502.01637) — SCONE adds frequent n-gram embeddings off-accelerator so a 1B model outperforms a 1.9B baseline at roughly half the inference FLOPs and accelerator memory (NeurIPS 2025).
    - [Tao et al., *LLMs are Also Effective Embedding Models: An In-depth Overview* (2024)](https://arxiv.org/abs/2412.12591) — surveys how decoder-only LLMs (GPT, LLaMA) now outperform BERT-style encoders for retrieval and semantic similarity.
    - [Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)](https://arxiv.org/abs/2411.09009) — Cut Cross-Entropy (Apple); computes the loss without materializing the `[N, V]` logit tensor in global memory, collapsing the dominant memory term of training a large-vocabulary model.

    **Open-source & tools**

    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — ~300-line reference for weight tying, embedding initialization, and the full token-ID-to-logit pipeline.
    - [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — the Rust-backed tokenizer library (BPE, WordPiece, Unigram) that feeds token IDs into embedding layers in production systems.
    - [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — Triton kernels for LLM training; `LigerFusedLinearCrossEntropyLoss` fuses the tied LM head with the loss and is exposed as `TrainingArguments(use_liger_kernel=True)` in HuggingFace `transformers`.
    - [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) — reference implementation of Cut Cross-Entropy, a drop-in replacement for the `lm_head` + `cross_entropy` pair.
    - [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — `VocabParallelEmbedding` and `vocab_parallel_cross_entropy`: the canonical implementation of sharding the vocabulary axis across a tensor-parallel group.

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

---

## Exercises

**1.** *(Conceptual.)* The chapter says the embedding operation $\mathbf{e}_i = \mathbf{W}_E \mathbf{o}_i$ is "mathematically equivalent" to a matrix multiply but is "implemented as an index select, not an actual matmul." If the two produce the same output, why does the distinction matter? State the asymptotic cost of each and explain the physical reason one is preferred on a GPU.

??? note "Solution"
    Both compute the same vector — column $i$ (or row $i$ in PyTorch's `[V, d]` convention) of the embedding table — so the *output* is identical. The distinction is about *cost*, not correctness.

    - A real GEMM multiplies the full one-hot matrix $\mathbf{O} \in \{0,1\}^{T \times V}$ by $\mathbf{W}_E^\top \in \mathbb{R}^{V \times d}$. This is $O(TVd)$ multiply-adds, and almost all of that arithmetic multiplies by the zeros in the one-hot rows — pure waste.
    - An index select just copies the $T$ needed rows out of the table: $O(Td)$ work, with no multiplication at all.

    The ratio of wasted-to-useful work is $V$, which is $10^4$–$10^5$ for real vocabularies. Physically, the GPU only has to move $T \times d$ floats out of HBM instead of performing $T \times V \times d$ fused multiply-adds, so the lookup is bound by a tiny memory copy rather than a huge compute kernel. That is why `nn.Embedding` (and `embedding.weight[ids]`) uses `index_select` and the chapter stresses the $O(Td)$ vs $O(TVd)$ gap.

**2.** *(Quantitative.)* Consider GPT-2 small with $V = 50{,}257$ and $d = 768$, using weight tying. (a) How many parameters live in the tied embedding/unembedding table? (b) In bytes, how much memory does that table occupy in float32 vs bfloat16? (c) The chapter states GPT-2 small has roughly 124 M parameters total; what fraction of the model is the embedding table? (d) If weights were *not* tied, how many extra parameters would the separate unembedding add?

??? note "Solution"
    (a) The table has $V \times d = 50{,}257 \times 768 = 38{,}597{,}376 \approx 38.6$ M parameters. With tying this single table serves both input and output.

    (b) float32 is 4 bytes/param: $38{,}597{,}376 \times 4 = 154{,}389{,}504$ bytes $\approx 154$ MB. bfloat16 is 2 bytes/param: $\approx 77$ MB. (These match the numbers printed in the chapter's code.)

    (c) $38.6\text{ M} / 124\text{ M} \approx 0.31$, i.e. about **31%** of the model is the embedding table. This is why weight tying — which the chapter notes saves ~30% of GPT-2 small's parameters — is so impactful for small models.

    (d) An untied unembedding is another $[V, d]$ matrix, so it adds another $38.6$ M parameters, taking the vocabulary-dependent parameter count from ~38.6 M to ~77.2 M.

**3.** *(Quantitative / conceptual.)* Take the gradient-sparsity code in the chapter: a single sentence of 7 distinct token IDs is embedded, and `loss = x.sum()` is backpropagated into an `nn.Embedding(50257, 768)`. (a) Exactly how many of the 50,257 rows of `embed.weight.grad` are nonzero, and why? (b) With `loss = x.sum()`, what is the numerical value of every nonzero gradient entry? (c) Now suppose the sentence were `[464, 464, 3797]` (the token 464 repeated). How many rows are nonzero, and what is the gradient value in row 464?

??? note "Solution"
    (a) Exactly **7** rows are nonzero — one per *unique* token ID that appeared (`464, 3797, 3332, 319, 262, 2603, 13`). The forward pass is an index select, so only the rows that were looked up participate in the computation; every other row is disconnected from the loss and gets exactly zero gradient.

    (b) `x.sum()` sums all $1 \times 7 \times 768$ entries of the output. Each embedded token's vector is copied directly from its table row, so $\partial(\sum x)/\partial W_E[t, j] = 1$ for every column $j$ of a row $t$ that was used. Hence every nonzero entry equals exactly **1.0**.

    (c) There are only **2** unique tokens (`464` and `3797`), so 2 rows are nonzero. Token 464 was looked up **twice**, and gradients from both occurrences accumulate into the same row. Each occurrence contributes 1.0 per column, so row 464 has the value **2.0** in every column; row 3797 has 1.0. This accumulation for repeated tokens is exactly the "hot row" effect the chapter mentions.

**4.** *(Implementation.)* The chapter's `InputPipeline` adds token and learned positional embeddings. Modify it into a `TiedInputPipeline` that (i) uses learned positional embeddings as before, and (ii) exposes a `logits(hidden)` method whose output projection is *weight-tied* to the token embedding table (reuse `token_embed.weight`, no separate `nn.Linear`, no bias). Then write an assertion that verifies the output projection truly shares storage with the token embedding.

??? note "Solution"
    Following the chapter's `TiedTransformerHead` (which uses `F.linear(hidden, self.embed.weight)`) and the `InputPipeline` structure:

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TiedInputPipeline(nn.Module):
        def __init__(self, vocab_size: int, d_model: int,
                     max_seq_len: int, dropout: float = 0.1):
            super().__init__()
            self.d_model = d_model
            self.token_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed   = nn.Embedding(max_seq_len, d_model)
            self.drop        = nn.Dropout(dropout)
            nn.init.normal_(self.token_embed.weight, std=0.02)
            nn.init.normal_(self.pos_embed.weight,   std=0.02)

        def forward(self, ids: torch.LongTensor) -> torch.Tensor:
            B, T = ids.shape
            pos = torch.arange(T, device=ids.device)          # [T]
            x = self.token_embed(ids) + self.pos_embed(pos)    # [B, T, d]
            return self.drop(x)

        def logits(self, hidden: torch.Tensor) -> torch.Tensor:
            # Weight-tied output projection: reuse the token table, no bias.
            return F.linear(hidden, self.token_embed.weight)   # [B, T, V]

    # --- verification ---
    model = TiedInputPipeline(vocab_size=32_000, d_model=256, max_seq_len=512)
    ids = torch.randint(0, 32_000, (2, 16))
    h = model(ids)                          # [2, 16, 256]
    lg = model.logits(h)                    # [2, 16, 32000]
    assert lg.shape == (2, 16, 32_000)

    # The output projection allocates NO separate [V, d] tensor: the only
    # parameters are the two embedding tables. If a second V*d matrix had
    # sneaked in (e.g. an untied nn.Linear head) this count would be larger.
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 32_000 * 256 + 512 * 256   # token table + pos table only

    # And logits() reads exactly the token-embedding storage: its gradient
    # lands on token_embed.weight, confirming the weights are shared.
    lg.sum().backward()
    assert model.token_embed.weight.grad is not None
    ```

    The key point is that `logits()` never allocates a second $[V, d]$ tensor — it calls `F.linear` on `self.token_embed.weight` directly, exactly as the chapter recommends to avoid the "copying instead of sharing" pitfall. During backward, this weight receives the sparse gradient from `forward` plus the dense gradient from `logits`.

**5.** *(Quantitative.)* The chapter's worked example prices the logit projection for Llama 3 8B ($V = 128{,}000$, $d = 4{,}096$). (a) Compute the multiply-add count for the logit projection of a *single* token, in GFLOP-equivalent multiply-adds. (b) For $T = 2048$ tokens, how many total, and why is this "comparable to multiple attention layers"? (c) During single-token autoregressive decode, the projection reads the full 524 M-parameter table from HBM. At 900 GB/s (A100) in bfloat16, estimate the time, and explain why decode is memory-bound here rather than compute-bound.

??? note "Solution"
    (a) The projection is $\mathbf{h} \cdot \mathbf{W}_U^\top$ with $\mathbf{h} \in \mathbb{R}^{4096}$ and $\mathbf{W}_U \in \mathbb{R}^{128000 \times 4096}$. That is one multiply-add per weight: $4{,}096 \times 128{,}000 = 524{,}288{,}000 \approx 0.5$ G multiply-adds per token.

    (b) For $T = 2048$: $0.5 \times 2048 \approx 1{,}024$ GFLOPs (multiply-adds) for the logit projection alone. It is "comparable to multiple attention layers" because a single dense $[T, d] \times [d, V]$ matmul with $V \gg d$ moves as much arithmetic as several of the model's internal $[T, d] \times [d, d]$ projections — the vocabulary dimension $V = 128{,}000$ is ~31x larger than $d = 4{,}096$, so one unembed matmul rivals a stack of $d \times d$ ops.

    (c) In bfloat16 the table is $524 \times 10^6 \times 2 = 1{,}048 \times 10^6$ bytes $\approx 1.05$ GB. Reading it at 900 GB/s takes $1.05\text{ GB} / 900\text{ GB/s} \approx 1.16$ ms. During decode we compute logits for just one token: the arithmetic is only ~0.5 GFLOP (microseconds of compute on an A100's tens of TFLOP/s), but we must still stream all 524 M weights from HBM. The bottleneck is therefore moving ~1 GB of parameters, not the multiply-adds — the operation is memory-bandwidth-bound. This is exactly why the chapter notes some inference systems use lower precision or approximate logit computation.

**6.** *(Conceptual, harder.)* Under weight tying and the residual-stream view $\mathbf{h}_T = \mathbf{e} + \sum_{\ell} \Delta_\ell$, argue why two tokens whose *input* embeddings are nearly parallel (high cosine similarity) will tend to receive similar *output* logits, and describe one situation where this coupling helps and one where it hurts. Tie your answer to the gradient asymmetry the chapter describes.

??? note "Solution"
    With tying, the logit for token $k$ is $\text{logit}_k = \mathbf{h}_T \cdot \mathbf{W}_E[k]$ — an inner product of the final residual state with token $k$'s *embedding* row (the same vector used at the input). If tokens $a$ and $b$ have nearly parallel embedding rows ($\mathbf{W}_E[a] \approx \mathbf{W}_E[b]$), then for any hidden state $\mathbf{h}_T$ the two inner products are close, so $\text{logit}_a \approx \text{logit}_b$. The output head literally "votes" for a token using that token's embedding direction, so geometric closeness at the input forces closeness at the output.

    **Where it helps:** for semantically interchangeable tokens (e.g., near-synonyms, or a word and its rare casing variant), coupling acts as a regularizer — the model predicts them with similar probability without needing to learn two independent output directions, which improves perplexity at fixed parameters, especially for small models (as the chapter and Press & Wolf note).

    **Where it hurts:** if two tokens *should* be sharply distinguished as predictions but happen to sit close in embedding space, their logits interfere and the model cannot cleanly separate them — the chapter's warning that "tokens close in embedding space will interfere with each other's logit scores."

    Gradient asymmetry connects here: a token's embedding row is pulled by a *dense* gradient from the unembed path (all $T$ hidden states, dominated by frequent tokens) and only a *sparse* gradient from the input lookup (present only when that token appears). So frequent tokens' embedding directions are shaped mostly by their role as prediction targets, while rare tokens get almost all their signal from the sparse input path — meaning the "voting direction" for rare tokens is under-trained, amplifying harmful interference precisely for the tokens that can least afford it.
