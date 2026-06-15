# 3.13 Long-Context Pretraining & Context Extension

What changes when you train a language model to read 128 thousand tokens at once instead of 2 thousand? The answer turns out to touch nearly every part of the stack: attention arithmetic, positional encoding mathematics, data curation, parallelism strategy, and inference memory management. This chapter develops each of those changes from first principles.

We will see why a naive extension to long contexts fails, how the community learned to patch positional encodings cheaply via RoPE interpolation, how continued pretraining on long documents consolidates those patches, and finally how ring/context parallelism makes the training computationally tractable. We close by connecting long-context training to the inference KV-cache problem you will likely be asked about in a systems design interview.

---

## The Two Fundamental Bottlenecks

### Quadratic Attention Cost

Standard self-attention computes a score matrix $S \in \mathbb{R}^{T \times T}$ for a sequence of length $T$:

$$
S = \frac{QK^\top}{\sqrt{d_k}}, \quad \text{Attention}(Q,K,V) = \operatorname{softmax}(S)\,V
$$

Both the FLOPs for this product and the memory to store $S$ grow as $\mathcal{O}(T^2)$. At $T = 2048$ this is manageable; at $T = 128{,}000$ it becomes painful:

| Sequence length | Score matrix (fp16) |
|---|---|
| 2,048 | 16 MB per head |
| 32,768 | ~4 GB per head |
| 128,000 | ~61 GB per head |

A model with 32 heads would need ~2 TB just for score matrices at 128 K. FlashAttention (covered in depth in [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) and [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html)) eliminates the need to materialise the full $S$ in HBM by fusing the computation in tiles, reducing the HBM requirement to $\mathcal{O}(T)$ activations. But the $\mathcal{O}(T^2)$ FLOPs remain — you simply compute them faster.

### Memory From Activations and KV Cache

During training, gradient checkpointing can trade memory for compute, but even with it, intermediate activations scale linearly with $T$ per layer. More acutely, training requires storing the full sequence per sample in the attention buffer. At long sequence lengths the per-token memory cost does not shrink; it just shifts from the score matrix to the gradient and activation buffers.

During inference, all previously computed keys and values must be kept in memory — the KV cache — so the context cost is $\mathcal{O}(T)$ per forward pass but persists for the entire generation episode. For a 70 B-parameter model with 8 KV heads (GQA), the KV cache for a single 128 K-token context is roughly:

$$
2 \times 8 \times 128{,}000 \times d_{\text{kv}} \times \text{bytes/value}
$$

With $d_{\text{kv}} = 128$ and bf16 (2 bytes): $2 \times 8 \times 128{,}000 \times 128 \times 2 \approx 524\;\text{MB}$ per layer. With 80 layers that is around 42 GB — comparable to the model weights themselves. Managing this is the focus of [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

---

## RoPE and the Extrapolation Problem

### Why Sinusoidal and Learned Encodings Fail

A model trained with absolute positional embeddings at positions $\{0, \ldots, T_{\text{train}}-1\}$ has never seen the embedding at position $T_{\text{train}} + k$ for any $k > 0$. Learned embeddings simply have no weight there. Sinusoidal embeddings produce a valid vector at any position, but the attention patterns were never trained to use those values, so performance degrades sharply. See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html) for the full encoding taxonomy.

### RoPE Refresher

RoPE (Rotary Position Embedding, Su et al., 2022) applies a position-dependent rotation to query and key vectors before the dot product. For a pair of scalars at dimension index $2i$ and $2i+1$:

$$
\begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}_{\text{rotated}} = \begin{bmatrix} \cos(m\,\theta_i) & -\sin(m\,\theta_i) \\ \sin(m\,\theta_i) & \cos(m\,\theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
$$

where $m$ is the absolute position and $\theta_i = b^{-2i/d}$ with base $b = 10{,}000$ by default. The key insight is that the dot product $q_m \cdot k_n$ depends only on the *relative* position $m - n$, because the rotation matrices satisfy $R_m^\top R_n = R_{m-n}$.

The problem for extrapolation: at training time, $m$ never exceeds $T_{\text{train}}$. The angles $m \theta_i$ stay within $[0, T_{\text{train}} \theta_i]$. For the low-frequency dimensions (large $i$, tiny $\theta_i$) this range can be small — those dimensions barely rotate even within the training window. For the high-frequency dimensions the rotation is well-covered. When you push $m > T_{\text{train}}$, high-frequency dimensions enter completely unseen phase territory, causing attention to produce pathological scores.

### Position Interpolation (PI)

Chen et al. (2023, *Extending Context Window of Large Language Models via Positional Interpolation*) propose a one-line fix: rather than feeding position $m$ to RoPE, feed the scaled position $m' = m \cdot \frac{T_{\text{train}}}{T_{\text{target}}}$. This linearly rescales positions so the full target window fits inside the training range.

If the training window is $T_{\text{train}} = 4{,}096$ and the target is $T_{\text{target}} = 32{,}768$, every position is divided by $32{,}768 / 4{,}096 = 8$. All angles stay within the trained range.

The cost: nearby tokens that were originally at relative distance 1 are now at effective distance $1/8$, so the model initially cannot distinguish adjacent tokens well. A small amount of fine-tuning (on the order of 1 000 steps) on long documents restores performance. The work reports good recovery at 8x extension.

### NTK-aware Interpolation

NTK-aware interpolation (bloc97, community research 2023) observes that PI uniformly compresses all frequencies, damaging the high-frequency (short-range) components that carry most of the positional signal. Instead, it rescales only the *base* $b$:

$$
b' = b \cdot \left(\frac{T_{\text{target}}}{T_{\text{train}}}\right)^{d/(d-2)}
$$

This increases the base (e.g. from 10 000 to ~500 000 for an 8x extension), making each $\theta_i$ smaller and thus each dimension rotate more slowly — effectively stretching the rope. High-frequency dimensions retain their resolution for local dependencies while low-frequency dimensions stretch to cover global distances. NTK-aware interpolation often works *without any fine-tuning*, giving it a zero-shot extension property.

### YaRN

YaRN (Peng et al., 2023, *YaRN: Efficient Context Window Extension of Large Language Models*) builds further on NTK-aware scaling by introducing:

1. **Frequency ramping.** A smooth interpolation factor $\gamma(i)$ that applies no scaling to the highest-frequency (most local) dimensions and full PI scaling to the lowest-frequency (most global) ones:

$$
\theta'_i = \begin{cases} \theta_i & \text{if } \lambda_i \leq \lambda_{\min} \\ \theta_i / s & \text{if } \lambda_i \geq \lambda_{\max} \\ \text{interpolated} & \text{otherwise} \end{cases}
$$

where $\lambda_i = 2\pi / \theta_i$ is the wavelength and $s = T_{\text{target}} / T_{\text{train}}$ is the scale factor.

2. **Attention temperature correction.** Extended positions produce larger dot products because the effective $d_k$ changes. YaRN multiplies the softmax logits by a temperature $t \approx 0.1 \ln s + 1$ to compensate.

YaRN requires only a small amount of continued training (around 400 steps in the original work) and often outperforms PI at the same training budget. It is widely adopted: Mistral, Qwen, and Deepseek use variants of YaRN.

```python
import torch
import math

def build_rope_freqs(dim: int, base: float, seq_len: int) -> torch.Tensor:
    """Build standard RoPE inverse frequencies for a given dimension and base."""
    # Shape: (dim // 2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # Outer product with positions → (seq_len, dim // 2)
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)      # shape: (seq_len, dim // 2)
    return freqs  # each row is the angles for one position

def build_yarn_freqs(
    dim: int,
    base: float,
    orig_max_seq_len: int,
    target_max_seq_len: int,
    beta_fast: float = 32.0,   # high-freq boundary (wavelengths)
    beta_slow: float = 1.0,    # low-freq boundary
) -> torch.Tensor:
    """
    YaRN frequency computation.
    Returns per-dimension scaling factor to apply to the standard freqs.
    """
    scale = target_max_seq_len / orig_max_seq_len

    # Standard inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Wavelength for each dimension: λ = 2π / θ_i
    wavelengths = 2 * math.pi / inv_freq

    # Ramp function: 0 for high-freq dims (leave unchanged), 1 for low-freq (full scale)
    ramp = (wavelengths - orig_max_seq_len / beta_fast) / (
        orig_max_seq_len / beta_slow - orig_max_seq_len / beta_fast
    )
    ramp = ramp.clamp(0.0, 1.0)

    # Blend: high-freq → no interpolation, low-freq → PI scaling
    # NTK base for fully-scaled dims: b' = b * scale^(dim/(dim-2))
    ntk_base = base * (scale ** (dim / (dim - 2)))
    ntk_inv_freq = 1.0 / (ntk_base ** (torch.arange(0, dim, 2).float() / dim))

    # Interpolate between unscaled (ramp=0) and NTK-scaled (ramp=1) per dimension
    blended_inv_freq = (1 - ramp) * inv_freq + ramp * ntk_inv_freq

    t = torch.arange(target_max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, blended_inv_freq)
    return freqs  # (target_max_seq_len, dim // 2)

# --- demo ---
# GPT-style model: dim=128, base=10000, trained at 4096, extending to 32768
orig_freqs  = build_rope_freqs(128, 10000, 4096)
yarn_freqs  = build_yarn_freqs(128, 10000, 4096, 32768)

print(f"Original freq range at pos 4096: [{orig_freqs[-1].min():.4f}, {orig_freqs[-1].max():.4f}]")
print(f"YaRN freq range at pos 32768:    [{yarn_freqs[-1].min():.4f}, {yarn_freqs[-1].max():.4f}]")
# YaRN keeps high-freq dims in known territory; low-freq dims gently scaled.
```

!!! example "Worked Example: Comparing PI vs YaRN angle ranges at 8x extension"

    Consider a model with $d = 128$ trained at $T_{\text{train}} = 4{,}096$. We extend to $T_{\text{target}} = 32{,}768$ (8x).

    **Standard RoPE** at position 32 768, dimension $i=0$ (highest frequency):
    $$\theta_0 = 10000^{0/128} = 1.0, \quad m\theta_0 = 32{,}768 \text{ rad}$$
    The model was trained only up to $4096 \text{ rad}$ for this dimension — over 8x out of distribution.

    **Position Interpolation** divides every position by 8:
    $$m' = 32{,}768 / 8 = 4{,}096, \quad m'\theta_0 = 4{,}096 \text{ rad}$$
    Now the highest-frequency dimension is safely in range, but nearby positions at $m=1$ and $m=2$ now appear at $m'=0.125$ and $m'=0.25$ — the model can barely tell them apart.

    **YaRN** with $\beta_{\text{fast}} = 32$ applies *no scaling* to the highest-frequency dimension (its wavelength $\lambda_0 = 2\pi \approx 6.3$ is far below the original context 4096), so nearby tokens remain distinguishable. It applies full NTK-style scaling only to low-frequency dimensions that actually need longer range. The result is the best of both worlds.

---

## Continued Pretraining on Long Documents

### Why Fine-Tuning Alone Is Insufficient

RoPE scaling patches the positional encoding but does not teach the model new attention patterns or update MLP weights to reason over document-scale structure. A 4 K-trained model re-encodes positions at 32 K but the attention heads have never learned which distant tokens to attend to: they were never penalised for ignoring tokens at offset >4 096. Continued pretraining on genuinely long documents rectifies this.

### Data Curation for Long-Context Training

The key challenge is obtaining sufficiently long training documents. The pretraining corpus — described in [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) — mostly consists of web paragraphs shorter than the target context. Long-context continued pretraining typically draws from:

- **Books** (Project Gutenberg, Books3): naturally multi-thousand-word.
- **Scientific papers** (arXiv full PDFs, Semantic Scholar): 6–20 K tokens each.
- **Code repositories** (The Stack, GitHub): single files or whole-repo concatenations.
- **Legal and financial documents**: contracts, 10-K filings.
- **Long web articles**: Wikipedia pages, journalism.

A useful rule of thumb from community experimentation: 5–20 billion tokens of continued pretraining at the target context length is often sufficient to unlock the new length. The ratio of long-to-short documents matters — too many short documents and the model never practices long-range attention.

### Learning Rate and Scheduler Strategy

Continued pretraining for context extension typically uses:

- A learning rate **significantly lower** than the original pretraining peak (on the order of 1–5% of it), to avoid catastrophic forgetting of short-context capabilities.
- A **cosine decay** or flat schedule, often without warmup since the model already has stable weights.
- Gradient clipping set conservatively (e.g., $\ell_2$ norm of 1.0).

See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html) for the general scheduling machinery.

---

## Document Packing for Long-Context Training

### The Problem With Naïve Packing

Training efficiency requires that every GPU sees a tensor of exactly $T_{\text{seq}}$ tokens per step. Short documents must be concatenated to fill the window, and long documents must be split across windows (or within one window). The naïve approach simply concatenates documents end-to-end and slices at $T_{\text{seq}}$ boundaries.

This creates a cross-document attention problem: the model computes attention between the end of document $A$ and the beginning of document $B$, even though they are unrelated. At short context lengths this is a minor source of noise; at 128 K context it can mean most of the window is cross-document garbage, actively harming training signal.

### Document Masking in Attention

The principled fix is a **document attention mask**: each token attends only to tokens that belong to the same document (and obeys causal ordering within that document). This is equivalent to block-diagonal causal masking over the packed sequence.

```python
import torch

def build_packed_attention_mask(
    doc_lengths: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a boolean causal attention mask for a packed sequence of multiple docs.

    doc_lengths: list of token counts per document, must sum to seq_len.
    Returns: (seq_len, seq_len) bool tensor where True = can attend.
    """
    seq_len = sum(doc_lengths)
    # Start with no attention allowed
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    offset = 0
    for length in doc_lengths:
        # Within this document's block, allow lower-triangular (causal) attention
        block_end = offset + length
        for row in range(offset, block_end):
            # Token at 'row' can attend to [offset .. row] (same doc, causal)
            mask[row, offset:row + 1] = True
        offset = block_end

    return mask  # (seq_len, seq_len)

# Example: two docs packed into a single context of length 8 (4 + 4)
mask = build_packed_attention_mask([4, 4])
print(mask.int())
# tensor([[1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 1, 0, 0],
#         [0, 0, 0, 0, 1, 1, 1, 0],
#         [0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
```

In practice, FlashAttention 2 supports "variable-length" (varlen) sequences via the `cu_seqlens` API, which achieves exactly this masking without materialising the full $T \times T$ mask tensor:

```python
# FlashAttention varlen API (flash_attn library)
from flash_attn import flash_attn_varlen_func
import torch

# Suppose we have a batch of two sequences packed: lengths 512 and 1024
batch_q = torch.randn(512 + 1024, 16, 64, device="cuda", dtype=torch.bfloat16)  # (total_tokens, heads, d_head)
batch_k = torch.randn_like(batch_q)
batch_v = torch.randn_like(batch_q)

# Cumulative sequence lengths (must start at 0 and end at total_tokens)
cu_seqlens = torch.tensor([0, 512, 512 + 1024], dtype=torch.int32, device="cuda")
max_seqlen = 1024

out = flash_attn_varlen_func(
    batch_q, batch_k, batch_v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,             # causal within each segment
)
# out.shape == (total_tokens, heads, d_head)
```

The varlen interface ensures that attention computations never cross document boundaries, at zero mask-storage cost.

### Loss Masking

The cross-entropy loss should also exclude the first token of each document's continuation *from* predicting the last token of the previous document. A simple `loss_mask` tensor of 0/1 per token accomplishes this — set the first token of each document to 0 (do not compute loss for it). See also [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) for how the same principle applies to fine-tuning packing.

---

## Evaluating Long-Context Models: Needle in a Haystack

### The Test

*Needle in a Haystack* (NIAH) is the canonical evaluation for long-context retrieval fidelity. A single "needle" sentence containing a unique fact (e.g., "The magic phrase is pineapple-strawberry-7.") is inserted at a controlled depth (e.g., 25%, 50%, 75%) inside a long "haystack" of unrelated text (often Paul Graham essays or Wikipedia). The model must retrieve the needle after reading the full context.

The result is plotted as a 2D heatmap: the x-axis is context length, the y-axis is needle depth, and color indicates whether the model retrieved it correctly. A well-trained model shows uniform success across the entire grid; a poorly extended model shows failures at long contexts and specific depths.

```python
import random

def build_haystack_with_needle(
    haystack_text: str,
    needle: str,
    context_length: int,           # target total tokens (approx by words)
    depth_fraction: float,         # 0.0 = beginning, 1.0 = end
    tokenizer_approx_ratio: float = 1.3,  # tokens per word (rough)
) -> tuple[str, int]:
    """
    Insert a needle sentence at a given depth in the haystack.
    Returns (full_text, needle_start_word_index).
    """
    target_words = int(context_length / tokenizer_approx_ratio)

    # Tile haystack words to fill the context
    words = haystack_text.split()
    if len(words) < target_words:
        multiplier = (target_words // len(words)) + 1
        words = words * multiplier
    words = words[:target_words]

    # Determine insertion point
    insert_idx = int(len(words) * depth_fraction)

    # Insert needle words at that point
    needle_words = needle.split()
    words = words[:insert_idx] + needle_words + words[insert_idx:]

    return " ".join(words), insert_idx

def evaluate_niah(
    model_fn,           # callable(text: str) -> str
    needle: str,
    answer_keyword: str,
    context_lengths: list[int],
    depths: list[float],
    haystack: str,
) -> dict:
    """
    Run a full NIAH evaluation grid.
    Returns dict[(context_len, depth)] -> bool
    """
    results = {}
    for ctx_len in context_lengths:
        for depth in depths:
            text, _ = build_haystack_with_needle(haystack, needle, ctx_len, depth)
            prompt = text + "\n\nWhat is the magic phrase?"
            response = model_fn(prompt)
            results[(ctx_len, depth)] = answer_keyword.lower() in response.lower()
    return results
```

NIAH is a necessary but not sufficient evaluation: a model can pass NIAH yet still fail at multi-hop reasoning over long contexts because NIAH only requires retrieval of a single salient fact. More challenging variants scatter multiple needles or require combining information from two distant locations.

---

## Ring Attention and Context Parallelism for Training

### Why We Need a New Parallelism Dimension

Even with FlashAttention, training a sequence of 128 K tokens on a single GPU is impossible: the tiled attention still needs $\mathcal{O}(T)$ intermediate buffers per layer. At 128 K tokens in bf16, the KV activations for one layer with $d_{\text{model}} = 4096$ are:

$$
2 \times 4096 \times 128{,}000 \times 2\;\text{bytes} \approx 2\;\text{GB per layer}
$$

For 80 layers that is 160 GB, exceeding a single A100 80 GB GPU. We need to shard the sequence across devices — a different axis than the data, tensor, or pipeline parallelism of [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html).

### Ring Attention

Ring Attention (Liu et al., 2023, *Ring Attention with Blockwise Transformers*) achieves sequence parallelism by distributing non-overlapping chunks of the sequence across $N$ GPUs arranged in a logical ring. Each GPU holds $T/N$ tokens' keys and values locally. Attention is computed as follows:

```text
Ring of 4 GPUs, each holding T/4 tokens:

Step 0: GPU-i computes attention of its Q chunk against its own KV chunk.
Step 1: Each GPU sends its KV chunk to the next GPU in the ring.
        While the data is in flight, each GPU continues computing
        attention of its Q chunk against the KV chunk it is about to
        receive (overlap communication with computation).
Step 2: Repeat for N-1 total rounds.
After N rounds, each GPU has accumulated the full softmax-normalised
attention output for its Q chunk.
```

The key algorithmic insight is that FlashAttention's *online softmax accumulation* is associative: you can accumulate the numerator sums and the log-sum-exp denominators independently across rounds and produce the exact same result as if all KV blocks were available at once. No approximation is involved.

Communication volume per round: $2 \times (T/N) \times d \times 2\;\text{bytes}$ per direction (sending K and V). With $N = 8$ GPUs and $T = 128{,}000$, $d = 4096$: each round moves $128{,}000/8 \times 4096 \times 2 \times 2 = 256\;\text{MB}$ per GPU. On NVLink at 400 GB/s this takes ~0.6 ms; a single attention kernel at this scale takes much longer, so the overlap is effective.

```python
# Pseudocode sketch of ring attention forward pass
# (actual implementation uses NCCL send/recv and a FlashAttention tile kernel)

import torch
import torch.distributed as dist

def ring_attention_forward(
    q_local: torch.Tensor,     # (T/N, H, D) — this GPU's query chunk
    k_local: torch.Tensor,     # (T/N, H, D) — this GPU's key chunk (starting KV)
    v_local: torch.Tensor,     # (T/N, H, D) — this GPU's value chunk
    causal: bool,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Simplified ring attention. Real implementations fuse this with FlashAttention.
    """
    T_local = q_local.shape[0]
    d_model = q_local.shape[-1]

    # Accumulators for online softmax (see FlashAttention chapter for derivation)
    output_acc = torch.zeros_like(q_local)        # O accumulator
    lse_acc = torch.full((q_local.shape[0], q_local.shape[1]), float('-inf'))  # log-sum-exp

    k_chunk = k_local.clone()
    v_chunk = v_local.clone()
    # We track which absolute position offset these k/v come from
    kv_rank = rank

    for step in range(world_size):
        # Compute attention of q_local against k_chunk / v_chunk
        # (in reality this calls flash_attn_with_kvcache or equivalent)
        scale = d_model ** -0.5
        scores = torch.einsum("thd,khd->thk", q_local, k_chunk) * scale  # (T, H, T_kv)

        if causal:
            # Mask out future KV positions relative to each Q position
            q_global_start  = rank * T_local
            kv_global_start = kv_rank * T_local
            for t in range(T_local):
                q_pos = q_global_start + t
                # Cannot attend to kv positions > q_pos
                cutoff = q_pos - kv_global_start + 1
                if cutoff <= 0:
                    scores[t] = float('-inf')
                elif cutoff < T_local:
                    scores[t, :, cutoff:] = float('-inf')

        # Online softmax accumulation (simplified; real code tracks m and l)
        new_lse = torch.logsumexp(scores, dim=-1)                    # (T, H)
        new_out = torch.softmax(scores, dim=-1) @ v_chunk            # (T, H, D)

        # Merge with running accumulator
        m = torch.maximum(lse_acc, new_lse)
        exp_old = torch.exp(lse_acc - m)
        exp_new = torch.exp(new_lse - m)
        lse_acc = m + torch.log(exp_old + exp_new)
        output_acc = (output_acc * exp_old.unsqueeze(-1) + new_out * exp_new.unsqueeze(-1)) / (exp_old + exp_new).unsqueeze(-1)

        # Rotate KV to next GPU in ring (overlap with next step's compute in real code)
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size
        dist.send(k_chunk, dst=next_rank)
        dist.recv(k_chunk, src=prev_rank)
        dist.send(v_chunk, dst=next_rank)
        dist.recv(v_chunk, src=prev_rank)
        kv_rank = (kv_rank - 1) % world_size  # the received chunk came from one step earlier

    return output_acc
```

### Sequence Parallelism in Megatron-LM

Megatron-LM (see [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html)) has a separate "sequence parallelism" mode that shards the LayerNorm and Dropout computations along the sequence dimension while using tensor parallelism for the attention projections. This is distinct from ring attention: it reduces activation memory for non-attention operations by splitting the sequence across tensor-parallel ranks, using `all-gather` and `reduce-scatter` at the sequence axis.

In practice, the two approaches compose: one might use tensor parallelism for weight matrices, ring/context attention for the attention itself, and pipeline parallelism for layers — the "4D parallelism" configuration used by some large-scale long-context runs.

!!! example "Worked Example: Memory Budget for 128 K Context Training"

    Configuration: 70 B parameter model, $d_{\text{model}} = 8192$, 80 layers, 64 heads, GQA with 8 KV heads, $d_{\text{head}} = 128$, batch size = 1 sequence of 128 K tokens, bf16 (2 bytes).

    **Model weights (parameters):** 70 B × 2 bytes = **140 GB**. Requires at minimum 8 × A100 80 GB GPUs under model parallelism.

    **Activations per attention layer (without ring attention):**
    - Q, K, V projections: $3 \times 128{,}000 \times 8192 \times 2 \approx 6.3\;\text{GB}$
    - Score matrix (if not FlashAttention): $64 \times 128{,}000^2 \times 2 \approx 2\;\text{TB}$ — clearly impossible.
    - With FlashAttention: score matrix never materialised; only $O(\text{block size})$ in SRAM.

    **Activations per attention layer (with FlashAttention + ring attention over 8 GPUs):**
    - Each GPU holds Q/K/V for $128{,}000 / 8 = 16{,}000$ tokens: $3 \times 16{,}000 \times 8192 \times 2 \approx 786\;\text{MB}$
    - The FlashAttention block size is ~128 tokens; only that block of scores lives in SRAM at once.
    - Per-layer ring overhead: 2 × KV chunk per round = $2 \times 16{,}000 \times 8 \times 128 \times 2 \approx 65\;\text{MB}$ in transit.

    **KV cache during inference** (separate from training):
    Each layer, each token: $2 \times 8 \times 128 \times 2 = 4096\;\text{bytes}$ = 4 KB.
    For 128 K tokens: $128{,}000 \times 4096 = 524\;\text{MB}$ per layer, $\times 80$ layers = **42 GB** total KV cache.

    This is why even at inference time, 128 K context with a 70 B model requires a multi-GPU setup.

---

## Connecting Long-Context Training to the Inference KV Cache

Training and inference share the same fundamental tension: both need to store or compute over $T$ keys and values per layer. The training solution (ring attention, gradient checkpointing) addresses the activation memory during the forward-backward pass. The inference solution (paged/chunked KV cache) addresses the persistent memory across generation steps.

The positional encoding choices made during training directly constrain what the inference engine can do:

- **RoPE base and scaling method**: The inference engine must apply the *same* RoPE modification as training. A mismatch (e.g., serving with standard RoPE a model trained with YaRN) produces garbage scores.
- **Maximum context length**: The inference engine's KV cache pre-allocates memory up to `max_model_len`. Extending this at inference time without retraining is limited by the interpolation range of the chosen scaling method.
- **GQA vs MHA**: GQA (8 KV heads instead of 64) cuts the KV cache footprint 8x and is a prerequisite for long contexts at scale. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

For chunked prefill — processing a 128 K context in chunks rather than all at once at inference startup — see [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

!!! interview "Interview Corner"
    **Q:** A model was pretrained with a 4 K context. You want to serve it at 32 K. Walk me through the approaches in order of increasing reliability, and explain the tradeoffs.

    **A:** Three approaches, ascending in reliability:

    1. **Zero-shot RoPE scaling (NTK-aware or Dynamic NTK):** Rescale the RoPE base at inference time with no additional training. Works surprisingly well for modest extensions (~4x) because it keeps high-frequency (short-range) dimensions in-distribution. The model's attention patterns were never trained at long range, so quality degrades on tasks requiring global reasoning, but short-range dependencies remain intact. Zero cost; no GPU required.

    2. **RoPE + continued pretraining (PI or YaRN):** Apply a scaling method such as YaRN to the positional embeddings, then continue pretraining on long documents for a few hundred to a few thousand steps at a reduced learning rate. The model learns genuine long-range attention patterns, not just in-distribution positional angles. Requires on the order of 5–20 B tokens of long training data and 100–1 000 GPU-hours for a 7 B model. Recovers most of the short-context performance if the mix includes sufficient short documents.

    3. **Train long contexts from scratch:** Incorporate long documents from the beginning of pretraining, using ring/context attention from the start. Expensive but produces models with consistent performance across the full context window. State-of-the-art long-context models (Claude 3, Gemini 1.5) use this strategy. Tradeoff: each token in a long sequence is more expensive per step (quadratic FLOPs), so you see fewer unique tokens per dollar.

    For a 32 K target, approach 2 with YaRN is the sweet spot — achievable by a mid-sized team with a few weeks of compute.

---

## Practical Training Recipe

Putting the above together, here is a concrete recipe to extend a pretrained 7 B RoPE model from 4 K to 32 K:

```bash
# 1. Curate a long-context fine-tuning dataset
# Mix: 60% books/papers/code >16 K tokens, 40% original pretraining mix (short docs)
# Target total: ~10 B tokens

# 2. Modify model config to use YaRN
# In HuggingFace config.json:
#   "rope_scaling": {"type": "yarn", "factor": 8.0,
#                    "original_max_position_embeddings": 4096}

# 3. Launch continued pretraining with reduced LR
torchrun --nproc_per_node=8 train.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --max_seq_length 32768 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \           # ~2% of original peak LR
  --lr_scheduler_type cosine \
  --num_train_epochs 1 \
  --warmup_steps 20 \
  --bf16 \
  --gradient_checkpointing \
  --attn_implementation flash_attention_2 \
  --dataloader_packing True \
  --packing_loss_mask True          # mask first token of each document
```

```python
# 4. Validate with NIAH sweep before serving
import matplotlib.pyplot as plt
import numpy as np

def plot_niah_heatmap(results: dict, context_lengths: list, depths: list):
    """
    Visualise needle-in-haystack results as a 2D heatmap.
    results: dict[(ctx_len, depth)] -> bool
    """
    grid = np.array([
        [float(results.get((c, d), False)) for c in context_lengths]
        for d in depths
    ])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(context_lengths)))
    ax.set_xticklabels([f"{c//1000}K" for c in context_lengths], rotation=45)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{int(d*100)}%" for d in depths])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Needle Depth")
    ax.set_title("Needle-in-Haystack Retrieval Accuracy")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("niah_heatmap.png", dpi=150)
    print("Saved niah_heatmap.png")
    return grid.mean()  # return overall accuracy
```

!!! warning "Common pitfall: forgetting to update max_position_embeddings"

    HuggingFace models cache `max_position_embeddings` in `config.json`. If you update the RoPE scaling factor but leave `max_position_embeddings = 4096`, the model will refuse to generate beyond 4 096 tokens at inference time even though it was trained at 32 K. Always set `max_position_embeddings` to the *new* target length. Similarly, vLLM reads `max_model_len` from the config or CLI; failing to set it will cause silent truncation.

!!! tip "Practitioner tip: use dynamic NTK for zero-shot quick evaluation"

    Before committing to a full continued pretraining run, validate that your model's architecture is RoPE-compatible with NTK-aware dynamic scaling. Many HuggingFace models accept `rope_scaling={"type": "dynamic", "factor": 8.0}` which automatically applies NTK scaling for sequences longer than the trained length. Run NIAH at your target length zero-shot. If accuracy is above ~60%, continued pretraining will likely bring it to >90%. If it is near 0%, investigate whether the model uses RoPE at all, or whether there is a positional encoding mismatch.

---

## Key Takeaways

!!! key "Key Takeaways"
    - Quadratic attention FLOPs are manageable with FlashAttention; the real bottleneck is activation memory and positional encoding extrapolation.
    - Standard RoPE extrapolates poorly. Position Interpolation (PI) divides positions by the extension ratio; NTK-aware scaling raises the RoPE base; YaRN blends per-frequency to protect both local and global dependencies.
    - YaRN is the current practical default: it requires only hundreds of continued pretraining steps and often works zero-shot for modest (~4x) extension.
    - Continued pretraining on long documents (5–20 B tokens, LR ~1–5% of peak) teaches genuine long-range attention patterns that positional scaling alone cannot provide.
    - Document-aware packing with FlashAttention's varlen API prevents cross-document attention contamination and is critical at long context lengths.
    - Ring attention (context parallelism) shards the sequence across GPUs in a ring, overlapping communication with the tiled attention compute to make 128 K+ contexts tractable on multi-GPU clusters.
    - Needle-in-haystack is a necessary evaluation sanity check but tests only retrieval, not multi-hop reasoning. Always supplement with longer-context benchmarks.
    - Every positional encoding decision made during training propagates directly to the inference KV cache: base, scaling factor, and max length must be consistent end-to-end.

---

!!! sota "State of the Art & Resources (2026)"
    Long-context pretraining has progressed rapidly: production models routinely support 128K–1M token windows using a combination of RoPE-based position extension, continued pretraining on long documents, and ring/context parallelism. The core techniques (PI, NTK-aware scaling, YaRN) are now standard ingredients in every major open-weight recipe, and context windows beyond 2M tokens have been demonstrated in research settings.

    **Foundational work**

    - [Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)](https://arxiv.org/abs/2104.09864) — original RoPE paper; the rotation-based encoding that all context-extension methods build on.
    - [Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023)](https://arxiv.org/abs/2306.15595) — introduced linear position interpolation (PI), the simplest and most-cited RoPE extension recipe.

    **Recent advances (2023–2026)**

    - [Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023)](https://arxiv.org/abs/2309.00071) — per-frequency interpolation + attention temperature scaling; became the practical default adopted by Mistral, Qwen, and DeepSeek.
    - [Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023)](https://arxiv.org/abs/2310.01889) — sequence parallelism via a GPU ring; enables training at 128K+ tokens without approximations.
    - [Chen et al., *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models* (2023)](https://arxiv.org/abs/2309.12307) — shifted sparse attention cuts the compute cost of long-context fine-tuning 16x while preserving full attention at inference.
    - [Ding et al., *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens* (2024)](https://arxiv.org/abs/2402.13753) — non-uniform positional interpolation search pushes verified context to 2M tokens with only 1K fine-tuning steps.

    **Open-source & tools**

    - [jquesnelle/yarn](https://github.com/jquesnelle/yarn) — reference YaRN implementation with fine-tuned checkpoints for LLaMA 2, Mistral, and SOLAR at 32K–128K.
    - [lhao499/llm_large_context](https://github.com/lhao499/llm_large_context) — official Ring Attention JAX implementation; pip-installable as `ringattention`.

    **Go deeper**

    - [EleutherAI Blog, *Extending the RoPE* (2023)](https://blog.eleuther.ai/yarn/) — the community write-up that introduced NTK-aware interpolation and motivated YaRN; excellent mathematical intuition.
    - [Bai et al., *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding* (2023)](https://arxiv.org/abs/2308.14508) — 21-task evaluation suite covering QA, summarisation, and code; goes well beyond needle-in-haystack retrieval.

## Further Reading

- **Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2022)** — original RoPE paper; introduces the rotation formulation and relative-position property.
- **Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (2023)** — the PI approach; simple and highly cited.
- **Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023)** — introduces frequency-ramped interpolation and temperature correction; practical state of the art.
- **Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)** — ring attention paper; the foundational sequence-parallelism approach.
- **Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)** — the varlen API for document packing is documented here.
- **Anthropic, "Claude's Long Context" (technical reports, 2024)** — high-level discussion of training recipes for very long contexts (100 K+).
- **LongBench (Bai et al., 2023)** — a comprehensive long-context evaluation benchmark covering summarisation, QA, few-shot learning, and code; more demanding than needle-in-haystack alone.
- **LLaMA-3 technical report (Meta, 2024)** — describes the multi-stage context extension recipe used in a public large model, including data mix and positional encoding choices.
