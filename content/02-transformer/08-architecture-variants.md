# 2.8 Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM

The transformer block — layer norm, multi-head attention, feed-forward, residual — is a fixed recipe. What varies enormously across model families is the *way those blocks are wired together* and, most critically, *which tokens can attend to which other tokens*. Three distinct wiring patterns dominate the field: the **encoder-only** model (BERT), the **encoder-decoder** model (T5, BART), and the **decoder-only** model (GPT). A fourth pattern, the **prefix language model** (PrefixLM), sits between encoder-decoder and decoder-only and is worth understanding in its own right.

This chapter builds each family from the masking pattern outward — because the mask is the thing that determines what information flows where, and getting that wrong at initialization or fine-tuning time is one of the most common, silent bugs in applied LLM work. We will also trace the industry's convergence on decoder-only for frontier models, explain why that happened, and give you the vocabulary and intuitions to answer interview questions cold.

For the mechanics of the individual block, see [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html). For the concrete GPT implementation, see [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html). For the attention mechanism itself, see [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

---

## Attention Masks: The Lingua Franca of Architecture

Before we look at families, we need a precise vocabulary for attention masks.

In any transformer layer the attention logit matrix is:

$$
L_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

where $i$ is the query position and $j$ is the key position. A **mask** is a boolean matrix $M \in \{0,1\}^{T \times T}$ (or equivalently a $\{0, -\infty\}$ additive mask) where $M_{ij} = 1$ means "position $i$ is allowed to attend to position $j$." After adding the mask, we apply softmax:

$$
A_{ij} = \operatorname{softmax}_j\!\left(L_{ij} + \underbrace{(1 - M_{ij}) \cdot (-\infty)}_{\text{mask out forbidden positions}}\right)
$$

Positions masked to $-\infty$ receive zero weight after softmax, so the value vector at that position contributes nothing to the output.

Three canonical mask shapes cover the entire design space:

{{fig:archvar-mask-shapes}}

The prefix-LM mask is the union: prefix tokens attend to all other prefix tokens (full block in the top-left), and all tokens attend to earlier tokens causally. This lets the model build rich representations of the prompt before generating.

---

## Encoder-Only Models (BERT and kin)

### Architecture

An encoder-only model is a stack of $N$ transformer blocks where every block uses **fully bidirectional attention** — no causal masking at all. Every token can directly attend to every other token in the sequence. The original BERT (Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2019) stacked 12 (BERT-base) or 24 (BERT-large) such blocks.

{{fig:archvar-encoder-only-stack}}

Because every token sees every other token, the final hidden state at each position encodes **contextual meaning of that token within the full sequence**. This makes encoder representations excellent for tasks where you need to understand input meaning: classification, named entity recognition, question answering (extract the answer span), and natural language inference.

### Masked Language Model (MLM) Pre-training Objective

BERT is pre-trained with a **masked language model** objective. During training, 15% of input tokens are selected at random; of those, 80% are replaced with a `[MASK]` token, 10% with a random token, and 10% are left unchanged. The model must predict the original token at each masked position using the full surrounding context.

The loss is cross-entropy over the masked positions only:

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|}\sum_{i \in M} \log p_\theta(x_i \mid \tilde{x})
$$

where $\tilde{x}$ is the corrupted sequence and $M$ is the set of masked positions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mlm_mask(input_ids: torch.Tensor,
                   vocab_size: int,
                   mask_token_id: int,
                   mask_prob: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply BERT-style MLM masking to a batch of token ids.

    Args:
        input_ids: shape (B, T)
        vocab_size: size of vocabulary
        mask_token_id: id of the [MASK] token
        mask_prob: fraction of tokens to select for masking

    Returns:
        masked_input: (B, T) — input with some tokens replaced
        labels:       (B, T) — original ids at masked positions, -100 elsewhere
                               (-100 is ignored by F.cross_entropy)
    """
    B, T = input_ids.shape
    # Draw a Bernoulli mask: which positions are selected (15%)
    selected = torch.rand(B, T) < mask_prob          # (B, T) bool

    # Of the selected positions:
    #   80% → [MASK]
    #   10% → random token
    #   10% → unchanged (but still included in loss)
    rand_roll = torch.rand(B, T)
    replace_with_mask   = selected & (rand_roll < 0.80)
    replace_with_random = selected & (rand_roll >= 0.80) & (rand_roll < 0.90)
    # The rest (0.90–1.0) remain as original — no action needed

    masked_input = input_ids.clone()
    masked_input[replace_with_mask]   = mask_token_id
    masked_input[replace_with_random] = torch.randint(
        0, vocab_size, (replace_with_random.sum().item(),)
    )

    # Labels: original token at selected positions, -100 elsewhere
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[selected] = input_ids[selected]

    return masked_input, labels


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V) — raw logits over vocab
    labels: (B, T)    — original token ids at masked positions, -100 elsewhere
    """
    B, T, V = logits.shape
    # F.cross_entropy ignores positions where label == -100
    return F.cross_entropy(logits.view(B * T, V), labels.view(B * T))
```

The 80/10/10 split is deliberate: training the model to recover tokens it has *not* seen as `[MASK]` (the 10% unchanged and 10% random) prevents the representation from being artificially conditioned on the `[MASK]` token at inference time, when no masking occurs.

### What Encoder-Only Is Good For (and What It Cannot Do)

The bidirectional nature makes it powerful for understanding, but it makes **autoregressive generation** impossible. To generate the $(t+1)$-th token you would need to compute attention over a sequence that includes the $(t+1)$-th position, which you haven't generated yet — circular. Encoder-only models are therefore not language models in the generative sense; they are representation models. For tasks like:

- Text classification (use `[CLS]` head)
- Named entity recognition (use per-token heads)
- Semantic similarity / embedding (mean-pool or `[CLS]`)
- Extractive QA (span prediction)

encoder-only models remain highly competitive, especially when data is limited and pre-trained encoders can be fine-tuned on small supervised sets.

---

## Encoder-Decoder Models (T5, BART)

### Architecture

An encoder-decoder model (Vaswani et al., *Attention Is All You Need*, 2017) pairs a fully bidirectional **encoder** stack with an autoregressive **decoder** stack. The encoder processes the full input sequence once. The decoder generates output tokens one at a time, attending to (a) its own previously generated tokens via **causal self-attention** and (b) the encoder's output via **cross-attention**.

{{fig:archvar-encoder-decoder-crossattn}}

The cross-attention in each decoder block has:
- **Queries** from the decoder's own hidden state at the current step
- **Keys and Values** from the encoder's output $H_\text{enc}$

This means each decoder position can attend fully (bidirectionally) to the entire source sequence. The causal mask in the decoder self-attention ensures the decoder cannot look ahead at future *output* tokens.

### T5: Text-to-Text Transfer Transformer

Raffel et al. (*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, 2020) pre-trained T5 with a **span corruption** objective: random contiguous spans of input are masked with sentinel tokens (e.g., `<extra_id_0>`), and the model must reconstruct those spans autoregressively in the decoder.

```text
Input (encoder):  "The <extra_id_0> sat on the <extra_id_1> mat."
Target (decoder): "<extra_id_0> cat <extra_id_1> brown"
```

This framing is more efficient than BERT's MLM because the decoder only generates masked spans (typically 15% of tokens), not the full sequence. T5's key insight is to reframe *all* NLP tasks as text-to-text: translation, summarization, classification, QA — every task feeds a textual prompt and expects a textual output. This makes fine-tuning uniform.

```python
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Encoder-decoder cross-attention: queries come from decoder,
    keys/values come from encoder hidden states.
    Fully standard attention — no causal mask on the encoder side.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # from decoder
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # from encoder
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # from encoder
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self,
                decoder_hidden: torch.Tensor,   # (B, T_dec, D)
                encoder_hidden: torch.Tensor,   # (B, T_enc, D)
                encoder_mask: torch.Tensor | None = None  # (B, T_enc) bool
                ) -> torch.Tensor:
        B, T_dec, D = decoder_hidden.shape
        T_enc = encoder_hidden.shape[1]
        H = self.n_heads

        # Project, then reshape to (B, H, T, d_k)
        def split_heads(x: torch.Tensor, T: int) -> torch.Tensor:
            return x.view(B, T, H, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(decoder_hidden), T_dec)   # (B, H, T_dec, d_k)
        K = split_heads(self.W_k(encoder_hidden), T_enc)   # (B, H, T_enc, d_k)
        V = split_heads(self.W_v(encoder_hidden), T_enc)   # (B, H, T_enc, d_k)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # (B, H, T_dec, T_enc)

        if encoder_mask is not None:
            # encoder_mask: (B, T_enc) — True where token is PAD
            pad_mask = encoder_mask[:, None, None, :]    # broadcast over H, T_dec
            scores = scores.masked_fill(pad_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)             # (B, H, T_dec, T_enc)
        out  = attn @ V                                   # (B, H, T_dec, d_k)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T_dec, D)
        return self.W_o(out)
```

### BART: Denoising Pre-training

BART (Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*, 2019) uses a more general corruption scheme: token masking, deletion, text infilling, sentence permutation, and document rotation. The encoder-decoder architecture is the same as T5; BART excels at summarization and generation tasks where the output is a compressed or stylistically altered version of the input.

### Memory Footprint of Encoder-Decoder

A significant practical consideration: encoder-decoder models carry *two* full transformer stacks. T5-large has around 770M parameters split roughly evenly. During generation, the decoder must re-run cross-attention at every step and either recompute or cache the encoder hidden states. If the encoder output is cached, the memory cost scales as $B \times T_\text{enc} \times d_\text{model} \times N_\text{dec}$ bytes. For a T5-3B model with a 1 024-token source:

$$
\text{encoder KV cache} \approx 1024 \times 1024 \times 2 \times 24 \times 2\text{ bytes (fp16)} \approx 96\text{ MB per batch element}
$$

At batch size 32 that is roughly 3 GB just for cross-attention keys and values — comparable to the KV cache budget in a mid-sized decoder-only model.

---

## Decoder-Only Models (GPT family)

### Architecture

A decoder-only model is a stack of blocks that use only **causal (lower-triangular) self-attention**. There is no encoder, no cross-attention, no separate source sequence. The model receives a sequence of tokens and predicts each token from the tokens before it:

$$
p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})
$$

This is exactly the **autoregressive language model** factorization. The training objective is **causal language modeling** (CLM), also called next-token prediction:

$$
\mathcal{L}_{\text{CLM}} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
$$

{{fig:archvar-decoder-only-causal}}

At inference time you feed a **prompt** (called the *prefix* or *context*), run the forward pass, sample or argmax the next token from the final logit, append it, and repeat. This is the KV-cache decode loop covered in detail in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

### From-Scratch Minimal Decoder-Only Transformer

The following is a self-contained, heavily commented decoder-only transformer that you can run. It intentionally omits optimizations (FlashAttention, fused kernels) to be readable.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k    = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        # Fused QKV projection for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Register causal mask as a buffer (not a parameter)
        # Lower-triangular: M[i,j]=1 iff j <= i
        causal_mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", causal_mask)  # (T_max, T_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k

        # Compute Q, K, V via fused projection, then split
        qkv = self.qkv(x)                       # (B, T, 3D)
        Q, K, V = qkv.split(D, dim=-1)          # each (B, T, D)

        # Reshape to (B, H, T, d_k) for multi-head attention
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, d_k).transpose(1, 2)

        Q, K, V = map(reshape, (Q, K, V))

        # Scaled dot-product attention with causal mask
        scores = Q @ K.transpose(-2, -1) * (d_k ** -0.5)   # (B, H, T, T)

        # Apply causal mask: positions where mask==False get -inf
        mask = self.causal_mask[:T, :T]          # (T, T)
        scores = scores.masked_fill(~mask, float('-inf'))

        attn  = F.softmax(scores, dim=-1)        # (B, H, T, T)
        out   = attn @ V                          # (B, H, T, d_k)

        # Merge heads: (B, H, T, d_k) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (decoder-only, no cross-attention)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 2048):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.attn   = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2  = nn.LayerNorm(d_model)
        # Feed-forward: expand to 4x, then contract
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalize before the sub-layer, add residual after
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Minimal GPT-style decoder-only language model.
    Uses learned absolute positional embeddings (GPT-2 style).
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int   = 256,
                 n_heads: int   = 4,
                 n_layers: int  = 6,
                 max_seq_len: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # learned pos emb

        d_ff = 4 * d_model
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm_f  = nn.LayerNorm(d_model)             # final norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and LM head share weights (saves params + improves quality)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) — token ids
        returns logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.tok_emb(idx) + self.pos_emb(positions)    # (B, T, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        return self.lm_head(x)                              # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 1.0) -> torch.Tensor:
        """Greedy/temperature sampling. Prompt: (1, T_prompt)."""
        for _ in range(max_new_tokens):
            logits = self.forward(prompt)[:, -1, :]         # (1, vocab_size)
            logits = logits / temperature
            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            prompt = torch.cat([prompt, next_tok], dim=1)
        return prompt


# --- Quick sanity check ---
if __name__ == "__main__":
    model = DecoderOnlyTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=4)
    x = torch.randint(0, 1000, (2, 32))   # batch=2, seq_len=32
    logits = model(x)
    print(f"Output shape: {logits.shape}")   # should be (2, 32, 1000)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
```

---

## Prefix Language Models

### What Is a Prefix-LM?

A **prefix language model** (prefix-LM) is a decoder-only model with a modified attention mask: the tokens belonging to the *input prompt* (the "prefix") attend to each other **bidirectionally**, while the tokens being *generated* attend causally. The mask is the block-diagonal hybrid we showed earlier.

This was the design used in models like **ULMFiT**-adjacent work and, more prominently, **PaLM** (Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022) in its fine-tuning phase, and was described explicitly in Raffel et al.'s T5 paper as a baseline worth studying. Google's **Gemini** architecture is also described as prefix-LM.

### Constructing the Prefix-LM Mask

```python
def make_prefix_lm_mask(prefix_len: int, total_len: int) -> torch.Tensor:
    """
    Returns additive mask for prefix-LM.
    Shape: (total_len, total_len)
    Value: 0 where attention is allowed, -inf where it is blocked.

    The prefix portion (positions 0..prefix_len-1) is fully bidirectional.
    Positions >= prefix_len are causal.
    """
    T = total_len
    # Start with a full causal mask
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))  # lower triangular

    # For the prefix block: all prefix positions can see all other prefix positions
    # (i.e., set the top-left sub-matrix to True)
    mask[:prefix_len, :prefix_len] = True

    # Convert bool mask to additive float mask: True→0, False→-inf
    additive_mask = torch.zeros(T, T, dtype=torch.float32)
    additive_mask[~mask] = float('-inf')
    return additive_mask


# Visualise for prefix_len=3, total_len=6
mask = make_prefix_lm_mask(3, 6)
for row in mask.tolist():
    print(" ".join("  0" if v == 0.0 else "-∞" for v in row))
```

```text
  0   0   0 -∞ -∞ -∞
  0   0   0 -∞ -∞ -∞
  0   0   0 -∞ -∞ -∞
  0   0   0   0 -∞ -∞
  0   0   0   0   0 -∞
  0   0   0   0   0   0
```

Prefix tokens (rows 0–2) can see all other prefix tokens but cannot see future generated tokens (the right-side −∞ values). Generated tokens (rows 3–5) can see the full prefix bidirectionally and all previously generated tokens, but not future generated tokens.

### Why Bother? The Tradeoff

| Property | Decoder-only (causal) | Prefix-LM | Encoder-Decoder |
|---|---|---|---|
| Prefix representation | Causal (sees only earlier prefix tokens) | Bidirectional | Fully bidirectional |
| Can generate autoregressively | Yes | Yes | Yes (decoder side) |
| Single model, no cross-attention overhead | Yes | Yes | No — two stacks |
| Fine-tuning simplicity | Simple | Simple | Moderate |
| Representation quality for encoding | Lower | Higher | Highest |

The key benefit of prefix-LM over pure causal: the model builds a richer contextual representation of the prompt before generating. The key benefit over encoder-decoder: there is only one model with shared weights, no cross-attention, and you can continue generating past the prefix boundary seamlessly.

The key limitation: during pre-training on pure text data, you must decide the prefix boundary. If you pre-train only with causal masking (standard), switching to prefix-LM masking at fine-tuning time creates a distribution mismatch: the model has never seen bidirectional attention during training, so the first few fine-tuning steps are spent adapting. Some models pre-train with randomly sampled prefix lengths to avoid this.

---

## Masking Patterns: A Worked Numerical Example

!!! example "Worked Example: Attention Scores Under Different Masks"

    Consider a tiny sequence of length $T=4$ with $d_k = 4$. Suppose after the QK projection we have (for a single head):

    $$
    L = \begin{bmatrix} 1.0 & 0.5 & 0.8 & 0.3 \\ 0.4 & 1.2 & 0.6 & 0.9 \\ 0.7 & 0.3 & 1.1 & 0.5 \\ 0.2 & 0.8 & 0.4 & 1.3 \end{bmatrix}
    $$

    **Bidirectional (encoder):** No masking. Softmax of each row. Row 0:
    $$\text{softmax}([1.0, 0.5, 0.8, 0.3]) = [0.38, 0.23, 0.31, 0.18]$$
    Token 0's representation is a blend of all four value vectors.

    **Causal (decoder):** Apply lower-triangular mask. For row 0 (query = position 0), only position 0 is visible:
    $$L'_{0,:} = [1.0, -\infty, -\infty, -\infty] \xrightarrow{\text{softmax}} [1.0, 0, 0, 0]$$
    Position 0's output is exactly its own value vector — it cannot see the future. For row 3:
    $$\text{softmax}([0.2, 0.8, 0.4, 1.3]) = [0.14, 0.27, 0.17, 0.42]$$
    Position 3 blends all four — it benefits from full left-context.

    **Prefix-LM with prefix length 2:** Rows 0 and 1 can see all other prefix positions (columns 0–1) bidirectionally:
    - Row 0: $\text{softmax}([1.0, 0.5, -\infty, -\infty]) = [0.62, 0.38, 0, 0]$
    - Row 2 (generation starts): causal from here, $\text{softmax}([0.7, 0.3, 1.1, -\infty]) = [0.26, 0.18, 0.56, 0]$

    Notice: prefix-LM gives prefix tokens *better* representations than pure causal (they mix with each other fully), while generation tokens remain strictly causal.

---

## The Decoder-Only Convergence

### Why the Field Moved to Decoder-Only

By 2022–2023, essentially all frontier models (GPT-3, PaLM, LLaMA, Mistral, Gemma, Claude, Gemini) converged on decoder-only architectures. This was not obvious a priori — encoder-decoder models like T5 showed strong results on many benchmarks, and encoder-only models like BERT dominated NLP for years. What drove the shift?

**1. Unified training objective.** Causal language modeling on raw text is simple, abundant, and self-supervised at planetary scale. There is no need to decide what to mask, how to construct pairs, or how to label anything. The training data is just the internet.

**2. In-context learning emerges naturally.** Because the prompt and the completion are treated identically as a sequence, the decoder naturally learns to condition its completions on the prompt. Few-shot learning (Brown et al., *Language Models are Few-Shot Learners*, 2020) arises as a capability from scale. Encoder-decoder models can do this too, but the mechanism is less seamless.

**3. Scalability.** As the context window grows (8K → 128K → 1M tokens), having a single attention stack is simpler than coordinating encoder length vs. decoder length. Cross-attention becomes more expensive as source length grows.

**4. RLHF / instruction tuning / preference optimization.** Post-training pipelines (see [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) and [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)) are straightforward to set up with a causal decoder: format inputs and outputs as a single sequence and compute the causal LM loss only on the output portion.

**5. KV cache simplicity.** The KV cache for inference (see [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)) is a single cache for a single stack. Encoder-decoder models require a separate cross-attention KV cache per decoder layer.

**What encoder-only is still good for.** Representation tasks with tight latency budgets: search re-ranking, embedding retrieval, token classification. A 110M-parameter BERT encoder produces high-quality contextual embeddings orders of magnitude cheaper than running a 70B decoder-only model.

**What encoder-decoder is still good for.** Constrained generation tasks where the output vocabulary is small relative to the input (document summarization with a known schema, structured extraction, code generation conditioned on long specs). The encoder-decoder can build a much richer representation of the source, which can matter when the source is long and the generation is short.

---

## Comparing the Pre-training Objectives

The pre-training objective is not just a loss function — it determines what the model learns to represent. Let us compare the three main objectives concretely.

| Objective | Task | Input to model | Loss computed over | Model type |
|---|---|---|---|---|
| Masked LM (MLM) | Predict masked tokens | Corrupted full sequence | Masked positions only | Encoder-only |
| Span corruption | Reconstruct masked spans | Corrupted encoder input | Full decoder output | Encoder-decoder |
| Causal LM (CLM) | Predict next token | All preceding tokens | All positions | Decoder-only |

A key efficiency point: CLM trains on **every position** in every sequence, while MLM trains only on the ~15% masked positions. This means that for a given sequence of $T$ tokens, CLM extracts $T$ gradient signals while MLM extracts only $\approx 0.15T$. Over a fixed compute budget, CLM sees more learning signal per FLOP on raw generation ability. MLM produces better per-token representations for discrimination tasks, but that advantage diminishes with scale.

{{fig:archvar-objective-signal-comparison}}

!!! note "The ELECTRA Alternative"
    Clark et al. (*ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators*, 2020) observed that MLM wastes compute on easy-to-predict unmasked tokens. ELECTRA uses a small generator to fill in masks, then trains a large discriminator to detect which tokens are "replaced." The discriminator trains on *all* tokens, achieving BERT-level performance at substantially lower compute. This is still encoder-only but with a more efficient objective.

---

## Masking in Practice: Implementation Gotchas

Getting the mask right in code is where many practitioners make silent mistakes. Here is a consolidated reference for the four patterns you will encounter.

```python
import torch


def causal_mask(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Standard lower-triangular causal mask. Returns (T, T) bool mask."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def bidirectional_mask(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """All-ones mask — every position attends to every position."""
    return torch.ones(T, T, dtype=torch.bool, device=device)


def prefix_lm_mask(prefix_len: int, total_len: int,
                   device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Prefix-LM mask: bidirectional within prefix, causal for generation."""
    mask = causal_mask(total_len, device)
    mask[:prefix_len, :prefix_len] = True   # full bidirectional for prefix block
    return mask


def encoder_decoder_mask(T_dec: int, T_enc: int,
                          pad_mask: torch.Tensor | None = None,
                          device: torch.device = torch.device("cpu")) -> dict:
    """
    Returns the two masks needed for an encoder-decoder model:
      - 'self':  causal mask for decoder self-attention  (T_dec, T_dec)
      - 'cross': encoder padding mask for cross-attention (T_enc,) bool: True=PAD
    """
    self_mask  = causal_mask(T_dec, device)
    cross_mask = pad_mask if pad_mask is not None else torch.zeros(T_enc, dtype=torch.bool, device=device)
    return {"self": self_mask, "cross": cross_mask}


# --- Common gotcha: using the wrong dtype ---
# torch.where and masked_fill expect a *bool* mask, not float.
# torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+) expects
# an *additive* attn_mask (float, 0 or -inf), NOT a bool mask.
# Always double-check which convention your attention function uses.

def apply_causal_mask_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                           ) -> torch.Tensor:
    """
    PyTorch 2.0+ scaled_dot_product_attention with causal mask.
    The is_causal=True flag efficiently generates the causal mask internally,
    avoiding the O(T^2) mask tensor allocation.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        is_causal=True,    # ← most efficient way to do causal masking in PyTorch 2+
        dropout_p=0.0,
    )
```

!!! warning "Mask Convention Mismatch"
    PyTorch's `nn.MultiheadAttention` uses a `key_padding_mask` where `True` means *ignore* (the opposite of "allowed"), while its `attn_mask` is an *additive* mask where $-\infty$ means blocked. `F.scaled_dot_product_attention` (PyTorch 2.0+) also uses an additive `attn_mask` and has `is_causal` as a shortcut. HuggingFace Transformers internally converts bool "attention masks" (1=attend, 0=ignore) into additive masks. Mixing these conventions is the most common source of silent accuracy bugs when implementing custom attention modules.

---

## Interview Corner

!!! interview "Interview Corner"
    **Q:** What is the fundamental difference between BERT and GPT architectures, and when would you choose one over the other in a production system?

    **A:** The core difference is the attention mask. BERT uses a **fully bidirectional** mask — every token attends to every other token — making it optimal for *understanding* tasks where you have the complete input available. GPT uses a **causal (lower-triangular)** mask so that token $t$ only attends to tokens $0, \ldots, t-1$, enabling **autoregressive generation**: you can extend the sequence one token at a time.

    Choose BERT-style when you need high-quality representations of fixed-length inputs: text classification, named entity recognition, extractive QA, semantic search embeddings. These tasks benefit from the richer per-token context from both directions. Choose GPT-style when you need to **generate** text — summarization, dialogue, code completion, instruction following — or when you want a single unified model that can handle both understanding and generation via prompting. For production deployment, decoder-only models also have a simpler KV-cache story: one cache, one stack, no cross-attention overhead.

    A nuance: both architectures can do retrieval if you pool the hidden states. But a fine-tuned BERT encoder at 110M parameters will produce better embeddings at lower latency than a 7B decoder-only model for the same compute budget, which is why bi-encoder retrieval systems still frequently use BERT-family models.

    **Follow-up:** Why can't you use BERT for generation? Because to predict token $t$, BERT's bidirectional attention would need to attend to token $t$ itself (it is in the input), which leaks the answer. You could run BERT autoregressively by masking future tokens, but then you are re-running the full encoder at every step, which is expensive, and you lose the bidirectional context benefit that justified using an encoder in the first place.

---

## Key Takeaways

!!! key "Key Takeaways"
    - The **attention mask** is the defining characteristic of each architecture family. Encoder-only = fully bidirectional; decoder-only = lower-triangular causal; encoder-decoder = bidirectional encoder + causal decoder + cross-attention; prefix-LM = hybrid.
    - **BERT (encoder-only)** trains with masked language modeling, gives the best per-token representations for fixed-length inputs, but cannot generate autoregressively.
    - **T5/BART (encoder-decoder)** trains with span corruption or denoising; excels at sequence-to-sequence tasks; carries the cost of two stacks and cross-attention KV caches.
    - **GPT (decoder-only)** trains with causal LM on every position, making it compute-efficient at scale; naturally handles both conditioning and generation via the context window.
    - **Prefix-LM** is a middle ground: bidirectional attention over the prompt, causal attention over the generation. It improves prompt representation at the cost of a distribution mismatch if the model was pre-trained purely causally.
    - The field converged on **decoder-only** for frontier models because: (1) CLM trains on all positions (more signal per FLOP), (2) in-context learning arises naturally from the sequence-continuation framing, (3) post-training pipelines (SFT, RLHF) are simpler, and (4) a single KV cache is operationally cleaner at inference.
    - Encoder-only models remain competitive for **embedding and classification** workloads where latency and cost matter more than generation capability.
    - A silent bug when implementing custom attention: mask conventions differ between PyTorch's `MultiheadAttention`, `F.scaled_dot_product_attention`, and HuggingFace — always verify whether `True` means "attend" or "ignore."
    - At large scale the distinction between architectures blurs: a decoder-only model with a very long context window and prompt caching behaves similarly to an encoder-decoder in many retrieval-augmented workloads.

---

!!! sota "State of the Art & Resources (2026)"
    Decoder-only transformers dominate frontier LLMs (GPT-4, Claude, Llama, Mistral, Gemini), while encoder-only models remain the backbone of fast retrieval and classification systems; encoder-decoder architectures are seeing a resurgence for parameter-efficient small models and structured generation tasks.

    **Foundational work**

    - [Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)](https://arxiv.org/abs/1810.04805) — defined the encoder-only paradigm and bidirectional masked language modeling.
    - [Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2020)](https://arxiv.org/abs/1910.10683) — T5 unified all NLP as text-to-text via span corruption and encoder-decoder design.
    - [Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training* (2019)](https://arxiv.org/abs/1910.13461) — generalized corruption objectives for the encoder-decoder family; excel at summarization.

    **Recent advances (2023–2026)**

    - [Jiang et al., *Mistral 7B* (2023)](https://arxiv.org/abs/2310.06825) — compact decoder-only model introducing grouped-query attention and sliding window attention; a reference design for efficient causal LMs.
    - [Warner et al., *ModernBERT: Smarter, Better, Faster, Longer* (2024)](https://arxiv.org/abs/2412.13663) — modernized encoder-only model with RoPE, FlashAttention, and 8 192-token context; the current state of the art for bidirectional encoders.
    - [Weller et al., *Return of the Encoder: Maximizing Parameter Efficiency for SLMs* (2025)](https://arxiv.org/abs/2501.16273) — shows encoder-decoder architectures achieve 47 % lower first-token latency and 4.7× higher throughput than decoder-only models at small parameter budgets.

    **Open-source & tools**

    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — minimal (~300-line) decoder-only GPT implementation; the canonical readable reference for the causal transformer.
    - [huggingface/transformers](https://github.com/huggingface/transformers) — hosts production implementations of BERT, T5, BART, GPT-2, Llama, Mistral, and more; the standard starting point for all three architecture families.

    **Go deeper**

    - [von Platen, *Transformer-based Encoder-Decoder Models* (Hugging Face Blog, 2020)](https://huggingface.co/blog/encoder-decoder) — thorough walkthrough of encoder, decoder, and cross-attention mechanics with code.
    - [Grattafiori et al., *The Llama 3 Herd of Models* (2024)](https://arxiv.org/abs/2407.21783) — detailed account of a production-scale decoder-only architecture, covering tokenization, attention, and scaling from 8 B to 405 B parameters.

## Further Reading

- Devlin, Chang, Lee, Toutanova. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019.
- Raffel, Shazeer, Roberts, et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*. JMLR 2020.
- Lewis, Liu, Goyal, et al. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020.
- Brown, Mann, Ryder, et al. *Language Models are Few-Shot Learners (GPT-3)*. NeurIPS 2020.
- Clark, Luong, Le, Manning. *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators*. ICLR 2020.
- Chowdhery, Narang, Devlin, et al. *PaLM: Scaling Language Modeling with Pathways*. JMLR 2023.
- Vaswani, Shazeer, Parmar, et al. *Attention Is All You Need*. NeurIPS 2017.
- Touvron, Lavril, Izacard, et al. *LLaMA: Open and Efficient Foundation Language Models*. arXiv 2023. (Exemplary decoder-only design at scale.)
- Andrej Karpathy. *nanoGPT* (GitHub). Reference implementation of a minimal decoder-only transformer.

---

## Exercises

**1.** (Conceptual) A colleague proposes taking a pre-trained BERT (encoder-only) model and using it to generate text one token at a time, exactly like GPT: feed the prompt, read the logits at the last position, sample a token, append it, and repeat. Explain precisely why this does not give you the same behaviour as a decoder-only model, referring to the attention mask. What are the *two* distinct problems?

??? note "Solution"
    The problem is the mask. Every BERT block uses a **fully bidirectional** mask, so the hidden state at any position $i$ is a function of *all* tokens in the input window, including tokens at positions $> i$.

    **Problem 1 — information leakage during training vs. inference mismatch.** BERT was never trained to predict a token from left context alone. Its representations are optimized to reconstruct a masked token using both sides. If you run it autoregressively and read the last-position logits, that position attended only to tokens that already exist (the prompt so far), which is exactly the regime BERT was *not* trained in for that head. The MLM head predicts a token *given that the position is a `[MASK]` surrounded by real context on both sides*; a bare next-position prediction is out of distribution.

    **Problem 2 — no causal structure means no cheap incremental decoding, and full recomputation.** To extend the sequence you must re-run the *entire* bidirectional stack over the whole sequence at every step, because a bidirectional model has no valid KV cache: adding a new token on the right changes the attention (and hence the hidden states) of *every earlier position*, since earlier positions are allowed to attend rightward. A causal model does not have this problem — earlier positions never see the new token, so their K/V entries are frozen and cacheable. So BERT-as-generator is both statistically wrong (out-of-distribution) and computationally $O(T)$ times more expensive per generated token.

    This is exactly the point made in the chapter's Interview Corner follow-up: to predict token $t$ bidirectionally you would need to attend to token $t$ itself, which leaks the answer, so you must mask future tokens — at which point you have thrown away the bidirectional context that justified using an encoder.

**2.** (Quantitative) Consider a single attention head over a sequence of length $T = 4$. For the **query at position 1** (0-indexed), the pre-softmax logits over the four key positions are

$$
L_{1,:} = [\,2.0,\; 3.0,\; 0.0,\; 1.0\,].
$$

Using $e^{2}\approx 7.389$, $e^{3}\approx 20.086$, $e^{1}\approx 2.718$, $e^{0}=1$, compute the attention weights this query places on each key under (a) an **encoder** (bidirectional) mask, (b) a **decoder** (causal) mask, and (c) a **prefix-LM** mask with `prefix_len = 3`. Round to three decimals.

??? note "Solution"
    We softmax the *visible* logits; masked keys get weight 0.

    **(a) Bidirectional — all four keys visible.**
    $$\text{sum} = 7.389 + 20.086 + 1 + 2.718 = 31.193$$
    $$A_{1,:} = \left[\tfrac{7.389}{31.193},\ \tfrac{20.086}{31.193},\ \tfrac{1}{31.193},\ \tfrac{2.718}{31.193}\right] = [\,0.237,\ 0.644,\ 0.032,\ 0.087\,].$$

    **(b) Causal — query at position 1 sees only keys $0$ and $1$.** Keys 2 and 3 are $-\infty$.
    $$\text{sum} = 7.389 + 20.086 = 27.475$$
    $$A_{1,:} = \left[\tfrac{7.389}{27.475},\ \tfrac{20.086}{27.475},\ 0,\ 0\right] = [\,0.269,\ 0.731,\ 0,\ 0\,].$$

    **(c) Prefix-LM, `prefix_len = 3`.** Position 1 lies inside the prefix (positions 0,1,2), so within the prefix block it attends *bidirectionally* — it can see keys 0, 1, and 2 — but key 3 is a generation token and stays masked.
    $$\text{sum} = 7.389 + 20.086 + 1 = 28.475$$
    $$A_{1,:} = \left[\tfrac{7.389}{28.475},\ \tfrac{20.086}{28.475},\ \tfrac{1}{28.475},\ 0\right] = [\,0.259,\ 0.705,\ 0.035,\ 0\,].$$

    Notice the trend from the chapter's worked example: prefix-LM lets this prefix token pull in key 2 (which the strictly causal model could not see), giving it a richer representation, while it still refuses to peek at the future generation token 3.

**3.** (Quantitative) The chapter's `make_prefix_lm_mask` starts from a full causal mask and then fills in the top-left prefix block. Derive a closed-form expression for the total number of *allowed* (query, key) attention pairs in a prefix-LM mask of total length $T$ with prefix length $p$. Then evaluate it for $T = 1024,\ p = 256$, and confirm your formula reproduces the count for the chapter's printed $T=6,\ p=3$ example.

??? note "Solution"
    A pure **causal** mask allows key $j \le i$, i.e. the lower triangle including the diagonal:
    $$\text{causal pairs} = \frac{T(T+1)}{2}.$$

    The prefix-LM mask additionally turns on the *upper* triangle of the top-left $p \times p$ block (the entries where query $i < p$, key $j < p$, and $j > i$ — the ones causal masking had left off). The number of such strictly-above-diagonal entries in a $p \times p$ block is
    $$\frac{p(p-1)}{2}.$$

    So the total allowed pairs are
    $$\boxed{\ \frac{T(T+1)}{2} + \frac{p(p-1)}{2}\ }.$$

    **Evaluate $T=1024,\ p=256$:**
    $$\frac{1024\cdot 1025}{2} = 524{,}800, \qquad \frac{256\cdot 255}{2} = 32{,}640,$$
    $$\text{total} = 524{,}800 + 32{,}640 = 557{,}440 \text{ allowed pairs}.$$

    **Check against $T=6,\ p=3$:**
    $$\frac{6\cdot 7}{2} + \frac{3\cdot 2}{2} = 21 + 3 = 24.$$
    Counting the `0` entries in the chapter's printed mask row by row gives $3+3+3+4+5+6 = 24$. The formula matches.

**4.** (Quantitative) The chapter notes that causal LM (CLM) extracts a learning signal at *every* position, while masked LM (MLM) computes loss only over the selected positions. For a training corpus fed in sequences of length $T = 512$ using BERT's default `mask_prob = 0.15`: (a) how many loss-contributing positions does each objective produce per sequence, and what is the ratio? (b) If both models are trained for the same number of sequences, roughly how many more supervised token-predictions does CLM see? (c) Give one reason MLM is nonetheless sometimes preferred despite this efficiency gap.

??? note "Solution"
    **(a)** CLM computes a next-token loss at all $T$ positions:
    $$\text{CLM signals} = T = 512 \text{ per sequence.}$$
    MLM computes loss only over the selected (~15%) positions (in the chapter's `apply_mlm_mask`, `labels[selected] = ...` covers *all* selected tokens, including the 10% left unchanged):
    $$\text{MLM signals} = 0.15 \times 512 = 76.8 \approx 77 \text{ per sequence.}$$
    $$\text{ratio} = \frac{512}{76.8} \approx 6.67.$$

    **(b)** For the same number of sequences, CLM produces about **6.7$\times$** as many token-prediction gradient signals. Over, say, 1 million sequences that is $512\text{M}$ CLM predictions vs. $\approx 76.8\text{M}$ MLM predictions — roughly $435$ million *more* supervised predictions for CLM.

    **(c)** MLM's per-token representations are **bidirectional** — each prediction is conditioned on both left and right context — which produces higher-quality contextual embeddings for *discrimination* tasks (classification, NER, retrieval) than causal representations that see only left context. The chapter notes this advantage is real but "diminishes with scale." (An orthogonal fix is ELECTRA's replaced-token-detection objective, which recovers the all-positions signal while keeping bidirectionality.)

**5.** (Implementation) The chapter ships a `CausalSelfAttention` module with the causal mask hard-wired into a buffer. Refactor it into a single **mask-agnostic** `MaskedSelfAttention` module whose `forward` accepts an explicit boolean mask of shape `(T, T)` (`True` = allowed), so that the *same* module can implement encoder (bidirectional), decoder (causal), and prefix-LM attention just by passing a different mask. Then, using the chapter's `bidirectional_mask`, `causal_mask`, and `prefix_lm_mask` helpers, show that all three run and that the causal and prefix-LM outputs differ. Keep the chapter's fused-QKV, multi-head style.

??? note "Solution"
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class MaskedSelfAttention(nn.Module):
        """Multi-head self-attention that takes the mask as a forward argument.
        mask: (T, T) bool, True where attention is allowed."""

        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            self.d_k     = d_model // n_heads
            self.n_heads = n_heads
            self.d_model = d_model
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            x:    (B, T, D)
            mask: (T, T) bool, True = allowed to attend
            returns: (B, T, D)
            """
            B, T, D = x.shape
            H, d_k = self.n_heads, self.d_k

            qkv = self.qkv(x)                     # (B, T, 3D)
            Q, K, V = qkv.split(D, dim=-1)        # each (B, T, D)

            def reshape(t: torch.Tensor) -> torch.Tensor:
                return t.view(B, T, H, d_k).transpose(1, 2)   # (B, H, T, d_k)

            Q, K, V = map(reshape, (Q, K, V))

            scores = Q @ K.transpose(-2, -1) * (d_k ** -0.5)  # (B, H, T, T)

            # Broadcast (T, T) mask over batch and heads; forbidden -> -inf.
            scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))

            attn = F.softmax(scores, dim=-1)      # (B, H, T, T)
            out  = attn @ V                        # (B, H, T, d_k)
            out  = out.transpose(1, 2).contiguous().view(B, T, D)
            return self.out(out)


    # --- Reuse the chapter's mask helpers ---
    def causal_mask(T, device=torch.device("cpu")):
        return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    def bidirectional_mask(T, device=torch.device("cpu")):
        return torch.ones(T, T, dtype=torch.bool, device=device)

    def prefix_lm_mask(prefix_len, total_len, device=torch.device("cpu")):
        m = causal_mask(total_len, device)
        m[:prefix_len, :prefix_len] = True
        return m


    if __name__ == "__main__":
        torch.manual_seed(0)
        B, T, D, H = 1, 6, 32, 4
        attn = MaskedSelfAttention(D, H)
        x = torch.randn(B, T, D)

        out_bi     = attn(x, bidirectional_mask(T))
        out_causal = attn(x, causal_mask(T))
        out_prefix = attn(x, prefix_lm_mask(3, T))

        print("shapes:", out_bi.shape, out_causal.shape, out_prefix.shape)
        # Position 0 under causal sees only itself; under bidirectional it sees all
        # -> the row-0 outputs must differ.
        print("bi vs causal differ at pos 0 :",
              not torch.allclose(out_bi[:, 0], out_causal[:, 0], atol=1e-6))
        # Prefix (prefix_len=3) lets rows 0..2 attend bidirectionally within the
        # prefix, so early rows differ from strictly-causal; row 5 is identical
        # (last position is causal-visible-to-all under both masks).
        print("causal vs prefix differ at pos 1:",
              not torch.allclose(out_causal[:, 1], out_prefix[:, 1], atol=1e-6))
        print("causal vs prefix same   at pos 5:",
              torch.allclose(out_causal[:, 5], out_prefix[:, 5], atol=1e-6))
    ```

    Expected output:

    ```text
    shapes: torch.Size([1, 6, 32]) torch.Size([1, 6, 32]) torch.Size([1, 6, 32])
    bi vs causal differ at pos 0 : True
    causal vs prefix differ at pos 1: True
    causal vs prefix same   at pos 5: True
    ```

    The single module now realizes all three architecture families. The only thing that changed between an encoder, a decoder, and a prefix-LM is the boolean mask handed to `forward` — which is exactly the chapter's central thesis: *the mask is the architecture*. Note that position 5 (the last token) yields identical causal and prefix-LM outputs because under both masks it is allowed to attend to every earlier position; the masks only diverge inside the prefix block.
