# 3.3 The Pretraining Objective & Loss

Before a language model can answer questions, write code, or follow instructions, it must first acquire a dense statistical model of language. That foundation is laid during pretraining, and the entire multi-trillion-token, multi-thousand-GPU campaign is guided by a single scalar number: the **cross-entropy loss** on the next-token prediction task. Understanding that loss — what it measures, why it is chosen, how it is computed in practice, and what its values mean — is arguably the most important prerequisite for every topic in the rest of this book.

This chapter dissects the pretraining objective from first principles. We cover the probabilistic framing, the exact mechanics of teacher forcing, the engineering of causal masking and label shifting, loss masking strategies for packing and multi-document batches, the bits-per-byte normalization, and the UL2 family of span-corruption alternatives. Throughout, we ground everything in concrete code you can run today.

---

## Why Next-Token Prediction?

A language model assigns probabilities to sequences. Given a vocabulary $\mathcal{V}$ of size $V$, we want to model the probability of a sequence $x = (x_1, x_2, \ldots, x_T)$. By the chain rule of probability this factorizes exactly:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

There is no approximation here — this is a theorem. It says that modeling a joint distribution over sequences is equivalent to modeling a sequence of conditional distributions, each predicting the *next* token given all previous tokens. A decoder-only transformer with a causal mask does exactly this.

Why this objective rather than, say, a reconstruction autoencoder or a contrastive loss?

1. **Every token in a long sequence contributes signal.** A 2048-token sequence yields 2047 gradient-carrying predictions per example, making pretraining extraordinarily data-efficient compared to objectives that emit one signal per sequence.
2. **It is a proper scoring rule.** Maximizing log-likelihood under the true data distribution is provably equivalent to minimizing the KL divergence from the model distribution to the data distribution. The model is incentivized to be calibrated, not just accurate on a held-out label.
3. **Scalability.** The computation is embarrassingly parallelizable across the time dimension during training (unlike autoregressive inference). The forward and backward passes can be batched over all positions simultaneously.
4. **Generality.** Nothing about this objective is task-specific. A model that can predict the next token of code, math, natural language, and structured data simultaneously acquires cross-domain representations that transfer well.

The connection to tokenization and vocabulary is covered in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html). Scaling behavior of this loss is analyzed in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).

---

## The Cross-Entropy Loss and Negative Log-Likelihood

### Formal Definition

Let $f_\theta : \mathcal{V}^{<t} \to \mathbb{R}^V$ be the transformer that maps a prefix to logits over the vocabulary. The model's predicted probability at position $t$ is:

$$
\hat{p}(x_t \mid x_{<t}) = \operatorname{softmax}(f_\theta(x_{<t}))_{x_t}
$$

The **negative log-likelihood (NLL)** loss for a single sequence of length $T$ is:

$$
\mathcal{L}_{\text{NLL}} = -\frac{1}{T} \sum_{t=1}^{T} \log \hat{p}(x_t \mid x_{<t})
$$

This is numerically identical to the categorical **cross-entropy** between the one-hot target distribution and the predicted softmax distribution:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{v \in \mathcal{V}} \mathbb{1}[x_t = v] \cdot \log \hat{p}(v \mid x_{<t})
$$

The inner sum collapses because all probability mass in the one-hot target is on the true token $x_t$.

### The Information-Theoretic Viewpoint

The per-token loss equals the **cross-entropy** $H(p_\text{data}, p_\theta)$. By decomposition:

$$
H(p_\text{data}, p_\theta) = H(p_\text{data}) + D_{\text{KL}}(p_\text{data} \| p_\theta)
$$

The entropy $H(p_\text{data})$ of the true data distribution is a constant. So minimizing cross-entropy is the same as minimizing the KL divergence from the model to the data. The irreducible entropy of natural language — roughly 1–1.5 bits per character or on the order of 2–4 nats per token for English — sets a hard lower bound the model can never beat.

**Perplexity** (PPL) is the exponentiated average loss:

$$
\text{PPL} = \exp\!\left(\mathcal{L}_{\text{NLL}}\right)
$$

Perplexity has the intuitive interpretation of the effective branching factor: a perplexity of 10 means the model is on average as uncertain as a uniform distribution over 10 equiprobable tokens. Well-pretrained 70B-class models reach perplexities on the order of 3–8 on standard benchmarks such as WikiText-103, depending on tokenizer and evaluation setup. (Always compare perplexity numbers only within the same tokenizer, as vocabulary size strongly affects the value.)

{{fig:ptobj-crossentropy-entropy-floor}}

---

## Teacher Forcing and the Causal Mask

### What is Teacher Forcing?

During training we use **teacher forcing**: at every position $t$ we feed the *ground-truth* token $x_{t-1}$ as input, not the model's own prediction $\hat{x}_{t-1}$. This is what makes the training forward pass efficient — all positions can be computed simultaneously rather than sequentially.

Without teacher forcing (scheduled sampling or curriculum approaches), each prediction at time $t$ depends on potentially erroneous earlier predictions, causing gradient signal to be noisy and training to be slow. Teacher forcing decouples positions and allows a single GPU kernel to produce all $T$ predictions in one shot.

### The Causal Mask

The quid pro quo for teacher forcing is that position $t$ must not be allowed to see tokens $x_{t+1}, \ldots, x_T$ — otherwise the model would trivially predict the next token by looking at it. This is enforced by the **causal (lower-triangular) attention mask**:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}
$$

Applied before the softmax in each attention layer, entries with $-\infty$ vanish after exponentiation, making the attention weight exactly zero. Position $i$ can only attend to positions $\le i$ (including itself).

The causal mask is discussed in detail with implementation in [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) and [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html).

```text
Input sequence:  [BOS] T h e   c a t   s a t
Positions:         0   1 2 3 4 5 6 7 8 9 10 11

Teacher-forced input  (i-th logit predicts token i+1):
 input:  [BOS] T h e   c a t   s a
 target:  T   h e     c a t   s a t

Causal mask (T=6 shown, ✓=attend, ✗=masked):
         pos0 pos1 pos2 pos3 pos4 pos5
  pos0:    ✓    ✗    ✗    ✗    ✗    ✗
  pos1:    ✓    ✓    ✗    ✗    ✗    ✗
  pos2:    ✓    ✓    ✓    ✗    ✗    ✗
  pos3:    ✓    ✓    ✓    ✓    ✗    ✗
  pos4:    ✓    ✓    ✓    ✓    ✓    ✗
  pos5:    ✓    ✓    ✓    ✓    ✓    ✓
```

### Shifting Labels

In PyTorch, the cross-entropy loss requires aligned `(logits, targets)` tensors. For a sequence of length $T$, the model produces $T$ logit vectors (one per input position), but the *first useful prediction* is at position 0 (predicting position 1). The standard idiom is:

- **Input to the model**: tokens $[x_0, x_1, \ldots, x_{T-1}]$
- **Targets**: tokens $[x_1, x_2, \ldots, x_T]$ — shifted by 1

```python
import torch
import torch.nn.functional as F

# Suppose tokens is shape (B, T+1): batch of sequences with one extra token
# e.g., tokens = tokenizer.encode(text) + [EOS]

def compute_lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    logits : (B, T, V)   — model output for each of T input positions
    tokens : (B, T+1)    — the full sequence including the token AFTER the last input

    We align by:
      inputs  = tokens[:, :-1]   shape (B, T)  — fed to the model
      targets = tokens[:, 1:]    shape (B, T)  — what each position should predict
    """
    # targets are the tokens one step ahead of each input position
    targets = tokens[:, 1:]          # (B, T)

    # logits come from running the model on tokens[:, :-1]
    # Reshape for F.cross_entropy: expects (N, C) or (N, C, ...) 
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),    # (B*T, V)
        targets.reshape(B * T),      # (B*T,)
        reduction='mean'             # average over all non-masked positions
    )
    return loss
```

This is the entirety of the loss function for standard causal language modeling. Everything else — masking, packing, weighting — is an elaboration of this core.

{{fig:ptobj-teacher-forcing-shift}}

### What `F.cross_entropy` Actually Computes

Every training-context code path in this chapter calls `F.cross_entropy` on a flattened `(B*T, V)` logit tensor, treating it as a black box. It isn't one: over class indices, it is exactly a numerically stable log-softmax followed by a gather of the true-class log-probability and a mean over the non-ignored rows. The stable log-softmax itself is the logsumexp shift trick derived in [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) — $\text{log\_softmax}(z)_i = z_i - (m + \log \sum_j \exp(z_j - m))$ with $m = \max_j z_j$ — which is also where that chapter's fused NumPy backward for this exact operation is worked out.

```python
import torch
import torch.nn.functional as F

def cross_entropy_from_scratch(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Reimplements F.cross_entropy(logits, targets, reduction='mean', ignore_index=ignore_index)
    from scratch: stable log-softmax + gather + masked mean.

    logits  : (N, V) float — raw scores, N = B*T when called on flattened LM logits
    targets : (N,)   long  — true class index per row, or ignore_index to skip that row
    returns : scalar — mean negative log-likelihood over rows where targets != ignore_index
    """
    # 1. Stable log-softmax via the max-shift trick (see foundations 1.4)
    m = logits.max(dim=-1, keepdim=True).values                    # (N,1)
    shifted = logits - m                                           # (N,V)
    logsumexp = shifted.exp().sum(dim=-1, keepdim=True).log()      # (N,1)
    log_probs = shifted - logsumexp                                # (N,V) == log_softmax(logits)

    # 2. Mask ignored rows: clamp targets so gather never indexes out of bounds
    #    on ignore_index=-100 rows; those rows are dropped in step 3 anyway.
    valid = targets != ignore_index                                # (N,) bool
    safe = targets.clamp_min(0)                                    # (N,) long, safe for gather

    # 3. Negative log-likelihood of the true class, averaged over active positions only
    nll = -log_probs.gather(1, safe.unsqueeze(1)).squeeze(1)       # (N,)
    return nll[valid].mean()                                       # scalar


# Verification against F.cross_entropy
torch.manual_seed(0)
N, V = 100, 32000
logits = torch.randn(N, V)
targets = torch.randint(0, V, (N,))
targets[::7] = -100  # ignore roughly 1 in 7 rows (padding / masked positions)

ref  = F.cross_entropy(logits, targets, ignore_index=-100, reduction='mean')
mine = cross_entropy_from_scratch(logits, targets, ignore_index=-100)

print(f"F.cross_entropy: {ref.item():.6f}")   # 11.051692
print(f"from scratch:    {mine.item():.6f}")  # 11.051691
assert torch.allclose(ref, mine, atol=1e-5)
```

Both lines print `~11.05` — a randomly-initialized model scoring a 32000-token vocabulary lands near $\log V$ plus a logit-variance term, here about 11.05 nats — and the assert passes to machine precision. That confirms the flattened `(B*T, V)` call used throughout this chapter is nothing more than stable log-softmax, a gather at the true-class index, and a masked mean.

---

## Loss Masking

### Why Mask?

Not all token positions carry equal pedagogical value. There are three major situations where we want to zero out (mask) certain positions in the loss:

1. **Padding tokens.** Sequences in a batch are padded to the same length. Padding positions have no linguistic content and including them would dilute the gradient.
2. **Prompt tokens in supervised fine-tuning.** During SFT or instruction tuning, we typically want the model to learn the response, not regurgitate the system prompt. (Covered in [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html).)
3. **Document boundary tokens in packed sequences.** When multiple documents are concatenated into a single long sequence (see §Packing below), we must prevent cross-document loss bleed.

### Implementation

```python
import torch
import torch.nn.functional as F

def compute_lm_loss_masked(
    logits: torch.Tensor,     # (B, T, V)
    tokens: torch.Tensor,     # (B, T+1) — full sequences
    mask: torch.Tensor,       # (B, T)   — 1 for positions to train on, 0 to skip
) -> torch.Tensor:
    """
    Masked causal language modeling loss.
    
    mask=0 at padding tokens, prompt tokens, or cross-document positions.
    We use ignore_index=-100 (PyTorch convention) to exclude masked positions.
    """
    targets = tokens[:, 1:].clone()  # (B, T) — shift targets

    # Replace masked positions with ignore_index so they contribute 0 to the loss
    targets[mask == 0] = -100        # -100 is the default ignore_index in F.cross_entropy

    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=-100,
        reduction='mean',    # averages only over non-ignored positions
    )
    return loss


# Quick sanity check -------------------------------------------------------
torch.manual_seed(42)
B, T, V = 2, 8, 32000
logits  = torch.randn(B, T, V)
tokens  = torch.randint(0, V, (B, T + 1))
mask    = torch.ones(B, T, dtype=torch.long)
mask[0, 6:] = 0   # mask last 2 positions of first example (padding)
mask[1, :3] = 0   # mask first 3 positions of second example (prompt)

loss = compute_lm_loss_masked(logits, tokens, mask)
print(f"Masked loss: {loss.item():.4f}")  # a finite float; masked positions don't contribute
```

A subtle point: when using `reduction='mean'`, PyTorch averages over *non-ignored* positions only. If you use `reduction='sum'` and divide manually, be careful to count only active tokens — dividing by the full sequence length (including masked positions) will produce a number that is systematically too small and will silently hurt training.

!!! warning "Mean vs. sum reduction pitfall"
    If you accumulate a sum loss across micro-batches for gradient accumulation, you must normalize by the total number of *active* (non-masked) tokens across all micro-batches, not the total sequence length. Many training bugs are caused by accidentally dividing by the wrong denominator, producing an effective learning rate that varies with the mask density.

---

## Document Packing

### The Efficiency Problem

Real pretraining datasets contain documents of wildly varying lengths — a sentence here, a book chapter there. Padding every example to the longest sequence in a batch wastes enormous amounts of compute on meaningless tokens. For a mixed-length corpus, naive padding can waste 30–60% of tokens.

**Document packing** (also called *sequence packing* or *bin packing*) solves this by concatenating multiple short documents end-to-end until the combined sequence reaches the target context length $L$. The concatenation uses a special separator token (e.g., `<|endoftext|>` in GPT-style models).

```text
Context window = 1024 tokens

Unpacked (padded):
  [Doc A: 300 tokens | PAD x 724]       — 71% waste
  [Doc B: 512 tokens | PAD x 512]       — 50% waste
  [Doc C: 128 tokens | PAD x 896]       — 88% waste

Packed:
  [Doc A: 300 | SEP | Doc B: 512 | SEP | Doc C: 128 | PAD x 83]
  — only 8% waste
```

### The Cross-Document Contamination Problem

Naive packing causes a subtle loss contamination issue: the model's prediction of the first token of document B is conditioned on the final tokens of document A, which is semantically meaningless. This inflates the loss on document-boundary tokens and, more insidiously, teaches the model to expect arbitrary tokens as context — potentially hurting coherence of long-form generation.

There are two ways to handle this:

**Option 1: Loss masking at boundaries.** Zero out the loss for the first token of each document in a packed sequence (its prediction is "poisoned" by the previous unrelated document). This is the most common approach.

**Option 2: Intra-document causal masking.** Use a block-diagonal attention mask so that each document only attends to itself. This is more expensive (cannot use standard FlashAttention without modification) but eliminates the contamination entirely. This is what some modern models use during fine-tuning (see [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html)).

```python
import torch

def build_packed_loss_mask(
    doc_ids: torch.Tensor,  # (B, T) — integer doc ID for each token position
) -> torch.Tensor:
    """
    Returns a loss mask (B, T) where position t is 1 (active) unless it is
    the first token of a new document (in which case its loss is contaminated
    by the previous document's context and should be excluded).

    doc_ids example for one sequence:
       [0, 0, 0, 1, 1, 1, 1, 2, 2]
    First positions of docs 1 and 2 (indices 3 and 7) get mask=0.
    """
    B, T = doc_ids.shape
    # A position starts a new document when its doc_id differs from the previous one
    # Position 0 is also the start of a document, but it has no "poisoned" context
    # so we keep it active (its input is just the BOS or the context start).
    mask = torch.ones(B, T, dtype=torch.long, device=doc_ids.device)

    # Detect document boundaries: where doc_id[t] != doc_id[t-1]
    # doc_ids[:, 1:] != doc_ids[:, :-1] gives True at boundary positions (t >= 1)
    boundary = (doc_ids[:, 1:] != doc_ids[:, :-1])  # (B, T-1)

    # The first token AFTER a boundary (i.e., position t where boundary[t-1] is True)
    # has its loss masked out. In the targets tensor (which is shifted by 1),
    # we mask the target at position t-1 when boundary[t-1] is True.
    # Equivalently: in the loss over targets[:, t], mask when doc changes at t.
    # Targets are tokens[:, 1:], so target[t] corresponds to predicting token t+1
    # from prefix up to token t. If token t+1 starts a new doc, mask it.
    new_doc_at_next = doc_ids[:, 1:] != doc_ids[:, :-1]  # (B, T-1): True when t+1 starts new doc
    mask[:, :-1][new_doc_at_next] = 0  # mask positions t where next token is a new doc

    return mask   # (B, T): 1 = train on this position, 0 = ignore
```

{{fig:ptobj-packing-loss-mask}}

---

## A Complete, Annotated Loss Computation

The following is a self-contained, runnable example that ties together all the pieces: packing, masking, and loss computation.

```python
"""
Minimal pretraining loss pipeline.

Demonstrates:
  - document packing with SEP tokens
  - loss mask construction (exclude cross-doc boundaries and padding)
  - causal LM loss computation

Runnable with: python -c "exec(open('this_file.py').read())"
Requires: torch >= 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


SEP_TOKEN_ID = 2   # <|endoftext|> or equivalent separator
PAD_TOKEN_ID = 0
VOCAB_SIZE    = 256  # tiny vocab for illustration


def pack_documents(
    documents: List[List[int]],
    context_len: int,
    sep_id: int = SEP_TOKEN_ID,
    pad_id: int = PAD_TOKEN_ID,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack a list of token-ID lists into a single sequence of length context_len.
    Returns:
      tokens  : (context_len,)  padded token sequence
      doc_ids : (context_len,)  document ID per position (-1 for padding)
    """
    tokens  = []
    doc_ids = []
    doc_idx = 0

    for doc in documents:
        # Add SEP before each document (except the very first)
        if tokens:
            tokens.append(sep_id)
            doc_ids.append(doc_idx - 1)  # SEP belongs to the preceding doc
        for tok in doc:
            if len(tokens) >= context_len:
                break
            tokens.append(tok)
            doc_ids.append(doc_idx)
        doc_idx += 1
        if len(tokens) >= context_len:
            break

    # Pad to context_len
    pad_len = context_len - len(tokens)
    tokens  = tokens  + [pad_id] * pad_len
    doc_ids = doc_ids + [-1]     * pad_len   # -1 marks padding

    return (
        torch.tensor(tokens,  dtype=torch.long),
        torch.tensor(doc_ids, dtype=torch.long),
    )


def loss_mask_from_doc_ids(
    doc_ids: torch.Tensor,   # (T,)  — -1 for padding
) -> torch.Tensor:
    """
    Build loss mask of shape (T,).
    Active (1) unless:
      - padding position (doc_id == -1)
      - first token of a new document that follows a different document
        (cross-doc context contamination)
    """
    T = doc_ids.shape[0]
    mask = (doc_ids >= 0).long()   # 0 at padding, 1 elsewhere

    # Also zero out the target positions where the *next* token starts a new doc.
    # Target at position t is tokens[t+1]; if tokens[t+1] belongs to a new doc,
    # the model's context (tokens[:t+1]) is from the wrong doc, so mask it.
    for t in range(T - 1):
        if doc_ids[t] >= 0 and doc_ids[t + 1] >= 0 and doc_ids[t] != doc_ids[t + 1]:
            mask[t] = 0   # predicting the first token of doc[t+1] from doc[t] context
    return mask


class TinyTransformerLM(nn.Module):
    """A minimal decoder-only LM for illustration (not optimized for performance)."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 128):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) → logits: (B, T, V)"""
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        h = self.embed(x) + self.pos_emb(positions)                 # (B, T, d_model)

        # Causal mask: upper-triangular with -inf (additive mask for PyTorch)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)   # (B, T, d_model)
        return self.head(h)                                           # (B, T, V)


def compute_causal_lm_loss(
    model:  nn.Module,
    tokens: torch.Tensor,   # (B, T+1)
    masks:  torch.Tensor,   # (B, T)
) -> torch.Tensor:
    inputs  = tokens[:, :-1]    # (B, T) — model input
    targets = tokens[:, 1:].clone()   # (B, T) — what to predict

    # Mask out undesired positions
    targets[masks == 0] = -100  # PyTorch's ignore_index convention

    logits = model(inputs)      # (B, T, V)
    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=-100,
    )
    return loss


# ---- Demo run ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    CONTEXT = 64

    # Simulate two documents with different lengths
    doc_a = list(range(10, 30))     # 20 tokens
    doc_b = list(range(50, 90))     # 40 tokens

    tokens_1d, doc_ids_1d = pack_documents([doc_a, doc_b], context_len=CONTEXT)
    mask_1d = loss_mask_from_doc_ids(doc_ids_1d)

    print(f"Tokens shape: {tokens_1d.shape}")
    print(f"Active positions: {mask_1d.sum().item()} / {CONTEXT}")
    print(f"Doc boundaries masked: {(mask_1d == 0).sum().item()} positions")

    # Batch of 2 sequences (in real training, batch of hundreds)
    tokens_batch   = tokens_1d.unsqueeze(0).expand(2, -1).clone()      # (2, 64)
    # mask_1d[t] flags whether the prediction of token t+1 (from doc[t]'s
    # context) is valid; it has one entry per input position (63 of them),
    # so the last entry (which has no "next token" in this array) is dropped
    # to align with targets = tokens_batch[:, 1:] of length T=63.
    mask_batch     = mask_1d[:-1].unsqueeze(0).expand(2, -1).clone()   # (2, 63)

    # tokens_batch has T+1=64 tokens; model sees first T=63, predicts last T=63
    model = TinyTransformerLM(vocab_size=VOCAB_SIZE, max_seq_len=CONTEXT)
    loss  = compute_causal_lm_loss(model, tokens_batch, mask_batch)

    print(f"Loss:       {loss.item():.4f} nats/token")
    print(f"Perplexity: {loss.exp().item():.2f}")
    # Expected: loss ≈ log(256) ≈ 5.55 for a random initialized model over 256-token vocab
```

---

## Bits-per-Byte: A Tokenizer-Agnostic Metric

Perplexity as defined above is tokenizer-dependent. A model trained on a byte-pair encoding (BPE) tokenizer with vocabulary size 50 000 will report different perplexity from one with a 100 000-token vocabulary, even if they have identical predictive power over raw text. This makes cross-model comparisons treacherous.

**Bits-per-byte (BPB)** normalizes by the number of *bytes* (or characters) each token represents, yielding a tokenizer-independent measure.

$$
\text{BPB} = \frac{\mathcal{L}_{\text{NLL}} \cdot \log_2 e}{\bar{r}}
$$

where $\mathcal{L}_{\text{NLL}}$ is the mean per-token NLL loss in nats, $\log_2 e \approx 1.4427$ converts nats to bits, and $\bar{r}$ is the average number of UTF-8 bytes per token for the tokenizer. Equivalently:

$$
\text{BPB} = -\frac{1}{N_\text{bytes}} \sum_{t} \log_2 \hat{p}(x_t \mid x_{<t})
$$

where the sum runs over all tokens and $N_\text{bytes}$ is the total byte count of the text.

!!! example "Worked Example: Converting Loss to BPB"
    Suppose a model trained on GPT-2's 50 257-token BPE vocabulary achieves a test-set NLL of **2.85 nats/token** on a English Wikipedia excerpt.
    
    The GPT-2 tokenizer has an average token length of approximately **4.0 bytes** for English text (empirically measured; varies by domain).
    
    **Converting to bits/token:**
    $$
    \text{bits/token} = 2.85 \times \log_2 e = 2.85 \times 1.4427 \approx 4.11 \text{ bits/token}
    $$
    
    **Converting to bits/byte (BPB):**
    $$
    \text{BPB} = \frac{4.11 \text{ bits/token}}{4.0 \text{ bytes/token}} \approx 1.03 \text{ bits/byte}
    $$
    
    A BPB below 1.0 on English Wikipedia is considered strong. Shannon estimated the entropy of English to be approximately 0.6–1.3 bits/character.
    
    Now compare with a model using a 100k vocabulary that achieves 3.20 nats/token but where each token averages 5.5 bytes. Its BPB would be:
    $$
    \text{BPB} = \frac{3.20 \times 1.4427}{5.5} \approx \frac{4.62}{5.5} \approx 0.84 \text{ bits/byte}
    $$
    
    Despite higher per-token loss, this model is **better** in a tokenizer-normalized sense. The comparison would be meaningless without BPB normalization.

```python
import math

def nats_per_token_to_bpb(
    loss_nats: float,        # average NLL in nats per token
    avg_bytes_per_token: float,   # tokenizer-specific compression ratio
) -> float:
    """Convert per-token NLL (nats) to bits-per-byte."""
    bits_per_token = loss_nats * math.log2(math.e)
    return bits_per_token / avg_bytes_per_token


def compute_avg_bytes_per_token(tokenizer, sample_texts: list[str]) -> float:
    """Estimate the average bytes-per-token ratio for a given tokenizer."""
    total_bytes  = 0
    total_tokens = 0
    for text in sample_texts:
        total_bytes  += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))
    return total_bytes / total_tokens


# Illustrative usage (requires `transformers` installed):
# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("gpt2")
# r = compute_avg_bytes_per_token(tok, ["Hello world.", "The cat sat on the mat."])
# bpb = nats_per_token_to_bpb(loss_nats=2.85, avg_bytes_per_token=r)

# Manual example matching the worked example above:
bpb = nats_per_token_to_bpb(2.85, 4.0)
print(f"BPB: {bpb:.3f}")   # → 1.028
```

---

## UL2 and Span Corruption Alternatives

### Beyond Causal LM

The causal language modeling objective is the dominant pretraining strategy for decoder-only models, but it is not the only option. Encoder-decoder models (T5, mT5) were pretrained on **masked span corruption** (also called **masked language modeling with contiguous spans**), popularized by Raffel et al. in the T5 paper (2020).

In span corruption, a fraction of token spans in the input is replaced with sentinel tokens (e.g., `<extra_id_0>`), and the model must reconstruct the original spans:

```text
Input:  "The quick <extra_id_0> over the <extra_id_1> dog."
Target: "<extra_id_0> brown fox jumps <extra_id_1> lazy <eos>"
```

This is more efficient than causal LM in one sense: only the masked-out tokens (typically 15% of the sequence) contribute to the loss, but each contributes a larger gradient signal because they require understanding the surrounding context.

### The UL2 Family

**UL2** (Tay et al., 2022, "Unifying Language Learning Paradigms") showed that a single model pretrained on a **mixture** of different denoising objectives can match or outperform models trained on any single objective, while gaining versatility.

UL2 defines three classes of denoising modes, labeled with special tokens:

| Mode | Sentinel | Description | Use case |
|------|----------|-------------|----------|
| **R-denoising** (Regular) | `[S2S]` | Short contiguous spans masked (~15%), low corruption | Recall-heavy tasks |
| **X-denoising** (Extreme) | `[S2S]` | Long spans masked (50–70%), high corruption | Strong generation |
| **S-denoising** (Sequential) | `[NLG]` | Causal prefix LM: attend to prefix, predict suffix | Causal/generative tasks |

During pretraining, the model sees all three modes with different sampling probabilities. At inference, users prefix their input with the appropriate sentinel to select the mode. **UL2 20B** (2022) achieved state-of-the-art on zero-shot generation benchmarks while retaining strong performance on classification, outperforming both T5 (span corruption only) and GPT-3 (causal LM only) at similar parameter counts.

**Flan-UL2** (2023) further instruction-tuned the UL2 checkpoint and was released publicly.

{{fig:ptobj-ul2-denoising-modes}}

### Prefix Language Modeling

A middle ground between masked LM and causal LM is the **prefix LM** (or **non-causal prefix** model): the input portion attends bidirectionally (full attention), while the output portion attends causally. This is the architecture of GLM (General Language Model, Du et al.) and PaLM's initial pretraining variant. The loss is computed only on the output (continuation) portion.

Architecture variants and their relationship to these objectives are covered in depth in [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html).

---

## Practical Loss Engineering: What Practitioners Actually Do

### Loss Normalization Across Heterogeneous Batches

In large-scale distributed training (see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)), each GPU holds a micro-batch. Two normalization choices are common:

**Token-normalized loss (recommended):**
$$
\mathcal{L} = \frac{\sum_{b,t} \mathbb{1}[\text{active}_{b,t}] \cdot \ell_{b,t}}{\sum_{b,t} \mathbb{1}[\text{active}_{b,t}]}
$$

This divides by the count of active (unmasked) tokens and is independent of batch construction.

**Sequence-normalized loss (legacy):**
$$
\mathcal{L} = \frac{1}{B} \sum_{b} \frac{1}{T_b} \sum_{t} \ell_{b,t}
$$

This averages per-sequence first, then averages sequences. It implicitly weights short sequences more heavily — usually undesired.

### Label Smoothing

**Label smoothing** replaces the one-hot target with a softer distribution: instead of probability 1 on the true token, the true token gets $1 - \epsilon$ and each other token gets $\epsilon / (V-1)$:

$$
\tilde{p}(v) = (1 - \epsilon) \cdot \mathbb{1}[v = x_t] + \frac{\epsilon}{V}
$$

With $\epsilon = 0.1$, this has been shown to improve calibration and slightly regularize training. PyTorch's `F.cross_entropy` supports it directly via the `label_smoothing` argument. Many large LM runs skip label smoothing or use very small $\epsilon \le 0.05$, since at scale the training signal is already rich enough.

### Z-Loss for Softmax Stability

At scale, the logits fed to the softmax can grow very large, causing numerical instability (the softmax exponentials overflow in float16). One solution is **z-loss** (Chowdhery et al., PaLM, 2022), which adds a regularizer to penalize large logit norms:

$$
\mathcal{L}_\text{z} = \alpha \cdot \log^2\!\left(\sum_{v} e^{z_v}\right)
$$

where $z_v$ are the pre-softmax logits. With a small coefficient $\alpha$ (e.g., $10^{-4}$), this penalizes large log-partition values without significantly affecting the primary loss. It dramatically reduces loss spikes during training, as documented in the PaLM technical report.

```python
def z_loss(logits: torch.Tensor, alpha: float = 1e-4) -> torch.Tensor:
    """
    Z-loss regularizer for softmax stability (PaLM / Chowdhery et al. 2022).
    
    logits : (*, V) — pre-softmax logit tensor
    alpha  : coefficient; typical value 1e-4 to 1e-5

    Returns a scalar to be added to the primary cross-entropy loss.
    """
    # log(sum_v exp(z_v)) = log-sum-exp, numerically stable via torch.logsumexp
    log_z = torch.logsumexp(logits, dim=-1)   # (*)  — one value per token
    return alpha * (log_z ** 2).mean()


# Integrate with main loss:
# total_loss = ce_loss + z_loss(logits)
```

Training stability issues and how loss diagnostics help debug them are covered in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

---

## Interview Corner

!!! interview "Interview Corner"
    **Q:** An interviewer asks: "What exactly does an LLM optimize during pretraining, and why is that sufficient to produce a model that can answer questions?"

    **A:** During pretraining a decoder-only LLM minimizes **cross-entropy (negative log-likelihood) on next-token prediction**: at each position $t$, the model predicts a probability distribution over the vocabulary given all preceding tokens, and the loss is $-\log \hat{p}(x_t \mid x_{<t})$, averaged over all tokens in the dataset. 

    This is the same as minimizing the KL divergence from the model distribution to the empirical data distribution. Crucially, we use **teacher forcing** — the true tokens are fed as input, not the model's own predictions — so all sequence positions are trained in parallel in a single forward pass with a causal attention mask.

    This objective is sufficient for downstream capability because: (1) language is a proxy for understanding — to predict the next token of a physics paper, a Python snippet, or a news article, the model must build an internal model of physics, programming semantics, and current events respectively; (2) question-answer pairs, reasoning chains, and instructions all appear in the pretraining corpus, so the model implicitly learns these patterns; (3) the data scale (trillions of tokens) and model scale (billions of parameters) ensure these patterns are compressed into the weights rather than memorized. Fine-tuning (SFT/RLHF) then shapes the *style* of output, not the underlying knowledge.

---

## Key Takeaways

!!! key "Key Takeaways"
    - The pretraining objective is **next-token prediction**: minimize $-\log \hat{p}(x_t \mid x_{<t})$ averaged over all tokens. This is exactly cross-entropy / NLL, and minimizing it is equivalent to minimizing KL divergence from the model to the data distribution.
    - **Teacher forcing** feeds ground-truth tokens as input, enabling fully parallel training across all sequence positions via the **causal (lower-triangular) attention mask**.
    - The label vector is the **input sequence shifted left by one**: inputs $= x_{0:T-1}$, targets $= x_{1:T}$.
    - **Loss masking** (setting targets to ignore_index=-100) is essential for padding, prompt regions (SFT), and cross-document boundaries in packed sequences. Always normalize by active token count, not total sequence length.
    - **Document packing** concatenates short documents end-to-end to maximize compute utilization, but requires boundary masking to avoid cross-document contamination.
    - **Bits-per-byte (BPB)** normalizes the loss by the bytes represented per token, making it a tokenizer-agnostic quality metric; use it for fair cross-model comparisons.
    - **UL2** showed that mixing causal (S-denoising), span-corruption (R- and X-denoising) objectives in a single model is both feasible and beneficial; the dominant architecture is still causal LM, but mixture objectives remain relevant for encoder-decoder systems.
    - **Z-loss** ($\alpha \cdot \log^2 Z$) is a practical add-on to penalize large logit norms and reduce loss spikes during large-scale training.
    - The pretraining loss is the single most important health metric for a training run: stable descent, predictable scaling with compute, and alignment between train and validation loss are the primary diagnostic signals.

---

!!! sota "State of the Art & Resources (2026)"
    Next-token prediction via cross-entropy remains the universal pretraining objective for all major decoder-only LLMs (GPT-4, Llama 3, Gemini, Claude). Active research is refining *how* the loss is computed (memory-efficient kernels, mixed-objective pretraining) and *what it should measure* (tokenizer-agnostic metrics like bits-per-byte), but the core mathematical framework from the 2010s is stable and foundational.

    **Foundational work**

    - [Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2020)](https://arxiv.org/abs/1910.10683) — systematic comparison of pretraining objectives (causal LM vs. span corruption) across architectures; defines the span-corruption baseline used by T5 and derivatives.
    - [Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020)](https://arxiv.org/abs/2005.14165) — establishes that scaling causal next-token prediction alone yields powerful few-shot models, cementing NTP as the dominant objective.
    - [Shannon, *A Mathematical Theory of Communication* (1948)](https://archive.org/details/bstj27-3-379) — original derivation of entropy as a lower bound on compression; foundational for understanding what cross-entropy loss measures and why bits-per-byte matters.

    **Recent advances (2023–2026)**

    - [Tay et al., *UL2: Unifying Language Learning Paradigms* (2022)](https://arxiv.org/abs/2205.05131) — shows a mixture of causal (S-denoising), short-span (R-denoising), and aggressive (X-denoising) objectives in a single model matches or beats specialist objectives across tasks.
    - [Wang et al., *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?* (2022)](https://arxiv.org/abs/2204.05832) — large-scale empirical comparison of causal vs. masked objectives across decoder-only and encoder-decoder architectures; key reference for practitioners choosing an objective.
    - [Grattafiori et al., *The Llama 3 Herd of Models* (Meta, 2024)](https://arxiv.org/abs/2407.21783) — details how a state-of-the-art open model is pretrained with standard next-token prediction at scale (15T tokens, 405B params), including packing and BPB evaluation.
    - [Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)](https://arxiv.org/abs/2411.09009) — proposes Cut Cross-Entropy (CCE), a fused kernel that computes cross-entropy without materializing the full logit matrix, reducing loss-layer memory from 24 GB to 1 MB for a 2B model.
    - [Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022)](https://arxiv.org/abs/2204.02311) — documents z-loss and other engineering stabilizations for cross-entropy at 540B-parameter scale.

    **Open-source & tools**

    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — ~300-line clean implementation of causal LM pretraining loss, teacher forcing, and the full training loop; the clearest code companion to this chapter.
    - [Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556) — uses per-token cross-entropy as the primary dependent variable to derive compute-optimal scaling laws; essential companion to the loss-as-health-metric theme.

    **Go deeper**

    - [Karpathy, *Let's build GPT: from scratch, in code, spelled out* (2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 2-hour video walkthrough building the causal LM loss, causal mask, and teacher forcing from scratch in PyTorch; pairs directly with this chapter's code.

## Further Reading

- **Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5), JMLR 2020.** Introduces span-corruption pretraining and systematically compares objectives, architectures, and data scales.
- **Tay et al., "Unifying Language Learning Paradigms" (UL2), ICLR 2023.** Defines the R/X/S-denoising taxonomy and shows that a single model benefits from a mixture of objectives.
- **Brown et al., "Language Models are Few-Shot Learners" (GPT-3), NeurIPS 2020.** Demonstrates that scale of causal LM pretraining alone produces powerful few-shot models, establishing the primacy of next-token prediction for general-purpose LLMs.
- **Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," JMLR 2023.** Documents z-loss and other engineering choices that stabilize cross-entropy at large scale.
- **Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla), NeurIPS 2022.** Uses the cross-entropy loss as the primary dependent variable to derive scaling laws; directly connected to [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).
- **Karpathy, "nanoGPT" (GitHub, 2022–present).** A ~300-line clean implementation of the full pretraining loop including the loss computation; the best codebase to read alongside this chapter.
- **Shannon, "A Mathematical Theory of Communication," Bell System Technical Journal, 1948.** The original derivation of entropy as a lower bound on compression, foundational to understanding what cross-entropy loss measures.

---

## Exercises

**1.** Perplexity intuition. A randomly-initialized model over a vocabulary of size $V = 32000$ has just been created; no training has happened. Assuming the softmax outputs are approximately uniform over the vocabulary at initialization, what per-token NLL loss (in nats) and what perplexity do you expect on the first batch? Then explain in one sentence why the chapter's from-scratch verification snippet actually prints $\approx 11.05$ rather than exactly $\log V$.

??? note "Solution"
    A uniform distribution over $V$ tokens assigns probability $1/V$ to the true token, so the per-token NLL is
    $$
    \mathcal{L}_{\text{NLL}} = -\log\frac{1}{V} = \log V = \log 32000 \approx 10.37 \text{ nats/token}.
    $$
    The corresponding perplexity is
    $$
    \text{PPL} = \exp(\mathcal{L}_{\text{NLL}}) = \exp(\log V) = V = 32000,
    $$
    which is exactly the "effective branching factor" interpretation from the chapter: an untrained model is as uncertain as a uniform choice among all 32000 tokens.

    The chapter's snippet prints $\approx 11.05$, not $10.37$, because its logits are drawn from `torch.randn` rather than being exactly zero. The softmax of non-zero i.i.d. Gaussian logits is *not* uniform; the logit variance adds an extra positive term on top of $\log V$ (as the chapter notes, "near $\log V$ plus a logit-variance term"). Only in the idealized zero-logit case does the loss equal $\log V$ exactly.

**2.** Label shifting and tensor shapes. In the chapter's `compute_lm_loss`, the input is `tokens` of shape `(B, T+1)`. Explain precisely why the model is fed `tokens[:, :-1]` and supervised with `tokens[:, 1:]`, and state how many gradient-carrying predictions a single unpadded sequence of length $T+1 = 2048$ produces. What goes wrong if you instead (incorrectly) align the logits with `tokens[:, :-1]` as targets?

??? note "Solution"
    A decoder-only model with a causal mask produces, at each input position $i$, a distribution over the *next* token. So the logit at position $i$ (computed from `tokens[:, :-1]`, i.e. from prefix $x_0 \ldots x_i$) must be scored against the ground-truth token at position $i+1$. Aligning `inputs = tokens[:, :-1]` with `targets = tokens[:, 1:]` implements exactly this one-step-ahead shift: position $i$'s logit is matched to $x_{i+1}$.

    A sequence of $T+1 = 2048$ tokens becomes $T = 2047$ input positions after dropping the last token, each producing one supervised prediction — so **2047 gradient-carrying predictions**, matching the chapter's "2047 predictions per 2048-token sequence" claim about data efficiency.

    If you instead used `tokens[:, :-1]` as the targets, every position would be trained to predict *its own input token*. Combined with the causal mask (position $i$ can already see $x_i$, including itself), this is a trivial identity task: the model can copy the input token to the output and drive the loss to zero without learning any language structure. Training would collapse to a useless copy function.

**3.** Bits-per-byte comparison. Two models are evaluated on the same English text.

    - Model A: BPE tokenizer, NLL $= 3.10$ nats/token, average $4.2$ UTF-8 bytes/token.
    - Model B: larger-vocab tokenizer, NLL $= 3.55$ nats/token, average $5.4$ UTF-8 bytes/token.

Compute the bits-per-byte for each (use $\log_2 e \approx 1.4427$) and state which model is better in a tokenizer-agnostic sense. Then explain in one sentence why comparing their raw per-token NLL values directly would be misleading.

??? note "Solution"
    Using $\text{BPB} = \dfrac{\mathcal{L}_{\text{NLL}} \cdot \log_2 e}{\bar r}$:

    Model A:
    $$
    \text{bits/token} = 3.10 \times 1.4427 \approx 4.472, \qquad
    \text{BPB}_A = \frac{4.472}{4.2} \approx 1.065 \text{ bits/byte}.
    $$

    Model B:
    $$
    \text{bits/token} = 3.55 \times 1.4427 \approx 5.122, \qquad
    \text{BPB}_B = \frac{5.122}{5.4} \approx 0.949 \text{ bits/byte}.
    $$

    Model B has the **lower** BPB ($0.949 < 1.065$), so it is the better model in a tokenizer-agnostic sense, even though its raw per-token NLL ($3.55$) is *higher* than Model A's ($3.10$).

    Comparing raw per-token NLL is misleading because each token covers a different amount of raw text: Model B's tokens are longer ($5.4$ vs $4.2$ bytes), so each of its predictions is "harder" (covers more content) and a higher per-token loss can still mean fewer bits spent per byte of actual text. BPB removes this tokenizer dependence by normalizing to a common unit (the byte).

**4.** Z-loss arithmetic and purpose. Consider a single token whose pre-softmax logits, before any normalization, all share a common additive offset $c$ — i.e. the logits are $z_v = a_v + c$ for some fixed base pattern $a_v$. (a) Show that the cross-entropy loss for the true token is invariant to $c$. (b) Show that the z-loss term $\alpha \log^2\!\big(\sum_v e^{z_v}\big)$ is *not* invariant to $c$, and compute the z-loss for $\alpha = 10^{-4}$ when $\log\sum_v e^{z_v} = 30$. (c) In one sentence, explain why penalizing this quantity improves numerical stability.

??? note "Solution"
    (a) The softmax probability of the true token $x_t$ is
    $$
    \hat p(x_t) = \frac{e^{z_{x_t}}}{\sum_v e^{z_v}} = \frac{e^{a_{x_t}+c}}{\sum_v e^{a_v+c}} = \frac{e^{c}e^{a_{x_t}}}{e^{c}\sum_v e^{a_v}} = \frac{e^{a_{x_t}}}{\sum_v e^{a_v}}.
    $$
    The common factor $e^{c}$ cancels, so $\hat p(x_t)$ — and hence the cross-entropy $-\log\hat p(x_t)$ — does not depend on $c$. (This is the familiar shift-invariance of softmax.)

    (b) The log-partition is
    $$
    \log\sum_v e^{z_v} = \log\Big(e^{c}\sum_v e^{a_v}\Big) = c + \log\sum_v e^{a_v},
    $$
    which grows linearly with $c$, so its square (and the z-loss) is *not* invariant. For $\log\sum_v e^{z_v} = 30$:
    $$
    \mathcal{L}_{\text{z}} = \alpha \cdot (30)^2 = 10^{-4} \times 900 = 0.09.
    $$

    (c) Because cross-entropy alone cannot "see" the common offset $c$, logits are free to drift to very large magnitudes during training; z-loss adds a gradient that pulls the log-partition (and thus the raw logit scale) back toward $0$, preventing the softmax exponentials from overflowing in low-precision arithmetic and reducing loss spikes.

**5.** Implement per-document (sequence-normalized) loss and compare to token-normalized loss. The chapter contrasts token-normalized and sequence-normalized loss and warns that sequence-normalization "implicitly weights short sequences more heavily." Implement a function `sequence_normalized_loss(logits, tokens, mask)` that computes the loss by first averaging over the active positions *within each sequence*, then averaging those per-sequence means across the batch. Then, using the chapter's masking conventions, construct a 2-sequence batch where one sequence has far fewer active tokens than the other and demonstrate numerically that the sequence-normalized value differs from the standard token-normalized `compute_lm_loss_masked`.

??? note "Solution"
    The key difference: `F.cross_entropy(..., reduction='mean')` divides the summed loss by the *total* active-token count across the whole batch (token-normalized). Sequence-normalization instead computes a per-sequence mean first, then a plain mean over sequences — giving every sequence equal weight regardless of how many active tokens it has.

    ```python
    import torch
    import torch.nn.functional as F

    def sequence_normalized_loss(
        logits: torch.Tensor,   # (B, T, V)
        tokens: torch.Tensor,   # (B, T+1)
        mask:   torch.Tensor,   # (B, T) — 1 = active, 0 = ignore
    ) -> torch.Tensor:
        targets = tokens[:, 1:].clone()          # (B, T)
        B, T, V = logits.shape

        # Per-position NLL with no reduction, so we can control the averaging.
        per_pos = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction='none',                    # (B*T,)
        ).reshape(B, T)                          # (B, T)

        m = mask.to(per_pos.dtype)               # (B, T)
        # Sum active loss per sequence, divide by active count per sequence.
        seq_active = m.sum(dim=1).clamp_min(1.0)         # (B,)
        seq_mean   = (per_pos * m).sum(dim=1) / seq_active   # (B,)
        # Then average the per-sequence means with equal weight.
        return seq_mean.mean()

    def compute_lm_loss_masked(logits, tokens, mask):
        targets = tokens[:, 1:].clone()
        targets[mask == 0] = -100
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=-100,
            reduction='mean',                    # token-normalized
        )

    # ---- Demonstration -----------------------------------------------------
    torch.manual_seed(0)
    B, T, V = 2, 8, 32000
    logits = torch.randn(B, T, V)
    tokens = torch.randint(0, V, (B, T + 1))

    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 2:] = 0   # sequence 0: only 2 active tokens (a short doc)
    mask[1, :]  = 1   # sequence 1: all 8 active tokens (a long doc)

    tok_norm = compute_lm_loss_masked(logits, tokens, mask)
    seq_norm = sequence_normalized_loss(logits, tokens, mask)

    print(f"token-normalized:    {tok_norm.item():.4f}")
    print(f"sequence-normalized: {seq_norm.item():.4f}")
    # The two values differ: the short 2-token sequence gets weight 1/2 under
    # sequence-normalization but only 2/10 of the tokens under token-normalization.
    ```

    Why they differ: with $2$ active tokens in sequence 0 and $8$ in sequence 1, token-normalization sums all $10$ per-token losses and divides by $10$ — sequence 1 contributes $8/10$ of the weight. Sequence-normalization gives each sequence's *mean* equal weight ($1/2$ each), so sequence 0's two tokens now carry $1/2$ of the total influence instead of $2/10$. Unless the two per-sequence means happen to coincide, the printed numbers differ, concretely demonstrating the chapter's warning that sequence-normalization over-weights short sequences. (If you set the masks so both sequences have the same active count, the two values become equal.)
