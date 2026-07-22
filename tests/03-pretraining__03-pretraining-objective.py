"""
Runs the CPU-runnable Python code blocks from:
    content/03-pretraining/03-pretraining-objective.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order. Later blocks reuse names from earlier blocks only where the chapter
itself does so; each block is otherwise self-contained.

Tested blocks:
    #1 (line ~121) -- compute_lm_loss(): shift-by-one label alignment +
        F.cross_entropy. The book defines it without calling it, so we add a
        tiny fixture call (small B,T,V random logits/tokens) to exercise it.
    #2 (line ~159) -- cross_entropy_from_scratch(): from-scratch stable
        log-softmax + gather + masked mean, verified against F.cross_entropy.
        Executed exactly as written, including the book's own assert.
    #3 (line ~219) -- compute_lm_loss_masked(): ignore_index masking for
        padding/prompt tokens. Executed exactly as written, including the
        book's own sanity-check call and print.
    #5 (line ~300) -- build_packed_loss_mask(): batched loss-mask
        construction from per-token doc_ids. The book defines it without
        calling it, so we add a tiny fixture call on a toy (B, T) doc_ids
        tensor with an explicit boundary to exercise it.
    #6 (line ~345) -- the complete annotated pipeline: pack_documents(),
        loss_mask_from_doc_ids(), TinyTransformerLM, compute_causal_lm_loss(),
        and the book's own `if __name__ == "__main__":` demo run, executed
        exactly as written (we call the demo body directly since this file's
        own __main__ guard is used for the overall test, not the chapter's).
        REAL BUG FOUND & FIXED (mirrored in content/03-pretraining/
        03-pretraining-objective.md): the book's demo built
        `mask_batch = mask_1d.unsqueeze(0).expand(2, -1).clone()` with shape
        (2, 64), but compute_causal_lm_loss's `targets = tokens[:, 1:]` has
        shape (2, 63) -- an IndexError at `targets[masks == 0] = -100`. Fixed
        by slicing `mask_1d[:-1]` (63 entries) before batching, since
        mask_1d's last entry has no corresponding target in this array.

Skipped blocks (per task brief, non-Python / non-standalone / needs-net):
    #0 (line ~96)  -- ```text``` diagram of teacher-forced input/target
                       alignment and the causal-mask triangle; not Python.
    #4 (line ~277) -- ```text``` diagram of packed vs. padded token layout;
                       not Python.
    #7 (line ~554) -- nats_per_token_to_bpb / compute_avg_bytes_per_token.
                       compute_avg_bytes_per_token() calls a HuggingFace
                       tokenizer (`transformers.AutoTokenizer`), which is a
                       network/model download at import+call time. We SKIP
                       that call. nats_per_token_to_bpb() has no network
                       dependency and matches the book's own worked example
                       (2.85 nats/token, 4.0 bytes/token -> BPB ~= 1.028), so
                       we run that half faithfully and assert against the
                       book's printed value.
    #8 (line ~670) -- z_loss(): the book never calls it standalone (only
                       sketches `total_loss = ce_loss + z_loss(logits)` in a
                       comment). Per the task's "fragment" allowance this is
                       skipped from the required-block list, but since it is
                       trivially CPU-safe and pure torch we exercise it too
                       (small logit tensor) as a bonus, not a required block.
    #9              -- not present as a distinct heuristic block in this
                       chapter's block list; nothing else to skip.

No third-party imports beyond torch/numpy/stdlib are required for the tested
blocks. `transformers` is guarded and never actually invoked (see block #7
note above), keeping this file network-free.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

try:
    from transformers import AutoTokenizer  # noqa: F401
    _HAVE_TRANSFORMERS = True
except Exception:
    AutoTokenizer = None
    _HAVE_TRANSFORMERS = False


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #1 (line ~121) -- compute_lm_loss()
# ============================================================================
_section("Block #1: compute_lm_loss")


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


# Tiny fixture call exercising compute_lm_loss (book defines but never calls it)
torch.manual_seed(0)
_B1, _T1, _V1 = 2, 5, 37
_logits1 = torch.randn(_B1, _T1, _V1)
_tokens1 = torch.randint(0, _V1, (_B1, _T1 + 1))
_loss1 = compute_lm_loss(_logits1, _tokens1)
print(f"compute_lm_loss: {_loss1.item():.4f}")
assert torch.isfinite(_loss1)


# ============================================================================
# Block #2 (line ~159) -- cross_entropy_from_scratch()
# ============================================================================
_section("Block #2: cross_entropy_from_scratch")


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


# ============================================================================
# Block #3 (line ~219) -- compute_lm_loss_masked()
# ============================================================================
_section("Block #3: compute_lm_loss_masked")


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
assert torch.isfinite(loss)


# ============================================================================
# Block #5 (line ~300) -- build_packed_loss_mask()
# ============================================================================
_section("Block #5: build_packed_loss_mask")


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


# Tiny fixture call exercising build_packed_loss_mask (book defines but never calls it)
_doc_ids5 = torch.tensor([
    [0, 0, 0, 1, 1, 1, 1, 2, 2],
    [0, 0, 1, 1, 1, 2, 2, 2, 2],
])
_mask5 = build_packed_loss_mask(_doc_ids5)
print(f"build_packed_loss_mask:\n{_mask5}")
# Boundaries for row 0 are at positions 3 (0->1) and 7 (1->2): predicting
# doc_ids[3]=1 from doc_ids[2]=0 context masks position 2, and predicting
# doc_ids[7]=2 from doc_ids[6]=1 context masks position 6.
assert _mask5[0, 2].item() == 0 and _mask5[0, 6].item() == 0
assert _mask5.sum().item() == _doc_ids5.numel() - 4  # two boundaries per row, two rows


# ============================================================================
# Block #6 (line ~345) -- the complete annotated pipeline
# ============================================================================
_section("Block #6: complete annotated pipeline (pack/mask/loss)")

"""
Minimal pretraining loss pipeline.

Demonstrates:
  - document packing with SEP tokens
  - loss mask construction (exclude cross-doc boundaries and padding)
  - causal LM loss computation

Runnable with: python -c "exec(open('this_file.py').read())"
Requires: torch >= 2.0
"""

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
# (The book gates this under `if __name__ == "__main__":`; this test file has
#  its own top-level __main__ guard below, so we run the body directly here,
#  verbatim, to actually exercise pack_documents/loss_mask_from_doc_ids/
#  TinyTransformerLM/compute_causal_lm_loss.)
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
# BUG FIX (mirrors content/03-pretraining/03-pretraining-objective.md):
# the book originally used `mask_1d.unsqueeze(0).expand(2, -1).clone()` here,
# giving mask_batch shape (2, 64). But compute_causal_lm_loss computes
# targets = tokens[:, 1:] of shape (2, 63), so masks must also be (2, 63) --
# the original code raised IndexError. mask_1d[t] flags whether predicting
# token t+1 (from doc[t]'s context) is valid, so its last entry (index 63,
# which has no "next token" within this array) is dropped to align with the
# 63 targets.
mask_batch     = mask_1d[:-1].unsqueeze(0).expand(2, -1).clone()   # (2, 63)

# tokens_batch has T+1=64 tokens; model sees first T=63, predicts last T=63
model = TinyTransformerLM(vocab_size=VOCAB_SIZE, max_seq_len=CONTEXT)
loss  = compute_causal_lm_loss(model, tokens_batch, mask_batch)

print(f"Loss:       {loss.item():.4f} nats/token")
print(f"Perplexity: {loss.exp().item():.2f}")
# Expected: loss ≈ log(256) ≈ 5.55 for a random initialized model over 256-token vocab
assert torch.isfinite(loss)
assert tokens_1d.shape == (CONTEXT,)


# ============================================================================
# Block #7 (line ~554) -- nats_per_token_to_bpb() only (SKIP the
# compute_avg_bytes_per_token() half: it calls a HuggingFace tokenizer,
# which is a network dependency; see module docstring).
# ============================================================================
_section("Block #7: nats_per_token_to_bpb (network-free half only)")


def nats_per_token_to_bpb(
    loss_nats: float,        # average NLL in nats per token
    avg_bytes_per_token: float,   # tokenizer-specific compression ratio
) -> float:
    """Convert per-token NLL (nats) to bits-per-byte."""
    bits_per_token = loss_nats * math.log2(math.e)
    return bits_per_token / avg_bytes_per_token


# SKIP(network): compute_avg_bytes_per_token(tokenizer, sample_texts) calls
# tokenizer.encode(), and the book's own usage example instantiates it via
# `AutoTokenizer.from_pretrained("gpt2")`, which downloads from the HF hub.
# We do not call compute_avg_bytes_per_token here.

# Manual example matching the worked example in the book's prose:
bpb = nats_per_token_to_bpb(2.85, 4.0)
print(f"BPB: {bpb:.3f}")   # → 1.028
assert math.isclose(bpb, 1.028, abs_tol=5e-4)


# ============================================================================
# Bonus (not in the required block list): z_loss() -- trivial, pure torch,
# CPU-safe. The book only sketches a usage comment
# (`total_loss = ce_loss + z_loss(logits)`) without calling it directly, so
# we exercise it here for completeness.
# ============================================================================
_section("Bonus: z_loss")


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


_zl = z_loss(torch.randn(4, 8, 100))
print(f"z_loss: {_zl.item():.6f}")
assert torch.isfinite(_zl) and _zl.item() >= 0.0


if __name__ == "__main__":
    print("\nAll blocks executed successfully.")
