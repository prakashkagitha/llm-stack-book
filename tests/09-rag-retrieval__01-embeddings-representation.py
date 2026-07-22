"""
Runs the CPU-runnable Python blocks from
content/09-rag-retrieval/01-embeddings-representation.md, concatenated in
order so that later blocks can rely on names defined by earlier ones (as
they do in the chapter itself). Each block is copied verbatim from the
chapter; minimal glue needed to make blocks that only *define* something
actually execute has been added, and is clearly marked "GLUE".

Blocks tested (5 heuristically CPU-runnable blocks):
  #0 (line ~49)  - mean_pool / normalize_embeddings
  #1 (line ~107) - infonce_loss + the book's own __main__ sanity check
  #2 (line ~261) - sentence-transformers encode + cosine similarity
  #5 (line ~355) - instruction-prefixed e5 embeddings
  #7 (line ~400) - MTEB evaluation loop

Blocks #2, #5, #7 call out to `sentence_transformers` / `mteb`, both of
which would otherwise download model weights / datasets over the network.
Per the network-forbidden rule, the `sentence_transformers` and `mteb`
modules are replaced with small deterministic fakes registered in
sys.modules *before* the book's own `import` statements run. Everything
downstream of that boundary (shapes, cosine-similarity math, MTEB glue
code) is the book's actual, unmodified logic executing against the fake
model's output.

Blocks skipped (not in the tested set; explicit reasons):
  #3 (line ~306) - SKIP(fragment): matryoshka_infonce_loss is a function
                   definition only; the orchestrator's heuristic flagged
                   it as a fragment and it is not among the 5 blocks this
                   harness is asked to prove out.
  #4 (line ~344) - SKIP(non-python): ```text``` instruction-prefix template,
                   not executable code.
  #6 (line ~396) - SKIP(shell): `pip install mteb`, a shell command.
  #8 (line ~430) - SKIP(needs-gpu): full BiEncoder training recipe -- pulls
                   a real `transformers.AutoModel` checkpoint (network) and
                   is explicitly commented in the book as pseudo-code
                   ("Example usage (pseudo-code, requires actual data and
                   GPU)"), so it is never meant to run standalone.
"""

import sys
import types

import numpy as np


# ======================================================================
# GLUE: mock the sentence_transformers / mteb boundary so blocks #2, #5,
# #7 exercise their own logic fully offline. No network call ever
# happens -- these fakes are registered in sys.modules before the book's
# `import sentence_transformers` / `import mteb` statements execute.
# ======================================================================

class _FakeSentenceTransformer:
    """Deterministic stand-in for sentence_transformers.SentenceTransformer.
    Returns fixed, L2-normalized pseudo-embeddings so the book's own
    downstream shape/cosine-similarity code runs unmodified."""

    _DIM = 384  # matches the bge-small-en-v1.5 shape the book comments show

    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        rng = np.random.RandomState(len(texts) * 7 + 1)
        emb = rng.randn(len(texts), self._DIM).astype("float32")
        if normalize_embeddings:
            emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        return emb


_fake_st_module = types.ModuleType("sentence_transformers")
_fake_st_module.SentenceTransformer = _FakeSentenceTransformer
_fake_st_module.InputExample = object
_fake_st_module.losses = types.SimpleNamespace()
sys.modules["sentence_transformers"] = _fake_st_module


class _FakeMTEBRunner:
    def __init__(self, tasks):
        self.tasks = tasks

    def run(self, model, output_folder=None, **kwargs):
        # No network, no disk writes -- just a plausible-shaped result.
        return {"NFCorpus": {"ndcg_at_10": 0.31}}


_fake_mteb_module = types.ModuleType("mteb")
_fake_mteb_module.get_model = lambda name: _FakeSentenceTransformer(name)
_fake_mteb_module.get_tasks = lambda tasks: list(tasks)
_fake_mteb_module.MTEB = _FakeMTEBRunner
sys.modules["mteb"] = _fake_mteb_module


print("=" * 70)
print("Block #0 (line ~49): mean_pool / normalize_embeddings")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn.functional as F


def mean_pool(token_embeddings: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute attention-mask-weighted mean of token embeddings.

    Args:
        token_embeddings: shape (batch, seq_len, hidden_dim)
        attention_mask:   shape (batch, seq_len), 1 for real tokens

    Returns:
        sentence_embeddings: shape (batch, hidden_dim)
    """
    # Expand mask to hidden_dim so we can multiply element-wise
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)

    # Zero out padding positions
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)  # (B, H)

    # Count real tokens per sample (clamp to avoid division by zero)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)             # (B, 1)

    return sum_embeddings / sum_mask  # (B, H)


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize so dot product == cosine similarity."""
    return F.normalize(embeddings, p=2, dim=-1)
# --- end verbatim ---

# GLUE: actually call the two functions on a tiny fixture (a 2-sample
# batch of 5-position, 8-dim toy "token embeddings" with padding).
_tok_emb = torch.randn(2, 5, 8)
_mask = torch.tensor([[1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1]])
_pooled = mean_pool(_tok_emb, _mask)
assert _pooled.shape == (2, 8)

_normed = normalize_embeddings(_pooled)
assert torch.allclose(_normed.norm(dim=-1), torch.ones(2), atol=1e-5)
print("mean_pool output shape:", _pooled.shape)
print("normalized norms:", _normed.norm(dim=-1))


print()
print("=" * 70)
print("Block #1 (line ~107): infonce_loss + book's own sanity check")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn.functional as F


def infonce_loss(
    query_emb: torch.Tensor,   # (B, D), already L2-normalized
    doc_emb: torch.Tensor,     # (B, D), already L2-normalized
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss with in-batch negatives.

    Each query is matched to the corresponding document (diagonal).
    All other documents in the batch serve as negatives.

    Args:
        query_emb:   Normalized query embeddings.
        doc_emb:     Normalized positive-document embeddings.
        temperature: Softmax temperature (lower = sharper contrast).

    Returns:
        Scalar loss (mean over batch).
    """
    B = query_emb.size(0)

    # Similarity matrix: (B, B)
    # sim[i][j] = cosine_sim(query_i, doc_j)
    sim = torch.matmul(query_emb, doc_emb.T) / temperature  # (B, B)

    # Targets: each query i matches document i (diagonal)
    labels = torch.arange(B, device=query_emb.device)

    # Query-to-document direction
    loss_q2d = F.cross_entropy(sim, labels)

    # Document-to-query direction (symmetric)
    loss_d2q = F.cross_entropy(sim.T, labels)

    return (loss_q2d + loss_d2q) / 2


# ---- Quick sanity check ----
# (This is the book's own `if __name__ == "__main__":` block; since this
# whole test module is itself run as __main__, it is inlined unconditionally
# here to guarantee it actually executes.)
torch.manual_seed(42)
B, D = 4, 768
# NOTE: the book's original toy example used D=64, but at temperature=0.05
# and only 64 dims, the random-init loss is inflated by variance amplification
# (observed ~2.8-3.2 across seeds, not close to log(B)). D=768 (BERT-base's
# actual dimension, used throughout this chapter) is large enough for the
# "~ log(B) ≈ 1.386" comment below to hold, matching the chapter's own claim.
# This was a real inaccuracy in the book's illustrative comment; fixed here
# and mirrored in content/09-rag-retrieval/01-embeddings-representation.md.

# Random normalized embeddings
q = F.normalize(torch.randn(B, D), dim=-1)
d = F.normalize(torch.randn(B, D), dim=-1)

loss = infonce_loss(q, d, temperature=0.05)
print(f"Loss (random init): {loss.item():.4f}")   # ~ log(B) ≈ 1.386

# Perfect embeddings: q[i] == d[i]
d_perfect = q.clone()
loss_perfect = infonce_loss(q, d_perfect, temperature=0.05)
print(f"Loss (perfect align): {loss_perfect.item():.6f}")  # ≈ 0.0
# --- end verbatim ---

assert abs(loss.item() - np.log(4)) < 0.5   # near log(4) at random init, per book
assert loss_perfect.item() < 1e-3           # near-zero once perfectly aligned


print()
print("=" * 70)
print("Block #2 (line ~261): sentence-transformers encode + cosine sim")
print("(SentenceTransformer mocked above -- no network)")
print("=" * 70)

# --- verbatim from the chapter ---
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load a pre-trained backbone (already fine-tuned for sentence embedding)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Encode a batch of texts -- returns numpy array by default
texts = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is located in Paris.",
]
embeddings = model.encode(texts, normalize_embeddings=True)

print(f"Embedding shape: {embeddings.shape}")  # (3, 384)

# Cosine similarity between query and each document
import numpy as np
query_emb = embeddings[0]
doc_embs  = embeddings[1:]
scores = doc_embs @ query_emb  # dot product of L2-normalized = cosine sim
print(f"Scores: {scores}")     # e.g. [0.71, 0.58] -- first doc more relevant
# --- end verbatim ---

assert embeddings.shape == (3, 384)
assert scores.shape == (2,)


print()
print("=" * 70)
print("Block #5 (line ~355): instruction-prefixed e5 embeddings")
print("(SentenceTransformer mocked above -- no network)")
print("=" * 70)

# --- verbatim from the chapter ---
from sentence_transformers import SentenceTransformer

# Instruction-aware embedding model
model = SentenceTransformer("intfloat/e5-large-v2")

# Prefix 'query:' vs 'passage:' tells the model the role
queries = ["query: How do transformers handle long sequences?"]
docs    = [
    "passage: Transformers scale quadratically with sequence length in attention.",
    "passage: The history of the Transformer architecture dates to 2017.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
d_emb = model.encode(docs,    normalize_embeddings=True)

scores = q_emb @ d_emb.T
print(scores)  # [[0.73, 0.41]] -- first document correctly ranked higher
# --- end verbatim ---

assert scores.shape == (1, 2)


print()
print("=" * 70)
print("Block #7 (line ~400): MTEB evaluation loop")
print("(mteb + SentenceTransformer mocked above -- no network)")
print("=" * 70)

# --- verbatim from the chapter ---
import mteb
from sentence_transformers import SentenceTransformer

# Load the model as an MTEB-compatible encoder
model_name = "BAAI/bge-small-en-v1.5"
model = mteb.get_model(model_name)

# Run a single retrieval task
tasks = mteb.get_tasks(tasks=["NFCorpus"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")

# results contains nDCG@10 and other metrics per task
# --- end verbatim ---

print("results:", results)
assert "NFCorpus" in results


print()
print("All tested blocks executed successfully.")
