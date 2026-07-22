"""
Runs the CPU-runnable Python blocks from:
  content/09-rag-retrieval/01-embeddings-representation.md

Blocks tested (book's code copied faithfully, concatenated in chapter order):
  - block #0 (line ~49):  mean_pool / normalize_embeddings (pooling functions)
  - block #1 (line ~107): infonce_loss + its own "Quick sanity check" section
  - block #2 (line ~261): sentence-transformers encode() + cosine-sim scoring
  - block #5 (line ~355): instruction-style e5-large-v2 query/passage scoring
  - block #7 (line ~400): mteb evaluation harness

Blocks intentionally NOT executed:
  - block #3 (line ~306, matryoshka_infonce_loss def): fragment in the book (no call
    site there); it is however exercised here anyway because block #8's
    EmbeddingModel.forward references it -- see below, we do call it with tiny data
    to make sure it actually runs, since it is trivially CPU-safe and only needs
    the infonce_loss already defined by block #1.
  - block #4 (line ~344): a `text` fenced block (instruction-prefix examples), not code.
  - block #6 (line ~396): `pip install mteb` shell block.
  - block #8 (line ~430): "Practical Training Recipe" -- imports
    transformers.AutoModel/AutoTokenizer and is meant to run on a GPU with real
    data/optimizer scheduling (the book itself calls it "pseudo-code, requires
    actual data and GPU"). SKIP(needs-gpu / non-standalone).

Network policy: blocks #2, #5, and #7 instantiate real hosted models
(SentenceTransformer(...), mteb.get_model(...)) or drive a full mteb evaluation
run (dataset + model download). Per the hard rules, instantiating a real model is
network even with a local cache, so:
  - block #2 and #5: SentenceTransformer is replaced with a tiny deterministic
    offline stub (`_FakeSentenceTransformer`) whose `.encode()` matches the real
    API and produces a shape-correct (N, dim) array. The surrounding logic that
    the chapter is actually teaching -- encoding a batch, then cosine/dot-product
    scoring of query vs. documents -- is executed for real, offline.
  - block #7: SKIP(network). Its only content is three chained calls
    (`mteb.get_model`, `mteb.get_tasks`, `MTEB(...).run(...)`) that each require a
    real network fetch (model weights, task metadata, and the NFCorpus dataset)
    with no independent logic in between to exercise offline -- mocking all three
    would just replay canned return values with nothing of the book's own code
    left to verify. `mteb` and `sentence_transformers` are still imported behind
    try/except so the module loads even when the packages are entirely absent
    (as they are under CI-sim / real CI).
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is guaranteed in CI, but be defensive
    torch = None
    F = None

# sentence_transformers / mteb are NOT installed in CI (see BLOCK list in
# scripts/ci_sim_run.py) -- guard the imports so the module still loads.
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses  # noqa: F401
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    InputExample = None
    losses = None
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import mteb  # noqa: F401
    HAS_MTEB = True
except Exception:
    mteb = None
    HAS_MTEB = False


# ===========================================================================
# Block #0 (line ~49): Max Pooling and Weighted Pooling section
# ===========================================================================

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


# ---- Glue: actually execute mean_pool / normalize_embeddings on a tiny fixture ----
torch.manual_seed(0)
_toy_tokens = torch.randn(2, 5, 8)              # (batch=2, seq_len=5, hidden=8)
_toy_mask = torch.tensor([[1, 1, 1, 0, 0],      # 3 real tokens
                           [1, 1, 1, 1, 1]])    # 5 real tokens
_pooled = mean_pool(_toy_tokens, _toy_mask)
assert _pooled.shape == (2, 8)
# Manually verify sample 0 against the mask-weighted mean formula in the book.
_expected0 = _toy_tokens[0, :3].mean(dim=0)
assert torch.allclose(_pooled[0], _expected0, atol=1e-5)

_normed = normalize_embeddings(_pooled)
assert _normed.shape == (2, 8)
_norms = _normed.norm(p=2, dim=-1)
assert torch.allclose(_norms, torch.ones(2), atol=1e-5)
print(f"[block#0] pooled shape={tuple(_pooled.shape)} normed L2 norms={_norms.tolist()}")


# ===========================================================================
# Block #1 (line ~107): Full In-Batch Negative Loss Implementation
# ===========================================================================

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


# ---- Quick sanity check ---- (this is the book's own __main__ block, run directly)
torch.manual_seed(42)
B, D = 4, 768

# Random normalized embeddings
q = F.normalize(torch.randn(B, D), dim=-1)
d = F.normalize(torch.randn(B, D), dim=-1)

loss = infonce_loss(q, d, temperature=0.05)
print(f"Loss (random init): {loss.item():.4f}")   # ~ log(B) ~= 1.386
# The book claims random-init loss ~= log(B). Verify order of magnitude, not an
# exact constant -- deterministic under the fixed seed above, but we assert the
# qualitative claim robustly rather than pin many decimal places.
import math
assert abs(loss.item() - math.log(B)) < 0.6

# Perfect embeddings: q[i] == d[i]
d_perfect = q.clone()
loss_perfect = infonce_loss(q, d_perfect, temperature=0.05)
print(f"Loss (perfect align): {loss_perfect.item():.6f}")  # ~= 0.0
assert loss_perfect.item() < 1e-3


# ===========================================================================
# Block #2 (line ~261): sentence-transformers encode() + cosine-sim scoring
#   NETWORK: SentenceTransformer(...) loads real hosted weights. Replaced with
#   a deterministic offline stub per the hard rules; the encode + scoring logic
#   that the chapter is teaching runs unmodified.
# ===========================================================================

class _FakeSentenceTransformer:
    """Deterministic offline stand-in for sentence_transformers.SentenceTransformer.

    Matches the slice of the real API used in this chapter: `.encode(texts,
    normalize_embeddings=True) -> np.ndarray` of shape (len(texts), dim).
    """

    def __init__(self, model_name_or_path: str, dim: int = 384):
        self._model_name = model_name_or_path
        self._dim = dim

    def encode(self, texts, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Seed from a stable hash (not Python's randomized str hash, which
            # varies per-process) so results are reproducible across runs.
            import zlib
            seed = zlib.crc32(text.encode("utf-8")) % (2 ** 32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            out[i] = vec
        return out


# Load a pre-trained backbone (already fine-tuned for sentence embedding)
model = _FakeSentenceTransformer("BAAI/bge-small-en-v1.5")  # SKIP(network): real load mocked

# Encode a batch of texts -- returns numpy array by default
texts = [
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is located in Paris.",
]
embeddings = model.encode(texts, normalize_embeddings=True)

print(f"Embedding shape: {embeddings.shape}")  # (3, 384)
assert embeddings.shape == (3, 384)

# Cosine similarity between query and each document
query_emb = embeddings[0]
doc_embs = embeddings[1:]
scores = doc_embs @ query_emb  # dot product of L2-normalized = cosine sim
print(f"Scores: {scores}")     # e.g. [0.71, 0.58] -- first doc more relevant
assert scores.shape == (2,)
assert np.all(scores >= -1.0 - 1e-6) and np.all(scores <= 1.0 + 1e-6)


# ===========================================================================
# Block #5 (line ~355): instruction-style query/passage scoring (e5-large-v2)
#   NETWORK: same treatment as block #2.
# ===========================================================================

# Instruction-aware embedding model
model = _FakeSentenceTransformer("intfloat/e5-large-v2", dim=1024)  # SKIP(network): mocked

# Prefix 'query:' vs 'passage:' tells the model the role
queries = ["query: How do transformers handle long sequences?"]
docs = [
    "passage: Transformers scale quadratically with sequence length in attention.",
    "passage: The history of the Transformer architecture dates to 2017.",
]

q_emb = model.encode(queries, normalize_embeddings=True)
d_emb = model.encode(docs, normalize_embeddings=True)

scores = q_emb @ d_emb.T
print(scores)  # [[0.73, 0.41]] -- first document correctly ranked higher (real model)
assert scores.shape == (1, 2)
assert np.all(scores >= -1.0 - 1e-6) and np.all(scores <= 1.0 + 1e-6)


# ===========================================================================
# Block #7 (line ~400): MTEB evaluation harness
#   SKIP(network): mteb.get_model() downloads real weights, mteb.get_tasks()
#   fetches task metadata, and evaluation.run() drives a full retrieval pass
#   over the NFCorpus dataset (also a download). All three calls are pure
#   network boundary with no interposed book logic to exercise offline, so
#   this block is left un-executed rather than reduced to mock-plumbing.
# ===========================================================================

if HAS_MTEB and HAS_SENTENCE_TRANSFORMERS:  # pragma: no cover - never true under CI-sim
    pass
    # import mteb
    # from sentence_transformers import SentenceTransformer
    #
    # model_name = "BAAI/bge-small-en-v1.5"
    # model = mteb.get_model(model_name)
    #
    # tasks = mteb.get_tasks(tasks=["NFCorpus"])
    # evaluation = mteb.MTEB(tasks=tasks)
    # results = evaluation.run(model, output_folder=f"results/{model_name}")


# ===========================================================================
# Supporting definition used by block #8's EmbeddingModel.forward (block #3,
# line ~306): matryoshka_infonce_loss. The def itself is a fragment with no
# call site in the chapter, but it only depends on infonce_loss (block #1)
# and is trivially CPU-safe, so it is exercised here with a tiny fixture to
# confirm it actually runs end to end.
# ===========================================================================

def matryoshka_infonce_loss(
    query_emb: torch.Tensor,      # (B, D) full dimension
    doc_emb: torch.Tensor,        # (B, D) full dimension
    dims=None,                    # prefix dimensions to train
    temperature: float = 0.05,
    weights=None,
) -> torch.Tensor:
    """
    Matryoshka contrastive loss: InfoNCE at each prefix dimension.
    Gradients flow through all prefix slices simultaneously.
    """
    D = query_emb.size(-1)
    if dims is None:
        dims = [32, 64, 128, 256, D]
    if weights is None:
        weights = [1.0] * len(dims)

    total_loss = torch.tensor(0.0, device=query_emb.device)

    for dim, w in zip(dims, weights):
        # Slice to prefix dimension and re-normalize
        q_slice = F.normalize(query_emb[:, :dim], dim=-1)
        d_slice = F.normalize(doc_emb[:, :dim], dim=-1)

        # Standard InfoNCE at this granularity
        loss_at_dim = infonce_loss(q_slice, d_slice, temperature)
        total_loss = total_loss + w * loss_at_dim

    return total_loss / sum(weights)


torch.manual_seed(7)
_mq = F.normalize(torch.randn(4, 128), dim=-1)
_md = F.normalize(torch.randn(4, 128), dim=-1)
_mloss = matryoshka_infonce_loss(_mq, _md, dims=[32, 64, 128], temperature=0.05)
assert _mloss.dim() == 0
assert _mloss.item() > 0.0
print(f"[block#3-supporting] matryoshka loss (random init) = {_mloss.item():.4f}")


print("All executed blocks completed successfully.")
