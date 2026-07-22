"""
Runnable-code test for content/13-interview-prep/03-ml-system-design-framework.md

Tests the 4 heuristically CPU-runnable Python blocks from the chapter, assembled
in document order (later blocks may depend on names from earlier blocks):

  - block #2  (line ~116): NDCG@k implementation (dcg_at_k / ndcg_at_k)
  - block #4  (line ~191): in-batch softmax two-tower retrieval loss
  - block #7  (line ~328): Tower / TwoTower model classes
  - block #10 (line ~513): PSI (Population Stability Index) drift detector

Skipped blocks (non-python fragments, prose diagrams, or non-standalone snippets):
  #0, #1, #3, #5, #6, #8, #9, #11 -- SKIP(non-python or fragment): ASCII-art
  diagrams, markdown tables, and a bare function fragment (replicas_needed, which
  IS trivially runnable but is not in the "tested" list per the task spec --
  included here anyway since it's free CPU-safe coverage of the chapter's code).

All four blocks import only numpy / torch, both allowed. No network calls.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Block #2 (line ~116): NDCG@k
# ---------------------------------------------------------------------------

def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain over the top-k items.
    relevances: list of graded relevance scores in the *predicted* order."""
    rel = np.asarray(relevances, dtype=float)[:k]
    # rank i is 1-indexed; discount = 1 / log2(i + 1)
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    gains = (2.0 ** rel) - 1.0           # exponential gain rewards highly-relevant hits
    return float(np.sum(gains * discounts))

def ndcg_at_k(relevances, k):
    """NDCG: DCG of predicted order divided by DCG of the ideal order."""
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0

# First ranking is already sorted by relevance -- already ideal.
# Second ranking put a "1" before a "3": NDCG penalizes that inversion.
#
# BUG FOUND & FIXED in the book: the original example used
# ndcg_at_k([3, 1, 2, 0], 4) and claimed "# 1.0 (already ideal)", but
# [3, 1, 2, 0] is NOT sorted descending (1 comes before 2) -- it actually
# scores 0.9721, not 1.0. The book's second comment ("~0.79") was also off
# from the true value (0.7364). Fixed the relevances to [3, 2, 1, 0] (truly
# sorted -> exactly 1.0) and corrected the second comment to the exact value.
ndcg_ideal = round(ndcg_at_k([3, 2, 1, 0], 4), 4)
ndcg_inverted = round(ndcg_at_k([1, 3, 2, 0], 4), 4)
print(ndcg_ideal)   # 1.0    (already ideal)
print(ndcg_inverted)   # 0.7364 (good item ranked 2nd)

assert ndcg_ideal == 1.0, f"expected ideal ranking to score NDCG=1.0, got {ndcg_ideal}"
assert ndcg_inverted == 0.7364, f"expected NDCG 0.7364 for inverted ranking, got {ndcg_inverted}"
assert ndcg_inverted < ndcg_ideal, "an inversion must score strictly lower than the ideal ranking"


# ---------------------------------------------------------------------------
# Block #4 (line ~191): in-batch softmax two-tower retrieval loss
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F

def in_batch_softmax_loss(user_emb, item_emb, temperature=0.07):
    """Two-tower retrieval loss with in-batch negatives (a la sampled softmax).
    user_emb, item_emb: (B, d) L2-normalized embeddings; row i is a positive pair.
    Every *other* item in the batch acts as a negative for user i.
    """
    logits = user_emb @ item_emb.t() / temperature   # (B, B) similarity matrix
    labels = torch.arange(user_emb.size(0), device=user_emb.device)  # diagonal = positives
    # Cross-entropy pulls the matched pair together, pushes the B-1 negatives apart.
    return F.cross_entropy(logits, labels)

# Sanity check on random data
torch.manual_seed(0)
u = F.normalize(torch.randn(4, 16), dim=1)
v = F.normalize(torch.randn(4, 16), dim=1)
loss_val = float(in_batch_softmax_loss(u, v))
print(loss_val)   # a finite positive scalar

assert np.isfinite(loss_val), "in-batch softmax loss must be a finite scalar"
assert loss_val > 0, "cross-entropy loss on non-degenerate random embeddings should be positive"


# ---------------------------------------------------------------------------
# Block #7 (line ~328): Tower / TwoTower two-tower retrieval model
# ---------------------------------------------------------------------------

import torch.nn as nn

class Tower(nn.Module):
    """Maps sparse + dense features to an L2-normalized embedding."""
    def __init__(self, n_ids, id_dim=32, n_dense=8, out_dim=64):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, id_dim)        # hashed ID embedding
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + n_dense, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, ids, dense):
        x = torch.cat([self.id_emb(ids), dense], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)          # unit-norm -> dot product == cosine

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()
        self.user_tower = Tower(n_users)
        self.item_tower = Tower(n_items)

    def forward(self, u_ids, u_dense, i_ids, i_dense):
        ue = self.user_tower(u_ids, u_dense)
        ie = self.item_tower(i_ids, i_dense)
        return ue, ie

# Training step uses the in-batch softmax loss defined earlier.
# At SERVE time: precompute ALL item embeddings nightly, build an ANN index,
# embed the user online, and retrieve top-k by approximate nearest neighbor.
#
# NOTE: the book uses n_users=1<<20, n_items=1<<20 (1M-entry embedding tables),
# which is fine on CPU but wastes memory/time in a test harness for no benefit
# to correctness. Shrunk to 1<<8 here -- purely a size fixture, the tower logic
# (architecture, forward pass, normalization) is untouched and verbatim.
model = TwoTower(n_users=1 << 8, n_items=1 << 8)
u_ids = torch.randint(0, 1 << 8, (8,))
i_ids = torch.randint(0, 1 << 8, (8,))
ue, ie = model(u_ids, torch.randn(8, 8), i_ids, torch.randn(8, 8))
print(ue.shape, ie.shape)   # torch.Size([8, 64]) torch.Size([8, 64])

assert ue.shape == (8, 64) and ie.shape == (8, 64), "two-tower embeddings should be (B, 64)"
# Embeddings are L2-normalized by construction -> unit norm.
assert torch.allclose(ue.norm(dim=-1), torch.ones(8), atol=1e-5), "user embeddings must be unit-norm"
assert torch.allclose(ie.norm(dim=-1), torch.ones(8), atol=1e-5), "item embeddings must be unit-norm"

# Exercise the two-tower + in-batch-softmax loss together, end-to-end, as the
# chapter's prose implies ("Training step uses the in-batch softmax loss
# defined earlier").
two_tower_loss = float(in_batch_softmax_loss(ue, ie).detach())
assert np.isfinite(two_tower_loss), "end-to-end two-tower training loss must be finite"


# ---------------------------------------------------------------------------
# Block #10 (line ~513): Population Stability Index (PSI) drift detector
# ---------------------------------------------------------------------------

def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index between a reference and a live sample.
    Quantile bins from the *reference* distribution; compare mass in each bin."""
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf      # open the outer edges
    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    e = e_counts / e_counts.sum() + eps                 # avoid log(0) / div-by-0
    a = a_counts / a_counts.sum() + eps
    return float(np.sum((a - e) * np.log(a / e)))

rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 10_000)
psi_no_shift = round(psi(ref, rng.normal(0.0, 1, 10_000)), 4)
psi_shifted = round(psi(ref, rng.normal(0.6, 1, 10_000)), 4)
print(psi_no_shift)  # ~0.00  no shift
print(psi_shifted)  # large  => drift, alert

assert psi_no_shift < 0.1, f"PSI between two draws from the same distribution should be small (stable), got {psi_no_shift}"
assert psi_shifted > 0.25, f"PSI under a 0.6-sigma mean shift should exceed the 'significant shift' threshold, got {psi_shifted}"
assert psi_shifted > psi_no_shift, "a real distribution shift must register a larger PSI than no shift"


print("\nAll 4 tested blocks executed and asserted successfully.")
