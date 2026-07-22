"""
Runnable-code smoke test for content/09-rag-retrieval/06-multimodal-visual-retrieval.md

Tests the 4 heuristically CPU-runnable Python blocks from the chapter:
  - block #1 (line ~90):  ColBERT MaxSim scoring function
  - block #2 (line ~141): vectorized/batched MaxSim + ColPali in-batch InfoNCE loss
  - block #3 (line ~229): PatchHNSWIndex (flatten patches into HNSW, exact MaxSim rerank)
  - block #5 (line ~359): reciprocal_rank_fusion (RRF over ColPali + BM25 rankings)
  - block #6 (line ~391): nDCG@k evaluation (dcg / ndcg_at_k)

Skipped blocks (see rationale inline at point of use):
  - block #0 (line ~35):  SigLIP embed_images/embed_text -- needs `transformers` model
                           download from the network. SKIP(network).
  - block #4 (line ~287): full colpali-engine + Qwen2-VL end-to-end pipeline -- needs
                           GPU-sized VLMs and network model downloads. SKIP(needs-gpu/network).
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

# hnswlib is not in the guaranteed-available CI import list (numpy/torch/einops/sklearn/
# stdlib only), so it must be import-guarded. Block #3's test is skipped if unavailable.
try:
    import hnswlib
except Exception:
    hnswlib = None


# =====================================================================================
# Block #1 (line ~90, 17 lines): ColBERT MaxSim scoring function
# =====================================================================================

def maxsim(Q, D):
    """
    Q: [n, d] query token embeddings (L2-normalized)
    D: [m, d] document token embeddings (L2-normalized)
    Returns the ColBERT late-interaction score (a scalar).
    """
    sim = Q @ D.T                  # [n, m] all query-token x doc-token sims
    per_query = sim.max(dim=1).values   # [n] best doc token per query token
    return per_query.sum()         # sum over query tokens

# Toy: 4 query tokens, 6 doc tokens, d=8
torch.manual_seed(0)
Q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
D = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
print("MaxSim score:", maxsim(Q, D).item())


# =====================================================================================
# Block #2 (line ~141, 28 lines): vectorized batched MaxSim + ColPali in-batch loss
# =====================================================================================

def colbert_scores(Qb, Db, q_mask, d_mask):
    """
    Vectorized MaxSim for a batch (used both in training and scoring).
    Qb: [B, n, d] query token embeddings (padded)
    Db: [B, m, d] doc patch embeddings (padded)
    q_mask: [B, n] 1 for real query tokens, 0 for padding
    d_mask: [B, m] 1 for real patches
    Returns: [B, B] score matrix S[i, j] = MaxSim(query_i, doc_j)
    """
    # Pairwise sims across the cross product of the batch:
    #   [B, n, d] x [B, m, d] -> [B(query), B(doc), n, m]
    sim = torch.einsum("ind,jmd->ijnm", Qb, Db)
    # Mask out padded doc patches before the max over patches.
    sim = sim.masked_fill(~d_mask[None, :, None, :].bool(), -1e4)
    sim = sim.max(dim=-1).values                 # [B, B, n] max over patches
    # Zero out padded query tokens before summing over query tokens.
    sim = sim * q_mask[:, None, :]
    return sim.sum(dim=-1)                        # [B, B]

def colpali_loss(Qb, Db, q_mask, d_mask):
    S = colbert_scores(Qb, Db, q_mask, d_mask)   # [B, B], diagonal = positives
    labels = torch.arange(S.size(0), device=S.device)
    # Standard in-batch InfoNCE over MaxSim scores (both directions optional).
    return F.cross_entropy(S, labels)

# Tiny fixture: batch of B=3, n=5 query tokens (padded to 5, all real here),
# m=7 doc patches (padded to 7, all real here), d=8.
torch.manual_seed(1)
B, n, m, d = 3, 5, 7, 8
Qb = F.normalize(torch.randn(B, n, d), dim=-1)
Db = F.normalize(torch.randn(B, m, d), dim=-1)
q_mask = torch.ones(B, n)
d_mask = torch.ones(B, m)
loss = colpali_loss(Qb, Db, q_mask, d_mask)
print("ColPali in-batch loss:", loss.item())
assert torch.isfinite(loss)


# =====================================================================================
# Block #3 (line ~229, 42 lines): PatchHNSWIndex -- flatten patches into HNSW,
# exact-MaxSim rerank over the candidate pages.
# =====================================================================================

class PatchHNSWIndex:
    """Flatten all page patches into one HNSW index; rerank pages with exact MaxSim."""
    def __init__(self, dim=128):
        self.dim = dim
        self.index = hnswlib.Index(space="ip", dim=dim)  # inner product
        self.page_patches = {}   # page_id -> [m, d] float32 patch matrix
        self.label_to_page = {}  # global patch label -> page_id
        self._next = 0

    def init(self, max_patches):
        self.index.init_index(max_elements=max_patches, ef_construction=200, M=16)

    def add_page(self, page_id, patches):
        patches = patches.astype(np.float32)
        n = patches.shape[0]
        labels = np.arange(self._next, self._next + n)
        self.index.add_items(patches, labels)
        for lab in labels:
            self.label_to_page[int(lab)] = page_id
        self.page_patches[page_id] = patches
        self._next += n

    def search(self, query_patches, k_per_token=50, topn=10):
        query_patches = query_patches.astype(np.float32)
        # 1) candidate generation: nearest patches per query token -> union of pages
        candidates = set()
        for qtok in query_patches:
            labels, _ = self.index.knn_query(qtok, k=k_per_token)
            for lab in labels[0]:
                candidates.add(self.label_to_page[int(lab)])
        # 2) exact MaxSim rerank over candidate pages only
        scored = []
        for pid in candidates:
            D = self.page_patches[pid]               # [m, d]
            sim = query_patches @ D.T                # [n, m]
            scored.append((pid, sim.max(axis=1).sum()))   # MaxSim
        scored.sort(key=lambda x: -x[1])
        return scored[:topn]

if hnswlib is not None:
    rng = np.random.default_rng(42)
    dim = 16
    n_pages, patches_per_page = 8, 20
    idx = PatchHNSWIndex(dim=dim)
    idx.init(max_patches=n_pages * patches_per_page)

    page_vectors = {}
    for pid in range(n_pages):
        vecs = rng.normal(size=(patches_per_page, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        page_vectors[pid] = vecs
        idx.add_page(pid, vecs)

    # Build a query whose tokens are near page 3's patches, so we expect it to rank near top.
    target_page = 3
    query = page_vectors[target_page][:4] + rng.normal(scale=0.05, size=(4, dim)).astype(np.float32)

    results = idx.search(query, k_per_token=10, topn=5)
    print("PatchHNSWIndex top results (page_id, score):", results)
    assert len(results) > 0
    top_page_ids = [pid for pid, _ in results]
    assert target_page in top_page_ids, "expected the near-duplicate page to be a top candidate"
else:
    print("SKIP(dependency): hnswlib not installed -- block #3 (PatchHNSWIndex) not executed.")


# =====================================================================================
# Block #0 (line ~35): SigLIP cross-modal embedding -- needs a `transformers` model
# download from the Hugging Face Hub over the network. Not executed.
# SKIP(network): would call AutoModel.from_pretrained("google/siglip-base-patch16-224").
# =====================================================================================


# =====================================================================================
# Block #4 (line ~287): full colpali-engine + Qwen2-VL end-to-end OCR-free RAG pipeline.
# Needs GPU-sized vision-language models and network downloads (ColPali + Qwen2-VL-7B).
# SKIP(needs-gpu / network): not executed.
# =====================================================================================


# =====================================================================================
# Block #5 (line ~359): reciprocal_rank_fusion -- pure-Python, standalone, CPU-runnable.
# =====================================================================================

def reciprocal_rank_fusion(rankings, k=60):
    """rankings: list of ranked lists, each a list of page_ids best-first."""
    scores = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda p: -scores[p])

colpali_order = ["p17", "p4", "p23", "p9"]
bm25_order    = ["p4", "p17", "p88", "p23"]   # from OCR + BM25
fused = reciprocal_rank_fusion([colpali_order, bm25_order])
print("RRF fused order:", fused)
# p17 and p4 each appear at ranks {0,1} across the two lists -> tied top scores; the
# remaining pages are strictly lower. Check the top-2 set and that all pages survive.
assert set(fused[:2]) == {"p17", "p4"}
assert set(fused) == {"p17", "p4", "p23", "p9", "p88"}


# =====================================================================================
# Block #6 (line ~391, 18 lines): nDCG@k evaluation
# =====================================================================================

def dcg(relevances):
    relevances = np.asarray(relevances, dtype=float)
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum((2**relevances - 1) / discounts)

def ndcg_at_k(retrieved_rels, ideal_rels, k=5):
    """retrieved_rels: relevance of each retrieved page, in retrieved order.
       ideal_rels: all true relevances sorted descending (for the ideal ranking)."""
    actual = dcg(retrieved_rels[:k])
    ideal  = dcg(sorted(ideal_rels, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0

# Retrieved pages had relevances [1, 0, 1, 0, 0]; one other relevant page (rel 1)
# existed but was missed, so the ideal top-5 is [1, 1, 1, 0, 0].
# NOTE: the book's original snippet passed ideal_rels=[1, 1, 1, 1, 0] (four 1s), which
# contradicts its own prose ("one other relevant page ... missed" implies three 1s
# total: two retrieved + one missed). Fixed here (and in the .md source) to
# [1, 1, 1, 0, 0] to match the stated narrative.
ndcg_value = round(ndcg_at_k([1, 0, 1, 0, 0], [1, 1, 1, 0, 0], k=5), 4)
print(ndcg_value)
assert 0.0 <= ndcg_value <= 1.0
assert abs(ndcg_value - 0.7039) < 1e-3


print("\nAll CPU-runnable blocks executed successfully.")
sys.exit(0)
