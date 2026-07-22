"""
Runs the CPU-runnable Python code blocks from:
    content/09-rag-retrieval/02-vector-databases-ann.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order -- exactly as the chapter's own capstone block (#8) does, since later
blocks reuse classes (IVFFlat, HNSW) defined by earlier blocks -- with small
glue/fixtures added so every block actually executes on tiny CPU data.

Tested blocks:
    #0 (line ~38)  brute_force_knn / brute_force_batch
    #1 (line ~93)  curse-of-dimensionality distance-concentration experiment
    #2 (line ~139) IVFFlat class (train + search)
    #3 (line ~232) ProductQuantizer class (train + encode + ADC search)
    #5 (line ~335) HNSW class (insert + search)
    #7 (line ~517) FAISS HNSWFlat + refine usage
    #8 (line ~560) evaluate() + Flat/IVF/HNSW end-to-end recall benchmark

Skipped blocks:
    #4 -- non-python: ASCII-art diagram of HNSW's layered graph structure.
    #6 -- non-python: plain-text printed recall/ef table (illustrative output).

`faiss` is an optional third-party dependency not guaranteed to be present in
CI. It is imported defensively at module scope; if unavailable, block #7's
logic is defined-not-executed and reported as SKIP(optional dep: faiss).

For blocks #7 and #8, the book's own example data sizes (N up to 200,000
vectors, 768-d; or building a full HNSW graph over 20,000 points in pure
Python) are far too large for a CPU test with a ~60s budget. Per the "minimal
honest glue" rule, the *algorithms* are copied verbatim; only the fixture
sizes (N, d, n_list, M, ef sweep values) are shrunk to tiny values that still
faithfully exercise every code path (train, assign, probe, insert, greedy
descent, beam search, LUT-based ADC scoring, recall computation).
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


# ============================================================================
# Block #0 (line ~38): brute-force k-NN -- verbatim from the book
# ============================================================================

def brute_force_knn(data: np.ndarray, q: np.ndarray, k: int = 10):
    """
    Exact k-NN under squared-L2 distance.
    data: (N, d) float32 corpus.   q: (d,) float32 query.
    Returns (indices, distances) of the k closest vectors, sorted ascending.
    """
    # ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2.  The ||q||^2 term is constant
    # across all x, so it does not affect the ranking and we can drop it.
    # We keep the full form here for an honest distance value.
    diffs = data - q                  # (N, d) broadcast subtraction
    dists = np.einsum("nd,nd->n", diffs, diffs)  # (N,) squared L2, fused

    # argpartition is O(N): it finds the k smallest without fully sorting.
    idx = np.argpartition(dists, k)[:k]
    # Then sort just those k for a clean ranked list.
    idx = idx[np.argsort(dists[idx])]
    return idx, dists[idx]


# For a *batch* of queries, prefer the GEMM trick:
def brute_force_batch(data: np.ndarray, queries: np.ndarray, k: int = 10):
    """data: (N,d), queries: (Q,d) -> top-k indices per query, (Q,k)."""
    # ||x-q||^2 = ||x||^2 - 2 q.x + ||q||^2. Compute the cross term with one GEMM.
    x_norm = np.einsum("nd,nd->n", data, data)        # (N,)
    cross  = queries @ data.T                          # (Q, N)  the expensive part
    dists  = x_norm[None, :] - 2.0 * cross             # (Q, N), q_norm dropped
    return np.argpartition(dists, k, axis=1)[:, :k]


def test_block0_brute_force():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((500, 16)).astype("float32")
    q = rng.standard_normal(16).astype("float32")

    idx, dists = brute_force_knn(data, q, k=5)
    assert idx.shape == (5,) and dists.shape == (5,)
    assert np.all(np.diff(dists) >= -1e-6), "results must be sorted ascending"

    # cross-check against an independent exact computation
    all_d = ((data - q) ** 2).sum(1)
    true_top5 = set(np.argsort(all_d)[:5].tolist())
    assert set(idx.tolist()) == true_top5, "brute_force_knn must match exact top-k"

    queries = rng.standard_normal((7, 16)).astype("float32")
    batch_idx = brute_force_batch(data, queries, k=5)
    assert batch_idx.shape == (7, 5)
    for qi in range(7):
        all_d_q = ((data - queries[qi]) ** 2).sum(1)
        true_top5_q = set(np.argsort(all_d_q)[:5].tolist())
        assert set(batch_idx[qi].tolist()) == true_top5_q

    print("[block #0] brute_force_knn / brute_force_batch match exact top-k -- OK")


# ============================================================================
# Block #1 (line ~93): curse-of-dimensionality distance-concentration demo
# -- verbatim from the book
# ============================================================================

def test_block1_curse_of_dimensionality():
    rng = np.random.default_rng(0)

    ratios = {}
    for d in [2, 10, 100, 1000]:
        X = rng.random((10_000, d))          # 10k uniform points in [0,1]^d
        q = rng.random(d)
        dists = np.linalg.norm(X - q, axis=1)
        dmin, dmax = dists.min(), dists.max()
        # "contrast": how much farther the farthest point is vs the nearest
        print(f"d={d:5d}  dmin={dmin:7.3f}  dmax={dmax:7.3f}  "
              f"ratio dmax/dmin={dmax/dmin:5.2f}")
        ratios[d] = dmax / dmin
    # d=    2  ... ratio ~ 50x      (nearest is dramatically nearer)
    # d= 1000  ... ratio ~ 1.1x     (everything is roughly equidistant)

    # Sanity: the contrast the book claims should collapse as d grows.
    assert ratios[2] > ratios[100] > ratios[1000], (
        "distance contrast (dmax/dmin) should shrink monotonically as d grows"
    )
    assert ratios[1000] < 2.0, "at d=1000 nearest/farthest should be nearly equidistant"
    print("[block #1] distance-concentration ratios collapse with dimension -- OK")


# ============================================================================
# Block #2 (line ~139): IVFFlat class -- verbatim from the book
# ============================================================================

class IVFFlat:
    """Inverted-file index with exact (flat) distances inside each cell."""
    def __init__(self, n_list: int = 256, n_iter: int = 25, seed: int = 0):
        self.n_list = n_list
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def _kmeans(self, X):
        n = X.shape[0]
        # init centroids from random data points (k-means++ is better in prod)
        C = X[self.rng.choice(n, self.n_list, replace=False)].copy()
        for _ in range(self.n_iter):
            # assign: nearest centroid for each point via the (a-b)^2 GEMM trick
            d2 = (np.einsum("nd,nd->n", X, X)[:, None]
                  - 2 * X @ C.T
                  + np.einsum("kd,kd->k", C, C)[None, :])   # (n, n_list)
            assign = d2.argmin(1)
            for j in range(self.n_list):                    # update step
                pts = X[assign == j]
                if len(pts):
                    C[j] = pts.mean(0)
        return C

    def train(self, X):
        self.centroids = self._kmeans(X)
        # assign every vector to its nearest centroid -> build posting lists
        d2 = (np.einsum("nd,nd->n", X, X)[:, None]
              - 2 * X @ self.centroids.T
              + np.einsum("kd,kd->k", self.centroids, self.centroids)[None, :])
        assign = d2.argmin(1)
        self.lists = [np.where(assign == j)[0] for j in range(self.n_list)]
        self.data = X
        return self

    def search(self, q, k=10, n_probe=8):
        # 1) find the n_probe nearest centroids to the query
        cd = np.einsum("kd,kd->k", self.centroids - q, self.centroids - q)
        probes = np.argpartition(cd, n_probe)[:n_probe]
        # 2) gather candidate ids from those lists and do exact distances
        cand = np.concatenate([self.lists[j] for j in probes])
        if len(cand) == 0:
            return np.array([], dtype=int)
        dists = np.einsum("nd,nd->n", self.data[cand] - q, self.data[cand] - q)
        order = np.argpartition(dists, min(k, len(cand) - 1))[:k]
        order = order[np.argsort(dists[order])]
        return cand[order]


# ============================================================================
# Block #3 (line ~232): ProductQuantizer class -- verbatim from the book
# ============================================================================

class ProductQuantizer:
    def __init__(self, m=8, k_star=256, n_iter=20, seed=0):
        self.m, self.k_star, self.n_iter = m, k_star, n_iter
        self.rng = np.random.default_rng(seed)

    def train(self, X):
        n, d = X.shape
        assert d % self.m == 0, "d must be divisible by m"
        self.dsub = d // self.m
        # one codebook per subspace: (m, k_star, dsub)
        self.codebooks = np.zeros((self.m, self.k_star, self.dsub), np.float32)
        for j in range(self.m):
            sub = X[:, j*self.dsub:(j+1)*self.dsub]          # (n, dsub)
            C = sub[self.rng.choice(n, self.k_star, replace=False)].copy()
            for _ in range(self.n_iter):                     # k-means per subspace
                d2 = (np.einsum("nd,nd->n", sub, sub)[:, None]
                      - 2 * sub @ C.T
                      + np.einsum("kd,kd->k", C, C)[None, :])
                a = d2.argmin(1)
                for c in range(self.k_star):
                    pts = sub[a == c]
                    if len(pts): C[c] = pts.mean(0)
            self.codebooks[j] = C
        return self

    def encode(self, X):
        """Map each vector to m uint8 codes."""
        n = X.shape[0]
        codes = np.zeros((n, self.m), np.uint8)
        for j in range(self.m):
            sub = X[:, j*self.dsub:(j+1)*self.dsub]
            C = self.codebooks[j]
            d2 = (np.einsum("nd,nd->n", sub, sub)[:, None]
                  - 2 * sub @ C.T
                  + np.einsum("kd,kd->k", C, C)[None, :])
            codes[:, j] = d2.argmin(1)                       # nearest centroid id
        return codes

    def search(self, q, codes, k=10):
        """ADC: build the per-subspace LUT, then sum lookups for every code."""
        # 1) precompute LUT: distance from each query sub-block to all centroids
        lut = np.empty((self.m, self.k_star), np.float32)
        for j in range(self.m):
            qsub = q[j*self.dsub:(j+1)*self.dsub]            # (dsub,)
            diff = self.codebooks[j] - qsub                  # (k_star, dsub)
            lut[j] = np.einsum("kd,kd->k", diff, diff)       # (k_star,)
        # 2) approximate distance to every db vector = sum of m table lookups
        approx = np.zeros(codes.shape[0], np.float32)
        for j in range(self.m):
            approx += lut[j, codes[:, j]]                    # vectorized gather
        idx = np.argpartition(approx, k)[:k]
        return idx[np.argsort(approx[idx])]


def test_block3_product_quantizer():
    rng = np.random.default_rng(1)
    # tiny fixture: d=16 split into m=4 subspaces of dsub=4; k_star shrunk
    # from the book's default 256 to 16 so it fits a small training set.
    n, d = 300, 16
    X = rng.standard_normal((n, d)).astype("float32")

    pq = ProductQuantizer(m=4, k_star=16, n_iter=10, seed=1)
    pq.train(X)
    assert pq.codebooks.shape == (4, 16, 4)

    codes = pq.encode(X)
    assert codes.shape == (n, 4)
    assert codes.dtype == np.uint8

    q = rng.standard_normal(d).astype("float32")
    approx_top5 = pq.search(q, codes, k=5)
    assert approx_top5.shape == (5,)

    # ADC is approximate but should still beat random guessing: overlap with
    # the true brute-force top-10 should be non-trivial on this tiny corpus.
    true_d = ((X - q) ** 2).sum(1)
    true_top10 = set(np.argsort(true_d)[:10].tolist())
    approx_top10 = set(pq.search(q, codes, k=10).tolist())
    overlap = len(true_top10 & approx_top10)
    assert overlap >= 1, "PQ/ADC search should recover at least some true neighbors"

    print(f"[block #3] ProductQuantizer trained, encoded {n} vectors to "
          f"{pq.m} bytes/vec, ADC top10 overlap with exact = {overlap}/10 -- OK")


# ============================================================================
# Block #5 (line ~335): HNSW class -- verbatim from the book
# ============================================================================

def l2(a, b):
    d = a - b
    return float(d @ d)               # squared L2

class HNSW:
    def __init__(self, data, M=16, ef_construction=100, seed=2):
        self.data = data
        self.M = M                        # max neighbors per node (layer > 0)
        self.M0 = 2 * M                   # layer 0 gets more links (paper heuristic)
        self.efc = ef_construction
        self.mL = 1.0 / math.log(M)       # level-generation normalizer
        self.rng = np.random.default_rng(seed)
        self.graph = []                   # graph[node][layer] -> list of neighbor ids
        self.levels = []                  # top layer index of each node
        self.entry = None                 # current global entry point
        self.top = -1                     # current top layer

    def _rand_level(self):
        # exponentially-decaying layer assignment (skip-list style)
        return int(-math.log(self.rng.random()) * self.mL)

    def _search_layer(self, q, entry_points, ef, layer):
        """Best-first search within one layer; returns ef nearest as (dist,id)."""
        visited = set(entry_points)
        cand = []   # min-heap of (dist, id): the frontier to expand
        res  = []   # max-heap of (-dist, id): the ef best found so far
        for e in entry_points:
            de = l2(q, self.data[e])
            heapq.heappush(cand, (de, e))
            heapq.heappush(res, (-de, e))
        while cand:
            dc, c = heapq.heappop(cand)
            # if the closest frontier node is worse than our current worst
            # kept result, we can stop expanding (the beam has converged).
            if -res[0][0] < dc:
                break
            for nb in self.graph[c][layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                dn = l2(q, self.data[nb])
                if dn < -res[0][0] or len(res) < ef:
                    heapq.heappush(cand, (dn, nb))
                    heapq.heappush(res, (-dn, nb))
                    if len(res) > ef:
                        heapq.heappop(res)     # keep only the ef best
        return [(-nd, n) for nd, n in res]

    def _select_neighbors(self, node, candidates, M):
        """Simple heuristic: keep the M closest candidates to `node`."""
        scored = sorted((l2(self.data[node], self.data[c]), c) for c in candidates)
        return [c for _, c in scored[:M]]

    def insert(self, i):
        level = self._rand_level()
        self.graph.append([[] for _ in range(level + 1)])
        self.levels.append(level)

        if self.entry is None:                  # first node ever
            self.entry, self.top = i, level
            return

        ep = [self.entry]
        # Phase 1: greedily descend the layers ABOVE the new node's top layer,
        # using a beam of width 1 (just find the single best entry per layer).
        for lc in range(self.top, level, -1):
            ep = [min(self._search_layer(self.data[i], ep, 1, lc))[1]]

        # Phase 2: for each layer the node lives in, find efc candidates and link.
        for lc in range(min(level, self.top), -1, -1):
            W = self._search_layer(self.data[i], ep, self.efc, lc)
            Mmax = self.M0 if lc == 0 else self.M
            neighbors = self._select_neighbors(i, [n for _, n in W], Mmax)
            self.graph[i][lc] = neighbors
            for n in neighbors:                 # add the reverse edges
                self.graph[n][lc].append(i)
                if len(self.graph[n][lc]) > Mmax:   # prune n if it got too full
                    self.graph[n][lc] = self._select_neighbors(
                        n, self.graph[n][lc], Mmax)
            ep = [n for _, n in W]

        if level > self.top:                    # this node becomes the new entry
            self.entry, self.top = i, level

    def search(self, q, k=10, ef=50):
        ep = [self.entry]
        for lc in range(self.top, 0, -1):       # descend with beam width 1
            ep = [min(self._search_layer(q, ep, 1, lc))[1]]
        W = sorted(self._search_layer(q, ep, ef, 0))   # widen the beam at layer 0
        return [n for _, n in W[:k]]


def test_block5_hnsw():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((400, 24)).astype("float32")

    hnsw = HNSW(data, M=8, ef_construction=40, seed=2)
    for i in range(len(data)):
        hnsw.insert(i)
    assert hnsw.entry is not None and hnsw.top >= 0

    q = data[7].copy()  # querying with a corpus point: it must retrieve itself
    result = hnsw.search(q, k=5, ef=50)
    assert 7 in result, "HNSW should retrieve an exact corpus point as its own neighbor"

    # recall@5 against brute force for a handful of random queries
    queries = rng.standard_normal((20, 24)).astype("float32")
    hits = 0
    for q in queries:
        true_d = ((data - q) ** 2).sum(1)
        true_top5 = set(np.argsort(true_d)[:5].tolist())
        approx = set(hnsw.search(q, k=5, ef=50))
        hits += len(true_top5 & approx)
    recall = hits / (20 * 5)
    assert recall > 0.3, f"HNSW recall@5 unexpectedly low on tiny data: {recall}"
    print(f"[block #5] HNSW built over {len(data)} points, self-retrieval OK, "
          f"recall@5={recall:.2f} -- OK")


# ============================================================================
# Block #7 (line ~517): FAISS HNSW + refine usage -- verbatim from the book,
# with fixture sizes shrunk from N=200_000/d=768 to something CPU-testable.
# SKIP(optional dep) if `faiss` is not installed in this environment.
# ============================================================================

def test_block7_faiss_hnsw():
    if faiss is None:
        print("[block #7] SKIP(optional dep): faiss is not installed in this "
              "environment; FAISS HNSW logic defined-not-executed.")
        return

    # Minimal real-library usage: FAISS HNSW with a re-ranking refine stage.
    # pip install faiss-cpu
    d, N = 32, 2_000   # shrunk from the book's d=768, N=200_000 for CPU test time
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((N, d)).astype("float32")
    faiss.normalize_L2(xb)                  # cosine == inner product on unit vectors

    # Build: HNSW with M=32 links, then wrap with exact refinement.
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40          # higher => better graph, slower build
    index.add(xb)
    index.hnsw.efSearch = 32                # the recall-latency knob at query time

    xq = rng.standard_normal((5, d)).astype("float32"); faiss.normalize_L2(xq)
    D, I = index.search(xq, k=10)           # I: (5,10) ids,  D: (5,10) similarities
    print(I.shape, D[0][:3])                # nearest-neighbor ids + scores for query 0

    assert I.shape == (5, 10) and D.shape == (5, 10)
    assert np.all(I >= 0), "all returned ids should be valid"
    print("[block #7] faiss.IndexHNSWFlat build + search -- OK")


# ============================================================================
# Block #8 (line ~560): evaluate() + Flat/IVF/HNSW end-to-end benchmark --
# verbatim logic from the book, with fixture sizes shrunk from N=20,000/d=64
# (pure-Python HNSW insert over 20k points is too slow for a CPU unit test)
# to a tiny corpus that still exercises the exact same code paths.
# ============================================================================

def evaluate(index_search_fn, data, queries, k=10, ground_truth=None):
    """Generic recall@k evaluator against a brute-force oracle."""
    if ground_truth is None:
        ground_truth = []
        for q in queries:
            dd = np.einsum("nd,nd->n", data - q, data - q)
            ground_truth.append(set(np.argpartition(dd, k)[:k].tolist()))
    recalls, t0 = [], time.perf_counter()
    for q, gt in zip(queries, ground_truth):
        pred = set(int(i) for i in index_search_fn(q, k))
        recalls.append(len(gt & pred) / k)
    dt = (time.perf_counter() - t0) / len(queries) * 1e3   # ms/query
    return float(np.mean(recalls)), dt, ground_truth


def test_block8_end_to_end_benchmark():
    rng = np.random.default_rng(7)
    # Shrunk from the book's (20_000, 64) / 200 queries so pure-Python HNSW
    # insertion finishes in a few seconds instead of minutes.
    data    = rng.standard_normal((1_200, 32)).astype("float32")
    queries = rng.standard_normal((30, 32)).astype("float32")

    # Flat oracle (recall = 1.0 by construction):
    def flat(q, k):
        dd = np.einsum("nd,nd->n", data - q, data - q)
        return np.argpartition(dd, k)[:k]
    r, ms, gt = evaluate(flat, data, queries)
    print(f"Flat     recall={r:.3f}  {ms:6.2f} ms/q")
    assert r == 1.0, "Flat search must have recall = 1.0 by construction"

    # IVF and HNSW (classes defined earlier in the chapter):
    ivf = IVFFlat(n_list=32, seed=0).train(data)
    ivf_recalls = {}
    for npb in (2, 8, 24):
        r, ms, _ = evaluate(lambda q, k: ivf.search(q, k, n_probe=npb),
                            data, queries, ground_truth=gt)
        print(f"IVF np={npb:<3d} recall={r:.3f}  {ms:6.2f} ms/q")
        ivf_recalls[npb] = r
    assert ivf_recalls[24] >= ivf_recalls[2], (
        "IVF recall should not decrease as n_probe grows toward n_list"
    )
    assert ivf_recalls[24] > 0.5, "with n_probe close to n_list, IVF recall should be high"

    hnsw = HNSW(data, M=8, ef_construction=40, seed=2)
    for i in range(len(data)):
        hnsw.insert(i)
    hnsw_recalls = {}
    for ef in (10, 50, 150):
        r, ms, _ = evaluate(lambda q, k: hnsw.search(q, k, ef=ef),
                            data, queries, ground_truth=gt)
        print(f"HNSW ef={ef:<3d} recall={r:.3f}  {ms:6.2f} ms/q")
        hnsw_recalls[ef] = r
    assert hnsw_recalls[150] >= hnsw_recalls[10] - 1e-9, (
        "HNSW recall should not decrease as ef (beam width) grows"
    )
    assert hnsw_recalls[150] > 0.5, "with a wide beam, HNSW recall should be high"

    print("[block #8] Flat/IVF/HNSW recall-latency benchmark reproduces the "
          "recall-vs-knob tradeoff -- OK")


# ============================================================================
# Run everything
# ============================================================================

if __name__ == "__main__":
    test_block0_brute_force()
    test_block1_curse_of_dimensionality()
    test_block3_product_quantizer()
    test_block5_hnsw()
    test_block7_faiss_hnsw()
    test_block8_end_to_end_benchmark()
    print("\nAll runnable blocks in 09-rag-retrieval/02-vector-databases-ann.md "
          "executed successfully.")
