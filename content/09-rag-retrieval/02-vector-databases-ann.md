# 9.2 Vector Databases & Approximate Nearest Neighbor Search

In the previous chapter, [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html), we learned how to turn a chunk of text into a dense vector $\mathbf{x} \in \mathbb{R}^d$ such that semantically similar passages land close together in vector space. That is half of a retrieval system. The other half is the problem this chapter solves: given a query vector $\mathbf{q}$ and a corpus of $N$ vectors — where $N$ might be ten million, or ten billion — find the $k$ vectors closest to $\mathbf{q}$ *fast enough to put in the critical path of an LLM request*. A user will not wait two seconds for retrieval before the model even starts generating. You have a few milliseconds.

This is the **nearest neighbor search** problem, and it is deceptively hard. The naive solution — compare the query to every vector — is correct but linear in $N$. At a billion vectors, that is a billion dot products *per query*. The entire field of **Approximate Nearest Neighbor (ANN)** search exists to trade a sliver of correctness for orders of magnitude in speed. The systems that productionize these algorithms — FAISS, Milvus, Qdrant, Weaviate, pgvector, ScaNN — are **vector databases**. By the end of this chapter you will understand the three algorithm families that power essentially all of them (graphs, inverted files, and quantization), you will have built a brute-force searcher, a product quantizer, and a working HNSW index from scratch, and you will be able to reason quantitatively about the recall-latency-memory triangle that governs every deployment decision.

This is foundational plumbing for [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html), [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html), and [Memory Systems for Agents](../08-agents-harness/05-agent-memory.html). It is also one of the most reliably asked ML-system-design topics in interviews, which we address directly.

## The Problem: Exact Nearest Neighbor Search

### Formalizing the query

We have a database of vectors $\mathcal{D} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$, each $\mathbf{x}_i \in \mathbb{R}^d$. Given a query $\mathbf{q} \in \mathbb{R}^d$ and a distance function $\operatorname{dist}(\cdot, \cdot)$, the **$k$-nearest-neighbor** ($k$-NN) problem is to return the set

$$
\operatorname{kNN}(\mathbf{q}) = \underset{S \subseteq \mathcal{D},\, |S|=k}{\arg\min} \sum_{\mathbf{x} \in S} \operatorname{dist}(\mathbf{q}, \mathbf{x}),
$$

i.e. the $k$ database vectors with the smallest distance to $\mathbf{q}$. The two distance functions that matter in practice are **squared Euclidean (L2)** distance and, far more commonly for normalized text embeddings, **cosine similarity** (equivalently, inner product on unit vectors):

$$
\operatorname{dist}_{L2}(\mathbf{q}, \mathbf{x}) = \|\mathbf{q} - \mathbf{x}\|_2^2 = \sum_{j=1}^{d}(q_j - x_j)^2, \qquad
\operatorname{sim}_{\cos}(\mathbf{q}, \mathbf{x}) = \frac{\mathbf{q}^\top \mathbf{x}}{\|\mathbf{q}\|\,\|\mathbf{x}\|}.
$$

A crucial identity ties these together. For unit-normalized vectors ($\|\mathbf{q}\| = \|\mathbf{x}\| = 1$):

$$
\|\mathbf{q} - \mathbf{x}\|_2^2 = \|\mathbf{q}\|^2 + \|\mathbf{x}\|^2 - 2\,\mathbf{q}^\top\mathbf{x} = 2 - 2\,\mathbf{q}^\top\mathbf{x}.
$$

So **minimizing L2 distance is equivalent to maximizing inner product** when everything is normalized. This is why most retrieval pipelines L2-normalize embeddings at index time: it lets you use a single index that is simultaneously a cosine index and a Euclidean index. The flip side — **maximum inner product search (MIPS)** on *un*-normalized vectors — is genuinely harder, because inner product is not a metric (it violates the triangle inequality), which breaks many tree- and graph-based pruning arguments.

### Brute force: the baseline you must always have

The exact answer is computed by scanning every vector. Done in a vectorized way, it is just a matrix-vector (or matrix-matrix, for a batch of queries) product:

```python
import numpy as np

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
```

Two lessons hide in this code. First, the dominant cost is the `queries @ data.T` GEMM (general matrix multiply), which is $O(Q \cdot N \cdot d)$ FLOPs but maps perfectly onto BLAS / a GPU — see [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html). On a modern GPU, brute force over a few million vectors is genuinely fast and is often the *right* answer; FAISS ships a `IndexFlatIP`/`IndexFlatL2` precisely for this. Second, brute force gives recall = 1.0 by definition, so it is your **ground-truth oracle** for measuring how approximate the approximate methods are. Never ship an ANN index without measuring its recall against brute force on a held-out query set.

!!! example "Worked example: when does brute force stop being free?"

    Suppose $N = 1{,}000{,}000$ documents, $d = 768$ (a typical sentence-embedding dimension), stored in `float32`. The corpus is $N \cdot d \cdot 4 = 10^6 \cdot 768 \cdot 4 \approx 3.07$ GB — it fits in RAM and even in a single GPU's memory.

    One query is a $(1 \times d) \times (d \times N)$ product: about $2 N d = 2 \cdot 10^6 \cdot 768 \approx 1.5 \times 10^9$ FLOPs. A GPU at, say, 20 TFLOP/s effective on this memory-bound op finishes one query in well under a millisecond — but the op is **memory-bound**: it must read all 3 GB of vectors, so latency is roughly $3.07\,\text{GB} / \text{(memory bandwidth)}$. At ~1.5 TB/s that is ~2 ms per query just to stream the data, regardless of FLOPs.

    Now scale to $N = 10^9$. The corpus is 3 TB — it no longer fits on one machine, the per-query scan reads 3 TB, and at 1.5 TB/s that is **2 seconds per query**. This is the wall. ANN exists to turn that 2 seconds into ~2 milliseconds by *not looking at most of the vectors*.

## The Curse of Dimensionality

Why can't we just use a $k$-d tree, the classic exact NN data structure from any algorithms course? Because in high dimensions, *space itself stops behaving the way our 2-D and 3-D intuitions expect*. This phenomenon is the **curse of dimensionality**, and understanding it is the key to understanding why ANN methods look the way they do.

### Distances concentrate

Consider $N$ points drawn uniformly at random in the unit hypercube $[0,1]^d$. As $d$ grows, the distance from a query to its *nearest* neighbor and to its *farthest* neighbor become nearly identical. Concretely, a classic result (Beyer et al., *When Is "Nearest Neighbor" Meaningful?*) shows that under broad conditions,

$$
\lim_{d \to \infty} \frac{\mathbb{E}[\operatorname{dist}_{\max}] - \mathbb{E}[\operatorname{dist}_{\min}]}{\mathbb{E}[\operatorname{dist}_{\min}]} \to 0.
$$

The *contrast* between near and far vanishes. We can feel this with a one-line experiment:

```python
import numpy as np
rng = np.random.default_rng(0)

for d in [2, 10, 100, 1000]:
    X = rng.random((10_000, d))          # 10k uniform points in [0,1]^d
    q = rng.random(d)
    dists = np.linalg.norm(X - q, axis=1)
    dmin, dmax = dists.min(), dists.max()
    # "contrast": how much farther the farthest point is vs the nearest
    print(f"d={d:5d}  dmin={dmin:7.3f}  dmax={dmax:7.3f}  "
          f"ratio dmax/dmin={dmax/dmin:5.2f}")
# d=    2  ... ratio ~ 50x      (nearest is dramatically nearer)
# d= 1000  ... ratio ~ 1.1x     (everything is roughly equidistant)
```

At $d=2$ the farthest point is dozens of times farther than the nearest, so "nearest neighbor" is meaningful and a tree can prune aggressively. At $d=1000$ the ratio collapses toward 1: the nearest and farthest neighbors are almost the same distance away.

{{fig:curse-distance-concentration}}

### Why this kills exact structures

A $k$-d tree prunes a branch when the splitting hyperplane is farther from the query than the best candidate found so far. In high dimensions, because distances concentrate and because a query's "ball" of radius (best-so-far) intersects almost every cell, **almost no branches can be pruned**. The tree degrades to a linear scan — with worse constants than brute force, because of the pointer chasing. Empirically, $k$-d trees lose to brute force somewhere around $d \approx 20$–$30$.

There is a second, geometric way to see the curse. The volume of a $d$-dimensional unit ball relative to the enclosing cube goes to zero super-exponentially:

$$
V_{\text{ball}}(d) = \frac{\pi^{d/2}}{\Gamma\!\left(\frac{d}{2}+1\right)}, \qquad \frac{V_{\text{ball}}}{V_{\text{cube}}} \xrightarrow{d \to \infty} 0.
$$

Almost all the volume of a high-dimensional cube is in its corners, far from the center. Random data is spread thin across an enormous space, and "locality" — the property tree structures exploit — barely exists.

!!! note "Aside: but real embeddings aren't uniform"

    The curse is stated for *uniformly random* data. Real text and image embeddings live on a low-dimensional **manifold** inside $\mathbb{R}^d$ — they have an *intrinsic dimensionality* far below the *ambient* $d$. This is exactly why ANN works at all in practice: the methods below exploit that hidden low-dimensional structure (clusters, neighborhoods, smooth directions). The curse tells you that *generic* exact methods fail; the *blessing* of structure is what the good ANN methods cash in on.

## Inverted Files (IVF): Search by Coarse Partition

The first practical idea is brutally simple: **don't search the whole database, search a neighborhood of it.** Partition the vector space into cells, figure out which cell the query falls in, and scan only that cell (plus a few neighbors). This is the **Inverted File** index, IVF, named by analogy to the inverted index of classical text search.

### Building an IVF index

1. **Train** by running $k$-means on a sample of the corpus to find $n_{\text{list}}$ **centroids** $\{\mathbf{c}_1, \dots, \mathbf{c}_{n_{\text{list}}}\}$. These centroids are the *coarse quantizer*. A common heuristic is $n_{\text{list}} \approx \sqrt{N}$.
2. **Assign** every database vector to its nearest centroid. The index is then a set of $n_{\text{list}}$ lists (the "inverted lists" or *posting lists*), where list $j$ holds the ids of all vectors assigned to centroid $\mathbf{c}_j$.
3. **Search** by finding the $n_{\text{probe}}$ centroids nearest to $\mathbf{q}$, then doing brute force *only* over the vectors in those $n_{\text{probe}}$ lists.

```python
import numpy as np

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
```

### The IVF tradeoff knob: `n_probe`

`n_probe` is the single most important search-time parameter. With `n_probe = 1` you scan only the query's own cell — blazing fast, but you miss any true neighbor that happens to sit just across a cell boundary (the **boundary problem**). With `n_probe = n_list` you scan everything and recover exact brute force. In between, recall rises monotonically with `n_probe` while latency rises roughly linearly. The expected number of vectors scanned is approximately

$$
\mathbb{E}[\text{scanned}] \approx N \cdot \frac{n_{\text{probe}}}{n_{\text{list}}},
$$

so with $n_{\text{list}} = \sqrt{N}$ and $n_{\text{probe}} = 8$, you scan on the order of $8\sqrt{N}$ vectors instead of $N$. For $N = 10^9$ that is $8 \cdot 31{,}623 \approx 2.5 \times 10^5$ vectors per query — a **4000×** reduction.

{{fig:ivf-voronoi-probe}}

IVF alone solves the *speed* problem but not the *memory* problem: every vector is still stored in full `float32`. For a billion 768-d vectors that is 3 TB of RAM. To break the memory wall we need quantization.

## Product Quantization (PQ): Compressing the Vectors

Product Quantization, introduced by Jégou, Douze & Schmid (*Product Quantization for Nearest Neighbor Search*, 2011), is the idea that lets a vector database hold *billions* of vectors in *gigabytes* of RAM. It is one of the most elegant ideas in the field.

### The core trick: split, then quantize each piece

A scalar quantizer maps a single number to one of $K$ codebook values. To quantize a whole $d$-dimensional vector to $K$ codewords, you would need $K$ exponentially large to cover the space — infeasible. PQ sidesteps this: **split the vector into $m$ contiguous subvectors and quantize each subvector independently** with its own small codebook of $k^*$ entries (typically $k^* = 256$, so each code fits in one byte).

Split $\mathbf{x} \in \mathbb{R}^d$ into $m$ subvectors $\mathbf{x} = (\mathbf{u}^1, \dots, \mathbf{u}^m)$, each of dimension $d^* = d/m$. Learn a separate codebook $\mathcal{C}^j = \{\mathbf{c}^j_1, \dots, \mathbf{c}^j_{k^*}\}$ per subspace via $k$-means. The PQ code of $\mathbf{x}$ is the tuple of nearest-centroid indices:

$$
q(\mathbf{x}) = \big(\, q_1(\mathbf{u}^1), \dots, q_m(\mathbf{u}^m)\,\big), \quad q_j(\mathbf{u}^j) = \underset{i \in \{1,\dots,k^*\}}{\arg\min}\ \|\mathbf{u}^j - \mathbf{c}^j_i\|^2.
$$

The genius is in the arithmetic. The reconstruction codebook implicitly represents $(k^*)^m$ distinct vectors — for $k^* = 256$, $m = 8$ that is $256^8 = 2^{64}$ possible reconstructions — but you only store $m \cdot k^* \cdot d^*$ floats of codebooks and **$m$ bytes per database vector**. A 768-d float32 vector (3072 bytes) compresses to $m = 8$ bytes: a **384× memory reduction**.

### Asymmetric Distance Computation (ADC)

How do you compute the distance from a *full-precision* query to a *compressed* database vector without decompressing? You exploit the separability of squared L2 distance across the subspaces. Approximate

$$
\|\mathbf{q} - \mathbf{x}\|^2 \approx \|\mathbf{q} - q(\mathbf{x})\|^2 = \sum_{j=1}^{m} \big\|\mathbf{q}^j - \mathbf{c}^j_{\,q_j(\mathbf{x})}\big\|^2,
$$

where $\mathbf{q}^j$ is the $j$-th sub-block of the query. The trick: at query time, precompute a small **lookup table (LUT)** of size $m \times k^*$ holding the squared distance from each query sub-block to every centroid in that subspace. Then the distance to *any* database vector is just $m$ table lookups and $m{-}1$ additions — no multiplications, no decompression. This is **ADC** (the query stays "asymmetric": full precision; only the database is quantized, which keeps accuracy higher than quantizing both sides).

{{fig:pq-adc-lookup}}

```python
import numpy as np

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
```

### IVF + PQ: the workhorse of billion-scale search

The legendary FAISS index `IVF{n_list},PQ{m}` combines all of the above. IVF restricts the search to a handful of cells (solving *speed*); PQ compresses each vector to $m$ bytes (solving *memory*). A common refinement, **IVFADC with residuals**, quantizes not the raw vector but its *residual* after subtracting its coarse centroid, which dramatically improves accuracy because residuals are small and well-clustered. A final, optional **re-ranking** step (`IndexRefineFlat`) fetches the top candidates' full vectors from disk and re-scores them exactly to recover the recall that quantization lost.

!!! example "Worked example: sizing a billion-vector IVF-PQ index"

    Target: $N = 10^9$ vectors, $d = 768$, in RAM on commodity hardware.

    **Flat float32:** $10^9 \cdot 768 \cdot 4 \approx 3.07$ TB. Infeasible on one box.

    **IVF-PQ with $m=64$ bytes/vector** (a typical accuracy-preserving choice for 768-d): each vector becomes 64 bytes of PQ code plus a few bytes of id, call it ~72 bytes. Total: $10^9 \cdot 72 \approx 72$ GB — fits on a single large-memory server. The codebooks ($n_{\text{list}}$ coarse centroids + $m \cdot 256 \cdot d^*$ PQ floats) are negligible by comparison.

    **Speed:** with $n_{\text{list}} = 2^{16} \approx 65{,}536$ and $n_{\text{probe}} = 64$, you scan $\approx 10^9 \cdot 64 / 65{,}536 \approx 9.8 \times 10^5$ codes per query. Each scored vector costs only $m=64$ byte-lookups-and-adds via the ADC LUT — no float multiplies. That is the difference between a billion full dot products and ~1M cache-friendly table lookups.

    **Cost:** the recall drop. With $m=64$ bytes and re-ranking the top ~1000 candidates against full vectors on disk, you can typically recover recall@10 into the high 0.9s — but you *measure* this against brute force, never assume it.

## HNSW: Navigable Small-World Graphs

{{fig:hnsw}}

Quantization compresses; IVF partitions. The third family — **graph-based ANN** — takes a different route and is, for most in-memory workloads under ~100M vectors, the recall-latency champion. The dominant algorithm is **Hierarchical Navigable Small World (HNSW)** graphs, from Malkov & Yashunin (*Efficient and robust approximate nearest neighbor search using HNSW graphs*, 2016/2018).

### Greedy search on a proximity graph

Build a graph where each vector is a node connected to some of its near neighbors. To search, start at any node and **greedily walk downhill**: repeatedly move to whichever neighbor of the current node is closer to the query, until no neighbor improves. On a well-constructed *navigable* graph, this greedy walk converges to (approximately) the true nearest neighbor in a number of hops that grows only *logarithmically* with $N$.

The naive version gets stuck in local minima and has long-range connectivity problems. HNSW fixes both with two ideas:

1. **Small-world structure.** Inspired by Kleinberg's navigable small-world networks (the "six degrees of separation" insight), the graph mixes short links (to immediate neighbors, for accuracy) with long links (for fast traversal across the space). This gives the logarithmic hop count.
2. **A hierarchy of layers.** Like a skip list, HNSW stacks multiple graph layers. The top layer is sparse with long-range links; each lower layer is denser. A node's maximum layer is drawn from an exponentially decaying distribution, $\ell = \lfloor -\ln(\text{Uniform}(0,1)) \cdot m_L \rfloor$ with $m_L = 1/\ln(M)$. Search starts at the single top-layer entry point, greedily descends to the nearest node in each layer, then drops down — zooming in like a coarse-to-fine map.

```text
  Layer 2:   E ------------------- A                (sparse, long hops)
                 \               /
  Layer 1:   E --- B ----- C --- A --- D            (medium)
              \    |       |     |    /
  Layer 0:   E-B-F-G-B-C-H-I-C-A-J-D-K...           (dense; every node lives here)
                          ^
              query enters at top entry E, greedily descends,
              refines in layer 0 to collect the final top-k.
```

### From-scratch HNSW

Here is a complete, runnable HNSW. It is faithful to the paper's two routines — `SEARCH-LAYER` (a best-first beam search with a candidate min-heap and a result max-heap) and the layered `INSERT` — and it achieves real recall. The two build parameters are $M$ (neighbors per node) and `ef_construction` (beam width during insertion); the one search parameter is `ef` (beam width during query, which trades recall for latency).

```python
import numpy as np, heapq, math

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
```

When we ran this on 8,000 random 32-d vectors and measured recall@10 against brute force, sweeping the search beam width `ef` produced exactly the behavior the theory predicts — a smooth recall-latency dial:

```text
ef= 10   recall@10 = 0.500     (fast, low recall)
ef= 25   recall@10 = 0.673
ef= 50   recall@10 = 0.823
ef=100   recall@10 = 0.895
ef=200   recall@10 = 0.955     (slow, high recall)
```

That table *is* the recall-latency tradeoff, made concrete. You turn one knob, `ef`, and slide along the curve.

### Why HNSW dominates — and its costs

HNSW gives the best recall-per-latency of any method for in-memory data, with no training step (unlike IVF/PQ's $k$-means). Its costs are real, though:

- **Memory.** It stores full vectors *plus* the graph. Each node holds up to $M$ neighbor ids per layer; layer 0 alone is $\sim 2M$ ids $\times$ 4–8 bytes. For $M=16$, that is ~64–128 bytes of graph overhead per vector *on top of* the $4d$ bytes of the vector itself. HNSW is memory-hungry, which is why billion-scale systems combine it with PQ (`IndexHNSWPQ`).
- **Build time.** Insertion is $O(\log N)$ per node but with a large constant; building a 100M-node graph takes hours and is the dominant offline cost.
- **Updates.** Deletes are awkward (you typically tombstone and periodically rebuild). High churn favors IVF, which re-clusters more gracefully.

## ScaNN and the Anisotropic Insight

Google's **ScaNN** (Scalable Nearest Neighbors; Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020) is worth a section because it changed how the field thinks about *what* quantization should preserve. Classic PQ minimizes reconstruction error $\|\mathbf{x} - q(\mathbf{x})\|^2$ uniformly. ScaNN's insight: for **maximum inner product search**, not all reconstruction errors are equally harmful. An error *parallel* to the database vector changes its inner product with queries; an error *orthogonal* to it mostly does not. So you should penalize parallel (anisotropic) error more heavily.

ScaNN replaces the isotropic loss with an **anisotropic** one that weights the parallel residual component more than the orthogonal one:

$$
\ell_{\text{aniso}}(\mathbf{x}, \tilde{\mathbf{x}}) = \eta_{\parallel}\,\big\|\,\text{proj}_{\mathbf{x}}(\mathbf{x} - \tilde{\mathbf{x}})\,\big\|^2 + \eta_{\perp}\,\big\|\,\text{proj}_{\mathbf{x}^\perp}(\mathbf{x} - \tilde{\mathbf{x}})\,\big\|^2, \quad \eta_{\parallel} > \eta_{\perp}.
$$

By learning codebooks under this loss, ScaNN preserves the inner products that determine ranking, achieving higher recall at the same compression. Combined with a partitioning tree and a heavily SIMD-optimized in-register distance computation, ScaNN is one of the fastest libraries on standard benchmarks. The broader lesson generalizes well beyond ScaNN: **optimize the quantizer for the *task metric* (ranking by inner product), not for a generic surrogate (reconstruction MSE).**

## The Recall–Latency–Memory Triangle

Every ANN deployment is a point inside a triangle whose three corners are **recall** (how often you find the true neighbors), **latency** (how fast), and **memory** (how much RAM/disk). You can have any two cheaply; the third is the price.

### Measuring recall

Recall@$k$ is the fraction of the true top-$k$ that your approximate search returns:

$$
\text{recall@}k = \frac{|\,\text{ANN}_k(\mathbf{q}) \cap \text{exact}_k(\mathbf{q})\,|}{k}.
$$

Always compute `exact_k` with brute force on a held-out query set. A subtle but important point: **recall is not accuracy of the downstream task.** In RAG, a missed neighbor only hurts if it was the document that contained the answer. Tune recall against your *end-to-end* eval (see [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html)), not in a vacuum — recall@10 = 0.85 may be plenty.

### The knobs, summarized

| Method | Speed knob | Memory knob | Build cost | Best regime |
|---|---|---|---|---|
| Flat (brute) | — (always exact) | none (full vectors) | none | $N \lesssim$ few M; GPU; need recall = 1.0 |
| IVF-Flat | `n_probe` | none (full vectors) | $k$-means | moderate $N$, want simple + exact-ish |
| IVF-PQ | `n_probe` | $m$ (bytes/vec) | $k$-means $\times (1{+}m)$ | billion-scale, RAM-constrained |
| HNSW | `ef` | $M$ (graph degree) | high (graph build) | in-memory, $\lesssim$ 100M, top recall/latency |
| HNSW-PQ | `ef` | $M$ + $m$ | high | large + RAM-constrained |
| ScaNN | leaves/reorder | aniso codebooks | $k$-means + tree | MIPS, throughput-critical |

The mental model: **HNSW spends memory to buy recall and latency. PQ spends recall to buy memory. IVF spends recall to buy latency.** Real systems stack them — `IVF65536_HNSW32,PQ64` is a real FAISS factory string that uses an HNSW graph as the coarse quantizer over IVF cells whose contents are PQ-compressed, hitting all three corners at once.

{{fig:recall-latency-memory-triangle}}

!!! tip "Practitioner tip: the order of operations that actually works"

    1. Start with **Flat** (brute force) on a sample. It is your correctness oracle and is often fast enough below a few million vectors, *especially on a GPU*. Do not add complexity you cannot justify.
    2. When Flat is too slow or too big, reach for **HNSW** if it fits in RAM — it needs no training and gives the best recall/latency.
    3. When it does not fit in RAM, add **PQ** (HNSW-PQ or IVF-PQ) and a **re-ranking** stage that re-scores the top candidates with full-precision vectors.
    4. **Always** measure recall@k against Flat on held-out queries, and tune the knob (`ef` / `n_probe`) to the *cheapest setting that clears your downstream eval bar* — not to the highest recall you can squeeze out.

## Vector Databases: From Algorithm to System

An ANN *algorithm* (FAISS, ScaNN, hnswlib) is a library. A vector *database* wraps that algorithm with the boring-but-essential machinery that makes it a production service: persistence, replication, CRUD with consistency, **metadata filtering**, multi-tenancy, sharding across machines, and horizontal scaling. Knowing the landscape matters for system-design interviews.

- **FAISS** (Facebook AI Similarity Search) — the canonical *library*, not a database. It implements every index above (Flat, IVF, PQ, HNSW, and combinations) with CPU and GPU backends. If you build retrieval infra, you will use or reimplement FAISS. It has no server, no persistence layer, no metadata filtering — you bring those.
- **Milvus** — a distributed vector *database* built on top of FAISS/HNSW/etc., with sharding, a write-ahead log, object-storage persistence, and a query/data/index node separation for independent scaling.
- **Qdrant** — Rust, HNSW-based, with strong **payload filtering** (combine vector search with structured predicates) and good single-node ergonomics.
- **Weaviate** — HNSW-based, with hybrid (vector + keyword) search and a built-in module ecosystem.
- **pgvector** — a PostgreSQL extension adding a `vector` column type with IVF-Flat and HNSW indexes. The pragmatic winner when you already run Postgres and your $N$ is modest: no new system to operate, transactional consistency for free, trivial joins between vectors and your relational data.

### The filtering problem (where naive ANN breaks)

Real queries are rarely "find similar vectors." They are "find similar vectors *where* `tenant_id = 42` *and* `published_at > 2025-01-01`." This **filtered ANN** problem is genuinely hard, because the graph/IVF structure was built over *all* vectors, and the nearest neighbors may all be filtered out, forcing the search to wander. The two strategies — and their failure modes — are interview-classic:

- **Pre-filter:** compute the matching id set first, then search only within it. Correct, but if the filter is selective, the matching set may be scattered across the whole HNSW graph; you can lose graph connectivity and recall craters, or you fall back to brute force over the filtered set.
- **Post-filter:** run unfiltered ANN, then drop results that fail the predicate. Fast, but if the filter is selective you may retrieve $k=10$ candidates and have *zero* survive — you have to over-fetch by an unknown factor.

Modern systems (Qdrant, Milvus) use **filtered HNSW** variants that consult the predicate *during* graph traversal, skipping non-matching nodes while still using them as bridges for connectivity. This is the production frontier and a great thing to mention in a design round.

```python
# Minimal real-library usage: FAISS HNSW with a re-ranking refine stage.
# pip install faiss-cpu
import faiss, numpy as np

d, N = 768, 200_000
rng = np.random.default_rng(0)
xb = rng.standard_normal((N, d)).astype("float32")
faiss.normalize_L2(xb)                  # cosine == inner product on unit vectors

# Build: HNSW with M=32 links, then wrap with exact refinement.
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200         # higher => better graph, slower build
index.add(xb)
index.hnsw.efSearch = 64                # the recall-latency knob at query time

xq = rng.standard_normal((5, d)).astype("float32"); faiss.normalize_L2(xq)
D, I = index.search(xq, k=10)           # I: (5,10) ids,  D: (5,10) similarities
print(I.shape, D[0][:3])                # nearest-neighbor ids + scores for query 0
```

!!! warning "Common pitfall: metric mismatch between embedding and index"

    The single most common ANN bug is a mismatch between how the embedding model was trained and how the index measures distance. If your model produces cosine-normalized embeddings but you build an L2 index on un-normalized vectors (or vice versa), you get *plausible-looking but wrong* neighbors and silently degraded RAG. Two rules: (1) decide your metric up front — for modern text embeddings it is almost always cosine, so **L2-normalize at both index and query time** and use inner product; (2) verify by checking that a vector retrieves *itself* as the #1 neighbor with similarity 1.0. If it does not, your normalization or metric is wrong.

!!! interview "Interview Corner"

    **Q:** You have 500 million document embeddings ($d = 1024$) and need sub-50 ms p99 retrieval latency at high QPS. Walk me through how approximate nearest neighbor search works and how you'd design the index. Where does the approximation come from, and how do you control the error?

    **A:** Exact search is a linear scan — 500M dot products per query, hundreds of milliseconds even on a GPU, and 2 TB of float32 that won't fit on one box. ANN trades a little recall for orders of magnitude in speed and memory by *not looking at most vectors*. Three mechanisms, which I'd combine:

    1. **Partition (IVF):** $k$-means the corpus into ~$\sqrt{N}\approx 22$k cells; at query time probe only the `n_probe` nearest cells. The approximation is the **boundary problem** — true neighbors just across a cell wall get missed — and I control it with `n_probe` (higher = more cells scanned = higher recall, higher latency).

    2. **Compress (PQ):** split each vector into $m$ sub-blocks, $k$-means each into 256 centroids, store one byte per block. 1024-d float32 (4 KB) → $m{=}64$ bytes, a ~64× memory cut, so 500M vectors fit in ~35 GB RAM. The approximation is **quantization error**; I control it with $m$ (more bytes = finer = higher recall) and recover it with a **re-ranking** stage that re-scores the top ~1000 candidates against full-precision vectors.

    3. **Graph (HNSW)** as the coarse quantizer for fast cell selection, with `efSearch` as the recall-latency dial.

    Concretely I'd build `IVF_HNSW,PQ64` in FAISS/Milvus, set the metric to inner product on L2-normalized vectors (so cosine = L2), shard across machines by id, replicate for QPS, and — critically — **measure recall@10 against brute force on a held-out set** and against the end-to-end RAG eval, then pick the *cheapest* `n_probe`/`efSearch`/$m$ that clears the bar. The approximation is principled and *measured*, never assumed. If there are metadata filters, I'd use filtered-HNSW that skips non-matching nodes during traversal rather than naive pre/post-filtering.

## Putting It Together: A Tiny End-to-End Benchmark

To cement the tradeoffs, here is a self-contained harness that builds Flat, IVF, PQ, and HNSW over the same data and reports each method's recall@10 against the exact baseline. Running it is the fastest way to *feel* the triangle.

```python
import numpy as np, time

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

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    data    = rng.standard_normal((20_000, 64)).astype("float32")
    queries = rng.standard_normal((200, 64)).astype("float32")

    # Flat oracle (recall = 1.0 by construction):
    def flat(q, k):
        dd = np.einsum("nd,nd->n", data - q, data - q)
        return np.argpartition(dd, k)[:k]
    r, ms, gt = evaluate(flat, data, queries)
    print(f"Flat     recall={r:.3f}  {ms:6.2f} ms/q")

    # IVF and HNSW (classes defined earlier in the chapter):
    ivf = IVFFlat(n_list=256).train(data)
    for npb in (4, 16, 64):
        r, ms, _ = evaluate(lambda q, k: ivf.search(q, k, n_probe=npb),
                            data, queries, ground_truth=gt)
        print(f"IVF np={npb:<3d} recall={r:.3f}  {ms:6.2f} ms/q")

    hnsw = HNSW(data, M=16, ef_construction=100)
    for i in range(len(data)):
        hnsw.insert(i)
    for ef in (16, 64, 200):
        r, ms, _ = evaluate(lambda q, k: hnsw.search(q, k, ef=ef),
                            data, queries, ground_truth=gt)
        print(f"HNSW ef={ef:<3d} recall={r:.3f}  {ms:6.2f} ms/q")
```

The shape of the output is always the same story: Flat is exact but its latency grows with $N$; IVF and HNSW each expose one knob (`n_probe`, `ef`) that buys recall with latency; and if you swapped the data store to PQ codes you would watch memory drop while recall takes a measured hit. That is the entire chapter, in one runnable file.

For how these retrievers slot into a generation pipeline — query construction, multi-vector retrieval, fusion with keyword search — continue to [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html) and [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html). For the embeddings that feed all of this, revisit [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html). And because retrieval lives in the latency budget of an LLM call, the serving concerns in [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html) apply directly.

!!! key "Key Takeaways"

    - **Exact NN is a linear scan.** It is the right answer below a few million vectors (especially on a GPU) and is your non-negotiable *recall oracle* — never ship ANN without measuring recall@k against brute force.
    - **The curse of dimensionality** makes exact tree structures ($k$-d trees) degrade to linear scan above $d \approx 20$, because distances concentrate and pruning fails. ANN works only because real embeddings live on a low-dimensional manifold.
    - **Three algorithm families** cover the field: **IVF** (partition into cells, probe a few — buys *latency*), **PQ** (split + quantize subvectors to bytes via ADC lookup tables — buys *memory*, ~100–400× compression), and **HNSW** (greedy search on a hierarchical small-world graph, log-$N$ hops — best *recall/latency* but memory-hungry).
    - **Every deployment is a point in the recall–latency–memory triangle.** You get two corners cheap; the third is the cost. One knob per method slides you along the curve: `n_probe` (IVF), `ef` (HNSW), bytes $m$ (PQ).
    - **Production systems stack the families** — e.g. `IVF_HNSW,PQ` with a full-precision **re-ranking** stage — to hit all three corners, and that is exactly what FAISS factory strings express.
    - **ScaNN's lesson generalizes:** optimize the quantizer for the *task metric* (ranking by inner product), not a generic reconstruction MSE.
    - **A vector database = ANN library + persistence + sharding + metadata filtering.** FAISS is a library; Milvus/Qdrant/Weaviate/pgvector are databases. **Filtered ANN** (search under structured predicates) is the production frontier and a frequent interview probe.
    - **Watch the metric mismatch footgun:** for text embeddings, L2-normalize at index and query time so cosine = inner product = L2, and sanity-check that a vector retrieves itself with similarity 1.0.

!!! sota "State of the Art & Resources (2026)"
    Vector ANN is a mature but actively evolving field: HNSW and IVF-PQ remain the workhorses, while recent work (DiskANN, RaBitQ) pushes recall and memory efficiency to new frontiers, and managed vector databases have made billion-scale retrieval a commodity infrastructure component.

    **Foundational work**

    - [Malkov & Yashunin, *Efficient and Robust ANN Search Using HNSW Graphs* (2016/2018)](https://arxiv.org/abs/1603.09320) — the HNSW paper; the dominant graph-based ANN algorithm in production.
    - [Johnson, Douze & Jégou, *Billion-Scale Similarity Search with GPUs* (2017)](https://arxiv.org/abs/1702.08734) — the FAISS paper; introduces GPU-accelerated flat and IVF-PQ indexes.
    - [Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ICML 2020)](https://arxiv.org/abs/1908.10396) — ScaNN; shows that optimizing quantization for inner-product ranking beats MSE-based PQ.

    **Recent advances (2019–2026)**

    - [Subramanya et al., *DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node* (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) — graph-based SSD index; 95%+ recall at <3 ms on 1B vectors on commodity hardware.
    - [Gao & Long, *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for ANN Search* (SIGMOD 2024)](https://arxiv.org/abs/2405.12497) — 1-bit-per-dimension quantization with provable error bounds; consistently outperforms PQ variants at the same memory budget.

    **Open-source & tools**

    - [facebookresearch/faiss](https://github.com/facebookresearch/faiss) — the canonical ANN library; every index family (Flat, IVF, PQ, HNSW, ScaNN-style) plus GPU backends and the factory-string API.
    - [nmslib/hnswlib](https://github.com/nmslib/hnswlib) — lightweight header-only C++/Python HNSW implementation by the algorithm's authors; the reference for embedding HNSW in your own system.
    - [microsoft/DiskANN](https://github.com/microsoft/DiskANN) — composable disk+memory ANN library (Vamana graph); supports real-time updates and attribute filtering.
    - [milvus-io/milvus](https://github.com/milvus-io/milvus) — cloud-native distributed vector database supporting HNSW, IVF, DiskANN and PQ indexes with sharding, WAL, and payload filtering.

    **Go deeper**

    - [erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks) — the standard recall-vs-QPS benchmark harness; run it locally to reproduce the recall-latency curves from this chapter on real data.
    - [Pinecone, *A Developer's Guide to Approximate Nearest Neighbor Algorithms* (2024)](https://www.pinecone.io/learn/a-developers-guide-to-ann-algorithms/) — practical survey of HNSW, IVF, DiskANN and SPANN with storage-tier trade-offs.

## Further reading

- Hervé Jégou, Matthijs Douze, Cordelia Schmid, *Product Quantization for Nearest Neighbor Search* (IEEE TPAMI, 2011) — the original PQ and ADC.
- Yury Malkov, Dmitry Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs* (2016/2018) — HNSW.
- Ruiqi Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ICML 2020) — ScaNN.
- Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, Uri Shaft, *When Is "Nearest Neighbor" Meaningful?* (ICDT 1999) — the curse of dimensionality, formalized.
- Jeff Johnson, Matthijs Douze, Hervé Jégou, *Billion-Scale Similarity Search with GPUs* (2017) — the FAISS paper.
- Jon Kleinberg, *Navigation in a Small World* (Nature, 2000) — the navigable small-world result underpinning graph ANN.
- The **FAISS** library (`facebookresearch/faiss`) and its wiki — the canonical reference implementation and index "factory" documentation.

## Exercises

**1.** (Conceptual) The chapter claims that once vectors are L2-normalized, a single index serves simultaneously as a cosine index and a Euclidean index. Starting from the identity $\|\mathbf{q} - \mathbf{x}\|_2^2 = 2 - 2\,\mathbf{q}^\top\mathbf{x}$ for unit vectors, show that ranking the corpus by *ascending* squared-L2 distance produces the *exact same order* as ranking by *descending* cosine similarity. Then explain what the "metric mismatch footgun" from the chapter's warning box would do to your results, and describe the one-line sanity check the chapter recommends.

??? note "Solution"
    For unit vectors, $\|\mathbf{q}-\mathbf{x}\|_2^2 = 2 - 2\,\mathbf{q}^\top\mathbf{x}$ and, since $\|\mathbf{q}\|=\|\mathbf{x}\|=1$, the cosine similarity is $\operatorname{sim}_{\cos}(\mathbf{q},\mathbf{x}) = \mathbf{q}^\top\mathbf{x}$. Substituting,

    $$
    \|\mathbf{q}-\mathbf{x}\|_2^2 = 2 - 2\,\operatorname{sim}_{\cos}(\mathbf{q},\mathbf{x}).
    $$

    This is an *affine, strictly decreasing* function of the cosine similarity: distance $= 2 - 2\,s$ with slope $-2 < 0$. A strictly decreasing map preserves the ranking but reverses its direction, so the vector with the *smallest* L2 distance is exactly the vector with the *largest* cosine similarity, and this holds for the entire ordering, not just the top-1. Concretely, for any two candidates $\mathbf{x}_a, \mathbf{x}_b$,

    $$
    \|\mathbf{q}-\mathbf{x}_a\|_2^2 < \|\mathbf{q}-\mathbf{x}_b\|_2^2 \iff \mathbf{q}^\top\mathbf{x}_a > \mathbf{q}^\top\mathbf{x}_b,
    $$

    so the top-$k$ sets are identical. That is why normalizing lets one index answer both metrics.

    The footgun: if the embedding model emits cosine-normalized vectors but you build an L2 index over *un-normalized* vectors (or mix an inner-product index with un-normalized data), the equivalence above no longer holds — the $\|\mathbf{x}\|^2$ term stops being constant across candidates, so longer vectors get unfairly penalized or favored. The neighbors look plausible but are wrong, silently degrading RAG. Sanity check: query the index with a vector that is *in* the corpus and confirm it retrieves *itself* as the rank-1 neighbor with similarity $1.0$ (distance $0$). If it does not, your normalization or metric is misconfigured.

**2.** (Quantitative) You deploy an IVF-Flat index over $N = 10^{8}$ vectors, choosing $n_{\text{list}} = \sqrt{N}$ centroids as the chapter's heuristic suggests, and search with $n_{\text{probe}} = 16$. (a) Roughly how many database vectors does one query scan? (b) What speedup is that over exact brute force? (c) The heuristic keeps $n_{\text{list}} = \sqrt{N}$; as $N$ grows, does the speedup at fixed $n_{\text{probe}}$ get better or worse, and why?

??? note "Solution"
    (a) With $n_{\text{list}} = \sqrt{10^{8}} = 10^{4}$ centroids, the expected number scanned is

    $$
    \mathbb{E}[\text{scanned}] \approx N \cdot \frac{n_{\text{probe}}}{n_{\text{list}}} = 10^{8} \cdot \frac{16}{10^{4}} = 1.6 \times 10^{5} \text{ vectors.}
    $$

    (b) Brute force scans all $N = 10^{8}$. The speedup is

    $$
    \frac{N}{\mathbb{E}[\text{scanned}]} = \frac{n_{\text{list}}}{n_{\text{probe}}} = \frac{10^{4}}{16} = 625\times.
    $$

    (This ignores the cost of the coarse step — finding the $n_{\text{probe}}$ nearest of $n_{\text{list}} = 10^4$ centroids — which is a small linear scan of $10^4$ dot products, negligible beside the $1.6\times 10^5$ candidate distances.)

    (c) Better. With $n_{\text{list}} = \sqrt{N}$, the speedup $= n_{\text{list}}/n_{\text{probe}} = \sqrt{N}/n_{\text{probe}}$, which grows as $\sqrt{N}$. Going from $N = 10^{8}$ to $N = 10^{9}$ raises the speedup from $625\times$ to $\sqrt{10^9}/16 \approx 1976\times$. The catch not captured by this arithmetic is recall: at fixed $n_{\text{probe}}$ you scan a *shrinking fraction* of the corpus, so the boundary problem worsens and recall tends to fall — which is why real deployments raise $n_{\text{probe}}$ as $N$ grows and always measure recall against brute force.

**3.** (Quantitative) You must fit $N = 10^{8}$ embeddings of dimension $d = 1024$ into RAM using Product Quantization with $m = 32$ subquantizers and $k^{*} = 256$ centroids per subspace. (a) How many bytes does one vector's PQ code take, and what is the compression ratio versus float32? (b) How many distinct vectors can the codebook *implicitly* represent? (c) How much RAM do the PQ codes for all $N$ vectors take, and how much do the codebooks themselves take? (d) What size is the per-query ADC lookup table, and what operations does scoring one database vector cost?

??? note "Solution"
    (a) Each subquantizer index is one of $k^{*} = 256$ values, i.e. exactly one byte, and there are $m = 32$ of them, so **32 bytes per vector**. A full-precision vector is $d \cdot 4 = 1024 \cdot 4 = 4096$ bytes. Compression ratio:

    $$
    \frac{4096}{32} = 128\times.
    $$

    (b) The reconstruction is the concatenation of one chosen centroid per subspace, so the number of representable combinations is

    $$
    (k^{*})^{m} = 256^{32} = (2^{8})^{32} = 2^{256} \approx 1.16 \times 10^{77},
    $$

    an astronomically large implicit codebook stored in only 32 bytes per vector.

    (c) Codes: $N \cdot 32 = 10^{8} \cdot 32 = 3.2 \times 10^{9}$ bytes $\approx 3.2$ GB. Codebooks: $m \cdot k^{*} \cdot d^{*}$ floats, where $d^{*} = d/m = 1024/32 = 32$, so $32 \cdot 256 \cdot 32 = 262{,}144$ floats $\times 4$ bytes $\approx 1.05$ MB — negligible next to the 3.2 GB of codes (and, importantly, *fixed*, independent of $N$).

    (d) The LUT has shape $m \times k^{*} = 32 \times 256 = 8192$ float32 entries ($\approx 32$ KB), computed once per query. Scoring one database vector is then $m = 32$ table lookups (one gather per subspace using that vector's code byte) and $m - 1 = 31$ additions — **no multiplications and no decompression**.

**4.** (Implementation) The chapter mentions a **re-ranking** stage (`IndexRefineFlat`) that "fetches the top candidates' full vectors and re-scores them exactly to recover the recall that quantization lost." Extend the `ProductQuantizer` class with a method `search_rerank(self, q, codes, data, k=10, rerank=100)` that (1) uses fast ADC to shortlist the `rerank` best candidates, then (2) re-scores *only those* against their full-precision vectors in `data` with exact squared-L2 and returns the true top-$k$. Explain the cost you added and why it helps.

??? note "Solution"
    The method reuses the ADC path to build a cheap shortlist, then pays for exact distances on just that shortlist:

    ```python
    def search_rerank(self, q, codes, data, k=10, rerank=100):
        """Two-stage: ADC shortlist -> exact re-rank on full vectors."""
        rerank = min(rerank, codes.shape[0])
        # --- Stage 1: ADC (same LUT trick as `search`), cheap over ALL codes ---
        lut = np.empty((self.m, self.k_star), np.float32)
        for j in range(self.m):
            qsub = q[j*self.dsub:(j+1)*self.dsub]
            diff = self.codebooks[j] - qsub
            lut[j] = np.einsum("kd,kd->k", diff, diff)
        approx = np.zeros(codes.shape[0], np.float32)
        for j in range(self.m):
            approx += lut[j, codes[:, j]]
        # keep the `rerank` smallest approximate distances (O(N), no full sort)
        cand = np.argpartition(approx, rerank - 1)[:rerank]
        # --- Stage 2: exact squared-L2 on the shortlist's FULL vectors ---
        diffs = data[cand] - q                       # (rerank, d)
        exact = np.einsum("nd,nd->n", diffs, diffs)  # (rerank,)
        order = np.argpartition(exact, min(k, rerank - 1))[:k]
        order = order[np.argsort(exact[order])]
        return cand[order]                           # exact top-k among shortlist
    ```

    Cost added: `rerank` full-precision squared-L2 evaluations, i.e. $O(\text{rerank} \cdot d)$ float multiply-adds plus the fetch of `rerank` full vectors (the very rows PQ was compressing away). Because `rerank` (e.g. 100–1000) is tiny next to $N$, this is a small fixed surcharge on top of the $O(N)$ ADC scan.

    Why it helps: ADC ranks by *approximate* distance, so quantization error can shuffle the true near neighbors slightly out of the top-$k$ — but they usually still land within a somewhat larger shortlist of size `rerank`. Re-scoring that shortlist with exact distances removes the quantization error *for the candidates that matter*, so the final top-$k$ is ordered by true distance. This recovers most of the recall PQ lost while keeping PQ's memory savings, since only the handful of shortlisted vectors ever need full precision (in production those full vectors typically live on disk/SSD). The knob is `rerank`: larger shortlist means higher recall for more re-scoring work.

**5.** (Quantitative) HNSW assigns each node a maximum layer by $\ell = \lfloor -\ln(U) \cdot m_L \rfloor$ with $U \sim \text{Uniform}(0,1)$ and $m_L = 1/\ln(M)$. Take $M = 16$. (a) Derive $P(\ell \geq L)$ as a function of $M$ and $L$. (b) For a graph of $N = 1{,}000{,}000$ nodes, how many nodes do you expect to appear in layer $\geq 1$, layer $\geq 2$, and layer $\geq 3$? (c) Explain in one or two sentences why this geometric thinning is what gives HNSW its logarithmic search cost.

??? note "Solution"
    (a) $\ell \geq L$ (with $L$ a positive integer) exactly when $-\ln(U)\, m_L \geq L$, because the floor of a value is $\geq L$ iff the value itself is $\geq L$. Rearranging with $m_L = 1/\ln M$:

    $$
    -\ln(U) \geq L\ln M \;\Longleftrightarrow\; \ln(U) \leq -L\ln M \;\Longleftrightarrow\; U \leq M^{-L}.
    $$

    Since $U$ is uniform on $(0,1)$, $P(U \leq M^{-L}) = M^{-L}$. So

    $$
    P(\ell \geq L) = M^{-L} = \left(\tfrac{1}{16}\right)^{L}.
    $$

    (b) The expected count at layer $\geq L$ is $N \cdot M^{-L}$:

    - Layer $\geq 1$: $10^{6} \cdot 16^{-1} = 10^{6}/16 = 62{,}500$ nodes.
    - Layer $\geq 2$: $10^{6} \cdot 16^{-2} = 10^{6}/256 \approx 3{,}906$ nodes.
    - Layer $\geq 3$: $10^{6} \cdot 16^{-3} = 10^{6}/4096 \approx 244$ nodes.

    (Layer 0, $L=0$, holds all $10^6$ nodes: $M^{0}=1$.)

    (c) Each layer up holds a factor $1/M$ fewer nodes, so the number of populated layers grows like $\log_M N$, and the top layers are sparse "express lanes" with long-range links. Search starts at the single top entry point and greedily descends, covering large distances in the sparse upper layers with few hops before refining in the dense layer 0 — so the total hop count scales with the number of layers, $O(\log N)$, rather than with $N$.

**6.** (Conceptual) A tenant runs the query "documents similar to $\mathbf{q}$ *where* `tenant_id = 42`", and `tenant_id = 42` matches only $0.1\%$ of the corpus. Contrast the **pre-filter** and **post-filter** strategies for this *filtered ANN* query: give the failure mode of each under such a selective predicate, and explain why a filter-aware HNSW traversal (as in Qdrant/Milvus) avoids both.

??? note "Solution"
    **Pre-filter** first computes the matching id set ($0.1\%$ of vectors), then searches only within it. The failure mode with a highly selective predicate: those matching vectors are scattered all over the HNSW graph, which was built over *all* vectors. Restricting traversal to the matching subset severs the graph's connectivity — the greedy walk can no longer use non-matching nodes as bridges — so recall craters, and the system often falls back to a brute-force scan over the (here, small) matching set, which is fine at $0.1\%$ but blows up when the filter is only moderately selective.

    **Post-filter** runs the normal unfiltered ANN for the top-$k$, then discards results failing the predicate. The failure mode: at $0.1\%$ selectivity, almost none of the unfiltered top-$k$ satisfy `tenant_id = 42`, so you may retrieve $k = 10$ and have *zero* survive. You are forced to over-fetch by an unknown, potentially huge factor (here on the order of $1/0.001 = 1000\times$) with no guarantee of enough survivors, wasting work and latency.

    **Filter-aware HNSW** consults the predicate *during* traversal: it still walks the full graph using non-matching nodes as connectivity bridges, but only *collects into the result set* nodes that satisfy the filter. This keeps the graph navigable (unlike pre-filter, which fragments it) while guaranteeing the returned neighbors all pass the predicate (unlike post-filter, which can return an empty set) — recovering recall without unbounded over-fetching. It is the production frontier the chapter flags as a strong design-round point.
