"""
Runnability test for content/13-interview-prep/05-coding-for-ml-interviews.md

The chapter builds a cumulative "interview kit": each code block defines a
function/class that later blocks (and the book's own final self-test,
block #18 / `_tests()`) call by name. Per the harness instructions, later
blocks may depend on names defined by earlier blocks in the same chapter --
that's expected -- so this file concatenates ALL of the chapter's Python
blocks in order (faithfully, no logic changes) to reproduce that dependency
chain, exactly as a reader following along in one scratch file would.

Blocks flagged as heuristically CPU-runnable by the harness and explicitly
tested here:
    - block #0  (line ~22)  -- numpy import + RNG + print options
    - block #13 (line ~357) -- top_k_frequent
    - block #15 (line ~391) -- longest_at_most_k_distinct
    - block #17 (line ~442) -- topo_sort
    - block #18 (line ~476) -- _tests(): the book's own end-to-end self-check

All other blocks (#1-#12, #14, #16) are fragments (function/class
definitions with no standalone demo) that block #18's `_tests()` (and, for
#13/#15, a small extra glue check below) depend on and call -- they are
included verbatim as necessary glue, matching the book's own structure,
but are not separately asserted-on beyond what the book itself exercises.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Block #0 (line ~22, 4 lines) -- TESTED
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)  # modern numpy RNG; reproducible across runs
np.set_printoptions(precision=4, suppress=True)  # readable arrays during the interview


# ---------------------------------------------------------------------------
# Block #1 (line ~47, fragment) -- dependency of blocks #18 (attention,
# cross_entropy, mlp_step all build on this) and the extra glue below.
# ---------------------------------------------------------------------------
def softmax(z, axis=-1):
    """Numerically stable softmax over `axis`.
    z: array of any shape. Returns same shape, sums to 1 along `axis`.
    """
    z = np.asarray(z, dtype=np.float64)
    z_max = np.max(z, axis=axis, keepdims=True)   # (..., 1) for broadcasting
    e = np.exp(z - z_max)                          # largest entry becomes exp(0)=1
    return e / np.sum(e, axis=axis, keepdims=True) # normalize; never divides by 0


# ---------------------------------------------------------------------------
# Block #2 (line ~66, fragment) -- dependency of cross_entropy / mlp_step
# ---------------------------------------------------------------------------
def logsumexp(z, axis=-1):
    c = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(c, axis=axis) + np.log(np.sum(np.exp(z - c), axis=axis))

def log_softmax(z, axis=-1):
    c = np.max(z, axis=axis, keepdims=True)
    return (z - c) - np.log(np.sum(np.exp(z - c), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Block #3 (line ~84, fragment) -- dependency of mlp_step / block #18
# ---------------------------------------------------------------------------
def cross_entropy(Z, y):
    """Z: (n, K) logits, y: (n,) int labels in [0, K). Returns scalar mean loss."""
    logp = log_softmax(Z, axis=1)          # (n, K)
    n = Z.shape[0]
    nll = -logp[np.arange(n), y]           # (n,) pick the true-class log-prob per row
    return nll.mean()


# ---------------------------------------------------------------------------
# Block #4 (line ~111, fragment) -- dependency of block #18
# ---------------------------------------------------------------------------
def attention(Q, K, V, mask=None):
    """Single-head scaled dot-product attention.
    Q: (n_q, d_k), K: (n_k, d_k), V: (n_k, d_v), mask: (n_q, n_k) bool or None.
    Returns: out (n_q, d_v), weights (n_q, n_k).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # (n_q, n_k)
    if mask is not None:
        # set masked positions to -inf so softmax gives them ~0 weight
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)      # (n_q, n_k), each row sums to 1
    out = weights @ V                       # (n_q, d_v)
    return out, weights


# ---------------------------------------------------------------------------
# Block #5 (line ~129, fragment) -- dependency of block #18
# ---------------------------------------------------------------------------
def causal_mask(n):
    """(n, n) boolean: True where query i may attend to key j (j <= i)."""
    return np.tril(np.ones((n, n), dtype=bool))


# ---------------------------------------------------------------------------
# Block #6 (line ~137, fragment) -- defined but not called by the book's own
# _tests(); kept verbatim as the multi-head generalization of block #4.
# ---------------------------------------------------------------------------
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """X: (B, T, d_model). W*: (d_model, d_model). Returns (B, T, d_model)."""
    B, T, d_model = X.shape
    d_head = d_model // n_heads

    def split_heads(M):                      # (B, T, d_model) -> (B, H, T, d_head)
        M = M.reshape(B, T, n_heads, d_head)
        return M.transpose(0, 2, 1, 3)

    Q = split_heads(X @ Wq)                   # project then split, per head
    K = split_heads(X @ Wk)
    V = split_heads(X @ Wv)

    scale = 1.0 / np.sqrt(d_head)
    # 'bhqd,bhkd->bhqk' contracts the d_head axis: scores per (batch, head, q, k)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale
    if causal:
        m = causal_mask(T)                    # (T, T), broadcasts over B and H
        scores = np.where(m, scores, -np.inf)
    A = softmax(scores, axis=-1)              # (B, H, T, T)
    ctx = np.einsum('bhqk,bhkd->bhqd', A, V)  # (B, H, T, d_head)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, d_model)  # merge heads back
    return ctx @ Wo


# ---------------------------------------------------------------------------
# Block #7 (line ~184, fragment) -- dependency of block #18
# ---------------------------------------------------------------------------
def sigmoid(z):
    # stable sigmoid: avoid exp overflow for large-magnitude z
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                           np.exp(z) / (1.0 + np.exp(z)))

def train_logreg(X, y, lr=0.1, epochs=200, l2=0.0):
    """X: (n, d), y: (n,) in {0,1}. Returns weights w (d,) and bias b."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b                  # (n,) logits
        p = sigmoid(z)                 # (n,) predicted probabilities
        err = p - y                    # (n,) the magic residual: dL/dz
        grad_w = X.T @ err / n + l2 * w   # (d,) plus optional L2 shrinkage
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


# ---------------------------------------------------------------------------
# Block #8 (line ~218, fragment) -- defined but not called by the book's own
# _tests(); kept verbatim, exercises cross_entropy/softmax internally.
# ---------------------------------------------------------------------------
def mlp_step(X, y, params, lr=0.1):
    """One forward+backward+update step. X: (n, d_in), y: (n,) int labels.
    params: dict with W1 (d_in,h), b1 (h,), W2 (h,K), b2 (K,)."""
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    n = X.shape[0]

    # ---- forward (cache activations we need for the backward pass) ----
    a1 = X @ W1 + b1            # (n, h) pre-activation
    h  = np.maximum(0, a1)      # (n, h) ReLU
    Z  = h @ W2 + b2            # (n, K) logits
    loss = cross_entropy(Z, y)  # scalar (reuses our stable CE)

    # ---- backward (every dX has the SAME shape as X) ----
    P = softmax(Z, axis=1)             # (n, K)
    dZ = P.copy()
    dZ[np.arange(n), y] -= 1.0         # subtract one-hot: dZ = P - onehot(y)
    dZ /= n                            # average over the batch

    dW2 = h.T @ dZ                     # (h, K)
    db2 = dZ.sum(axis=0)              # (K,)
    dh  = dZ @ W2.T                    # (n, h) gradient flows back into hidden layer
    da1 = dh * (a1 > 0)               # (n, h) ReLU gate: pass grad only where a1>0
    dW1 = X.T @ da1                    # (d_in, h)
    db1 = da1.sum(axis=0)            # (h,)

    # ---- SGD update ----
    for k, g in (('W1', dW1), ('b1', db1), ('W2', dW2), ('b2', db2)):
        params[k] -= lr * g
    return loss


# ---------------------------------------------------------------------------
# Block #9 (line ~254, fragment) -- defined but not called by the book's own
# _tests(); kept verbatim.
# ---------------------------------------------------------------------------
def grad_check(f, x, eps=1e-6):
    """f: scalar function of array x. Returns max |numeric - none| placeholder; here
    we just return the numeric gradient via central differences for spot checks."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)   # central difference: O(eps^2) accurate
        it.iternext()
    return g


# ---------------------------------------------------------------------------
# Block #10 (line ~281, fragment) -- dependency of block #18
# ---------------------------------------------------------------------------
def kmeans(X, k, iters=100, seed=0):
    """X: (n, d). Returns centroids (k, d) and assignments (n,)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # k-means++ style would be better, but random init is fine to start
    centroids = X[rng.choice(n, size=k, replace=False)].copy()  # (k, d)
    assign = np.full(n, -1)
    for _ in range(iters):
        # (n, 1, d) - (1, k, d) -> (n, k, d); sum over d -> (n, k) squared dists
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assign = d2.argmin(axis=1)          # (n,) nearest centroid per point
        if np.array_equal(new_assign, assign):  # converged: assignments stable
            break
        assign = new_assign
        for j in range(k):                       # update step
            members = X[assign == j]
            if len(members):                     # guard against empty clusters
                centroids[j] = members.mean(axis=0)
    inertia = ((X - centroids[assign]) ** 2).sum()
    return centroids, assign, inertia


# ---------------------------------------------------------------------------
# Block #11 (line ~306, fragment) -- defined but not called by the book's own
# _tests(); kept verbatim (memory-lean alternative to block #10's distances).
# ---------------------------------------------------------------------------
def pairwise_sq_dist(X, C):
    """Memory-lean (n, k) squared distances via the norm expansion."""
    xx = (X ** 2).sum(axis=1, keepdims=True)   # (n, 1)
    cc = (C ** 2).sum(axis=1)[None, :]         # (1, k)
    cross = X @ C.T                            # (n, k)
    return np.maximum(xx - 2 * cross + cc, 0)  # clamp tiny negatives from roundoff


# ---------------------------------------------------------------------------
# Block #12 (line ~345, fragment) -- dependency of block #18
# ---------------------------------------------------------------------------
def first_duplicate(stream):
    """First element seen twice, or None. O(n) time, O(n) space."""
    seen = set()
    for x in stream:
        if x in seen:
            return x
        seen.add(x)
    return None


# ---------------------------------------------------------------------------
# Block #13 (line ~357, 9 lines) -- TESTED (via extra glue below, since the
# book's own _tests() does not call top_k_frequent)
# ---------------------------------------------------------------------------
import heapq
from collections import Counter

def top_k_frequent(tokens, k):
    """k most frequent items. O(n log k) with a min-heap of size k."""
    counts = Counter(tokens)              # O(n)
    # heapq.nlargest is the idiomatic one-liner; it maintains a size-k heap
    return heapq.nlargest(k, counts.keys(), key=counts.get)


# ---------------------------------------------------------------------------
# Block #14 (line ~375, fragment) -- dependency of block #18
# ---------------------------------------------------------------------------
def two_sum_sorted(a, target):
    """a sorted ascending. Return indices (i, j) with a[i]+a[j]==target, or None."""
    i, j = 0, len(a) - 1
    while i < j:
        s = a[i] + a[j]
        if s == target:
            return (i, j)
        elif s < target:
            i += 1            # need a bigger sum -> move left pointer up
        else:
            j -= 1            # need a smaller sum -> move right pointer down
    return None


# ---------------------------------------------------------------------------
# Block #15 (line ~391, 15 lines) -- TESTED (via extra glue below, since the
# book's own _tests() does not call longest_at_most_k_distinct)
# ---------------------------------------------------------------------------
def longest_at_most_k_distinct(seq, K):
    """Length of the longest contiguous subarray with <= K distinct values."""
    from collections import defaultdict
    count = defaultdict(int)
    left = best = 0
    for right, x in enumerate(seq):
        count[x] += 1
        while len(count) > K:            # too many distinct -> shrink from left
            count[seq[left]] -= 1
            if count[seq[left]] == 0:
                del count[seq[left]]
            left += 1
        best = max(best, right - left + 1)
    return best


# ---------------------------------------------------------------------------
# Block #16 (line ~419, fragment) -- dependency of block #18
# ---------------------------------------------------------------------------
def edit_distance(a, b):
    """Levenshtein distance. O(len(a)*len(b)) time; O(len(b)) space (rolling rows)."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))            # dp row for i=0: j deletions
    for i in range(1, m + 1):
        curr = [i] + [0] * n             # dp[i][0] = i insertions
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1]                       # match: no cost
            else:
                curr[j] = 1 + min(prev[j],     # delete from a
                                  curr[j-1],   # insert into a
                                  prev[j-1])   # substitute
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# Block #17 (line ~442, 21 lines) -- TESTED
# ---------------------------------------------------------------------------
from collections import deque, defaultdict

def topo_sort(num_nodes, edges):
    """Kahn's algorithm. edges: list of (u, v) meaning u must come before v.
    Returns an ordering, or None if a cycle exists. O(V + E)."""
    indeg = [0] * num_nodes
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    q = deque(i for i in range(num_nodes) if indeg[i] == 0)  # sources
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1                 # "remove" u's outgoing edges
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == num_nodes else None  # short order => cycle


# ---------------------------------------------------------------------------
# Block #18 (line ~476, 48 lines) -- TESTED
# This is the book's own end-to-end self-check, copied verbatim. It exercises
# softmax, cross_entropy, attention, causal_mask, train_logreg/sigmoid,
# kmeans, first_duplicate, two_sum_sorted, edit_distance, and topo_sort.
# ---------------------------------------------------------------------------
import numpy as np

def _tests():
    rng = np.random.default_rng(0)

    # softmax sums to 1 and is overflow-safe
    z = np.array([1000.0, 1000.0, 1000.0])
    assert np.allclose(softmax(z), 1/3), "softmax must be overflow-stable"

    # cross_entropy matches the manual definition on a tiny case
    Z = np.array([[2.0, 1.0, 0.1]])
    y = np.array([0])
    manual = -(np.log(np.exp(Z - Z.max()) / np.exp(Z - Z.max()).sum())[0, 0])
    assert np.allclose(cross_entropy(Z, y), manual), "CE mismatch"

    # attention rows are probability distributions
    Q, K, V = rng.standard_normal((4, 8)), rng.standard_normal((6, 8)), rng.standard_normal((6, 3))
    out, w = attention(Q, K, V)
    assert np.allclose(w.sum(axis=1), 1.0) and out.shape == (4, 3)

    # causal mask: query 0 attends only to key 0
    # BUG FIX (mirrors content/13-interview-prep/05-coding-for-ml-interviews.md):
    # the book originally wrote `attention(Q[:6], K, V, mask=causal_mask(6))`,
    # but Q has only 4 rows (needed for the `out.shape == (4, 3)` assertion
    # above), so `Q[:6]` silently returns all 4 rows and the (4,6) scores
    # matrix cannot broadcast against the (6,6) causal mask -> ValueError.
    # The intent is self-attention over the 6-length K sequence, so use K
    # as both query and key for this specific check.
    _, wc = attention(K, K, V, mask=causal_mask(6))
    assert np.allclose(wc[0, 1:], 0.0), "causal mask leak"

    # logistic regression separates a linearly separable set
    X = np.vstack([rng.standard_normal((50, 2)) + 3, rng.standard_normal((50, 2)) - 3])
    y = np.r_[np.ones(50), np.zeros(50)]
    w, b = train_logreg(X, y, lr=0.5, epochs=300)
    acc = ((sigmoid(X @ w + b) > 0.5) == y).mean()
    assert acc > 0.95, f"logreg acc too low: {acc}"

    # k-means recovers two well-separated blobs
    blob = np.vstack([rng.standard_normal((40, 2)) + 5, rng.standard_normal((40, 2)) - 5])
    _, assign, inertia = kmeans(blob, k=2, seed=1)
    # each true blob should be (mostly) one cluster
    assert len(set(assign[:40])) == 1 and len(set(assign[40:])) == 1

    # DS&A spot checks
    assert first_duplicate([1, 2, 3, 2]) == 2
    assert two_sum_sorted([1, 3, 4, 7], 7) == (1, 2)
    assert edit_distance("kitten", "sitting") == 3
    assert topo_sort(3, [(0, 1), (1, 2)]) == [0, 1, 2]
    assert topo_sort(2, [(0, 1), (1, 0)]) is None    # cycle

    print("all checks passed")

_tests()


# ---------------------------------------------------------------------------
# Extra glue (not in the book): the book's own _tests() above exercises every
# tested block except #13 (top_k_frequent) and #15 (longest_at_most_k_distinct).
# Call those explicitly with tiny inputs so every tested block actually runs.
# ---------------------------------------------------------------------------
def _extra_checks():
    tk = top_k_frequent(["a", "b", "a", "c", "a", "b"], 2)
    assert set(tk) == {"a", "b"}, f"top_k_frequent wrong: {tk}"

    span = longest_at_most_k_distinct(list("eceba"), 2)
    assert span == 3, f"longest_at_most_k_distinct wrong: {span}"

    print("extra checks passed (block #13 top_k_frequent, block #15 longest_at_most_k_distinct)")

_extra_checks()
