# 13.5 Coding for ML Interviews

The coding round is where many strong ML candidates stumble — not because the problems are hard, but because they prepared for the *wrong* coding. A Google ML domain interview almost always includes a live-coding segment, and it splits into two flavors. The first is **"implement this ML primitive from scratch in numpy"**: softmax, attention, k-means, logistic regression, backprop through a tiny MLP. The second is **classic data-structures-and-algorithms (DS&A)** — arrays, hashing, two pointers, dynamic programming, graphs — sometimes with an ML flavor stapled on ("dedupe these embeddings," "batch these requests"). You need both in your fingers.

This chapter is a drill book. We build the ML primitives the way an interviewer wants to see them built — vectorized, numerically stable, shape-annotated — and we work the DS&A patterns that recur in ML interviews. Every block of code here is runnable. The goal is not to memorize solutions but to internalize the *shapes of the answers* so that under pressure you write the stable softmax, not the one that overflows.

We assume you know Python and basic linear algebra. For the underlying theory, lean on the foundations chapters: [Machine Learning Fundamentals](../01-foundations/05-ml-fundamentals.html), [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html), and [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html). Here we focus on *typing it correctly in 25 minutes while talking*.

## How the ML coding round actually works

Before the code, the meta-game. The interviewer is evaluating four things, and only one of them is "did it pass the test."

1. **Correctness under shapes.** Can you keep `(batch, seq, dim)` straight? Vectorized code lives or dies by broadcasting. State your shapes out loud and annotate them in comments — `# x: (n, d)`. This single habit prevents most bugs and signals seniority.
2. **Numerical stability.** Naive softmax overflows; naive log-likelihood underflows; naive variance loses precision. Interviewers *plant* these traps. Reach for the max-subtraction trick and `logsumexp` reflexively.
3. **Communication.** Narrate the plan before typing. "I'll subtract the row max for stability, exponentiate, then normalize along the last axis." Silence reads as guessing.
4. **Complexity awareness.** State time and space. Know that attention is $O(n^2 d)$, k-means is $O(n k d)$ per iteration, and a hash set lookup is $O(1)$ amortized.

A practical rule: **never use a Python loop where a numpy/torch broadcast will do**, *except* to make a first version correct. It is perfectly fine to say "let me write the loop version to nail correctness, then vectorize" — that is exactly how a thoughtful engineer works. Just make sure you actually finish the vectorization.

Set up your scratch environment with explicit imports and a fixed seed so your worked examples are reproducible:

```python
import numpy as np
rng = np.random.default_rng(0)  # modern numpy RNG; reproducible across runs
np.set_printoptions(precision=4, suppress=True)  # readable arrays during the interview
```

We use `numpy` for the from-scratch primitives (no autograd to hide behind) and bring in `torch` only where the question is explicitly "do it in PyTorch." Knowing both fluently — and knowing when each is expected — is itself a signal.

## Softmax, log-sum-exp, and cross-entropy

Softmax is the single most-asked from-scratch primitive because it tests numerical stability in three lines. The definition for a vector $z \in \mathbb{R}^d$ is

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{d} e^{z_j}}.
$$

The trap: if any $z_i$ is large (say 1000), $e^{z_i}$ overflows to `inf` and you get `nan`. The fix exploits **shift invariance** — subtracting a constant $c$ from every logit leaves the output unchanged, because the $e^{-c}$ factors cancel:

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} e^{z_i}}{e^{-c}\sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Choosing $c = \max_i z_i$ guarantees the largest exponent is $e^0 = 1$, so nothing overflows. Underflow in the small terms is harmless (they round to 0, which is the right answer).

```python
def softmax(z, axis=-1):
    """Numerically stable softmax over `axis`.
    z: array of any shape. Returns same shape, sums to 1 along `axis`.
    """
    z = np.asarray(z, dtype=np.float64)
    z_max = np.max(z, axis=axis, keepdims=True)   # (..., 1) for broadcasting
    e = np.exp(z - z_max)                          # largest entry becomes exp(0)=1
    return e / np.sum(e, axis=axis, keepdims=True) # normalize; never divides by 0
```

The `keepdims=True` is the part candidates forget. Without it, `z_max` has the reduced shape and broadcasting against `z` either errors or — worse — silently broadcasts along the wrong axis. Say "keepdims so the subtraction broadcasts row-wise" as you type it.

Closely related is **log-softmax**, which you want whenever the result feeds a log-likelihood. Computing `np.log(softmax(z))` works but wastes precision (you exponentiate then take a log). Compute it directly via the log-sum-exp identity:

$$
\log \operatorname{softmax}(z)_i = z_i - \operatorname{logsumexp}(z), \qquad \operatorname{logsumexp}(z) = c + \log \sum_j e^{z_j - c},\; c=\max_j z_j.
$$

```python
def logsumexp(z, axis=-1):
    c = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(c, axis=axis) + np.log(np.sum(np.exp(z - c), axis=axis))

def log_softmax(z, axis=-1):
    c = np.max(z, axis=axis, keepdims=True)
    return (z - c) - np.log(np.sum(np.exp(z - c), axis=axis, keepdims=True))
```

Now cross-entropy. For a batch of logits `Z` of shape `(n, K)` and integer labels `y` of shape `(n,)`, the mean cross-entropy loss is

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n} \log \operatorname{softmax}(Z_i)_{y_i}.
$$

Build it on top of `log_softmax` so it inherits stability, and use **fancy indexing** to pull out the correct-class log-probabilities without a loop:

```python
def cross_entropy(Z, y):
    """Z: (n, K) logits, y: (n,) int labels in [0, K). Returns scalar mean loss."""
    logp = log_softmax(Z, axis=1)          # (n, K)
    n = Z.shape[0]
    nll = -logp[np.arange(n), y]           # (n,) pick the true-class log-prob per row
    return nll.mean()
```

The expression `logp[np.arange(n), y]` is the idiom to internalize: paired index arrays select element `(i, y[i])` for each `i`. Interviewers love it because it shows you think in vectors.

!!! warning "Common pitfall"
    Do **not** compute `cross_entropy(softmax(Z), y)` by then taking `np.log`. Combining softmax and log into one `log_softmax` is more stable and is also exactly what `torch.nn.functional.cross_entropy` does internally (it takes raw logits, not probabilities). If an interviewer hands you probabilities that already sum to 1, clamp before the log: `np.log(np.clip(p, 1e-12, 1.0))`.

!!! example "Worked example: softmax magnitudes"
    Take logits `z = [2.0, 1.0, 0.1]`. Subtract the max (2.0) to get `[0.0, -1.0, -1.9]`. Exponentiate: `[1.0, 0.3679, 0.1496]`, summing to `1.5175`. Divide: `[0.659, 0.242, 0.099]` — a confident-but-not-certain distribution. Now try `z = [1000, 1000, 1000]` with the *naive* formula: `exp(1000)` is `inf`, so you get `[nan, nan, nan]`. The stable version subtracts 1000 first, exponentiates `[0,0,0]` to `[1,1,1]`, and returns the correct uniform `[0.333, 0.333, 0.333]`. This is the entire reason the trick exists, and it is a 30-second thing to demonstrate at the whiteboard.

## Scaled dot-product attention from scratch

Attention is the flagship ML-coding question for LLM roles. The interviewer wants the scaled dot-product form, optionally multi-head, optionally causally masked. The core equation:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

$Q$ is `(n_q, d_k)`, $K$ is `(n_k, d_k)`, $V$ is `(n_k, d_v)`. The scores $QK^\top$ are `(n_q, n_k)`; softmax runs over the **last axis** (the keys), so each query forms a distribution over keys; the weighted sum with $V$ gives `(n_q, d_v)`. The $\sqrt{d_k}$ divisor keeps the dot products from growing with dimension and saturating the softmax — derived and motivated in [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

```python
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
```

For a **causal** (autoregressive) mask, position $i$ may only attend to positions $j \le i$. Build the lower-triangular allow-mask with `np.tril`:

```python
def causal_mask(n):
    """(n, n) boolean: True where query i may attend to key j (j <= i)."""
    return np.tril(np.ones((n, n), dtype=bool))
```

The batched, multi-head version is the senior-level ask. The trick is reshaping `(batch, seq, d_model)` into `(batch, heads, seq, d_head)` so a single `einsum` handles every head and every batch element at once. Keep `d_model = n_heads * d_head`.

```python
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
```

Three habits make this answer land. First, the `einsum` strings *are* the math — say "contract over `d`" as you write `bhqd,bhkd->bhqk`. Second, `transpose` then `reshape` to merge heads is the inverse of `reshape` then `transpose` to split them; getting the order wrong silently scrambles heads. Third, name the complexity unprompted: the score matrix is $O(B \cdot H \cdot T^2 \cdot d_{\text{head}})$ in time and $O(B \cdot H \cdot T^2)$ in memory — the quadratic-in-sequence-length term that motivates [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html).

!!! tip "Practitioner tip"
    If asked the same problem in PyTorch, the entire body collapses to `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`, which dispatches to a fused FlashAttention kernel. State that you *know* the one-liner, then offer to write it from scratch — interviewers want to see you can do both, and leading with the library call alone can read as not understanding the internals.

## Logistic regression and backprop by hand

These two are the classic "do you understand gradients" probes. Both come down to writing a forward pass, a loss, the analytic gradient, and a training loop — without an autograd framework.

### Logistic regression with gradient descent

Binary logistic regression models $P(y=1 \mid x) = \sigma(w^\top x + b)$ with the sigmoid $\sigma(z) = 1/(1+e^{-z})$, trained by minimizing binary cross-entropy. The gradient has a famously clean form: for the cross-entropy loss, $\partial \mathcal{L} / \partial z = \sigma(z) - y$. That cancellation — the messy sigmoid derivative meets the log in the loss and simplifies — is the thing to be able to recite.

$$
\mathcal{L}(w,b) = -\frac{1}{n}\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big], \quad p_i = \sigma(w^\top x_i + b).
$$

$$
\nabla_w \mathcal{L} = \frac{1}{n} X^\top (p - y), \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n}\sum_i (p_i - y_i).
$$

```python
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
```

The stable `sigmoid` (branching on the sign of `z`) is a detail worth showing: for very negative `z`, `exp(-z)` overflows, so we switch to the algebraically equal `exp(z)/(1+exp(z))`. Mention it; you do not have to belabor it.

### Backprop through a two-layer MLP

The from-scratch backprop question is the deepest of the ML-coding asks. Build a one-hidden-layer network with a ReLU and a softmax-cross-entropy head, then derive the backward pass by hand. The forward pass:

$$
h = \operatorname{ReLU}(X W_1 + b_1), \qquad Z = h W_2 + b_2, \qquad \mathcal{L} = \operatorname{CE}(\operatorname{softmax}(Z), y).
$$

The single fact that makes backprop tractable here is the same cancellation as logistic regression: the gradient of softmax-cross-entropy with respect to the logits is $\partial \mathcal{L}/\partial Z = (\hat{p} - \text{onehot}(y))/n$, where $\hat p = \operatorname{softmax}(Z)$. From there it is the chain rule, layer by layer, reusing the cached forward activations.

```python
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
```

Two checkpoints to verbalize. **Shape symmetry:** every gradient `dW` matches the shape of its `W`, and every `dX` matches `X` — if a `@` does not produce the right shape, the chain rule order is wrong. **The ReLU gate:** `da1 = dh * (a1 > 0)` zeroes the gradient wherever the unit was inactive in the forward pass; this is the discrete switch that makes ReLU networks trainable. The full theory, including why these shapes work out, is in [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html) and [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html).

A senior touch: offer a **gradient check**. Numerically perturb one parameter by $\epsilon$ and compare the finite-difference slope to your analytic gradient. It costs three lines and proves your backward pass is correct.

```python
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
```

## K-means and vectorized numpy fluency

K-means is the canonical unsupervised-learning coding question and a pure test of numpy broadcasting. Given $n$ points in $\mathbb{R}^d$ and $k$ clusters, alternate two steps until assignments stop changing: **assign** each point to its nearest centroid, then **update** each centroid to the mean of its assigned points. The objective being minimized is within-cluster sum of squares (inertia):

$$
J = \sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert_2^2, \qquad c_i = \arg\min_j \lVert x_i - \mu_j \rVert_2^2.
$$

The interview-grade move is computing the full `(n, k)` distance matrix **without any loop**, using broadcasting. The clean way avoids the squared-norm expansion and just broadcasts a `(n,1,d)` against a `(1,k,d)`:

```python
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
```

The `X[:, None, :] - centroids[None, :, :]` line is the whole interview. Walk through the shapes aloud: inserting a length-1 axis lets numpy broadcast all `n × k` pairwise differences in one C-level operation. The memory cost is $O(n k d)$ for that intermediate tensor — fine for an interview, but flag it: for large `n` you would use the expansion $\lVert x - \mu \rVert^2 = \lVert x \rVert^2 - 2 x^\top \mu + \lVert \mu \rVert^2$ to avoid materializing the 3-D array:

```python
def pairwise_sq_dist(X, C):
    """Memory-lean (n, k) squared distances via the norm expansion."""
    xx = (X ** 2).sum(axis=1, keepdims=True)   # (n, 1)
    cc = (C ** 2).sum(axis=1)[None, :]         # (1, k)
    cross = X @ C.T                            # (n, k)
    return np.maximum(xx - 2 * cross + cc, 0)  # clamp tiny negatives from roundoff
```

The `np.maximum(..., 0)` clamp is a subtle stability point: floating-point roundoff in `xx - 2*cross + cc` can produce small negatives that become `nan` under a later `sqrt`. Catching that detail unprompted reads as someone who has actually shipped numerical code — see [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html).

!!! example "Worked example: one k-means iteration"
    Points `[[0,0],[0,1],[10,10],[11,9]]`, init centroids at the first and third points: `μ0=[0,0]`, `μ1=[10,10]`. Squared distances: point `[0,1]` is `1` from `μ0` and `181` from `μ1` → cluster 0; point `[11,9]` is `202` from `μ0` and `2` from `μ1` → cluster 1. So assignments are `[0,0,1,1]`. Update: `μ0 = mean([0,0],[0,1]) = [0,0.5]`, `μ1 = mean([10,10],[11,9]) = [10.5,9.5]`. The next iteration produces the same assignments → converged. Inertia is `0.5 + 0.5 = 1.0`. Doing this by hand on two clusters convinces the interviewer you understand the loop, not just the API.

### A numpy fluency checklist

Vectorized thinking is itself graded. These idioms come up constantly; have them automatic:

| Task | Idiom | Note |
|---|---|---|
| Row-wise softmax | `e/e.sum(1, keepdims=True)` | always `keepdims` |
| One-hot from labels | `np.eye(K)[y]` | fancy indexing |
| Pairwise distances | `((A[:,None]-B[None])**2).sum(-1)` | $O(nm d)$ memory |
| Select per-row element | `M[np.arange(n), idx]` | paired index arrays |
| Masked fill | `np.where(mask, x, -np.inf)` | for attention masks |
| Batched matmul | `np.einsum('bij,bjk->bik', A, B)` | or `A @ B` (batched) |
| Normalize rows to unit L2 | `X / np.linalg.norm(X,axis=1,keepdims=True)` | for cosine sim |
| Cumulative without loop | `np.cumsum`, `np.cumprod` | DP-flavored |

The single highest-leverage habit: whenever you reach for a `for` loop over rows, pause and ask whether broadcasting or `einsum` does it in one line. In an ML interview, the vectorized answer is the expected answer.

## DS&A staples, framed for ML candidates

Half of ML coding rounds are ordinary algorithmic problems — but the interviewer often dresses them in ML clothing. Recognizing the underlying pattern is the skill. Here are the patterns that recur, each with a runnable solution and the ML framing you will actually hear.

### Hashing and arrays: dedup, counting, top-k

**Pattern — hash set for membership / dedup.** "You have a stream of document fingerprints; return the first duplicate." This is the near-duplicate detection step from [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html), reduced to its algorithmic core. A `set` gives $O(1)$ lookups and one pass:

```python
def first_duplicate(stream):
    """First element seen twice, or None. O(n) time, O(n) space."""
    seen = set()
    for x in stream:
        if x in seen:
            return x
        seen.add(x)
    return None
```

**Pattern — counting with a hash map → top-k with a heap.** "Return the $k$ most frequent tokens in a corpus" is the vocabulary-building step of [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html). Count in a `Counter`, then use a size-$k$ heap to get top-$k$ in $O(n \log k)$ instead of sorting everything in $O(n \log n)$:

```python
import heapq
from collections import Counter

def top_k_frequent(tokens, k):
    """k most frequent items. O(n log k) with a min-heap of size k."""
    counts = Counter(tokens)              # O(n)
    # heapq.nlargest is the idiomatic one-liner; it maintains a size-k heap
    return heapq.nlargest(k, counts.keys(), key=counts.get)
```

State the trade-off out loud: `Counter.most_common(k)` is fine and uses a heap under the hood, but if asked to implement it, the size-$k$ min-heap is the answer that shows you know why it beats a full sort.

### Two pointers and sliding windows

**Pattern — sorted-array two pointers.** "Given sorted similarity scores, find a pair summing to a target threshold." The two-pointer sweep is $O(n)$ and beats the $O(n^2)$ brute force or the $O(n)$-space hash approach:

```python
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
```

**Pattern — sliding window.** "Find the longest span of a token stream with at most $K$ distinct token types" — directly the kind of constraint that shows up in context-window and chunking logic ([Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)). Grow the window on the right, shrink from the left when the constraint breaks; each element enters and leaves at most once, so it is $O(n)$:

```python
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
```

### Dynamic programming

DP is the pattern candidates fear most, yet ML interviews favor a small, recognizable set. The skeleton is always: define the state, write the recurrence, identify base cases, then either memoize (top-down) or tabulate (bottom-up).

**Edit distance** (Levenshtein) is the most ML-relevant — it underlies token-level evaluation metrics and fuzzy matching. The state $dp[i][j]$ is the minimum edits to turn the first $i$ characters of `a` into the first $j$ of `b`:

$$
dp[i][j] = \begin{cases} \max(i,j) & \text{if } \min(i,j)=0 \\ dp[i-1][j-1] & \text{if } a_i = b_j \\ 1 + \min(dp[i-1][j],\, dp[i][j-1],\, dp[i-1][j-1]) & \text{otherwise.}\end{cases}
$$

```python
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
```

The rolling-array optimization (keeping only two rows, $O(n)$ space instead of $O(mn)$) is the senior flourish — mention it even if you first write the full table. **Longest common subsequence**, **coin change**, and **0/1 knapsack** share this exact skeleton; if you can derive edit distance live, you can derive those.

### Graphs: BFS, DFS, and topological sort

Graph problems appear as **dependency resolution** — and the model-parallel and pipeline-scheduling chapters ([Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html)) are literally about scheduling a computation DAG. The bread-and-butter is BFS/DFS for connectivity and topological sort for ordering.

```python
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
```

The detection of cycles via "did we emit every node?" is the part to articulate: a DAG has a valid topological order; if Kahn's algorithm terminates with fewer than `num_nodes` emitted, a cycle blocked the rest. For shortest-path-on-unweighted-graph, plain BFS suffices; for weighted, you would reach for Dijkstra with a heap — know which tool fits which graph.

!!! interview "Interview Corner"
    **Q:** Implement scaled dot-product attention with a causal mask in numpy, and tell me its time and memory complexity. Then: how would you cut the memory?

    **A:** *(Writes the `attention` function above with `mask = np.tril(...)` and `np.where(mask, scores, -inf)`.)* Time is $O(n^2 d)$ — the $QK^\top$ matmul and the $AV$ matmul both scale with the $n \times n$ score matrix times the $d$ feature dimension. Memory is $O(n^2)$ for the explicit score/weight matrices, which dominates for long sequences. To cut the memory you avoid materializing the full $n \times n$ matrix: process keys/values in blocks and maintain a running softmax via the **online-softmax** trick — track the running max, the running denominator, and a rescaled running output, updating them block by block. That is exactly FlashAttention; it brings the activation memory down to $O(n)$ while keeping the math identical, and it is faster in practice because it is IO-aware about the GPU memory hierarchy. The key numerical subtlety is the same max-subtraction we use in plain softmax, applied incrementally so that merging two blocks rescales the older partial sums by $e^{m_{\text{old}} - m_{\text{new}}}$.

## Putting it together: a tested interview kit

The final skill is *self-verification*. Strong candidates finish by running their code on a small case and stating the result — never "I think this works." Here is a single script that exercises every primitive above against a known-good reference (PyTorch / sklearn-style checks), the way you would sanity-check at the end of a round:

```python
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

    # causal mask: query 0 attends only to key 0 (self-attention over the
    # 6-length K sequence, so query count matches key count for the mask)
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
```

Running this prints `all checks passed`. The `edit_distance("kitten", "sitting") == 3` line is the textbook example (substitute k→s, e→i, append g). Ending a round by *running your own tests* — especially the adversarial ones like the overflow softmax and the cycle in topo-sort — is the single most reliable way to signal engineering maturity.

A few closing tactics for the live round, distilled from the patterns above:

- **Restate the problem and the shapes** before coding. "So `X` is `(n, d)`, I return `(n,)` labels."
- **Write the brute force, then optimize**, narrating the complexity at each step. A correct $O(n^2)$ beats a broken $O(n)$.
- **Reach for stability reflexively**: max-subtraction in softmax, sign-branched sigmoid, clamp-before-log, clamp-negative-before-sqrt.
- **Vectorize**: a `for` loop over rows is a smell; broadcasting or `einsum` is the expected answer.
- **Test at the end** on a tiny case with a known answer, including an adversarial input.

For the surrounding interview strategy — pacing, how the loop is structured, and what each interviewer is scoring — see [The Google ML Domain Interview: Format & Strategy](../13-interview-prep/01-google-ml-interview-format.html) and the rapid-fire concept drills in [ML Breadth: Rapid-Fire Concepts & Model Answers](../13-interview-prep/02-ml-breadth-rapidfire.html). For the deeper conceptual questions that bracket the coding, see [LLM-Specific Deep-Dive Questions](../13-interview-prep/06-llm-deepdive-questions.html).

!!! key "Key Takeaways"
    - The ML coding round tests two tracks: **implement-the-primitive in numpy** (softmax, attention, k-means, logistic regression, backprop) and **classic DS&A** (hashing, two pointers, sliding window, DP, graphs) — prepare both.
    - **Numerical stability is graded.** Subtract the max in softmax, use `log_softmax`/`logsumexp` for losses, branch the sigmoid on the sign of `z`, and clamp before `log` and `sqrt`. Interviewers plant overflow/underflow traps.
    - **Shapes are your debugger.** Annotate every array's shape, keep `keepdims=True` on reductions you broadcast against, and remember every `dW` matches its `W` and every `dX` matches `X`.
    - The **softmax-cross-entropy gradient simplifies to $\hat p - \text{onehot}(y)$** — this single cancellation drives both logistic regression and MLP backprop; be able to recite and use it.
    - **Vectorize instead of looping.** Master broadcasting, fancy indexing (`M[np.arange(n), idx]`), and `einsum`; a Python loop over rows signals you have not internalized numpy.
    - For DS&A, **recognize the pattern under the ML costume**: dedup → hash set, top-k tokens → heap, longest-valid-span → sliding window, fuzzy match → edit-distance DP, dependency order → topological sort.
    - **Always state time and space complexity**, and know the headline numbers: attention $O(n^2 d)$, k-means $O(nkd)$ per iteration, heap top-k $O(n \log k)$, hash lookup $O(1)$ amortized.
    - **Finish by running your own tests** on tiny known-answer cases, including adversarial inputs — self-verification is the clearest seniority signal you can give.

## Further reading

- Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* — from-scratch implementations of logistic regression, k-means, and gradient descent with the numpy idioms used here.
- Vaswani et al., *Attention Is All You Need* (2017) — the source of the scaled dot-product attention you implement in this chapter.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — the online-softmax method behind the memory-reduction answer in the Interview Corner.
- Cormen, Leiserson, Rivest & Stein, *Introduction to Algorithms* (CLRS) — the canonical reference for the DP, graph, and complexity material.
- Goodfellow, Bengio & Courville, *Deep Learning* — backpropagation and the chain-rule mechanics underlying the MLP backward pass.
- The official NumPy documentation on broadcasting and `numpy.einsum` — the two features that turn loop-heavy code into one-liners under interview pressure.
