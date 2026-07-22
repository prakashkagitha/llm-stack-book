# 4.2 FlashAttention I: IO-Awareness & The Online Softmax

In [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) we wrote the canonical formula and the canonical loop:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V .
$$

That implementation is *correct* and it is *slow* — not because the arithmetic is expensive, but because of where the numbers live. The standard implementation builds the full $N \times N$ score matrix $S = QK^\top/\sqrt{d}$ in GPU main memory (HBM), reads it back to apply softmax, reads it again to multiply by $V$. For a sequence of $N = 8192$ tokens, that intermediate matrix is 64 million entries *per head per layer*, and every byte of it has to travel across the slowest wire in the machine. The matrix multiplies that everyone worries about are not the bottleneck. The traffic to and from HBM is.

FlashAttention (Dao, Fu, Ermon, Rudra, Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022) is the algorithm that removed that bottleneck. Its central trick is to **never materialize the $N \times N$ matrix at all** — to compute attention in tiles small enough to live in on-chip SRAM, fusing the three steps into a single pass. To make that possible it needs one beautiful piece of numerical machinery: the **online softmax**, a way to compute a numerically-stable softmax incrementally, while streaming the inputs, without ever seeing all of them at once. That algorithm — running maxima, running denominators, and a correction factor that retroactively rescales partial results — is the heart of this chapter, and one of the most elegant ideas in all of systems-for-ML.

This chapter is a marquee chapter. We will derive the online softmax from scratch, prove it gives *exactly* the same answer as the naive version (FlashAttention is exact, not an approximation), build the full forward and backward passes in NumPy so you can run them, and then do the IO-complexity accounting that explains *why* it is faster. It builds directly on [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) and [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html); [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html) then takes the same core and squeezes out the remaining inefficiencies.

## The memory bottleneck: why naive attention is IO-bound

Let us be precise about what the standard kernel does. Take a single attention head with sequence length $N$ and head dimension $d$ (think $d = 64$ or $128$). The inputs $Q, K, V$ are each $N \times d$. The reference computation is three operations:

```python
import numpy as np

def attention_naive(Q, K, V):
    # Q, K, V: (N, d) arrays already on the device
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)          # (N, N)  -- the score matrix, WRITTEN to HBM
    P = softmax_rows(S)               # (N, N)  -- read S, write P to HBM
    O = P @ V                         # (N, d)  -- read P, read V, write O
    return O

def softmax_rows(S):
    S = S - S.max(axis=-1, keepdims=True)   # numerical stability: subtract row max
    e = np.exp(S)
    return e / e.sum(axis=-1, keepdims=True)
```

The arithmetic is two matrix multiplies ($QK^\top$ and $PV$), each $O(N^2 d)$ floating-point operations, plus an $O(N^2)$ softmax. On a modern GPU those FLOPs run on the tensor cores at, on the order of, hundreds of teraFLOP/s. So compute is cheap. What is expensive is the data movement, and the data movement is dominated by the two $N \times N$ intermediates $S$ and $P$.

Recall the memory hierarchy from [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html). A GPU has a small amount of extremely fast on-chip **SRAM** (the shared memory / L1 cache of each streaming multiprocessor, on the order of 100–200 KB per SM, with aggregate bandwidth in the tens of TB/s) and a large amount of much slower off-chip **HBM** (tens of GB at, on the order of, 2–3.5 TB/s). The ratio matters: SRAM is roughly an order of magnitude faster per byte than HBM, and there is *far* less of it.

The naive kernel makes the worst possible use of this hierarchy. Look at the HBM traffic:

```text
  operation        reads from HBM            writes to HBM       bytes (fp16)
  ---------------   -----------------------   -----------------   --------------
  S = Q K^T         Q (N·d), K (N·d)          S (N·N)             ~2·N² + 4·N·d
  P = softmax(S)    S (N·N)                   P (N·N)             ~4·N²
  O = P V           P (N·N), V (N·d)          O (N·d)             ~2·N² + 4·N·d
  ---------------------------------------------------------------------------
  TOTAL HBM traffic dominated by the N² terms:               Θ(N²)  elements
```

Every $N \times N$ matrix is born in HBM, dies in HBM, and is dragged across the bus in between. The total HBM traffic scales as $\Theta(N^2)$ — quadratic in sequence length, the same order as the FLOPs, but the *bytes* are moving over a channel that is an order of magnitude slower than the one the *FLOPs* run on. Plug that into the roofline picture from [The Roofline Model](../04-kernels-efficiency/01-roofline-performance.html): the **arithmetic intensity** (FLOPs per byte of HBM traffic) is low and roughly constant in $N$, which puts attention firmly on the **memory-bound** (bandwidth-limited) side of the roofline. The hardware's tensor cores sit mostly idle, starved, waiting for HBM.

{{fig:naive-vs-flash-hbm}}

There is a second, even harsher cost: **memory capacity.** The $S$ and $P$ matrices are $O(N^2)$ in size. At $N = 8192$ and fp16, a single $N \times N$ matrix is $8192^2 \times 2 \approx 134$ MB — per head, per layer. With dozens of heads and dozens of layers, and a backward pass that wants to keep $P$ around for the gradient, you blow through HBM capacity long before you run out of compute. This is the wall that capped context lengths for years.

!!! example "The N² wall, in real numbers"
    Take a 1-batch, single-head forward pass at $d = 128$ and let the sequence length grow. The $S$ matrix alone (fp16, 2 bytes/element):

    | $N$ | $N^2$ entries | $S$ size (fp16) |
    |---|---|---|
    | 1,024 | ~1.05 M | ~2 MB |
    | 8,192 | ~67 M | ~134 MB |
    | 32,768 | ~1.07 B | ~2.1 GB |
    | 131,072 | ~17.2 B | ~34 GB |

    Now remember a real model has, on the order of, 32 layers and 32 heads — but those run sequentially or in modest parallel, so the per-head matrix is what hits memory at any instant. Still, at $N=131{,}072$ a *single* score matrix does not fit in a 24 GB consumer GPU, and barely fits in an 80 GB datacenter GPU with nothing else loaded. FlashAttention's peak extra memory is instead $O(N)$ (it keeps only the output and two length-$N$ statistics), independent of $N^2$. That is the difference between "long context is impossible" and "long context is routine."

The diagnosis is now sharp. Attention is slow because (1) it moves $\Theta(N^2)$ bytes across the slow HBM bus, and (2) it allocates $\Theta(N^2)$ bytes of HBM it does not fundamentally need. Both pathologies share one cause: the algorithm **materializes the full score matrix in HBM.** If we could compute the exact same output without ever writing $S$ or $P$ to HBM — keeping the working set in SRAM — both problems vanish. The obstacle to doing that is the softmax, because softmax over a row needs to see the *whole* row (to normalize) before it can produce any output. The online softmax dissolves that obstacle.

## The online softmax: computing a normalized sum in one streaming pass

Forget attention for a moment and consider the bare numerical problem. We have a long vector of logits $x_1, x_2, \dots, x_N$ and we want the softmax weights $p_i = e^{x_i} / \sum_j e^{x_j}$ — but in practice we never want $p$ alone; in attention we want the *weighted sum* $\sum_i p_i v_i$ for value vectors $v_i$. The question: can we compute this **while streaming** the $x_i$ (and $v_i$) one block at a time, using $O(1)$ extra state, instead of making two full passes (one to find the normalizer, one to accumulate)?

### Why the naive online attempt fails

Softmax must be computed in a numerically stable way. The naive expression $e^{x_i}/\sum_j e^{x_j}$ overflows the moment any $x_i$ exceeds ~88 (for fp32, $e^{88} \approx 10^{38}$ is near the float max) or underflows to zero for very negative logits. The standard fix is the **max-subtraction** trick: subtract the maximum logit $m = \max_j x_j$ before exponentiating,

$$
p_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}},
$$

which is algebraically identical (the $e^{-m}$ cancels top and bottom) but keeps every exponent $\le 0$, so $e^{x_i - m} \in (0, 1]$ — no overflow, and the largest term is exactly 1. The catch: $m$ depends on the *whole* vector. If we are streaming, we do not know the global max until we have seen everything. That is exactly the dependency that forces the naive kernel to write $S$ to HBM and read it back.

### The key idea: a running max and a correction factor

The online softmax breaks the dependency with a retroactive rescale. We process the logits in blocks. We keep two running statistics:

- $m$ — the maximum logit seen *so far*,
- $\ell$ — the running sum of exponentials, $\sum e^{x_j - m}$, **relative to the current running max** $m$,

and (for attention) an unnormalized output accumulator $o = \sum e^{x_j - m}\, v_j$, also relative to $m$. When a new block arrives and contains a logit larger than the current $m$, the running max jumps from $m_{\text{old}}$ to $m_{\text{new}}$. Every quantity we accumulated was scaled by $e^{-m_{\text{old}}}$; to put it on the new scale we must multiply it by

$$
\frac{e^{-m_{\text{new}}}}{e^{-m_{\text{old}}}} = e^{m_{\text{old}} - m_{\text{new}}} .
$$

Because $m_{\text{new}} \ge m_{\text{old}}$, this **correction factor** $\alpha = e^{m_{\text{old}} - m_{\text{new}}} \in (0, 1]$ shrinks the old accumulators down onto the new, larger scale. That single multiply is the entire trick. It lets us "change our mind" about the normalizer after the fact, in $O(1)$ work, every time we discover a larger logit.

Here is the recurrence in full. Suppose after processing some prefix we hold $(m, \ell, o)$, and a new block contributes logits $\{x_j\}$ with values $\{v_j\}$ (in attention these are one tile of $K, V$). Let the block's local max be $\tilde m = \max_j x_j$.

$$
\begin{aligned}
m_{\text{new}} &= \max(m, \tilde m), \\
\alpha &= e^{\,m - m_{\text{new}}}, \qquad \text{(rescale prior state)}\\
\ell_{\text{new}} &= \alpha\,\ell \;+\; \sum_j e^{\,x_j - m_{\text{new}}}, \\
o_{\text{new}} &= \alpha\,o \;+\; \sum_j e^{\,x_j - m_{\text{new}}}\, v_j .
\end{aligned}
$$

{{fig:online-softmax-rescale}}

After the final block, the true softmax-weighted output is $o / \ell$ — we divide by the denominator exactly once, at the very end. Notice what each line does: $m_{\text{new}}$ updates the max; $\alpha$ rescales the *already-accumulated* sum and output down onto the new max; then we add the new block's contributions, exponentiated against the new max so they are on the same scale.

### A from-scratch implementation and a correctness proof

Let us make this concrete and runnable. The function below computes a streaming, numerically-stable softmax-weighted sum and we assert it equals the two-pass reference.

```python
import numpy as np

def online_softmax_weighted_sum(x, V, block_size=4):
    """
    Compute  sum_i softmax(x)_i * V[i]   in a single streaming pass,
    processing the logits x in blocks, using only O(d) running state.

    x : (N,)   logits
    V : (N, d) value vectors
    returns o/ell : (d,)  the softmax-weighted average of the rows of V
    """
    N, d = V.shape
    m = -np.inf            # running max of logits seen so far
    ell = 0.0              # running sum of exp(x - m)
    o = np.zeros(d)        # running sum of exp(x - m) * v, UNnormalized

    for start in range(0, N, block_size):
        xb = x[start:start + block_size]          # block of logits
        Vb = V[start:start + block_size]          # block of values

        m_block = xb.max()                        # local max of this block
        m_new = max(m, m_block)                   # updated running max

        alpha = np.exp(m - m_new)                 # correction for OLD state
        # exp(m - m_new) with m = -inf at the very first block -> exp(-inf) = 0,
        # which correctly zeroes the (empty) initial accumulators.

        p = np.exp(xb - m_new)                    # block weights vs new max, in (0,1]
        ell = alpha * ell + p.sum()               # rescale old sum, add new
        o   = alpha * o   + p @ Vb                # rescale old output, add new
        m = m_new

    return o / ell

# ---- correctness check against the honest two-pass softmax ----
rng = np.random.default_rng(0)
N, d = 37, 8
x = rng.normal(0, 5, size=N)        # wide spread -> stresses the max-subtraction
V = rng.normal(0, 1, size=(N, d))

# reference: stable softmax then weighted sum
p_ref = np.exp(x - x.max()); p_ref /= p_ref.sum()
o_ref = p_ref @ V

o_online = online_softmax_weighted_sum(x, V, block_size=4)
print("max abs error:", np.abs(o_online - o_ref).max())   # ~1e-16, machine epsilon
```

Run it: the maximum absolute error is on the order of $10^{-16}$ — floating-point round-off, nothing more. The online algorithm is **bit-for-bit equivalent up to round-off**, not an approximation. Here is why, by induction on the blocks.

**Claim.** After processing the first $k$ blocks (covering indices in a set $I_k$), the state satisfies, exactly,

$$
m = \max_{i \in I_k} x_i, \qquad
\ell = \sum_{i \in I_k} e^{x_i - m}, \qquad
o = \sum_{i \in I_k} e^{x_i - m}\, v_i .
$$

**Base case** ($k=0$, no blocks): $m = -\infty$, and the empty sums are $0$, matching $\ell = o = 0$. ✓

**Inductive step.** Assume the claim holds after $k$ blocks. Process block $k{+}1$ with indices $J$ and local max $\tilde m$. The update sets $m_{\text{new}} = \max(m, \tilde m) = \max_{i \in I_k \cup J} x_i$, which is correct. For the denominator, the old sum was relative to the old $m$; multiplying by $\alpha = e^{m - m_{\text{new}}}$ re-bases it:

$$
\alpha \, \ell = e^{m - m_{\text{new}}} \sum_{i \in I_k} e^{x_i - m} = \sum_{i \in I_k} e^{x_i - m_{\text{new}}},
$$

and adding $\sum_{j \in J} e^{x_j - m_{\text{new}}}$ gives $\ell_{\text{new}} = \sum_{i \in I_k \cup J} e^{x_i - m_{\text{new}}}$, exactly the claim for $k{+}1$. The identical argument holds for $o$ with the $v_i$ carried along. ∎

So at the end $m = \max_i x_i$, $\ell = \sum_i e^{x_i - m}$, and $o = \sum_i e^{x_i - m} v_i$, hence $o/\ell = \sum_i \frac{e^{x_i - m}}{\sum_j e^{x_j - m}} v_i = \sum_i p_i v_i$. The streaming computation equals the reference. The whole point is that we paid only $O(d)$ state and one extra multiply per block to avoid ever holding all $N$ logits at once. The numerical stability is preserved at every step because every exponent $x_j - m_{\text{new}} \le 0$, so we never exponentiate a positive number — the running max guarantees it.

!!! example "Watch the correction factor work"
    Stream the logits $x = [1,\,2,\,8,\,3]$ with block size 2 (blocks $[1,2]$ then $[8,3]$). First, the reference. The global max is $8$, so the stable exponentials are $e^{1-8}, e^{2-8}, e^{8-8}, e^{3-8} = 0.000912,\ 0.002479,\ 1,\ 0.006738$, summing to $\ell^\star = 1.010129$. We will reproduce this value by streaming, never seeing all four logits at once.

    **Block 1** $[1, 2]$: local max $\tilde m = 2$, so $m = 2$, $\alpha = e^{-\infty - 2}\to 0$ (no prior state). $\ell = e^{1-2} + e^{2-2} = 0.3679 + 1 = 1.3679$.

    **Block 2** $[8, 3]$: local max $\tilde m = 8$, new $m_{\text{new}} = 8$. Correction $\alpha = e^{2 - 8} = e^{-6} = 0.002479$. Rescale old denominator: $\alpha \cdot 1.3679 = 0.003391$. Add new terms $e^{8-8} + e^{3-8} = 1 + 0.006738 = 1.006738$. Total $\ell = 0.003391 + 1.006738 = 1.010129$.

    The streaming denominator $1.010129$ equals the reference $\ell^\star = 1.010129$ exactly. Watch what the correction factor bought us: block 1 was accumulated against its *local* max of $2$ (giving $\ell = 1.3679$), yet when block 2 revealed the true max of $8$, the single multiply $\alpha = e^{2-8}$ re-based that entire prior contribution down onto the final scale — no second pass over the data, no re-reading of block 1. The running max guarantees every exponent stays $\le 0$, and the final divide by $\ell$ happens exactly once.

## FlashAttention forward: tiling + online softmax fused into one pass

{{fig:flash-tiling}}

Now we lift the online softmax from a single row to the full attention matrix. The observation that makes the row-wise online softmax into FlashAttention: **every output row is an independent online softmax over the key/value blocks.** Row $i$ of the output is $\sum_j p_{ij} v_j$ where $p_{ij} = \operatorname{softmax}_j(S_{ij})$, and we can compute that row by streaming over column-blocks of $K, V$ exactly as above — running max $m_i$, running denominator $\ell_i$, running output $o_i$ — all in SRAM. We do this for a *block of query rows* at a time, so that the $Q$ tile, the $K$/$V$ tiles, and the accumulators all fit on-chip simultaneously.

The structure is a double loop. The **outer loop** is over blocks of query rows ($B_r$ rows at a time). The **inner loop** is over blocks of key/value rows ($B_c$ columns at a time). For each $(i, j)$ tile we load $Q_i, K_j, V_j$ from HBM into SRAM, compute the small $B_r \times B_c$ score tile, update the running statistics for the $B_r$ rows, and move on. The score tile is born and consumed entirely in SRAM; it is **never written to HBM.** That is the whole game.

{{fig:flash1-forward-dataflow}}

Here is a faithful, runnable NumPy implementation of the forward pass. It mirrors the structure of the real CUDA kernel — the same loops, the same statistics, the same rescale — just without the GPU. (Causal masking and the actual SRAM tiling are noted in comments; we will add the mask in the code that follows.)

```python
import numpy as np

def flash_attention_forward(Q, K, V, Br=32, Bc=32, causal=False):
    """
    Exact attention via tiling + online softmax. Mirrors the FlashAttention-1
    forward pass. No N×N matrix is ever fully materialized: the largest
    intermediate is one Br×Bc score tile.

    Q, K, V : (N, d)
    returns:
      O : (N, d)         attention output
      L : (N,)           logsumexp statistic per row  (= m + log ell), saved for backward
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    O = np.zeros((N, d))
    L = np.zeros(N)                      # row logsumexp, needed by the backward pass

    # OUTER LOOP over query blocks (each maps to a CUDA thread block / program)
    for i0 in range(0, N, Br):
        i1 = min(i0 + Br, N)
        Qi = Q[i0:i1]                    # (br, d)  -- stays in SRAM for the whole inner loop
        br = i1 - i0

        # Per-row running statistics, initialized for "nothing seen yet"
        m_i  = np.full(br, -np.inf)      # running max logit per row
        l_i  = np.zeros(br)              # running denominator per row
        O_i  = np.zeros((br, d))         # running UNnormalized output per row

        # INNER LOOP over key/value blocks
        for j0 in range(0, N, Bc):
            j1 = min(j0 + Bc, N)
            Kj = K[j0:j1]                # (bc, d)
            Vj = V[j0:j1]                # (bc, d)

            # 1) score tile in SRAM (Br×Bc) -- the only "big" intermediate, and it is tiny
            Sij = (Qi @ Kj.T) * scale    # (br, bc)

            if causal:
                # mask out keys j > query i  (positions are absolute indices)
                rows = np.arange(i0, i1)[:, None]
                cols = np.arange(j0, j1)[None, :]
                Sij = np.where(cols <= rows, Sij, -np.inf)

            # 2) online-softmax update for these br rows over this block
            m_block = Sij.max(axis=1)               # (br,) local max of the tile
            m_new   = np.maximum(m_i, m_block)       # (br,) updated running max
            # guard rows that are entirely -inf (e.g. fully-masked causal blocks)
            m_new   = np.where(np.isneginf(m_new), 0.0, m_new)

            P = np.exp(Sij - m_new[:, None])         # (br, bc) tile weights, in [0,1]
            alpha = np.exp(m_i - m_new)              # (br,) correction for old state
            alpha = np.where(np.isneginf(m_i), 0.0, alpha)

            l_i = alpha * l_i + P.sum(axis=1)        # rescale + accumulate denominator
            O_i = alpha[:, None] * O_i + P @ Vj      # rescale + accumulate output
            m_i = m_new

        # 3) finalize: normalize once, and save the logsumexp for backward
        l_safe = np.where(l_i == 0.0, 1.0, l_i)      # avoid 0/0 for fully-masked rows
        O[i0:i1] = O_i / l_safe[:, None]
        L[i0:i1] = m_i + np.log(l_safe)              # logsumexp = m + log(sum exp(.-m))

    return O, L


# ---- verify against the reference, with and without causal masking ----
def reference_attention(Q, K, V, causal=False):
    d = Q.shape[-1]
    S = (Q @ K.T) / np.sqrt(d)
    if causal:
        N = Q.shape[0]
        mask = np.tril(np.ones((N, N), bool))
        S = np.where(mask, S, -np.inf)
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
    return P @ V

rng = np.random.default_rng(1)
N, d = 100, 16
Q = rng.normal(size=(N, d)); K = rng.normal(size=(N, d)); V = rng.normal(size=(N, d))

for causal in (False, True):
    O_flash, _ = flash_attention_forward(Q, K, V, Br=32, Bc=32, causal=causal)
    O_ref = reference_attention(Q, K, V, causal=causal)
    print(f"causal={causal}: max abs error = {np.abs(O_flash - O_ref).max():.2e}")
# both print ~1e-15: exact up to floating-point round-off
```

A few details worth pausing on.

- **The accumulators are the entire memory footprint per query block.** For a block of $B_r$ rows we hold $m_i$ ($B_r$ floats), $\ell_i$ ($B_r$ floats), and $O_i$ ($B_r \times d$ floats) — and the transient $B_r \times B_c$ score tile. Choose $B_r, B_c$ so all of this fits in SRAM. Nothing scales with $N$. The peak *extra* HBM allocation across the whole kernel is just $O$ (size $N \times d$) and $L$ (size $N$) — linear in $N$, never quadratic.

- **The logsumexp $L_i = m_i + \log \ell_i$ is the one statistic we save.** It is the log of the softmax denominator for row $i$, computed in a stable way. We will see in the next section that the backward pass needs exactly this scalar per row to reconstruct the softmax weights on the fly — so storing $L$ (size $N$) lets us *recompute* $P$ (size $N^2$) in backward instead of storing it. This is the recomputation trade we make.

- **Causal masking is free.** With a causal mask, query $i$ only attends to keys $j \le i$. In the tiled loop that means the outer query-block $i$ can simply **skip all key-blocks $j$ that lie entirely above the diagonal** ($j_0 > i_1$), and only the diagonal block needs the elementwise triangular mask. This roughly halves the work for long sequences — a major reason causal FlashAttention is so fast — and the real kernel exploits it explicitly.

!!! note "Why the inner loop is over keys, not queries"
    FlashAttention-1 puts the **query block** in the outer loop and **key/value blocks** in the inner loop, accumulating each output row to completion before moving on. This keeps the running statistics ($m_i, \ell_i, O_i$) for a query block resident in registers/SRAM across the whole inner loop, and writes each output row to HBM exactly once. [FlashAttention 2](../04-kernels-efficiency/03-flash-attention-2-3.html) revisits this loop ordering and the placement of the rescale to reduce non-matmul FLOPs and improve GPU occupancy — but the online-softmax core is identical.

## The backward pass: recomputation instead of storage

Training needs gradients. The standard backward pass for attention stores the $N \times N$ probability matrix $P$ from the forward pass so it can compute $\partial \mathcal{L}/\partial Q$, $\partial \mathcal{L}/\partial K$, $\partial \mathcal{L}/\partial V$. But $P$ is exactly the $O(N^2)$ object we worked so hard not to materialize. FlashAttention's answer is **recomputation** (a.k.a. activation checkpointing, see [Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html)): do *not* store $P$; instead, recompute each $P$ tile from $Q, K$ and the saved logsumexp $L$ during the backward pass, again block by block in SRAM. We trade extra FLOPs (recomputing the scores) for not touching HBM with an $N^2$ matrix. Because attention is memory-bound, that trade is a clear win — the "wasted" recomputation FLOPs run on otherwise-idle tensor cores.

First, the math. With $S = QK^\top/\sqrt{d}$, $P = \operatorname{softmax}_{\text{row}}(S)$, and $O = PV$, and given the upstream gradient $\mathrm{d}O = \partial \mathcal{L} / \partial O$, the chain rule gives:

$$
\mathrm{d}V = P^\top \,\mathrm{d}O, \qquad \mathrm{d}P = \mathrm{d}O\, V^\top .
$$

The softmax Jacobian (per row, since softmax is applied row-wise) turns $\mathrm{d}P$ into $\mathrm{d}S$. For a single softmax row $p = \operatorname{softmax}(s)$ with upstream $\mathrm{d}p$, the standard result is

$$
\mathrm{d}s_i = p_i\left(\mathrm{d}p_i - \textstyle\sum_k p_k\, \mathrm{d}p_k\right).
$$

In attention, $\mathrm{d}p_{ij} = (\mathrm{d}O\,V^\top)_{ij}$ and $\sum_k p_{ik}\,\mathrm{d}p_{ik} = \sum_k p_{ik}(\mathrm{d}O_i \cdot V_k) = \mathrm{d}O_i \cdot (\sum_k p_{ik} V_k) = \mathrm{d}O_i \cdot O_i$. Define the per-row scalar

$$
D_i \;=\; \sum_k p_{ik}\,\mathrm{d}p_{ik} \;=\; \mathrm{d}O_i \cdot O_i \;=\; \sum_c \mathrm{d}O_{ic}\,O_{ic},
$$

which is just the row-wise dot product of the output gradient with the output — a cheap $O(Nd)$ quantity computable up front. Then

$$
\mathrm{d}S_{ij} = P_{ij}\big( (\mathrm{d}O\,V^\top)_{ij} - D_i \big), \qquad
\mathrm{d}Q = \frac{\mathrm{d}S\, K}{\sqrt d}, \qquad
\mathrm{d}K = \frac{\mathrm{d}S^\top Q}{\sqrt d}.
$$

The crucial enabling fact: we can reconstruct $P_{ij}$ in any tile without storing it, because $P_{ij} = \exp(S_{ij} - L_i)$, where $S_{ij} = (Q_i K_j^\top)/\sqrt d$ is recomputed and $L_i$ is the saved logsumexp. The logsumexp absorbs both the max-subtraction and the normalization into one number — $\exp(S_{ij} - L_i)$ is *already* the normalized probability, numerically stable. This is precisely why the forward pass bothered to save $L$.

```python
import numpy as np

def flash_attention_backward(Q, K, V, O, L, dO, Br=32, Bc=32, causal=False):
    """
    Backward pass for FlashAttention. Recomputes P tile-by-tile from (Q,K,L)
    instead of storing the N×N matrix. Returns dQ, dK, dV.

    O, L are the forward outputs; dO is the upstream gradient (N, d).
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    # Per-row scalar D_i = sum_c dO_{ic} O_{ic}  (rowwise dot of dO and O)
    D = np.sum(dO * O, axis=1)                    # (N,)

    for i0 in range(0, N, Br):
        i1 = min(i0 + Br, N)
        Qi, dOi, Li, Di = Q[i0:i1], dO[i0:i1], L[i0:i1], D[i0:i1]
        dQi = np.zeros_like(Qi)

        for j0 in range(0, N, Bc):
            j1 = min(j0 + Bc, N)
            Kj, Vj = K[j0:j1], V[j0:j1]

            # RECOMPUTE the score tile and the probability tile (no storage)
            Sij = (Qi @ Kj.T) * scale             # (br, bc)
            if causal:
                rows = np.arange(i0, i1)[:, None]
                cols = np.arange(j0, j1)[None, :]
                Sij = np.where(cols <= rows, Sij, -np.inf)
            Pij = np.exp(Sij - Li[:, None])       # = softmax weights, since L is logsumexp

            # gradients flowing through this tile
            dV[j0:j1] += Pij.T @ dOi              # dV = P^T dO   (accumulate over query blocks)
            dPij = dOi @ Vj.T                      # (br, bc)  dP = dO V^T
            dSij = Pij * (dPij - Di[:, None])      # softmax-Jacobian: P*(dP - D)
            dQi   += (dSij @ Kj) * scale           # dQ = dS K / sqrt(d)
            dK[j0:j1] += (dSij.T @ Qi) * scale     # dK = dS^T Q / sqrt(d)

        dQ[i0:i1] = dQi

    return dQ, dK, dV


# ---- gradient check against autograd-style finite differences ----
def reference_attention_loss(Q, K, V, causal=False):
    O = reference_attention(Q, K, V, causal=causal)   # defined earlier
    return O

rng = np.random.default_rng(2)
N, d = 48, 8
Q = rng.normal(size=(N, d)); K = rng.normal(size=(N, d)); V = rng.normal(size=(N, d))
causal = True

O, L = flash_attention_forward(Q, K, V, Br=16, Bc=16, causal=causal)
dO = rng.normal(size=(N, d))                            # arbitrary upstream grad
dQ, dK, dV = flash_attention_backward(Q, K, V, O, L, dO, Br=16, Bc=16, causal=causal)

# numerical gradient of  scalar = sum(dO * O)  w.r.t. Q, via central differences
def num_grad(param, idx):
    eps = 1e-5
    p = param.copy(); orig = p[idx]
    p[idx] = orig + eps; Op = reference_attention_loss(*( (p,K,V) if param is Q else
                                  (Q,p,V) if param is K else (Q,K,p)), causal=causal)
    fp = np.sum(dO * Op)
    p[idx] = orig - eps; Om = reference_attention_loss(*( (p,K,V) if param is Q else
                                  (Q,p,V) if param is K else (Q,K,p)), causal=causal)
    fm = np.sum(dO * Om)
    return (fp - fm) / (2 * eps)

err = max(abs(num_grad(Q, (3, 2)) - dQ[3, 2]),
          abs(num_grad(K, (5, 1)) - dK[5, 1]),
          abs(num_grad(V, (7, 4)) - dV[7, 4]))
print("max grad-check error:", err)     # ~1e-7: analytic backward matches finite differences
```

The gradient check passes to $\sim 10^{-7}$, the expected precision of central differences in fp64. The backward pass is genuinely exact. Two practical notes mirror the real kernel:

- **$\mathrm{d}V$ and $\mathrm{d}K$ accumulate across query blocks** (every query attends to a given key), so in a parallel implementation they require atomic adds or a separate reduction pass. FlashAttention-1 handles this by looping keys in the outer loop for the backward pass (the opposite of the forward), so each key block's $\mathrm{d}K_j, \mathrm{d}V_j$ is finalized in one go while $\mathrm{d}Q$ is accumulated with atomics. The math above is loop-order-agnostic; the implementation chooses an order that minimizes atomics.

- **Recomputation cost is small and overlaps.** The backward pass redoes the $QK^\top$ matmul (the score tile) it could have stored. That is one extra $O(N^2 d)$ matmul — but matmuls run on tensor cores that the memory-bound kernel was leaving idle anyway, and we *save* the $O(N^2)$ HBM read/write of the stored $P$. On the memory-bound side of the roofline, trading FLOPs for bytes is exactly the right direction.

## IO-aware analysis: counting the bytes

We have claimed FlashAttention is faster because it moves fewer bytes. Let us prove it with an IO-complexity argument — the kind of analysis the paper is named for, and the kind an interviewer loves.

The model: HBM is large and slow; SRAM has size $M$ (in elements) and is fast. We count **HBM accesses** (reads + writes), because that is what dominates runtime for a memory-bound kernel. The inputs $Q, K, V$ are $N \times d$; assume $d \le M$ (a few head-dimensions of vectors fit in SRAM, which is always true in practice).

**Naive attention.** It writes and reads the $N \times N$ matrices $S$ and $P$. Even with perfect overlap, $S$ is written once and read once, $P$ is written once and read once: that is $\Theta(N^2)$ HBM accesses, plus $\Theta(Nd)$ for $Q,K,V,O$. Total:

$$
\text{HBM}_{\text{naive}} = \Theta\!\left(N^2 + N d\right) = \Theta(N^2 d \, / \, d) \approx \Theta(N^2).
$$

**FlashAttention.** Choose block sizes so that one $Q$ tile, one $K$ tile, one $V$ tile, and the accumulators all fit in SRAM. With SRAM size $M$, the column block can be $B_c = \Theta(M/d)$ and the row block $B_r = \Theta(M/d)$ as well (bounded so the $B_r \times B_c$ score tile $\le M$). Now count: the outer loop runs $N / B_r$ times; for *each* query block, the inner loop streams **all** of $K$ and $V$ from HBM once — that is $\Theta(Nd)$ bytes per query block. So:

$$
\text{HBM}_{\text{flash}} = \frac{N}{B_r} \cdot \Theta(Nd) = \Theta\!\left(\frac{N^2 d}{B_r}\right) = \Theta\!\left(\frac{N^2 d^2}{M}\right).
$$

The last step uses $B_r = \Theta(M/d)$. Compare the two: FlashAttention's HBM traffic is smaller than naive's $\Theta(N^2)$ by a factor of $\Theta(M/d^2)$. With SRAM on the order of $M \sim 10^5$ elements and $d^2 \sim 10^4$, that is an order-of-magnitude reduction in HBM traffic — and it is provably (the paper shows) optimal up to constants over all algorithms that compute exact attention by this tiling. The bigger the SRAM, the fewer passes over $K, V$, the less traffic. This is the formal content of "IO-awareness": the algorithm's cost is dominated by HBM accesses, so we design the algorithm to minimize them, not to minimize FLOPs.

```text
                  HBM accesses        SRAM scratch        extra HBM memory
  ---------------  -----------------   ----------------    ----------------
  Naive attention   Θ(N²)              O(Br·Bc) but it      Θ(N²)  (S and P)
                                       still spills S,P
  FlashAttention    Θ(N² d² / M)       O(Br·Bc) tile +      Θ(N)   (O and L)
                                       O(Br·d) accumulators

  M = SRAM size (elements). For typical M and d, FlashAttention moves ~M/d²
  fewer bytes — about an order of magnitude — and uses O(N) instead of O(N²)
  scratch memory. Same FLOPs, same answer, far fewer HBM trips.
```

Two consequences fall out of this accounting:

1. **It is exact, and the speedup is "free."** FlashAttention computes the identical function as naive attention (we proved it above, twice). The speedup comes purely from data movement, not from any approximation, sparsity, or low-rank trick. This is what distinguishes it from earlier "efficient attention" methods (Linformer, Performer, sparse attention — see [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html)) that traded accuracy for speed. FlashAttention trades nothing.

2. **Memory drops from $O(N^2)$ to $O(N)$.** The only persistent extra allocations are $O$ ($N \times d$) and $L$ ($N$). This linear scaling is what unlocked long-context training and inference; it is the foundation under nearly every long-context model shipped since 2022, and under the prefill phase of [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html).

!!! warning "FlashAttention is memory-bound, not compute-bound — measure the right thing"
    A common mistake when benchmarking is to count FLOPs and conclude FlashAttention "does the same work" as naive attention (true) and therefore "can't be faster" (false). The FLOPs are identical; the *wall-clock* difference is all in HBM traffic. If you profile and see the kernel spending its time on memory transactions rather than tensor-core issue, that is expected. The corollary for [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html): once you have killed the HBM traffic, the *next* bottleneck becomes the non-matmul work (the rescales and exponentials) and GPU occupancy — which is exactly what FA-2 and FA-3 attack.

## Practical use: you almost never write this yourself

You now understand the algorithm well enough to implement it. In practice you call it. PyTorch ships a fused, hardware-dispatched attention that uses a FlashAttention backend when shapes and dtypes allow:

```python
import torch
import torch.nn.functional as F

# q, k, v : (batch, heads, seq_len, head_dim), on CUDA, in fp16/bf16
q = torch.randn(2, 16, 8192, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Single fused call. PyTorch dispatches to a FlashAttention kernel when eligible:
# contiguous, supported head_dim, fp16/bf16, etc. is_causal applies the triangular mask
# WITHOUT materializing it (the kernel skips upper-triangular key blocks, as we discussed).
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (2, 16, 8192, 128)

# You can inspect / force which backend is used:
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Things that silently *disable* the FlashAttention backend and fall back to a slower math path — worth memorizing because they show up in profiling:

- **Unsupported `head_dim`.** Older kernels support head dims up to 128 (later up to 256). A head dim of 160, say, may fall back.
- **fp32 inputs.** FlashAttention kernels target fp16/bf16. Pass fp32 and you get the math fallback.
- **An additive float mask with awkward shape**, or a mask that the kernel cannot fuse, can force the materialized path. Prefer `is_causal=True` over hand-built masks when possible.
- **Non-contiguous tensors** or unsupported strides.

The official `flash-attn` package (Dao et al.) exposes the kernels directly (`flash_attn_func`, `flash_attn_varlen_func` for ragged/packed batches — relevant to [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html)) with more knobs: sliding-window attention, ALiBi slopes, dropout fused into the kernel, and the variable-length API that avoids padding waste. For writing your *own* fused attention variants (custom masks, custom biases) the modern path is [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html), where a FlashAttention-style kernel is the canonical tutorial example.

!!! tip "Block size is a tuning knob, not a constant"
    $B_r$ and $B_c$ are chosen to fill SRAM without spilling, and the sweet spot depends on $d$ and the GPU's SRAM-per-SM. The IO analysis says bigger blocks → fewer passes over $K,V$ → less HBM traffic, but blocks too big spill registers/SRAM and tank occupancy. Real kernels autotune these (often 64 or 128). When you write a Triton version, sweep `BLOCK_M` and `BLOCK_N` — it is routinely a 1.5–2× swing.

!!! interview "Interview Corner"
    **Q:** Naive attention and FlashAttention do the *same* number of floating-point operations and compute the *same* result. So where does the speedup come from, and why doesn't it apply to, say, a single large matrix multiply?

    **A:** The speedup is entirely from reduced HBM traffic, not reduced FLOPs. Naive attention materializes the $N \times N$ score and probability matrices in HBM, so it pays $\Theta(N^2)$ slow memory accesses; attention is *memory-bound*, so that traffic — not the matmuls — dominates wall-clock time. FlashAttention tiles the computation so the score matrix lives only in fast on-chip SRAM and is never written to HBM; using the online softmax (running max, running denominator, and a correction factor that rescales partial accumulators when a larger logit appears) it produces the exact same softmax-weighted output in a single streaming pass. The IO complexity drops from $\Theta(N^2)$ to $\Theta(N^2 d^2/M)$ where $M$ is SRAM size — about an order of magnitude fewer bytes — and peak memory drops from $O(N^2)$ to $O(N)$. It does *not* help a single large dense matmul because a big matmul is already *compute-bound* with high arithmetic intensity: its bottleneck is the tensor cores, not HBM, so there is no IO to eliminate. The win is specific to operations like attention that have low arithmetic intensity *and* a large intermediate that standard implementations spill to HBM. A sharp follow-up: the backward pass uses recomputation — it does not store $P$ but reconstructs each tile from $Q$, $K$, and the saved per-row logsumexp $L$ — trading extra (idle) tensor-core FLOPs for avoiding the $O(N^2)$ HBM read of a stored probability matrix.

## Putting it together: the mental model

Step back and hold the whole thing at once. Standard attention is correct but pathological on a GPU: it writes a quadratic intermediate to the slow memory and reads it back twice, so it is bandwidth-starved and capacity-limited. The single obstacle to fixing this is that softmax normalization seems to need the whole row before it can emit anything. The online softmax dissolves that obstacle with a running maximum, a running denominator, and a correction factor $e^{m_{\text{old}} - m_{\text{new}}}$ that retroactively re-bases partial results whenever a bigger logit shows up — letting us compute a numerically-stable, exactly-correct softmax-weighted sum in one streaming pass over blocks. Wrap that streaming softmax in a tile loop that keeps every working block in SRAM, normalize once at the end, and save a single logsumexp per row so the backward pass can recompute the probabilities instead of storing them. The result moves an order of magnitude fewer bytes, uses linear instead of quadratic memory, and computes precisely the same function. No approximation. That combination — an elegant numerical recurrence married to a hardware-aware data layout — is why FlashAttention is one of the most consequential systems papers of the LLM era, and why "IO-awareness" is now a design principle, not just a kernel.

!!! key "Key Takeaways"
    - Naive attention is **memory-bound, not compute-bound**: it materializes the $N \times N$ score and probability matrices in HBM, paying $\Theta(N^2)$ slow memory accesses and $O(N^2)$ peak memory. The matmuls were never the problem.
    - The **online softmax** computes a numerically-stable, exactly-correct softmax-weighted sum in one streaming pass using three running statistics — max $m$, denominator $\ell$, output $o$ — and a **correction factor** $\alpha = e^{m_{\text{old}} - m_{\text{new}}} \in (0,1]$ that rescales prior accumulators whenever a larger logit appears.
    - FlashAttention **fuses** $QK^\top$, softmax, and $\cdot V$ into a single tiled kernel; the $B_r \times B_c$ score tile lives only in **SRAM** and is never written to HBM. Peak extra memory falls from $O(N^2)$ to $O(N)$ (just the output $O$ and the per-row logsumexp $L$).
    - It is **exact**, not approximate — bit-for-bit equal to naive attention up to floating-point round-off. The speedup is purely from reduced data movement.
    - The **backward pass uses recomputation**: it does not store $P$; it reconstructs each probability tile as $\exp(S_{ij} - L_i)$ from $Q$, $K$, and the saved logsumexp, trading idle-tensor-core FLOPs for avoiding an $O(N^2)$ HBM read. The per-row scalar $D_i = \mathrm{d}O_i \cdot O_i$ collapses the softmax Jacobian.
    - **IO complexity** drops from $\Theta(N^2)$ to $\Theta(N^2 d^2 / M)$ where $M$ is SRAM size — about an order of magnitude — and is provably near-optimal for exact tiled attention.
    - **Causal masking is nearly free**: tiled kernels skip key-blocks entirely above the diagonal, roughly halving the work for long sequences.
    - In practice you call `F.scaled_dot_product_attention` or the `flash-attn` library; know the fallback triggers (fp32, unsupported head dim, awkward masks). Custom variants are written in [Triton](../04-kernels-efficiency/04-triton-kernels.html).

!!! sota "State of the Art & Resources (2026)"
    FlashAttention is now the universal baseline for exact attention on modern GPUs: FA-2 ships inside PyTorch's `scaled_dot_product_attention`, FA-3 targets Hopper's async tensor cores and FP8, and FA-4 (beta) extends the approach to Blackwell. The core IO-aware tiling idea has spawned a broad ecosystem of inference-optimized and flexibly-programmable attention kernels.

    **Foundational papers**

    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — the original IO-aware tiling algorithm; introduces the online-softmax forward/backward and the IO-complexity theorem.
    - [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691) — revises the loop ordering and work partitioning, roughly doubling throughput to 50–73% of A100 peak FLOPs/s.
    - [Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018)](https://arxiv.org/abs/1805.02867) — the streaming-softmax recurrence that FlashAttention's online softmax is built on.

    **Pushing the frontier (2024–2026)**

    - [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608) — exploits Hopper warp-specialization and TMA to overlap compute with data movement; FP16 reaches 740 TFLOPs/s and FP8 approaches 1.2 PFLOPs/s on H100.
    - [Bikshandi & Shah, *A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library* (2023)](https://arxiv.org/abs/2312.11918) — deep dive into Hopper-specific CUTLASS primitives underlying FA-3; 20–50% gains over Ampere-optimized FA-2.
    - [Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving* (2025)](https://arxiv.org/abs/2501.01005) — block-sparse KV-cache formatting and JIT kernel generation for variable-length decode; adopted by vLLM and SGLang.

    **Open-source & tools**

    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the canonical production library: FA-1 through FA-4 (beta), `flash_attn_varlen_func` for packed sequences, sliding-window and ALiBi support.
    - [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) — inference-focused attention engine with JIT kernel composition, block-sparse KV layout, and support for decode-phase workloads.

    **Go deeper**

    - [PyTorch Blog: *FlexAttention — The Flexibility of PyTorch with the Performance of FlashAttention* (2024)](https://pytorch.org/blog/flexattention/) — how PyTorch's `flex_attention` API lowers user-supplied `score_mod` / `mask_mod` functions into a fused FA-style kernel via `torch.compile`.

## Further reading

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022). The original paper; read it for the IO-complexity theorem and the forward/backward algorithm boxes.
- Tri Dao — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023). The follow-up on loop ordering, non-matmul FLOP reduction, and occupancy; covered in [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html).
- Maxim Milakov and Natalia Gimelshein — *Online normalizer calculation for softmax* (2018). The streaming softmax recurrence that FlashAttention builds on.
- Markus N. Rabe and Charles Staats — *Self-attention Does Not Need $O(n^2)$ Memory* (2021). An earlier memory-reduction-by-recomputation result for attention.
- Ashish Vaswani et al. — *Attention Is All You Need* (2017). The attention operator FlashAttention accelerates; the from-scratch version is in [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).
- The `Dao-AILab/flash-attention` GitHub repository — production CUDA kernels, the `flash_attn` Python API, and the varlen/packed-sequence entry points.
- The OpenAI Triton tutorials — the fused-attention example is the canonical place to see a FlashAttention-style kernel written in a high-level GPU language; see [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html).
