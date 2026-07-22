"""
Executable test for content/04-kernels-efficiency/02-flash-attention-1.md

Assembles the chapter's CPU-runnable Python blocks, in the order they appear
in the chapter, into one runnable module. Later blocks depend on names
defined by earlier blocks (e.g. block #4's gradient check calls
`reference_attention`, which is defined inside block #3).

Blocks tested (chapter order):
    #0 (line ~19)  - attention_naive() + softmax_rows(): the naive reference
                     kernel that materializes the full N x N score matrix.
    #2 (line ~120) - online_softmax_weighted_sum(): the streaming, O(d)-state
                     online-softmax core, checked against a two-pass reference.
    #3 (line ~212) - flash_attention_forward() + reference_attention(): the
                     full tiled forward pass, checked against the naive
                     reference with and without causal masking.
    #4 (line ~346) - flash_attention_backward(): the recomputation-based
                     backward pass, checked against central-difference
                     numerical gradients.

The task brief's heuristic classifier flagged blocks #3 and #4 as
"needs-gpu". On inspection both are pure NumPy (no `torch`, no `device=`,
no CUDA of any kind) -- the heuristic likely tripped on in-comment mentions
of "CUDA thread block" / "SRAM" / "GPU" in the docstrings and prose. Per the
task's "Other blocks (default SKIP unless trivially CPU-safe)" rule, both
are included here since they are trivially CPU-safe and are the heart of
the chapter's demonstration (the tiled forward/backward FlashAttention
algorithm in NumPy, explicitly described in the text as "runnable").

Blocks skipped (chapter order):
    #1 (line ~42)  - SKIP(non-python): an ASCII HBM-traffic accounting table
                     in a ```text fence, not executable code.
    #5 (line ~453) - SKIP(non-python): an ASCII IO-complexity accounting
                     table in a ```text fence, not executable code.
    #6 (line ~479) - SKIP(needs-gpu): `F.scaled_dot_product_attention` /
                     `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` demo that
                     constructs tensors with `device="cuda"` and requires an
                     actual FlashAttention-capable GPU kernel; cannot run on
                     CPU-only CI.

Every CPU-runnable code block in the chapter is executed here.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Block #0 (line ~19): the naive reference kernel -- materializes S and P.
# ---------------------------------------------------------------------------


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


# ---- glue: actually call the naive kernel on a tiny input and check it ----
_rng0 = np.random.default_rng(42)
_N0, _d0 = 12, 6
_Q0 = _rng0.normal(size=(_N0, _d0))
_K0 = _rng0.normal(size=(_N0, _d0))
_V0 = _rng0.normal(size=(_N0, _d0))

_O0 = attention_naive(_Q0, _K0, _V0)
assert _O0.shape == (_N0, _d0)

# softmax_rows must produce a proper row-stochastic matrix
_P0 = softmax_rows(_Q0 @ _K0.T / np.sqrt(_d0))
assert np.allclose(_P0.sum(axis=-1), 1.0, atol=1e-10)
assert np.all(_P0 >= 0.0)
print("block #0 attention_naive output shape:", _O0.shape, "row-sums of P ~1:", True)


# ---------------------------------------------------------------------------
# Block #2 (line ~120): online softmax, streaming, O(d) state.
# ---------------------------------------------------------------------------


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
_err2 = np.abs(o_online - o_ref).max()
print("max abs error:", _err2)   # ~1e-16, machine epsilon
assert _err2 < 1e-9, f"online softmax diverges from two-pass reference: {_err2}"


# ---------------------------------------------------------------------------
# Block #3 (line ~212): tiled forward pass + online softmax, fused.
# ---------------------------------------------------------------------------


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
    _err3 = np.abs(O_flash - O_ref).max()
    print(f"causal={causal}: max abs error = {_err3:.2e}")
    assert _err3 < 1e-9, f"flash_attention_forward diverges from reference (causal={causal}): {_err3}"
# both print ~1e-15: exact up to floating-point round-off


# ---------------------------------------------------------------------------
# Block #4 (line ~346): backward pass via recomputation, no P storage.
# ---------------------------------------------------------------------------


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
assert err < 1e-5, f"flash_attention_backward gradient check failed: {err}"


# ---------------------------------------------------------------------------
# SKIP(non-python): block #1 (line ~42) -- an ASCII HBM-traffic accounting
# table in a ```text fence; not executable code.
#
# SKIP(non-python): block #5 (line ~453) -- an ASCII IO-complexity
# accounting table in a ```text fence; not executable code.
#
# SKIP(needs-gpu): block #6 (line ~479) -- constructs tensors with
# device="cuda" and dispatches through torch's FlashAttention CUDA backend
# (`F.scaled_dot_product_attention` under `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`);
# there is no CPU FlashAttention kernel to fall back to, and CI has no GPU.
# ---------------------------------------------------------------------------

print("\nAll CPU-runnable blocks in 04-kernels-efficiency/02-flash-attention-1.md executed successfully.")
