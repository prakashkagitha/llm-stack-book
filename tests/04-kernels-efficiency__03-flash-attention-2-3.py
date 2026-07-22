"""
Runnable-code test for content/04-kernels-efficiency/03-flash-attention-2-3.md

Blocks tested (CPU-runnable, executed in chapter order):
  - block #0 (line ~43):  flash_attention_2_reference() — pure-PyTorch FA2
    dataflow reference (deferred normalization). Purely CPU code (no device=
    "cuda" anywhere); heuristically flagged "needs-gpu" by the scanner but is
    trivially CPU-safe, so it is included here as a bonus in addition to the
    3 required blocks below.
  - block #6 (line ~251): fa3_pipeline_model() — dataflow model of FA3's
    producer/consumer ring-buffer pipeline, computing exact attention for one
    query block.
  - block #7 (line ~336): hadamard(), incoherent_rotation(), fake_fp8_quant()
    — the incoherent-processing (Hadamard rotation) demo for FP8 attention.
  - block #8 (line ~409): xformers memory_efficient_attention snippet.
    SKIP(dependency): xformers is not in the guaranteed-available import
    list for CI (numpy/torch/einops/sklearn/stdlib only) and is not
    installed here. The import is guarded at module scope; the call is only
    made if xformers is actually importable, so the file always loads.

Blocks intentionally SKIPPED (not tested):
  - block #1 (line ~116): ```text``` gridDim pseudocode — not Python.
  - block #2 (line ~135): ```text``` split-K vs split-Q ASCII diagram — not Python.
  - block #3 (line ~155): PyTorch SDPA example — hardcodes device="cuda" and
    torch.nn.attention.sdpa_kernel(...FLASH_ATTENTION...), which requires an
    actual CUDA flash-attention backend. SKIP(needs-gpu).
  - block #4 (line ~210): ```text``` producer/consumer ring-buffer diagram — not Python.
  - block #5 (line ~231): ```text``` ping-pong scheduling diagram — not Python.
"""

import torch
from collections import deque

try:
    from xformers.ops import memory_efficient_attention, LowerTriangularMask
except Exception:
    memory_efficient_attention = None
    LowerTriangularMask = None


# ---------------------------------------------------------------------------
# Block #0 (line ~43): flash_attention_2_reference — FA2 deferred-rescale
# dataflow, verified against dense softmax attention.
# ---------------------------------------------------------------------------


def flash_attention_2_reference(Q, K, V, Br=64, Bc=64, causal=False):
    """
    Exact attention via the FlashAttention-2 recurrence.
    Q, K, V: (N, d). Returns O: (N, d).

    Mirrors the FA2 dataflow:
      - outer loop over QUERY blocks (this is the parallel axis on the GPU)
      - inner loop over KEY/VALUE blocks
      - the output accumulator is kept UNNORMALIZED and divided by l only at the end
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)
    # running statistics, one per query row
    for i in range(0, N, Br):
        qi = Q[i:i+Br]                          # (Br, d)
        Oi = torch.zeros(qi.shape[0], d)        # UNNORMALIZED accumulator
        mi = torch.full((qi.shape[0],), -float('inf'))   # running max
        li = torch.zeros(qi.shape[0])           # running denominator (sum of exp)

        for j in range(0, N, Bc):
            if causal and j > i + Br - 1:
                break                            # whole key block is in the future
            kj = K[j:j+Bc]                       # (Bc, d)
            vj = V[j:j+Bc]                       # (Bc, d)

            # ---- matmul 1: scores ----
            Sij = (qi @ kj.T) / (d ** 0.5)       # (Br, Bc)
            if causal:
                # mask future positions inside the diagonal block
                qpos = torch.arange(i, i+qi.shape[0]).unsqueeze(1)
                kpos = torch.arange(j, j+kj.shape[0]).unsqueeze(0)
                Sij = Sij.masked_fill(kpos > qpos, -float('inf'))

            # ---- online softmax statistics ----
            mij = Sij.max(dim=1).values                  # block rowmax
            mi_new = torch.maximum(mi, mij)
            Pij = torch.exp(Sij - mi_new.unsqueeze(1))   # (Br, Bc)
            alpha = torch.exp(mi - mi_new)               # correction factor

            # rescale running denom and accumulator by alpha
            li = alpha * li + Pij.sum(dim=1)

            # ---- matmul 2: weighted values, accumulated UNNORMALIZED ----
            Oi = alpha.unsqueeze(1) * Oi + Pij @ vj      # (Br, d)

            mi = mi_new

        # single normalization at the very end (FA2's deferred division)
        O[i:i+qi.shape[0]] = Oi / li.unsqueeze(1)
    return O


torch.manual_seed(0)
N, d = 256, 64
Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
ref = torch.softmax(Q @ K.T / d**0.5, dim=-1) @ V
out = flash_attention_2_reference(Q, K, V)
err0 = (ref - out).abs().max().item()
print("block#0 max abs error vs dense softmax:", err0)
assert err0 < 1e-4, f"flash_attention_2_reference diverged from dense softmax: {err0}"

# also exercise the causal branch, since it is otherwise never taken above
out_causal = flash_attention_2_reference(Q, K, V, Br=64, Bc=64, causal=True)
causal_mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
scores = (Q @ K.T / d**0.5).masked_fill(~causal_mask, -float('inf'))
ref_causal = torch.softmax(scores, dim=-1) @ V
err0c = (ref_causal - out_causal).abs().max().item()
print("block#0 causal max abs error vs dense softmax:", err0c)
assert err0c < 1e-4, f"flash_attention_2_reference causal path diverged: {err0c}"


# ---------------------------------------------------------------------------
# Block #6 (line ~251): fa3_pipeline_model — producer/consumer ring-buffer
# dataflow model, verified against dense softmax attention.
# ---------------------------------------------------------------------------


def fa3_pipeline_model(Q, K, V, Bc=64, n_stages=2):
    """
    A *dataflow model* of FA3's producer/consumer pipeline for one query block.
    It does NOT use real async hardware; it shows the ordering:
      - producer 'prefetches' up to n_stages KV tiles into a ring buffer
      - consumer pops a tile only after it is present, then does the GEMMs/softmax
    Result is exact attention for the single query block Q.
    """
    N, d = K.shape
    ring = deque()                      # the shared-memory ring buffer of (Kj, Vj)
    Oi = torch.zeros(Q.shape[0], d)     # unnormalized accumulator
    mi = torch.full((Q.shape[0],), -float('inf'))
    li = torch.zeros(Q.shape[0])

    key_blocks = [(j, K[j:j+Bc], V[j:j+Bc]) for j in range(0, N, Bc)]
    it = iter(key_blocks)

    def produce():                      # PRODUCER: issue one "TMA load"
        try:
            ring.append(next(it))       # copy a KV tile into the ring
            return True
        except StopIteration:
            return False

    # prime the pipeline: producer runs ahead by n_stages
    for _ in range(n_stages):
        produce()

    while ring:                         # CONSUMER drains the ring
        j, kj, vj = ring.popleft()      # wait-for-tile then take it
        produce()                       # producer refills behind the consumer

        Sij = (Q @ kj.T) / (d ** 0.5)          # wgmma #1 (async on real HW)
        mij = Sij.max(dim=1).values
        mi_new = torch.maximum(mi, mij)
        Pij = torch.exp(Sij - mi_new.unsqueeze(1))   # softmax on SFU (overlapped)
        alpha = torch.exp(mi - mi_new)
        li = alpha * li + Pij.sum(dim=1)
        Oi = alpha.unsqueeze(1) * Oi + Pij @ vj      # wgmma #2 (async on real HW)
        mi = mi_new

    return Oi / li.unsqueeze(1)


torch.manual_seed(1)
N6, d6 = 512, 64
Q6 = torch.randn(128, d6); K6 = torch.randn(N6, d6); V6 = torch.randn(N6, d6)
ref6 = torch.softmax(Q6 @ K6.T / d6**0.5, dim=-1) @ V6
out6 = fa3_pipeline_model(Q6, K6, V6)
err6 = (ref6 - out6).abs().max().item()
print("block#6 max abs error:", err6)
assert err6 < 1e-4, f"fa3_pipeline_model diverged from dense softmax: {err6}"


# ---------------------------------------------------------------------------
# Block #7 (line ~336): hadamard / incoherent_rotation / fake_fp8_quant —
# incoherent processing reduces FP8 quantization error under an outlier.
# ---------------------------------------------------------------------------


def hadamard(n):
    """Build the n x n (Sylvester) Hadamard matrix; n must be a power of 2."""
    assert n & (n - 1) == 0, "n must be a power of two"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H,  H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / (H.shape[0] ** 0.5)            # orthonormal: H^T H = I

def incoherent_rotation(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    H = hadamard(d)
    D = torch.where(torch.rand(d, generator=g) > 0.5, 1.0, -1.0)  # random +-1 signs
    return H @ torch.diag(D)                   # M = H D, orthogonal

def fake_fp8_quant(x):
    """Crude e4m3-like quantization: per-tensor scale, ~3 mantissa bits."""
    s = x.abs().max() / 448.0                  # 448 = max finite e4m3 magnitude
    if s == 0:
        s = torch.tensor(1.0)
    xs = x / s                                 # bring values into FP8 range
    # simulate ~3 mantissa bits by rounding to 2^-3 relative steps
    # (illustrative, not a bit-exact e4m3 emulator)
    mant = torch.round(xs * 8) / 8
    return mant * s

torch.manual_seed(0)
d7 = 128
Q7 = torch.randn(64, d7); K7 = torch.randn(64, d7)
Q7[:, 7] *= 25.0; K7[:, 7] *= 25.0           # inject an outlier feature
ref7 = Q7 @ K7.T

# --- FP8 WITHOUT incoherent processing ---
err_plain = (fake_fp8_quant(Q7) @ fake_fp8_quant(K7).T - ref7).abs().mean()

# --- FP8 WITH incoherent processing (rotate, quantize, scores invariant) ---
M7 = incoherent_rotation(d7)
Qr, Kr = Q7 @ M7, K7 @ M7                       # (QM)(KM)^T == QK^T exactly
err_rot = (fake_fp8_quant(Qr) @ fake_fp8_quant(Kr).T - ref7).abs().mean()

print(f"block#7 mean score error  plain FP8 : {err_plain:.4f}")
print(f"block#7 mean score error  rotated   : {err_rot:.4f}")
# the rotated version has substantially lower error because the
# outlier in feature 7 is spread across all d coordinates
assert err_rot < err_plain, (
    f"incoherent processing should reduce FP8 error under an outlier: "
    f"plain={err_plain:.4f} rotated={err_rot:.4f}"
)

# The rotation must also leave the pre-quantization scores exactly invariant
# ((QM)(KM)^T == QK^T), which is the whole premise of the technique.
rot_only_scores = Qr @ Kr.T
assert torch.allclose(rot_only_scores, ref7, atol=1e-4), (
    "Hadamard rotation should leave QK^T unchanged before quantization"
)


# ---------------------------------------------------------------------------
# Block #8 (line ~409): xformers memory_efficient_attention.
# SKIP(dependency): xformers is not in the guaranteed CI import set and is
# not installed in this environment; the block's own logic is only executed
# if the import actually succeeded above.
# ---------------------------------------------------------------------------

if memory_efficient_attention is not None:
    batch, seqlen, num_heads, head_dim = 1, 32, 2, 16
    q8 = torch.randn(batch, seqlen, num_heads, head_dim)
    k8 = torch.randn(batch, seqlen, num_heads, head_dim)
    v8 = torch.randn(batch, seqlen, num_heads, head_dim)
    out8 = memory_efficient_attention(q8, k8, v8, attn_bias=LowerTriangularMask())
    print("block#8 xformers output shape:", tuple(out8.shape))
    assert out8.shape == (batch, seqlen, num_heads, head_dim)
else:
    print("block#8 SKIP(dependency): xformers not installed, skipping "
          "memory_efficient_attention call.")


print("All executable blocks in 04-kernels-efficiency/03-flash-attention-2-3.md ran successfully.")
