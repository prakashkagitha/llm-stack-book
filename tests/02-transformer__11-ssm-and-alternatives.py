"""
Runs the CPU-runnable Python blocks from
content/02-transformer/11-ssm-and-alternatives.md, concatenated in order so
that later blocks can rely on names defined by earlier ones (as they do in
the chapter itself). Each block is copied verbatim from the chapter; only
minimal glue needed to make blocks that only *define* something (a function
or class) actually execute has been added, and is clearly marked "GLUE".

Blocks covered (9 CPU-runnable blocks):
  #0  (line ~56)  - sliding_window_causal_mask
  #1  (line ~76)  - FlexAttention sliding_window_mod + create_block_mask
  #4  (line ~199) - elu_feature_map, linear_attention_parallel, recurrent step
  #6  (line ~309) - __main__ smoke test: parallel/recurrent/chunked agree
  #8  (line ~462) - SelectiveSSM, MambaBlock
  #9  (line ~590) - __main__ smoke test: SelectiveSSM/MambaBlock + closed form
  #10 (line ~652) - associative_scan + __main__ smoke test
  #11 (line ~748) - RWKVTimeMixing (pure-PyTorch WKV scan; runs on CPU)
  #12 (line ~918) - HybridBlock, HybridLanguageModel + sanity check

Blocks explicitly SKIPPED (per task spec / hard rules):
  #2  (line ~98)  - SKIP(fragment): RollingKVCache is a standalone class
                    fragment demonstrating a KV-cache eviction API; it is
                    never wired into a full attention loop in the chapter,
                    so instantiating it in isolation would not exercise any
                    interesting logic beyond a slice-and-cat, and the task
                    spec marks it "fragment" (default skip).
  #3  (line ~130) - SKIP(non-python): ```text ascii-art mask diagram.
  #5  (line ~273) - SKIP(fragment): linear_attention_chunked is defined here
                    but only exercised by block #6's __main__ test below
                    (which we DO run), so its logic is in fact executed as
                    part of block #6 — we still define it as part of that
                    block's dependency chain rather than re-declaring it a
                    separate tested block, per the task's block numbering.
  #7  (line ~433) - SKIP(non-python): ```text Mamba block-diagram fragment.

Note on block #5: the chapter's own __main__ test in block #6 (line ~343-354)
calls `linear_attention_chunked`, so to run block #6 faithfully we must also
define block #5's function. We include its body as glue-dependency (verbatim,
unmodified) immediately before block #6, and do not claim it as a separately
"tested" block on its own -- it is exercised only via block #6's assertions.
"""

import torch

# ======================================================================
# Block #0 (line ~56): sliding_window_causal_mask
# ======================================================================
print("=" * 70)
print("Block #0 (line ~56): sliding_window_causal_mask")
print("=" * 70)

# --- verbatim from the chapter ---
import torch

def sliding_window_causal_mask(seq_len: int, window: int, device="cpu") -> torch.Tensor:
    """
    Additive attention mask for causal sliding-window attention.
    Query i may attend key j  iff  j <= i (causal)  AND  i - j < window (local).
    Returns (seq_len, seq_len) with 0 where allowed and -inf where masked;
    add it to QK^T / sqrt(d_k) before the softmax.
    """
    i = torch.arange(seq_len, device=device)[:, None]   # (N, 1)
    j = torch.arange(seq_len, device=device)[None, :]    # (1, N)
    allowed = (j <= i) & (i - j < window)                # banded lower-triangular
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask
# --- end verbatim ---

# GLUE: actually call it and sanity-check the banded structure.
mask = sliding_window_causal_mask(seq_len=8, window=3)
assert mask.shape == (8, 8)
# Diagonal always allowed (i - j == 0 < window, j == i <= i).
assert torch.all(mask.diagonal() == 0.0)
# Future positions always masked.
assert mask[0, 1].item() == float("-inf")
# Position exactly at the window edge is masked; one inside is allowed.
assert mask[5, 2].item() == float("-inf")   # i-j = 3, not < window=3
assert mask[5, 3].item() == 0.0             # i-j = 2, < window=3
print("sliding_window_causal_mask: OK, shape", tuple(mask.shape))


print()
print("=" * 70)
print("Block #1 (line ~76): FlexAttention sliding_window_mod + create_block_mask")
print("=" * 70)

# --- verbatim from the chapter ---
# PyTorch 2.5+: FlexAttention skips fully-masked 128x128 blocks for a real speedup.
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def sliding_window_mod(window: int):
    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        local  = q_idx - kv_idx < window
        return causal & local
    return mask_mod

# Build the block mask once, reuse across the forward pass:
block_mask = create_block_mask(
    sliding_window_mod(window=1024),
    B=None, H=None, Q_LEN=8192, KV_LEN=8192,
)
# out = flex_attention(q, k, v, block_mask=block_mask)   # q,k,v: (B, H, N, d_k)
# --- end verbatim ---

assert block_mask is not None
assert tuple(block_mask.shape) == (1, 1, 8192, 8192)
print("FlexAttention block_mask built: OK, shape", tuple(block_mask.shape))

# GLUE: also actually run flex_attention end-to-end on tiny tensors (the
# book leaves the call commented out as a usage note; we execute it here
# with a small block_mask sized to match small q/k/v so the block's logic
# is genuinely exercised, not just defined).
small_block_mask = create_block_mask(
    sliding_window_mod(window=4),
    B=None, H=None, Q_LEN=16, KV_LEN=16,
    device="cpu",  # GLUE: create_block_mask defaults to CUDA if available;
                   # force CPU to match the CPU q/k/v tensors below.
)
Bq, Hq, Nq, dk = 1, 2, 16, 8
q = torch.randn(Bq, Hq, Nq, dk)
k = torch.randn(Bq, Hq, Nq, dk)
v = torch.randn(Bq, Hq, Nq, dk)
out = flex_attention(q, k, v, block_mask=small_block_mask)
assert out.shape == (Bq, Hq, Nq, dk)
print("flex_attention forward: OK, out shape", tuple(out.shape))


print()
print("=" * 70)
print("Block #4 (line ~199): elu_feature_map, linear_attention_parallel, "
      "linear_attention_recurrent_step")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn.functional as F

def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map: keeps positivity, cheap to compute."""
    return F.elu(x) + 1.0  # shape preserved; always > 0

def linear_attention_parallel(
    Q: torch.Tensor,  # (B, N, H, d_k)
    K: torch.Tensor,  # (B, N, H, d_k)
    V: torch.Tensor,  # (B, N, H, d_v)
    causal: bool = True,
) -> torch.Tensor:
    """
    Parallel (training) form of linear attention.
    For causal (autoregressive) models we cannot simply do K^T V first —
    we need the cumulative version. Here we show the non-causal form for clarity.
    The causal form requires a cumulative outer-product scan.
    """
    B, N, H, d_k = Q.shape
    d_v = V.shape[-1]

    phi_Q = elu_feature_map(Q)  # (B, N, H, d_k)
    phi_K = elu_feature_map(K)  # (B, N, H, d_k)

    # Compute K^T V: (B, H, d_k, d_v)
    # phi_K: (B, N, H, d_k) -> reshape to (B, H, N, d_k)
    phi_K_t = phi_K.permute(0, 2, 1, 3)  # (B, H, N, d_k)
    V_t     = V.permute(0, 2, 1, 3)       # (B, H, N, d_v)
    KV      = torch.einsum('bhnk,bhnv->bhkv', phi_K_t, V_t)  # (B, H, d_k, d_v)

    # Normalizer: sum of phi_K over N
    z = phi_K_t.sum(dim=2)  # (B, H, d_k)

    # Compute output
    phi_Q_t = phi_Q.permute(0, 2, 1, 3)  # (B, H, N, d_k)
    # Numerator: phi(Q) @ KV -> (B, H, N, d_v)
    num = torch.einsum('bhnk,bhkv->bhnv', phi_Q_t, KV)
    # Denominator: phi(Q) @ z -> (B, H, N)
    den = torch.einsum('bhnk,bhk->bhn', phi_Q_t, z).unsqueeze(-1)
    # Clamp for numerical stability
    den = den.clamp(min=1e-6)

    out = (num / den).permute(0, 2, 1, 3)  # back to (B, N, H, d_v)
    return out


def linear_attention_recurrent_step(
    q_t: torch.Tensor,  # (B, H, d_k) — current query
    k_t: torch.Tensor,  # (B, H, d_k) — current key
    v_t: torch.Tensor,  # (B, H, d_v) — current value
    S:   torch.Tensor,  # (B, H, d_k, d_v) — running state
    z:   torch.Tensor,  # (B, H, d_k)       — running normalizer
):
    """Single-step recurrent update — O(d_k * d_v) per step, O(d_k * d_v) memory."""
    phi_q = elu_feature_map(q_t)  # (B, H, d_k)
    phi_k = elu_feature_map(k_t)  # (B, H, d_k)

    # Update state: S += phi_k outer v
    S = S + torch.einsum('bhk,bhv->bhkv', phi_k, v_t)
    z = z + phi_k  # (B, H, d_k)

    # Output
    num = torch.einsum('bhk,bhkv->bhv', phi_q, S)   # (B, H, d_v)
    den = torch.einsum('bhk,bhk->bh', phi_q, z).unsqueeze(-1).clamp(min=1e-6)
    return num / den, S, z
# --- end verbatim ---

# GLUE: call both functions on tiny tensors so the block actually executes
# (block #6 below also exercises them via the chapter's own __main__ tests,
# but we call them here too to confirm the definitions work standalone).
_B, _N, _H, _dk, _dv = 1, 4, 2, 3, 3
_Q = torch.randn(_B, _N, _H, _dk)
_K = torch.randn(_B, _N, _H, _dk)
_V = torch.randn(_B, _N, _H, _dv)
_out = linear_attention_parallel(_Q, _K, _V, causal=False)
assert _out.shape == (_B, _N, _H, _dv)
print("linear_attention_parallel: OK, out shape", tuple(_out.shape))

_S0 = torch.zeros(_B, _H, _dk, _dv)
_z0 = torch.zeros(_B, _H, _dk)
_y, _S1, _z1 = linear_attention_recurrent_step(_Q[:, 0], _K[:, 0], _V[:, 0], _S0, _z0)
assert _y.shape == (_B, _H, _dv)
print("linear_attention_recurrent_step: OK, y shape", tuple(_y.shape))


# ----------------------------------------------------------------------
# Dependency for block #6: linear_attention_chunked (chapter block at
# line ~273, marked "fragment"/#5 in the task's skip list). The chapter's
# own __main__ test in block #6 calls this function, so we define it here,
# verbatim, purely as glue so block #6 can run as written.
# ----------------------------------------------------------------------
# --- verbatim from the chapter (line ~273) ---
def linear_attention_chunked(
    Q: torch.Tensor,  # (B, H, N, d_k)  -- head-major here for clean einsums
    K: torch.Tensor,  # (B, H, N, d_k)
    V: torch.Tensor,  # (B, H, N, d_v)
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunked *causal* linear attention (unnormalized numerator).
    Assumes N is divisible by chunk_size. Cost: O(N * C * d) intra-chunk
    matmuls + O((N/C) * d_k * d_v) state carries -- fully parallel within a chunk.
    Divide by the same scan run with V replaced by ones to get the normalizer.
    """
    B, H, N, d_k = Q.shape
    d_v = V.shape[-1]
    phi_Q = elu_feature_map(Q)
    phi_K = elu_feature_map(K)
    n_chunks = N // chunk_size
    phi_Q = phi_Q.view(B, H, n_chunks, chunk_size, d_k)
    phi_K = phi_K.view(B, H, n_chunks, chunk_size, d_k)
    Vc    = V.view(B, H, n_chunks, chunk_size, d_v)

    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=Q.device))  # intra-chunk causal
    S = torch.zeros(B, H, d_k, d_v, device=Q.device, dtype=Q.dtype)         # inter-chunk state
    outs = []
    for c in range(n_chunks):
        q, k, v = phi_Q[:, :, c], phi_K[:, :, c], Vc[:, :, c]   # (B, H, C, *)
        inter = torch.einsum('bhck,bhkv->bhcv', q, S)           # from all previous chunks
        A     = torch.einsum('bhck,bhdk->bhcd', q, k) * mask    # masked within-chunk scores
        intra = torch.einsum('bhcd,bhdv->bhcv', A, v)           # within-chunk contribution
        outs.append(inter + intra)
        S = S + torch.einsum('bhck,bhcv->bhkv', k, v)           # fold chunk into running state
    return torch.cat(outs, dim=2)                                # (B, H, N, d_v)
# --- end verbatim ---


print()
print("=" * 70)
print("Block #6 (line ~309): __main__ smoke test - parallel/recurrent/chunked agree")
print("=" * 70)

# --- verbatim from the chapter (was guarded by `if __name__ == "__main__":`;
# unwrapped here since this whole file IS the main module) ---
torch.manual_seed(0)

# (a) Non-causal parallel form matches an explicit softmax-free reference.
B, N, H, d_k, d_v = 2, 6, 2, 4, 5
Q = torch.randn(B, N, H, d_k); K = torch.randn(B, N, H, d_k); V = torch.randn(B, N, H, d_v)
pQ, pK = elu_feature_map(Q), elu_feature_map(K)
scores = torch.einsum('bnhk,bmhk->bhnm', pQ, pK)
scores = scores / scores.sum(-1, keepdim=True).clamp(min=1e-6)
ref = torch.einsum('bhnm,bmhv->bnhv', scores, V)
assert torch.allclose(linear_attention_parallel(Q, K, V, causal=False), ref, atol=1e-5)
print("non-causal parallel form matches reference: OK")

# (b) Recurrent step, rolled over t, reproduces the *causal* output.
Sr = torch.zeros(B, H, d_k, d_v); zr = torch.zeros(B, H, d_k); rec = []
for t in range(N):
    y, Sr, zr = linear_attention_recurrent_step(Q[:, t], K[:, t], V[:, t], Sr, zr)
    rec.append(y)
rec = torch.stack(rec, dim=1)   # (B, N, H, d_v)

def causal_reference(Q, K, V):
    B, N, H, d_k = Q.shape; d_v = V.shape[-1]
    pQ, pK = elu_feature_map(Q), elu_feature_map(K)
    out = torch.zeros(B, N, H, d_v)
    for t in range(N):
        s = torch.einsum('bhk,bihk->bih', pQ[:, t], pK[:, :t + 1])   # (B, t+1, H)
        num = torch.einsum('bih,bihv->bhv', s, V[:, :t + 1])
        den = s.sum(1).clamp(min=1e-6)                               # (B, H)
        out[:, t] = num / den.unsqueeze(-1)
    return out
assert torch.allclose(rec, causal_reference(Q, K, V), atol=1e-5)
print("recurrent rollout reproduces causal output: OK")

# (c) Chunked numerator == recurrent (unnormalized) numerator.
B, N, H, d_k, d_v = 2, 32, 2, 4, 5
Q = torch.randn(B, N, H, d_k); K = torch.randn(B, N, H, d_k); V = torch.randn(B, N, H, d_v)
Qh, Kh, Vh = [t.permute(0, 2, 1, 3).contiguous() for t in (Q, K, V)]   # (B, H, N, *)
chk = linear_attention_chunked(Qh, Kh, Vh, chunk_size=8)
S = torch.zeros(B, H, d_k, d_v); num = []
for t in range(N):
    pk = elu_feature_map(Kh[:, :, t]); pq = elu_feature_map(Qh[:, :, t])
    S = S + torch.einsum('bhk,bhv->bhkv', pk, Vh[:, :, t])
    num.append(torch.einsum('bhk,bhkv->bhv', pq, S))
assert torch.allclose(chk, torch.stack(num, dim=2), atol=1e-4)
print("chunked numerator matches recurrent numerator: OK")
# --- end verbatim ---


print()
print("=" * 70)
print("Block #8 (line ~462): SelectiveSSM, MambaBlock")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn as nn
import math

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style selective SSM for one feature dimension.
    Full Mamba also uses depthwise conv before this and expand*D channels;
    this illustrates the core selective scan logic.

    d_model: input/output dimension
    d_state: SSM state dimension N (Mamba uses 16)
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank = dt_rank or math.ceil(d_model / 16)

        # Log of A diagonal: initialized to evenly spaced negative reals
        # A = -exp(log_A), kept negative for stability
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                          .unsqueeze(0).repeat(d_model, 1))
        self.A_log = nn.Parameter(A_log)  # (d_model, d_state)

        # D skip connection (one per channel)
        self.D = nn.Parameter(torch.ones(d_model))

        # Projections for input-dependent parameters
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        B, L, D = x.shape
        N = self.d_state

        # A: (D, N), kept negative for stability
        A = -torch.exp(self.A_log)  # (D, N)

        # Compute input-dependent B, C, delta
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt, B_ssm, C = x_dbl.split([self.dt_proj.in_features, N, N], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt))  # (B, L, D), positive

        # Discretize: zero-order hold
        # dA[t] = exp(dt[t] * A)  — shape (B, L, D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,D,N)
        # dB[t] = dt[t] * B[t] (simplified ZOH for diagonal A)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, D, N)

        # Sequential scan (simple version — production uses parallel scan)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # h: (B, D, N), dA[:,t]: (B, D, N), dB[:,t]: (B, D, N)
            # x[:,t]: (B, D)
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # y: (B, D) via C projection
            y = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y)

        # Stack and add skip connection
        y_seq = torch.stack(ys, dim=1)  # (B, L, D)
        y_seq = y_seq + x * self.D.unsqueeze(0).unsqueeze(0)
        return y_seq


class MambaBlock(nn.Module):
    """
    Full Mamba block with expansion, depthwise conv, gating, and SSM.
    expand: expansion factor (Mamba uses 2)
    d_conv: depthwise conv width (Mamba uses 4)
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # Input projection: split into SSM branch (d_inner) and gate (d_inner)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise convolution over sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,          # left-pad for causality
            groups=self.d_inner,         # depthwise
            bias=True,
        )
        self.act = nn.SiLU()

        # SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape

        # Split into SSM branch u and gate z
        xz = self.in_proj(x)                   # (B, L, 2*d_inner)
        u, z = xz.chunk(2, dim=-1)             # each (B, L, d_inner)

        # Depthwise conv (operates on L dimension)
        u = u.transpose(1, 2)                  # (B, d_inner, L)
        u = self.conv1d(u)[..., :L]            # causal: trim right
        u = u.transpose(1, 2)                  # (B, L, d_inner)
        u = self.act(u)

        # SSM
        y = self.ssm(u)                        # (B, L, d_inner)

        # Gate: multiply by SiLU(z)
        y = y * self.act(z)

        # Output projection
        return self.out_proj(y)                # (B, L, d_model)
# --- end verbatim ---
print("SelectiveSSM, MambaBlock defined: OK (instantiated/called in block #9 below)")


print()
print("=" * 70)
print("Block #9 (line ~590): __main__ smoke test - SelectiveSSM/MambaBlock "
      "+ closed form")
print("=" * 70)

# --- verbatim from the chapter (was guarded by `if __name__ == "__main__":`;
# unwrapped here since this whole file IS the main module) ---
torch.manual_seed(0)
B, L, d_model = 2, 16, 32
x = torch.randn(B, L, d_model)

ssm = SelectiveSSM(d_model, d_state=8)       # note d_state != d_model on purpose
y = ssm(x)
print("SelectiveSSM:", tuple(x.shape), "->", tuple(y.shape))   # (2,16,32) -> (2,16,32)
assert y.shape == x.shape

block = MambaBlock(d_model, d_state=8)
y = block(x)
print("MambaBlock:  ", tuple(x.shape), "->", tuple(y.shape))    # (2,16,32) -> (2,16,32)
assert y.shape == x.shape

# Selective scan vs closed form:
#   h_t = sum_{i<=t} (prod_{j=i+1}^{t} dA_j) * dB_i * x_i,   y_t = sum_n C[t,n] * h[t,n]
torch.manual_seed(1)
B, L, D, N = 1, 5, 2, 3
dA = torch.rand(B, L, D, N) * 0.5 + 0.4      # per-step decays in (0.4, 0.9)
dB = torch.randn(B, L, D, N)
xin = torch.randn(B, L, D)
C = torch.randn(B, L, N)

# (i) sequential scan -- the exact loop used inside SelectiveSSM.forward
h = torch.zeros(B, D, N); seq = []
for t in range(L):
    h = dA[:, t] * h + dB[:, t] * xin[:, t].unsqueeze(-1)
    seq.append((h * C[:, t].unsqueeze(1)).sum(-1))
seq = torch.stack(seq, dim=1)

# (ii) cumulative-product closed form
clo = []
for t in range(L):
    h = torch.zeros(B, D, N)
    for i in range(t + 1):
        prod = torch.ones(B, D, N)
        for j in range(i + 1, t + 1):
            prod = prod * dA[:, j]
        h = h + prod * dB[:, i] * xin[:, i].unsqueeze(-1)
    clo.append((h * C[:, t].unsqueeze(1)).sum(-1))
clo = torch.stack(clo, dim=1)

assert torch.allclose(seq, clo, atol=1e-5)
print("selective scan matches cumulative-product closed form: OK")
# --- end verbatim ---


print()
print("=" * 70)
print("Block #10 (line ~652): associative_scan + __main__ smoke test")
print("=" * 70)

# --- verbatim from the chapter ---
import torch

def associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel (Hillis-Steele) inclusive scan of the affine recurrence
        h_t = a_t * h_{t-1} + b_t,   h_{-1} = 0
    a, b: (L, ...) with matching trailing shape, e.g. (L, D, N).
    Runs in O(log L) doubling steps; returns h stacked along dim 0.
    Combine (earlier=(a_e,b_e) then later=(a_l,b_l)) -> (a_l*a_e, a_l*b_e + b_l).
    """
    L = a.shape[0]
    a = a.clone(); b = b.clone()
    d = 1
    while d < L:
        a_prev, b_prev = a[:-d], b[:-d]        # partials ending d steps earlier
        a_new, b_new = a.clone(), b.clone()
        a_new[d:] = a[d:] * a_prev
        b_new[d:] = a[d:] * b_prev + b[d:]
        a, b = a_new, b_new
        d *= 2
    return b   # h_t == accumulated b_t because h_{-1} = 0

torch.manual_seed(0)
L, D, N = 128, 4, 3
a = torch.rand(L, D, N) * 0.9              # per-step decays in (0, 0.9)
b = torch.randn(L, D, N)

par = associative_scan(a, b)

h = torch.zeros(D, N); ref = []            # ground-truth sequential loop
for t in range(L):
    h = a[t] * h + b[t]
    ref.append(h.clone())
ref = torch.stack(ref, dim=0)

assert torch.allclose(par, ref, atol=1e-4)
print("parallel associative scan matches sequential loop: OK")
# --- end verbatim ---


print()
print("=" * 70)
print("Block #11 (line ~748): RWKVTimeMixing")
print("=" * 70)

# --- verbatim from the chapter (line ~748) ---
import torch
import torch.nn as nn

class RWKVTimeMixing(nn.Module):
    """
    RWKV-4 style time mixing block.
    n_embd: model dimension
    layer_id: used to initialize w differently per layer
    """
    def __init__(self, n_embd: int, layer_id: int = 0):
        super().__init__()
        self.n_embd = n_embd

        # Learned time decay (w): initialized to negative values (decay)
        # Stored as log(-w) for stability; w should be < 0
        self.w = nn.Parameter(torch.zeros(n_embd))
        nn.init.uniform_(self.w, -8.0, -0.5)

        # Time-first (u): bonus for current token
        self.u = nn.Parameter(torch.zeros(n_embd))
        nn.init.uniform_(self.u, -0.5, 0.5)

        # Token shift mixing ratios
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd))

        # Projections
        self.key    = nn.Linear(n_embd, n_embd, bias=False)
        self.value  = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_embd)
        Token shift: mix current and previous token for k, v, r inputs.
        Then run the WKV scan.
        """
        B, T, C = x.shape

        # Token shift: concat zero-padded version of x (shifted right by 1)
        x_prev = torch.cat([torch.zeros(B, 1, C, device=x.device), x[:, :-1]], dim=1)

        # Compute K, V, R with time-mixed inputs
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)        # (B, T, C)
        v = self.value(xv)      # (B, T, C)
        r = torch.sigmoid(self.receptance(xr))  # gating

        # WKV computation (simplified sequential loop over T)
        # In practice this is a custom CUDA kernel for speed
        w = -torch.exp(self.w)   # (C,), negative → exponential decay
        u = self.u               # (C,)

        # Running accumulators
        aa = torch.zeros(B, C, device=x.device)  # numerator state
        bb = torch.zeros(B, C, device=x.device)  # denominator state
        pp = torch.full((B, C), -1e38, device=x.device)  # log-sum for numerical stability

        wkv_outputs = []
        for t in range(T):
            kt = k[:, t]  # (B, C)
            vt = v[:, t]  # (B, C)

            # Numerically stable WKV update using log-space
            # Output uses the *undecayed* running state a_{t-1}, b_{t-1};
            # the decay w is applied only in the state update below.
            # (Matches the displayed WKV recurrence and the official RWKV-4 kernel.)
            qq = torch.maximum(pp, kt + u)         # (B, C)
            a  = torch.exp(pp - qq) * aa + torch.exp(kt + u - qq) * vt
            b  = torch.exp(pp - qq) * bb + torch.exp(kt + u - qq)
            wkv_outputs.append(a / b.clamp(min=1e-8))

            # Update state
            qq2 = torch.maximum(pp + w, kt)
            aa  = torch.exp(pp + w - qq2) * aa + torch.exp(kt - qq2) * vt
            bb  = torch.exp(pp + w - qq2) * bb + torch.exp(kt - qq2)
            pp  = qq2

        wkv = torch.stack(wkv_outputs, dim=1)  # (B, T, C)
        out = r * wkv
        return self.output(out)
# --- end verbatim ---

# GLUE: instantiate and run the RWKV-4 time-mixing block on CPU (the chapter's
# "custom CUDA kernel" note is only a production speedup remark; the shown code
# is pure-PyTorch and runs on CPU). Also cross-check the log-space WKV scan
# against a direct (non-log) evaluation of the displayed recurrence to confirm
# the numerically-stable formulation is faithful.
torch.manual_seed(0)
_rwkv = RWKVTimeMixing(n_embd=16)
_x = torch.randn(2, 8, 16)
_y = _rwkv(_x)
assert _y.shape == _x.shape

# Direct evaluation of the displayed recurrence:
#   wkv_t = (e^{u+k_t} v_t + a_{t-1}) / (e^{u+k_t} + b_{t-1})
#   a_t   = e^{w} a_{t-1} + e^{k_t} v_t,   b_t = e^{w} b_{t-1} + e^{k_t}
_w = -torch.exp(_rwkv.w); _u = _rwkv.u
_xp = torch.cat([torch.zeros(2, 1, 16), _x[:, :-1]], dim=1)
_k = _rwkv.key(_x * _rwkv.time_mix_k + _xp * (1 - _rwkv.time_mix_k))
_v = _rwkv.value(_x * _rwkv.time_mix_v + _xp * (1 - _rwkv.time_mix_v))
_B, _T, _C = _x.shape
_aa = torch.zeros(_B, _C); _bb = torch.zeros(_B, _C); _ref = []
for _t in range(_T):
    _kt, _vt = _k[:, _t], _v[:, _t]
    _num = torch.exp(_u + _kt) * _vt + _aa
    _den = torch.exp(_u + _kt) + _bb
    _ref.append(_num / _den.clamp(min=1e-8))
    _aa = torch.exp(_w) * _aa + torch.exp(_kt) * _vt
    _bb = torch.exp(_w) * _bb + torch.exp(_kt)
_ref = torch.stack(_ref, dim=1)
_r = torch.sigmoid(_rwkv.receptance(_x * _rwkv.time_mix_r + _xp * (1 - _rwkv.time_mix_r)))
_ref_out = _rwkv.output(_r * _ref)
assert torch.allclose(_y, _ref_out, atol=1e-4)
print("RWKVTimeMixing: OK, log-space WKV matches direct recurrence, out shape",
      tuple(_y.shape))


print()
print("=" * 70)
print("Block #12 (line ~918): HybridBlock, HybridLanguageModel + sanity check")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn as nn
from typing import Literal

# We assume MambaBlock and a standard TransformerBlock are already defined.
# (MambaBlock is defined above in block #8, exactly as the chapter intends
# via its "already defined" comment.)
# Below shows how to compose them in a hybrid model.

class HybridBlock(nn.Module):
    """
    A single hybrid layer that is either an SSM block or an Attention block.
    block_type: 'mamba' | 'attention'
    """
    def __init__(
        self,
        d_model: int,
        block_type: Literal["mamba", "attention"],
        n_heads: int = 8,
        d_state: int = 16,
    ):
        super().__init__()
        self.block_type = block_type
        self.norm = nn.RMSNorm(d_model)

        if block_type == "mamba":
            self.block = MambaBlock(d_model, d_state=d_state)
        elif block_type == "attention":
            # Minimal multi-head self-attention
            self.block = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.block_type == "mamba":
            x = self.block(x)
        else:
            x, _ = self.block(x, x, x, attn_mask=attn_mask)
        return residual + x


class HybridLanguageModel(nn.Module):
    """
    A hybrid SSM-Attention language model.
    attn_every: place an attention layer every N layers (rest are Mamba).
    E.g. attn_every=8 → 1 attention layer per 7 Mamba layers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        attn_every: int = 8,
        n_heads: int = 8,
        d_state: int = 16,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        layers = []
        for i in range(n_layers):
            # Place attention layers at fixed intervals
            if (i + 1) % attn_every == 0:
                block_type = "attention"
            else:
                block_type = "mamba"
            layers.append(
                HybridBlock(d_model, block_type, n_heads=n_heads, d_state=d_state)
            )

        self.layers = nn.ModuleList(layers)
        self.norm_out = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) -> logits (B, L, vocab_size)"""
        x = self.embed(tokens)  # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return self.lm_head(x)


# Quick sanity check
# (chapter uses vocab_size=32000, d_model=512, n_layers=12; we keep these but
#  they run comfortably on CPU in well under a second)
model = HybridLanguageModel(
    vocab_size=32000, d_model=512, n_layers=12, attn_every=4
)
# Count attention vs Mamba layers
attn_count  = sum(1 for l in model.layers if l.block_type == "attention")
mamba_count = sum(1 for l in model.layers if l.block_type == "mamba")
print(f"Attention layers: {attn_count}, Mamba layers: {mamba_count}")
# → Attention layers: 3, Mamba layers: 9

tokens = torch.randint(0, 32000, (2, 128))
logits = model(tokens)
print(f"Output shape: {logits.shape}")  # (2, 128, 32000)
# --- end verbatim ---

assert attn_count == 3 and mamba_count == 9
assert logits.shape == (2, 128, 32000)
print("HybridLanguageModel forward: OK")


print()
print("=" * 70)
print("ALL BLOCKS PASSED")
print("=" * 70)
