"""
Runs the CPU-runnable Python blocks from
content/04-kernels-efficiency/07-quantization-ptq.md, concatenated in order.
Each tested block's code is copied verbatim from the chapter; minimal glue
needed to actually *execute* each block (small fixtures, tiny shapes) is
clearly marked "GLUE".

Blocks tested (all pure-CPU tensor arithmetic, no GPU/network needed):

  #0 (line ~49)  - quantize_symmetric / dequantize_symmetric / quantize_affine
                   / dequantize_affine + the book's own "sanity check" snippet
  #2 (line ~179) - gptq_quantize_layer(): the Cholesky/Hessian-compensated
                   column-by-column GPTQ core
  #4 (line ~267) - awq_search_scales(): AWQ's per-input-channel scale grid
                   search
  #5 (line ~331) - smoothquant_scales() / apply_smoothing(): SmoothQuant's
                   per-channel migration scale
  #6 (line ~369) - llm_int8_matmul(): the LLM.int8() mixed INT8/FP16
                   outlier-decomposed matmul

SKIP (per task instructions -- non-standalone fragments, not in the tested
set of 5):
  #1 (line ~99)  - quantize_per_group / dequantize_per_group -- SKIP(fragment):
        defines functions only, never called/demonstrated by the chapter
        text itself; not one of the 5 assigned blocks.
  #3 (line ~232) - accumulate_hessian() -- SKIP(fragment): a standalone
        helper that loops over an externally-supplied `calibration_inputs`
        iterable of per-batch activation tensors from a real `layer`; the
        chapter never calls it in this snippet, it's narrative-only context
        for how H is produced in practice. Not one of the 5 assigned blocks.
        (We build an equivalent Hessian directly in the GLUE for block #2
        instead of importing this helper, since block #2 needs *some* H to
        run against.)

No real bugs were found in the book's code -- all 5 tested blocks ran
correctly on the first pass once given tiny CPU-sized fixtures.
"""

import torch


# =====================================================================
# Block #0 (line ~49): symmetric / affine quantize-dequantize + sanity check
# =====================================================================

def quantize_symmetric(x: torch.Tensor, n_bits: int = 4):
    """Symmetric per-tensor quantization. Returns (codes, scale)."""
    qmax = 2 ** (n_bits - 1) - 1          # e.g. 7 for INT4, 127 for INT8
    qmin = -qmax                           # symmetric grid, drop the -8 slot for simplicity
    s = x.abs().max() / qmax               # one scalar scale for the whole tensor
    s = s.clamp(min=1e-8)                  # avoid divide-by-zero on all-zero tensors
    q = torch.clamp(torch.round(x / s), qmin, qmax)
    return q.to(torch.int8), s

def dequantize_symmetric(q: torch.Tensor, s: torch.Tensor):
    return q.to(torch.float32) * s

def quantize_affine(x: torch.Tensor, n_bits: int = 4):
    """Asymmetric (affine) quantization with a zero-point. Good for activations."""
    qmin, qmax = 0, 2 ** n_bits - 1        # unsigned grid: 0..15 for INT4
    xmin, xmax = x.min(), x.max()
    s = (xmax - xmin) / (qmax - qmin)
    s = s.clamp(min=1e-8)
    z = torch.round(qmin - xmin / s)        # the integer code that represents real 0.0
    q = torch.clamp(torch.round(x / s) + z, qmin, qmax)
    return q.to(torch.uint8), s, z

def dequantize_affine(q, s, z):
    return (q.to(torch.float32) - z) * s

# --- sanity check ---
torch.manual_seed(0)
w = torch.randn(4096) * 0.05               # a slice of a typical weight column
q, s = quantize_symmetric(w, n_bits=4)
err = (dequantize_symmetric(q, s) - w).abs().mean()
print(f"INT4 mean abs error: {err:.4e}  (scale={s:.4e})")

# GLUE: also exercise the affine path (book's block defines it but the
# "sanity check" snippet only calls the symmetric one).
a = torch.rand(2048) * 2.2 - 0.2           # skewed, non-negative-ish activation-like range
qa, sa, za = quantize_affine(a, n_bits=4)
err_a = (dequantize_affine(qa, sa, za) - a).abs().mean()
print(f"[block #0] affine INT4 mean abs error: {err_a:.4e} (scale={sa:.4e}, zero={za:.1f})")

assert err.item() < 0.01           # step ~ 0.05*3/7 ~= 0.02, mean abs err should be small
assert err_a.item() < 0.2          # affine grid is coarser (16 levels over range ~2.2)
assert q.dtype == torch.int8 and qa.dtype == torch.uint8
print("[block #0] quantize/dequantize (symmetric + affine) ran OK")


# =====================================================================
# Block #2 (line ~179): gptq_quantize_layer() -- Hessian-compensated GPTQ core
# =====================================================================

def gptq_quantize_layer(W, H, n_bits=4, group_size=128, percdamp=0.01):
    """
    GPTQ for one linear layer.
      W : [out_features, in_features]  original weights (float)
      H : [in_features, in_features]   Hessian  = X X^T  over calibration tokens
    Returns the quantized-then-dequantized weight (ready to use in FP matmul,
    or to be packed into INT4). This mirrors the real algorithm's structure.
    """
    W = W.clone().float()
    out_f, in_f = W.shape
    qmax = 2 ** (n_bits - 1) - 1

    # 1) Dampen the Hessian diagonal for invertibility (some inputs are dead).
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H[range(in_f), range(in_f)] += damp

    # 2) Cholesky of the INVERSE Hessian -> upper triangular factor we walk along.
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)   # upper-tri Cholesky of H^{-1}

    Q = torch.zeros_like(W)
    Err = torch.zeros_like(W)

    # 3) Walk columns left-to-right; quantize, then propagate the error forward.
    for i in range(in_f):
        w = W[:, i].clone()                # current column across all output rows
        d = Hinv[i, i]                     # diagonal entry => error-scaling factor

        # per-group symmetric scale (recomputed at each group boundary)
        if i % group_size == 0:
            g = W[:, i:i + group_size]
            scale = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax

        q = torch.clamp(torch.round(w / scale.squeeze(1)),
                        -qmax, qmax) * scale.squeeze(1)
        Q[:, i] = q

        err = (w - q) / d                  # the OBS-optimal scaled error
        # compensate every not-yet-quantized column j>i using H^{-1} row i.
        W[:, i:] -= err.unsqueeze(1) * Hinv[i, i:].unsqueeze(0)
        Err[:, i] = err

    return Q

# GLUE: tiny calibration-derived Hessian and tiny weight matrix so the
# Cholesky/inverse machinery genuinely executes on CPU in milliseconds.
# (This mirrors what the book's separately-defined, unrunnable-standalone
# accumulate_hessian() helper would produce: H = (2/n) * sum_t x_t x_t^T.)
torch.manual_seed(1)
_out_f, _in_f, _group = 6, 16, 8
_X_calib = torch.randn(64, _in_f)                       # 64 calibration "tokens"
_H = (2.0 / _X_calib.shape[0]) * (_X_calib.t() @ _X_calib)
_W = torch.randn(_out_f, _in_f) * 0.05

_Q = gptq_quantize_layer(_W, _H, n_bits=4, group_size=_group, percdamp=0.01)

assert _Q.shape == (_out_f, _in_f)
assert torch.isfinite(_Q).all()
_recon_err = (_Q - _W).abs().mean().item()
print(f"[block #2] gptq_quantize_layer ran OK, shape={tuple(_Q.shape)}, "
      f"mean abs recon error={_recon_err:.4e}")


# =====================================================================
# Block #4 (line ~267): awq_search_scales() -- AWQ per-channel scale search
# =====================================================================

def awq_search_scales(W, X, n_bits=4, group_size=128, grid=20):
    """
    AWQ per-input-channel scale search for one linear layer.
      W : [out, in]   weights
      X : [n_tokens, in]   calibration activations into this layer
    Returns scale vector s of shape [in]; you then quantize (W * s) and fold
    (1/s) into the upstream layernorm / previous linear.
    """
    W = W.float()
    x_mag = X.abs().float().mean(dim=0)          # [in]  per-channel activation scale
    w_mag = W.abs().mean(dim=0).clamp(min=1e-6)  # [in]  per-channel weight scale

    def quant_dequant(w):                         # group-wise INT4 RTN
        out, inf = w.shape
        wg = w.reshape(out, inf // group_size, group_size)
        qmax = 2 ** (n_bits - 1) - 1
        s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(wg / s), -qmax, qmax) * s
        return q.reshape(out, inf)

    ref = X.float() @ W.t()                       # FP16 reference output
    best_err, best_s = float("inf"), torch.ones_like(x_mag)
    for i in range(grid):
        beta = i / grid                           # interpolate importance exponent
        s = (x_mag ** beta) / (w_mag ** (1 - beta))
        s = s.clamp(min=1e-4)
        s = s / s.max()                           # normalize to avoid blowup
        Wq = quant_dequant(W * s)                 # quantize the *scaled* weights
        out = (X.float() / s) @ Wq.t()            # input rescaled by 1/s
        err = (out - ref).pow(2).mean().item()
        if err < best_err:
            best_err, best_s = err, s
    return best_s

# GLUE: tiny weight/activation matrices; group_size must divide in_features.
torch.manual_seed(2)
_W4 = torch.randn(8, 16) * 0.05
_X4 = torch.randn(32, 16)
_best_s = awq_search_scales(_W4, _X4, n_bits=4, group_size=8, grid=5)

assert _best_s.shape == (16,)
assert torch.isfinite(_best_s).all()
assert (_best_s > 0).all()
print(f"[block #4] awq_search_scales ran OK, scale range=[{_best_s.min():.4f}, {_best_s.max():.4f}]")


# =====================================================================
# Block #5 (line ~331): smoothquant_scales() / apply_smoothing()
# =====================================================================

@torch.no_grad()
def smoothquant_scales(act_abs_max, weight, alpha=0.5):
    """
    Compute the SmoothQuant per-channel migration scale.
      act_abs_max : [in]   per-channel max |activation| from calibration
      weight      : [out, in]
    Returns s of shape [in]: divide activations by s, multiply weight cols by s.
    """
    w_abs_max = weight.abs().amax(dim=0).clamp(min=1e-5)   # [in]
    a_abs_max = act_abs_max.clamp(min=1e-5)
    s = (a_abs_max.pow(alpha) / w_abs_max.pow(1 - alpha)).clamp(min=1e-5)
    return s

@torch.no_grad()
def apply_smoothing(weight, ln_weight, s):
    """Fold 1/s into the upstream norm, s into this linear's weight columns."""
    ln_weight = ln_weight / s          # activations get divided by s for free
    weight = weight * s.unsqueeze(0)   # weight columns multiplied by s
    return weight, ln_weight

# GLUE: tiny weight matrix + synthetic activation stats with a couple of
# outlier channels, exactly the scenario the chapter's SmoothQuant section
# describes.
torch.manual_seed(3)
_W5 = torch.randn(8, 16) * 0.05
_act_abs_max = torch.rand(16) * 0.5
_act_abs_max[3] = 40.0    # an outlier channel, 100x the rest
_act_abs_max[9] = 25.0
_ln_weight = torch.ones(16)

_s5 = smoothquant_scales(_act_abs_max, _W5, alpha=0.5)
_W5_smoothed, _ln_smoothed = apply_smoothing(_W5, _ln_weight, _s5)

assert _s5.shape == (16,)
assert _W5_smoothed.shape == _W5.shape
assert _ln_smoothed.shape == _ln_weight.shape
# outlier channels should get a larger migration scale
assert _s5[3] > _s5[0] and _s5[9] > _s5[0]
# folding is exact: (X/s) @ (W*s)^T == X @ W^T (the whole point of the trick)
_X5 = torch.randn(4, 16)
_orig_out = _X5 @ _W5.t()
_smoothed_out = (_X5 / _s5) @ _W5_smoothed.t()
assert torch.allclose(_orig_out, _smoothed_out, atol=1e-5)
print(f"[block #5] smoothquant_scales/apply_smoothing ran OK, "
      f"outlier scale s[3]={_s5[3]:.4f} vs normal s[0]={_s5[0]:.4f}")


# =====================================================================
# Block #6 (line ~369): llm_int8_matmul() -- mixed INT8/FP16 decomposition
# =====================================================================

@torch.no_grad()
def llm_int8_matmul(X, W, threshold=6.0):
    """
    Mixed INT8 / FP16 matmul, the LLM.int8() decomposition.
      X : [tokens, in]   activations (fp16)
      W : [in, out]      weights (fp16)
    """
    # 1) find outlier feature columns by magnitude across the batch
    col_max = X.abs().amax(dim=0)                # [in]
    outlier = col_max > threshold               # boolean mask
    reg = ~outlier

    # 2) FP16 path for the few outlier dimensions (exact, no quant error)
    y_fp16 = X[:, outlier] @ W[outlier, :]

    # 3) INT8 path for the regular dimensions
    Xr, Wr = X[:, reg], W[reg, :]
    sx = Xr.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127   # per-row act scale
    sw = Wr.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127   # per-col wgt scale
    Xq = torch.clamp(torch.round(Xr / sx), -127, 127).to(torch.int8)
    Wq = torch.clamp(torch.round(Wr / sw), -127, 127).to(torch.int8)
    acc = (Xq.float() @ Wq.float())             # stands in for an INT8 tensor-core matmul
    y_int8 = acc * sx * sw                       # dequantize with outer-product of scales

    return y_int8 + y_fp16

# GLUE: tiny activation matrix with a couple of injected outlier channels
# (magnitude > threshold) and a tiny weight matrix, matching the chapter's
# "outliers live in a handful of feature dimensions" scenario.
torch.manual_seed(4)
_tokens, _in6, _out6 = 5, 20, 4
_X6 = torch.randn(_tokens, _in6) * 0.5
_X6[:, 2] = 12.0 + torch.randn(_tokens) * 0.1     # outlier channel #2
_X6[:, 15] = -9.0 + torch.randn(_tokens) * 0.1    # outlier channel #15
_W6 = torch.randn(_in6, _out6) * 0.3

_y6 = llm_int8_matmul(_X6, _W6, threshold=6.0)
_y6_ref = _X6 @ _W6   # true FP result for comparison

assert _y6.shape == (_tokens, _out6)
assert torch.isfinite(_y6).all()
# INT8 quantization of the regular (non-outlier) dims plus exact FP16 handling
# of the outlier dims should stay reasonably close to the true FP matmul.
_rel_err = (_y6 - _y6_ref).abs().mean() / _y6_ref.abs().mean()
assert _rel_err.item() < 0.1
print(f"[block #6] llm_int8_matmul ran OK, shape={tuple(_y6.shape)}, "
      f"relative error vs FP={_rel_err.item():.4e}")


print("\nAll CPU-runnable blocks in 04-kernels-efficiency/07-quantization-ptq.md executed successfully.")
