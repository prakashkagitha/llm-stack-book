"""
Runs the CPU-runnable Python blocks from
content/13-interview-prep/02-ml-breadth-rapidfire.md, concatenated in order,
with minimal glue to actually EXECUTE every function/class the chapter defines.

Blocks tested: 0-12 (all 13 heuristically CPU-runnable blocks in the chapter).
No blocks skipped.
"""

import numpy as np
import torch
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------------
# Block #0 (line ~25): bias-variance decomposition via polynomial fits
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)

def true_f(x):
    return np.sin(2 * np.pi * x)          # the function we wish we knew

def make_dataset(n=20, noise=0.25):
    x = rng.uniform(0, 1, n)
    y = true_f(x) + rng.normal(0, noise, n)  # irreducible noise sigma=0.25
    return x, y

# Fixed grid of test points where we measure bias/variance.
x_test = np.linspace(0, 1, 100)
f_test = true_f(x_test)

for degree in [1, 3, 9]:
    preds = []                            # predictions across many resampled datasets
    for _ in range(200):                  # 200 independent training sets
        x, y = make_dataset()
        coeffs = np.polyfit(x, y, degree)  # least-squares polynomial fit
        preds.append(np.polyval(coeffs, x_test))
    preds = np.stack(preds)               # shape (200, 100)

    mean_pred = preds.mean(axis=0)        # E_D[ f_hat(x) ]
    bias2 = np.mean((mean_pred - f_test) ** 2)
    variance = np.mean(preds.var(axis=0))
    print(f"degree={degree:>2}  bias^2={bias2:.4f}  variance={variance:.4f}  "
          f"sum={bias2+variance:.4f}")

print("[block 0] bias-variance sweep done")

# ---------------------------------------------------------------------------
# Block #1 (line ~93): L1 vs L2 proximal operators (soft-thresholding vs shrinkage)
# ---------------------------------------------------------------------------

# Why L1 zeros things out and L2 only shrinks: the proximal operators.
def prox_l1(w, t):           # soft-thresholding: solution of min 0.5(x-w)^2 + t|x|
    return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)

def prox_l2(w, t):           # shrinkage: solution of min 0.5(x-w)^2 + 0.5 t x^2
    return w / (1.0 + t)

w = np.array([0.05, 0.5, -0.3, 0.02, 2.0])
print("L1 (t=0.1):", prox_l1(w, 0.1))   # -> [ 0.  0.4 -0.2  0.  1.9] small coords killed
print("L2 (t=0.1):", prox_l2(w, 0.1))   # -> [0.045 0.455 ...]        all shrunk, none zero

assert (prox_l1(w, 0.1)[[0, 3]] == 0.0).all()  # near-zero coords killed exactly
assert (prox_l2(w, 0.1) != 0.0).all()          # L2 never zeroes anyone

print("[block 1] L1/L2 proximal operators done")

# ---------------------------------------------------------------------------
# Block #2 (line ~131): Adam optimizer step from scratch
# ---------------------------------------------------------------------------

def adam_step(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """One AdamW-style step. `state` holds m, v, and the step count t."""
    state.setdefault("t", 0); state["t"] += 1
    t = state["t"]
    for k in params:
        g = grads[k]
        m = state.setdefault(f"m_{k}", np.zeros_like(g))
        v = state.setdefault(f"v_{k}", np.zeros_like(g))
        m[:] = b1 * m + (1 - b1) * g            # first moment (momentum)
        v[:] = b2 * v + (1 - b2) * g * g        # second moment (uncentered variance)
        m_hat = m / (1 - b1 ** t)               # bias-corrected
        v_hat = v / (1 - b2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params

# Glue: actually call adam_step for a few steps on a toy parameter.
_params = {"w": np.array([1.0, 2.0, 3.0])}
_grads = {"w": np.array([0.1, -0.2, 0.05])}
_state = {}
for _ in range(5):
    adam_step(_params, _grads, _state)
print("adam w after 5 steps:", _params["w"])
assert _state["t"] == 5

print("[block 2] Adam step done")

# ---------------------------------------------------------------------------
# Block #3 (line ~156): learning-rate schedule (linear warmup + cosine decay)
# ---------------------------------------------------------------------------

def lr_schedule(step, base_lr=3e-4, warmup=2000, total=100_000, min_ratio=0.1):
    if step < warmup:                           # linear warmup
        return base_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))   # 1 -> 0
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)

for s in [0, 1000, 2000, 50_000, 100_000]:
    print(f"step {s:>7}: lr = {lr_schedule(s):.2e}")

assert lr_schedule(0) == 0.0
assert abs(lr_schedule(2000) - 3e-4) < 1e-12  # end of warmup hits base_lr

print("[block 3] LR schedule done")

# ---------------------------------------------------------------------------
# Block #4 (line ~184): init variance through depth + global-norm grad clipping
# ---------------------------------------------------------------------------

# 1) WHY init matters: variance of activations through 50 linear layers.
def forward_chain(scale, depth=50, width=512):
    x = np.random.randn(width)
    for _ in range(depth):
        W = np.random.randn(width, width) * scale
        x = np.maximum(W @ x, 0)               # ReLU
    return x.std()

print("naive  std≈", forward_chain(0.05))      # collapses toward 0 (vanishing)
print("He init std≈", forward_chain(np.sqrt(2/512)))  # stays O(1): Var=2/fan_in for ReLU

# 2) gradient clipping by global norm — the standard explosion guard.
def clip_grad_norm(grads, max_norm=1.0):
    total = np.sqrt(sum((g ** 2).sum() for g in grads))
    if total > max_norm:
        grads = [g * (max_norm / (total + 1e-6)) for g in grads]
    return grads

# Glue: actually call clip_grad_norm on an exploding toy gradient.
_big_grads = [np.array([3.0, 4.0]), np.array([0.0, 0.0])]  # norm = 5.0
_clipped = clip_grad_norm(_big_grads, max_norm=1.0)
_clipped_norm = np.sqrt(sum((g ** 2).sum() for g in _clipped))
print("clipped grad norm:", _clipped_norm)
assert _clipped_norm <= 1.0 + 1e-6

print("[block 4] init variance + grad clipping done")

# ---------------------------------------------------------------------------
# Block #5 (line ~224): LayerNorm vs. RMSNorm
# ---------------------------------------------------------------------------

x = torch.randn(4, 8, 16)   # (batch=4, seq=8, features=16)

# LayerNorm: stats over the LAST dim (features), per (batch, position).
mu = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)
ln = (x - mu) / torch.sqrt(var + 1e-5)
assert torch.allclose(ln.mean(-1), torch.zeros(4, 8), atol=1e-5)  # each token normalized

# RMSNorm (used in LLaMA/PaLM): no mean-subtraction, just divide by RMS. Cheaper.
rms = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

print("[block 5] LayerNorm/RMSNorm done, rms shape:", tuple(rms.shape))

# ---------------------------------------------------------------------------
# Block #6 (line ~245): inverted dropout
# ---------------------------------------------------------------------------

def inverted_dropout(x, p=0.1, training=True):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()   # keep with prob (1-p)
    return x * mask / (1 - p)                  # rescale so E[output] == E[input]

x_do = torch.ones(100_000)
out = inverted_dropout(x_do, p=0.3, training=True)
print(out.mean().item())   # ≈ 1.0 — expectation preserved despite 30% zeroed
assert abs(out.mean().item() - 1.0) < 0.05

print("[block 6] inverted dropout done")

# ---------------------------------------------------------------------------
# Block #7 (line ~294): scaled dot-product attention + causal mask
# ---------------------------------------------------------------------------

def attention(q, k, v, mask=None):
    # q,k,v: (batch, heads, seq, d_k)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5      # (b, h, seq, seq)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # causal mask
    weights = F.softmax(scores, dim=-1)                # rows sum to 1
    return weights @ v                                  # weighted sum of values

# Causal mask: position i may attend only to positions <= i (lower triangular).
seq = 5
causal = torch.tril(torch.ones(seq, seq))
print(causal)   # the mask that makes a decoder autoregressive

# Glue: actually run attention() with tiny (batch=1, heads=2, seq=5, d_k=4) tensors.
_g = torch.Generator().manual_seed(0)
q = torch.randn(1, 2, seq, 4, generator=_g)
k = torch.randn(1, 2, seq, 4, generator=_g)
v = torch.randn(1, 2, seq, 4, generator=_g)
mask4d = causal.unsqueeze(0).unsqueeze(0)  # broadcast over batch/heads
out_attn = attention(q, k, v, mask=mask4d)
assert out_attn.shape == (1, 2, seq, 4)
print("attention output shape:", tuple(out_attn.shape))

print("[block 7] attention done")

# ---------------------------------------------------------------------------
# Block #8 (line ~330): cosine similarity vs. dot product
# ---------------------------------------------------------------------------

# Cosine similarity vs. dot product: when do they differ?
a = torch.tensor([1.0, 0.0])
b = torch.tensor([2.0, 0.0])   # same direction, 2x magnitude
c = torch.tensor([0.0, 1.0])   # orthogonal

print(F.cosine_similarity(a, b, dim=0))  # 1.0 — direction identical
print(a @ b)                             # 2.0 — magnitude inflates dot product
print(F.cosine_similarity(a, c, dim=0))  # 0.0 — orthogonal = unrelated

print("[block 8] cosine similarity done")

# ---------------------------------------------------------------------------
# Block #9 (line ~361): confusion-matrix metrics
# ---------------------------------------------------------------------------

def metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return dict(precision=precision, recall=recall, f1=f1,
                accuracy=(tp + tn) / len(y_true))

# A 99%-negative dataset: accuracy is a liar.
y_true = np.array([0] * 990 + [1] * 10)
y_pred_lazy = np.zeros(1000)              # predict "negative" for everything
print(metrics(y_true, y_pred_lazy))
# accuracy=0.99, but recall=0.0 — the model is useless and accuracy hides it.

_m = metrics(y_true, y_pred_lazy)
assert abs(_m["accuracy"] - 0.99) < 1e-6
assert _m["recall"] == 0.0

print("[block 9] confusion-matrix metrics done")

# ---------------------------------------------------------------------------
# Block #10 (line ~388): sweep thresholds for best F-beta
# ---------------------------------------------------------------------------

def best_threshold(y_true, scores, beta=1.0):
    """Sweep thresholds, return the one maximizing F-beta."""
    thresholds = np.unique(scores)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-9)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f

# Glue: actually call best_threshold on a tiny toy scored dataset.
_y_true_small = np.array([0, 0, 1, 1, 1])
_scores_small = np.array([0.1, 0.4, 0.35, 0.8, 0.65])
_t, _f = best_threshold(_y_true_small, _scores_small)
print("best threshold:", _t, "best F1:", _f)
assert 0.0 <= _t <= 1.0
assert _f > 0.0

print("[block 10] threshold sweep done")

# ---------------------------------------------------------------------------
# Block #11 (line ~420): focal loss
# ---------------------------------------------------------------------------

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    # logits: (N, C), targets: (N,) class indices
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    pt = p.gather(1, targets[:, None]).squeeze(1)     # prob of the true class
    logpt = logp.gather(1, targets[:, None]).squeeze(1)
    focal = -alpha * (1 - pt) ** gamma * logpt        # easy examples (pt→1) contribute ≈0
    return focal.mean()

# Glue: actually call focal_loss on tiny toy logits/targets.
_g2 = torch.Generator().manual_seed(1)
_logits = torch.randn(4, 3, generator=_g2)
_targets = torch.tensor([0, 1, 2, 1])
_floss = focal_loss(_logits, _targets)
print("focal loss:", _floss.item())
assert _floss.item() >= 0.0

print("[block 11] focal loss done")

# ---------------------------------------------------------------------------
# Block #12 (line ~462): softmax + cross-entropy with the (p - y) gradient
# ---------------------------------------------------------------------------

def softmax_cross_entropy(logits, y_onehot):
    z = logits - logits.max(axis=-1, keepdims=True)   # numerical stability
    p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    loss = -(y_onehot * np.log(p + 1e-12)).sum(axis=-1).mean()
    grad = (p - y_onehot) / len(logits)                # the famous (p - y)
    return loss, grad

# Glue: actually call softmax_cross_entropy on tiny toy logits/one-hot targets.
_ce_logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
_ce_y = np.array([[1, 0, 0], [0, 1, 0]])
_ce_loss, _ce_grad = softmax_cross_entropy(_ce_logits, _ce_y)
print("softmax-CE loss:", _ce_loss, "grad:", _ce_grad)
assert _ce_loss > 0.0
assert _ce_grad.shape == _ce_logits.shape

print("[block 12] softmax cross-entropy done")

print("\nALL BLOCKS EXECUTED SUCCESSFULLY")
