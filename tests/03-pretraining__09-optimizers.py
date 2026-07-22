"""
Executable test for content/03-pretraining/09-optimizers.md

Concatenates the chapter's 5 CPU-runnable Python blocks in order and exercises
each one with tiny CPU tensors so the book's actual code runs end to end.

Blocks covered:
  #0 (line ~27)  sgd_momentum_step
  #1 (line ~101) AdamW (from-scratch optimizer class)
  #2 (line ~206) adafactor_matrix_step
  #3 (line ~252) lion_step
  #4 (line ~319) newton_schulz5 + muon_step
"""

import torch

# ============================================================
# Block #0 (line ~27) -- SGD / Nesterov momentum step
# ============================================================
import torch  # noqa: F811 (book repeats this import per-block; kept verbatim)

def sgd_momentum_step(params, grads, velocities, lr=0.1, mu=0.9, nesterov=False):
    """One step of (Nesterov) momentum SGD, in-place. Pure PyTorch tensors."""
    for p, g, v in zip(params, grads, velocities):
        v.mul_(mu).add_(g)              # v <- mu*v + g
        if nesterov:
            update = g.add(v, alpha=mu)  # g + mu*v  (look-ahead)
        else:
            update = v
        p.add_(update, alpha=-lr)        # theta <- theta - lr * update


# ============================================================
# Block #1 (line ~101) -- From-scratch AdamW optimizer
# ============================================================
import torch  # noqa: F811
from torch.optim import Optimizer

class AdamW(Optimizer):
    """From-scratch AdamW (Loshchilov & Hutter, 2019).

    Stores two state tensors per parameter: exp_avg (m) and exp_avg_sq (v).
    This is the memory tax we account for in the next section.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95),
                 eps=1e-8, weight_decay=0.1):
        # betas: (beta1, beta2). For LLMs beta2=0.95 is the common choice.
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2) = group["lr"], group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdamW does not support sparse grads")

                state = self.state[p]
                if len(state) == 0:                    # lazy init
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)      # m
                    state["exp_avg_sq"] = torch.zeros_like(p)   # v
                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # --- Decoupled weight decay: multiplicative shrink toward 0 ---
                # Applied to the *weights*, NOT folded into the gradient.
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # --- Update biased first and second moment estimates ---
                m.mul_(b1).add_(g, alpha=1.0 - b1)          # m = b1*m + (1-b1)*g
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)   # v = b2*v + (1-b2)*g^2

                # --- Bias correction ---
                bias_c1 = 1.0 - b1 ** t
                bias_c2 = 1.0 - b2 ** t
                # Fold bias_c2 into the denominator; step_size folds bias_c1.
                denom = (v.sqrt() / (bias_c2 ** 0.5)).add_(eps)
                step_size = lr / bias_c1

                # theta <- theta - step_size * m / denom
                p.addcdiv_(m, denom, value=-step_size)
        return loss


# ============================================================
# Block #2 (line ~206) -- Adafactor factored second moment
# ============================================================
import torch  # noqa: F811

def adafactor_matrix_step(W, G, R, C, t, lr, beta2=0.999, eps1=1e-30, eps2=1e-3):
    """One Adafactor step for a 2D weight W with grad G.
    R: row accumulator (n,), C: col accumulator (m,). No first moment here.
    """
    n, m = W.shape
    g2 = G * G + eps1                       # squared grad, floored

    # Decayed running averages of row sums and column sums of g^2
    beta2_t = 1.0 - t ** (-0.8)             # Adafactor's time-dependent decay
    R.mul_(beta2_t).add_(g2.mean(dim=1), alpha=1 - beta2_t)   # (n,)
    C.mul_(beta2_t).add_(g2.mean(dim=0), alpha=1 - beta2_t)   # (m,)

    # Rank-1 reconstruction of the second-moment estimate V_hat (n,m)
    R_factor = (R / R.mean()).rsqrt().unsqueeze(1)   # (n,1)
    C_factor = C.rsqrt().unsqueeze(0)                # (1,m)
    update = G * R_factor * C_factor                 # G / sqrt(V_hat)

    # RMS-clip the update (Adafactor's stability trick)
    rms = update.pow(2).mean().sqrt()
    update = update / max(1.0, (rms / 1.0).item())

    # Relative step size scaled by parameter RMS
    param_rms = W.pow(2).mean().sqrt().clamp_min(eps2)
    W.add_(update, alpha=-lr * param_rms.item())


# ============================================================
# Block #3 (line ~252) -- Lion optimizer step
# ============================================================
import torch  # noqa: F811

def lion_step(p, g, m, lr=1e-4, beta1=0.9, beta2=0.99, wd=0.0):
    """One Lion update, in-place. Only ONE state tensor m per parameter."""
    # Update direction uses an interpolation with beta1...
    c = m.mul(beta1).add(g, alpha=1.0 - beta1)   # beta1*m + (1-beta1)*g (temp)
    update = c.sign()                            # +/-1 per coordinate
    if wd != 0:
        p.mul_(1.0 - lr * wd)                    # decoupled weight decay
    p.add_(update, alpha=-lr)                    # theta <- theta - lr*sign(c)
    # ...but the stored momentum uses beta2 (note: g, not c)
    m.mul_(beta2).add_(g, alpha=1.0 - beta2)


# ============================================================
# Block #4 (line ~319) -- Muon: Newton-Schulz orthogonalization
# ============================================================
import torch  # noqa: F811

@torch.no_grad()
def newton_schulz5(G, steps=5, eps=1e-7):
    """Compute an approximate UV^T (orthogonalization) of G via Newton-Schulz.
    Matmul-only; runs in bf16. Coefficients tuned to converge singular values->1.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):           # iterate on the smaller dimension
        X = X.T
        transposed = True
    X = X / (X.norm() + eps)            # spectral pre-normalization
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X

@torch.no_grad()
def muon_step(W, G, momentum_buf, lr=0.02, mu=0.95, ns_steps=5):
    """One Muon update for a 2D weight W. Only ONE state buffer (momentum)."""
    momentum_buf.mul_(mu).add_(G)               # standard heavy-ball momentum
    update = newton_schulz5(momentum_buf, steps=ns_steps)  # orthogonalize
    # Scale by sqrt(max(rows,cols)) so RMS of update ~ 1, matching AdamW's scale
    scale = (max(W.shape) ** 0.5)
    W.add_(update, alpha=-lr * scale)


# ============================================================
# Exercise every block with tiny CPU fixtures
# ============================================================

def main():
    torch.manual_seed(0)

    # --- Block #0: sgd_momentum_step -------------------------------------
    params = [torch.randn(4, 4), torch.randn(3)]
    grads = [torch.randn(4, 4), torch.randn(3)]
    velocities = [torch.zeros(4, 4), torch.zeros(3)]
    before = [p.clone() for p in params]
    sgd_momentum_step(params, grads, velocities, lr=0.1, mu=0.9, nesterov=False)
    for p, b in zip(params, before):
        assert not torch.allclose(p, b), "sgd_momentum_step did not update params"
    # Also exercise the Nesterov branch.
    params2 = [torch.randn(4, 4)]
    grads2 = [torch.randn(4, 4)]
    velocities2 = [torch.zeros(4, 4)]
    sgd_momentum_step(params2, grads2, velocities2, lr=0.1, mu=0.9, nesterov=True)
    assert velocities2[0].abs().sum().item() > 0
    print("[OK] block #0 sgd_momentum_step ran (plain + nesterov)")

    # --- Block #1: AdamW class --------------------------------------------
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(6, 6))
    opt = AdamW([w], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    w0 = w.detach().clone()
    for _ in range(5):
        opt.zero_grad()
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
    assert not torch.allclose(w.detach(), w0), "AdamW did not move the weight"
    assert loss.item() >= 0
    # Sanity: state has the two expected buffers.
    state = opt.state[w]
    assert "exp_avg" in state and "exp_avg_sq" in state
    print(f"[OK] block #1 AdamW ran 5 steps, loss={loss.item():.4f}")

    # --- Block #2: adafactor_matrix_step -----------------------------------
    torch.manual_seed(0)
    W = torch.randn(5, 4)
    R = torch.zeros(5)
    C = torch.zeros(4)
    W0 = W.clone()
    for t in range(1, 6):
        G = torch.randn(5, 4) * 0.1
        adafactor_matrix_step(W, G, R, C, t, lr=0.5, beta2=0.999)
    assert not torch.allclose(W, W0), "adafactor_matrix_step did not update W"
    assert torch.isfinite(W).all()
    print("[OK] block #2 adafactor_matrix_step ran 5 steps")

    # --- Block #3: lion_step -------------------------------------------------
    torch.manual_seed(0)
    p = torch.randn(4, 4)
    m = torch.zeros(4, 4)
    p0 = p.clone()
    for _ in range(5):
        g = torch.randn(4, 4) * 0.1
        lion_step(p, g, m, lr=1e-2, beta1=0.9, beta2=0.99, wd=0.01)
    assert not torch.allclose(p, p0), "lion_step did not update p"
    # Every coordinate should have moved by a multiple of lr (sign update) plus
    # decay -- just confirm updates stayed bounded/finite.
    assert torch.isfinite(p).all()
    print("[OK] block #3 lion_step ran 5 steps")

    # --- Block #4: newton_schulz5 + muon_step --------------------------------
    torch.manual_seed(0)
    G = torch.randn(6, 4)
    O = newton_schulz5(G, steps=5)
    assert O.shape == G.shape
    # Singular values of the orthogonalized (non-square, so semi-orthogonal)
    # output should be close to 1 for the min(rows,cols) directions.
    svals = torch.linalg.svdvals(O.float())
    # bf16 + 5 Newton-Schulz iterations only *approximately* whitens the
    # singular values toward 1 -- allow a generous tolerance.
    assert torch.allclose(svals, torch.ones_like(svals), atol=0.35), svals

    Wm = torch.randn(6, 4)
    Gm = torch.randn(6, 4)
    momentum_buf = torch.zeros(6, 4)
    Wm0 = Wm.clone()
    muon_step(Wm, Gm, momentum_buf, lr=0.02, mu=0.95, ns_steps=5)
    assert not torch.allclose(Wm, Wm0), "muon_step did not update W"
    assert momentum_buf.abs().sum().item() > 0
    print("[OK] block #4 newton_schulz5 + muon_step ran")

    print("\nAll 5 blocks executed successfully.")


if __name__ == "__main__":
    main()
