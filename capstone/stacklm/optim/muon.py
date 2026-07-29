"""Muon optimizer (Jordan et al., 2024): Newton-Schulz orthogonalization of the
momentum update, for the 2D hidden weight matrices. Used at scale by Kimi K2.

We run the Newton-Schulz iteration in fp32 (the reference runs bf16). fp32 is
plenty for a quintic whose job is to condition a direction, and it keeps the CPU
smoke test agreeing with the GPU path to fp32 tolerance -- not bit-for-bit (CUDA
and CPU GEMMs use different reduction orders), but far closer than bf16 would.

`batched=True` switches `Muon.step()` to the shape-bucketed fast path: all
identically-shaped update directions are stacked and orthogonalized with one
`bmm` chain instead of one `mm` chain each. For Stack-100M that collapses 210
matrices into 3 shape classes (Ch. 14.6).
"""
import torch
from torch.optim.optimizer import Optimizer

_NS_COEFFS = (3.4445, -4.7750, 2.0315)   # tuned quintic coefficients


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2, "Newton-Schulz orthogonalization is only for 2D matrices"
    a, b, c = _NS_COEFFS
    X = G.float()
    X = X / (X.norm() + 1e-7)                    # normalize so all sigma <= 1
    transposed = G.size(0) > G.size(1)          # keep the short side as rows
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        P = b * A + c * (A @ A)                  # name P, not b (avoid shadowing)
        X = a * X + P @ X
    if transposed:
        X = X.T
    return X


@torch.no_grad()
def zeropower_batched(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Same quintic on a STACK of identically-shaped matrices: G is (K, m, n).

    Collapses K*steps*4 kernel launches into steps*4 batched ones. Algebraically
    identical to looping `zeropower_via_newtonschulz5`; expect last-ulp
    differences, because `bmm` and `mm` use different reduction orders.
    """
    assert G.ndim == 3, "zeropower_batched expects a (K, m, n) stack"
    a, b, c = _NS_COEFFS
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)   # per-matrix Frobenius
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT                                          # batched transpose
    for _ in range(steps):
        A = X @ X.mT                                      # (K, r, r) via bmm
        P = b * A + c * (A @ A)
        X = a * X + P @ X
    if transposed:
        X = X.mT
    return X


@torch.no_grad()
def _orthogonalize_bucketed(dirs, steps: int):
    """Orthogonalize a list of 2D directions, bucketing identical shapes.

    Matrices are keyed by their SORTED shape, so a (1408, 512) and a (512, 1408)
    land in the same bucket after the orientation-normalizing transpose -- which
    is what collapses Stack-100M's 210 matrices into 3 classes rather than 4.
    """
    out = [None] * len(dirs)
    buckets: dict[tuple, list[int]] = {}
    for i, d in enumerate(dirs):
        m, n = d.shape
        buckets.setdefault((min(m, n), max(m, n)), []).append(i)
    for idxs in buckets.values():
        flip = [dirs[i].size(0) > dirs[i].size(1) for i in idxs]
        stack = torch.stack([dirs[i].T if f else dirs[i] for i, f in zip(idxs, flip)])
        O = zeropower_batched(stack, steps=steps)
        for j, (i, f) in enumerate(zip(idxs, flip)):
            out[i] = O[j].T if f else O[j]
    return out


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.1, ns_steps=5, batched=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.batched = batched

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            nesterov, wd, ns = group["nesterov"], group["weight_decay"], group["ns_steps"]

            live, dirs = [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon received a non-2D param; check routing"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)                    # B <- mu*B + g
                # Nesterov look-ahead: orthogonalize g + mu*B, not the bare buffer.
                live.append(p)
                dirs.append(g.add(buf, alpha=mu) if nesterov else buf)

            if not live:
                continue
            if self.batched:
                orths = _orthogonalize_bucketed(dirs, ns)
            else:
                orths = [zeropower_via_newtonschulz5(d, steps=ns) for d in dirs]

            for p, o in zip(live, orths):
                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5   # RMS-match to AdamW
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)              # decoupled weight decay
                p.add_(o.to(p.dtype), alpha=-lr * scale)   # W <- W - lr*scale*O
        return loss
