"""Muon optimizer (Jordan et al., 2024): Newton-Schulz orthogonalization of the
momentum update, for the 2D hidden weight matrices. Used at scale by Kimi K2.

We run the Newton-Schulz iteration in fp32 (the chapter uses bf16) so it is
identical on CPU and GPU -- precision is more than enough for the quintic.
"""
import torch
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim == 2, "Newton-Schulz orthogonalization is only for 2D matrices"
    a, b, c = 3.4445, -4.7750, 2.0315          # tuned quintic coefficients
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


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.1, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            nesterov, wd, ns = group["nesterov"], group["weight_decay"], group["ns_steps"]
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
                d = g.add(buf, alpha=mu) if nesterov else buf
                o = zeropower_via_newtonschulz5(d, steps=ns).to(p.dtype)
                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5   # RMS-match to AdamW
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)              # decoupled weight decay
                p.add_(o, alpha=-lr * scale)          # W <- W - lr*scale*O
        return loss
