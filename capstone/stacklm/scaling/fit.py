"""Mini scaling laws (Ch. 14.5). Fit L(N,D) = E + A/N^alpha + B/D^beta from a
ladder of small runs, then extrapolate to the 100M config.

The chapter fits this with `scipy.optimize.minimize(L-BFGS-B)`; CI has no scipy.
We exploit structure instead: for FIXED (alpha, beta) the model is LINEAR in
(E, A, B), so we grid over the two exponents (constrained to the empirical
0.25-0.42 band) and solve each inner problem with numpy least-squares. Fully
deterministic, no scipy.
"""
import numpy as np

from .ladder import family
from .flops import flops_per_token


def _linear_fit_for_exponents(N, D, L, alpha, beta):
    """Least-squares E, A, B for fixed exponents; returns (E, A, B, sse)."""
    X = np.stack([np.ones_like(N), N ** (-alpha), D ** (-beta)], axis=1)
    coef, *_ = np.linalg.lstsq(X, L, rcond=None)
    pred = X @ coef
    sse = float(((pred - L) ** 2).sum())
    return coef[0], coef[1], coef[2], sse


def fit_scaling_law(N, D, L, alpha_grid=None, beta_grid=None):
    """Fit the Chinchilla-style law. Returns a dict with E, A, B, alpha, beta, sse."""
    N = np.asarray(N, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    if alpha_grid is None:
        alpha_grid = np.linspace(0.25, 0.42, 35)
    if beta_grid is None:
        beta_grid = np.linspace(0.25, 0.42, 35)

    best = None
    for alpha in alpha_grid:
        for beta in beta_grid:
            E, A, B, sse = _linear_fit_for_exponents(N, D, L, alpha, beta)
            if E < 0 or A < 0 or B < 0:      # nonphysical: reject
                continue
            if best is None or sse < best["sse"]:
                best = dict(E=float(E), A=float(A), B=float(B),
                            alpha=float(alpha), beta=float(beta), sse=sse)
    if best is None:                         # fall back to unconstrained best
        E, A, B, sse = _linear_fit_for_exponents(N, D, L, 0.34, 0.28)
        best = dict(E=float(E), A=float(A), B=float(B), alpha=0.34, beta=0.28, sse=sse)
    return best


def predicted_loss(fit: dict, N: float, D: float) -> float:
    return float(fit["E"] + fit["A"] * N ** (-fit["alpha"]) + fit["B"] * D ** (-fit["beta"]))


def compute_optimal_allocation(C: float, fit: dict, use_6nd: bool = False,
                               d_grid=range(128, 2049, 64)) -> dict:
    """Min-loss allocation at a fixed compute budget C, searching REAL configs.

    D is forced by the FLOP constraint. By default the constraint uses the full
    per-token cost (blocks + causal attention + tied head) from `flops.py`, so an
    "IsoFLOP" slice really is iso-compute. `use_6nd=True` reproduces the naive
    6ND constraint for contrast (Ch. 14.5, Exercise 5) -- it credits the model
    with ~1.6x more tokens than it can actually afford at this scale.
    """
    E, A, B, alpha, beta = fit["E"], fit["A"], fit["B"], fit["alpha"], fit["beta"]
    best = None
    for d in d_grid:
        cfg = family(d)
        N = cfg.nonembed_params()
        D = C / (6.0 * N) if use_6nd else C / flops_per_token(cfg)["total"]
        Lpred = E + A * N ** (-alpha) + B * D ** (-beta)
        if best is None or Lpred < best["L"]:
            best = dict(d_model=d, n_layers=cfg.n_layers, N=float(N), D=float(D),
                        L=float(Lpred), tpp=float(D / N))
    return best


def isoflop_points(fit: dict, budgets, **kw) -> list:
    """IsoFLOP profiles (Hoffmann et al., 2022): min-loss (N*, D*) per budget."""
    return [{"C": float(C), **compute_optimal_allocation(C, fit, **kw)}
            for C in budgets]
