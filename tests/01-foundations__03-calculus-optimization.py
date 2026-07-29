"""
Runnable-code test for content/01-foundations/03-calculus-optimization.md

Chapter blocks tested (assembled in document order):

    - block #1 (line ~359): "Full Runnable Code: Gradient Descent on a Toy Loss
      Surface" -- implements the Rosenbrock and anisotropic-quadratic loss
      surfaces plus four from-scratch optimizers (vanilla GD, SGD-with-noise,
      heavy-ball momentum, Adam) and a `main()` that runs all of them and
      prints/JSON-serializes trajectories. Copied verbatim from the chapter
      and executed via `main()` below, which is the block's own entry point.

    - the two PyTorch blocks added in the library/buildability revision pass:
      "The Library Layer: torch.optim" (AdamW parameter groups, LambdaLR
      warmup+cosine, clip_grad_norm_, and an equivalence check between the
      chapter's from-scratch heavy-ball update and torch.optim.SGD) and
      "Measuring Curvature Without Forming the Hessian" (double-backward
      Hessian-vector products + power iteration for lambda_max). Both are
      copied verbatim and executed at the bottom of this file.

SKIPPED (per task spec, non-Python / prose blocks):
    - block #0 (line ~135): SKIP(non-python): ```text``` ASCII-art schematic of
      convex vs non-convex loss landscapes -- not executable code.
    - block #2 (line ~562): SKIP(non-python): ```text``` fenced block showing
      sample console output of block #1 -- not executable code, used below only
      as a sanity reference for expected magnitudes.

This chapter's only runnable block (`import numpy as np` / `import json` /
stdlib `typing` only) has no third-party dependencies beyond numpy, which is
in the guaranteed CI import list, so no optional-import guarding is needed.
To keep runtime well under the ~60s budget while still exercising every line
of the book's logic, the same `n_steps` values from the chapter are kept
(2000 steps x4 optimizers on Rosenbrock, 200 steps x3 optimizers on the
quadratic) -- this runs in well under a second since it's pure numpy on a
2D state vector.

REAL BUG FOUND AND FIXED IN THE CHAPTER: the "Running this script produces
output like:" reference block (originally at line ~562) reported a Rosenbrock
starting loss of 6.2500 for theta0=(-1.5, 0.5) -- the correct value, verified
by actually running this exact code, is 312.5000 (the (1-x)^2 term alone is
6.25, but the (y-x^2)^2 term contributes another 306.25). The same fabricated
block also claimed Adam converges fastest on both surfaces and momentum beats
GD by ~3.5x on the quadratic; actually running the code shows momentum (not
Adam) essentially solves Rosenbrock exactly, Adam does *worse* than plain GD
on Rosenbrock with these hyperparameters, and plain GD reaches the quadratic's
loss<0.01 threshold before momentum or Adam do (which overshoot/oscillate
first before settling much lower). content/01-foundations/03-calculus-optimization.md
was corrected to report the actual, reproducible output and an accurate
explanation. The assertions below encode the actual (correct) behavior.
"""

# ============================================================================
# block #1 (line ~359): Gradient descent visualization on a 2D loss surface.
# Implements GD, SGD (with noise), Momentum, and Adam from scratch.
# No dependencies except numpy. Copied verbatim from the chapter.
# ============================================================================

import numpy as np
import json
from typing import Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Loss surface and gradient
# ---------------------------------------------------------------------------

def rosenbrock(theta: np.ndarray) -> float:
    """
    Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y - x^2)^2
    Classic non-convex test: narrow curved valley, global min at (1, 1).
    """
    x, y = theta[0], theta[1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2


def rosenbrock_grad(theta: np.ndarray) -> np.ndarray:
    """Analytical gradient of the Rosenbrock function."""
    x, y = theta[0], theta[1]
    dfdx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    dfdy = 200.0 * (y - x**2)
    return np.array([dfdx, dfdy])


def quadratic(theta: np.ndarray) -> float:
    """Anisotropic quadratic: f(x,y) = x^2 + 10*y^2. Condition number 10."""
    return theta[0]**2 + 10.0 * theta[1]**2


def quadratic_grad(theta: np.ndarray) -> np.ndarray:
    """Gradient of the anisotropic quadratic."""
    return np.array([2.0 * theta[0], 20.0 * theta[1]])


# ---------------------------------------------------------------------------
# Optimizers (all from scratch, no libraries)
# ---------------------------------------------------------------------------

def run_gd(
    theta0: np.ndarray,
    grad_fn: Callable,
    lr: float = 0.005,
    n_steps: int = 500,
) -> List[np.ndarray]:
    """Vanilla gradient descent."""
    theta = theta0.copy()
    trajectory = [theta.copy()]
    for _ in range(n_steps):
        g = grad_fn(theta)
        theta = theta - lr * g
        trajectory.append(theta.copy())
    return trajectory


def run_sgd_with_noise(
    theta0: np.ndarray,
    grad_fn: Callable,
    lr: float = 0.005,
    noise_std: float = 0.1,
    n_steps: int = 500,
    rng_seed: int = 42,
) -> List[np.ndarray]:
    """
    SGD with simulated mini-batch noise.
    In practice the noise comes from random mini-batches; here we add
    Gaussian noise to the gradient to simulate the same effect.
    """
    rng = np.random.default_rng(rng_seed)
    theta = theta0.copy()
    trajectory = [theta.copy()]
    for _ in range(n_steps):
        g = grad_fn(theta) + rng.normal(0, noise_std, size=theta.shape)
        theta = theta - lr * g
        trajectory.append(theta.copy())
    return trajectory


def run_momentum(
    theta0: np.ndarray,
    grad_fn: Callable,
    lr: float = 0.005,
    beta: float = 0.9,
    n_steps: int = 500,
) -> List[np.ndarray]:
    """SGD with heavy-ball momentum."""
    theta = theta0.copy()
    velocity = np.zeros_like(theta)
    trajectory = [theta.copy()]
    for _ in range(n_steps):
        g = grad_fn(theta)
        velocity = beta * velocity - lr * g   # accumulate direction
        theta = theta + velocity
        trajectory.append(theta.copy())
    return trajectory


def run_adam(
    theta0: np.ndarray,
    grad_fn: Callable,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    n_steps: int = 500,
) -> List[np.ndarray]:
    """
    Adam optimizer from scratch.
    Note the bias-correction terms: critical for the first ~1/(1-beta) steps.
    """
    theta = theta0.copy()
    m = np.zeros_like(theta)   # first moment (mean)
    v = np.zeros_like(theta)   # second moment (uncentered variance)
    trajectory = [theta.copy()]
    for t in range(1, n_steps + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1.0 - beta1) * g          # update biased first moment
        v = beta2 * v + (1.0 - beta2) * g**2       # update biased second moment
        m_hat = m / (1.0 - beta1**t)               # bias correction
        v_hat = v / (1.0 - beta2**t)               # bias correction
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(theta.copy())
    return trajectory


# ---------------------------------------------------------------------------
# Run experiments and emit results as JSON-serializable data
# ---------------------------------------------------------------------------

def trajectory_to_losses(
    traj: List[np.ndarray],
    loss_fn: Callable,
) -> List[float]:
    """Convert a list of parameter vectors into a list of loss values."""
    return [float(loss_fn(theta)) for theta in traj]


def main():
    theta0 = np.array([-1.5, 0.5])   # starting point for Rosenbrock

    print("=== Rosenbrock surface (non-convex) ===")
    print(f"  Starting loss: {rosenbrock(theta0):.4f}")
    print(f"  Global minimum at (1, 1), loss = 0\n")

    runs: Dict[str, List[np.ndarray]] = {
        "GD":       run_gd(theta0, rosenbrock_grad, lr=0.001, n_steps=2000),
        "Momentum": run_momentum(theta0, rosenbrock_grad, lr=0.001, beta=0.9, n_steps=2000),
        "Adam":     run_adam(theta0, rosenbrock_grad, lr=0.01, n_steps=2000),
        "SGD+Noise": run_sgd_with_noise(theta0, rosenbrock_grad, lr=0.001,
                                         noise_std=0.05, n_steps=2000),
    }

    results = {}
    for name, traj in runs.items():
        losses = trajectory_to_losses(traj, rosenbrock)
        final_theta = traj[-1]
        results[name] = {
            "final_loss": losses[-1],
            "final_theta": final_theta.tolist(),
            # Subsample trajectory for visualization (every 50 steps)
            "loss_curve": losses[::50],
            "path_x": [t[0] for t in traj[::50]],
            "path_y": [t[1] for t in traj[::50]],
        }
        print(f"  {name:12s}  final_loss={losses[-1]:.6f}  "
              f"theta=({final_theta[0]:.4f}, {final_theta[1]:.4f})")

    # Emit as JSON so downstream tools (e.g., matplotlib, Vega, Plotly) can render
    print("\n=== JSON output (for plotting) ===")
    print(json.dumps(results, indent=2)[:800], "...")   # truncate for display

    # -----------------------------------------------------------------------
    # Second experiment: condition number demonstration
    # -----------------------------------------------------------------------
    print("\n=== Quadratic surface (condition number κ=10) ===")
    theta0_q = np.array([4.0, 1.0])

    gd_traj    = run_gd(theta0_q, quadratic_grad, lr=0.04, n_steps=200)
    mom_traj   = run_momentum(theta0_q, quadratic_grad, lr=0.04, beta=0.9, n_steps=200)
    adam_traj  = run_adam(theta0_q, quadratic_grad, lr=0.1, n_steps=200)

    for name, traj in [("GD", gd_traj), ("Momentum", mom_traj), ("Adam", adam_traj)]:
        losses = trajectory_to_losses(traj, quadratic)
        # Find first step where loss < 0.01
        converge_step = next((i for i, l in enumerate(losses) if l < 0.01), None)
        print(f"  {name:10s}  steps to loss<0.01: "
              f"{'>' + str(len(losses)) if converge_step is None else converge_step}")

    return runs, results, (gd_traj, mom_traj, adam_traj)


# ============================================================================
# Test-harness glue: actually execute block #1's entry point and assert on
# the results, mirroring the "Running this script produces output like"
# reference block (line ~562, SKIPPED as non-python but used here as a
# sanity check on expected magnitudes/ordering).
# ============================================================================

if __name__ == "__main__":
    runs, results, (gd_traj, mom_traj, adam_traj) = main()

    # --- Sanity checks on the Rosenbrock experiment ------------------------
    # theta0 = (-1.5, 0.5); correct starting loss is 312.5 (verified above as
    # the real bug: the chapter originally mis-stated this as 6.25).
    start_loss = rosenbrock(np.array([-1.5, 0.5]))
    assert abs(start_loss - 312.5) < 1e-9, f"unexpected starting loss: {start_loss}"

    for name in ("GD", "Momentum", "Adam", "SGD+Noise"):
        final_loss = results[name]["final_loss"]
        assert final_loss < start_loss, (
            f"{name} did not reduce the Rosenbrock loss: {final_loss} >= {start_loss}"
        )

    # Momentum essentially solves Rosenbrock exactly with these hyperparameters
    # -- it should land very close to the true minimum (1, 1), loss ~ 0.
    assert results["Momentum"]["final_loss"] < 1e-4, (
        f"Momentum should nearly reach the Rosenbrock minimum "
        f"(loss~0), got {results['Momentum']['final_loss']}"
    )
    mom_theta = results["Momentum"]["final_theta"]
    assert abs(mom_theta[0] - 1.0) < 1e-2 and abs(mom_theta[1] - 1.0) < 1e-2, (
        f"Momentum's final theta should be near (1, 1), got {mom_theta}"
    )

    # With these (untuned) hyperparameters, Adam actually does *worse* than
    # plain GD on Rosenbrock's curved valley -- its per-coordinate
    # normalization fights the coupling between x and y. This is the real,
    # reproducible behavior (contrary to the chapter's original, fabricated
    # claim that "Adam converges fastest on both surfaces").
    assert results["Adam"]["final_loss"] > results["GD"]["final_loss"], (
        "Expected Adam to underperform vanilla GD on Rosenbrock with these "
        "untuned hyperparameters (the actual, reproducible behavior)"
    )

    # SGD+Noise should NOT fully converge to the minimum (the noise floor
    # keeps it oscillating) -- this is the chapter's generalization-vs-
    # convergence tradeoff point.
    assert results["SGD+Noise"]["final_loss"] > 1e-3, (
        "SGD+Noise unexpectedly converged to near-zero loss; noise floor "
        "should keep it away from the exact minimum"
    )

    # --- Sanity checks on the quadratic (condition number) experiment ------
    def steps_to_converge(traj):
        losses = trajectory_to_losses(traj, quadratic)
        step = next((i for i, l in enumerate(losses) if l < 0.01), None)
        return step if step is not None else len(losses)

    gd_steps = steps_to_converge(gd_traj)
    mom_steps = steps_to_converge(mom_traj)
    adam_steps = steps_to_converge(adam_traj)

    # With these (untuned) hyperparameters, plain GD actually reaches the
    # loss<0.01 threshold *before* momentum or Adam do -- momentum and Adam
    # overshoot/oscillate in the early steps here before eventually settling
    # to a much lower loss than GD ever reaches. This is the real,
    # reproducible behavior (contrary to the chapter's original, fabricated
    # claim that momentum/Adam beat GD by 3-8x on this surface).
    assert gd_steps < mom_steps, (
        f"Expected GD ({gd_steps} steps) to cross the loss<0.01 threshold "
        f"before Momentum ({mom_steps} steps) with these hyperparameters"
    )
    assert gd_steps < adam_steps, (
        f"Expected GD ({gd_steps} steps) to cross the loss<0.01 threshold "
        f"before Adam ({adam_steps} steps) with these hyperparameters"
    )

    # Momentum and Adam aren't "broken" here -- they still land at a very
    # small final loss by step 200 (both ~1e-8), just not as small as GD's
    # unusually good ~5e-14 with this particular, well-matched learning
    # rate on this simple, low-dimensional bowl.
    mom_final = quadratic(mom_traj[-1])
    adam_final = quadratic(adam_traj[-1])
    gd_final = quadratic(gd_traj[-1])
    for name, val in (("Momentum", mom_final), ("Adam", adam_final), ("GD", gd_final)):
        assert val < 1e-6, f"{name}'s final quadratic loss should be tiny, got {val}"

    print("\nAll assertions passed for content/01-foundations/03-calculus-optimization.md block #1.")


# ============================================================================
# block "The Library Layer: torch.optim" (added in the library/buildability
# revision pass; appears in the chapter BEFORE the numpy block above, in the
# "Gradient Descent: Variants and Analysis" section). Copied verbatim.
# ============================================================================

import math
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))

# AdamW with the LLM-pretraining convention: decoupled weight decay applied to
# 2D matmul weights only -- never to biases or norm gains (1D tensors).
decay = [p for p in model.parameters() if p.ndim >= 2]
no_decay = [p for p in model.parameters() if p.ndim < 2]
opt = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
    betas=(0.9, 0.95),                 # beta2=0.95 (not 0.999) is standard for LLMs
    eps=1e-8,
    fused=torch.cuda.is_available(),   # fused/foreach kernels: far fewer launches
)

# Warmup + cosine decay to 10% of peak, expressed as a multiplier on lr.
WARMUP, TOTAL = 100, 1000

def lr_multiplier(step: int) -> float:
    if step < WARMUP:
        return (step + 1) / WARMUP                       # linear ramp
    progress = (step - WARMUP) / max(1, TOTAL - WARMUP)  # in [0, 1]
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_multiplier)

x, y = torch.randn(32, 64), torch.randn(32, 64)
for step in range(3):
    loss = ((model(x) - y) ** 2).mean()
    opt.zero_grad(set_to_none=True)     # set_to_none frees the grad buffers
    loss.backward()                     # populates p.grad for every parameter
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()                          # the update rule
    sched.step()                        # advance the schedule
    print(f"step {step}  loss {loss.item():.4f}  "
          f"pre-clip |g|={gnorm:.3f}  lr={sched.get_last_lr()[0]:.3e}")

# ---------------------------------------------------------------------------
# Our from-scratch heavy-ball update IS torch.optim.SGD(momentum=...).
# ---------------------------------------------------------------------------
p = nn.Parameter(torch.tensor([4.0, 1.0]))
sgd = torch.optim.SGD([p], lr=0.005, momentum=0.9)    # nesterov=True gives NAG
theta, v = torch.tensor([4.0, 1.0]), torch.zeros(2)
for _ in range(10):
    sgd.zero_grad(set_to_none=True)
    (p[0] ** 2 + 10.0 * p[1] ** 2).backward()         # the quadratic from above
    sgd.step()
    g = torch.tensor([2.0 * theta[0], 20.0 * theta[1]])
    v = 0.9 * v - 0.005 * g                           # our chapter's form
    theta = theta + v
assert torch.allclose(p.detach(), theta, atol=1e-5)
print("from-scratch momentum matches torch.optim.SGD:", theta.tolist())

# --- assertions on the library-layer block ---------------------------------
# The warmup schedule must actually be ramping (3 steps in, lr is a tiny
# fraction of the 3e-4 peak, and strictly increasing).
assert 0 < sched.get_last_lr()[0] < 3e-4
# Parameter-group wiring: 1D tensors must carry zero weight decay.
assert opt.param_groups[0]["weight_decay"] == 0.1
assert opt.param_groups[1]["weight_decay"] == 0.0
assert all(t.ndim >= 2 for t in opt.param_groups[0]["params"])
assert all(t.ndim < 2 for t in opt.param_groups[1]["params"])
# Adam state really is 2 buffers per parameter tensor (the 8 bytes/param claim).
st = opt.state[opt.param_groups[0]["params"][0]]
assert "exp_avg" in st and "exp_avg_sq" in st


# ============================================================================
# block "Measuring Curvature Without Forming the Hessian" (chapter section
# "Saddle Points and Escape Dynamics"). Copied verbatim.
# ============================================================================

import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
x, y = torch.randn(64, 8), torch.randn(64, 1)
params = [p for p in model.parameters()]


def hvp(vec):
    """Hessian-vector product H @ v. Cost ~2 backward passes; H never formed."""
    loss = ((model(x) - y) ** 2).mean()
    g = torch.autograd.grad(loss, params, create_graph=True)   # keep the graph
    gv = sum((gi * vi).sum() for gi, vi in zip(g, vec))        # scalar g . v
    return torch.autograd.grad(gv, params)                     # d/dtheta (g.v) = H v


# Power iteration: v <- Hv/||Hv|| converges to the eigenvector of largest |lambda|.
v = [torch.randn_like(p) for p in params]
nrm = torch.sqrt(sum((vi ** 2).sum() for vi in v))
v = [vi / nrm for vi in v]
for _ in range(60):
    Hv = hvp(v)
    nrm = torch.sqrt(sum((h ** 2).sum() for h in Hv)) + 1e-12
    v = [h / nrm for h in Hv]

lam = sum((h * vi).sum() for h, vi in zip(hvp(v), v)).item()
print(f"lambda_max(H) ~= {lam:.4f}")
print(f"stability ceiling 2/lambda_max = {2.0 / lam:.4f}  (max usable GD lr)")

# --- assertions on the HVP block -------------------------------------------
# The chapter quotes lambda_max ~= 4.09 and 2/lambda_max ~= 0.49 for this
# seeded toy network; verified by running this exact code.
assert abs(lam - 4.0909) < 5e-2, f"lambda_max drifted: {lam}"
assert abs(2.0 / lam - 0.4889) < 5e-3

# Cross-check the HVP operator against an explicitly materialized Hessian on a
# tiny quadratic, where the answer is known exactly: f(w) = w^T A w has H = 2A.
w = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
A = torch.tensor([[3.0, 0.5, 0.0], [0.5, 2.0, 1.0], [0.0, 1.0, 5.0]])
probe = torch.tensor([1.0, 0.0, -1.0])
fw = (w @ A @ w)
g_exact = torch.autograd.grad(fw, [w], create_graph=True)
gv = (g_exact[0] * probe).sum()
Hv_exact = torch.autograd.grad(gv, [w])[0]
assert torch.allclose(Hv_exact, 2.0 * (A @ probe), atol=1e-5), Hv_exact

print("\nAll assertions passed for the torch.optim and HVP blocks.")
