# 1.3 Calculus, Optimization & Convexity

Training a neural network is fundamentally a calculus problem: given a scalar loss $\mathcal{L}(\theta)$ that measures how wrong our model is, find the $\theta$ that makes it as small as possible. Everything else — architectures, data pipelines, learning rate schedules — exists in service of that search. This chapter builds the mathematical scaffolding from first principles: how gradients generalize derivatives to high dimensions, what the Hessian tells you about the loss landscape, why the landscape is almost never convex in practice, and which algorithmic tricks let us navigate it efficiently anyway. We close with a worked numerical example, a full runnable implementation of gradient descent on a 2D loss surface, and an answer to one of the most satisfying questions in ML theory: why does SGD generalize?

Prerequisites: basic single-variable calculus and the matrix notation from [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html). We will build on these ideas directly in [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html) and [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html).

---

## Gradients, Jacobians, and Hessians

### The Gradient: Steepest Ascent in Parameter Space

Let $f : \mathbb{R}^n \to \mathbb{R}$ be a differentiable scalar-valued function. The **gradient** $\nabla_\theta f$ is the vector of partial derivatives:

$$
\nabla_\theta f = \begin{bmatrix} \frac{\partial f}{\partial \theta_1} \\ \frac{\partial f}{\partial \theta_2} \\ \vdots \\ \frac{\partial f}{\partial \theta_n} \end{bmatrix}
$$

The gradient has a geometric meaning that makes it indispensable: at any point $\theta$, $\nabla_\theta f$ points in the direction of steepest *increase* of $f$. Moving in the direction $-\nabla_\theta f$ therefore decreases $f$ most rapidly — that single insight is all of gradient descent.

The directional derivative of $f$ along unit vector $\mathbf{u}$ is $D_\mathbf{u} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos\phi$, where $\phi$ is the angle between $\mathbf{u}$ and $\nabla f$. It is maximized when $\phi = 0$ (moving along $\nabla f$) and minimized when $\phi = \pi$ (moving against $\nabla f$).

For a modern language model, $\theta \in \mathbb{R}^n$ with $n$ on the order of $10^9$ to $10^{12}$. The gradient is a vector of the same dimensionality — one scalar sensitivity per parameter.

{{fig:gradient-contour-steepest-descent}}

### The Jacobian: Gradient for Vector-Valued Maps

When the function maps $\mathbb{R}^n \to \mathbb{R}^m$, the first-order derivative is the **Jacobian matrix** $J \in \mathbb{R}^{m \times n}$:

$$
J_{ij} = \frac{\partial f_i}{\partial \theta_j}
$$

Each row $i$ of $J$ is the gradient of the $i$-th output with respect to all inputs. The gradient is the special case $m=1$, i.e., $J \in \mathbb{R}^{1 \times n}$ (a row vector; transposing gives the usual column-vector gradient).

In neural network backpropagation, the backward pass through each layer is essentially a Jacobian-vector product (JVP) or vector-Jacobian product (VJP). PyTorch's autograd computes VJPs by default (reverse-mode AD), which is efficient when $m \ll n$ — exactly the situation when we have a scalar loss. See [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html) for the full derivation.

### The Hessian: Second-Order Curvature

The **Hessian** $H \in \mathbb{R}^{n \times n}$ collects all second-order partial derivatives:

$$
H_{ij} = \frac{\partial^2 f}{\partial \theta_i \, \partial \theta_j}
$$

The Hessian is symmetric (under mild regularity conditions, by Clairaut's theorem). Its eigenvalues characterize the local curvature of the loss:

- Positive eigenvalues → the loss curves upward in that direction (locally convex).
- Negative eigenvalues → the loss curves downward (local maximum in that direction).
- Zero eigenvalues → flat directions.

For a network with $n = 10^9$ parameters, storing $H$ explicitly requires $n^2 \approx 10^{18}$ floats — physically impossible. This is why second-order optimizers for large networks either approximate $H$ (L-BFGS) or work with diagonal/structured approximations (Shampoo, K-FAC). We revisit these in [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

---

## The Chain Rule and Backpropagation

The **chain rule** is the engine of all gradient computation in deep learning. For composite functions $z = f(g(x))$:

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$

In the multivariate case, let $\mathbf{y} = g(\mathbf{x})$ and $z = f(\mathbf{y})$. Then:

$$
\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}
$$

which in matrix form is the VJP $\frac{\partial z}{\partial \mathbf{x}} = J_g^\top \frac{\partial z}{\partial \mathbf{y}}$.

For a deep network, the loss $\mathcal{L}$ is a composition of $L$ layer functions. Unwinding the chain rule layer by layer — from the output back to the input — is backpropagation. The key efficiency insight is that each intermediate Jacobian never needs to be materialized; we only need the VJP, which costs $O(n)$ time per layer rather than $O(n^2)$.

### A Three-Layer Chain Rule Worked Out

Suppose $\mathcal{L} = \text{MSE}(W_3 \sigma(W_2 \sigma(W_1 x)))$. Define:

$$
h_1 = \sigma(W_1 x), \quad h_2 = \sigma(W_2 h_1), \quad \hat{y} = W_3 h_2, \quad \mathcal{L} = \|\hat{y} - y\|^2
$$

Backward pass (pseudocode logic):

$$
\delta_3 = 2(\hat{y} - y), \quad \frac{\partial \mathcal{L}}{\partial W_3} = \delta_3 h_2^\top
$$

$$
\delta_2 = W_3^\top \delta_3 \odot \sigma'(W_2 h_1), \quad \frac{\partial \mathcal{L}}{\partial W_2} = \delta_2 h_1^\top
$$

$$
\delta_1 = W_2^\top \delta_2 \odot \sigma'(W_1 x), \quad \frac{\partial \mathcal{L}}{\partial W_1} = \delta_1 x^\top
$$

The backward pass mirrors the forward pass in structure — each layer's contribution is gated by its local Jacobian — and costs roughly the same FLOPs as the forward pass.

---

## Convex vs Non-Convex Landscapes

### Convexity Defined

A set $\mathcal{C}$ is **convex** if for any two points $x, y \in \mathcal{C}$ and $\lambda \in [0,1]$, the interpolant $\lambda x + (1-\lambda)y \in \mathcal{C}$. A function $f$ is **convex** if its domain is convex and:

$$
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)
$$

Equivalently, $f$ is convex if $\nabla^2 f \succeq 0$ everywhere (positive semidefinite Hessian). For convex functions, any local minimum is a global minimum — gradient descent with a suitable step size is guaranteed to converge.

Logistic regression and linear regression with squared loss are convex. Almost every useful deep learning objective is **not** convex.

### Why Neural Networks Are Non-Convex

Several structural reasons conspire to make neural loss landscapes non-convex:

1. **Permutation symmetry.** Swapping two hidden units in a layer with corresponding weight adjustments yields an identical function but a different parameter vector. This creates $n!$ equivalent global optima for a layer with $n$ units — the loss surface has many valleys, not one.

2. **Composition of non-linearities.** Even a simple two-layer ReLU network has regions where the loss surface is neither convex nor concave.

3. **Saddle points at scale.** In high-dimensional spaces, critical points (where $\nabla \mathcal{L} = 0$) are overwhelmingly likely to be saddle points rather than local minima. This was formalized by Dauphin et al. (2014) using random matrix theory: a random function of $n$ variables has local minima that are exponentially rare; most critical points have some directions curving up and some curving down.

4. **Flat regions and loss plateaus.** Dead ReLU neurons, saturating sigmoids, and residual scaling can all create large flat regions where $\nabla \mathcal{L} \approx 0$ but the current point is far from a minimum.

### What Saves Us

Empirically, large neural networks trained with SGD find solutions that generalize well, despite the non-convexity. Several phenomena explain this:

- **Overparameterization:** When $n \gg \text{data size}$, the loss landscape has many equivalent minima and gradient descent finds one efficiently.
- **SGD noise** (see the Interview Corner below) biases toward wider minima.
- **The loss landscape is actually well-behaved locally** for modern architectures: residual connections and layer normalization smooth the surface dramatically (Li et al., *Visualizing the Loss Landscape of Neural Nets*, 2018).

```text
Convex landscape:          Non-convex landscape (schematic):

Loss                        Loss
 |                           |
 |     U-shaped              |  /\  saddle  /\
 |   /        \              | /  \___/\___/  \
 |  /          \             |/    local min   \
 +--x*-----------> θ        +--------------------> θ
   one global min            many local min + saddles
```

{{fig:loss-landscape-convex-vs-nonconvex}}

---

## Gradient Descent: Variants and Analysis

### Vanilla Gradient Descent (GD)

The update rule is simple: move opposite to the gradient, scaled by learning rate $\eta$:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

For a convex function with $L$-Lipschitz gradient (see section on Lipschitz constants below), GD with $\eta \leq 1/L$ converges at rate $O(1/t)$. For strongly convex functions (Hessian eigenvalues $\geq \mu > 0$), it converges exponentially: $O(\exp(-\mu t / L))$.

**Full-batch GD** computes the gradient over the entire dataset, which is exact but absurdly expensive for large datasets (a billion-token corpus requires a full pass per step).

### Stochastic Gradient Descent (SGD)

**SGD** replaces the full gradient with a noisy estimate computed on a single sample (or a small mini-batch of $B$ samples):

$$
\nabla_\theta \hat{\mathcal{L}} = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \ell(\theta; x_i, y_i)
$$

This is an unbiased estimator of the true gradient: $\mathbb{E}[\nabla_\theta \hat{\mathcal{L}}] = \nabla_\theta \mathcal{L}$. Variance scales as $O(1/B)$.

The practical magic of SGD:
- Each step costs $O(B)$ rather than $O(N)$, enabling many more updates per wall-clock second.
- The gradient noise acts as implicit regularization (more on this in the Interview Corner).
- It escapes sharp minima and saddle points more easily than full-batch GD.

### SGD with Momentum

Pure SGD can oscillate and converge slowly in narrow valleys. **Heavy-ball momentum** (Polyak, 1964) introduces a velocity term:

$$
v_{t+1} = \beta v_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

$$
\theta_{t+1} = \theta_t + v_{t+1}
$$

With $\beta = 0.9$, each update is an exponentially weighted moving average of past gradients. In the direction of consistent gradient signal, velocity accumulates; in oscillating directions, gradients cancel. The effective learning rate in a consistent direction is $\eta / (1 - \beta)$ — with $\beta = 0.9$, it is $10\eta$.

**Nesterov Accelerated Gradient (NAG)** computes the gradient at the "lookahead" position rather than the current position:

$$
v_{t+1} = \beta v_t - \eta \nabla_\theta \mathcal{L}(\theta_t + \beta v_t)
$$

$$
\theta_{t+1} = \theta_t + v_{t+1}
$$

This provides a corrective anticipation and achieves the optimal $O(1/t^2)$ convergence rate for convex functions (versus $O(1/t)$ for GD) — the first-order oracle lower bound.

### Adaptive Learning Rate Methods

**AdaGrad** (Duchi et al., 2011) accumulates squared gradients to scale the learning rate per parameter:

$$
G_t = G_{t-1} + (\nabla_\theta \mathcal{L})^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla_\theta \mathcal{L}
$$

Parameters with historically large gradients get smaller effective learning rates. This is powerful for sparse features but the accumulating denominator means the effective learning rate shrinks to zero over time.

**RMSProp** fixes this with an exponential moving average:

$$
v_t = \rho v_{t-1} + (1-\rho)(\nabla_\theta \mathcal{L})^2
$$

**Adam** (Kingma & Ba, 2015) combines momentum (first moment) with RMSProp (second moment):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Typical defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 3 \times 10^{-4}$. The bias-correction terms ($\hat{m}_t, \hat{v}_t$) are crucial in early training when the moving averages have not yet warmed up.

Adam is the workhorse optimizer for LLM pretraining. Its variants (AdamW, Adafactor, Lion) are covered in depth in [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

### Summary of Variants

| Method | Per-step cost | Memory | Adapts per-param LR? | Typical use |
|---|---|---|---|---|
| GD | $O(N)$ | $O(n)$ | No | Small convex problems |
| SGD | $O(B)$ | $O(n)$ | No | Large-scale / online |
| SGD+Momentum | $O(B)$ | $O(2n)$ | No | CV training |
| AdaGrad | $O(B)$ | $O(2n)$ | Yes (cumulative) | Sparse features |
| RMSProp | $O(B)$ | $O(2n)$ | Yes (EMA) | RNN training |
| Adam | $O(B)$ | $O(3n)$ | Yes (EMA) | LLM pretraining |

---

## Saddle Points and Escape Dynamics

In a $n$-dimensional landscape, a **saddle point** is a critical point ($\nabla \mathcal{L} = 0$) where the Hessian has both positive and negative eigenvalues. By random matrix theory arguments (Choromanska et al., 2015; Dauphin et al., 2014), the fraction of critical points that are true local minima decreases exponentially with $n$. For $n \sim 10^9$, essentially all critical points are saddle points.

### How SGD Escapes Saddle Points

Near a strict saddle (one with a negative eigenvalue $\lambda_{\min} < 0$), the gradient noise in SGD will eventually kick the trajectory into the descent direction. The escape time scales as $O(1/|\lambda_{\min}|)$ — fast escape from sharp saddles, slow from near-flat ones.

**Gradient perturbation tricks** add explicit noise: $\theta \leftarrow \theta + \xi$ where $\xi \sim \mathcal{N}(0, \sigma^2 I)$ every $k$ steps. This is occasionally used in practice to escape persistent flat regions.

**Negative curvature descent** (used in some second-order methods) explicitly computes the minimum-curvature direction of the Hessian and steps along it. This escapes saddles in $O(1)$ steps but requires Hessian-vector products, making it expensive.

### Flat Regions vs Sharp Minima

Not all minima are equal. The **sharpness** of a minimum is captured by the maximum eigenvalue of the Hessian $\lambda_{\max}(H)$ at that point. Sharp minima (large $\lambda_{\max}$) generalize worse than flat minima (small $\lambda_{\max}$) — intuitively, a flat basin is robust to small perturbations in $\theta$, while a sharp spike requires precision.

This observation (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017) motivates Sharpness-Aware Minimization (SAM), which explicitly penalizes sharp minima, and also explains why large-batch SGD (which has lower gradient noise) tends to find sharper minima than small-batch SGD.

{{fig:flat-vs-sharp-minima-generalization}}

---

## Lipschitz Constants and the Condition Number

### $L$-Smooth Functions

A function $f$ has an $L$-**Lipschitz gradient** (is $L$-smooth) if:

$$
\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\| \quad \forall x, y
$$

Equivalently, $\lambda_{\max}(H) \leq L$ everywhere. The gradient doesn't change "too fast." This is the condition required for gradient descent to make guaranteed progress: with $\eta \leq 1/L$, each step decreases the loss by at least $\|\nabla f\|^2 / (2L)$.

If $\eta > 1/L$, GD can overshoot and diverge. A common rule of thumb: start with $\eta = 1/L$ and use a learning rate finder or warmup (see [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html)) to tune from there.

### The Condition Number

For strongly convex functions (Hessian eigenvalues bounded below by $\mu > 0$), the **condition number** $\kappa = L / \mu$ governs convergence:

- GD converges at rate $\left(1 - \frac{\mu}{L}\right)^t = \left(1 - \frac{1}{\kappa}\right)^t$.
- With Nesterov momentum, the rate improves to $\left(1 - \frac{1}{\sqrt{\kappa}}\right)^t$.

A condition number of $\kappa = 100$ means GD takes about $100 \ln(1/\epsilon)$ steps while Nesterov takes $10 \ln(1/\epsilon)$ — a $10\times$ speedup.

For neural networks, the condition number of the loss Hessian can be in the thousands or higher, which is one reason adaptive methods like Adam work better in practice: they implicitly precondition the gradient by the square root of the second moment, effectively rescaling each parameter direction to have condition number closer to 1.

{{fig:optimizer-paths-conditioning}}

### Layer-Wise Conditioning

A residual network with skip connections has a better-conditioned loss landscape than a plain deep network (He et al., 2016). The gradient signal passes directly through skip connections without squashing by weight matrices, keeping $\lambda_{\min}(H)$ from collapsing to zero (vanishing gradients). This is a geometric explanation for why residual networks are easier to train: they reduce the condition number of the optimization problem.

---

## Worked Numerical Example: GD on a Quadratic

!!! example "Gradient Descent on a 2D Quadratic"
    Consider the loss surface $\mathcal{L}(\theta_1, \theta_2) = \theta_1^2 + 10 \theta_2^2$. This is a bowl stretched by 10x in the $\theta_2$ direction — it has condition number $\kappa = 10 / 1 = 10$.

    The Hessian is $H = \text{diag}(2, 20)$, so $L = 20$, $\mu = 2$.

    **Learning rate choice:** $\eta = 1/L = 0.05$.

    **Starting point:** $\theta_0 = (4, 1)$.

    **Step 1:**
    $$
    \nabla \mathcal{L}(4, 1) = (2 \cdot 4, \; 20 \cdot 1) = (8, 20)
    $$
    $$
    \theta_1 = (4 - 0.05 \cdot 8, \; 1 - 0.05 \cdot 20) = (3.6, 0.0)
    $$
    $$
    \mathcal{L}(\theta_1) = 3.6^2 + 10 \cdot 0^2 = 12.96 \quad \text{(was } 16 + 10 = 26\text{)}
    $$

    **Step 2:**
    $$
    \nabla \mathcal{L}(3.6, 0) = (7.2, 0)
    $$
    $$
    \theta_2 = (3.6 - 0.05 \cdot 7.2, \; 0) = (3.24, 0)
    $$
    $$
    \mathcal{L}(\theta_2) = 3.24^2 = 10.50
    $$

    After step 1 the $\theta_2$ component (high-curvature direction) is already zeroed out. The remaining convergence is geometric: $\mathcal{L}$ decreases by factor $(1 - 2 \cdot 0.05)^2 = 0.81$ per step. To reach $\mathcal{L} < 0.01$ from the initial loss $\mathcal{L}(\theta_0) = 4^2 + 10 \cdot 1^2 = 26$: $t \geq \log(0.01/26) / \log(0.81) \approx 38$ steps. (Counting instead from the post-step-1 loss of 12.96 gives $\approx 34$ *additional* steps — the same order of magnitude, since step 1 already did most of the work by zeroing the high-curvature component.)

    With Nesterov (rate $O(1/\sqrt{\kappa})$): $\approx 12$ steps to the same tolerance — $3\times$ faster.

---

## Full Runnable Code: Gradient Descent on a Toy Loss Surface

The code below implements vanilla GD, SGD, momentum, and Adam from scratch on a 2D landscape, then generates "plots as data" (JSON-serializable trajectories you can feed into any visualization library or print directly).

```python
"""
Gradient descent visualization on a 2D loss surface.
Implements GD, SGD (with noise), Momentum, and Adam from scratch.
No dependencies except numpy.
"""

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


if __name__ == "__main__":
    main()
```

Running this script produces output like:

```text
=== Rosenbrock surface (non-convex) ===
  Starting loss: 312.5000
  Global minimum at (1, 1), loss = 0

  GD            final_loss=0.053318  theta=(0.7693, 0.5908)
  Momentum      final_loss=0.000000  theta=(0.9999, 0.9998)
  Adam          final_loss=2.681244  theta=(-0.6368, 0.4102)
  SGD+Noise     final_loss=0.053240  theta=(0.7695, 0.5911)

=== Quadratic surface (condition number κ=10) ===
  GD          steps to loss<0.01: 45
  Momentum    steps to loss<0.01: 60
  Adam        steps to loss<0.01: 61
```

(Note: with $\theta_0 = (-1.5, 0.5)$ the actual starting Rosenbrock loss is $(1-(-1.5))^2 + 100(0.5-(-1.5)^2)^2 = 6.25 + 306.25 = 312.5$, not $6.25$ — the smaller term alone is only the $(1-x)^2$ piece.)

These numbers are worth sitting with, because they cut against the "Adam always wins" intuition. On the Rosenbrock surface, **momentum** — not Adam — reaches the global minimum almost exactly, because the curved valley rewards accumulating velocity along a consistent direction. Adam actually does *worse* than plain GD here: its per-coordinate normalization by $\sqrt{\hat v_t}$ treats $x$ and $y$ independently, which fights against Rosenbrock's tightly coupled, curved valley, and with these particular (untuned) hyperparameters it stalls well short of the minimum. On the quadratic, plain GD wins on *both* metrics with these hyperparameters: it not only reaches the loss $<0.01$ threshold first (45 steps vs. 60–61), it also ends up at a lower final loss after 200 steps ($5\times 10^{-14}$ vs. $\sim 10^{-8}$ for momentum and Adam) — its learning rate happens to be unusually well-matched to this simple, low-dimensional bowl, while momentum and Adam overshoot and oscillate for the first several dozen steps (visible if you print the early loss curve) before settling down. The lesson: the textbook convergence-rate rankings (Nesterov beats GD by $O(\sqrt{\kappa})$, Adam preconditions ill-conditioned directions) describe *asymptotic, well-tuned* behavior and are landscape-dependent — naive default learning rates do not automatically realize those speedups on every problem, and which optimizer "wins" for a given step budget and hyperparameter choice is an empirical question you have to check on your actual loss surface, not something to assume from the algorithm's name.

---

## Why SGD Generalizes: The Interview Question

!!! interview "Interview Corner"
    **Q:** Stochastic Gradient Descent uses a noisier, less accurate gradient estimate than full-batch gradient descent. Why does this noise actually *help* generalization on test data?

    **A:** There are several complementary explanations, each backed by theory or empirical evidence:

    1. **Implicit bias toward flat minima.** SGD's gradient noise effectively prevents the optimizer from settling into sharp, narrow minima of the training loss. Flat minima correspond to solutions that are robust to small perturbations in the weights — and weight perturbations are analogous to distribution shift or sampling variability. A flat minimum generalizes better because nearby parameter vectors have similar loss (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017).

    2. **Langevin dynamics interpretation.** With a constant learning rate $\eta$ and gradient noise of variance $\sigma^2$, SGD approximates a continuous-time stochastic process called Stochastic Gradient Langevin Dynamics (SGLD). The stationary distribution of this process concentrates around minima proportionally to $\exp(-\mathcal{L}/(\eta \sigma^2))$. Wide, low-loss basins have more volume and therefore higher stationary probability — SGD spends more time in flat regions.

    3. **Regularization via path length.** Noisier optimization paths tend to explore a wider region of parameter space before converging, effectively averaging over many candidate solutions. This is related to ensemble methods, where averaging improves generalization.

    4. **Empirical confirmation.** Keskar et al. (2017) directly demonstrated that increasing batch size degrades test accuracy on CIFAR and ImageNet (up to $\sim$1-2% on the setups tested) without changing train accuracy, attributing the gap to sharper minima found by large-batch SGD. Techniques like learning rate warmup and linear learning rate scaling recover some of this gap but not all.

    5. **PAC-Bayes framing.** The generalization gap can be bounded in terms of the "volume" of good parameters consistent with the training data. Wide minima correspond to larger volumes, giving tighter PAC-Bayes bounds.

    In an interview, lead with points 1 and 4 (most concrete), then offer the Langevin connection as a theoretical justification if asked for depth.

---

## Learning Rate, Convergence, and Practical Guidance

### The Learning Rate as the Most Important Hyperparameter

The learning rate $\eta$ controls the scale of each update. Too large: the loss diverges or oscillates. Too small: training is impossibly slow. For a quadratic with maximum curvature $L$, the critical threshold is exactly $\eta < 2/L$.

**Learning rate warmup** is standard for LLM training: start with $\eta \approx 0$ and linearly ramp to the target over the first few hundred to few thousand steps. This matters because at initialization, gradients are large and noisy — a large learning rate early in training frequently causes loss spikes. Warmup effectively applies a smaller effective learning rate while the model is in its most sensitive initial state.

**Cosine annealing** then decays $\eta$ to near zero following a cosine curve:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)
$$

The gradual decay helps the optimizer settle into a flat minimum rather than oscillating around it. The full landscape of schedules (linear decay, polynomial, constant-then-cosine, etc.) is covered in [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

### Gradient Clipping

Large gradient norms can destabilize training, especially early in training or after a data spike. **Global gradient norm clipping** rescales the gradient if its norm exceeds a threshold $\tau$:

$$
g \leftarrow g \cdot \min\!\left(1, \frac{\tau}{\|g\|_2}\right)
$$

This is not a cure for bad initialization or bad data, but it prevents one catastrophically large step from derailing an otherwise healthy run. Typical values: $\tau \in [0.5, 5.0]$, with $\tau = 1.0$ common for LLM training.

### Connection to Second-Order Methods

The ideal update would be $\theta \leftarrow \theta - H^{-1} \nabla \mathcal{L}$ (Newton's method), which solves the local quadratic approximation exactly. This converges in $O(\log(1/\epsilon))$ steps regardless of condition number — it completely handles ill-conditioning. The problem is cost: inverting $H$ for $n = 10^9$ is $O(n^3)$, completely intractable.

**Quasi-Newton methods** (L-BFGS) approximate $H^{-1}$ using the last $k \approx 20$ gradient differences, costing $O(kn)$ per step. They work well for small-to-medium networks but are rarely used for LLMs.

**Distributed Shampoo** (Anil et al., 2020) maintains full matrix preconditioners per layer using Kronecker products, achieving near-second-order convergence at manageable memory cost. It has been used in production at Google for pretraining runs. See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) for details.

---

## Putting It Together: The Optimization Landscape of an LLM

Training a large language model is not just applying Adam to a convex problem. The loss landscape has structure that the practitioner must respect:

1. **Early training (steps 0–1000):** High gradient variance, potential for large spikes. Learning rate warmup and gradient clipping are essential. The Hessian has many near-zero eigenvalues (the network is massively under-utilized early).

2. **Mid training (bulk of steps):** The loss landscape is well-behaved near a low-loss manifold. Adam efficiently navigates the per-parameter curvature variations. The condition number is high in some directions (embedding layers, which see many more gradient updates for common tokens) and low in others.

3. **Late training / convergence:** The learning rate decay helps the optimizer settle into a flat basin. The final loss is determined not by the optimization algorithm per se, but by the flatness of the basin found — a generalization property.

4. **Loss spikes:** Occasional sharp increases in loss, sometimes by $2-10\times$, can result from bad batches, gradient accumulation bugs, or instability in normalization layers. Monitoring the gradient norm, the Adam second moment, and the per-layer update magnitudes is key. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

The full probabilistic picture of why the loss landscape is navigable — the statistical mechanics viewpoint, the role of overparameterization, and the relationship to the Neural Tangent Kernel (NTK) — goes beyond our scope here. For the practitioner, the takeaways from optimization theory are: use Adam, clip gradients, warm up the learning rate, and monitor training closely.

---

!!! key "Key Takeaways"
    - The **gradient** $\nabla_\theta \mathcal{L}$ is a vector pointing in the direction of steepest increase; gradient descent steps opposite to it. The gradient of an $n$-parameter model is itself an $n$-dimensional vector.
    - The **Jacobian** generalizes the gradient to vector-valued functions ($\mathbb{R}^n \to \mathbb{R}^m$); backpropagation computes vector-Jacobian products efficiently via the chain rule.
    - The **Hessian** captures second-order curvature. Its eigenvalue spectrum determines whether a critical point is a minimum, maximum, or saddle. For large models, the Hessian is never formed explicitly — approximations (Adam's second moment, Shampoo's Kronecker factors) are used instead.
    - Neural network loss surfaces are **non-convex**: they have permutation symmetry, saddle points, and flat regions. Most critical points are saddles, not local minima, especially in high dimension.
    - **SGD momentum** accumulates velocity in consistent gradient directions, accelerating convergence in narrow valleys. The theoretical speedup over GD is $O(\sqrt{\kappa})$ via Nesterov acceleration.
    - **Adam** applies per-parameter adaptive learning rates, effectively precondition the gradient by the square root of the second moment. This dramatically reduces sensitivity to the condition number and is the standard optimizer for LLM pretraining.
    - The **condition number** $\kappa = L/\mu$ governs how many steps are needed to converge. Well-conditioned problems ($\kappa \approx 1$) converge fast; ill-conditioned ones ($\kappa \gg 1$) benefit from preconditioning.
    - **SGD generalizes better than large-batch GD** because gradient noise biases the optimizer toward flat minima, which are more robust to distribution shift.
    - **Gradient clipping** and **learning rate warmup** are practical necessities for stable LLM training, preventing early-phase instability from derailing otherwise healthy runs.

!!! sota "State of the Art & Resources (2026)"
    Optimization for deep learning is a mature field with well-established foundations — the core algorithms (SGD, Adam, Nesterov) are from the 1960s–2010s — but active research continues on flat-minima geometry, sharpness-aware methods, and second-order approximations for billion-parameter models.

    **Textbooks & courses**

    - [Boyd & Vandenberghe, *Convex Optimization* (2004)](https://stanford.edu/~boyd/cvxbook/) — the definitive reference on convexity, duality, and gradient methods; free PDF from the authors.
    - [Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* (2020)](https://mml-book.github.io/) — Chapter 7 covers continuous optimization; free PDF with Jupyter notebooks.

    **Foundational papers**

    - [Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015)](https://arxiv.org/abs/1412.6980) — introduces Adam with bias-corrected moment estimates; the dominant LLM pretraining optimizer.
    - [Dauphin et al., *Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization* (2014)](https://arxiv.org/abs/1406.2572) — establishes via random matrix theory that critical points in deep networks are overwhelmingly saddles, not local minima.
    - [Keskar et al., *On Large-Batch Training for Deep Learning* (2017)](https://arxiv.org/abs/1609.04836) — empirical proof that large-batch SGD finds sharper minima and generalizes worse; motivates gradient noise as implicit regularization.
    - [Li et al., *Visualizing the Loss Landscape of Neural Nets* (2018)](https://arxiv.org/abs/1712.09913) — filter-normalized 2D projections show how skip connections smooth the loss surface.

    **Recent advances (2021–2026)**

    - [Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization* (2021)](https://arxiv.org/abs/2010.01412) — SAM adds a min-max perturbation step to explicitly seek flat minima; consistent gains across vision and NLP benchmarks.

    **Open-source & tools**

    - [tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape) — PyTorch code to compute and plot 1D/2D loss-surface slices for any model; companion to the Li et al. NeurIPS 2018 paper.

    **Go deeper**

    - [Ruder, *An Overview of Gradient Descent Optimization Algorithms* (2016)](https://www.ruder.io/optimizing-gradient-descent/) — comprehensive illustrated survey of SGD, Momentum, AdaGrad, RMSProp, Adam, and friends.
    - [Goh, *Why Momentum Really Works* — Distill (2017)](https://distill.pub/2017/momentum/) — interactive visual explainer on momentum through the lens of convex quadratics; builds deep intuition for the $\sqrt{\kappa}$ speedup.
    - [3Blue1Brown, *Backpropagation Calculus* (2017)](https://www.3blue1brown.com/lessons/backpropagation-calculus) — animated walkthrough of the chain rule in neural networks; the clearest visual introduction available.

## Further Reading

- Ruder, Sebastian. "An Overview of Gradient Descent Optimization Algorithms." arXiv, 2016. A comprehensive survey of SGD variants with intuition-building diagrams.
- Kingma, Diederik P. and Ba, Jimmy. "Adam: A Method for Stochastic Optimization." ICLR 2015. The original Adam paper; essential reading.
- Dauphin, Yann N., et al. "Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization." NeurIPS 2014. Establishes why saddle points dominate over local minima in deep networks.
- Keskar, Nitish Shirish, et al. "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." ICLR 2017. Empirical and theoretical case that large-batch SGD finds sharper minima.
- Li, Hao, et al. "Visualizing the Loss Landscape of Neural Nets." NeurIPS 2018. Shows how skip connections smooth the loss landscape.
- Nesterov, Yurii. "A Method of Solving a Convex Programming Problem with Convergence Rate $O(1/k^2)$." Soviet Mathematics Doklady, 1983. The original Nesterov acceleration paper.
- Polyak, Boris T. "Some Methods of Speeding Up the Convergence of Iteration Methods." USSR Computational Mathematics and Mathematical Physics, 1964. Origin of heavy-ball momentum.
- Boyd, Stephen and Vandenberghe, Lieven. *Convex Optimization*. Cambridge University Press, 2004. The definitive reference on convexity, duality, and gradient methods. Available free online.
