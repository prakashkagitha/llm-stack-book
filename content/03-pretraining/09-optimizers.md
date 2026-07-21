# 3.9 Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo

The optimizer is the engine that turns gradients into weight updates. Choose it well and a 70-billion-parameter model converges smoothly on a fixed token budget; choose it poorly and you either waste GPU-months or watch the loss spike into NaNs. For most of the deep-learning era one optimizer — Adam, and its weight-decay-fixed cousin AdamW — has been the default for training transformers, and for good reason. But Adam carries a hidden tax: it stores **two extra full-precision tensors per parameter**, and at scale that memory cost rivals the model weights themselves. That tension — *fast, robust convergence* versus *memory and compute overhead* — is the throughline of this chapter, and it is exactly what newer optimizers like Adafactor, Lion, Shampoo, and Muon attack from different angles.

We will build up from the gradient-descent first principles you saw in [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html), derive Adam and its bias correction carefully, implement AdamW from scratch, account for optimizer-state memory the way a systems engineer must, and then tour the modern menagerie: factored second moments (Adafactor), sign-based updates (Lion), full preconditioners (Shampoo), and the newest newsmaker, **Muon**, which orthogonalizes the update of every weight matrix. By the end you should be able to pick an optimizer for a given memory and stability budget, and defend that choice in an interview.

## From Gradient Descent to Momentum

The starting point is plain stochastic gradient descent (SGD). Given a loss $L(\theta)$ and a minibatch estimate $g_t = \nabla_\theta L_{\mathcal{B}_t}(\theta_{t-1})$ of its gradient, SGD takes the step

$$
\theta_t = \theta_{t-1} - \eta\, g_t,
$$

where $\eta$ is the learning rate. This is the cheapest possible optimizer: zero extra state, one tensor of gradients, one fused multiply-subtract. Its weakness is that the raw gradient is a noisy, badly-scaled descent direction. In a ravine — a loss surface that is steep across one axis and nearly flat along another, which is the *typical* geometry of deep nets — SGD zig-zags across the steep walls while crawling along the flat valley floor. The condition number of the Hessian, the ratio of its largest to smallest eigenvalue, controls how bad this is; for transformers it can be enormous.

**Momentum** fixes the zig-zag by accumulating an exponentially-weighted average of past gradients, a "velocity" $v_t$, and stepping along *that* instead:

$$
v_t = \mu\, v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\, v_t.
$$

Here $\mu \in [0,1)$ (typically 0.9) is the momentum coefficient. The intuition is physical: the velocity behaves like a heavy ball rolling downhill. Oscillating components of the gradient cancel across steps, while the consistent down-valley component accumulates. The geometric-series identity tells us that a steady gradient $g$ produces a terminal velocity of $g/(1-\mu)$, so momentum with $\mu=0.9$ effectively multiplies the step length along consistent directions by $10\times$. That is also why you usually drop the learning rate when you increase momentum.

A subtle but important variant is **Nesterov momentum**, which evaluates the gradient at the *look-ahead* point $\theta_{t-1} - \eta\mu v_{t-1}$ rather than at $\theta_{t-1}$. The correction term gives a slightly more responsive update and a better convergence rate on convex problems. SGD with Nesterov momentum and a well-tuned learning-rate schedule remains the gold standard for training convolutional vision models, and it generalizes beautifully. So why do we not train transformers with it?

```python
import torch

def sgd_momentum_step(params, grads, velocities, lr=0.1, mu=0.9, nesterov=False):
    """One step of (Nesterov) momentum SGD, in-place. Pure PyTorch tensors."""
    for p, g, v in zip(params, grads, velocities):
        v.mul_(mu).add_(g)              # v <- mu*v + g
        if nesterov:
            update = g.add(v, alpha=mu)  # g + mu*v  (look-ahead)
        else:
            update = v
        p.add_(update, alpha=-lr)        # theta <- theta - lr * update
```

The answer is scale heterogeneity. In a transformer the gradient magnitudes differ wildly across parameters — embedding rows that fire rarely versus LayerNorm gains versus attention projections — and a *single* global learning rate cannot serve all of them. We need a **per-parameter adaptive** learning rate. That is what Adam provides.

!!! note "Aside: SGD's generalization edge"
    A persistent empirical finding is that SGD-trained models often generalize slightly better than adaptive ones, plausibly because the implicit regularization of SGD's noise biases solutions toward flatter minima. For LLMs this edge is outweighed by Adam's vastly faster and more robust convergence on the ill-conditioned, sparse-gradient landscape of language. The frontier optimizers later in this chapter (Muon, Shampoo) are partly attempts to recover both: adaptive speed *and* SGD-like generalization.

## Adam and AdamW: Derivation, Bias Correction, Decoupled Decay

Adam (Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2015) combines two ideas: momentum on the gradient (the **first moment**), and a per-coordinate rescaling by the running magnitude of the gradient (the **second moment**, an idea inherited from RMSProp and AdaGrad). It maintains two exponential moving averages:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$

where $g_t^2$ is elementwise. Typical defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$ (for LLMs $\beta_2 = 0.95$ is common — more on that below). $m_t$ estimates $\mathbb{E}[g]$ and $v_t$ estimates $\mathbb{E}[g^2]$.

### Why bias correction is necessary

Both averages are initialized at zero, which biases them toward zero in early steps. Unroll $v_t$ assuming a stationary gradient distribution:

$$
v_t = (1-\beta_2)\sum_{i=1}^{t} \beta_2^{\,t-i}\, g_i^2.
$$

Taking expectations and assuming $\mathbb{E}[g_i^2]$ is approximately constant $\approx \mathbb{E}[g_t^2]$,

$$
\mathbb{E}[v_t] = \mathbb{E}[g_t^2]\,(1-\beta_2)\sum_{i=1}^{t}\beta_2^{\,t-i} = \mathbb{E}[g_t^2]\,(1 - \beta_2^{\,t}).
$$

The factor $(1-\beta_2^{\,t})$ is the bias: at $t=1$ with $\beta_2=0.999$, $v_1$ is only $0.1\%$ of the true second moment, so an *uncorrected* step would be roughly $\sqrt{1000}\approx 31\times$ too large. Adam divides it out:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}}.
$$

The final update normalizes the first moment by the root second moment, with a small $\epsilon$ (e.g. $10^{-8}$) for numerical safety:

$$
\theta_t = \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$

The deep insight is in the ratio $\hat{m}_t / \sqrt{\hat{v}_t}$. For a coordinate with a consistent gradient, $\hat{m}_t \approx \sqrt{\hat{v}_t}$, so the update magnitude is $\approx \eta$ — Adam takes a step of *roughly unit size in each coordinate*, regardless of the gradient's absolute scale. This is the **scale-invariance** that makes Adam so robust: you can change the loss scaling (or use mixed precision, see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)) and Adam's effective step is unchanged. It is also why Adam tolerates the wildly heterogeneous gradient scales of a transformer that defeat plain SGD.

{{fig:opt-adam-scale-invariance}}

### Decoupled weight decay: the W in AdamW

L2 regularization adds $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ to the loss, which contributes $\lambda\theta$ to the gradient. In *plain* SGD this is identical to weight decay (shrinking $\theta$ toward zero each step). But in Adam they are **not** equivalent: the $\lambda\theta$ term flows through the $m_t$ and $v_t$ machinery and gets divided by $\sqrt{\hat v_t}$, so parameters with large gradient magnitude (large $v_t$) get *less* decay — exactly backwards from what you want. Loshchilov & Hutter (*Decoupled Weight Decay Regularization*, 2019) showed this hurts, and proposed **AdamW**, which applies the decay directly to the weights, decoupled from the adaptive term:

$$
\theta_t = \theta_{t-1} - \eta\left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\, \theta_{t-1}\right).
$$

Equivalently $\theta_t = (1-\eta\lambda)\,\theta_{t-1} - \eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$: a clean multiplicative shrink toward zero plus the adaptive step. AdamW is the de-facto standard for pretraining every modern LLM. A crucial practical detail: **do not decay 1-D parameters** — biases, LayerNorm/RMSNorm gains, and usually embeddings — only the 2-D weight matrices. Decaying norm gains pulls them toward zero and destabilizes training.

### Implementing AdamW from scratch

Here is a complete, correct, heavily-commented AdamW that you could drop into a real training loop. It mirrors the math above and matches `torch.optim.AdamW` numerically.

```python
import torch
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
```

A few implementation notes that separate a toy from a production optimizer. The `@torch.no_grad()` decorator is mandatory — the update itself must not build an autograd graph. Operations are **in-place** (`mul_`, `add_`, `addcmul_`, `addcdiv_`) to avoid allocating new tensors every step; for a 7B model a single full-size temporary is 14 GB in bf16. Real implementations go further with **fused** or **foreach** kernels (`torch.optim.AdamW(..., fused=True)`) that batch the elementwise math across all parameters into one CUDA launch, which is a large throughput win when you have thousands of parameter tensors.

!!! warning "Common pitfall: $\beta_2$ and loss spikes"
    The default $\beta_2 = 0.999$ averages the second moment over $\sim 1/(1-\beta_2) = 1000$ steps. If a single batch produces a large gradient (a "bad" document, a tokenization artifact), $v_t$ reacts slowly, so $\sqrt{\hat v_t}$ stays small and the update explodes — a classic loss spike. Lowering to $\beta_2 = 0.95$ (a $\sim 20$-step window) makes the denominator respond faster and is standard for large LLM pretraining. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

## The Memory Cost of Optimizer States

Here is the systems reality that motivates everything in the rest of the chapter. Consider a model with $P$ parameters. With AdamW you store, at minimum:

| Tensor | Count | Typical precision | Bytes/param |
|---|---|---|---|
| Parameters (weights) | $P$ | bf16 | 2 |
| Gradients | $P$ | bf16 | 2 |
| Adam $m$ (first moment) | $P$ | fp32 | 4 |
| Adam $v$ (second moment) | $P$ | fp32 | 4 |
| Master weights (fp32 copy) | $P$ | fp32 | 4 |

That last row appears because mixed-precision training keeps a high-precision master copy of the weights so that tiny updates are not lost to bf16 rounding (see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)). The headline number from the ZeRO/DeepSpeed analysis is **16 bytes per parameter** for the model+optimizer state (2+2+4+4+4 = 16), of which the *optimizer alone* (m, v, master weights) is **12 bytes** — three times the 4 bytes of bf16 weights and gradients together.

!!! example "Worked example: optimizer memory for a 7B model"
    Take $P = 7\times 10^9$ parameters.

    - bf16 weights: $7\text{e}9 \times 2 = 14$ GB
    - bf16 gradients: $14$ GB
    - Adam $m$ (fp32): $7\text{e}9 \times 4 = 28$ GB
    - Adam $v$ (fp32): $28$ GB
    - fp32 master weights: $28$ GB

    Total $= 14 + 14 + 28 + 28 + 28 = 112$ GB, i.e. $16$ bytes/param. The **optimizer states alone are 84 GB** — they do not fit on a single 80 GB H100, and they dwarf the 14 GB of actual weights. This is the single biggest reason large-model training needs ZeRO/FSDP sharding (see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)), and the single biggest motivation for memory-frugal optimizers. Halving optimizer memory can be the difference between needing 16 versus 8 GPUs.

{{fig:opt-memory-tax}}

Two orthogonal strategies attack this. The first is **systems-level**: ZeRO/FSDP *shards* the 12 bytes of optimizer state across $N$ data-parallel ranks, so each rank holds only $12P/N$ bytes — no change to the math, pure distribution. The second is **algorithmic**: redesign the optimizer to store *less state per parameter*. Adafactor, Lion, and the sign-based family live here, and that is where we turn next. The two strategies compose: you can shard a Lion optimizer too.

## Adafactor: Factoring the Second Moment

Adafactor (Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, 2018) was born from exactly the memory pressure above, originally for training large T5 models on TPUs. Its key observation: the costliest state is the second moment $v$, a full $P$-element tensor. For a weight *matrix* $W \in \mathbb{R}^{n\times m}$, instead of storing the full $n\times m$ matrix of second moments, Adafactor stores only a **row vector** $R\in\mathbb{R}^{n}$ and a **column vector** $C\in\mathbb{R}^{m}$ and reconstructs a rank-1 approximation:

$$
\hat V_{ij} \approx \frac{R_i\, C_j}{\sum_k R_k}.
$$

This is the best rank-1 (in a generalized-KL sense) factorization of the true second-moment matrix. The memory for the second moment drops from $O(nm)$ to $O(n+m)$ — sublinear in the parameter count. For a $4096\times 4096$ matrix that is $16.7$M numbers versus $8192$, a $2000\times$ reduction *for that tensor's second-moment state*.

Adafactor pairs this with two more memory moves: it can **drop the first moment entirely** ($\beta_1 = 0$, no momentum), and it uses **relative step sizes** scaled by the RMS of the parameters so it needs no external learning-rate tuning. It also adds **update clipping** by RMS norm for stability. The factored update for a matrix, in pseudocode:

```python
import torch

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
```

The trade-off is real: Adafactor's factored second moment and missing momentum make it slightly noisier and sometimes less stable than AdamW, and it can need more babysitting (warmup, the `eps2` floor). But it cut optimizer memory roughly in half-to-two-thirds and made T5-scale training feasible. It remains popular for fine-tuning large models on memory-constrained hardware, and its rank-1 factorization idea echoes in later work. A related modern option, **8-bit Adam** (Dettmers et al.), takes the orthogonal route of keeping full $m$ and $v$ but quantizing them to 8 bits with block-wise scaling, cutting their footprint $4\times$ with almost no quality loss — and composes with everything else.

## Lion: Learning the Sign of the Update

Lion (Chen et al., *Symbolic Discovery of Optimization Algorithms*, 2023) was discovered by a program-search procedure over optimizer programs, and the winner is startlingly simple. "Lion" stands for **Evo**lved **Si**gn **Mo**me**n**tum. It keeps a *single* momentum buffer (so only **4 extra bytes/param** versus Adam's 8) and the update direction is the **sign** of an interpolated momentum:

$$
c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
\theta_t = \theta_{t-1} - \eta\big(\operatorname{sign}(c_t) + \lambda\theta_{t-1}\big),
$$

$$
m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t.
$$

Note the two different interpolations: the *update* uses $\beta_1$ (e.g. 0.9) while the *momentum buffer* is updated with $\beta_2$ (e.g. 0.99). The $\operatorname{sign}$ is the heart of it: every parameter moves by exactly $\pm\eta$ (plus decay), independent of gradient magnitude. This is an extreme form of the scale-invariance Adam approximates — Lion makes it exact and uniform.

```python
import torch

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
```

Because the update magnitude is uniformly $\eta$, Lion's effective step is **larger and more uniform** than AdamW's, so the recommended learning rate is roughly $3$–$10\times$ *smaller* than Adam's, and the weight decay correspondingly $3$–$10\times$ *larger* (to keep $\eta\lambda$ in a sane range). When tuned, Lion matches or beats AdamW on many vision and language pretraining tasks while using half the optimizer memory and slightly less compute (no square root, no second buffer). Its weaknesses: the sign update injects more gradient noise, so it tends to need **larger batch sizes** to behave, and it can be touchier near the end of training. Still, Lion is the cleanest demonstration that you do not need a per-coordinate magnitude estimate at all — a good *sign* plus momentum is often enough.

!!! tip "Practitioner tip: re-tune LR and decay when switching optimizers"
    You cannot drop a new optimizer into an existing recipe and keep the hyperparameters. Lion needs a much smaller LR and larger decay than AdamW; Muon needs its own LR for matrices and a *separate* AdamW for embeddings and the LM head. Always re-sweep learning rate (and warmup) when changing optimizer family. See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

## Second-Order Methods: Shampoo and Preconditioning

Everything so far uses only *diagonal* curvature information — Adam's $\sqrt{v}$ is a diagonal preconditioner, one scalar per coordinate. The ideal update, from a second-order Taylor expansion, is **Newton's method**:

$$
\theta_t = \theta_{t-1} - \eta\, H^{-1} g_t,
$$

where $H$ is the Hessian (or, in practice, the Fisher / empirical second-moment matrix). $H^{-1}$ rotates and rescales the gradient to undo the loss surface's anisotropy, fixing the ill-conditioning that diagonal methods can only partially address. The catch is fatal at scale: for $P$ parameters $H$ is $P\times P$, which is $\sim 5\times 10^{19}$ entries for a 7B model. Storing, let alone inverting, the full Hessian is impossible.

**Shampoo** (Gupta, Koren & Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization*, 2018) makes second-order methods tractable for *matrix*-shaped parameters by using a **Kronecker-factored** preconditioner. For a weight matrix $W\in\mathbb{R}^{n\times m}$ with gradient $G$, it maintains two much smaller matrices:

$$
L_t = L_{t-1} + G G^\top \in \mathbb{R}^{n\times n}, \qquad
R_t = R_{t-1} + G^\top G \in \mathbb{R}^{m\times m},
$$

and preconditions on both sides with their inverse fourth roots:

$$
W_t = W_{t-1} - \eta\, L_t^{-1/4}\, G\, R_t^{-1/4}.
$$

This approximates the full $nm\times nm$ preconditioner by the Kronecker product $L^{1/2}\otimes R^{1/2}$, capturing curvature *between rows* and *between columns* separately. The memory is $O(n^2 + m^2)$ instead of $O(n^2 m^2)$, and Shampoo provably converges faster per step on many problems. Its costs are the periodic computation of matrix inverse-roots (eigendecompositions of $n\times n$ and $m\times m$ matrices, done every $K$ steps to amortize) and the $L,R$ accumulators themselves. A distributed implementation (**Distributed Shampoo**, Anil et al.) won the 2024 AlgoPerf optimization benchmark, demonstrating that well-engineered second-order methods can beat AdamW on wall-clock time, not just step count. The practical barriers — kernel complexity, inverse-root numerics, and integrating with parameter sharding — have kept it out of most mainstream LLM recipes, but it directly inspired the optimizer we cover last.

## Muon: Orthogonalizing the Update

**Muon** (Jordan et al., 2024) is the newest entrant and has driven much of the recent optimizer excitement, including reported speed records on nanoGPT and use in frontier-scale training. The name stands for **M**oment**U**m **O**rthogonalized by **N**ewton-Schulz. Its premise is geometric: for a 2-D weight matrix, the *momentum* update $M_t$ is typically dominated by a few large singular directions — it is effectively low-rank, so most of the update's "energy" pushes along a handful of directions and starves the rest. Muon fixes this by replacing the momentum update with its **orthogonalization**: the nearest semi-orthogonal matrix, which has all singular values equal to 1.

Formally, if $M_t = U\Sigma V^\top$ is the singular value decomposition (SVD) of the momentum, Muon's update is

$$
O_t = U V^\top, \qquad W_t = W_{t-1} - \eta\, O_t.
$$

Setting every singular value to 1 makes the update *spectrally uniform* — it pushes equally along every singular direction of the momentum, rather than letting the top few dominate. This is the matrix-valued analogue of Lion's per-coordinate sign: Lion normalizes each scalar entry to $\pm 1$; Muon normalizes each *singular value* to 1. Both are aggressive ways of discarding magnitude and keeping direction, but Muon respects the matrix structure of the weight.

{{fig:opt-discard-magnitude-family}}

The genius is *how* it computes $UV^\top$ without an SVD (which is slow and hard to do well in bf16 on GPUs). It uses a **Newton-Schulz iteration**: a fixed sequence of matrix-multiply-only steps that drives the singular values toward 1. Starting from a spectrally-normalized $X_0 = M/\lVert M\rVert_F$, it repeats a cubic polynomial in $X$:

$$
X_{k+1} = a X_k + b\,(X_k X_k^\top) X_k + c\,(X_k X_k^\top)^2 X_k,
$$

with carefully chosen coefficients $(a,b,c)\approx(3.4445, -4.7750, 2.0315)$ tuned so that roughly 5 iterations push all singular values close to 1. Crucially these are just `matmul`s — they run at full GPU throughput in bf16, so the orthogonalization adds only modest overhead.

```python
import torch

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
```

{{fig:opt-muon-newton-schulz}}

Muon's properties make it a compelling AdamW replacement for the bulk of an LLM's parameters:

- **Memory.** Like momentum SGD and Lion, it stores **one** buffer per parameter (momentum), not two — half of Adam's optimizer-state memory. There is no second-moment tensor at all.
- **Only for 2-D weights.** Orthogonalization is defined for matrices. The standard recipe is **hybrid**: use Muon for the 2-D hidden weight matrices (attention and MLP projections) and a small **AdamW for the 1-D parameters and the input embedding / output head**, which are not matrix-multiplied in the same sense and behave better under Adam.
- **Scale matching.** The $\sqrt{\max(n,m)}$ factor makes Muon's update RMS comparable to AdamW's, so learning-rate intuition transfers and you can reuse much of an AdamW schedule.
- **Reported gains.** On small-scale benchmarks (nanoGPT speedruns) and in some larger reports, Muon reaches a target loss in meaningfully fewer steps/tokens than tuned AdamW, while using less memory.

The mechanism connects cleanly to Shampoo: orthogonalizing $M = U\Sigma V^\top$ to $UV^\top$ is exactly applying the preconditioner $(MM^\top)^{-1/2}M$, a "whitening" of the update — the same spectral idea as Shampoo's inverse-root preconditioning, but computed cheaply with matmuls and applied to the momentum rather than accumulated second moments. Muon can be read as a streamlined, GPU-friendly descendant of the Shampoo line.

!!! note "Aside: why the embeddings get AdamW"
    Embedding and unembedding (LM head) parameters have rows indexed by token, with extremely sparse and uneven gradients — a rare token's row is updated only when that token appears. Orthogonalizing across the vocabulary dimension mixes unrelated tokens and behaves poorly, and these layers benefit from Adam's per-coordinate magnitude adaptation. Hence the hybrid Muon+AdamW recipe rather than Muon everywhere. The same logic explains why 1-D norm/bias parameters stay on AdamW.

## Choosing an Optimizer: A Practical Comparison

The table below summarizes the state per parameter (the memory tax) and the character of each method. "Buffers/param" counts the optimizer-specific state tensors beyond weights and gradients.

| Optimizer | Buffers/param | Extra bytes/param (fp32) | Update character | Where it shines |
|---|---|---|---|---|
| SGD | 0 | 0 | raw gradient | cheapest; rarely used for LLMs |
| SGD+momentum | 1 ($v$) | 4 | velocity | vision models, great generalization |
| AdamW | 2 ($m,v$) | 8 | per-coord adaptive | **LLM default**; robust, well-understood |
| 8-bit AdamW | 2 (quantized) | ~2 | same as AdamW | AdamW quality, $4\times$ less state |
| Adafactor | factored | ~$O(n{+}m)$ | factored 2nd moment | memory-bound fine-tuning, T5-scale |
| Lion | 1 ($m$) | 4 | sign of momentum | half AdamW memory; needs big batches |
| Shampoo | 2 factors | $O(n^2{+}m^2)$ | Kronecker 2nd-order | fewest steps; heavy compute/eng |
| Muon (2-D) | 1 ($m$) | 4 | orthogonalized momentum | speed + half memory; hybrid recipe |

A pragmatic decision procedure for a new pretraining run:

1. **Default to AdamW** with $\beta=(0.9, 0.95)$, weight decay $0.1$ on 2-D weights only, gradient clipping at norm 1.0. It is the most documented and forgiving choice and the safe baseline for an interview answer.
2. **If optimizer memory is the binding constraint** and you cannot shard further, reach for 8-bit AdamW (smallest behavioral change) or Adafactor (most aggressive, more babysitting).
3. **If you want speed and have appetite for tuning**, try Muon for the matrices + AdamW for embeddings/head; it is the most exciting current option and halves optimizer-state memory on the bulk of parameters.
4. **Always re-sweep learning rate, warmup, and weight decay** when you change family — the hyperparameters do not transfer.

!!! interview "Interview Corner"
    **Q:** Why is Adam (or AdamW) the default for training transformers instead of SGD with momentum, and what is the cost of that choice?

    **A:** Three reasons. (1) **Per-parameter adaptivity / scale invariance.** Transformer gradients span many orders of magnitude across parameters — sparse embedding rows, norm gains, dense projections — and Adam's division by $\sqrt{\hat v}$ rescales every coordinate to a near-unit step, so a single global learning rate works. SGD has one rate for all and zig-zags on the resulting ill-conditioned, anisotropic loss surface. (2) **Robustness to sparse and noisy gradients**, which dominate at the embedding layer. (3) **Fast, reliable early convergence**, helped by bias correction that prevents huge steps in the first iterations. The cost is memory: AdamW stores two extra fp32 tensors per parameter (first and second moment), so optimizer states are ~12 bytes/param including the fp32 master copy — three times the bf16 weights. For a 7B model that's ~84 GB of optimizer state alone, which is why we need ZeRO/FSDP sharding and why memory-frugal optimizers (Lion, Adafactor, Muon, 8-bit Adam) exist. A secondary cost: SGD often generalizes slightly better, but for LLMs Adam's convergence speed and robustness win decisively.

    **Follow-up Q:** What's the difference between L2 regularization and weight decay in Adam, and why does AdamW matter?

    **A:** In plain SGD they're identical. In Adam they're not: L2 adds $\lambda\theta$ to the gradient, which then passes through the adaptive denominator $\sqrt{\hat v}$ — so high-gradient parameters get *less* effective decay, the opposite of the intent. AdamW decouples decay, applying $\theta \leftarrow (1-\eta\lambda)\theta$ directly to the weights independent of the adaptive term. This consistently improves generalization and is why every modern LLM uses AdamW, not Adam-with-L2. And you apply decay only to 2-D weight matrices, never to norm gains or biases.

!!! key "Key Takeaways"
    - **SGD+momentum** is cheapest (0–1 buffers) and generalizes well, but a single global learning rate cannot handle the wildly heterogeneous gradient scales of a transformer — hence adaptive methods.
    - **Adam/AdamW** rescale each coordinate by $\hat m/\sqrt{\hat v}$, giving near-unit, scale-invariant steps; **bias correction** ($1-\beta^t$ factors) prevents huge early steps. For LLMs use $\beta_2 = 0.95$ to react faster to gradient spikes.
    - **AdamW decouples weight decay** from the adaptive term ($\theta\leftarrow(1-\eta\lambda)\theta$); L2-in-the-gradient under-decays high-gradient params. Decay only 2-D weights, never norms or biases.
    - **Optimizer state is the memory hog:** AdamW costs ~12 bytes/param ($m$, $v$, fp32 master) — 3× the bf16 weights, ~84 GB for a 7B model. This forces ZeRO/FSDP sharding and motivates frugal optimizers.
    - **Adafactor** factors the second moment into row×column vectors ($O(n{+}m)$ memory) and can drop momentum — sublinear optimizer memory, at some stability cost. **8-bit Adam** quantizes $m,v$ for a $4\times$ cut with little quality loss.
    - **Lion** stores one buffer and steps by the *sign* of momentum (uniform $\pm\eta$); half AdamW memory, needs a smaller LR, larger decay, and bigger batches.
    - **Shampoo** is a tractable second-order method via Kronecker-factored preconditioners ($L^{-1/4}GR^{-1/4}$); fewest steps, but heavy compute and engineering.
    - **Muon** orthogonalizes the momentum of 2-D weights via a matmul-only Newton-Schulz iteration (singular values → 1), the matrix analogue of Lion's sign. One buffer per param, hybrid with AdamW for embeddings/head; a fast, memory-light, newsmaking AdamW alternative.

!!! sota "State of the Art & Resources (2026)"
    As of 2026, AdamW remains the default for LLM pretraining, but orthogonalization-based optimizers—Muon in particular—have moved from nanoGPT speedruns to frontier-scale production use (Kimi K2, GLM-4.5), offering roughly 2× compute efficiency gains over AdamW. The field is converging on hybrid recipes: Muon for 2-D weight matrices, AdamW for embeddings and 1-D parameters.

    **Foundational work**

    - [Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015)](https://arxiv.org/abs/1412.6980) — original Adam derivation with bias-corrected moment estimates; the paper every LLM optimizer builds on.
    - [Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2019)](https://arxiv.org/abs/1711.05101) — shows L2 regularization ≠ weight decay under Adam and introduces AdamW, now the universal LLM training standard.
    - [Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* (2018)](https://arxiv.org/abs/1804.04235) — rank-1 factored second moments cut optimizer memory from O(nm) to O(n+m); first practical large-scale memory-frugal adaptive optimizer.

    **Recent advances (2023–2026)**

    - [Chen et al., *Symbolic Discovery of Optimization Algorithms* (2023)](https://arxiv.org/abs/2302.06675) — program-search discovers Lion (Evolved Sign Momentum): one buffer, sign update, half Adam's optimizer-state memory.
    - [Liu et al., *Muon is Scalable for LLM Training* (2025)](https://arxiv.org/abs/2502.16982) — proves Muon scales to large models with weight decay + per-parameter update scaling, achieving ~2× compute efficiency vs. AdamW on compute-optimal runs.
    - [Vyas et al., *SOAP: Improving and Stabilizing Shampoo using Adam* (2024)](https://arxiv.org/abs/2409.11321) — runs Adam in Shampoo's eigenbasis, reducing iterations by 40% and wall-clock time by 35% on language model training.
    - [Anil et al., *Scalable Second Order Optimization for Deep Learning* (2020)](https://arxiv.org/abs/2002.09018) — Distributed Shampoo: Kronecker-factored preconditioners made tractable via CPU-distributed inverse-root computation.

    **Open-source & tools**

    - [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — drop-in 8-bit AdamW (and other optimizers) via block-wise quantization; cuts optimizer-state memory 4× with negligible quality loss.
    - [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) — open-source distributed Muon implementation used to train the 16B Moonlight MoE on 5.7T tokens; includes pretrained checkpoints.
    - [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch) — clean PyTorch implementation of Lion with optional Triton fused kernels.

    **Go deeper**

    - [Keller Jordan, *Muon: An optimizer for hidden layers in neural networks* (2024)](https://kellerjordan.github.io/posts/muon/) — the original Muon blog post explaining Newton-Schulz orthogonalization, nanoGPT results, and the connection to Shampoo.
    - [Dettmers et al., *8-bit Optimizers via Block-wise Quantization* (2022)](https://arxiv.org/abs/2110.02861) — ICLR 2022 spotlight; the paper behind bitsandbytes' 8-bit Adam, with block-wise dynamic quantization preserving 32-bit fidelity.

## Further reading

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015) — the original Adam derivation and bias correction.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2019) — AdamW and why decoupling matters.
- Shazeer & Stern, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* (2018) — factored second moments.
- Chen et al., *Symbolic Discovery of Optimization Algorithms* (2023) — the Lion optimizer.
- Gupta, Koren & Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018); Anil et al., *Scalable Second Order Optimization for Deep Learning* (Distributed Shampoo).
- Jordan et al., *Muon* (2024) — orthogonalized momentum via Newton-Schulz; see also the nanoGPT speedrun writeups.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) — the optimizer-state memory analysis and sharding.
- Dettmers et al., *8-bit Optimizers via Block-wise Quantization* (2022) — quantized Adam states.
