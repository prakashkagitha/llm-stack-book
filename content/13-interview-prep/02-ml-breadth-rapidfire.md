# 13.2 ML Breadth: Rapid-Fire Concepts & Model Answers

The Google ML "breadth" round is not a quiz about whether you memorized definitions. It is a probe of whether you *understand mechanisms* well enough to reason about them under pressure — to say not just "dropout prevents overfitting" but *why*, *how much*, *what it costs at inference*, and *when it actively hurts*. The format strategy is covered in [The Google ML Domain Interview: Format & Strategy](../13-interview-prep/01-google-ml-interview-format.html); this chapter is the dense reference you drill the night before.

The structure is deliberate. Each concept gets a **crisp model answer** — the two-to-four sentences you would actually say out loud — followed by enough mechanism, math, and runnable code that the answer is *yours* and not a recited card. Treat the model answers as the floor, not the ceiling: a great candidate hits the model answer and then volunteers the trade-off the interviewer was about to ask for.

A unifying lens runs through everything below. Almost every "ML breadth" question is secretly about one of three tensions: **bias vs. variance** (capacity and generalization), **signal vs. noise** (optimization and regularization), or **what you actually measure vs. what you care about** (evaluation). Keep those three axes in mind and the forty-odd sub-topics collapse into a handful of ideas applied repeatedly.

## Generalization: Bias, Variance, Overfitting & Regularization

### The bias–variance decomposition

**Model answer.** For squared-error loss, expected test error at a point decomposes into bias², variance, and irreducible noise. Bias is error from wrong assumptions (too simple a model — *underfitting*); variance is sensitivity to the particular training sample (too flexible a model — *overfitting*); noise is the Bayes-error floor you cannot beat. Increasing capacity trades bias for variance; regularization, more data, and ensembling trade variance for a little bias.

Formally, for a target $y = f(x) + \varepsilon$ with $\mathbb{E}[\varepsilon]=0$, $\operatorname{Var}(\varepsilon)=\sigma^2$, and an estimator $\hat f$ trained on a random dataset $D$:

$$
\mathbb{E}_{D}\big[(y - \hat f(x))^2\big] = \underbrace{\big(f(x) - \mathbb{E}_D[\hat f(x)]\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\big[(\hat f(x) - \mathbb{E}_D[\hat f(x)])^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}
$$

The decomposition is *exact* for squared error; for cross-entropy and 0–1 loss there are analogous but messier decompositions, so in interviews quote the squared-error version and say the *intuition* carries over.

Let us see it empirically. We fit polynomials of increasing degree to a noisy sine and measure how bias and variance trade off across resampled training sets.

```python
import numpy as np

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
```

Running this prints something like `degree= 1  bias^2=0.21  variance=0.03`, `degree= 3  bias^2=0.01  variance=0.02`, `degree= 9  bias^2=330  variance=64251`. The degree-1 line is too stiff (high bias, low variance); degree-3 is the sweet spot (both small); and the degree-9 polynomial — fit through only 20 points on an ill-conditioned Vandermonde basis — wiggles so wildly to chase noise that both its bias² *and* variance explode into the thousands. That runaway is high variance in its most vivid form. **This U-shaped total-error curve is the single most important picture in the breadth round** — the interviewer wants to see you draw it and name the axes.

!!! note "Aside: double descent and why the classic U-curve isn't the whole story"
    The classic bias–variance U-curve predicts catastrophe as you overparameterize. Yet modern networks with far more parameters than data points generalize *well*. The resolution is **double descent**: test error rises to a peak at the *interpolation threshold* (model just barely fits the training data), then *descends again* as you keep adding capacity. The over-parameterized regime is governed by implicit regularization from SGD, which among the infinitely many zero-training-loss solutions tends to find low-norm, smooth ones. In an interview, mention double descent as the "modern footnote" to bias–variance — it signals you read past the textbook.

### Diagnosing overfitting vs. underfitting

**Model answer.** Plot training and validation loss together. *Underfitting:* both are high and close — the model lacks capacity or hasn't trained long enough. *Overfitting:* training loss keeps dropping while validation loss flattens then rises — the gap between the two curves *is* the variance. The fix differs: underfitting wants more capacity / less regularization / longer training; overfitting wants more data, more regularization, or early stopping.

| Symptom | Train loss | Val loss | Train–val gap | Likely cause | Fix |
|---|---|---|---|---|---|
| Underfitting | high | high | small | too little capacity / too much reg | bigger model, train longer, lower $\lambda$ |
| Good fit | low | low | small | — | ship it |
| Overfitting | low | high | large | too much capacity / too little data | more data, augmentation, reg, early stop |
| Distribution shift | low | high | large but val *flat from start* | train ≠ test distribution | fix data, domain adaptation |

The last row is the subtle one and a favorite trap: a large train–val gap is *not* always overfitting. If validation loss is high from the very first epoch and never tracked training, you likely have a **distribution mismatch or a leak**, not a variance problem, and throwing regularization at it won't help.

### Regularization: the full toolbox

**Model answer.** Regularization is anything that reduces variance at the cost of some bias — it constrains the hypothesis space toward simpler functions. The big families are: (1) *norm penalties* — L2/weight decay shrinks weights toward zero, L1 induces sparsity; (2) *stochastic* — dropout, DropConnect, stochastic depth, data augmentation; (3) *early stopping*, which limits the effective number of optimization steps; (4) *normalization* (batch/layer norm) and *architectural* priors (convolutions, attention); and (5) *ensembling*, which averages away variance directly.

The L2-penalized objective and its gradient make the "shrinkage" literal:

$$
\tilde{\mathcal{L}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2
\quad\Longrightarrow\quad
\nabla\tilde{\mathcal{L}} = \nabla\mathcal{L} + \lambda\theta
\quad\Longrightarrow\quad
\theta \leftarrow (1 - \eta\lambda)\,\theta - \eta\nabla\mathcal{L}
$$

Every step *multiplicatively decays* the weights by $(1 - \eta\lambda)$ before applying the data gradient — hence "weight decay." For plain SGD, the L2 penalty and weight decay are identical. **For Adam they are not**, because Adam divides the gradient by a per-parameter running RMS, which also rescales the penalty's contribution. *AdamW* fixes this by applying decay *directly to the weights*, decoupled from the adaptive moment estimate — this is why AdamW is the default for transformers and why "decoupled weight decay" is a sharp thing to mention. More on optimizers in [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

L1 vs. L2 geometry is worth one crisp sentence: the L1 ball has corners on the axes, so the loss contour first touches it at a vertex where some coordinates are exactly zero — that is *why* L1 gives sparse solutions while L2 only shrinks.

```python
import numpy as np

# Why L1 zeros things out and L2 only shrinks: the proximal operators.
def prox_l1(w, t):           # soft-thresholding: solution of min 0.5(x-w)^2 + t|x|
    return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)

def prox_l2(w, t):           # shrinkage: solution of min 0.5(x-w)^2 + 0.5 t x^2
    return w / (1.0 + t)

w = np.array([0.05, 0.5, -0.3, 0.02, 2.0])
print("L1 (t=0.1):", prox_l1(w, 0.1))   # -> [ 0.  0.4 -0.2  0.  1.9] small coords killed
print("L2 (t=0.1):", prox_l2(w, 0.1))   # -> [0.045 0.455 ...]        all shrunk, none zero
```

L1 sends the two near-zero coordinates *exactly* to 0; L2 shrinks everyone proportionally but never zeroes anyone. That difference — feature selection vs. smooth shrinkage — is the entire L1-vs-L2 answer.

!!! warning "Common pitfall: regularizing the wrong things"
    Do **not** apply weight decay to bias terms, LayerNorm scale/shift parameters, or embeddings indiscriminately — decaying a LayerNorm gain toward zero fights the normalization it is supposed to enable, and decaying biases just adds noise with no generalization benefit. Production training scripts (nanoGPT, Megatron) build *two parameter groups*: one with weight decay for the matmul weights, one with `weight_decay=0` for norms and biases. If you claim "we use weight decay" in an interview, be ready to say *which* parameters.

## Optimization: Gradient Descent, Adam & Gradient Pathologies

### From SGD to Adam in one mental model

**Model answer.** SGD follows the noisy negative gradient; the noise from minibatching is a feature, not a bug — it helps escape sharp minima. Momentum adds a velocity term that accumulates consistent gradient directions and damps oscillation across narrow ravines. RMSProp divides each coordinate's step by a running root-mean-square of its gradients, giving every parameter its own adaptive learning rate. Adam is *momentum + RMSProp with bias correction*. AdamW additionally decouples weight decay.

The Adam update, with bias correction, is worth being able to write from memory:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, & \hat m_t &= \frac{m_t}{1-\beta_1^t} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2, & \hat v_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} & &
\end{aligned}
$$

The bias correction matters most in the first few hundred steps: because $m_0 = v_0 = 0$, the raw moments are biased toward zero, and dividing by $(1-\beta_1^t)$ and $(1-\beta_2^t)$ un-biases them. Here is Adam from scratch — 12 lines that you should be able to reproduce on a whiteboard:

```python
import numpy as np

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
```

Why does Adam dominate transformer training while SGD+momentum dominates ResNets? Transformer gradients are *heavy-tailed and wildly different in scale across parameters* (embeddings vs. attention vs. LayerNorm), and the per-coordinate adaptive scaling is essential for stability. The deep dive is in [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

### Learning rate: the single most important hyperparameter

**Model answer.** Too high and you diverge or oscillate; too low and you crawl and may stall in a bad region. The standard transformer recipe is *linear warmup* (ramp from 0 over a few hundred to a few thousand steps, because early Adam moment estimates are unreliable and large embeddings are untrained) followed by *cosine decay* to a small floor. Tune it first, on a log scale, and tune it before anything else.

```python
import math

def lr_schedule(step, base_lr=3e-4, warmup=2000, total=100_000, min_ratio=0.1):
    if step < warmup:                           # linear warmup
        return base_lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))   # 1 -> 0
    return base_lr * (min_ratio + (1 - min_ratio) * cosine)

for s in [0, 1000, 2000, 50_000, 100_000]:
    print(f"step {s:>7}: lr = {lr_schedule(s):.2e}")
```

Schedules, warmup, and the batch-size / learning-rate coupling (the linear and square-root scaling rules) are in [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

### Vanishing and exploding gradients

**Model answer.** In a deep network the gradient is a long product of Jacobians (chain rule). If the typical singular values of those Jacobians are below 1, the product shrinks exponentially with depth — *vanishing gradients*, so early layers barely learn. If above 1, it grows exponentially — *exploding gradients*, so updates blow up and loss goes to NaN. The fixes are structural: residual connections (so gradients have a "highway" of identity paths), normalization (keeps activations and thus Jacobians well-scaled), careful initialization, non-saturating activations, and — for explosions — gradient clipping.

The math, for a network $h_t = \phi(W h_{t-1})$:

$$
\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L}\prod_{t=1}^{L} \frac{\partial h_t}{\partial h_{t-1}}, \qquad \left\|\prod_t J_t\right\| \sim \rho^{L}
$$

where $\rho$ is the typical spectral radius of the layer Jacobians. The exponential in $L$ is the whole problem. Three concrete mitigations:

```python
import numpy as np

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
```

**He/Kaiming initialization** ($\operatorname{Var}(W) = 2/\text{fan\_in}$ for ReLU) keeps activation variance roughly constant through depth; **Xavier/Glorot** ($2/(\text{fan\_in}+\text{fan\_out})$) targets the same for symmetric activations. The residual connection — the reason transformers can be hundreds of layers deep — gives $\partial h_t/\partial h_{t-1} = I + \partial F/\partial h_{t-1}$, so the identity term keeps the gradient from vanishing no matter how small $F$'s Jacobian gets. This is the single most important architectural trick for trainability; see [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html). Loss spikes and large-run debugging are covered in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

!!! interview "Interview Corner"
    **Q:** Your transformer's training loss is decreasing nicely, then suddenly spikes to a huge value and the run NaNs out. Walk me through how you debug it.

    **A:** First I check the gradient norm logs around the spike: a spike in grad-norm before the loss spike points to an exploding-gradient event, often from a single bad batch (e.g. a very long or degenerate sequence) or numerical overflow in attention. Immediate guards: global gradient clipping (norm ≤ 1.0), lowering the learning rate or extending warmup, and checking for fp16 overflow — switching attention/softmax and the loss to fp32 or moving to bf16, whose wider exponent range rarely overflows. I'd also inspect whether a specific data shard correlates with spikes, and confirm the LR schedule isn't ramping too aggressively. Structurally, I'd verify residuals and normalization are correctly placed (pre-norm is more stable than post-norm at depth) and that no LayerNorm gain has drifted huge. The systematic version of this checklist is the "training stability" playbook: clip, warm up, bf16, pre-norm, inspect data, lower LR.

## Normalization, Dropout & Activations

### BatchNorm vs. LayerNorm — and why transformers use LayerNorm

**Model answer.** Both normalize activations to roughly zero-mean/unit-variance then apply a learnable scale and shift, which smooths the loss landscape and lets you use higher learning rates. *BatchNorm normalizes across the batch dimension per feature* — so its statistics depend on other examples in the batch, which breaks for small/variable batches, for sequence models where the "batch" of tokens is ragged, and at inference (it needs running averages). *LayerNorm normalizes across the feature dimension per example* — independent of batch size, identical at train and test, and naturally suited to variable-length sequences. That is why transformers use LayerNorm (and RMSNorm).

$$
\text{BN: } \hat x_{i} = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \quad(\mu,\sigma \text{ over the batch}) \qquad
\text{LN: } \hat x_{i} = \frac{x_i - \mu_{\text{feat}}}{\sqrt{\sigma_{\text{feat}}^2 + \epsilon}} \quad(\mu,\sigma \text{ over features})
$$

```python
import torch

x = torch.randn(4, 8, 16)   # (batch=4, seq=8, features=16)

# LayerNorm: stats over the LAST dim (features), per (batch, position).
mu = x.mean(-1, keepdim=True)
var = x.var(-1, unbiased=False, keepdim=True)
ln = (x - mu) / torch.sqrt(var + 1e-5)
assert torch.allclose(ln.mean(-1), torch.zeros(4, 8), atol=1e-5)  # each token normalized

# RMSNorm (used in LLaMA/PaLM): no mean-subtraction, just divide by RMS. Cheaper.
rms = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
```

Three sharp follow-ups to volunteer: (1) **RMSNorm** drops the mean-centering entirely — empirically the re-centering contributes little, so modern LLMs save the compute. (2) **Pre-norm vs. post-norm**: putting the norm *inside* the residual branch (pre-norm, $x + F(\text{LN}(x))$) keeps a clean identity path and is far more stable at depth than the original post-norm transformer. (3) BatchNorm's batch-dependence is a *correctness* problem in RL and contrastive setups where examples within a batch are correlated. Details in [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

### Dropout — train-time noise, test-time scaling

**Model answer.** Dropout randomly zeros each activation with probability $p$ during training, forcing the network not to rely on any single feature — it approximates training an exponential ensemble of sub-networks and sharing their weights. At test time you use all units but must rescale so the expected activation matches; the standard *inverted dropout* divides by $(1-p)$ at train time so test-time is a plain forward pass.

```python
import torch

def inverted_dropout(x, p=0.1, training=True):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()   # keep with prob (1-p)
    return x * mask / (1 - p)                  # rescale so E[output] == E[input]

x = torch.ones(100_000)
out = inverted_dropout(x, p=0.3, training=True)
print(out.mean().item())   # ≈ 1.0 — expectation preserved despite 30% zeroed
```

The expectation argument is exact: each unit survives with probability $(1-p)$ and is then divided by $(1-p)$, so $\mathbb{E}[\text{out}] = (1-p)\cdot \frac{x}{1-p} = x$. **In modern LLM pretraining, dropout is often set to 0** — with web-scale data the model is data-bound, not variance-bound, so the regularization just slows learning. Dropout reappears during *fine-tuning* on small datasets where overfitting is real. Knowing when *not* to use a technique is a strong signal.

!!! warning "Common pitfall: forgetting eval mode"
    `model.eval()` in PyTorch does two things at once: it switches dropout *off* and switches BatchNorm to its *running statistics* instead of batch statistics. Forgetting it at inference is a classic bug — your outputs become nondeterministic (dropout) and batch-size-dependent (BN), and metrics silently degrade. Conversely, forgetting `model.train()` after an eval loop means you accidentally train with dropout disabled. Always pair them.

### Activation functions

**Model answer.** Activations inject nonlinearity (without one, a deep net collapses to a single linear map). ReLU ($\max(0,x)$) is cheap and non-saturating for positive inputs, which fixes vanishing gradients for deep nets, but can "die" (a unit stuck at zero gradient forever). GELU and SiLU/Swish are smooth approximations that are now standard in transformers; gated variants like **SwiGLU** (used in LLaMA/PaLM) split the MLP into a gate and a value branch and consistently improve quality per parameter.

| Activation | Formula | Property |
|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | saturates both ends → vanishing gradients; avoid in hidden layers |
| Tanh | $\tanh(x)$ | zero-centered but still saturates |
| ReLU | $\max(0,x)$ | cheap, non-saturating; risk of dead units |
| LeakyReLU | $\max(\alpha x, x)$ | small negative slope $\alpha$ avoids dead units |
| GELU | $x\,\Phi(x)$ | smooth; transformer default |
| SiLU/Swish | $x\,\sigma(x)$ | smooth, self-gated |
| SwiGLU | $(\text{Swish}(xW_1)\odot xW_3)W_2$ | gated MLP; best quality/param in LLMs |

The "dying ReLU" mechanism is worth one sentence: if a large gradient pushes a unit's pre-activation permanently negative, its output is 0, its gradient is 0, and it never recovers — LeakyReLU/GELU avoid this by keeping a nonzero gradient for negative inputs.

## Transformers, Attention & Embeddings

This section compresses Part II into interview-sized answers. For the full from-scratch builds, see [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html), [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html), and [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html).

### Self-attention in one breath

**Model answer.** Each token emits a query, a key, and a value (linear projections of its embedding). Attention computes, for every query, a softmax-weighted average of all values, weighted by query–key similarity (scaled dot product). So each token's output is a content-based, learned mixture of every other token's value — the mechanism that lets the model route information across arbitrary distances in one layer. It is $O(n^2)$ in sequence length because every token attends to every other.

$$
\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Two sub-questions interviewers love. **Why divide by $\sqrt{d_k}$?** The dot product of two independent $d_k$-dimensional vectors with unit-variance entries has variance $d_k$; without scaling, the logits grow with dimension, push softmax into a saturated regime where one weight is ≈1 and the rest ≈0, and the gradient through softmax nearly vanishes. Dividing by $\sqrt{d_k}$ restores unit-variance logits. **Why softmax?** It produces a valid probability distribution (non-negative, sums to 1) so the output is a convex combination of values, and it is differentiable.

```python
import torch
import torch.nn.functional as F

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
```

### Multi-head attention, and the KV-cache flavors

**Model answer.** Instead of one attention with dimension $d$, we run $h$ heads of dimension $d/h$ in parallel and concatenate — each head can specialize (syntax, coreference, positional patterns) and attend to a different subspace. The cost is the same FLOPs but more expressive routing. *MQA/GQA* share keys and values across heads to shrink the KV cache (the dominant memory cost at inference); *MLA* compresses KV into a low-rank latent.

The KV-cache trade-off is a *systems* answer the interviewer will reward: with $h$ heads, naive multi-head attention (MHA) stores $h$ separate K and V vectors per token, which at long context dominates GPU memory during decode. **Multi-Query Attention (MQA)** uses one shared K/V for all heads (≈$h\times$ smaller cache, slight quality loss); **Grouped-Query Attention (GQA)** interpolates with $g$ groups. This is the central reason modern LLMs ship GQA. The math and KV-cache sizing live in [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) and [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

!!! example "Worked example: KV-cache size, and why GQA matters"
    Take a model with $L = 32$ layers, $h = 32$ heads, head dimension $d_h = 128$ (so model dim $d = 4096$), serving in bf16 (2 bytes). The KV cache stores **2** tensors (K and V) per layer per token.

    **Per token, full MHA:** $2 \times L \times h \times d_h \times 2\text{ bytes} = 2 \times 32 \times 32 \times 128 \times 2 = 2{,}097{,}152$ bytes $\approx$ **2 MiB per token**.

    For a single sequence of **8{,}192 tokens**: $8192 \times 2\text{ MiB} = 16\text{ GiB}$ — for *one* request's KV cache. That is most of an A100-40GB before you've batched anything. **Now switch to GQA with 8 KV groups** instead of 32 heads: the K/V projection shrinks by $32/8 = 4\times$, so the cache drops to **4 GiB**, letting you batch ~4× more concurrent requests on the same card. This single architectural choice is the difference between serving 1 user and 4 — which is exactly why every production LLM since ~2023 uses GQA or MLA. The throughput consequences are worked out in [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

### Embeddings: what they are and why dot products mean "similar"

**Model answer.** An embedding is a learned dense vector representing a discrete object (token, word, user, item) in a continuous space where geometric proximity encodes semantic similarity. They are trained so that objects appearing in similar contexts get similar vectors (the distributional hypothesis). We compare them with cosine similarity or dot product; the magic is that *meaning becomes arithmetic* — `king - man + woman ≈ queen`.

```python
import torch
import torch.nn.functional as F

# Cosine similarity vs. dot product: when do they differ?
a = torch.tensor([1.0, 0.0])
b = torch.tensor([2.0, 0.0])   # same direction, 2x magnitude
c = torch.tensor([0.0, 1.0])   # orthogonal

print(F.cosine_similarity(a, b, dim=0))  # 1.0 — direction identical
print(a @ b)                             # 2.0 — magnitude inflates dot product
print(F.cosine_similarity(a, c, dim=0))  # 0.0 — orthogonal = unrelated
```

The crisp distinction: **cosine similarity ignores magnitude (pure direction); dot product rewards both alignment and magnitude.** For retrieval where vectors are L2-normalized, the two are equivalent. Token embeddings, the input pipeline, and positional information are in [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html); representation learning, contrastive training, and retrieval embeddings in [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html).

!!! interview "Interview Corner"
    **Q:** Why does positional encoding exist at all — what would break without it?

    **A:** Self-attention is *permutation-equivariant*: it computes a weighted sum over a *set* of tokens with no notion of order, so "dog bites man" and "man bites dog" would produce identical representations. Positional encodings inject order. Sinusoidal (Vaswani) and learned absolute encodings add a position-dependent vector to the input. Modern LLMs prefer **RoPE** (rotary), which rotates the query/key vectors by an angle proportional to position, so the attention dot product depends only on *relative* offset $(i-j)$ — this generalizes better to longer contexts and is what enables context-length extension. The taxonomy is in [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).

## Evaluation Metrics, Class Imbalance & Calibration

### The confusion matrix and its derived metrics

**Model answer.** From the four cells — true positive, false positive, true negative, false negative — you derive everything. *Precision* = TP/(TP+FP): of what I flagged positive, how much was right (penalizes false alarms). *Recall* = TP/(TP+FN): of all actual positives, how many I caught (penalizes misses). *F1* is their harmonic mean. Which you optimize depends on the cost of each error type: a cancer screen wants recall, a spam filter that must not bury real mail wants precision.

$$
P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}, \quad F_1 = \frac{2PR}{P+R}, \quad F_\beta = (1+\beta^2)\frac{PR}{\beta^2 P + R}
$$

```python
import numpy as np

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
```

### ROC-AUC vs. PR-AUC, and the threshold question

**Model answer.** A classifier outputs a score; the *threshold* turns it into a decision, and every threshold gives a different (precision, recall) and (TPR, FPR). The **ROC curve** plots TPR vs. FPR across all thresholds; its area (ROC-AUC) is the probability the model ranks a random positive above a random negative — a *threshold-independent* ranking quality. **PR-AUC** plots precision vs. recall and is the better summary under heavy class imbalance, because ROC-AUC can look deceptively good when negatives vastly outnumber positives (a few extra false positives barely move FPR but crater precision).

The threshold-tuning loop is what you actually ship:

```python
import numpy as np

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
```

The default 0.5 threshold is arbitrary — it is optimal only for a calibrated model with equal error costs and balanced classes. **Choosing the operating point is a product decision, not a modeling one**; say that and the interviewer relaxes.

### Class imbalance: a menu of fixes

**Model answer.** Imbalance hurts because the loss is dominated by the majority class, so the model can score well by ignoring the minority. The fixes, roughly in order of preference: (1) *use the right metric* (PR-AUC, per-class recall — never raw accuracy); (2) *resample* — oversample the minority (SMOTE synthesizes interpolated examples) or undersample the majority; (3) *reweight the loss* so minority errors cost more; (4) *focal loss*, which down-weights easy, well-classified examples and focuses gradient on the hard minority; (5) *threshold-move* at inference. Often a combination.

Class-weighted and focal loss are two lines each:

$$
\mathcal{L}_{\text{weighted}} = -\sum_i w_{y_i}\log p_{i,y_i}, \qquad
\mathcal{L}_{\text{focal}} = -\sum_i (1 - p_{i,y_i})^{\gamma}\log p_{i,y_i}
$$

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    # logits: (N, C), targets: (N,) class indices
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    pt = p.gather(1, targets[:, None]).squeeze(1)     # prob of the true class
    logpt = logp.gather(1, targets[:, None]).squeeze(1)
    focal = -alpha * (1 - pt) ** gamma * logpt        # easy examples (pt→1) contribute ≈0
    return focal.mean()
```

The focal-loss insight is the gem: $(1-p_t)^\gamma$ is ≈0 when the model is already confident and correct ($p_t \to 1$), so gradient flows almost entirely from hard or misclassified examples — exactly the minority class in an imbalanced problem.

!!! warning "Common pitfall: resampling before splitting"
    If you oversample (or run SMOTE) *before* the train/validation split, synthetic or duplicated copies of the same example land in *both* sets — a data leak that inflates validation scores and collapses in production. **Always split first, then resample only the training fold.** Same rule for any imbalance handling, feature scaling fit on train only, and target encoding. This "fit on train, apply to val/test" discipline is the most common interview trap in the whole evaluation section.

### Calibration

**Model answer.** A model is *calibrated* if its confidence matches its accuracy — among predictions made with 0.8 probability, 80% should be correct. Modern deep networks are usually *over-confident*. We measure it with a reliability diagram and **Expected Calibration Error (ECE)**, which bins predictions by confidence and averages the gap between confidence and accuracy in each bin. Cheap fixes: **temperature scaling** (divide logits by a single learned scalar $T>1$ to soften), Platt scaling, or isotonic regression — all fit on a held-out set.

$$
\text{ECE} = \sum_{b=1}^{B}\frac{|B_b|}{N}\,\big|\,\text{acc}(B_b) - \text{conf}(B_b)\,\big|
$$

Temperature scaling matters directly for LLMs: the sampling temperature in decoding *is* this $T$, dividing logits before softmax to trade off diversity vs. determinism — see [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html). Honest measurement of models is the subject of [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html).

## Loss Functions, Information Theory & A Rapid-Fire Round

### Cross-entropy, log-loss, and what they minimize

**Model answer.** Cross-entropy is the negative log-likelihood of the correct class under the model's predicted distribution; minimizing it is maximum-likelihood estimation. It pairs with softmax because the softmax+cross-entropy gradient is beautifully simple — $(\hat p - y)$ — and because log-loss heavily penalizes confident wrong predictions (the $-\log$ goes to infinity as the predicted probability of the truth goes to 0). This is *the* loss for LLM pretraining: predict the next token, minimize its negative log-likelihood.

$$
H(p,q) = -\sum_x p(x)\log q(x), \qquad
\frac{\partial}{\partial z_j}\big[\text{softmax-CE}\big] = \hat p_j - y_j
$$

The clean gradient is why softmax+CE is the canonical classifier head — derive it once and you never forget it:

```python
import numpy as np

def softmax_cross_entropy(logits, y_onehot):
    z = logits - logits.max(axis=-1, keepdims=True)   # numerical stability
    p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    loss = -(y_onehot * np.log(p + 1e-12)).sum(axis=-1).mean()
    grad = (p - y_onehot) / len(logits)                # the famous (p - y)
    return loss, grad
```

The `logits - max` trick is itself an interview-worthy detail: it prevents `exp` overflow without changing the softmax output, because softmax is invariant to adding a constant to all logits. The pretraining loss in full is in [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html).

### KL divergence, entropy, and perplexity

**Model answer.** Entropy $H(p)$ is the average bits to encode samples from $p$; cross-entropy $H(p,q)$ is the cost of using a wrong code $q$; their difference is the **KL divergence** $D_{\text{KL}}(p\,\|\,q) \ge 0$, the extra bits from the mismatch. KL is *not* symmetric — forward KL (used in MLE) is mass-covering, reverse KL (used in variational inference and as the RLHF penalty) is mode-seeking. **Perplexity** is just $\exp(\text{cross-entropy})$ — the effective branching factor, "how many equally-likely next tokens is the model confused among."

$$
D_{\text{KL}}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p), \qquad \text{PPL} = e^{H(p,q)}
$$

KL divergence is load-bearing across the LLM stack: it is the regularizer keeping an RLHF policy near its reference model (so it doesn't reward-hack into gibberish), and it is the distillation objective. See [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) and [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html). The information-theory foundations are in [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html).

!!! example "Worked example: perplexity intuition"
    A language model assigns probability $0.1$ to each of the actual next tokens in a held-out sentence of length 10 (so cross-entropy per token is $-\log(0.1) = \ln 10 \approx 2.303$ nats). Then perplexity $= e^{2.303} = 10$ — the model is, on average, as confused as if it were guessing uniformly among **10** equally likely tokens. If we improve it so it assigns $0.5$ to the right tokens, perplexity drops to $e^{-\ln 0.5} = 2$: down to 2 effective choices. A perplexity of 1 means perfect, deterministic prediction; a vocabulary-size perplexity (~50{,}000) means the model learned nothing. This is why a drop from PPL 20 to PPL 12 is a *big* deal — the relationship to loss is logarithmic.

### Generative vs. discriminative, parametric vs. non-parametric

**Model answer.** A **discriminative** model learns $p(y\mid x)$ directly (logistic regression, most classifiers, the discriminative head of a fine-tuned LLM) — it draws decision boundaries. A **generative** model learns the joint $p(x,y)$ or $p(x)$ and can *sample new data* (naive Bayes, GANs, diffusion, and *autoregressive LLMs themselves*, which model $p(\text{tokens})$). Discriminative models usually win at pure classification accuracy with enough data; generative models give you sampling, anomaly detection, and work better with little data or missing features. **Parametric** models have a fixed parameter count regardless of dataset size (linear/logistic regression, neural nets); **non-parametric** models grow with the data (k-NN, kernel SVMs, decision trees, Gaussian processes).

### The rapid-fire lightning round

These come fast; have a one-liner ready for each.

- **Why is a validation set separate from test?** Validation tunes hyperparameters and triggers early stopping (you "fit" to it indirectly); test is touched *once* for an unbiased final estimate. Touch test repeatedly and you overfit to it too.
- **What is k-fold cross-validation for?** Reuse scarce data: rotate the held-out fold $k$ times, average the metric — lower-variance estimate than a single split, at $k\times$ compute. Use it when data is small.
- **Generative model in a few words?** Learns to *sample*; an LLM is generative — it models $p(x_t \mid x_{<t})$.
- **One-hot vs. embedding?** One-hot is sparse, orthogonal, no notion of similarity, dimension = vocab. Embedding is dense, low-dim, learned, encodes similarity. Embedding = one-hot times a learned weight matrix (a lookup).
- **Why mini-batches and not full-batch GD?** Full-batch is expensive per step and the gradient noise from mini-batches both regularizes and helps escape sharp minima; pure SGD (batch=1) is too noisy/slow.
- **Bagging vs. boosting?** Bagging (random forests) trains models in *parallel* on bootstrap samples and averages — reduces *variance*. Boosting (gradient boosting, XGBoost) trains models *sequentially*, each correcting the last's residuals — reduces *bias*.
- **Curse of dimensionality?** In high dimensions, volume explodes, all points become roughly equidistant, and distance-based methods (k-NN) and density estimation degrade — you need exponentially more data to fill the space. This is *why* we learn low-dimensional embeddings.
- **L2 vs. L1 in one line?** L2 = smooth shrinkage (all weights small); L1 = sparsity (many weights exactly zero, feature selection).
- **Why does a deeper/wider model not always help?** Past the data's capacity you trade bias for variance and start overfitting; you also hit optimization and compute walls. Scaling laws say compute, data, and parameters must grow *together* — see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).
- **What is the reparameterization trick?** Rewrite a stochastic node $z\sim\mathcal{N}(\mu,\sigma^2)$ as $z = \mu + \sigma\,\epsilon,\ \epsilon\sim\mathcal{N}(0,1)$ so gradients flow through $\mu,\sigma$ — the trick that makes VAEs trainable.
- **Precision/recall trade-off knob?** The decision threshold. Raise it → higher precision, lower recall, and vice versa.
- **What is label smoothing?** Replace hard one-hot targets with $(1-\epsilon)$ on the true class and $\epsilon/(K-1)$ elsewhere; prevents over-confidence, improves calibration, mild regularizer.

!!! interview "Interview Corner"
    **Q:** You train a model and get 98% accuracy, but in production it performs terribly on the minority class you actually care about. Diagnose and fix.

    **A:** The 98% almost certainly reflects a class imbalance where the majority class dominates accuracy — accuracy is the wrong metric. I'd first re-evaluate with per-class precision/recall, the confusion matrix, and PR-AUC for the minority class; I expect to find recall near zero there. Fixes, layered: (1) switch the reported metric to PR-AUC / minority-recall so we optimize the right thing; (2) rebalance training via class-weighted or focal loss, or minority oversampling / SMOTE — but resample *only the training fold after splitting* to avoid leakage; (3) move the decision threshold using a held-out set to hit the precision/recall operating point the product needs; (4) check for distribution shift between train and production and for label noise in the minority class. I'd also verify the train/val split is stratified so the minority class is represented in both.

!!! key "Key Takeaways"
    - **Bias–variance is the master axis.** Underfitting = high bias (both losses high, small gap); overfitting = high variance (low train, high val, large gap). Regularization, more data, and ensembling buy variance reduction for a little bias; double descent is the modern footnote.
    - **Regularization is anything that shrinks variance.** L2/weight decay (use AdamW for *decoupled* decay; never decay norms/biases), L1 for sparsity, dropout (inverted, rescale by $1/(1-p)$, off in eval), early stopping, augmentation, normalization, ensembling.
    - **Gradient pathologies are exponential in depth.** Fix vanishing/exploding with residual connections (identity highway), normalization, He/Xavier init, non-saturating activations, and global-norm gradient clipping.
    - **LayerNorm/RMSNorm beat BatchNorm for sequences** because they normalize per-example over features — batch-independent and identical at train and test. Pre-norm is more stable than post-norm at depth.
    - **Attention is content-based routing:** $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. The $\sqrt{d_k}$ keeps logits unit-variance so softmax doesn't saturate; multi-head adds subspace specialization; GQA/MLA shrink the KV cache that dominates inference memory.
    - **Pick the metric before the model.** Accuracy lies under imbalance; use precision/recall/F1, PR-AUC, and a *tuned threshold*. The 0.5 default is only optimal for a calibrated, balanced, equal-cost problem.
    - **Cross-entropy = MLE; its softmax gradient is $(\hat p - y)$.** KL divergence, entropy, and perplexity ($e^{\text{loss}}$) are the same idea in different clothes — and KL is the load-bearing regularizer across RLHF and distillation.
    - **The deadliest interview trap is data leakage:** fit scalers, resamplers, and encoders on the training fold only, split before you resample, and touch the test set exactly once.

## Further reading

- Vaswani et al., *Attention Is All You Need* (2017) — the transformer and scaled dot-product attention.
- Goodfellow, Bengio & Courville, *Deep Learning* (2016) — chapters on regularization (Ch. 7) and optimization (Ch. 8) are the canonical breadth reference.
- Bishop, *Pattern Recognition and Machine Learning* (2006) — the definitive bias–variance and probabilistic-modeling treatment.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* — bias–variance, regularization paths, and ensembles.
- Ioffe & Szegedy, *Batch Normalization* (2015); Ba, Kiros & Hinton, *Layer Normalization* (2016) — the two normalization schemes contrasted above.
- Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (2014).
- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015); Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2019).
- Lin et al., *Focal Loss for Dense Object Detection* (2017) — class imbalance done right.
- Guo et al., *On Calibration of Modern Neural Networks* (2017) — temperature scaling and ECE.
- Belkin et al., *Reconciling Modern Machine-Learning Practice and the Bias–Variance Trade-off* (2019) — the double-descent paper.
