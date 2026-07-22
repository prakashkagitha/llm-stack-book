# 3.10 Learning Rate Schedules, Warmup, Batch Size & Hyperparameters

Hyperparameter tuning is one of the most impactful — and most under-documented — parts of pretraining a large language model. A 10x learning rate error will destroy a run that would otherwise converge cleanly; the right schedule can shave percentage points off final perplexity. Yet the wisdom lives mostly in appendices of papers and in the institutional memory of ML engineering teams.

This chapter makes that implicit knowledge explicit. We cover the full pipeline: why warmup is mandatory at scale, the major schedule families and when to use each, how batch size interacts with learning rate and what the *critical batch size* tells you about compute efficiency, and finally how muP (maximal-update parameterization) lets you tune hyperparameters on a small model and transfer them to a large one. Every section includes runnable code and concrete numbers.

Related chapters you should read in tandem: [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) covers the adaptive optimizer mechanics that schedules ride on top of; [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html) covers what happens when you get the schedule wrong; [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) gives the big picture that informs budget allocation decisions.

## Why Schedules and Warmup Exist

Deep learning optimizers like Adam carry two moving-average estimates: first moment $m_t$ (gradient direction) and second moment $v_t$ (gradient magnitude squared). Both are initialized to zero. During the first few hundred steps, $v_t$ is still a noisy underestimate of the true gradient variance — the bias-correction terms in Adam partially compensate, but the effective learning rate is still erratic early in training.

For large models, this cold-start instability is catastrophic. At step 1 of a run with 1 billion parameters and a 4096-token context, the weight matrices are random; the gradient norms are large and wildly variable across layers. Applying the full target learning rate immediately produces parameter updates large enough to push weights into regions where softmax logits saturate, norms explode, or residual magnitudes collapse — none of which recover easily. Empirically, runs without warmup frequently spike or diverge within the first thousand steps.

Warmup solves this by linearly ramping the effective learning rate from near-zero to the target value over a set number of steps, giving the optimizer time to calibrate its momentum estimates and giving the network time to find a reasonable initialization basin before the full update magnitude kicks in.

{{fig:lrsched-why-warmup}}

The interplay with weight initialization is important. Standard initialization schemes ([Transformers: The Transformer Block](../02-transformer/06-transformer-block.html)) ensure variance-preserving forward passes at step 0, but they do not ensure that the *gradient landscape* is well-behaved. Warmup effectively treats the first $T_w$ steps as a coarser form of initialization.

## The Major Schedule Families

### Linear Schedule

The simplest useful schedule: ramp linearly from $\eta_{\min}$ to $\eta_{\max}$ during warmup, then decay linearly to $\eta_{\min}$ over the remaining steps.

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot \dfrac{t}{T_w} & t \leq T_w \\[6pt]
\eta_{\max} \cdot \dfrac{T - t}{T - T_w} & t > T_w
\end{cases}
$$

Linear decay is fast to implement and interpretable. It was widely used in BERT-era fine-tuning but is now rarely the first choice for pretraining because it decays too aggressively in the middle of the run.

### Cosine Annealing

The dominant pretraining schedule as of 2024. After warmup, the learning rate follows the right half of a cosine curve, smoothly decaying to a floor $\eta_{\min}$ (usually $\eta_{\max}/10$ or a small constant like $1\text{e-}5$):

{{fig:lr-schedule}}

$$
\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\pi \cdot \frac{t - T_w}{T - T_w}\right)\right)
$$

Key properties:
- Spends most of the budget near the peak learning rate (the cosine curve is flat near its maximum), which means the network sees aggressive gradient steps for most of training — good for exploration.
- The tail naturally slows down near the end, allowing fine-grained convergence.
- The exact shape is not sensitive to the precise $T$ you use, as long as $T$ is approximately correct.

The main weakness: cosine requires knowing the total token budget $T$ upfront. If you extend the run or add a second phase, you need to restart the schedule or accept a discontinuity.

### Cosine with Restarts (SGDR)

Introduced by Loshchilov & Hutter (*SGDR: Stochastic Gradient Descent with Warm Restarts*, 2016), this runs multiple cosine cycles with exponentially increasing cycle length. Each restart re-warms the LR to $\eta_{\max}$ and decays again. It was influential in CV but is rarely used in modern LLM pretraining because the LR spike at each restart causes loss spikes and the benefits for language modeling are unclear.

### Warmup-Stable-Decay (WSD)

WSD, popularized by MiniCPM (Hu et al., 2024) and used in several other recent models, divides training into three explicit phases:

1. **Warmup** ($T_w$ steps): linear ramp from near-zero to $\eta_{\max}$.
2. **Stable** ($T_s$ steps): constant $\eta_{\max}$.
3. **Decay** ($T_d$ steps): cosine or linear decay to $\eta_{\min}$.

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot \dfrac{t}{T_w} & t \leq T_w \\[6pt]
\eta_{\max} & T_w < t \leq T_w + T_s \\[6pt]
\eta_{\max} \cdot f\!\left(\dfrac{t - T_w - T_s}{T_d}\right) & t > T_w + T_s
\end{cases}
$$

where $f$ is a cosine or linear decay function. The insight driving WSD is that most of the loss reduction happens in the stable phase, and the decay phase mainly "polishes" the model. This decoupling is practically valuable: you can train in the stable phase for as long as resources allow, then trigger the decay when you're ready to finalize — without having committed to a specific total step count at the start.

{{fig:lrsched-wsd-shape}}

### RSqrt (Inverse Square Root)

$$
\eta(t) = \eta_{\max} \cdot \sqrt{\frac{T_w}{\max(t, T_w)}}
$$

Used in original Transformer training (Vaswani et al., 2017) with the combined formula $\eta(t) = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5},\; t \cdot T_w^{-1.5})$. The rsqrt schedule never fully plateaus — the LR continues to slowly decrease throughout training. It works well for smaller models and shorter runs but tends to decay too quickly for billion-scale pretraining.

## Implementing Schedules From Scratch

```python
import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_fraction: float = 0.1,   # eta_min = eta_max * min_lr_fraction
) -> LambdaLR:
    """
    Cosine annealing with linear warmup.
    The LambdaLR multiplier is relative to the base LR in the optimizer.
    """
    def lr_lambda(current_step: int) -> float:
        # --- Warmup phase ---
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # --- Cosine decay phase ---
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # progress in [0, 1]; cosine from 1 → min_lr_fraction
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Rescale so the floor is min_lr_fraction
        return min_lr_fraction + (1.0 - min_lr_fraction) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_fraction: float = 0.1,
) -> LambdaLR:
    """
    Warmup-Stable-Decay (WSD) schedule.
    Advantage: total training length can be decided late — just extend stable phase.
    """
    T_w = num_warmup_steps
    T_s = num_stable_steps
    T_d = num_decay_steps

    def lr_lambda(step: int) -> float:
        if step < T_w:
            # Linear warmup
            return float(step) / float(max(1, T_w))
        elif step < T_w + T_s:
            # Stable plateau at peak LR
            return 1.0
        else:
            # Cosine decay to floor
            decay_progress = float(step - T_w - T_s) / float(max(1, T_d))
            decay_progress = min(decay_progress, 1.0)  # clamp at end
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---- Quick smoke test ----
if __name__ == "__main__":
    model = torch.nn.Linear(10, 10)
    # Base LR that the scheduler multiplies against
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
        min_lr_fraction=0.1,
    )

    lrs = []
    for step in range(1000):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Verify: step 50 should be ~50% of peak; step 999 should be near min
    assert abs(lrs[50] / lrs[99] - 50/100) < 0.01, "warmup slope wrong"
    assert lrs[-1] < lrs[99] * 0.15, "floor not reached"
    print(f"Peak LR: {max(lrs):.2e}, Final LR: {lrs[-1]:.2e}")
    # Output: Peak LR: 3.00e-04, Final LR: 3.00e-05
```

## Learning Rate vs. Batch Size Scaling

### The Linear Scaling Rule

When you increase the batch size $B$ by a factor $k$, each gradient step is an average over $k$ times more samples, reducing variance by $\sqrt{k}$ and the signal-to-noise ratio effectively improves. To maintain the same training dynamics — the same total parameter update magnitude per unit of data — you should also scale the learning rate:

$$
\eta' = k \cdot \eta \quad \text{(linear scaling rule, Goyal et al., 2017)}
$$

This rule works well for small-to-moderate batch size changes (say, 256 to 4096) with SGD and has been observed to hold approximately for Adam as well. The intuition: with $k\times$ larger batches, each step is $k\times$ more expensive in wall time (same flops per sample), but covers $k\times$ more data, so to move at the same "rate through the data manifold," the step size should scale linearly.

### The Square-Root Scaling Rule

For very large batch sizes, linear scaling breaks down — the gradient variance no longer falls as $1/B$ because you start hitting the intrinsic noise floor of the data distribution, not just sampling noise. The square-root rule offers a more conservative alternative:

$$
\eta' = \sqrt{k} \cdot \eta
$$

In practice, OpenAI GPT-3 and similar runs used variants of this rule when scaling from thousands to millions of tokens per batch.

### Critical Batch Size

The *critical batch size* $B^*$ is the regime boundary between these two laws. It was formalized by McCandlish et al. (*An Empirical Model of Large-Batch Training*, 2018). The key quantity is the *gradient noise scale*:

$$
B_{\text{noise}} = \frac{\text{tr}(\Sigma)}{\|G\|^2}
$$

where $\Sigma$ is the covariance of the per-sample gradient and $G$ is the mean gradient. When $B \ll B_{\text{noise}}$, batches are too small to average out noise — increasing batch size linearly reduces steps needed. When $B \gg B_{\text{noise}}$, you are in the *saturated* regime where more data per step doesn't help; gradient noise is already small and increasing $B$ wastes compute.

The practical takeaway: there is an optimal batch size for a given compute budget. Doubling the batch size beyond $B^*$ halves your throughput efficiency. For typical LLM pretraining, $B^*$ for cross-entropy loss on language is on the order of a few million tokens — consistent with the token batches used in Chinchilla-optimal runs.

{{fig:lrsched-critical-batch-size}}

!!! example "Worked Example: Batch-LR Pair for a 7B Pretraining Run"

    Suppose your baseline is:
    - $\eta_{\max} = 3\text{e-}4$, batch size $B = 512$ samples × 2048 tokens = ~1M tokens/step.

    You want to scale to $B' = 2048$ samples × 2048 tokens = ~4M tokens/step, a factor $k = 4$.

    **Linear scaling rule:** $\eta' = 4 \times 3\text{e-}4 = 1.2\text{e-}3$.

    At 4M tokens/step, you'll also converge in roughly $1/4$ the steps for the same total token count. If original run had 100K steps, new run has 25K steps. With cosine schedule, the peak LR and total steps both halve/quarter.

    **Check: does 1.2e-3 violate any rule of thumb?** Under standard parameterization the optimal global LR shrinks as width grows (roughly $1/d$ for hidden matrices); for a 7B model with hidden dim $d = 4096$, published peak LRs fall in the range $1\text{e-}4$ to $3\text{e-}3$. (Under muP the *base* LR you tune is width-invariant instead — the $1/d$ factor lives in the per-layer LR multiplier, not in the number you sweep.) So 1.2e-3 is well within range.

    **Square-root rule (conservative):** $\eta' = 2 \times 3\text{e-}4 = 6\text{e-}4$. This is safer if you're uncertain whether you've exceeded the critical batch size.

## Gradient Accumulation

When you can't fit the full desired batch size in GPU memory in a single forward-backward pass, *gradient accumulation* (GA) simulates a larger effective batch by running $k$ micro-batches before calling `optimizer.step()`:

$$
g_{\text{eff}} = \frac{1}{k} \sum_{i=1}^{k} g_i
$$

The effective batch size is `micro_batch_size * accumulation_steps * world_size`. Gradient accumulation is mathematically equivalent to a full batch if loss is averaged (not summed) — a subtle but critical distinction.

```python
import torch
import torch.nn as nn
from contextlib import nullcontext

def train_step_with_grad_accumulation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    accumulation_steps: int = 4,
    scaler=None,  # Optional GradScaler for mixed precision
    device: str = "cuda",
) -> float:
    """
    One effective step = accumulation_steps micro-forward-backward passes.
    Returns mean loss over the effective batch.
    """
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for micro_step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Only sync gradients across DDP replicas on the LAST micro-step.
        # This avoids expensive all-reduce on every micro-step.
        sync_ctx = (
            model.no_sync()
            if hasattr(model, "no_sync") and micro_step < accumulation_steps - 1
            else nullcontext()
        )

        with sync_ctx:
            # Use autocast for bf16/fp16 if scaler is provided
            amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if scaler else nullcontext()
            with amp_ctx:
                logits = model(input_ids)
                # CRITICAL: divide by accumulation_steps so effective batch
                # average == sum-of-microbatch-averages / k.
                # If your loss already averages over tokens in the microbatch,
                # this gives the right weight for each sample.
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                ) / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        total_loss += loss.item()

        if (micro_step + 1) == accumulation_steps:
            break  # done accumulating

    # Unscale, clip, step
    if scaler:
        scaler.unscale_(optimizer)
    # Gradient clipping (see next section)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return total_loss * accumulation_steps  # report unscaled mean loss
```

One important caveat with distributed training: gradient synchronization (the all-reduce across data-parallel ranks) should happen only at the final accumulation step. Using `model.no_sync()` in PyTorch DDP avoids the all-reduce on intermediate steps. See [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) for the full picture.

## Weight Decay and Gradient Clipping

### Weight Decay

Weight decay (L2 regularization) penalizes large weights and prevents any single parameter from dominating. In modern deep learning it is implemented as *decoupled weight decay* (as in AdamW, Loshchilov & Hutter, 2019):

$$
\theta_{t+1} = (1 - \lambda \eta) \theta_t - \eta \cdot m_t / (\sqrt{v_t} + \epsilon)
$$

The decay is applied to the raw parameter values, not the gradient estimate, which prevents the adaptive scaling of Adam from interfering with regularization. Typical values: $\lambda = 0.1$ for pretraining (used in GPT-3, Llama, and most modern runs). Embeddings and bias terms are usually excluded from decay since they have a different scale and semantics.

```python
def get_optimizer_with_decay(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),  # common for LLM pretraining
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    AdamW with weight decay applied only to weight matrices (not biases/norms).
    betas=(0.9, 0.95) is standard for LLM pretraining — beta2=0.999 (default)
    can slow adaptation to gradient changes late in training.
    """
    # Partition params: decay weights but NOT biases, LayerNorm params, embeddings
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Common no-decay criteria: 1D params (bias, LN scale/bias),
        # and sometimes embedding matrices
        if param.ndim == 1 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, betas=betas, eps=eps, fused=True  # fused=True uses fast CUDA kernel
    )
    return optimizer
```

Setting `betas=(0.9, 0.95)` rather than the default `(0.9, 0.999)` is a commonly used pretraining choice — the lower $\beta_2$ makes the optimizer more responsive to recent gradient magnitude changes, which helps during the warmup phase and when learning rate jumps occur.

### Gradient Clipping

Gradient clipping prevents parameter updates from being catastrophically large due to occasional gradient spikes (common when training on noisy web data or when the schedule is aggressive). The global-norm clip is by far the most common form:

$$
g \leftarrow g \cdot \min\!\left(1,\; \frac{c}{\|g\|_2}\right)
$$

where $c$ is the clip threshold (typically 1.0) and $\|g\|_2 = \sqrt{\sum_i g_i^2}$ is the global L2 norm across all parameters. This preserves gradient direction while bounding the step magnitude.

A persistent misconception: clipping is *not* a substitute for a well-designed schedule. If your global gradient norm is routinely hitting the clip threshold on more than ~20% of steps, something is wrong with your initialization, learning rate, or data distribution.

```python
# Gradient clipping with norm tracking for monitoring
def clip_and_log_grad_norm(
    model: nn.Module,
    max_norm: float = 1.0,
) -> float:
    """Returns the pre-clip gradient norm for monitoring dashboards."""
    # Computes global L2 norm across all parameters
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    return float(total_norm)
```

`clip_grad_norm_` is a one-liner in PyTorch, but it's worth implementing from scratch once so the mechanics are unambiguous. The whole thing is two steps: (1) compute a single *global* L2 norm as the square root of the summed sum-of-squares across **all** parameters — not a separate norm per parameter — and (2) scale every gradient in place by $\min(1,\, c / \|g\|)$, i.e. you only ever scale gradients *down*, never up.

```python
@torch.no_grad()
def clip_grad_norm_from_scratch(params, max_norm: float = 1.0, eps: float = 1e-6) -> float:
    """From-scratch global-norm clip; matches torch.nn.utils.clip_grad_norm_."""
    grads = [p.grad for p in params if p.grad is not None]
    # ONE global L2 norm across ALL params, not per-parameter norms.
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    clip_coef = max_norm / (total_norm + eps)   # torch adds eps for stability
    if clip_coef < 1.0:                          # only ever scale DOWN
        for g in grads:
            g.mul_(clip_coef)
    return float(total_norm)


if __name__ == "__main__":
    torch.manual_seed(0)
    layer_a = nn.Linear(64, 64)
    layer_b = nn.Linear(64, 64)
    params = list(layer_a.parameters()) + list(layer_b.parameters())

    x = torch.randn(8, 64)
    loss = (layer_b(layer_a(x)) ** 2).sum()
    loss.backward()

    # Save the un-clipped grads so both implementations start from the same state.
    original_grads = [p.grad.clone() for p in params]

    total_norm_scratch = clip_grad_norm_from_scratch(params, max_norm=1.0)
    scratch_clipped_grads = [p.grad.clone() for p in params]

    for p, g in zip(params, original_grads):
        p.grad.copy_(g)  # reset to un-clipped values before the torch reference call
    total_norm_torch = float(nn.utils.clip_grad_norm_(params, max_norm=1.0))
    torch_clipped_grads = [p.grad.clone() for p in params]

    assert abs(total_norm_scratch - total_norm_torch) < 1e-5
    for g_scratch, g_torch in zip(scratch_clipped_grads, torch_clipped_grads):
        assert torch.allclose(g_scratch, g_torch, atol=1e-6)
    print(f"scratch total_norm={total_norm_scratch:.6f}, torch total_norm={total_norm_torch:.6f}")
    # Expected: both report the same total_norm and identical clipped grads
```

Always log the pre-clip gradient norm at every step. A sudden spike — say, from 0.5 to 50 — is an early warning of a loss spike before it becomes visible in the loss itself. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html) for the full debugging playbook.

## muP: Maximal-Update Parameterization

### The Problem with Standard Parameterization at Scale

When you tune hyperparameters on a small model (say, 125M parameters) and transfer them to a large model (7B+), something breaks. The optimal learning rate changes because the width of the network changes: wider networks have larger forward activations and can receive larger gradient signal, so they need smaller learning rates to maintain stable training dynamics.

In standard parameterization (SP), the optimal LR scales roughly as $1/\sqrt{d}$ or $1/d$ depending on the layer type, where $d$ is the hidden dimension. This means every time you scale the model, you have to re-tune LR.

### muP: What Changes

Maximal-update parameterization (Yang et al., *Tensor Programs V*, 2022) is a reparameterization of the network that makes the optimal hyperparameters — especially learning rate and initialization scale — *independent of model width*. You can then:

1. Run a full HP sweep on a tiny "proxy model" (e.g., width 256).
2. Transfer the optimal HPs directly to the full-scale run at width 4096 or beyond.

The key changes relative to standard PyTorch initialization:

| Component | Standard Param | muP |
|---|---|---|
| Input embedding | $\mathcal{N}(0, 1)$ | $\mathcal{N}(0, 1)$ |
| Hidden weight $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ | $\mathcal{N}(0, \sigma^2/d_{\text{in}})$ | $\mathcal{N}(0, \sigma^2/d_{\text{in}})$ |
| Output / readout weight | $\mathcal{N}(0, 1/d_{\text{in}})$ | $\mathcal{N}(0, 1/d_{\text{in}}^2)$ scaled by $1/d$ |
| Per-layer LR multiplier | 1 | $1/d_{\text{in}}$ for hidden; $1/d$ for readout |
| Attention logit scale | $1/\sqrt{d_k}$ | $1/d_k$ |

The precise prescription comes from requiring that all feature updates $\Delta h^{(l)}$ (the pre-activation change per step) remain $O(1)$ as width $d \to \infty$ — the "maximal" in muP means every neuron participates maximally in learning without causing instability.

```python
import torch
import torch.nn as nn
import math


class MuPLinear(nn.Linear):
    """
    Linear layer with muP-compatible initialization and LR scaling.
    In muP:
      - hidden layers: init std = base_std / sqrt(fan_in), LR *= 1/fan_in
      - readout layer: init std = base_std / fan_in, LR *= 1/fan_in
    We implement LR scaling via a per-parameter LR multiplier convention
    compatible with mup (microsoft/mup on GitHub).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_readout: bool = False,
        base_std: float = 1.0,
        inf_width: int = None,  # width at "infinite" (reference) model scale
    ):
        super().__init__(in_features, out_features, bias)
        self.is_readout = is_readout
        self.inf_width = inf_width or in_features

        # muP initialization
        if is_readout:
            # Readout: std ∝ 1/d so activations stay O(1)
            std = base_std / in_features
        else:
            # Hidden: same as standard He/fan-in but with explicit inf_width scaling
            std = base_std / math.sqrt(in_features)

        nn.init.normal_(self.weight, mean=0.0, std=std)
        if bias:
            nn.init.zeros_(self.bias)

    def get_lr_multiplier(self) -> float:
        """
        Returns the per-layer LR multiplier.
        Base LR should be tuned at proxy (small) model; this scales it correctly.
        At proxy model width d_proxy, multiplier = 1.
        At width d >> d_proxy, multiplier = d_proxy / d.
        For simplicity, return 1/in_features (absorbed into optimizer param groups).
        """
        return 1.0 / self.in_features


def build_mup_optimizer(
    model: nn.Module,
    base_lr: float,
    proxy_width: int,
    weight_decay: float = 0.1,
) -> torch.optim.AdamW:
    """
    Build AdamW where each layer's effective LR = base_lr * (proxy_width / layer_width).
    base_lr is tuned at proxy_width; this transfers the HP to any larger model.
    """
    param_groups = []

    for name, module in model.named_modules():
        if isinstance(module, MuPLinear):
            # Scale LR inversely with layer width to maintain muP invariance
            actual_width = module.in_features
            lr_scale = proxy_width / actual_width  # == 1 at proxy, <1 at larger models
            # muP scales ONLY the 2D matrix weight by 1/width. Vector params
            # (biases, ndim==1) stay width-invariant under muP+Adam, so exclude
            # them here; they fall through to the base-LR group below.
            matrix_params = [p for p in module.parameters() if p.ndim >= 2]
            param_groups.append({
                "params": matrix_params,
                "lr": base_lr * lr_scale,
                "weight_decay": weight_decay,
                "name": name,
            })

    # All other parameters (norms, embeddings) get base LR
    # Only the width-scaled matrix weights were grouped above; MuPLinear biases
    # (ndim==1) deliberately fall through to the width-invariant group below.
    named_param_set = {
        id(p)
        for m in model.modules()
        if isinstance(m, MuPLinear)
        for p in m.parameters()
        if p.ndim >= 2
    }
    other_params = [p for p in model.parameters() if id(p) not in named_param_set]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "other",
        })

    return torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95))
```

{{fig:lrsched-mup-transfer}}

### Verifying muP: The Coordinate Check

The coordinate check is the single test that catches the large majority of real-world muP bugs. It plots the per-layer activation (or update) scale against width. Under a **correct** muP implementation, those curves are flat across widths — the whole point of the parameterization. Under standard parameterization (SP), the same curves blow up or shrink monotonically as width grows, because activation scale is exactly what SP fails to control.

```python
import torch
import torch.nn as nn


def make_mlp(width: int, mup: bool) -> nn.Module:
    """4-layer MLP: input -> hidden -> hidden -> readout, ReLU between."""
    if mup:
        layers = [
            MuPLinear(64, width),
            nn.ReLU(),
            MuPLinear(width, width),
            nn.ReLU(),
            MuPLinear(width, width),
            nn.ReLU(),
            MuPLinear(width, 64, is_readout=True),
        ]
    else:
        # Standard parameterization contrast: plain nn.Linear, default init.
        layers = [
            nn.Linear(64, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 64),
        ]
    return nn.Sequential(*layers)


def last_hidden_mean_abs_act(model: nn.Module, x: torch.Tensor) -> float:
    """Captures mean(abs(activation)) at the output of the last hidden ReLU."""
    activations = {}

    def hook(module, inp, out):
        activations["last_hidden"] = out.detach().abs().mean().item()

    # Index 5 is the ReLU right after the third (last hidden) linear layer.
    handle = model[5].register_forward_hook(hook)
    model(x)
    handle.remove()
    return activations["last_hidden"]


if __name__ == "__main__":
    print(f"{'width':>6} | {'muP mean|act|':>14} | {'SP mean|act|':>14}")
    for width in [256, 512, 1024, 2048]:
        for mup_flag, label in [(True, "mup"), (False, "sp")]:
            torch.manual_seed(0)
            model = make_mlp(width, mup=mup_flag)
            optimizer = (
                build_mup_optimizer(model, base_lr=1e-2, proxy_width=256)
                if mup_flag
                else torch.optim.AdamW(model.parameters(), lr=1e-2)
            )

            torch.manual_seed(0)
            x = torch.randn(32, 64)
            target = torch.randn(32, 64)

            for _ in range(5):
                optimizer.zero_grad()
                out = model(x)
                loss = nn.functional.mse_loss(out, target)
                loss.backward()
                optimizer.step()

            act = last_hidden_mean_abs_act(model, x)
            if mup_flag:
                mup_act = act
            else:
                sp_act = act
        print(f"{width:>6} | {mup_act:>14.4f} | {sp_act:>14.4f}")
    # Expected: the muP column stays roughly constant (within ~2x) across the
    # 8x width sweep 256 -> 2048; the SP column drifts several-fold over the
    # same sweep (in a typical run it shrinks ~15-20x, e.g. ~0.24 -> ~0.014),
    # i.e. it is NOT flat.
```

If your muP column is not flat, the bug is almost always in the init std, the per-layer LR multiplier, or the attention/readout scaling.

{{fig:lrsched-coordinate-check}}

### Practical muP Workflow

The Microsoft `mup` library (github.com/microsoft/mup) provides a plug-and-play implementation. The typical workflow is:

1. **Define a base config** with a small proxy width (e.g., 256 hidden dim).
2. **Run a dense HP sweep** over LR, initialization std, and possibly weight decay — this is cheap at small width.
3. **Identify the optimal HP at proxy width.**
4. **Scale width** to the full model. muP guarantees the same HP is optimal (up to reasonable approximation in practice).
5. **Validate on a medium model** (e.g., 1B) before launching the full run.

The evidence that this works is now substantial: Microsoft's Phi models, various internal runs at other labs, and controlled ablations in the *Tensor Programs V* paper all show that muP-transferred LRs closely match the empirically optimal LRs found by grid search at the large scale — saving orders-of-magnitude in tuning compute.

!!! warning "What muP Transfers - and What It Does Not"

    muP guarantees hyperparameter transfer across **width** only. Everything else is empirical and weaker, so a width-256 proxy sweep does *not* automatically transfer to a model that is also deeper, trained longer, or run at a different batch size.

    - **Width:** guaranteed by construction (Yang et al., *Tensor Programs V*, 2022). A width-256 sweep transfers to width 4096+.
    - **Depth:** approximate only. Vanilla muP does not stabilize optimal HPs as you add layers; residual-branch scaling (depth-muP / "complete-P", Yang et al. 2023; Bordelon et al. 2023) is a separate line of work. Do not assume a shallow proxy transfers to a 4x-deeper model.
    - **Training horizon:** not covered. Optimal LR and schedule shift with the total token budget - retune, or use WSD to decouple length from schedule shape.
    - **Batch size:** not covered. Apply the linear/sqrt scaling rules from earlier in this chapter and re-check against the critical batch size.

    Rule of thumb: transfer LR and init across width with muP; retune (or use this chapter's scaling rules) whenever depth, horizon, or batch size change.

## Practical Hyperparameter Recipes

Consolidating the above into a reference table for common pretraining scales:

| Model scale | Peak LR | Warmup steps | Batch (tokens) | Grad clip | Weight decay | $\beta_1, \beta_2$ |
|---|---|---|---|---|---|---|
| 125M | $6\text{e-}4$ | 2000 | ~0.5M | 1.0 | 0.1 | 0.9, 0.95 |
| 1B | $3\text{e-}4$ | 2000 | ~2M | 1.0 | 0.1 | 0.9, 0.95 |
| 7B | $3\text{e-}4$ | 2000 | ~4M | 1.0 | 0.1 | 0.9, 0.95 |
| 70B | $1.5\text{e-}4$ | 4000 | ~8M | 1.0 | 0.1 | 0.9, 0.95 |
| 400B+ | $\sim 1\text{e-}4$ | 4000–8000 | ~16M | 1.0 | 0.1 | 0.9, 0.95 |

These are starting points synthesized from published literature (GPT-3, Llama 1/2/3, Mistral, Falcon, OLMo) — treat them as reasonable defaults, not ground truth. The right value for any specific run depends on architecture choices (RMSNorm vs LayerNorm, activation function, depth/width ratio) and the data mixture.

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PretrainingHParams:
    """
    Reference hyperparameter config for LLM pretraining.
    Start here and tune with muP proxy sweeps.
    """
    # Optimizer
    optimizer: Literal["adamw", "lion", "adafactor"] = "adamw"
    peak_lr: float = 3e-4
    min_lr_fraction: float = 0.1       # eta_min = peak_lr * min_lr_fraction
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1

    # Schedule
    schedule: Literal["cosine", "wsd", "linear", "rsqrt"] = "cosine"
    warmup_steps: int = 2000
    # For cosine: total_steps must be set before training
    # For WSD: set stable_steps and decay_steps instead
    total_steps: int = 100_000
    wsd_stable_fraction: float = 0.85  # fraction of (total-warmup) in stable phase

    # Gradient
    grad_clip_norm: float = 1.0
    grad_accumulation_steps: int = 1

    # Batch
    micro_batch_size: int = 4          # per-GPU, per-step
    tokens_per_sample: int = 2048

    def effective_batch_tokens(self, world_size: int) -> int:
        return (
            self.micro_batch_size
            * self.tokens_per_sample
            * self.grad_accumulation_steps
            * world_size
        )

    def total_tokens(self, world_size: int) -> int:
        return self.effective_batch_tokens(world_size) * self.total_steps


def scale_lr_for_batch_size(
    base_lr: float,
    base_batch: int,
    target_batch: int,
    rule: Literal["linear", "sqrt"] = "linear",
) -> float:
    """
    Scale learning rate when changing effective batch size.
    Use 'linear' when target_batch / base_batch <= 8x.
    Use 'sqrt' for more aggressive batch scaling.
    """
    ratio = target_batch / base_batch
    if rule == "linear":
        return base_lr * ratio
    elif rule == "sqrt":
        return base_lr * math.sqrt(ratio)
    else:
        raise ValueError(f"Unknown rule: {rule}")


# Example: scale 1M-token/step config to 4M tokens/step
base_cfg = PretrainingHParams(peak_lr=3e-4)
new_lr = scale_lr_for_batch_size(
    base_lr=base_cfg.peak_lr,
    base_batch=1_000_000,
    target_batch=4_000_000,
    rule="linear",
)
print(f"Scaled LR: {new_lr:.2e}")  # 1.20e-03
```

### Warmup Duration Heuristics

A useful rule of thumb: warm up for at least $T_w = \max(1000, 0.02 \times T_{\text{total}})$ steps — that is, at least 1000 steps or 2% of the total budget, whichever is larger. Shorter warmups are fine for fine-tuning (where the model is already initialized near a good basin) but dangerous for pretraining from scratch.

For continues pretraining (e.g., domain adaptation starting from a released checkpoint), a short warmup of 100–500 steps is usually sufficient — the parameters are already in a well-behaved regime.

!!! warning "Common Pitfall: Forgetting to Warm Up After a Checkpoint Restart"

    When you resume from a checkpoint, the learning rate resumes at whatever value the schedule produced at the checkpoint step. This is correct if the run is a clean continuation.

    But if you restart with a new optimizer state (e.g., because you ran out of memory and had to restart ZeRO-3), the optimizer's $m_t$ and $v_t$ estimates are reset to zero — yet the LR is at its full value. This is exactly the dangerous cold-start situation warmup is designed to prevent. Always re-warm for a few hundred steps when reinitializing optimizer state.

!!! interview "Interview Corner"

    **Q:** You're scaling a 1B model run to 10B parameters. Keeping all other hyperparameters fixed, how would you adjust the learning rate and why? What framework would you use to make this decision more systematic?

    **A:** With standard parameterization, the optimal LR typically decreases with model width because wider networks produce larger activations and gradient signals, so a proportionally smaller step size is needed to maintain stable training dynamics. A rough empirical rule is to scale LR as $1/\sqrt{d}$ or $1/d$ (depending on the layer type), though the exact exponent varies by architecture. The more principled answer is to use muP (maximal-update parameterization, Yang et al. 2022): define a small proxy model of width 256 or 512, run a grid search over LR there, then transfer the optimal LR directly to the 10B model because muP guarantees width-invariant optimal hyperparameters. This saves enormous compute compared to grid-searching at scale. Additionally, when scaling batch size — which often grows with model scale — apply the linear or sqrt scaling rule accordingly, and re-validate warmup duration since larger models are more sensitive to cold-start instability.

## Putting It All Together: A Training Launch Checklist

Before launching a large run, verify each item:

```text
Hyperparameter Launch Checklist
================================
[ ] Peak LR set (use muP proxy sweep if scaling architecture)
[ ] Warmup steps >= max(1000, 2% of total) for full pretraining
[ ] Batch size chosen: effective tokens/step in range [0.5M, 16M]
    depending on model scale; check against critical batch size estimate
[ ] LR scaled for batch size if changed from reference run (linear or sqrt)
[ ] Cosine or WSD schedule: total_steps (cosine) or stable_steps set
[ ] Min LR floor = 10% of peak LR
[ ] Weight decay = 0.1 (decoupled, excluded from embeddings/norms/biases)
[ ] Grad clip norm = 1.0; grad norm logged every step
[ ] beta2 = 0.95 (not 0.999 default)
[ ] Gradient accumulation: effective_batch = micro_batch * accum_steps * world_size
[ ] DDP: model.no_sync() used on non-final accumulation steps
[ ] Optimizer state saved in checkpoint; LR scheduler state saved separately
[ ] Monitoring: log {lr, grad_norm, loss, step} every step
```

!!! tip "Practitioner Tip: Curriculum for Batch Size"

    Some teams ramp the batch size up alongside the learning rate during warmup — starting with a small batch (say 128K tokens/step) and linearly increasing to the full target batch over the first 2000 steps. This provides a stronger implicit regularization signal early in training (small batches have higher gradient noise, which acts like data augmentation) and can improve final evaluation loss slightly. The learning rate warmup and batch warmup interact: keep them synchronized so the effective LR-per-sample stays roughly constant during the warmup phase.

!!! key "Key Takeaways"

    - Warmup is mandatory at large scale because Adam's momentum estimates are cold-started at zero; ramping LR over 1000–4000 steps prevents early instability.
    - Cosine annealing is the dominant pretraining schedule; Warmup-Stable-Decay (WSD) is increasingly popular because it decouples training length from schedule shape.
    - Linearly scale LR with batch size for moderate scaling ($k \leq 8\times$); use sqrt scaling for larger changes. Beyond the critical batch size, increasing batch size no longer reduces the step count proportionally.
    - Gradient accumulation simulates large batches; always divide loss by accumulation steps and use `model.no_sync()` on intermediate steps in DDP.
    - Use `AdamW` with $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay 0.1 (decoupled, excluding 1D params), and gradient clip norm 1.0 as the default pretraining recipe.
    - muP (maximal-update parameterization) makes hyperparameters — especially peak LR — invariant to model width, enabling HP transfer from small proxy models to billion-parameter runs.
    - Always log pre-clip gradient norm every step; spikes in grad norm are early warnings of loss spikes.
    - When restarting training with reset optimizer state, always re-warm the LR — a full warm-start LR with cold optimizer moments is as dangerous as the original cold start.

!!! sota "State of the Art & Resources (2026)"
    Learning rate schedules, warmup, and hyperparameter transfer are now well-understood engineering disciplines: cosine annealing and Warmup-Stable-Decay (WSD) dominate LLM pretraining, muP has become the standard framework for transferring hyperparameters from proxy to full-scale models, and the theoretical underpinnings of why these schedules work are being rigorously established (2024–2026).

    **Foundational work**

    - [Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (2017)](https://arxiv.org/abs/1608.03983) — introduced cosine annealing with warm restarts; the cosine curve became the default LLM pretraining schedule.
    - [Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (2017)](https://arxiv.org/abs/1706.02677) — established the linear scaling rule for batch size / learning rate.
    - [McCandlish et al., *An Empirical Model of Large-Batch Training* (2018)](https://arxiv.org/abs/1812.06162) — defined the gradient noise scale and critical batch size; the conceptual backbone of batch-size tuning.
    - [Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2019)](https://arxiv.org/abs/1711.05101) — decoupled weight decay from gradient updates; now the universal pretraining optimizer recipe.
    - [Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (2022)](https://arxiv.org/abs/2203.03466) — muP; proved that optimal hyperparameters (especially LR) can be made width-invariant and transferred from small proxy models.

    **Recent advances (2023–2026)**

    - [Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024)](https://arxiv.org/abs/2404.06395) — popularized the Warmup-Stable-Decay (WSD) schedule, demonstrating that decoupling training length from schedule shape enables flexible, efficient pretraining.
    - [Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective* (2024)](https://arxiv.org/abs/2410.05192) — theoretical explanation of WSD via the "river valley" loss landscape; explains why large LR oscillations during the stable phase are benign.
    - [Li et al., *Optimal Learning-Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay* (2026)](https://arxiv.org/abs/2602.06797) — rigorous theory showing a phase transition between power-decay and WSD as optimal schedules depending on task difficulty.

    **Open-source & tools**

    - [microsoft/mup](https://github.com/microsoft/mup) — reference PyTorch implementation of muP with MuAdam/MuSGD optimizers, coordinate-check utilities, and Transformer examples; the standard starting point for HP transfer workflows.

    **Go deeper**

    - [Groeneveld et al., *OLMo: Accelerating the Science of Language Models* (2024)](https://arxiv.org/abs/2402.00838) — fully open pretraining paper with detailed hyperparameter tables (LR, warmup, batch, weight decay) for 1B and 7B runs; a clean reference implementation of the standard recipe.

## Further Reading

- Yang, G. et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." NeurIPS 2022. (The muP paper; defines maximal-update parameterization and the theoretical framework behind HP transfer.)
- McCandlish, S. et al. "An Empirical Model of Large-Batch Training." arXiv, 2018. (Introduces the gradient noise scale and critical batch size.)
- Goyal, P. et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv, 2017. (The linear scaling rule for batch size.)
- Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019. (Defines AdamW and decoupled weight decay.)
- Loshchilov, I. & Hutter, F. "SGDR: Stochastic Gradient Descent with Warm Restarts." ICLR 2017. (Cosine annealing with warm restarts.)
- Hu, S. et al. "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies." arXiv, 2024. (Popularizes the WSD schedule and demonstrates its practical advantages.)
- Brown, T. et al. "Language Models are Few-Shot Learners." NeurIPS 2020. (GPT-3; documents the hyperparameter choices used for large-scale pretraining at the time.)
- microsoft/mup GitHub repository. Reference implementation of maximal-update parameterization in PyTorch.

## Exercises

**1.** Adam carries first- and second-moment estimates $m_t$ and $v_t$ that are both initialized to zero. Using this fact, explain (a) why pretraining a billion-parameter model *without* warmup frequently diverges in the first thousand steps, and (b) why the "Common Pitfall" admonition insists you re-warm the learning rate after a checkpoint restart that *resets the optimizer state* — even though the same admonition says a clean continuation (which keeps optimizer state) needs no re-warm.

??? note "Solution"
    (a) At step 0 both $m_t$ and $v_t$ are zero. The second moment $v_t$ is the running estimate of gradient magnitude squared, and Adam's update divides by $\sqrt{v_t}+\epsilon$. In the first few hundred steps $v_t$ is still a noisy underestimate of the true gradient variance, so the *effective* per-parameter step size is erratic and, on average, too large. At the same time the weight matrices are freshly random: gradient norms are large and vary wildly across layers. Applying the full target learning rate on top of an uncalibrated $v_t$ produces updates big enough to push weights into regimes where softmax logits saturate, norms explode, or residual magnitudes collapse — none of which recover. So the run spikes or diverges. Warmup ramps the effective LR from near-zero to the target over $T_w$ steps, buying the optimizer time to calibrate $m_t, v_t$ and the network time to settle into a reasonable basin before full-magnitude updates arrive.

    (b) The danger is specifically the *combination* of a cold optimizer state (zero $m_t, v_t$) with a full-magnitude learning rate — that is exactly the step-0 situation. A clean continuation reloads the saved $m_t, v_t$, so the moments are already calibrated and the schedule's LR at that step is appropriate; no re-warm is needed. But if you restart with a fresh optimizer (e.g., you had to re-init ZeRO-3 after an OOM), $m_t$ and $v_t$ are back to zero while the schedule places the LR at its full mid-run value. That reproduces the cold-start instability warmup was invented to prevent, so you must re-warm for a few hundred steps.

**2.** Use the chapter's cosine-with-warmup schedule with $\eta_{\max} = 3\text{e-}4$, `num_warmup_steps` $=100$, `num_training_steps` $=1000$, and `min_lr_fraction` $=0.1$. Compute the learning rate at (a) step 50, (b) step 550, and (c) step 1000. Give each to three significant figures.

??? note "Solution"
    The `LambdaLR` multiplier is applied to the base LR $\eta_{\max}=3\text{e-}4$.

    (a) **Step 50 (warmup, since $50 < 100$).** Multiplier $= t/T_w = 50/100 = 0.5$.
    $\eta = 0.5 \times 3\text{e-}4 = 1.50\text{e-}4$.

    (b) **Step 550 (cosine phase).** Progress $= (t - T_w)/(T - T_w) = (550-100)/(1000-100) = 450/900 = 0.5$.
    Cosine term: $0.5\,(1 + \cos(\pi \cdot 0.5)) = 0.5\,(1 + 0) = 0.5$.
    Multiplier $=$ `min_lr_fraction` $+ (1 - $ `min_lr_fraction`$) \times 0.5 = 0.1 + 0.9 \times 0.5 = 0.55$.
    $\eta = 0.55 \times 3\text{e-}4 = 1.65\text{e-}4$.

    (c) **Step 1000.** Progress $= (1000-100)/900 = 1.0$. Cosine term: $0.5\,(1 + \cos(\pi)) = 0.5\,(1 - 1) = 0$.
    Multiplier $= 0.1 + 0.9 \times 0 = 0.1$.
    $\eta = 0.1 \times 3\text{e-}4 = 3.00\text{e-}5$.

    Note that step 1000 lands exactly on the floor `min_lr_fraction` $\times \eta_{\max}$, matching the smoke test's reported final LR of $3.00\text{e-}5$.

**3.** Your reference run uses $\eta_{\max} = 3\text{e-}4$ at an effective batch of $0.5\text{M}$ tokens/step. You want to scale to $4\text{M}$ tokens/step. (a) Give the peak LR the linear scaling rule prescribes and the peak LR the square-root rule prescribes. (b) The chapter says the critical batch size $B^*$ for language cross-entropy is "on the order of a few million tokens." Given that, which of your two candidate LRs is the safer choice for the $4\text{M}$-token batch, and why? (c) If you hold the *total* token budget fixed while going from $0.5\text{M}$ to $4\text{M}$ tokens/step, by what factor does the number of optimizer steps change?

??? note "Solution"
    The scaling factor is $k = 4\text{M} / 0.5\text{M} = 8$.

    (a) **Linear rule:** $\eta' = k \cdot \eta = 8 \times 3\text{e-}4 = 2.4\text{e-}3$.
    **Square-root rule:** $\eta' = \sqrt{k}\cdot \eta = \sqrt{8}\times 3\text{e-}4 \approx 2.828 \times 3\text{e-}4 \approx 8.49\text{e-}4$.

    (b) The target batch of $4\text{M}$ tokens is right around the stated critical batch size $B^*$ (a few million tokens). Near or beyond $B^*$ you are leaving the small-batch regime where linear scaling holds and entering the saturated regime where gradient variance no longer falls as $1/B$; linear scaling then over-scales the LR and risks instability. So the **square-root value, $\approx 8.49\text{e-}4$**, is the safer, more conservative choice — the chapter explicitly recommends sqrt scaling "if you're uncertain whether you've exceeded the critical batch size." (For reference, both candidates still sit within the published $1\text{e-}4$ to $3\text{e-}3$ peak-LR band for a 7B model, so neither is absurd — but sqrt is the prudent pick this close to $B^*$.)

    (c) With the total token budget fixed, steps $=$ total tokens / tokens-per-step, so multiplying tokens/step by 8 divides the step count by **8** (e.g., 200K steps becomes 25K steps). If you keep a cosine schedule, `num_training_steps` must be updated to this new, smaller value so the decay still lands correctly at the end.

**4.** The gradient-accumulation code divides the per-microbatch loss by `accumulation_steps` before calling `.backward()`. (a) Assume each of $k$ microbatches contains exactly $m$ tokens and its loss is the mean cross-entropy over those $m$ tokens. Show that summing the $k$ divided-and-backpropagated microbatch gradients equals the gradient of the mean loss over all $km$ tokens. (b) Now suppose the microbatches have *different* token counts $m_1, \dots, m_k$. Explain why dividing every microbatch loss by the same constant $k$ no longer reproduces the true full-batch mean gradient, and state the correct weighting.

??? note "Solution"
    Let $\ell_j(\theta)$ be the per-token cross-entropy on token $j$. Gradients are linear, so $\nabla$ of a sum is the sum of $\nabla$s.

    (a) Microbatch $i$ has loss $L_i = \frac{1}{m}\sum_{j \in \text{mb}_i} \ell_j$. The code backpropagates $L_i / k$, and since `.backward()` *accumulates* into `.grad`, after all $k$ microbatches the stored gradient is
    $$
    \sum_{i=1}^{k} \nabla \frac{L_i}{k} = \frac{1}{k}\sum_{i=1}^{k} \nabla\!\left(\frac{1}{m}\sum_{j\in\text{mb}_i}\ell_j\right) = \frac{1}{km}\sum_{j=1}^{km}\nabla \ell_j = \nabla\!\left(\frac{1}{km}\sum_{j=1}^{km}\ell_j\right).
    $$
    The right-hand side is exactly the gradient of the mean loss over the full effective batch of $km$ tokens. So dividing by $k$ makes accumulation mathematically identical to one big averaged batch — this is the "mathematically equivalent... if loss is averaged (not summed)" point in the text.

    (b) With unequal counts, microbatch $i$'s mean loss $L_i = \frac{1}{m_i}\sum_{j\in\text{mb}_i}\ell_j$ already normalizes by its *own* $m_i$. Dividing again by the constant $k$ gives accumulated gradient $\frac{1}{k}\sum_i \frac{1}{m_i}\sum_{j\in\text{mb}_i}\nabla\ell_j$, which weights each *token* by $\frac{1}{k\,m_i}$ — tokens in a small microbatch get more weight than tokens in a large one. The true full-batch mean weights every token equally by $\frac{1}{\sum_i m_i}$. To recover it you must weight each microbatch by its token share: scale microbatch $i$'s loss by $m_i / \sum_i m_i$ (equivalently, sum the *token-summed* losses and divide once by the total token count $\sum_i m_i$), not by a flat $1/k$.

**5.** Implement a `get_linear_schedule_with_warmup` function in the same `LambdaLR` style as the chapter's `get_cosine_schedule_with_warmup`: linear ramp over `num_warmup_steps`, then a *linear* decay to a floor of `min_lr_fraction` at `num_training_steps`. Add a smoke test that checks the midpoint of the decay phase and the final value.

??? note "Solution"
    During warmup the multiplier is $t/T_w$, identical to the cosine version. During decay, `progress` runs $0 \to 1$ and the multiplier interpolates *linearly* from $1$ down to `min_lr_fraction`: multiplier $= 1 - (1 - f)\,\text{progress}$, where $f=$ `min_lr_fraction`.

    ```python
    import math
    import torch
    from torch.optim.lr_scheduler import LambdaLR


    def get_linear_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_fraction: float = 0.1,
    ) -> LambdaLR:
        """Linear warmup, then linear decay from peak to min_lr_fraction * peak."""
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            progress = min(progress, 1.0)  # clamp so we never go below the floor
            # Linear interpolation from 1.0 down to min_lr_fraction.
            return 1.0 - (1.0 - min_lr_fraction) * progress

        return LambdaLR(optimizer, lr_lambda)


    if __name__ == "__main__":
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=1000,
            min_lr_fraction=0.1,
        )

        lrs = []
        for _ in range(1000):
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # Peak is 3e-4. Decay midpoint is step 550 (progress = 0.5):
        # multiplier = 1 - 0.9 * 0.5 = 0.55  ->  1.65e-4.
        assert abs(lrs[550] - 3e-4 * 0.55) < 1e-9, "decay midpoint wrong"
        # Step 999 (progress = 899/900 ~ 0.999) is just above the 3e-5 floor.
        assert lrs[-1] < 3e-4 * 0.101, "floor not reached"
        print(f"Peak LR: {max(lrs):.2e}, Step 550 LR: {lrs[550]:.2e}, "
              f"Final LR: {lrs[-1]:.2e}")
        # Output: Peak LR: 3.00e-04, Step 550 LR: 1.65e-04, Final LR: 3.03e-05
    ```

    Contrast with cosine: at the decay midpoint both schedules happen to give the same $1.65\text{e-}4$ here (cosine's $0.5(1+\cos\tfrac{\pi}{2}) = 0.5$ coincides with linear's $0.5$), but away from the midpoint linear decays at a constant rate while cosine stays flatter near the peak and steeper only near the tail — which is precisely why the chapter says linear "decays too aggressively in the middle of the run."

**6.** You tune hyperparameters on a muP proxy model of width $d_{\text{proxy}} = 256$ and find an optimal base LR of $1\text{e-}2$. (a) Using the chapter's `build_mup_optimizer` convention (`lr_scale = proxy_width / actual_width`), what effective LR does a hidden `MuPLinear` layer of width $2048$ receive when you scale up? (b) In *standard* parameterization the optimal LR for hidden matrices scales roughly as $1/d$. If you instead grid-searched at width 256 in SP and naively reused that LR at width $4096$, by what factor would you likely be *off*? (c) When you run the chapter's coordinate check, what qualitative signature in the "muP mean|act|" column versus the "SP mean|act|" column tells you the muP implementation is correct?

??? note "Solution"
    (a) `lr_scale` $= d_{\text{proxy}} / d_{\text{actual}} = 256 / 2048 = 1/8$. Effective LR $= 1\text{e-}2 \times 1/8 = 1.25\text{e-}3$. The base number you *swept* stays $1\text{e-}2$; muP folds the width dependence into the per-layer multiplier, so you never re-tune it.

    (b) Going from width 256 to 4096 is a $16\times$ increase. Under SP the optimal hidden-matrix LR scales as $1/d$, so it should drop by about $16\times$. Reusing the width-256 value unchanged at width 4096 would leave you roughly **$16\times$ too high** — squarely the kind of blow-up that motivates muP. (That is exactly the width re-tuning muP eliminates: the whole point is that the *tuned* number is width-invariant.)

    (c) Under correct muP the per-layer activation (or update) scale is **flat across width** — the "muP mean|act|" column stays roughly constant (within ~2x) across the 256 -> 2048 sweep. Under SP the same quantity **drifts monotonically** as width grows (in the chapter's example it shrinks ~15-20x, e.g. ~0.24 -> ~0.014). So the signature of a correct implementation is: muP column flat, SP column clearly not flat. If your muP column is *not* flat, the bug is almost always in the init std, the per-layer LR multiplier, or the attention/readout scaling.
