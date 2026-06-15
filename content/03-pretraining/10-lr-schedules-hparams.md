# 3.10 Learning Rate Schedules, Warmup, Batch Size & Hyperparameters

Hyperparameter tuning is one of the most impactful — and most under-documented — parts of pretraining a large language model. A 10x learning rate error will destroy a run that would otherwise converge cleanly; the right schedule can shave percentage points off final perplexity. Yet the wisdom lives mostly in appendices of papers and in the institutional memory of ML engineering teams.

This chapter makes that implicit knowledge explicit. We cover the full pipeline: why warmup is mandatory at scale, the major schedule families and when to use each, how batch size interacts with learning rate and what the *critical batch size* tells you about compute efficiency, and finally how muP (maximal-update parameterization) lets you tune hyperparameters on a small model and transfer them to a large one. Every section includes runnable code and concrete numbers.

Related chapters you should read in tandem: [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) covers the adaptive optimizer mechanics that schedules ride on top of; [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html) covers what happens when you get the schedule wrong; [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) gives the big picture that informs budget allocation decisions.

## Why Schedules and Warmup Exist

Deep learning optimizers like Adam carry two moving-average estimates: first moment $m_t$ (gradient direction) and second moment $v_t$ (gradient magnitude squared). Both are initialized to zero. During the first few hundred steps, $v_t$ is still a noisy underestimate of the true gradient variance — the bias-correction terms in Adam partially compensate, but the effective learning rate is still erratic early in training.

For large models, this cold-start instability is catastrophic. At step 1 of a run with 1 billion parameters and a 4096-token context, the weight matrices are random; the gradient norms are large and wildly variable across layers. Applying the full target learning rate immediately produces parameter updates large enough to push weights into regions where softmax logits saturate, norms explode, or residual magnitudes collapse — none of which recover easily. Empirically, runs without warmup frequently spike or diverge within the first thousand steps.

Warmup solves this by linearly ramping the effective learning rate from near-zero to the target value over a set number of steps, giving the optimizer time to calibrate its momentum estimates and giving the network time to find a reasonable initialization basin before the full update magnitude kicks in.

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

    **Check: does 1.2e-3 violate any rule of thumb?** For a 7B model with hidden dim $d = 4096$, muP-optimal LR scales as $1/d$ — at $d=4096$ we expect peak LRs in the range $1\text{e-}4$ to $3\text{e-}3$. So 1.2e-3 is well within range.

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
            param_groups.append({
                "params": list(module.parameters()),
                "lr": base_lr * lr_scale,
                "weight_decay": weight_decay,
                "name": name,
            })

    # All other parameters (norms, embeddings) get base LR
    named_param_set = {
        id(p)
        for m in model.modules()
        if isinstance(m, MuPLinear)
        for p in m.parameters()
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

### Practical muP Workflow

The Microsoft `mup` library (github.com/microsoft/mup) provides a plug-and-play implementation. The typical workflow is:

1. **Define a base config** with a small proxy width (e.g., 256 hidden dim).
2. **Run a dense HP sweep** over LR, initialization std, and possibly weight decay — this is cheap at small width.
3. **Identify the optimal HP at proxy width.**
4. **Scale width** to the full model. muP guarantees the same HP is optimal (up to reasonable approximation in practice).
5. **Validate on a medium model** (e.g., 1B) before launching the full run.

The evidence that this works is now substantial: Microsoft's Phi models, various internal runs at other labs, and controlled ablations in the *Tensor Programs V* paper all show that muP-transferred LRs closely match the empirically optimal LRs found by grid search at the large scale — saving orders-of-magnitude in tuning compute.

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
