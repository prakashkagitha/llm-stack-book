# 3.10 Learning Rate Schedules, Warmup, Batch Size & Hyperparameters

Hyperparameter tuning is one of the most impactful — and most under-documented — parts of pretraining a large language model. A 10x learning rate error will destroy a run that would otherwise converge cleanly; the right schedule can shave percentage points off final perplexity. Yet the wisdom lives mostly in appendices of papers and in the institutional memory of ML engineering teams.

This chapter makes that implicit knowledge explicit. We cover the full pipeline: why warmup is mandatory at scale, the major schedule families and when to use each, how batch size interacts with learning rate and what the *critical batch size* tells you about compute efficiency, and finally how muP (maximal-update parameterization) lets you tune hyperparameters on a small model and transfer them to a large one. Every section includes runnable code and concrete numbers.

Related chapters you should read in tandem: [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) covers the adaptive optimizer mechanics that schedules ride on top of; [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html) covers what happens when you get the schedule wrong; [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) gives the big picture that informs budget allocation decisions. For every number in this chapter pinned down for one concrete run, see the capstone's [Optimizer & Schedule: Muon + MuonClip and Warmup-Stable-Decay](../14-capstone/06-optimizer-and-schedule.html), which sets the peak LR, warmup, and WSD phase lengths for the ~100M-parameter Stack-100M model by measurement rather than by folklore.

## Why Schedules and Warmup Exist

Deep learning optimizers like Adam carry two moving-average estimates: first moment $m_t$ (gradient direction) and second moment $v_t$ (gradient magnitude squared). Both are initialized to zero. During the first few hundred steps, $v_t$ is still a noisy underestimate of the true gradient variance — the bias-correction terms in Adam partially compensate, but the effective learning rate is still erratic early in training.

For large models, this cold-start instability is catastrophic. At step 1 of a run with 1 billion parameters and a 4096-token context, the weight matrices are random; the gradient norms are large and wildly variable across layers. Applying the full target learning rate immediately produces parameter updates large enough to push weights into regions where softmax logits saturate, norms explode, or residual magnitudes collapse — none of which recover easily. Empirically, runs without warmup frequently spike or diverge within the first thousand steps.

Warmup solves this by linearly ramping the effective learning rate from near-zero to the target value over a set number of steps, giving the optimizer time to calibrate its momentum estimates and giving the network time to find a reasonable initialization basin before the full update magnitude kicks in.

There is a second, complementary mechanism worth holding in your head, because it explains why warmup helps even for optimizers with no bias-correction problem. What actually matters for stability is the *relative* update size $\lVert \Delta\theta \rVert / \lVert \theta \rVert$ per layer. Adam's update has a roughly fixed per-element magnitude of about $\eta$ (the $\sqrt{v_t}$ denominator normalizes the gradient scale away), while the weight norm $\lVert \theta \rVert$ at initialization is small and only grows as training proceeds. So at step 0 a full-magnitude LR is an enormous *fractional* change to every layer; a few thousand steps later the same LR is a modest one. Warmup is the schedule that keeps that ratio bounded while the weights are still small — an analysis developed carefully by Kosson et al. (*Why Warmup the Learning Rate?*, 2024). This framing also predicts the practical corollary you will meet below: anything that resets the weight/optimizer scale relationship (a fresh optimizer state, a sudden batch-size jump) calls for re-warming.

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

For years the default pretraining schedule, and still one of the two standard choices in 2026 (alongside WSD, below). After warmup, the learning rate follows the right half of a cosine curve, smoothly decaying to a floor $\eta_{\min}$ (usually $\eta_{\max}/10$ or a small constant like $1\text{e-}5$):

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

where $f$ is a decay function. Cosine and linear both work; MiniCPM reported that a $1-\sqrt{\cdot}$ shape (fast at first, flattening at the end) beats linear, and unlike cosine pretraining practice the WSD decay usually goes all the way to (near) zero rather than stopping at $\eta_{\max}/10$ — the last bit of decay is where the characteristic extra loss drop lives. A common budget is $T_d \approx 10\%$ of total steps.

The insight driving WSD is that most of the loss reduction happens in the stable phase, and the decay phase mainly "polishes" the model. This decoupling buys three things:

- **Late binding of the run length.** You can train in the stable phase for as long as resources allow, then trigger the decay when you're ready to finalize — no commitment to a total step count at the start.
- **Branching.** One stable checkpoint can seed several independent decay runs (different data mixes, different decay lengths) that you compare head-to-head for a fraction of a full run's cost. A *decayed* checkpoint, by contrast, is spent: resuming high-LR training from it throws away the polish.
- **A natural seam for a data-mixture switch.** Because the highest-value tokens are the ones seen at the lowest LR, the decay window is exactly where you swap the broad web mix for premium data. This is the whole idea behind mid-training/annealing ([Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html)), and the capstone does precisely this: the WSD decay phase *is* the mid-training phase ([Mid-Training](../14-capstone/08-mid-training.html)).

Hägele et al. (*Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*, 2024) ran the careful comparison and found that constant-LR-plus-cooldown matches cosine's final loss at equal compute while producing a whole family of usable checkpoints along the way — which is why WSD is now a first-class choice rather than a curiosity.

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

### The Library Equivalents

You should write the above once to own the mechanics; in a real run you will usually call a library. Every schedule in this chapter already exists in the standard stack, and knowing the exact knob name in each framework is most of the practical battle.

```python
import torch

model = torch.nn.Linear(16, 16)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

# --- HuggingFace transformers: the most-used scheduler factory in the ecosystem ---
# (requires `pip install transformers`; get_wsd_schedule needs a recent version)
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_inverse_sqrt_schedule,   # the rsqrt schedule from Vaswani et al.
    get_wsd_schedule,            # Warmup-Stable-Decay
)

sched = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=2000, num_training_steps=100_000
)
# NOTE: get_cosine_schedule_with_warmup decays to ZERO, not to a 10% floor.
# For a floor, pass a min-LR argument if your version exposes one, or use the
# from-scratch LambdaLR above. Check the signature for the version you pinned —
# get_wsd_schedule's phase arguments in particular have changed across releases.

# Via the Trainer, the same thing is one string:
#   TrainingArguments(lr_scheduler_type="cosine", warmup_ratio=0.02,
#                     learning_rate=3e-4, max_grad_norm=1.0, weight_decay=0.1)

# --- Pure PyTorch: compose warmup and decay with SequentialLR ---
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=2000)
decay = CosineAnnealingLR(optimizer, T_max=100_000 - 2000, eta_min=3e-5)
sched = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[2000])
```

Framework-level trainers expose the same schedules as config rather than code:

```yaml
# DeepSpeed config (ds_config.json): warmup + linear decay, no Python needed.
scheduler:
  type: WarmupDecayLR
  params:
    warmup_min_lr: 0.0
    warmup_max_lr: 3.0e-4
    warmup_num_steps: 2000
    total_num_steps: 100000
```

```bash
# Megatron-LM: the schedule is entirely CLI flags on pretrain_gpt.py.
# --lr-decay-style accepts constant | linear | cosine | inverse-square-root;
# recent versions also ship a WSD style with its own decay-length flags.
# Use --lr-warmup-fraction 0.02 instead of --lr-warmup-iters to express warmup
# as a fraction of --lr-decay-iters.
SCHEDULE_ARGS=(
  --lr 3.0e-4
  --min-lr 3.0e-5
  --lr-decay-style cosine
  --lr-warmup-iters 2000
  --lr-decay-iters 100000
  --clip-grad 1.0
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)
echo "${SCHEDULE_ARGS[@]}"   # torchrun ... pretrain_gpt.py "${SCHEDULE_ARGS[@]}"
```

PyTorch's `torchtitan` reference stack similarly exposes warmup/stable/decay phase lengths and a decay type in its TOML job config, and TRL/Axolotl fine-tuning configs forward `lr_scheduler_type` straight through to the `transformers` factories above. See [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html) for the surrounding launch machinery.

### Making the Schedule Restart-Safe

Long pretraining runs get preempted, and the schedule must survive the restart exactly. There is a footgun here: `LambdaLR.state_dict()` cannot serialize a plain-function `lr_lambda` (it is skipped, with a warning), so a naive `torch.save(scheduler.state_dict())` round-trip can silently restore only `last_epoch` — and if you rebuild the scheduler with different `num_training_steps` on resume, you get a *different curve* with no error.

The robust pattern used by most production trainers, including the capstone's `wsd_lr`, is to make the schedule a **pure function of the global step** and set the LR by hand each iteration. There is no hidden state to checkpoint beyond the step counter you were already saving:

```python
import math

def lr_at_step(step: int, peak_lr: float, warmup: int, total: int,
               min_lr_fraction: float = 0.1) -> float:
    """Stateless cosine-with-warmup: the ONLY input is the global step."""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_fraction + (1.0 - min_lr_fraction) * cos)


# In the training loop, before optimizer.step():
#   lr = lr_at_step(global_step, 3e-4, 2000, 100_000)
#   for g in optimizer.param_groups:
#       g["lr"] = lr * g.get("lr_scale", 1.0)   # lr_scale carries muP / per-group ratios
```

The `lr_scale` multiplier in that last line matters whenever different parameter groups run at different peak LRs — muP's per-layer width factors, or a Muon/AdamW hybrid where the two groups sit at a fixed ratio to each other. Writing one scalar LR into *every* group collapses the ratio and is a classic silent bug; see [Optimizer & Schedule](../14-capstone/06-optimizer-and-schedule.html) for the concrete Muon-plus-AdamW case. Checkpointing more broadly is covered in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html).

## Learning Rate vs. Batch Size Scaling

### The Linear Scaling Rule

When you increase the batch size $B$ by a factor $k$, each gradient step is an average over $k$ times more samples, reducing variance by $\sqrt{k}$ and the signal-to-noise ratio effectively improves. To maintain the same training dynamics — the same total parameter update magnitude per unit of data — you should also scale the learning rate:

$$
\eta' = k \cdot \eta \quad \text{(linear scaling rule, Goyal et al., 2017)}
$$

This rule was derived and validated for **SGD with momentum**, for small-to-moderate batch changes (say, 256 to 4096). The intuition: with $k\times$ larger batches, each step is $k\times$ more expensive in wall time (same flops per sample), but covers $k\times$ more data, so to move at the same "rate through the data manifold," the step size should scale linearly.

### The Square-Root Scaling Rule

$$
\eta' = \sqrt{k} \cdot \eta
$$

Two independent arguments land on the square root, and for LLM pretraining they are the ones that matter.

**It is the correct rule for adaptive optimizers.** This is the point most treatments of the linear rule get wrong. Adam's update divides the gradient by $\sqrt{v_t}$, an estimate of the gradient's *second* moment — so the noise that averaging removes appears inside a square root, and the noise-preserving scaling becomes $\eta \propto \sqrt{k}$ rather than $\eta \propto k$. Malladi et al. (*On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*, 2022) derive this from the stochastic-differential-equation limit of Adam and RMSProp and verify it empirically. Since essentially all LLM pretraining uses AdamW (or a normalized-update optimizer like Muon, whose update magnitude is fixed by construction and so behaves similarly), **square-root scaling should be your default when changing batch size, and linear scaling the special case you reach for only with SGD.**

**It is also the conservative choice near saturation.** Independently of the optimizer, at very large batch the gradient variance stops falling as $1/B$ — you hit the intrinsic noise floor of the data distribution rather than sampling noise — so any rule that keeps growing the LR with $B$ eventually over-scales. The square root degrades gracefully where linear does not.

The capstone uses exactly this rule to move a peak LR measured on a small probe batch (65,536 tokens/step) to the real run's 524,288 tokens/step: $\times\sqrt{8} \approx 2.83$ ([Optimizer & Schedule](../14-capstone/06-optimizer-and-schedule.html)).

### Critical Batch Size

The *critical batch size* $B^*$ is the regime boundary between these two laws. It was formalized by McCandlish et al. (*An Empirical Model of Large-Batch Training*, 2018). The key quantity is the *gradient noise scale*:

$$
B_{\text{noise}} = \frac{\text{tr}(\Sigma)}{\|G\|^2}
$$

where $\Sigma$ is the covariance of the per-sample gradient and $G$ is the mean gradient. When $B \ll B_{\text{noise}}$, batches are too small to average out noise — increasing batch size linearly reduces steps needed. When $B \gg B_{\text{noise}}$, you are in the *saturated* regime where more data per step doesn't help; gradient noise is already small and increasing $B$ wastes compute.

The practical takeaway: there is an optimal batch size for a given compute budget. Doubling the batch size beyond $B^*$ halves your throughput efficiency. For typical LLM pretraining, $B^*$ for cross-entropy loss on language is on the order of a few million tokens — consistent with the token batches used in Chinchilla-optimal runs — while for a ~100M model it sits closer to a few hundred thousand tokens, which is why the capstone lands on 524,288 tokens/step.

An important 2024–2025 refinement: the critical batch size scales primarily with the **amount of data trained on**, not with parameter count. Zhang et al. (*How Does Critical Batch Size Scale in Pre-training?*, 2024) hold one of the two fixed at a time and find $B^*$ grows with tokens seen and is close to flat in model size once data is controlled. The practical consequence is that a long over-trained run of a small model can tolerate a surprisingly large batch, and that $B^*$ *increases as your run proceeds* — a reason ramping batch size over training (see the practitioner tip at the end of this chapter) is principled rather than folklore.

{{fig:lrsched-critical-batch-size}}

You do not have to take $B^*$ on faith: the noise scale is directly measurable from two batch sizes. McCandlish et al. give unbiased estimators for $\lVert G\rVert^2$ and $\operatorname{tr}(\Sigma)$ from the squared gradient norms at a small and a large batch, and the ratio is $B_{\text{noise}}$. Run this for a few hundred steps on your actual model and data before you pick a batch size.

```python
import torch


@torch.no_grad()
def _sq_grad_norm(model) -> float:
    """Squared global L2 norm of the current .grad tensors."""
    return float(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                     if p.grad is not None))


def gradient_noise_scale(model, loss_fn, batch, b_small: int, b_big: int) -> float:
    """
    Estimate B_noise = tr(Sigma) / ||G||^2 (McCandlish et al., 2018) from one
    large batch and its first `b_small` examples.

    The estimators exploit E[||g_B||^2] = ||G||^2 + tr(Sigma)/B, evaluated at two
    batch sizes and solved as a 2x2 linear system:
        ||G||^2   ~ (b_big*||g_big||^2 - b_small*||g_small||^2) / (b_big - b_small)
        tr(Sigma) ~ (||g_small||^2 - ||g_big||^2) / (1/b_small - 1/b_big)
    Both are noisy per-step; average them over ~100 steps before dividing.
    """
    model.zero_grad(set_to_none=True)
    loss_fn(model, batch[:b_small]).backward()
    g_small = _sq_grad_norm(model)

    model.zero_grad(set_to_none=True)
    loss_fn(model, batch[:b_big]).backward()
    g_big = _sq_grad_norm(model)

    g_norm_sq = (b_big * g_big - b_small * g_small) / (b_big - b_small)
    trace_sigma = (g_small - g_big) / (1.0 / b_small - 1.0 / b_big)
    return trace_sigma / max(g_norm_sq, 1e-12)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Linear(32, 4)
    data = torch.randn(256, 32)
    targets = torch.randint(0, 4, (256,))

    def loss_fn(m, idx_slice):
        n = idx_slice.shape[0]
        return torch.nn.functional.cross_entropy(m(data[:n]), targets[:n])

    # Average the two moments over several draws, then divide (never average ratios).
    est = sum(gradient_noise_scale(model, loss_fn, data, 8, 128) for _ in range(20)) / 20
    print(f"B_noise estimate: {est:.1f} examples")
    # In a real run: repeat every few hundred steps and watch B_noise GROW as
    # the gradient shrinks -- that growth is the signal to ramp the batch size.
```

In a distributed run the same estimate falls out for free: compare the per-rank gradient norm (small batch) with the post-all-reduce global gradient norm (large batch), which is what production monitoring typically logs. See [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html).

!!! example "Worked Example: Batch-LR Pair for a 7B Pretraining Run"

    Suppose your baseline is:
    - $\eta_{\max} = 3\text{e-}4$, batch size $B = 512$ samples × 2048 tokens = ~1M tokens/step.

    You want to scale to $B' = 2048$ samples × 2048 tokens = ~4M tokens/step, a factor $k = 4$.

    **Square-root rule (the AdamW default):** $\eta' = \sqrt{4} \times 3\text{e-}4 = 6\text{e-}4$.

    At 4M tokens/step, you'll also converge in roughly $1/4$ the steps for the same total token count. If original run had 100K steps, new run has 25K steps — so with a cosine schedule you must also divide `num_training_steps` by 4, or the decay will never reach the floor.

    **Check: does 6e-4 violate any rule of thumb?** Under standard parameterization the optimal global LR shrinks as width grows (roughly $1/d$ for hidden matrices); for a 7B model with hidden dim $d = 4096$, published peak LRs fall in the range $1\text{e-}4$ to $3\text{e-}3$. (Under muP the *base* LR you tune is width-invariant instead — the $1/d$ factor lives in the per-layer LR multiplier, not in the number you sweep.) So 6e-4 sits comfortably mid-range.

    **Linear rule (SGD-style, aggressive here):** $\eta' = 4 \times 3\text{e-}4 = 1.2\text{e-}3$. Still inside the published band, so it will not blow up on step one — but with AdamW it over-scales, and at 4M tokens/step you are already near the critical batch size where the extra factor buys nothing. Prefer the square root unless a probe run says otherwise.

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

Notice what the update equation implies: the shrinkage per step is $\lambda\eta$, **not** $\lambda$. Weight decay and the learning rate are therefore not independent knobs — they multiply. Two consequences follow, and both bite in practice:

- **Your schedule decays your regularization too.** As $\eta$ falls along a cosine or a WSD decay phase, the effective pull toward zero falls with it. The characteristic timescale over which a weight is forgotten is roughly $\tau \approx 1/(\lambda\eta)$ steps; at $\lambda=0.1,\ \eta=3\text{e-}4$ that is about 33,000 steps — comparable to a whole run, which is why weight decay in LLM pretraining is a slow, global effect rather than a per-batch one.
- **If you change the peak LR, you have changed the regularization strength.** When comparing two LR settings on equal footing, either hold $\lambda\eta$ fixed or report that you did not.

This coupling is also why weight decay works at all in a regime with essentially no overfitting: for one-epoch LLM pretraining, decay is not primarily preventing memorization of a re-seen training set. Andriushchenko et al. (*Why Do We Need Weight Decay in Modern Deep Learning?*, 2023) argue its main role there is to keep weight norms — and hence the *effective* per-step learning rate seen by each layer — in a favorable range, i.e. it is an optimization-shaping term more than a classical regularizer. Practically: keep $\lambda = 0.1$, exclude 1D parameters, and treat $\lambda$ and $\eta$ as one joint decision.

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

The Microsoft `mup` library (github.com/microsoft/mup) provides a plug-and-play implementation. Its central concept is that you never describe a model in absolute terms — you give it a *base* model and a *delta* model differing only in the widths you intend to scale, and it infers which dimensions are "infinite" and rewrites init and per-parameter LR accordingly:

```python
# Sketch of the mup API (pip install mup). Requires MuReadout on the output head.
import torch.nn as nn
import mup

def MyTransformer(d_model):                      # stand-in for your real model
    return nn.Sequential(nn.Linear(64, d_model), nn.ReLU(),
                         mup.MuReadout(d_model, 64))

base_lr = 1e-2                                   # tuned at the base width
base = MyTransformer(d_model=256)                # base shapes
delta = MyTransformer(d_model=512)               # differs ONLY in scaled dims
model = MyTransformer(d_model=4096)              # the model you will train

mup.set_base_shapes(model, base, delta=delta)    # MUST come before init + optimizer
for p in model.parameters():
    if p.ndim >= 2:
        mup.init.normal_(p, mean=0.0, std=0.02)  # mup-aware init, reads base shapes

optimizer = mup.MuAdam(model.parameters(), lr=base_lr)   # or MuAdamW / MuSGD
# mup.coord_check.get_coord_data(...) produces the coordinate-check plot above.
```

Two ordering rules cause most `mup` bugs: `set_base_shapes` must run *before* you initialize weights and *before* you construct the optimizer, and the readout layer must be `mup.MuReadout` (or hand-scaled by $1/d$) or the width invariance silently fails.

The typical workflow is:

1. **Define a base config** with a small proxy width (e.g., 256 hidden dim).
2. **Run a dense HP sweep** over LR, initialization std, and possibly weight decay — this is cheap at small width.
3. **Identify the optimal HP at proxy width.**
4. **Scale width** to the full model. muP guarantees the same HP is optimal (up to reasonable approximation in practice).
5. **Validate on a medium model** (e.g., 1B) before launching the full run.

The evidence that this works is now substantial: Microsoft's Phi models, various internal runs at other labs, and controlled ablations in the *Tensor Programs V* paper all show that muP-transferred LRs closely match the empirically optimal LRs found by grid search at the large scale — saving orders-of-magnitude in tuning compute.

Two notes on how this looks in 2026 practice. First, many teams no longer depend on the `mup` package: because the prescription reduces to a handful of rules (init std $\propto 1/\sqrt{\text{fan\_in}}$, readout scaled by $1/d$, per-matrix LR $\propto 1/\text{fan\_in}$, attention logits divided by $d_k$ rather than $\sqrt{d_k}$), it is often written directly into the model definition — which is also what makes it survive `torch.compile` and FSDP wrapping without surprises. Second, **normalized-update optimizers get part of this for free.** Muon's orthogonalized update has a fixed per-element RMS by construction, independent of gradient scale and largely of width, so its peak LR transfers across width far better than Adam's — which is why the capstone can carry a peak LR measured on a 43M proxy up to full width with only a short confirmation run instead of a full muP apparatus ([Optimizer & Schedule](../14-capstone/06-optimizer-and-schedule.html)). muP and normalized optimizers are attacking the same problem — making the update magnitude scale-invariant — from opposite ends.

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
| ~100M | $6\text{e-}4$ | 1000–2000 | ~0.5M | 1.0 | 0.1 | 0.9, 0.95 |
| 125M | $6\text{e-}4$ | 2000 | ~0.5M | 1.0 | 0.1 | 0.9, 0.95 |
| 1B | $3\text{e-}4$ | 2000 | ~2M | 1.0 | 0.1 | 0.9, 0.95 |
| 7B | $3\text{e-}4$ | 2000 | ~4M | 1.0 | 0.1 | 0.9, 0.95 |
| 70B | $1.5\text{e-}4$ | 4000 | ~8M | 1.0 | 0.1 | 0.9, 0.95 |
| 400B+ | $\sim 1\text{e-}4$ | 4000–8000 | ~16M | 1.0 | 0.1 | 0.9, 0.95 |

These are starting points synthesized from published literature (GPT-3, Llama 1/2/3, Mistral, Falcon, OLMo) — treat them as reasonable defaults, not ground truth. The right value for any specific run depends on architecture choices (RMSNorm vs LayerNorm, activation function, depth/width ratio), the optimizer, and the data mixture. The peak-LR column assumes **AdamW on every parameter**; a Muon/AdamW hybrid runs the 2D matrices an order of magnitude higher (the capstone's ~100M model uses Muon at $0.02$ with AdamW at $3\text{e-}3$ for embeddings and norms), because Muon's orthogonalized update has a fixed per-element magnitude rather than a gradient-scaled one. See [Optimizers](../03-pretraining/09-optimizers.html) and, for the full worked recipe at this scale, [Optimizer & Schedule](../14-capstone/06-optimizer-and-schedule.html).

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
    Use 'sqrt' for AdamW/Muon (the SDE-derived rule for adaptive and
    normalized-update optimizers) -- this is the LLM-pretraining default.
    Use 'linear' only for SGD-with-momentum, and only well below B*.
    """
    ratio = target_batch / base_batch
    if rule == "linear":
        return base_lr * ratio
    elif rule == "sqrt":
        return base_lr * math.sqrt(ratio)
    else:
        raise ValueError(f"Unknown rule: {rule}")


# Example: scale 1M-token/step config to 4M tokens/step under AdamW
base_cfg = PretrainingHParams(peak_lr=3e-4)
new_lr = scale_lr_for_batch_size(
    base_lr=base_cfg.peak_lr,
    base_batch=1_000_000,
    target_batch=4_000_000,
    rule="sqrt",
)
print(f"Scaled LR (sqrt): {new_lr:.2e}")  # 6.00e-04
# The SGD-style linear rule would give 1.20e-03 -- 2x higher, and near B* that
# is exactly the over-scaling that shows up as a loss spike a few thousand
# steps in rather than as an immediate divergence.
```

### Warmup Duration Heuristics

A useful rule of thumb: warm up for at least $T_w = \max(1000, 0.02 \times T_{\text{total}})$ steps — that is, at least 1000 steps or 2% of the total budget, whichever is larger. Shorter warmups are fine for fine-tuning (where the model is already initialized near a good basin) but dangerous for pretraining from scratch.

State it in **tokens** when you compare across runs, since that is the quantity that is actually invariant: 2% of a 38,147-step run at 524,288 tokens/step is ~1B warmup tokens, which is squarely the ~0.5–2B band used by open pretraining recipes at every scale from 100M to 70B ([The Pretraining Run](../14-capstone/07-pretraining-run.html) uses exactly 2,000 steps ≈ 1.05B tokens). Warmup is cheap insurance: over-warming costs you a fraction of a percent of final loss, while under-warming can cost you the run. When a run diverges early, **lengthen warmup before you lower the peak LR** — it preserves the peak you tuned.

For continued pretraining (e.g., domain adaptation starting from a released checkpoint), a short warmup of 100–500 steps is usually sufficient — the parameters are already in a well-behaved regime. See [Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html).

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
[ ] LR scaled for batch size if changed from reference run (sqrt for AdamW/Muon)
[ ] Cosine or WSD schedule: total_steps (cosine) or stable_steps set
[ ] Min LR floor = 10% of peak LR
[ ] Weight decay = 0.1 (decoupled, excluded from embeddings/norms/biases)
[ ] Grad clip norm = 1.0; grad norm logged every step
[ ] beta2 = 0.95 (not 0.999 default)
[ ] Gradient accumulation: effective_batch = micro_batch * accum_steps * world_size
[ ] DDP: model.no_sync() used on non-final accumulation steps
[ ] Optimizer state saved in checkpoint; schedule reproducible from global_step
    alone (prefer a stateless lr_at_step over a pickled LambdaLR)
[ ] Per-group LR ratios (muP multipliers / Muon-vs-AdamW) preserved on resume
[ ] Monitoring: log {lr, grad_norm, loss, step} every step
```

!!! tip "Practitioner Tip: Curriculum for Batch Size"

    Some teams ramp the batch size up alongside the learning rate during warmup — starting with a small batch (say 128K tokens/step) and linearly increasing to the full target batch over the first 2000 steps. GPT-3 did this, and the critical-batch-size picture explains why it is principled rather than a trick: early in training the gradient is large and $B_{\text{noise}} = \operatorname{tr}(\Sigma)/\lVert G\rVert^2$ is *small*, so a big batch is genuinely wasted compute; as the gradient shrinks, $B_{\text{noise}}$ grows and the large batch starts paying for itself. Logging the noise-scale estimator above every few hundred steps turns this from a schedule you guess into one you read off a curve.

    Two implementation notes. The learning rate warmup and batch warmup interact — keep them synchronized so the update magnitude per token stays roughly constant. And ramping batch by raising `grad_accumulation_steps` (rather than `micro_batch_size`) keeps the memory footprint fixed, which is what you want on a fixed cluster allocation.

!!! key "Key Takeaways"

    - Warmup is mandatory at large scale because Adam's momentum estimates are cold-started at zero; ramping LR over 1000–4000 steps prevents early instability.
    - Cosine annealing is the dominant pretraining schedule; Warmup-Stable-Decay (WSD) is increasingly popular because it decouples training length from schedule shape.
    - Scale LR with batch size as $\sqrt{k}$ under AdamW (the SDE-derived rule for adaptive optimizers); the linear rule is the SGD-with-momentum special case. Beyond the critical batch size, increasing batch size no longer reduces the step count proportionally — and $B^*$ is measurable from the gradient noise scale, and grows with tokens seen.
    - Weight decay and LR multiply: the per-step shrinkage is $\lambda\eta$, so changing the peak LR silently changes the regularization strength.
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
    - [Malladi et al., *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms* (2022)](https://arxiv.org/abs/2205.10287) — derives the **square-root** LR scaling rule for Adam/RMSProp from their SDE limits; the reason LLM practice does not use Goyal's linear rule.
    - [Andriushchenko et al., *Why Do We Need Weight Decay in Modern Deep Learning?* (2023)](https://arxiv.org/abs/2310.04415) — argues that in one-epoch LLM pretraining weight decay is an optimization-shaping term (controlling the effective LR via weight norm) rather than a classical regularizer.
    - [Hägele et al., *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (2024)](https://arxiv.org/abs/2405.18392) — careful head-to-head showing constant-LR-plus-cooldown matches cosine at equal compute while yielding a family of usable checkpoints; the empirical case for WSD.
    - [Kosson et al., *Why Warmup the Learning Rate? Underlying Mechanisms and Improvements* (2024)](https://arxiv.org/abs/2406.09405) — explains warmup as bounding the *relative* per-step update size while weight norms are still small at initialization.
    - [Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective* (2024)](https://arxiv.org/abs/2410.05192) — theoretical explanation of WSD via the "river valley" loss landscape; explains why large LR oscillations during the stable phase are benign.
    - [Zhang et al., *How Does Critical Batch Size Scale in Pre-training?* (2024)](https://arxiv.org/abs/2410.21676) — controlled study finding that critical batch size scales with data seen rather than parameter count.
    - [Li et al., *Optimal Learning-Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay* (2026)](https://arxiv.org/abs/2602.06797) — rigorous theory showing a phase transition between power-decay and WSD as optimal schedules depending on task difficulty.

    **Open-source & tools**

    - [microsoft/mup](https://github.com/microsoft/mup) — reference PyTorch implementation of muP with MuAdam/MuSGD optimizers, coordinate-check utilities, and Transformer examples; the standard starting point for HP transfer workflows.
    - **HuggingFace `transformers`** — `get_{linear,cosine,constant}_schedule_with_warmup`, `get_inverse_sqrt_schedule`, and `get_wsd_schedule`; the same schedules are reachable from `TrainingArguments(lr_scheduler_type=..., warmup_ratio=...)` and are what TRL and Axolotl forward to under the hood.
    - **PyTorch** — `torch.optim.lr_scheduler` (`LambdaLR`, `LinearLR`, `CosineAnnealingLR`, `SequentialLR`) and `torch.nn.utils.clip_grad_norm_`; `torchtitan` exposes warmup/stable/decay phases as job config.
    - **Megatron-LM / DeepSpeed** — `--lr-decay-style` plus `--lr-warmup-iters`/`--lr-warmup-fraction` on the Megatron side, and the `WarmupDecayLR`/`WarmupLR` scheduler blocks in a DeepSpeed JSON config; both are how these schedules are actually configured on large clusters.

    **Go deeper**

    - [Groeneveld et al., *OLMo: Accelerating the Science of Language Models* (2024)](https://arxiv.org/abs/2402.00838) — fully open pretraining paper with detailed hyperparameter tables (LR, warmup, batch, weight decay) for 1B and 7B runs; a clean reference implementation of the standard recipe.

## Further Reading

- Yang, G. et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." NeurIPS 2022. (The muP paper; defines maximal-update parameterization and the theoretical framework behind HP transfer.)
- McCandlish, S. et al. "An Empirical Model of Large-Batch Training." arXiv, 2018. (Introduces the gradient noise scale and critical batch size.)
- Goyal, P. et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv, 2017. (The linear scaling rule for batch size — derived for SGD with momentum.)
- Malladi, S. et al. "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms." NeurIPS 2022. (The square-root scaling rule for Adam/RMSProp; the correct default for LLM pretraining.)
- Hägele, A. et al. "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations." NeurIPS 2024. (Constant LR plus cooldown matches cosine; the empirical case for WSD.)
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

    (b) Two independent reasons both point at the **square-root value, $\approx 8.49\text{e-}4$**. First, the optimizer: pretraining runs on AdamW, and the SDE analysis of adaptive methods (Malladi et al., 2022) gives $\eta \propto \sqrt{B}$, not $\eta \propto B$ — the linear rule was derived for SGD with momentum. Second, the regime: $4\text{M}$ tokens is right around the stated critical batch size $B^*$ (a few million tokens), and near or beyond $B^*$ gradient variance no longer falls as $1/B$, so any rule that keeps growing the LR with $B$ over-scales. (For reference, both candidates still sit within the published $1\text{e-}4$ to $3\text{e-}3$ peak-LR band for a 7B model, so neither is absurd — but sqrt is both the principled and the prudent pick here.)

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
