# 3.11 Training Stability, Loss Spikes & Debugging Large Runs

Pretraining a large language model is one of the most expensive experiments a human team can run. At the scale of hundreds of billions of parameters and trillions of tokens, a single unrecovered divergence can waste millions of dollars of compute and weeks of calendar time. Yet instability is not rare — it is an almost universal companion of large-scale training. Every serious pretraining team has stories of mysterious loss spikes, subtle data bugs discovered only after 10 days of training, and GPU clusters operating at 40 % throughput while engineers hunt a deadlock.

This chapter is the field manual for that experience. We cover the mechanisms behind instability, the mitigations built into modern architectures and training pipelines, the monitoring infrastructure that surfaces problems before they become catastrophes, and the decision-making playbook you need when things go wrong at 3 AM.

Related context that we assume you have read or will read alongside this chapter: [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) for floating-point root causes, [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) for optimizer dynamics, [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html) for schedule interactions, and [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html) for recovery mechanics.

---

## What a Stable Run Looks Like — and When to Worry

Before diagnosing instability, you need a baseline for health. A well-configured pretraining run exhibits a smooth loss curve that decreases monotonically on a log scale, with mild stochastic noise but no sustained plateaus or upward excursions larger than roughly 0.05–0.10 nats over a few hundred steps.

```text
Healthy run:                        Spike event:
 loss                                loss
  |                                   |
3.5|*                              3.5|*
   | **                                | **
3.0|   ***                         3.0|   ***
   |      *****                        |      *****
2.5|           *******             2.5|           ***  /\  ****
   |                 ****              |                \/
2.0|                     ****      2.0|
   +----------------------- step      +----------------------- step
```

**Warning thresholds (guidelines, not hard rules):**

| Signal | Normal | Yellow | Red |
|---|---|---|---|
| Per-step loss jump | < 0.03 nats | 0.03–0.15 | > 0.15 |
| Gradient norm | < 1× peak warmup norm | 2–5× | > 10× |
| Loss plateau (no decrease) | — | 200 steps | 1 000 steps |
| Max activation value (fp16) | < 300 | 300–1 000 | > 1 000 (overflow risk) |
| NaN or Inf count per step | 0 | 0 | any nonzero |

These numbers come from practical experience across GPT-class, Llama-class, and similar training runs. The exact thresholds depend on model size, learning rate, and optimizer — but the *relative* ratios are robust.

---

## Anatomy of a Loss Spike

A loss spike is a sudden increase in training loss, typically resolved by the optimizer over tens to hundreds of subsequent steps. Understanding what actually happens mechanically is important for choosing the right mitigation.

### The causal chain

A spike almost always follows this causal chain:

$$
\text{bad batch OR bad state} \;\longrightarrow\; \text{large gradient} \;\longrightarrow\; \text{large parameter update} \;\longrightarrow\; \text{bad model state} \;\longrightarrow\; \text{elevated loss}
$$

The "bad state" can be a transient — the model recovers on its own — or it can be absorbing: the optimizer finds a new basin with higher loss, and the run effectively diverges. Whether a spike is transient or absorbing depends heavily on the magnitude relative to the Adam/AdamW second-moment estimate, which we discuss in the next section.

### Adam's role in amplifying spikes

Adam (and its variants) maintains a running estimate of the second moment $\hat{v}_t$ for each parameter gradient. When a gradient is unusually large, the effective learning rate for that parameter is:

$$
\Delta \theta = -\frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

If the large gradient is *new* (not predicted by the historical $\hat{v}_t$), the denominator is small (accumulated from previous, smaller gradients), so the parameter update is disproportionately large. This is the primary amplification mechanism. After the spike, $\hat{v}_t$ quickly absorbs the new large value, damping future updates — which is why spikes are usually transient. But if the first large update throws the model into a region of high curvature, recovery may take hundreds of steps or fail entirely.

!!! example "Worked example: spike magnitude with Adam"
    Take AdamW with $\beta_1 = 0.9$ (so $1-\beta_1 = 0.1$), learning rate $\alpha = 3 \times 10^{-4}$, and $\epsilon = 10^{-8}$. Suppose a parameter has historical gradient RMS $g_\text{rms} = 0.01$, so the second-moment estimate is $\hat{v} \approx g_\text{rms}^2 = 10^{-4}$ and $\sqrt{\hat{v}} \approx 0.01$ (the $\epsilon$ term is negligible here). Crucially, $\hat{v}$ updates slowly ($\beta_2$ close to 1, e.g. $0.999$, so $1-\beta_2 \approx 10^{-3}$): on the step a spike arrives, $\sqrt{\hat{v}}$ still reflects the *historical* gradient scale, not the spike.

    Model the first moment as tracking the current gradient with the fresh-moment factor $\hat{m} \approx (1-\beta_1)\,g$, and hold the stale denominator $\sqrt{\hat{v}} \approx 0.01$ fixed for the arriving step. The update magnitude is then $|\Delta\theta| \approx \alpha\,(1-\beta_1)\,g / \sqrt{\hat{v}}$.

    A **normal** step with $g = g_\text{rms} = 0.01$:

    $$
    |\Delta \theta|_\text{normal} \approx \frac{\alpha\,(1-\beta_1)\,g}{\sqrt{\hat{v}}} = \frac{3\times10^{-4} \times 0.1 \times 0.01}{0.01} = 3\times10^{-5}
    $$

    A **spike** step where a bad batch produces $g = 1.0$ (100x the historical RMS), with the *same* stale $\sqrt{\hat{v}} = 0.01$:

    $$
    |\Delta \theta|_\text{spike} \approx \frac{\alpha\,(1-\beta_1)\,g}{\sqrt{\hat{v}}} = \frac{3\times10^{-4} \times 0.1 \times 1.0}{0.01} = 3\times10^{-3}
    $$

    The spike update is **100x larger** than a normal step — exactly the ratio of the gradients ($1.0 / 0.01$), because the denominator $\sqrt{\hat{v}}$ has not yet absorbed the spike. That factor is enough to blow the model out of a good basin. Once $\hat{v}$ catches up over the next few hundred steps the denominator grows and the amplification fades, which is why spikes are usually transient.

    **With gradient clipping** at global norm $\tau = 1.0$: during the spike the *global* gradient norm is large — say $\|g\| \approx 10$ — so clipping rescales every gradient by $\tau/\|g\| \approx 0.1$, dropping this parameter's gradient from $1.0$ to $\approx 0.1$. The update becomes $\alpha\,(1-\beta_1)\times 0.1 / \sqrt{\hat{v}} = 3\times10^{-4}$, i.e. **~10x normal** instead of 100x — an order of magnitude of damage removed, survivable in most cases.

{{fig:adam-spike-amplification}}

---

## Root Causes of Instability

### Bad data batches

The single most common cause of spikes in practice is a batch containing anomalous content: repeated tokens, extremely long documents (padding or truncation bugs), encoding corruption, or near-duplicate toxic sequences that the model assigns very low probability to. The resulting loss is high, the gradient is large, and Adam amplifies it.

**Canonical bad-data patterns:**

- **Repeated n-grams or copy-paste artifacts.** A sequence like `aaaa...aaaa` (10 000 repetitions) has near-zero cross-entropy under a well-trained model but produces high loss on an in-progress model, and the gradient is sharply peaked at the repetition token.
- **Mixed-language documents with incorrect tokenization.** A Chinese document tokenized with a primarily-English BPE vocabulary produces absurdly long token sequences, often hitting the sequence-length limit mid-word. See [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) for how this manifests.
- **Numeric/code sequences with large literal integers.** Sequences like `0000000000000000000000001` produce degenerate token sequences that confuse positional encodings.
- **HTML/boilerplate leaking through filtering.** A batch with 30 % `<div class="...">` repeating patterns is effectively adversarial input.

```python
# Tool: detect anomalous batches before they enter the training loop.
# Run as a filter in the data loader or offline as a data audit.

import torch
from collections import Counter

def batch_anomaly_score(
    input_ids: torch.Tensor,  # (B, T)
    ngram_n: int = 4,
    repetition_threshold: float = 0.4,
) -> torch.Tensor:
    """
    Returns a per-example anomaly score in [0, 1].
    High score = potentially bad batch element.
    
    Two signals:
      1. Repetition: fraction of n-grams that are duplicates.
      2. Entropy: low token-level entropy suggests degenerate text.
    """
    B, T = input_ids.shape
    scores = torch.zeros(B)

    for b in range(B):
        tokens = input_ids[b].tolist()

        # Signal 1: n-gram repetition fraction
        ngrams = [tuple(tokens[i:i+ngram_n]) for i in range(T - ngram_n)]
        if ngrams:
            counts = Counter(ngrams)
            # fraction of positions that are a repeated n-gram
            repeated = sum(v - 1 for v in counts.values() if v > 1)
            rep_frac = repeated / len(ngrams)
        else:
            rep_frac = 0.0

        # Signal 2: unigram entropy (in bits)
        tok_counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum(
            (c / total) * (torch.log2(torch.tensor(c / total)).item())
            for c in tok_counts.values()
        )
        # For a typical English document, entropy > 8 bits; < 3 is suspicious.
        entropy_score = max(0.0, 1.0 - entropy / 8.0)

        scores[b] = 0.5 * rep_frac + 0.5 * entropy_score

    return scores


def should_skip_batch(input_ids: torch.Tensor, threshold: float = 0.35) -> bool:
    """Return True if the batch contains too many anomalous examples."""
    scores = batch_anomaly_score(input_ids)
    # Skip if average score is high OR if any single example is very bad
    return bool(scores.mean() > threshold or scores.max() > 0.75)
```

### Learning rate issues

An LR too high produces large gradients on every batch, not just anomalous ones. The tell-tale sign: the loss is healthy during warmup but explodes as soon as the LR reaches its peak value. A too-short warmup (reaching full LR before the Adam moments have stabilized) has the same symptom. See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html) for schedule mechanics.

**Rule of thumb for initial LR selection:** For AdamW with global batch size $B$ tokens and a cosine schedule, a reasonable starting LR scales as $\alpha \approx C / \sqrt{d_\text{model}}$ where $C$ is typically around $3 \times 10^{-3}$ to $6 \times 10^{-3}$. This is loosely justified by the maximal update parametrization (μP) analysis of Yang et al. — the key insight being that you want the *feature learning* scale to be approximately 1 regardless of model width.

### Floating-point issues

At bf16, the representable range is roughly $\pm 3.4 \times 10^{38}$ (same exponent bits as fp32), but precision is limited to ~3 decimal digits of mantissa. Overflow is rare but not impossible; underflow to zero is more common and more insidious.

At fp16, overflow occurs above $65\,504$, and activations can silently become inf or NaN during the forward pass if any intermediate value — typically in the attention softmax or MLP feedforward — exceeds this. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) for the full picture.

**The attention logit overflow problem.** Without QK normalization or attention bias clipping, the logits $QK^\top / \sqrt{d_k}$ can grow arbitrarily large. For a model with $d_k = 128$, the scale factor is $1/\sqrt{128} \approx 0.088$. If query and key vectors both have L2 norm $\approx 100$ (feasible in a large model at late training), the maximum logit magnitude is $\approx 100 \times 100 \times 0.088 = 880$, which is within fp16 range but causes the softmax distribution to collapse to a single-token delta. The exponential sum underflows, producing NaN gradients.

### Embedding table instabilities

The input embedding matrix is updated on every step via the language model head (tied weights) but each *row* is only updated when the corresponding token appears. Rare tokens accumulate large Adam second moments slowly, so their effective LR can be anomalously high when they do appear. The output embedding rows also receive gradients from every token in the vocabulary through the CE loss, but the magnitude varies wildly with token frequency.

---

## Architectural Mitigations

Modern LLM architectures have baked in a portfolio of stability techniques. Understanding each one individually helps you reason about which to apply when you start from scratch or debug an existing run.

{{fig:spike-causal-chain-mitigations}}

### QK-Norm

Instead of relying on weight initialization to keep query and key norms bounded, QK-Norm explicitly normalizes the query and key projections before computing attention logits:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{\operatorname{RMSNorm}(Q) \cdot \operatorname{RMSNorm}(K)^\top}{\sqrt{d_k}}\right) V
$$

This ensures the logit magnitudes grow at most as $O(\sqrt{d_k})$ regardless of the raw activations. Introduced in normalized attention variants and used in production by Gemma, Qwen2.5, and others, it is now a standard recommendation for fp16 training at large scale.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QKNormAttention(nn.Module):
    """
    Multi-head attention with per-head QK normalization.
    Prevents attention logit overflow, a common source of training spikes.
    """
    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learnable scale parameters, one per head dimension.
        # RMSNorm without learned scale would fix norm to 1.0;
        # a learnable scale restores representational flexibility.
        self.q_norm = nn.RMSNorm(self.d_k, eps=eps)
        self.k_norm = nn.RMSNorm(self.d_k, eps=eps)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V and reshape to (B, n_heads, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))  # (B, H, T, d_k)
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        # --- QK Norm: the key stability ingredient ---
        # Normalize along the head dimension; norms are now bounded.
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Scaled dot-product attention (safe now that Q, K are normalized)
        scale = self.d_k ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale  # (B, H, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)
```

{{fig:attention-logit-overflow-qknorm}}

### Z-loss

The z-loss is a small auxiliary penalty on the log-partition function of the softmax:

$$
\mathcal{L}_z = \frac{\beta_z}{|B|} \sum_{i \in B} \bigl(\log \sum_v e^{z_{i,v}}\big)^2
$$

where $z_{i,v}$ are the pre-softmax logits for position $i$ and vocabulary token $v$. If the logits grow large, the log-partition function grows, and the z-loss penalizes this. This is especially useful for Mixture-of-Experts models (see [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html)) where the router softmax is a common source of collapse. A typical $\beta_z = 10^{-4}$ adds negligible loss overhead but provides a gradient pressure that keeps logit norms bounded.

```python
def z_loss(logits: torch.Tensor, beta: float = 1e-4) -> torch.Tensor:
    """
    Z-loss regularizer (Zoph et al., ST-MoE 2022).
    logits: (B, T, V) or (B, V) pre-softmax values.
    
    Penalizes large log-partition values to prevent logit explosion.
    Typical beta: 1e-4 for MoE router; 1e-5 for LM head.
    """
    # log(sum_v exp(z_v)) = log-partition function per position
    # torch.logsumexp is numerically stable
    log_z = torch.logsumexp(logits, dim=-1)  # (B, T) or (B,)
    return beta * (log_z ** 2).mean()

# Usage in training loop:
# lm_loss = cross_entropy_loss(logits, targets)
# aux_loss = z_loss(logits, beta=1e-5)
# total_loss = lm_loss + aux_loss
```

### Careful initialization

The variance of activations through the network is set by initialization. The classical analysis by He et al. and the subsequent improvements (μP, Transformers specific scaling) give concrete recipes:

- **Embedding matrix:** Initialize with $\mathcal{N}(0, \sigma^2)$ where $\sigma = d_\text{model}^{-0.5}$ (this keeps embedding norms $\approx 1$ immediately, avoiding early logit explosions).
- **Residual projections (output of attention and MLP):** Scale down by $1/\sqrt{2L}$ where $L$ is the number of layers. This is the "depth scaling" trick — each layer contributes $1/\sqrt{2L}$ to the residual stream, keeping the cumulative norm growth bounded.
- **QKV projections:** Initialize so that the expected logit variance is $\approx 1$. With head dimension $d_k$, this means $\sigma_{QK} = d_k^{-0.25}$ (so that $QK^\top / \sqrt{d_k}$ has variance 1).

```python
import math

def init_transformer_weights(model: nn.Module, n_layers: int, d_model: int):
    """
    Stability-oriented weight initialization for a GPT-style transformer.
    Based on the GPT-2/NanoGPT pattern with depth scaling.
    """
    for name, param in model.named_parameters():
        if param.dim() < 2:
            # Biases, norms — leave as default (zeros / ones)
            continue

        if 'embedding' in name:
            # Embedding table: small init to keep logits bounded at step 0
            nn.init.normal_(param, mean=0.0, std=d_model ** -0.5)

        elif 'c_proj' in name or 'out_proj' in name:
            # Residual-path output projections (attn output + MLP output).
            # Scaled down by 1/sqrt(2 * n_layers) so that the residual stream
            # norm grows as O(1) rather than O(sqrt(L)) at initialization.
            std = (2 * n_layers) ** -0.5
            nn.init.normal_(param, mean=0.0, std=std)

        elif 'q_proj' in name or 'k_proj' in name:
            # Query/Key projections: initialize so logit std ≈ 1
            d_k = param.shape[0]  # assuming (d_k, d_model) layout
            nn.init.normal_(param, mean=0.0, std=d_k ** -0.25)

        else:
            # Default: Kaiming/He for everything else
            nn.init.normal_(param, mean=0.0, std=0.02)
```

### Embedding norm clipping

Even with small init, embeddings can grow unbounded during training. A simple but effective technique is to project embeddings back to a unit ball (or ball of radius $r$) after each optimizer step:

```python
@torch.no_grad()
def clip_embedding_norm(embedding: nn.Embedding, max_norm: float = 1.0):
    """
    After optimizer step: clip embedding row norms to max_norm.
    Prevents rare-token embedding rows from drifting far from the manifold.
    """
    norms = embedding.weight.norm(dim=-1, keepdim=True)  # (V, 1)
    # Only clip rows that exceed max_norm; leave smaller ones untouched.
    clipped = embedding.weight * (max_norm / norms.clamp(min=max_norm))
    embedding.weight.copy_(clipped)
```

---

## Training-Time Mitigations

### Gradient norm clipping

Gradient clipping is the first line of defense against spike propagation. The global gradient norm is:

$$
\|g\|_2 = \sqrt{\sum_i g_i^2}
$$

and we rescale all gradients by $\min(1, \tau / \|g\|_2)$ where $\tau$ is the clip threshold. This prevents large gradients from causing large parameter updates but preserves their *direction*. The standard value is $\tau = 1.0$; some works use $\tau = 0.5$ for extra stability at the cost of slightly slower early learning.

!!! warning "Gradient clipping with distributed training"
    With data parallelism (DDP, ZeRO), each rank holds a shard of the gradients. You must compute the *global* gradient norm across all ranks before clipping — otherwise each rank clips to its local norm, which may differ wildly. PyTorch's `torch.nn.utils.clip_grad_norm_` handles this automatically if called after `loss.backward()` and before `optimizer.step()`, but only if gradients are synchronized. With ZeRO-3, gradients are sharded, so you need DeepSpeed's `clip_grad_norm_` or Fully Sharded Data Parallel (FSDP)'s equivalent.

### Skip-batch logic

When a batch produces an anomalous gradient — detected either by norm threshold or by explicit batch quality scoring — you can skip the optimizer step for that batch. This is conservative: you still do the forward and backward pass (wasting compute), but you do not update the model state. The Adam moments are also not updated.

```python
def training_step(
    model,
    optimizer,
    batch: dict,
    scaler,  # GradScaler for mixed precision
    grad_clip: float = 1.0,
    grad_skip_threshold: float = 5.0,  # skip if norm > 5x clip threshold
    anomaly_threshold: float = 0.35,
) -> dict:
    """
    One training step with skip-batch and gradient norm monitoring.
    Returns a metrics dict for logging.
    """
    # --- Optional: skip anomalous batch early (before forward pass) ---
    if should_skip_batch(batch['input_ids'], threshold=anomaly_threshold):
        return {'loss': float('nan'), 'skipped': True, 'reason': 'bad_data'}

    optimizer.zero_grad()

    # Forward pass under autocast
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(batch['input_ids'])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch['labels'].view(-1),
            ignore_index=-100,
        )

    # Backward pass
    scaler.scale(loss).backward()

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)

    # Compute global gradient norm BEFORE clipping (for monitoring)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=grad_clip,
    ).item()

    # --- Skip-step logic: if norm is astronomically large, skip optimizer ---
    if grad_norm > grad_skip_threshold * grad_clip:
        # Reset gradients so they don't corrupt the next step
        optimizer.zero_grad()
        scaler.update()
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'skipped': True,
            'reason': 'grad_spike',
        }

    # Normal step
    scaler.step(optimizer)
    scaler.update()

    return {
        'loss': loss.item(),
        'grad_norm': grad_norm,
        'skipped': False,
    }
```

### Loss spike recovery: rolling back vs. continuing

When a spike is detected (loss increases by more than, say, 0.2 nats over 20 steps), you have three options:

1. **Continue and hope.** Works for mild, transient spikes. The Adam moments adapt within 100–300 steps. Monitor closely.
2. **Resume from the last good checkpoint.** If the spike is severe or you can identify a batch to blame, roll back to the checkpoint before the bad batch and skip it. See [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html) for checkpoint strategy.
3. **Resume with lower LR.** Load the last good checkpoint and continue with the LR reduced by 20–40 %. This treats the spike as a symptom of the LR being at its peak (common mid-run), and the reduced LR prevents recurrence while the moments stabilize.

```text
Spike decision tree:

spike detected (loss > baseline + 0.15 nats for > 50 steps)
     |
     +-- grad_norm normal (< 2x clip)?
     |        YES: likely LR/optimizer issue → reduce LR by 20%, continue
     |        NO:  large grad norm → check batch data
     |
     +-- specific batch identifiable (from data logs)?
     |        YES: roll back to pre-spike checkpoint, skip that batch
     |        NO:  roll back to pre-spike checkpoint, reduce LR by 30%
     |
     +-- spike severity?
              MILD (< 0.3 nats, < 200 steps): continue + monitor
              SEVERE (> 0.5 nats, model not recovering): roll back
```

---

## Monitoring Infrastructure

You cannot debug what you cannot see. A production pretraining run should log at minimum the following signals every N steps (typically every 5–50 steps, depending on cluster size and logging overhead).

### Essential metrics

```python
import torch
import wandb  # or any logging framework

class TrainingMonitor:
    """
    Collects and logs training health signals.
    Designed to add minimal overhead: most metrics are computed from
    tensors already in memory.
    """
    def __init__(self, log_every: int = 10, spike_window: int = 50):
        self.log_every = log_every
        self.spike_window = spike_window
        self.loss_history = []
        self.step = 0

    def log_step(
        self,
        loss: float,
        grad_norm: float,
        model: torch.nn.Module,
        lr: float,
    ):
        self.step += 1
        self.loss_history.append(loss)

        if self.step % self.log_every != 0:
            return

        metrics = {
            'train/loss': loss,
            'train/perplexity': math.exp(min(loss, 20)),  # clamp to avoid overflow
            'train/grad_norm': grad_norm,
            'train/lr': lr,
            'train/step': self.step,
        }

        # --- Activation statistics (sampled from a few layers) ---
        # Hook-based; only active when we log.
        act_stats = self._sample_activation_stats(model)
        metrics.update(act_stats)

        # --- Spike detector ---
        if len(self.loss_history) >= self.spike_window:
            window = self.loss_history[-self.spike_window:]
            baseline = sum(window[:self.spike_window // 2]) / (self.spike_window // 2)
            recent = sum(window[self.spike_window // 2:]) / (self.spike_window // 2)
            metrics['train/spike_delta'] = recent - baseline

        wandb.log(metrics)

    @torch.no_grad()
    def _sample_activation_stats(self, model: torch.nn.Module) -> dict:
        """
        Compute per-layer activation norms for the most recent forward pass.
        In practice, wire this to forward hooks registered on a few layers.
        Here we illustrate with a direct parameter-norm proxy.
        """
        stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Track weight matrix spectral norm proxy (Frobenius / sqrt(numel))
                rms = param.norm() / (param.numel() ** 0.5)
                short = name.replace('.weight', '').replace('model.', '')
                stats[f'weights/{short}_rms'] = rms.item()
        return stats
```

### What to watch and why

| Metric | What it tells you | Action threshold |
|---|---|---|
| `grad_norm` | Health of each step; spike precursor | > 5×clip → investigate |
| `loss` smoothed over 100 steps | Run progress vs. scaling law prediction | Flat for > 500 steps → reduce LR or check data |
| `loss_spike_delta` (window) | Spike severity in progress | > 0.1 nats → alert |
| Embedding row norm max | Embedding divergence | > 10 → apply clip |
| Attention logit max (sampled) | fp16 overflow risk | > 50 000 in fp16 → add QK norm |
| MLP pre-activation RMS | Activation explosion | > 100 → add norm before MLP |
| LM head output logit std | Logit scale | > 30 → add z-loss |
| `skipped_batches` fraction | Data quality / grad instability | > 2% → audit dataset |

### Distributed monitoring considerations

In a multi-node run across hundreds of GPUs (see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)), you should:

- Log metrics only from rank 0 to avoid redundant writes.
- But compute gradient norms globally (all-reduce) before logging, since rank-0's local norm may not represent the global norm.
- Alert on NaN/Inf using `torch.isnan(loss).any()` — in ZeRO-3, a NaN on any rank will propagate during the gradient all-reduce, so catching it early saves wasted compute.

```python
def check_for_nan_distributed(loss: torch.Tensor, grad_norm: float) -> bool:
    """
    Check for NaN/Inf on all ranks and broadcast the result.
    Returns True if any rank has a problem.
    """
    import torch.distributed as dist

    # Flag: 1.0 if problematic, 0.0 if OK
    bad = torch.tensor(
        1.0 if (not torch.isfinite(loss) or not math.isfinite(grad_norm)) else 0.0,
        device=loss.device,
    )
    # Max-reduce: any rank with a problem sets the flag for all
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    return bad.item() > 0.5
```

---

## The Lived Experience: War Stories & Common Failure Modes

This section distills patterns from real large-scale training runs. No precise benchmark numbers are attributed, but the patterns are real and well-documented in public technical reports from teams including those behind Llama, PaLM, Gemini, and similar.

### Story 1: The silent data corruption

A team ran a 70B model to 500B tokens with no obvious spikes, but evaluation metrics on a standard benchmark plateau early and never recover. Post-mortem: a deduplication bug in the data pipeline had removed 40 % of the math and code content, replacing it with duplicate web text. The loss curve was *smooth* because the model simply memorized the duplicates efficiently. Lesson: **loss alone is not a sufficient health signal.** Hold out a small fixed eval set covering each domain (code, math, multilingual, factual QA) and report it every 10B tokens.

### Story 2: The creeping LR spike

Every modern LLM training run has seen this: things are fine for weeks, then at precisely the step where the warmup ends and the cosine peak is reached, loss jumps 0.4 nats and the grad norm hits 20×. The model recovers, but with a shifted loss baseline. Root cause: the Adam moments were well-calibrated for the warmup LR, not the peak LR. The effective LR at the peak is $\alpha_\text{peak} / \sqrt{\hat{v}}$, and $\hat{v}$ was too small. Fix: use a longer warmup (1–2 % of total steps rather than 0.1 %) or apply a sqrt-scaled warmup that grows $\alpha$ slower than Adam's $\sqrt{t}$ moment term.

### Story 3: The fp16 attention NaN cascade

In a 13B model trained in fp16 (not bf16), at step 42 000 the loss becomes NaN. Debugging with activation hooks reveals that the attention logits for the last layer's head 7 are producing values near 60 000 before the softmax — approaching the fp16 max of 65 504. A batch with a 32 000-token nearly-identical sequence (a repeated copyright boilerplate) pushed query and key norms to an extreme. The softmax then produces Inf, the backward pass propagates NaN, and all parameters are corrupted. Fix: add QK-Norm (see above) and switch to bf16, which has a much larger dynamic range. The retrospective also added a maximum logit monitor to the activation hooks.

### Story 4: The zombie GPU

During a 256-GPU training run, GPU #183 silently corrupts its computation starting at step 15 000 (the gradient it contributes is numerically wrong but finite). Because the corrupt gradient is averaged in during the DDP all-reduce, the model trains fine for another 20 000 steps — just slightly worse than it should. Discovered only when the team compared two runs that should have been identical under different partitioning and found a large discrepancy. Fix: implement periodic determinism checks (run a single fixed micro-batch through each GPU independently and compare; see also chapter [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html)).

---

## Debugging Playbook: A Step-by-Step Checklist

When something goes wrong, follow this ordered checklist. Each step narrows the hypothesis space.

```text
===== TRAINING STABILITY DEBUGGING CHECKLIST =====

STEP 1: ESTABLISH GROUND TRUTH
  [ ] Is the loss spike visible on ALL tracked metrics (val loss, ppl)?
  [ ] Is the spike visible from ALL ranks (or just one shard)?
  [ ] Is there a NaN or Inf anywhere in the loss or grad norm logs?
  [ ] What is the grad_norm at the spike step vs. baseline?

STEP 2: NARROW TO CATEGORY
  [ ] Grad norm normal, loss spiked → data issue (anomalous batch)
  [ ] Grad norm large (>5x), loss spiked → optimizer amplification
  [ ] Grad norm NaN → activation overflow (fp issue)
  [ ] Grad norm zero or near-zero → vanishing gradient or bad init
  [ ] Spike correlates with LR peak → warmup length issue

STEP 3: ISOLATE THE STEP
  [ ] Roll back to the checkpoint BEFORE the spike
  [ ] Replay forward+backward for the exact batch at the spike step
  [ ] Log per-layer gradient norms (which layer blows up first?)
  [ ] Log per-head attention logit max (which head has overflow?)

STEP 4: IDENTIFY ROOT CAUSE
  [ ] Inspect the batch: run batch_anomaly_score, look at raw tokens
  [ ] Check the data pipeline: was this batch from a specific shard/source?
  [ ] Check fp overflow: run the forward pass in fp32 and compare
  [ ] Check LR: what was the LR at the spike step?

STEP 5: MITIGATE AND RESUME
  [ ] Implement the fix (QK-norm, skip-batch, lower LR, fix data)
  [ ] Resume from pre-spike checkpoint
  [ ] Monitor for 500 steps before lowering alert thresholds
  [ ] Document the incident (what failed, why, what was fixed)

STEP 6: PREVENT RECURRENCE
  [ ] Add the root cause to the pre-training data quality audit
  [ ] Add the relevant monitoring signal permanently
  [ ] Consider adding skip-batch logic if not already present
  [ ] Consider adding QK-norm / z-loss if not already present
```

{{fig:spike-diagnostic-decision-tree}}

Here is a minimal but complete diagnostic script to run against a checkpoint and a batch:

```python
"""
stability_probe.py — Diagnose a training spike post-hoc.

Usage:
  python stability_probe.py \
    --ckpt /path/to/checkpoint_before_spike.pt \
    --batch /path/to/spike_batch.pt \
    --model_config /path/to/config.json

Outputs: per-layer gradient norms, attention logit statistics,
         and a verdict on likely root cause.
"""
import torch
import json
import argparse
import math
from typing import Dict

def load_model_and_batch(ckpt_path: str, batch_path: str, config_path: str):
    """Load model from checkpoint and batch from saved tensor file."""
    # In practice: instantiate your model class, load state dict.
    # Here we use a placeholder to show the diagnostic logic.
    raise NotImplementedError("Wire to your model class")

@torch.no_grad()
def probe_activation_norms(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float32,  # Always probe in fp32 for accuracy
) -> Dict[str, float]:
    """
    Register forward hooks to capture activation norms at each transformer block.
    Returns a dict: layer_name → max_activation_norm.
    """
    act_norms = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            act_norms[name] = out.abs().max().item()
        return hook

    for name, module in model.named_modules():
        if 'attn' in name or 'mlp' in name:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.to(dtype=dtype)
    with torch.autocast(device_type='cuda', enabled=False):
        _ = model(**batch)

    for h in hooks:
        h.remove()

    return act_norms


def compute_per_layer_grad_norms(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Run a backward pass and report per-parameter gradient norms."""
    model.train()
    logits = model(**batch)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms


def diagnose(act_norms: Dict[str, float], grad_norms: Dict[str, float]) -> str:
    """Heuristic verdict based on activation and gradient diagnostics."""
    max_act = max(act_norms.values()) if act_norms else 0
    max_grad = max(grad_norms.values()) if grad_norms else 0

    if max_act > 60000:
        return "VERDICT: fp16 overflow likely (max_act={:.0f}). Add QK-Norm or switch to bf16.".format(max_act)
    elif max_act > 1000:
        return "VERDICT: activation explosion (max_act={:.0f}). Check init & norms.".format(max_act)
    elif max_grad > 50:
        layer = max(grad_norms, key=grad_norms.get)
        return "VERDICT: gradient explosion at '{}' (norm={:.1f}). Check data batch or reduce LR.".format(layer, max_grad)
    elif max_grad < 1e-6:
        layer = max(grad_norms, key=grad_norms.get)
        return "VERDICT: vanishing gradient (max_grad={:.2e}). Check depth or init.".format(max_grad)
    else:
        return "VERDICT: no clear activation/gradient anomaly. Check data content and LR schedule."
```

!!! interview "Interview Corner"
    **Q:** Your 100B model training run experiences a sudden loss spike of 0.4 nats at step 80 000. The gradient norm at that step is 18× the normal baseline. After 200 steps the loss mostly recovers, but remains 0.05 nats above where it was before. How would you diagnose and mitigate this, and what does the partial recovery tell you?

    **A:** The large gradient norm points to an optimizer amplification event rather than a model architecture issue. The most likely causes in order of probability are: (1) an anomalous data batch — possibly a corrupted or repetitive sequence that produces a very high cross-entropy loss; (2) the run reaching its peak learning rate before Adam's second moments have fully converged (especially if this is near the end of warmup).

    Diagnosis: roll back to the checkpoint before step 80 000, replay the exact batch at that step in full fp32 with per-layer gradient norms logged, and inspect the batch content with a token-level quality check.

    Mitigation: if a specific bad batch is identified, skip it and resume; add skip-batch logic for future batches. If the cause is LR, reduce the peak LR by 20 % and extend warmup. Also add z-loss and QK-Norm if not already present.

    The partial recovery (loss 0.05 nats above baseline) tells us the large update moved the model into a slightly worse local minimum — it did not fully return to its previous trajectory. This is a common outcome for moderate spikes: the Adam moments absorbed the new gradient scale but the weight values settled in a slightly different basin. With a longer run (more tokens to go), the gap typically closes. If this were a severe spike (> 1 nat, no recovery), a full checkpoint roll-back would be required.

---

## Pre-Run Stability Hardening Checklist

Before starting a large, expensive pretraining run, validate every item on this list. Discovering a bug at step 5 000 of a 500B-token run is far cheaper than discovering it at step 500 000.

```text
===== PRE-RUN HARDENING CHECKLIST =====

ARCHITECTURE
  [ ] QK-Norm applied to all attention layers
  [ ] Residual output projections initialized with 1/sqrt(2L) scaling
  [ ] Embedding init std ≤ 1/sqrt(d_model)
  [ ] Z-loss enabled (beta ≈ 1e-5 for LM head, 1e-4 for MoE router)
  [ ] Pre-norm (RMSNorm before attention/MLP) rather than post-norm

OPTIMIZER
  [ ] Gradient clipping at 1.0 (or 0.5 for extra stability)
  [ ] Adam epsilon = 1e-8 (not 1e-6; smaller epsilon = more stability)
  [ ] Warmup ≥ 1% of total steps (for 1T-token run: ≥ 1B tokens)
  [ ] Weight decay ≠ 0 (0.1 is standard; reduces weight growth)
  [ ] No weight decay on embeddings / normalization parameters

PRECISION
  [ ] Using bf16 (not fp16) for forward/backward passes
  [ ] Loss scaling disabled when using bf16 (not needed)
  [ ] Master weights in fp32 (bf16 master weights can accumulate error)

DATA PIPELINE
  [ ] Batch anomaly detection enabled (or offline data audit complete)
  [ ] Sequence length distribution checked (no extreme outliers)
  [ ] Dataset mixture ratios validated against intent
  [ ] Fixed held-out eval batches per domain (code, math, web, multilingual)

MONITORING
  [ ] Grad norm logged every N steps (N ≤ 50)
  [ ] Loss spike detector with alerting (threshold: > 0.1 nat window delta)
  [ ] NaN/Inf detection with distributed reduce
  [ ] Checkpoint every K steps with verified restore test
  [ ] Compute and log MFU (Model FLOP Utilization) — sudden drops signal hardware issues
```

---

## Key Takeaways

!!! key "Key Takeaways"
    - Loss spikes are caused by large gradients (from bad data, high LR, or floating-point issues) amplified by Adam's first-moment-to-second-moment ratio. Understanding this mechanism guides every mitigation.
    - **QK-Norm** and **z-loss** are inexpensive architectural additions that prevent the two most common sources of catastrophic instability: attention logit overflow and logit explosion. Add them by default.
    - **Gradient clipping** (norm threshold 1.0) and **skip-batch logic** (skip optimizer steps when gradient norm exceeds 5×clip) form the first line of operational defense during training.
    - **Careful initialization** — depth-scaled residual projections, small embedding init, well-tuned QK init — keeps the model in a stable regime from step 0, reducing the number of spikes encountered in the first few thousand steps.
    - **Monitoring is not optional.** Track gradient norms, activation statistics, per-domain eval loss, and an explicit spike delta at every 10–50 steps. You cannot recover from what you cannot see.
    - When a spike occurs, the debugging protocol is: establish ground truth → narrow to category → isolate to the step → identify root cause → mitigate and document. Never skip steps.
    - The **partial recovery** pattern (loss returns to near-baseline but not exactly) indicates the optimizer settled in a slightly worse basin — usually acceptable for long runs but worth monitoring. A full non-recovery is the signal to roll back.
    - Data quality issues (repetition, encoding bugs, deduplication failures) are the silent killers: they may not produce visible loss spikes but consistently degrade model quality. Run per-domain eval metrics, not just aggregate loss.
    - The engineering discipline of **pre-run hardening** — checking every architectural, optimizer, precision, data, and monitoring setting before starting — saves orders of magnitude more compute than it costs.

---

!!! sota "State of the Art & Resources (2026)"
    Training stability for large-scale LLMs is now a well-mapped engineering discipline: the core failure modes (Adam logit amplification, fp16 overflow, bad-data spikes) are understood analytically, and a standard toolkit of QK-norm, z-loss, gradient clipping, and skip-batch logic has been validated at scales from 7B to 400B+ parameters. Active research focuses on optimizer-level spike detection, adaptive clipping, and principled hyperparameter transfer across scales.

    **Foundational work**

    - [Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (2022)](https://arxiv.org/abs/2202.08906) — Introduces z-loss regularization and provides the canonical analysis of MoE training instability; z-loss is now standard in dense and MoE runs alike.
    - [Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters* (2023)](https://arxiv.org/abs/2302.05442) — Popularises QK-norm as the key technique for preventing attention-logit overflow at scale.
    - [Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (2022)](https://arxiv.org/abs/2203.03466) — The μP framework that justifies $\alpha \propto 1/\sqrt{d}$ LR scaling and provides a principled basis for stable init across model widths.

    **Recent advances (2023–2026)**

    - [Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (2023)](https://arxiv.org/abs/2309.14322) — Shows that attention-logit and output-logit instabilities can be reproduced cheaply at small scale, enabling systematic study of mitigations before committing to large runs.
    - [Molybog et al., *A Theory on Adam Instability in Large-Scale Machine Learning* (2023)](https://arxiv.org/abs/2304.09871) — Analytical theory showing how Adam enters a regime where updates are large and uncorrelated with the loss gradient; validated on 7B–546B models.
    - [Rybakov et al., *Methods of Improving LLM Training Stability* (2024)](https://arxiv.org/abs/2410.16682) — Systematic study of per-layer norm placement (QK, Proj, FC2); shows combined QK-norm + softmax capping allows 1.5× higher LR without divergence.
    - [Huang et al., *SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training* (2025)](https://arxiv.org/abs/2501.06842) — Optimizer extension that detects gradient spikes and resets momentum at the spike step, eliminating manual rollback for many spike events; ICLR 2025.
    - [OLMo Team, *2 OLMo 2 Furious* (2025)](https://arxiv.org/abs/2501.00656) — Fully open pretraining run detailing practical stability choices (RMSNorm reordering, QK-norm, z-loss, data filtering) at 7B–32B scale with all training artifacts released.
    - [Grattafiori et al., *The Llama 3 Herd of Models* (2024)](https://arxiv.org/abs/2407.21783) — Meta's candid technical report on instabilities encountered during Llama 3 pretraining and the mitigations applied in production.

    **Open-source & tools**

    - [microsoft/mup](https://github.com/microsoft/mup) — PyTorch implementation of maximal update parametrization (μP) for stable, scale-transferable LR and init.
    - [allenai/OLMo](https://github.com/allenai/OLMo) — Fully open pretraining codebase with QK-norm, z-loss, and monitoring baked in; the most transparent reference implementation of production stability practices.

## Further Reading

- **Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (2022)** — Introduces z-loss and provides a thorough analysis of instability in MoE training, with ablations on every stability technique discussed in this chapter.
- **Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities" (2023)** — Systematically studies which instabilities are predictable at small scale, providing a framework for the pre-run hardening approach.
- **Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2022)** — The μP framework that underpins principled LR and init scaling, explaining why $\alpha \propto 1/\sqrt{d}$ keeps training stable across widths.
- **Grattafiori et al., "The Llama 3 Herd of Models" (Meta AI, 2024)** — The technical report contains candid discussion of training instabilities encountered during Llama 3 pretraining and the mitigations applied.
- **Anil et al., "PaLM 2 Technical Report" (Google, 2023)** — Documents the training stability experience for a series of large models including the use of bf16, careful init, and monitoring infrastructure.
- **Molybog et al., "A Theory of Loss Landscape and Training Stability" (2023)** — Theoretical treatment connecting loss landscape curvature to spike behavior, providing analytical backing for the Adam amplification model described in this chapter.
- **nanoGPT (Andrej Karpathy, GitHub)** — The canonical minimal GPT implementation. The `train.py` file is a useful starting point for understanding gradient clipping, skip-batch, and monitoring in a single-file, readable codebase.
