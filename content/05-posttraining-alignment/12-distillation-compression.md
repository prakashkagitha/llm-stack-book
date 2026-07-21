# 5.12 Distillation, Model Compression & Knowledge Transfer

Training a 70-billion-parameter model costs millions of dollars and requires a cluster of GPUs. Deploying it on a developer laptop or inside a latency-sensitive API costs even more. The gap between what you can afford to train and what you can afford to serve motivates an entire family of techniques grouped under the banner of *model compression*. This chapter covers the three most important ones — knowledge distillation, weight pruning, and their synergy with speculative decoding — and shows you how to implement them from scratch.

The key insight that ties everything together is this: a large, expensive model contains far more "knowledge" than its parameter count strictly requires. Redundant neurons, near-zero weights, and over-parameterized attention heads all suggest that a smaller model, if trained carefully, can approximate the same function at a fraction of the cost. Our job is to extract that knowledge efficiently.

## 5.1 Knowledge Distillation: Soft Targets and Temperature

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean introduced *knowledge distillation* (KD) in a landmark 2015 paper. The idea is deceptively simple: instead of training a student model on one-hot (hard) labels, train it to match the full output distribution (soft targets) of a larger *teacher* model.

### Why Soft Targets Contain More Information

Consider a cat-vs-dog classifier. A hard label says `cat = 1, dog = 0`. But a well-trained teacher might output `cat = 0.92, dog = 0.06, tiger = 0.02`. That distribution says: this cat looks vaguely tiger-like and not at all dog-like. This inter-class similarity is information the student can use to generalize better from less data.

The same principle applies to language models. When a teacher assigns probability 0.3 to "Paris", 0.2 to "London", and 0.15 to "Berlin" as the next token after "The capital of France is", the student learns a richer, more structured representation than it would from the hard label "Paris" alone.

{{fig:distill-soft-targets-temperature}}

### The KD Loss

For a language model the per-token distillation loss is the Kullback-Leibler divergence between the teacher's soft distribution $q$ and the student's distribution $p$:

$$
\mathcal{L}_\text{KD} = \tau^2 \sum_{v} q_\tau(v) \log \frac{q_\tau(v)}{p_\tau(v)}
$$

where the temperature-scaled distributions are:

$$
q_\tau(v) = \frac{\exp(z^T_v / \tau)}{\sum_{v'} \exp(z^T_{v'} / \tau)}, \quad p_\tau(v) = \frac{\exp(z^S_v / \tau)}{\sum_{v'} \exp(z^S_{v'} / \tau)}
$$

Here $z^T$ and $z^S$ are the teacher's and student's logit vectors over the vocabulary, and $\tau > 1$ is the *temperature*. The $\tau^2$ prefactor ensures that the gradient magnitude stays constant as $\tau$ changes — without it, increasing $\tau$ softens the distribution but also shrinks the gradient by $1/\tau^2$, making training effectively slower.

The full training loss blends distillation with standard cross-entropy on ground-truth labels:

$$
\mathcal{L} = \alpha \, \mathcal{L}_\text{CE}(p_1, y) + (1 - \alpha) \, \mathcal{L}_\text{KD}(p_\tau, q_\tau)
$$

Common choices are $\tau \in [2, 5]$ and $\alpha \in [0.1, 0.5]$.

!!! example "Worked Example: Temperature Effect on Soft Targets"

    Suppose a teacher has logits $z^T = [3.0, 1.0, 0.5]$ for three tokens.

    At $\tau = 1$: softmax gives $[0.825, 0.112, 0.068]$ — nearly all mass on token 0.

    At $\tau = 4$: logits become $[0.75, 0.25, 0.125]$, softmax gives $[0.388, 0.317, 0.295]$ — much softer.

    The soft distribution at $\tau = 4$ tells the student that tokens 1 and 2 are plausible alternatives, carrying meaningful signal about inter-token similarity. At $\tau = 1$ this information is almost entirely suppressed. Setting $\tau$ too high (say, 20) eventually flattens the distribution toward uniform, losing the ordering information — this is why values of 2–5 are typical.

## 5.2 On-Policy vs Off-Policy Distillation

The terminology "on-policy" vs "off-policy" in distillation borrows from RL and describes *who generated the context being trained on*.

### Off-Policy Distillation

In the standard formulation, you take a fixed dataset of (context, continuation) pairs, run the teacher forward to get $q_\tau$, then train the student to match those distributions. The student never influences which sequences it is trained on — it is trained purely on the teacher's preferred distribution. This is called *off-policy* distillation (or sometimes *teacher-guided* distillation).

It is cheap and simple: compute teacher logits once, save them to disk, train the student on top. The downside is *distribution mismatch*: the student is trained on sequences the teacher "liked" but at test time the student must generate its own sequences, which may drift into regions where the teacher's signals don't transfer well.

### On-Policy Distillation

In on-policy distillation the student generates text, the teacher scores those generations, and the student is trained on its own outputs. This is a form of imitation learning / behavioral cloning applied at the sequence level.

The simplest on-policy algorithm:
1. Sample a batch of prompts $x \sim \mathcal{D}$.
2. Generate completions $\hat{y} \sim p_S(\cdot|x)$ using the student.
3. For each token position, compute the teacher's distribution $q_\tau$ conditioned on $(x, \hat{y}_{<t})$.
4. Minimize $\mathcal{L}_\text{KD}(p_{S,\tau}, q_\tau)$ over the student's own rollouts.

On-policy distillation is closely related to methods like [RLHF](../05-posttraining-alignment/05-rlhf-reward-modeling.html) and [DPO](../05-posttraining-alignment/07-dpo-and-variants.html) — you are training the student to behave like the teacher on its own generations rather than on a fixed corpus. The statistical consistency argument for on-policy methods is the same one motivating PPO in RL: you want gradients to reflect the distribution the policy actually encounters.

On-policy distillation typically outperforms off-policy on longer generations and tasks that require multi-step reasoning, at the cost of running the student (and evaluating the teacher) online during training.

### Imitation-Gap and Capacity Gap

A practical tension: if the teacher is vastly larger than the student, the student cannot represent the teacher's distribution accurately. Hinton et al. called this the *capacity gap*. Empirically, distilling a 70B teacher into a 1B student often underperforms distilling a 13B teacher into the same 1B student, because the 70B model's distribution is "too complex" for the student to model. Progressive distillation — chaining 70B → 13B → 3B → 1B — often produces better final results.

## 5.3 Sequence-Level Knowledge Distillation

Token-level KD teaches the student to match the teacher at each position. But language generation is a sequential process and errors compound: a student wrong at position $t$ diverges from the teacher at position $t+1$ in a way that per-token loss doesn't penalize.

*Sequence-level KD* (SeqKD), introduced by Kim & Rush (2016), addresses this by distilling at the sequence level rather than the token level.

### SeqKD: Data Augmentation View

The simplest SeqKD recipe:
1. Run the teacher in greedy decoding (or top-k sampling) over a training prompt set.
2. Use the teacher's *output text* as the student's training target (standard cross-entropy, hard labels).
3. Optionally mix original gold labels with teacher-generated pseudo-labels.

The student now learns to imitate the teacher's complete output behavior. A critical nuance: greedy teacher outputs are often different from gold labels, and the student may actually score better on downstream tasks by following the teacher's idioms rather than the gold data.

### Word-Level vs Sequence-Level vs Intermediate Features

{{fig:distill-granularity-tiers}}

**FitNet-style distillation** goes beyond output distributions to match *intermediate representations*: hidden states, attention patterns, or even specific layer outputs. A projector head (typically a linear map) aligns the student's smaller hidden dimension to the teacher's:

$$
\mathcal{L}_\text{feat} = \| W_\text{proj} \, h^S_l - h^T_{l'} \|_2^2
$$

This works well when you want the student's representations to be semantically aligned with the teacher's, which is important for tasks that involve intermediate reasoning steps.

## 5.4 Distilling Reasoning: From Large to Small Reasoning Models

The emergence of chain-of-thought (CoT) and extended reasoning models (see [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html)) created a new distillation problem: how do you transfer *reasoning behavior* (long CoT traces) from a large model to a small one?

### R1-Style Reasoning Distillation

DeepSeek-R1 demonstrated a recipe for creating small reasoning models that generalizes well:

1. **Sample long reasoning traces from the large model.** Run the teacher (e.g., a 671B MoE reasoning model) on math, coding, and logic problems. Collect the full `<think>...</think>` traces.
2. **Filter for correctness and quality.** Keep traces where the final answer is verifiable (e.g., answer matches a reference), and discard traces that are overly repetitive or contain hallucinations.
3. **SFT the student on (prompt, trace, answer) triples.** This is standard supervised fine-tuning (see [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html)) but with the reasoning trace included in the target.
4. **Optionally apply RL with verifiable rewards** to refine the distilled student further, using the same GRPO/RLOO infrastructure described in [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html).

The resulting small models (e.g., on the order of 7–14B parameters) can solve problems that require multi-step reasoning at a level that was previously only achievable with much larger models. The key insight is that reasoning *behaviors* transfer via imitation learning more efficiently than raw capability transfers via weight matching.

### Pitfalls in Reasoning Distillation

- **Trace length mismatch.** A teacher that spends 2,000 tokens reasoning may produce traces that are too long for a student to generate reliably during RL fine-tuning. Filtering for concise-but-correct traces helps.
- **Format overfitting.** Students may learn to produce *the teacher's format* without understanding the underlying reasoning. Augment with diverse prompt phrasings.
- **Capability ceiling.** A 1B student will hit a hard ceiling regardless of trace quality — if the intermediate steps require knowledge the student doesn't have, imitation fails. Scaling to at least 7B is typically recommended for non-trivial reasoning tasks.

## 5.5 Pruning: Structured and Unstructured

Pruning removes weights or entire structures from a trained model. Unlike distillation (which trains a new small model), pruning modifies the existing model. The two main flavors are *unstructured* (individual weights) and *structured* (entire neurons, heads, or layers).

### Unstructured Pruning and the Magnitude Baseline

The simplest approach: set the smallest-magnitude weights to zero. A weight $w$ is pruned if $|w| < \theta$ for some threshold $\theta$ chosen to achieve a target sparsity level $s$ (e.g., 50% of weights are zero).

Unstructured sparsity at 50–70% has minimal accuracy impact on large models but provides limited wall-clock speedup on standard GPUs, because hardware is optimized for dense matrix multiplications. The benefit is mainly in model file size and in specialized sparse-compute hardware.

### SparseGPT: One-Shot Unstructured Pruning

SparseGPT (Frantar & Alistarh, 2023) enables high sparsity in LLMs with a single forward pass, with no gradient computation. It is based on the *Optimal Brain Surgeon* (OBS) framework, which computes the second-order (Hessian-based) reconstruction error after removing a weight and compensates by updating remaining weights.

For each linear layer with weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, SparseGPT processes the columns of $W$ sequentially:

1. Collect activation statistics $H = X^T X / N$ using calibration data (typically 128 samples).
2. For each column $q$: compute the pruning score $\text{score}(w_{ij}) = w_{ij}^2 / [H^{-1}]_{jj}$ (analogous to the OBS weight saliency).
3. Prune the lowest-score weights in that column to zero.
4. Update the remaining weights in the column to compensate: $\delta w = -\frac{w_q}{[H^{-1}]_{qq}} H^{-1}_{:,q}$.
5. Update $H$ using Cholesky rank-1 updates.

SparseGPT achieves 50–60% sparsity on models like LLaMA with near-zero perplexity increase, and can be extended to 2:4 structured sparsity (2 nonzeros per 4 weights) that maps directly to NVIDIA's sparse tensor core format and yields about 1.5–2x throughput improvement.

### Wanda: Pruning Without Hessians

Wanda (Sun et al., 2023) ("Pruning by Weights and Activations") shows that a much simpler pruning criterion often matches SparseGPT quality:

$$
\text{score}(w_{ij}) = |w_{ij}| \cdot \|x_j\|_2
$$

where $\|x_j\|_2$ is the RMS magnitude of the $j$-th input feature computed over calibration data. The score combines weight magnitude (what OBS uses) with activation magnitude (how important that feature actually is at runtime). Wanda requires no Hessian inversion — just one forward pass — making it extremely fast to apply even to 70B models.

### Structured Pruning

Structured pruning removes entire components: attention heads, MLP neurons, or even full transformer layers. This yields speedup on any hardware without needing sparse kernels.

**Attention head pruning.** Michel et al. (2019) showed that many attention heads can be removed with limited performance degradation. A simple sensitivity analysis: mask one head at a time, measure validation loss increase, prune heads with lowest importance.

**Layer dropping.** Some layers in deep transformers are near-identity mappings (the residual stream barely changes). Dropping such layers yields surprisingly small accuracy drops and significant latency reductions.

**Width pruning.** Prune MLP intermediate dimensions or the hidden dimension of attention projections. This requires careful co-pruning of the weight matrices on both sides of a pruned feature.

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

# ─────────────────────────────────────────────────────
# Simple magnitude + activation (Wanda-style) pruning
# for a single linear layer.
# ─────────────────────────────────────────────────────

def compute_activation_norms(
    model: nn.Module,
    layer_name: str,
    calibration_tokens: torch.Tensor,   # shape: (n_samples, seq_len)
    device: str = "cuda",
) -> torch.Tensor:
    """
    Collect per-feature activation L2 norms for one linear layer
    by running a small calibration set through the model.
    Returns a vector of shape (in_features,).
    """
    activation_sq_sum = None
    n_tokens = 0

    def hook_fn(module, inp, out):
        nonlocal activation_sq_sum, n_tokens
        x = inp[0].detach().float()          # (batch, seq, in_features)
        # Flatten batch and sequence dimensions
        x = x.reshape(-1, x.shape[-1])       # (B*T, in_features)
        sq = (x ** 2).sum(dim=0)             # (in_features,)
        if activation_sq_sum is None:
            activation_sq_sum = sq
        else:
            activation_sq_sum += sq
        n_tokens += x.shape[0]

    # Register hook on the target layer
    target = dict(model.named_modules())[layer_name]
    handle = target.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch in calibration_tokens.split(8):        # micro-batches
            model(batch.to(device))

    handle.remove()
    # RMS activation magnitude per input feature
    return (activation_sq_sum / n_tokens).sqrt()         # shape: (in_features,)


def wanda_prune_layer(
    weight: torch.Tensor,           # (out_features, in_features)
    act_norms: torch.Tensor,        # (in_features,)
    sparsity: float = 0.5,
) -> torch.Tensor:
    """
    Apply Wanda pruning to a weight matrix.
    Returns a binary mask (1 = keep, 0 = prune).
    """
    # Score = |W_ij| * ||x_j||_2  (broadcast act_norms across rows)
    scores = weight.abs() * act_norms.unsqueeze(0)       # (out, in)

    # Determine threshold so that `sparsity` fraction are below it
    n_prune = int(sparsity * scores.numel())
    threshold = scores.flatten().kthvalue(n_prune).values

    mask = (scores > threshold).float()                  # 1 = keep
    return mask


# ─────────────────────────────────────────────────────
# Demonstration: prune a tiny test linear layer
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    W = torch.randn(64, 128)           # small layer for illustration
    act_norms = torch.rand(128) * 2.0  # simulated per-feature activation norms

    mask = wanda_prune_layer(W, act_norms, sparsity=0.5)
    sparsity_achieved = 1.0 - mask.mean().item()
    print(f"Sparsity achieved: {sparsity_achieved:.1%}")  # → ~50.0%

    W_pruned = W * mask
    print(f"Non-zero params: {mask.sum().int()} / {mask.numel()}")
```

## 5.6 Knowledge Distillation: Full Implementation

The following is a complete, runnable KD training loop for language models. It handles both the soft-target KD loss and the hard-label cross-entropy loss, with temperature scaling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────
# Knowledge Distillation Loss
# ─────────────────────────────────────────────────────

def kd_loss(
    student_logits: torch.Tensor,    # (B, T, V)
    teacher_logits: torch.Tensor,    # (B, T, V)
    labels: torch.Tensor,            # (B, T)  — ground-truth token ids
    temperature: float = 2.0,
    alpha: float = 0.3,              # weight on the hard CE loss
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Combined KD + CE loss.

    alpha * CE(student, hard_labels) + (1-alpha) * KL(teacher_soft || student_soft)

    The tau^2 factor is included so gradient scale is invariant to temperature.
    """
    B, T, V = student_logits.shape

    # ── 1. Hard-label cross-entropy ──────────────────────────────────────────
    ce = F.cross_entropy(
        student_logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=ignore_index,
    )

    # ── 2. Soft-target KL divergence ─────────────────────────────────────────
    # Mask out ignored positions so they don't contribute to the KD loss
    valid_mask = (labels != ignore_index).reshape(B * T)

    s_flat = student_logits.reshape(B * T, V)[valid_mask]  # (N_valid, V)
    t_flat = teacher_logits.reshape(B * T, V)[valid_mask]  # (N_valid, V)

    # Temperature-scaled log-softmax for student, softmax for teacher
    s_log_probs = F.log_softmax(s_flat / temperature, dim=-1)   # (N, V)
    t_probs     = F.softmax(t_flat / temperature, dim=-1)       # (N, V)

    # KL(teacher || student) = sum_v q*log(q/p) = sum_v q*(log q - log p)
    # F.kl_div(input=log_p, target=q) computes sum_v q*(log q - log p)
    # reduction='batchmean' divides by batch size N_valid
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")

    # Multiply by tau^2 to maintain gradient magnitude
    kl = kl * (temperature ** 2)

    # ── 3. Blend ─────────────────────────────────────────────────────────────
    loss = alpha * ce + (1.0 - alpha) * kl
    return loss, ce.detach(), kl.detach()


# ─────────────────────────────────────────────────────
# Distillation Training Loop
# ─────────────────────────────────────────────────────

@dataclass
class DistillConfig:
    temperature: float = 2.0
    alpha: float = 0.3           # weight on hard-label CE
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 8
    max_seq_len: int = 512
    grad_clip: float = 1.0
    save_path: str = "student_distilled.pt"


def distill(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: DataLoader,
    config: DistillConfig,
    device: str = "cuda",
):
    """
    Off-policy KD training loop.
    Teacher is frozen; student is updated.
    """
    teacher.to(device).eval()
    student.to(device).train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    scaler = torch.cuda.amp.GradScaler()          # bf16/fp16 mixed precision

    for epoch in range(config.epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)    # (B, T)
            labels    = batch["labels"].to(device)       # (B, T)

            # ── Teacher forward (no grad, optional float32 for stability) ──
            with torch.no_grad():
                teacher_out = teacher(input_ids)
                t_logits = teacher_out.logits.float()    # (B, T, V)

            # ── Student forward (with AMP) ─────────────────────────────────
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                student_out = student(input_ids)
                s_logits = student_out.logits             # (B, T, V)

                loss, ce, kl = kd_loss(
                    student_logits=s_logits.float(),      # upcasted for KL
                    teacher_logits=t_logits,
                    labels=labels,
                    temperature=config.temperature,
                    alpha=config.alpha,
                )

            # ── Backward ──────────────────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch} | step {step:5d} | "
                    f"loss={loss:.4f}  ce={ce:.4f}  kl={kl:.4f}"
                )

        print(f"Epoch {epoch} done — avg loss {total_loss/(step+1):.4f}")

    torch.save(student.state_dict(), config.save_path)
    print(f"Student saved to {config.save_path}")
```

!!! warning "Common pitfall: forgetting the τ² scale factor"

    A common bug is to compute the KL divergence with temperature-scaled distributions but omit the $\tau^2$ multiplicative factor. At $\tau = 4$, this makes the KD gradient 16× smaller than the CE gradient, effectively reducing distillation to near-zero influence. Always include `kl * (temperature ** 2)` in the loss.

## 5.7 Speculative Decoding's Draft Models as Distillation

Speculative decoding (covered in full in [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)) uses a small, fast *draft model* to propose tokens that a larger *verifier model* then accepts or rejects. The draft model is effectively a distilled student of the target model.

### The KD Connection

The draft model should approximate the target model's next-token distribution as closely as possible — because the acceptance rate of speculative decoding is:

$$
\alpha = \mathbb{E}_{x \sim p_d} \left[ \min\!\left(1, \frac{p_t(x|c)}{p_d(x|c)}\right) \right]
$$

where $p_t$ is the target distribution and $p_d$ is the draft distribution. When $p_d \approx p_t$, most proposals are accepted. This is exactly the goal of distillation: minimize $\text{KL}(p_t \| p_d)$.

Training draft models with KD from the target model (rather than from scratch) measurably improves acceptance rates. The target model is available at inference time to provide soft-target signals during training.

### Self-Speculative Decoding: Exit Layers

A related technique that makes the KD connection even tighter is *self-speculative decoding* (or "early exit"). The same model uses its own early-layer hidden states as a draft:

{{fig:distill-selfspec-earlyexit}}

The early-exit head is trained with KD to match the final-layer distribution. At inference time, the model runs the first 16 layers quickly, samples a draft token, then optionally runs the remaining 16 layers to verify. The KD objective here is:

$$
\mathcal{L}_\text{EE} = \text{KL}(p_\text{final} \| p_\text{early})
$$

This is distillation within a single model — the final layers teach the early-exit head — and is a good example of how KD ideas permeate modern LLM engineering far beyond the classic teacher→student setup.

### EAGLE: Speculative Drafting with Feature Distillation

EAGLE (Li et al., 2024) takes this further: the draft model conditions on the target model's hidden states (feature distillation) rather than just its output tokens. The draft model is a single transformer layer trained to predict the next token conditioned on the target model's feature map at layer $L-1$. Because the draft model has access to the verifier's internal representations, it achieves acceptance rates in the range of 2–3× speedup on typical text generation tasks.

!!! interview "Interview Corner"

    **Q:** What is knowledge distillation, and why does it work? How does temperature affect the distillation signal?

    **A:** Knowledge distillation trains a small *student* model to match the soft output distribution of a larger *teacher* instead of hard one-hot labels. It works because the teacher's distribution encodes inter-class similarity — e.g., "cat" is closer to "tiger" than to "car" — giving the student a richer learning signal per example than hard labels provide. Temperature $\tau > 1$ is applied to both teacher and student logits before softmax. Raising $\tau$ makes both distributions softer and brings out the teacher's "dark knowledge" (the non-dominant token probabilities). However the gradients shrink by $1/\tau^2$, so we multiply the KL loss by $\tau^2$ to compensate. A $\tau$ of 2–4 is typical for language model distillation — high enough to expose inter-token structure, low enough to retain meaningful signal.

## 5.8 Combining Compression Techniques

In practice, the biggest efficiency wins come from combining multiple compression techniques.

### The Compression Pipeline

{{fig:distill-compression-pipeline}}

### LoRA + Distillation: LoRA-KD

A practical pattern for adapting a distilled student to a new domain is to freeze the base student and train only LoRA adapters (see [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)) with KD from a domain-specialized teacher. This combines:
- **Low parameter count** of PEFT
- **Task-specific knowledge** from the teacher
- **Base capability** of the pretrained student

The distillation loss is computed on the teacher's logits for domain-specific data while only the LoRA adapter parameters receive gradients.

### Quantization-Aware Distillation

Quantization (covered in [Quantization I: Post-Training Quantization](../04-kernels-efficiency/07-quantization-ptq.html)) can degrade model quality, especially at INT4. Quantization-aware distillation (QAD) fine-tunes the quantized student with KD from the full-precision teacher:

$$
\mathcal{L}_\text{QAD} = \text{KL}(q_\tau^\text{FP16-teacher} \| p_\tau^\text{INT4-student})
$$

The teacher's soft targets act as a "correction signal" that helps the INT4 student recover the precision lost by quantization. In practice, QAD can recover 0.5–1.5 perplexity points compared to vanilla quantization of the same model.

### Scaling Law for Distillation

An approximate empirical scaling rule: given a teacher with $N_T$ parameters and a student with $N_S$ parameters (where $N_S \ll N_T$), and a distillation dataset of $D$ tokens, the student achieves roughly the quality of a model of the same size trained from scratch on $D' > D$ tokens. The quality boost from distillation is equivalent to having access to more data, which explains why distilled small models often outperform matched-size models trained from scratch on the same data budget.

This connects to scaling laws (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)): distillation shifts the effective Chinchilla compute-optimal point by reducing the "data requirement" for the student.

!!! example "Worked Example: Memory Budget for Distillation Training"

    Suppose you want to distill a LLaMA-3-70B teacher into a 7B student.

    **Teacher forward pass (inference only, no gradient):**
    - 70B params × 2 bytes (BF16) = 140 GB. Requires a minimum of 2 × A100 80GB or 4 × A100 40GB.
    - Teacher activations for a batch of 8 × 512 tokens at 8,192 hidden dim: roughly 8 × 512 × 8192 × 80 layers × 2 bytes ≈ 4 GB. Manageable.

    **Student forward + backward:**
    - 7B params × 2 bytes = 14 GB for weights.
    - Gradients: another 14 GB (fp32 = 28 GB, or bf16 = 14 GB).
    - Adam optimizer states: 2× gradients = 28 GB (fp32).
    - Activations for the same batch: ≈ 400 MB with gradient checkpointing.
    - Total student-side memory: roughly 55–60 GB — fits on a single A100 80GB with gradient checkpointing.

    **Practical setup:** teacher on 2× A100 (tensor parallel), student on 1× A100. Pre-compute and cache teacher logits on disk (about 40 GB per 1 billion tokens at BF16 for a 32,000-vocab model) to avoid re-running the teacher every epoch.

!!! sota "State of the Art & Resources (2026)"
    Knowledge distillation, pruning, and compression are now standard components of every production LLM pipeline: small reasoning models distilled from 70B+ teachers routinely match earlier frontier performance, and one-shot pruning methods (SparseGPT, Wanda) can halve parameter counts with negligible accuracy loss. The field has converged on combining distillation → structured pruning → quantization for edge deployment.

    **Foundational work**

    - [Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (2015)](https://arxiv.org/abs/1503.02531) — the original KD paper introducing soft targets and temperature scaling.
    - [Sanh et al., *DistilBERT: smaller, faster, cheaper and lighter* (2019)](https://arxiv.org/abs/1910.01108) — seminal application of KD to pre-training a 40%-smaller BERT that retains 97% of performance.

    **Recent advances (2023–2026)**

    - [Frantar & Alistarh, *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot* (2023)](https://arxiv.org/abs/2301.00774) — Hessian-based one-shot pruning to 50–60% sparsity on LLaMA/OPT with negligible perplexity loss.
    - [Sun et al., *A Simple and Effective Pruning Approach for Large Language Models* (2024)](https://arxiv.org/abs/2306.11695) — Wanda: prune by |weight| × activation norm, no Hessian inversion needed.
    - [Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2024)](https://arxiv.org/abs/2306.13649) — GKD: trains student on its own rollouts with teacher feedback, fixing distribution mismatch in standard KD.
    - [Gu et al., *MiniLLM: On-Policy Distillation of Large Language Models* (2024)](https://arxiv.org/abs/2306.08543) — replaces forward KL with reverse KL to prevent student from over-spreading onto low-probability teacher regions.
    - [Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* (2024)](https://arxiv.org/abs/2401.15077) — draft model trained on target model's hidden states achieves 2.7–3.5× inference speedup via feature-level distillation.
    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — shows that verified CoT traces from a 671B teacher can SFT-distill strong reasoning into 7–32B student models.

    **Open-source & tools**

    - [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt) — reference implementation of SparseGPT; supports OPT, BLOOM, and LLaMA with unstructured and 2:4 structured sparsity.
    - [locuslab/wanda](https://github.com/locuslab/wanda) — Wanda pruning code for LLaMA/LLaMA-2/OPT; minimal setup, no retraining required.

    **Go deeper**

    - [Xu et al., *A Survey on Knowledge Distillation of Large Language Models* (2024)](https://arxiv.org/abs/2402.13116) — comprehensive taxonomy covering algorithm design, skill transfer, and enterprise applications of LLM KD.

## Further Reading

- Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network," NIPS 2014 Workshop.
- Kim & Rush, "Sequence-Level Knowledge Distillation," EMNLP 2016.
- Sanh et al., "DistilBERT, a distilled version of BERT," arXiv 2019.
- Touvron et al., "Training data-efficient image transformers & distillation through attention," ICML 2021. (DeiT — a foundational vision distillation paper.)
- Frantar & Alistarh, "SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot," ICML 2023.
- Sun et al., "A Simple and Effective Pruning Approach for Large Language Models," ICLR 2024. (Wanda)
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv 2025. (Section on distillation of reasoning traces.)
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," arXiv 2024.
- Leviathan, Kalman & Matias, "Fast Inference from Transformers via Speculative Decoding," ICML 2023.

!!! key "Key Takeaways"

    - Knowledge distillation trains a student to match a teacher's *soft probability distribution*, not just its hard predictions. The KD loss is a temperature-scaled KL divergence, and the $\tau^2$ factor must be included to keep gradient magnitude invariant to temperature.
    - Temperature $\tau > 1$ softens both distributions, exposing the teacher's "dark knowledge" (inter-token similarities). A value of 2–4 is typical; too high and the signal becomes noise.
    - Off-policy distillation is cheap but suffers from distribution mismatch. On-policy distillation trains on the student's own generations (evaluated by the teacher) and closes the gap for long, multi-step tasks.
    - Sequence-level KD (SeqKD) uses the teacher's greedy output as hard training targets — a simple, cheap alternative to per-token KL that still captures teacher behavior.
    - Reasoning distillation (e.g., R1-style) works by collecting verified chain-of-thought traces from a large model and using them as SFT targets for a small model. The student learns *behavior*, not just output distributions.
    - SparseGPT and Wanda enable one-shot unstructured pruning of LLMs at 50%+ sparsity with near-zero perplexity degradation. Wanda's criterion (|w| × activation norm) requires no Hessian inversion and is extremely fast.
    - Speculative decoding's draft models are conceptually distilled students: a good draft model minimizes $\text{KL}(p_\text{target} \| p_\text{draft})$, and training the draft with KD from the target measurably improves acceptance rates.
    - Compression techniques stack: distillation → structured pruning → quantization → speculative decoding can take a 70B model to a practical on-device deployment. The quality at each step depends heavily on the ordering and the calibration data.
