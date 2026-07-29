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

On-policy distillation typically outperforms off-policy on longer generations and tasks that require multi-step reasoning, at the cost of running the student (and evaluating the teacher) online during training. It is worth seeing how it compares to RL on the same rollouts: RLVR (see [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)) scores an entire rollout with *one* scalar, so a 500-token trace carries a single bit or two of learning signal, while on-policy distillation gives the student a full corrective distribution at *every* token of that same trace. That density is why on-policy distillation is usually far more sample-efficient than RL when a competent teacher exists — and why it is not a substitute for RL when no teacher is better than the student.

{{fig:distill-onpolicy-vs-offpolicy}}

### Which Divergence? Forward KL vs Reverse KL

The KD loss in Section 5.1 uses the *forward* KL, $\text{KL}(q \| p)$ — teacher first. Forward KL is **mode-covering**: it blows up wherever the teacher has mass and the student has none, so the student is pushed to spread probability across everything the teacher might plausibly say. With a high-capacity student that is exactly right. With a small student it is a liability: the smeared student ends up placing real mass *between* the teacher's modes, on text the teacher would never generate — the "fluent nonsense" failure mode.

The *reverse* KL, $\text{KL}(p \| q)$ — student first — is **mode-seeking**: it only penalizes the student where the student itself puts mass, so the student commits to a subset of the teacher's behavior and generates it cleanly. MiniLLM (Gu et al.) showed this is usually the better objective for open-ended LLM generation. Note that SeqKD (Section 5.3) is implicitly mode-seeking too: greedy teacher text *is* one mode. The rule of thumb: **the larger the capacity gap, the further toward the reverse-KL end you should sit.**

### GKD: One Algorithm With Both Dials

Generalized Knowledge Distillation (Agarwal et al., 2024) turns both of the above choices into continuous knobs rather than forks:

- $\lambda$ — the **student data fraction**. Each batch is drawn from the student's own rollouts with probability $\lambda$, and from the fixed teacher-labeled dataset otherwise. $\lambda = 0$ is pure off-policy, $\lambda = 1$ pure on-policy.
- $\beta$ — the **divergence**, via the generalized Jensen–Shannon divergence between teacher $q$ and student $p$:

$$
\mathcal{D}_\beta(q, p) = \beta\,\text{KL}\!\left(q \,\|\, m\right) + (1-\beta)\,\text{KL}\!\left(p \,\|\, m\right), \qquad m = \beta q + (1-\beta) p
$$

Unlike either KL, $\mathcal{D}_\beta$ is bounded and always finite, which makes it numerically well-behaved even when the two distributions barely overlap early in training. Its endpoints degenerate (in the appropriate limit) to the two KL directions, and that is how the library exposes it.

The reference implementation is Hugging Face **TRL**'s `GKDTrainer`, a subclass of `SFTTrainer` that additionally takes a teacher and handles the on-policy generation loop for you:

```python
# pip install "trl>=0.12" transformers datasets accelerate
#
# NOTE: in current TRL the trainer lives under the experimental namespace;
# older releases exported it directly as `from trl import GKDConfig, GKDTrainer`.
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

student_id = "Qwen/Qwen2.5-0.5B-Instruct"
teacher_id = "Qwen/Qwen2.5-7B-Instruct"     # MUST share the student's tokenizer

student = AutoModelForCausalLM.from_pretrained(student_id)
teacher = AutoModelForCausalLM.from_pretrained(teacher_id)
tok     = AutoTokenizer.from_pretrained(student_id)

# Chat-format rows; the teacher will be asked to score the student's rollouts.
train_dataset = Dataset.from_dict({
    "messages": [[
        {"role": "user",      "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]] * 64
})

args = GKDConfig(
    output_dir="gkd-student",
    lmbda=1.0,        # 1.0 -> fully on-policy: train on the student's own samples
    beta=0.5,         # 0.0 -> forward KL, 1.0 -> reverse KL, 0.5 -> symmetric JSD
    temperature=0.9,  # sampling temperature for the student's rollouts
    max_new_tokens=256,
    seq_kd=False,     # True -> plain SFT on teacher-generated text (Section 5.3)
    per_device_train_batch_size=2,
)

trainer = GKDTrainer(
    model=student,
    teacher_model=teacher,
    args=args,
    processing_class=tok,
    train_dataset=train_dataset,
)
trainer.train()
```

Two practical notes. First, `lmbda=1.0` makes every step pay for a generation pass, so throughput is dominated by decoding, not by the backward pass — the same bottleneck that motivates vLLM/SGLang rollout engines in RL training (see [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html)). Second, `GKDTrainer` assumes teacher and student share a tokenizer; see the warning in Section 5.3 for what to do when they do not.

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

!!! warning "Common pitfall: token-level KD requires a shared tokenizer"

    Every loss in Sections 5.1–5.2 assumes $z^T$ and $z^S$ index the *same* vocabulary and that position $t$ denotes the same text span in both models. If your teacher is a Qwen model and your student has its own byte-level BPE — as Stack-100M does, see [A Byte-Level BPE Tokenizer From Scratch](../14-capstone/03-tokenizer.html) — neither assumption holds and the KL is simply meaningless. Three ways out, in increasing order of effort:

    1. **Give the student the teacher's tokenizer.** This is why the DeepSeek-R1-Distill releases keep the Qwen and Llama vocabularies intact. Cheapest option, and it constrains your architecture choices very little.
    2. **Use SeqKD / trajectory distillation.** These need only the teacher's *text*, so any teacher can teach any student — including a closed API model you can never get logits from. This is the route the capstone takes in [A Narrow Auto-Research Agent](../14-capstone/10-agentic-narrow.html), and it is the right default at 100M scale.
    3. **Cross-tokenizer logit distillation.** The Universal Logit Distillation loss (Boizard et al., TMLR 2025) matches the two distributions with an optimal-transport cost instead of assuming a shared index set. Powerful, but adds real machinery.

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
- **Capability ceiling.** A 1B student will hit a hard ceiling regardless of trace quality — if the intermediate steps require knowledge the student doesn't have, imitation fails. Scaling to at least 7B is typically recommended for *general* reasoning. The ceiling is about *breadth*, not about reasoning being impossible below 7B: on a narrow, closed domain where every fact the trace needs is either in the prompt or retrievable, much smaller students work. That is exactly the bet Stack-100M makes in [A Narrow Auto-Research Agent](../14-capstone/10-agentic-narrow.html), and the honest framing there is worth internalizing — a 100M model does not learn to reason about tool use, it learns to reproduce one well-worn groove of it.

!!! tip "Practitioner tip: distillation is the *only* post-training that works at 100M"

    At the scale of the capstone model, the ordering of techniques inverts relative to frontier practice. RLVR cannot bootstrap a behavior the base model never emits, and a 100M model emits essentially no correct multi-step traces to reinforce. So the pipeline is: **distill first** (rejection-sampled teacher trajectories as SFT targets), *then* apply a small dose of RLVR to sharpen the distilled groove — see [Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M](../14-capstone/09-post-training.html). Filtering matters more than volume: a few hundred verified traces in a rigidly consistent format beat tens of thousands of unfiltered ones, because a small student spends most of its capacity learning the format.

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

{{fig:distill-wanda-vs-magnitude}}

### Structured Pruning

Structured pruning removes entire components: attention heads, MLP neurons, or even full transformer layers. This yields speedup on any hardware without needing sparse kernels.

**Attention head pruning.** Michel et al. (2019) showed that many attention heads can be removed with limited performance degradation. A simple sensitivity analysis: mask one head at a time, measure validation loss increase, prune heads with lowest importance.

**Layer dropping.** Some layers in deep transformers are near-identity mappings (the residual stream barely changes). Dropping such layers yields surprisingly small accuracy drops and significant latency reductions.

**Width pruning.** Prune MLP intermediate dimensions or the hidden dimension of attention projections. This requires careful co-pruning of the weight matrices on both sides of a pruned feature.

**Prune, then distill — the Minitron recipe.** Structured pruning on its own always costs accuracy, because you have physically deleted computation. The fix that has become standard practice is to treat the *unpruned* model as a teacher and the *pruned* model as a student: NVIDIA's Minitron work (Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation*, 2024) estimates per-component importance from activation statistics on a small calibration set, prunes depth and width to a target size, then distills from the original model on a modest retraining budget — reported as on the order of tens of billions of tokens, versus the trillions a from-scratch model of that size would need. This is why the two halves of this chapter belong together, and it is the cheapest known way to produce a *family* of sizes from one expensive pretraining run: train once at the large size, then prune-and-distill down the ladder. The recipe is implemented in NVIDIA's Megatron-LM/NeMo toolchain, and it is the direct ancestor of the "prune a 1B down to 100M" shortcut you could take instead of pretraining Stack-100M from scratch (see [The Capstone: Building Stack-100M](../14-capstone/01-overview-and-landscape.html) for why we pretrain anyway — the point of the capstone is to see every stage).

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

    Wanda's comparison group is *per output row*, not global: within each row we
    drop the `sparsity` fraction of lowest-scoring input connections. This detail
    matters — the paper shows per-output grouping clearly beats a single global
    threshold, because a global threshold lets a few "loud" output neurons consume
    the whole keep-budget and leaves other neurons almost entirely disconnected.
    It is also what makes an n:m mask (e.g. 2:4) a one-line variation: group the
    row into consecutive blocks of 4 and keep the top 2 in each.
    """
    # Score = |W_ij| * ||x_j||_2  (broadcast act_norms across rows)
    scores = weight.abs() * act_norms.unsqueeze(0)       # (out, in)

    n_prune = int(sparsity * weight.shape[1])            # per row, not global
    if n_prune == 0:
        return torch.ones_like(weight)

    # Indices of the lowest-scoring n_prune entries in each row
    _, prune_idx = scores.topk(n_prune, dim=1, largest=False)  # (out, n_prune)
    mask = torch.ones_like(scores)
    mask.scatter_(1, prune_idx, 0.0)                     # 1 = keep, 0 = prune
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

### Doing This For Real: `llm-compressor`

You would not hand-roll the loop above for a real checkpoint — hooking every linear layer, streaming calibration data, and keeping a 70B model's Hessians off the GPU is the hard part, not the scoring formula. The maintained toolkit is the vLLM Project's [`llm-compressor`](https://github.com/vllm-project/llm-compressor) (the successor to Neural Magic's SparseML), which implements SparseGPT, Wanda, magnitude pruning, GPTQ/AWQ/SmoothQuant, and FP8/NVFP4 behind one `oneshot()` entry point and writes checkpoints vLLM and SGLang can load directly. The same library is used for quantization in [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html); here we drive its pruning modifiers.

```python
# pip install llmcompressor
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier  # or WandaPruningModifier

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
N_CALIB, MAX_LEN = 512, 2048

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── Calibration set: a few hundred in-distribution sequences is enough ────────
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N_CALIB}]")
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"],
                                                             tokenize=False)})
ds = ds.map(lambda s: tokenizer(s["text"], max_length=MAX_LEN, truncation=True,
                                add_special_tokens=False),
            remove_columns=ds.column_names)

# ── Recipe: 2:4 semi-structured sparsity, never touching the output head ─────
recipe = SparseGPTModifier(
    sparsity=0.5,
    mask_structure="2:4",     # "unstructured" for plain 50% sparsity
    targets=["Linear"],
    ignore=["lm_head"],       # pruning the head costs far more than it saves
)

oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=MAX_LEN, num_calibration_samples=N_CALIB)

model.save_pretrained("Llama-3.2-1B-2of4", save_compressed=True)
tokenizer.save_pretrained("Llama-3.2-1B-2of4")
```

Swap in `WandaPruningModifier(sparsity=0.5, mask_structure="2:4", targets=["Linear"], ignore=["lm_head"])` for the Hessian-free variant — same call, seconds instead of minutes per layer. Both modifiers also accept `sparsity_profile="owl"` (Outlier-Weighed Layerwise sparsity), which allocates *non-uniform* sparsity across layers based on how many activation outliers each one carries; at aggressive ratios (70%+) this recovers a meaningful chunk of the quality that uniform sparsity throws away.

!!! warning "Common pitfall: 2:4 sparsity only pays off with the right kernel"

    A 2:4-sparse checkpoint stored as a dense tensor with zeros in it is *slower* than the dense original — you saved nothing and added bookkeeping. The speedup comes from NVIDIA's sparse tensor cores (Ampere and later) reached through a compressed storage format, which is why `save_compressed=True` and a serving stack that understands the format both matter. Always benchmark end-to-end tokens/second, not parameter counts.

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined KD + CE loss. Returns (total_loss, ce_detached, kl_detached).

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
    # No GradScaler: bf16 keeps fp32's exponent range, so gradients do not
    # underflow the way they do in fp16. (Use torch.amp.GradScaler("cuda")
    # only if you switch the autocast dtype below to torch.float16.)

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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
            optimizer.step()

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

### Top-k Teacher Logits: What Makes Off-Policy KD Affordable

The loop above re-runs the teacher on every batch of every epoch. The obvious fix — precompute the teacher's logits once and cache them — runs straight into arithmetic. A full logit row for a 32,000-token vocabulary in BF16 is 64 KB **per token**. One billion tokens is therefore about 64 TB of cache. Nobody does this.

What everyone actually does is cache the **top-k** entries. Language-model next-token distributions are extremely concentrated: for a well-trained teacher, the top 64 tokens typically carry the overwhelming majority of the probability mass, and the tail is exactly the part the student cannot learn anything useful from anyway. At $k = 64$, storing BF16 values plus `uint16` indices (valid whenever the vocabulary fits in 65,536 entries) costs $64 \times 4 = 256$ bytes per token — a **250× reduction**, turning 100M cached tokens into roughly 26 GB. That fits on one NVMe drive and can be memory-mapped by the dataloader.

The loss then compares two distributions restricted to the *same* $k$-element support and renormalized identically:

```python
import torch
import torch.nn.functional as F


@torch.no_grad()
def cache_teacher_topk(teacher, input_ids, k: int = 64):
    """Run the teacher once and keep only its top-k logits per position.

    Save `values` as bf16 and, on disk, `indices` as np.uint16 when V < 65536 —
    that halves the index cost relative to int32.
    """
    logits = teacher(input_ids).logits            # (B, T, V)
    values, indices = logits.topk(k, dim=-1)      # (B, T, k) each
    return values.to(torch.bfloat16), indices.to(torch.int32)


def topk_kd_loss(
    student_logits: torch.Tensor,   # (B, T, V)
    t_values: torch.Tensor,         # (B, T, k) cached teacher logits
    t_indices: torch.Tensor,        # (B, T, k) their vocabulary ids
    labels: torch.Tensor,           # (B, T)
    temperature: float = 2.0,
    alpha: float = 0.3,
    ignore_index: int = -100,
):
    """KD against a top-k teacher cache.

    Both sides are softmaxed over the SAME k vocabulary entries, which makes the
    teacher's truncated distribution and the student's gathered one comparable.
    (Renormalizing over the top-k support is the standard choice; the alternative
    is to add a k+1-th "everything else" bucket holding the residual mass.)
    """
    B, T, V = student_logits.shape

    ce = F.cross_entropy(
        student_logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=ignore_index,
    )

    valid = (labels != ignore_index).reshape(B * T)
    s_flat = student_logits.reshape(B * T, V)[valid]              # (N, V)
    tv = t_values.reshape(B * T, -1)[valid].float()               # (N, k)
    ti = t_indices.reshape(B * T, -1)[valid].long()               # (N, k)

    t_probs = F.softmax(tv / temperature, dim=-1)                 # (N, k)
    s_sel = torch.gather(s_flat, 1, ti)                           # (N, k)
    s_log_probs = F.log_softmax(s_sel / temperature, dim=-1)      # (N, k)

    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
    kl = kl * (temperature ** 2)

    return alpha * ce + (1.0 - alpha) * kl, ce.detach(), kl.detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V, k = 2, 8, 512, 64
    s_logits = torch.randn(B, T, V, requires_grad=True)
    t_logits = torch.randn(B, T, V) * 2.0                 # a sharper "teacher"
    tv, ti = t_logits.topk(k, dim=-1)
    labels = torch.randint(0, V, (B, T))

    loss, ce, kl = topk_kd_loss(s_logits, tv, ti, labels)
    loss.backward()
    print(f"loss={loss:.4f}  ce={ce:.4f}  kl={kl:.4f}")
```

!!! tip "Practitioner tip: pick k by measuring captured mass, not by guessing"

    Before committing to a cache, run the teacher on a few thousand tokens and histogram $\sum_{v \in \text{top-}k} q(v)$. If the median captured mass at $k = 64$ is above ~0.99 you are fine; if your teacher is unusually high-entropy (early-training checkpoints, or high sampling temperature in the data) you may need $k = 128$ or 256. Note the interaction with temperature: the KD temperature $\tau$ is applied *after* truncation here, so a large $\tau$ redistributes mass only within the kept support — another reason not to push $\tau$ past ~5.

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

EAGLE (Li et al., 2024) takes this further: the draft model conditions on the target model's hidden states (feature distillation) rather than just its output tokens. The draft model is a single transformer layer trained to predict the next token conditioned on the target model's feature map at layer $L-1$. Because the draft model has access to the verifier's internal representations, it achieves acceptance rates in the range of 2–3× speedup on typical text generation tasks. EAGLE-3 (Li et al., 2025) — the current standard-bearer, integrated into vLLM and SGLang — drops feature prediction in favor of direct token prediction with multi-layer feature fusion ("training-time test"), which lets acceptance keep improving as you scale draft-training data and pushes speedups up to ~6.5×.

Training your own draft head is packaged too: the SGLang project ships [SpecForge](https://github.com/sgl-project/SpecForge), a training framework for EAGLE-style draft models that exports checkpoints SGLang can serve directly, and vLLM loads EAGLE/EAGLE-3 heads through its speculative-decoding config. In other words, the whole loop of this section — *distill a draft from your target model, then serve the pair* — is now a two-library workflow rather than a research project.

!!! interview "Interview Corner"

    **Q:** What is knowledge distillation, and why does it work? How does temperature affect the distillation signal?

    **A:** Knowledge distillation trains a small *student* model to match the soft output distribution of a larger *teacher* instead of hard one-hot labels. It works because the teacher's distribution encodes inter-class similarity — e.g., "cat" is closer to "tiger" than to "car" — giving the student a richer learning signal per example than hard labels provide. Temperature $\tau > 1$ is applied to both teacher and student logits before softmax. Raising $\tau$ makes both distributions softer and brings out the teacher's "dark knowledge" (the non-dominant token probabilities). However the gradients shrink by $1/\tau^2$, so we multiply the KL loss by $\tau^2$ to compensate. A $\tau$ of 2–4 is typical for language model distillation — high enough to expose inter-token structure, low enough to retain meaningful signal.

## 5.8 Combining Compression Techniques

In practice, the biggest efficiency wins come from combining multiple compression techniques.

### The Compression Pipeline

{{fig:distill-compression-pipeline}}

Ordering is not arbitrary. **Distill or prune before you quantize**, because both operations change the weight distribution and any quantization scales you calibrated beforehand become wrong. Within compression, **prune before you quantize**: SparseGPT and GPTQ share the same OBS machinery, and applying sparsity first lets the quantizer's error compensation absorb some of the pruning damage rather than fight it (`llm-compressor` supports exactly this ordering in a single multi-modifier recipe). Finally, **quantize before you export to GGUF/llama.cpp** — the GGUF conversion is the last step, not a stage you compress after. The capstone walks the concrete end of this for a 100M model in [Evaluation & Serving: Honest Benchmarks, int4 Quantization, and Running on a Laptop](../14-capstone/11-evaluation-and-serving.html).

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

    **Practical setup:** teacher on 2× A100 (tensor parallel), student on 1× A100. To avoid re-running the teacher every epoch you want a logit cache — but caching the *full* distribution is hopeless: 32,000 vocab × 2 bytes = 64 KB per token, i.e. ~64 TB per billion tokens. Cache the **top-k** instead (Section 5.6): $k = 64$ at BF16 values plus `uint16` indices is 256 bytes per token, so 100M tokens costs ~26 GB — the difference between "impossible" and "one NVMe drive."

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
    - [Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (2024)](https://arxiv.org/abs/2407.14679) — the Minitron recipe: activation-based importance estimation, structured depth/width pruning, then distillation from the unpruned parent; the standard way to derive a family of model sizes from one pretraining run.
    - [Boizard et al., *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs* (TMLR 2025)](https://arxiv.org/abs/2402.12030) — an optimal-transport loss that removes the shared-tokenizer requirement of token-level KD.
    - [Li et al., *EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test* (2025)](https://arxiv.org/abs/2503.01840) — the 2026 draft-model standard (in vLLM/SGLang): direct token prediction + multi-layer feature fusion, speedups up to ~6.5× with a scaling law in draft-training data. See the earlier [EAGLE (2024)](https://arxiv.org/abs/2401.15077) for the original feature-distillation formulation.
    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — shows that verified CoT traces from a 671B teacher can SFT-distill strong reasoning into 7–32B student models; the R1-Distill-Qwen/Llama releases (1.5B–70B) remain the reference open recipe, now commonly reproduced on Qwen3 backbones.

    **Open-source & tools**

    - [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) — the maintained production toolkit (successor to Neural Magic's SparseML): `SparseGPTModifier`, `WandaPruningModifier`, magnitude pruning, OWL non-uniform sparsity profiles, plus GPTQ/AWQ/SmoothQuant/FP8, all behind one `oneshot()` call and writing vLLM/SGLang-loadable checkpoints.
    - [huggingface/trl](https://github.com/huggingface/trl) — `GKDTrainer` implements generalized KD with the `lmbda` (on-policy fraction) and `beta` (forward↔reverse KL) dials, plus `seq_kd` for sequence-level KD; the fastest path from a teacher checkpoint to a distilled student.
    - [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt) — reference implementation of SparseGPT; supports OPT, BLOOM, and LLaMA with unstructured and 2:4 structured sparsity.
    - [locuslab/wanda](https://github.com/locuslab/wanda) — Wanda pruning code for LLaMA/LLaMA-2/OPT; minimal setup, no retraining required.
    - [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) — training framework for EAGLE-style speculative draft models, exporting checkpoints that plug straight into SGLang serving.

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
- Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes," ICLR 2024. (GKD; implemented as `GKDTrainer` in Hugging Face TRL.)
- Muralidharan et al., "Compact Language Models via Pruning and Knowledge Distillation," NeurIPS 2024. (Minitron — prune-then-distill.)
- The `vllm-project/llm-compressor` repository — the maintained implementation of SparseGPT/Wanda pruning and PTQ for vLLM-served checkpoints.

## Exercises

**1.** In the KD loss the temperature-scaled KL term is multiplied by $\tau^2$. Explain in your own words *why* this factor is needed, and describe concretely what goes wrong if you distill at $\tau = 4$ but forget the factor while keeping the blend weight $\alpha = 0.3$.

??? note "Solution"
    Applying a temperature $\tau > 1$ before the softmax divides every logit by $\tau$. Because the softmax gradient with respect to the logits scales like $1/\tau$ on each side (student and teacher), the gradient of the KL term scales like $1/\tau^2$ overall. So *softening* the distributions to expose the teacher's "dark knowledge" comes with an unwanted side effect: the gradient magnitude collapses by $1/\tau^2$. Multiplying the KL loss by $\tau^2$ exactly cancels this shrinkage, so the size of the distillation gradient is invariant to the choice of $\tau$. That decouples the two roles of temperature — *how soft the targets are* vs *how strong the update is*.

    Concretely at $\tau = 4$ the factor is $\tau^2 = 16$. Forgetting it makes the KD gradient $16\times$ smaller than it should be. In the blended loss $\mathcal{L} = \alpha\,\mathcal{L}_\text{CE} + (1-\alpha)\,\mathcal{L}_\text{KD}$ with $\alpha = 0.3$, the hard-label CE term keeps its full gradient while the soft KD term is effectively down-weighted by another factor of 16. The $(1-\alpha)=0.7$ nominal weight on distillation is silently reduced to about $0.7/16 \approx 0.044$ in gradient terms, so the student is trained almost entirely on hard labels and the distillation signal is nearly wasted — you pay the full cost of running the teacher for almost no benefit.

**2.** A teacher outputs logits $z^T = [2.0, 1.0, 0.0]$ over three tokens. Compute the softmax probabilities at $\tau = 1$ and at $\tau = 2$ (work to 3 decimals). Comment on what the change does to the "dark knowledge" available to the student.

??? note "Solution"
    At $\tau = 1$ we exponentiate the raw logits:

    $$
    e^{2.0}=7.389,\quad e^{1.0}=2.718,\quad e^{0.0}=1.000,\quad \text{sum}=11.107
    $$

    $$
    p = [7.389, 2.718, 1.000]/11.107 = [0.665,\ 0.245,\ 0.090]
    $$

    At $\tau = 2$ the scaled logits are $z^T/\tau = [1.0, 0.5, 0.0]$:

    $$
    e^{1.0}=2.718,\quad e^{0.5}=1.649,\quad e^{0.0}=1.000,\quad \text{sum}=5.367
    $$

    $$
    p_\tau = [2.718, 1.649, 1.000]/5.367 = [0.506,\ 0.307,\ 0.186]
    $$

    Raising the temperature moves probability mass off the top token (0.665 -> 0.506) and onto the two non-dominant tokens (0.245 -> 0.307 and 0.090 -> 0.186). The *relative ordering* is preserved, but the runner-up tokens now carry much more probability, so the KL target tells the student explicitly that token 1 is a strong alternative and token 2 a weaker-but-real one. That inter-token structure is the "dark knowledge" a hard one-hot label ($[1,0,0]$) throws away entirely.

**3.** Speculative decoding accepts a draft token with probability $\min\!\left(1, p_t/p_d\right)$, giving expected acceptance $\alpha = \mathbb{E}_{x \sim p_d}\!\big[\min(1, p_t(x)/p_d(x))\big]$. For a target distribution $p_t = [0.5, 0.3, 0.2]$, compute $\alpha$ for two candidate draft models: draft A $= [0.4, 0.4, 0.2]$ and draft B $= [0.7, 0.2, 0.1]$. Which draft is better, and how does the answer relate to the distillation objective?

??? note "Solution"
    Using the identity $\alpha = \sum_x p_d(x)\,\min(1, p_t(x)/p_d(x)) = \sum_x \min(p_d(x), p_t(x))$ (the total overlapping mass of the two distributions):

    Draft A vs target:

    $$
    \min(0.4,0.5)+\min(0.4,0.3)+\min(0.2,0.2) = 0.4+0.3+0.2 = 0.9
    $$

    Draft B vs target:

    $$
    \min(0.7,0.5)+\min(0.2,0.3)+\min(0.1,0.2) = 0.5+0.2+0.1 = 0.8
    $$

    Draft A has the higher acceptance rate ($\alpha = 0.9$ vs $0.8$), so it is the better draft. Draft A's distribution is closer to the target's, which is exactly what distillation optimizes: minimizing $\text{KL}(p_t \,\|\, p_d)$ drives $p_d \to p_t$, increases the overlapping mass, and therefore raises the acceptance rate. This is why training a draft model with KD from the target model measurably improves speculative-decoding throughput.

**4.** During one training step a distillation run reports a hard-label cross-entropy of $\mathcal{L}_\text{CE} = 2.0$ and a *temperature-scaled but not yet $\tau^2$-corrected* KL value of $0.05$, using $\tau = 4$ and $\alpha = 0.3$. Compute the total loss (a) with the $\tau^2$ factor applied and (b) without it. What fraction of the total loss comes from distillation in each case?

??? note "Solution"
    The blended loss is $\mathcal{L} = \alpha\,\mathcal{L}_\text{CE} + (1-\alpha)\,\mathcal{L}_\text{KD}$ with $\alpha = 0.3$, so the CE contribution is $0.3 \times 2.0 = 0.6$ in both cases.

    (a) With the factor, $\mathcal{L}_\text{KD} = \tau^2 \times 0.05 = 16 \times 0.05 = 0.8$. Distillation contribution $= 0.7 \times 0.8 = 0.56$.

    $$
    \mathcal{L} = 0.6 + 0.56 = 1.16,\qquad \text{KD share} = 0.56/1.16 \approx 48.3\%
    $$

    (b) Without the factor, $\mathcal{L}_\text{KD} = 0.05$. Distillation contribution $= 0.7 \times 0.05 = 0.035$.

    $$
    \mathcal{L} = 0.6 + 0.035 = 0.635,\qquad \text{KD share} = 0.035/0.635 \approx 5.5\%
    $$

    With the correct $\tau^2$ factor distillation and CE contribute roughly equally (48% vs 52%); without it the KD term is a negligible ~5.5% of the loss, matching the pitfall from Exercise 1 — the student is trained almost entirely on hard labels.

**5.** *Implementation.* MiniLLM argues that language-model distillation should minimize the *reverse* KL $\text{KL}(p_\text{student} \,\|\, q_\text{teacher})$ instead of the forward KL used in the chapter's `kd_loss`, so the student avoids spreading probability onto low-probability teacher regions. Modify `kd_loss` to compute the reverse-KL variant while keeping temperature scaling, the $\tau^2$ factor, the ignore-index masking, and the blend with hard-label CE.

??? note "Solution"
    Forward KL is $\sum_v q\log(q/p)$; reverse KL is $\sum_v p\log(p/q)$, i.e. we swap the roles of student and teacher. We now need the student's *probabilities* (weighted average) and both log-probabilities. `F.kl_div(input=log_target, target=source)` computes $\sum \text{source}\cdot(\log \text{source} - \log\text{target})$, so to get $\text{KL}(p\|q)$ we pass `input = log_q` and `target = p`.

    ```python
    import torch
    import torch.nn.functional as F

    def kd_loss_reverse(
        student_logits: torch.Tensor,    # (B, T, V)
        teacher_logits: torch.Tensor,    # (B, T, V)
        labels: torch.Tensor,            # (B, T)
        temperature: float = 2.0,
        alpha: float = 0.3,
        ignore_index: int = -100,
    ):
        """alpha * CE(student, hard) + (1-alpha) * KL(student_soft || teacher_soft)."""
        B, T, V = student_logits.shape

        # 1. Hard-label CE (unchanged)
        ce = F.cross_entropy(
            student_logits.reshape(B * T, V),
            labels.reshape(B * T),
            ignore_index=ignore_index,
        )

        # 2. Reverse KL on valid (non-ignored) positions
        valid = (labels != ignore_index).reshape(B * T)
        s_flat = student_logits.reshape(B * T, V)[valid]
        t_flat = teacher_logits.reshape(B * T, V)[valid]

        # Student provides the "source" distribution p; teacher the log-target log q
        s_log_probs = F.log_softmax(s_flat / temperature, dim=-1)   # log p
        s_probs     = s_log_probs.exp()                            # p
        t_log_probs = F.log_softmax(t_flat / temperature, dim=-1)  # log q

        # KL(p || q) = sum_v p * (log p - log q)
        # F.kl_div(input=log q, target=p) = sum_v p*(log p - log q)
        kl = F.kl_div(t_log_probs, s_probs, reduction="batchmean")
        kl = kl * (temperature ** 2)   # keep gradient scale invariant to tau

        loss = alpha * ce + (1.0 - alpha) * kl
        return loss, ce.detach(), kl.detach()
    ```

    The three substantive changes versus the forward-KL version: (i) we compute the student's `s_probs` as the distribution being averaged over; (ii) we call `F.kl_div` with the *teacher's* log-probs as `input` and the *student's* probs as `target`, which flips the divergence direction; (iii) everything else — temperature scaling, the `* temperature**2` factor, `ignore_index` masking, and the CE blend — is preserved. Note that because `s_probs` now depends on the student and multiplies the gradient, reverse KL is "mode-seeking": it heavily penalizes the student for putting mass where the teacher assigns near-zero probability, which is the behavior MiniLLM wants.

**6.** *Implementation.* The chapter's `wanda_prune_layer` scores weights by $|w_{ij}| \cdot \|x_j\|_2$. Implement the plain **magnitude** baseline (`magnitude_prune_layer`, score $= |w_{ij}|$, a single 50% *global* threshold, same mask convention where 1 = keep) and write a short check that constructs a weight matrix where magnitude and Wanda disagree on which weights to keep. Explain what feature of the activations produces the disagreement.

??? note "Solution"
    The magnitude baseline is the Wanda function with the activation term removed — score by absolute weight only:

    ```python
    import torch

    def magnitude_prune_layer(
        weight: torch.Tensor,       # (out_features, in_features)
        sparsity: float = 0.5,
    ) -> torch.Tensor:
        """Unstructured magnitude pruning. Returns a binary mask (1 = keep, 0 = prune)."""
        scores = weight.abs()                                # no activation term
        n_prune = int(sparsity * scores.numel())
        threshold = scores.flatten().kthvalue(n_prune).values
        mask = (scores > threshold).float()                  # 1 = keep
        return mask
    ```

    A check where the two criteria disagree. Give one input feature a large weight but a tiny activation norm, and another a smaller weight but a large activation norm:

    ```python
    from types import SimpleNamespace  # not needed; illustrative only

    # 1 output, 2 input features
    W = torch.tensor([[1.0, 0.5]])          # feature 0 has the larger weight
    act = torch.tensor([0.1, 10.0])         # but feature 1 dominates at runtime

    # Magnitude keeps the top 50% (1 of 2) purely by |w|: keeps feature 0.
    m_mag = magnitude_prune_layer(W, sparsity=0.5)

    # Wanda scores: |1.0|*0.1 = 0.10  vs  |0.5|*10.0 = 5.0  -> keeps feature 1.
    m_wanda = wanda_prune_layer(W, act, sparsity=0.5)

    print("magnitude keeps:", m_mag.tolist())   # [[1.0, 0.0]] -> feature 0
    print("wanda keeps:    ", m_wanda.tolist())  # [[0.0, 1.0]] -> feature 1
    ```

    The disagreement is produced by *activation scale*. Magnitude pruning assumes every input feature is equally important, so it keeps the numerically larger weight (feature 0). Wanda multiplies by $\|x_j\|_2$, so it recognizes that feature 1, despite its smaller weight, contributes far more to the layer's output because its activations are ~100x larger ($0.5 \times 10 = 5.0$ vs $1.0 \times 0.1 = 0.1$). Whenever the per-feature activation norms are highly non-uniform — which is common in LLMs, where a few "outlier" feature dimensions carry disproportionate magnitude — the two criteria diverge, and Wanda's activation-aware score is the better predictor of which weights actually matter at runtime.

    One more difference worth noticing in the two implementations: `magnitude_prune_layer` uses a *single global* threshold over the whole matrix, while `wanda_prune_layer` prunes *per output row*. That is deliberate and is itself part of Wanda's contribution — a global threshold lets a few high-magnitude output neurons monopolize the keep-budget and leaves other neurons nearly disconnected, whereas per-row grouping guarantees every output keeps the same number of inputs. Try re-running the magnitude baseline with per-row grouping on a matrix whose rows have very different scales to see how much of the gap that one change closes.

!!! key "Key Takeaways"

    - Knowledge distillation trains a student to match a teacher's *soft probability distribution*, not just its hard predictions. The KD loss is a temperature-scaled KL divergence, and the $\tau^2$ factor must be included to keep gradient magnitude invariant to temperature.
    - Temperature $\tau > 1$ softens both distributions, exposing the teacher's "dark knowledge" (inter-token similarities). A value of 2–4 is typical; too high and the signal becomes noise.
    - Off-policy distillation is cheap but suffers from distribution mismatch. On-policy distillation trains on the student's own generations (evaluated by the teacher) and closes the gap for long, multi-step tasks; GKD (`trl`'s `GKDTrainer`) exposes both this choice (`lmbda`) and the divergence (`beta`) as continuous dials. Forward KL is mode-covering, reverse KL is mode-seeking — the bigger the capacity gap, the more you want reverse.
    - Token-level KD requires teacher and student to share a tokenizer, and caching *full* teacher logits is infeasible (64 KB/token at 32k vocab). Cache the top-k instead — $k=64$ costs 256 bytes/token, a ~250× reduction — and renormalize both distributions over that same support.
    - Sequence-level KD (SeqKD) uses the teacher's greedy output as hard training targets — a simple, cheap alternative to per-token KL that still captures teacher behavior.
    - Reasoning distillation (e.g., R1-style) works by collecting verified chain-of-thought traces from a large model and using them as SFT targets for a small model. The student learns *behavior*, not just output distributions.
    - SparseGPT and Wanda enable one-shot unstructured pruning of LLMs at 50%+ sparsity with near-zero perplexity degradation. Wanda's criterion (|w| × activation norm) requires no Hessian inversion and is extremely fast.
    - Speculative decoding's draft models are conceptually distilled students: a good draft model minimizes $\text{KL}(p_\text{target} \| p_\text{draft})$, and training the draft with KD from the target measurably improves acceptance rates.
    - Compression techniques stack: distillation → structured pruning → quantization → speculative decoding can take a 70B model to a practical on-device deployment. The quality at each step depends heavily on the ordering and the calibration data.
