# 6.3 TRL: HuggingFace's RL Library

TRL (Transformer Reinforcement Learning) is the most widely used open-source library for post-training large language models with reinforcement learning. Where frameworks like [veRL](../06-rl-infra/04-verl.html) or [OpenRLHF](../06-rl-infra/05-openrlhf-nemo-ray.html) target multi-node, throughput-optimized production runs, TRL occupies a different niche: it is the fastest path from a research idea to a working experiment, and the most accessible entry point for engineers who know the HuggingFace ecosystem.

This chapter dissects TRL from the inside out: its component trainers, how they compose with `accelerate`, PEFT, and vLLM, and — critically — what happens at the code level when you fire off a GRPO or DPO run. We give you enough detail to debug failures, tune performance, and adapt the library to custom reward functions.

## Why TRL Exists

Before TRL (released 2022 by Leandro von Werra and collaborators at HuggingFace), reproducing RLHF required stitching together a policy gradient loop, a KL penalty, reward model inference, reference model inference, and a PPO optimizer — all while handling variable-length sequences, packing, half-precision, and distributed training. The surface area for bugs was enormous.

TRL packages each stage of the alignment pipeline as a standalone `Trainer` subclass that inherits from HuggingFace `transformers.Trainer`. This means:

- All the `transformers` tooling (dataset loading, tokenizers, logging, evaluation hooks) works out of the box.
- Distributed training is handled by `accelerate` — TRL trainers are unaware of whether they run on one GPU or 64.
- PEFT (LoRA, QLoRA) integrates transparently; the trainer detects an adapter-wrapped model and handles merging for reference-model inference automatically.

The cost of this design is that TRL is not the fastest option for very large runs. The generation loop is colocated with training on the same GPUs (no disaggregated rollout workers), and throughput is limited by the sequential generate-then-train-step cycle. For the research and small-team use-case, this is exactly the right trade-off.

## TRL's Trainer Landscape

The classic view is five trainers spanning the alignment pipeline:


{{fig:trl-trainer-landscape-pipeline}}


Each trainer is independently usable; you do not need to run them all in sequence. Many modern recipes (DeepSeek-R1, Qwen3, etc.) skip the standalone reward model and go straight to GRPO with a verifiable reward function.

### What the v1 API surface actually looks like

TRL v1 (2026) drew a hard line between a small **stable** trainer suite and a much larger **experimental** namespace. Everything importable from the top-level `trl` package carries semantic-versioning guarantees:

```python
from trl import (
    SFTTrainer,      # supervised fine-tuning
    RewardTrainer,   # Bradley-Terry reward model
    DPOTrainer,      # offline preference optimization
    KTOTrainer,      # unpaired (thumbs up/down) preference optimization
    GRPOTrainer,     # critic-free online RL, group-relative advantages
    RLOOTrainer,     # critic-free online RL, leave-one-out baseline
)
```

Everything else — including **`PPOTrainer`** — now lives under `trl.experimental.*` and emits a `TRLExperimentalWarning` on import, signalling that its API may change without a deprecation cycle:

```python
from trl.experimental.ppo import PPOTrainer, PPOConfig      # classic actor-critic PPO
from trl.experimental.online_dpo import OnlineDPOTrainer    # on-policy DPO with a judge
from trl.experimental.gkd import GKDTrainer                 # generalized knowledge distillation
from trl.experimental.orpo import ORPOTrainer               # odds-ratio PO (SFT + preference in one)
from trl.experimental.prm import PRMTrainer                 # process reward models (step-level)
from trl.experimental.async_grpo import AsyncGRPOTrainer    # decoupled rollout/train workers
```

That reorganization is itself the headline fact about TRL in 2026: PPO with a learned critic is no longer the default path for LLM RL. The stable suite is SFT → (optional RM) → DPO/KTO offline, or GRPO/RLOO online with programmatic rewards.

Every stable trainer also has a first-class CLI wrapper — `trl sft`, `trl dpo`, `trl grpo`, `trl kto`, `trl rloo`, `trl reward`, plus `trl vllm-serve` and `trl env` — which shells out to `accelerate launch` under the hood, so every distributed option in the next section applies unchanged. Each subcommand accepts `--config recipe.yaml`, which is how open-r1 and most published recipes ship their hyperparameters. We use it end to end in the last section.

### SFTTrainer

`SFTTrainer` wraps `transformers.Trainer` with quality-of-life features for supervised fine-tuning:

- **Automatic sequence packing.** `packing=True` bin-packs examples into `max_length` chunks, eliminating padding waste. Modern TRL defaults to `packing_strategy="bfd"` (best-fit-decreasing: sort examples by length, place each into the fullest bin it still fits) rather than the older naive concatenation, and it emits `position_ids` so FlashAttention treats each packed example as its own sequence — no cross-contamination of attention between neighbours. Alternatives are `"bfd_split"` (split rather than truncate overflow) and `"wrapped"` (the old aggressive cut-anywhere behaviour). For a 2048-token context with typical 200-token instruction examples, packing can increase GPU utilization by 4–6x.
- **Padding-free batching.** `padding_free=True` is the middle road: instead of packing, it flattens the batch into one long unpadded sequence plus `position_ids`, again relying on FlashAttention's variable-length kernel. Use it when you want packing's efficiency without merging distinct examples into one training sample.
- **Chat template application.** Pass a `formatting_func` or set `dataset_text_field` and TRL handles tokenization. `chat_template_path` lets you swap in a template for a base model that ships without one.
- **PEFT integration.** Pass a `PeftConfig` (e.g., `LoraConfig`) and the trainer wraps the model automatically.
- **Loss masking.** `completion_only_loss=True` masks prompt tokens out of the loss on prompt-completion datasets (the default `None` infers this from the dataset shape). For multi-turn conversational data, `assistant_only_loss=True` uses the chat template's generation markers to train on *every* assistant turn while masking user turns.

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Load a model and tokenizer (e.g., Qwen-2.5-7B-Instruct)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype="auto",                # bfloat16 on Ampere+ (was `torch_dtype` before transformers v5)
    device_map="auto",           # naive tensor parallel across GPUs
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# LoRA config: rank-16 adapters on q_proj and v_proj only
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# SFT config inherits from TrainingArguments — all standard HF args apply
sft_config = SFTConfig(
    output_dir="./sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # effective batch = 4 * 8 * num_gpus
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_length=2048,                  # hard cap; packing fills bins to this
    packing=True,                     # bin-pack examples (see packing_strategy)
    packing_strategy="bfd",           # best-fit-decreasing (TRL default)
    dataset_text_field="text",        # column containing formatted text
)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model()          # saves merged model or adapter, depending on config
```

### RewardTrainer

`RewardTrainer` trains a scalar reward model from comparison pairs `(chosen, rejected)`. It uses the Bradley-Terry objective:

$$
\mathcal{L}_\text{BT} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]
$$

where $r_\phi(x, y)$ is the scalar reward head applied to the final token's hidden state. The underlying backbone is any causal or seq-to-seq model with a linear head added by TRL.

```python
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# A reward model is a classifier backbone — AutoModelForSequenceClassification
# adds a single linear head on top of the last hidden state.
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-7B",
    num_labels=1,         # single scalar output (the reward)
    dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer.pad_token = tokenizer.eos_token

reward_config = RewardConfig(
    output_dir="./reward-model",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    bf16=True,
    max_length=1024,
    gradient_checkpointing=True,  # saves ~50% memory; moderate speed hit
    center_rewards_coefficient=0.01,  # adds coef * mean((r_w + r_l)^2) to the loss,
                                      # pushing rewards mean-zero. BT only constrains
                                      # *differences*, so the additive offset is
                                      # otherwise unidentifiable and can drift.
)

# Dataset must have columns: "chosen" and "rejected" (formatted strings)
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

trainer = RewardTrainer(
    model=reward_model,
    args=reward_config,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()
```

## DPOTrainer: Direct Preference Optimization

DPO (Rafailov et al., 2023) sidesteps a separate reward model entirely. It reparameterizes the reward as the log-ratio between policy and reference model:

$$
r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)} + \text{const}
$$

and substitutes this into the Bradley-Terry loss to yield the closed-form DPO objective:

$$
\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right]
$$

TRL's `DPOTrainer` handles the tricky plumbing of keeping a frozen reference model in memory simultaneously with the training model, computing per-token log-probabilities for both, and applying the loss. For the theory and variants (SimPO, IPO, KTO, cDPO), see [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

### How DPOTrainer manages memory

When you pass a `model` to `DPOTrainer` without a `ref_model`, TRL clones the model's adapter weights (if PEFT) or does a deepcopy (if full fine-tuning) at trainer initialization and keeps it frozen. During forward, both passes can share the same GPU memory for the frozen backbone if LoRA is used — the base weights are identical; only the adapter delta changes.

Two `DPOConfig` knobs address the case where a second full copy will not fit:

- `precompute_ref_log_probs=True` runs the reference model over the entire training set *once* before training, caches $\log \pi_\text{ref}(y_w \mid x)$ and $\log \pi_\text{ref}(y_l \mid x)$, then frees the reference model from GPU memory. Since $\pi_\text{ref}$ is frozen, its log-probs never change — this is pure win for full fine-tuning, at the cost of a preprocessing pass (tune it with `precompute_ref_batch_size`). It is incompatible with a moving reference.
- `sync_ref_model=True` with `ref_model_sync_steps=N` and `ref_model_mixup_alpha=α` implements the TR-DPO moving reference: every `N` steps the reference weights are updated as $\pi_\text{ref} \leftarrow \alpha \pi_\theta + (1-\alpha)\pi_\text{ref}$, an EMA-style snapshot that lets the policy travel further from the initial SFT model without the KL term exploding.

```python
from trl import DPOConfig, DPOTrainer, LogCompletionsCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Policy model wrapped with LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16"
)
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")
model = get_peft_model(base_model, lora_config)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,                          # KL regularization temperature
    loss_type="sigmoid",               # standard DPO. Others: "ipo", "hinge", "robust",
                                       # "apo_zero", "apo_down", "sppo_hard", "discopop",
                                       # "sft". Pass a list + `loss_weights` to blend
                                       # (e.g. ["sigmoid", "sft"] for a DPO+SFT anchor).
    label_smoothing=0.0,               # >0 gives conservative DPO (cDPO)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True,
    max_length=2048,                   # max total len (prompt + completion)
    truncation_mode="keep_start",      # drop the tail, not the prompt, when over budget
    padding_free=True,                 # flatten batch + position_ids (needs FlashAttention)
    # Memory-saving: reuse base weights for reference model (LoRA only)
    # ref_model=None means TRL auto-creates from base weights
)

# Dataset with columns: prompt, chosen, rejected
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

trainer = DPOTrainer(
    model=model,
    ref_model=None,      # auto-derives reference from base (without LoRA adapters)
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# Eyeball actual generations during training. (The old `generate_during_eval`
# config flag was replaced by this callback, which logs a completions table
# to W&B / Trackio every `num_prompts` eval prompts.)
trainer.add_callback(
    LogCompletionsCallback(trainer, num_prompts=8, freq=100)
)
trainer.train()
```

!!! example "Worked Example: DPO loss computation"

    Suppose $\beta = 0.1$ and for a single (prompt, chosen, rejected) triple we compute the following per-sequence log-probabilities:

    | Quantity | Value |
    |---|---|
    | $\log \pi_\theta(y_w \mid x)$ | $-12.4$ |
    | $\log \pi_\text{ref}(y_w \mid x)$ | $-14.1$ |
    | $\log \pi_\theta(y_l \mid x)$ | $-10.8$ |
    | $\log \pi_\text{ref}(y_l \mid x)$ | $-10.5$ |

    The log-ratios are:

    $$\log \frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} = -12.4 - (-14.1) = +1.7$$

    $$\log \frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)} = -10.8 - (-10.5) = -0.3$$

    The implicit reward margin:

    $$\beta \cdot (1.7 - (-0.3)) = 0.1 \times 2.0 = 0.2$$

    The DPO loss for this example:

    $$\mathcal{L} = -\log \sigma(0.2) = -\log(0.5498) \approx 0.598$$

    This is a fairly high loss — the policy barely prefers the chosen response. After sufficient training steps you would expect the margin to grow toward 2–4 and the loss to drop toward 0.1–0.2.

## PPOTrainer: Online RL with a Reward Signal

PPO (Schulman et al., Proximal Policy Optimization, 2017) is the classical RL algorithm used in InstructGPT and the original RLHF pipeline. TRL's `PPOTrainer` implements the actor-critic loop adapted to language model sequences. For the full theory of policy gradients and PPO clipping, see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) and [The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html).

!!! warning "API churn: PPO has moved twice"

    Older tutorials show a manual loop built around `AutoModelForCausalLMWithValueHead`, `PPOTrainer(config=..., tokenizer=...)`, and an explicit `trainer.step(queries, responses, rewards)` call. That API was retired in TRL 0.12 and replaced by a `transformers.Trainer`-shaped `PPOTrainer` you drive with `.train()`. In TRL v1 it moved again — out of the stable namespace and into `trl.experimental.ppo`. Any snippet you find calling `trainer.step(...)` predates 2024 and will not run. Treat PPO as legacy: reach for `GRPOTrainer` or `RLOOTrainer` unless you specifically need a learned value function.

The TRL PPO loop at each iteration:

1. **Rollout.** Sample a batch of prompts; call `model.generate()` to produce completions.
2. **Reward scoring.** Pass `(prompt, completion)` pairs through the reward model.
3. **KL penalty.** Compute per-token KL divergence between the policy and a frozen reference model; subtract it from the reward.
4. **Advantage estimation.** Run a value head (a separate linear layer on top of the policy backbone) through the rollout to compute GAE (Generalized Advantage Estimation) advantages.
5. **PPO update.** Run $K$ mini-batch gradient steps with the clipped surrogate objective.

The modern (v1, experimental) shape wires four *separate* models — policy, reference, reward, and value — and then just calls `.train()`:

```python
# PPO now lives in the experimental namespace; importing it warns loudly.
from trl.experimental.ppo import PPOConfig, PPOTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_dataset

base = "EleutherAI/pythia-160m"          # small enough to run on one GPU
rm_path = "EleutherAI/pythia-160m"        # in practice: your RewardTrainer output

tokenizer = AutoTokenizer.from_pretrained(base, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Four models, each with a distinct role:
policy     = AutoModelForCausalLM.from_pretrained(base)                 # trained
ref_policy = AutoModelForCausalLM.from_pretrained(base)                 # frozen, for KL
# The critic and the reward model are both scalar-head classifiers.
# The value model IS trained (it is the critic); the reward model is frozen.
value_model  = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1)
reward_model = AutoModelForSequenceClassification.from_pretrained(rm_path, num_labels=1)

# Prompt-only dataset: PPO generates the completions itself.
dataset = load_dataset("trl-lib/tldr", split="train")

ppo_config = PPOConfig(
    output_dir="./ppo-output",
    learning_rate=3e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    total_episodes=10_000,           # rollouts to collect over the whole run
    num_ppo_epochs=4,                # inner passes over each rollout batch (K)
    num_mini_batches=1,              # splits of the batch per inner epoch
    response_length=53,              # generated tokens per rollout
    kl_coef=0.05,                    # fixed per-token KL penalty vs. ref_policy
    kl_estimator="k1",               # k1 = logp_ref - logp_policy; "k3" is lower-variance
    cliprange=0.2,                   # PPO policy clip epsilon
    cliprange_value=0.2,             # value-function clip
    vf_coef=0.1,                     # value-loss weight
    gamma=1.0, lam=0.95,             # GAE discount / trace decay
    whiten_rewards=False,            # optional batch-level reward normalization
    missing_eos_penalty=1.0,         # subtract from reward if EOS was never emitted
    local_rollout_forward_batch_size=8,
)

trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,     # pass None when using PEFT: adapters get disabled instead
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset,
)
trainer.train()
```

The `missing_eos_penalty` knob is worth internalizing: without it, a policy learns that truncated, run-on completions are never penalized, because the reward model only ever sees a cut-off string. It is the simplest instance of a general rule — *every* degenerate behaviour your reward function cannot see, it implicitly rewards.

### PPOTrainer's memory footprint

PPO is expensive: you hold *four* models simultaneously — the trained policy, the trained critic (`value_model`, itself a full backbone plus scalar head, not just a head bolted onto the policy), the frozen reference, and the frozen reward model — plus optimizer states for the two trainable ones. For a 7B policy in bf16:
- Policy + optimizer states (Adam): roughly $7 \times 10^9 \times 2 + 7 \times 10^9 \times 8 = 70$ GB
- Reference model (inference-only, bf16): ~14 GB
- Reward model + critic: another ~14 GB each in bf16, and the critic carries its own Adam states
- Activations and rollout buffer: varies

Total can easily exceed 100 GB for a 7B model, requiring at least two A100-80GB cards. This cost motivated the GRPO and DPO approaches that eliminate the critic (and, with verifiable rewards, the reward model too — leaving a single trainable model).

{{fig:rl-trainer-memory-footprint}}

## GRPOTrainer: The DeepSeek-R1 Recipe

GRPO (Group Relative Policy Optimization, Shao et al., 2024) eliminates the value function entirely. Instead of estimating the advantage for each response with a critic, it generates a group of $G$ responses to the same prompt and uses the group's mean reward as a baseline:

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j
$$

{{fig:grpo-group-relative-advantage}}

The policy gradient objective is then:

$$
\mathcal{L}_\text{GRPO} = -\mathbb{E}\!\left[\sum_{i=1}^{G} \min\!\left(\rho_i A_i,\; \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i\right) - \beta \mathbb{D}_\text{KL}[\pi_\theta \| \pi_\text{ref}]\right]
$$

where $\rho_i = \pi_\theta(y_i \mid x) / \pi_{\text{old}}(y_i \mid x)$ is the importance ratio for response $i$. For the complete derivation and comparisons with RLOO, see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html).

### Configuring and running GRPOTrainer

TRL's `GRPOTrainer` is the most important trainer for reasoning-focused alignment. Here is a fully annotated, runnable example for a math reasoning task using a verifiable reward function.

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig
import re

# ----------------------------------------------------------------
# 1. Load model and tokenizer
# ----------------------------------------------------------------
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",           # bfloat16 on Ampere+
    attn_implementation="flash_attention_2",  # requires flash-attn installed
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ----------------------------------------------------------------
# 2. Reward function: verifiable math reward
#    TRL calls every reward function with KEYWORD arguments:
#      f(prompts=..., completions=..., completion_ids=..., trainer_state=...,
#        **every_other_dataset_column)
#    so parameter names matter. Always accept **kwargs. Return one float per
#    completion — or None for a row you want excluded from the loss entirely
#    (e.g. unparseable ground truth), which is different from returning 0.0.
# ----------------------------------------------------------------
def extract_boxed_answer(text: str) -> str | None:
    """Parse LaTeX \boxed{...} from model output."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None

def math_reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Reward +1.0 if the boxed answer matches ground truth, else 0.0.
    kwargs may contain extra dataset columns (e.g., 'answer').
    """
    ground_truths = kwargs.get("answer", [""] * len(completions))
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        pred = extract_boxed_answer(completion)
        if pred is not None and pred.strip() == str(gt).strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# ----------------------------------------------------------------
# 3. Build a toy dataset (replace with GSM8K / MATH in practice)
# ----------------------------------------------------------------
raw_data = [
    {"prompt": "What is 23 + 47?", "answer": "70"},
    {"prompt": "What is 15 * 8?",  "answer": "120"},
    {"prompt": "What is 144 / 12?", "answer": "12"},
    {"prompt": "What is 2^10?",    "answer": "1024"},
]
# TRL expects the dataset to have a "prompt" column at minimum.
# Extra columns are forwarded to the reward function as kwargs.
dataset = Dataset.from_list(raw_data * 250)   # repeat to simulate real dataset

# ----------------------------------------------------------------
# 4. GRPO configuration
# ----------------------------------------------------------------
grpo_config = GRPOConfig(
    output_dir="./grpo-math",
    # --- Group sampling ---
    num_generations=8,           # G: completions per prompt (more = stabler baseline)
    # --- Sampling params (config fields, NOT generate() kwargs) ---
    max_completion_length=512,   # cap on generated tokens (there is no `max_new_tokens`)
    temperature=0.9,             # diversity in rollouts; sampling is always on
    top_p=1.0,
    # --- Training ---
    # NOTE: per_device_train_batch_size counts *completions* per device, not prompts.
    # TRL requires num_processes * per_device_train_batch_size *
    # gradient_accumulation_steps to be divisible by num_generations, so that every
    # group stays intact within one optimizer step.
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-6,              # small LR is critical for stability
    bf16=True,
    # --- Objective ---
    epsilon=0.2,                 # PPO clip ratio ε (lower bound)
    epsilon_high=0.28,           # DAPO's asymmetric upper clip: more room to raise
                                 # the probability of rare-but-correct tokens
    beta=0.0,                    # KL coefficient. TRL's default is 0.0, which skips
                                 # loading the reference model entirely — standard for
                                 # verifiable-reward RL. Set ~0.001-0.04 to keep a leash.
    loss_type="dapo",            # token-level normalization over the whole batch.
                                 # "grpo" normalizes per sequence and is length-biased;
                                 # "dr_grpo" uses a constant normalizer (Dr. GRPO).
    scale_rewards="group",       # divide advantages by within-group std. "none" is the
                                 # Dr. GRPO recommendation (removes difficulty bias).
    mask_truncated_completions=True,   # DAPO: don't penalize answers that merely ran
                                       # out of budget — they are noise, not errors.
    num_iterations=1,            # μ: gradient passes per generation batch. >1 makes the
                                 # update genuinely off-policy (and the clip term active).
    # --- Logging ---
    logging_steps=5,
    save_steps=50,
    log_completions=True,        # dump sample rollouts to the console / W&B table
    # --- vLLM backend for faster generation (optional, see below) ---
    use_vllm=False,              # set True if vllm installed; see §Integration
)

# ----------------------------------------------------------------
# 5. Optional: LoRA to reduce GPU memory
# ----------------------------------------------------------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

# ----------------------------------------------------------------
# 6. Train
# ----------------------------------------------------------------
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=math_reward_fn,    # can also pass a list for composite rewards
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./grpo-math-final")
```

!!! tip "Practitioner tip"

    Set `num_generations` to at least 8 for stable advantage estimates. With only 4 samples, the within-group variance is high enough to produce noisy gradients (TRL rejects `num_generations < 2` outright, since a group of one has zero advantage by construction).

    The batch arithmetic is the single most common source of `ValueError`s in GRPO. TRL derives

    ```text
    generation_batch_size = per_device_train_batch_size × num_processes × steps_per_generation
    steps_per_generation  = gradient_accumulation_steps        (unless you set it)
    ```

    and then requires `generation_batch_size % num_generations == 0`, because a generation batch must contain *whole* groups — a half-group has no valid baseline. Note that `generation_batch_size` is counted in **completions**, so the number of unique prompts per generation batch is `generation_batch_size / num_generations`. (`auto_find_batch_size` is rejected for the same reason: halving the batch on OOM would split groups.)

    To shrink memory, lower `per_device_train_batch_size` to a value that still keeps the product divisible by `num_generations`, or raise `steps_per_generation` above `gradient_accumulation_steps` — the latter keeps the same number of prompts per *generation* while spreading the backward pass over more micro-steps.

### The GRPO training loop internals

When `trainer.train()` runs, TRL executes the following per-step logic:


{{fig:trl-grpo-train-loop-steps}}


The most expensive step is step 1: generating $P \times G$ completions sequentially on the same GPUs doing training. This is the core throughput bottleneck; the vLLM backend (§6) addresses it.

One step in that diagram is now conditional. With TRL's default `beta=0.0` the reference-model forward pass and the KL term are **skipped entirely**: `π_ref` is never loaded, saving a full model's worth of weights and one forward pass per token. This is not a shortcut — it is what DeepSeek-R1-Zero, DAPO, and most 2025–2026 RLVR recipes actually do. The KL leash exists to stop a policy from drifting into regions where a *learned* reward model is miscalibrated; a verifiable checker (does the answer equal 70? does the test suite pass?) has no such off-distribution failure mode, so the leash mostly just slows learning. Keep `beta > 0` when your reward comes from a neural reward model or a judge; drop it to zero when it comes from a verifier.

!!! tip "GRPO at 100M scale"

    Everything above is written for 7B models, but `GRPOTrainer` is if anything *easier* at the capstone's scale. For a ~100M-parameter policy in bf16 the whole model is ~0.2 GB, a reference copy is another ~0.2 GB, and full fine-tuning is cheaper and better-behaved than LoRA (there is no memory pressure to relieve, and low-rank updates only slow adaptation). You can afford `num_generations=16` and `num_iterations=1` on a single consumer GPU. The binding constraint flips from memory to *reward density*: a 100M model solves so few open-ended math problems that most groups come back all-zero and contribute nothing (watch `frac_reward_zero_std`). The capstone therefore narrows the task until the base model's pass rate sits in a usable band — see [Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M](../14-capstone/09-post-training.html).

## Integration with Accelerate, PEFT, and vLLM

### Accelerate

All TRL trainers run under `accelerate`'s process group. You launch distributed runs as:

```bash
# 4-GPU DDP training
accelerate launch --num_processes 4 train_grpo.py

# With a config file (recommended for multi-node)
accelerate config   # generates ~/.cache/huggingface/accelerate/default_config.yaml
accelerate launch --config_file my_accelerate.yaml train_grpo.py
```

A typical `accelerate` config for GRPO on 8 H100s:

```yaml
# my_accelerate.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP           # or DEEPSPEED
num_processes: 8
mixed_precision: bf16
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1       # FULL_SHARD (ZeRO-3 equivalent)
  fsdp_state_dict_type: FULL_STATE_DICT
```

With FSDP, model shards are split across GPUs during the forward/backward pass and re-gathered as needed — see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) for the mechanics. TRL's trainers call `accelerate.prepare()` on the model, optimizer, and dataloader internally.

### PEFT / LoRA

When a `peft_config` is passed to any TRL trainer:

1. The trainer calls `get_peft_model(model, peft_config)` and trains only the adapter weights.
2. For DPO and GRPO, the reference model is derived from the base model **without** the adapters: `model.disable_adapter_layers()`. This is memory-free because adapter and base weights share the same GPU tensors; only the adapter delta is excluded from the reference forward pass.
3. At the end of training, `trainer.save_model()` saves only the adapter weights (a few hundred MB for rank-64 LoRA on a 7B model). Optionally merge with `model.merge_and_unload()`.

For QLoRA (4-bit base + LoRA), pass a `BitsAndBytesConfig`:

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,   # nested quantization; saves ~0.4 GB per 7B
    bnb_4bit_quant_type="nf4",        # NormalFloat4 distribution
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)  # cast norms to fp32, enable grad ckpt
```

See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) for the mathematical foundation of LoRA and the memory analysis of QLoRA.

### vLLM backend for generation

The primary bottleneck in GRPO (and PPO) is `model.generate()`, which is a batch-of-one-token-at-a-time loop with no paged KV cache and no continuous batching. TRL integrates vLLM as the generation backend, in **two modes** selected by `vllm_mode`:

- `"server"` — a separate `trl vllm-serve` process owns its own GPUs; the trainer is an HTTP client. Clean isolation, but the training GPUs sit idle during generation and the server GPUs sit idle during the backward pass.
- `"colocate"` (the default) — vLLM runs inside the training process and shares the same GPUs, capped at `vllm_gpu_memory_utilization` (default 0.3). No GPU is ever idle; this is the "no GPU left behind" configuration, reported by HuggingFace at roughly 1.3–1.7x wall-clock on large enough models.

```python
# Mode A: separate server. First, in another terminal:
#   trl vllm-serve --model Qwen/Qwen2.5-7B-Instruct \
#                  --tensor-parallel-size 2 --data-parallel-size 2 \
#                  --port 8000 --gpu-memory-utilization 0.9
grpo_config = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",
    vllm_server_host="localhost",    # host running the vLLM inference server
    vllm_server_port=8000,
    vllm_server_timeout=120,         # seconds to wait for server readiness
    # vLLM generation kwargs
    temperature=1.0,
    top_p=0.95,
)

# Mode B: colocated in the training process — no server to launch.
grpo_config = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,  # fraction of each GPU handed to the vLLM engine
                                      # (its weight copy + KV cache); the rest stays
                                      # with the trainer's weights/optimizer/activations
    vllm_tensor_parallel_size=1,      # must divide the training world size
    vllm_enable_sleep_mode=True,      # offload vLLM weights/KV between rollout phases
    temperature=1.0,
    top_p=0.95,
)
```

The workflow with vLLM:


{{fig:trl-vllm-colocate-rollout-flow}}


**Weight synchronization.** After each optimizer step the policy in vLLM is stale, so TRL pushes the new weights before the next rollout. In server mode this goes through `VLLMClient`: the trainer and the server first build a side-channel process group (`init_communicator`, a `StatelessProcessGroup` wrapping a `PyNcclCommunicator`), and then `update_model_params(model)` walks `model.named_parameters()` and NCCL-broadcasts each tensor GPU-to-GPU — the weights never touch the HTTP body or host RAM. In colocate mode there is no network hop at all: TRL calls vLLM's `collective_rpc` to load the tensors directly into the resident engine. Under FSDP/ZeRO-3 the parameters must be re-gathered shard-by-shard before broadcast, which is why `ds3_gather_for_generation` exists.

!!! warning "The rollout–training log-prob mismatch"

    vLLM and the training forward pass compute *different* log-probabilities for the very same tokens. They use different kernels, different batching, and different reduction orders, so bf16 rounding diverges — and vLLM may be running a quantized or differently-fused path entirely. GRPO's importance ratio $\rho_i = \pi_\theta / \pi_\text{old}$ silently assumes $\pi_\text{old}$ *is* the sampler, so this discrepancy injects bias and, at scale, causes runs to collapse after thousands of steps.

    TRL corrects for it explicitly: `vllm_importance_sampling_correction=True` (on by default) multiplies each token's loss by $\exp\!\left(\log \pi_\text{train}(y_t) - \log \pi_\text{vLLM}(y_t)\right)$ — the ratio between the log-prob the trainer recomputes and the one vLLM actually sampled with — clamped to `[vllm_importance_sampling_clip_min, vllm_importance_sampling_clip_max]`. This is truncated importance sampling (TIS). Watch `sampling/sampling_logp_difference/mean` — if it grows over training, your inference and training stacks have drifted apart and the run is on borrowed time. The general problem is covered in [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html).

For a deep dive on the PagedAttention mechanism powering vLLM's generation, see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html). Note that the diagram's overlap of generation with back-propagation is *not* what stock synchronous GRPO does — each step still waits for all rollouts. True overlap requires `steps_per_generation > gradient_accumulation_steps` (train on rollouts that are one step stale) or the decoupled `trl.experimental.async_grpo` workers; see [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html).

## Practical Configuration Reference

The key numerical hyperparameters and their typical ranges:

| Parameter | DPO | GRPO | PPO | Notes |
|---|---|---|---|---|
| `beta` (KL coefficient) | 0.01–0.5 | 0 (verifiable) / 0.001–0.04 (RM) | `kl_coef` ≈ 0.05 | Higher = stay closer to reference |
| `learning_rate` | 1e-5–5e-5 | 5e-7–2e-6 | 1e-6–3e-6 | GRPO/PPO need very small LR |
| `num_generations` (G) | — | 4–16 | — | More = stable baseline, more memory |
| `epsilon` / `epsilon_high` | — | 0.2 / 0.28 | `cliprange` 0.1–0.2 | DAPO decouples the two bounds |
| `max_completion_length` | — | 256–8192 | `response_length` 53–512 | Task-dependent |
| `num_iterations` (μ) / `num_ppo_epochs` | — | 1 | 1–4 | Reuse ratio; 1 is safer |
| Effective batch size | 32–128 | 64–512 completions | 64–256 | must be divisible by G for GRPO |

!!! warning "Common pitfall"

    The classic advice — "`beta` too low means reward hacking" — is right for *learned* rewards and wrong for verifiable ones. With a reward model or an LLM judge, a weak KL leash lets the policy walk off-distribution into the region where the reward model is miscalibrated and score beautifully on garbage; monitor `kl` and stop if it exceeds roughly 20 nats. With a verifier (exact-match, unit tests, a compiler) there is no off-distribution failure mode to exploit, and `beta=0` — TRL's default — is standard practice.

    What replaces the KL leash in the verifiable setting is *reward-surface* hygiene: watch `completions/mean_length` for length exploitation, `entropy` for mode collapse, and read actual rollouts (`log_completions=True`). Models reliably discover that `\boxed{}` matching can be satisfied by emitting every plausible answer, or that a test suite passes if you monkey-patch the test. See [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html) for detailed diagnostics.

### Monitoring training health

TRL logs a rich set of metrics to Weights & Biases or TensorBoard:

```text
# DPO metrics (prefixed train/ or eval/)
rewards/chosen        # beta * log-ratio for chosen responses (the implicit reward)
rewards/rejected
rewards/margins       # chosen - rejected; should grow steadily
rewards/accuracies    # fraction of examples where chosen reward > rejected reward
logps/chosen          # mean log-probability of chosen completions
logits/chosen         # mean logit; a runaway value signals numerical trouble

# GRPO metrics
reward                          # mean reward across all rollouts
reward_std                      # std of rewards within each group
rewards/<func_name>/mean        # one series PER reward function — indispensable for
                                # composite rewards (is format or correctness moving?)
frac_reward_zero_std            # fraction of groups with ZERO within-group variance,
                                # i.e. all-correct or all-wrong -> no gradient at all
completions/mean_length         # mean completion length (watch length exploitation)
completions/clipped_ratio       # fraction hitting max_completion_length
entropy                         # policy entropy; a collapse to ~0 means mode collapse
kl                              # mean KL penalty (only logged when beta > 0)
clip_ratio/low_mean             # fraction of tokens clipped at the lower bound
clip_ratio/high_mean            # ... and at the upper bound (epsilon_high)
sampling/importance_sampling_ratio/mean   # vLLM-vs-training correction (see above)
```

A healthy GRPO run shows `reward` trending up, `entropy` decaying slowly rather than crashing, `clip_ratio/*` around 0.1–0.2, and `kl` below 10–15 when it is enabled at all. Two metrics deserve more attention than they usually get:

- **`frac_reward_zero_std`** is your rollout-efficiency gauge. Every group it counts is compute you paid for and received no gradient from (Exercise 4 works through why). If it sits above ~0.5, your dataset is mostly too easy or too hard for the current policy and the fix is curriculum/filtering, not hyperparameters — see [RL Data, Curriculum & Replay Management](../06-rl-infra/12-rl-data-curriculum-replay.html).
- **`completions/clipped_ratio`** rising alongside `completions/mean_length` is the classic length-exploitation signature. Add an explicit length penalty, or set `mask_truncated_completions=True` so truncated rollouts stop contributing noise to the loss.

## Strengths and Limits of TRL

### Where TRL excels

- **Ease of use.** A working GRPO experiment requires fewer than 100 lines of Python. The HuggingFace ecosystem (datasets, tokenizers, model hub, PEFT) integrates without boilerplate.
- **Breadth.** A single library covers the entire alignment pipeline: SFT → RM → PPO/DPO/GRPO. You can ablate algorithms quickly by swapping trainer classes.
- **PEFT integration.** QLoRA + GRPO on a single A100-40GB is routine; the reference model is free.
- **Custom reward functions.** Passing a Python callable means any verifiable reward — code execution, math verification, format checking — works without infrastructure overhead.
- **Community and maintenance.** As part of the HuggingFace organization, TRL receives rapid fixes, new algorithms (SimPO, KTO, ORPO were all added within months of publication), and broad community support.

### Where TRL hits limits

- **Single-node generation bottleneck.** Without the vLLM backend, generation runs through `model.generate()` on the training GPUs — no paged KV cache, no continuous batching. For a 70B model, rollout latency dwarfs training compute.
- **Only one rollout fleet, and no placement control.** `trl vllm-serve` gives you *a* disaggregated generation server, but not the general resource-placement model of [veRL](../06-rl-infra/04-verl.html)'s single controller or [OpenRLHF](../06-rl-infra/05-openrlhf-nemo-ray.html)'s Ray actors, where policy, critic, reward, and rollout each get an independently sized, independently parallelized placement group.
- **Memory for PPO.** Four concurrent models (policy, critic, reference, reward). At 70B, PPO requires a large cluster even with FSDP; GRPO is much more practical — and PPO is now experimental in TRL anyway.
- **Synchronous by default.** Each training step waits for the slowest rollout in the batch, so one 8k-token generation stalls hundreds of short ones (the long-tail straggler problem). `trl.experimental.async_grpo` decouples rollout and training workers, but it is experimental; [Prime-RL](../06-rl-infra/06-prime-rl-async.html) treats async as the design centre rather than an add-on.
- **Sequence packing in RL.** Packing works beautifully for SFT but is harder in GRPO/PPO because each rollout in a group must be associated with its prompt for advantage computation. TRL handles this but the implementation is more complex than SFT packing.

!!! interview "Interview Corner"

    **Q:** A colleague proposes using TRL's `GRPOTrainer` with `num_generations=16` to train a 13B reasoning model on GSM8K. You have 4 × A100-80GB GPUs. What bottlenecks do you anticipate, and how would you address them?

    **A:** Three main bottlenecks arise. First, **memory**: 13B in bf16 is ~26 GB; with LoRA adapters the model fits on one GPU, and setting `beta=0.0` (the default) means no reference model is loaded at all — appropriate here, since GSM8K's reward is a verifier, not a learned RM. Enable `gradient_checkpointing=True` to halve activation memory. Second, **generation throughput**: generating 16 completions per prompt is the dominant wall-clock cost. Enable `use_vllm=True`; on only 4 GPUs prefer `vllm_mode="colocate"` (share all 4, `vllm_gpu_memory_utilization≈0.3`) over `"server"`, which would idle whole GPUs on each side of the loop. Third, **batch arithmetic**: `generation_batch_size = per_device_train_batch_size × 4 × steps_per_generation` counts *completions* and must be divisible by 16, so `per_device_train_batch_size=4` with `gradient_accumulation_steps=8` gives 128 completions = 8 prompts per optimizer step — thin on prompt diversity. Raise `steps_per_generation` to 32 for 512 completions (32 prompts) per step instead of shrinking the group. Monitor `frac_reward_zero_std` (GSM8K rows a 13B model always or never solves are pure waste) and `clip_ratio/high_mean` (target 0.1–0.2).

## Building a Custom Reward Function Pipeline

One of TRL's most powerful features is that reward functions are plain Python. Before writing your own, check `trl.rewards` — the library ships the standard RLVR components, already hardened against the parsing edge cases that eat an afternoon:

```python
from trl.rewards import (
    accuracy_reward,               # math-verify equivalence check vs. a `solution` column;
                                   # returns None (not 0.0) when the ground truth is
                                   # unparseable, so bad rows are excluded, not punished
    think_format_reward,           # 1.0 iff the completion is wrapped in <think>...</think>
    get_soft_overlong_punishment,  # DAPO's soft length penalty (factory -> callable)
    get_repetition_penalty_reward, # penalizes repeated n-grams (degenerate loops)
    get_cosine_scaled_reward,      # scales correctness by length on a cosine schedule:
                                   # short correct answers > long correct answers
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,   # needs a "solution" column for accuracy_reward
    reward_funcs=[
        accuracy_reward,
        think_format_reward,
        get_soft_overlong_punishment(max_completion_len=512, soft_punish_cache=64),
        get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5),
    ],
    # Weighted sum, one weight per function (default: all 1.0)
    processing_class=tokenizer,
)
grpo_config.reward_weights = [1.0, 0.2, 1.0, 1.0]
```

Note the factory pattern: functions that need configuration (`get_*`) return a closure, because `reward_funcs` entries are called as `f(prompts=..., completions=..., **dataset_columns)` with no room for extra arguments. You can also pass a *model* (or a Hub model id) in `reward_funcs`, in which case TRL scores completions with that sequence classifier — this is how you mix a learned reward model with programmatic checks in one run.

Writing your own is still the common case. Here is a production-grade reward function that combines a format reward with a correctness reward, plus a length penalty:

```python
import re
from typing import Any

def composite_reward(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """
    Composite reward for math reasoning:
    - +0.5 if completion contains a <think>...</think> block (format reward)
    - +1.0 if the boxed answer matches the ground truth (correctness reward)
    - -0.3 penalty if completion exceeds 800 tokens (length penalty)

    This mirrors the structure used in DeepSeek-R1-Zero and Qwen-QwQ.
    """
    ground_truths = kwargs.get("answer", [""] * len(completions))
    rewards = []

    for completion, gt in zip(completions, ground_truths):
        r = 0.0

        # --- Format reward ---
        has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        if has_think:
            r += 0.5

        # --- Correctness reward ---
        pred = extract_boxed_answer(completion)
        if pred is not None and pred.strip() == str(gt).strip():
            r += 1.0

        # --- Length penalty (rough token count via whitespace split) ---
        if len(completion.split()) > 800:
            r -= 0.3

        rewards.append(r)

    return rewards


def extract_boxed_answer(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None
```

You can pass a list of reward functions to `GRPOTrainer`; TRL sums them with optional per-function weights:

```python
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    # List of reward functions — TRL sums their outputs
    reward_funcs=[composite_reward],
    processing_class=tokenizer,
    peft_config=lora_config,
)
```

For tasks requiring code execution or external API calls, the reward function runs in the same Python process as training. For sandboxed execution (preventing the model from writing malicious code that runs during training), use a subprocess or a containerized reward server — see [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html) for safe execution patterns.

## End-to-End: Reproducing a DeepSeek-R1-Style Run

The minimal recipe to reproduce the RLVR reasoning alignment approach from DeepSeek-R1-Zero with TRL:

```bash
# 1. Install dependencies (math-verify powers trl.rewards.accuracy_reward)
pip install "trl[vllm]" transformers peft accelerate flash-attn math-verify

# 2. Sanity-check the environment before burning GPU hours
trl env          # prints trl/transformers/accelerate/peft/vllm/torch versions + GPU info

# 3. Terminal A — dedicate GPUs 4-7 to the rollout server
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve \
  --model Qwen/Qwen2.5-7B \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --port 8000

# 4. Terminal B — train on GPUs 0-3 using TRL's own GRPO script
CUDA_VISIBLE_DEVICES=0,1,2,3 trl grpo \
  --model_name_or_path Qwen/Qwen2.5-7B \
  --dataset_name trl-lib/DeepMath-103K \
  --reward_funcs accuracy_reward think_format_reward \
  --num_generations 8 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_completion_length 2048 \
  --beta 0.0 \
  --learning_rate 1e-6 \
  --num_train_epochs 1 \
  --bf16 \
  --use_vllm --vllm_mode server \
  --vllm_server_host localhost --vllm_server_port 8000 \
  --log_completions \
  --output_dir ./grpo-r1-repro
```

`trl grpo` is `trl/scripts/grpo.py`, which is essentially the `GRPOTrainer` example from §4 wired to `TrlParser`. `--reward_funcs` takes names from a small built-in registry (`accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`, `get_soft_overlong_punishment`) **or any dotted import path** — `--reward_funcs my_rewards.composite_reward` imports from the current working directory, so your own callable from the previous section drops straight into the CLI with no script to copy. Add `--config recipe.yaml` and you have exactly the shape of an [open-r1](https://github.com/huggingface/open-r1) recipe. The RLVR recipe and its relation to chain-of-thought emergence are covered in depth in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html); the scaled-down version that actually fits the capstone budget is in [Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M](../14-capstone/09-post-training.html).

## Key Takeaways

!!! key "Key Takeaways"

    - TRL v1's **stable** suite is `SFTTrainer`, `RewardTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer`, and `RLOOTrainer`; everything else — including `PPOTrainer`, ORPO, Online DPO, GKD, and PRM — now lives in `trl.experimental.*` with no API guarantees. Each stable trainer also has a CLI (`trl sft`, `trl grpo`, …) plus `trl vllm-serve` and `trl env`.
    - All trainers inherit from `transformers.Trainer` and use `accelerate` for distribution — any `accelerate` backend (FSDP, DeepSpeed, DDP) works without trainer-level code changes.
    - PEFT (LoRA, QLoRA) integrates transparently: the reference model shares the base backbone with the policy, making the reference model nearly memory-free for DPO and GRPO. For full fine-tuning, `precompute_ref_log_probs=True` frees the reference model outright.
    - `GRPOTrainer` is the recommended entry point for reasoning alignment (replacing PPO): no critic, group-relative advantages, and verifiable reward functions as plain Python callables — with `trl.rewards` shipping the standard ones.
    - Modern GRPO defaults encode hard-won lessons: `beta=0.0` (no reference model at all with verifiable rewards), `loss_type="dapo"` (token-level normalization, no length bias), `epsilon_high=0.28` (asymmetric clipping), and `mask_truncated_completions=True`.
    - The batch arithmetic is the top footgun: `generation_batch_size = per_device_train_batch_size × num_processes × steps_per_generation`, counted in **completions**, and it must be divisible by `num_generations` so every group stays whole.
    - The generation bottleneck is TRL's main throughput limitation; `use_vllm=True` with `vllm_mode="colocate"` shares the training GPUs with a vLLM engine for roughly 1.3–1.7x wall-clock speedup, syncing weights by NCCL broadcast — and `vllm_importance_sampling_correction` compensates for the sampler-vs-trainer log-prob mismatch.
    - Monitor `reward`, `frac_reward_zero_std` (wasted rollouts), `entropy` (mode collapse), and `completions/mean_length` (length exploitation); `kl` only exists when `beta > 0`.
    - TRL is the fastest path from research paper to running experiment; for production-scale multi-node runs (70B+), consider veRL or OpenRLHF which offer general resource placement and higher throughput.

!!! sota "State of the Art & Resources (2026)"
    TRL has matured into the standard entry point for LLM post-training, with v1.0 (March 2026) stabilising its CLI, config system, and trainer suite. The v1 line draws a hard boundary: `SFTTrainer`, `RewardTrainer`, `DPOTrainer`, `KTOTrainer`, `GRPOTrainer`, and `RLOOTrainer` are stable and semver-guaranteed, while `PPOTrainer` and the long tail of preference algorithms moved to `trl.experimental.*` — a clean statement that critic-free RL has won for LLMs. `GRPOTrainer` with co-located vLLM is the dominant single-node recipe for reasoning alignment, and its defaults now bake in DAPO-style loss normalization and asymmetric clipping; disaggregated frameworks (veRL, OpenRLHF) still handle the 70B+ regime.

    **Foundational work**

    - [Ziegler et al., *Fine-Tuning Language Models from Human Preferences* (2019)](https://arxiv.org/abs/1909.08593) — the original RLHF paper combining reward models and PPO for language, the conceptual root of TRL's pipeline.
    - [Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347) — the clipped surrogate objective underlying TRL's `PPOTrainer` and GRPO's importance-ratio update.
    - [Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023)](https://arxiv.org/abs/2305.18290) — derives the closed-form DPO loss that TRL's `DPOTrainer` implements, eliminating the need for a separate reward model.

    **Recent advances (2023–2026)**

    - [Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024)](https://arxiv.org/abs/2402.03300) — introduces GRPO (group-relative advantage, no critic), the algorithm behind TRL's `GRPOTrainer`.
    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — demonstrates RLVR with verifiable rewards using GRPO at scale; the defining blueprint for TRL's reasoning alignment use-case.
    - [*DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — source of TRL's default `loss_type="dapo"` (token-level loss normalization), the asymmetric `epsilon_high=0.28` clip, `mask_truncated_completions`, and the soft overlong penalty in `trl.rewards`.
    - [*Understanding R1-Zero-Like Training: A Critical Perspective* (2025)](https://arxiv.org/abs/2503.20783) — the "Dr. GRPO" analysis of length and difficulty bias in the standard GRPO estimator; the reason `scale_rewards` and `loss_type` are configurable at all.
    - [*Group Sequence Policy Optimization* (2025)](https://arxiv.org/abs/2507.18071) — motivates TRL's `importance_sampling_level="sequence"`: sequence-level ratios are better matched to sequence-level rewards and markedly more stable than per-token ones.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — the TRL library itself: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `PPOTrainer`, `RewardTrainer`, and more, all built on Transformers + Accelerate.
    - [huggingface/open-r1](https://github.com/huggingface/open-r1) — fully open reproduction of the DeepSeek-R1 training pipeline using TRL's GRPO, including datasets and training scripts.

    **Go deeper**

    - [TRL v1 blog post — *Post-Training Library Built to Move with the Field* (2026)](https://huggingface.co/blog/trl-v1) — covers the v1.0 redesign: unified CLI, config system, and the expanded trainer suite.
    - [HuggingFace Blog, *No GPU Left Behind: Co-located vLLM in TRL* (2025)](https://huggingface.co/blog/vllm-colocate) — explains the `vllm_mode="colocate"` feature that embeds generation inside the training process for 1.3–1.7× wall-clock speedup.
    - [HuggingFace Cookbook, *Post-training an LLM for Reasoning with GRPO in TRL*](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl) — end-to-end notebook for reasoning fine-tuning with verifiable math rewards, mirroring the DeepSeek-R1-Zero recipe.

## Further Reading

- **Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019)** — the original RLHF paper using reward models and PPO with language models; foundational for understanding why the pipeline exists.
- **Schulman et al., "Proximal Policy Optimization Algorithms" (2017)** — the PPO algorithm underlying TRL's `PPOTrainer`; essential reading for the clipped surrogate objective.
- **Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)** — derives DPO and is required context for `DPOTrainer`.
- **Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)** — introduces GRPO and the group-relative advantage estimation used in `GRPOTrainer`.
- **DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)** — demonstrates the RLVR recipe with verifiable rewards; the most influential recent application of GRPO.
- **TRL GitHub repository (HuggingFace/trl)** — source code, examples, and trainer documentation; the `examples/` directory contains complete training scripts for each trainer.
- **HuggingFace Accelerate documentation** — for multi-GPU/multi-node launch configurations used with all TRL trainers.

## Exercises

**1.** When you pass a `peft_config` (LoRA) to `DPOTrainer` or `GRPOTrainer` and set `ref_model=None`, the chapter claims the reference model is "nearly memory-free." Explain *why* this is true for LoRA but would **not** be true if you were doing full fine-tuning. What does TRL do at the code level to obtain the reference forward pass?

??? note "Solution"

    With LoRA, the policy is the frozen base model plus a small set of trainable adapter weights (the low-rank delta $\Delta W = BA$). The reference distribution $\pi_\text{ref}$ that DPO/GRPO needs is exactly the *original* model — i.e., the base weights **without** the adapter contribution. Because the policy and reference share the identical base tensors already resident on the GPU, TRL does not allocate a second copy of the model. Instead it computes the reference forward pass by temporarily switching the adapters off:

    ```python
    model.disable_adapter_layers()          # switch to base weights only -> reference
    ref_logits = model(input_ids).logits    # forward without the adapter delta
    model.enable_adapter_layers()           # restore policy for the training pass
    ```

    The only extra memory is the adapter delta itself (a few hundred MB for rank-64 on a 7B model), which is negligible. So the reference is "free."

    Under **full** fine-tuning there are no adapters to disable — every parameter of the policy is being updated, so the reference (the pre-update weights) genuinely differs from the policy everywhere. TRL must therefore hold a second, frozen `deepcopy` of the entire model (e.g., another ~14 GB in bf16 for a 7B model). That is the case the chapter describes when it says TRL "does a deepcopy (if full fine-tuning) at trainer initialization." The alternative for very large full-FT models is the EMA-style `sync_ref_model=True` / `ref_model_sync_steps=N` scheme, which periodically snapshots the policy as the new reference instead of keeping a permanent second copy.

**2.** You launch a GRPO run on **8 GPUs** with `per_device_train_batch_size=16`, `num_generations=8`, `gradient_accumulation_steps=4`, and `steps_per_generation` left unset. (a) What is `generation_batch_size`, and is it in prompts or completions? (b) How many *unique prompts* contribute to one optimizer step? (c) If each completion is capped at `max_completion_length=512`, how many generated tokens does one optimizer step cost in the worst case? (d) Your colleague raises `num_generations` from 8 to 12 for a stabler baseline, leaving everything else alone, and TRL refuses to start. Reproduce the error message, and give two different one-line fixes. (e) Hitting OOM, they then set `auto_find_batch_size=True`. TRL refuses that too — why is this refusal *specific to GRPO* rather than a general `Trainer` restriction?

??? note "Solution"

    Recall the two rules from the practitioner tip: `generation_batch_size = per_device_train_batch_size × num_processes × steps_per_generation`, with `steps_per_generation` defaulting to `gradient_accumulation_steps`; and `generation_batch_size` is counted in **completions**, and must be divisible by `num_generations` so no group is split.

    (a) With `steps_per_generation` unset it inherits `gradient_accumulation_steps = 4`:
    $$16 \times 8 \times 4 = 512 \text{ completions}.$$

    (b) Each prompt yields $G = 8$ completions, so:
    $$512 / 8 = 64 \text{ unique prompts}.$$

    (c) Worst case every completion runs to the 512-token cap:
    $$512 \text{ rollouts} \times 512 \text{ tokens} = 262{,}144 \text{ generated tokens per optimizer step}.$$

    This is exactly why the chapter flags generation (step 1 of the GRPO loop, $P \times G$ completions) as the dominant wall-clock cost and motivates the vLLM backend.

    (d) The generation batch is unchanged at 512 completions, but $512 \bmod 12 = 8 \ne 0$, so TRL raises

    ```text
    ValueError: generation_batch_size (512) must be divisible by num_generations (12).
    ```

    A partial group has no valid baseline — you cannot subtract "the group mean" from a group you only half-generated. Two one-line fixes: pick a $G$ that divides 512 (`num_generations=16`, which also strengthens the baseline), or adjust the batch so 12 divides it (`steps_per_generation=3` gives $16 \times 8 \times 3 = 384 = 12 \times 32$). Either way the constraint is on the *product*, not on any single knob.

    (e) `auto_find_batch_size` halves `per_device_train_batch_size` on OOM and retries. For ordinary supervised training that is harmless — examples are i.i.d. and the loss decomposes per example. For GRPO it is not: the advantage $A_i = r_i - \bar{r}$ couples the $G$ completions of a prompt, so a batch that silently loses half its rows can split a group and change the baseline. There is no way to shrink the batch while preserving the whole-groups invariant, so TRL rejects the flag up front rather than producing quietly wrong advantages. Reduce `per_device_train_batch_size` yourself (keeping divisibility), enable `gradient_checkpointing=True`, or move generation off the training memory budget with `use_vllm=True`.

**3.** Using the DPO objective from the chapter, compute the loss for a single triple with $\beta = 0.2$ and the following per-sequence log-probabilities:

| Quantity | Value |
|---|---|
| $\log \pi_\theta(y_w \mid x)$ | $-8.0$ |
| $\log \pi_\text{ref}(y_w \mid x)$ | $-9.0$ |
| $\log \pi_\theta(y_l \mid x)$ | $-11.0$ |
| $\log \pi_\text{ref}(y_l \mid x)$ | $-9.5$ |

Report the two log-ratios, the implicit reward margin, and the final loss. Is the policy currently ranking this pair correctly?

??? note "Solution"

    Chosen log-ratio:
    $$\log \frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} = -8.0 - (-9.0) = +1.0.$$

    Rejected log-ratio:
    $$\log \frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)} = -11.0 - (-9.5) = -1.5.$$

    Implicit reward margin:
    $$\beta \cdot (1.0 - (-1.5)) = 0.2 \times 2.5 = 0.5.$$

    DPO loss:
    $$\mathcal{L} = -\log \sigma(0.5).$$
    With $\sigma(0.5) = \dfrac{1}{1 + e^{-0.5}} = \dfrac{1}{1 + 0.6065} = 0.6225$,
    $$\mathcal{L} = -\log(0.6225) \approx 0.474.$$

    The margin is **positive**, so the policy already assigns relatively more probability mass (versus the reference) to $y_w$ than to $y_l$ — it ranks the pair correctly. This would register as a hit in the `train/rewards/accuracies` metric. The loss (0.474) is still well above the 0.1-0.2 "well-trained" range, so there is room to push the margin higher.

**4.** For one GRPO prompt you sample a group of $G = 8$ completions and score them with the verifiable `math_reward_fn` (reward $1.0$ for a correct boxed answer, else $0.0$). Three completions are correct: $r = [1, 1, 1, 0, 0, 0, 0, 0]$. (a) Compute the group baseline and the advantage $A_i$ for a correct and for an incorrect completion. (b) Now suppose a *different* prompt is so easy that **all 8** completions are correct ($r = [1,1,1,1,1,1,1,1]$). What are the advantages, and what does this imply about the gradient contribution from that prompt? (c) Why does this make dataset difficulty selection important for GRPO?

??? note "Solution"

    (a) Group baseline (mean reward):
    $$\bar{r} = \frac{1}{8}(1+1+1+0+0+0+0+0) = \frac{3}{8} = 0.375.$$
    Advantage $A_i = r_i - \bar{r}$:
    - Correct completion: $A = 1.0 - 0.375 = +0.625$.
    - Incorrect completion: $A = 0.0 - 0.375 = -0.375$.

    So correct responses are pushed up and incorrect ones down, with no critic needed.

    (b) If all 8 are correct, $\bar{r} = 1.0$, and every advantage is
    $$A_i = 1.0 - 1.0 = 0.$$
    Since the GRPO objective multiplies the (clipped) importance ratio by $A_i$, a zero advantage means this prompt contributes **zero policy-gradient signal**. The same is true for a prompt where all completions are wrong ($\bar{r}=0 \Rightarrow A_i = 0$ for all). This is the flip side of the group-relative baseline: learning signal comes only from *within-group reward variance*.

    (c) Prompts that are always solved or never solved waste rollout compute — you pay for $G$ generations but get no gradient. Effective GRPO training wants prompts of intermediate difficulty (mixed success within a group), which maximizes `train/reward_std` and hence the useful signal. This is why curated, difficulty-balanced datasets (and curriculum/filtering) matter, and it connects to the practitioner tip that small $G$ gives noisy baselines: with few samples you also more often land on the all-correct or all-wrong degenerate cases.

**5.** The chapter estimates PPO memory for a 7B model at ~100 GB. (a) Reproduce the policy + optimizer and reference-model figures using the chapter's byte accounting, then redo the calculation for a **13B** model. (b) With that 13B number, how many A100-80GB cards does full-FT PPO minimally need? (c) Explain, in memory terms, how switching to GRPO + LoRA lets the same 13B model train on a single 80 GB card.

??? note "Solution"

    (a) The chapter's accounting for the policy under full fine-tuning is bf16 weights (2 bytes/param) plus Adam optimizer states (8 bytes/param, i.e. fp32 first + second moment), giving 10 bytes/param; the reference is inference-only bf16 (2 bytes/param).

    7B check:
    $$\text{policy+opt} = 7\times10^9 \times (2 + 8) = 70 \text{ GB}, \qquad \text{ref} = 7\times10^9 \times 2 = 14 \text{ GB},$$
    totaling ~84 GB before activations/rollout buffers — consistent with the chapter's "easily exceed 100 GB."

    13B:
    $$\text{policy+opt} = 13\times10^9 \times 10 = 130 \text{ GB}, \qquad \text{ref} = 13\times10^9 \times 2 = 26 \text{ GB},$$
    totaling ~156 GB before activations, value-head, and rollout buffers.

    (b) 156 GB already exceeds one 80 GB card and, once activations and the rollout buffer are added, comfortably needs **at least two** A100-80GB cards (160 GB aggregate) — and realistically FSDP sharding across more.

    (c) GRPO + LoRA collapses this on three fronts:
    - **No value head / critic.** GRPO replaces the learned value function with the group-mean baseline, so there is no critic model or its optimizer states to hold.
    - **Optimizer states only on adapters.** Only the LoRA delta is trainable, so the 8-bytes/param Adam cost applies to a few million adapter params, not all 13B. The frozen base is just 26 GB in bf16 (or ~6.5 GB under QLoRA 4-bit).
    - **Free reference.** As in Exercise 1, the reference is the base model with adapters disabled — no second copy.

    What remains is roughly one copy of the base weights plus tiny adapter optimizer states plus rollout activations, which fits in 80 GB (and gradient checkpointing / QLoRA give further headroom).

**6.** The `composite_reward` function in the chapter applies the length penalty as a hard cliff: `-0.3` as soon as a completion exceeds 800 whitespace tokens, and `0` otherwise. This creates a discontinuity the model can sit just under. **Modify** `composite_reward` so the length penalty is a *smooth linear ramp*: no penalty up to 800 tokens, then a penalty that grows linearly with the overage and saturates at `-0.3` once the completion is 400 tokens over the threshold (i.e. at 1200 tokens). Keep the format and correctness rewards unchanged, and keep the function signature and return type identical.

??? note "Solution"

    Replace only the length-penalty branch with a clamped linear ramp. Let $n$ be the token count; the penalty is $0$ for $n \le 800$, grows linearly as $-0.3 \cdot (n - 800)/400$, and is clamped to $-0.3$ for $n \ge 1200$.

    ```python
    import re
    from typing import Any

    def composite_reward(
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """
        Composite reward for math reasoning:
        - +0.5 if completion contains a <think>...</think> block (format reward)
        - +1.0 if the boxed answer matches the ground truth (correctness reward)
        - smooth length penalty: 0 up to 800 tokens, ramping linearly to
          -0.3 at 1200 tokens (and staying at -0.3 beyond).
        """
        ground_truths = kwargs.get("answer", [""] * len(completions))
        rewards = []

        for completion, gt in zip(completions, ground_truths):
            r = 0.0

            # --- Format reward ---
            if re.search(r"<think>.*?</think>", completion, re.DOTALL):
                r += 0.5

            # --- Correctness reward ---
            pred = extract_boxed_answer(completion)
            if pred is not None and pred.strip() == str(gt).strip():
                r += 1.0

            # --- Smooth length penalty ---
            n_tokens = len(completion.split())
            overage = n_tokens - 800
            if overage > 0:
                frac = min(overage / 400.0, 1.0)   # 0 -> 1 over [800, 1200]
                r -= 0.3 * frac

            rewards.append(r)

        return rewards


    def extract_boxed_answer(text: str) -> str | None:
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        return match.group(1).strip() if match else None
    ```

    Quick sanity checks: a 700-token answer incurs `0`; an 800-token answer incurs `0`; a 1000-token answer incurs $-0.3 \times (200/400) = -0.15$; a 1200-token (or longer) answer incurs the full $-0.3$. Because the ramp is continuous, there is no single token boundary the model can exploit, and the gradient of the reward w.r.t. length is now informative rather than a step. The signature `(prompts, completions, **kwargs) -> list[float]` is unchanged, so it still drops into `GRPOTrainer(reward_funcs=[composite_reward], ...)` exactly as before.

    This is exactly the shape of DAPO's soft overlong punishment, which TRL ships as `trl.rewards.get_soft_overlong_punishment(max_completion_len, soft_punish_cache)` — the ramp there runs over the last `soft_punish_cache` tokens before `max_completion_len` and saturates at $-1$. Two differences worth noting in the real implementation: it measures length in *tokens* (via the `completion_ids` kwarg TRL passes to every reward function) rather than whitespace-split words, and it is registered as a separate reward function rather than folded into the composite, so `rewards/soft_overlong_punishment_reward/mean` gets its own logged series (the key comes from the callable's `__name__`).
