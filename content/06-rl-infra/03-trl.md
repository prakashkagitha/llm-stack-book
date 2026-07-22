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

TRL exposes five major trainers relevant to alignment work. They sit at different points in the alignment pipeline:


{{fig:trl-trainer-landscape-pipeline}}


Each trainer is independently usable; you do not need to run them all in sequence. Many modern recipes (DeepSeek-R1, Qwen-2.5, etc.) skip the standalone reward model and go straight to GRPO with a verifiable reward function.

### SFTTrainer

`SFTTrainer` wraps `transformers.Trainer` with quality-of-life features for supervised fine-tuning:

- **Automatic sequence packing.** It uses `ConstantLengthDataset` to concatenate examples into fixed-length chunks, eliminating padding waste. For a 2048-token context with typical 200-token instruction examples, packing can increase GPU utilization by 4–6x.
- **Chat template application.** Pass a `formatting_func` or set `dataset_text_field` and TRL handles tokenization.
- **PEFT integration.** Pass a `PeftConfig` (e.g., `LoraConfig`) and the trainer wraps the model automatically.
- **Loss masking.** The trainer supports masking prompt tokens from the loss (only computing loss on the completion side) by setting `completion_only_trainer=True` or using `DataCollatorForCompletionOnlyLM`.

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Load a model and tokenizer (e.g., Qwen-2.5-7B-Instruct)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",          # bfloat16 on Ampere+
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
    max_seq_length=2048,              # hard cap; packing fills to this
    packing=True,                     # enable ConstantLengthDataset packing
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
    torch_dtype="bfloat16",
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

If the model is too large to hold two copies, pass `ref_model=None` and set `create_reference_model=False` together with `sync_ref_model=True` and `ref_model_sync_steps=N`: TRL will periodically snapshot the training model as the new reference (Exponential Moving Average–style).

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Policy model wrapped with LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16"
)
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")
model = get_peft_model(base_model, lora_config)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,                          # KL regularization temperature
    loss_type="sigmoid",               # standard DPO; alternatives: "ipo", "kto_pair"
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True,
    max_length=2048,                   # max total len (prompt + completion)
    max_prompt_length=1024,
    generate_during_eval=True,         # sample from policy during eval for inspection
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

The TRL PPO loop at each iteration:

1. **Rollout.** Sample a batch of prompts; call `model.generate()` to produce completions.
2. **Reward scoring.** Pass `(prompt, completion)` pairs through the reward model.
3. **KL penalty.** Compute per-token KL divergence between the policy and a frozen reference model; subtract it from the reward.
4. **Advantage estimation.** Run a value head (a separate linear layer on top of the policy backbone) through the rollout to compute GAE (Generalized Advantage Estimation) advantages.
5. **PPO update.** Run $K$ mini-batch gradient steps with the clipped surrogate objective.

```python
from trl import PPOConfig, PPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# The policy model also has a value head (AutoModelForCausalLMWithValueHead)
from trl import AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    "gpt2",         # small example; swap for Llama/Qwen in practice
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=256,         # total rollouts per update step
    mini_batch_size=32,     # mini-batch size for gradient steps
    gradient_accumulation_steps=1,
    ppo_epochs=4,           # inner PPO mini-batch epochs
    max_grad_norm=0.5,
    kl_penalty="kl",        # per-token adaptive KL penalty
    init_kl_coef=0.2,       # starting KL coefficient
    target_kl=6.0,          # target KL for adaptive controller
    adap_kl_ctrl=True,      # adaptive KL coefficient
    horizon=10000,          # steps for adaptive controller
)

# Reward pipeline (sentiment as a toy example)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="lvwerra/distilbert-imdb",
    device=0,
)

trainer = PPOTrainer(
    config=ppo_config,
    model=policy,
    ref_model=None,   # auto-created from policy weights
    tokenizer=tokenizer,
)

# Training loop (simplified)
prompts = ["The movie was", "I thought the film", "Overall, this production"]
for epoch in range(10):
    # --- Rollout phase ---
    query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze() for p in prompts]

    # generate() returns a list of response tensors
    response_tensors = trainer.generate(
        query_tensors,
        max_new_tokens=50,
        do_sample=True,
        top_k=0,
        top_p=0.9,
    )

    # Decode completions for reward scoring
    completions = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # --- Reward scoring ---
    pipe_outputs = sentiment_pipe(completions, top_k=None)
    rewards = [
        torch.tensor(
            next(d["score"] for d in out if d["label"] == "POSITIVE"), dtype=torch.float
        )
        for out in pipe_outputs
    ]

    # --- PPO update (handles advantages, value loss, policy loss internally) ---
    stats = trainer.step(query_tensors, response_tensors, rewards)
    trainer.log_stats(stats, {}, rewards)
```

### PPOTrainer's memory footprint

PPO is expensive: at inference time you need the policy, the value head, and the frozen reference model, plus all their optimizer states during the update. For a 7B model in bf16:
- Policy + optimizer states (Adam): roughly $7 \times 10^9 \times 2 + 7 \times 10^9 \times 8 = 70$ GB
- Reference model (inference-only, bf16): ~14 GB
- Activations and rollout buffer: varies

Total can easily exceed 100 GB for a 7B model, requiring at least two A100-80GB cards. This cost motivated the GRPO and DPO approaches that eliminate the value head.

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
    torch_dtype="auto",     # bfloat16 on Ampere+
    attn_implementation="flash_attention_2",  # requires flash-attn installed
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ----------------------------------------------------------------
# 2. Reward function: verifiable math reward
#    Returns a float for each (prompt, completion) pair.
#    TRL calls this as: reward_fn(prompts, completions) -> List[float]
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
    num_generations=8,           # G: responses per prompt (more = stabler baseline)
    # --- Generation kwargs passed to model.generate() ---
    max_new_tokens=512,          # max completion length
    temperature=0.9,             # diversity in rollouts
    do_sample=True,
    # --- Training ---
    per_device_train_batch_size=2,   # prompts per device per step
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-6,              # small LR is critical for stability
    bf16=True,
    # --- PPO/GRPO-specific ---
    epsilon=0.2,                 # PPO clip ratio ε
    beta=0.04,                   # KL penalty coefficient
    # --- Logging ---
    logging_steps=5,
    save_steps=50,
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

    Set `num_generations` to at least 8 for stable advantage estimates. With only 4 samples, the within-group variance is high enough to produce noisy gradients. The memory cost scales linearly with `num_generations`, so on memory-constrained hardware use GRPO with LoRA and reduce the generation batch via `per_device_train_batch_size=1` first.

### The GRPO training loop internals

When `trainer.train()` runs, TRL executes the following per-step logic:


{{fig:trl-grpo-train-loop-steps}}


The most expensive step is step 1: generating $P \times G$ completions sequentially on the same GPUs doing training. This is the core throughput bottleneck; the vLLM backend (§6) addresses it.

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

The primary bottleneck in GRPO (and PPO) is `model.generate()`, which runs on the same GPUs as training. TRL 0.12+ integrates vLLM as an optional generation backend:

```python
grpo_config = GRPOConfig(
    use_vllm=True,
    vllm_server_host="localhost",    # host running the vLLM inference server
    vllm_server_port=8000,
    vllm_server_timeout=120,         # seconds to wait for server readiness
    # vLLM generation kwargs
    temperature=1.0,
    top_p=0.95,
)
```

The workflow with vLLM:


{{fig:trl-vllm-colocate-rollout-flow}}


After each training step, TRL calls `vllm_client.update_model_weights(trainer.model)` to push the latest policy weights to the vLLM server via shared memory or RDMA. The vLLM server then generates the next batch of rollouts while the training GPUs continue back-propagation. This overlap of generation and gradient computation can improve wall-clock throughput by 1.5–2x on large enough models.

For a deep dive on the PagedAttention mechanism powering vLLM's generation, see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html).

## Practical Configuration Reference

The key numerical hyperparameters and their typical ranges:

| Parameter | DPO | GRPO | PPO | Notes |
|---|---|---|---|---|
| `beta` (KL coefficient) | 0.01–0.5 | 0.01–0.1 | adaptive | Higher = stay closer to reference |
| `learning_rate` | 1e-5–5e-5 | 5e-7–2e-6 | 1e-5–3e-5 | GRPO needs very small LR |
| `num_generations` (G) | — | 4–16 | — | More = stable baseline, more memory |
| `epsilon` (PPO clip) | — | 0.1–0.2 | 0.1–0.2 | Standard PPO value |
| `max_new_tokens` | — | 256–2048 | 128–512 | Task-dependent |
| `ppo_epochs` | — | — | 1–4 | Reuse ratio; 1 is safer |
| Effective batch size | 32–128 | 64–512 | 64–256 | prompts × G for GRPO |

!!! warning "Common pitfall"

    Setting `beta` too low (< 0.01) during GRPO or DPO training leads to reward hacking — the model finds degenerate completions that score well on the reward function but have diverged catastrophically from the reference distribution. Monitor the KL divergence metric (logged as `kl` or `mean_kl`) and stop training if it exceeds ~20 nats for most tasks. See [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html) for detailed diagnostics.

### Monitoring training health

TRL logs a rich set of metrics to Weights & Biases or TensorBoard:

```text
# DPO metrics
train/rewards/chosen       # mean reward for chosen responses (proxy: log-ratio margin)
train/rewards/rejected
train/rewards/accuracies   # fraction of examples where chosen reward > rejected reward
train/logps/chosen         # mean log-probability of chosen completions
train/kl                   # KL divergence from reference

# GRPO metrics
train/reward               # mean reward across all rollouts
train/reward_std           # within-group reward variance
train/kl                   # mean KL penalty
train/clip_ratio           # fraction of tokens where PPO clip is active (target ~0.1–0.2)
train/response_length      # mean completion length (watch for length exploitation)
```

A healthy GRPO run shows `train/reward` monotonically increasing, `train/kl` staying below 10–15, and `train/clip_ratio` around 0.1–0.2. If `train/response_length` explodes, the model is gaming a length-correlated reward — add an explicit length penalty to your reward function.

## Strengths and Limits of TRL

### Where TRL excels

- **Ease of use.** A working GRPO experiment requires fewer than 100 lines of Python. The HuggingFace ecosystem (datasets, tokenizers, model hub, PEFT) integrates without boilerplate.
- **Breadth.** A single library covers the entire alignment pipeline: SFT → RM → PPO/DPO/GRPO. You can ablate algorithms quickly by swapping trainer classes.
- **PEFT integration.** QLoRA + GRPO on a single A100-40GB is routine; the reference model is free.
- **Custom reward functions.** Passing a Python callable means any verifiable reward — code execution, math verification, format checking — works without infrastructure overhead.
- **Community and maintenance.** As part of the HuggingFace organization, TRL receives rapid fixes, new algorithms (SimPO, KTO, ORPO were all added within months of publication), and broad community support.

### Where TRL hits limits

- **Single-node generation bottleneck.** Without the vLLM backend, generation and training share GPUs. For a 70B model, rollout latency can dwarf training compute, making per-step wall-clock time impractical.
- **No disaggregated workers.** TRL does not support a separate fleet of rollout workers consuming prompts from a queue, as [veRL](../06-rl-infra/04-verl.html) or [OpenRLHF](../06-rl-infra/05-openrlhf-nemo-ray.html) do. The single-process design limits throughput scaling.
- **Memory for PPO.** The value head and reference model double the memory footprint. At 70B, PPO requires a large cluster even with FSDP; GRPO is much more practical.
- **No async rollouts.** Each training step waits for all rollouts to complete before updating. For tasks with variable-length or slow reward evaluation (e.g., running code), this serialization wastes GPU time. [Prime-RL](../06-rl-infra/06-prime-rl-async.html) addresses this with asynchronous rollout workers.
- **Sequence packing in RL.** Packing works beautifully for SFT but is harder in GRPO/PPO because each rollout in a group must be associated with its prompt for advantage computation. TRL handles this but the implementation is more complex than SFT packing.

!!! interview "Interview Corner"

    **Q:** A colleague proposes using TRL's `GRPOTrainer` with `num_generations=16` to train a 13B reasoning model on GSM8K. You have 4 × A100-80GB GPUs. What bottlenecks do you anticipate, and how would you address them?

    **A:** Three main bottlenecks arise. First, **memory**: 13B in bf16 is ~26 GB; with LoRA adapters the model fits on one GPU, but GRPO must forward 16 completions per prompt through the policy and reference model. Use QLoRA (4-bit base) or enable `gradient_checkpointing=True` to halve activation memory. Second, **generation throughput**: generating 16 completions sequentially is the dominant wall-clock cost. Enable `use_vllm=True` and dedicate 1–2 GPUs to the vLLM server while 2–3 GPUs train. Third, **batch size / advantage stability**: with 4 GPUs and `per_device_train_batch_size=1`, each step processes 4 × 1 × 16 = 64 rollouts. The group baseline is computed per-prompt over 16 responses, which is adequate. Use `gradient_accumulation_steps=8` to increase the effective batch to 512 rollouts and stabilize gradients. Monitor `train/kl` (keep below 10) and `train/clip_ratio` (target 0.1–0.2).

## Building a Custom Reward Function Pipeline

One of TRL's most powerful features is that reward functions are plain Python. Here is a production-grade reward function that combines a format reward with a correctness reward, plus a length penalty:

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
# 1. Install dependencies
pip install trl[vllm] transformers peft accelerate flash-attn bitsandbytes

# 2. Launch with accelerate (8 GPUs: 4 train + 4 vLLM)
accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  train_grpo_math.py \
  --model_name Qwen/Qwen2.5-7B \
  --dataset gsm8k \
  --num_generations 8 \
  --beta 0.04 \
  --learning_rate 1e-6 \
  --num_train_epochs 3 \
  --use_vllm true \
  --vllm_server_host localhost \
  --vllm_server_port 8000
```

The script `train_grpo_math.py` is essentially the GRPOTrainer example from §4, with the dataset replaced by GSM8K and a proper verifiable reward function. The RLVR recipe and its relation to chain-of-thought emergence are covered in depth in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

## Key Takeaways

!!! key "Key Takeaways"

    - TRL provides `SFTTrainer`, `RewardTrainer`, `DPOTrainer`, `PPOTrainer`, and `GRPOTrainer` as modular, independently usable components covering the full alignment pipeline.
    - All trainers inherit from `transformers.Trainer` and use `accelerate` for distribution — any `accelerate` backend (FSDP, DeepSpeed, DDP) works without trainer-level code changes.
    - PEFT (LoRA, QLoRA) integrates transparently: the reference model shares the base backbone with the policy, making the reference model nearly memory-free for DPO and GRPO.
    - `GRPOTrainer` is the recommended entry point for reasoning alignment (replacing PPO): no value head, group-relative advantages, and verifiable reward functions as plain Python callables.
    - The generation bottleneck is TRL's main throughput limitation; the vLLM integration (`use_vllm=True`) overlaps rollout generation with gradient computation for 1.5–2x wall-clock speedup.
    - Monitor `train/kl`, `train/reward`, and `train/clip_ratio`; a KL spike above 20 or a clip ratio above 0.5 signals instability.
    - TRL is the fastest path from research paper to running experiment; for production-scale multi-node runs (70B+), consider veRL or OpenRLHF which offer disaggregated rollout workers and higher throughput.
    - Composite reward functions (format + correctness + length penalty) are straightforward to implement as Python callables — no infrastructure overhead required.

!!! sota "State of the Art & Resources (2026)"
    TRL has matured into the standard entry point for LLM post-training, with v1.0 (March 2026) stabilising its CLI, config system, and trainer suite; `GRPOTrainer` with co-located vLLM is now the dominant single-node recipe for reasoning alignment, while disaggregated frameworks (veRL, OpenRLHF) handle the 70B+ regime.

    **Foundational work**

    - [Ziegler et al., *Fine-Tuning Language Models from Human Preferences* (2019)](https://arxiv.org/abs/1909.08593) — the original RLHF paper combining reward models and PPO for language, the conceptual root of TRL's pipeline.
    - [Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347) — the clipped surrogate objective underlying TRL's `PPOTrainer` and GRPO's importance-ratio update.
    - [Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023)](https://arxiv.org/abs/2305.18290) — derives the closed-form DPO loss that TRL's `DPOTrainer` implements, eliminating the need for a separate reward model.

    **Recent advances (2023–2026)**

    - [Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024)](https://arxiv.org/abs/2402.03300) — introduces GRPO (group-relative advantage, no critic), the algorithm behind TRL's `GRPOTrainer`.
    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — demonstrates RLVR with verifiable rewards using GRPO at scale; the defining blueprint for TRL's reasoning alignment use-case.

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

**2.** You launch a GRPO run on **8 GPUs** with `per_device_train_batch_size=2`, `num_generations=8`, and `gradient_accumulation_steps=4`. (a) How many *unique prompts* contribute to one optimizer step? (b) How many total *rollouts* (completions) are generated per optimizer step? (c) If each completion is capped at `max_new_tokens=512`, how many generated tokens does one optimizer step cost in the worst case?

??? note "Solution"

    Recall from the chapter that for GRPO the effective batch is `prompts x G`, that `per_device_train_batch_size` counts *prompts per device*, and that gradient accumulation multiplies the work done before a single optimizer step.

    (a) Unique prompts per optimizer step:
    $$8 \text{ GPUs} \times 2 \text{ prompts/GPU} \times 4 \text{ accum steps} = 64 \text{ prompts}.$$

    (b) Each prompt is expanded into $G = 8$ completions, so total rollouts:
    $$64 \times 8 = 512 \text{ rollouts}.$$

    (c) Worst case every completion runs to the 512-token cap:
    $$512 \text{ rollouts} \times 512 \text{ tokens} = 262{,}144 \text{ generated tokens per optimizer step}.$$

    This is exactly why the chapter flags generation (step 1 of the GRPO loop, $P \times G$ completions) as the dominant wall-clock cost and motivates the vLLM backend.

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
