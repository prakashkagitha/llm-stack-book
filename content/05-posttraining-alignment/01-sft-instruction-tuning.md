# 5.1 Supervised Fine-Tuning & Instruction Tuning

A freshly pretrained language model is a remarkable thing: it has absorbed syntax, facts, reasoning patterns, and writing styles from hundreds of billions of tokens of text. And yet if you give it the prompt "Explain quantum entanglement to a 10-year-old," a raw base model is just as likely to continue the sentence with another question, an unrelated anecdote, or a bibliographic citation as it is to produce a helpful explanation. The model has learned *language*; it has not learned to be *helpful*.

This is the problem supervised fine-tuning (SFT) solves. SFT teaches the model to follow instructions, adopt a conversational format, and suppress unhelpful completions — all by showing it examples of the behavior we want. In this chapter we work through the mechanics of SFT from first principles: the objective, the data, the full-versus-partial finetuning spectrum, the risk of catastrophic forgetting, and the three-stage post-training recipe that is now standard across frontier labs. We include a complete, runnable SFT training loop.

SFT is the first stage of the alignment pipeline. It is followed by preference learning ([The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)) and optionally by policy-optimization steps ([Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) and [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)). But none of those later stages work well without a solid SFT foundation.

## Why Base Models Need Fine-Tuning

Pretraining (see [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html)) optimizes for next-token prediction over a diverse web corpus. The model learns to mimic the statistical properties of text — including plenty of text that continues questions with more questions, lists instructions without following them, or meanders off-topic. It has no concept of a "user" who wants something useful.

The gap between a base model and a useful assistant has three dimensions:

1. **Format mismatch.** Users send instructions; the internet mostly contains prose, code, and documents. The model hasn't been rewarded for recognizing and following an imperative sentence.
2. **Behavior mismatch.** Even when the model "knows" an answer, it may produce the answer embedded in a Wikipedia-style article rather than as a direct reply.
3. **Value mismatch.** Base models generate whatever is most probable; a helpful assistant should refuse harmful requests, acknowledge uncertainty, and exhibit certain social norms.

SFT addresses the first two problems by maximum-likelihood training on high-quality (instruction, response) pairs. It partially addresses the third; the remainder is the job of RLHF/DPO.

## The SFT Objective

Given a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ where $x_i$ is an instruction (prompt) and $y_i$ is the target response, SFT fine-tunes the pretrained model $\theta$ by minimizing the standard negative log-likelihood, but *only over the response tokens*:

$$
\mathcal{L}_\text{SFT}(\theta) = -\sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log p_\theta\!\left(y_i^{(t)} \mid x_i, y_i^{(<t)}\right)
$$

This is identical to the pretraining causal language-modeling loss, with one critical difference: the loss mask. Tokens belonging to the instruction $x_i$ are masked out (weight zero); only the response $y_i$ tokens contribute to the gradient. The model is trained to predict what a good assistant would say, not to re-predict the instruction it just received.

The concatenated sequence seen by the transformer is:

{{fig:sft-loss-mask-sequence}}

The exact chat template varies by model family — see [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) for the full treatment of tokenization and formatting.

### Why Not Train on the Full Sequence?

A natural question: why mask the instruction? In principle, training on the full sequence also works — and some practitioners do it. But there are two reasons to prefer response-only supervision:

- **Signal concentration.** Instructions are often short; responses are longer. Response-only supervision gives the optimizer a cleaner gradient signal that specifically rewards good reply quality rather than re-encoding the input.
- **Prompt contamination.** If the model is penalized for "wrong" instruction tokens, it may learn to prefer certain prompt phrasings over others in ways that generalize poorly.

In practice, the difference is small for well-formatted data but response masking is the standard convention.

## Instruction Datasets: A Field Guide

The quality of SFT output depends overwhelmingly on the data. Here we survey the landmark datasets that shaped how the field thinks about instruction tuning.

### FLAN (Finetuned Language Models Are Zero-Shot Learners, Wei et al., 2021)

FLAN was among the first large-scale demonstrations that instruction tuning dramatically improves zero-shot task performance. It took 62 existing NLP benchmark datasets (sentiment, QA, translation, commonsense, etc.) and reformulated each as a set of natural-language instruction templates: for example, a sentiment classification sample might be phrased as "Does this review express a positive or negative opinion? Review: ..." followed by the label as the response.

The key insight from FLAN: task diversity matters enormously. A model finetuned on many task types generalizes to held-out tasks; a model finetuned on a narrow set does not. The follow-on FLAN-T5 and FLAN-v2 scaled this to thousands of task mixtures.

### Alpaca (Taori et al., Stanford, 2023)

Alpaca made instruction tuning accessible to the broader research community. The data generation method is now iconic: feed 175 seed instruction-response pairs to a capable API model (text-davinci-003) and prompt it to generate 52,000 more using a self-instruct procedure. Fine-tune LLaMA-7B on the result. The fine-tuned model followed instructions surprisingly well, demonstrating that even a small model benefits enormously from instruction tuning.

Alpaca revealed two things: (1) you can synthesize instruction data cheaply using a stronger teacher; and (2) even 52k examples is enough to meaningfully shift model behavior if the examples are diverse.

!!! warning "Data quality is a moving target"
    Alpaca data is known to contain factual errors and repetitive patterns introduced by the teacher model. Several studies found that training on cleaner subsets of 5–10k examples outperformed the full 52k set. This foreshadows LIMA.

### ShareGPT & OpenHermes

ShareGPT is a crowd-sourced corpus of real ChatGPT conversations shared voluntarily by users. Unlike Alpaca's single-turn format, ShareGPT contains multi-turn dialogues with genuine diversity of topic and user intent. This makes it much better at teaching the model how to handle follow-up questions, clarifications, and context accumulation.

OpenHermes (Teknium, 2023) is a curated blend of high-quality synthetic conversations from multiple sources: code exercises, reasoning problems, creative writing, roleplay, and factual Q&A. It demonstrated that careful curation and blending of several source datasets — even with no human annotation — can produce a very capable SFT model.

### LIMA: Less Is More for Alignment (Zhou et al., 2023)

LIMA is possibly the most important alignment paper for practitioners. The authors curated exactly 1,000 high-quality prompt-response pairs — drawn from Stack Exchange, wikiHow, and Reddit, plus some hand-written examples — and fine-tuned a LLaMA-65B model on nothing else.

The result outperformed SFT models trained on hundreds of thousands of examples in human preference studies. The conclusion: **data quality dominates data quantity**. A model that has absorbed vast knowledge during pretraining needs only a relatively small number of well-structured demonstrations to learn the format and style of helpful responses. The alignment tax for SFT is surprisingly cheap — if the data is good.

The LIMA hypothesis is sometimes called the *superficial alignment hypothesis*: the core "knowledge" of the model lives in the pretraining weights; SFT teaches the model to *access and present* that knowledge in the right format.

!!! note "Implications for practitioners"
    Before scaling your SFT dataset to millions of examples, invest heavily in data quality metrics: deduplication, response length distribution, instruction diversity, and human spot-checking. Starting with 1–5k meticulously verified examples often beats starting with 100k noisy ones.

{{fig:superficial-alignment-quality-over-quantity}}

## Data Quality > Quantity: Practical Data Engineering

Given the LIMA insight, how do you build high-quality SFT data in practice?

**Instruction diversity.** Cluster your instructions (e.g., by embedding them with a sentence encoder) and measure coverage. An instruction dataset that is 70% "write a Python function to..." is not diverse and will produce a lopsided model.

**Response quality filters.** Use heuristics (minimum length, no refusal boilerplate, no truncations) and reward model scoring. If you already have a reward model (chapter [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)), filter to the top-scoring responses per instruction cluster.

**Format consistency.** All examples should follow the same chat template. Mixed templates in a training batch cause the model to learn an inconsistent format.

**Deduplication.** Near-duplicate instructions with slightly different phrasings inflate dataset size while contributing almost no new learning signal. Min-hash or embedding-based deduplication is standard; see [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html).

**Long-tail coverage.** Ensure examples cover rare but important topics: safety refusals, citations/uncertainty, multi-hop reasoning, code debugging. These are underrepresented in organic data but disproportionately important for the model's edge-case behavior.

### Comparing Key SFT Datasets

| Dataset | Size | Source | Multi-turn | Key property |
|---|---|---|---|---|
| FLAN | ~100k+ | NLP benchmarks, templated | No | Task diversity |
| Alpaca | 52k | GPT-3 self-instruct | No | Cheap synthesis |
| ShareGPT | ~70k | Real ChatGPT conversations | Yes | Authentic multi-turn |
| OpenHermes | ~900k | Multi-source synthetic blend | Yes | Quality curation at scale |
| LIMA | 1k | Human curated | No | Quality > quantity demo |

## Refusal Training: A Safety Data Recipe

The "Long-tail coverage" bullet above gestures at "safety refusals" as one underrepresented category — but refusal data deserves its own recipe, because getting it wrong in either direction is easy: too little and refusals are unreliable, too much and the model over-refuses.

**Mixing ratio.** Safety demonstrations should be a small fraction of the overall SFT mix — roughly 1–5% (Llama-2 and Tulu-style recipes sit in the low single digits). This is a calibration problem, not a "more is better" axis: pushing the fraction up doesn't make the model safer past a point, it makes it learn to decline on keyword triggers regardless of context.

**Refusal response format.** A good refusal is short and non-preachy: (a) a brief decline, (b) one clause of reason, (c) optionally a safe redirection or partial, harm-reducing help for dual-use asks. For example: *"I can't help with synthesizing that compound, but I can point you to general lab-safety resources if that's useful."* Long moralizing refusals are bad — they teach the model verbosity, they're trivially detected and steered around by jailbreaks, and they annoy users. Wherever the request is dual-use rather than clearly malicious, prefer a "safe completion" (partial, harm-reducing help) over a flat refusal.

**Contrast sets to control over-refusal.** For every harmful prompt in the mix, include several benign look-alike prompts as COMPLY examples — XSTest-style pairs such as "How do I whittle a knife?" (comply) versus "How do I whittle a knife to kill my sister?" (refuse). Without these, the model learns keyword matching and starts refusing benign medical, history, fiction, or chemistry questions. Within any topic cluster, the comply contrast examples should outnumber the refusals — that ratio is what keeps the model from generalizing "knife" to "refuse."

**Safety preference pairs.** For the reward model or DPO stage (see [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)), construct pairs on both axes:

- **Harm axis:** chosen = calibrated refusal or safe completion, rejected = harmful compliance.
- **Over-refusal axis:** chosen = helpful compliance on a benign look-alike, rejected = an unnecessary refusal of that same benign prompt.

If you only build harm-axis pairs, the reward model learns that refusing is always the safe choice and pushes the policy toward blanket refusal. The over-refusal pairs are the counterweight that keeps refusal calibrated rather than maximal.

**Scaling the data.** Hand-writing thousands of refusal and safe-completion examples doesn't scale. See [Constitutional AI, RLAIF & Self-Improvement](../05-posttraining-alignment/11-constitutional-rlaif.html) for generating refusal and revision data at scale via model self-critique instead.

**Evaluation.** A safety change is only good if attack success rate (ASR) drops without benign compliance dropping — report both as one operating point, not in isolation. Evaluate with the red-teaming and safety harness: ASR on HarmBench-style jailbreaks and over-refusal false-positive rate / XSTest compliance on benign look-alikes. See [Red-Teaming & Safety Evaluation](../11-evaluation/05-redteaming-safety-eval.html).

!!! warning "The over-refusal failure mode"
    A naive "add more refusals" approach reliably produces a model that refuses benign requests. It happens by default because refusal examples are cheap to write and comply examples on adjacent benign topics are easy to forget. The contrast set (comply examples outnumbering refusals per cluster) and the over-refusal preference pairs are what keep the operating point calibrated — without them, safety training silently trades helpfulness for a false sense of security.

{{fig:refusal-calibration-contrast-set}}

## The Three-Stage Post-Training Recipe

Modern instruction-following models are not trained in one step. The standard recipe, used across frontier labs, involves three stages:

{{fig:sft-three-stage-posttraining-recipe}}

SFT is indispensable as Stage 1 because:

- It gives the RL algorithm a strong behavioral prior to start from. RL optimization on a raw base model is extremely sample-inefficient and unstable — the policy space is too large.
- The SFT model already speaks the right "language" (instruction-following format), so the reward model can produce meaningful gradients from day one.
- Some tasks (code generation, structured output) are almost entirely learned in SFT; RL fine-tunes the margins.

The SFT → RM → RL pipeline is covered in depth in [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) and [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html). DPO skips the explicit RM step; see [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

## Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning

When fine-tuning a pretrained model, you have a spectrum of choices about which parameters to update.

### Full Fine-Tuning

All $N$ parameters of the model are updated by gradient descent. For a 7B-parameter model in float32, the parameters alone require ~28 GB. Full fine-tuning additionally needs optimizer states: AdamW maintains a first and second moment per parameter, adding another ~56 GB for a total of ~84 GB at float32 — or ~42 GB at bf16/float32 mixed precision. This typically requires multiple high-memory GPUs.

Full fine-tuning gives the model the most flexibility to adapt, and it is the preferred choice when:
- You have abundant compute.
- You are doing domain adaptation that requires broad weight updates (e.g., medical notes for a model trained only on general web text).
- The target distribution differs substantially from pretraining.

### Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods freeze most model weights and train only a small adapter. The canonical method is LoRA (Low-Rank Adaptation, Hu et al., 2021): for a weight matrix $W \in \mathbb{R}^{d \times k}$, learn a low-rank update $\Delta W = AB$ where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$.

The modified forward pass is:

$$
h = W x + \Delta W x = W x + A B x
$$

with $B$ initialized to zero so that $\Delta W = 0$ at the start of training. During inference the adapter can be merged back: $W' = W + AB$, adding zero latency overhead.

LoRA at rank $r=16$ on a 7B model trains roughly 0.5–1% of parameters, reducing memory requirements dramatically. QLoRA (Dettmers et al., 2023) additionally quantizes the frozen base model weights to 4-bit NF4, enabling SFT of 7B models on a single consumer GPU.

See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) for the full PEFT treatment. For this chapter we focus on the SFT objective and data pipeline, which are identical whether you use full fine-tuning or LoRA — the only difference is which parameters receive gradients.

!!! example "Memory budget: 7B full fine-tuning vs. QLoRA"
    Consider a 7B-parameter model. Parameters have roughly 7 × 10⁹ entries.

    **Full fine-tuning (bf16 params + fp32 optimizer states):**
    - Model weights: 7B × 2 bytes = 14 GB
    - Gradients (fp32): 7B × 4 bytes = 28 GB
    - AdamW m₁ + m₂ (fp32): 7B × 2 × 4 bytes = 56 GB
    - Activations (sequence length 2048, batch 4): ~12–20 GB depending on architecture
    - **Total: ~110–120 GB** → requires 2–4 × 80 GB A100s

    **QLoRA (4-bit frozen base + fp32 LoRA adapters, r=64):**
    - Quantized base: 7B × 0.5 bytes ≈ 3.5 GB
    - LoRA trainable params ≈ 80M × 4 bytes ≈ 0.3 GB
    - AdamW states for LoRA ≈ 0.6 GB
    - Activations: ~4–8 GB
    - **Total: ~10–15 GB** → fits on a single 24 GB RTX 3090 or 4090

{{fig:sft-memory-budget-full-vs-qlora}}

## Catastrophic Forgetting

One of the central risks of fine-tuning a pretrained model is catastrophic forgetting (CF): updating weights to improve performance on the target task can degrade performance on other tasks the model previously handled well. The weights that encode general knowledge are overwritten by the gradient updates for the specific SFT distribution.

### Why It Happens

During pretraining, the model's weights settle into a configuration that is jointly optimal for a huge variety of tasks. SFT data is a narrow sample of that space. If the learning rate is too large, or the SFT distribution is too far from the pretraining distribution, gradient descent will "forget" the pretraining solution in favor of the fine-tuning target.

The phenomenon is well-studied in continual learning: when a neural network is trained sequentially on task A then task B, performance on task A degrades roughly in proportion to the distance between the loss landscapes of A and B.

{{fig:catastrophic-forgetting-loss-landscape}}

### Mitigation Strategies

**Low learning rate.** SFT learning rates are typically one to two orders of magnitude below pretraining rates. If pretraining used a peak LR of $3 \times 10^{-4}$, SFT might use $1 \times 10^{-5}$ to $5 \times 10^{-5}$. This limits the step size of weight updates, preserving most of the pretrained knowledge.

**Short training (1–3 epochs).** Running SFT for many epochs on a small dataset causes the model to overfit the SFT distribution and forget pretraining. One to three epochs over a well-curated dataset is the standard.

**Small dataset advantage.** Counter-intuitively, training on a *smaller* high-quality dataset (LIMA-style) for fewer steps causes less forgetting than training on a large noisy dataset for many steps.

**Data mixing.** Blending a small fraction (5–10%) of pretraining data back into the SFT mix preserves general capabilities. This is sometimes called replay or data mixing and is common in practice.

**LoRA/PEFT.** Because LoRA freezes the base weights, it is structurally immune to overwriting pretrained knowledge. The base model's general knowledge is perfectly preserved; only the low-rank adapter changes. This is a major practical advantage of PEFT beyond just memory efficiency.

**Elastic Weight Consolidation (EWC, Kirkpatrick et al., 2017).** Adds a penalty term to the loss that discourages changes to parameters that were important for previous tasks, weighted by the Fisher information matrix. Rarely used in LLM SFT today (too expensive to compute exactly) but conceptually important.

!!! interview "Interview Corner"
    **Q:** You're fine-tuning a 7B instruction model on 50,000 domain-specific Q&A pairs for a medical assistant. After fine-tuning, users report that the model can no longer do basic arithmetic and has lost some of its general conversational ability. What went wrong, and how would you fix it?

    **A:** This is classic catastrophic forgetting. Several things likely went wrong: (1) the learning rate was too high, causing large weight updates that overwrote general-capability weights; (2) training ran for too many epochs, overfitting the medical distribution; and (3) there was no data mixing to maintain coverage of general tasks.

    To fix it: reduce the learning rate to around 1×10⁻⁵, train for 1–2 epochs maximum, and add a data mix — include 5–10% of a general instruction dataset (e.g., ShareGPT or FLAN subset) alongside the medical data. Alternatively, switch to LoRA/QLoRA, which freezes the base weights and prevents forgetting structurally. Evaluate on a held-out general benchmark (e.g., MMLU or MT-Bench) alongside the domain-specific eval to track both capabilities simultaneously.

## A Complete SFT Training Loop

Below is a from-scratch SFT training loop in PyTorch using the HuggingFace `transformers` and `datasets` libraries. It implements: response-only loss masking, gradient accumulation, learning rate warmup and cosine decay, checkpoint saving, and W&B logging hooks.

```python
"""
sft_train.py — A production-quality SFT training loop.
Trains a causal language model on (instruction, response) pairs
with response-only loss masking.

Usage:
    python sft_train.py \
        --model_name meta-llama/Llama-2-7b-hf \
        --dataset_path ./data/sft_data.jsonl \
        --output_dir ./checkpoints/sft_run1 \
        --num_epochs 3 \
        --batch_size 4 \
        --grad_accum_steps 8 \
        --lr 2e-5
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
import datasets

# ---------------------------------------------------------------------------
# 1. Dataset class with response-only loss masking
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignores this label index


class InstructionDataset(Dataset):
    """
    Loads JSONL with fields {"instruction": str, "response": str}.
    Tokenizes and builds input_ids + labels where instruction tokens
    are masked (IGNORE_INDEX) so only response tokens incur loss.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

        # Load raw data
        raw = datasets.load_dataset("json", data_files=data_path, split="train")
        self.samples = [row for row in raw]

    def _format(self, instruction: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Build input_ids + labels with response-only loss masking.
        We tokenize the prompt and the response SEPARATELY and concatenate
        their ids. This makes the mask boundary exact: there is no
        cross-boundary merge and no off-by-one from an auto-prepended BOS.
        The template contains NO literal <s>/</s> -- the tokenizer adds a
        single BOS to the prompt, and we append EOS by id so the model
        learns to stop.
        """
        prompt = (
            f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n"
            f"{instruction} [/INST]"
        )
        # add_special_tokens=True -> prompt_ids begins with exactly one BOS.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # add_special_tokens=False -> response carries NO BOS of its own.
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [self.tokenizer.eos_token_id]  # teach stopping

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[: self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Verify the mask boundary -- the exact bug class this chapter warns
        # about: prompt fully masked, response labels equal response ids.
        prompt_len = len(prompt_ids)
        assert bool((labels[:prompt_len] == IGNORE_INDEX).all()), "prompt not masked"
        if prompt_len < input_ids.shape[0]:
            assert bool((labels[prompt_len:] == input_ids[prompt_len:]).all()), \
                "response labels must equal response ids"

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        return self._format(row["instruction"], row["response"])


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Pads a batch to the longest sequence in the batch.
    Pads input_ids with pad_token_id, labels with IGNORE_INDEX.
    """
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids_padded, labels_padded, attention_mask = [], [], []
    for item in batch:
        L = item["input_ids"].shape[0]
        pad_len = max_len - L

        # Right-pad input_ids
        input_ids_padded.append(
            F.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
        )
        # Right-pad labels with IGNORE_INDEX so padding doesn't contribute loss
        labels_padded.append(
            F.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX)
        )
        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask.append(
            torch.cat([torch.ones(L, dtype=torch.long),
                       torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
        "attention_mask": torch.stack(attention_mask),
    }


# ---------------------------------------------------------------------------
# 2. Loss function with explicit response masking
# ---------------------------------------------------------------------------

def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the causal language modeling loss over response tokens only.
    logits: (B, L, V)
    labels: (B, L) with IGNORE_INDEX for prompt tokens

    Standard next-token prediction: predict token t from context 0..t-1.
    We shift logits left by one and labels right by one.
    """
    # Shift so that token i predicts token i+1
    shift_logits = logits[:, :-1, :].contiguous()   # (B, L-1, V)
    shift_labels = labels[:, 1:].contiguous()         # (B, L-1)

    # Flatten for cross-entropy; ignore_index silently skips masked positions
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    return loss


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ---- Load model and tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # LLaMA-style models don't have a pad token; use EOS
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,  # use bf16 to save ~50% memory vs fp32
        device_map="auto",           # auto-shards across available GPUs
    )
    model.config.use_cache = False   # disable KV-cache during training

    # ---- Build dataset and dataloader ----
    dataset = InstructionDataset(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # ---- Optimizer and LR scheduler ----
    # Weight decay is typically applied only to weight matrices, not biases/norms
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=args.lr)

    total_steps = (len(dataloader) // args.grad_accum_steps) * args.num_epochs
    warmup_steps = int(0.03 * total_steps)  # 3% warmup, a common heuristic

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Training ----
    global_step = 0
    model.train()

    for epoch in range(args.num_epochs):
        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, L, V)

            # Compute response-only loss
            loss = compute_sft_loss(logits, labels)

            # Scale loss for gradient accumulation
            loss = loss / args.grad_accum_steps
            loss.backward()

            running_loss += loss.item()

            # Optimizer step every grad_accum_steps mini-batches
            if (step + 1) % args.grad_accum_steps == 0:
                # Gradient clipping: prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = running_loss
                running_loss = 0.0

                if global_step % 10 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1}/{args.num_epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e}"
                    )

        # ---- Save checkpoint after each epoch ----
        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# 4. Argument parsing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
```

### Verifying the Loss Mask

Before launching any real run, dump token ids alongside labels for one example and eyeball the boundary. This is the single cheapest way to catch the off-by-one and double-BOS mask bugs that the "Assistant prefix contamination" pitfall below warns about.

```python
# Sanity-check the mask on one example BEFORE launching a run.
ds = InstructionDataset("data/sft_data.jsonl", tokenizer)
ex = ds[0]
ids, labs = ex["input_ids"], ex["labels"]
for tok, lab in zip(ids.tolist(), labs.tolist()):
    piece = tokenizer.convert_ids_to_tokens(tok)
    flag = "MASK" if lab == IGNORE_INDEX else "LOSS"
    print(f"{tok:>6}  {flag}  {piece!r}")

# Expected: ids[0] is the BOS id and it appears exactly once; every prompt
# token is flagged MASK; every response token (including the trailing EOS)
# is flagged LOSS. The unmasked region must decode back to the response:
resp = tokenizer.decode(ids[labs != IGNORE_INDEX])
assert resp.rstrip().endswith(tokenizer.eos_token)
assert (ids == tokenizer.bos_token_id).sum().item() == 1  # no double BOS
```

The expected result: exactly one BOS token at position 0 (never two), every prompt token flagged MASK, every response token including the final EOS flagged LOSS, and `tokenizer.decode` of the unmasked ids reproducing the response text followed by the EOS marker.

### Key Implementation Notes

**Gradient accumulation.** With `batch_size=4` and `grad_accum_steps=8`, the effective batch size is 32. Accumulation is critical for SFT because (a) individual examples vary widely in length, and (b) a larger effective batch reduces gradient noise, which matters for a small dataset.

**BF16 training.** We use `torch_dtype=torch.bfloat16` for the model. BF16 has the same dynamic range as float32 (8 exponent bits) but less precision (7 mantissa bits vs. 23). This is the preferred format for SFT on modern GPUs with bf16 tensor cores — see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

**Gradient clipping.** `clip_grad_norm_(max_norm=1.0)` is standard. SFT on a small dataset can produce occasional large gradients (long responses, unusual tokens), and clipping prevents loss spikes.

**Disabling the KV cache.** `model.config.use_cache = False` is required during training; the KV cache is only useful during autoregressive inference.

!!! example "Computing effective batch size and tokens per second"
    Suppose we train a 7B model on a dataset of 10,000 examples with an average length of 512 tokens. We use:
    - Physical batch size: 4 sequences
    - Gradient accumulation: 8 steps
    - Effective batch size: 4 × 8 = 32 sequences = 32 × 512 ≈ 16,384 tokens per optimizer step

    Total tokens in the dataset: 10,000 × 512 = 5,120,000 tokens.
    For 3 epochs: 15,360,000 total tokens processed.

    On a single A100 80GB GPU, a 7B model in bf16 achieves roughly 10,000–20,000 tokens/second during forward+backward (very roughly). At 15,000 tokens/second: ~1,024 seconds ≈ 17 minutes per epoch, or about 51 minutes for 3 epochs. (Real numbers depend heavily on sequence packing efficiency — see [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html).)

    Number of optimizer steps ≈ 15,360,000 / 16,384 ≈ 937 steps.
    With warmup_steps = 3% × 937 ≈ 28 warmup steps, the learning rate climbs linearly for the first 28 steps then follows a cosine decay.

## Evaluating SFT Models

Evaluating instruction-following quality is genuinely hard. There is no single-number metric that captures all the dimensions we care about. The standard suite includes:

**MT-Bench (Zheng et al., 2023).** An 80-question multi-turn benchmark covering reasoning, math, coding, writing, and roleplay. Answers are scored by a judge LLM (typically GPT-4 or a reward model) on a 1–10 scale. MT-Bench is widely reported and gives a good first-pass quality signal.

**AlpacaEval.** A single-turn benchmark that compares model responses to a reference (GPT-4 or Davinci-003) using win-rate from an LLM judge. Quick to run, good for iteration.

**MMLU (Hendrycks et al.).** Multiple-choice knowledge benchmark across 57 subjects. Good for measuring whether SFT caused catastrophic forgetting on knowledge tasks — an MMLU drop post-SFT signals over-training.

**Perplexity on a held-out pretraining slice.** A quick internal signal: if perplexity on general web text rises sharply after SFT, the model has drifted too far from the pretraining distribution.

**Human preference evaluation.** The gold standard. Present pairs of model outputs to annotators and collect preference votes. Slow and expensive, but essential for production models.

!!! tip "Practitioner tip"
    Run MT-Bench and MMLU every checkpoint. Set an MMLU floor — for example, require that MMLU accuracy does not drop by more than 1.5 percentage points from the base model baseline. This catches catastrophic forgetting early without requiring human evaluators.

## Common Pitfalls and Best Practices

**Overfitting short responses.** If many of your training examples are short, the model will learn to produce short responses even when length is appropriate. Ensure your dataset has a healthy distribution of response lengths.

**Assistant prefix contamination.** Some tokenizers and templates include an "Assistant:" prefix in the prompt. If this is tokenized inconsistently (sometimes included in the instruction mask, sometimes in the response labels), the model will produce garbage outputs at inference. Always verify your mask boundaries explicitly by printing token IDs alongside labels.

**Data ordering.** Shuffling the dataset is essential. If you accidentally train epoch 1 on simple examples and epoch 2 on complex ones, the model will appear to "learn" during epoch 1 but regress during epoch 2. Use `shuffle=True` in the DataLoader.

**Tokenizer mismatch.** Always use the same tokenizer and special tokens that the base model was pretrained with. Replacing the tokenizer or adding new special tokens requires embedding re-initialization and substantially more training.

**Chat template parity.** At inference time, apply exactly the same chat template you used at training time. A common mistake is training with the LLaMA-2 Alpaca template but inferring with the LLaMA-2 Chat template — the model will produce incoherent outputs.

!!! warning "The length bias trap"
    SFT on a dataset where "correct" responses are consistently long will produce a model that gives verbose answers even when brevity is preferred. This is a form of shortcut learning: the model learns that long responses reduce training loss (because long responses contain more plausible next tokens). Prefer a response length distribution that matches your target use case, and consider length-normalizing your loss.

!!! sota "State of the Art & Resources (2026)"
    Instruction tuning has matured into a well-understood first stage of the post-training pipeline: the field has converged on response-only loss masking, curated data over raw quantity (the LIMA finding), and parameter-efficient adapters (LoRA/QLoRA) as the default compute strategy. Current frontier work focuses on data curation at scale, verifiable-reward RL layered on top of SFT, and fully open replication of the entire post-training stack.

    **Foundational work**

    - [Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN, 2022)](https://arxiv.org/abs/2109.01652) — the paper that established instruction tuning as a general paradigm, showing task diversity drives zero-shot generalization.
    - [Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions* (2023)](https://arxiv.org/abs/2212.10560) — the bootstrap recipe behind Alpaca and countless derivative datasets: use the model itself to generate training data.
    - [Zhou et al., *LIMA: Less Is More for Alignment* (2023)](https://arxiv.org/abs/2305.11206) — 1,000 carefully curated examples beat hundreds of thousands of noisy ones; introduced the superficial alignment hypothesis.

    **Recent advances (2023–2026)**

    - [Chung et al., *Scaling Instruction-Finetuned Language Models* (FLAN-T5/v2, 2024)](https://arxiv.org/abs/2210.11416) — systematic study of how task count, model scale, and chain-of-thought data interact during instruction tuning.
    - [Xu et al., *WizardLM: Empowering Large Language Models to Follow Complex Instructions* (2024)](https://arxiv.org/abs/2304.12244) — Evol-Instruct: automatically rewrite seed instructions to progressively higher complexity, improving instruction-following on hard tasks.
    - [Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model Post-Training* (2024)](https://arxiv.org/abs/2411.15124) — fully open SFT → DPO → RLVR recipe from AllenAI that matches or exceeds proprietary fine-tuned models; includes training data, code, and evals.
    - [Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023)](https://arxiv.org/abs/2306.05685) — introduced MT-Bench, the standard multi-turn benchmark for evaluating instruction-tuned models.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — HuggingFace's post-training library; `SFTTrainer` is the most-used one-stop SFT entry point in the ecosystem.
    - [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — unified fine-tuning of 100+ LLMs with a web UI; supports SFT, DPO, GRPO, LoRA, QLoRA, and full fine-tuning (ACL 2024).
    - [OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) — YAML-driven fine-tuning framework with Flash Attention, sequence parallelism, and multi-GPU support; popular for research runs.
    - [allenai/open-instruct](https://github.com/allenai/open-instruct) — AllenAI's fully open post-training codebase backing the Tulu series; covers SFT, DPO, and RLVR end-to-end.

## Further Reading

- Wei et al., "Finetuned Language Models Are Zero-Shot Learners" (FLAN), ICLR 2022.
- Taori et al., "Alpaca: A Strong, Replicable Instruction-Following Model," Stanford CRFM blog, 2023.
- Chung et al., "Scaling Instruction-Finetuned Language Models" (FLAN-T5/FLAN-v2), JMLR 2024.
- Zhou et al., "LIMA: Less Is More for Alignment," NeurIPS 2023.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
- Wang et al., "Self-Instruct: Aligning Language Models with Self-Generated Instructions," ACL 2023.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023.
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC), PNAS 2017.
- OpenHermes-2.5 dataset and model by Teknium, available on HuggingFace Hub, 2023.

!!! key "Key Takeaways"
    - SFT converts a base model into an instruction follower by training on (instruction, response) pairs with **response-only loss masking** — the model is only penalized for poor replies, not for re-predicting the input.
    - The SFT objective is identical to pretraining NLL; the only changes are data format and the loss mask.
    - **Data quality dominates data quantity** (the LIMA finding): 1k high-quality examples often outperforms 100k noisy ones because the base model already contains the knowledge; SFT teaches access and format.
    - Landmark datasets — FLAN, Alpaca, ShareGPT, OpenHermes — each introduced a key insight: task diversity, cheap synthesis, multi-turn realism, and quality curation respectively.
    - SFT is Stage 1 of the three-stage recipe: SFT → Reward Modeling → RL alignment. A strong SFT model is a prerequisite for stable and effective RLHF/DPO.
    - **Catastrophic forgetting** is the main risk: mitigate with low learning rates (~1–5 × 10⁻⁵), short training (1–3 epochs), data mixing, and/or LoRA.
    - LoRA/QLoRA freeze base weights structurally, preventing forgetting and reducing GPU memory from ~120 GB (full 7B) to ~12–15 GB — enabling SFT on a single consumer GPU.
    - Always evaluate SFT models on both a capability benchmark (MT-Bench) and a forgetting probe (MMLU delta from base) simultaneously.
    - At inference time, apply exactly the chat template used during training — template mismatch is the most common cause of degraded SFT model outputs in production.
