# 5.1 Supervised Fine-Tuning & Instruction Tuning

A freshly pretrained language model is a remarkable thing: it has absorbed syntax, facts, reasoning patterns, and writing styles from hundreds of billions of tokens of text. And yet if you give it the prompt "Explain quantum entanglement to a 10-year-old," a raw base model is just as likely to continue the sentence with another question, an unrelated anecdote, or a bibliographic citation as it is to produce a helpful explanation. The model has learned *language*; it has not learned to be *helpful*.

This is the problem supervised fine-tuning (SFT) solves. SFT teaches the model to follow instructions, adopt a conversational format, and suppress unhelpful completions — all by showing it examples of the behavior we want. In this chapter we work through the mechanics of SFT from first principles: the objective, the data, the full-versus-partial finetuning spectrum, the risk of catastrophic forgetting, and the three-stage post-training recipe that is now standard across frontier labs. We include a complete, runnable SFT training loop.

SFT is the first stage of the alignment pipeline. It is followed by preference learning ([The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)) and optionally by policy-optimization steps ([Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) and [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)). But none of those later stages work well without a solid SFT foundation. For this chapter's machinery applied end to end on a model you can actually train yourself, see [Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M](../14-capstone/09-post-training.html), which runs the SFT stage of the book's 100M-parameter capstone model.

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

**How this scales down.** LIMA's thousand examples worked on a 65B base whose capability only needed *unlocking*. A ~100M base has far less latent skill to surface, so its SFT stage is mostly about installing the chat format, turn-taking, and stopping behavior reliably — which needs more repetitions than a 65B model does, even though the curation bar is unchanged. The capstone budgets roughly 100k curated conversations (a SmolTalk-style public mix) for three epochs at a peak LR of $2 \times 10^{-5}$; every other hyperparameter in this chapter transfers directly. See [Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M](../14-capstone/09-post-training.html).

## Data Quality > Quantity: Practical Data Engineering

Given the LIMA insight, how do you build high-quality SFT data in practice?

**Instruction diversity.** Cluster your instructions (e.g., by embedding them with a sentence encoder) and measure coverage. An instruction dataset that is 70% "write a Python function to..." is not diverse and will produce a lopsided model.

**Response quality filters.** Use heuristics (minimum length, no refusal boilerplate, no truncations) and reward model scoring. If you already have a reward model (chapter [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)), filter to the top-scoring responses per instruction cluster.

**Format consistency.** All examples should follow the same chat template. Mixed templates in a training batch cause the model to learn an inconsistent format.

**Deduplication.** Near-duplicate instructions with slightly different phrasings inflate dataset size while contributing almost no new learning signal. Min-hash or embedding-based deduplication is standard; see [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html).

**Long-tail coverage.** Ensure examples cover rare but important topics: safety refusals, citations/uncertainty, multi-hop reasoning, code debugging. These are underrepresented in organic data but disproportionately important for the model's edge-case behavior.

**The tools that do this.** None of the above needs custom code. Embed and cluster instructions with `sentence-transformers` (a MiniLM or BGE encoder) plus `scikit-learn` k-means or `faiss` for nearest-neighbour dedup; run MinHash-LSH near-duplicate removal with `huggingface/datatrove` or `ChenghaoMou/text-dedup`; generate, evolve, and LLM-judge-filter synthetic instructions with `argilla-io/distilabel` (which implements Self-Instruct, Evol-Instruct, and UltraFeedback-style pipelines as composable steps) and review the survivors in `argilla`. Store and version the resulting mix as a HuggingFace `datasets` `Dataset` so the exact revision is reproducible — the same discipline as the pretraining pipeline in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html).

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

    (This accounting assumes gradients are kept in fp32 and omits the fp32 *master* copy of the weights that some mixed-precision setups also hold, which would add another ~28 GB. Activation checkpointing trades roughly 30% extra compute for most of the activation term — see [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).)

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
from typing import Dict, List, Optional, Tuple

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

def compute_sft_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """
    Causal-LM loss over response tokens only.
    logits: (B, L, V)
    labels: (B, L) with IGNORE_INDEX for prompt tokens

    Standard next-token prediction: predict token t from context 0..t-1.
    We shift logits left by one and labels right by one.

    Returns (loss_SUM, n_response_tokens) rather than a mean. The sum is what
    lets gradient accumulation normalize by the TRUE token count of the whole
    accumulation window -- see "Loss normalization under gradient
    accumulation" below. Returning the count makes the denominator explicit.
    """
    # Shift so that token i predicts token i+1
    shift_logits = logits[:, :-1, :].contiguous()   # (B, L-1, V)
    shift_labels = labels[:, 1:].contiguous()         # (B, L-1)

    n_tokens = int((shift_labels != IGNORE_INDEX).sum().item())

    # Flatten for cross-entropy; ignore_index silently skips masked positions
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum, n_tokens


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
        # Running sums for the CURRENT accumulation window.
        window_loss, window_tokens = 0.0, 0

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

            # Compute response-only loss as a SUM plus its token count
            loss_sum, n_tokens = compute_sft_loss(logits, labels)

            if n_tokens > 0:
                # Backward on the UNNORMALIZED sum: gradients accumulate as
                # sums too, and we rescale ONCE at the accumulation boundary.
                # (A microbatch with zero response tokens would give 0/0.)
                loss_sum.backward()
                window_loss += loss_sum.item()
                window_tokens += n_tokens

            # Optimizer step every grad_accum_steps mini-batches
            if (step + 1) % args.grad_accum_steps == 0:
                if window_tokens > 0:
                    # Divide the accumulated gradient by the window's true
                    # response-token count -> the exact per-token mean
                    # gradient over the effective batch.
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(window_tokens)

                    # Gradient clipping: prevents exploding gradients.
                    # Done AFTER rescaling, so max_norm=1.0 means what it says.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    if global_step % 10 == 0:
                        avg_loss = window_loss / window_tokens  # nats/token
                        lr_now = scheduler.get_last_lr()[0]
                        print(
                            f"Epoch {epoch+1}/{args.num_epochs} | "
                            f"Step {global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e}"
                        )

                optimizer.zero_grad(set_to_none=True)
                window_loss, window_tokens = 0.0, 0

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

**Loss normalization under gradient accumulation.** This is the subtlest correctness issue in the whole loop, and the reason `compute_sft_loss` returns a sum rather than a mean. The naive implementation computes `F.cross_entropy(..., reduction="mean")` — the mean over *this microbatch's* response tokens — and divides by `grad_accum_steps`. That weights every microbatch equally, so a microbatch holding 40 response tokens contributes as much gradient as one holding 900. The gradient you take is then a *mean of means*, $\frac{1}{G}\sum_g \frac{L_g}{n_g}$, not the gradient of the loss over the accumulation window, $\frac{\sum_g L_g}{\sum_g n_g}$ — and changing `grad_accum_steps` silently changes the objective. Response-only masking is exactly what makes $n_g$ vary wildly, so SFT suffers far more than pretraining (where every packed window has the same number of targets). This is the gradient-accumulation normalization bug that HuggingFace and Unsloth publicized in late 2024 and subsequently fixed across `transformers` and TRL. The fix above accumulates unnormalized sums and divides the accumulated *gradient* by the window's true token count — exact, single-pass, no extra forward. Backpropagating an unnormalized sum makes raw gradients on the order of $10^3\times$ larger than a mean's, which is safe here because bf16 carries fp32's dynamic range, and we rescale *before* clipping so `max_norm=1.0` still means what it says. Under DDP, all-reduce `window_tokens` across ranks and divide by the global total instead, since DDP averages gradients across ranks. The same fix, with the same reasoning, appears in the capstone's SFT loop ([Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M](../14-capstone/09-post-training.html)).

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

### The Same Run in TRL

You should write the loop above once, to know what every line does. For production you use a library, and the ecosystem default is HuggingFace **TRL** (`huggingface/trl`), whose `SFTTrainer` wraps `transformers.Trainer` with the masking, packing, PEFT, and distributed plumbing already correct — including the loss normalization discussed above. The entire script becomes:

```python
# pip install "trl>=0.15" transformers datasets peft
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# A dataset with "prompt"/"completion" columns is a *prompt-completion*
# dataset: TRL builds the mask for us, so no manual IGNORE_INDEX bookkeeping.
ds = load_dataset("json", data_files="data/sft_data.jsonl", split="train")
ds = ds.rename_columns({"instruction": "prompt", "response": "completion"})

cfg = SFTConfig(
    output_dir="./checkpoints/sft_trl",
    max_length=2048,
    completion_only_loss=True,      # response-only masking (our label mask)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # effective batch = 32 sequences
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=10,
)

trainer = SFTTrainer(
    model="meta-llama/Llama-2-7b-hf",
    args=cfg,
    train_dataset=ds,
    # Drop peft_config for full fine-tuning; keep it for LoRA/QLoRA.
    peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM"),
)
trainer.train()
```

The mapping is one-to-one with what we built: `completion_only_loss` is our label mask, `gradient_accumulation_steps` is our accumulation window, `peft_config` is the LoRA branch of the memory table. Two flags worth knowing beyond this snippet: for *conversational* datasets (a `messages` column of role/content turns) use `assistant_only_loss=True` instead, which masks every non-assistant turn but requires the tokenizer's chat template to wrap assistant content in a `{% generation %}` block — if the template lacks it, the flag silently trains on everything. And `packing=True` concatenates examples to fill `max_length`, the single biggest throughput win for short SFT data; both are covered in [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html), and TRL itself in [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html). Argument names do move between TRL releases — read the `SFTConfig` dataclass in the version you install rather than trusting a snippet. Above TRL sit config-driven wrappers that need no Python at all: **axolotl** (YAML), **LLaMA-Factory** (YAML + web UI), **Unsloth** (fused Triton kernels for single-GPU LoRA), and **allenai/open-instruct** (the Tulu recipes end to end). For multi-GPU full fine-tuning, TRL delegates sharding to Accelerate with DeepSpeed ZeRO-3 or PyTorch FSDP — see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

## Evaluating SFT Models

Evaluating instruction-following quality is genuinely hard. There is no single-number metric that captures all the dimensions we care about. The standard suite includes:

**IFEval (Zhou et al., 2023).** *Verifiable* instruction following: prompts carry programmatically checkable constraints ("write at least 400 words," "respond in all lowercase," "wrap your answer in double quotes"), and a script checks compliance. No judge model, no judge bias, no cost — which makes it the cheapest and most reliable regression test for whether SFT actually taught constraint-following rather than surface style. Run it first on every checkpoint.

**MT-Bench (Zheng et al., 2023).** An 80-question multi-turn benchmark covering reasoning, math, coding, writing, and roleplay. Answers are scored by a judge LLM (originally GPT-4, now typically a strong frontier model or a reward model) on a 1–10 scale. Historically the standard first-pass quality signal, but it saturates: strong models cluster near the ceiling, so it now discriminates poorly at the top and is best used as a floor check.

**AlpacaEval.** A single-turn benchmark that compares model responses to a reference using win-rate from an LLM judge. Report the **length-controlled** win rate of AlpacaEval 2.0 (Dubois et al., 2024), which regresses out the judge's well-documented preference for longer answers — an uncontrolled win rate rewards exactly the length bias this chapter warns about. **Arena-Hard-Auto** (LMSYS) is the harder companion, built from difficult real Chatbot Arena prompts and designed to correlate with human Arena rankings. See [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html) for judge bias and how to calibrate around it.

**MMLU (Hendrycks et al.).** Multiple-choice knowledge benchmark across 57 subjects. Good for measuring whether SFT caused catastrophic forgetting on knowledge tasks — an MMLU drop post-SFT signals over-training. Run it (and GSM8K, HellaSwag, ARC, and the harder MMLU-Pro) with EleutherAI's **`lm-evaluation-harness`**, the de-facto standard runner: `lm_eval --model hf --model_args pretrained=./checkpoints/sft_run1 --tasks mmlu,gsm8k,ifeval --batch_size auto`. Pinning the harness version matters — prompt formats and normalization change between releases, and a "regression" is often just a harness bump. See [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html).

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
    Instruction tuning has matured into a well-understood first stage of the post-training pipeline: the field has converged on response-only loss masking, curated data over raw quantity (the LIMA finding), and parameter-efficient adapters (LoRA/QLoRA) as the default compute strategy. Current frontier work focuses on data curation at scale, verifiable-reward RL layered on top of SFT, distilling long chain-of-thought reasoning traces into smaller models via plain SFT, and fully open replication of the entire post-training stack.

    **Foundational work**

    - [Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN, 2022)](https://arxiv.org/abs/2109.01652) — the paper that established instruction tuning as a general paradigm, showing task diversity drives zero-shot generalization.
    - [Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions* (2023)](https://arxiv.org/abs/2212.10560) — the bootstrap recipe behind Alpaca and countless derivative datasets: use the model itself to generate training data.
    - [Zhou et al., *LIMA: Less Is More for Alignment* (2023)](https://arxiv.org/abs/2305.11206) — 1,000 carefully curated examples beat hundreds of thousands of noisy ones; introduced the superficial alignment hypothesis.

    **Recent advances (2023–2026)**

    - [Chung et al., *Scaling Instruction-Finetuned Language Models* (FLAN-T5/v2, 2024)](https://arxiv.org/abs/2210.11416) — systematic study of how task count, model scale, and chain-of-thought data interact during instruction tuning.
    - [Xu et al., *WizardLM: Empowering Large Language Models to Follow Complex Instructions* (2024)](https://arxiv.org/abs/2304.12244) — Evol-Instruct: automatically rewrite seed instructions to progressively higher complexity, improving instruction-following on hard tasks.
    - [Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model Post-Training* (2024)](https://arxiv.org/abs/2411.15124) — fully open SFT → DPO → RLVR recipe from AllenAI that matches or exceeds proprietary fine-tuned models; includes training data, code, and evals.
    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — showed that plain SFT on reasoning traces distilled from a strong RL-trained teacher gives smaller dense models (1.5B–70B) most of the teacher's reasoning ability, making trace distillation a mainstream SFT recipe.
    - [Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023)](https://arxiv.org/abs/2306.05685) — introduced MT-Bench, the standard multi-turn benchmark for evaluating instruction-tuned models.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — HuggingFace's post-training library; `SFTTrainer` is the most-used one-stop SFT entry point in the ecosystem.
    - [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — unified fine-tuning of 100+ LLMs with a web UI; supports SFT, DPO, GRPO, LoRA, QLoRA, and full fine-tuning (ACL 2024).
    - [axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl) — YAML-driven fine-tuning framework with Flash Attention, multi-GPU support, and (as of 2025) GRPO and multimodal fine-tuning; popular for research runs.
    - [allenai/open-instruct](https://github.com/allenai/open-instruct) — AllenAI's fully open post-training codebase backing the Tulu series; covers SFT, DPO, and RLVR end-to-end.
    - [unslothai/unsloth](https://github.com/unslothai/unsloth) — fused Triton kernels for single-GPU LoRA/QLoRA SFT; drop-in with TRL's `SFTTrainer` and the fastest path to fine-tuning on one consumer card.
    - [argilla-io/distilabel](https://github.com/argilla-io/distilabel) — composable pipelines for synthesizing, evolving, and LLM-judge-filtering instruction data (Self-Instruct, Evol-Instruct, UltraFeedback as reusable steps).
    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the standard runner for MMLU/GSM8K/IFEval and the forgetting probes you should gate every SFT checkpoint on.

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
    - The SFT objective is identical to pretraining NLL; the only changes are data format and the loss mask. Normalize by the **total response-token count of the whole accumulation window**, not per microbatch — the mean-of-means shortcut is the classic gradient-accumulation bug, and response-only masking (which makes per-microbatch token counts vary wildly) is what makes it bite.
    - **Data quality dominates data quantity** (the LIMA finding): 1k high-quality examples often outperforms 100k noisy ones because the base model already contains the knowledge; SFT teaches access and format.
    - Landmark datasets — FLAN, Alpaca, ShareGPT, OpenHermes — each introduced a key insight: task diversity, cheap synthesis, multi-turn realism, and quality curation respectively.
    - SFT is Stage 1 of the three-stage recipe: SFT → Reward Modeling → RL alignment. A strong SFT model is a prerequisite for stable and effective RLHF/DPO.
    - **Catastrophic forgetting** is the main risk: mitigate with low learning rates (~1–5 × 10⁻⁵), short training (1–3 epochs), data mixing, and/or LoRA.
    - LoRA/QLoRA freeze base weights structurally, preventing forgetting and reducing GPU memory from ~120 GB (full 7B) to ~12–15 GB — enabling SFT on a single consumer GPU.
    - Always evaluate SFT models on both a capability benchmark and a forgetting probe simultaneously: verifiable IFEval plus a length-controlled AlpacaEval 2.0 / Arena-Hard win rate for quality, MMLU delta from base for forgetting — all runnable from a pinned `lm-evaluation-harness`.
    - Write the training loop once to understand it, then run production SFT through **TRL**'s `SFTTrainer` (`completion_only_loss` / `assistant_only_loss`, `packing`, `peft_config`) or a YAML wrapper over it (axolotl, LLaMA-Factory, Unsloth, open-instruct) — and at inference apply exactly the chat template used during training, since template mismatch is the most common cause of degraded SFT outputs in production.

## Exercises

**1.** *(Response-only masking.)* The SFT objective masks the instruction tokens and, in the chapter's training code, appends an EOS token to every response before computing the loss. (a) Give the two reasons the chapter offers for masking the instruction rather than training on the full concatenated sequence. (b) Suppose you removed the line `response_ids = response_ids + [self.tokenizer.eos_token_id]` so responses no longer end in EOS. What failure would you expect at inference time, and why?

??? note "Solution"
    **(a)** The chapter gives two reasons to prefer response-only supervision:

    - *Signal concentration.* Instructions are often short while responses are longer. Masking the instruction gives the optimizer a gradient that specifically rewards reply quality instead of spending capacity re-encoding the input.
    - *Prompt contamination.* If the model is penalized for "wrong" instruction tokens, it can learn to prefer particular prompt phrasings in ways that generalize poorly.

    (In practice the gap is small for well-formatted data, but response masking is the standard convention.)

    **(b)** The EOS token is what teaches the model to *stop*. Its label is one of the response tokens that incur loss, so training on it makes $p_\theta(\text{EOS} \mid \text{full response})$ large — the model learns that a completed answer is followed by end-of-sequence. If you never include EOS in the labels, the model is never supervised to emit it at the end of an answer. At inference the decoder has no learned signal to terminate, so generation runs on past the natural end of the reply — rambling, repeating, or drifting into a new turn until it hits the max-token cap. This is the runaway-generation failure.

**2.** *(Learning-rate scaling.)* The chapter says SFT learning rates are "one to two orders of magnitude below pretraining rates." If a model was pretrained with a peak learning rate of $3 \times 10^{-4}$, what SFT peak-LR range does that rule imply? How does it compare to the concrete range the chapter recommends, and why is a low LR the first-line defense against catastrophic forgetting?

??? note "Solution"
    "One order of magnitude below" means dividing by 10; "two orders" means dividing by 100. Applied to $3 \times 10^{-4}$:

    $$
    \frac{3 \times 10^{-4}}{10} = 3 \times 10^{-5}, \qquad \frac{3 \times 10^{-4}}{100} = 3 \times 10^{-6}.
    $$

    So the rule implies a peak SFT LR roughly in $[3 \times 10^{-6},\ 3 \times 10^{-5}]$. This overlaps the chapter's concrete recommendation of $1 \times 10^{-5}$ to $5 \times 10^{-5}$ (the upper end sits just above the "one order down" figure, which is fine — these are heuristics).

    A low LR is the first-line defense against catastrophic forgetting because pretraining leaves the weights in a configuration that is jointly good for many tasks, and SFT data is a narrow slice of that space. The size of each gradient step scales with the LR, so a small LR keeps updates small and prevents the optimizer from moving far enough to overwrite the pretraining solution in favor of the narrow SFT target.

**3.** *(Effective batch size, tokens, and schedule.)* You run the chapter's `sft_train.py` on a dataset of 20,000 examples averaging 500 tokens each, with `--batch_size 4`, `--grad_accum_steps 8`, and `--num_epochs 3`. Compute: (a) the effective batch size in sequences and in tokens; (b) the total number of tokens processed over the whole run; (c) the number of optimizer steps; (d) the number of warmup steps under the code's 3% warmup heuristic.

??? note "Solution"
    **(a)** Effective batch size in sequences is `batch_size × grad_accum_steps`:

    $$
    4 \times 8 = 32 \text{ sequences.}
    $$

    In tokens, at 500 tokens/sequence: $32 \times 500 = 16{,}000$ tokens per optimizer step.

    **(b)** Tokens in one epoch: $20{,}000 \times 500 = 10{,}000{,}000$. Over 3 epochs:

    $$
    3 \times 10{,}000{,}000 = 30{,}000{,}000 \text{ tokens.}
    $$

    **(c)** Optimizer steps = total tokens / tokens-per-step:

    $$
    \frac{30{,}000{,}000}{16{,}000} = 1{,}875 \text{ steps.}
    $$

    (Cross-check: steps per epoch $= 20{,}000 / 32 = 625$, and $625 \times 3 = 1{,}875$.)

    **(d)** Warmup at 3% of total steps:

    $$
    0.03 \times 1{,}875 = 56.25 \approx 56 \text{ warmup steps}
    $$

    (the code takes `int(0.03 * total_steps)`, which truncates to 56). The LR climbs linearly for ~56 steps, then follows cosine decay for the remaining ~1,819.

**4.** *(Memory budget.)* Using the chapter's per-parameter accounting (bf16 weights, fp32 gradients, and AdamW's two fp32 moments), estimate the memory for full fine-tuning a **13B**-parameter model, ignoring activations. Then estimate the frozen-base memory for QLoRA (4-bit NF4). What reduction factor does QLoRA give on the base-storage line alone, and roughly how many 80 GB A100s does the full run need?

??? note "Solution"
    Take $N = 13 \times 10^{9}$ parameters and use the chapter's byte counts.

    **Full fine-tuning (weights + grads + optimizer):**

    - Weights, bf16: $13\text{e}9 \times 2 = 26$ GB
    - Gradients, fp32: $13\text{e}9 \times 4 = 52$ GB
    - AdamW $m_1 + m_2$, fp32: $13\text{e}9 \times 2 \times 4 = 104$ GB

    $$
    26 + 52 + 104 = 182 \text{ GB (before activations).}
    $$

    Adding a realistic ~15–25 GB of activations pushes the run to roughly 200 GB, so it needs $\lceil 200 / 80 \rceil = 3$ — realistically **3–4 × 80 GB A100s** once you leave headroom.

    **QLoRA frozen base (4-bit NF4 = 0.5 bytes/param):**

    $$
    13\text{e}9 \times 0.5 = 6.5 \text{ GB.}
    $$

    The LoRA adapters and their optimizer states add only ~1 GB (they touch a fraction of a percent of parameters), so the whole run fits comfortably on a single 24 GB GPU.

    **Reduction factor on base storage alone:** comparing the 26 GB bf16 base to the 6.5 GB quantized base gives $26 / 6.5 = 4\times$. Comparing the *full* training footprint (182 GB) to the quantized base (6.5 GB) is about $28\times$ — which is why QLoRA moves a job that needed a multi-GPU node onto one consumer card.

**5.** *(Implement length-normalized loss.)* The chapter's `compute_sft_loss` returns a *sum* over response tokens plus the token count, which the training loop turns into an exact per-**token** mean over the accumulation window. The "length bias trap" warning notes that this lets long responses dominate. Rewrite the function so that each *sequence* contributes equally regardless of its response length: compute a per-token loss, average it *within* each example over that example's own response tokens, then return the result in the same `(sum, count)` form the loop expects so it drops in unchanged. Explain in one line how the gradient weighting differs from the original.

??? note "Solution"
    Use `reduction="none"` to get a per-token loss, reshape to `(B, L-1)`, build a mask from the non-ignored labels, and reduce in two stages:

    ```python
    def compute_sft_loss_length_normalized(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Per-sequence-mean SFT loss: each example weighted equally,
        regardless of how many response tokens it has. Returns
        (sum over sequences, number of sequences) so the chapter's
        accumulation loop -- which divides the accumulated gradient by the
        accumulated count -- yields an exact per-SEQUENCE mean."""
        shift_logits = logits[:, :-1, :].contiguous()   # (B, L-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, L-1)
        B = shift_labels.size(0)

        # Per-token loss, no reduction. Masked positions still produce a
        # value here, so we zero them out with the mask below.
        per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(B, -1)                                    # (B, L-1)

        mask = (shift_labels != IGNORE_INDEX).float()    # (B, L-1)
        # Response-token count per example; clamp avoids divide-by-zero
        # for a (degenerate) all-masked row.
        seq_counts = mask.sum(dim=1).clamp(min=1.0)      # (B,)
        seq_loss = (per_token * mask).sum(dim=1) / seq_counts  # (B,)
        return seq_loss.sum(), B                         # (sum, count)
    ```

    Note that `ignore_index` already forces the loss at masked positions to 0 in the `reduction="none"` output, but multiplying by `mask` before the per-sequence sum is what makes the denominator (`seq_counts`) match the numerator exactly.

    **Gradient weighting difference:** the original token-mean divides by the *total* response-token count in the window, so a 400-token answer contributes ~8x the gradient of a 50-token answer; the per-sequence version gives every example weight $1/B$, so short and long responses pull the update equally — which is what removes the length bias. Note the loop needs no change: just accumulate the returned count (now a sequence count, not a token count) and divide the gradient by it at the accumulation boundary, exactly as before.

**6.** *(Implement replay / data mixing.)* One mitigation for catastrophic forgetting is to blend a small fraction (the chapter suggests 5–10%) of general or pretraining-style data back into the SFT mix. Write a `Dataset` wrapper `ReplayMixedDataset(primary, replay, replay_frac)` that presents a shuffled blend in which the replay examples make up `replay_frac` of the total, drawing replay items with replacement if there are too few. It must be drop-in compatible with the chapter's `DataLoader`/`collate_fn`. What must be true about how the `replay` examples are formatted for the mix to be valid?

??? note "Solution"
    The wrapper builds an index of `(source, local_index)` pairs — all of the primary items plus enough replay items to hit the target fraction — then shuffles once. Each `__getitem__` delegates to the underlying dataset, so every returned item is the same `{"input_ids", "labels"}` dict the existing `collate_fn` expects.

    Solving `n_replay / (n_primary + n_replay) = replay_frac` for the replay count gives $n_\text{replay} = \dfrac{\texttt{replay\_frac}}{1 - \texttt{replay\_frac}} \, n_\text{primary}$.

    ```python
    import random
    from torch.utils.data import Dataset

    class ReplayMixedDataset(Dataset):
        """Blend a `replay_frac` fraction of general/pretraining-style
        examples into the SFT set to mitigate catastrophic forgetting.
        Both `primary` and `replay` must yield the same
        {"input_ids", "labels"} dicts (e.g. InstructionDataset instances)."""

        def __init__(self, primary, replay, replay_frac=0.1, seed=0):
            assert 0.0 <= replay_frac < 1.0
            self.primary, self.replay = primary, replay
            n_primary = len(primary)
            # n_replay / (n_primary + n_replay) == replay_frac
            n_replay = int(round(n_primary * replay_frac / (1.0 - replay_frac)))

            rng = random.Random(seed)
            self.index = [("p", i) for i in range(n_primary)]
            # Sample replay WITH replacement so a small replay pool still
            # fills the quota; each SFT example still appears exactly once.
            self.index += [("r", rng.randrange(len(replay)))
                           for _ in range(n_replay)]
            rng.shuffle(self.index)

        def __len__(self):
            return len(self.index)

        def __getitem__(self, idx):
            src, i = self.index[idx]
            return self.primary[i] if src == "p" else self.replay[i]
    ```

    Usage keeps the rest of the chapter's loop unchanged:

    ```python
    sft = InstructionDataset("data/sft_data.jsonl", tokenizer)
    replay = InstructionDataset("data/general_replay.jsonl", tokenizer)
    dataset = ReplayMixedDataset(sft, replay, replay_frac=0.1)
    # dataloader = DataLoader(dataset, ..., collate_fn=lambda b: collate_fn(b, pad_id))
    ```

    **Formatting requirement:** the replay items must be tokenized with the *same tokenizer, chat template, and loss-masking convention* as the primary data, so that mixed batches don't teach an inconsistent format (the chapter's "format consistency" and "chat template parity" points). If the replay source is raw pretraining text rather than instruction pairs, it should be tokenized so that *all* of its tokens are targets (no prompt to mask) — i.e. `labels == input_ids` for those examples — since there is no instruction/response boundary; it still returns the identical dict shape so `collate_fn` and `compute_sft_loss` handle it unchanged.
