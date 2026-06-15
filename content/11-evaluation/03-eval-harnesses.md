# 11.3 Building Eval Harnesses

Every claimed benchmark score has a story behind it: which prompt template, which few-shot examples, whether the answer was extracted from a generation or scored as log-likelihood, how ties were broken, and whether the random seed was fixed. Get any of these wrong and your numbers become incomparable to everyone else's — even on the same dataset. This chapter is the engineering manual for that story.

We cover the two dominant open-source harnesses (lm-evaluation-harness and HELM), dissect the mechanics that separate correct from incorrect evaluations, show you how to build a custom eval from scratch, and close with the statistics you need to know whether a difference in accuracy is real or noise.

Related chapters provide necessary context: [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) surveys what we are measuring, [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html) covers neural judges as an alternative to harness-based scoring, and [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html) extends these ideas to more complex task types.

## Why Harnesses Exist

Imagine benchmarking a new model on MMLU. You write a script, get 72.1% accuracy, and compare it to the leaderboard showing another model at 74.3%. Are you 2.2 points behind? Possibly not. The other result may have been measured with:

- A different prompt template (e.g., "The answer is:" vs. "Answer:")
- 5-shot examples from a fixed seed vs. 0-shot
- Log-likelihood scoring vs. constrained generation
- A different answer-extraction regex
- An evaluation set that had been de-duplicated to remove contamination overlap with the model's training data

All of these choices are legitimate, but they produce incomparable numbers. An **eval harness** is a framework that standardizes every one of these choices for a library of tasks, so that running model A and model B through the same harness produces genuinely comparable scores.

The secondary benefit is velocity: a well-designed harness lets a practitioner add a new task in ~50 lines of YAML/Python rather than re-implementing tokenization, batching, log-likelihood computation, and result aggregation from scratch.

## lm-evaluation-harness: Architecture and Mechanics

The [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (Gao et al.) is the de facto standard for open-model evaluation. It supports hundreds of tasks and is the backend for most open leaderboards, including the Open LLM Leaderboard on Hugging Face.

### High-Level Architecture

{{fig:evalharness-lm-eval-architecture}}

The interface is cleanly separated: the `LM` abstract class exposes `loglikelihood`, `loglikelihood_rolling`, and `generate_until`. Tasks are described declaratively; the harness handles batching, tokenization, and aggregation.

### Installing and Running

```bash
# Install with recommended extras
pip install lm-eval[vllm,wandb]

# Evaluate a model on MMLU and HellaSwag with 5-shot prompting
lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks mmlu,hellaswag \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path results/llama3-8b \
    --log_samples   # save raw predictions for auditing

# Use vLLM backend for large models (much faster)
lm_eval \
    --model vllm \
    --model_args pretrained=meta-llama/Llama-3.1-70B-Instruct,tensor_parallel_size=4 \
    --tasks mmlu_pro \
    --num_fewshot 0 \
    --batch_size 32 \
    --output_path results/llama3-70b
```

### How Log-Likelihood Scoring Works

For multiple-choice tasks like MMLU, the harness does **not** generate text. Instead, for each answer choice $c_i$ it computes the log-likelihood of that choice given the context:

$$
\text{score}(c_i) = \log p_\theta(c_i \mid \text{context})
= \sum_{t=1}^{|c_i|} \log p_\theta(w_t \mid \text{context}, w_{1:t-1})
$$

The predicted answer is the choice with the highest score. This is sometimes called **length-normalized** log-likelihood: because longer answers accumulate more log-probability mass, many harnesses divide by the token length of each choice:

$$
\text{score}_{\text{norm}}(c_i) = \frac{\log p_\theta(c_i \mid \text{context})}{|c_i|_{\text{tokens}}}
$$

The harness exposes `acc` (raw argmax) and `acc_norm` (length-normalized) as separate metrics, and you need to know which one is being reported before comparing to a third-party result.

!!! example "Worked example: MMLU scoring"

    Consider a 4-choice MMLU question with context $x$ and choices:
    - A: "osmosis" (2 tokens)
    - B: "active transport" (3 tokens)
    - C: "diffusion through a lipid bilayer" (7 tokens)
    - D: "endocytosis" (3 tokens)

    Suppose the model assigns (unnormalized) log-likelihoods:
    - $\log p(\text{A} \mid x) = -1.2$
    - $\log p(\text{B} \mid x) = -3.6$
    - $\log p(\text{C} \mid x) = -4.9$
    - $\log p(\text{D} \mid x) = -1.5$

    Without normalization, choice A wins (least negative). After dividing by token count:
    - A: $-1.2 / 2 = -0.60$
    - B: $-3.6 / 3 = -1.20$
    - C: $-4.9 / 7 = -0.70$
    - D: $-1.5 / 3 = -0.50$

    With normalization, choice D wins. If D is correct, `acc` and `acc_norm` disagree, and the reported accuracy will differ by at least this one example. At scale across thousands of questions, these differences can easily amount to 1–3 percentage points on MMLU.

### Generation Scoring

Some tasks require the model to generate text freely, then extract an answer for comparison. Typical examples: GSM8K (math word problems), code generation benchmarks, and open-ended QA.

The harness handles this via `generate_until`, which calls the model's generation endpoint with specified stopping strings, then passes the output to a `process_results` function that extracts the answer and computes the metric.

```python
# Snippet from a typical generation-based task definition
# (equivalent of what lives inside the harness YAML/Task class)

def doc_to_text(doc):
    """Format a single document into a prompt string."""
    return (
        "Question: " + doc["question"] +
        "\nLet's think step by step.\nAnswer:"
    )

def process_results(doc, results):
    """Extract the numeric answer from a chain-of-thought generation."""
    import re
    # results[0] is the generated string
    gen = results[0]
    # Look for the last number in the generation
    matches = re.findall(r"[-+]?\d*\.?\d+", gen)
    if matches:
        predicted = float(matches[-1])
        gold = float(doc["answer"])
        return {"exact_match": predicted == gold}
    return {"exact_match": 0.0}
```

### Few-Shot Formatting

Few-shot examples are prepended to the context. The harness selects examples from a fixed set (typically the training split of the dataset) and formats them identically to the evaluation document. The number of examples is controlled by `--num_fewshot`.

```python
# Pseudocode of what the harness does internally for K-shot
def build_prompt(task, doc, k, fewshot_docs):
    """Build a K-shot prompt for a document."""
    parts = []

    # System prompt (if the task defines one)
    if task.has_system_prompt():
        parts.append(task.system_prompt())

    # K few-shot examples
    for fewshot_doc in fewshot_docs[:k]:
        # doc_to_text formats the question
        # doc_to_target formats the gold answer
        parts.append(task.doc_to_text(fewshot_doc) +
                     task.doc_to_target(fewshot_doc))

    # The actual test document (no answer appended)
    parts.append(task.doc_to_text(doc))

    return task.fewshot_delimiter().join(parts)
```

A critical detail: the **log-likelihood is computed only over the target tokens**, not the full context. The harness implements this by computing $\log p(\text{target} \mid \text{prefix})$, where the prefix includes all few-shot examples plus the question. This matches how humans interpret the task — the model is given context and evaluated on whether it can produce the right completion.

## HELM: The Holistic Evaluation Framework

HELM (Holistic Evaluation of Language Models, Liang et al., 2022) takes a different philosophical stance from lm-evaluation-harness. Rather than maximizing task coverage within a single scoring paradigm, HELM emphasizes:

1. **Multiple metrics per scenario**: accuracy, calibration, robustness to perturbations, fairness across demographic groups, efficiency (tokens used), and toxicity.
2. **Standardized adaptation**: few-shot demonstrations are sampled with a fixed seed, and the prompt format is documented in a machine-readable schema.
3. **Scenario × Adaptation × Metric orthogonality**: you can combine any scenario with any adaptation strategy and get a well-defined result.

### HELM's Three-Layer Structure

{{fig:evalharness-helm-three-layer}}

HELM's output is a structured JSON with a per-model, per-scenario, per-metric score table. This makes it easy to compare models on a specific capability rather than a single aggregate number.

### Running HELM

```bash
# Install HELM
pip install crfm-helm

# Run a subset of scenarios
helm-run \
    --conf-path src/helm/benchmark/presentation/run_specs_lite.conf \
    --suite my_eval \
    --max-eval-instances 500 \
    --num-threads 4 \
    --models-to-run meta/llama-3-8b

# Summarize results
helm-summarize --suite my_eval

# Start the web UI
helm-server --suite my_eval
```

## Prompt Formatting: The Invisible Variable

The single most common source of incomparable results is prompt format. The same model can vary by 5–10 percentage points on MMLU depending on whether the prompt uses:

- "Question: ... Answer:" vs. "Q: ... A:"
- An explicit "The best answer is:" suffix vs. none
- Chat template application vs. raw concatenation
- A system prompt saying "You are a helpful assistant" vs. nothing

### Chat Template Pitfalls

Instruction-tuned models expect prompts to be wrapped in the chat template used during training. When you evaluate a model like Llama-3-Instruct with raw concatenation, you are sending it out-of-distribution prompts, which degrades performance unpredictably.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

question = "What is the capital of France?"
choices = ["London", "Berlin", "Paris", "Madrid"]

# WRONG: raw concatenation (no chat template)
raw_prompt = f"Question: {question}\nChoices: {choices}\nAnswer:"

# RIGHT: use the model's chat template
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Answer with the letter only."
    },
    {
        "role": "user",
        "content": (
            f"{question}\n"
            + "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(choices))
        )
    }
]

# apply_chat_template adds special tokens and role delimiters
templated_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,        # return string, not token ids
    add_generation_prompt=True  # append the assistant turn starter
)
print(templated_prompt)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant. Answer with the letter only.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# What is the capital of France?
# A) London
# B) Berlin
# C) Paris
# D) Madrid<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
```

The lm-evaluation-harness exposes `--apply_chat_template` for exactly this reason. Enable it for instruction-tuned models and disable it for base models.

### Normalization

After generation, answers need to be normalized before comparison. The standard pipeline:

1. Strip leading/trailing whitespace
2. Lowercase
3. Remove punctuation (for open-ended tasks)
4. For multiple-choice: extract the letter or the choice text

```python
import re
import string

def normalize_answer(s: str) -> str:
    """Normalize a string answer for comparison.
    
    Follows the same normalization as the SQuAD evaluation script,
    used widely in QA benchmarks.
    """
    # Lowercase
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s

def extract_mc_answer(generation: str, choices: list[str]) -> str | None:
    """Extract a multiple-choice answer from a free-form generation.
    
    Handles both letter-based ("A", "B") and text-based answers.
    Returns the matched choice text, or None if no match.
    """
    gen = generation.strip()
    
    # Try to match a leading letter like "A" or "A."
    letter_match = re.match(r"^([A-Da-d])[\.\):\s]?", gen)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    
    # Try to match the choice text directly
    gen_norm = normalize_answer(gen)
    for choice in choices:
        if normalize_answer(choice) in gen_norm:
            return choice
    
    return None
```

## Building a Custom Eval from Scratch

The harnesses are excellent for standard tasks, but production use cases frequently require custom evaluations: domain-specific knowledge, proprietary test sets, novel task formats, or metrics the harnesses don't expose. Here is a complete worked example.

### Task: Domain-Specific Medical QA

We will build a harness for a hypothetical medical multiple-choice dataset stored in JSONL format, then integrate it with lm-evaluation-harness.

**Step 1: Define the data format.**

```json
{"id": "medq_001", "question": "Which enzyme is deficient in classic phenylketonuria?", "choices": ["Tyrosine hydroxylase", "Phenylalanine hydroxylase", "Homogentisate oxidase", "DOPA decarboxylase"], "answer": 1}
```

**Step 2: Write the YAML task config.**

```yaml
# tasks/medqa_custom.yaml
task: medqa_custom
dataset_path: json                    # HuggingFace datasets loader
dataset_kwargs:
  data_files:
    test: data/medqa_test.jsonl
test_split: test
output_type: multiple_choice          # use log-likelihood scoring over choices
doc_to_text: "Question: {{question}}\n"
doc_to_choice: "{{choices}}"         # the choices list field
doc_to_target: "{{answer}}"          # integer index of the correct choice
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
num_fewshot: 0
```

**Step 3: Register and run.**

```bash
# Point the harness at your custom task directory
lm_eval \
    --model hf \
    --model_args pretrained=mistralai/Mistral-7B-v0.1 \
    --tasks medqa_custom \
    --include_path ./tasks \
    --num_fewshot 0 \
    --output_path results/medqa
```

### Custom Metric: Clinical Entity F1

For tasks requiring structured extraction, you need to register a custom aggregation function.

```python
# custom_tasks/medner/task.py
"""Custom NER task for the lm-evaluation-harness.

Evaluates medical named entity recognition as exact-match entity-level F1.
"""
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.metrics import mean


def entity_f1(items):
    """Compute macro-averaged entity-level F1 over a list of (pred, gold) pairs.
    
    Args:
        items: list of (prediction_set, gold_set) tuples, where each set
               contains normalized entity strings.
    
    Returns:
        Macro-averaged F1 score in [0, 1].
    """
    f1_scores = []
    for pred_set, gold_set in items:
        if not gold_set:
            # No gold entities: perfect if model also predicts nothing
            f1_scores.append(1.0 if not pred_set else 0.0)
            continue
        
        true_pos = len(pred_set & gold_set)
        precision = true_pos / len(pred_set) if pred_set else 0.0
        recall    = true_pos / len(gold_set)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    
    return sum(f1_scores) / len(f1_scores)


class MedNERTask(ConfigurableTask):
    """Medical NER task with entity-level F1 metric."""

    VERSION = 1
    DATASET_PATH = "json"
    DATASET_NAME = None

    def doc_to_text(self, doc):
        return f"Extract all medical entities from this text:\n{doc['text']}\nEntities:"

    def doc_to_target(self, doc):
        # Return the list as a comma-separated string for generation
        return ", ".join(doc["entities"])

    def process_results(self, doc, results):
        # Parse the generation into a set of entities
        generation = results[0].strip()
        pred_entities = {
            e.strip().lower()
            for e in generation.split(",")
            if e.strip()
        }
        gold_entities = {e.lower() for e in doc["entities"]}
        return {
            "entity_f1": (pred_entities, gold_entities)
        }

    def aggregation(self):
        return {"entity_f1": entity_f1}

    def higher_is_better(self):
        return {"entity_f1": True}
```

### Rolling Your Own Lightweight Harness

Sometimes you need maximum control. Here is a minimal but production-grade harness that handles batching, log-likelihood scoring, and result serialization without any framework dependency.

```python
"""
minimal_harness.py

A self-contained eval harness for multiple-choice tasks.
Handles: batching, log-likelihood scoring (raw + normalized),
few-shot assembly, deterministic seeds, and JSON result output.

Usage:
    python minimal_harness.py \
        --model meta-llama/Llama-3.1-8B \
        --task_file data/task.jsonl \
        --num_fewshot 5 \
        --batch_size 8 \
        --output results.json
"""

import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MCDoc:
    id: str
    question: str
    choices: list[str]
    answer: int  # 0-indexed correct choice


@dataclass
class MCResult:
    id: str
    acc: int        # 1 if raw argmax is correct
    acc_norm: int   # 1 if length-normalized argmax is correct
    predicted_raw: int
    predicted_norm: int
    gold: int
    log_likelihoods: list[float]
    log_likelihoods_norm: list[float]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def format_doc(doc: MCDoc, include_answer: bool = False) -> str:
    """Format a single document as a prompt string."""
    letters = "ABCD"
    lines = [f"Question: {doc.question}"]
    for i, choice in enumerate(doc.choices):
        lines.append(f"{letters[i]}) {choice}")
    prompt = "\n".join(lines) + "\nAnswer:"
    if include_answer:
        # Append the correct answer letter for few-shot examples
        prompt += f" {letters[doc.answer]}"
    return prompt


def build_fewshot_prompt(
    test_doc: MCDoc,
    fewshot_docs: list[MCDoc],
    k: int
) -> str:
    """Build a K-shot prompt for a test document.
    
    Returns: (prefix, continuation) where prefix is the full context
    and continuation is the answer text to score.
    """
    shots = fewshot_docs[:k]
    parts = [format_doc(d, include_answer=True) for d in shots]
    parts.append(format_doc(test_doc, include_answer=False))
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_choices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    choices: list[str],
    device: str = "cuda"
) -> tuple[list[float], list[float]]:
    """Compute raw and length-normalized log-likelihoods for each choice.
    
    The trick: for each choice we tokenize [context + choice] and compute
    the sum of log-probs over only the choice tokens, not the context.
    
    Returns:
        log_liks: raw sum of log-probs per choice
        log_liks_norm: log_liks divided by choice token count
    """
    letters = "ABCD"
    context_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    context_len = context_ids.shape[1]

    log_liks = []
    log_liks_norm = []

    for i, choice in enumerate(choices):
        # The model expects " A", " B", etc. as the continuation
        # (leading space is important for tokenization)
        continuation = f" {letters[i]}"
        full_text = context + continuation
        full_ids = tokenizer.encode(
            full_text, return_tensors="pt", add_special_tokens=False
        ).to(device)

        # We need to include at least context_ids plus continuation tokens
        # Prepend BOS if the tokenizer uses it
        if tokenizer.bos_token_id is not None:
            bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
            full_ids = torch.cat([bos, full_ids], dim=1)
            ctx_len = context_ids.shape[1] + 1  # +1 for BOS
        else:
            ctx_len = context_len

        # Forward pass; model returns logits at each position
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: logits[t] predicts token[t+1]
        # We want log-probs for tokens from ctx_len onward
        log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)

        # Continuation tokens are at positions [ctx_len, seq_len)
        # Their log-probs are at positions [ctx_len-1, seq_len-1)
        cont_ids = full_ids[0, ctx_len:]  # the actual continuation token ids
        cont_log_probs = log_probs[ctx_len - 1: ctx_len - 1 + len(cont_ids)]

        # Gather the log-prob of each actual token
        token_log_probs = cont_log_probs.gather(
            1, cont_ids.unsqueeze(1)
        ).squeeze(1)

        ll = token_log_probs.sum().item()
        n_tokens = len(cont_ids)

        log_liks.append(ll)
        log_liks_norm.append(ll / n_tokens if n_tokens > 0 else ll)

    return log_liks, log_liks_norm


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_name: str,
    task_file: str,
    num_fewshot: int = 0,
    batch_size: int = 1,
    seed: int = 42,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """Run the full evaluation and return a results dict."""
    
    # Deterministic reproducibility: fix all relevant seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load task documents
    docs = []
    with open(task_file) as f:
        for line in f:
            d = json.loads(line)
            docs.append(MCDoc(**d))

    # Separate fewshot pool (use first 20% for fewshot, rest for eval)
    # In a real harness, this uses the training split, never the test split.
    n_fewshot_pool = max(num_fewshot * 5, 20)
    fewshot_pool = docs[:n_fewshot_pool]
    eval_docs = docs[n_fewshot_pool:]

    # Sample fixed few-shot examples per evaluation instance
    rng = random.Random(seed)
    fewshot_docs = rng.sample(fewshot_pool, min(num_fewshot, len(fewshot_pool)))

    # Load model and tokenizer
    print(f"Loading {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # BF16 is the safe default for LLMs
        device_map="auto",
    )
    model.eval()

    # Evaluate each document
    results = []
    for i, doc in enumerate(eval_docs):
        if i % 100 == 0:
            print(f"  [{i}/{len(eval_docs)}] …")

        context = build_fewshot_prompt(doc, fewshot_docs, num_fewshot)
        ll, ll_norm = score_choices(model, tokenizer, context, doc.choices, device)

        pred_raw = int(np.argmax(ll))
        pred_norm = int(np.argmax(ll_norm))

        results.append(MCResult(
            id=doc.id,
            acc=int(pred_raw == doc.answer),
            acc_norm=int(pred_norm == doc.answer),
            predicted_raw=pred_raw,
            predicted_norm=pred_norm,
            gold=doc.answer,
            log_likelihoods=ll,
            log_likelihoods_norm=ll_norm,
        ))

    # Aggregate
    acc    = np.mean([r.acc for r in results])
    acc_norm = np.mean([r.acc_norm for r in results])
    n = len(results)

    # Standard error: SE = sqrt(p*(1-p)/n) for a Bernoulli proportion
    se    = np.sqrt(acc * (1 - acc) / n)
    se_norm = np.sqrt(acc_norm * (1 - acc_norm) / n)

    summary = {
        "model": model_name,
        "task": task_file,
        "num_fewshot": num_fewshot,
        "seed": seed,
        "n_docs": n,
        "acc": round(float(acc), 4),
        "acc_se": round(float(se), 4),
        "acc_norm": round(float(acc_norm), 4),
        "acc_norm_se": round(float(se_norm), 4),
        "samples": [asdict(r) for r in results],  # full sample log
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {output_path}")

    print(f"\nacc={acc:.4f} ±{se:.4f}  acc_norm={acc_norm:.4f} ±{se_norm:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task_file", required=True)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        task_file=args.task_file,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        seed=args.seed,
        output_path=args.output,
    )
```

## Reproducibility Engineering

A benchmark number is only valuable if someone else can reproduce it. The checklist below separates reproducible from irreproducible evaluations.

### The Reproducibility Checklist

```text
MUST record for every eval run:
  ✓ Model name + revision hash (not just "llama3-8b")
  ✓ Harness version + commit SHA (lm_eval.__version__ is not enough)
  ✓ Task version numbers (each task has a VERSION field)
  ✓ num_fewshot and the exact few-shot seed
  ✓ Prompt template (or YAML hash)
  ✓ Whether apply_chat_template was used
  ✓ Scoring mode: loglikelihood vs generation
  ✓ Length normalization: yes / no
  ✓ Random seed for sampling (--gen_kwargs seed=X)
  ✓ Hardware (GPU model, driver, CUDA version)
  ✓ Floating-point format (BF16 vs FP16 vs FP32)
  ✓ Whether --limit was used (partial eval is not full eval)
  ✓ Dataset split and subset (MMLU has 57 subjects)
```

### Storing Results

Always emit raw per-sample predictions alongside aggregate scores. A sample log enables post-hoc analysis: debugging individual failures, checking for task-level patterns, and recomputing metrics with different normalization without re-running the model.

```bash
# lm_eval produces this structure under --output_path:
results/
├── results.json          # aggregate scores by task and metric
├── samples_mmlu_0.jsonl  # per-sample log for mmlu, seed 0
└── samples_hellaswag_0.jsonl
```

```json
// One line from samples_mmlu_0.jsonl
{
  "doc_id": 1234,
  "doc": {"question": "...", "choices": [...], "answer": 2},
  "target": 2,
  "arguments": [["context + choice A"], ["context + choice B"], ...],
  "resps": [[-3.12, false], [-1.87, false], [-1.42, true], [-4.01, false]],
  "filtered_resps": [[-3.12, false], [-1.87, false], [-1.42, true], [-4.01, false]],
  "acc": 1.0,
  "acc_norm": 1.0
}
```

### Contamination and Dataset Splits

One of the thorniest reproducibility issues is contamination: if a model was trained on data that overlaps with the evaluation set, the eval score is inflated. Harnesses cannot detect this automatically — it requires explicit deduplication between the model's training set and the benchmark. See [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) for the techniques used during pretraining.

For a custom eval, use a test set that was created after the model's training cutoff, or use n-gram overlap tools (e.g., MinHash LSH) to filter training data.

## Statistical Significance and Error Bars

A 1-point improvement on a benchmark is almost never a headline result. You need to understand whether the difference is statistically meaningful.

### The Bernoulli Standard Error

For accuracy (a proportion), the standard error is:

$$
\text{SE} = \sqrt{\frac{p(1-p)}{n}}
$$

where $p$ is the observed accuracy and $n$ is the number of evaluation examples.

!!! example "Worked example: Is 74.1% really better than 73.0%?"

    Suppose model A scores $p_A = 0.741$ on MMLU (14,079 test questions) and model B scores $p_B = 0.730$.

    $$
    \text{SE}_A = \sqrt{\frac{0.741 \times 0.259}{14079}} \approx \sqrt{\frac{0.1919}{14079}} \approx 0.00369
    $$

    $$
    \text{SE}_B = \sqrt{\frac{0.730 \times 0.270}{14079}} \approx \sqrt{\frac{0.1971}{14079}} \approx 0.00374
    $$

    The standard error of the difference (assuming independence):
    $$
    \text{SE}_{A-B} = \sqrt{\text{SE}_A^2 + \text{SE}_B^2} \approx \sqrt{0.0000136 + 0.0000140} \approx 0.00525
    $$

    The observed difference is $0.741 - 0.730 = 0.011$. The z-score is:
    $$
    z = \frac{0.011}{0.00525} \approx 2.10
    $$

    A z-score of 2.10 corresponds to a p-value of about 0.036, so the difference is statistically significant at the 0.05 level — but only barely. On a smaller dataset (say, n=1000), the same 1.1-point difference would not be significant.

### McNemar's Test for Paired Evaluations

When comparing two models on the **same** documents, the samples are not independent (the same document may be harder for both models). McNemar's test uses paired data and is more powerful:

$$
\chi^2 = \frac{(n_{01} - n_{10})^2}{n_{01} + n_{10}}
$$

where $n_{01}$ is the number of examples where model A is wrong and model B is right, and $n_{10}$ is the reverse.

```python
from scipy.stats import chi2

def mcnemar_test(results_a: list[int], results_b: list[int]) -> dict:
    """Perform McNemar's test for two paired binary result sequences.
    
    Args:
        results_a: list of 0/1 correctness for model A (one per doc)
        results_b: list of 0/1 correctness for model B (one per doc)
    
    Returns:
        dict with chi2 statistic, p-value, and cell counts
    """
    assert len(results_a) == len(results_b), "Must be paired on the same docs"
    
    n_01 = sum(a == 0 and b == 1 for a, b in zip(results_a, results_b))
    n_10 = sum(a == 1 and b == 0 for a, b in zip(results_a, results_b))
    
    if n_01 + n_10 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_01": 0, "n_10": 0}
    
    # Apply continuity correction (Yates) for small cell counts
    statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    p_value = 1 - chi2.cdf(statistic, df=1)
    
    return {
        "chi2": round(statistic, 4),
        "p_value": round(p_value, 4),
        "n_01": n_01,  # B correct, A wrong
        "n_10": n_10,  # A correct, B wrong
        "acc_a": sum(results_a) / len(results_a),
        "acc_b": sum(results_b) / len(results_b),
    }

# Example usage
results_a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  # model A correct/wrong
results_b = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]  # model B correct/wrong
print(mcnemar_test(results_a, results_b))
# {'chi2': 0.0, 'p_value': 1.0, 'n_01': 2, 'n_10': 2, 'acc_a': 0.6, 'acc_b': 0.6}
```

### Bootstrap Confidence Intervals

Bootstrap is the most flexible option because it makes no distributional assumptions and extends to any metric (F1, ROUGE, pass@k):

```python
import numpy as np
from typing import Callable

def bootstrap_ci(
    scores: list[float],
    metric_fn: Callable[[list[float]], float] = np.mean,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for any metric.
    
    Args:
        scores: per-example metric values (e.g., list of 0/1 for accuracy)
        metric_fn: aggregation function (default: mean)
        n_bootstrap: number of bootstrap resamples
        alpha: significance level (0.05 → 95% CI)
        seed: random seed for reproducibility
    
    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    
    point_estimate = metric_fn(arr)
    
    # Resample with replacement n_bootstrap times
    boot_stats = []
    for _ in range(n_bootstrap):
        resample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(metric_fn(resample))
    
    boot_stats = np.array(boot_stats)
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, lower, upper

# Example: accuracy scores for 500 examples
scores = [1] * 370 + [0] * 130  # 74% accuracy
pt, lo, hi = bootstrap_ci(scores, n_bootstrap=10_000, seed=42)
print(f"acc = {pt:.3f}  95% CI: [{lo:.3f}, {hi:.3f}]")
# acc = 0.740  95% CI: [0.703, 0.776]
```

### Multiple Comparison Corrections

When running a model on 50+ benchmarks and cherry-picking the best ones, you inflate the false-positive rate. Use the Bonferroni correction (divide the significance threshold by the number of comparisons) or the Benjamini-Hochberg procedure for controlling the false discovery rate. For a thorough model comparison across 10 benchmarks, report all 10 p-values with a Bonferroni-corrected threshold of $\alpha' = 0.05/10 = 0.005$.

!!! warning "Common pitfall: leaking the test set"

    The single most dangerous mistake in eval harness design is using the test split for any kind of hyperparameter tuning — including choosing the prompt template. If you try 10 prompt variants on the test set and report the best, you have effectively turned the test set into a validation set, and the reported score is optimistic. Always tune on a held-out dev set, then do a single final run on test.

!!! interview "Interview Corner"

    **Q:** You have a new model that scores 75.2% on MMLU and a baseline that scores 74.8%. How do you decide whether this is a real improvement?

    **A:** First, I compute the standard error for each score: $\text{SE} = \sqrt{p(1-p)/n}$. MMLU has about 14,000 questions, so the SE for both models is roughly 0.37%. The observed difference (0.4%) is barely more than one SE, giving a z-score around 1.1, which is not statistically significant (p ≈ 0.27). I would not claim this as a real improvement without either (1) a larger eval set, (2) a paired test like McNemar's on the same examples, or (3) consistent gains across several benchmarks. I would also check that both scores were produced by the same harness version, prompt format, and scoring mode — a 0.4-point gap can easily be a formatting artifact rather than a model difference.

## Comparing Harnesses: lm-evaluation-harness vs HELM

| Dimension | lm-evaluation-harness | HELM |
|---|---|---|
| Task coverage | 200+ tasks, community-maintained | ~60 core scenarios, curated |
| Scoring modes | log-likelihood + generation | primarily generation |
| Multi-metric per task | acc / acc_norm | accuracy + calibration + fairness + toxicity + efficiency |
| Speed | Very fast (batched LL scoring) | Slower (full generation for most tasks) |
| Reproducibility artifacts | JSON results + JSONL sample logs | structured JSON + web UI |
| Leaderboard integration | Open LLM Leaderboard (HuggingFace) | HELM Leaderboard (Stanford CRFM) |
| Custom task difficulty | Easy (YAML + optional Python) | Moderate (Python subclassing) |
| Best for | Quick model comparisons, CI/CD integration | Multi-dimensional capability profiling |

For most production use cases, lm-evaluation-harness is the right starting point. Use HELM when you need detailed multi-axis analysis or want to align with Stanford CRFM benchmarking methodology.

## Continuous Evaluation in CI/CD

A mature LLM development workflow runs evals automatically on every significant model checkpoint, not just at release time. Here is a practical CI/CD integration pattern.

```yaml
# .github/workflows/eval.yml  (GitHub Actions)
name: Model Evaluation

on:
  push:
    paths:
      - "checkpoints/**"
  schedule:
    - cron: "0 4 * * *"   # nightly eval at 04:00 UTC

jobs:
  evaluate:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: pip install lm-eval[vllm]==0.4.3  # pin version for reproducibility

      - name: Run core benchmark suite
        run: |
          lm_eval \
            --model vllm \
            --model_args pretrained=${{ env.MODEL_PATH }} \
            --tasks mmlu,hellaswag,arc_challenge,winogrande \
            --num_fewshot 5 \
            --batch_size 32 \
            --seed 1234 \
            --output_path eval_results/${{ github.sha }}

      - name: Check regression threshold
        run: |
          # Fail the pipeline if any task regresses > 0.5%
          python scripts/check_regression.py \
            --baseline results/baseline.json \
            --current eval_results/${{ github.sha }}/results.json \
            --threshold 0.005

      - name: Upload results to tracking system
        run: |
          python scripts/log_to_mlflow.py \
            --run_name ${{ github.sha }} \
            --results_dir eval_results/${{ github.sha }}
```

The `check_regression.py` script reads aggregate results, computes the difference from a stored baseline, and exits with a non-zero code if any task drops below threshold — blocking the merge and alerting the team.

This connects to broader MLOps concerns covered in [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html).

---

!!! key "Key Takeaways"
    - Eval harnesses standardize prompt formatting, scoring mode, few-shot sampling, and normalization — without them, cross-model comparisons are meaningless.
    - lm-evaluation-harness uses log-likelihood scoring for multiple-choice tasks (computing $\sum \log p(\text{choice} \mid \text{context})$); length normalization by token count changes which answer wins and must be reported explicitly.
    - HELM adds multi-dimensional scoring (accuracy + calibration + fairness + efficiency) and structured reproducibility artifacts; lm-eval is faster and has broader task coverage.
    - Chat-templated models must be evaluated with `apply_chat_template`; using raw concatenation sends them out-of-distribution and can suppress performance by several points.
    - A 1-point accuracy difference is only statistically significant at typical MMLU scale (~14k examples) if $z = \Delta / \text{SE}_{diff} > 2$; compute standard errors and report them alongside every number.
    - McNemar's test is the correct statistical test for paired model comparisons; bootstrap confidence intervals generalize to any non-standard metric.
    - Every eval run must log: model revision, harness version, task version, few-shot seed, prompt template, and per-sample predictions — not just aggregate scores.
    - Never tune prompt templates on the test split; doing so converts test accuracy into a training signal and produces inflated, non-reproducible results.
    - CI/CD integration with regression thresholds catches quality regressions before they ship to users.

!!! sota "State of the Art & Resources (2026)"
    Eval harnesses are now a standard fixture of LLM development: lm-evaluation-harness underpins almost every public open-model leaderboard, while HELM, Inspect AI, and newer tooling are pushing evaluation toward multi-axis, contamination-aware, and domain-specific scoring. The field's central challenge has shifted from "how do we score models" to "how do we prevent benchmark saturation and gaming."

    **Foundational work**

    - [Hendrycks et al., *Measuring Massive Multitask Language Understanding* (2021)](https://arxiv.org/abs/2009.03300) — introduced MMLU, the most widely used multiple-choice benchmark, and established log-likelihood scoring as the default evaluation paradigm.
    - [Liang et al., *Holistic Evaluation of Language Models* (2022)](https://arxiv.org/abs/2211.09110) — HELM's scenario × adaptation × metric decomposition; argues accuracy alone is insufficient and defines calibration, fairness, and efficiency as first-class metrics.
    - [Srivastava et al., *Beyond the Imitation Game Benchmark* (2022)](https://arxiv.org/abs/2206.04615) — BIG-bench's 204-task collaborative effort; documented how task design, few-shot protocol, and human baselines interact at scale.

    **Recent advances (2023–2026)**

    - [Biderman et al., *Lessons from the Trenches on Reproducible Evaluation of Language Models* (2024)](https://arxiv.org/abs/2405.14782) — three years of lm-eval experience distilled into concrete reproducibility recommendations; the authoritative reference for prompt sensitivity, versioning, and per-sample logging.
    - [Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark* (NeurIPS 2024)](https://arxiv.org/abs/2406.01574) — extends MMLU to 10-choice questions and shows 16–33% accuracy drops vs. original MMLU, reducing prompt-sensitivity from ~5% to ~2%.
    - [Shashidhar et al., *YourBench: Easy Custom Evaluation Sets for Everyone* (2025)](https://arxiv.org/abs/2504.01833) — automated generation of domain-tailored benchmarks from documents; replicates MMLU subsets for under $15, addressing contamination by making fresh evals cheap.

    **Open-source & tools**

    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the de facto standard open-model eval framework; backs the Hugging Face Open LLM Leaderboard and supports 200+ tasks with log-likelihood and generation scoring.
    - [stanford-crfm/helm](https://github.com/stanford-crfm/helm) — HELM's official Python package; multi-axis leaderboards covering capabilities, safety, MedHELM, and long-context evaluation.
    - [openai/evals](https://github.com/openai/evals) — OpenAI's eval framework and open-source benchmark registry; useful reference for generation-based and LLM-as-judge eval patterns.
    - [Inspect AI](https://inspect.aisi.org.uk/) — UK AISI's open-source eval framework with 200+ pre-built evals, agent sandboxing, and multi-provider model support; particularly strong for agentic and safety evaluations.

    **Go deeper**

    - [lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md) — official YAML task-authoring reference; shows exactly how `doc_to_text`, `doc_to_target`, and `process_results` are wired together.

## Further Reading

- **EleutherAI lm-evaluation-harness** — Gao et al., *A Framework for Few-Shot Language Model Evaluation*, 2021 (GitHub: EleutherAI/lm-evaluation-harness). The canonical reference for open-model evaluation at scale.
- **HELM** — Liang et al., *Holistic Evaluation of Language Models*, NeurIPS 2022. Introduces the scenario × adaptation × metric decomposition and multi-axis evaluation.
- **BIG-bench** — Srivastava et al., *Beyond the Imitation Game*, 2022. A collaborative benchmark with 200+ tasks and careful attention to task design, few-shot protocols, and human baselines.
- **MMLU** — Hendrycks et al., *Measuring Massive Multitask Language Understanding*, ICLR 2021. The most widely-used multiple-choice benchmark; the source code demonstrates log-likelihood evaluation mechanics clearly.
- **Calibration in NLP** — Desai and Durrett, *Calibration of Pre-trained Transformers*, EMNLP 2020. Explains why accuracy alone is insufficient and how to measure uncertainty quality.
- **Statistical significance in NLP** — Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing*, ACL 2018. Practical guidance on test selection and multiple comparison correction for benchmark comparisons.
