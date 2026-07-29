# 14.11 Evaluation & Serving: Honest Benchmarks, int4 Quantization, and Running on a Laptop

By the end of [14.9 Post-Training](09-post-training.html) and [14.10 The Narrow Agent](10-agentic-narrow.html), you have a checkpoint: `Stack-100M`, pretrained on ~20B tokens, mid-trained for long context and capability injection, SFT+DPO-aligned, and lightly RLVR-sharpened on tool use. It is tempting to declare victory and ship it. Resist that. A checkpoint is not a result — a **number you can defend**, next to an **honest description of what the model cannot do**, is a result. This chapter builds that: a small, principled evaluation harness, and then the second half of the payoff — squeezing the ~101M-parameter, ~406MB fp32 checkpoint down to something that runs, from cold start to generated text, on the CPU of the laptop you are reading this on.

Two halves, one spirit: **measure honestly, then serve efficiently**. Neither half is about chasing a leaderboard. A 100M model trained on a single GPU for the price of a nice dinner is never going to touch a frontier benchmark, and pretending otherwise — via a leaky eval set or a cherry-picked prompt — teaches the wrong lesson. The right lesson is that a small model, evaluated honestly and served efficiently, is still a genuinely useful, genuinely *yours* piece of software. That is what we build here.

## 1. What "Done" Means for Stack-100M

Recall the full pipeline from [14.1 The Capstone Overview](01-overview-and-landscape.html): tokenizer → pretraining (~20B tokens, WSD schedule, Muon+AdamW) → mid-training (long-context extension, quality annealing) → post-training (SFT, DPO, narrow RLVR) → agentic distillation. Every one of those stages produced a checkpoint. This chapter is the final gate before you call the project finished:

1. **Evaluate** the final checkpoint against a small, fixed battery: held-out perplexity, an arithmetic probe, a tiny multiple-choice set, and retrieval-QA exact match. Report numbers, not vibes.
2. **Write down what the model cannot do.** This is not optional padding — it is the deliverable that makes the numbers trustworthy.
3. **Quantize** the checkpoint to int8, then int4, understanding exactly what post-training quantization (PTQ) does to the weights and why round-to-nearest (RTN) is a baseline rather than the state of the art.
4. **Export and serve** the quantized model on CPU, with measured (not estimated) latency and memory on your own machine.

We deliberately evaluate the *final* post-trained checkpoint, but every function below also runs against the raw pretrained base checkpoint from [14.7 The Pretraining Run](07-pretraining-run.html) — comparing base vs. instruct/agent perplexity and probe scores is itself a useful sanity check that post-training didn't quietly regress core language modeling ability (a real failure mode called *alignment tax*).

## 2. Held-Out Perplexity: The One Number You Can Trust

Perplexity is the exponentiated average negative log-likelihood the model assigns to held-out text it never trained on:

$$
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i})\right)
$$

It is the single most trustworthy number in this chapter precisely because it is *cheap to make honest*: as long as your held-out shard was excluded from the training manifest (see the data pipeline's manifest/hash bookkeeping in [14.2 The Data Pipeline](02-data-pipeline.html)), there is no way to game it by prompt-crafting or answer-formatting tricks. It directly measures the thing pretraining optimizes. Its weakness is exactly the flip side: it tells you how well the model predicts *held-out web text*, not whether it can multiply two-digit numbers or answer a question correctly — for that you need the probes in §3.

```python
"""
stacklm/eval.py — held-out perplexity.

Assumes the checkpoint format and data-loading conventions established in
Ch. 14.7 (uint16 memmap .bin shards, document-aware packing to seq_len 2048).
"""
import math
import torch
import torch.nn.functional as F

from stacklm.model import StackLM, StackLMConfig  # Ch. 14.4
from stacklm.data import ShardDataset            # Ch. 14.2 / 14.7


@torch.no_grad()
def compute_perplexity(
    model: StackLM,
    val_shard_paths: list[str],
    seq_len: int = 2048,
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "cuda",
) -> dict:
    """Compute held-out cross-entropy loss and perplexity.

    Returns both the mean loss (nats/token, directly comparable to the
    training curve in Ch. 14.7) and perplexity (exp of that loss).
    """
    model.eval()
    ds = ShardDataset(val_shard_paths, seq_len=seq_len)  # never seen in training
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    total_nll, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)          # (B, T)
        targets = batch["targets"].to(device)               # (B, T), shifted by 1
        doc_ids = batch["doc_ids"].to(device)                # for no-cross-doc masking

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, doc_ids=doc_ids)       # (B, T, V)
            # sum (not mean) so we can correctly average over the whole eval set
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                reduction="sum",
                ignore_index=-100,  # padded / cross-doc positions
            )
        n_valid = (targets != -100).sum().item()
        total_nll += loss.item()
        total_tokens += n_valid

    mean_nll = total_nll / total_tokens
    return {
        "loss_nats_per_token": mean_nll,
        "perplexity": math.exp(mean_nll),
        "n_tokens_evaluated": total_tokens,
    }
```

!!! example "Worked example: reading a perplexity number"

    Suppose `compute_perplexity` reports a held-out loss of **2.9 nats/token**, in the "on the order of ~2.8–3.2 nats/token" range documented for the flagship run in [14.7](07-pretraining-run.html). That converts to $\text{PPL} = e^{2.9} \approx 18.2$. In plain terms: on held-out text from the training mix, the model's predictive distribution has an effective branching factor of about 18 tokens at each position — not "18 equally likely tokens" (real text is far more skewed than that), but a useful single-number summary of average surprise. A larger, better-trained model on the same mix might sit at PPL ≈ 8–12; a random-guessing model over the 32768-token vocabulary would sit at PPL ≈ 32768. The honest framing: this number tells you the model learned real structure in FineWeb-Edu-like text, and nothing more.

Two practical rules make this number defensible rather than decorative. First, **the held-out shard must be provably excluded** from every stage of training — pretraining, mid-training annealing, and SFT/DPO data — which is why the data pipeline in Ch. 14.2 hashes and manifests every source file up front. Second, **report loss in nats/token, not just perplexity**, since it composes linearly and is what your training curve already plots — you can literally draw an "eval loss" point on the same chart as the training loss and see the generalization gap.

## 3. Lightweight Capability Probes

Perplexity says nothing about whether the model can do anything *useful*. We add three small, cheap, interpretable probes — each deliberately tiny (tens to low hundreds of examples), because at 100M parameters, a large eval suite would cost more compute than the pretraining run itself and buy you very little additional signal. These are diagnostic probes, not benchmark claims; treat every number as "this model, on this exact small set, on this exact day."

### 3.1 Arithmetic accuracy

Integer arithmetic is the cleanest probe of *reliable reasoning*: the answer is unambiguous, easy to grade exactly, and — because [14.9](09-post-training.html) ran narrow RLVR on exactly this task with exact-match reward — a direct check of whether that RL stage actually stuck.

```python
import random
import re


def make_arithmetic_probe(n: int = 200, seed: int = 0, max_digits: int = 2) -> list[dict]:
    """Generate n random addition/subtraction problems with known ground truth."""
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        a = rng.randint(0, 10 ** max_digits - 1)
        b = rng.randint(0, 10 ** max_digits - 1)
        op = rng.choice(["+", "-"])
        answer = a + b if op == "+" else a - b
        problems.append({"prompt": f"{a} {op} {b} = ", "answer": answer})
    return problems


_NUM_RE = re.compile(r"-?\d+")


@torch.no_grad()
def eval_arithmetic(model, tokenizer, generate_fn, problems: list[dict]) -> dict:
    """Greedy-generate a short completion per problem, parse the first integer,
    and score exact match against the ground-truth answer."""
    n_correct = 0
    for p in problems:
        completion = generate_fn(
            model, tokenizer, prompt=p["prompt"], max_new_tokens=8, temperature=0.0
        )
        match = _NUM_RE.search(completion)
        predicted = int(match.group()) if match else None
        n_correct += int(predicted == p["answer"])
    return {"accuracy": n_correct / len(problems), "n": len(problems)}
```

### 3.2 A tiny multiple-choice set

We score multiple-choice the way most eval harnesses do it under the hood: not by asking the model to output the letter "A"/"B"/"C"/"D" (a 100M model's instruction-following is too weak to reliably format that), but by **cloze scoring** — computing the model's total log-probability of each full answer string appended to the question, and picking the highest-probability option. This is the same technique used by the EleutherAI `lm-evaluation-harness` (Gao et al.) for MMLU-style tasks (Hendrycks et al., *Measuring Massive Multitask Language Understanding*, 2021), and it sidesteps the formatting-fragility that would otherwise dominate the result at this model scale.

```python
@torch.no_grad()
def _sequence_logprob(model, tokenizer, prompt: str, continuation: str, device="cuda") -> float:
    """Sum of log p(token | prefix) for every token in `continuation`."""
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + continuation)
    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids)  # (1, T, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Score only the continuation tokens: positions len(prompt_ids)-1 .. len(full_ids)-2
    # predict tokens at positions len(prompt_ids) .. len(full_ids)-1.
    start = len(prompt_ids) - 1
    end = len(full_ids) - 1
    target_ids = torch.tensor(full_ids[start + 1 : end + 1], device=device)
    token_logps = log_probs[0, start:end, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum().item()


# A hand-written, deliberately tiny probe (10 items shown; the full set used in
# practice is ~50-100 items spanning basic science, geography, and word sense —
# small enough to eyeball every item for leakage).
TINY_MC_SET = [
    {
        "question": "The chemical symbol for water is",
        "choices": [" H2O", " CO2", " NaCl", " O2"],
        "answer_idx": 0,
    },
    {
        "question": "The capital of France is",
        "choices": [" Paris", " Berlin", " Madrid", " Rome"],
        "answer_idx": 0,
    },
    # ... remaining items omitted for brevity; same {question, choices, answer_idx} shape.
]


def eval_mc_probe(model, tokenizer, mc_set: list[dict] = TINY_MC_SET) -> dict:
    n_correct = 0
    for item in mc_set:
        scores = [
            _sequence_logprob(model, tokenizer, item["question"], choice)
            for choice in item["choices"]
        ]
        predicted_idx = max(range(len(scores)), key=lambda i: scores[i])
        n_correct += int(predicted_idx == item["answer_idx"])
    return {"accuracy": n_correct / len(mc_set), "n": len(mc_set)}
```

### 3.3 Retrieval-QA exact match

This probe reuses the small local corpus and retriever built for the narrow agent in [14.10](10-agentic-narrow.html). It measures a *different* skill than the closed-book MC set above: given a **retrieved passage that actually contains the answer**, can the model extract and state it correctly? This isolates reading comprehension from parametric knowledge — a fairer test for a model this small, which has nowhere near the capacity to memorize broad factual knowledge in its weights (Natural Questions-style QA, Kwiatkowski et al., 2019, is the large-scale ancestor of this idea).

```python
def normalize_answer(s: str) -> str:
    """Standard EM normalization: lowercase, strip punctuation/articles/extra spaces."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def eval_retrieval_qa(model, tokenizer, generate_fn, retriever, qa_pairs: list[dict]) -> dict:
    """qa_pairs: [{"question": str, "gold_answer": str}, ...]
    retriever: the BM25/embedding-lite retriever from Ch. 14.10, exposing .search(query, k)."""
    n_correct = 0
    for pair in qa_pairs:
        passage = retriever.search(pair["question"], k=1)[0]
        prompt = (
            f"<|system|>Answer using only the passage below.<|end|>\n"
            f"<|user|>Passage: {passage}\nQuestion: {pair['question']}<|end|>\n"
            f"<|assistant|>"
        )
        completion = generate_fn(model, tokenizer, prompt=prompt, max_new_tokens=16, temperature=0.0)
        n_correct += int(normalize_answer(completion) == normalize_answer(pair["gold_answer"]))
    return {"exact_match": n_correct / len(qa_pairs), "n": len(qa_pairs)}
```

!!! example "Worked example: a plausible probe report"

    Running all three probes against the post-trained Stack-100M checkpoint might produce a table like this (illustrative structure — replace with your own run's numbers):

    | Probe | Metric | Score | n |
    |---|---|---|---|
    | Held-out perplexity | PPL | ~18 | 2M tokens |
    | Arithmetic (2-digit +/-) | exact match | on the order of 60-85% | 200 |
    | Tiny MC set | accuracy | on the order of 40-60% | ~80 |
    | Retrieval-QA | exact match | on the order of 50-75% | ~50 |

    Notice the pattern such a table typically reveals: arithmetic and retrieval-QA — both narrowly scoped, both RL/SFT-targeted, both with an unambiguous grading function — score meaningfully above chance, while the closed-book MC set, which requires broad parametric world knowledge the model never had the capacity to store, sits closer to a coin flip. That asymmetry is itself the finding, and it should be reported, not smoothed over.

## 4. The Honest Capability Report

The single most important artifact of this chapter is not a number — it is a paragraph. Every capability probe above measures a narrow slice; the reader of your model card needs the slices assembled into an honest picture. Write it down explicitly, next to the numbers:

- **Stack-100M is a narrow tool, not a general oracle.** At ~101M parameters and ~20B training tokens, it sits roughly three orders of magnitude below a frontier model in both parameters and effective training FLOPs. It will confidently state incorrect facts, struggle with anything requiring multi-step reasoning beyond what narrow RLVR explicitly trained, and its "knowledge" is a lossy compression of a filtered web+synthetic corpus, not a queryable database.
- **What it is reliably good at** is precisely the narrow, scaffolded tasks it was pointed at: short-form chat in the SFT/DPO style, two-digit arithmetic, and grounded retrieval-QA when the answer is handed to it in context — the ReAct-style agent loop from [14.10](10-agentic-narrow.html) exists specifically because retrieval + a small model beats a small model alone on knowledge-heavy questions.
- **What it is not good at**: long-horizon reasoning, anything resembling closed-book trivia outside the training mix's coverage, code beyond the toy StarCoder-subset flavor it saw, and — like every language model — it will hallucinate fluently and without any internal signal of uncertainty that a downstream system can cheaply detect.

This is the same evaluation ethic developed at length in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) and [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html): a benchmark score is only as trustworthy as the disclosure that accompanies it.

!!! warning "Contamination: the probe you build is the probe you must audit"

    Because you constructed the tiny MC set and the arithmetic generator yourself, contamination risk is lower than for a model evaluated on a public leaderboard — but it is not zero. Two concrete leaks to check for:

    1. **Data-pipeline leakage.** If FineWeb-Edu or Cosmopedia happens to contain near-duplicates of your hand-written MC questions (surprisingly plausible for well-known trivia like "capital of France"), the model may be pattern-matching memorized web text rather than reasoning. Cross-check MC items against the deduplication index built in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html).
    2. **Retrieval-QA leakage.** If the small local corpus used for the retrieval probe overlaps with pretraining data verbatim, the model may already have the passage memorized and the probe silently degrades into a closed-book test wearing an open-book costume. Build the retrieval corpus from sources *excluded* from the pretraining manifest, the same discipline used for the held-out perplexity shard in §2.

    More generally: any number you cannot regenerate from a documented, hashed, versioned eval set is not a number — it is an anecdote. See [Statistical Rigor in Evaluation](../11-evaluation/06-statistical-rigor-eval.html) for how to attach confidence intervals to small-n probes like these (n=50-200 is small enough that a single-digit-percentage-point swing is well within noise — report it).

{{fig:honest-eval-four-probes-asymmetry}}

## 5. Post-Training Quantization: RTN, GPTQ, and AWQ

With honest numbers in hand, the second half of the chapter is serving. The fp32 checkpoint is ~406MB (101.4M params × 4 bytes — the exact accounting is in [14.4](04-architecture.html)); even at bf16 that is ~203MB, comfortably in RAM on any laptop but wasteful given how little precision a well-trained weight actually needs at inference time. **Post-training quantization (PTQ)** converts the trained fp32/bf16 weights to low-bit integers *after* training, no gradient updates required — the natural complement to the quantization-aware and mixed-precision *training* techniques covered in [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html). This chapter is a hands-on companion to the book's dedicated PTQ chapters — [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html) and [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html) — read those for the full derivations; here we implement the baseline end to end and *use* it.

### 5.1 Round-to-nearest (RTN): the baseline

RTN is exactly what it sounds like: pick a scale (and, for asymmetric schemes, a zero-point) per group of weights from that group's min/max, then round each weight to the nearest representable integer. For a $b$-bit **symmetric** scheme over a group of weights $w$:

$$
s = \frac{\max(|w|)}{2^{b-1}-1}, \qquad q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right),\, -(2^{b-1}-1),\, 2^{b-1}-1\right), \qquad \hat w = s\,q
$$

For a $b$-bit **asymmetric** scheme (uses the full unsigned range, better for skewed distributions):

$$
s = \frac{\max(w)-\min(w)}{2^b-1}, \qquad z = \operatorname{round}\!\left(\frac{-\min(w)}{s}\right), \qquad q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right)+z,\, 0,\, 2^b-1\right), \qquad \hat w = s\,(q-z)
$$

RTN's virtue is that it is a single pass over the weights with no calibration data and no per-layer optimization — you can quantize a checkpoint in seconds. Its vice is that it treats every weight independently: it has no notion that some weights matter more to the model's output than others, so at 4 bits (only 16 representable values per group) it can measurably hurt quality, especially on outlier-heavy channels.

{{fig:rtn-quantization-number-line}}

### 5.2 GPTQ: reconstruction-aware quantization

**GPTQ** (Frantar, Ashkboos, Hoefler & Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022) fixes RTN's blind spot by quantizing each linear layer's weight matrix **column by column**, and after quantizing each column, *updating all not-yet-quantized columns to compensate for the error just introduced*. Concretely, for a layer with calibration activations $X$ (a batch of real hidden states run through the layer) and weight $W$, GPTQ builds the Hessian of the layer's squared-error reconstruction objective, $H = 2XX^\top + \lambda I$ (the damping term $\lambda I$ keeps it invertible), and processes input-feature columns $q = 1, 2, \dots$ in order:

$$
\hat w_{:,q} = \operatorname{quant}(w_{:,q}) \quad\text{(RTN on this column)}, \qquad
\delta = \frac{w_{:,q} - \hat w_{:,q}}{[H^{-1}]_{qq}}, \qquad
w_{:,j} \mathrel{-}= \delta \cdot [H^{-1}]_{qj} \;\; \forall\, j > q
$$

The update spreads each column's quantization error onto the columns not yet quantized, weighted by how strongly the Hessian says those columns interact with this one — an efficient, layer-local instance of the classic Optimal Brain Surgeon idea (LeCun et al., 1990; Hassibi & Stork, 1993) applied to quantization instead of pruning. The payoff is that GPTQ can push to 4 or even 3 bits with far less quality loss than RTN, at the cost of needing calibration data and $O(d_{in}^3)$ work per layer for the Hessian inverse (GPTQ's actual implementation batches this efficiently via Cholesky decomposition; the sketch below is the pedagogical, unoptimized version):

```python
import torch


def gptq_quantize_column_by_column(W: torch.Tensor, X_calib: torch.Tensor, bits: int = 4,
                                     damp: float = 1e-2) -> torch.Tensor:
    """Simplified, unoptimized reference implementation of GPTQ's per-layer
    reconstruction. W: (d_out, d_in) weight matrix. X_calib: (n_samples, d_in)
    calibration activations captured from a real forward pass on held-out text.

    This is pedagogical: real GPTQ batches columns and uses a running Cholesky
    factorization for speed. Use bitsandbytes/AutoGPTQ/GPTQModel for production.
    """
    d_out, d_in = W.shape
    W = W.clone().float()

    # Hessian of the layer's local least-squares reconstruction objective.
    H = 2 * (X_calib.T @ X_calib) / X_calib.shape[0]
    H += damp * torch.eye(d_in, device=W.device)  # damping for numerical stability
    H_inv = torch.linalg.inv(H)

    qmax = 2 ** (bits - 1) - 1
    for q in range(d_in):
        col = W[:, q]
        scale = col.abs().max() / qmax if col.abs().max() > 0 else 1.0
        q_col = torch.clamp(torch.round(col / scale), -qmax, qmax)
        w_hat = q_col * scale
        error = (col - w_hat) / H_inv[q, q]           # per-output-row error, scalar Hessian term
        W[:, q] = w_hat
        if q + 1 < d_in:
            # Spread the reconstruction error onto not-yet-quantized columns.
            W[:, q + 1 :] -= torch.outer(error, H_inv[q, q + 1 :])
    return W
```

### 5.3 AWQ: protect the salient channels instead of correcting for them

**AWQ** (Lin, Tang, Tang, Yang, Dang & Han, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023) takes a different, cheaper angle. Its empirical observation: a small fraction of *input channels* (about 0.1–1% in practice) receive systematically large activation magnitudes and disproportionately affect the layer's output — quantization error on the weights feeding those channels hurts far more than error elsewhere. Rather than correcting for errors after the fact like GPTQ, AWQ **protects the salient channels before quantizing** by rescaling: it multiplies the weight columns corresponding to high-activation channels by a per-channel scale $s>1$ (making them larger and therefore relatively less perturbed by rounding) and divides the corresponding activations by the same $s$ — a transformation that leaves the mathematical output of $W^\top x$ unchanged but shifts quantization error away from the channels that matter most:

$$
y = W^\top x = (W \cdot \operatorname{diag}(s))^\top (x / s)
$$

The scale vector $s$ is found by a small calibration-data grid search (no backprop, no Hessian inversion), which makes AWQ substantially cheaper to run than GPTQ while empirically competitive at 4-bit — a good default when calibration-compute budget is tight, which is exactly our situation on a single rented GPU.

```python
def awq_find_channel_scales(W: torch.Tensor, X_calib: torch.Tensor, n_grid: int = 20) -> torch.Tensor:
    """Sketch of AWQ's salient-channel protection (no backprop; small grid search
    over a single global scaling strength alpha, per the AWQ paper's simplification).
    W: (d_out, d_in). X_calib: (n_samples, d_in)."""
    # Per-input-channel average activation magnitude — the saliency signal.
    act_scale = X_calib.abs().mean(dim=0)              # (d_in,)
    act_scale = act_scale / act_scale.mean()            # normalize

    best_scale, best_err = None, float("inf")
    for i in range(1, n_grid + 1):
        alpha = i / n_grid                               # search strength in [0, 1]
        s = act_scale.clamp(min=1e-5).pow(alpha)          # per-channel scale, s >= 1 on salient chans
        W_scaled = W * s.unsqueeze(0)                     # scale weight columns up
        # RTN-quantize the *rescaled* weight, then dequantize back.
        scale = W_scaled.abs().max() / 7                  # 4-bit symmetric, qmax=7
        W_q = torch.clamp(torch.round(W_scaled / scale), -7, 7) * scale
        # The real AWQ objective is the layer's OUTPUT error ||(W/s)^T x - dequant||;
        # this sketch uses the cheaper weight-space proxy (W_scaled - W_q) for clarity.
        err = (W_scaled - W_q).pow(2).mean().item()
        if err < best_err:
            best_err, best_scale = err, s
    return best_scale  # fold s into W before quant; fold 1/s into the preceding norm/layer
```

Stack-100M ships with RTN as the reference, fully-implemented path in §6 (correct, simple, and sufficient at int8; a real quality drop shows up at int4 on the harder tail of the arithmetic and MC probes — measure it yourself with §3's harness). GPTQ and AWQ are the production upgrade path once you notice that gap and want to close it.

{{fig:gptq-vs-awq-two-philosophies}}

## 6. Implementing int8 and int4 Weight-Only Quantization

The implementation below quantizes every `nn.Linear` weight in `StackLM` (attention projections, SwiGLU gate/up/down, and the tied embedding/output projection) and leaves norms and biases at fp32 — they are a vanishingly small fraction of total parameters and extremely sensitive to error. int8 uses per-output-channel (row-wise) symmetric scales, since a scale-per-row is nearly free in memory and already captures most of the benefit at 8 bits. int4 uses per-group asymmetric scales with `group_size=64` — both `d_model=512` and `intermediate=1408` from the [14.4 architecture](04-architecture.html) config divide evenly by 64 — because at 4 bits the coarser per-row scale is not enough precision to avoid a visible quality hit.

```python
"""
stacklm/quantize.py — round-to-nearest int8/int4 weight-only PTQ, with export
to a packed on-disk format for CPU inference (Ch. 14.11 reference path).
"""
import json
import struct
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core RTN quantize/dequantize primitives
# ---------------------------------------------------------------------------

def quantize_int8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization.
    weight: (d_out, d_in) fp32/bf16. Returns (int8 tensor, fp32 scales of shape (d_out,))."""
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0          # (d_out,)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127)
    return q.to(torch.int8), scale


def dequantize_int8_per_row(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(1)


def quantize_int4_grouped(weight: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-group int4 quantization, packed two values per uint8 byte.
    weight: (d_out, d_in), d_in must be divisible by group_size.
    Returns (packed uint8 tensor of shape (d_out, d_in//2), scales, zero_points),
    both of shape (d_out, d_in // group_size)."""
    d_out, d_in = weight.shape
    assert d_in % group_size == 0, f"d_in={d_in} not divisible by group_size={group_size}"
    n_groups = d_in // group_size

    w = weight.float().view(d_out, n_groups, group_size)          # (d_out, n_groups, group_size)
    w_min = w.amin(dim=2)                                          # (d_out, n_groups)
    w_max = w.amax(dim=2)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)               # 4-bit unsigned range: 0..15
    zero_point = torch.round(-w_min / scale)                       # integer zero-point

    q = torch.round(w / scale.unsqueeze(2) + zero_point.unsqueeze(2)).clamp(0, 15)
    q = q.view(d_out, d_in).to(torch.uint8)                        # (d_out, d_in), values in [0, 15]

    # Pack two 4-bit values into each byte: low nibble = even column, high nibble = odd column.
    q_even, q_odd = q[:, 0::2], q[:, 1::2]
    packed = (q_even | (q_odd << 4)).to(torch.uint8)               # (d_out, d_in // 2)
    return packed, scale, zero_point


def dequantize_int4_grouped(packed: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                             d_in: int, group_size: int = 64) -> torch.Tensor:
    d_out = packed.shape[0]
    n_groups = d_in // group_size
    q_even = (packed & 0x0F).to(torch.float32)
    q_odd = ((packed >> 4) & 0x0F).to(torch.float32)
    q = torch.empty(d_out, d_in, device=packed.device)
    q[:, 0::2], q[:, 1::2] = q_even, q_odd
    q = q.view(d_out, n_groups, group_size)
    w = (q - zero_point.unsqueeze(2)) * scale.unsqueeze(2)
    return w.view(d_out, d_in)


# ---------------------------------------------------------------------------
# Drop-in quantized Linear layer (dequantize-then-matmul reference path)
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """A memory-efficient stand-in for nn.Linear. Stores weights packed at
    int8 or int4 precision; on each forward call, dequantizes to fp32 and
    performs a standard matmul. This is the *pedagogically* correct approach
    — it is exactly what the weight compression buys you — but note it does
    NOT by itself give you a matmul FLOPs speedup on CPU, since PyTorch still
    computes in fp32. Real speedups need fused int-matmul kernels (torch's
    `quantize_dynamic` for int8, or GGML/llama.cpp-style kernels for int4);
    see the closing note in Ch. 14.11 §7."""

    def __init__(self, bits: int, group_size: int = 64):
        super().__init__()
        assert bits in (4, 8)
        self.bits = bits
        self.group_size = group_size
        self.d_in = None  # set at load time

    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int, group_size: int = 64):
        layer = cls(bits, group_size)
        layer.d_in = linear.in_features
        # register (not plain-assign) so the bias round-trips through state_dict;
        # StackLM is LLaMA-style bias-free, so this is usually None.
        layer.register_buffer(
            "bias", linear.bias.detach().clone() if linear.bias is not None else None
        )
        if bits == 8:
            q, scale = quantize_int8_per_row(linear.weight.data)
            layer.register_buffer("q_weight", q)
            layer.register_buffer("scale", scale)
        else:
            packed, scale, zp = quantize_int4_grouped(linear.weight.data, group_size)
            layer.register_buffer("q_weight", packed)
            layer.register_buffer("scale", scale)
            layer.register_buffer("zero_point", zp)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 8:
            w = dequantize_int8_per_row(self.q_weight, self.scale)
        else:
            w = dequantize_int4_grouped(self.q_weight, self.scale, self.zero_point,
                                          self.d_in, self.group_size)
        out = torch.nn.functional.linear(x, w.to(x.dtype), self.bias)
        return out


def quantize_stacklm(model: nn.Module, bits: int, group_size: int = 64) -> nn.Module:
    """Walk the model and replace every nn.Linear with a QuantizedLinear.
    RMSNorm scales and any biases stay at fp32 — a negligible parameter
    fraction, and error-sensitive. The tied input/output embedding is a single
    shared weight exposed as the LM-head nn.Linear (see Ch. 14.4), so it IS
    quantized by this loop; the input-side token lookup reads back that same
    quantized weight (dequantizing only the rows it gathers), so there is no
    separate fp32 embedding table — which is why §6's budget can quantize the
    full 101.4M parameters."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinear.from_float(module, bits, group_size))
        else:
            quantize_stacklm(module, bits, group_size)  # recurse into submodules
    return model


def export_quantized(model: nn.Module, path: str, bits: int, config: dict) -> None:
    """Serialize a quantized StackLM to disk: one .pt for tensors, one .json
    for metadata (bit-width, group size, and the architecture config from
    Ch. 14.4, needed to reconstruct the model shape at load time)."""
    torch.save(model.state_dict(), path + ".pt")
    with open(path + ".json", "w") as f:
        json.dump({"bits": bits, "group_size": 64, "architecture": config}, f, indent=2)
```

!!! example "Worked example: the memory budget, computed exactly"

    Stack-100M has ≈101.4M total parameters ([14.4](04-architecture.html)'s accounting). Here is what each format actually costs, computed rather than guessed:

    | Format | Weight storage | Scale/zero-point overhead | Total | vs. fp32 |
    |---|---|---|---|---|
    | fp32 | 101.4M × 4 B = 405.6 MB | — | **≈406 MB** | 1.0× |
    | bf16 | 101.4M × 2 B = 202.8 MB | — | **≈203 MB** | 2.0× |
    | int8 (row-wise) | 101.4M × 1 B = 101.4 MB | ~171K scales × 4 B ≈ 0.7 MB | **≈102 MB** | 4.0× |
    | int4 (group=64) | 101.4M × 0.5 B = 50.7 MB | ~1.58M groups × (fp32 scale + fp32 zp, 8 B) ≈ 12.7 MB | **≈63 MB** | 6.4× |

    (Group count for int4: 101.4M params ÷ 64 ≈ 1.58M groups; each group stores one fp32 scale **and** one fp32 zero-point in this reference implementation, i.e. 8 bytes/group ≈ 12.7 MB — a real chunk of the total, and exactly why production formats compress it. A typical export stores the scale at fp16 and the zero-point at int8 (≈5 B/group ≈ 8 MB, dropping int4 to ~59 MB); `GGUF`'s block-quantization formats go further with an int8-with-a-shared-super-scale layout.) The headline: **the whole model, quantized, is smaller than a typical high-resolution JPEG photo folder** — it sits comfortably in RAM alongside a browser and an IDE on any laptop.

{{fig:quant-memory-ladder-and-packing}}

## 7. Running Stack-100M on a Laptop CPU

The final step: load the quantized checkpoint and generate text with no GPU involved. The KV-cache decode loop below follows the same anatomy taught in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) — a prefill pass over the prompt, then autoregressive decode steps each attending only to the growing cache — with GQA's 2 KV heads keeping that cache small, as covered in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) and [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

```python
"""
stacklm/generate.py — CPU text generation from a quantized checkpoint, with
measured latency and memory. Run: `python -m stacklm.generate --bits 4`
"""
import argparse
import json
import os
import time
import tracemalloc

import torch

from stacklm.model import StackLM, StackLMConfig
from stacklm.tokenizer import Tokenizer                 # Ch. 14.3
from stacklm.quantize import quantize_stacklm            # loads QuantizedLinear in-place


def load_quantized_model(path_prefix: str) -> StackLM:
    with open(path_prefix + ".json") as f:
        meta = json.load(f)
    config = StackLMConfig(**meta["architecture"])
    model = StackLM(config)
    # Replace nn.Linear modules with the (empty) QuantizedLinear shape first,
    # so load_state_dict's tensor shapes match the packed/quantized buffers.
    quantize_stacklm(model, bits=meta["bits"], group_size=meta["group_size"])
    state_dict = torch.load(path_prefix + ".pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def generate(model: StackLM, tokenizer: Tokenizer, prompt: str,
             max_new_tokens: int = 64, temperature: float = 0.7) -> str:
    """Greedy/temperature sampling with KV cache — same interface used by the
    eval probes in §3. torch.set_num_threads() upstream controls CPU parallelism."""
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    kv_cache = model.init_kv_cache(batch_size=1, max_seq_len=input_ids.shape[1] + max_new_tokens)

    # Prefill: one forward pass over the whole prompt, filling the KV cache.
    logits, kv_cache = model(input_ids, kv_cache=kv_cache, start_pos=0)
    next_token = _sample(logits[:, -1, :], temperature)
    generated = [next_token.item()]

    # Decode: one token at a time, each step attends prompt + all prior generated tokens.
    for step in range(max_new_tokens - 1):
        logits, kv_cache = model(next_token.unsqueeze(0), kv_cache=kv_cache,
                                   start_pos=input_ids.shape[1] + step)
        next_token = _sample(logits[:, -1, :], temperature)
        if next_token.item() == tokenizer.eos_id:
            break
        generated.append(next_token.item())

    return tokenizer.decode(generated)


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stack100m")
    parser.add_argument("--prompt", type=str, default="The mitochondria is")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count())  # use all available CPU cores

    tracemalloc.start()
    t_load_start = time.perf_counter()
    model = load_quantized_model(f"{args.checkpoint_dir}/stack100m_int{args.bits}")
    tokenizer = Tokenizer.load(f"{args.checkpoint_dir}/tokenizer.json")
    load_time = time.perf_counter() - t_load_start

    t_gen_start = time.perf_counter()
    text = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)
    gen_time = time.perf_counter() - t_gen_start
    peak_mem = tracemalloc.get_traced_memory()[1] / 1e6  # MB (Python-object peak; add
    tracemalloc.stop()                                    # weight bytes from §6's table)

    tok_per_sec = args.max_new_tokens / gen_time
    print(f"--- Stack-100M (int{args.bits}) on CPU ---")
    print(f"Load time:        {load_time:.2f} s")
    print(f"Generation time:  {gen_time:.2f} s  ({tok_per_sec:.1f} tok/s)")
    print(f"Peak Python heap: {peak_mem:.1f} MB  (add ~{63 if args.bits == 4 else 102} MB weights)")
    print(f"Output: {args.prompt}{text}")


if __name__ == "__main__":
    main()
```

!!! example "Worked example: the full laptop memory budget"

    Put together every piece measured or computed in this chapter, for a generation call with `max_seq_len = 2048` at int4:

    - **Quantized weights**: ≈63 MB (§6 table — 50.7 MB packed int4 + 12.7 MB fp32 scales/zero-points).
    - **KV cache**, GQA with `n_kv_heads=2`, `head_dim=64`, 30 layers, bf16: per token per layer, K+V together cost $2 \times 64 \times 2 = 256$ elements $\times$ 2 bytes = 512 bytes; across 30 layers that's $512 \times 30 = 15{,}360$ bytes (15 KB) per token; at the full 2048-token context, $15{,}360 \times 2048 \approx 31.5$ MB.
    - **Activations** (single-token decode step, transient): a few MB at most — nowhere near the weight or cache budget.

    Total: **on the order of 100 MB** to hold the entire model plus a full 2048-token conversation in memory — small enough to run comfortably alongside a browser and an IDE on any laptop from the last decade, and small enough to fit on a Raspberry Pi. This is the concrete payoff of stacking GQA (4× smaller KV cache than plain multi-head attention), a modest 32768-token vocabulary (small embedding table), and int4 weight quantization (≈6× smaller than fp32, scale overhead included) — each individually a modest win, compounding into "runs anywhere."

!!! tip "Practitioner tip: why int4 weights don't automatically mean int4 speed"

    The `QuantizedLinear` above dequantizes to fp32 before every matmul — it buys **memory** (RAM and disk), not automatically **compute** speed, since the matmul itself still runs in fp32. For a real CPU throughput win at int8, PyTorch's native `torch.ao.quantization.quantize_dynamic` routes matmuls through Intel's FBGEMM or ARM's QNNPACK backend, which perform the multiply-accumulate directly in integer arithmetic. For int4 on CPU, the ecosystem's fastest production path is not a stock PyTorch op at all — it is llama.cpp's GGUF format (Gerganov et al.), which fuses the unpack-and-matmul into a single hand-written kernel per architecture (AVX2/AVX-512/NEON). Our reference implementation is deliberately the transparent, correct, *slow-path* version, so you can see exactly what quantization does to the numbers; converting the exported `stack100m_int4.pt` to GGUF and running it under llama.cpp is the natural "now make it fast" follow-up exercise, and a good place to reread [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) and [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) with this exact model in mind. Measure your own `tok/s` with the script above before and after — the gap is the lesson.

!!! interview "Interview Corner"

    **Q:** You quantize a model to int4 with round-to-nearest and perplexity barely moves, but a downstream multi-step reasoning eval drops sharply. What's going on, and how would GPTQ or AWQ help?

    **A:** Perplexity averages error across the entire vocabulary and every position, so it's dominated by the easy majority of predictions and can mask damage to a small number of high-leverage weights — the ones a specific reasoning chain depends on at a specific step. RTN quantizes every weight independently with no notion of which weights the model's *output* is most sensitive to, so it can silently wreck a handful of outlier channels that barely move the average loss but matter enormously for a token the model must get exactly right mid-chain. GPTQ addresses this directly: after quantizing each column, it uses the layer's Hessian to redistribute the resulting error onto the columns not yet quantized, explicitly minimizing the *layer's output reconstruction error*, not treating weights as independent. AWQ takes a cheaper, complementary approach: it identifies the small set of activation channels with systematically large magnitude (empirically the ones that matter most) and rescales weights so those specific channels see less quantization error in the first place. Both are strictly better than RTN at preserving exactly the kind of high-leverage precision that a coarse, position-blind metric like perplexity would never surface as a problem — which is why production int4 deployments use GPTQ/AWQ rather than RTN, and why this chapter treats RTN as the pedagogical baseline, not the final answer.

## 8. Closing the Loop

You now have every artifact the capstone promised: a trained, mid-trained, aligned, tool-using ~101M-parameter model; an honest, reproducible evaluation report with explicit disclosure of what it cannot do; and a quantized export that runs, measured, on ordinary CPU hardware. [14.12 Retrospective & Scale-Up](12-retrospective-and-scaleup.html) closes the capstone with the full cost accounting and a concrete map of what changes if you decide to push past 100M parameters. Before that: run `stacklm.generate` on your own laptop, on your own checkpoint, and read what it wrote. That sentence is the actual point of the entire capstone — not the perplexity number, not the tok/s number, but the fact that a model you trained end to end, on hardware you rented for less than a dinner out, just generated a sentence on the machine in front of you.

!!! key "Key Takeaways"

    - A model is not "done" at a loss curve. It is done at a defensible number (held-out perplexity, provably excluded from every training stage) plus a set of narrow, cheap probes (arithmetic exact-match, cloze-scored multiple-choice, retrieval-QA exact-match) plus an honest, explicit statement of what the model cannot do.
    - Perplexity measures predictive fit to held-out text and nothing else; it can stay flat while a downstream capability collapses, because it averages over the whole vocabulary and masks damage to a small number of high-leverage weights or tokens.
    - Cloze-style scoring (comparing full-sequence log-probabilities of candidate answers) is more reliable than asking a 100M model to emit a formatted letter choice — instruction-following that precise is not something a model this small reliably has.
    - Contamination is a real risk even for hand-built probes: audit your MC set and retrieval corpus against the training data's deduplication index, not just against public benchmarks.
    - Round-to-nearest (RTN) quantization is a fast, calibration-free baseline; GPTQ (Frantar et al., 2022) corrects each column's quantization error onto not-yet-quantized columns via the layer's Hessian; AWQ (Lin et al., 2023) instead protects a small set of high-activation "salient" channels by rescaling before quantization. Both beat RTN at 4 bits precisely where RTN is weakest: high-leverage weights.
    - Weight-only quantization computed exactly for Stack-100M: fp32 ≈406MB → bf16 ≈203MB → int8 ≈102MB (4×) → int4 ≈63MB (6.4×, counting the fp32 scale/zero-point overhead honestly) — the whole model smaller than a folder of photos.
    - Quantizing weights to int4 shrinks memory, but does not automatically speed up matmul on stock PyTorch CPU — real throughput gains need integer-native kernels (FBGEMM/QNNPACK for int8 via `quantize_dynamic`, or llama.cpp/GGUF-style fused kernels for int4).
    - GQA's 4× smaller KV cache, a right-sized 32768-token vocabulary, and int4 weights compound: the full model plus a 2048-token conversation fits in roughly 100MB of RAM — small enough for any laptop, and small enough for a Raspberry Pi.
    - The payoff of the entire capstone is concrete and checkable in one command: run `stacklm.generate --bits 4` and read text your own model produced, on your own machine.

!!! sota "State of the Art & Resources (2026)"
    Weight-only PTQ has matured into a commodity operation — GPTQ and AWQ are now standard export paths in every major serving stack, GGUF's block-quantized formats dominate CPU/edge deployment, and the frontier has pushed toward sub-4-bit and even ternary weights while small-model evaluation has consolidated around a handful of reproducible open harnesses.

    **Foundational work**

    - [Frantar, Ashkboos, Hoefler & Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022)](https://arxiv.org/abs/2210.17323) — the Hessian-based, column-by-column reconstruction method this chapter's §5.2 implements a pedagogical version of.
    - [Dettmers, Lewis, Belkada & Zettlemoyer, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022)](https://arxiv.org/abs/2208.07339) — the mixed-precision decomposition that made int8 inference practical for transformers with outlier activation features.

    **Recent advances (2023–2026)**

    - [Lin, Tang, Tang, Yang, Dang & Han, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023)](https://arxiv.org/abs/2306.00978) — the salient-channel-protection method §5.3 implements; MLSys 2024 Best Paper.
    - [Xiao, Lin, Seznec, Wu, Demouth & Han, *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (2023)](https://arxiv.org/abs/2211.10438) — migrates quantization difficulty from activations to weights via a per-channel smoothing factor, enabling accurate W8A8 (not just weight-only) quantization.
    - [Ma, Wang, Ma, Wang, Wang, Huang, Dong, Wang, Xue & Wei, *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (2024)](https://arxiv.org/abs/2402.17764) — ternary {-1, 0, 1} weights matching full-precision perplexity at equal size/tokens, the current frontier past int4.

    **Open-source & tools**

    - [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the production reference for fused, quantized CPU/edge inference and the GGUF format this chapter's §7 practitioner tip points to as the "make it fast" follow-up.
    - [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — the library behind `torch.ao`-adjacent 8-bit/4-bit `Linear` layers and QLoRA, referenced in §7 as the CPU/GPU integer-kernel path.
    - [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) — actively maintained production GPTQ/AWQ/GGUF quantization toolkit with Hugging Face Transformers, vLLM, and SGLang integration.
    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the framework whose cloze-scoring approach §3.2 borrows for MC probes; backs the Hugging Face Open LLM Leaderboard.

    **Go deeper**

    - [GGUF (Hugging Face Hub docs)](https://huggingface.co/docs/hub/en/gguf) — the on-disk block-quantization format (Q4_K, Q6_K, IQ-series, and more) referenced throughout §5–§7 as the production upgrade from this chapter's reference int8/int4 packing.

## Further reading

- Frantar, Ashkboos, Hoefler & Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022.
- Lin, Tang, Tang, Yang, Dang & Han, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023.
- Dettmers, Lewis, Belkada & Zettlemoyer, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, 2022.
- Jacob, Kligys, Chen, Zhu, Tang, Howard, Adam & Kalenichenko, *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, 2018.
- Gao et al. (EleutherAI), *A Framework for Few-Shot Language Model Evaluation* (`lm-evaluation-harness`).
- Hendrycks, Burns, Basart, Zou, Mazeika, Song & Steinhardt, *Measuring Massive Multitask Language Understanding* (MMLU), 2021.
- Kwiatkowski et al., *Natural Questions: A Benchmark for Question Answering Research*, 2019.
- Gerganov et al., `llama.cpp` and the GGUF format — the production reference for fused, quantized CPU/edge inference.
- Cross-reference: [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html), [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html), [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html), [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html), [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html).

## Exercises

**1.** Your `compute_perplexity` run on the held-out shard reports a mean loss of **3.1 nats/token**. (a) Convert this to perplexity. (b) The function returns *both* `loss_nats_per_token` and `perplexity`. Give the specific reason §2 offers for reporting the nats/token number rather than perplexity alone. (c) A random-guessing model over Stack-100M's vocabulary would sit at what perplexity, and what does the gap between that and your number tell you?

??? note "Solution"
    (a) $\text{PPL} = \exp(3.1)$. Since $e^{3} \approx 20.09$ and $e^{0.1} \approx 1.105$, $\text{PPL} \approx 20.09 \times 1.105 \approx 22.2$. The model's predictive distribution has an effective branching factor of about 22 tokens per position on this held-out text.

    (b) §2 gives two reasons: loss in nats/token **composes linearly** (perplexity does not — it is an exponential), and it is **exactly the quantity the training curve in [14.7] already plots**. So you can drop an "eval loss" point directly onto the same chart as the training loss and read off the generalization gap by eye. Perplexity is the human-facing summary; nats/token is the composable, chart-compatible one.

    (c) A model that assigns uniform probability $1/V$ to every token over the $V = 32768$-vocabulary has $\text{PPL} = V = 32768$. Your $\approx 22.2$ is roughly three orders of magnitude below that, which is the concrete evidence that the model learned real predictive structure in the training-mix text (and, per §2, *nothing more than that* — it says nothing about arithmetic or factual accuracy).

**2.** Using the exact memory accounting in §6 (101.4M total parameters, quantized end to end), compute the int4 on-disk size and compression-vs-fp32 ratio for **`group_size = 128`** instead of the default 64. The reference int4 layout stores one fp32 scale plus one fp32 zero-point per group. Compare against the default (group=64) numbers in the §6 table and state the tradeoff you are making by widening the group.

??? note "Solution"
    Packed 4-bit weights are independent of group size: $101.4\text{M} \times 0.5\ \text{B} = 50.7\ \text{MB}$.

    Group count at `group_size = 128`: $101.4\text{M} \div 128 \approx 792{,}000$ groups (half as many as the $\approx 1.58\text{M}$ groups at group=64). Each group stores one fp32 scale + one fp32 zero-point $= 8\ \text{B}$:

    $$792{,}000 \times 8\ \text{B} \approx 6.34\ \text{MB}.$$

    Total $\approx 50.7 + 6.34 \approx 57.0\ \text{MB}$. Compression vs. fp32: $406 \div 57.0 \approx 7.1\times$ (versus $\approx 63\ \text{MB}$ / $6.4\times$ at group=64).

    Tradeoff: doubling the group size **halves the scale/zero-point overhead** ($\approx 12.7\ \text{MB} \to \approx 6.3\ \text{MB}$), but each scale/zero-point now has to cover 128 weights with a single min/max instead of 64, so the quantization grid is **coarser** and the reconstruction error grows — exactly the "not enough precision to avoid a visible quality hit" concern §6 raises for 4-bit. You would confirm the quality cost empirically with the §3 probe harness before shipping the wider group.

**3.** Work RTN by hand. Take a single group of four weights $w = [\,0.90,\ -0.40,\ 0.10,\ -0.85\,]$ and apply the **$b=4$ symmetric** scheme from §5.1 (so $2^{b-1}-1 = 7$). (a) Compute the scale $s$. (b) Compute the integer codes $q$ and the dequantized weights $\hat w$. (c) Report the maximum absolute reconstruction error.

??? note "Solution"
    (a) Symmetric scale: $s = \dfrac{\max(|w|)}{2^{b-1}-1} = \dfrac{0.90}{7} \approx 0.12857$.

    (b) $q_i = \operatorname{clip}(\operatorname{round}(w_i/s),\,-7,\,7)$:

    - $0.90/0.12857 = 7.00 \to 7$
    - $-0.40/0.12857 = -3.11 \to -3$
    - $0.10/0.12857 = 0.78 \to 1$
    - $-0.85/0.12857 = -6.61 \to -7$

    So $q = [\,7,\ -3,\ 1,\ -7\,]$. Dequantize $\hat w_i = s\,q_i$:

    - $7 \times 0.12857 = 0.9000$
    - $-3 \times 0.12857 = -0.3857$
    - $1 \times 0.12857 = 0.1286$
    - $-7 \times 0.12857 = -0.9000$

    So $\hat w = [\,0.900,\ -0.386,\ 0.129,\ -0.900\,]$.

    (c) Per-element errors $w_i - \hat w_i$: $0.000,\ -0.0143,\ -0.0286,\ +0.0500$. The **maximum absolute error is $0.050$** (on the $-0.85$ weight). Note the largest error falls on $-0.85$, whose scaled value $-6.61$ lands nearest a rounding boundary (a $\approx 0.39$-step residual), not on the largest-magnitude weight $0.90$, which sits exactly on the grid endpoint with zero error — a small illustration of RTN's blind spot from §5.1: it rounds every weight independently by grid proximity alone, with no notion of which one matters more to the layer's output.

**4.** `_sequence_logprob` in §3.2 scores a candidate answer by the **sum** of its per-token log-probabilities, and `eval_mc_probe` picks the choice with the highest score. Suppose one item's choices tokenize to very different lengths — e.g. one option is 1 token and another is 5 tokens. (a) Explain the systematic bias this introduces. (b) Why is it not a problem for the `TINY_MC_SET` items actually shown in the chapter? (c) Name one thing you would check or change before adding a longer free-form option.

??? note "Solution"
    (a) Every token contributes a $\log p \le 0$ term, so each additional continuation token can only push the summed score **downward** (more negative). Longer candidate strings therefore accumulate more negative mass purely by being longer, independent of correctness. With the raw-sum scoring shown, `max(scores)` is biased toward the **shorter** option — a 1-token choice can win over a correct-but-longer 5-token choice just because it stops accumulating penalty sooner.

    (b) The chapter's `TINY_MC_SET` choices are deliberately near-uniform in length (`" H2O"`, `" CO2"`, `" NaCl"`, `" O2"`; `" Paris"`, `" Berlin"`, `" Madrid"`, `" Rome"`) — all short and comparable — so the length bias is roughly constant across options and cancels out of the arg-max. The construction ("small enough to eyeball every item") keeps this invariant by hand.

    (c) You would either (i) verify the choices tokenize to comparable lengths, or (ii) switch from the raw sum to a **length-normalized** score, e.g. dividing the summed log-prob by the number of continuation tokens (mean log-prob per token), so that a longer correct answer is not penalized for its length. Either keeps the arg-max honest once options differ in length.

**5.** Compute the KV-cache budget the way §7's worked example does, using the same GQA config (`n_kv_heads = 2`, `head_dim = 64`, 30 layers, bf16 = 2 bytes/element). (a) Bytes per token per layer, and per token across all layers. (b) Total cache for a **512-token** context. (c) The chapter says GQA gives a "4x smaller KV cache than plain multi-head attention." What would the same 512-token cache cost under plain MHA, and what does that imply about the number of query heads?

??? note "Solution"
    (a) Per token per layer, we store both K and V, each of shape `n_kv_heads x head_dim`:

    $$2\ (\text{K and V}) \times 2\ (\text{kv heads}) \times 64\ (\text{head dim}) = 256\ \text{elements} \times 2\ \text{B} = 512\ \text{B}.$$

    Across 30 layers: $512 \times 30 = 15{,}360\ \text{B} \approx 15\ \text{KB per token}$.

    (b) At 512 tokens: $15{,}360 \times 512 = 7{,}864{,}320\ \text{B} \approx 7.86\ \text{MB}$.

    (c) "4x smaller" means plain MHA would cost $4 \times 7.86 \approx 31.5\ \text{MB}$ for the same context, because MHA stores K and V for every query head rather than sharing 2 KV heads. A 4x-larger cache with `head_dim` fixed means MHA uses $4 \times 2 = 8$ KV heads — i.e. the model has **8 query heads**, and GQA collapses them onto 2 shared KV heads (a 4:1 grouping). This is the compounding win §7 highlights: the smaller cache is what keeps "model + full 2048-token conversation" near 100 MB.

**6.** Implement a **symmetric** per-group int4 variant of `quantize_int4_grouped` / `dequantize_int4_grouped` (call them `..._symmetric`), following §5.1's symmetric scheme and reusing §6's two-nibbles-per-byte packing. Then answer: (a) how much on-disk memory does dropping the zero-point save for the full 101.4M-parameter Stack-100M at `group_size = 64`, and (b) why did §6 nonetheless choose *asymmetric* int4 as the reference path?

??? note "Solution"
    Symmetric int4 uses codes in $[-7, 7]$ with $s = \max(|w|)/7$ and no zero-point. The signed range spans 15 levels, which still fits a 4-bit nibble; we shift by $+7$ to $[0, 14]$ only for packing, and undo it on dequantize.

    ```python
    def quantize_int4_grouped_symmetric(weight: torch.Tensor, group_size: int = 64):
        """Symmetric per-group int4, packed two values per uint8 byte.
        Returns (packed uint8 (d_out, d_in//2), scales (d_out, d_in//group_size))."""
        d_out, d_in = weight.shape
        assert d_in % group_size == 0, f"d_in={d_in} not divisible by {group_size}"
        n_groups = d_in // group_size

        w = weight.float().view(d_out, n_groups, group_size)
        scale = (w.abs().amax(dim=2) / 7.0).clamp(min=1e-8)        # (d_out, n_groups)
        q = torch.round(w / scale.unsqueeze(2)).clamp(-7, 7)        # codes in [-7, 7]
        q = (q + 7).to(torch.uint8).view(d_out, d_in)              # shift to [0, 14] for packing

        q_even, q_odd = q[:, 0::2], q[:, 1::2]
        packed = (q_even | (q_odd << 4)).to(torch.uint8)           # (d_out, d_in // 2)
        return packed, scale


    def dequantize_int4_grouped_symmetric(packed: torch.Tensor, scale: torch.Tensor,
                                          d_in: int, group_size: int = 64) -> torch.Tensor:
        d_out = packed.shape[0]
        n_groups = d_in // group_size
        q_even = (packed & 0x0F).to(torch.float32)
        q_odd = ((packed >> 4) & 0x0F).to(torch.float32)
        q = torch.empty(d_out, d_in, device=packed.device)
        q[:, 0::2], q[:, 1::2] = q_even, q_odd
        q = (q - 7.0).view(d_out, n_groups, group_size)            # undo the +7 shift
        w = q * scale.unsqueeze(2)                                 # no zero-point term
        return w.view(d_out, d_in)
    ```

    (This drops straight into `QuantizedLinear.from_float`/`forward`: for `bits == 4` you would register `q_weight` and `scale` only, with no `zero_point` buffer.)

    (a) The reference asymmetric layout stores an fp32 scale **and** an fp32 zero-point (8 B) per group; symmetric stores only the fp32 scale (4 B). At $\approx 1.58\text{M}$ groups that saves $1.58\text{M} \times 4\ \text{B} \approx 6.3\ \text{MB}$ — the int4 total drops from $\approx 63\ \text{MB}$ to $\approx 57\ \text{MB}$ (about $7.1\times$ vs. fp32).

    (b) §6 chose asymmetric because at only 4 bits the extra precision matters more than the 6 MB. A symmetric scheme forces the grid to be centered at 0 and cover $[-\max|w|,\ +\max|w|]$; for a **skewed** group (per §5.1, "better for skewed distributions") this wastes codes on a range the weights never occupy. Asymmetric fits the actual $[\min(w),\ \max(w)]$ with a learned zero-point, giving finer effective resolution exactly where 4-bit RTN is most fragile. The memory saving is real but small relative to that quality risk, so the reference path keeps the zero-point.
