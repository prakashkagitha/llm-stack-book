# 3.15 Synthetic Data for Pre- and Post-Training

For most of deep learning's history, "data" meant something you *found*: ImageNet was scraped and hand-labeled, Common Crawl was harvested from the open web, and the implicit contract was that the internet would always be bigger than your model's appetite. That contract is breaking. Frontier models are now trained on tens of trillions of tokens, and the supply of *high-quality* human-written text — clean prose, correct code, well-structured explanations — is finite and largely exhausted. The Chinchilla math from [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) says we want roughly 20 tokens per parameter; a 1-trillion-parameter model wants ~20T tokens, and the clean, deduplicated, English subset of the web is smaller than that. We are running into a **data wall**.

Synthetic data is the response. Instead of (or in addition to) scraping tokens, we *generate* them with a model — rephrasing messy web text into clean prose, writing textbook-style explanations of concepts, manufacturing instruction–answer pairs, or distilling the step-by-step reasoning of a strong teacher into a smaller student. Done well, this is one of the highest-leverage techniques in the modern stack: it converts *compute* (cheap, scalable) into *high-quality tokens* (scarce), and it lets you target exactly the distribution you want. Done badly, it is a slow-motion catastrophe — **model collapse**, where models trained on their own outputs progressively forget the tails of the distribution and converge to bland, narrowed mush.

This chapter is about doing it well. We will cover the four dominant recipes — web rephrasing (WRAP), textbook-style generation (Phi, Cosmopedia, FineWeb-Edu), instruction-augmented pretraining, and reasoning-trace distillation for post-training. We will derive when mixing synthetic with natural data helps versus hurts, build the math of model collapse from first principles, write runnable generation-and-verification pipelines, and catalog the failure modes — contamination, distributional narrowing, fact drift — that separate a useful synthetic corpus from a poisoned one.

This chapter builds on [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) (where natural tokens come from), [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) (the filters synthetic data must also pass), and [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html) (how to weight a natural/synthetic blend). The post-training half connects to [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html), and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

---

## Why Generate Tokens At All?

### The data wall, quantified

Let us make the scarcity concrete. Estimates of the high-quality public text stock vary, but a useful order-of-magnitude figure is that the *clean, deduplicated, English* web is on the order of $10^{13}$ tokens — call it tens of trillions, with a long tail of lower-quality text beyond that. Meanwhile training token counts have grown roughly an order of magnitude every couple of years. When the model's appetite catches up to the supply, you have three moves:

1. **Train more epochs over the same data.** Repetition gives diminishing returns and eventually *hurts* — re-seeing the same tokens four-plus times yields little new signal and risks memorization (see [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html)). The Muennighoff et al. "data-constrained scaling" work quantifies this: up to ~4 epochs is roughly as good as fresh data, after which value decays fast.
2. **Lower the quality bar** and ingest more of the noisy tail. This dilutes the average token's value and can degrade the model.
3. **Manufacture new high-quality tokens** with a generator model. This is synthetic data.

The third option is attractive because of an asymmetry: **verifying or restyling text is easier than writing it from scratch**, and a model that is mediocre at open-ended generation can still be excellent at *transforming* existing good text. That asymmetry — generation-easier-than-from-scratch, verification-easier-than-generation — is the engine behind every recipe in this chapter.

### A taxonomy of synthetic data

It helps to classify by *how far* the synthetic text departs from a human-written source. The further you go, the more value you can add — and the more risk you take on.


{{fig:syndata-grounding-spectrum}}


- **Source-grounded transformation** (WRAP-style rephrasing). Take a real document; rewrite it in a target style (clean, formal, Q&A). Facts are inherited from the source, so hallucination risk is bounded by the source's accuracy plus paraphrase drift.
- **Knowledge-augmented generation.** Seed with a real passage but *add* structure the model knows — worked examples, contrasting cases, follow-up questions. Risk rises because added content is not grounded.
- **Prompt-seeded generation** (Cosmopedia, Phi textbooks). Seed only with a topic, audience, and format ("write a textbook section on photosynthesis for a high-schooler"). The model writes from its parametric knowledge. Highest density of clean pedagogical text, highest hallucination risk.
- **Reasoning-trace distillation** (post-training). Have a strong teacher solve problems with chain-of-thought, then train a student on those traces — ideally keeping only traces whose final answer is *verified* correct.

---

## Recipe 1 — Rephrasing the Web (WRAP)

The lowest-risk, highest-ROI recipe is **Web Rephrasing Augmented Pretraining (WRAP)**, introduced by Maini et al. The insight: raw web text is information-rich but *stylistically poor* — boilerplate, broken HTML remnants, SEO spam, inconsistent formatting. Loss on such text is dominated by predicting noise. If you take each web document and ask an instruction-tuned model to rewrite it in one of a few clean styles, you get tokens that carry the same information at much higher "signal per token."

WRAP uses a small set of rewrite styles, for example:

- **Easy / Wikipedia-style**: clear expository prose, like an encyclopedia entry.
- **Q&A**: turn the passage into question–answer pairs.
- **Concise / formal**: tighten and formalize.

The reported result is striking: pretraining on a *mix* of original web text plus its rephrasings reaches a target perplexity with roughly **3× fewer tokens** than web text alone, and improves downstream zero-shot accuracy — without needing any human-written corpus beyond the original web. Crucially, you **keep the original document too**; the rephrasing augments rather than replaces, which preserves the factual grounding and the natural diversity of the source.

### A runnable WRAP pipeline

Here is the core loop, using a vLLM-style batched generator (see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html)). The economics matter — you are running inference over *trillions* of tokens, so batching and a small generator are essential.

```python
"""
WRAP-style web rephrasing. For each web doc, sample one of a few rewrite
styles and generate a clean version. Keep BOTH original and rephrasing.
"""
import random
from dataclasses import dataclass

# A small, fixed menu of rewrite styles. Diversity of styles is what keeps
# the synthetic distribution from collapsing onto one bland mode.
STYLE_PROMPTS = {
    "wiki":    "Rewrite the following text as a clear, encyclopedic passage. "
               "Preserve all factual content; remove boilerplate and noise.\n\n{doc}",
    "qa":      "Convert the following text into 2-4 self-contained "
               "question-and-answer pairs covering its key facts.\n\n{doc}",
    "concise": "Rewrite the following text concisely and formally, keeping "
               "every fact but removing redundancy.\n\n{doc}",
}

@dataclass
class WrapConfig:
    max_input_chars: int = 6000      # truncate very long docs before rewriting
    max_new_tokens: int = 1024
    temperature: float = 0.7         # some entropy -> stylistic diversity
    styles_per_doc: int = 1          # how many rephrasings per source doc

def build_prompts(docs, cfg: WrapConfig):
    """Yield (doc_id, style, prompt) for batched generation."""
    for doc_id, doc in docs:
        doc = doc[: cfg.max_input_chars]
        for _ in range(cfg.styles_per_doc):
            style = random.choice(list(STYLE_PROMPTS))
            yield doc_id, style, STYLE_PROMPTS[style].format(doc=doc)

def rephrase_corpus(docs, llm, cfg: WrapConfig):
    """
    `llm.generate(prompts, ...)` is the vLLM batched API. We process the whole
    shard at once so the GPU stays saturated -- throughput, not latency, is
    what dominates cost when rewriting trillions of tokens.
    """
    items = list(build_prompts(docs, cfg))
    prompts = [p for (_, _, p) in items]
    outputs = llm.generate(
        prompts,
        sampling_params=dict(temperature=cfg.temperature,
                             max_tokens=cfg.max_new_tokens),
    )
    for (doc_id, style, _), out in zip(items, outputs):
        text = out.outputs[0].text.strip()
        # Emit BOTH the original (once) and the rephrasing. Keeping the
        # original preserves grounding and natural diversity.
        yield {"doc_id": doc_id, "kind": "synthetic",
               "style": style, "text": text}
```

### Why keep the original — and how much to mix

If you trained *only* on rephrasings, you would inherit every quirk of the rewriter: its preferred sentence rhythms, its vocabulary, its tendency to "tidy away" rare facts. Keeping the original web text anchors the model to the true data distribution. The empirical sweet spot in WRAP-style work is roughly a **1:1 mix** of natural and synthetic — enough synthetic to lift quality, enough natural to preserve the tails. We will formalize this trade-off in the model-collapse section below.

!!! tip "Practitioner tip"
    Use the *smallest* rewriter that produces clean output. The rewrite task is easy relative to open-ended generation, so a 1–8B instruction model is often plenty, and at trillion-token scale the difference between a 7B and a 70B rewriter is the difference between a feasible and an infeasible budget. Quantize the rewriter (see [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html)) and serve it with continuous batching.

---

## Recipe 2 — Textbook-Style Generation (Phi, Cosmopedia, FineWeb-Edu)

The second recipe goes further: generate *new* expository content rather than rewriting a specific source. The thesis, from Microsoft's **Phi** series ("Textbooks Are All You Need," Gunasekar et al.), is that data *quality* can substitute for *quantity* and *model size*. A small model trained on a curated diet of "textbook-quality" text — clear, pedagogically structured, low-noise — can match much larger models trained on raw web scrapings. Phi-1 (1.3B) reached strong code performance on a tiny, curated corpus; later Phi models extended the recipe to general reasoning.

There are two ways to get textbook-quality tokens:

1. **Filter the web for them** (FineWeb-Edu). Train a lightweight classifier to score how "educational" a web page is, then keep only the high-scoring pages. This is *selection*, not generation — but it is the natural-data analogue of textbook generation and pairs with it. See [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html).
2. **Generate them** (Cosmopedia, the open reproduction of Phi's synthetic data). Prompt a strong model with a *seed* (a topic, an audience, a format) and have it write the passage. The art is in the *prompt diversity* — without it, you generate the same fifty essays a million times.

### The FineWeb-Edu educational classifier

The filter half is cheap and worth always doing. You score the web with a small regression head on top of frozen embeddings, trained on a few hundred thousand LLM-labeled examples.

```python
"""
FineWeb-Edu-style educational quality filter. A strong LLM labels a seed set
with an 'educational value' score 0-5; we distill that into a fast linear head
on frozen sentence embeddings, then score the whole web with the cheap head.
"""
import numpy as np
from sklearn.linear_model import Ridge

def make_labeling_prompt(text: str) -> str:
    # The rubric is the product. Be explicit about what 'educational' means
    # so the labels are consistent across millions of pages.
    return (
        "Rate the educational value of the text for teaching a student, "
        "on an additive 0-5 scale:\n"
        "+1 if it contains some educational content,\n"
        "+1 if it addresses a topic relevant to school/college curricula,\n"
        "+1 if it is coherent and well-written,\n"
        "+1 if it is comprehensive and self-contained,\n"
        "+1 if it is outstanding, like a high-quality textbook.\n"
        "Reply with ONLY the integer.\n\nTEXT:\n" + text[:4000]
    )

def train_edu_head(embeds: np.ndarray, scores: np.ndarray) -> Ridge:
    """embeds: (N, d) frozen embeddings; scores: (N,) LLM labels in [0,5]."""
    head = Ridge(alpha=1.0)
    head.fit(embeds, scores)
    return head

def keep_mask(head: Ridge, embeds: np.ndarray, threshold: float = 3.0):
    """Keep only pages the head scores >= threshold (e.g. 3/5)."""
    preds = head.predict(embeds)
    return preds >= threshold
```

A threshold around 3/5 typically keeps a small single-digit percentage of the web but yields a corpus on which models learn dramatically faster per token. The lesson generalizes: **a cheap classifier distilled from expensive LLM labels is one of the best dollar-for-dollar quality levers in pretraining.**

### Cosmopedia-style generation with prompt diversity

For generation, the central engineering problem is **avoiding repetition at scale**. If you prompt "write a textbook section about biology" a million times, deduplication will throw away 99% of it. The fix is a combinatorial prompt template seeded by *web snippets* (so topics track real-world distribution) crossed with audiences, styles, and formats.

```python
"""
Cosmopedia-style synthetic textbook generation. The key is PROMPT DIVERSITY:
seed each prompt with a real web snippet (for topical grounding) and sample an
audience + format. This combinatorial explosion keeps near-duplicates rare.
"""
import random, hashlib

AUDIENCES = ["a curious middle-school student", "an undergraduate",
             "a professional in an unrelated field", "a young child"]
FORMATS   = ["a textbook section with a worked example",
             "a Socratic dialogue", "a how-to tutorial with steps",
             "a blog post that builds intuition first"]
STYLES    = ["rigorous and precise", "warm and intuitive",
             "concise and example-driven"]

def make_generation_prompt(seed_snippet: str) -> str:
    aud, fmt, sty = (random.choice(AUDIENCES),
                     random.choice(FORMATS), random.choice(STYLES))
    return (
        f"Using the topic implied by the snippet below, write {fmt} aimed at "
        f"{aud}. Be {sty}. Do not mention the snippet; teach the underlying "
        f"concept thoroughly and accurately. If you state a fact, make sure it "
        f"is correct.\n\nSNIPPET:\n{seed_snippet[:1500]}\n\nPASSAGE:\n"
    )

def dedup_key(text: str, n: int = 8) -> str:
    """MinHash-lite: hash the first n normalized words as a near-dup bucket.
    In production use real MinHash/LSH (see the dedup chapter)."""
    words = " ".join(text.lower().split()[:n])
    return hashlib.md5(words.encode()).hexdigest()

def generate_textbook(seeds, llm, seen=None):
    seen = seen if seen is not None else set()
    prompts = [make_generation_prompt(s) for s in seeds]
    outs = llm.generate(prompts, sampling_params=dict(
        temperature=1.0, top_p=0.95, max_tokens=1500))  # high entropy = diversity
    for out in outs:
        text = out.outputs[0].text.strip()
        k = dedup_key(text)
        if k in seen:          # cheap near-dup guard; real dedup is downstream
            continue
        seen.add(k)
        yield {"kind": "synthetic_textbook", "text": text}
```

Note the deliberately high temperature and `top_p`: for *expository* generation we *want* entropy, because diversity is the whole game. This is the opposite of reasoning distillation (Recipe 4), where we will often want low temperature and verified correctness.

!!! warning "Common pitfall: hallucinated facts in textbook generation"
    Prompt-seeded generation writes from the model's parametric memory, so it *will* state plausible-sounding falsehoods — wrong dates, invented citations, subtly broken proofs. Unlike WRAP, there is no source document to anchor against. Mitigations: (1) seed with real snippets so claims track reality; (2) restrict to domains where the generator is strong and verifiable (math, code, well-trodden science); (3) run a verification pass (next section). Never generate "textbook" content in a domain your generator is weak in — you are just laundering its errors into the next model.

---

## Recipe 3 — Instruction- and QA-Augmented Pretraining

A third recipe blurs the line between pretraining and post-training: inject **instruction-formatted** data (questions, tasks, and their answers) directly into the *pretraining* mix. The UL2 / FLAN line of work and, more pointedly, the "instruction pretraining" results show that interleaving QA pairs derived from raw text during pretraining improves both the base model's quality and its downstream instruction-following — the model arrives at the SFT stage already fluent in the question-answer shape.

The mechanism is **reading-comprehension synthesis**: for each raw passage, generate questions whose answers are *grounded in that passage*, then train on `passage → questions+answers`. Because the answers come from the passage, grounding is strong and hallucination is low — this is closer to WRAP than to free generation.

```python
"""
Instruction-augmented pretraining: turn raw passages into grounded
reading-comprehension QA, so the base model sees instruction-shaped data
DURING pretraining. Answers are extracted from the passage -> low hallucination.
"""
QA_SYNTH_PROMPT = """\
From the passage, write {k} diverse question/answer pairs. Each ANSWER must be
fully supported by the passage -- do not use outside knowledge. Mix factual,
inferential, and "why/how" questions. Format strictly as:
Q: ...
A: ...

PASSAGE:
{passage}
"""

def synth_qa(passages, llm, k=3):
    prompts = [QA_SYNTH_PROMPT.format(k=k, passage=p[:4000]) for p in passages]
    outs = llm.generate(prompts, sampling_params=dict(
        temperature=0.7, max_tokens=512))
    for p, out in zip(passages, outs):
        qa_text = out.outputs[0].text.strip()
        # Interleave the ORIGINAL passage with its QA so the model learns the
        # link between source text and answerable questions.
        yield {"kind": "instruct_pretrain",
               "text": p.strip() + "\n\n" + qa_text}
```

This recipe is the cheapest way to give a base model "a head start" on instruction following, and it composes with everything in [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html). The key constraint is the **grounding instruction** ("ANSWER must be supported by the passage") — drop it and you slide from augmentation into free generation, with all the hallucination risk that entails.

---

## Recipe 4 — Reasoning-Trace Distillation for Post-Training

The post-training half of synthetic data is where the most dramatic recent gains live. The recipe: take a strong **teacher** that can reason (produce chain-of-thought, write and check code), have it solve a large bank of problems, **filter to keep only verified-correct traces**, and fine-tune a **student** on those traces. This is **rejection-sampling fine-tuning** (sometimes called STaR-style after Zelikman et al.'s "Self-Taught Reasoner," or RFT), and it is how reasoning capability gets compressed from a frontier model into a small one. It connects deeply to [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html), [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html), and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

The crucial difference from the pretraining recipes: here we can often **verify the output objectively**. A math answer either matches the gold answer or it does not; code either passes the unit tests or it does not. Verification turns noisy generation into a high-precision data source — we *over-generate* and *filter hard*.

### Rejection-sampling fine-tuning, end to end

```python
"""
Reasoning-trace distillation via rejection sampling (STaR/RFT-style).
For each problem with a known answer, sample K chain-of-thought traces from a
teacher, KEEP only traces whose final answer is verified correct, dedup, and
build an SFT dataset of (problem -> correct trace).
"""
import re
from collections import defaultdict

def extract_final_answer(trace: str):
    """Pull the boxed/'final answer' from a CoT trace. Robust parsing matters;
    a bad extractor silently throws away good traces or keeps bad ones."""
    m = re.search(r"\\boxed\{([^}]*)\}", trace)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", trace, re.I)
    return m.group(1).strip() if m else None

def verify(pred, gold) -> bool:
    """Domain-specific. For math, normalize then compare; for code, run tests
    in a sandbox (see the reward-verifiers chapter). Here: normalized string eq."""
    if pred is None:
        return False
    norm = lambda s: re.sub(r"\s+", "", str(s)).rstrip(".").lower()
    return norm(pred) == norm(gold)

def distill_reasoning(problems, llm, K=8, max_keep_per_problem=2):
    """
    problems: list of dicts {"question": str, "gold": str}
    Over-generate K traces/problem, verify, keep up to `max_keep` correct +
    DIVERSE traces per problem.
    """
    # Sample K completions per problem in one big batched call.
    prompts, owners = [], []
    for i, p in enumerate(problems):
        cot = (p["question"] +
               "\n\nThink step by step, then give the final answer "
               "in \\boxed{}.")
        for _ in range(K):
            prompts.append(cot); owners.append(i)

    outs = llm.generate(prompts, sampling_params=dict(
        temperature=0.8, top_p=0.95, max_tokens=2048))  # entropy -> diverse traces

    kept = defaultdict(list)
    for owner, out in zip(owners, outs):
        trace = out.outputs[0].text
        pred  = extract_final_answer(trace)
        if verify(pred, problems[owner]["gold"]):
            kept[owner].append(trace.strip())

    dataset = []
    for i, traces in kept.items():
        # Light diversity: dedup identical first lines, cap per problem so easy
        # problems don't swamp the dataset (a curriculum/balance concern).
        seen, chosen = set(), []
        for t in traces:
            key = t.split("\n", 1)[0]
            if key in seen:
                continue
            seen.add(key); chosen.append(t)
            if len(chosen) >= max_keep_per_problem:
                break
        for t in chosen:
            dataset.append({"question": problems[i]["question"],
                            "trace": t, "kind": "reasoning_sft"})
    return dataset
```

Three design points are doing the heavy lifting:

- **Over-generate, filter hard.** Sampling $K=8$ traces and keeping only correct ones is the verification asymmetry in action. Even if the teacher is only ~40% correct per attempt, with $K=8$ you cover a large fraction of solvable problems at least once: the probability of at least one success is $1-(1-0.4)^8 \approx 0.98$.
- **Cap per problem.** Easy problems get solved every time and would dominate the dataset; capping enforces balance, a curriculum concern shared with [RL Data, Curriculum & Replay Management](../06-rl-infra/12-rl-data-curriculum-replay.html).
- **Keep diverse correct traces, not just one.** Multiple valid solution paths teach the student that reasoning is a search, not a single memorized script.

### Distillation vs. RL: where synthetic SFT stops

Rejection-sampling distillation is **off-policy SFT on filtered samples**. It is cheap, stable, and gets you most of the way. But it has a ceiling: the student can only imitate traces the teacher already produces, and SFT on someone else's tokens can teach *style* without *competence* (the student parrots "Let me think step by step" without the underlying search). When you want the student to *exceed* its imitation ceiling, you move to RL with verifiable rewards, where the student generates its *own* traces and is rewarded for correctness — see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html) and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html). A common modern pipeline is: rejection-sampling SFT to bootstrap, *then* RLVR to push past the teacher.

---

## Mixing, Scaling, and the Math of Model Collapse

We now turn to the central question: **how much synthetic data, mixed how, before things go wrong?**

### The accumulate-vs-replace distinction

The most important result in the model-collapse literature (Shumailov et al., "The Curse of Recursion"; Gerstgrasser et al., "Is Model Collapse Inevitable?") is a sharp dichotomy:

- **Replace.** If each generation trains *only* on the previous generation's synthetic output, discarding the original data, errors compound and the distribution narrows generation after generation. Variance shrinks, the tails vanish, and eventually the model produces near-constant gibberish. This is true **model collapse**.
- **Accumulate.** If you *keep the original real data* and *add* synthetic data on top — train each generation on real + all-synthetic-so-far — collapse is largely *avoided*. The real data acts as an anchor that pins the distribution in place.

This is exactly why every good recipe above keeps the original: WRAP keeps the source document, instruction-augmentation interleaves the real passage, FineWeb-Edu *is* real data. **The failure mode is recursive replacement, not synthetic data per se.**

### A toy model of collapse: the shrinking Gaussian

The cleanest way to *feel* collapse is the Gaussian re-estimation chain. Suppose generation 0 is the true distribution $\mathcal{N}(\mu, \sigma^2)$. Each generation draws $n$ samples, fits a Gaussian by maximum likelihood, and the *next* generation samples from that fitted Gaussian — the "replace" regime. The fitted variance $\hat\sigma^2_{t}$ is an unbiased estimate of $\sigma^2_{t-1}$, but it has *sampling noise*, and that noise systematically erodes variance over time. The expected variance decays:

$$
\mathbb{E}[\hat\sigma^2_t] \approx \sigma^2 \left(1 - \frac{1}{n}\right)^{t}
$$

so after $t$ generations the variance has shrunk by a factor $(1-1/n)^t$. With finite samples the tails progressively disappear; the distribution collapses toward its mean. Now contrast the **accumulate** regime, where each fit uses the *original* real samples plus all synthetic so far — the real samples keep re-injecting the true variance, and the decay halts. Let us watch it happen.

```python
"""
Toy model of model collapse: recursive Gaussian re-estimation.
REPLACE: each generation fits only on the previous generation's samples.
ACCUMULATE: each generation fits on the ORIGINAL data plus all synthetic so far.
"""
import numpy as np

def collapse_chain(mode="replace", n=50, generations=30, seed=0):
    rng = np.random.default_rng(seed)
    true_mu, true_sigma = 0.0, 1.0
    real = rng.normal(true_mu, true_sigma, size=n)   # generation-0 REAL data

    mu, sigma = real.mean(), real.std()
    history = [(mu, sigma)]
    pool = list(real)  # for ACCUMULATE we grow this with synthetic samples

    for _ in range(generations):
        synth = rng.normal(mu, sigma, size=n)        # sample from current model
        if mode == "replace":
            fit = synth                              # forget the real data
        else:  # accumulate
            pool.extend(synth)                       # keep real + all synthetic
            fit = np.array(pool)
        mu, sigma = fit.mean(), fit.std()
        history.append((mu, sigma))
    return history

for mode in ("replace", "accumulate"):
    h = collapse_chain(mode=mode)
    s0, s_end = h[0][1], h[-1][1]
    print(f"{mode:>10}: sigma {s0:.3f} -> {s_end:.3f}  "
          f"({100*(s0-s_end)/s0:+.0f}% change after 30 generations)")
```

Running this, the `replace` chain shows the standard deviation steadily shrinking toward zero (variance collapse), while the `accumulate` chain holds its standard deviation near 1.0 — the real anchor prevents collapse. The same dynamic plays out in language models, with "variance" standing in for *diversity of vocabulary, syntax, and ideas*: replacement-trained models drift toward generic, repetitive, low-perplexity text, losing the rare words and unusual constructions that live in the tails.

!!! example "Worked example: variance decay and the collapse half-life"
    Take the replace regime with $n = 50$ samples per generation. The per-generation variance retention is $(1 - 1/n) = 0.98$. After $t$ generations the expected variance is $\sigma^2 \cdot 0.98^{\,t}$.

    How many generations until variance halves? Solve $0.98^{\,t} = 0.5$:

    $$
    t = \frac{\ln 0.5}{\ln 0.98} = \frac{-0.693}{-0.0202} \approx 34 \text{ generations.}
    $$

    So with 50 samples per round, you lose half your diversity in ~34 recursive generations — slow but inexorable. Now *double the sample size* to $n = 100$: retention becomes $0.99$, and the half-life stretches to $t = \ln 0.5 / \ln 0.99 \approx 69$ generations. **More samples per generation slows collapse but never stops it** — only re-injecting real data (the accumulate regime) actually halts the decay. This is the quantitative case for the 1:1-ish natural/synthetic mixes used in practice, and for never letting a fully-synthetic recursive loop run unchecked.

### Scaling behavior of mixed corpora

Two empirical regularities guide the mixing ratio (and connect to [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html)):

1. **Synthetic data improves the *effective* token efficiency** — you reach a target loss with fewer total tokens — but the gains *saturate* as the synthetic fraction grows. Past some fraction (often cited in the 30–50% range for rephrase-style data, higher for verified reasoning data), adding more synthetic stops helping and starts narrowing the distribution.
2. **The optimal synthetic fraction depends on the quality gap.** If your generator is much stronger than your target student (reasoning distillation from a frontier teacher), a high synthetic fraction is great. If your generator is the *same* class as your target (self-rephrasing), keep the fraction moderate and always anchor with real data, because you cannot distill capability the generator does not have — you can only restyle.

A simple, defensible default for *pretraining* mixes: **1–2 parts natural to 1 part synthetic**, with synthetic capped around one-third to one-half of the blend. For *post-training* reasoning SFT, the dataset can be almost entirely synthetic *because every example is verified* — verification breaks the collapse dynamic by removing the error-accumulation channel.

---

## Verification, Filtering, and Contamination Control

Synthetic data is only as good as its filters. The generation step is cheap; the **verification and decontamination** steps are where quality is won or lost. Treat the whole thing as a high-throughput data pipeline with quality gates (the philosophy of [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) and [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html)).

### The verification ladder

Order your filters cheapest-first so you discard junk before paying for expensive checks:


{{fig:syndata-verification-ladder}}


```python
"""
A cheapest-first verification ladder for a synthetic shard. Each stage drops
records so later, costlier stages process fewer items. Returns survivors + stats.
"""
def verify_shard(records, edu_head, embed_fn, judge_llm=None):
    stats = {"in": len(records)}
    out = []
    seen_ngrams = set()

    for r in records:
        t = r["text"]
        # Stage 1: format / length / language (cheap string ops)
        if not (50 <= len(t.split()) <= 4000):
            continue
        # Stage 2: cheap near-dup via 13-gram fingerprint set
        toks = t.lower().split()
        fp = hash(" ".join(toks[:13]))
        if fp in seen_ngrams:
            continue
        seen_ngrams.add(fp)
        out.append(r)
    stats["after_dedup"] = len(out)

    # Stage 3: quality classifier (one batched embedding + linear head)
    embeds = embed_fn([r["text"] for r in out])
    keep = edu_head.predict(embeds) >= 3.0
    out = [r for r, k in zip(out, keep) if k]
    stats["after_quality"] = len(out)

    # Stage 4: correctness is already enforced upstream for reasoning traces
    # (rejection sampling); for free-gen we may run an LLM judge here.
    if judge_llm is not None:
        out = [r for r in out if llm_judge_ok(judge_llm, r["text"])]
    stats["out"] = len(out)
    return out, stats

def llm_judge_ok(judge_llm, text: str) -> bool:
    """Ask a judge model for a factuality/coherence verdict. Expensive -> last."""
    prompt = ("Is the following passage coherent and free of obvious factual "
              "errors? Answer YES or NO.\n\n" + text[:3000])
    verdict = judge_llm.generate([prompt], sampling_params=dict(
        temperature=0.0, max_tokens=4))[0].outputs[0].text.strip().upper()
    return verdict.startswith("YES")
```

For *verifiable* domains (math, code), stage 4 is a real executor in a sandbox — see [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html). For everything else, stage 5 is an LLM-as-judge pass; understand its biases before you trust it ([LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html)).

### Contamination: the silent killer of synthetic data

The single most dangerous failure mode is **benchmark contamination**. When your generator was itself trained on, or has memorized, the test sets you evaluate on (MMLU, GSM8K, HumanEval, …), its synthetic output can leak those test items verbatim or near-verbatim into your training data. You then score brilliantly on the benchmark and learn nothing — a contaminated number is worse than no number, because it actively misleads. Synthetic data *amplifies* this risk because the generator can regurgitate memorized eval items even when you never asked for them.

Decontamination is mandatory and must run against your eval suites:

```python
"""
Decontamination: drop any synthetic record that overlaps an eval item by a
high-n-gram match. Use a generous n (e.g. 13-grams) and check against ALL
benchmarks you report on. This is non-negotiable for credible numbers.
"""
def ngrams(text, n=13):
    toks = text.lower().split()
    return {tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)}

def build_eval_ngram_index(eval_items, n=13):
    idx = set()
    for item in eval_items:           # questions AND answers from every benchmark
        idx |= ngrams(item, n)
    return idx

def decontaminate(records, eval_index, n=13, max_overlap=0):
    """Drop a record if it shares more than `max_overlap` 13-grams with any
    eval item -- i.e. it has likely memorized/leaked a test question."""
    clean = []
    for r in records:
        overlap = len(ngrams(r["text"], n) & eval_index)
        if overlap <= max_overlap:
            clean.append(r)
    return clean
```

!!! warning "Common pitfall: trusting synthetic benchmark scores"
    If you generate training data with model X and then report results on benchmark B, you *must* assume X may have seen B. Always (1) decontaminate the synthetic corpus against B's n-grams, (2) report on at least one *held-out, freshly authored* eval the generator could not have seen, and (3) be suspicious of any benchmark where the synthetic-trained model jumps far more than its general capability did. Contamination produces exactly that signature: a spike on the measured benchmark with no transfer.

### Distributional narrowing and fact drift

Two subtler failure modes round out the catalog:

- **Distributional narrowing.** Even without full collapse, synthetic data tends to over-represent the *center* of the distribution — common topics, standard phrasings, the generator's stylistic tics — and under-represent rare entities, dialects, code-switching, and long-tail facts. Mitigate by *seeding generation from real data* (so topic frequency tracks reality) and by measuring diversity directly: track type-token ratio, self-BLEU (lower is more diverse), and embedding-space coverage of your synthetic corpus against the natural one.
- **Fact drift / error laundering.** A wrong fact generated once, kept, and trained on becomes "knowledge" the next generation confidently repeats and elaborates. This is how a single hallucination metastasizes across a model lineage. The defense is grounding (WRAP, instruction-augmentation), verification (reasoning traces), and source seeding — and never running a recursive self-generation loop without real-data anchoring and decontamination at every step.

```python
"""
Diversity monitoring: a cheap self-BLEU proxy (lexical overlap among samples)
plus type-token ratio. Run these on every synthetic shard; a falling TTR or a
rising self-overlap is an early warning of narrowing/collapse.
"""
def type_token_ratio(texts):
    toks = [w for t in texts for w in t.lower().split()]
    return len(set(toks)) / max(1, len(toks))   # higher = more diverse

def self_overlap(texts, k=200):
    """Average fraction of shared unigrams between random pairs. Higher = more
    repetitive (a collapse warning). Sampled for speed."""
    import random
    sets = [set(t.lower().split()) for t in texts[:k]]
    pairs, tot = 0, 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j]) or 1
            tot += inter / union; pairs += 1
    return tot / max(1, pairs)                  # ~ Jaccard self-similarity
```

Wire these into your pipeline as **monitored metrics**: if type-token ratio falls or self-overlap rises shard-over-shard, your generator is narrowing and you should raise temperature, broaden seeds, or cut the synthetic fraction.

---

## Interview Corner

!!! interview "Interview Corner"
    **Q:** You are out of high-quality pretraining tokens and decide to add synthetic data by rephrasing your existing corpus with your own model. A colleague warns this will cause "model collapse." Are they right, and how would you design the mix to be safe?

    **A:** Partially right — collapse is a real risk but it is *avoidable*, and the key variable is whether you **replace** real data or **accumulate** on top of it. The Shumailov-style collapse results are about *recursive replacement*: training generation $t+1$ only on generation $t$'s outputs, discarding the original. That compounds sampling noise and narrows the distribution — the toy Gaussian shows variance decaying like $(1-1/n)^t$. But the Gerstgrasser "accumulate" results show that if you *keep all the real data* and *add* synthetic to it, collapse is largely prevented because the real data re-anchors the distribution each round.

    So the design: (1) **never replace** — always keep the full natural corpus and treat synthetic as augmentation, targeting roughly a 1:1 to 2:1 natural:synthetic ratio. (2) **Use rephrasing, not free generation**, so every synthetic token is grounded in a real source and facts are inherited, not invented (WRAP-style). (3) **Diversify the rewrite styles** and use a moderate temperature so you do not collapse onto one bland mode. (4) **Decontaminate** the synthetic output against every benchmark you will report — a self-rephraser can leak memorized eval items. (5) **Monitor diversity** (type-token ratio, self-overlap) shard-over-shard as an early-warning signal. The colleague's fear applies to a naive recursive loop; a grounded, accumulating, decontaminated, diversity-monitored mix is safe and is exactly what shops doing this in practice run.

---

## Putting It Together: A Synthetic-Data Playbook

Stitching the recipes into one decision procedure:

| Goal | Recipe | Grounding | Filter | Mix |
|---|---|---|---|---|
| Stretch a scarce corpus | WRAP rephrasing | Source doc | Quality + dedup | ~1:1 natural:synth |
| Inject clean pedagogy | Cosmopedia gen + FineWeb-Edu filter | Seed snippet (gen); real (filter) | Quality, dedup, judge | Cap synth ≤ ~1/3 |
| Head-start instruction following | QA-augmented pretrain | Passage | Grounding check | Small fraction |
| Distill reasoning into a student | Rejection-sampling SFT | Verified answer | Exec/verify (hard) | Mostly synth (verified) |
| Exceed the teacher | RLVR (not SFT) | Verifiable reward | Reward = verifier | On-policy |

The throughline across all five: **synthetic data converts compute into targeted, high-quality tokens, and its safety is governed by grounding, verification, and the refusal to ever recursively replace real data.** Generation is the easy 10%; the verification ladder, decontamination, and diversity monitoring are the 90% that decides whether your synthetic corpus is a moat or a landmine.

!!! key "Key Takeaways"
    - The **data wall** is real: high-quality human text is finite, and synthetic data trades cheap compute for scarce high-quality tokens — but only if generated and filtered carefully.
    - Four recipes, by increasing risk: **WRAP rephrasing** (source-grounded, ~3× token efficiency), **textbook generation** (Phi/Cosmopedia, highest quality density, highest hallucination risk), **instruction/QA augmentation** (grounded, gives base models a head start), and **reasoning-trace distillation** (verified, the post-training workhorse).
    - **Model collapse is caused by recursive *replacement*, not by synthetic data itself.** Keeping the real data and *accumulating* synthetic on top largely prevents it; the toy Gaussian shows variance decaying like $(1-1/n)^t$ only in the replace regime.
    - **Verification breaks the collapse dynamic.** For math/code you can keep only objectively-correct traces, which is why post-training datasets can be almost fully synthetic while pretraining mixes cannot.
    - **Over-generate, filter hard.** Sample $K$ candidates, keep the verified-correct ones; with the verification asymmetry, an unreliable generator still yields a high-precision dataset.
    - **Decontamination against every reported benchmark is mandatory** — synthetic generators can leak memorized eval items, producing spurious score spikes with no real transfer.
    - **Monitor diversity** (type-token ratio, self-overlap, embedding coverage). Falling diversity is the early-warning sign of narrowing before full collapse.
    - Default pretraining mix: **1–2 parts natural to 1 part synthetic**, synthetic capped around one-third to one-half; always anchor with real data, always seed generation from real data so topic frequencies track reality.

!!! sota "State of the Art & Resources (2026)"
    Synthetic data has become a first-class ingredient in frontier LLM training: WRAP-style rephrasing, textbook-quality generation (Phi/Cosmopedia), and verified reasoning-trace distillation (RFT/STaR) are now standard practice, while the model-collapse literature has largely settled on "accumulate, don't replace" as the governing principle for safe mixing.

    **Foundational work**

    - [Gunasekar et al., *Textbooks Are All You Need* (2023)](https://arxiv.org/abs/2306.11644) — introduced Phi-1 and the thesis that data quality can substitute for scale; spawned the textbook-generation paradigm.
    - [Li et al., *Textbooks Are All You Need II: phi-1.5* (2023)](https://arxiv.org/abs/2309.05463) — extended the recipe to general reasoning, showing 1.3B models matching 5× larger ones on NLP tasks.
    - [Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning* (2022)](https://arxiv.org/abs/2203.14465) — the canonical rejection-sampling fine-tuning loop: over-generate, verify, keep correct traces.

    **Recent advances (2023–2026)**

    - [Maini et al., *Rephrasing the Web (WRAP)* (2024)](https://arxiv.org/abs/2401.16380) — source-grounded web rephrasing achieves ~3× pretraining token efficiency; the lowest-risk synthetic recipe.
    - [Shumailov et al., *The Curse of Recursion* (2023)](https://arxiv.org/abs/2305.17493) — quantifies model collapse under recursive replacement; tails of the distribution vanish irreversibly.
    - [Gerstgrasser et al., *Is Model Collapse Inevitable?* (2024)](https://arxiv.org/abs/2404.01413) — shows accumulating real + synthetic data breaks the collapse dynamic; test risk bounded by π²/6 of the baseline.
    - [Muennighoff et al., *Scaling Data-Constrained Language Models* (2023)](https://arxiv.org/abs/2305.16264) — NeurIPS 2023 outstanding paper; quantifies diminishing returns of repeated epochs vs. fresh tokens.
    - [Cheng et al., *Instruction Pre-Training* (2024)](https://arxiv.org/abs/2406.14491) — injecting 200M synthesized instruction-response pairs during pretraining; enables Llama3-8B to match 70B on downstream tasks.

    **Open-source & tools**

    - [huggingface/cosmopedia](https://github.com/huggingface/cosmopedia) — fully open pipeline (prompt generation, llm-swarm inference, deduplication, decontamination) that produced 25B tokens of synthetic textbooks.
    - [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — 1.3T-token educational web corpus (arxiv: 2406.17557); the classifier distillation approach is a reusable quality-filtering recipe.

## Further reading

- Maini, Seto, et al., *Rephrasing the Web (WRAP): A Recipe for Compute and Data-Efficient Language Modeling*.
- Gunasekar et al., *Textbooks Are All You Need* (Phi-1); Li et al., *Textbooks Are All You Need II* (phi-1.5); the broader Microsoft **Phi** technical reports.
- Ben Allal, Lozhkov, et al., **Cosmopedia** (HuggingFace) — open reproduction of Phi-style synthetic textbook data; and the **FineWeb-Edu** dataset and educational-classifier report.
- Shumailov et al., *The Curse of Recursion: Training on Generated Data Makes Models Forget* (model collapse).
- Gerstgrasser et al., *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*.
- Zelikman et al., *STaR: Bootstrapping Reasoning with Reasoning*; and rejection-sampling fine-tuning as used in the **Llama** and **DeepSeek-R1** post-training reports.
- Muennighoff et al., *Scaling Data-Constrained Language Models* (how many epochs of real data are worth fresh tokens).
- Cheng et al., *Instruction Pre-Training: Language Models Are Supervised Multitask Learners*.
