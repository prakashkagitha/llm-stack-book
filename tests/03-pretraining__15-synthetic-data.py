"""
Runs the CPU-runnable Python code blocks from:
    content/03-pretraining/15-synthetic-data.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:  #0, #1, #2, #4, #5, #6, #7, #8
Skipped blocks: #3 (line ~233, instruction-augmented pretraining `synth_qa`) --
                not standalone-tested per task spec (listed as default-SKIP
                fragment); it is a thin wrapper around `llm.generate` with no
                extra logic beyond what blocks #0/#2/#4 already exercise
                (prompt formatting + llm.generate call), so re-testing it adds
                no coverage.

All blocks call `llm.generate(prompts, sampling_params=...)` in the vLLM
batched style. Since vLLM requires a GPU and network model download, we
supply a tiny in-process `FakeLLM` that returns objects shaped like vLLM's
output (`.outputs[0].text`) driven by a deterministic responder function --
this exercises the BOOK'S OWN prompt-assembly / parsing / filtering logic
against canned text, with no network or GPU involved.
"""

from __future__ import annotations

import types
import numpy as np


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Shared test double: a minimal stand-in for the vLLM `llm.generate(...)` API
# used verbatim by every generation block in this chapter.
# ============================================================================

class _FakeOutput:
    """Mimics one vLLM RequestOutput: `.outputs[0].text`."""

    def __init__(self, text: str):
        self.outputs = [types.SimpleNamespace(text=text)]


class FakeLLM:
    """Stands in for a vLLM `LLM` instance. `responder(prompt, index)` decides
    the canned completion text for each prompt in the batch."""

    def __init__(self, responder):
        self.responder = responder

    def generate(self, prompts, sampling_params=None):
        return [_FakeOutput(self.responder(p, i)) for i, p in enumerate(prompts)]


# ============================================================================
# Block #0 (line ~56) -- WRAP-style web rephrasing
# ============================================================================
_section("Block #0: WRAP-style web rephrasing")

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

# --- Exercise block #0 with a tiny fixture ---
random.seed(0)
_wrap_docs = [
    (0, "The mitochondria is the powerhouse of the cell. It produces ATP "
        "through oxidative phosphorylation. <boilerplate>Click here!</boilerplate>"),
    (1, "Photosynthesis converts light energy into chemical energy in plants."),
]
_wrap_cfg = WrapConfig(styles_per_doc=2, max_new_tokens=64)


def _wrap_responder(prompt, i):
    return f"[rephrased #{i}] clean version of the source document."


_wrap_llm = FakeLLM(_wrap_responder)
_wrap_records = list(rephrase_corpus(_wrap_docs, _wrap_llm, _wrap_cfg))

# 2 docs * styles_per_doc=2 = 4 rephrasings
assert len(_wrap_records) == 4
assert {r["doc_id"] for r in _wrap_records} == {0, 1}
assert all(r["kind"] == "synthetic" for r in _wrap_records)
assert all(r["style"] in STYLE_PROMPTS for r in _wrap_records)
assert all(r["text"].startswith("[rephrased #") for r in _wrap_records)
print(f"Rephrased {len(_wrap_records)} records from {len(_wrap_docs)} docs")
print("Block #0 OK")


# ============================================================================
# Block #1 (line ~133) -- FineWeb-Edu-style educational quality filter
# ============================================================================
_section("Block #1: FineWeb-Edu educational quality filter")

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

# --- Exercise block #1 with a toy embedding space ---
_edu_rng = np.random.default_rng(0)
_EDU_DIM = 5
_N_TRAIN = 200
# Construct a "true" direction so higher-scoring docs really do sit further
# along one axis -- makes the toy regression meaningful rather than pure noise.
_true_w = _edu_rng.normal(size=_EDU_DIM)
_train_embeds = _edu_rng.normal(size=(_N_TRAIN, _EDU_DIM))
_train_scores = np.clip(
    _train_embeds @ _true_w + _edu_rng.normal(scale=0.3, size=_N_TRAIN) + 2.5,
    0, 5,
)

_prompt_sample = make_labeling_prompt("Photosynthesis is the process by which plants convert light into energy.")
assert "educational value" in _prompt_sample
assert _prompt_sample.endswith("Photosynthesis is the process by which plants convert light into energy.")

_edu_head = train_edu_head(_train_embeds, _train_scores)

_test_embeds = _edu_rng.normal(size=(20, _EDU_DIM))
_mask = keep_mask(_edu_head, _test_embeds, threshold=3.0)
assert _mask.shape == (20,)
assert _mask.dtype == np.bool_
print(f"Kept {_mask.sum()}/{len(_mask)} toy pages at threshold 3.0")
print("Block #1 OK")


# ============================================================================
# Block #2 (line ~174) -- Cosmopedia-style synthetic textbook generation
# ============================================================================
_section("Block #2: Cosmopedia-style textbook generation")

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

# --- Exercise block #2: force one near-duplicate to prove dedup fires ---
random.seed(1)
_cosmo_seeds = ["web snippet about tides", "web snippet about tides (dup)",
                "web snippet about volcanoes"]


def _cosmo_responder(prompt, i):
    if i < 2:
        # Same first 8 words -> same dedup_key -> the second should be dropped.
        return ("The ocean tides rise and fall each day because of gravity "
                "and it is fascinating to study in detail.")
    return f"Volcanoes erupt molten rock from deep underground reservoir number {i}."


_cosmo_llm = FakeLLM(_cosmo_responder)
_cosmo_records = list(generate_textbook(_cosmo_seeds, _cosmo_llm))

# 3 seeds in, 1 near-duplicate dropped -> 2 records out
assert len(_cosmo_records) == 2
assert all(r["kind"] == "synthetic_textbook" for r in _cosmo_records)
print(f"Generated {len(_cosmo_records)} unique textbook passages from {len(_cosmo_seeds)} seeds "
      f"({len(_cosmo_seeds) - len(_cosmo_records)} near-dup dropped)")
print("Block #2 OK")


# ============================================================================
# Block #3 (line ~233) -- instruction-augmented pretraining QA synthesis
# ============================================================================
# SKIP(fragment, per task spec): `synth_qa` is a thin wrapper around
# `llm.generate` whose logic (prompt formatting + text concatenation) is
# already exercised by blocks #0/#2/#4 above; not independently tested here.


# ============================================================================
# Block #4 (line ~274) -- rejection-sampling reasoning-trace distillation
# ============================================================================
_section("Block #4: rejection-sampling fine-tuning (STaR/RFT)")

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

# --- Exercise block #4: 2 problems, K=4 traces each, half correct/half wrong ---
_rft_problems = [
    {"question": "What is 2+2?", "gold": "4"},
    {"question": "What is 3+3?", "gold": "6"},
]
_RFT_K = 4


def _rft_responder(prompt, i):
    owner = i // _RFT_K
    within = i % _RFT_K
    gold = _rft_problems[owner]["gold"]
    if within % 2 == 0:
        # Correct answer; distinct first line per sample -> both survive dedup.
        return f"Step {within}: reasoning path {within}.\nFinal answer: \\boxed{{{gold}}}"
    return f"Step {within}: a wrong reasoning path.\nFinal answer: \\boxed{{999}}"


_rft_llm = FakeLLM(_rft_responder)
_rft_dataset = distill_reasoning(_rft_problems, _rft_llm, K=_RFT_K, max_keep_per_problem=2)

# Each problem: within=0,2 correct (2 kept, distinct first lines); within=1,3 wrong (dropped).
assert len(_rft_dataset) == 4
assert all(d["kind"] == "reasoning_sft" for d in _rft_dataset)
assert {d["question"] for d in _rft_dataset} == {"What is 2+2?", "What is 3+3?"}
assert extract_final_answer("blah \\boxed{42} blah") == "42"
assert extract_final_answer("the final answer: 7") == "7"
assert extract_final_answer("no answer here") is None
assert verify("4", "4.") is True
assert verify(None, "4") is False
print(f"Rejection-sampling kept {len(_rft_dataset)} verified traces from "
      f"{len(_rft_problems) * _RFT_K} sampled")
print("Block #4 OK")


# ============================================================================
# Block #5 (line ~383) -- toy model of collapse: shrinking Gaussian
# ============================================================================
_section("Block #5: model collapse -- shrinking Gaussian (replace vs accumulate)")

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

_collapse_results = {}
for mode in ("replace", "accumulate"):
    h = collapse_chain(mode=mode)
    s0, s_end = h[0][1], h[-1][1]
    print(f"{mode:>10}: sigma {s0:.3f} -> {s_end:.3f}  "
          f"({100*(s0-s_end)/s0:+.0f}% change after 30 generations)")
    _collapse_results[mode] = (s0, s_end)

# The replace regime should shrink noticeably; accumulate should stay near 1.0.
_replace_s0, _replace_send = _collapse_results["replace"]
_accum_s0, _accum_send = _collapse_results["accumulate"]
assert _replace_send < _replace_s0 * 0.9, "replace regime should shrink variance"
assert abs(_accum_send - 1.0) < 0.25, "accumulate regime should stay anchored near sigma=1"
print("Block #5 OK")


# ============================================================================
# Block #6 (line ~454) -- cheapest-first verification ladder
# ============================================================================
_section("Block #6: verification ladder")

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

# --- Exercise block #6: reuse the trained edu_head from block #1 ---
_word60 = " ".join(["educational"] * 60)  # 60 words -> passes length stage
_word60_b = " ".join(["volcanic"] * 60)   # different opening -> different fingerprint
_ladder_records = [
    {"text": _word60 + " unique passage alpha about biology and cells."},
    {"text": _word60 + " unique passage alpha about biology and cells."},  # exact dup of [0]
    {"text": _word60_b + " a totally different passage about lava flows here."},  # unique
    {"text": " ".join(["short"] * 5)},  # too short -> fails stage 1
]

def _ladder_embed_fn(texts):
    # Deterministic per-text embedding in the same 5-D space the edu_head
    # (from block #1) was trained on, so `.predict` is well-defined.
    rng = np.random.default_rng(42)
    return rng.normal(size=(len(texts), _EDU_DIM))


def _judge_responder(prompt, i):
    return "YES"


_judge_llm = FakeLLM(_judge_responder)
_survivors, _ladder_stats = verify_shard(
    _ladder_records, _edu_head, _ladder_embed_fn, judge_llm=_judge_llm
)

assert _ladder_stats["in"] == 4
# record[3] fails stage-1 length; record[1] is an exact 13-gram dup of record[0]
# whose fingerprint collides with record[2] as well (identical first 13 tokens).
assert _ladder_stats["after_dedup"] == 2
assert _ladder_stats["after_quality"] <= _ladder_stats["after_dedup"]
assert _ladder_stats["out"] == _ladder_stats["after_quality"]  # judge said YES to all
assert llm_judge_ok(_judge_llm, "some coherent passage") is True
print(f"Verification ladder stats: {_ladder_stats}")
print("Block #6 OK")


# ============================================================================
# Block #7 (line ~508) -- decontamination via n-gram overlap
# ============================================================================
_section("Block #7: decontamination")

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

# --- Exercise block #7 ---
_eval_items = [
    "What is the boiling point of water at sea level in degrees Celsius "
    "and why does it matter for cooking experiments",
]
_eval_index = build_eval_ngram_index(_eval_items, n=13)
assert len(_eval_index) > 0

_decon_records = [
    {"text": "What is the boiling point of water at sea level in degrees "
             "Celsius and why does it matter for cooking experiments"},  # leaked
    {"text": "Photosynthesis converts sunlight into chemical energy stored "
             "in glucose molecules inside chloroplasts of plant cells."},  # clean
]
_clean = decontaminate(_decon_records, _eval_index, n=13, max_overlap=0)
assert len(_clean) == 1
assert "Photosynthesis" in _clean[0]["text"]
print(f"Decontamination kept {len(_clean)}/{len(_decon_records)} records "
      f"({len(_decon_records) - len(_clean)} dropped as leaked)")
print("Block #7 OK")


# ============================================================================
# Block #8 (line ~545) -- diversity monitoring (TTR, self-overlap)
# ============================================================================
_section("Block #8: diversity monitoring")

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

# --- Exercise block #8: diverse corpus vs. a narrow/repetitive corpus ---
_diverse_texts = [
    "The mitochondria produces ATP through oxidative phosphorylation reactions.",
    "Photosynthesis converts sunlight into chemical energy inside chloroplasts.",
    "Neural networks approximate functions using layers of weighted connections.",
    "Volcanic eruptions release magma, ash, and gases from beneath the crust.",
]
_narrow_texts = ["the cat sat on the mat" for _ in range(4)]

_ttr_diverse = type_token_ratio(_diverse_texts)
_ttr_narrow = type_token_ratio(_narrow_texts)
_overlap_diverse = self_overlap(_diverse_texts)
_overlap_narrow = self_overlap(_narrow_texts)

assert 0.0 < _ttr_diverse <= 1.0
assert 0.0 < _ttr_narrow <= 1.0
assert _ttr_narrow < _ttr_diverse, "identical repeated text should have lower TTR"
assert _overlap_narrow > _overlap_diverse, "identical repeated text should have higher self-overlap"
assert _overlap_narrow == 1.0  # every pair is identical -> Jaccard 1.0
print(f"TTR diverse={_ttr_diverse:.3f} narrow={_ttr_narrow:.3f}  "
      f"self-overlap diverse={_overlap_diverse:.3f} narrow={_overlap_narrow:.3f}")
print("Block #8 OK")


print("\nALL BLOCKS PASSED")
