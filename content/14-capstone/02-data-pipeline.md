# 14.2 Data: Sourcing, Filtering, Dedup, Tokenize & Pack ~20B Tokens

Every chapter so far in this book has treated data as an input — something that arrives already tokenized, already packed, already a clean tensor of `input_ids`. This chapter is where we build that tensor. We are going to source, filter, deduplicate, tokenize, and pack roughly 20 billion tokens for `Stack-100M`, the ~101.4M-parameter model this capstone builds end to end (architecture fixed in `capstone/PLAN.md` §1; full parameter count derived in the next chapter). Everything here lives under the `stacklm` package, in `capstone/stacklm/data/`.

The number "20 billion" is not arbitrary, and it is larger than you might expect for a 100M-parameter model. The first order of business is explaining why — because the answer is the single most important lesson this capstone teaches about how small models are actually trained in 2025–2026, and it shapes every design decision that follows: which corpora we mix, how aggressively we deduplicate, and how densely we pack tokens onto disk.

This chapter builds directly on [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) and [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html), which cover the general theory in far more depth than we repeat here; treat this chapter as their applied, at-scale, single-GPU-budget instantiation.

## Why Over-Train? 20B Tokens for a 100M-Parameter Model

The Chinchilla scaling law (Hoffmann et al., 2022 — see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for the full derivation) says that for a fixed training-compute budget $C$, loss is minimized when parameters $N$ and tokens $D$ are scaled together, with the empirical optimum landing near

$$
D^\*(N) \approx 20\,N.
$$

For our $N \approx 101.4\text{M}$ parameters, that gives $D^\*\approx 2.0\text{B}$ tokens — the *training-compute-optimal* budget. If we only cared about minimizing loss per FLOP spent during training, we would stop at 2B tokens.

But we do not only care about training FLOPs. We care about the model we end up with, because we are going to *serve* it — quantize it, run it on a laptop, wire it into an agent loop (Ch. 14.9–14.10). Chinchilla's compute-optimal frontier answers "what's the cheapest way to reach a given loss during training?" It says nothing about inference cost, and inference cost is what actually gates a small model. Every additional token you train on is compute you pay for exactly once, at training time. Every parameter you add is compute you pay for on *every single forward pass*, forever. Once serving volume is large enough, over-training a smaller model past its Chinchilla-optimal token budget beats training a larger model to Chinchilla-optimal, because the smaller model's inference savings compound while the extra pretraining tokens are a one-time cost.

This is exactly the logic that produced the current generation of small, extremely capable open models — SmolLM2/SmolLM3 (HuggingFace), Qwen3's small variants, and Llama 3's 8B model (trained on roughly 15T tokens against a Chinchilla-optimal budget more than an order of magnitude smaller) all deliberately over-train. `Stack-100M` adopts the same philosophy at a scale a single GPU can afford:

$$
D_{\text{Stack-100M}} = 20\times 10^9 \text{ tokens} \approx 197 \text{ tokens/param} \approx 10\times \text{Chinchilla's } D^\*.
$$

We will round this to "≈200 tokens/param" throughout, matching `capstone/PLAN.md`. The FLOP cost of this choice is easy to see with the standard $C\approx 6ND$ approximation (see the roofline chapter, [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html), for where the factor of 6 comes from):

$$
C = 6\,N\,D \approx 6 \times (1.014\times10^{8}) \times (2\times10^{10}) \approx 1.22\times10^{19}\ \text{FLOPs}.
$$

That is roughly $10\times$ the compute of the Chinchilla-optimal 2B-token run for the same $N$ — but it is still cheap in absolute terms, because $N$ is small. On a single A100 at bf16, this is on the order of the 15–25 GPU-hours (≈USD 40–100) quoted as the flagship budget in `capstone/PLAN.md`; we verify that consistency numerically in the worked example below. The over-training bet only pays off because $N$ is small enough that $10\times$ the tokens is still an affordable afternoon of GPU time — at 70B parameters the same ratio would cost a fortune. This is the deployment-economics argument in one sentence: **spend the extra compute where it's cheap (training, once) to save it where it's expensive (inference, forever).**

!!! interview "Interview Corner"

    **Q:** Chinchilla says $D^\*\approx 20N$ minimizes loss for a fixed *training* compute budget. Why would anyone deliberately train past that point — isn't that compute-inefficient?

    **A:** It's training-compute-inefficient but can be deployment-compute-efficient, and those are different objectives. Chinchilla optimizes $\min_{N,D} L(N,D)$ subject to $C=6ND$ fixed — it never sees an inference cost term. If the model will be served many times, the relevant objective is closer to $\min_N \big[L(N, D) + \lambda \cdot (\text{inference cost as a function of } N)\big]$ for whatever token budget $D$ you're willing to spend once. Since inference cost scales with $N$ (roughly linearly in FLOPs/token, and directly in memory footprint), shrinking $N$ and over-training on more $D$ trades a one-time training cost for a permanent inference-cost reduction. This is exactly the reasoning behind Llama 3 8B's ~15T-token run and small models like SmolLM2 — and it's why "Chinchilla-optimal" and "the model you should actually ship" are frequently different points on the loss curve.

{{fig:overtrain-deployment-economics}}

## The Stack-100M Data Mix

We follow the recipe popularized by HuggingFace's SmolLM series: a large base of filtered educational web text, topped up with synthetic textbooks and a slice of code and math, rather than raw, undifferentiated Common Crawl. The mix, fixed in `capstone/PLAN.md` §2:

| Source | HF dataset (production) | Weight | Tokens (of 20B) | Domain |
|---|---|---:|---:|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 70% | 14.0B | Filtered educational web |
| Cosmopedia v2 | `HuggingFaceTB/cosmopedia` | 15% | 3.0B | Synthetic textbooks/stories |
| StarCoder (subset) | `bigcode/starcoderdata` | 10% | 2.0B | Permissively-licensed code |
| FineMath / OpenWebMath | `HuggingFaceTB/finemath` | 5% | 1.0B | Math text and problem sets |

**FineWeb-Edu** (Penedo et al., HuggingFace, 2024) is Common Crawl filtered down to the subset a lightweight classifier judges educational, where the classifier was itself trained on quality labels distilled from a large teacher LLM's judgments. It is the bulk of the mix because raw web text at trillion-token scale, once aggressively quality-filtered, remains the best source of broad linguistic and world knowledge per token.

**Cosmopedia v2** (HuggingFaceTB) is entirely synthetic: textbooks, blog posts, and stories generated by a large model from curated seed topics and reading levels. Synthetic textbooks are denser in unambiguous, well-structured knowledge than the median web page — the tradeoff is diversity, which is why it is 15% of the mix rather than the majority (see [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html) for the general case for and against synthetic pretraining data).

**StarCoder** (BigCode) and **FineMath/OpenWebMath** are the "capability injection" slices: code and math text noticeably improve a small model's structured-reasoning ability even when the downstream tasks are not code or math, and they are prerequisites for the narrow tool-using agent we build in Ch. 14.9–14.10. At only 10% and 5% of the mix, they are seasoning, not the entrée.

These weights are a starting point, not a law — Chapter 14.5's mini scaling-law ladder and Chapter 14.8's mid-training annealing both revisit and re-weight this mix; see [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html) for the general theory of how and why to reweight a training mix over the course of a run.

## Sourcing: Streaming Downloads and an Offline Synthetic Fallback

At 20B tokens, none of these corpora fit on a laptop disk in raw form, and CI must run with no network access at all. Both constraints point to the same design: **stream** documents rather than downloading full datasets, and give every source a **deterministic, in-process synthetic fallback** with the same schema, so the entire pipeline — filtering, dedup, packing, sharding — is exercised end to end without ever touching the network.

### Streaming from HuggingFace Datasets

In production, each source is a `datasets.load_dataset(..., streaming=True)` iterator, which pulls shards over HTTP on demand instead of materializing the whole corpus:

```python
"""
capstone/stacklm/data/sources.py

Streaming source abstraction for the Stack-100M data mix (capstone/PLAN.md sec.2).
Every source has a synthetic, network-free fallback so this module is hermetic
in CI and runnable on an offline laptop.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class DataMixEntry:
    name: str     # short id, e.g. "fineweb_edu"
    hf_path: str  # HuggingFace dataset id used in production
    weight: float  # fraction of the 20B-token budget
    domain: str    # "web" | "synthetic" | "code" | "math" -- routes the filter (below)


# The exact mix fixed in capstone/PLAN.md sec.2. Weights sum to 1.0.
STACK100M_MIX = [
    DataMixEntry("fineweb_edu",   "HuggingFaceFW/fineweb-edu",     0.70, "web"),
    DataMixEntry("cosmopedia_v2", "HuggingFaceTB/cosmopedia",      0.15, "synthetic"),
    DataMixEntry("starcoder",     "bigcode/starcoderdata",         0.10, "code"),
    DataMixEntry("finemath",      "HuggingFaceTB/finemath",        0.05, "math"),
]

TOTAL_TOKEN_BUDGET = 20_000_000_000  # ~20B tokens, ~200 tok/param (PLAN.md sec.2)


def stream_source(entry: DataMixEntry, offline: bool = False) -> Iterator[dict]:
    """
    Yield {"text", "source", "domain"} documents from one mix entry.

    offline=True (or a missing `datasets` install, or no network) always falls
    back to the synthetic generator below, so this function never blocks a
    hermetic CI run or a laptop with the wifi off.
    """
    if not offline:
        try:
            from datasets import load_dataset  # heavy optional dependency
            ds = load_dataset(entry.hf_path, split="train", streaming=True)
            for row in ds:
                text = row.get("text", "")
                if text:
                    yield {"text": text, "source": entry.name, "domain": entry.domain}
            return
        except Exception:
            # Network unavailable, dataset gated behind auth, or `datasets`
            # not installed. Fall through to the synthetic generator.
            pass
    yield from synthetic_corpus(entry)
```

### The Offline Synthetic Fallback

The synthetic generator is deliberately small and deterministic (seeded per source), and it intentionally injects a handful of exact and near-duplicate documents — so that Section 4's deduplication code has something real to catch even when there is no network:

```python
_dup_cache: dict[str, str] = {}


def synthetic_corpus(entry: DataMixEntry, n_docs: int = 2000) -> Iterator[dict]:
    """
    A tiny, deterministic, in-process corpus generator with no network dependency.
    It teaches the model nothing -- it exists so every downstream stage (filter,
    dedup, pack, shard) is exercised by CI as a hermetic stand-in for the real
    FineWeb-Edu / Cosmopedia / StarCoder / FineMath streams.
    """
    seed = int(hashlib.blake2b(entry.name.encode(), digest_size=4).hexdigest(), 16)
    rng = random.Random(seed)
    vocab = {
        "web": ["photosynthesis", "converts", "sunlight", "into", "chemical", "energy",
                "plants", "use", "carbon", "dioxide", "and", "water", "to", "produce",
                "glucose", "the", "process", "occurs", "inside", "chloroplasts"],
        "synthetic": ["chapter", "one", "introduces", "the", "concept", "of", "gravity",
                      "as", "a", "force", "that", "attracts", "objects", "with", "mass",
                      "toward", "each", "other", "consider", "an", "example"],
        "code": ["def", "compute", "(", "x", ")", ":", "return", "x", "*", "x", "+", "1",
                 "for", "i", "in", "range", "(", "10", ")", ":", "print", "(", "i", ")"],
        "math": ["let", "f", "(", "x", ")", "=", "x^2", "then", "the", "derivative",
                 "is", "2x", "solve", "for", "x", "when", "3x", "+", "5", "=", "20"],
    }[entry.domain]

    for i in range(n_docs):
        # Every 97th doc is an *exact* repeat of an earlier one; every 53rd doc
        # is a *near*-duplicate (a ~5% word-level edit of an earlier one).
        if entry.domain in _dup_cache and i % 97 == 0:
            text = _dup_cache[entry.domain]
        elif entry.domain in _dup_cache and i % 53 == 0:
            base = _dup_cache[entry.domain].split()
            for _ in range(max(1, len(base) // 20)):
                base[rng.randrange(len(base))] = rng.choice(vocab)
            text = " ".join(base)
        else:
            n_words = rng.randint(60, 400)
            text = " ".join(rng.choice(vocab) for _ in range(n_words)) + "."
            _dup_cache[entry.domain] = text
        doc_id = hashlib.sha1(f"{entry.name}-{i}".encode()).hexdigest()[:12]
        yield {"text": text, "source": entry.name, "domain": entry.domain, "doc_id": doc_id}
```

{{fig:data-pipeline-funnel}}

## Quality Filtering and Deduplication

### Quality Filtering at Ingest

[Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) covers the full battery of heuristic filters (length bounds, character-class ratios, repeated-n-gram detection, language ID) used to clean web-scale corpora. We re-implement a lean, domain-routed subset here: web/synthetic prose gets the generic filter, but code and math have such different character distributions that the generic filter would wrongly reject nearly all of them (a `def` block has almost no English stop words and a high symbol density; a proof has a much higher digit fraction than prose).

```python
"""
capstone/stacklm/data/filters.py

Fast, dependency-free quality gates, domain-routed. This is a lean subset of
Data Cleaning, Deduplication & Quality Filtering (Ch. 3.2), tuned so that
filtering compute doesn't compete with training compute at a 20B-token budget.
"""
from __future__ import annotations
import re

_WORD_RE = re.compile(r"\S+")


def basic_stats(text: str) -> dict:
    words = _WORD_RE.findall(text)
    n_words = len(words) or 1
    n_chars = len(text) or 1
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    return dict(
        n_words=n_words,
        alpha_frac=alpha / n_chars,
        digit_frac=digit / n_chars,
        mean_word_len=sum(len(w) for w in words) / n_words,
    )


def passes_web_filter(text: str) -> bool:
    """Generic prose gate for FineWeb-Edu / Cosmopedia documents."""
    s = basic_stats(text)
    return (
        50 <= s["n_words"] <= 100_000
        and 3.0 <= s["mean_word_len"] <= 10.0
        and s["alpha_frac"] >= 0.60
        and s["digit_frac"] <= 0.20
    )


def passes_code_filter(text: str) -> bool:
    """Loose gate for StarCoder: reject empty/binary-looking or generated-
    looking files (dominated by one repeated character); keep everything
    else, since code has a very different character distribution than
    prose and would be wrongly rejected by `passes_web_filter`."""
    if not (20 <= len(text) <= 200_000):
        return False
    head = text[:2000]
    most_common_frac = max(head.count(c) for c in set(head)) / max(len(head), 1)
    return most_common_frac <= 0.30


def passes_math_filter(text: str) -> bool:
    """FineMath/OpenWebMath gate: require mathematical density (digits,
    operators, or LaTeX-ish markers), not generic prose fluency."""
    s = basic_stats(text)
    markers = sum(text.count(m) for m in ("=", "\\frac", "$", "^", "\\sum"))
    return s["n_words"] >= 20 and (s["digit_frac"] >= 0.03 or markers >= 3)


_FILTERS = {
    "web": passes_web_filter,
    "synthetic": passes_web_filter,
    "code": passes_code_filter,
    "math": passes_math_filter,
}


def quality_filter(doc: dict) -> bool:
    """Route a document to the filter for its source domain."""
    fn = _FILTERS.get(doc.get("domain", "web"), passes_web_filter)
    return fn(doc["text"])
```

### Deduplication: Exact Hashing and MinHash Near-Duplicates

Even after quality filtering, web-scale corpora contain enormous amounts of exact and near-duplicate content — mirrored pages, boilerplate legal text, syndicated news, forum threads quoting each other. Duplicate content wastes token budget and, worse, causes the model to overfit to memorized strings rather than generalizing (Lee et al., *Deduplicating Training Data Makes Language Models Better*, 2022). We run two passes.

**Exact dedup** hashes the normalized text of every document and drops repeats — cheap, and it catches mirrors and copy-pasted boilerplate outright:

```python
"""
capstone/stacklm/data/dedup.py

Two-stage deduplication:
  1. Exact dedup -- hash normalized text, drop exact repeats.
  2. Near-dedup -- MinHash + LSH banding over character 5-shingles, drop
     documents whose estimated Jaccard similarity to an already-kept document
     exceeds a threshold. Follows the resemblance-estimation technique of
     Broder (1997) as applied to web-text corpora by Lee et al. (2022) and
     used in RefinedWeb/FineWeb-style pipelines.

Implemented from scratch (stdlib only) so this chapter's code has no external
dependency beyond numpy/torch used elsewhere in the capstone.
"""
from __future__ import annotations
import hashlib
import random
import re
from typing import Iterable, Iterator

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for a stable exact-dup hash key."""
    return _WS_RE.sub(" ", text.lower()).strip()


def exact_dedup(docs: Iterable[dict]) -> Iterator[dict]:
    """Drop documents whose normalized-text hash has been seen before.
    O(1) memory per unique document: we keep 16-byte digests, not raw text."""
    seen: set = set()
    for doc in docs:
        h = hashlib.blake2b(normalize(doc["text"]).encode("utf-8"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        yield doc


def shingles(text: str, k: int = 5) -> set:
    """Character k-shingles: robust to small word-level edits, unlike whole-word
    shingles, which is exactly the failure mode near-duplicates exploit."""
    t = normalize(text)
    return {t[i:i + k] for i in range(len(t) - k + 1)} if len(t) >= k else {t}


class MinHasher:
    """
    `num_perm` independent hash functions h_i(x) = (a_i*x + b_i) mod p define a
    signature of `num_perm` minima over a shingle set. Broder (1997):
    P(min_i(A) == min_i(B)) = Jaccard(A, B) in expectation, so the fraction of
    matching signature positions between two documents is an unbiased estimator
    of their true Jaccard similarity -- without ever materializing the full
    shingle sets to compare them.
    """
    def __init__(self, num_perm: int = 128, seed: int = 1234):
        self.num_perm = num_perm
        self._p = (1 << 61) - 1  # a Mersenne prime, large enough to avoid collisions
        rng = random.Random(seed)
        self.a = [rng.randrange(1, self._p) for _ in range(num_perm)]
        self.b = [rng.randrange(0, self._p) for _ in range(num_perm)]

    def signature(self, shingle_set: set) -> tuple:
        if not shingle_set:
            return tuple([0] * self.num_perm)
        hashes = [
            int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
            for s in shingle_set
        ]
        return tuple(
            min((a_i * h + b_i) % self._p for h in hashes)
            for a_i, b_i in zip(self.a, self.b)
        )


class LSHIndex:
    """
    Banding: split the num_perm-length signature into `bands` bands of `rows`
    rows each. Two documents are *candidates* if any band matches exactly --
    turning an O(n^2) all-pairs comparison into near-linear bucket lookups, at
    the cost of a probabilistic threshold governed by the classic
    (1/bands)^(1/rows) S-curve (more bands = more sensitive to weak similarity).
    """
    def __init__(self, num_perm: int = 128, bands: int = 16):
        assert num_perm % bands == 0
        self.bands, self.rows = bands, num_perm // bands
        self.buckets: list = [dict() for _ in range(bands)]

    def _band_keys(self, sig: tuple) -> list:
        return [sig[i * self.rows:(i + 1) * self.rows] for i in range(self.bands)]

    def query_candidates(self, sig: tuple) -> set:
        cands: set = set()
        for b, key in enumerate(self._band_keys(sig)):
            cands.update(self.buckets[b].get(key, []))
        return cands

    def insert(self, doc_idx: int, sig: tuple) -> None:
        for b, key in enumerate(self._band_keys(sig)):
            self.buckets[b].setdefault(key, []).append(doc_idx)


def estimate_jaccard(sig_a: tuple, sig_b: tuple) -> float:
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


def near_dedup(docs: list, num_perm: int = 128, bands: int = 16,
               threshold: float = 0.8) -> list:
    """
    Single-pass streaming near-dedup: for each document, look up LSH candidates
    among already-kept documents; drop it if any candidate's estimated Jaccard
    similarity meets `threshold`. Otherwise keep it and index its signature.
    """
    hasher = MinHasher(num_perm=num_perm)
    index = LSHIndex(num_perm=num_perm, bands=bands)
    kept: list = []
    kept_sigs: list = []
    for doc in docs:
        sig = hasher.signature(shingles(doc["text"]))
        is_dup = any(
            estimate_jaccard(sig, kept_sigs[c]) >= threshold
            for c in index.query_candidates(sig)
        )
        if not is_dup:
            index.insert(len(kept), sig)
            kept_sigs.append(sig)
            kept.append(doc)
    return kept
```

Run in sequence — `near_dedup(list(exact_dedup(stream_source(entry))))` — this removes the exact and near-duplicate synthetic documents injected by `synthetic_corpus` above, exactly the way it would remove mirrored pages and syndicated articles in the real FineWeb-Edu stream. In production, `near_dedup` is run *within* each source and, ideally, *across* sources too (Cosmopedia occasionally paraphrases the same underlying facts as FineWeb-Edu); the cross-source pass is identical code, just fed a concatenated stream.

!!! warning "Common pitfall: MinHash band/row choice silently changes your recall"

    The `(bands, rows)` split determines the near-duplicate detection threshold, not just a performance knob. With `num_perm=128` and `bands=16` (so `rows=8`), the probability two documents at true Jaccard similarity $J$ are flagged as candidates is approximately $1-(1-J^{8})^{16}$ — an S-curve whose steep transition sits near the classic LSH threshold approximation $(1/\text{bands})^{1/\text{rows}} = (1/16)^{1/8}\approx 0.71$ (numerically the curve climbs from $\approx0.06$ at $J=0.5$ to $\approx0.62$ at $J=0.7$ to $\approx0.95$ at $J=0.8$). Fewer, fatter bands (say `bands=8`, `rows=16`) push that threshold higher — to $(1/8)^{1/16}\approx 0.88$ — and catch fewer near-duplicates; more, thinner bands lower it, catching more but also more false positives that then need the `estimate_jaccard` re-check to filter out. Note that candidate generation and the final decision are deliberately decoupled: the band structure is tuned to over-generate candidates around $J\approx0.7$, and `near_dedup`'s explicit `threshold=0.8` re-check on the full signature is what actually decides a drop. Always plot or sanity-check this curve for your chosen `(bands, rows)` before trusting the dedup rate — a silently wrong threshold either wastes token budget on undetected duplicates or discards genuinely distinct documents that happen to share common phrasing.

{{fig:minhash-lsh-band-scurve}}

## Tokenizing and Packing to 2048 with Document-Aware Attention

Two things happen at once in this stage: text becomes token IDs, and token IDs from many short documents get concatenated into fixed-length windows of `SEQ_LEN=2048` (`Stack-100M`'s pretraining context, per `capstone/PLAN.md` §1) — because training on ragged, individually-padded sequences wastes enormous compute on pad tokens when the median FineWeb-Edu document is far shorter than 2048 tokens. Packing multiple documents into one window recovers that compute, but naively concatenating documents lets attention flow *across* document boundaries: token 40 of document B would attend to token 2000 of unrelated document A, and RoPE's relative-position machinery ([Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)) would treat B's tokens as continuing A's position sequence. Both are wrong, and both need fixing at pack time:

1. **Position-id reset.** Every document's position ids restart at 0, regardless of where in the packed window it lands.
2. **Document-aware attention masking.** A token may only attend to earlier tokens *within its own document* — never across a packed boundary.

The tokenizer itself (byte-level BPE, `vocab_size=32768`) is trained from scratch in the next chapter; see [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) for the deeper mechanics of BPE training. Here we only need its *interface* — `encode`, plus `bos_id`/`eos_id`/`pad_id` — so this chapter's code defines that interface as a `Protocol` and, for a demo that runs without the trained artifact, a minimal byte-level fallback:

```python
"""
capstone/stacklm/data/pack.py

Tokenize documents and pack them into fixed-length SEQ_LEN=2048 sequences
with document-aware attention: tokens from different documents that land in
the same packed window never attend to each other, and each document's
position ids restart at 0. The same "pack without cross-contamination"
recipe reappears for instruction data in Chat Templates, Data Formatting &
Sequence Packing (Ch. 5.2).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol
import numpy as np

SEQ_LEN = 2048  # Stack-100M pretraining max_seq_len (capstone/PLAN.md sec.1)


class Tokenizer(Protocol):
    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int
    def encode(self, text: str) -> list: ...


class ByteTokenizer:
    """
    Dependency-free stand-in implementing the Tokenizer interface that
    Chapter 14.3's from-scratch BPE tokenizer (vocab_size=32768) implements
    for real. NOT what Stack-100M actually trains with -- swap in
    `stacklm.tokenizer.load_tokenizer()` for a real run; this class exists
    purely so the code in this chapter executes without that artifact.
    """
    bos_id, eos_id, pad_id = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))


def pack_documents(docs: Iterable[dict], tokenizer: Tokenizer,
                    seq_len: int = SEQ_LEN) -> Iterator[tuple]:
    """
    Greedily concatenate <bos> doc_tokens <eos> across documents into
    fixed-length windows. Yields (input_ids, position_ids) per full window;
    a final partial window is padded (pad_id / position 0) and yielded too,
    so no tokens are silently dropped.

    Documents longer than the window are chunked into seq_len-sized pieces,
    each treated as its own "document" for position-reset purposes -- this
    keeps every position id inside [0, seq_len), which is the range RoPE's
    theta=10000 base (capstone/PLAN.md sec.1) is tuned for during pretraining.
    """
    buf_ids: list = []
    buf_pos: list = []
    max_body = seq_len - 2  # room for <bos> and <eos> in every chunk

    for doc in docs:
        raw = tokenizer.encode(doc["text"])
        chunks = [raw[i:i + max_body] for i in range(0, len(raw), max_body)] or [[]]
        for chunk in chunks:
            toks = [tokenizer.bos_id, *chunk, tokenizer.eos_id]
            pos = list(range(len(toks)))  # this chunk's own position clock, from 0
            buf_ids.extend(toks)
            buf_pos.extend(pos)
            while len(buf_ids) >= seq_len:
                yield buf_ids[:seq_len], buf_pos[:seq_len]
                buf_ids, buf_pos = buf_ids[seq_len:], buf_pos[seq_len:]

    if buf_ids:  # flush a final, padded window
        pad_n = seq_len - len(buf_ids)
        buf_ids.extend([tokenizer.pad_id] * pad_n)
        buf_pos.extend([0] * pad_n)
        yield buf_ids, buf_pos


def build_intra_doc_causal_mask(position_ids: np.ndarray) -> np.ndarray:
    """
    Reconstruct the boolean (seq_len, seq_len) attention mask from position_ids
    alone -- True means "token i may attend to token j". Two conditions:
      (a) causal: j <= i
      (b) same document: i and j fall in the same position-reset segment,
          where a reset (position_id == 0) starts a new segment.
    In production this dense mask is replaced by FlashAttention's `cu_seqlens`
    varlen API for O(seq_len) memory instead of O(seq_len^2) -- see
    FlashAttention I: IO-Awareness & The Online Softmax (Ch. 4.2) and
    Multi-Head Attention, MQA, GQA & MLA (Ch. 2.4). This dense version is for
    teaching, unit tests, and small-scale CPU eval.
    """
    seq_len = position_ids.shape[0]
    doc_id = np.cumsum(position_ids == 0)  # monotonically increasing per document
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    same_doc = doc_id[:, None] == doc_id[None, :]
    return causal & same_doc
```

`build_intra_doc_causal_mask` is worth staring at: it needs *only* the position ids to reconstruct which tokens belong to which document, because every document boundary is exactly the points where `position_id == 0`. That is the payoff of resetting positions at pack time — document boundaries and position resets become the same signal, so we never need to store a separate document-id array on disk (Section 6 exploits this directly).

!!! example "Packing three short documents into one window"

    Suppose `seq_len=8` (tiny, for illustration) and three documents tokenize to 3, 2, and 5 body tokens respectively (using a 1-token `bos`/`eos`, so 5, 4, and 7 tokens including specials). Packing greedily:

    ```text
    doc A (5 tok): [bos a1 a2 a3 eos]      positions [0 1 2 3 4]
    doc B (4 tok): [bos b1 b2 eos]         positions [0 1 2 3]
    doc C (7 tok): [bos c1 c2 c3 c4 c5 eos] positions [0 1 2 3 4 5 6]
    ```

    Concatenated: `[bos a1 a2 a3 eos bos b1 b2 eos bos c1 c2 c3 c4 c5 eos]` (16 tokens), positions `[0 1 2 3 4 0 1 2 3 0 1 2 3 4 5 6]`. Two `seq_len=8` windows come out of `pack_documents`:

    - Window 1: tokens `[bos a1 a2 a3 eos bos b1 b2]`, positions `[0 1 2 3 4 0 1 2]`
    - Window 2: tokens `[eos bos c1 c2 c3 c4 c5 eos]`, positions `[3 0 1 2 3 4 5 6]`

    In window 1, `doc_id = cumsum(position==0) = [1 1 1 1 1 2 2 2]` — token 5 (`bos` of B) starts segment 2. Token 7 (`b2`, position 2) may attend to tokens 5 and 6 (B's own `bos`/`b1`) but **not** to tokens 0–4 (all of document A), even though they sit earlier in the same physical window. That is the entire mechanism: one `cumsum` over a boolean array recovers exact document boundaries from position ids alone.

{{fig:doc-aware-packing-mask}}

## Sharding to uint16 memmap Files and a Streaming Dataset

The last step turns a stream of `(input_ids, position_ids)` windows into files a `torch.utils.data.Dataset` can read without ever loading the full 20B-token corpus into RAM. We use the same trick as nanoGPT / llm.c: flat binary files, memory-mapped with `np.memmap`, no serialization format to parse. `vocab_size=32768` and `SEQ_LEN=2048` both fit comfortably under `uint16`'s 65,535 ceiling, so every array is 2 bytes/token — half of `int32`, doubling effective disk and page-cache bandwidth for free.

```python
"""
capstone/stacklm/data/shard.py

Write packed (input_ids, position_ids) windows to flat uint16 memmap .bin
shards. Two arrays per shard: tokens and position ids, side by side --
position ids double as the document-boundary signal consumed by
build_intra_doc_causal_mask, so no third array is needed.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

DTYPE = np.uint16  # vocab_size=32768 and seq_len=2048 both fit; see worked example


class ShardWriter:
    """Buffers packed windows and flushes a new shard every `seqs_per_shard`
    sequences (default target: ~100M tokens/shard, see Section 7)."""

    def __init__(self, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.seqs_per_shard = max(1, tokens_per_shard // seq_len)
        self._buf_ids: list = []
        self._buf_pos: list = []
        self._shard_idx = 0

    def add(self, input_ids: list, position_ids: list) -> None:
        self._buf_ids.append(np.asarray(input_ids, dtype=DTYPE))
        self._buf_pos.append(np.asarray(position_ids, dtype=DTYPE))
        if len(self._buf_ids) >= self.seqs_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_ids:
            return
        ids = np.stack(self._buf_ids)  # (n_seq, seq_len)
        pos = np.stack(self._buf_pos)  # (n_seq, seq_len)
        stem = str(self.out_dir / f"shard_{self._shard_idx:05d}")
        ids.tofile(stem + ".tokens.bin")
        pos.tofile(stem + ".pos.bin")
        np.array([ids.shape[0], ids.shape[1]], dtype=np.int64).tofile(stem + ".meta.bin")
        self._shard_idx += 1
        self._buf_ids.clear()
        self._buf_pos.clear()

    def close(self) -> None:
        self._flush()  # flush the trailing partial shard


def build_shards(docs, tokenizer, out_dir: str, seq_len: int = 2048,
                  tokens_per_shard: int = 100_000_000) -> int:
    """End-to-end: pack a document stream and write it out. Returns shard count."""
    from .pack import pack_documents  # local import to keep this snippet standalone
    writer = ShardWriter(out_dir, seq_len=seq_len, tokens_per_shard=tokens_per_shard)
    for input_ids, position_ids in pack_documents(docs, tokenizer, seq_len=seq_len):
        writer.add(input_ids, position_ids)
    writer.close()
    return writer._shard_idx
```

At train time, `PackedMemmapDataset` maps every shard's `.tokens.bin` and `.pos.bin` straight into address space. No shard is ever fully resident in RAM at once — the OS page cache serves whatever windows the dataloader touches, which is what makes an 80GB corpus (Section 7) trainable on a machine with far less than 80GB of RAM:

```python
"""
capstone/stacklm/data/dataset.py

torch Dataset over the sharded uint16 .bin files. `position_ids` double as
the document-boundary signal: `build_intra_doc_causal_mask` (pack.py)
reconstructs the attention mask on the fly from `position_ids == 0`, so no
extra array needs to be stored or loaded per batch.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PackedMemmapDataset(Dataset):
    def __init__(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        self._shards = []       # list of (tokens_memmap, pos_memmap)
        self._cum_seqs = [0]    # prefix sums of sequence counts, for indexing
        for meta_path in sorted(self.shard_dir.glob("shard_*.meta.bin")):
            n_seq, seq_len = (int(x) for x in np.fromfile(meta_path, dtype=np.int64))
            stem = str(meta_path)[: -len(".meta.bin")]
            tok_mm = np.memmap(stem + ".tokens.bin", dtype=np.uint16, mode="r",
                                shape=(n_seq, seq_len))
            pos_mm = np.memmap(stem + ".pos.bin", dtype=np.uint16, mode="r",
                                shape=(n_seq, seq_len))
            self._shards.append((tok_mm, pos_mm))
            self._cum_seqs.append(self._cum_seqs[-1] + n_seq)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self._cum_seqs[-1]

    def _locate(self, idx: int) -> tuple:
        for s, (start, end) in enumerate(zip(self._cum_seqs, self._cum_seqs[1:])):
            if start <= idx < end:
                return s, idx - start
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> dict:
        s, row = self._locate(idx)
        tok_mm, pos_mm = self._shards[s]
        ids = torch.from_numpy(tok_mm[row].astype(np.int64))
        pos = torch.from_numpy(pos_mm[row].astype(np.int64))
        # Standard next-token target: shift by one. The training loop (Ch. 14.7)
        # masks the loss on pad_id targets so the trailing padded window costs
        # nothing extra beyond the (small) wasted compute of the forward pass.
        return {"input_ids": ids[:-1], "position_ids": pos[:-1], "targets": ids[1:]}
```

## Worked Example: Token, Byte, and Shard Accounting

!!! example "Sizing the full 20B-token Stack-100M corpus"

    **Mix breakdown.** 20B tokens split 70/15/10/5 gives exactly 14.0B (FineWeb-Edu), 3.0B (Cosmopedia v2), 2.0B (StarCoder), and 1.0B (FineMath/OpenWebMath) tokens — the weights were chosen to divide the budget cleanly.

    **Raw text volume.** Byte-level BPE at `vocab_size=32768` typically compresses English-heavy prose to on the order of 4–4.5 bytes/token (similar to GPT-2's byte-level tokenizer). At 20B tokens that implies roughly

    $$
    20\times10^{9}\ \text{tokens} \times 4.5\ \tfrac{\text{bytes}}{\text{token}} \approx 9.0\times10^{10}\ \text{bytes} \approx 90\ \text{GB}
    $$

    of *kept* raw UTF-8 text. The raw volume you must *stream and filter* to end up with 90GB of kept text is substantially larger — quality filters and deduplication routinely discard a large majority of raw Common Crawl before FineWeb-Edu-style filtering, which is exactly why sourcing is a streaming pass over a much bigger corpus (Section 3), not a one-shot download.

    **Sequence count.** At `SEQ_LEN=2048`, the token budget divides *exactly*:

    $$
    \frac{20\times10^{9}}{2048} = 9{,}765{,}625 \text{ packed sequences.}
    $$

    **Shard sizes.** Each `uint16` token occupies 2 bytes, so the full `tokens.bin` corpus across all shards is $20\times10^{9}\times 2 = 4.0\times10^{10}$ bytes = **40 GB**, and `pos.bin` is another **40 GB** — **≈80 GB total** on disk for the packed corpus. Sharding at a target of ~100M tokens/shard gives `seqs_per_shard = 100_000_000 // 2048 = 48{,}828` sequences/shard (≈99.99M tokens after flooring), so the corpus splits into

    $$
    \left\lceil \frac{9{,}765{,}625}{48{,}828} \right\rceil \approx 200 \text{ shards},
    $$

    each roughly 200MB `tokens.bin` + 200MB `pos.bin` ≈ **400MB/shard**. 200 shards of manageable size are easy to distribute, resume, and spot-check individually — a corruption in one shard costs you 0.5% of the run, not the whole thing (see [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html) for the training-time half of this story).

    **Compute consistency check.** Section 1 derived $C\approx 1.22\times10^{19}$ FLOPs for this token budget at $N\approx101.4$M. `capstone/PLAN.md`'s flagship figure is 15–25 GPU-hours on one A100. Solving for the sustained throughput each end of that range implies: the 20-hour midpoint needs $1.22\times10^{19}\ \text{FLOPs} / (20\,\text{hr}\times 3600\,\text{s/hr}) \approx 1.7\times10^{14}\ \text{FLOP/s}$ — on the order of 170 TFLOP/s, or roughly 55% of an A100's bf16 peak (312 TFLOP/s); the 25-hour end needs only ~135 TFLOP/s (~43% MFU). The lower, more conservative end is the honest expectation for a *deep-and-thin* model: at `d_model=512` the matmuls are small enough that tensor cores are under-fed, so model-FLOP utilization (MFU) at this shape typically lands well below what a wide model of the same parameter count would hit — which is exactly why the budget carries a range rather than a point estimate. This is a rough consistency check, not a benchmark claim: the actual achieved MFU is measured and discussed in [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) and in the pretraining-loop chapter.

## Key Takeaways & Further Reading

!!! key "Key Takeaways"

    - `Stack-100M` deliberately over-trains: ~20B tokens against a ~101.4M-parameter model is ≈200 tokens/param, roughly 10× Chinchilla's compute-optimal ≈20 tokens/param (≈2B tokens) for this size — a deployment-economics bet, not a training-compute-optimal one, because inference cost scales with $N$ while the extra pretraining tokens are a one-time cost.
    - The mix follows the SmolLM/FineWeb recipe: 70% FineWeb-Edu (filtered web), 15% Cosmopedia v2 (synthetic textbooks), 10% StarCoder (code), 5% FineMath/OpenWebMath — broad web knowledge as the base, synthetic data for density, code and math for structured-reasoning transfer.
    - Every streaming source has a deterministic, network-free synthetic fallback with the same document schema, so the entire pipeline is hermetic in CI and runnable offline.
    - Deduplication is two-stage: cheap exact-hash dedup catches mirrors and boilerplate; MinHash + LSH banding catches near-duplicates in near-linear time by estimating Jaccard similarity from signature agreement (Broder, 1997).
    - Packing concatenates many short documents into fixed `SEQ_LEN=2048` windows for compute efficiency, but requires document-aware attention masking and per-document position-id resets to prevent cross-document attention leakage and RoPE position aliasing.
    - Position ids double as the document-boundary signal: `position_id == 0` marks a new document, so `build_intra_doc_causal_mask` reconstructs the exact attention mask from position ids alone, no separate document-id array required.
    - `uint16` is the right dtype for both token ids (`vocab_size=32768`) and position ids (`< 2048`) — 2 bytes/token, half the size of `int32`, for zero loss of information at this scale: the full packed corpus is ≈80GB (40GB tokens + 40GB positions) across roughly 200 shards.
    - `PackedMemmapDataset` memory-maps shards directly, so the OS page cache — not application RAM — serves the working set, making an 80GB corpus trainable on modest hardware.

**Further reading**

- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla"), 2022 — the $D^\*\approx 20N$ result this chapter deliberately trains past.
- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, HuggingFace, 2024 — FineWeb and FineWeb-Edu.
- Ben Allal et al. and the HuggingFace SmolLM/SmolLM2 team's technical reports and blog posts — the small-model, high-quality-mix recipe this capstone's data mix follows, and the origin of Cosmopedia.
- Li et al. and the BigCode community, *StarCoder: May the Source Be With You!*, 2023.
- Broder, *On the Resemblance and Containment of Documents*, 1997 — the MinHash resemblance-estimation technique used for near-dedup.
- Lee et al., *Deduplicating Training Data Makes Language Models Better*, 2022 — the empirical case for aggressive corpus deduplication.
- Karpathy, nanoGPT and llm.c — the flat uint16/uint32 memmap `.bin` sharding convention this chapter's `ShardWriter`/`PackedMemmapDataset` follow.
- [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) and [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) — the full theory behind the lean, at-scale versions of filtering and dedup used in this chapter.
