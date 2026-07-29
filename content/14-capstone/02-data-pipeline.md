# 14.2 Data: Sourcing, Filtering, Dedup, Tokenize & Pack ~20B Tokens

Every chapter so far in this book has treated data as an input — something that arrives already tokenized, already packed, already a clean tensor of `input_ids`. This chapter is where we build that tensor. We are going to source, filter, deduplicate, tokenize, pack, and shard roughly 20 billion tokens for `Stack-100M`, the ~101.4M-parameter model this capstone builds end to end (architecture fixed in `capstone/PLAN.md` §1; full parameter count derived in Ch. 14.4).

The deliverable of this chapter is not a set of illustrative snippets — it is a runnable corpus builder. Everything lives under the `stacklm` package:

| File | Role |
|---|---|
| `capstone/stacklm/data/synthetic.py` | Source registry (repo id, config, column, revision) + streaming + the offline synthetic fallback |
| `capstone/stacklm/data/filters.py` | Domain-routed quality gates and the hashable filter config |
| `capstone/stacklm/data/dedup.py` | Exact hashing; MinHash + LSH near-duplicate detection |
| `capstone/stacklm/data/pack.py` | Tokenize and pack into `SEQ_LEN=2048` windows; document-aware masking |
| `capstone/stacklm/data/shard.py` | `uint16` memmap shard writer + `manifest.json` |
| `capstone/stacklm/data/dataset.py` | `torch` `Dataset` over the shards |
| `capstone/stacklm/data/build_corpus.py` | The driver: per-source budgets, interleaving, shuffle, held-out split, manifest |

The number "20 billion" is not arbitrary, and it is larger than you might expect for a 100M-parameter model. The first order of business is explaining why — because the answer is the single most important lesson this capstone teaches about how small models are actually trained in 2025–2026, and it shapes every decision that follows.

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

We round this to "≈200 tokens/param" throughout, matching `capstone/PLAN.md`. The bet only pays off because $N$ is small enough that $10\times$ the tokens is still an affordable day of GPU time — at 70B parameters the same ratio would cost a fortune. That is the deployment-economics argument in one sentence: **spend the extra compute where it's cheap (training, once) to save it where it's expensive (inference, forever).**

### What the run actually costs: 6ND is not enough here

The usual shorthand for training compute is $C\approx 6ND$: two FLOPs per multiply-accumulate, forward plus backward being roughly $3\times$ forward, applied to every parameter for every token (see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)). That term covers all the *weight* matmuls, including the tied `lm_head`:

$$
6N = 6 \times 1.014\times10^{8} = 6.08\times10^{8}\ \text{FLOPs/token}.
$$

But $6ND$ counts only matmuls against weights. The attention score and value matmuls — $QK^\top$ and $\text{softmax}(\cdot)V$ — have no parameters at all; their cost scales with *context length*, not with $N$. For a sequence of length $T$ and per-layer model width $d$, the forward pass costs $4T^2 d$ FLOPs per layer, so per token it is $4Td$; multiplying by 3 for forward+backward and halving for causal masking (we only compute the lower triangle) gives $6\,T\,d$ per layer per token, i.e.

$$
\text{FLOPs}_{\text{attn}}/\text{token} = 6\,n_{\text{layers}}\,n_{\text{ctx}}\,d_{\text{model}}
= 6\times 30\times 2048\times 512 = 1.89\times10^{8}.
$$

That is **31% of $6N$**, not a rounding error. The usual justification for dropping the attention term — "$d_{\text{model}} \gg n_{\text{ctx}}$, so it's negligible" — is exactly backwards for `Stack-100M`: at $d_{\text{model}}=512$ and $n_{\text{ctx}}=2048$ the context is four times the width. Deep-and-thin models pay proportionally more for attention than wide ones. So:

$$
C \approx \big(6N + 6\,n_{\text{layers}}\,n_{\text{ctx}}\,d_{\text{model}}\big)\,D
= (6.08 + 1.89)\times10^{8} \times 2\times10^{10} \approx 1.6\times10^{19}\ \text{FLOPs.}
$$

(If you use Kaplan's convention, which does not halve for causality, the attention term is $12\,n_{\text{layers}}\,n_{\text{ctx}}\,d_{\text{model}} = 3.78\times10^{8}$ and the total is $\approx 2.0\times10^{19}$. Both conventions appear in the literature; state which one you are using whenever you quote an MFU number, because they differ by 25%.)

!!! warning "Common pitfall: quoting MFU against the wrong FLOP count"

    Model-FLOP utilization is $\text{MFU} = \frac{\text{FLOPs/token}\times\text{tokens/s}}{\text{peak FLOP/s}}$, and *every* term is a choice. Report MFU against $6N$ alone and you will understate your utilization by ~24% here ($6N / (6N + \text{attn}) = 0.76$); report it against the Kaplan convention and you overstate it. Worse, some tools quote "hardware FLOPs utilization" (HFU), which counts recomputed activations from gradient checkpointing as useful work — HFU is always ≥ MFU and can exceed it by 20–30% in a checkpointed run. When you compare your throughput number to a published one, first check that the two are the same number. Ch. 14.7 measures `Stack-100M`'s MFU with the $6N + 6\,L\,T\,d$ convention used here, and reports both.

Wall-clock is $C / (\text{MFU}\times\text{peak})$, so the budget table is:

| GPU (bf16 dense peak) | 40% MFU | 50% MFU | 58% MFU |
|---|---:|---:|---:|
| **A100 80GB (312 TFLOP/s)** — the flagship tier | ≈ 35 hr | ≈ 28 hr | **≈ 24 hr** |
| H100 SXM (≈ 495 TFLOP/s dense) | ≈ 22 hr | ≈ 18 hr | ≈ 15 hr |

`capstone/PLAN.md` fixes the flagship tier as **a single A100 80GB, 22–29 GPU-hours, USD 25–50**, and the bolded cell is where Ch. 14.7's measured loop lands: a `torch.compile`d, FlashAttention-backed step at ≈228k tokens/s is **44.5% MFU under the $6ND$ convention and 58.2% under the $6N+\text{attn}$ convention** — one run, two numbers — giving $2\times10^{10}/2.28\times10^{5} \approx 8.8\times10^{4}$ s ≈ **24.4 A100-hours** for the full 20B tokens, the same figure as $C/(0.582\times3.12\times10^{14})$. That sits mid-band in PLAN's envelope; reaching the lower end means pushing tokens/s up, not inventing utilization. At USD 1–2/A100-hour that is roughly USD 25–50, comfortably inside "the USD 100 model" once you add mid-training's decay phase and the failed first attempt everybody has. The H100 row is a comparison, not a requirement. Ch. 14.7 measures; Ch. 14.12 does the final cost accounting.

!!! interview "Interview Corner"

    **Q:** Chinchilla says $D^\*\approx 20N$ minimizes loss for a fixed *training* compute budget. Why would anyone deliberately train past that point — isn't that compute-inefficient?

    **A:** It's training-compute-inefficient but can be deployment-compute-efficient, and those are different optimization problems. Chinchilla solves $\min_{N,D} L(N,D)$ subject to $C=6ND$ — an objective with no inference term in it at all. If the model will be served, the relevant objective is closer to

    $$
    \min_{N}\ \big[\,L(N, D) + \lambda \cdot \text{cost}_{\text{inf}}(N)\,\big]
    $$

    for whatever one-time token budget $D$ you are willing to spend, where $\lambda$ encodes expected serving volume. Because $\text{cost}_{\text{inf}}$ grows with $N$ (linearly in FLOPs/token, and in memory footprint, and in KV-cache size) while extra $D$ is paid once, the minimizer sits at a smaller $N$ and a larger $D$ than Chinchilla's. The bet is an *amortization* argument, so it inverts when $\lambda \to 0$: for a model you evaluate once and discard — a scaling-law data point, a one-off research artifact — over-training is simply wasted compute and you should stop near $D^\*$.

    **Follow-up they will ask:** *"You said $C=6ND$. When is that wrong?"* When context is long relative to width. The attention score/value matmuls cost $\approx 6\,n_{\text{layers}}\,n_{\text{ctx}}\,d_{\text{model}}$ FLOPs/token on top of $6N$ — negligible for a 70B model at 4k context (well under 5%), but 31% of the budget for a 100M-parameter model at 2k context. Any time you plan a run for a small or deep-and-thin model, carry the attention term.

{{fig:overtrain-deployment-economics}}

## The Stack-100M Data Mix

We follow the recipe popularized by HuggingFace's SmolLM series: a large base of filtered educational web text, topped up with synthetic textbooks and slices of code and math, rather than raw, undifferentiated Common Crawl. The mix is fixed in `capstone/PLAN.md` §2 — but a mix table that lists only dataset *names* is not reproducible. Here is the full loading coordinate for each source, which is what the code actually needs:

| Source | HF repo | config (`name=`) | `data_dir` | text column | Weight | Tokens |
|---|---|---|---|---|---:|---:|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-100BT` | — | `text` | 70% | 14.0B |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` | — | `text` | 15% | 3.0B |
| StarCoder (Python) | `bigcode/starcoderdata` | — | `python` | **`content`** | 10% | 2.0B |
| FineMath | `HuggingFaceTB/finemath` | `finemath-4plus` | — | `text` | 5% | 1.0B |

!!! warning "Common pitfall: the dataset id is not enough"

    Three of the four entries above will fail — or, far worse, *silently succeed and yield nothing* — if you pass only the repo id to `load_dataset`.

    - **Multi-config repos have no default.** `HuggingFaceFW/fineweb-edu`, `HuggingFaceTB/finemath`, and `HuggingFaceTB/smollm-corpus` all ship several configs; calling `load_dataset(repo, split="train")` without `name=` raises a `ValueError` listing the available configs. Pick deliberately: FineWeb-Edu's default config is the full multi-terabyte dump, while `sample-100BT` is a pre-sampled 100B-token slice — far more than the 14B we need and vastly cheaper to stream. FineMath exposes quality tiers (`finemath-3plus`, `finemath-4plus`, and `infiwebmath-*` variants); higher tiers are smaller and cleaner.
    - **"Cosmopedia v2" is not in the `cosmopedia` repo.** `HuggingFaceTB/cosmopedia` is *v1* (configs `web_samples_v1`, `web_samples_v2`, `stories`, `stanford`, `openstax`, `khanacademy`, `auto_math_text`, `wikihow`). The v2 regeneration that SmolLM2 actually trained on lives in `HuggingFaceTB/smollm-corpus` under config `cosmopedia-v2`.
    - **The text column is not always `text`.** `bigcode/starcoderdata` stores source under **`content`**. A pipeline that does `row.get("text", "")` against it drops every single row and produces a corpus with 0% code — with no error, no warning, and no way to notice until your model cannot write a `for` loop. This is why the code below *asserts* on non-empty output instead of using `.get(..., "")`.
    - **Some sources are gated.** `bigcode/starcoderdata` requires accepting terms on the Hub and `huggingface_hub.login()` first; it is also sharded by language via `data_dir=`, so you request `python` (or `java`, `javascript`, …) rather than the whole 800GB. The newer `bigcode/the-stack-v2` family stores *file pointers* rather than file contents, and requires a separate fetch from Software Heritage's S3 bucket to materialize the code — a real pipeline step, not a `load_dataset` call. `starcoderdata` is the lower-friction choice at our scale.
    - **Pin a revision.** Datasets on the Hub are mutable. Passing `revision=<commit sha>` is what makes "I trained on FineWeb-Edu" a reproducible statement; our registry defaults to `"main"` and records whatever you pass in the corpus manifest.

**FineWeb-Edu** (Penedo et al., HuggingFace, 2024) is Common Crawl filtered to the subset a lightweight classifier judges educational, that classifier having been trained on quality labels distilled from a large teacher LLM. It is the bulk of the mix because aggressively quality-filtered web text remains the best source of broad linguistic and world knowledge per token. **Cosmopedia v2** (HuggingFaceTB) is entirely synthetic — textbooks, blog posts, and stories generated by a large model from curated seed topics and reading levels — denser in unambiguous, well-structured knowledge than the median web page, but less diverse, which is why it is 15% and not the majority (see [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)). **StarCoder** (BigCode) and **FineMath** are the "capability injection" slices: code and math text noticeably improve a small model's structured reasoning even on non-code, non-math tasks, and they are prerequisites for the tool-using agent of Ch. 14.9–14.10. At 10% and 5%, they are seasoning, not the entrée.

These weights are a starting point, not a law — Ch. 14.5's mini scaling-law ladder and Ch. 14.8's mid-training annealing both revisit and re-weight this mix; see [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html) for the general theory of how and why to reweight a training mix over the course of a run.

## Sourcing: Streaming, Failing Loudly, and an Offline Fallback

At 20B tokens, none of these corpora fit on a laptop disk in raw form, and CI must run with no network access at all. Both constraints point to the same design: **stream** documents rather than downloading full datasets, and give every source a **deterministic, in-process synthetic fallback** with the same schema, so the entire pipeline — filtering, dedup, packing, sharding — is exercised end to end without ever touching the network.

The registry below carries the full loading coordinates from the table above, so a source can never be opened with a guessed column name:

```python
"""
capstone/stacklm/data/synthetic.py

Source registry + offline synthetic corpus (Ch. 14.2). Each entry carries the
FULL coordinates needed to load it: repo id, config name, optional data_dir, and
the column the text actually lives in -- these differ per dataset, and getting
them wrong is the most common way a data pipeline silently produces zero
documents.
"""
import hashlib
import random
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class DataMixEntry:
    name: str                      # short id, e.g. "fineweb_edu"
    hf_path: str                   # HuggingFace repo id
    weight: float                  # fraction of the 20B-token budget
    domain: str                    # "web"|"synthetic"|"code"|"math" -- routes the filter
    hf_config: Optional[str] = None    # config/subset name (`name=` in load_dataset)
    hf_data_dir: Optional[str] = None  # directory-sharded repos (starcoderdata)
    text_column: str = "text"          # NOT always "text" (starcoderdata: "content")
    revision: str = "main"             # pin to a commit sha for a reproducible corpus
    gated: bool = False                # requires huggingface_hub.login()


# The exact mix fixed in capstone/PLAN.md sec. 2. Weights sum to 1.0.
STACK100M_MIX = [
    DataMixEntry("fineweb_edu", "HuggingFaceFW/fineweb-edu", 0.70, "web",
                 hf_config="sample-100BT"),
    DataMixEntry("cosmopedia_v2", "HuggingFaceTB/smollm-corpus", 0.15, "synthetic",
                 hf_config="cosmopedia-v2"),
    DataMixEntry("starcoder", "bigcode/starcoderdata", 0.10, "code",
                 hf_data_dir="python", text_column="content", gated=True),
    DataMixEntry("finemath", "HuggingFaceTB/finemath", 0.05, "math",
                 hf_config="finemath-4plus"),
]

TOTAL_TOKEN_BUDGET = 20_000_000_000  # ~20B tokens, ~200 tok/param (PLAN.md sec. 2)


def load_hf_stream(entry: DataMixEntry):
    """Open one source as a streaming `datasets.IterableDataset`.

    Raises rather than returning an empty stream: a mistyped config or column is
    a bug, and a pipeline that silently yields nothing is the worst possible
    failure mode -- you find out 20 GPU-hours later.
    """
    from datasets import load_dataset  # heavy optional dependency; not in CI

    ds = load_dataset(
        entry.hf_path,
        name=entry.hf_config,
        data_dir=entry.hf_data_dir,
        split="train",
        streaming=True,
        revision=entry.revision,
    )
    cols = getattr(ds, "column_names", None)
    if cols is not None and entry.text_column not in cols:
        raise KeyError(
            f"{entry.name}: text column {entry.text_column!r} not in {cols}. "
            "Most HF text corpora use 'text', but bigcode/starcoderdata "
            "uses 'content'."
        )
    return ds


def stream_hf(entry: DataMixEntry, probe: int = 8) -> Iterator[dict]:
    """Yield normalized {"text","source","domain"} docs from the real dataset.
    Asserts that the first `probe` rows are not all empty, so a misconfigured
    source fails loudly instead of contributing zero tokens."""
    ds = load_hf_stream(entry)
    n_seen = n_nonempty = 0
    for row in ds:
        text = row.get(entry.text_column) or ""
        if n_seen < probe:
            n_seen += 1
            n_nonempty += bool(text)
            if n_seen == probe and n_nonempty == 0:
                raise ValueError(
                    f"{entry.name}: first {probe} rows had an empty "
                    f"{entry.text_column!r} field -- wrong column or wrong config?"
                )
        if text:
            yield {"text": text, "source": entry.name, "domain": entry.domain}
```

The fallback logic deserves one deliberate design note. It is tempting to wrap the whole streaming loop in `try/except Exception: pass` so the pipeline "always works". Do not. That converts a transient HTTP 503 in hour six of a download into a *silently truncated corpus* — you get shards, you get a loss curve, and you never learn that 40% of FineWeb-Edu is missing. We guard only the *opening* of the stream, and only against the two conditions that genuinely mean "there is no network here": `datasets` not installed, or the Hub unreachable.

```python
def stream_source(entry: DataMixEntry, offline: bool = False,
                  n_docs: int = 2000) -> Iterator[dict]:
    """Real HF stream when `offline=False`, synthetic fallback otherwise.

    Only the *opening* of the stream is guarded: if `datasets` is missing or the
    Hub is unreachable we fall back to the synthetic corpus, but once the stream
    is open, errors propagate. Swallowing mid-stream exceptions would silently
    truncate the corpus.
    """
    if not offline:
        try:
            gen = stream_hf(entry)
            first = next(gen)
        except (ImportError, OSError, ConnectionError):
            pass                                   # no `datasets` / no network
        else:
            yield first
            yield from gen
            return
    yield from synthetic_corpus(entry, n_docs=n_docs)
```

Because `stream_hf` is a generator, `load_dataset` does not actually fire until the first `next(gen)` — which is inside the `try`. That is why the guard catches a dead Hub at all, and why nothing after that first row is protected.

### The offline synthetic fallback

The synthetic generator is deliberately small and deterministic (seeded per source), and it intentionally injects a handful of exact and near-duplicate documents — so the deduplication code below has something real to catch even when there is no network:

```python
_dup_cache: dict = {}

_VOCAB = {
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
}


def synthetic_corpus(entry: DataMixEntry, n_docs: int = 2000) -> Iterator[dict]:
    """Deterministic in-process corpus with injected exact (every 97th) and near
    (every 53rd) duplicates, so the dedup stages have something real to catch.
    It teaches the model nothing -- it exists so every downstream stage is
    exercised by CI as a hermetic stand-in for the real streams."""
    seed = int(hashlib.blake2b(entry.name.encode(), digest_size=4).hexdigest(), 16)
    rng = random.Random(seed)
    vocab = _VOCAB[entry.domain]
    for i in range(n_docs):
        if entry.domain in _dup_cache and i % 97 == 0:
            text = _dup_cache[entry.domain]                        # exact duplicate
        elif entry.domain in _dup_cache and i % 53 == 0:
            base = _dup_cache[entry.domain].split()                # near-duplicate
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

### Quality filtering at ingest

[Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) covers the full battery of heuristic filters (length bounds, character-class ratios, repeated-n-gram detection, language ID) used to clean web-scale corpora. We re-implement a lean, domain-routed subset here: web and synthetic prose get the generic filter, but code and math have such different character distributions that the generic filter would wrongly reject nearly all of them (a `def` block has almost no English stop words and a high symbol density; a proof has a much higher digit fraction than prose).

Every threshold lives in one dict, and that dict is hashed into the corpus manifest — so "which filter produced this corpus?" has an answer you can check, not a memory you can misremember.

```python
"""
capstone/stacklm/data/filters.py

Fast, dependency-free quality gates, domain-routed. A lean subset of Ch. 3.2,
tuned so filtering compute doesn't compete with training compute at 20B tokens.
"""
import hashlib
import json
import re

_WORD_RE = re.compile(r"\S+")

FILTER_CONFIG = {
    "web": {"min_words": 50, "max_words": 100_000, "min_mean_word_len": 3.0,
            "max_mean_word_len": 10.0, "min_alpha_frac": 0.60, "max_digit_frac": 0.20,
            "max_repeat_line_frac": 0.30},
    "code": {"min_chars": 20, "max_chars": 200_000, "max_char_frac": 0.30},
    "math": {"min_words": 20, "min_digit_frac": 0.03, "min_markers": 3},
}


def filter_config_hash() -> str:
    """Stable 12-hex-char digest of FILTER_CONFIG, recorded in the manifest."""
    blob = json.dumps(FILTER_CONFIG, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=6).hexdigest()


def basic_stats(text: str) -> dict:
    words = _WORD_RE.findall(text)
    n_words = len(words) or 1
    n_chars = len(text) or 1
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    lines = text.splitlines() or [text]
    uniq = len(set(lines))
    return dict(
        n_words=n_words,
        alpha_frac=alpha / n_chars,
        digit_frac=digit / n_chars,
        mean_word_len=sum(len(w) for w in words) / n_words,
        dup_line_frac=1.0 - uniq / len(lines),   # boilerplate / nav-bar detector
    )


def passes_web_filter(text: str) -> bool:
    """Generic prose gate for FineWeb-Edu / Cosmopedia documents."""
    c = FILTER_CONFIG["web"]
    s = basic_stats(text)
    return (
        c["min_words"] <= s["n_words"] <= c["max_words"]
        and c["min_mean_word_len"] <= s["mean_word_len"] <= c["max_mean_word_len"]
        and s["alpha_frac"] >= c["min_alpha_frac"]
        and s["digit_frac"] <= c["max_digit_frac"]
        and s["dup_line_frac"] <= c["max_repeat_line_frac"]
    )


def passes_code_filter(text: str) -> bool:
    """Loose gate for StarCoder: reject empty/binary/minified-looking files
    (dominated by one repeated character); keep everything else, since code has
    a very different character distribution than prose and would be wrongly
    rejected by `passes_web_filter`."""
    c = FILTER_CONFIG["code"]
    if not (c["min_chars"] <= len(text) <= c["max_chars"]):
        return False
    head = text[:2000]
    most_common_frac = max(head.count(ch) for ch in set(head)) / max(len(head), 1)
    return most_common_frac <= c["max_char_frac"]


def passes_math_filter(text: str) -> bool:
    """FineMath gate: require mathematical density (digits, operators, LaTeX-ish
    markers), not generic prose fluency."""
    c = FILTER_CONFIG["math"]
    s = basic_stats(text)
    markers = sum(text.count(m) for m in ("=", "\\frac", "$", "^", "\\sum"))
    return s["n_words"] >= c["min_words"] and (
        s["digit_frac"] >= c["min_digit_frac"] or markers >= c["min_markers"]
    )


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

### Deduplication: exact hashing and MinHash near-duplicates

Even after quality filtering, web-scale corpora contain enormous amounts of exact and near-duplicate content — mirrored pages, boilerplate legal text, syndicated news, forum threads quoting each other. Duplicate content wastes token budget and, worse, causes the model to memorize rather than generalize (Lee et al., *Deduplicating Training Data Makes Language Models Better*, 2022). We run two passes.

**Exact dedup** hashes the normalized text of every document and drops repeats — cheap, streaming, and it catches mirrors and copy-pasted boilerplate outright. **Near dedup** estimates Jaccard similarity between shingle sets with MinHash signatures, and uses LSH banding to avoid the $O(n^2)$ all-pairs comparison.

Two implementation details matter enough to be their own lessons, and both are about *representation*, not algorithm.

First, the permutation sweep is **vectorized with numpy**. The textbook formulation — `min((a_i*h + b_i) % p for h in hashes)` inside a Python loop over 128 permutations — does `num_perm × n_shingles` Python-level modular multiplies. On a 5.5 KB document (5,541 shingles) we measure ~104 ms that way and ~8 ms with the `(num_perm, n_shingles)` matrix built once and reduced by `.min(axis=1)`: a ~12× speedup, and over 20M documents the difference between ~580 core-hours and ~44. To keep the products inside `int64`, we use the Mersenne prime $2^{31}-1$ rather than $2^{61}-1$: $a\cdot h < 2^{62}$ then fits without overflow.

Second, **how you store a signature decides how many you can hold**. A signature stored as a Python tuple of 128 ints measures ~4.9 KB (tuple header plus 128 boxed `int` objects, none of them small enough for CPython's small-int cache); the identical information as a row of a `uint32` numpy array is $128\times 4 = 512$ bytes. That single change is a ~10× difference in how much of the corpus a bounded streaming index can cover, so `MinHasher.signature` returns an array and `SignatureStore` keeps those rows contiguous.

```python
"""
capstone/stacklm/data/dedup.py

Two-stage deduplication:
  1. Exact dedup  -- blake2b over normalized text, streaming.
  2. Near dedup   -- MinHash (Broder, 1997) + LSH banding over character
                     5-shingles, streaming with a bounded index.

Implemented from scratch (stdlib + numpy) so the mechanism is visible. For a
real 20B-token corpus use `datatrove`'s Minhash* pipeline -- see "Production
path" below for the throughput and memory arithmetic that forces it.
"""
import hashlib
import logging
import random
import re
from typing import Iterable, Iterator, List

import numpy as np

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_MERSENNE_31 = (1 << 31) - 1  # keeps a*h+b inside int64 for vectorized minhashing


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for a stable exact-dup hash key."""
    return _WS_RE.sub(" ", text.lower()).strip()


def exact_dedup(docs: Iterable[dict]) -> Iterator[dict]:
    """Drop documents whose normalized-text digest has been seen before.

    The payload is 16 bytes per *unique* document, but a CPython `set` of
    `bytes` costs ~80-90 B/element once the hash-table slot and object header
    are counted: budget ~1.7 GB at 20M unique documents, not the 320 MB the
    payload alone suggests.
    """
    seen: set = set()
    for doc in docs:
        h = hashlib.blake2b(normalize(doc["text"]).encode("utf-8"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        yield doc


def shingles(text: str, k: int = 5) -> List[str]:
    """Character k-shingles: robust to the small word-level edits that
    near-duplicates are made of, unlike whole-word shingles."""
    t = normalize(text)
    if len(t) < k:
        return [t]
    return list({t[i:i + k] for i in range(len(t) - k + 1)})


class MinHasher:
    """`num_perm` hash functions h_i(x) = (a_i*x + b_i) mod p give a signature of
    `num_perm` minima over a shingle set. Broder (1997):
    P(min_i(A) == min_i(B)) = Jaccard(A, B), so the fraction of matching
    signature positions is an unbiased estimator of the true Jaccard similarity
    -- without ever materializing the shingle sets to compare them.

    The permutation sweep is vectorized: the (num_perm, n_shingles) matrix is
    built once and reduced with `.min(axis=1)`, ~12x faster than the Python loop
    (measured ~8 ms vs ~104 ms on a 5.5 KB document).
    """

    def __init__(self, num_perm: int = 128, seed: int = 1234):
        self.num_perm = num_perm
        self._p = _MERSENNE_31
        rng = random.Random(seed)
        self.a = np.array([rng.randrange(1, self._p) for _ in range(num_perm)], dtype=np.int64)
        self.b = np.array([rng.randrange(0, self._p) for _ in range(num_perm)], dtype=np.int64)

    def _shingle_hashes(self, shingle_list) -> np.ndarray:
        raw = b"".join(
            hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest() for s in shingle_list
        )
        h = np.frombuffer(raw, dtype=">u4").astype(np.int64)
        return h % self._p

    def signature(self, shingle_list) -> np.ndarray:
        """(num_perm,) uint32 signature. Deliberately NOT a tuple of Python
        ints: a 128-int tuple measures ~4.9 KB, this row is 128*4 = 512 B."""
        if not shingle_list:
            return np.zeros(self.num_perm, dtype=np.uint32)
        h = self._shingle_hashes(shingle_list)                       # (n_shingles,)
        mixed = (self.a[:, None] * h[None, :] + self.b[:, None]) % self._p
        return mixed.min(axis=1).astype(np.uint32)                   # (num_perm,)


class SignatureStore:
    """Contiguous (n, num_perm) uint32 store, grown by doubling up to `capacity`.

    Row cost is exactly num_perm*4 bytes -- 512 B at num_perm=128, versus ~4.9 KB
    for the tuple-of-ints it replaces.
    """

    def __init__(self, num_perm: int, capacity: int, initial: int = 4096):
        self.num_perm, self.capacity = num_perm, capacity
        self._buf = np.empty((max(1, min(initial, capacity)), num_perm), dtype=np.uint32)
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> np.ndarray:
        return self._buf[i]

    def append(self, sig: np.ndarray) -> None:
        assert self.n < self.capacity, "SignatureStore is full"
        if self.n == self._buf.shape[0]:
            grown = np.empty((min(2 * self.n, self.capacity), self.num_perm),
                             dtype=np.uint32)
            grown[: self.n] = self._buf
            self._buf = grown
        self._buf[self.n] = sig
        self.n += 1


class LSHIndex:
    """Banding: split the signature into `bands` bands of `rows` rows. Two docs
    are *candidates* if any band matches exactly, turning an O(n^2) all-pairs
    scan into near-linear bucket lookups at the cost of a probabilistic
    threshold governed by the (1/bands)^(1/rows) S-curve."""

    def __init__(self, num_perm: int = 128, bands: int = 16):
        assert num_perm % bands == 0
        self.bands, self.rows = bands, num_perm // bands
        self.buckets: list = [dict() for _ in range(bands)]

    def _band_keys(self, sig) -> list:
        """32-byte `bytes` keys, not tuples of Python ints: ~65 B vs ~340 B each,
        and byte-identical across processes and runs."""
        sig = np.asarray(sig, dtype=np.uint32)
        return [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def query_candidates(self, sig) -> set:
        cands: set = set()
        for b, key in enumerate(self._band_keys(sig)):
            cands.update(self.buckets[b].get(key, ()))
        return cands

    def insert(self, doc_idx: int, sig) -> None:
        for b, key in enumerate(self._band_keys(sig)):
            self.buckets[b].setdefault(key, []).append(doc_idx)


def estimate_jaccard(sig_a, sig_b) -> float:
    """Fraction of agreeing signature positions -- an unbiased estimator of the
    true Jaccard similarity (Broder, 1997)."""
    return float(np.mean(np.asarray(sig_a) == np.asarray(sig_b)))


def lsh_candidate_prob(jaccard: float, bands: int, rows: int) -> float:
    """P(at least one band collides) at true similarity J -- the LSH S-curve."""
    return 1.0 - (1.0 - jaccard ** rows) ** bands


def near_dedup_stream(docs: Iterable[dict], num_perm: int = 128, bands: int = 16,
                      threshold: float = 0.8, index_capacity: int = 500_000
                      ) -> Iterator[dict]:
    """Streaming near-dedup: yield documents that are not near-duplicates of an
    already-kept document. The corpus is never resident; memory is bounded by
    the index, measured at ~512 B/signature plus ~3.3 KB/document of LSH buckets
    (16 bands), i.e. ~1.9 GB at the 500k default.

    HARD CEILING: past `index_capacity` the index stops growing and near-dup
    recall for every later document silently drops to zero. We log loudly once.
    At 20B tokens (~20M documents) this ceiling is what forces the `datatrove`
    path -- do not just raise the number.
    """
    hasher = MinHasher(num_perm=num_perm)
    index = LSHIndex(num_perm=num_perm, bands=bands)
    store = SignatureStore(num_perm, index_capacity)
    warned = False
    for doc in docs:
        sig = hasher.signature(shingles(doc["text"]))
        if any(estimate_jaccard(sig, store[c]) >= threshold
               for c in index.query_candidates(sig)):
            continue
        if len(store) < index_capacity:
            index.insert(len(store), sig)
            store.append(sig)
        elif not warned:
            warned = True
            log.warning("near_dedup_stream: index_capacity=%d reached; near-dup "
                        "recall is now ZERO for the rest of this stream. Shard the "
                        "input or switch to datatrove's MinhashDedup* pipeline.",
                        index_capacity)
        yield doc


def near_dedup(docs, num_perm: int = 128, bands: int = 16,
               threshold: float = 0.8) -> list:
    """List-returning convenience wrapper around `near_dedup_stream`."""
    return list(near_dedup_stream(docs, num_perm=num_perm, bands=bands,
                                  threshold=threshold))
```

`near_dedup_stream` is a *generator*, not a function that takes a list: a `near_dedup(docs: list)` signature would need the whole corpus in RAM — ~84 GB of text for our budget — which is a hard wall, not a stylistic quibble. But "streaming means bounded memory" is often quoted as if bounded meant small, and it is worth being blunt about the constant. Measured on CPython 3.12, at `num_perm=128, bands=16`:

| Structure | Per document | At 500k docs |
|---|---:|---:|
| `exact_dedup`'s digest set | ~83 B | ~0.04 GB |
| `SignatureStore` rows (`uint32`) | 512 B | ~0.26 GB |
| LSH buckets (16 bands × `bytes` key + list) | ~3.3 KB | ~1.66 GB |

So the *buckets*, not the signatures, dominate — a ~1.9 GB index at the 500k default, and ~7.6 GB if you naively raise `index_capacity` to 2M. (With tuple signatures and tuple band keys, the same 2M index would be ~25 GB; that is the representation lesson, quantified.) A 20M-document corpus does not fit under any of these, and the failure is *silent*: once the store is full the code keeps yielding documents and simply stops detecting duplicates. Hence the loud one-shot log, and hence the production path below.

!!! warning "Common pitfall: MinHash band/row choice silently changes your recall"

    The `(bands, rows)` split determines the near-duplicate detection threshold, not just a performance knob. With `num_perm=128` and `bands=16` (so `rows=8`), the probability two documents at true Jaccard similarity $J$ are flagged as candidates is $1-(1-J^{8})^{16}$ — an S-curve whose steep transition sits near the classic LSH threshold approximation $(1/\text{bands})^{1/\text{rows}} = (1/16)^{1/8}\approx 0.71$ (numerically the curve climbs from $\approx0.06$ at $J=0.5$ to $\approx0.61$ at $J=0.7$ to $\approx0.95$ at $J=0.8$). Fewer, fatter bands (say `bands=8`, `rows=16`) push that threshold higher — to $(1/8)^{1/16}\approx 0.88$ — and catch fewer near-duplicates; more, thinner bands lower it, catching more but also more false positives. Candidate generation and the final decision are deliberately decoupled: the band structure over-generates candidates around $J\approx0.7$, and `near_dedup_stream`'s explicit `threshold=0.8` re-check on the full signature is what actually decides a drop. Always plot `lsh_candidate_prob` for your chosen `(bands, rows)` before trusting the dedup rate — a silently wrong threshold either wastes token budget on undetected duplicates or discards genuinely distinct documents that happen to share common phrasing.

{{fig:minhash-lsh-band-scurve}}

### Production path: running this at 20B tokens with datatrove

The code above exists to make MinHash *concrete*. It is not what you should run over 84 GB of text, and two pieces of arithmetic say why. The memory ceiling is one, measured above. The other is throughput: the vectorized `MinHasher.signature` measures ~8 ms on a ~5 KB document on one modern core, and FineWeb-Edu averages roughly a thousand tokens (~4 KB) per document, so 20B kept tokens is on the order of 20M documents:

$$
20\times10^{6}\ \text{docs}\times 8\ \text{ms} \approx 1.6\times10^{5}\ \text{s} \approx 44\ \text{core-hours},
$$

plus shingling, I/O, and the fact that you must sign *more* documents than you keep (dedup and filtering both reject). Single-threaded that is the better part of a core-week, it does not parallelize by itself, it does not checkpoint, and it does not survive a crash at hour 30.

The production tool for exactly this job is **[`datatrove`](https://github.com/huggingface/datatrove)** — HuggingFace's own pipeline library, the one FineWeb was actually built with. It expresses a corpus build as a list of pipeline blocks executed by a runner, and swapping `LocalPipelineExecutor` for `SlurmPipelineExecutor` is the only change needed to go from one box to a cluster:

```python
"""
capstone/scripts/dedup_datatrove.py  (production path -- needs `pip install datatrove[all]`)

MinHash deduplication of the Stack-100M corpus, using the same 4-stage pipeline
HuggingFace used for FineWeb. Not run in CI (heavy, multi-process, on-disk).
"""
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter,
)
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig

# FineWeb's settings: 5-grams, 112 permutations as 14 buckets x 8 hashes.
# The band/row S-curve of (1/14)^(1/8) ~ 0.72 targets documents ~75%+ similar --
# the same knob analyzed in the pitfall box above, just at production defaults.
cfg = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)

IN, WORK, OUT = "s3://.../filtered", "/scratch/minhash", "/scratch/deduped"
TASKS = 64  # one task per CPU core; SlurmPipelineExecutor scales this to a cluster

stage1 = LocalPipelineExecutor(
    pipeline=[JsonlReader(IN),
              MinhashDedupSignature(output_folder=f"{WORK}/signatures", config=cfg)],
    tasks=TASKS, logging_dir=f"{WORK}/logs/sig")

stage2 = LocalPipelineExecutor(          # one task per bucket: buckets are independent
    pipeline=[MinhashDedupBuckets(input_folder=f"{WORK}/signatures",
                                  output_folder=f"{WORK}/buckets", config=cfg)],
    tasks=cfg.num_buckets, logging_dir=f"{WORK}/logs/buckets", depends=stage1)

stage3 = LocalPipelineExecutor(          # union-find over all candidate pairs: single task
    pipeline=[MinhashDedupCluster(input_folder=f"{WORK}/buckets",
                                  output_folder=f"{WORK}/remove_ids", config=cfg)],
    tasks=1, logging_dir=f"{WORK}/logs/cluster", depends=stage2)

stage4 = LocalPipelineExecutor(          # re-read the corpus, drop the flagged ids
    pipeline=[JsonlReader(IN),
              MinhashDedupFilter(input_folder=f"{WORK}/remove_ids"),
              JsonlWriter(OUT)],
    tasks=TASKS, logging_dir=f"{WORK}/logs/filter", depends=stage3)

if __name__ == "__main__":
    stage4.run()   # `depends` chains the whole DAG; running the last stage runs all four
```

Three things to notice, because they are the reasons the from-scratch version cannot simply be scaled up. First, the work is **split into four stages with an on-disk hand-off**, so a crash costs one stage, not the whole run — the same fault-tolerance instinct as sharded checkpoints in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html). Second, signatures live **on disk, keyed by bucket**, so there is no in-RAM index to overflow and no silent recall cliff. Third, clustering is a **global** union-find over candidate pairs rather than our greedy "keep the first, drop later matches" rule; greedy streaming dedup is order-dependent (which member of a duplicate cluster survives depends on stream order), while union-find picks a canonical representative deterministically. At 20B tokens with `tasks=64`, expect the whole dedup to be on the order of an afternoon rather than a fortnight.

`datatrove` also supplies the filtering blocks our `filters.py` re-implements by hand — `GopherQualityFilter`, `GopherRepetitionFilter`, `C4QualityFilter`, `FineWebQualityFilter`, `LanguageFilter` (fastText language ID), `URLFilter` — plus `Trafilatura` for HTML extraction if you are starting from WARC files rather than a curated dataset. The alternatives worth knowing: **[`text-dedup`](https://github.com/ChenghaoMou/text-dedup)** (a focused collection of MinHash/SimHash/suffix-array dedup implementations), **NVIDIA NeMo Curator** (GPU-accelerated fuzzy dedup and classifier filtering), and AI2's **Dolma toolkit** (the pipeline behind the Dolma corpus). Use the from-scratch code in this chapter to understand what they do; use them to build your corpus.

## Tokenizing and Packing to 2048 with Document-Aware Attention

Two things happen at once in this stage: text becomes token IDs, and token IDs from many short documents get concatenated into fixed-length windows of `SEQ_LEN=2048` (`Stack-100M`'s pretraining context, per `capstone/PLAN.md` §1) — because training on ragged, individually-padded sequences wastes enormous compute on pad tokens when the median FineWeb-Edu document is far shorter than 2048 tokens. Packing multiple documents into one window recovers that compute, but naive concatenation lets attention flow *across* document boundaries: token 40 of document B would attend to token 2000 of unrelated document A. So packing has one non-negotiable requirement and one bookkeeping convention:

1. **Document-aware attention masking (non-negotiable).** A token may only attend to earlier tokens *within its own document* — never across a packed boundary.
2. **Position-id reset (bookkeeping).** Every document's position ids restart at 0, regardless of where in the packed window it lands.

It is worth being precise about why (1) is load-bearing and (2) is not — the folk explanation, "otherwise RoPE positions alias across documents", is wrong for a RoPE model. `Stack100M.forward` *does* consume `position_ids`: it indexes the rotary tables with them (`cos = self.rope_cos[position_ids]`, Ch. 14.4 — positions are never hard-coded to `arange(T)`), so the dataset must supply the array, and every value in it must lie in $[0, \texttt{max\_seq\_len})$. But RoPE encodes *relative* offsets: the attention logit between positions $i$ and $j$ depends only on $i-j$ ([Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)). Shifting a whole segment's clock by a constant therefore leaves every within-segment $i-j$ — and hence every logit inside that segment — unchanged. Given a correct block-diagonal mask, whether document B's clock restarts at 0 or continues from A's is invisible to the model. That shift-invariance is what will let us *derive* the position array from tokens instead of storing it; it is not a licence to omit it. The reset does genuinely matter for (a) *learned* or absolute positional encodings, where the embedding looked up is the absolute index; (b) length-generalization bookkeeping, where you want the true within-document offset; and (c) as a convenient boundary signal. Cross-document *attention* is the actual bug, and the mask is what fixes it.

The tokenizer itself (byte-level BPE, `vocab_size=32768`) is trained from scratch in the next chapter; see [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) for the deeper mechanics of BPE training. Here we need only its *interface* — `encode`, plus `bos_id`/`eos_id`/`pad_id` — so this chapter's code declares that interface as a `Protocol`.

```python
"""
capstone/stacklm/data/pack.py

Pack tokenized documents into fixed-length SEQ_LEN=2048 windows with
document-aware segmentation. Every packed chunk begins with <bos>, so the token
array alone carries the document boundaries. Position ids are DERIVED, never
stored -- see `segments_from_bos`. The same "pack without cross-contamination"
recipe reappears for instruction data in Chat Templates, Data Formatting &
Sequence Packing (Ch. 5.2).
"""
from typing import Iterable, Iterator, Protocol
import numpy as np

SEQ_LEN = 2048  # Stack-100M pretraining max_seq_len (capstone/PLAN.md sec. 1)


class Tokenizer(Protocol):
    bos_id: int
    eos_id: int
    pad_id: int
    def encode(self, text: str) -> list: ...


def pack_documents(docs: Iterable[dict], tokenizer, seq_len: int = SEQ_LEN) -> Iterator[tuple]:
    """Greedily concatenate `<bos> body <eos>` across documents into fixed-length
    windows. Yields (input_ids, position_ids) per window; the final partial window
    is padded so no tokens are silently dropped. A document dict may carry
    pre-computed `ids` (the corpus builder tokenizes once, for budgeting).

    Documents longer than the window are chunked into seq_len-sized pieces, each
    treated as its own "document" for position-reset purposes -- keeping every
    position id inside [0, seq_len), which is both the range RoPE's theta=10000
    base (capstone/PLAN.md sec. 1) is tuned for and the valid index range of the
    model's precomputed rope_cos/rope_sin tables.
    """
    buf_ids: list = []
    buf_pos: list = []
    max_body = seq_len - 2  # room for <bos> and <eos> in every chunk

    for doc in docs:
        raw = doc["ids"] if "ids" in doc else tokenizer.encode(doc["text"])
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


def segments_from_bos(input_ids: np.ndarray, bos_id: int) -> tuple:
    """Derive (seq_ids, position_ids) from a packed window's tokens alone.

    `seq_ids[i]` is the index of the document token i belongs to (-1 for a
    leading fragment continued from the previous window); `position_ids[i]` is
    the offset of token i inside its document. Storing these on disk would
    double the corpus for information the token array already contains.
    """
    starts = input_ids == bos_id
    seq_ids = np.cumsum(starts) - 1                       # -1 for a leading tail
    idx = np.arange(input_ids.shape[0])
    seg_start = np.maximum.accumulate(np.where(starts, idx, -1))
    return seq_ids, idx - np.maximum(seg_start, 0)


def build_intra_doc_causal_mask(position_ids: np.ndarray) -> np.ndarray:
    """Reconstruct the boolean (seq_len, seq_len) attention mask -- True means
    "token i may attend to token j". Two conditions: (a) causal, j <= i;
    (b) same document, i.e. i and j fall in the same position-reset segment.

    In production this dense mask is replaced by FlashAttention's `cu_seqlens`
    varlen API for O(seq_len) memory instead of O(seq_len^2) -- see
    FlashAttention I: IO-Awareness & The Online Softmax (Ch. 4.2) and Exercise 2.
    This dense version is for teaching, unit tests, and small-scale CPU eval.
    """
    seq_len = position_ids.shape[0]
    doc_id = np.cumsum(position_ids == 0)  # monotonically increasing per document
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    same_doc = doc_id[:, None] == doc_id[None, :]
    return causal & same_doc


def segment_ids_from_positions(position_ids: np.ndarray) -> np.ndarray:
    """The per-token segment id (0,1,2,...) that `Stack100M.forward(seq_ids=...)`
    consumes to build its (B, 1, T, T) block-diagonal mask."""
    return np.cumsum(position_ids == 0) - 1
```

`segment_ids_from_positions` is the function the model actually calls into: `Stack100M.forward(idx, targets, position_ids, seq_ids)` compares `seq_ids[:, :, None] == seq_ids[:, None, :]`, intersects that with a causal `tril`, and passes the result to scaled-dot-product attention. Everything in this chapter's packing machinery exists to produce that one integer array (and the position array that indexes the rotary tables).

!!! example "Packing three short documents into one window"

    Suppose `seq_len=8` (tiny, for illustration) and three documents tokenize to 3, 2, and 5 body tokens respectively (with 1-token `bos`/`eos`, so 5, 4, and 7 tokens including specials). Packing greedily:

    ```text
    doc A (5 tok): [bos a1 a2 a3 eos]       positions [0 1 2 3 4]
    doc B (4 tok): [bos b1 b2 eos]          positions [0 1 2 3]
    doc C (7 tok): [bos c1 c2 c3 c4 c5 eos] positions [0 1 2 3 4 5 6]
    ```

    Concatenated: `[bos a1 a2 a3 eos bos b1 b2 eos bos c1 c2 c3 c4 c5 eos]` (16 tokens), positions `[0 1 2 3 4 0 1 2 3 0 1 2 3 4 5 6]`. Two `seq_len=8` windows come out of `pack_documents`:

    - Window 1: tokens `[bos a1 a2 a3 eos bos b1 b2]`, positions `[0 1 2 3 4 0 1 2]`
    - Window 2: tokens `[eos bos c1 c2 c3 c4 c5 eos]`, positions `[3 0 1 2 3 4 5 6]`

    In window 1, `doc_id = cumsum(position == 0) = [1 1 1 1 1 2 2 2]` — token 5 (`bos` of B) starts segment 2. Token 7 (`b2`, position 2) may attend to tokens 5 and 6 (B's own `bos`/`b1`) but **not** to tokens 0–4 (all of document A), even though they sit earlier in the same physical window. That is the entire mechanism: one `cumsum` over a boolean array recovers exact document boundaries.

    Now run `segments_from_bos` on window 2's *tokens*, with no stored positions: `starts = [F T F F F F F F]`, so `seq_ids = cumsum(starts) - 1 = [-1 0 0 0 0 0 0 0]` and `position_ids = [0 0 1 2 3 4 5 6]`. The leading `eos` is correctly isolated as its own segment (`-1`, the tail of document B carried over from window 1), and C's clock restarts at its `bos`. The only difference from the stored version is that B's tail restarts at 0 instead of continuing at 3 — a constant shift of one segment's clock, which a relative encoding cannot see. This is why the `.pos.bin` array is redundant.

{{fig:doc-aware-packing-mask}}

## Sharding to uint16 memmap Files and a Streaming Dataset

The last storage step turns a stream of packed windows into files a `torch.utils.data.Dataset` can read without ever loading the corpus into RAM. We use the same trick as nanoGPT / llm.c: flat binary files, memory-mapped with `np.memmap`, no serialization format to parse. `vocab_size=32768` fits comfortably under `uint16`'s 65,535 ceiling, so every token is 2 bytes — half of `int32`, doubling effective disk and page-cache bandwidth for free.

**We store only the token array.** A `.pos.bin` companion would be a 100% storage and read-bandwidth overhead for information `segments_from_bos` reconstructs in two numpy ops — 40 GB saved on a 20B-token corpus, and 40 GB less page cache competing with the training working set. `store_positions=True` remains available for debugging and inspection.

Alongside the shards we write a `manifest.json`. That file is not decoration: it carries `bos_id` (without which the shards are uninterpretable), the realized data mix, the dataset revisions, the filter-config hash, and the seed — precisely the contents of Ch. 14.12's reproducibility checklist.

```python
"""
capstone/stacklm/data/shard.py

One shard = `shard_XXXXX.tokens.bin` (flat uint16, n_seq x seq_len) plus a tiny
`.meta.bin` holding [n_seq, seq_len]. Position ids are NOT stored: they are
derived on read from `input_ids == bos_id`, halving the corpus on disk and in
page cache.
"""
import json
import numpy as np
from pathlib import Path

DTYPE = np.uint16  # vocab_size=32768 fits; see worked example


class ShardWriter:
    """Buffers packed windows and flushes a new shard every `seqs_per_shard`
    sequences (default target: ~100M tokens/shard, see the worked example)."""

    def __init__(self, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000,
                 store_positions: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.seqs_per_shard = max(1, tokens_per_shard // seq_len)
        self.store_positions = store_positions
        self._buf_ids: list = []
        self._buf_pos: list = []
        self._shard_idx = 0
        self.n_sequences = 0

    def add(self, input_ids, position_ids=None) -> None:
        self._buf_ids.append(np.asarray(input_ids, dtype=DTYPE))
        if self.store_positions:
            self._buf_pos.append(np.asarray(position_ids, dtype=DTYPE))
        self.n_sequences += 1
        if len(self._buf_ids) >= self.seqs_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_ids:
            return
        ids = np.stack(self._buf_ids)  # (n_seq, seq_len)
        stem = str(self.out_dir / f"shard_{self._shard_idx:05d}")
        ids.tofile(stem + ".tokens.bin")
        if self.store_positions:
            np.stack(self._buf_pos).tofile(stem + ".pos.bin")
        np.array([ids.shape[0], ids.shape[1]], dtype=np.int64).tofile(stem + ".meta.bin")
        self._shard_idx += 1
        self._buf_ids.clear()
        self._buf_pos.clear()

    def close(self) -> None:
        self._flush()  # flush the trailing partial shard

    def write_manifest(self, tokenizer=None, extra: dict = None) -> dict:
        """Write manifest.json beside the shards. `bos_id` is the load-bearing
        field: the dataset needs it to recover document boundaries on read."""
        man = {
            "seq_len": self.seq_len,
            "n_shards": self._shard_idx,
            "n_sequences": self.n_sequences,
            "n_tokens": self.n_sequences * self.seq_len,
            "store_positions": self.store_positions,
        }
        if tokenizer is not None:
            man.update(bos_id=int(tokenizer.bos_id), eos_id=int(tokenizer.eos_id),
                       pad_id=int(tokenizer.pad_id))
        if extra:
            man.update(extra)
        (self.out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        return man


def build_shards(docs, tokenizer, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000,
                 store_positions: bool = False, manifest_extra: dict = None) -> int:
    """End-to-end: pack a document stream, write shards + manifest. Returns the
    number of shards written."""
    from .pack import pack_documents
    writer = ShardWriter(out_dir, seq_len=seq_len, tokens_per_shard=tokens_per_shard,
                         store_positions=store_positions)
    for input_ids, position_ids in pack_documents(docs, tokenizer, seq_len=seq_len):
        writer.add(input_ids, position_ids)
    writer.close()
    writer.write_manifest(tokenizer=tokenizer, extra=manifest_extra)
    return writer._shard_idx
```

At train time, `PackedMemmapDataset` maps every shard's `.tokens.bin` straight into address space. No shard is ever fully resident in RAM — the OS page cache serves whatever windows the dataloader touches, which is what makes a 40 GB corpus trainable on a machine with far less than 40 GB of RAM. Note the exact keys it returns: `seq_ids` is the one the training loop forwards to the model for masking, `position_ids` is the one that indexes the rotary tables, and a dataset that omits either produces a `KeyError` in Ch. 14.7.

```python
"""
capstone/stacklm/data/dataset.py

torch Dataset over the sharded uint16 .bin files. Only tokens are stored;
`seq_ids` (the segment id `Stack100M.forward` consumes for document-aware
masking) and `position_ids` (which index its RoPE tables) are derived per item
from `input_ids == bos_id`.
"""
import bisect
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .pack import segments_from_bos


class PackedMemmapDataset(Dataset):
    def __init__(self, shard_dir: str, bos_id: int = None):
        self.shard_dir = Path(shard_dir)
        self._shards = []       # list of (tokens_memmap, pos_memmap_or_None)
        self._cum_seqs = [0]    # prefix sums of sequence counts, for indexing
        self.seq_len = None     # None, not a loop variable: an empty dir must not crash

        man_path = self.shard_dir / "manifest.json"
        self.manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
        self.bos_id = bos_id if bos_id is not None else self.manifest.get("bos_id")

        for meta_path in sorted(self.shard_dir.glob("shard_*.meta.bin")):
            n_seq, seq_len = (int(x) for x in np.fromfile(meta_path, dtype=np.int64))
            stem = str(meta_path)[: -len(".meta.bin")]
            tok_mm = np.memmap(stem + ".tokens.bin", dtype=np.uint16, mode="r",
                               shape=(n_seq, seq_len))
            pos_path = Path(stem + ".pos.bin")
            pos_mm = (np.memmap(pos_path, dtype=np.uint16, mode="r", shape=(n_seq, seq_len))
                      if pos_path.exists() else None)
            self._shards.append((tok_mm, pos_mm))
            self._cum_seqs.append(self._cum_seqs[-1] + n_seq)
            self.seq_len = seq_len

        if self._shards and self._shards[0][1] is None and self.bos_id is None:
            raise ValueError(
                f"{shard_dir}: no .pos.bin and no bos_id (manifest.json missing?). "
                "Pass PackedMemmapDataset(dir, bos_id=tok.bos_id)."
            )

    def __len__(self) -> int:
        return self._cum_seqs[-1]

    def _locate(self, idx: int) -> tuple:
        """(shard, row) for a global sequence index. Binary search, not a linear
        scan: with 200 shards and ~10M samples/epoch a scan would cost billions
        of pointless Python comparisons per epoch."""
        if not 0 <= idx < self._cum_seqs[-1]:
            raise IndexError(idx)
        s = bisect.bisect_right(self._cum_seqs, idx) - 1
        return s, idx - self._cum_seqs[s]

    def __getitem__(self, idx: int) -> dict:
        s, row = self._locate(idx)
        tok_mm, pos_mm = self._shards[s]
        ids_np = tok_mm[row].astype(np.int64)
        if pos_mm is not None:                        # legacy shards with .pos.bin
            pos_np = pos_mm[row].astype(np.int64)
            seq_np = np.cumsum(pos_np == 0) - 1
        else:                                         # derive from the tokens alone
            seq_np, pos_np = segments_from_bos(ids_np, self.bos_id)
        ids = torch.from_numpy(ids_np)
        # Standard next-token target: shift by one. The training loop (Ch. 14.7)
        # masks the loss on pad targets, so the trailing padded window costs
        # nothing beyond the (small) wasted compute of its forward pass.
        return {
            "input_ids": ids[:-1],
            "position_ids": torch.from_numpy(np.ascontiguousarray(pos_np))[:-1],
            "seq_ids": torch.from_numpy(np.ascontiguousarray(seq_np))[:-1],
            "targets": ids[1:],
        }
```

## The Driver: Budgets, Interleaving, Held-Out Split, Manifest

Everything above is a stage. Something has to *compose* them into a 20B-token corpus that actually honours the 70/15/10/5 mix, and that something is easy to get catastrophically wrong in two specific ways:

- **No budget enforcement.** Streaming a source to exhaustion and moving to the next produces 14B FineWeb-Edu tokens, then 3B Cosmopedia, then 2B code, then 1B math. The model would see zero code for the first 85% of training and then a sudden pure-code phase at the end — an accidental curriculum nobody designed, and a reliable way to blow up the loss late in a run.
- **No shuffle.** Shards are written in stream order, so shard 0 would be pure FineWeb-Edu. Any resume-from-shard-*k* restart, any "evaluate on the first 1% of shards" habit, and any data-parallel rank assignment then sees a biased slice of the mix.

`build_corpus.py` fixes both, plus emits the held-out split and the manifest:

```python
"""
capstone/stacklm/data/build_corpus.py

The corpus driver: turn the four sources into sharded, packed `train/` + `val/`
corpora honouring the 70/15/10/5 mix and a total token budget.

Stages, per source: stream -> quality filter -> exact dedup -> near dedup ->
tokenize (once, in batches) -> stop at that source's token budget. The four
streams are then interleaved by weight, pushed through a reservoir shuffle
buffer, split into a deterministic document-level held-out set, packed, and
written as uint16 shards with a `manifest.json` recording the realized mix.
"""
import hashlib
import json
import random
from pathlib import Path

from .dedup import exact_dedup, near_dedup_stream
from .filters import filter_config_hash, quality_filter
from .pack import pack_documents
from .shard import ShardWriter
from .synthetic import STACK100M_MIX, TOTAL_TOKEN_BUDGET, stream_source


def encode_batched(docs, tokenizer, batch: int = 1024):
    """Attach `ids` to each document, encoding in batches.

    Every fast encoder in the ecosystem is a BATCH API -- Ch. 14.3's
    `encode_corpus` (multiprocessing.Pool over documents), HF `tokenizers`'
    `encode_batch`, `tiktoken`'s `encode_ordinary_batch`. Calling `encode` one
    document at a time leaves the process-parallel speedup on the table, and at
    84 GB of text that is ~4 core-hours instead of ~35 minutes.
    """
    encode_batch = getattr(tokenizer, "encode_batch", None)
    buf = []

    def flush():
        texts = [d["text"] for d in buf]
        id_lists = (encode_batch(texts) if encode_batch is not None
                    else [tokenizer.encode(t) for t in texts])
        for d, ids in zip(buf, id_lists):
            yield {**d, "ids": list(ids)}
        buf.clear()

    for doc in docs:
        buf.append(doc)
        if len(buf) >= batch:
            yield from flush()
    if buf:
        yield from flush()


def _source_pipeline(entry, tokenizer, budget_tokens, offline, dedup_kwargs, stats,
                     encode_batch_size: int = 1024):
    """Filtered, deduplicated, tokenized documents from one source, capped at
    `budget_tokens`. Tokenizing here (once) is what makes the budget exact."""
    docs = stream_source(entry, offline=offline)
    docs = (d for d in docs if quality_filter(d))
    docs = exact_dedup(docs)
    docs = near_dedup_stream(docs, **dedup_kwargs)
    used = 0
    for doc in encode_batched(docs, tokenizer, batch=encode_batch_size):
        if not doc["ids"]:
            continue
        used += len(doc["ids"]) + 2              # +2 for <bos>/<eos> added at pack time
        stats[entry.name] = used
        yield doc
        if used >= budget_tokens:
            return


def interleave_budgeted(entries, tokenizer, total_tokens, offline=True,
                        seed=1337, dedup_kwargs=None, stats=None):
    """Weighted round-robin over the per-source pipelines. Each source gets
    `weight * total_tokens`; a source that runs dry is dropped and the remaining
    weights renormalize -- deliberately UNLIKE
    `datasets.interleave_datasets(stopping_strategy=...)`, whose
    `"first_exhausted"` truncates the whole mix and whose `"all_exhausted"`
    oversamples the drained source. See the comparison at the end of this
    section."""
    stats = {} if stats is None else stats
    dedup_kwargs = dedup_kwargs or {}
    rng = random.Random(seed)
    gens, weights = {}, {}
    for e in entries:
        budget = int(round(e.weight * total_tokens))
        gens[e.name] = _source_pipeline(e, tokenizer, budget, offline, dedup_kwargs, stats)
        weights[e.name] = e.weight
        stats.setdefault(e.name, 0)
    alive = list(gens)
    while alive:
        name = rng.choices(alive, weights=[weights[n] for n in alive], k=1)[0]
        try:
            yield next(gens[name])
        except StopIteration:
            alive.remove(name)


def shuffle_buffer(docs, size=100_000, seed=1337):
    """Reservoir shuffle: without it, shard 0 would be ordered exactly as the
    sources were interleaved, and any resume-from-shard-k restart would see a
    biased slice of the mix."""
    rng = random.Random(seed)
    buf = []
    for doc in docs:
        if len(buf) < size:
            buf.append(doc)
            continue
        j = rng.randrange(size)
        yield buf[j]
        buf[j] = doc
    rng.shuffle(buf)
    yield from buf


def is_holdout(doc, per_mille: int = 1) -> bool:
    """Deterministic document-level held-out assignment. Hashing the *text*
    (not a counter) means the same document lands in the same split on every
    rebuild, and a document is never split across train and val -- the failure
    mode that silently contaminates a held-out perplexity."""
    h = int(hashlib.blake2b(doc["text"].encode("utf-8"), digest_size=8).hexdigest(), 16)
    return (h % 1000) < per_mille


def build_corpus(out_dir, tokenizer, total_tokens=TOTAL_TOKEN_BUDGET, entries=None,
                 seq_len=2048, tokens_per_shard=100_000_000, offline=True,
                 holdout_per_mille=1, holdout_tokens=10_000_000,
                 shuffle_size=100_000, seed=1337, dedup_kwargs=None) -> dict:
    """Build `out_dir/train` and `out_dir/val` shards. Returns the manifest."""
    entries = entries if entries is not None else STACK100M_MIX
    assert abs(sum(e.weight for e in entries) - 1.0) < 1e-9, "mix weights must sum to 1"
    out = Path(out_dir)
    stats: dict = {}
    val_docs: list = []
    held = {"tokens": 0}

    docs = interleave_budgeted(entries, tokenizer, total_tokens, offline=offline,
                               seed=seed, dedup_kwargs=dedup_kwargs, stats=stats)
    docs = shuffle_buffer(docs, size=shuffle_size, seed=seed)

    def train_stream():
        for doc in docs:
            if is_holdout(doc, holdout_per_mille):
                if held["tokens"] < holdout_tokens:
                    val_docs.append(doc)
                    held["tokens"] += len(doc["ids"]) + 2
                continue                     # held-out docs NEVER enter training
            yield doc

    train_writer = ShardWriter(out / "train", seq_len=seq_len,
                               tokens_per_shard=tokens_per_shard)
    for ids, pos in pack_documents(train_stream(), tokenizer, seq_len=seq_len):
        train_writer.add(ids, pos)
    train_writer.close()

    val_writer = ShardWriter(out / "val", seq_len=seq_len,
                             tokens_per_shard=tokens_per_shard)
    for ids, pos in pack_documents(iter(val_docs), tokenizer, seq_len=seq_len):
        val_writer.add(ids, pos)
    val_writer.close()

    realized = sum(stats.values()) or 1
    provenance = {
        "seed": seed,
        "token_budget": total_tokens,
        "filter_config_hash": filter_config_hash(),
        "dedup": {"num_perm": 128, "bands": 16, "threshold": 0.8, **(dedup_kwargs or {})},
        "sources": [
            {"name": e.name, "hf_path": e.hf_path, "hf_config": e.hf_config,
             "hf_data_dir": e.hf_data_dir, "revision": e.revision,
             "target_weight": e.weight,
             "realized_tokens": stats.get(e.name, 0),
             "realized_weight": round(stats.get(e.name, 0) / realized, 4)}
            for e in entries
        ],
        "holdout": {"per_mille": holdout_per_mille, "tokens": held["tokens"]},
        "offline_synthetic": offline,
    }
    man = train_writer.write_manifest(tokenizer=tokenizer, extra=provenance)
    val_writer.write_manifest(tokenizer=tokenizer, extra=provenance)
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    return man
```

Three details are worth dwelling on.

**Tokenize once, and in batches.** The budget is in *tokens*, so the driver must know each document's token count; tokenizing in `_source_pipeline` and carrying `ids` forward (which `pack_documents` picks up) avoids encoding the whole corpus twice. Batching matters as much as not repeating: a serial `encode`-per-document pass over the ~84 GB this budget implies runs at Ch. 14.3's measured ~5.8 MB/s, i.e. **≈4 core-hours**, while its `encode_corpus` with `multiprocessing.Pool(16)` measures ~39.8 MB/s, i.e. **≈35 minutes**. Run that pass on a cheap CPU box *before* you rent the GPU — the `.bin` shards are the hand-off, and the pretraining loop should never call `encode` at all.

**Held-out by document hash, not by slicing.** Taking "the last 1% of shards" as validation is the standard way to contaminate an eval set: packed windows straddle document boundaries, so the same document can appear in both splits. Hashing the document text routes a document *atomically* to exactly one split, deterministically, on every rebuild — and any hashed-to-holdout document beyond the `holdout_tokens` cap is dropped entirely rather than falling back into training. See [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html) and the contamination discussion in Ch. 3.2 for why this matters more than it looks.

**Realized vs target mix.** `stats` records what each source actually contributed. Sources run dry (StarCoder's Python subset is finite), filters reject at different rates per domain, and the realized weights will not exactly equal the targets. Printing the realized mix — and storing it in the manifest — turns "we trained on 10% code" from an assumption into a measurement.

In production, the interleave step also has a first-class library implementation. If you are already holding `datasets.IterableDataset` objects, this is the exact right call:

```python
from datasets import interleave_datasets, load_dataset

streams = [load_dataset(e.hf_path, name=e.hf_config, data_dir=e.hf_data_dir,
                        split="train", streaming=True, revision=e.revision)
           for e in STACK100M_MIX]

mixed = interleave_datasets(
    streams,
    probabilities=[e.weight for e in STACK100M_MIX],   # 0.70 / 0.15 / 0.10 / 0.05
    seed=1337,
    stopping_strategy="all_exhausted",  # keep sampling until every source is drained
).shuffle(seed=1337, buffer_size=100_000)              # streaming reservoir shuffle
```

`stopping_strategy="first_exhausted"` (the default) stops as soon as *any* source is drained — which, with a 5% math weight over a small math corpus, would truncate the whole mix early. `"all_exhausted"` instead re-cycles exhausted sources until every source has been drained at least once, so use it deliberately: it means repeated math data rather than a short corpus, and repeats interact badly with the dedup you just ran. Our hand-rolled driver takes the third option — drop the drained source and renormalize the remaining weights — which is why it reports the realized mix rather than promising the target one.

### Running the whole pipeline offline, in ten seconds

Nothing above needs a network, a GPU, or the trained tokenizer to execute. The real byte-level BPE arrives in Ch. 14.3; until then a raw-bytes stand-in implements the same `Tokenizer` protocol, and the pipeline runs end to end at toy scale — which is exactly what the book's CI does:

```python
"""
Toy end-to-end run of the Ch. 14.2 pipeline. No network, no GPU, ~10 seconds.
Swap ByteTokenizer for `stacklm.tokenizer.load_tokenizer()` for a real corpus.
"""
import tempfile
from stacklm.data import build_corpus, PackedMemmapDataset, STACK100M_MIX


class ByteTokenizer:
    """Dependency-free stand-in for Ch. 14.3's BPE tokenizer (vocab_size=32768).
    NOT what Stack-100M trains with -- it exists so this chapter's code runs
    before that artifact exists. It has no `encode_batch`, so `encode_batched`
    transparently falls back to the per-document path."""
    bos_id, eos_id, pad_id = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))


tok = ByteTokenizer()
out = tempfile.mkdtemp(prefix="stack100m_toy_")
man = build_corpus(out, tok,
                   total_tokens=200_000,      # 20B in the real run
                   seq_len=128,               # 2048 in the real run
                   tokens_per_shard=128 * 64,
                   offline=True,              # synthetic sources, hermetic
                   holdout_per_mille=20, holdout_tokens=20_000,
                   shuffle_size=256)

train = PackedMemmapDataset(f"{out}/train")   # bos_id comes from manifest.json
val = PackedMemmapDataset(f"{out}/val")
batch = train[0]

print(f"{man['n_shards']} shard(s), {len(train)} train seqs, {len(val)} val seqs")
print("realized mix:", {s["name"]: s["realized_weight"] for s in man["sources"]})
print("batch keys:", sorted(batch.keys()))    # input_ids, position_ids, seq_ids, targets

assert abs(sum(e.weight for e in STACK100M_MIX) - 1.0) < 1e-9
assert batch["input_ids"].shape == batch["targets"].shape == (man["seq_len"] - 1,)
assert int(batch["position_ids"].max()) < man["seq_len"]   # positions stay in range
assert int(batch["seq_ids"].max()) >= 0                    # at least one document
```

`seq_ids` is the key that matters downstream: Ch. 14.7's training loop reads `batch["seq_ids"]` and passes it to `Stack100M.forward(..., seq_ids=seq_ids)`. A dataset that returns everything *except* `seq_ids` will train — with cross-document attention silently enabled, and a loss curve just plausible enough that you will not notice.

## Worked Example: Token, Byte, and Shard Accounting

!!! example "Sizing the full 20B-token Stack-100M corpus"

    **Mix breakdown.** 20B tokens split 70/15/10/5 gives exactly 14.0B (FineWeb-Edu), 3.0B (Cosmopedia v2), 2.0B (StarCoder), and 1.0B (FineMath) tokens — the weights were chosen to divide the budget cleanly.

    **Raw text volume.** Ch. 14.3 *measures* the `Stack-100M` byte-level BPE at **4.196 bytes/token** on held-out prose, so the budget in bytes is

    $$
    20\times10^{9}\ \text{tokens} \times 4.196\ \tfrac{\text{bytes}}{\text{token}} \approx 8.4\times10^{10}\ \text{bytes} \approx 84\ \text{GB}
    $$

    of *kept* raw UTF-8 text — on the order of 20M documents at FineWeb-Edu's average length. That ratio is domain-dependent (Ch. 14.3 measures ~3.89 bytes/token on Python and ~4.84 on plain English), so the 10% StarCoder slice pulls the corpus a little under 84 GB in practice; use 84 GB as the planning figure. The raw volume you must *stream and filter* to end up with 84 GB of kept text is substantially larger: quality filtering and deduplication routinely discard the large majority of raw Common Crawl. That is exactly why sourcing is a streaming pass over a much bigger corpus, not a one-shot download.

    **Sequence count.** At `SEQ_LEN=2048`, the token budget divides *exactly*:

    $$
    \frac{20\times10^{9}}{2048} = 9{,}765{,}625 \text{ packed sequences.}
    $$

    **Shard sizes.** Each `uint16` token occupies 2 bytes, so the packed corpus is $20\times10^{9}\times 2 = 4.0\times10^{10}$ bytes = **40 GB** — tokens only; the derived positions cost 0 bytes on disk and two numpy ops per item on read. Sharding at ~100M tokens/shard gives `seqs_per_shard = 100_000_000 // 2048 = 48{,}828` sequences/shard (≈99.99M tokens after flooring), so the corpus splits into

    $$
    \left\lceil \frac{9{,}765{,}625}{48{,}828} \right\rceil = 200 \text{ shards of } \approx 200\ \text{MB each.}
    $$

    200 shards of manageable size are easy to distribute, resume, and spot-check individually — a corruption in one shard costs you 0.5% of the run, not the whole thing (see [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html) for the training-time half of this story).

    **CPU budget for the build.** Tokenizing 84 GB is ~35 minutes on 16 cores (Ch. 14.3); MinHashing ~20M documents is ~44 core-hours single-threaded, which is why dedup goes through `datatrove` with `tasks=64`. Neither of these should overlap with GPU rental — the whole point of the `.bin` hand-off is that data prep is a CPU job and training is a GPU job. Section 1 has the GPU side of the budget.

## Key Takeaways & Further Reading

!!! key "Key Takeaways"

    - `Stack-100M` deliberately over-trains: ~20B tokens against ~101.4M parameters is ≈200 tokens/param, roughly 10× Chinchilla's compute-optimal ≈2B — a deployment-economics bet, because inference cost scales with $N$ forever while extra pretraining tokens are a one-time cost. The bet inverts if the model is served rarely.
    - $C = 6ND$ undercounts this shape. Attention costs $\approx 6\,n_{\text{layers}}\,n_{\text{ctx}}\,d_{\text{model}}$ FLOPs/token — 31% on top of $6N$ at $d_{\text{model}}=512$, $n_{\text{ctx}}=2048$ — putting the real budget near $1.6\times10^{19}$ FLOPs, i.e. ≈24 A100-hours at the 58% MFU Ch. 14.7 measures (the same run reports 44.5% under the $6ND$ convention). Always state which convention you are quoting.
    - A dataset id is not a loading recipe: multi-config repos need `name=`, StarCoder's text column is `content` (not `text`), Cosmopedia **v2** lives in `HuggingFaceTB/smollm-corpus`, and gated repos need `huggingface_hub.login()`. Assert on non-empty output; a silently empty source is the worst failure mode in a data pipeline.
    - Dedup is two-stage: cheap streaming exact-hash catches mirrors and boilerplate; MinHash + LSH banding catches near-duplicates in near-linear time by estimating Jaccard from signature agreement (Broder, 1997). The `(bands, rows)` split *is* the similarity threshold, via the $(1/b)^{1/r}$ S-curve.
    - Representation decides feasibility: a signature as a tuple of 128 Python ints is ~4.9 KB, as a `uint32` row it is 512 B; a streaming LSH index costs ~1.9 GB at 500k documents and its buckets — not its signatures — dominate. Past `index_capacity` recall silently drops to zero, which is precisely why the 20M-document corpus goes through `datatrove`'s four-stage `MinhashDedupSignature → Buckets → Cluster → Filter` pipeline: it spills to disk, checkpoints between stages, and clusters globally instead of greedily.
    - A corpus needs a *driver*, not just stages: per-source token budgets, weighted interleaving (so code appears from step 0), batched tokenization (~35 min on 16 cores vs ~4 core-hours serially), a reservoir shuffle before sharding, a document-hash held-out split that can never straddle a boundary, and a `manifest.json` recording realized mix, revisions, filter hash, and seed.
    - Packing requires document-aware attention masking; the per-document position reset is *bookkeeping*. The model does index its RoPE tables with `position_ids`, but RoPE is relative, so shifting one segment's clock by a constant changes nothing given a correct block-diagonal mask. Cross-document attention is the actual bug.
    - Store tokens only. `seq_ids` and `position_ids` are recovered from `input_ids == bos_id` with a `cumsum` and a `maximum.accumulate`, saving 40 GB on a 20B-token corpus and the page cache to match — `uint16` memmap shards of ~200 MB, 200 of them, served by the OS page cache rather than application RAM.

**Further reading**

- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla"), 2022 — the $D^\*\approx 20N$ result this chapter deliberately trains past.
- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 — the appendix where the $6N + 12\,n_{\text{layers}}\,n_{\text{ctx}}\,d_{\text{model}}$ per-token FLOP accounting comes from.
- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, HuggingFace, 2024 — FineWeb and FineWeb-Edu, and the MinHash settings the production pipeline above reuses.
- Ben Allal et al., *SmolLM2*, HuggingFace — the small-model, high-quality-mix recipe this capstone's data mix follows, and the origin of Cosmopedia v2 / `smollm-corpus`.
- Li et al. and the BigCode community, *StarCoder: May the Source Be With You!*, 2023; and the StarCoder2 / The Stack v2 follow-up.
- Broder, *On the Resemblance and Containment of Documents*, 1997 — the MinHash resemblance-estimation technique used for near-dedup.
- Lee et al., *Deduplicating Training Data Makes Language Models Better*, 2022 — the empirical case for aggressive corpus deduplication.
- Karpathy, nanoGPT and llm.c — the flat `uint16` memmap `.bin` sharding convention `ShardWriter`/`PackedMemmapDataset` follow.
- [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) and [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) — the full theory behind the lean, at-scale versions used here.

## Exercises

**1.** (Quantitative) Using $N = 101.4\text{M}$, $n_{\text{layers}}=30$, $n_{\text{ctx}}=2048$, $d_{\text{model}}=512$, and $D=20\text{B}$: (a) verify the budget is ≈200 tokens/param and ≈10× Chinchilla-optimal; (b) compute total training FLOPs with the naive $6ND$ rule *and* with the attention term; (c) Ch. 14.7 measures 227,951 tokens/s on one A100 80GB — report the MFU under both conventions and the wall-clock for the full budget; (d) name one deployment scenario in which the over-training bet does **not** pay off.

??? note "Solution"
    (a) $D/N = 2\times10^{10} / 1.014\times10^{8} \approx 197.2 \approx 200$ tokens/param. Chinchilla-optimal is $D^\* \approx 20N = 2.028\times10^{9}$, so the ratio is $2\times10^{10}/2.028\times10^{9} \approx 9.86 \approx 10\times$.

    (b) Naive: $6ND = 6 \times 1.014\times10^{8} \times 2\times10^{10} = 1.22\times10^{19}$ FLOPs. Attention adds $6\,n_{\text{layers}}\,n_{\text{ctx}}\,d_{\text{model}} = 6\times30\times2048\times512 = 1.887\times10^{8}$ FLOPs/token against $6N = 6.084\times10^{8}$ — a 31% surcharge. Total per token $\approx 7.97\times10^{8}$, so $C \approx 1.59\times10^{19}\approx 1.6\times10^{19}$ FLOPs. (Kaplan's non-causal convention doubles the attention term to $3.78\times10^{8}$, giving $\approx 2.0\times10^{19}$.)

    (c) Achieved FLOP/s under $6ND$: $6.084\times10^{8}\times 2.27951\times10^{5} \approx 1.387\times10^{14}$, i.e. 138.7 TFLOP/s, or $138.7/312 \approx \mathbf{44.5\%}$ MFU against the A100's bf16 dense peak. With the attention term: $7.97\times10^{8}\times 2.27951\times10^{5}\approx 1.817\times10^{14}$, i.e. $181.7/312 \approx \mathbf{58.2\%}$. Wall-clock does not depend on the convention, because it comes from the rate: $2\times10^{10}/2.27951\times10^{5}\approx 8.77\times10^{4}$ s $\approx \mathbf{24.4}$ **A100-hours** — mid-band of PLAN's 22–29 hr envelope, and identical to $C/(0.582\times 3.12\times10^{14})$, which is the consistency check worth doing. The lesson: two people can quote 44.5% and 58.2% for the same loop; only the tokens/s number is convention-free.

    (d) When serving volume is near zero — a scaling-law data point, a one-off ablation, a model you evaluate once and discard. Over-training is an amortization bet ($\lambda \to 0$ in the Interview Corner's objective), so with no inference to amortize against you should stop near $D^\*\approx 2$B and spend the saved 90% of compute elsewhere.

**2.** (Implementation) `build_intra_doc_causal_mask` materializes a dense $T\times T$ boolean mask — $2048^2$ = 4.2M entries per sequence, which FlashAttention's *varlen* API avoids entirely by taking a `cu_seqlens` array of cumulative segment boundaries instead (Ch. 4.2). (a) Implement `cu_seqlens_from_seq_ids(seq_ids) -> np.ndarray` returning an `int32` array with `cu_seqlens[0] == 0`, `cu_seqlens[-1] == T`, and one entry per segment boundary. (b) Verify it against the dense mask: rebuild the mask from `cu_seqlens` and assert equality on both windows of the worked example. (c) Explain why `seq_ids`, not `position_ids`, is the natural input — and what happens to the leading `-1` tail segment.

??? note "Solution"
    ```python
    """capstone/stacklm/data/varlen.py -- FlashAttention varlen boundaries."""
    import numpy as np


    def cu_seqlens_from_seq_ids(seq_ids: np.ndarray) -> np.ndarray:
        """Cumulative segment boundaries for flash_attn_varlen_* .

        Segments are contiguous by construction (`seq_ids` is a cumsum of bos
        flags), so every boundary is a position where the id changes.
        """
        bounds = np.flatnonzero(np.diff(seq_ids)) + 1
        return np.concatenate(([0], bounds, [seq_ids.shape[0]])).astype(np.int32)


    def mask_from_cu_seqlens(cu: np.ndarray, seq_len: int) -> np.ndarray:
        """Reference expansion: the dense mask varlen attention never builds."""
        seg = np.zeros(seq_len, dtype=np.int64)
        for k in range(len(cu) - 1):
            seg[cu[k]:cu[k + 1]] = k
        causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        return causal & (seg[:, None] == seg[None, :])
    ```

    (b) Window 1 of the worked example is `[bos a1 a2 a3 eos bos b1 b2]`, so `seq_ids = [0 0 0 0 0 1 1 1]` and `cu_seqlens = [0, 5, 8]` — two segments of length 5 and 3, summing to $T=8$. Window 2 is `[eos bos c1 c2 c3 c4 c5 eos]`, so `seq_ids = [-1 0 0 0 0 0 0 0]` and `cu_seqlens = [0, 1, 8]`. In both cases `mask_from_cu_seqlens(cu, 8)` is bit-identical to `build_intra_doc_causal_mask(position_ids)` — the assertion the chapter's CI test runs. Sanity properties worth asserting in your own code: strictly increasing, first entry 0, last entry $T$, and `len(cu) - 1 == n_segments`.

    (c) `seq_ids` is already a *segment label per token*, so boundaries are exactly its change points — one `diff` and one `flatnonzero`. Deriving the same thing from `position_ids` requires looking for zeros, which is only valid because resets happen at zero; with a stored (non-reset) tail such as window 2's `[3, 0, 1, …]` the zero-scan would miss the first boundary entirely, while the `seq_ids` route gets it right. The leading `-1` tail is simply segment 0 as far as `cu_seqlens` is concerned: it becomes a length-1 sequence that attends only to itself, which is the correct behaviour — its true context lives in the previous window and is not available here. In real varlen usage you pass `cu_seqlens_q = cu_seqlens_k = cu` and `max_seqlen = int(np.diff(cu).max())`, and the memory cost drops from $O(T^2)$ to $O(T)$.

**3.** (Quantitative + implementation) The chapter stores only `tokens.bin`, deriving positions on read. (a) Prove that the stored `position_ids` array is redundant — show that `segments_from_bos` recovers the same document segmentation from the tokens alone, and explain the one case where the derived positions differ from the stored ones and why it does not matter for `Stack-100M`. (b) Compute the disk saved on the 20B-token corpus. (c) Under what change to the model would the stored array stop being redundant?

??? note "Solution"
    (a) `pack_documents` emits `<bos> body <eos>` for every chunk, and the BPE tokenizer never produces `bos_id` from ordinary text (it is a reserved special token that `encode` only recognizes when the caller passes it explicitly — see Ch. 14.3), so `input_ids == bos_id` is true at exactly the document starts, the same positions where the stored `position_ids` reset to 0. Hence `cumsum(input_ids == bos_id)` and `cumsum(position_ids == 0)` induce the *same* partition of the window into segments, and the block-diagonal mask built from either is identical.

    The one difference is a window whose first document is a *tail* carried over from the previous window (window 2 in the worked example, which starts at document B's position 3). The stored array says `[3, 0, 1, …]`; the derived array says `[0, 0, 1, …]`. `Stack100M.forward` really does use this array — `cos = self.rope_cos[position_ids]` — so it is not ignored; it is *shift-invariant*. RoPE contributes only through $i-j$ within a segment, and both arrays give the same within-segment differences, so every attention logit inside the tail is unchanged. Note the derived version has a second, mundane virtue: it can never index outside the rotary tables, because it restarts at 0 in every window.

    (b) One `uint16` per token: $20\times10^{9}\times 2 = 4.0\times10^{10}$ bytes = **40 GB saved**, halving the corpus from 80 GB to 40 GB — and halving the read bandwidth and page-cache pressure during training.

    (c) Any model whose positional encoding is *absolute* rather than relative: learned position embeddings, sinusoidal-added-to-input embeddings, or a length-extrapolation rule keyed on the absolute index. Then the actual index matters, not just the difference, and a tail segment restarting at 0 would place document B's continuation at the wrong absolute offset. It would also matter if you wanted to log or filter by true within-document offset (e.g. "evaluate only on tokens beyond position 1024"), since the derived value is wrong for carried-over tails.

**4.** (Quantitative) (a) The LSH candidate probability at true Jaccard $J$ is $P(J) = 1 - (1 - J^{\text{rows}})^{\text{bands}}$. Compute $P(0.9)$ for **A** = (`bands=16, rows=8`, our default) and **B** = (`bands=8, rows=16`), give each configuration's approximate threshold $(1/b)^{1/r}$ alongside FineWeb's **C** = (`bands=14, rows=8`), and say what the difference means for recall. (b) You have a 32 GB box and want the streaming near-dedup to use at most 24 GB. Using the chapter's measured costs (512 B per `SignatureStore` row at `num_perm=128`; ~3.3 KB/document of LSH buckets at `bands=16`; ~83 B/document for the exact-dedup set), what `index_capacity` can you afford, what fraction of a 20M-document corpus does that cover, and what exactly goes wrong beyond it?

??? note "Solution"
    (a) Config A (`rows=8, bands=16`) at $J=0.9$: $0.9^{8} = 0.43047$, so $P_A = 1-(0.56953)^{16}$; since $(0.56953)^{16} = e^{-9.01} \approx 1.2\times10^{-4}$, $P_A \approx 0.9999$. Config B (`rows=16, bands=8`): $0.9^{16} = 0.18530$, so $P_B = 1-(0.81470)^{8} = 1 - e^{-1.640} \approx 0.806$. Thresholds: A $=(1/16)^{1/8}\approx 0.71$; B $=(1/8)^{1/16}\approx 0.88$; C $=(1/14)^{1/8}\approx 0.72$. At $J=0.9$, A flags the pair essentially always while B misses it about one time in five. Fewer, fatter bands raise the effective threshold and *lower* recall; more, thinner bands lower it, catching more true near-duplicates at the cost of spurious candidates that the explicit `threshold=0.8` re-check then filters. C sits essentially where A does (0.72 vs 0.71) — FineWeb's 14×8 and our 16×8 target the same "≈75% similar or more" region; the difference is 112 vs 128 permutations, i.e. signature cost, not semantics.

    (b) Per kept document the index costs $512 + 3{,}300 = 3{,}812$ B. The exact-dedup set runs over the *whole* corpus, not just the index, so it costs $20\times10^{6}\times 83 \approx 1.7$ GB regardless. That leaves $24 - 1.7 = 22.3$ GB, hence $22.3\times10^{9}/3{,}812 \approx 5.85\times10^{6}$: set `index_capacity ≈ 5{,}500{,}000` and keep the rest as headroom for the interpreter, the shuffle buffer, and page cache. That covers **~28%** of a 20M-document corpus. Beyond it, `near_dedup_stream` stops inserting: it still yields every document, still spends ~8 ms MinHashing each one, and detects nothing — recall for the remaining ~72% of the corpus is exactly zero, and the only evidence is the one-shot `log.warning`. Two legitimate fixes: shard the input and dedup each shard independently (accepting missed cross-shard duplicates), or use `datatrove`, whose signatures live on disk keyed by bucket so there is no in-RAM ceiling at all. Raising `index_capacity` until the box swaps is not a fix.

**5.** (Implementation) The driver runs dedup *within* each source. In production you also want a cross-source pass, since Cosmopedia occasionally paraphrases facts that also appear in FineWeb-Edu. Implement `dedup_all_sources(entries, offline=True)` that concatenates every mix entry's stream, applies quality filtering, then exact dedup, then cross-source near-dedup, and returns the kept documents. Explain (a) why the exact pass must come before the near pass, and (b) why the streaming generator form matters at 20B tokens.

??? note "Solution"
    ```python
    """capstone/stacklm/data/dedup_all.py -- cross-source deduplication."""
    from itertools import chain

    from .synthetic import STACK100M_MIX, stream_source
    from .filters import quality_filter
    from .dedup import exact_dedup, near_dedup_stream


    def dedup_all_sources(entries=None, offline: bool = True, num_perm: int = 128,
                          bands: int = 16, threshold: float = 0.8):
        """Concatenate every source's stream and deduplicate globally.

        Returns a GENERATOR: at real scale the caller pipes this straight into
        `pack_documents`, never into a list.
        """
        entries = entries if entries is not None else STACK100M_MIX
        combined = chain.from_iterable(
            stream_source(entry, offline=offline) for entry in entries
        )
        kept = (d for d in combined if quality_filter(d))   # 1. cheap per-doc gate
        kept = exact_dedup(kept)                            # 2. cheap streaming hash
        return near_dedup_stream(kept, num_perm=num_perm,   # 3. expensive MinHash
                                 bands=bands, threshold=threshold)


    if __name__ == "__main__":
        print(f"kept {sum(1 for _ in dedup_all_sources(offline=True))} documents")
    ```

    (a) **Exact before near.** Exact dedup is a single `blake2b` hash plus a set lookup per document — microseconds. Near-dedup costs a ~8 ms MinHash signature plus an LSH insert and a candidate re-check, roughly a thousand times more, *and* consumes a slot in the bounded index. Running exact first strips every verbatim repeat (mirrors, copy-pasted boilerplate, and the exact duplicates `synthetic_corpus` injects every 97th doc) so the expensive pass sees a thinner stream and its scarce index capacity is spent on genuinely distinct documents. The near pass then catches only what a hash cannot: the ~5%-edited near-duplicates (every 53rd doc) and cross-source paraphrases, whose normalized text differs and therefore hashes to a different key.

    (b) **Streaming.** A `near_dedup(docs: list)` signature forces the caller to materialize the corpus: ~20M documents and ~84 GB of text, which no single machine holds. Every stage here — `chain`, the filter genexp, `exact_dedup`, `near_dedup_stream` — is a generator, so memory is bounded by the exact-hash set (~1.7 GB at 20M unique documents) plus the LSH index (~1.9 GB at the 500k default), not by corpus size. The same reasoning is why `PackedMemmapDataset` memory-maps rather than loads. And note the honest caveat this exercise cannot fix: even streaming, this is ~44 single-core-hours of MinHashing for 20M documents *and* an index that covers only the first ~500k of them, which is why the production path hands the job to `datatrove`.

!!! sota "State of the Art & Resources (2026)"
    The "filtered web + synthetic textbooks + code/math" recipe this chapter follows, and the practice of over-training small models past Chinchilla-optimal, are now the mainstream way small open LLMs are built — the links below trace both threads from their founding papers to the current tools.

    **Foundational work**

    - [Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022)](https://arxiv.org/abs/2203.15556) — the Chinchilla scaling law this chapter deliberately trains past, and the reason "tokens per parameter" is the right unit to reason in.
    - [Kaplan et al., *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361) — the source of the $6N$ + attention-term FLOP accounting corrected in Section 1.
    - [Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022)](https://arxiv.org/abs/2107.06499) — the empirical case for the two-stage exact + near dedup implemented here.

    **Recent advances (2023–2026)**

    - [Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (2024)](https://arxiv.org/abs/2406.17557) — the paper behind FineWeb and FineWeb-Edu, 70% of the Stack-100M mix, and the source of the 14×8 MinHash configuration used in the production pipeline above.
    - [Li et al., *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models* (2024)](https://arxiv.org/abs/2406.11794) — a rigorous benchmark for comparing data-curation pipelines at fixed compute, the same design space this chapter navigates by hand.
    - [Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* (2025)](https://arxiv.org/abs/2502.02737) — the technical report behind the small-model, high-quality-mix recipe (and Cosmopedia v2 / `smollm-corpus`) this capstone's data mix follows.
    - [Dubey et al., *The Llama 3 Herd of Models* (2024)](https://arxiv.org/abs/2407.21783) — documents Llama 3 8B's ~15T-token over-training run, the large-scale precedent for the deployment-economics argument in Section 1.
    - [Soldaini et al., *Dolma: an Open Corpus of Three Trillion Tokens…* (2024)](https://arxiv.org/abs/2402.00159) — an open corpus released *with* its curation toolkit, the other main reference implementation of this pipeline.

    **Open-source & tools**

    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — HuggingFace's production pipeline library for large-scale filtering, deduplication, and dataset construction; the real-world version of this chapter's `filters.py`/`dedup.py`, and what FineWeb was built with.
    - [huggingface/datasets](https://github.com/huggingface/datasets) — `load_dataset(..., streaming=True)`, `interleave_datasets(probabilities=...)`, and `.shuffle(buffer_size=...)`: the streaming/mixing layer the driver sits on.
    - [ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) — ready-to-use MinHash/SimHash/suffix-array near-dedup implementations, a drop-in alternative to the from-scratch `MinHasher`/`LSHIndex` here.
    - [NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) — GPU-accelerated fuzzy deduplication and classifier-based quality filtering, when the CPU pipeline becomes the bottleneck.
    - [allenai/dolma](https://github.com/allenai/dolma) — AI2's curation toolkit (taggers, mixers, dedupers) behind the Dolma corpus.
    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and [karpathy/llm.c](https://github.com/karpathy/llm.c) — the flat-binary, memory-mapped `.bin` shard convention `ShardWriter`/`PackedMemmapDataset` follow, end to end without any framework.

    **Go deeper**

    - [FineWeb: Decanting the Web for the Finest Text Data at Scale (blog)](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — HuggingFace's narrative walkthrough of the exact filtering/dedup pipeline FineWeb-Edu was built with, including the MinHash parameter choices.
    - [HuggingFaceTB/smollm-corpus (dataset)](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) — the `cosmopedia-v2` / `fineweb-edu-dedup` / `python-edu` configs behind SmolLM2, and the dataset card that documents which config is which.
