# 3.1 Pretraining Data: Sources, Crawling & The Data Pipeline

Every capability an LLM has — from grammar to reasoning to coding — is first learned from data. Before a single gradient step is taken, a research or engineering team must answer a deceptively difficult question: *what text should the model read, and in what proportions?* Get this wrong and no optimizer, no architecture change, and no amount of compute will fix it.

This chapter is the engineering foundation for Part III. We examine every layer of the data stack: where the text comes from, how it travels from a raw crawl to a packed training shard, and how teams at the frontier have made (and published) their data choices. The adjacent chapter, [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html), covers the transformation step in detail; here we focus on *sourcing, acquisition, and pipeline architecture*.

---

## The Data Problem at Pretraining Scale

Training a frontier model on, say, 10 trillion tokens sounds straightforward until you realize what that means in physical terms. A token is roughly 4 bytes in a UTF-8 BPE vocabulary. Ten trillion tokens is approximately 40 terabytes of raw Unicode text. Assembling that much high-quality, diverse, legally defensible text is a multi-month infrastructure project.

The challenge is not just volume. It is *quality heterogeneity*: the web contains academic prose, Python source code, forum arguments, machine-generated spam, adult content, and plagiarized boilerplate — all interleaved. A model trained on an indiscriminate dump will learn the distribution of *all* of those things. Careful curation of the data mixture is how practitioners steer a model's knowledge, language coverage, and safety properties before a single training step.

Three design axes frame every data decision:

1. **Breadth vs. depth.** Web crawls offer unmatched breadth but noisy quality. Curated corpora (books, arXiv, Wikipedia) offer depth but limited scale.
2. **Compute efficiency.** The Chinchilla scaling law (covered in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)) shows that for a fixed compute budget, it is usually better to train on *more* tokens from a *smaller* model than to over-train a giant model. Data becomes the binding constraint.
3. **Legal exposure.** Copyright, license compliance, and the emerging regulatory landscape around training data consent are real engineering constraints, not afterthoughts.

---

## Common Crawl: The Backbone of LLM Data

[Common Crawl](https://commoncrawl.org) is a nonprofit that continuously crawls the web and publishes monthly snapshots under an open-access license. It is the primary raw material for almost every large open dataset. As of 2025, the Common Crawl corpus spans over 250 billion web pages accumulated across more than a decade of monthly releases.

### File Formats: WARC, WET, and WAT

Each monthly crawl is published in three complementary formats:

| Format | Content | Typical use |
|--------|---------|-------------|
| **WARC** (Web ARChive) | Full HTTP responses including headers, HTML, images | Re-rendering, link analysis |
| **WET** (WARC Encapsulated Text) | Extracted plain text only | Language model training |
| **WAT** (Web Archive Transformation) | Metadata JSON only (links, language tags, etc.) | Filtering, graph analysis |

For LLM pretraining, WET files are the convenient starting point — but with an important caveat. WET text is produced by Common Crawl's *own* basic built-in extractor (historically based on Apache Tika / jsoup, via the `ia-web-commons` library), **not** by a modern main-content extractor like `trafilatura`. That built-in extractor dumps nearly all visible text — including navigation menus, footers, cookie banners, and other boilerplate — and its quality is noticeably lower than modern extractors. This is exactly why the highest-quality pipelines (RefinedWeb, FineWeb, Dolma) *re-extract* text from the raw WARC HTML rather than trusting WET, as the *Extracting Text from WARC* section below shows. Each monthly dump produces roughly 9 TB of WET files compressed with gzip (the raw WARC set for the same crawl is about 90 TB).

A WET record looks like this:

```text
WARC/1.0
WARC-Type: conversion
WARC-Target-URI: https://example.com/article
WARC-Date: 2024-04-15T12:34:56Z
WARC-Refers-To: <urn:uuid:...>
Content-Type: text/plain
Content-Length: 2048

The quick brown fox jumps over the lazy dog. Lorem ipsum ...
```

### Crawl Anatomy

Common Crawl uses Apache Nutch and Heritrix to discover pages. The crawler follows links breadth-first from a seed list of highly-linked root domains. Because it respects `robots.txt` and applies politeness delays, the crawl takes weeks per snapshot. The resulting corpus is *not* a uniform sample of the web — high-PageRank domains are over-represented, low-resource languages are under-represented, and large static-content sites (PDFs, images) are excluded.

This non-uniformity is both a feature and a bug. English Wikipedia is crawled in its entirety every month. Some low-resource languages appear only in a handful of documents. Any downstream model will inherit these imbalances unless they are explicitly corrected through domain up/down-weighting.

### Extracting Text from WARC

WET is convenient, but as noted above its built-in extractor keeps boilerplate and is lower quality than modern tools. The modern pipelines — RefinedWeb, FineWeb, and Dolma — therefore skip WET and re-extract text straight from the raw **WARC** HTML. This is also the first stage of the CS336 data assignment. The recipe: iterate the WARC's HTTP *response* records, decode the HTML payload with the right charset, and run a main-content extractor (`trafilatura` or `resiliparse`) to drop navigation, sidebars, and footers.

```python
"""
Extract main-content text from raw Common Crawl WARC files.
    pip install warcio trafilatura        # portable; what we use here
    # 10-50x faster alternative for cluster-scale runs:
    #   pip install fastwarc resiliparse
"""
import trafilatura
from warcio.archiveiterator import ArchiveIterator


def extract_text_from_warc(warc_path: str):
    """Yield (url, main_text) for each HTML response record in a WARC file."""
    with open(warc_path, "rb") as fh:
        for record in ArchiveIterator(fh):
            if record.rec_type != "response":          # skip request/metadata
                continue
            ctype = record.http_headers.get_header("Content-Type", "") or ""
            if "html" not in ctype.lower():             # skip PDFs, images, ...
                continue

            url = record.rec_headers.get_header("WARC-Target-URI")
            raw = record.content_stream().read()        # raw HTTP response body
            html = _decode_html(raw, ctype)
            if html is None:
                continue

            # trafilatura strips boilerplate and returns clean main content.
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_precision=True,   # prefer clean text over recall
            )
            if text:
                yield url, text


def _decode_html(raw: bytes, content_type: str):
    """Bytes -> str using the HTTP charset if given, then utf-8, then latin-1."""
    charset = None
    lc = content_type.lower()
    if "charset=" in lc:
        charset = lc.split("charset=", 1)[1].split(";")[0].strip()
    for enc in (charset, "utf-8", "latin-1"):
        if enc:
            try:
                return raw.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
    return None
```

**Why bother, if WET already gives you text?** Because the two disagree, and the difference shows up downstream. On the same page, the built-in WET extractor emits the article *plus* the nav menu, the "related stories" rail, and the cookie banner; `trafilatura` on the WARC HTML emits just the article body:

| Extractor | What you get on a typical news page | Boilerplate |
|-----------|-------------------------------------|-------------|
| CC WET (built-in) | article body + nav + footer + "related" + cookie notice | High |
| `trafilatura` on WARC | article body only | Low |

FineWeb's ablations found that training on text re-extracted from WARC with `trafilatura` measurably beat training on the WET text for the same pages — which is why FineWeb (and RefinedWeb before it) never trains on WET at all.

The cost is compute: reading WET is nearly free, whereas WARC re-extraction parses full HTML for every page. On a **laptop/CPU**, `warcio` + `trafilatura` handles roughly 50–200 pages/sec/core — fine for one WARC file (~1 GB, ~30–50k records) as a learning exercise. At **crawl scale** (one monthly snapshot is ~90k WARC files, ~90 TB compressed, ~2.4B pages), switch to `fastwarc` + `resiliparse` (10–50x faster C-backed parsing) and fan out across a cluster with a work queue, one WARC per task — the same embarrassingly parallel pattern used for WET below.

---

## Curated Corpora: Beyond the Raw Crawl

Web text alone produces capable but uneven models. Practitioners supplement it with curated corpora that provide depth in specific high-value domains.

### Books

**Books3** (part of The Pile, EleutherAI, 2020) contains roughly 197,000 full-length books scraped from a shadow library. It offers multi-chapter long-form coherent text that web pages rarely provide — an important signal for long-range narrative reasoning. However, Books3 has faced significant copyright challenges, leading many subsequent projects to omit it or replace it with licensed alternatives.

**Project Gutenberg** provides around 60,000 public-domain books. Smaller in scale but legally unambiguous.

**OpenLibrary / Internet Archive Book Scans** offer scanned OCR text from millions of physical books, though quality is variable and OCR noise requires careful cleaning.

### Scientific and Technical Text

**arXiv** preprints: The arXiv bulk access API provides LaTeX source for millions of papers across physics, mathematics, computer science, and adjacent fields. LaTeX source is richer than rendered PDFs because it contains explicit semantic structure (theorems, proofs, equations). The S2ORC project (Semantic Scholar Open Research Corpus) provides cleaned, parsed versions of the open-access scientific literature.

**PubMed Central** (PMC) provides open-access biomedical literature. This domain is valuable for medical reasoning but requires understanding the distinction between the structured abstract and the full paper.

**GitHub code**: Code is now a standard pretraining ingredient. The Stack (BigCode, 2022) assembled 358 GB of permissively licensed code across 86 programming languages by filtering GitHub by license type. Code is valuable not only for coding tasks but also for teaching formal reasoning patterns — many researchers believe that code-heavy pretraining substantially improves mathematical performance.

### Wikipedia and Reference Content

Wikipedia is the single highest-quality per-token training signal in most data mixes. Its prose is grammatical, encyclopedic, cross-referenced, and maintained by human editors. However, the entire English Wikipedia dumps to only about 20 GB of plain text — a tiny fraction of a modern training corpus by volume. Its contribution is disproportionate to its weight, which is why data mixtures almost universally up-weight it.

**Wikidata, Freebase, DBpedia** provide structured knowledge in triple or JSON form. Some pipelines convert these to natural language sentences (e.g., "Paris is the capital of France") to inject factual grounding.

### Multilingual Sources

**CC-100** (Conneau et al., 2020) extracted 100 language-specific corpora from Common Crawl using language identification. **mC4** (Raffel et al., T5 paper follow-up) is the multilingual variant of C4. **CulturaX** (Nguyen et al., 2023) provides a deduplicated, filtered multilingual corpus spanning 167 languages built from mC4 and the Oscar corpus.

---

## The Data Recipe: Mixture Weights

Assembling sources is only the first step. The second — and often more consequential — step is deciding *how much* of each source to include per training step. This is called the **data recipe** or **mixture**.

Let $D = \{D_1, D_2, \ldots, D_k\}$ be a collection of $k$ domain corpora. At each training step, a batch is sampled from the mixture using domain weights $w = (w_1, \ldots, w_k)$ with $\sum_i w_i = 1$. These weights determine the effective number of tokens seen from each domain over the full training run.

If the model is trained for $N$ total tokens and domain $i$ has weight $w_i$, then it sees approximately $N \cdot w_i$ tokens from domain $i$. If domain $i$ contains $|D_i|$ tokens, the **epoch count** for that domain is:

$$
\text{epochs}_i = \frac{N \cdot w_i}{|D_i|}
$$

Repeating data is not free — multiple passes over the same text lead to memorization and reduced generalization, particularly noticeable after about 4 epochs (Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023). This creates a practical ceiling: for domains like Wikipedia (small corpus), high weights cause many-epoch repetition; for web (huge corpus), even a modest weight can represent less than 1 epoch.

!!! example "Worked example: data mixture sizing"

    Suppose we are training a 7B parameter model on 2 trillion tokens ($N = 2 \times 10^{12}$) with the following simplified mixture:

    | Domain | Weight $w_i$ | Corpus size $|D_i|$ | Epochs $= N \cdot w_i / |D_i|$ |
    |--------|-------------|---------------------|-------------------------------|
    | Web (CC) | 0.65 | 3.0T tokens | $\approx 0.43$ |
    | Code (GitHub) | 0.15 | 350B tokens | $\approx 0.86$ |
    | Books | 0.08 | 90B tokens | $\approx 1.78$ |
    | arXiv | 0.04 | 40B tokens | $\approx 2.00$ |
    | Wikipedia | 0.04 | 4B tokens | $\approx 20.0$ |
    | Other curated | 0.04 | 60B tokens | $\approx 1.33$ |

    Wikipedia is seen roughly 20 times despite comprising only 0.2% of the raw token count. This is by design — its per-token quality justifies heavy up-weighting. Code and arXiv approach 1–2 epochs, which is acceptable. Web crawl is comfortably under 1 epoch, so no repetition memorization.

    The practical implication: if you reduce $N$ (train shorter, e.g., 1T tokens), you cut every epoch count proportionally, meaning Wikipedia moves to ~10 epochs and you may see slightly more memorization of Wikipedia text.

---

## Open Dataset Case Studies

Several research groups have released their full data pipelines. These are the landmark datasets every practitioner should know.

### The Pile (EleutherAI, Gao et al., 2020)

The Pile was one of the first carefully documented, publicly released pretraining datasets. It assembled 825 GB of text from 22 diverse sources, including Pile-CC (a cleaned Common Crawl subset), Books3, OpenWebText2, GitHub, arXiv, FreeLaw, Wikipedia, DM Mathematics, and more. The Pile's documentation of per-domain composition established a template that all subsequent efforts followed. GPT-Neo, GPT-J, and GPT-NeoX were trained on The Pile.

### C4 (Raffel et al., T5, 2019)

Colossal Clean Crawled Corpus (C4) is a 750 GB English-only filtered version of a single Common Crawl snapshot. The cleaning pipeline applied heuristic filters: remove lines without terminal punctuation, remove documents under 5 sentences, remove documents containing JavaScript warnings, and deduplicate at the three-sentence-span level (any span of three consecutive sentences occurring more than once in the corpus was removed). C4 became the standard pretraining corpus for the T5 family and remains a useful ablation baseline.

### RedPajama (Together AI, 2023)

RedPajama replicated and open-sourced the LLaMA training data recipe. It documented the exact composition used by Meta's LLaMA: CommonCrawl (67%), C4 (15%), GitHub (4.5%), Wikipedia (4.5%), Books (4.5%), arXiv (2.5%), StackExchange (2%). RedPajama v2 extended this to over 30 trillion tokens with quality signals attached to each document (e.g., perplexity score against a reference model, URL quality signals) so downstream users could apply their own filtering thresholds.

### Dolma (AI2, Soldaini et al., 2024)

Dolma is a 3 trillion token open corpus assembled by the Allen Institute for AI for training OLMo. It is notable for:
- **Extreme documentation**: every filtering step, model card, and design decision is described in the accompanying paper.
- **Legal care**: documents are tagged with license information; Books3 is excluded.
- **Taggers**: Dolma attaches Gopher-quality signals, language identification scores, and toxicity scores to every document without removing them — allowing users to filter at different thresholds.

The Dolma toolkit (a separate open-source tool) supports streaming processing of WARC/WET files, deduplication, and mixing — making it the most production-ready open data pipeline as of 2025.

### RefinedWeb (TII, Penedo et al., 2023)

RefinedWeb is the web corpus behind the Falcon models, and the dataset that first demonstrated the thesis FineWeb later scaled: *properly filtered and deduplicated web data alone can match or beat curated multi-source mixtures like The Pile*. It is produced by the **MacroData Refinement (MDR)** pipeline, whose stages are the modern template for web curation:

1. **URL filtering** — a blocklist of adult/spam/low-quality domains plus a URL-word score, applied *before* any expensive processing.
2. **Text extraction from WARC** — main-content extraction with `trafilatura` directly from the raw HTML (not WET), precisely to avoid the boilerplate that Common Crawl's built-in WET extractor leaves behind.
3. **Quality filtering** — MassiveText/Gopher-style document heuristics (length, symbol-to-word ratio, repetition, stop-word presence) plus line-level heuristics that strip leftover navigation and boilerplate.
4. **Deduplication** — both exact (suffix-array substring) and fuzzy (MinHash) dedup, applied aggressively; RefinedWeb found dedup to be one of the single largest quality levers.

The public release is a 600B-token extract of a ~5-trillion-token internal corpus. Its headline result — a model trained on web-only RefinedWeb outperforming one trained on The Pile — reframed the field's assumption that curated corpora were indispensable, and set up FineWeb's 15T-token follow-through a year later.

### FineWeb (HuggingFace, Penedo et al., 2024)

FineWeb is a 15 trillion token dataset derived entirely from Common Crawl (96 monthly snapshots through 2024). Its key innovation is showing that *aggressive quality filtering alone* — without exotic curated corpora — can match or exceed the downstream performance of carefully hand-curated mixes. The FineWeb-Edu subset (1.3T tokens) applies an educational quality classifier (a fine-tuned LLM scoring each page on a 0–5 scale for educational value) and dramatically outperforms base FineWeb on knowledge-intensive benchmarks at small model scales (1B–7B parameters).

FineWeb is released under the Common Crawl terms of service, making it the most legally accessible large-scale web corpus.

---

## Data Engineering at Scale

Assembling trillions of tokens requires industrial-grade data engineering. A naive approach — load everything into memory, filter, write out — fails immediately. We need streaming pipelines that process hundreds of terabytes without materializing more than a few GB at a time.

### Distributed Processing Architecture

The standard architecture uses a distributed compute layer (Apache Spark, or custom Python with Ray/Dask) on top of object storage (S3, GCS). A typical pipeline stage looks like:

```text
  S3 / GCS (raw WET files)
       |
       v
  [Worker pool]  — stream and decompress WET
       |
       v
  [Language ID]  — fastText / CLD3 per document
       |
       v
  [Quality filter]  — heuristic + model-based
       |
       v
  [Deduplication]  — MinHash LSH (see next chapter)
       |
       v
  [Tokenize + pack]  — HuggingFace tokenizers
       |
       v
  [Write shards]  — .bin / .arrow / .jsonl.zst
       |
       v
  S3 / GCS (training shards, ready to stream)
```

The key design principle is **embarrassing parallelism**: each WET file (typically 400–700 MB compressed) is an independent processing unit. Workers download, decompress, and process one file at a time, emitting filtered documents to an output bucket. No inter-worker coordination is needed until the deduplication stage.

### Streaming Data Pipeline in Python

Below is a self-contained, production-inspired streaming pipeline that processes a directory of compressed WET files, applies basic quality filters, tokenizes, and writes packed binary shards. This sketches the core logic you would extend with deduplication and more sophisticated filtering.

```python
"""
Streaming pretraining data pipeline.

Processes gzipped WET files from a directory,
applies quality heuristics, tokenizes with a BPE tokenizer,
packs tokens into fixed-length context windows, and
writes binary numpy shards.

Usage:
    python pipeline.py --input /data/wet --output /data/shards \
                       --context_len 4096 --shard_size 500_000_000
"""

import argparse
import gzip
import os
import re
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer  # HuggingFace fast tokenizers


# ── WET record parser ──────────────────────────────────────────────────────

def _iter_wet_records(lines):
    """
    Core WET parser over any iterator of text lines.

    Each WET record is a WARC/1.0 record:

        WARC/1.0                     <- record boundary
        WARC-Type: conversion
        WARC-Target-URI: <url>
        ... more headers (Content-Type, Content-Length, ...) ...
                                     <- ONE blank line ends the header block
        <body: the record's text payload>
                                     <- blank line(s), then the next 'WARC/1.0'

    The parser resets ALL per-record state at every 'WARC/1.0' boundary and
    ignores every header line, so header text (e.g. 'Content-Length: 2048')
    and the next record's boundary lines can never leak into a body. The
    first record in a WET file is a 'warcinfo' record with no
    WARC-Target-URI; it is skipped because it has no URL.

    (Content-Length is ignored in this didactic version; see the production
    note below for why real pipelines honor it instead.)
    """
    url = None
    in_header = False        # inside the header block of the current record
    body_lines = []
    started = False          # have we seen the first 'WARC/1.0' yet?

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line == "WARC/1.0":
            # Record boundary: flush the record we just finished ...
            if started:
                text = "\n".join(body_lines).strip("\n")
                if url is not None and text:
                    yield url, text
            # ... then reset ALL state for the new record.
            url = None
            in_header = True
            body_lines = []
            started = True
            continue

        if in_header:
            if line == "":
                in_header = False              # blank line ends the headers
            elif line.startswith("WARC-Target-URI:"):
                url = line.split(":", 1)[1].strip()
            # All other header lines are ignored and never reach body_lines.
        else:
            body_lines.append(line)

    # Flush the final record.
    if started:
        text = "\n".join(body_lines).strip("\n")
        if url is not None and text:
            yield url, text


def parse_wet_records(filepath: str):
    """Yield (url, text) pairs from a gzipped WET file (WARC/1.0 format)."""
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
        yield from _iter_wet_records(fh)


# ── Quality heuristics ─────────────────────────────────────────────────────

def passes_quality_filter(text: str, min_chars: int = 200) -> bool:
    """
    A simplified version of the heuristics used by C4 and Gopher.
    Returns True if the document passes all checks.

    Real pipelines also run language identification (fastText) and
    a reference model perplexity filter — omitted here for brevity.
    """
    # (1) Minimum length
    if len(text) < min_chars:
        return False

    # (2) Must contain at least 3 lines with terminal punctuation
    sentences_with_punct = [
        l for l in text.splitlines()
        if len(l) > 10 and l[-1] in ".!?\""
    ]
    if len(sentences_with_punct) < 3:
        return False

    # (3) Word-level repetition ratio (Gopher signal)
    # If the most common word accounts for > 20% of words, likely spam
    words = text.lower().split()
    if not words:
        return False
    most_common_freq = max(
        words.count(w) for w in set(words)
    ) / len(words)
    if most_common_freq > 0.20:
        return False

    # (4) Bullet/symbol density (JS code / menu spam heuristic)
    non_alpha = sum(1 for c in text if not c.isalnum() and c not in " \n.,;:!?'\"-()")
    if non_alpha / max(len(text), 1) > 0.25:
        return False

    return True


# ── Token packer ──────────────────────────────────────────────────────────

class TokenShard:
    """
    Accumulates tokenized documents and flushes complete rows to a binary
    shard file. Each row is (ctx_len + 1) int32 tokens: ctx_len inputs plus
    one extra token for the label shift (target = row[1:]). This matches the
    reader in ShardedTokenDataset exactly — the practitioner tip below is
    implemented here, so there is no off-by-one at read time. The final
    partial row (< ctx_len + 1 tokens) is dropped.
    """

    def __init__(self, output_dir: str, shard_size: int, ctx_len: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ctx_len = ctx_len
        self.row_len = ctx_len + 1                       # tokens per training row
        # Round the requested shard size down to a whole number of rows.
        self.rows_per_shard = max(1, shard_size // self.row_len)
        self.shard_tokens = self.rows_per_shard * self.row_len
        self._buffer = []
        self._shard_idx = 0
        self._total_tokens = 0

    def add_tokens(self, token_ids: list[int], eos_id: int):
        """Append document tokens followed by an EOS separator."""
        self._buffer.extend(token_ids)
        self._buffer.append(eos_id)
        while len(self._buffer) >= self.shard_tokens:
            self._flush_shard(self.shard_tokens)

    def _flush_shard(self, n_tokens: int):
        """Write one shard containing a whole number of (ctx_len + 1) rows."""
        n_rows = n_tokens // self.row_len
        n_keep = n_rows * self.row_len                   # carry remainder forward
        tokens = self._buffer[:n_keep]
        self._buffer = self._buffer[n_keep:]
        if not tokens:
            return
        arr = np.array(tokens, dtype=np.int32)
        shard_path = self.output_dir / f"shard_{self._shard_idx:05d}.bin"
        arr.tofile(shard_path)
        self._total_tokens += len(tokens)
        self._shard_idx += 1
        print(f"  Wrote {shard_path.name}: {len(tokens):,} tokens "
              f"({n_rows:,} rows of {self.row_len})")

    def finalize(self):
        """Flush any remaining full rows from the buffer."""
        while len(self._buffer) >= self.shard_tokens:
            self._flush_shard(self.shard_tokens)
        if len(self._buffer) >= self.row_len:
            self._flush_shard(len(self._buffer))
        print(f"\nTotal tokens written: {self._total_tokens:,}")


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str,
    context_len: int = 4096,
    shard_size: int = 500_000_000,
):
    """
    Main entry point.  In production, this loop is distributed across
    hundreds of workers, each handling a distinct subset of WET files.
    Here we run single-threaded for clarity.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    packer = TokenShard(output_dir, shard_size, context_len)

    wet_files = sorted(Path(input_dir).glob("*.wet.gz"))
    print(f"Found {len(wet_files)} WET files to process.")

    total_docs = 0
    kept_docs = 0

    for wet_path in wet_files:
        print(f"Processing {wet_path.name} ...")
        for url, text in parse_wet_records(str(wet_path)):
            total_docs += 1
            if not passes_quality_filter(text):
                continue
            # Tokenize (encode returns a huggingface Encoding object)
            encoding = tokenizer.encode(text)
            packer.add_tokens(encoding.ids, eos_id)
            kept_docs += 1

    packer.finalize()
    retention = 100 * kept_docs / max(total_docs, 1)
    print(f"Retention rate: {kept_docs:,}/{total_docs:,} = {retention:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--context_len", type=int, default=4096)
    parser.add_argument("--shard_size", type=int, default=500_000_000)
    args = parser.parse_args()

    run_pipeline(
        args.input, args.output, args.tokenizer,
        args.context_len, args.shard_size,
    )
```

In production, the outer loop over WET files is parallelized — each worker picks a file from an SQS queue or equivalent, processes it, and writes results to a shared output bucket. Workers are stateless, so they are trivially recoverable on node failure.

!!! note "The WET parser above is didactic — use a real WARC reader in production"

    `parse_wet_records` is a hand-rolled line scanner. It is correct for well-formed WET files, but it leans on string heuristics: a body line that happens to equal `WARC/1.0`, a truncated record, or unusual header casing could fool it. Production pipelines never parse WARC/WET by hand — they use `warcio`'s `ArchiveIterator` or the much faster `fastwarc`, both of which honor each record's `Content-Length` and read exactly that many payload bytes, so body content can never be mistaken for a header or a record boundary. To read WET this way, iterate exactly as in the *Extracting Text from WARC* section and keep records with `record.rec_type == "conversion"`.

### Shard Layout for Streaming Training

Training reads shards in random order to approximate i.i.d. sampling. Each shard is a flat binary file of `int32` token IDs arranged in rows of `context_len` tokens. The training dataloader memory-maps these files:

```python
import numpy as np
import torch
from torch.utils.data import IterableDataset


class ShardedTokenDataset(IterableDataset):
    """
    Memory-mapped streaming dataset over pre-tokenized binary shards.

    Shards are laid out as flat int32 arrays; each context window
    is a contiguous slice of length (context_len + 1) — the +1 gives
    us the label shift: input = tokens[:-1], target = tokens[1:].
    """

    def __init__(
        self,
        shard_dir: str,
        context_len: int = 4096,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_paths = sorted(Path(shard_dir).glob("*.bin"))
        self.context_len = context_len
        self.shuffle_shards = shuffle_shards

    def __iter__(self):
        shard_order = list(self.shard_paths)

        # Split shards across DataLoader workers. WITHOUT this, every worker
        # (num_workers > 1) iterates the *entire* shard list, so the DataLoader
        # yields each example once per worker — silently duplicating your data.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard_order = shard_order[worker_info.id :: worker_info.num_workers]
            # If num_workers > number of shards, some workers get nothing.

        if self.shuffle_shards:
            import random
            random.shuffle(shard_order)   # seed per (epoch, worker) in real code

        for shard_path in shard_order:
            # Memory-map: no data is loaded until a row is accessed
            data = np.memmap(shard_path, dtype=np.int32, mode="r")
            n_contexts = len(data) // (self.context_len + 1)

            # Shuffle context order within the shard
            indices = np.random.permutation(n_contexts)

            for idx in indices:
                start = idx * (self.context_len + 1)
                chunk = data[start : start + self.context_len + 1]
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield x, y
```

!!! tip "Practitioner tip"

    Write shards with `context_len + 1` tokens per row so that every fetch yields a complete `(input, label)` pair with no off-by-one indexing at read time. This eliminates a common source of subtle bugs when labels wrap across rows.

### Verifying the Pipeline

Before running on real crawl data, sanity-check the three moving parts on tiny synthetic inputs — no tokenizer or network needed. These are deterministic and should print `All pipeline checks passed.`:

```python
# verify_pipeline.py
# Deterministic, offline checks for the pipeline above.
import tempfile
from pathlib import Path

import numpy as np
from pipeline import _iter_wet_records, passes_quality_filter, TokenShard


def test_parse_wet_records():
    """2-record WET sample -> exact (url, text); warcinfo skipped, no leakage."""
    sample = (
        "WARC/1.0\nWARC-Type: warcinfo\n"
        "Content-Type: application/warc-fields\nContent-Length: 26\n\n"
        "isPartOf: CC-MAIN-2024-18\n\n"
        "WARC/1.0\nWARC-Type: conversion\n"
        "WARC-Target-URI: https://example.com/a\n"
        "WARC-Date: 2024-04-15T12:34:56Z\nContent-Type: text/plain\n"
        "Content-Length: 28\n\nHello world.\nThis is page A.\n\n"
        "WARC/1.0\nWARC-Type: conversion\n"
        "WARC-Target-URI: https://example.com/b\n"
        "Content-Type: text/plain\nContent-Length: 12\n\nPage B only.\n\n"
    )
    got = list(_iter_wet_records(sample.splitlines(keepends=True)))
    assert got == [
        ("https://example.com/a", "Hello world.\nThis is page A."),
        ("https://example.com/b", "Page B only."),
    ], got
    for _, text in got:                      # no header / boundary leakage
        assert "Content-Length" not in text and "WARC/1.0" not in text
    print("test_parse_wet_records: OK (2 docs, warcinfo skipped, no leakage)")


def test_quality_filter():
    good = "\n".join([
        "The history of computing spans several centuries of innovation.",
        "Early mechanical calculators gave way to electronic machines.",
        "Modern processors execute billions of instructions each second.",
        "Software links these machines into a global information network.",
        "Researchers keep pushing the boundaries of what is possible.",
    ])
    assert passes_quality_filter(good) is True           # clean prose passes
    assert passes_quality_filter("Hi there.") is False    # too short
    assert passes_quality_filter("buy " * 300) is False   # repetition + no punct
    print("test_quality_filter: OK")


def test_token_shard():
    # ctx_len=4 -> row_len=5; shard_size=15 -> 3 rows/shard.
    # Feed 4+1, 4+1, 4+1 (flush 15), then 7+1; finalize keeps 1 row (5), drops 3.
    with tempfile.TemporaryDirectory() as d:
        shard = TokenShard(d, shard_size=15, ctx_len=4)
        shard.add_tokens([1, 2, 3, 4], eos_id=0)                   # buffer 5
        shard.add_tokens([5, 6, 7, 8], eos_id=0)                   # buffer 10
        shard.add_tokens([9, 10, 11, 12], eos_id=0)               # 15 -> flush
        shard.add_tokens([13, 14, 15, 16, 17, 18, 19], eos_id=0)  # buffer 8
        shard.finalize()                                          # flush 5, drop 3
        assert shard._total_tokens == 20, shard._total_tokens
        rows = np.fromfile(sorted(Path(d).glob("*.bin"))[0], dtype=np.int32)
        assert len(rows) % 5 == 0                                  # whole rows
    print("test_token_shard: OK (20 tokens, 3 dropped, rows of ctx_len+1)")


if __name__ == "__main__":
    test_parse_wet_records()
    test_quality_filter()
    test_token_shard()
    print("All pipeline checks passed.")
```

With a real tokenizer you can add one more round-trip check: take any written row, and confirm `tokenizer.decode(row[:-1].tolist())` reproduces readable text from your corpus — the fastest way to catch a dtype or endianness mismatch in the shard format.

---

## Domain Mixing Strategies and Curriculum

Beyond static mixture weights, researchers have explored *dynamic* data curricula — changing the mixture as training progresses.

**Skill-it! (Chen et al., 2023)** showed that ordering data by increasing difficulty (measured by loss on a held-out probe set) can improve downstream performance, analogous to curriculum learning in classical ML.

**DoReMi (Xie et al., 2023)** framed mixture weight selection as a distributionally robust optimization (DRO) problem: train a small proxy model and a domain weight learner simultaneously, with the weight learner trying to equalize worst-case domain loss. The resulting weights outperform human-tuned weights on average across downstream tasks.

**Online data mixing** is used by some teams to re-weight domains mid-training based on validation loss trends — domains where validation loss plateaus early get down-weighted. This requires periodic synchronization with a validation loop but can squeeze meaningful quality improvements from a fixed corpus.

The key insight from all of these approaches is that the optimal mixture is *not* the natural distribution of the web. The web has too much low-quality text and too little high-quality structured knowledge. Every serious training run explicitly tilts the distribution toward quality.

---

## Licensing, Legal Considerations & Data Ethics

Licensing is increasingly a first-class engineering concern, not a legal afterthought. The 2023–2024 wave of copyright litigation against AI companies (NYT v. OpenAI, Getty v. Stability AI, and others) has made data provenance a real risk factor.

### License Taxonomy

Training data sources fall into a rough hierarchy of legal clarity:

| Tier | Examples | Risk level |
|------|---------|-----------|
| Public domain | Project Gutenberg, US government works, pre-1928 texts | Lowest |
| Permissive open licenses | Apache 2.0, MIT, CC-BY code/text | Low |
| Share-alike licenses | CC BY-SA, GPL | Medium (copyleft may propagate) |
| Unclear / no license | Most of the web, Common Crawl | High — jurisdiction-dependent |
| Opted-out or robots.txt blocked | Varies by site | Should exclude |

Most frontier training corpora rely heavily on Tier 3–4 material, betting on fair-use arguments in the US and equivalent doctrines elsewhere. The EU AI Act (effective 2025–2026) adds new transparency requirements: providers of general-purpose AI models must publish a "sufficiently detailed summary" of training data, including copyright opt-outs.

### C2PA and Data Consent Infrastructure

The Content Credentials (C2PA) standard and emerging "robots.txt for AI" (the `ai-disallow` token in robots.txt, as proposed by several working groups) are nascent infrastructure for consent signaling. The AI2 Dolma team introduced the concept of **domain-level opt-out lists** — documents from sites that have explicitly opted out of AI training are tagged and excluded. This is currently voluntary, but is likely to become a legal requirement in multiple jurisdictions.

!!! warning "Common pitfall"

    Using web data with no license audit is fine for research prototypes but creates material legal risk for deployed products. Build document-level provenance tracking (source URL, crawl date, inferred license tier) into your pipeline from day one — retrofitting it after training is far more expensive.

---

## Interview Corner

!!! interview "Interview Corner"

    **Q:** A team is building a 7B model and asks you to design the pretraining data pipeline from scratch. Walk through the major decisions, from raw Common Crawl data to packed training shards.

    **A:** I would structure the work in four phases:

    1. **Source selection and acquisition**: Start with a recent Common Crawl WET snapshot (~9 TB compressed). Add curated high-quality corpora: Wikipedia (up-weight heavily), deduplicated GitHub code filtered for permissive licenses, arXiv LaTeX source, and a cleaned books corpus. Each source is tracked with a provenance record (URL, license tier, crawl date).

    2. **Per-document filtering**: Stream each WET file through (a) language ID (fastText; keep only desired languages), (b) Gopher heuristics (word count, symbol ratio, line-ending punctuation, repetition rate), and (c) an optional perplexity filter against a small reference LM. Expect to retain roughly 30–50% of raw Common Crawl by document count but higher by quality-adjusted token value.

    3. **Deduplication**: Apply MinHash LSH at 13-gram shingle level to remove near-duplicate documents across the full corpus. This is the most computationally intensive step — covered in the [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) chapter. Expect 15–30% additional token reduction.

    4. **Mixing and packing**: Assign domain weights (roughly: 65% web, 15% code, 8% books, 4% arXiv, 4% Wikipedia, 4% other) and sample proportionally when writing training shards. Tokenize with the model's BPE tokenizer, concatenate documents with an EOS separator, pack into fixed `context_len + 1` rows, and write as memory-mappable `int32` binary shards (~500M tokens each).

    The resulting pipeline should be embarrassingly parallel at the file level. Track total token counts and per-domain epoch counts throughout — the epoch counts will tell you whether you need to source more data or accept repetition.

---

## Key Takeaways

!!! key "Key Takeaways"

    - Common Crawl WET files are the raw material for most open LLM training corpora; the monthly snapshots total hundreds of terabytes and require industrial-strength streaming pipelines to process.
    - No one trains on raw crawl data. Every serious corpus applies at minimum: language filtering, Gopher-style quality heuristics, and MinHash deduplication.
    - The data recipe — the mixture weights across domains — matters as much as total token count. Up-weighting Wikipedia, code, and curated scientific text is standard practice even though they are a tiny fraction of raw volume.
    - Epoch counts are a first-class design parameter: Wikipedia at high weight may be seen 10–20 times in a trillion-token run; repeating data beyond ~4 epochs degrades generalization.
    - FineWeb demonstrated that aggressive quality filtering of Common Crawl alone can match carefully hand-curated mixes; FineWeb-Edu showed that educational quality classifiers further improve knowledge benchmarks.
    - Dolma and RedPajama are the most thoroughly documented open pretraining datasets; their papers are essential reading for understanding real-world pipeline decisions.
    - Licensing and data provenance are production engineering constraints, not research afterthoughts. Build document-level provenance tracking from day one; EU AI Act requirements make it legally necessary.
    - A streaming pipeline with memory-mapped binary shards, embarrassing per-file parallelism, and domain-balanced sampling is the standard architecture; it scales from research to trillion-token production runs.

---

!!! sota "State of the Art & Resources (2026)"
    Pretraining data curation has matured into a rigorous engineering discipline: the field now understands that aggressive quality filtering of Common Crawl alone can match hand-curated mixes (FineWeb), that data mixture weights should be optimized rather than guessed (DoReMi, DCLM), and that transparent, reproducible pipelines (Dolma, RedPajama-v2) are the new baseline for open research.

    **Foundational work**

    - [Gao et al., *The Pile: An 800 GB Dataset of Diverse Text for Language Modeling* (2020)](https://arxiv.org/abs/2101.00027) — established the template of documenting multi-source open pretraining corpora.
    - [Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019)](https://arxiv.org/abs/1910.10683) — introduced C4 and the web-cleaning heuristics still used today.

    **Recent advances (2023–2026)**

    - [Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (2024)](https://arxiv.org/abs/2406.17557) — 15T-token web corpus; shows aggressive quality filtering alone matches curated mixes; FineWeb-Edu classifier boosts knowledge benchmarks.
    - [Li et al., *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models* (2024)](https://arxiv.org/abs/2406.11794) — controlled benchmark for data-curation strategies; DCLM-Baseline enables 64% MMLU at 7B with 2.6T tokens.
    - [Soldaini et al., *Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* (2024)](https://arxiv.org/abs/2402.00159) — most thoroughly documented open pipeline; per-document provenance, license tags, and multi-threshold quality signals.
    - [Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* (2023)](https://arxiv.org/abs/2305.10429) — group DRO on a proxy model finds domain weights that outperform human-tuned mixes by 6.5 pp on downstream tasks.
    - [Muennighoff et al., *Scaling Data-Constrained Language Models* (2023)](https://arxiv.org/abs/2305.16264) — quantifies how data repetition beyond ~4 epochs degrades generalization; introduces scaling laws for repeated-token regimes.

    **Open-source & tools**

    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — platform-agnostic, memory-efficient pipeline library (readers, filters, dedup, writers) used to build FineWeb; runs locally or on SLURM unchanged.
    - [allenai/dolma](https://github.com/allenai/dolma) — production-ready data curation toolkit with Rust Bloom-filter dedup, parallel taggers, and S3 support; backs the OLMo training corpus.
    - [mlfoundations/dclm](https://github.com/mlfoundations/dclm) — DCLM benchmark framework; 300T-token Common Crawl pool, 53-task evaluation suite, and OpenLM pretraining recipes across 411M–7B scales.
    - [togethercomputer/RedPajama-Data](https://github.com/togethercomputer/RedPajama-Data) — open replication of the LLaMA data recipe; v2 adds 40+ pre-computed quality signals to 30T tokens across five languages.

    **Go deeper**

    - [FineWeb: decanting the web for the finest text data at scale (HuggingFace blog, 2024)](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html) — detailed walkthrough of every filtering and deduplication decision, with ablation charts; excellent companion to the paper.

## Further Reading

- **Gao et al.** — *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*, EleutherAI, 2020. The template for documented open pretraining data.
- **Raffel et al.** — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*, JMLR, 2020. Introduced C4 and documented web-cleaning heuristics.
- **Soldaini et al.** — *Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research*, AI2, 2024. Most thoroughly documented open pipeline.
- **Penedo et al.** — *FineWeb: Decanting the Web for the Finest Text Data at Scale*, HuggingFace, 2024. Shows aggressive quality filtering can match curated mixes.
- **Xie et al.** — *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*, NeurIPS, 2023. Principled approach to mixture weight selection.
- **Muennighoff et al.** — *Scaling Data-Constrained Language Models*, NeurIPS, 2023. Quantifies the cost of data repetition.
- **Together AI** — *RedPajama*, 2023. Open replication of the LLaMA data recipe, including v2 with attached quality signals.
- **BigCode Project** — *The Stack*, 2022. Permissively licensed code corpus across 86 programming languages.
- **Common Crawl** — `commoncrawl.org`. Primary source of web data for nearly all open LLM corpora.
