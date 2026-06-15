# 3.2 Data Cleaning, Deduplication & Quality Filtering

Raw internet crawl data is not a dataset — it is a chaos of HTML artifacts, boilerplate menus, foreign-language text, spam, near-duplicate paragraphs, personally identifiable information (PII), toxic content, and accidental benchmark answers embedded in forum threads. The gap between a raw CommonCrawl snapshot and the clean, diverse, high-signal corpus that produces a capable language model is where most of the unreported work in pretraining happens. This chapter is a precise guide to every step in that gap.

We begin with the upstream work described in [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html), then hand off to [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) downstream. Quality of data directly governs sample efficiency: as [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) shows, the optimal token budget depends critically on the token distribution being worth training on.

## Language Identification

Before any quality heuristic is applied you need to know what language a document is written in. A heuristic that works for English prose (average word length > 4 characters, high fraction of common words) will misfire on German, Finnish, or Vietnamese. Language identification (LangID) is therefore the gating step.

**fastText LangID.** The `fastText` model trained on Wikipedia and other multilingual sources can classify text into 176 languages in roughly 1 microsecond per document. It operates on character $n$-grams, making it robust to typos and code-switching. A single call returns a label and a confidence score:

```python
import fasttext
import urllib.request, pathlib, tempfile

# Download the pre-trained fastText LangID model (lid.176.bin, ~131 MB).
# In production, bake this into a Docker image or model registry.
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

def load_langid_model(local_path="lid.176.bin"):
    """Download and cache the fastText LangID model."""
    if not pathlib.Path(local_path).exists():
        print(f"Downloading LangID model to {local_path} ...")
        urllib.request.urlretrieve(MODEL_URL, local_path)
    # IMPORTANT: suppress_stdou=True silences fastText's noisy banner.
    model = fasttext.load_model(local_path)
    return model

def detect_language(model, text: str, threshold: float = 0.65):
    """
    Return (lang_code, confidence) or ('__unknown__', 0.0) if below threshold.
    text: raw document string (newlines replaced with spaces for fastText).
    """
    # fastText expects a single line; truncate to 1000 chars for speed.
    snippet = text[:1000].replace("\n", " ")
    labels, probs = model.predict(snippet, k=1)
    lang = labels[0].replace("__label__", "")
    conf = float(probs[0])
    if conf < threshold:
        return "__unknown__", conf
    return lang, conf

# ---------- example usage ----------
# model = load_langid_model()
# lang, conf = detect_language(model, "Le chat est sur le tapis.")
# assert lang == "fr" and conf > 0.99
```

**Filtering strategy.** For English-centric corpora, retain documents where `lang == "en"` and `confidence >= 0.65`. For multilingual corpora (e.g., training a model like mT5 or BLOOM), keep all languages above threshold and track per-language token counts to enable upsampling of low-resource languages later.

## Heuristic Quality Filters

Heuristic filters are fast rule-based tests applied before any classifier. They remove the most obvious garbage — empty pages, boilerplate, machine-generated spam — at near-zero cost.

### Document-Level Heuristics

The following table lists standard heuristics used in production pipelines (inspired by C4, RefinedWeb, and Dolma):

| Filter | Rule | Typical threshold |
|---|---|---|
| Minimum length | `len(tokens) >= 50` | Drop very short docs |
| Maximum length | `len(tokens) <= 100_000` | Drop pathological docs |
| Mean word length | `3.0 <= mean_word_len <= 10.0` | Catches encoding garbage |
| Digit fraction | `digit_chars / total_chars <= 0.15` | Catches numeric tables/logs |
| Alphabetic fraction | `alpha_chars / total_chars >= 0.60` | Catches code/spam |
| Stop-word coverage | `stop_words_in_doc / total_words >= 0.10` | Catches word lists |
| Repeated n-gram ratio | `top_2gram_frac <= 0.20` | Catches boilerplate/footers |
| Bullet fraction | `lines_starting_with_bullet / lines <= 0.90` | Catches lists without prose |
| Curly brace fraction | `curly_braces / chars <= 0.01` | Catches leaked JSON/code |

```python
import re
import unicodedata
from collections import Counter

# Standard English stop words (abbreviated here for compactness).
STOP_WORDS = frozenset([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "she", "they", "we", "you", "this",
    "but", "from", "or", "an", "all", "so", "at", "if", "one", "would",
    "there", "their", "what", "out", "about", "up", "do", "was", "were",
])

def compute_heuristics(text: str) -> dict:
    """Compute a dictionary of quality signals for a single document."""
    words = text.split()
    chars = list(text)
    n_chars = len(chars) or 1
    n_words = len(words) or 1

    alpha_count = sum(1 for c in chars if c.isalpha())
    digit_count = sum(1 for c in chars if c.isdigit())
    curly_count = text.count("{") + text.count("}")

    # Mean word length (skip empty splits).
    word_lengths = [len(w) for w in words if w]
    mean_wl = sum(word_lengths) / n_words

    # Stop-word coverage.
    sw_count = sum(1 for w in words if w.lower() in STOP_WORDS)

    # Repeated 2-gram ratio.
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    bg_counter = Counter(bigrams)
    top_bg_frac = (bg_counter.most_common(1)[0][1] / len(bigrams)
                   if bigrams else 0.0)

    lines = text.splitlines()
    bullet_lines = sum(
        1 for l in lines if l.lstrip().startswith(("•", "-", "*", "·"))
    )

    return {
        "n_tokens": n_words,
        "mean_word_len": mean_wl,
        "alpha_frac": alpha_count / n_chars,
        "digit_frac": digit_count / n_chars,
        "stop_word_frac": sw_count / n_words,
        "top_bigram_frac": top_bg_frac,
        "bullet_frac": bullet_lines / max(len(lines), 1),
        "curly_frac": curly_count / n_chars,
    }

def passes_heuristic_filters(text: str) -> bool:
    """Return True if the document passes all heuristic quality filters."""
    h = compute_heuristics(text)
    return (
        50 <= h["n_tokens"] <= 100_000
        and 3.0 <= h["mean_word_len"] <= 10.0
        and h["alpha_frac"] >= 0.60
        and h["digit_frac"] <= 0.15
        and h["stop_word_frac"] >= 0.10
        and h["top_bigram_frac"] <= 0.20
        and h["bullet_frac"] <= 0.90
        and h["curly_frac"] <= 0.01
    )
```

### Line-Level and Paragraph-Level Filtering

Some noise lives at the sub-document level. For example, a Wikipedia article is mostly high quality, but the "References" section at the bottom is mostly citation boilerplate. C4 drops any line containing the phrase "javascript must be enabled" (a cookie/warning banner). RefinedWeb removes lines that contain mostly non-alphabetic characters or that are very short (under 20 characters) and lack terminal punctuation.

A practical approach: split each document into paragraphs, apply per-paragraph filters, then reassemble and re-check document-level minimums. This keeps the good parts of a noisy document rather than discarding the whole thing.

## Quality Classifiers

Heuristics filter the worst tail, but they cannot distinguish a mediocre product review from a well-reasoned essay. For that, we use quality classifiers: trained models that score a document's value as training data.

### Wikipedia-Trained Classifiers

The simplest approach, popularized by the GPT-3 data pipeline, trains a binary classifier to distinguish Wikipedia/Books (positive class) from random CommonCrawl text (negative class), then retains only documents above some score threshold. In practice, n-gram language models (using `KenLM`, for example) work surprisingly well for this: compute the perplexity of a document under a model trained on Wikipedia, and keep only low-perplexity documents.

$$
\text{score}(d) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P_\theta(w_i \mid w_{<i})\right)^{-1}
$$

A lower perplexity means the document resembles Wikipedia-quality text.

### Fasttext and Linear Classifiers

For speed at billion-document scale, fastText classifiers (with TF-IDF or hashing bag-of-words features) are common. They run in microseconds per document, enabling an entire CommonCrawl WET dump (on the order of a billion URLs, several hundred GB) to be classified in hours on a single machine.

The `CCNet` pipeline and its successors train a per-language classifier using Wikipedia as the positive signal, then keep documents above a chosen percentile threshold (e.g., keep the top 30% by score). The ROOTS corpus (used for BLOOM) went further, adding domain experts to annotate quality labels.

### Instruction-Following / Reward Model Classifiers

More recent pipelines use a reward model or an LLM-as-a-judge signal: a small fine-tuned model scores each document on dimensions like "educational value," "coherence," and "uniqueness." Phi-1 (Gunasekar et al., 2023) famously used GPT-4 to generate "textbook quality" synthetic documents and to score web crawl text for educational value, dramatically boosting sample efficiency. This approach is expensive but effective for smaller, higher-quality subsets.

## Toxicity Filtering and PII Removal

Training on toxic or private content is both an alignment risk and a legal risk. These two concerns require different tooling.

### Toxicity Filtering

Toxicity classifiers (Perspective API, Jigsaw's models, or custom fastText classifiers trained on annotated hate speech datasets) assign a probability that text is harmful, harassing, or offensive. The typical approach:

1. Score each document with a lightweight classifier.
2. Drop documents above a threshold (e.g., toxicity probability > 0.5).
3. For borderline cases, apply a more expensive model.

!!! warning "Toxicity filtering can introduce bias"
    Toxicity classifiers trained on English internet data systematically over-flag text mentioning LGBTQ+ identities, African-American Vernacular English (AAVE), and discussions of discrimination. Dropping all high-scoring documents can reduce representation of marginalized communities in the training corpus. Use per-category thresholds and audit the filter's demographic impact.

### PII Removal

Personally identifiable information (PII) includes names, email addresses, phone numbers, IP addresses, social security numbers, and similar. The standard pipeline:

```python
import re

# Compiled regex patterns for common PII types.
# In production, add a Named Entity Recognition (NER) model
# (e.g., spaCy en_core_web_trf) for name/address detection.

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
)
PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

PLACEHOLDER = {
    "email":   "EMAIL_ADDRESS",
    "phone":   "PHONE_NUMBER",
    "ssn":     "SSN",
    "ip":      "IP_ADDRESS",
}

def redact_pii(text: str) -> str:
    """Replace PII with category-specific placeholders."""
    text = EMAIL_RE.sub(PLACEHOLDER["email"], text)
    text = PHONE_RE.sub(PLACEHOLDER["phone"], text)
    text = SSN_RE.sub(PLACEHOLDER["ssn"], text)
    text = IP_RE.sub(PLACEHOLDER["ip"], text)
    return text

# Example:
# redact_pii("Contact me at alice@example.com or 555-867-5309.")
# -> "Contact me at EMAIL_ADDRESS or PHONE_NUMBER."
```

Regex handles structured PII well. Unstructured PII (names, addresses, partial account numbers) requires an NER model. The Dolma pipeline uses a combination of regex rules and a fine-tuned `mBERT`-based NER model. Bigram blocking (maintaining a blocklist of known high-frequency real names scraped from public records and blocking documents with many hits) is a cheaper complement.

## Exact Deduplication

Duplicate text is wasteful and actively harmful. A model that sees "The quick brown fox jumps over the lazy dog" ten thousand times will assign it disproportionate probability mass and may memorize it verbatim. Lee et al. (2022, "Deduplicating Training Data Makes Language Models Better") showed that deduplication reduces memorization, improves downstream benchmark performance, and allows training on smaller datasets without quality degradation.

**Exact deduplication** removes documents (or paragraphs) that share an identical byte sequence (or identical hash). Implementation:

```python
import hashlib
from collections import defaultdict

def exact_dedup_sha256(documents: list[str]) -> list[str]:
    """
    Remove exact duplicate documents using SHA-256 content hashing.
    Returns the first-seen copy of each unique document.
    """
    seen_hashes: set[str] = set()
    unique_docs: list[str] = []

    for doc in documents:
        # Normalize: strip leading/trailing whitespace and collapse
        # internal runs of whitespace to a single space before hashing.
        # This catches "same document, different formatting."
        normalized = " ".join(doc.split())
        h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    return unique_docs
```

For billion-document corpora, exact dedup is done with a distributed hash map (e.g., using Apache Spark with `groupByKey` on the hash, then dropping all but one per group) or with a sorted file of `(hash, doc_id)` pairs. The memory overhead is approximately 32 bytes per document for SHA-256 hashes, so 1 billion documents cost about 32 GB of memory — manageable with distributed tools.

## Fuzzy Deduplication: MinHash and LSH

Exact deduplication misses documents that are "almost" the same — two scraped copies of the same news article with slightly different HTML rendering, two forum threads with a shared boilerplate header, or a Wikipedia article and its mirror. For these we need *fuzzy* near-deduplication.

### Shingling and Jaccard Similarity

The Jaccard similarity between two sets $A$ and $B$ is:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

We model each document as a set of overlapping character $n$-grams (called *shingles*). Two documents with $J \geq 0.8$ are considered near-duplicates. The problem: computing all pairwise Jaccard similarities over $N$ documents takes $O(N^2)$ time — infeasible for billions of documents.

### MinHash

MinHash approximates $J(A, B)$ using random hashing. For each of $k$ independent hash functions $h_j$, compute:

$$
m_j(A) = \min_{s \in A} h_j(s)
$$

The *MinHash signature* of document $A$ is the vector $[m_1(A), m_2(A), \ldots, m_k(A)]$. The key property:

$$
P[m_j(A) = m_j(B)] = J(A, B)
$$

So the fraction of matching entries in two signatures is an unbiased estimator of Jaccard similarity.

### Locality Sensitive Hashing (LSH)

With $k=128$ hash functions and $N=10^9$ documents, we still cannot check every pair. LSH solves this by organizing signatures into *bands*: divide the 128-row signature into $b$ bands of $r$ rows each ($b \times r = 128$). Two documents are candidate pairs if their signatures agree on *all* $r$ rows in *at least one band*. The probability that a pair with true similarity $s$ becomes a candidate is:

$$
P(\text{candidate} \mid s) = 1 - (1 - s^r)^b
$$

This is an S-shaped function: pairs with similarity below a threshold $t^* \approx (1/b)^{1/r}$ rarely become candidates; pairs above rarely miss each other. By choosing $b$ and $r$ you tune the threshold.

!!! example "Worked example: MinHash parameter selection"
    Suppose we want a near-dedup threshold of $t^* = 0.8$ Jaccard with $k = 128$ hashes.

    Try $b = 16$ bands, $r = 8$ rows each. Then:

    $$
    t^* \approx \left(\frac{1}{b}\right)^{1/r} = \left(\frac{1}{16}\right)^{1/8} = 16^{-0.125} \approx 0.794
    $$

    - For a pair with $s = 0.80$: $P(\text{candidate}) = 1 - (1 - 0.80^8)^{16} = 1 - (1 - 0.168)^{16} \approx 1 - 0.046 = 0.954$. Nearly all such pairs are detected.
    - For a pair with $s = 0.50$: $P(\text{candidate}) = 1 - (1 - 0.50^8)^{16} = 1 - (1 - 0.0039)^{16} \approx 0.060$. Only 6% of these become candidates, limiting false-positive work.

    With $N = 10^9$ documents, the number of (band, hash-bucket) entries is $N \times b = 1.6 \times 10^{10}$. Collisions in a bucket trigger the expensive Jaccard verification step — but because the threshold is high, the number of buckets with more than one document is small.

```python
import hashlib
import struct
import random
from collections import defaultdict
from typing import List, Set, Dict, Tuple

# -----------------------------------------------------------------------
# MinHash from scratch — no external library required.
# -----------------------------------------------------------------------

def get_shingles(text: str, k: int = 5) -> Set[int]:
    """
    Compute character k-gram shingles as 32-bit integers.
    Using integers instead of strings saves memory and speeds hashing.
    k=5 works well for long documents; use k=3 for short texts.
    """
    normalized = " ".join(text.lower().split())
    shingles = set()
    for i in range(len(normalized) - k + 1):
        gram = normalized[i : i + k]
        # Map the gram to a 32-bit int via CRC-style hash.
        h = int(hashlib.md5(gram.encode()).hexdigest(), 16) & 0xFFFFFFFF
        shingles.add(h)
    return shingles


def make_hash_functions(num_hashes: int, seed: int = 42) -> List[Tuple[int, int]]:
    """
    Generate (a, b) parameters for the universal hash family:
        h_{a,b}(x) = (a * x + b) mod LARGE_PRIME
    LARGE_PRIME is a Mersenne prime (2^61 - 1).
    Returns a list of (a, b) pairs — one per hash function.
    """
    LARGE_PRIME = (1 << 61) - 1  # 2^61 - 1
    rng = random.Random(seed)
    params = []
    for _ in range(num_hashes):
        a = rng.randint(1, LARGE_PRIME - 1)
        b = rng.randint(0, LARGE_PRIME - 1)
        params.append((a, b))
    return params


def minhash_signature(
    shingles: Set[int],
    hash_params: List[Tuple[int, int]],
) -> List[int]:
    """
    Compute the MinHash signature for a set of shingles.
    For each hash function, the signature entry is min_{x in shingles} h(x).
    Time: O(|shingles| * num_hashes).  Can be vectorized with NumPy.
    """
    LARGE_PRIME = (1 << 61) - 1
    MAX_VAL = float("inf")
    sig = [MAX_VAL] * len(hash_params)

    for shingle in shingles:
        for j, (a, b) in enumerate(hash_params):
            h = (a * shingle + b) % LARGE_PRIME
            if h < sig[j]:
                sig[j] = h

    return [int(v) for v in sig]


def lsh_buckets(
    signatures: Dict[str, List[int]],
    num_bands: int,
    rows_per_band: int,
) -> Dict[Tuple, List[str]]:
    """
    Assign each document to LSH buckets (one per band).
    Returns a dict: bucket_id -> list of doc_ids that fall in that bucket.
    Two docs in the same bucket are candidate near-duplicates.
    """
    buckets: Dict[Tuple, List[str]] = defaultdict(list)
    for doc_id, sig in signatures.items():
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_key = (band_idx, tuple(sig[start:end]))
            buckets[band_key].append(doc_id)
    return buckets


def jaccard_from_sigs(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


def find_near_duplicates(
    documents: Dict[str, str],
    num_hashes: int = 128,
    num_bands: int = 16,
    threshold: float = 0.8,
    shingle_k: int = 5,
) -> List[Tuple[str, str, float]]:
    """
    Full MinHash-LSH near-dedup pipeline.
    Returns list of (doc_id_a, doc_id_b, estimated_jaccard) pairs
    that are near-duplicates (estimated Jaccard >= threshold).
    """
    assert num_hashes % num_bands == 0, "num_hashes must be divisible by num_bands"
    rows_per_band = num_hashes // num_bands

    print(f"[1/4] Computing shingles for {len(documents)} documents...")
    shingles_map = {doc_id: get_shingles(text, k=shingle_k)
                    for doc_id, text in documents.items()}

    print(f"[2/4] Computing MinHash signatures ({num_hashes} hashes)...")
    hash_params = make_hash_functions(num_hashes)
    sigs = {doc_id: minhash_signature(shingles, hash_params)
            for doc_id, shingles in shingles_map.items()}

    print(f"[3/4] Building LSH buckets ({num_bands} bands x {rows_per_band} rows)...")
    buckets = lsh_buckets(sigs, num_bands, rows_per_band)

    print(f"[4/4] Finding candidate pairs and verifying...")
    candidate_pairs: set = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                pair = tuple(sorted([bucket_docs[i], bucket_docs[j]]))
                candidate_pairs.add(pair)

    near_dups = []
    for (doc_a, doc_b) in candidate_pairs:
        est_j = jaccard_from_sigs(sigs[doc_a], sigs[doc_b])
        if est_j >= threshold:
            near_dups.append((doc_a, doc_b, est_j))

    return near_dups


# -----------------------------------------------------------------------
# Demo: five documents with known near-duplicate structure.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    docs = {
        "doc1": "The quick brown fox jumps over the lazy dog near the river bank.",
        "doc2": "The quick brown fox jumps over the lazy dog near the river banks.",  # near-dup of doc1
        "doc3": "A completely different document about machine learning and neural networks.",
        "doc4": "Another unique document discussing the history of the Roman Empire.",
        "doc5": "The quick brown fox jumps over a lazy dog close to the river bank.",  # near-dup of doc1/doc2
    }

    pairs = find_near_duplicates(docs, num_hashes=64, num_bands=8, threshold=0.5)
    for a, b, j in sorted(pairs, key=lambda x: -x[2]):
        print(f"  {a} <-> {b}  estimated Jaccard = {j:.3f}")
```

## Suffix Arrays for Exact Substring Deduplication

MinHash works at the document level. A more aggressive strategy deduplicates at the *substring* level: find long repeated sequences of tokens anywhere in the corpus and remove them. This catches boilerplate paragraphs that appear in otherwise unique documents, such as legal disclaimers, copyright notices, and license headers.

The canonical tool is the **suffix array** approach used by Lee et al. (2022) in their `deduplicate-text-datasets` toolkit. The algorithm:

1. Concatenate all documents into a single token sequence separated by sentinel tokens.
2. Build a suffix array (a sorted array of all suffixes).
3. Use the longest common prefix (LCP) array to identify runs of suffixes sharing a long prefix — these correspond to repeated sequences.
4. Mark and remove repeated spans.

Building a suffix array over tens of billions of tokens is non-trivial. The SA-IS algorithm runs in $O(n)$ time and $O(n)$ space, but with a large constant. For a 100-billion-token corpus, this requires on the order of 400 GB of RAM (4 bytes per token index). In practice, sharded suffix arrays (one per 10-billion-token shard, then cross-shard matching) are used.

```text
Example:
Corpus (simplified, characters): "abcde abc fg abc"
                                   0123456789...

Sorted suffixes (partial):
  "abc"         at positions 0, 6, 13
  "abcde abc fg" at position 0
  ...

LCP between consecutive entries tells us the length of the shared prefix.
Any shared prefix of length >= L (e.g., L=50 tokens) is a candidate repeated span.
```

The key tuning parameter is the minimum duplicate span length $L$. Shorter $L$ catches more duplicates but risks removing legitimately common short phrases. A common choice is $L = 50$ tokens (roughly one sentence).

## Near-Duplicate Detection Across Documents

A subtler problem is detecting near-duplicate *paragraphs* across different *unique* documents. For example, the same boilerplate "About Us" paragraph might appear in thousands of different company websites that are otherwise unique. Treating each company's website as a unique document would miss this repeated content.

The solution is to apply MinHash at the *paragraph* level rather than (or in addition to) the document level. The pipeline becomes:

{{fig:dedup-paragraph-minhash-pipeline}}

This two-level deduplication (document-level and paragraph-level) is used by the Dolma and DCLM pipelines and can reduce corpus size by 20–40% on CommonCrawl while retaining diverse documents.

## Benchmark Decontamination

A critical and often overlooked step is **benchmark decontamination**: removing from the training set any document that overlaps with evaluation benchmarks (HellaSwag, MMLU, GSM8K, HumanEval, etc.). If a model has seen the exact test questions or their answers during pretraining, its benchmark scores are inflated and meaningless.

### Detection

The standard approach uses $n$-gram overlap. For each benchmark example, extract a distinctive $n$-gram (typically $n = 13$ tokens), and scan the training corpus for documents containing that $n$-gram. Any training document with a match is flagged and removed.

```python
from collections import defaultdict

def build_ngram_index(
    benchmark_examples: list[str],
    n: int = 13,
) -> dict[tuple, list[int]]:
    """
    Build an index of n-grams from benchmark examples.
    Returns: ngram_tuple -> list of benchmark example indices.
    """
    index = defaultdict(list)
    for ex_idx, text in enumerate(benchmark_examples):
        tokens = text.lower().split()
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            index[gram].append(ex_idx)
    return index


def is_contaminated(
    doc: str,
    ngram_index: dict[tuple, list[int]],
    n: int = 13,
) -> bool:
    """
    Return True if any n-gram in doc matches a benchmark example.
    O(len(doc) * n) but n is constant, so O(len(doc)).
    """
    tokens = doc.lower().split()
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        if gram in ngram_index:
            return True
    return False
```

### How Much Contamination Exists?

Researchers studying C4, The Pile, and RedPajama consistently find that a non-trivial fraction of popular benchmark test sets appears verbatim in CommonCrawl-derived corpora. Removing contamination typically lowers reported benchmark numbers slightly, reflecting more honest evaluation. The important lesson: always report whether decontamination was performed and against which benchmarks.

!!! interview "Interview Corner"
    **Q:** A candidate claims their 7B model achieves state-of-the-art on MMLU without decontamination. Should you trust this number? How would you check?

    **A:** No — you should be skeptical. MMLU questions and answer keys appear in many internet sources; a model trained on contaminated data will score inflated numbers. To check: (1) extract the 13-gram fingerprint of each MMLU test example, (2) scan the training corpus for matches using the n-gram index approach, (3) if contamination rate exceeds ~1–2% of test examples, re-evaluate on a held-out contamination-free split or a different benchmark entirely. Legitimate model cards should report decontamination status.

## The Impact of Deduplication on Model Quality

Deduplication is not just about storage efficiency — it has measurable effects on model quality and behavior:

1. **Reduced memorization.** Carlini et al. (2021, "Extracting Training Data from Large Language Models") showed that verbatim memorization scales with duplication frequency. Removing near-duplicates is the most effective tool against training data extraction attacks.

2. **Improved perplexity and downstream performance.** Lee et al. (2022) showed that training on deduplicated data achieves the same loss as training on more data without deduplication — or equivalently, that deduplication improves sample efficiency by roughly 1.5–2x on their benchmarks.

3. **Better calibration.** When a model has seen the same passage many times, it assigns disproportionately high probability to that passage regardless of context. This hurts calibration and can cause models to hallucinate by completing "familiar" sequences even when they are wrong in context.

4. **Effective data budget.** For a Chinchilla-optimal run (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)), the training token budget is $T \approx 20 \times N$ (tokens $\approx$ 20 times parameters). If your corpus before dedup is, say, 5 trillion tokens but 40% are near-duplicates, your effective unique token budget is 3 trillion — a significant difference for a very large model.

5. **Faster training convergence.** Unique tokens per step carry more gradient signal per compute dollar than repeated tokens.

!!! example "Worked example: deduplication at CommonCrawl scale"
    Consider processing one CommonCrawl snapshot:

    - Raw WET files: approximately 100 billion tokens before filtering.
    - After language ID (keep English, threshold 0.65): approximately 60 billion tokens.
    - After heuristic filters: approximately 45 billion tokens.
    - After exact dedup (SHA-256 on normalized text): approximately 40 billion tokens (removes ~10% exact duplicates — mostly mirror sites and syndicated content).
    - After MinHash fuzzy dedup (threshold 0.8, 128 hashes, 16 bands): approximately 30 billion tokens (removes another ~25% near-duplicates).
    - After benchmark decontamination (13-gram, against 20 standard benchmarks): approximately 29.9 billion tokens (removes <0.1% but dramatically improves eval integrity).

    Final yield: roughly 30% of raw tokens, with substantially higher diversity, lower memorization risk, and honest benchmark evaluations. The yield ratio will vary — older crawls (2014–2018) tend to have more boilerplate; newer crawls have more unique content.

## Putting It All Together: A Production Pipeline

A production data-cleaning pipeline for pretraining typically runs in several distributed stages. The following pseudocode shows the logical order and the typical tool choices:

```bash
# Stage 0: Download WET files from CommonCrawl S3
aws s3 sync s3://commoncrawl/crawl-data/CC-MAIN-2024-10/segments/ ./wet/ \
  --include "*.warc.wet.gz" --request-payer requester

# Stage 1: Text extraction + language ID + heuristic filtering
# (Spark or Ray job, ~hours on 100-node cluster)
spark-submit data_pipeline/01_filter.py \
  --input ./wet/ \
  --output ./filtered/ \
  --lang en --lang-threshold 0.65

# Stage 2: Exact deduplication (distributed SHA-256 hash + groupBy)
spark-submit data_pipeline/02_exact_dedup.py \
  --input ./filtered/ \
  --output ./exact_dedup/

# Stage 3: MinHash fuzzy deduplication
# The deduplicate-text-datasets tool (Lee et al.) is the reference implementation.
python -m text_dedup.minhash \
  --path ./exact_dedup/ \
  --output ./fuzzy_dedup/ \
  --threshold 0.8 \
  --num_perm 128 \
  --b 16 --r 8

# Stage 4: Quality classification (fastText or KenLM scorer)
spark-submit data_pipeline/04_quality_score.py \
  --input ./fuzzy_dedup/ \
  --output ./quality_scored/ \
  --model quality_classifier.bin \
  --percentile 70  # keep top 30% by quality score

# Stage 5: PII removal + toxicity filter
python data_pipeline/05_pii_toxicity.py \
  --input ./quality_scored/ \
  --output ./clean/ \
  --toxicity-model perspective  # or local classifier

# Stage 6: Benchmark decontamination
python data_pipeline/06_decontaminate.py \
  --input ./clean/ \
  --benchmarks benchmarks/mmlu.jsonl benchmarks/gsm8k.jsonl \
  --ngram 13 \
  --output ./final/

# Stage 7: Tokenize and shard for training
python data_pipeline/07_tokenize.py \
  --input ./final/ \
  --tokenizer tokenizer.json \
  --output ./tokenized/ \
  --shard-size-gb 10
```

The total wall-clock time for this pipeline on one full CommonCrawl snapshot (roughly 80 TB of compressed WET files) is on the order of 24–72 hours on a 100-node Spark cluster, with the MinHash dedup stage being the most expensive due to the large number of (band, bucket) groupings.

For multi-source corpora (combining CommonCrawl with Books, GitHub, Wikipedia, arXiv, etc.), each source runs through its own adapted pipeline (e.g., GitHub needs a different toxicity model and no stop-word heuristics), and then cross-source deduplication removes documents that appear in multiple sources before merging.

!!! tip "Practitioner tip"
    Run the full pipeline on a 1% sample first. Check the yield at each stage, the language distribution, the quality score distribution, and spot-check 50–100 random documents from the final output. A bug in the heuristic filter (e.g., an off-by-one in the digit fraction check) can silently drop 30% of your corpus before you notice.

!!! key "Key Takeaways"
    - Language identification (fastText lid.176.bin) is the first gate; apply per-language heuristics only after LangID.
    - Heuristic filters (word length, digit fraction, stop-word coverage, repeated bigram ratio) cheaply remove the worst-quality 50–70% of a raw crawl.
    - Quality classifiers (KenLM perplexity, fastText trained on Wikipedia vs. crawl, or LLM-rated educational value) further improve data quality at the cost of compute.
    - Exact deduplication via SHA-256 hashing is trivial but removes only ~10% of duplicates; fuzzy dedup with MinHash + LSH is essential to catch the other 25–35%.
    - MinHash signatures estimate Jaccard similarity in $O(k)$ time; LSH reduces candidate pairs from $O(N^2)$ to near-linear by organizing signatures into bands.
    - Suffix array substring dedup (Lee et al., 2022) finds repeated spans across documents and is the most aggressive deduplication strategy.
    - Benchmark decontamination via 13-gram fingerprinting is a mandatory step for honest evaluation; contamination in CommonCrawl-based corpora is non-trivial.
    - Deduplication improves model quality beyond just storage savings: it reduces memorization, improves downstream task performance, and makes calibration better.
    - Always run the pipeline on a sample first and audit yield, language distribution, and spot-checked documents before committing to a full run.

!!! sota "State of the Art & Resources (2026)"
    Web-scale data curation has matured into a disciplined engineering subdiscipline: MinHash + suffix-array deduplication, LLM-scored quality classifiers, and rigorous benchmark decontamination are now standard practice, with open tooling (Dolma toolkit, datatrove, DCLM) making reproducible pipelines accessible to independent researchers.

    **Foundational work**

    - [Lee et al., *Deduplicating Training Data Makes Language Models Better* (2022)](https://arxiv.org/abs/2107.06499) — landmark study establishing MinHash + suffix-array dedup as the standard; releases the `deduplicate-text-datasets` Rust toolkit.
    - [Wenzek et al., *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data* (2020)](https://arxiv.org/abs/1911.00359) — introduced the LangID → KenLM perplexity scoring pipeline that underlies most subsequent web-crawl filters.
    - [Carlini et al., *Extracting Training Data from Large Language Models* (2021)](https://arxiv.org/abs/2012.07805) — demonstrates how duplication frequency drives verbatim memorization, providing the privacy motivation for aggressive dedup.

    **Recent advances (2023–2026)**

    - [Penedo et al., *The RefinedWeb Dataset for Falcon LLM* (2023)](https://arxiv.org/abs/2306.01116) — shows that aggressive heuristic filtering and near-dedup on web data alone can outperform curated multi-source corpora; describes a 5T-token production pipeline.
    - [Soldaini et al., *Dolma: an Open Corpus of Three Trillion Tokens* (2024)](https://arxiv.org/abs/2402.00159) — fully open 3T-token corpus with detailed ablations of each cleaning stage (heuristics, PII, dedup, decontamination).
    - [Li et al., *DataComp-LM: In Search of the Next Generation of Training Sets* (2024)](https://arxiv.org/abs/2406.11794) — controlled benchmark for data-curation strategies across model scales; finds model-based filtering outperforms heuristics alone.
    - [Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (2024)](https://arxiv.org/abs/2406.17557) — 15T-token open dataset built with full pipeline transparency; FineWeb-Edu shows LLM-annotated quality scores dramatically improve downstream educational benchmarks.

    **Open-source & tools**

    - [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) — reference Rust + Python implementation of suffix-array exact-substring dedup from Lee et al.
    - [allenai/dolma](https://github.com/allenai/dolma) — production-grade data curation toolkit (Bloom-filter dedup, Gopher/C4 taggers, PII removal) used to build the OLMo pretraining corpus.
    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — modular pipeline library (filtering, MinHash dedup, extraction) that runs locally or on SLURM/Ray; used to build FineWeb.

## Further Reading

- **Lee et al., "Deduplicating Training Data Makes Language Models Better" (2022)** — the landmark study on MinHash + suffix array dedup for LLM pretraining; includes the `deduplicate-text-datasets` open-source toolkit.
- **Joulin et al., "FastText.zip: Compressing text classification models" (2016)** and the companion **"Bag of Tricks for Efficient Text Classification" (2016)** — the basis for fastText LangID and quality classifiers.
- **Wenzek et al., "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data" (2020)** — the CCNet pipeline: LangID + KenLM quality scoring for multilingual data.
- **Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020, "C4" dataset section)** — describes the C4 dataset filters that became an industry baseline.
- **Soldaini et al., "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (2024)** — detailed discussion of heuristic filters, PII removal, and decontamination at scale.
- **Gunasekar et al., "Textbooks Are All You Need" (2023)** — shows how LLM-rated quality classifiers can dramatically improve sample efficiency.
- **Carlini et al., "Extracting Training Data from Large Language Models" (2021)** — demonstrates the connection between duplication frequency and memorization; motivates dedup as a privacy tool.
- **Penedo et al., "The RefinedWeb Dataset for Falcon LLM" (2023)** — production-quality pipeline with aggressive near-dedup and quality filters for a 5-trillion-token English corpus.
