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
    # fasttext.load_model takes no quiet/suppress argument; silence its noisy
    # C++ banner by monkeypatching the eprint hook before loading instead.
    fasttext.FastText.eprint = lambda *args, **kwargs: None
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
\text{PPL}(d) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P_\theta(w_i \mid w_{<i})\right)
$$

A lower perplexity means the document resembles Wikipedia-quality text, so CCNet-style pipelines keep the documents whose per-language perplexity falls below a percentile threshold (for example, the lowest-perplexity 30%). If you prefer a score where higher is better, use the reciprocal $1/\text{PPL}(d)$ and keep the top percentile instead -- just state the direction explicitly so it matches the threshold logic.

### Fasttext and Linear Classifiers

For speed at billion-document scale, fastText classifiers (with TF-IDF or hashing bag-of-words features) are common. They run in microseconds per document, enabling an entire CommonCrawl WET dump (on the order of a billion URLs, several hundred GB) to be classified in hours on a single machine.

The `CCNet` pipeline and its successors train a per-language classifier using Wikipedia as the positive signal, then keep documents above a chosen percentile threshold (e.g., keep the top 30% by score). The ROOTS corpus (used for BLOOM) went further, adding domain experts to annotate quality labels.

Here is a compact but complete trainable classifier: Wikipedia paragraphs are the positive class, raw CommonCrawl paragraphs the negative class, and we keep the top percentile by predicted quality on a held-out split (the CCNet recipe). It uses `HashingVectorizer` so there is no vocabulary to store -- fixed memory regardless of corpus size -- with a `LogisticRegression` head; the commented block shows the faster `fasttext.train_supervised` alternative used at billion-document scale.

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_quality_classifier(wiki_docs, crawl_docs, n_features=2**20):
    # Hashed word 1+2-grams -> no stored vocab, constant memory.
    vec = HashingVectorizer(n_features=n_features, ngram_range=(1, 2),
                            alternate_sign=False, norm="l2")
    X = vec.transform(wiki_docs + crawl_docs)
    y = np.array([1] * len(wiki_docs) + [0] * len(crawl_docs))
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
    return vec, clf

def quality_scores(vec, clf, docs):
    return clf.predict_proba(vec.transform(docs))[:, 1]    # P(quality) in [0, 1]

def pick_threshold(heldout_scores, keep_frac=0.30):
    # Keep the top `keep_frac` by score: threshold = the (1 - keep_frac) quantile.
    return float(np.quantile(heldout_scores, 1.0 - keep_frac))

# ---- usage ----
# vec, clf = train_quality_classifier(wiki_paragraphs, raw_cc_paragraphs)
# thr = pick_threshold(quality_scores(vec, clf, heldout_docs), keep_frac=0.30)
# keep = [d for d in corpus if quality_scores(vec, clf, [d])[0] >= thr]  # top 30%
#
# fastText alternative (microseconds/doc at billion-doc scale):
#   with open("train.txt", "w") as f:
#       for d in wiki_docs:  f.write("__label__hq " + d.replace("\n", " ") + "\n")
#       for d in crawl_docs: f.write("__label__lq " + d.replace("\n", " ") + "\n")
#   m = fasttext.train_supervised("train.txt", epoch=5, wordNgrams=2, dim=100)
#   labels, probs = m.predict(doc, k=2)          # score = P(__label__hq)
#   p_hq = dict(zip(labels, probs)).get("__label__hq", 0.0)
```

For an embedding-based classifier -- the FineWeb-Edu recipe, where a strong LLM labels a seed set and a light head learns to reproduce the scores -- see the Ridge-head implementation in [Synthetic Data for Pretraining](../03-pretraining/15-synthetic-data.html). There the `embed_fn` is left abstract; a concrete choice is `sentence-transformers/all-MiniLM-L6-v2` (call `SentenceTransformer("all-MiniLM-L6-v2").encode(list_of_texts)` to get the embedding matrix), while FineWeb-Edu itself embedded with `Snowflake/snowflake-arctic-embed-m` before the Ridge head.

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

## Bloom-Filter Deduplication

SHA-256 exact dedup needs a hash *set* in memory -- 32 bytes per document, so a billion documents cost about 32 GB (above). A **Bloom filter** does the same membership test probabilistically in a fraction of the space. It is a bit array of $m$ bits with $k$ independent hash functions: to insert an item, set the $k$ bits it hashes to; to test membership, check whether all $k$ bits are already set.

The filter has **no false negatives** -- a set bit is never cleared, so a true duplicate is always reported as "seen" -- but a tunable **false-positive** rate. After inserting $n$ items into $m$ bits with $k$ hashes, the probability that a fresh item collides on all $k$ bits is

$$
p \approx \left(1 - e^{-kn/m}\right)^{k}.
$$

Minimizing over $k$ gives the optimal sizing

$$
\frac{m}{n} = -\frac{\ln p}{(\ln 2)^2}, \qquad k = \frac{m}{n}\ln 2.
$$

For a target $p = 10^{-6}$ this is $m/n = -\ln(10^{-6})/(\ln 2)^2 \approx 28.8$ bits per element and $k \approx 20$ hash functions. For $n = 10^9$ documents that is $2.88 \times 10^{10}$ bits $\approx 3.6$ GB -- versus 32 GB for the full SHA-256 set, roughly a 9x saving. The asymmetry is exactly what dedup wants: a false positive wrongly drops a *unique* document (you lose about 1 in $10^{6}$ unique docs at $p = 10^{-6}$), while a true duplicate is *never* kept. Choose $p$ to bound the unique-document loss you can tolerate.

```python
import hashlib, math

class BloomFilter:
    def __init__(self, n_items: int, fp_rate: float = 1e-6):
        # Optimal sizing (Bloom, 1970):
        #   m = -n ln p / (ln 2)^2  bits ;  k = (m/n) ln 2  hash functions
        self.m = max(1, math.ceil(-n_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.k = max(1, round((self.m / max(n_items, 1)) * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _indices(self, item: str):
        # Kirsch-Mitzenmacher double hashing: derive k indices from two 64-bit
        # halves of one SHA-256 digest. No false negatives are possible.
        d = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(d[:8], "big")
        h2 = int.from_bytes(d[8:16], "big") | 1        # keep h2 odd
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: str) -> None:
        for idx in self._indices(item):
            self.bits[idx >> 3] |= (1 << (idx & 7))

    def __contains__(self, item: str) -> bool:
        return all(self.bits[idx >> 3] & (1 << (idx & 7))
                   for idx in self._indices(item))


def bloom_dedup(documents: list[str]) -> list[str]:
    """Single-pass streaming dedup in O(m) memory, independent of doc size.
    No false negatives => a true duplicate is never emitted twice; a false
    positive (rate ~fp_rate) drops a unique doc. Uses the same whitespace
    normalization as exact_dedup_sha256 so the two agree on exact duplicates."""
    bf = BloomFilter(n_items=len(documents), fp_rate=1e-6)
    out = []
    for doc in documents:
        key = " ".join(doc.split())
        if key in bf:            # almost certainly a duplicate -> skip
            continue
        bf.add(key)
        out.append(doc)
    return out


# Verification: zero false negatives, and empirical FP rate near the target.
if __name__ == "__main__":
    bf = BloomFilter(n_items=100_000, fp_rate=1e-3)
    inserted = {f"doc-{i}" for i in range(100_000)}
    for s in inserted:
        bf.add(s)
    assert all(s in bf for s in inserted)                 # zero false negatives
    fp = sum((f"absent-{i}" in bf) for i in range(100_000)) / 100_000
    print(f"k={bf.k}  bits/elt={bf.m/100_000:.2f}  empirical FP={fp:.4f}")
    # Prints: k=10  bits/elt=14.38  empirical FP=0.0010  (target 0.0010)
```

This is exactly the mechanism `allenai/dolma` uses (a Rust Bloom filter) for both exact-document and paragraph-level dedup: it streams the corpus once, keeping only a bit array in memory rather than a growing hash set, which is what makes single-machine dedup of trillion-token corpora feasible. The trade-off versus MinHash is that a Bloom filter tests *exact* (normalized) equality -- it replaces the SHA-256 set, not the fuzzy near-dedup that follows.

{{fig:bloom-filter-dedup-asymmetry}}

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

{{fig:minhash-jaccard-estimator}}

### Locality Sensitive Hashing (LSH)

With $k=128$ hash functions and $N=10^9$ documents, we still cannot check every pair. LSH solves this by organizing signatures into *bands*: divide the 128-row signature into $b$ bands of $r$ rows each ($b \times r = 128$). Two documents are candidate pairs if their signatures agree on *all* $r$ rows in *at least one band*. The probability that a pair with true similarity $s$ becomes a candidate is:

$$
P(\text{candidate} \mid s) = 1 - (1 - s^r)^b
$$

This is an S-shaped function: pairs with similarity below a threshold $t^* \approx (1/b)^{1/r}$ rarely become candidates; pairs above rarely miss each other. By choosing $b$ and $r$ you tune the threshold.

!!! example "Worked example: MinHash parameter selection"
    Suppose we want to catch near-duplicates around $0.8$ Jaccard with $k = 128$ hashes, and we will size the bands to sit just under that target.

    Try $b = 16$ bands, $r = 8$ rows each ($b \times r = 128$). Then:

    $$
    t^* \approx \left(\frac{1}{b}\right)^{1/r} = \left(\frac{1}{16}\right)^{1/8} = 16^{-1/8} = 2^{-1/2} \approx 0.707
    $$

    So $b = 16, r = 8$ actually places the S-curve inflection near $0.71$ Jaccard -- slightly *below* the 0.8 target, not at it. That is a deliberately conservative choice: it turns pairs down to about $0.71$ into candidates, and the exact signature-match verification step (the `est_j >= threshold` check with `threshold = 0.8` in the code below) then discards those that fall short of 0.8. Sizing the bands so the LSH threshold sits just under your true target is standard practice -- you would rather pay to verify a few extra candidates than miss a genuine duplicate.

    - For a pair with $s = 0.80$: $P(\text{candidate}) = 1 - (1 - 0.80^8)^{16} = 1 - (1 - 0.168)^{16} \approx 1 - 0.053 = 0.947$. Nearly all such pairs become candidates.
    - For a pair with $s = 0.50$: $P(\text{candidate}) = 1 - (1 - 0.50^8)^{16} = 1 - (1 - 0.0039)^{16} \approx 1 - 0.939 = 0.061$. Only ~6% of these reach the verification step, limiting false-positive work.

    If you instead wanted the inflection right at 0.8, pick fewer, longer bands -- e.g. $b = 8, r = 16$ gives $t^* = 8^{-1/16} = 2^{-3/16} \approx 0.878$, catching only very close duplicates. The $(b, r)$ split is the knob that trades recall against verification cost.

    With $N = 10^9$ documents, the number of (band, hash-bucket) entries is $N \times b = 1.6 \times 10^{10}$. Collisions in a bucket trigger the expensive Jaccard verification step — but because the threshold is high, the number of buckets with more than one document is small.

{{fig:lsh-banding-scurve}}

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
    # Edge case: a document shorter than the shingle length k (or an empty /
    # whitespace-only document) produces no shingles. The int(float('inf'))
    # conversion below would then raise OverflowError, so return a sentinel
    # signature and let the caller route such docs to exact dedup only.
    if not shingles:
        return [-1] * len(hash_params)
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
    # Drop docs too short to yield any shingle (len(text) < shingle_k): they
    # cannot be fuzzy-deduped and would otherwise crash minhash_signature.
    # Handle these via exact dedup only.
    short = [d for d, s in shingles_map.items() if not s]
    if short:
        print(f"      skipping {len(short)} doc(s) shorter than k={shingle_k}")
    shingles_map = {d: s for d, s in shingles_map.items() if s}

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

    # r = num_hashes / num_bands = 64 / 16 = 4, so the LSH threshold is
    # t* = (1/16)^(1/4) = 0.5 -- low enough to also surface doc5 (J ~ 0.6).
    # With num_bands=8 the threshold is ~0.77 and doc5 would be missed.
    pairs = find_near_duplicates(docs, num_hashes=64, num_bands=16, threshold=0.5)
    for a, b, j in sorted(pairs, key=lambda x: -x[2]):
        print(f"  {a} <-> {b}  estimated Jaccard = {j:.3f}")

    # ---- Expected output (deterministic: make_hash_functions uses seed=42) ----
    #   doc1 <-> doc2  estimated Jaccard = 0.938
    #   doc1 <-> doc5  estimated Jaccard = 0.594
    #   doc2 <-> doc5  estimated Jaccard = 0.547
    # doc3 and doc4 share no near-duplicate, so they never appear.

    # ---- Verification: MinHash is an unbiased Jaccard estimator whose standard
    # error is ~1/sqrt(num_hashes) = 1/sqrt(64) = 0.125. Check the signature
    # estimate against the exact set Jaccard for the closest pair. ----
    def exact_jaccard(a: str, b: str, k: int = 5) -> float:
        sa, sb = get_shingles(a, k), get_shingles(b, k)
        return len(sa & sb) / len(sa | sb)

    hp = make_hash_functions(64)
    s1 = minhash_signature(get_shingles(docs["doc1"]), hp)
    s2 = minhash_signature(get_shingles(docs["doc2"]), hp)
    est = jaccard_from_sigs(s1, s2)
    exact = exact_jaccard(docs["doc1"], docs["doc2"])
    print(f"doc1<->doc2 check: est={est:.3f} exact={exact:.3f} err={abs(est-exact):.3f}")
    assert abs(est - exact) <= 3 / (64 ** 0.5), "estimate outside 3 standard errors"
    # Prints: doc1<->doc2 check: est=0.938 exact=0.950 err=0.012
```

## Suffix Arrays for Exact Substring Deduplication

MinHash works at the document level. A more aggressive strategy deduplicates at the *substring* level: find long repeated sequences of tokens anywhere in the corpus and remove them. This catches boilerplate paragraphs that appear in otherwise unique documents, such as legal disclaimers, copyright notices, and license headers.

The canonical tool is the **suffix array** approach used by Lee et al. (2022) in their `deduplicate-text-datasets` toolkit. The algorithm:

1. Concatenate all documents into a single token sequence separated by sentinel tokens.
2. Build a suffix array (a sorted array of all suffixes).
3. Use the longest common prefix (LCP) array to identify runs of suffixes sharing a long prefix — these correspond to repeated sequences.
4. Mark and remove repeated spans.

Building a suffix array over tens of billions of tokens is non-trivial. The SA-IS algorithm runs in $O(n)$ time and $O(n)$ space, but with a large constant. For a 100-billion-token corpus a 32-bit index is not enough: it can address only $2^{32} \approx 4.3$ billion positions, far short of $10^{11}$. You need 8-byte (int64) indices, so the suffix array alone is about $8 \times 10^{11}$ bytes $\approx 800$ GB of RAM -- plus the token stream itself. This is exactly why sharded suffix arrays (one per 10-billion-token shard, then cross-shard matching) are used.

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

### Measuring the effect: a runnable ablation

The numbers above are quoted from the literature. To *measure* the effect yourself, train two identical small models -- one on raw data, one on filtered+deduped data -- for the **same token budget**, then compare held-out perplexity and a zero-shot benchmark. The harness below reuses the filtering and dedup functions from this chapter to build the two corpora, trains a GPT-2-small (~124M) twice, and reports the deltas.

```python
# pip install torch transformers datasets
import math, torch
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
                          get_cosine_schedule_with_warmup)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

def build_model(n_layer=12, n_embd=768, n_head=12):
    # ~124M GPT-2 small. Shrink (n_layer=4, n_embd=256, n_head=4 -> ~11M)
    # for a CPU/laptop toy run.
    cfg = GPT2Config(vocab_size=len(tok), n_positions=1024, n_ctx=1024,
                     n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    return GPT2LMHeadModel(cfg).to(DEVICE)

def pack_tokens(docs, block=1024):
    ids = []
    for d in docs:
        ids.extend(tok(d)["input_ids"] + [tok.eos_token_id])
    n = (len(ids) // block) * block
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, block)

def train(model, blocks, steps, bs=8, lr=3e-4):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    for step in range(steps):
        idx = torch.randint(0, blocks.size(0), (bs,))
        batch = blocks[idx].to(DEVICE)
        loss = model(input_ids=batch, labels=batch).loss   # HF shifts internally
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % 500 == 0:
            print(f"  step {step:6d}  train loss {loss.item():.3f}")
    return model

@torch.no_grad()
def heldout_ppl(model, blocks, bs=8):
    model.eval(); tot, ntok = 0.0, 0
    for i in range(0, blocks.size(0), bs):
        batch = blocks[i:i + bs].to(DEVICE)
        loss = model(input_ids=batch, labels=batch).loss   # mean NLL/token
        tot += loss.item() * batch.numel(); ntok += batch.numel()
    return math.exp(tot / ntok)

@torch.no_grad()
def loglikelihood(model, context, continuation):
    # sum log P(continuation | context) over continuation tokens
    ctx = tok(context)["input_ids"]
    cont = tok(continuation)["input_ids"]
    ids = torch.tensor([ctx + cont], device=DEVICE)
    logp = torch.log_softmax(model(ids).logits[0], dim=-1)  # (T, V)
    total = 0.0
    for i, tid in enumerate(cont):
        total += logp[len(ctx) + i - 1, tid].item()        # predicted from prev pos
    return total, len(cont)

def eval_multiple_choice(model, examples):
    # examples: [{"ctx": str, "endings": [str, ...], "label": int}, ...]
    # length-normalized log-likelihood == lm-eval-harness `acc_norm`.
    correct = 0
    for ex in examples:
        best_j, best_norm = -1, -1e30
        for j, end in enumerate(ex["endings"]):
            ll, n = loglikelihood(model, ex["ctx"], " " + end)
            norm = ll / max(n, 1)
            if norm > best_norm:
                best_norm, best_j = norm, j
        correct += (best_j == ex["label"])
    return correct / len(examples)

def run_ablation(raw_docs, clean_docs, heldout_docs, mc_examples,
                 steps=4000, bs=8, block=1024):
    results = {}
    for name, docs in [("raw", raw_docs), ("filtered+deduped", clean_docs)]:
        print(f"=== training on {name} ({len(docs)} docs) ===")
        blocks = pack_tokens(docs, block)
        model = train(build_model(), blocks, steps=steps, bs=bs)
        ppl = heldout_ppl(model, pack_tokens(heldout_docs, block), bs=bs)
        acc = eval_multiple_choice(model, mc_examples)
        results[name] = (ppl, acc)
        print(f"  {name}: held-out PPL={ppl:.2f}  acc_norm={acc:.3f}")
    (p_raw, a_raw), (p_cl, a_cl) = results["raw"], results["filtered+deduped"]
    print(f"\nDELTA held-out PPL: {p_raw:.2f} -> {p_cl:.2f} "
          f"({100 * (p_raw - p_cl) / p_raw:+.1f}%)")
    print(f"DELTA benchmark acc: {a_raw:.3f} -> {a_cl:.3f} "
          f"({100 * (a_cl - a_raw):+.1f} pts)")
    return results

# raw_docs   = list of raw crawl strings
# clean_docs = [d for d in raw_docs if passes_heuristic_filters(d)]
#              then exact_dedup_sha256(...) then MinHash near-dedup (this chapter)
# heldout_docs = a clean, deduped, held-out slice NOT in either training set
# mc_examples: load real benchmarks, e.g.
#   from datasets import load_dataset
#   hs = load_dataset("hellaswag", split="validation").select(range(500))
#   mc = [{"ctx": e["ctx"], "endings": e["endings"], "label": int(e["label"])}
#         for e in hs]
# run_ablation(raw_docs, clean_docs, heldout_docs, mc, steps=4000)
```

**Hardware spectrum and expected magnitudes.**

- *Laptop / CPU toy* (~11M model via `build_model(4, 256, 4)`, `block=256`, `steps=500`, a few tens of MB of text): runs in minutes. The held-out PPL delta is directionally correct but noisy; benchmark accuracy sits at chance (HellaSwag `acc_norm` ~25%, PIQA ~50%) -- too small to move the benchmark, so read the PPL delta only.
- *Single GPU* (A100/4090, 124M model, `bs=8`, `block=1024`, `steps=122000` ~ 1B tokens, ~7 hours): expect a held-out PPL improvement of roughly **5-15%** for filtered+deduped vs raw; the benchmark move is small and often within noise at this scale, so average over 3 seeds.
- *8-GPU node* (DDP, 160M model, effective `bs=64`, 5-10B tokens, a few hours): the PPL gap is clear and HellaSwag/PIQA `acc_norm` typically moves **+1-3 points**.
- *Multi-node* (1B params, 30B+ tokens): this is the DCLM/FineWeb regime where good filtering vs raw CommonCrawl moves an aggregate benchmark suite by **several points** reliably -- the scale at which the published 1.5-2x sample-efficiency and downstream gains show up cleanly.

**Verification -- establish a noise floor first.** Before trusting a raw-vs-clean delta, train two models on the *same* corpus with different seeds; the PPL gap between them is your noise floor. Only a raw-vs-clean delta that exceeds that floor is real. This same-data control is why single-GPU ablations should report a 3-seed mean rather than a single number. The multiple-choice scorer above is the length-normalized log-likelihood metric implemented by [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) (`acc_norm`); for publishable numbers run that harness rather than the toy scorer, which omits its request-batching and prompt templates.

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

The total wall-clock time for this pipeline on one full CommonCrawl snapshot (roughly 9 TB of compressed WET text extracts -- the full WARC set for the same monthly crawl is about 90 TB) is on the order of 24–72 hours on a 100-node Spark cluster, with the MinHash dedup stage being the most expensive due to the large number of (band, bucket) groupings.

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

{{fig:data-cleaning-cascade-funnel}}

## Further Reading

- **Lee et al., "Deduplicating Training Data Makes Language Models Better" (2022)** — the landmark study on MinHash + suffix array dedup for LLM pretraining; includes the `deduplicate-text-datasets` open-source toolkit.
- **Joulin et al., "FastText.zip: Compressing text classification models" (2016)** and the companion **"Bag of Tricks for Efficient Text Classification" (2016)** — the basis for fastText LangID and quality classifiers.
- **Wenzek et al., "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data" (2020)** — the CCNet pipeline: LangID + KenLM quality scoring for multilingual data.
- **Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020, "C4" dataset section)** — describes the C4 dataset filters that became an industry baseline.
- **Soldaini et al., "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research" (2024)** — detailed discussion of heuristic filters, PII removal, and decontamination at scale.
- **Gunasekar et al., "Textbooks Are All You Need" (2023)** — shows how LLM-rated quality classifiers can dramatically improve sample efficiency.
- **Carlini et al., "Extracting Training Data from Large Language Models" (2021)** — demonstrates the connection between duplication frequency and memorization; motivates dedup as a privacy tool.
- **Penedo et al., "The RefinedWeb Dataset for Falcon LLM" (2023)** — production-quality pipeline with aggressive near-dedup and quality filters for a 5-trillion-token English corpus.

## Exercises

**1.** Language identification is described as the *gating* step: LangID runs before any heuristic quality filter. Explain why the order matters. What specifically goes wrong if you apply the English-tuned document-level heuristics from the table (mean word length in $[3.0, 10.0]$, stop-word coverage $\geq 0.10$) directly to a raw multilingual crawl without a LangID gate first?

??? note "Solution"
    The heuristics in the table encode assumptions about English prose. Two of them are language-specific in a way that makes them misfire badly on other languages:

    - **Stop-word coverage $\geq 0.10$.** The `STOP_WORDS` set is English function words ("the", "of", "and", ...). A perfectly clean German, Finnish, or Vietnamese document contains essentially none of them, so `stop_word_frac` is near $0$ and the document is dropped as a "word list" even though it is high-quality prose. Without a LangID gate you would silently delete most non-English text.
    - **Mean word length in $[3.0, 10.0]$.** This is calibrated to English. Agglutinative or compounding languages (Finnish, German, Turkish) routinely exceed a mean of $10$ characters per word; languages written without spaces between words (Chinese, Japanese, Thai) break `text.split()` in the opposite direction. Either way the filter's threshold is meaningless without knowing the language.

    LangID first lets the pipeline (a) route each document to the correct per-language thresholds (or the correct stop-word list), and (b) for an English-centric corpus, cheaply discard non-English documents *before* spending any compute on heuristics that would reject them anyway for the wrong reason. This is exactly the CCNet ordering: LangID, then per-language filtering. It also makes per-language token accounting possible, which is what enables later upsampling of low-resource languages.

**2.** Consider the spam document `text = "cheap deals cheap deals cheap deals cheap deals cheap deals"` (the two-word phrase "cheap deals" repeated five times). Using the definitions in `compute_heuristics`, compute by hand: (a) `mean_word_len`, (b) `stop_word_frac`, and (c) `top_bigram_frac`. Then, using `passes_heuristic_filters`, list every check this document fails.

??? note "Solution"
    The document has 10 whitespace-separated tokens: five copies each of "cheap" (5 letters) and "deals" (5 letters).

    **(a) `mean_word_len`.** Every word has length $5$, and `n_words = 10`, so

    $$
    \text{mean\_word\_len} = \frac{\sum_i |w_i|}{n\_words} = \frac{10 \times 5}{10} = 5.0 .
    $$

    This is inside $[3.0, 10.0]$, so it passes that check.

    **(b) `stop_word_frac`.** Neither "cheap" nor "deals" is in `STOP_WORDS`, so `sw_count = 0` and

    $$
    \text{stop\_word\_frac} = \frac{0}{10} = 0.0 .
    $$

    **(c) `top_bigram_frac`.** There are `len(words) - 1 = 9` bigrams. They alternate `"cheap deals"` and `"deals cheap"`; `"cheap deals"` occurs at the 5 even positions ($0,2,4,6,8$), so its count is $5$:

    $$
    \text{top\_bigram\_frac} = \frac{5}{9} \approx 0.556 .
    $$

    **Failed checks.** Running `passes_heuristic_filters`:

    - `50 <= n_tokens` fails: `n_tokens = 10 < 50` (minimum length).
    - `stop_word_frac >= 0.10` fails: $0.0 < 0.10$.
    - `top_bigram_frac <= 0.20` fails: $0.556 > 0.20$.

    (The other checks pass: `alpha_frac = 50/59 \approx 0.85 \geq 0.60`, `digit_frac = 0`, `mean_word_len = 5.0`, `bullet_frac = 0`, `curly_frac = 0`.) The document trips three independent filters, which is the point: obvious keyword-spam is caught redundantly, so no single threshold has to be perfectly tuned.

**3.** You are Bloom-filter deduplicating $n = 2 \times 10^{8}$ documents at a target false-positive rate $p = 10^{-4}$. Using the chapter's sizing formulas, compute: (a) the bits per element $m/n$, (b) the number of hash functions $k$, (c) the total memory in GB, and (d) the expected number of *unique* documents wrongly dropped. Compare the memory to the SHA-256 hash-set approach (32 bytes per document).

??? note "Solution"
    **(a) Bits per element.** From $\dfrac{m}{n} = -\dfrac{\ln p}{(\ln 2)^2}$ with $p = 10^{-4}$:

    $$
    \frac{m}{n} = -\frac{\ln(10^{-4})}{(\ln 2)^2} = \frac{9.210}{0.4805} \approx 19.2 \text{ bits/element}.
    $$

    **(b) Hash functions.** $k = \dfrac{m}{n}\ln 2 = 19.2 \times 0.6931 \approx 13.3$, so $k = 13$ (rounded, as in `BloomFilter.__init__`).

    **(c) Total memory.** $m = 19.2 \times 2\times10^{8} = 3.83 \times 10^{9}$ bits $= 4.79 \times 10^{8}$ bytes $\approx 0.48$ GB.

    **(d) Unique documents wrongly dropped.** A false positive drops a genuinely unique document, and each document is tested once before insertion. If essentially all $n$ are unique, the expected number of false positives is

    $$
    p \times n = 10^{-4} \times 2\times10^{8} = 2\times10^{4} = 20{,}000 \text{ documents}.
    $$

    There are **no** false negatives, so a true duplicate is never kept — the asymmetry the chapter highlights.

    **Comparison.** The SHA-256 set stores $32$ bytes $= 256$ bits per document $= 32 \times 2\times10^{8} = 6.4$ GB. The Bloom filter uses $0.48$ GB, a saving of $256 / 19.2 \approx 13\times$. (The chapter's $9\times$ figure is for the stricter $p = 10^{-6}$; loosening $p$ to $10^{-4}$ shrinks the filter further, at the cost of dropping more unique docs.)

**4.** MinHash with $k = 128$ hashes. Instead of the chapter's $b = 16, r = 8$ banding, you try $b = 32, r = 4$. (a) Compute the LSH threshold $t^{*}$. (b) Compute $P(\text{candidate})$ for a pair at true similarity $s = 0.9$ and at $s = 0.6$. (c) Explain, relative to the chapter's $16 \times 8$ split, what this does to recall and to verification cost, and what happens to the number of (band, bucket) entries.

??? note "Solution"
    With $b \times r = 32 \times 4 = 128$ the constraint is satisfied.

    **(a) Threshold.** $t^{*} \approx \left(\dfrac{1}{b}\right)^{1/r} = \left(\dfrac{1}{32}\right)^{1/4} = 32^{-1/4} = 2^{-5/4} \approx 0.420.$

    So the S-curve inflection sits near $0.42$ Jaccard — much lower than the $16\times8$ value of $t^{*} = 16^{-1/8} = 2^{-1/2} \approx 0.707$.

    **(b) Candidate probabilities**, using $P = 1 - (1 - s^{r})^{b}$ with $r = 4, b = 32$:

    - $s = 0.9$: $0.9^{4} = 0.6561$, $\;(1 - 0.6561)^{32} = 0.3439^{32} \approx 1.5\times10^{-15}$, so $P \approx 1.000$.
    - $s = 0.6$: $0.6^{4} = 0.1296$, $\;(1 - 0.1296)^{32} = 0.8704^{32} \approx 0.0118$, so $P \approx 0.988$.

    For contrast, the $16\times8$ split gives $P(\text{candidate}\mid s{=}0.6) = 1 - (1 - 0.6^{8})^{16} \approx 0.24$.

    **(c) Interpretation.** Lowering $t^{*}$ from $0.707$ to $0.420$ makes the banding far more permissive: a moderately similar pair ($s = 0.6$) becomes a candidate $\sim 99\%$ of the time instead of $\sim 24\%$. This **raises recall** (fewer genuine near-duplicates slip through) but **greatly raises verification cost**, because many more pairs — including ones far below the $0.8$ target — collide in a band and must go through the exact `est_j >= threshold` check. The number of (band, bucket) entries is $N \times b$: doubling $b$ from $16$ to $32$ literally doubles the number of hashed entries and buckets that must be built and scanned. The $(b, r)$ split is the knob trading recall against verification/index cost; you size it so $t^{*}$ sits just below your true similarity target, not far below it.

**5.** The document-level heuristics catch a page whose *bigrams* repeat, but not a page assembled from many identical *lines* (e.g. a navigation footer duplicated across sections). Implement a Gopher-style `duplicate_line_fraction(text)` — the fraction of non-empty lines that are exact repeats of an earlier identical line — add it to `compute_heuristics` as `dup_line_frac`, and add a `dup_line_frac <= 0.30` check to `passes_heuristic_filters`. Show it rejects a document that is mostly a repeated footer.

??? note "Solution"
    For each distinct line value with count $c$, exactly $c - 1$ of its occurrences are repeats, so the number of repeated lines is $\sum_v (c_v - 1) = (\text{total lines}) - (\text{distinct lines})$.

    ```python
    from collections import Counter

    def duplicate_line_fraction(text: str) -> float:
        """Fraction of non-empty lines that are duplicates of an earlier
        identical line (Gopher-style). 0.0 if there are no lines."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return 0.0
        distinct = len(Counter(lines))
        repeated = len(lines) - distinct
        return repeated / len(lines)
    ```

    Add the signal inside `compute_heuristics` (just before the `return`) and include it in the returned dict:

    ```python
        dup_line_frac = duplicate_line_fraction(text)
        # ... existing keys ...
        "dup_line_frac": dup_line_frac,
    ```

    Then extend the filter:

    ```python
    def passes_heuristic_filters(text: str) -> bool:
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
            and h["dup_line_frac"] <= 0.30      # new: reject repeated-line pages
        )
    ```

    Demonstration — a page that is one real paragraph plus the same footer line repeated nine times:

    ```python
    footer = "Home | About | Contact | Privacy Policy | Terms of Service"
    body = "This paragraph is the only genuine content on the page."
    doc = body + "\n" + "\n".join([footer] * 9)

    lines = [l for l in doc.splitlines() if l.strip()]   # 10 lines
    # distinct = {body, footer} = 2  ->  repeated = 10 - 2 = 8
    print(duplicate_line_fraction(doc))   # 8 / 10 = 0.8
    ```

    `dup_line_frac = 0.8 > 0.30`, so the new check fails and the document is dropped, even though its bigram and word-length signals look fine. This complements `top_bigram_frac`, which measures phrase-level rather than whole-line repetition.

**6.** `find_near_duplicates` returns *pairs* $(a, b, \hat{J})$, but near-duplication is transitive at the cluster level: if `doc1`~`doc2` and `doc2`~`doc5`, all three are the same content and only one should survive. Implement `dedup_keep_set(all_ids, near_dup_pairs)` that groups the pairs into connected components (clusters) and returns the set of document IDs to keep — one representative per cluster. Run it on the demo output from the chapter's MinHash code and state the resulting keep-set.

??? note "Solution"
    Connected components over the "is-a-near-duplicate-of" graph are exactly a union-find (disjoint-set) problem. Union every reported pair, then keep one canonical representative (the component root) per cluster. Singletons — documents in no pair — are their own root and are always kept.

    ```python
    def dedup_keep_set(all_ids, near_dup_pairs):
        """Cluster near-duplicate pairs into connected components and return
        the set of IDs to keep: the smallest ID in each component (its root),
        plus every document that appears in no pair."""
        parent = {i: i for i in all_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                # Keep the lexicographically smaller ID as the surviving root.
                lo, hi = (ra, rb) if ra <= rb else (rb, ra)
                parent[hi] = lo

        for a, b, _ in near_dup_pairs:
            union(a, b)

        return {find(i) for i in all_ids}
    ```

    Running on the chapter's demo, whose near-duplicate pairs are
    `[("doc1","doc2",0.938), ("doc1","doc5",0.594), ("doc2","doc5",0.547)]`
    over `all_ids = ["doc1","doc2","doc3","doc4","doc5"]`:

    ```python
    all_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    pairs = [("doc1", "doc2", 0.938), ("doc1", "doc5", 0.594), ("doc2", "doc5", 0.547)]
    print(sorted(dedup_keep_set(all_ids, pairs)))
    # -> ['doc1', 'doc3', 'doc4']
    ```

    `doc1`, `doc2`, `doc5` collapse into one component (root `doc1`), while `doc3` and `doc4` are singletons. The keep-set is `{doc1, doc3, doc4}` — the corpus shrinks from 5 to 3 documents. The transitive grouping is what makes `doc2` and `doc5` both get dropped in favor of a single representative, even though `doc2`~`doc5` was only surfaced indirectly through their shared similarity to `doc1`.
