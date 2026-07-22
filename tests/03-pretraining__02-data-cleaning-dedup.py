"""
Runs the CPU-runnable Python code blocks from:
    content/03-pretraining/02-data-cleaning-dedup.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order. Each block's functions/classes are then actually exercised with tiny
fixtures so every tested block EXECUTES, not just defines names.

Tested blocks:
    #1 (line ~74)  -- compute_heuristics / passes_heuristic_filters
    #2 (line ~170) -- train_quality_classifier / quality_scores / pick_threshold
    #3 (line ~230) -- redact_pii (regex PII redaction)
    #4 (line ~277) -- exact_dedup_sha256
    #5 (line ~323) -- BloomFilter / bloom_dedup (book's own __main__ demo,
                       which asserts zero false negatives, runs verbatim
                       since this file is itself run as __main__)
    #6 (line ~446) -- MinHash + LSH near-dedup pipeline (book's own __main__
                       demo, which asserts the MinHash estimate is within
                       3 standard errors of the exact Jaccard, runs verbatim)
    #8 (line ~686) -- build_ngram_index / is_contaminated (benchmark decontam)

Skipped blocks:
    #0  (line ~13)  -- SKIP(network): downloads lid.176.bin via urllib and
                        loads it with `fasttext`, an optional third-party
                        package not in the guaranteed CI import list. Both a
                        network call and an unavailable import; skipped.
    #7  (line ~652) -- non-python ```text``` diagram of a suffix array /
                        sorted-suffix example, nothing to execute.
    #9  (line ~762) -- SKIP(gpu/heavy): the GPT-2 ablation harness needs
                        `transformers`/`datasets` (not guaranteed in CI) and
                        trains two real GPT-2-style models for thousands of
                        steps -- far outside a ~60s CPU budget even at "toy"
                        size. `transformers`/`datasets` imports are guarded so
                        the module still loads; the functions are defined but
                        not called.
    #10 (line ~880) -- SKIP(shell): a ```bash``` production-pipeline script
                        (aws s3 sync / spark-submit / python CLI stages), not
                        Python.

No network access and no optional third-party imports are exercised at
runtime for the blocks that DO run -- only numpy, scikit-learn, and the
standard library, all of which are in the guaranteed CI list.
"""

from __future__ import annotations

# Optional third-party deps used only by SKIPPED blocks -- guarded so the
# module still loads in CI even without them.
try:
    import fasttext  # noqa: F401  (used only by skipped block #0)
except Exception:
    fasttext = None

try:
    import torch  # noqa: F401  (used only by skipped block #9)
    from transformers import (  # noqa: F401
        GPT2Config,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        get_cosine_schedule_with_warmup,
    )
except Exception:
    torch = None
    GPT2Config = GPT2LMHeadModel = GPT2TokenizerFast = None
    get_cosine_schedule_with_warmup = None


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #1 (line ~74) -- Heuristic quality filters
# ============================================================================
_section("Block #1: compute_heuristics / passes_heuristic_filters")

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

# ---- exercise block #1 ----
GOOD_DOC = (
    "The history of the printing press is one of the most important "
    "stories in the history of human communication. Before it existed, "
    "books were copied by hand, a slow and costly process that meant "
    "only the wealthy or the church could own them. When the press "
    "arrived, it changed how people shared knowledge, how they learned "
    "about the world, and how ideas spread from one city to another. "
    "Many historians argue that this single invention did more to shape "
    "the modern world than almost any other technology that came before it."
)
BAD_DOC_SHORT = "Buy now! Click here!!! 12345 12345 {a} {b}"

good_h = compute_heuristics(GOOD_DOC)
print("good doc heuristics:", good_h)
assert passes_heuristic_filters(GOOD_DOC) is True
assert passes_heuristic_filters(BAD_DOC_SHORT) is False
print("passes_heuristic_filters: good=True, bad=False -- OK")


# ============================================================================
# Block #2 (line ~170) -- Wikipedia-vs-crawl trainable quality classifier
# ============================================================================
_section("Block #2: train_quality_classifier / quality_scores / pick_threshold")

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

# ---- exercise block #2 (tiny fixture: 20 Wikipedia-ish vs. 20 crawl-ish docs,
#      n_features shrunk from the book's 2**20 to 2**14 purely for test speed
#      -- HashingVectorizer's whole point is that n_features is a free knob) ----
wiki_docs = [
    "The mitochondrion is the organelle responsible for producing ATP "
    "through oxidative phosphorylation in eukaryotic cells.",
    "The French Revolution began in 1789 and led to the end of the "
    "Bourbon monarchy in France.",
    "Photosynthesis converts light energy into chemical energy stored "
    "in glucose molecules within chloroplasts.",
    "The Roman Empire reached its greatest territorial extent under "
    "the emperor Trajan in the early second century.",
    "Quantum mechanics describes the behavior of matter and energy at "
    "the smallest scales, including atoms and subatomic particles.",
] * 4
crawl_docs = [
    "BUY NOW!! best deals cheap cheap cheap click here for FREE prize!!!",
    "asdkj asldkj asd 123123 !!!! subscribe like comment SALE SALE SALE",
    "lorem ipsum dolor sit amet consectetur spam spam spam buy buy buy",
    "!!! CLICK HERE !!! WIN A FREE IPHONE NOW LIMITED TIME OFFER !!!",
    "xxxx yyyy zzzz random keyword stuffing seo seo seo cheap cheap",
] * 4
heldout_docs = wiki_docs[:3] + crawl_docs[:3]

vec, clf = train_quality_classifier(wiki_docs, crawl_docs, n_features=2**14)
scores = quality_scores(vec, clf, heldout_docs)
thr = pick_threshold(scores, keep_frac=0.30)
print("heldout quality scores:", np.round(scores, 3))
print("threshold (top 30%):", round(thr, 3))
assert scores.shape == (len(heldout_docs),)
assert 0.0 <= thr <= 1.0
# The classifier should score Wikipedia-like text above spammy crawl text.
assert quality_scores(vec, clf, wiki_docs[:1])[0] > quality_scores(vec, clf, crawl_docs[:1])[0]
print("quality classifier ranks wiki-like text above spam -- OK")


# ============================================================================
# Block #3 (line ~230) -- PII regex redaction
# ============================================================================
_section("Block #3: redact_pii")

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

# ---- exercise block #3 (the chapter's own worked example) ----
redacted = redact_pii(
    "Contact me at alice@example.com or 555-867-5309. "
    "SSN on file: 123-45-6789, server IP 10.0.0.1."
)
print("redacted:", redacted)
assert redacted == (
    "Contact me at EMAIL_ADDRESS or PHONE_NUMBER. "
    "SSN on file: SSN, server IP IP_ADDRESS."
)
print("redact_pii matches chapter's worked example -- OK")


# ============================================================================
# Block #4 (line ~277) -- Exact SHA-256 deduplication
# ============================================================================
_section("Block #4: exact_dedup_sha256")

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

# ---- exercise block #4 ----
raw_docs = [
    "The quick brown fox jumps over the lazy dog.",
    "The   quick brown fox jumps over the lazy dog.",   # dup via whitespace
    "  The quick brown fox jumps over the lazy dog.  ",  # dup via whitespace
    "A totally different sentence about cats.",
    "The quick brown fox jumps over the lazy dog.",     # exact dup
]
deduped = exact_dedup_sha256(raw_docs)
print("raw:", len(raw_docs), "-> deduped:", len(deduped))
assert deduped == [
    "The quick brown fox jumps over the lazy dog.",
    "A totally different sentence about cats.",
]
print("exact_dedup_sha256 removes exact + whitespace-normalized dups -- OK")


# ============================================================================
# Block #5 (line ~323) -- BloomFilter / bloom_dedup
# ============================================================================
_section("Block #5: BloomFilter / bloom_dedup")

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

    # Also exercise bloom_dedup itself (not in the book's own __main__ demo,
    # but the function it defines above it in the same block).
    docs_with_dups = raw_docs  # reuse block #4's fixture
    bd = bloom_dedup(docs_with_dups)
    print("bloom_dedup:", len(docs_with_dups), "->", len(bd))
    assert bd == deduped, "bloom_dedup should agree with exact_dedup_sha256 here"
    print("bloom_dedup agrees with exact_dedup_sha256 -- OK")


# ============================================================================
# Block #6 (line ~446) -- MinHash + LSH near-duplicate detection
# ============================================================================
_section("Block #6: MinHash + LSH near-dedup pipeline")

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
    assert {(a, b) for a, b, _ in pairs} == {("doc1", "doc2"), ("doc1", "doc5"), ("doc2", "doc5")}
    assert ("doc3", "doc4") not in {(a, b) for a, b, _ in pairs}

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


# ============================================================================
# Block #8 (line ~686) -- Benchmark decontamination via n-gram overlap
# ============================================================================
_section("Block #8: build_ngram_index / is_contaminated")

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

# ---- exercise block #8 (n=13 default, per the chapter's own recommendation) ----
benchmark_examples = [
    "What is the capital of France and what river runs through the middle of it",
    "The mitochondria is the powerhouse of the cell and produces energy via ATP",
]
ngram_index = build_ngram_index(benchmark_examples, n=13)

contaminated_doc = (
    "Some forum post text before. What is the capital of France and what "
    "river runs through the middle of it, asked the student. More text after."
)
clean_doc = (
    "This is an entirely unrelated document about gardening tips for "
    "growing tomatoes in a small backyard plot during the summer months."
)
print("ngram_index has", len(ngram_index), "distinct 13-grams")
assert is_contaminated(contaminated_doc, ngram_index, n=13) is True
assert is_contaminated(clean_doc, ngram_index, n=13) is False
print("is_contaminated: contaminated=True, clean=False -- OK")


print("\nAll tested blocks (#1, #2, #3, #4, #5, #6, #8) executed successfully.")
