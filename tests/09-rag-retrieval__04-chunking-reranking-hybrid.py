"""
Runs the CPU-runnable Python code blocks from:
    content/09-rag-retrieval/04-chunking-reranking-hybrid.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:    #0, #1, #2, #4, #5, #7, #8, #9, #10, #11, #12
Skipped blocks:   #3  (needs GPU/HF model download: AutoModel/AutoTokenizer
                       for jina-embeddings-v2, 8k-context transformer forward pass)
                  #6  (illustrative fine-tuning "sketch": calls CrossEncoder.fit()
                       on a real HF model with a real training loop -- not a
                       standalone unit of logic, needs network + real data)

Two third-party libraries used by the book's code (`sentence_transformers`,
`qdrant_client`) are not guaranteed in CI and normally require a network call
to download model weights / a live server. Rather than skip the blocks that
use them outright, each import is guarded with try/except; when the real
package is unavailable (always true under the CI-sim harness, which blocks
these imports outright) we fall back to a tiny, deterministic, offline stand-in
class with the same call surface (CrossEncoder.predict, Filter/FieldCondition/
MatchValue/Range as plain data holders, a fake client's .search()) so that the
*book's own logic* (pair construction, sorting, filter-condition assembly)
executes for real. Only the external network/model call is faked.
"""

from __future__ import annotations

import zlib

import numpy as np

# ============================================================================
# Guarded imports for third-party libraries not guaranteed in CI
# (`sentence_transformers`, `qdrant_client`). When unavailable -- which is
# always true under the CI-sim harness, since it blocks these imports outright
# -- fall back to tiny, deterministic, offline stand-ins that preserve the
# exact call surface the book's code below uses (CrossEncoder.predict,
# Filter/FieldCondition/MatchValue/Range as plain data holders, a client with
# .search()). This lets the book's OWN logic (pair construction, sorting,
# filter-condition assembly) execute for real; only the external network/model
# call is faked.
# ============================================================================
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

if CrossEncoder is None:

    class CrossEncoder:
        """Stands in for sentence_transformers.CrossEncoder (no network/model download)."""

        def __init__(self, model_name, max_length=512, num_labels=1):
            self.model_name = model_name
            self.max_length = max_length
            self.num_labels = num_labels

        def predict(self, pairs, batch_size=32):
            # Deterministic, content-dependent "relevance" score so sorting is
            # non-trivial: passages that share more words with the query
            # score higher. This is a stand-in, but it exercises real sorting.
            scores = []
            for query, passage in pairs:
                q_words = set(query.lower().split())
                p_words = passage.lower().split()
                overlap = sum(1 for w in p_words if w in q_words)
                scores.append(float(overlap) + 0.001 * len(passage))
            return np.array(scores, dtype=np.float32)


try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
except Exception:
    QdrantClient = Filter = FieldCondition = MatchValue = Range = None

if QdrantClient is None:

    class QdrantClient:
        """Stands in for qdrant_client.QdrantClient (used only as a type hint here)."""

        pass

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class Range:
        def __init__(self, gte=None, lte=None):
            self.gte = gte
            self.lte = lte

    class FieldCondition:
        def __init__(self, key, match=None, range=None):
            self.key = key
            self.match = match
            self.range = range

    class Filter:
        def __init__(self, must=None):
            self.must = must or []


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #0 (line ~32) -- fixed_chunking.py
# ============================================================================
_section("Block #0: fixed_chunking.py")

from typing import Iterator


def fixed_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    tokenizer=None,      # a callable str->list[int]; falls back to whitespace split
) -> Iterator[str]:
    """
    Yield overlapping token-level chunks of `text`.

    Args:
        text:        The raw document string.
        chunk_size:  Maximum tokens per chunk.
        overlap:     Number of tokens to repeat from the previous chunk.
        tokenizer:   Optional callable returning token ids. When None, we split
                     on whitespace as a proxy (fast for prototyping).

    Yields:
        Decoded string chunks.

    Example:
        >>> chunks = list(fixed_chunk("word " * 1000, chunk_size=10, overlap=2))
        >>> len(chunks)   # ceil((1000-10) / (10-2)) + 1 = 125 chunks
        125
    """
    if tokenizer is None:
        # Whitespace proxy: each "token" is a word
        tokens = text.split()
        decode = lambda ids: " ".join(ids)  # noqa: E731
    else:
        tokens = tokenizer(text)
        decode = tokenizer.decode  # type: ignore[attr-defined]

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be strictly greater than overlap")

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        yield decode(tokens[start:end])
        if end == len(tokens):
            break
        start += step


# --- Quick demo (book's __main__ block, executed directly here) ---
doc = " ".join([f"word{i}" for i in range(200)])
chunks = list(fixed_chunk(doc, chunk_size=50, overlap=10))
print(f"Produced {len(chunks)} chunks from 200 tokens")
print(f"First chunk length: {len(chunks[0].split())} words")
print(f"Last chunk:         {chunks[-1][:60]}...")

assert len(chunks) == 5, f"expected 5 chunks, got {len(chunks)}"
assert len(chunks[0].split()) == 50
# consecutive chunks must overlap by exactly `overlap` words
assert chunks[0].split()[-10:] == chunks[1].split()[:10]

# Also verify the docstring's own worked example (book line ~57-59):
# ceil((1000-10)/(10-2)) + 1 = 125 chunks.
example_chunks = list(fixed_chunk("word " * 1000, chunk_size=10, overlap=2))
assert len(example_chunks) == 125, f"docstring example: expected 125, got {len(example_chunks)}"

print("Block #0 OK")


# ============================================================================
# Block #1 (line ~103) -- semantic_chunking.py
# ============================================================================
_section("Block #1: semantic_chunking.py")

from typing import Callable


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def semantic_chunk(
    sentences: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    threshold: float = 0.75,
    min_sentences: int = 3,
) -> list[str]:
    """
    Group consecutive sentences into chunks based on embedding similarity.

    Args:
        sentences:     Pre-tokenized list of sentence strings.
        embed_fn:      Function mapping list[str] -> np.ndarray of shape (N, D).
        threshold:     Similarity threshold; boundaries placed where sim < threshold.
        min_sentences: Minimum sentences per chunk to avoid micro-chunks.

    Returns:
        List of joined chunk strings.

    The key insight: when cos-sim between sentence[i] and sentence[i+1]
    drops sharply, we've hit a topic boundary. This is much more robust
    than arbitrary token counts for structured prose.
    """
    if not sentences:
        return []

    # Embed all sentences in one batch (efficient for API calls)
    embeddings = embed_fn(sentences)  # shape (N, D)

    # Compute consecutive similarities
    sims = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(sentences) - 1)
    ]

    # Find boundary positions (where similarity is low AND gap >= min_sentences)
    chunks: list[str] = []
    current: list[str] = []
    for i, sent in enumerate(sentences):
        current.append(sent)
        if i < len(sims):
            is_boundary = sims[i] < threshold and len(current) >= min_sentences
            if is_boundary:
                chunks.append(" ".join(current))
                current = []

    if current:  # flush remaining sentences
        chunks.append(" ".join(current))

    return chunks


# --- Illustrative usage (book's __main__ block, executed directly here) ---
def mock_embed(texts: list[str]) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((len(texts), 384)).astype(np.float32)


sample = [f"Sentence {i} about {'topic A' if i < 5 else 'topic B'}."
          for i in range(10)]
result = semantic_chunk(sample, mock_embed, threshold=0.5, min_sentences=2)
print(f"Produced {len(result)} semantic chunks")

assert isinstance(result, list) and len(result) > 0
# every original sentence must appear in exactly one output chunk
joined = " ".join(result)
for s in sample:
    assert s in joined

print("Block #1 OK")


# ============================================================================
# Block #2 (line ~187) -- structural_chunking.py
# ============================================================================
_section("Block #2: structural_chunking.py")

import re


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def split_markdown(
    text: str,
    max_tokens: int = 512,
    words_per_token: float = 0.75,  # rough approximation
) -> list[dict]:
    """
    Split a Markdown document respecting heading hierarchy.

    Each returned dict has:
      - 'text':     the chunk content
      - 'heading':  the nearest parent heading (for metadata)
      - 'level':    heading depth (1-6; 0 = preamble)

    Strategy:
      1. Split on headings first.
      2. If a section exceeds max_tokens, sub-split on paragraph breaks.
      3. Preserve heading as metadata for downstream metadata filtering.
    """
    max_words = int(max_tokens / words_per_token)
    chunks: list[dict] = []

    # Find all heading positions
    boundaries = [
        (m.start(), m.group(1), m.group(2).strip()) for m in HEADING_RE.finditer(text)
    ]
    boundaries.append((len(text), "", ""))  # sentinel

    prev_end = 0
    current_heading = ""
    current_level = 0

    for i, (pos, hashes, heading_text) in enumerate(boundaries):
        section = text[prev_end:pos].strip()
        if section:
            words = section.split()
            if len(words) <= max_words:
                chunks.append({
                    "text": section,
                    "heading": current_heading,
                    "level": current_level,
                })
            else:
                # Sub-split on blank lines (paragraphs)
                paragraphs = re.split(r"\n\s*\n", section)
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        chunks.append({
                            "text": para,
                            "heading": current_heading,
                            "level": current_level,
                        })

        current_heading = heading_text
        current_level = len(hashes)
        prev_end = pos

    return chunks


# --- Minimal exercise (no __main__ block in the book; added here) ---
_long_section = "word " * 40  # exceeds max_words for max_tokens=20 below
md_text = (
    "# Introduction\n"
    "This is the intro paragraph explaining background.\n\n"
    "## Details\n"
    f"{_long_section}\n\n"
    "more detail text after the blank line.\n\n"
    "# Conclusion\n"
    "Wrap it up.\n"
)

md_chunks = split_markdown(md_text, max_tokens=20)
print(f"Produced {len(md_chunks)} markdown chunks")
for c in md_chunks:
    print(f"  heading={c['heading']!r} level={c['level']} text[:40]={c['text'][:40]!r}")

assert len(md_chunks) > 0
headings_seen = {c["heading"] for c in md_chunks}
assert "Introduction" in headings_seen, headings_seen
assert "Details" in headings_seen, headings_seen
levels_seen = {c["level"] for c in md_chunks}
assert 1 in levels_seen and 2 in levels_seen, levels_seen
# the oversized "## Details" section must have been sub-split into >1 chunk
details_chunks = [c for c in md_chunks if c["heading"] == "Details"]
assert len(details_chunks) >= 2, "oversized section should have been paragraph-split"

print("Block #2 OK")


# ============================================================================
# Block #3 (line ~267) -- late_chunking.py
# SKIP(needs-gpu/network): loads a real HuggingFace encoder
# (AutoTokenizer/AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en"))
# and runs a full transformer forward pass over an 8k-token context. This
# requires downloading multi-hundred-MB model weights over the network and is
# explicitly out of scope for a CPU/offline smoke test.
# ============================================================================


# ============================================================================
# Block #4 (line ~384) -- hybrid_search.py
# ============================================================================
_section("Block #4: hybrid_search.py")

from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class Document:
    id: str
    text: str
    embedding: np.ndarray | None = field(default=None, repr=False)


# ─── BM25 implementation ────────────────────────────────────────────────────

class BM25Index:
    """Minimal BM25 index for a list of Documents."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self._build(docs)

    def _build(self, docs: list[Document]) -> None:
        # Term frequency per document
        self.tf: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        self.df: dict[str, int] = defaultdict(int)

        for doc in docs:
            tokens = doc.text.lower().split()
            freq: dict[str, int] = defaultdict(int)
            for tok in tokens:
                freq[tok] += 1
            self.tf.append(dict(freq))
            self.doc_len.append(len(tokens))
            for tok in freq:
                self.df[tok] += 1

        self.N = len(docs)
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, doc_idx: int) -> float:
        tokens = query.lower().split()
        dl = self.doc_len[doc_idx]
        score = 0.0
        for tok in tokens:
            f = self.tf[doc_idx].get(tok, 0)
            idf = self._idf(tok)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_idx, bm25_score) sorted descending."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ─── Dense retrieval ─────────────────────────────────────────────────────────

def dense_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,  # (N, D)
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Return (doc_idx, cosine_sim) sorted descending."""
    sims = doc_embeddings @ query_embedding  # dot product = cosine sim if normalized
    idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idx]


# ─── Reciprocal Rank Fusion ──────────────────────────────────────────────────

def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Fuse multiple ranked lists via RRF.

    Args:
        rankings: Each element is a list of (doc_idx, score) sorted by score desc.
                  The actual scores are ignored; only ranks matter.
        k:        RRF smoothing constant (default 60, from Cormack et al. 2009).

    Returns:
        Fused list of (doc_idx, rrf_score) sorted descending.
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked_list in rankings:
        for rank, (doc_idx, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_idx] += 1.0 / (k + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return fused


# ─── Putting it together ──────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    bm25_index: BM25Index,
    doc_embeddings: np.ndarray,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Hybrid BM25 + dense search with RRF fusion.

    Returns top_k (Document, rrf_score) pairs sorted by descending fused rank.
    """
    bm25_results = bm25_index.search(query, top_k=top_k * 2)
    dense_results = dense_search(query_embedding, doc_embeddings, top_k=top_k * 2)

    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=rrf_k)
    docs = bm25_index.docs
    return [(docs[idx], score) for idx, score in fused[:top_k]]


# --- Exercise with a tiny toy corpus ---
_toy_texts = [
    "The transformer architecture uses self-attention layers.",
    "Reciprocal rank fusion combines BM25 and dense retrieval rankings.",
    "The capital of France is Paris, a city on the Seine.",
    "BM25 handles exact keyword and rare-term matches well.",
    "Cross-encoder rerankers score query-document pairs jointly.",
]
_toy_docs = [Document(id=f"d{i}", text=t) for i, t in enumerate(_toy_texts)]
_rng = np.random.default_rng(0)
_toy_embeddings = _rng.standard_normal((len(_toy_docs), 16)).astype(np.float32)
_toy_embeddings /= np.linalg.norm(_toy_embeddings, axis=1, keepdims=True)
_query_embedding = _toy_embeddings[1] + 0.01 * _rng.standard_normal(16).astype(np.float32)
_query_embedding /= np.linalg.norm(_query_embedding)

bm25_index = BM25Index(_toy_docs)
results = hybrid_search("reciprocal rank fusion BM25", _query_embedding, bm25_index, _toy_embeddings, top_k=3)
print("Hybrid search results:")
for doc, score in results:
    print(f"  {doc.id}: {score:.4f}  {doc.text[:50]!r}")

assert len(results) == 3
assert results[0][1] >= results[1][1] >= results[2][1], "results must be sorted descending"
# doc 1 (the RRF sentence) should rank first: it wins BM25 (keyword match) AND
# dense (it's the query embedding's nearest neighbor by construction)
assert results[0][0].id == "d1"

# also sanity check the docstring's own worked RRF example (Section 3.1 table)
_bm25_ranking = [(2, 0.0), (1, 0.0), (3, 0.0), (4, 0.0), (5, 0.0)]   # D2,D1,D3,D4,D5
_dense_ranking = [(1, 0.0), (3, 0.0), (2, 0.0), (5, 0.0), (4, 0.0)]  # D1,D3,D2,D5,D4
_fused = reciprocal_rank_fusion([_bm25_ranking, _dense_ranking], k=60)
assert _fused[0][0] == 1, f"expected D1 to win the fused ranking, got D{_fused[0][0]}"
_d1_score = dict(_fused)[1]
assert abs(_d1_score - (1 / 62 + 1 / 61)) < 1e-9

print("Block #4 OK")


# ============================================================================
# Block #5 (line ~545) -- cross_encoder_rerank.py
# Uses a stand-in sentence_transformers.CrossEncoder (see top of file) so the
# book's own pair-construction/sort logic runs without a network model download.
# ============================================================================
_section("Block #5: cross_encoder_rerank.py")

# `CrossEncoder` here resolves to the real sentence_transformers class if the
# package happens to be installed, otherwise to the offline stand-in defined
# near the top of this file (see the guarded-import section above).


def rerank(
    query: str,
    candidates: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """
    Rerank a list of candidate passages using a cross-encoder.

    Args:
        query:      The user query string.
        candidates: List of passage strings to rerank.
        model_name: HuggingFace cross-encoder model. Common choices:
                    - 'cross-encoder/ms-marco-MiniLM-L-6-v2'  (fast, strong)
                    - 'cross-encoder/ms-marco-electra-base'    (more accurate)
                    - 'mixedbread-ai/mxbai-rerank-large-v1'   (multilingual)
        top_k:      Number of top passages to return after reranking.
        batch_size: Batch size for model inference.

    Returns:
        List of (passage_text, relevance_score) sorted by descending score.

    Notes:
        - Scores are logits (not probabilities) from the final classification head.
        - Higher score = more relevant.
        - For latency-sensitive applications, prefer MiniLM (6 layers) over ELECTRA.
    """
    model = CrossEncoder(model_name, max_length=512)

    # Build (query, passage) pairs — one per candidate
    pairs = [(query, passage) for passage in candidates]

    # Score all pairs; model handles batching internally
    scores: np.ndarray = model.predict(pairs, batch_size=batch_size)

    # Sort by descending score
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return ranked[:top_k]


# --- Exercise ---
_rerank_query = "What is reciprocal rank fusion?"
_rerank_candidates = [
    "The capital of France is Paris.",
    "Reciprocal rank fusion combines multiple ranked lists using rank position.",
    "Cats are small domesticated carnivorous mammals.",
]
reranked = rerank(_rerank_query, _rerank_candidates, top_k=2)
print("Reranked:")
for text, score in reranked:
    print(f"  {score:.3f}  {text}")

assert len(reranked) == 2
assert reranked[0][1] >= reranked[1][1]
# the passage that actually discusses RRF should be ranked first
assert "Reciprocal rank fusion" in reranked[0][0]

print("Block #5 OK")


# ============================================================================
# Block #6 (line ~605) -- reranker_finetune.py
# SKIP(fragment/network): this is explicitly an illustrative "sketch"
# (book's own words) that builds a real sentence-transformers CrossEncoder
# fine-tuning loop with model.fit(). It needs a real downloaded HF model plus
# a torch DataLoader training loop -- not a standalone unit of retrieval
# logic to unit-test, and running .fit() against a fake model would not
# verify anything the book actually claims.
# ============================================================================


# ============================================================================
# Block #7 (line ~642) -- query_rewriting.py
# ============================================================================
_section("Block #7: query_rewriting.py")

import json


REWRITE_PROMPT = """\
You are a retrieval expert. Given a user question, produce 3 alternative phrasings
that together cover the semantic space of the question. Return a JSON list of strings.

User question: {question}

Alternative phrasings (JSON list):"""


def expand_query(
    question: str,
    llm_call: callable,  # fn(prompt: str) -> str
    n_variants: int = 3,
) -> list[str]:
    """
    Use an LLM to generate query variants for multi-query retrieval.

    Args:
        question:   Original user question.
        llm_call:   Callable that takes a prompt and returns a completion string.
        n_variants: Number of alternative phrasings requested (soft limit).

    Returns:
        List of query strings including the original.
    """
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm_call(prompt)

    # Extract JSON list from LLM output (handle markdown code fences)
    json_match = re.search(r"\[.*?\]", response, re.DOTALL)
    if json_match:
        try:
            variants = json.loads(json_match.group())
            if isinstance(variants, list):
                return [question] + variants[:n_variants]
        except json.JSONDecodeError:
            pass

    return [question]  # fallback: return original query only


# --- Exercise: a stub LLM that returns well-formed JSON ---
def _stub_llm_good(prompt: str) -> str:
    return (
        "Here are some alternatives:\n"
        '```json\n["How does RRF combine rankings?", '
        '"Explain reciprocal rank fusion.", "What is the RRF formula?"]\n```'
    )


variants = expand_query("What is RRF?", _stub_llm_good)
print("Query variants:", variants)
assert variants[0] == "What is RRF?"
assert len(variants) == 4

# --- Exercise the fallback path: a stub LLM that returns garbage ---
def _stub_llm_bad(prompt: str) -> str:
    return "I'm not sure how to answer that."


fallback_variants = expand_query("What is RRF?", _stub_llm_bad)
assert fallback_variants == ["What is RRF?"]

print("Block #7 OK")


# ============================================================================
# Block #8 (line ~698) -- hyde.py
# ============================================================================
_section("Block #8: hyde.py")

HYDE_PROMPT = """\
Write a short, factual paragraph (2-4 sentences) that directly answers the
following question. Do not say "I don't know" — write the best answer you can,
even if uncertain.

Question: {question}

Answer:"""


def hyde_embed(
    question: str,
    llm_call: callable,           # fn(prompt: str) -> str
    embed_fn: callable,           # fn(list[str]) -> np.ndarray
    n_hypotheses: int = 1,
) -> np.ndarray:
    """
    Generate HyDE embedding for a question.

    When n_hypotheses > 1, generate multiple hypothetical documents and
    average their embeddings. This reduces variance from stochastic generation.

    Args:
        question:      User query.
        llm_call:      LLM completion function.
        embed_fn:      Embedding function mapping list[str] -> (N, D) array.
        n_hypotheses:  Number of hypothetical documents to generate.

    Returns:
        Averaged embedding vector of shape (D,), L2-normalized.
    """
    hypotheses = []
    prompt = HYDE_PROMPT.format(question=question)
    for _ in range(n_hypotheses):
        hypothetical_doc = llm_call(prompt)
        hypotheses.append(hypothetical_doc.strip())

    # Embed all hypotheses and average
    embeddings = embed_fn(hypotheses)  # (n_hypotheses, D)
    mean_embedding = embeddings.mean(axis=0)

    # L2-normalize so downstream cosine search still works
    norm = np.linalg.norm(mean_embedding)
    return mean_embedding / (norm + 1e-9)


# --- Exercise ---
def _stub_llm_hyde(prompt: str) -> str:
    return "Reciprocal rank fusion is a method for combining ranked lists using rank position."


def _stub_embed(texts: list[str]) -> np.ndarray:
    rng = np.random.default_rng(len(texts))
    return rng.standard_normal((len(texts), 8)).astype(np.float32)


hyde_vec = hyde_embed("What is RRF?", _stub_llm_hyde, _stub_embed, n_hypotheses=3)
print("HyDE embedding shape:", hyde_vec.shape, "norm:", np.linalg.norm(hyde_vec))
assert hyde_vec.shape == (8,)
assert abs(np.linalg.norm(hyde_vec) - 1.0) < 1e-5

print("Block #8 OK")


# ============================================================================
# Block #9 (line ~786) -- metadata_filtering.py
# Uses a stand-in qdrant_client (see top of file) so the book's own
# filter-condition-assembly logic runs without a live Qdrant server.
# ============================================================================
_section("Block #9: metadata_filtering.py")

# `QdrantClient` / `Filter` / `FieldCondition` / `MatchValue` / `Range` here
# resolve to the real qdrant_client classes if the package happens to be
# installed, otherwise to the offline stand-ins defined near the top of this
# file (see the guarded-import section above). The client instance passed to
# filtered_search() below is always the fake `_FakeSearchClient` regardless,
# since a real QdrantClient would need a live server.


def filtered_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: np.ndarray,
    filters: dict,              # simplified filter spec (see below)
    top_k: int = 10,
) -> list[dict]:
    """
    Perform vector search with metadata pre-filtering in Qdrant.

    Args:
        client:          Qdrant client connected to your instance.
        collection_name: Name of the Qdrant collection.
        query_vector:    Query embedding, shape (D,).
        filters:         Dict with optional keys:
                         - 'category': str (exact match)
                         - 'date_after': int (Unix timestamp)
                         - 'date_before': int (Unix timestamp)
        top_k:           Number of results to return.

    Returns:
        List of payload dicts for matched documents.

    Qdrant supports filtering directly on HNSW traversal, making pre-filtering
    efficient even on large corpora (millions of documents).
    """
    conditions = []

    if "category" in filters:
        conditions.append(
            FieldCondition(key="category", match=MatchValue(value=filters["category"]))
        )

    if "date_after" in filters or "date_before" in filters:
        conditions.append(
            FieldCondition(
                key="timestamp",
                range=Range(
                    gte=filters.get("date_after"),
                    lte=filters.get("date_before"),
                ),
            )
        )

    qdrant_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return [hit.payload for hit in results]


# --- Exercise with a fake Qdrant client that records the call and returns hits ---
class _FakeHit:
    def __init__(self, payload):
        self.payload = payload


class _FakeSearchClient:
    def __init__(self):
        self.last_call = None

    def search(self, collection_name, query_vector, query_filter, limit, with_payload):
        self.last_call = dict(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        )
        return [_FakeHit({"id": "doc1", "category": "legal"}), _FakeHit({"id": "doc2", "category": "legal"})]


_fake_client = _FakeSearchClient()
_qvec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
payloads = filtered_search(
    _fake_client, "my_collection", _qvec,
    filters={"category": "legal", "date_after": 1700000000},
    top_k=5,
)
print("Filtered search payloads:", payloads)

assert payloads == [{"id": "doc1", "category": "legal"}, {"id": "doc2", "category": "legal"}]
assert _fake_client.last_call["limit"] == 5
assert _fake_client.last_call["query_filter"] is not None
assert len(_fake_client.last_call["query_filter"].must) == 2
assert _fake_client.last_call["query_filter"].must[0].match.value == "legal"
assert _fake_client.last_call["query_filter"].must[1].range.gte == 1700000000

# no filters -> query_filter must be None
_fake_client2 = _FakeSearchClient()
filtered_search(_fake_client2, "my_collection", _qvec, filters={}, top_k=3)
assert _fake_client2.last_call["query_filter"] is None

print("Block #9 OK")


# ============================================================================
# Block #10 (line ~855) -- parent_child_retrieval.py
# ============================================================================
_section("Block #10: parent_child_retrieval.py")

from typing import Any


@dataclass
class ChildChunk:
    id: str
    parent_id: str   # foreign key to parent document
    text: str        # small chunk used for embedding/retrieval
    embedding: Any   # np.ndarray


@dataclass
class ParentDocument:
    id: str
    text: str        # larger context returned to the LLM
    metadata: dict


def parent_child_retrieve(
    query_embedding,
    child_index,           # your vector index over ChildChunk objects
    parent_store: dict,    # parent_id -> ParentDocument
    top_k: int = 5,
) -> list[ParentDocument]:
    """
    Retrieve top-k child chunks, then return their parent documents.

    Deduplicates parents so the same parent is not returned twice even if
    multiple of its children ranked highly.
    """
    # Retrieve top-k*3 children to ensure we have enough unique parents
    child_hits = child_index.search(query_embedding, top_k=top_k * 3)

    seen_parent_ids: set[str] = set()
    parents: list[ParentDocument] = []

    for child in child_hits:
        pid = child.parent_id
        if pid not in seen_parent_ids:
            seen_parent_ids.add(pid)
            if pid in parent_store:
                parents.append(parent_store[pid])
        if len(parents) >= top_k:
            break

    return parents


# --- Exercise ---
class _FakeChildIndex:
    def __init__(self, children: list[ChildChunk]):
        self.children = children

    def search(self, query_embedding, top_k):
        # ignore the actual embedding; just return children in stored order
        return self.children[:top_k]


_children = [
    ChildChunk(id="c1", parent_id="p1", text="child 1 of parent 1", embedding=None),
    ChildChunk(id="c2", parent_id="p1", text="child 2 of parent 1", embedding=None),
    ChildChunk(id="c3", parent_id="p2", text="child 1 of parent 2", embedding=None),
    ChildChunk(id="c4", parent_id="p3", text="child 1 of parent 3", embedding=None),
]
_parent_store = {
    "p1": ParentDocument(id="p1", text="Full parent document 1 text.", metadata={}),
    "p2": ParentDocument(id="p2", text="Full parent document 2 text.", metadata={}),
    "p3": ParentDocument(id="p3", text="Full parent document 3 text.", metadata={}),
}
_child_index = _FakeChildIndex(_children)
parents = parent_child_retrieve(np.zeros(4), _child_index, _parent_store, top_k=2)
print("Retrieved parents:", [p.id for p in parents])

assert [p.id for p in parents] == ["p1", "p2"], "must dedupe c1/c2 -> single p1, then p2"
assert len(parents) == 2

print("Block #10 OK")


# ============================================================================
# Block #11 (line ~913) -- rag_pipeline.py
# ============================================================================
_section("Block #11: rag_pipeline.py")

import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: str = "fixed"  # "fixed" | "semantic" | "structural"

    # Retrieval
    bm25_top_k: int = 40          # BM25 candidates before fusion
    dense_top_k: int = 40         # Dense candidates before fusion
    rrf_k: int = 60                # RRF smoothing constant

    # Query expansion
    use_hyde: bool = False          # enable HyDE
    n_query_variants: int = 1       # 1 = no expansion

    # Reranking
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5         # final chunks passed to LLM

    # Metadata filter
    metadata_filter: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    query: str
    retrieved_chunks: list[str]
    rrf_scores: list[float]
    reranker_scores: list[float]
    hyde_hypothesis: str | None = None


def run_rag_pipeline(
    query: str,
    bm25_index,           # BM25Index from Section 3
    dense_index,          # vector store with .search(embedding, top_k) -> list[tuple[int, float]]
    embed_fn: Callable,   # str -> np.ndarray
    llm_fn: Callable,     # str -> str (for HyDE / query expansion)
    reranker,             # CrossEncoder or None
    config: RAGConfig,
) -> RAGResult:
    """
    Full RAG retrieval pipeline: query expansion → hybrid retrieval → rerank.
    """

    # ── Step 1: Query representation ──────────────────────────────────────────
    hyde_hypothesis = None
    if config.use_hyde:
        query_embedding = hyde_embed(query, llm_fn, embed_fn)
        hyde_hypothesis = "(HyDE embedding used)"
        logger.info("Using HyDE embedding for query: %s", query[:60])
    else:
        query_embedding = embed_fn([query])[0]

    # ── Step 2: Hybrid retrieval ───────────────────────────────────────────────
    bm25_results = bm25_index.search(query, top_k=config.bm25_top_k)
    dense_results_raw = dense_index.search(query_embedding, top_k=config.dense_top_k)
    # Normalize dense results to (idx, score) format (indices must match the
    # corpus indices used by bm25_index.docs, same contract as dense_search())
    dense_results = [(int(idx), float(score)) for idx, score in dense_results_raw]

    fused = reciprocal_rank_fusion(
        [bm25_results, dense_results], k=config.rrf_k
    )
    # Map back to document texts
    all_docs = bm25_index.docs
    fused_chunks = [(all_docs[idx].text, score) for idx, score in fused]
    rrf_scores = [s for _, s in fused_chunks]
    candidate_texts = [t for t, _ in fused_chunks]

    logger.info(
        "Hybrid retrieval: %d BM25 + %d dense → %d fused candidates",
        config.bm25_top_k, config.dense_top_k, len(fused_chunks),
    )

    # ── Step 3: Reranking ──────────────────────────────────────────────────────
    if config.use_reranker and reranker is not None:
        reranked = rerank(
            query, candidate_texts,
            top_k=config.reranker_top_k,
        )
        final_chunks = [t for t, _ in reranked]
        reranker_scores = [float(s) for _, s in reranked]
    else:
        final_chunks = candidate_texts[:config.reranker_top_k]
        reranker_scores = rrf_scores[:config.reranker_top_k]

    return RAGResult(
        query=query,
        retrieved_chunks=final_chunks,
        rrf_scores=rrf_scores[:config.reranker_top_k],
        reranker_scores=reranker_scores,
        hyde_hypothesis=hyde_hypothesis,
    )


# --- Exercise: reuse the toy BM25 corpus/embeddings from Block #4 ---
class _FakeDenseIndex:
    """vector store with .search(embedding, top_k) -> list[tuple[int, float]]"""

    def __init__(self, doc_embeddings: np.ndarray):
        self.doc_embeddings = doc_embeddings

    def search(self, query_embedding, top_k):
        return dense_search(query_embedding, self.doc_embeddings, top_k=top_k)


def _pipeline_embed_fn(texts: list[str]) -> np.ndarray:
    # Deterministic pseudo-embedding derived from a text hash, so the query
    # embedding is reproducible across runs. Uses zlib.crc32 rather than the
    # builtin hash(), which is randomized per-process (PYTHONHASHSEED) for
    # strings and would make this non-deterministic across runs/machines.
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(zlib.crc32(t.encode("utf-8")))
        out[i] = rng.standard_normal(16).astype(np.float32)
        out[i] /= np.linalg.norm(out[i]) + 1e-9
    return out


dense_index = _FakeDenseIndex(_toy_embeddings)
config = RAGConfig(bm25_top_k=5, dense_top_k=5, reranker_top_k=3, use_reranker=True)
pipeline_result = run_rag_pipeline(
    query="reciprocal rank fusion BM25",
    bm25_index=bm25_index,
    dense_index=dense_index,
    embed_fn=_pipeline_embed_fn,
    llm_fn=_stub_llm_hyde,
    reranker=CrossEncoder(config.reranker_model),
    config=config,
)
print("RAG pipeline result:")
print(" retrieved_chunks:", pipeline_result.retrieved_chunks)
print(" rrf_scores:      ", pipeline_result.rrf_scores)
print(" reranker_scores: ", pipeline_result.reranker_scores)

assert isinstance(pipeline_result, RAGResult)
assert len(pipeline_result.retrieved_chunks) <= config.reranker_top_k
assert len(pipeline_result.retrieved_chunks) > 0
assert pipeline_result.hyde_hypothesis is None  # use_hyde=False

# also exercise the use_hyde=True branch
config_hyde = RAGConfig(bm25_top_k=5, dense_top_k=5, reranker_top_k=2, use_reranker=False, use_hyde=True)
pipeline_result_hyde = run_rag_pipeline(
    query="What is RRF?",
    bm25_index=bm25_index,
    dense_index=dense_index,
    embed_fn=_pipeline_embed_fn,
    llm_fn=_stub_llm_hyde,
    reranker=None,
    config=config_hyde,
)
assert pipeline_result_hyde.hyde_hypothesis == "(HyDE embedding used)"
assert len(pipeline_result_hyde.retrieved_chunks) <= config_hyde.reranker_top_k

print("Block #11 OK")


# ============================================================================
# Block #12 (line ~1040) -- rag_diagnostics.py
# ============================================================================
_section("Block #12: rag_diagnostics.py")

from typing import NamedTuple


class RetrievalMetrics(NamedTuple):
    recall_at_k: float        # fraction of queries where relevant doc is in top-k
    mrr: float                # mean reciprocal rank
    ndcg_at_k: float          # normalized discounted cumulative gain at k
    latency_p50_ms: float
    latency_p99_ms: float


def compute_recall_at_k(
    relevant_ids: list[set[str]],    # per-query sets of relevant document ids
    retrieved_ids: list[list[str]],  # per-query ranked retrieved document ids
    k: int,
) -> float:
    """
    Recall@k: for each query, was at least one relevant document in top-k?
    Averaged over all queries.
    """
    assert len(relevant_ids) == len(retrieved_ids), "Must align by query"
    hits = sum(
        1 for rel, ret in zip(relevant_ids, retrieved_ids)
        if rel & set(ret[:k])
    )
    return hits / len(relevant_ids)


def compute_mrr(
    relevant_ids: list[set[str]],
    retrieved_ids: list[list[str]],
) -> float:
    """
    Mean Reciprocal Rank: reciprocal of rank of first relevant document.
    """
    rrs = []
    for rel, ret in zip(relevant_ids, retrieved_ids):
        for rank, doc_id in enumerate(ret, start=1):
            if doc_id in rel:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return sum(rrs) / len(rrs)


# --- Exercise with a tiny hand-computed fixture ---
_relevant = [{"docA"}, {"docX", "docY"}, {"docZ"}]
_retrieved = [
    ["doc1", "docA", "doc2"],   # docA at rank 2 -> recall@3 hit, RR=1/2
    ["docX", "doc3", "doc4"],   # docX at rank 1 -> recall hit, RR=1
    ["doc5", "doc6", "doc7"],   # no relevant doc retrieved -> miss, RR=0
]

recall_at_3 = compute_recall_at_k(_relevant, _retrieved, k=3)
mrr = compute_mrr(_relevant, _retrieved)
print(f"Recall@3: {recall_at_3:.4f}  MRR: {mrr:.4f}")

assert abs(recall_at_3 - (2 / 3)) < 1e-9
assert abs(mrr - ((1 / 2 + 1 / 1 + 0) / 3)) < 1e-9

metrics = RetrievalMetrics(
    recall_at_k=recall_at_3, mrr=mrr, ndcg_at_k=0.75,
    latency_p50_ms=12.0, latency_p99_ms=45.0,
)
assert metrics.recall_at_k == recall_at_3

print("Block #12 OK")


print("\nALL BLOCKS PASSED")
