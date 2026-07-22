# 9.4 Chunking, Reranking & Hybrid Search

Retrieval-Augmented Generation (RAG) feels deceptively simple on paper: split your documents into pieces, embed them, and fetch the most similar pieces at query time. In practice, that description hides half a dozen hard sub-problems. Split too coarsely and you retrieve paragraphs that bury the relevant sentence in noise. Split too finely and no single chunk carries enough context to be useful. Use only dense retrieval and you miss documents where an exact product code or a rare proper noun is the critical signal. Use only keyword search and you miss paraphrases. Retrieve the top-k blindly and the final context window fills with redundant or marginally relevant chunks.

This chapter is about the practical levers that close the gap between a toy RAG prototype and a production-grade system. We cover the full retrieval pipeline: chunking strategies, hybrid search combining BM25 and dense retrieval, reciprocal rank fusion, cross-encoder rerankers, query rewriting and HyDE, and metadata filtering. Each section explains the mechanism, shows real code, and calls out the failure modes.

For the broader RAG system architecture see [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html). The embedding models that power the dense retrieval leg are covered in [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html), and the ANN indexes that scale to millions of vectors are described in [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html). Advanced topics like GraphRAG and long-context alternatives appear in [Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html).

---

## 1. Why Chunking Is the Hardest Part Nobody Talks About

Every embedding model has a maximum token length — typically 512 tokens for older bi-encoders, 4096–8192 for modern ones (e.g., `text-embedding-3-large`, `nomic-embed-text-v1.5`). But even if your model supports 8k tokens, shoving an entire chapter into a single vector is a mistake: the embedding averages the semantics of every sentence, so a query about one narrow fact will score poorly against a dense multi-topic chunk that happens to contain that fact.

The chunking decision controls a fundamental quality–recall trade-off:

- **Too large chunks** — high recall (the fact is somewhere in there), but low precision (the LLM gets a lot of noise and may hallucinate or ignore the signal).
- **Too small chunks** — high precision, but context loss: a sentence like "It increased by 12%" is meaningless without its referent.

Getting chunking wrong is probably the single most common cause of poor RAG performance in the wild. The optimal chunk size is domain-dependent, but the strategies below give you a principled path through the search space.

{{fig:chunkrerank-chunk-size-tradeoff}}

---

## 2. Chunking Strategies

### 2.1 Fixed-Length Chunking with Overlap

The simplest approach: split every `chunk_size` tokens, with an `overlap` of tokens shared between adjacent chunks so that sentence boundaries do not cut off context.

```python
# fixed_chunking.py — minimal, dependency-free fixed-length chunker
from __future__ import annotations
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
        >>> len(chunks)   # ceil((1000-2) / (10-2)) = 123 chunks
        123
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


# --- Quick demo ---
if __name__ == "__main__":
    doc = " ".join([f"word{i}" for i in range(200)])
    chunks = list(fixed_chunk(doc, chunk_size=50, overlap=10))
    print(f"Produced {len(chunks)} chunks from 200 tokens")
    print(f"First chunk length: {len(chunks[0].split())} words")
    print(f"Last chunk:         {chunks[-1][:60]}...")
```

The overlap parameter is crucial: without it, a sentence split across two chunks may lose its grammatical antecedent in both halves. Typical production values are `chunk_size=256–512` tokens with `overlap=32–64`.

### 2.2 Semantic Chunking

Fixed-length chunking is agnostic to content structure. Semantic chunking exploits embedding similarity to find natural topic breaks.

The algorithm:
1. Split text into sentences (using a sentence tokenizer like NLTK or spaCy).
2. Embed each sentence.
3. Compute the cosine similarity between consecutive sentence embeddings.
4. Mark a chunk boundary wherever the similarity drops below a threshold (or is a local minimum).

```python
# semantic_chunking.py — embed-based semantic chunker
from __future__ import annotations
import numpy as np
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


# --- Illustrative usage (replace embed_fn with your actual model) ---
if __name__ == "__main__":
    import random

    # Mock embed function: random vectors — replace with real embeddings
    def mock_embed(texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(texts), 384)).astype(np.float32)

    sample = [f"Sentence {i} about {'topic A' if i < 5 else 'topic B'}."
              for i in range(10)]
    result = semantic_chunk(sample, mock_embed, threshold=0.5, min_sentences=2)
    print(f"Produced {len(result)} semantic chunks")
```

Semantic chunking produces more coherent chunks but is slower (requires an embedding call) and more sensitive to the threshold. A common tuning strategy: run the chunker on a held-out set, visualize the similarity curve, and set the threshold at the 15th percentile of observed similarities.

### 2.3 Structure-Aware Chunking

Real documents have structure: headings, paragraphs, bullet lists, code blocks. Respecting that structure almost always outperforms purely statistical methods.

```python
# structural_chunking.py — Markdown-aware recursive splitter
from __future__ import annotations
import re


HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


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
    boundaries = [(m.start(), m.group().strip()) for m in HEADING_RE.finditer(text)]
    boundaries.append((len(text), ""))  # sentinel

    prev_end = 0
    current_heading = ""
    current_level = 0

    for i, (pos, heading_text) in enumerate(boundaries):
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
        current_level = len(heading_text) - len(heading_text.lstrip("#"))
        prev_end = pos

    return chunks
```

### 2.4 Late Chunking

Late chunking, introduced by Günther et al. (2024) and popularized in the context of `jina-embeddings-v2`, inverts the usual order of operations. Instead of chunking first and then embedding, we:

1. Pass the **entire document** (or a large passage) through the transformer encoder to produce contextualized token embeddings.
2. Mean-pool over the token spans that correspond to each logical chunk.

This is valuable because the attention mechanism lets every token see its full document context before the pooling boundary is applied. A sentence like "It increased by 12%" gets an embedding that encodes what "it" refers to, because the self-attention already saw the preceding paragraph.

{{fig:chunkrerank-late-chunking-arch}}

The constraint: the full document must fit within the model's context window. For documents longer than ~8k tokens you can apply late chunking within sliding windows.

```python
# late_chunking.py — illustrative late chunking with a HuggingFace model
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModel
from typing import NamedTuple


class LateChunk(NamedTuple):
    text: str
    embedding: torch.Tensor  # shape (D,)
    start_char: int
    end_char: int


def late_chunk_document(
    document: str,
    chunk_boundaries: list[tuple[int, int]],  # (start_char, end_char) pairs
    model_name: str = "jinaai/jina-embeddings-v2-base-en",
    device: str = "cpu",
) -> list[LateChunk]:
    """
    Apply late chunking: encode full document once, pool over chunk spans.

    Args:
        document:          Full document string.
        chunk_boundaries:  List of (start_char, end_char) defining each chunk.
                           These can be obtained from structural_chunking above,
                           or any other boundary detection method.
        model_name:        HuggingFace encoder model (must support long context).
        device:            'cpu' or 'cuda'.

    Returns:
        List of LateChunk named tuples with text + contextual embedding.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Tokenize entire document, keep char-to-token mapping
    encoding = tokenizer(
        document,
        return_tensors="pt",
        return_offsets_mapping=True,  # crucial: gives (char_start, char_end) per token
        truncation=True,              # truncate to model max length if needed
        max_length=8192,
    )
    offset_mapping = encoding.pop("offset_mapping")[0]  # (seq_len, 2)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # token_embeddings: (1, seq_len, hidden_size)
        token_embeddings = outputs.last_hidden_state[0]  # (seq_len, D)

    chunks: list[LateChunk] = []
    for start_char, end_char in chunk_boundaries:
        # Find which token indices correspond to this character span
        token_mask = (
            (offset_mapping[:, 0] >= start_char) &
            (offset_mapping[:, 1] <= end_char)
        )
        span_embeddings = token_embeddings[token_mask]  # (span_len, D)

        if span_embeddings.shape[0] == 0:
            continue  # skip empty spans (e.g., punctuation-only)

        # Mean pooling over the span's token embeddings
        chunk_embedding = span_embeddings.mean(dim=0)  # (D,)

        # L2-normalize for cosine similarity
        chunk_embedding = chunk_embedding / (chunk_embedding.norm() + 1e-9)

        chunks.append(LateChunk(
            text=document[start_char:end_char],
            embedding=chunk_embedding.cpu(),
            start_char=start_char,
            end_char=end_char,
        ))

    return chunks
```

---

## 3. Hybrid Search: BM25 + Dense Retrieval

Dense retrieval (see [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html)) excels at semantic similarity but struggles with exact-match recall. If a user asks about "CVE-2023-44487" or a specific product SKU, the embedding of that query may not land near the embedding of the document that contains that exact string — because the model generalizes. BM25, the classic TF-IDF variant, handles these cases natively.

**BM25 score** for query $q$ against document $d$:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

where $f(t, d)$ is term frequency in $d$, $|d|$ is document length in words, $\text{avgdl}$ is the average document length in the corpus, and $k_1 \approx 1.2\text{–}2.0$, $b \approx 0.75$ are tuning constants. The IDF is:

$$
\text{IDF}(t) = \log\!\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)
$$

with $N$ the corpus size and $n(t)$ the number of documents containing term $t$.

### 3.1 Reciprocal Rank Fusion

Given a BM25 ranking and a dense ranking, how do you combine them? Cormack et al. (2009) introduced **Reciprocal Rank Fusion (RRF)**, which is both elegant and robust:

$$
\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}
$$

where $k = 60$ is the smoothing constant (documents that don't appear in a ranking are simply omitted from that term). The document with the highest combined RRF score wins.

RRF does not require score normalization — it only uses the rank ordinal, which makes it immune to the scale mismatch between BM25 scores (roughly 0–20) and cosine similarities (roughly 0.5–1.0).

{{fig:chunkrerank-rrf-fusion}}

```python
# hybrid_search.py — BM25 + dense retrieval with RRF fusion
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Callable
import math
import numpy as np


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
```

!!! example "Worked example: RRF magnitudes"
    Suppose we have 5 documents. BM25 ranks them [D2, D1, D3, D4, D5] and dense
    retrieval ranks them [D1, D3, D2, D5, D4].

    With $k=60$:

    | Doc | BM25 rank | Dense rank | BM25 term | Dense term | RRF score |
    |-----|-----------|-----------|-----------|------------|-----------|
    | D1  | 2         | 1         | 1/62      | 1/61       | 0.0323    |
    | D2  | 1         | 3         | 1/61      | 1/63       | 0.0321    |
    | D3  | 3         | 2         | 1/63      | 1/62       | 0.0319    |
    | D4  | 4         | 5         | 1/64      | 1/65       | 0.0311    |
    | D5  | 5         | 4         | 1/65      | 1/64       | 0.0311    |

    D1 wins in the fused ranking despite being second in BM25, because its
    dense rank of 1 contributes a large term. The $k=60$ constant prevents any
    single top-1 ranking from dominating completely; increasing $k$ makes the
    fusion more conservative and score-stable.

---

## 4. Cross-Encoder Rerankers

The retrieval stage (BM25 or dense) must be fast — often processing millions of documents in tens of milliseconds. Speed requires a bi-encoder architecture: query and document are embedded independently, and similarity is a cheap dot product. The downside: the query and document tokens never "see" each other during encoding, so nuanced relevance signals (especially multi-hop or contrastive relevance) can be missed.

A **cross-encoder** receives the concatenation `[query; document]` as a single sequence and produces a scalar relevance score. Because every query token attends to every document token, the model can reason about fine-grained relevance at the cost of $O(N)$ forward passes (one per candidate). This makes cross-encoders unsuitable for first-stage retrieval but ideal for **reranking** a small shortlist of, say, 20–100 candidates.

The two-stage pipeline:

{{fig:chunkrerank-two-stage-rerank-pipeline}}

```python
# cross_encoder_rerank.py — cross-encoder reranking with sentence-transformers
from __future__ import annotations
from sentence_transformers import CrossEncoder
import numpy as np


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


# --- Illustrative latency numbers ---
# On a single A100:
#   cross-encoder/ms-marco-MiniLM-L-6-v2, 100 candidates, avg passage 200 tokens:
#   ~60ms per query batch
# On CPU (Intel Xeon):
#   ~400ms for the same 100 candidates
```

### 4.1 Training Your Own Reranker

If you have domain-specific relevance labels (from click logs, human annotations, or distillation from a more powerful model), you can fine-tune a cross-encoder with a binary cross-entropy loss on (query, positive, negative) triples, or a listwise ranking loss like ListNet.

```python
# reranker_finetune.py — pointwise cross-entropy fine-tuning sketch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# Training data: list of InputExample(texts=[query, passage], label=1.0 or 0.0)
train_examples = [
    InputExample(texts=["What is RRF?", "Reciprocal rank fusion combines..."], label=1.0),
    InputExample(texts=["What is RRF?", "The capital of France is Paris."], label=0.0),
    # ... many more examples from your domain
]

# CrossEncoder with a binary classification head
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Fine-tune: 1 epoch is often enough for domain adaptation
model.fit(
    train_dataloader=train_dataloader,
    epochs=1,
    warmup_steps=50,
    output_path="./my-domain-reranker",
    show_progress_bar=True,
)
```

---

## 5. Query Rewriting and HyDE

The query the user types is rarely optimal for retrieval. It may be short, colloquial, or implicit (assuming context from earlier conversation turns). Two complementary techniques address this.

### 5.1 Query Rewriting and Expansion

**Multi-query expansion** generates several paraphrases of the original query, retrieves for each, and unions the result sets (deduplicating by document ID). This dramatically improves recall for queries that can be stated multiple ways.

```python
# query_rewriting.py — LLM-powered multi-query expansion
from __future__ import annotations
import json
import re


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
```

### 5.2 HyDE — Hypothetical Document Embeddings

HyDE (Gao et al., 2022) takes a different approach: instead of embedding the (short, vague) query, ask the LLM to generate a **hypothetical document** that would answer the query, then embed that document and retrieve using its embedding.

The intuition: a hypothetical answer document and a real answer document inhabit closer regions of embedding space than the query and the answer document do, because they share vocabulary, entity mentions, and syntactic structure.

{{fig:chunkrerank-hyde-embedding-space}}

```python
# hyde.py — Hypothetical Document Embeddings for improved dense retrieval
from __future__ import annotations
import numpy as np


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


# --- When to use HyDE vs raw query embedding ---
#
# Use HyDE when:
#  - Your corpus uses formal or technical language that differs from query style
#  - Queries are short and ambiguous (e.g., "transformer memory management")
#  - You can afford 1-2 extra LLM calls per query (latency budget ~200ms)
#
# Avoid HyDE when:
#  - Queries are already long and specific
#  - LLM hallucinations in the hypothesis could mislead retrieval
#  - Low-latency SLA (< 50ms) prevents extra LLM calls
```

!!! warning "HyDE hallucination risk"
    The LLM's hypothetical document will contain plausible-sounding but possibly
    incorrect facts. This is fine for retrieval (you only use the embedding, not
    the text), but be careful not to accidentally include the hypothesis in the
    LLM context. Always retrieve from the real corpus using the HyDE embedding,
    then pass the retrieved real documents to the generator.

---

## 6. Metadata Filtering

Purely semantic retrieval treats all documents as equally eligible. In practice, many queries have hard constraints: "only show me documents from Q4 2024", "only from the 'legal' category", "only for product model XYZ-500".

Metadata filtering applies these constraints **before** or **during** the ANN search, drastically reducing the candidate pool and improving precision.

### 6.1 Pre-filtering vs Post-filtering

**Pre-filter** (filter before ANN search): restrict the index to only documents matching the metadata predicate, then run ANN on that subset. This is exact but slow if the filter is selective (you need an efficient inverted index on metadata fields, not just vectors).

**Post-filter** (retrieve top-k, then filter): fast, but you may need to retrieve a large k to guarantee that filtered results cover the top-k meaningful hits. Risk: if the filter is very selective, you waste most of your retrieval budget.

Modern vector databases (Qdrant, Pinecone, Weaviate, Milvus) support **hybrid pre/post-filtering** with HNSW graph filterable attributes. For implementation details see [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

```python
# metadata_filtering.py — metadata-aware retrieval with Qdrant
from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import numpy as np


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
```

### 6.2 Parent-Child Document Retrieval

A useful pattern: index fine-grained **child chunks** (e.g., single sentences or 128-token windows) for high-precision retrieval, but when a child chunk is retrieved, return its **parent document** (e.g., the full paragraph or section) as the LLM context. This gives the LLM the surrounding context it needs while keeping the retrieval signal sharp.

```python
# parent_child_retrieval.py — parent document retrieval pattern
from __future__ import annotations
from dataclasses import dataclass
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
```

---

## 7. Putting It All Together: A Production RAG Pipeline

Here is a complete, annotated pipeline that combines all the techniques discussed above, with configurable stages.

```python
# rag_pipeline.py — production-grade RAG pipeline
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
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
    dense_index,          # vector store with .search(embedding, top_k) -> list[Document]
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
    # Normalize dense results to (idx, score) format
    dense_results = [(i, float(s)) for i, s in enumerate(dense_results_raw)]

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
```

### 7.1 Choosing Configuration for Your Domain

The table below summarizes empirical guidance for configuring the pipeline. These are starting points, not guarantees — always evaluate on your domain.

| Domain characteristic | Recommended setting |
|---|---|
| Precise entity lookup (medical codes, SKUs, legal citations) | BM25 weight high; use hybrid with RRF; add metadata filter on document type |
| Conversational QA over prose | Dense retrieval dominant; enable HyDE; semantic chunking |
| Long documents with clear section structure | Structural chunking + parent-child retrieval |
| Multilingual corpus | Use a multilingual cross-encoder reranker (e.g., `cross-encoder/ms-marco-electra-base` fine-tuned with mMARCO) |
| Low-latency SLA (< 100ms) | Skip HyDE; use MiniLM reranker or skip reranking; pre-filter metadata |
| High-stakes accuracy requirement | Enable all stages: HyDE + hybrid + full reranker; re-evaluate every 30 days |

---

## 8. Diagnostics and Iteration

A RAG pipeline has many knobs, and it is easy to tune one component in isolation while unknowingly degrading another. The rigorous approach is to evaluate each stage separately with stage-specific metrics, then measure the end-to-end quality.

```python
# rag_diagnostics.py — evaluate each pipeline stage independently
from __future__ import annotations
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


# Typical baseline numbers to aim for on a reasonably clean corpus:
#   Recall@10:  > 0.85 (retrieval stage)
#   MRR:        > 0.70
#   After rerank, Recall@5: > 0.80
```

!!! interview "Interview Corner"
    **Q:** You have a RAG system where retrieval recall is high but generation
    quality is poor — the LLM often ignores the retrieved context or produces
    hallucinations. What might be wrong, and how would you debug it?

    **A:** Several failure modes can cause this. First, check chunk quality: if
    chunks are too large, the relevant sentence may be buried in noise; if too
    small, they may lack the context the LLM needs to interpret them. Second,
    check for retrieval-generation mismatch: the retrieved chunks might be
    superficially relevant (high embedding similarity) but not actually answer
    the question — a cross-encoder reranker that scores precise relevance often
    fixes this. Third, inspect the prompt format: if the context is pasted at
    the end of a very long prompt, the LLM may exhibit lost-in-the-middle
    behavior (Liu et al., 2023) and downweight it; try placing the most
    relevant chunks at the top or bottom of the context window. Fourth, consider
    whether a faithfulness metric (e.g., RAGAS) shows the answer is entailed by
    the retrieved text — if not, the retrieval is fetching the wrong documents
    regardless of cosine score. Finally, for factual queries, adding metadata
    filters to restrict freshness or source authority can dramatically improve
    generation quality by reducing noisy candidates.

---

!!! key "Key Takeaways"
    - **Chunking strategy is the single highest-leverage RAG decision.** Fixed chunking is fast but naive; semantic and structural chunking better preserve coherence. Late chunking gives context-aware embeddings at the cost of requiring the full document to fit in the encoder.
    - **Hybrid search (BM25 + dense) consistently outperforms either alone.** BM25 handles exact-match queries (rare terms, product codes, names); dense retrieval handles paraphrase and semantic queries. Reciprocal Rank Fusion (RRF) is a robust, parameter-light way to combine the two ranked lists.
    - **RRF only uses rank ordinals**, not raw scores, making it immune to scale mismatches between BM25 and cosine similarities. The smoothing constant $k=60$ prevents any single top-ranked document from dominating.
    - **Cross-encoder rerankers improve precision dramatically** at the cost of latency. Run retrieval with a large top-k (40–100), rerank with a cross-encoder, and pass only the top-5 to the LLM. The two-stage pipeline amortizes the encoder cost over a small set.
    - **HyDE** shifts the retrieval problem from query-to-document to document-to-document matching, which is easier for bi-encoders. Use it when your query vocabulary diverges from document vocabulary, but never include the hypothetical document in the LLM prompt — only use its embedding.
    - **Metadata filtering** is essential for production systems with heterogeneous corpora. Pre-filtering in the vector index (via Qdrant, Weaviate, etc.) is generally more efficient than post-filtering for selective predicates.
    - **Parent-child retrieval** gives you the best of both worlds: fine-grained retrieval signal from small child chunks and rich LLM context from large parent documents.
    - **Measure each stage independently** (Recall@k, MRR) before optimizing end-to-end RAGAS or LLM judge scores. A component that looks good in isolation may be bottlenecked by the stage before it.

---

!!! sota "State of the Art & Resources (2026)"
    Hybrid retrieval (BM25 + dense) with cross-encoder reranking is now the production standard for RAG, with late chunking and LLM-based query rewriting closing the remaining gap between prototype and production quality. Evaluation benchmarks like BEIR and frameworks like RAGAS have made systematic pipeline comparison routine.

    **Foundational work**

    - [Nogueira & Cho, *Passage Re-ranking with BERT* (2019)](https://arxiv.org/abs/1901.04085) — introduced the cross-encoder paradigm that underpins every modern reranker; MS-MARCO models trace directly to this paper.
    - [Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (SIGIR 2009)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf) — the original RRF paper; the k=60 constant is still used unchanged in virtually all hybrid search stacks.
    - [Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022)](https://arxiv.org/abs/2212.10496) — the HyDE paper; shows that embedding a hypothetical answer document consistently outperforms embedding the raw query.

    **Recent advances (2023–2026)**

    - [Günther et al., *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models* (2024)](https://arxiv.org/abs/2409.04701) — late chunking lets every chunk embedding see its full document context via self-attention before pooling, often beating fixed-window chunking without retraining.
    - [Thakur et al., *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (NeurIPS 2021)](https://arxiv.org/abs/2104.08663) — 18-dataset benchmark that exposed BM25 as a surprisingly hard-to-beat baseline and set the standard for evaluating retrieval generalization.
    - [Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (TACL 2024)](https://arxiv.org/abs/2307.03172) — empirical evidence that LLMs downweight middle-context information; directly motivates placing the highest-ranked chunks at the top or bottom of the RAG prompt.

    **Open-source & tools**

    - [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) — canonical Python library for bi-encoder embeddings and cross-encoder rerankers; includes `cross-encoder/ms-marco-MiniLM-L-6-v2` and dozens of production-ready models.
    - [jina-ai/late-chunking](https://github.com/jina-ai/late-chunking) — reference implementation and evaluation code for the late chunking method.
    - [xhluca/bm25s](https://github.com/xhluca/bm25s) — ultrafast BM25 in pure Python backed by sparse matrices; orders of magnitude faster than rank-bm25 for large corpora.
    - [RUC-NLPIR/FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) — modular RAG research toolkit with 36 benchmark datasets and 23 RAG algorithms; excellent for ablating chunking/retrieval/reranking choices.

    **Go deeper**

    - [explodinggradients/ragas](https://github.com/explodinggradients/ragas) — the standard framework for reference-free RAG evaluation (faithfulness, answer relevancy, context precision); integrates with LangChain and LlamaIndex.

## Further Reading

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) — the canonical reference for BM25 derivation and tuning.
- Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR 2009) — original RRF paper with empirical comparisons.
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) — introduced the cross-encoder reranking paradigm for neural IR; the MS MARCO models trace to this work.
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2022) — the HyDE paper, with ablations showing when it helps and when it hurts.
- Günther et al., "Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents" (2023) — introduces late chunking and benchmarks it against standard chunking.
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023) — empirical evidence that LLMs underweight information placed in the middle of long contexts; directly motivates context ordering in RAG prompts.
- Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training" (ICML 2020) — early end-to-end trainable RAG architecture that motivates the field.
- Ma et al., "Query Rewriting in Retrieval-Augmented Large Language Models" (EMNLP 2023) — systematic study of query rewriting strategies including multi-query expansion.
- Sentence Transformers library (`UKPLab/sentence-transformers`) — the go-to Python package for cross-encoder models and bi-encoder fine-tuning.
