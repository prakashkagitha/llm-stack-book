# 9.3 Retrieval-Augmented Generation Architectures

A language model trained six months ago does not know about last week's earnings call. A 7-billion-parameter model that has memorized Wikipedia cannot tell you what is in your company's internal runbook. Even a frontier model with a 128 k-token context window will confidently confabulate a citation if the relevant fact falls outside its training distribution. Retrieval-Augmented Generation (RAG) addresses all three problems with a single architectural shift: rather than forcing the model to answer from parametric memory alone, we first retrieve the most relevant documents from an external store and inject them into the context before the model generates its answer.

This chapter dissects the full RAG pipeline — from corpus ingestion to final generation — explains why each design decision matters, catalogs the most important failure modes, and shows you how to build a minimal but production-faithful implementation from scratch. We also cover how to measure whether your RAG system is actually working with the RAGAS framework of faithfulness, answer relevance, and context precision.

Related chapters: [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html) covers how dense encoders produce the vectors we retrieve against. [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html) covers the indexing and search algorithms in depth. [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html) extends the ideas here with more sophisticated retrieval pipelines. [Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html) covers the frontier.

## Why RAG Exists: Three Hard Problems

### The Parametric Memory Wall

A transformer's weights are a lossy, distributed compression of the training corpus. Factual knowledge is stored implicitly across billions of parameters, which means:

1. **Temporal staleness.** Pre-training is expensive and infrequent. A model trained on data through mid-2024 cannot answer questions about events after that date.
2. **Capacity limits.** Even a 70 B-parameter model has finite capacity. Long-tail facts, private documents, and domain-specific knowledge may never be reliably memorized.
3. **No attribution.** When a model answers from parametric memory, there is no provenance trail — no page number, no document, no timestamp. For enterprise or legal contexts, this is often a blocker.

### What RAG Changes

RAG augments each inference call with a retrieval step. The model's parametric knowledge becomes a reasoning engine rather than a fact store:

$$
P(y \mid q) \;\longrightarrow\; P(y \mid q,\, \mathcal{D}_{q})
$$

where $q$ is the user query and $\mathcal{D}_{q} = \{d_1, d_2, \ldots, d_k\}$ is the set of $k$ documents retrieved for that specific query. The generation is conditioned on fresh, verifiable context. Updating the knowledge base requires only re-indexing documents, not re-training the model.

This trades off context-window tokens for factual accuracy, freshness, and citability — a trade that is almost always worth making for knowledge-intensive tasks.

## The RAG Pipeline: Five Stages

{{fig:rag-pipeline}}

The full pipeline from raw documents to final answer has five conceptually distinct stages. Each is a design space in its own right.

{{fig:rag-arch-pipeline-five-stages}}

### Stage 1 — Chunking

A raw document (PDF, HTML, code file, transcript) must be split into chunks that fit inside a retrieval unit. Chunks that are too large decrease retrieval precision because a 2,000-token chunk retrieved for one sentence drags in irrelevant text. Chunks that are too small lose local context — a sentence about "the acquisition" means nothing without the surrounding paragraph.

Common strategies:

| Strategy | Size | When to use |
|---|---|---|
| Fixed-size sliding window | 256–512 tokens, 10–20% overlap | Baseline; works for structured prose |
| Sentence/paragraph boundary | Varies | Better coherence, avoids splitting mid-idea |
| Semantic chunking | Dynamic | Splits where embedding similarity drops |
| Document-aware (section headers) | Varies | PDFs, wikis, Markdown with structure |

Chapter [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html) covers the design space in full detail.

### Stage 2 — Embedding

Each chunk is encoded into a dense vector $\mathbf{v} \in \mathbb{R}^d$ by a bi-encoder (also called a dual-encoder). The same encoder maps the query to $\mathbf{q} \in \mathbb{R}^d$. Retrieval is then a nearest-neighbor search in this space. See [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html) for the full treatment of encoder architectures and training.

Popular open-source choices include models from the `sentence-transformers` family, `e5-large-v2`, `bge-m3`, and others. Typical dimensionality $d$ ranges from 384 to 1536.

### Stage 3 — Indexing

The chunk vectors are stored in a vector database (FAISS, Pinecone, Weaviate, Qdrant, pgvector, etc.) with an Approximate Nearest Neighbor (ANN) index such as HNSW or IVF. See [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html) for internals.

At query time, the query vector is compared against all indexed vectors, and the top-$k$ most similar chunks (by cosine similarity or dot product) are returned in sub-millisecond to low-millisecond time even for corpora of tens of millions of chunks.

### Stage 4 — Retrieve (and optionally Rerank)

The ANN index returns approximate top-$k$ results, typically $k = 5$–$20$. A cross-encoder reranker (e.g., `bge-reranker-large`) then scores the query alongside each candidate chunk jointly, producing a more accurate relevance ranking. The top-$k'$ (typically $k' = 3$–$5$) chunks after reranking form the retrieved context.

{{fig:bi-encoder-vs-cross-encoder}}

### Stage 5 — Generate

The retrieved chunks are concatenated into a context block, formatted with a prompt template, and sent to the LLM. The model generates an answer grounded in the retrieved evidence.

```text
System: You are a helpful assistant. Answer based on the context below.
        If the answer is not in the context, say "I don't know."

Context:
[Chunk 1 text]
---
[Chunk 2 text]
---
[Chunk 3 text]

User: {query}
```

## A Minimal RAG Implementation

The following implementation is self-contained and runnable. It uses `sentence-transformers` for embedding, `faiss-cpu` for indexing, and the `openai` client for generation. Every design decision is annotated.

```python
"""
Minimal RAG from scratch.
Dependencies: sentence-transformers, faiss-cpu, openai
  pip install sentence-transformers faiss-cpu openai
"""

import re
import textwrap
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# 1. Corpus (replace with your real documents)
# ─────────────────────────────────────────────
CORPUS = [
    """FlashAttention is an IO-aware exact attention algorithm. It tiles the
    Q, K, V matrices into blocks that fit in SRAM and computes attention
    without materializing the full N×N attention matrix, reducing memory
    from O(N²) to O(N). Published by Dao et al. in 2022.""",

    """RLHF (Reinforcement Learning from Human Feedback) is a post-training
    technique that fine-tunes a language model to align with human
    preferences. It requires a reward model trained on comparison data and
    uses PPO to optimize the policy against that reward signal.""",

    """RAG (Retrieval-Augmented Generation) was introduced by Lewis et al.
    in 2020. It combines a dense retriever (DPR) with a seq2seq generator
    (BART) and trains both end-to-end. The retriever is frozen during
    inference but the generator is conditioned on retrieved documents.""",

    """The Chinchilla scaling law (Hoffmann et al., 2022) showed that for a
    given compute budget, training tokens should scale roughly 1:1 with
    model parameters. A 70 B model is optimally trained on about 1.4 T
    tokens, not the ~300 B tokens used for the original GPT-3 scale models.""",

    """FAISS (Facebook AI Similarity Search) is a library for efficient
    similarity search. Its IVF index (Inverted File Index) partitions
    vectors into Voronoi cells using k-means clustering. At query time,
    only the closest nprobe cells are searched, trading recall for speed.""",
]


# ─────────────────────────────────────────────
# 2. Chunking (trivial for this demo — each doc
#    is already one chunk; in practice you would
#    split long docs into overlapping windows)
# ─────────────────────────────────────────────
def chunk_text(text: str, max_tokens: int = 200, overlap: int = 20) -> List[str]:
    """
    Simple word-level sliding window chunker.
    For production use a sentence boundary splitter or spacy.
    """
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap  # overlap keeps context across boundaries
    return chunks


# Flatten corpus into chunks
all_chunks: List[str] = []
for doc in CORPUS:
    all_chunks.extend(chunk_text(doc))

print(f"Total chunks: {len(all_chunks)}")


# ─────────────────────────────────────────────
# 3. Embed all chunks (bi-encoder)
# ─────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
encoder = SentenceTransformer(EMBED_MODEL)

# encode() returns (n_chunks, d) float32 numpy array
chunk_embeddings = encoder.encode(
    all_chunks,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # unit-norm so dot product == cosine similarity
)
d = chunk_embeddings.shape[1]   # embedding dimensionality (384 for MiniLM)
print(f"Embedding dim: {d}, corpus size: {chunk_embeddings.shape[0]}")


# ─────────────────────────────────────────────
# 4. Build FAISS index
# ─────────────────────────────────────────────
# IndexFlatIP = exact inner product (cosine on unit vectors) — fine for small corpora.
# For production use IndexIVFPQ for speed and memory efficiency at scale.
index = faiss.IndexFlatIP(d)
index.add(chunk_embeddings.astype(np.float32))
print(f"FAISS index has {index.ntotal} vectors")


# ─────────────────────────────────────────────
# 5. Retrieval function
# ─────────────────────────────────────────────
def retrieve(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """
    Encode the query and return the top-k (chunk, score) pairs.
    Score is cosine similarity in [0, 1] because vectors are unit-normed.
    """
    q_vec = encoder.encode(
        [query],
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(q_vec, k)
    # scores shape: (1, k), indices shape: (1, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:  # FAISS returns -1 for missing results
            results.append((all_chunks[idx], float(score)))
    return results


# ─────────────────────────────────────────────
# 6. Generation with injected context
# ─────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise technical assistant. Answer the user's question using
    ONLY the provided context passages. If the answer cannot be found in the
    context, respond with "I don't know based on the provided context."
    Cite the relevant passage(s) inline as [1], [2], etc.
""")


def build_context_block(retrieved: List[Tuple[str, float]]) -> str:
    lines = []
    for i, (chunk, score) in enumerate(retrieved, 1):
        lines.append(f"[{i}] (relevance={score:.3f})\n{chunk}")
    return "\n\n".join(lines)


def rag_query(query: str, k: int = 3) -> str:
    # Step A: retrieve
    retrieved = retrieve(query, k=k)
    context_block = build_context_block(retrieved)

    # Step B: generate
    client = OpenAI()  # reads OPENAI_API_KEY from env
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Question: {query}"
            ),
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,   # deterministic for RAG — no creativity needed
        max_tokens=512,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# 7. Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query = "How does FlashAttention reduce memory usage?"
    print(f"\nQ: {query}")
    answer = rag_query(query)
    print(f"A: {answer}")
```

!!! example "Worked example: memory and latency budget"
    Suppose you have a corpus of 100,000 documents, each split into roughly 10 chunks of 256 tokens. That gives $N = 10^6$ chunk vectors.

    **Index memory.** With embedding dimension $d = 1536$ (OpenAI `text-embedding-3-small`) and float32 storage:

    $$
    \text{Memory} = N \times d \times 4\,\text{bytes} = 10^6 \times 1536 \times 4 = 6.14\,\text{GB}
    $$

    That fits comfortably on a single A10G GPU with 24 GB VRAM, or on a machine with 32 GB RAM using CPU inference. For $d = 384$ (MiniLM), the same corpus is only 1.5 GB.

    **Latency budget.** An exact `IndexFlatIP` search over $10^6$ vectors at $d = 384$ takes roughly 20–50 ms on a single CPU core (FAISS is BLAS-accelerated). With an HNSW index, the same search takes under 2 ms. The embedding step for the query adds another 5–10 ms. Total retrieval latency before generation: on the order of 10–60 ms depending on index type and hardware — typically negligible compared to LLM generation time.

    **Reranker cost.** A cross-encoder reranker scoring 20 candidates against the query takes roughly 50–100 ms on a single GPU (batched). If you retrieve $k = 20$ for ANN and rerank to $k' = 5$, this is the dominant retrieval cost.

## Naive RAG and Its Failure Modes

"Naive RAG" refers to the minimal pipeline: fixed-size chunks, single-pass dense retrieval, no reranking, concatenate-and-prompt generation. Practitioners who deploy this baseline quickly run into a characteristic set of failure modes.

### Retrieval Failures

**Semantic mismatch between query and chunk vocabulary.** Dense embeddings capture meaning, not keywords. A query "What is the interest rate hike in Q3?" may miss a chunk that uses the phrase "Federal Reserve raised the federal funds rate by 75 bps in the third quarter" because the two sentences are in different parts of the semantic space if the encoder was not fine-tuned on financial text.

**Chunking boundary problems.** A chunk that begins mid-sentence ("...increased revenue, driven by strong performance in the cloud segment") provides poor context to the retriever and the reader. The embedding may not capture the key entity because it appears in the previous chunk.

**Top-k may not contain the answer at all.** For long-tail or multi-hop questions ("Who acquired the company that made the chip used in the iPhone 6?"), a single retrieval step may not find all necessary evidence. The answer requires stitching together two or more documents.

**Stale index.** If documents are updated but the index is not rebuilt, retrieved chunks may contain outdated information.

### Context Utilization Failures

**Lost-in-the-middle.** Liu et al. (2023) documented that LLMs are significantly worse at using information in the middle of a long context compared to information at the beginning or end. If you inject $k = 10$ chunks, the ones in positions 4–6 are likely to be ignored.

**Context-answer inconsistency (hallucination despite context).** The model may generate information that is *not* in the retrieved chunks, especially when the retrieved context is ambiguous or partially relevant.

**Prompt over-crowding.** Too many retrieved chunks consume the context budget, leaving insufficient room for chain-of-thought reasoning or multi-turn history. For agents, this competes with tool call outputs and conversation history. See [Context Engineering & Management](../08-agents-harness/04-context-engineering.html) for strategies.

### Ranking and Diversity Failures

**Semantic redundancy.** ANN retrieval can return five near-duplicate chunks from the same section of a document. Each chunk has a high similarity score to the query, but they provide no additional information. A Maximal Marginal Relevance (MMR) or diversity filter is needed.

**No cross-encoder reranking.** Bi-encoders embed query and document independently, so they cannot capture fine-grained interaction. A query "disadvantages of batch normalization" may retrieve the highest-scoring chunk being a general overview of batch norm that only tangentially mentions its disadvantages.

!!! warning "Common pitfall: trusting cosine similarity as a quality signal"
    A cosine similarity of 0.85 between a query and a chunk does not guarantee the chunk is relevant. In high-dimensional embedding spaces, cosine similarities cluster in a narrow band (often 0.7–0.95 for any reasonable pair). Always pair vector retrieval with a reranker for high-stakes applications.

## RAG Evaluation: RAGAS and the Three Metrics

{{fig:ragas-metric-decomposition}}

Measuring whether your RAG system is working requires disaggregating the pipeline into retrieval quality and generation quality. The RAGAS framework (Es et al., 2023) defines three core metrics that cover both concerns, all computable without human labels using an LLM-as-judge.

### Faithfulness

**Definition.** Given the retrieved context $\mathcal{D}_{q}$ and the generated answer $y$, faithfulness measures whether every claim in $y$ is entailed by $\mathcal{D}_{q}$. It is computed as:

$$
\text{Faithfulness} = \frac{|\{\text{claims in } y \text{ that are supported by } \mathcal{D}_{q}\}|}{|\{\text{claims in } y\}|}
$$

A faithfulness score of 1.0 means the answer is fully grounded. A score below 0.8 is a strong signal of hallucination.

**Implementation.** An LLM (the judge) first decomposes $y$ into a list of atomic claims, then classifies each claim as supported or unsupported by the context passages.

### Answer Relevance

**Definition.** Answer relevance measures whether the generated answer is relevant to the query, independent of whether it is faithful. A system could achieve high faithfulness (all claims supported by context) but low answer relevance (the context and answer are about a different aspect of the query).

Formally, the judge generates $n$ hypothetical questions $q_1, \ldots, q_n$ that the answer $y$ appears to address, then measures:

$$
\text{AnswerRelevance} = \frac{1}{n} \sum_{i=1}^{n} \cos(\mathbf{e}_{q_i},\, \mathbf{e}_{q})
$$

where $\mathbf{e}_{q}$ is the embedding of the original query. A high score means the hypothetical questions regenerated from the answer closely resemble the original query.

### Context Precision and Recall

Two retrieval-quality metrics round out the picture:

- **Context Precision.** Of the $k$ retrieved chunks, how many are actually useful for answering the question? Precision penalizes noisy retrieval.
- **Context Recall.** Given ground-truth relevant passages, how many were retrieved? Recall penalizes missed evidence.

$$
\text{ContextPrecision@k} = \frac{|\{\text{relevant chunks in top-}k\}|}{k}
$$

In the LLM-as-judge variant (no ground truth), the judge is asked: "Is this chunk necessary to produce the correct answer?" for each retrieved chunk.

### RAGAS in Practice

```python
"""
Minimal RAGAS-style evaluation loop.
Requires: pip install ragas datasets openai
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Build evaluation dataset in RAGAS format
# ground_truth is optional (needed for context_recall)
eval_data = [
    {
        "question": "How does FlashAttention reduce memory?",
        "answer": "FlashAttention tiles Q, K, V into SRAM blocks and avoids materializing the O(N²) attention matrix, reducing memory to O(N).",
        "contexts": [
            "FlashAttention is an IO-aware exact attention algorithm. It tiles the Q, K, V matrices into blocks that fit in SRAM...",
        ],
        "ground_truth": "FlashAttention avoids storing the full N×N attention matrix by using tiled SRAM computation.",
    },
]

dataset = Dataset.from_list(eval_data)
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(results)
# Example output:
# {'faithfulness': 0.97, 'answer_relevancy': 0.92,
#  'context_precision': 1.00, 'context_recall': 0.95}
```

!!! note "Aside: RAGAS requires an LLM judge"
    RAGAS calls an LLM (default: GPT-4) to decompose answers into claims and to score them. This means evaluation costs money and is subject to judge bias. For large-scale offline evaluation, cache judge responses or use a cheaper model for preliminary sweeps.

## The RAG Design Space

Naive RAG is a starting point, not a destination. The design space is large and each axis can be tuned independently.

### Retrieval Strategy

| Axis | Options |
|---|---|
| Index type | Dense only, sparse (BM25) only, hybrid (dense + sparse) |
| Retrieval granularity | Paragraph, sentence, document, hierarchical |
| Query rewriting | None, HyDE, step-back, multi-query expansion |
| Multi-hop | Single retrieval, iterative retrieval, chain-of-thought retrieval |
| Filtering | Metadata filters, date range, access control |

**Hybrid search** combines a BM25 sparse index (exact keyword matching) with a dense embedding index, fusing scores via Reciprocal Rank Fusion (RRF). This is almost universally better than either alone for general-purpose corpora:

$$
\text{RRF}(d, R_1, R_2) = \frac{1}{k + \text{rank}_{R_1}(d)} + \frac{1}{k + \text{rank}_{R_2}(d)}
$$

where $k = 60$ is a smoothing constant, $R_1$ is the BM25 ranking, and $R_2$ is the dense ranking.

{{fig:hybrid-rrf-fusion}}

**HyDE (Hypothetical Document Embedding)**, introduced by Gao et al. (2022), flips the retrieval problem: instead of embedding the short query directly, the LLM generates a hypothetical answer document, and that document is used as the retrieval query. Since the hypothetical document uses the same vocabulary and style as the corpus, the retrieval often improves substantially for queries where the question and answer have very different surface forms.

### Indexing Strategies

**Parent-child chunks.** Index small child chunks (128 tokens) for high retrieval precision, but when a child chunk is retrieved, look up and return its parent chunk (512 tokens) for generation context. This avoids losing context while keeping retrieval sharp.

**Summary indexes.** For each document, store a summary chunk alongside the raw chunks. The retriever can match against the summary (capturing document-level semantics), then return the full document or relevant sections.

**Knowledge graph indexes (GraphRAG).** Convert the corpus into a knowledge graph of entities and relationships, enabling retrieval by entity traversal rather than pure semantic similarity. See [Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html).

### Generator Configuration

The way retrieved context is formatted and prompted has a large impact on generation quality:

1. **Positional bias mitigation.** Shuffle retrieved chunks between runs, or put the most relevant chunk first *and* last.
2. **Explicit grounding instruction.** "Answer ONLY using the provided passages. If the answer is not there, say 'I don't know'." This reduces hallucination significantly compared to implicit grounding.
3. **Citation format.** Instructing the model to cite "[chunk index]" inline allows post-processing to verify every cited claim.
4. **Temperature.** For factual QA, temperature 0.0 or very low temperatures are preferred — we do not want creative paraphrasing of facts.

!!! interview "Interview Corner"
    **Q:** You are designing a RAG system for a customer-support chatbot. The retrieval precision is high (users get relevant passages) but faithfulness is low (the model often adds information not in the retrieved context). How would you diagnose and fix this?

    **A:** Low faithfulness with high precision is almost always a generation problem, not a retrieval problem. The model is generating from its parametric memory instead of strictly following the context.

    Diagnosis: Run RAGAS faithfulness on a sample. Decompose generated answers into atomic claims and check which claims are unsupported. Look for patterns — does the model add details about products, prices, or policies that are not in the chunks?

    Fixes, in order of effort:
    1. Strengthen the system prompt: "You MUST answer using only the passages below. Do not add information from your own knowledge."
    2. Add chain-of-thought grounding: ask the model to first quote the relevant sentence, then answer based on the quote.
    3. Use a model fine-tuned for RAG (e.g., a model fine-tuned with RAFT — Retrieval-Augmented Fine Tuning — which trains the model to distinguish relevant from irrelevant context).
    4. Use a smaller, more instruction-following model that has less parametric knowledge to "leak."
    5. Post-process outputs with an NLI classifier to flag ungrounded claims before serving.

## Putting It Together: A More Complete Pipeline

Here is a pipeline that incorporates hybrid retrieval, parent-child chunk lookup, and explicit faithfulness checking. This is close to a production-quality open-source RAG system.

```python
"""
Production-style RAG pipeline:
  - BM25 + dense hybrid retrieval (RRF fusion)
  - Parent-child chunk hierarchy
  - Faithfulness check (NLI-based)
Dependencies: rank_bm25, sentence-transformers, faiss-cpu, transformers
  pip install rank_bm25 sentence-transformers faiss-cpu transformers
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline as hf_pipeline
import faiss
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
class Chunk:
    def __init__(self, text: str, chunk_id: int, parent_id: int):
        self.text = text
        self.chunk_id = chunk_id
        self.parent_id = parent_id   # index into parent_docs list

class Document:
    def __init__(self, full_text: str, doc_id: int):
        self.full_text = full_text
        self.doc_id = doc_id


# ─────────────────────────────────────────────
# Build parent-child index
# ─────────────────────────────────────────────
def build_parent_child_index(
    documents: List[str],
    child_size: int = 128,    # words per child chunk
    overlap: int = 16,
) -> Tuple[List[Document], List[Chunk]]:
    parent_docs = [Document(doc, i) for i, doc in enumerate(documents)]
    child_chunks = []
    chunk_id = 0
    for doc in parent_docs:
        words = doc.full_text.split()
        start = 0
        while start < len(words):
            end = min(start + child_size, len(words))
            text = " ".join(words[start:end])
            child_chunks.append(Chunk(text, chunk_id, doc.doc_id))
            chunk_id += 1
            if end == len(words):
                break
            start = end - overlap
    return parent_docs, child_chunks


# ─────────────────────────────────────────────
# Hybrid retrieval: BM25 + dense, fused by RRF
# ─────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embed_model: str):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]

        # BM25
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Dense (FAISS)
        self.encoder = SentenceTransformer(embed_model)
        embeddings = self.encoder.encode(
            self.texts, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    def _rrf(self, *rank_lists, k: int = 60) -> Dict[int, float]:
        """Reciprocal Rank Fusion over multiple ranked lists of chunk indices."""
        scores: Dict[int, float] = {}
        for ranked in rank_lists:
            for rank, idx in enumerate(ranked):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def retrieve(self, query: str, top_n: int = 10) -> List[Tuple[Chunk, float]]:
        # BM25 ranking
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranked = list(np.argsort(bm25_scores)[::-1][:top_n])

        # Dense ranking
        q_vec = self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        _, dense_indices = self.index.search(q_vec, top_n)
        dense_ranked = list(dense_indices[0])

        # Fuse
        fused = self._rrf(bm25_ranked, dense_ranked, k=60)
        sorted_ids = sorted(fused.items(), key=lambda x: -x[1])[:top_n]
        return [(self.chunks[idx], score) for idx, score in sorted_ids]


# ─────────────────────────────────────────────
# Cross-encoder reranker
# ─────────────────────────────────────────────
class Reranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 3,
    ) -> List[Tuple[Chunk, float]]:
        pairs = [(query, c.text) for c, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda x: -x[1],
        )
        return ranked[:top_k]


# ─────────────────────────────────────────────
# NLI faithfulness checker (optional post-filter)
# ─────────────────────────────────────────────
class FaithfulnessChecker:
    """
    Uses an NLI model to check whether the generated answer
    is entailed by the retrieved context.
    Returns a float in [0, 1]; <0.5 suggests hallucination.
    """
    def __init__(self):
        # mnli model: entailment, neutral, contradiction
        self.nli = hf_pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
        )

    def score(self, premise: str, hypothesis: str) -> float:
        """Returns probability of entailment."""
        result = self.nli(
            f"{premise} [SEP] {hypothesis}",
            truncation=True,
            max_length=512,
            top_k=None,  # return scores for all labels, not just the top-1
        )
        label_score = {r["label"]: r["score"] for r in result}
        return label_score.get("entailment", 0.0)

    def check_answer(self, answer: str, context: str, threshold: float = 0.5) -> bool:
        """Return True if answer appears to be grounded in context."""
        return self.score(premise=context, hypothesis=answer) >= threshold


# ─────────────────────────────────────────────
# Assemble the full pipeline
# ─────────────────────────────────────────────
def run_pipeline(
    query: str,
    retriever: HybridRetriever,
    reranker: Reranker,
    parent_docs: List[Document],
    faithfulness_checker: FaithfulnessChecker,
    generate_fn,  # callable(prompt: str) -> str
) -> Dict:
    # Step 1: Retrieve child chunks
    candidates = retriever.retrieve(query, top_n=20)

    # Step 2: Rerank
    top_chunks = reranker.rerank(query, candidates, top_k=3)

    # Step 3: Look up parent documents for richer context
    seen_parents = set()
    context_parts = []
    for chunk, score in top_chunks:
        parent_id = chunk.parent_id
        if parent_id not in seen_parents:
            context_parts.append(parent_docs[parent_id].full_text)
            seen_parents.add(parent_id)

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Generate
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer the question based ONLY on the context above.\n"
        f"Question: {query}\nAnswer:"
    )
    answer = generate_fn(prompt)

    # Step 5: Faithfulness check
    is_faithful = faithfulness_checker.check_answer(answer, context)

    return {
        "answer": answer,
        "context": context,
        "top_chunks": [(c.text, s) for c, s in top_chunks],
        "faithful": is_faithful,
    }
```

## Long-Context LLMs vs RAG: The Design Decision

A recurring question for practitioners is: "If the model has a 128 k-token context window, why do I need RAG at all — can't I just stuff the whole knowledge base into the prompt?"

The honest answer is: it depends.

| Consideration | Long-Context LLM | RAG |
|---|---|---|
| Corpus size | Must fit in context; cost scales $O(N)$ with tokens | Scales to billions of documents |
| Freshness | Re-prompt each time (cheap) | Re-index on document update |
| Latency | Prefilling 100 k tokens takes seconds | Retrieval adds ~50–100 ms |
| Inference cost | Very high (attention is $O(N^2)$ in prefill) | Cheap: only top-$k$ docs injected |
| Retrieval precision | Perfect (nothing is missed) | Recall depends on retriever quality |
| Lost-in-middle | Significant beyond ~32 k tokens | Controlled; inject 1–5 k tokens |

For corpora that fit in a long context (e.g., a single legal contract, a codebase under 100 k tokens), long-context prompting is simpler and more reliable. For large, dynamic, multi-document corpora (knowledge bases, enterprise wikis, customer support databases), RAG is the right tool. See [Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html) for a detailed comparison.

The two approaches also compose: **iterative RAG** retrieves a small context, reasons over it, then decides whether to retrieve more, effectively using long-context reasoning to stitch together multi-hop evidence. [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html) covers this pattern.

## Serving a RAG System in Production

A few operational concerns that come up in every production RAG deployment:

**Indexing latency vs freshness.** Real-time re-embedding and re-indexing is expensive. Common patterns: batch re-index nightly; maintain a "hot" real-time index for recent documents and a "cold" indexed corpus for historical ones; serve queries against both and merge results.

**Multi-tenancy and access control.** Retrieved documents must respect document-level permissions. FAISS alone has no ACL support. Solutions include: metadata-filtered retrieval (Qdrant, Weaviate, Pinecone all support filter expressions), per-tenant index shards, or post-retrieval filtering.

**Caching.** Embedding the same query repeatedly is wasteful. Cache query embeddings keyed by the (normalized) query string. Also cache retrieval results for frequent queries. See [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html).

**Monitoring.** Log every query, retrieved chunk IDs, faithfulness scores, and user feedback. A RAGAS-style offline eval batch should run on a sample of production logs nightly. See [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html).

**Embedding model upgrades.** If you switch encoder models, you must re-embed the entire corpus — the new model's embedding space is incompatible with the old one. Plan for full re-indexing as a first-class operation. Dual-write during migration: index new documents with both old and new models until the old index is retired.

!!! tip "Practitioner tip: start with BM25 only"
    Before investing in a dense embedding pipeline, run BM25 (sparse retrieval) alone on your corpus. For many enterprise knowledge bases with domain-specific terminology, BM25 outperforms a generic dense model. Once you have a BM25 baseline, add a dense retriever and measure whether hybrid search beats the baseline. Never optimize before measuring.

!!! key "Key Takeaways"
    - RAG addresses three core limitations of parametric LLMs: temporal staleness, capacity limits, and lack of attribution. It conditions generation on freshly retrieved external documents rather than memorized facts.
    - The pipeline has five stages: chunk, embed, index, retrieve, and generate. Each stage is a distinct design dimension with its own failure modes.
    - Naive RAG fails due to chunking boundary issues, semantic mismatch, lost-in-the-middle generation degradation, and semantic redundancy in retrieved results.
    - Hybrid retrieval (BM25 + dense, fused by RRF) and cross-encoder reranking are the two single highest-leverage improvements over naive RAG.
    - RAGAS provides three core evaluation metrics — faithfulness, answer relevance, and context precision — that decompose system quality into retrieval and generation components without requiring human labels.
    - Parent-child chunking decouples retrieval precision (small child chunks) from generation context quality (full parent document).
    - Long-context LLMs and RAG are complementary: use long-context for corpora that fit in the window; use RAG for large, dynamic, or multi-tenant knowledge bases.
    - Production RAG requires solving indexing freshness, access control, embedding model versioning, and per-query observability — these are often harder than the core retrieval logic.

!!! sota "State of the Art & Resources (2026)"
    RAG has evolved from a single-pass dense-retrieval pipeline into a rich ecosystem of hybrid search, agentic multi-hop retrieval, and standardized evaluation frameworks — making it one of the most production-deployed LLM architectural patterns as of 2026.

    **Foundational work**

    - [Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020)](https://arxiv.org/abs/2005.11401) — the original RAG paper combining DPR retrieval with BART generation end-to-end; defines the retrieve-then-generate paradigm.
    - [Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020)](https://arxiv.org/abs/2004.04906) — the DPR dual-encoder that underlies most modern dense retrievers; showed dense representations outperform BM25 by 9–19% on top-20 recall.

    **Recent advances (2023–2026)**

    - [Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE, 2022)](https://arxiv.org/abs/2212.10496) — generates a hypothetical answer document as the retrieval query, bridging the vocabulary gap between short queries and long documents.
    - [Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023)](https://arxiv.org/abs/2307.03172) — empirical evidence that LLMs degrade on information in the middle of long contexts; directly motivates careful chunk ordering in RAG prompts.
    - [Es et al., *Ragas: Automated Evaluation of Retrieval Augmented Generation* (2023)](https://arxiv.org/abs/2309.15217) — reference-free faithfulness, answer relevancy, and context precision metrics; the de facto RAG evaluation standard.
    - [Zhang et al., *RAFT: Adapting Language Model to Domain Specific RAG* (2024)](https://arxiv.org/abs/2403.10131) — fine-tuning recipe that teaches models to cite supporting passages and ignore distractor documents in RAG settings.
    - [Gupta et al., *A Comprehensive Survey of RAG: Evolution, Current Landscape and Future Directions* (2024)](https://arxiv.org/abs/2410.12837) — broad survey covering modular RAG, GraphRAG, agentic retrieval, and evaluation benchmarks up to late 2024.
    - [Singh et al., *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG* (2025)](https://arxiv.org/abs/2501.09136) — surveys how autonomous agents dynamically decide when and what to retrieve, enabling multi-hop and self-correcting pipelines.

    **Open-source & tools**

    - [explodinggradients/ragas](https://github.com/explodinggradients/ragas) — the ragas Python library implementing faithfulness, answer relevancy, and context precision; integrates with LangChain and LlamaIndex.
    - [langchain-ai/langchain](https://github.com/langchain-ai/langchain) — the dominant RAG orchestration framework (138 k GitHub stars); provides document loaders, text splitters, retrievers, and LLM chains with 300+ integrations.
    - [run-llama/llama_index](https://github.com/run-llama/llama_index) — LlamaIndex, specialized for advanced indexing patterns (parent-child, summary indexes, knowledge graphs) and agentic retrieval workflows.

## Further Reading

- Lewis et al., **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**, NeurIPS 2020 — the original RAG paper combining DPR retrieval with BART generation end-to-end.
- Karpukhin et al., **"Dense Passage Retrieval for Open-Domain Question Answering"** (DPR), EMNLP 2020 — the bi-encoder retrieval model that underlies most dense RAG systems.
- Gao et al., **"Precise Zero-Shot Dense Retrieval without Relevance Labels"** (HyDE), ACL 2023 — hypothetical document embedding for improved query-document alignment.
- Es et al., **"RAGAS: Automated Evaluation of Retrieval Augmented Generation"**, 2023 — defines the faithfulness, answer relevance, and context precision metrics implemented by the `ragas` library.
- Liu et al., **"Lost in the Middle: How Language Models Use Long Contexts"**, TACL 2024 — empirical study of positional bias in long-context generation, directly relevant to multi-chunk RAG.
- Shi et al., **"REPLUG: Retrieval-Augmented Black-Box Language Models"**, NAACL 2023 — treats the LLM as a black box and trains only the retriever via LM likelihood signals.
- Zhang et al., **"RAFT: Adapting Language Model to Domain Specific RAG"**, 2024 — fine-tuning recipe for making models better at extracting answers from retrieved context while ignoring distractor documents.
- LangChain and LlamaIndex open-source repos — the two most widely used RAG orchestration frameworks, with extensive examples of advanced retrieval patterns.
