"""
Test for content/09-rag-retrieval/03-rag-architectures.md

Heuristic scan found 4 fenced blocks (#0-#3):
  - #0 (line ~76):  ```text``` block, non-python -> SKIP(non-python)
  - #1 (line ~94):  "Minimal RAG from scratch" -- uses sentence-transformers,
                     faiss-cpu, AND makes a real `openai` client call for
                     generation -> SKIP(network): real LLM API call, no network in CI.
  - #2 (line ~364): "Minimal RAGAS-style evaluation loop" -- calls ragas.evaluate(),
                     which invokes a real LLM judge (default GPT-4) over the network
                     -> SKIP(network): real LLM-judge API call, no network/keys in CI.
  - #3 (line ~467): "Production-style RAG pipeline" (BM25 + dense hybrid retrieval
                     w/ RRF fusion, parent-child chunk hierarchy, cross-encoder
                     rerank, NLI faithfulness check) -> TESTED below.

Block #3 imports rank_bm25, sentence_transformers, transformers, and faiss --
none of which are in the CI allowlist (numpy, torch, einops, sklearn, stdlib).
Per the hard rules, instantiating SentenceTransformer/CrossEncoder/transformers
pipeline() is treated as "network" even with a local cache, because CI has
neither the packages nor the weights. So each import is guarded, and when the
real package is unavailable we substitute a tiny, deterministic, offline,
shape-correct stub that preserves the exact interface the book's code calls
(BM25Okapi.get_scores, SentenceTransformer.encode, CrossEncoder.predict,
faiss.IndexFlatIP.add/search, transformers.pipeline "text-classification").
The book's own classes/functions (Chunk, Document, build_parent_child_index,
HybridRetriever, Reranker, FaithfulnessChecker, run_pipeline) are copied
verbatim and exercised end to end using these stubs.
"""

import zlib
from typing import Dict, List, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:
    SentenceTransformer = None
    CrossEncoder = None

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

try:
    import faiss
except Exception:
    faiss = None


# ─────────────────────────────────────────────────────────────
# Offline, deterministic stand-ins used only when the real
# packages are unavailable (as in CI). Each preserves the exact
# method signatures the book's code below calls.
# ─────────────────────────────────────────────────────────────

if BM25Okapi is None:

    class BM25Okapi:
        """Deterministic BM25 stand-in: score = token-overlap count."""

        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query_tokens):
            qset = set(query_tokens)
            return np.array(
                [float(sum(1 for t in doc if t in qset)) for doc in self.corpus],
                dtype=np.float32,
            )


if SentenceTransformer is None:

    class SentenceTransformer:
        """Deterministic fake embedder: stable-hash text into a fixed-dim vector."""

        _DIM = 16

        def __init__(self, model_name):
            self.model_name = model_name

        def _embed_one(self, text):
            seed = zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            return rng.randn(self._DIM).astype(np.float32)

        def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
            single = isinstance(texts, str)
            if single:
                texts = [texts]
            vecs = [self._embed_one(t) for t in texts]
            arr = np.stack(vecs).astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
            return arr[0] if single else arr


if CrossEncoder is None:

    class CrossEncoder:
        """Deterministic fake cross-encoder: score = normalized token overlap."""

        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            scores = []
            for q, d in pairs:
                qset = set(q.lower().split())
                dset = set(d.lower().split())
                inter = len(qset & dset)
                scores.append(float(inter) / (len(qset) + 1e-6))
            return np.array(scores, dtype=np.float32)


if faiss is None:

    class _FlatIPIndex:
        def __init__(self, d):
            self.d = d
            self._vecs = None

        def add(self, vecs):
            self._vecs = vecs.astype(np.float32)

        def search(self, queries, k):
            sims = queries @ self._vecs.T
            k = min(k, sims.shape[1])
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
            return scores, idx

    class _FaissStub:
        IndexFlatIP = _FlatIPIndex

    faiss = _FaissStub()


if hf_pipeline is None:

    def hf_pipeline(task, model=None):
        """Deterministic fake text-classification (NLI) pipeline."""

        def _call(text, truncation=True, max_length=512, top_k=None):
            premise, _, hypothesis = text.partition(" [SEP] ")
            pset = set(premise.lower().split())
            hset = set(hypothesis.lower().split())
            overlap = len(pset & hset) / (len(hset) + 1e-6)
            entail = float(min(1.0, overlap))
            rest = max(0.0, 1.0 - entail) / 2.0
            return [
                {"label": "entailment", "score": entail},
                {"label": "neutral", "score": rest},
                {"label": "contradiction", "score": rest},
            ]

        return _call


# ============================================================
# BEGIN block #3 verbatim (content/09-rag-retrieval/03-rag-architectures.md, line ~467)
# ============================================================

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

# ============================================================
# END block #3
# ============================================================


def main():
    np.random.seed(0)

    documents = [
        (
            "Python is a dynamically typed interpreted programming language "
            "widely used for data science machine learning and web development. "
            "Python programming emphasizes readability and a clean simple syntax "
            "that programmers find easy to learn. The Python interpreter runs "
            "bytecode compiled from source code at runtime."
        ),
        (
            "Golden retrievers are friendly loyal dogs that make excellent family "
            "pets. Golden retriever puppies need regular exercise and training. "
            "These dogs are known for their gentle temperament around children "
            "and other animals, and they love to play fetch in the yard."
        ),
    ]

    # build_parent_child_index
    parent_docs, child_chunks = build_parent_child_index(
        documents, child_size=20, overlap=4
    )
    assert len(parent_docs) == 2
    assert len(child_chunks) > 2
    assert all(isinstance(c, Chunk) for c in child_chunks)

    # HybridRetriever (BM25 + dense w/ RRF fusion)
    retriever = HybridRetriever(child_chunks, embed_model="fake-embed-model")
    results = retriever.retrieve("Python programming language", top_n=5)
    assert len(results) > 0
    assert all(isinstance(c, Chunk) and isinstance(s, float) for c, s in results)
    # RRF scores should be sorted descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # Lexically-overlapping (Python) chunks should be fused/ranked ahead of
    # the completely unrelated (dogs) chunks.
    assert "python" in results[0][0].text.lower()

    # Reranker (cross-encoder)
    reranker = Reranker(model="fake-cross-encoder")
    top = reranker.rerank("Python programming language", results, top_k=2)
    assert 0 < len(top) <= 2

    # FaithfulnessChecker (NLI)
    checker = FaithfulnessChecker()
    grounded = checker.score(
        premise="Python is a dynamically typed interpreted programming language.",
        hypothesis="Python is a dynamically typed programming language.",
    )
    unrelated = checker.score(
        premise="Python is a dynamically typed interpreted programming language.",
        hypothesis="Golden retrievers love to play fetch in the yard.",
    )
    assert 0.0 <= grounded <= 1.0 and 0.0 <= unrelated <= 1.0
    assert grounded > unrelated, (
        "a paraphrase of the premise should score higher entailment than an unrelated sentence"
    )

    def fake_generate(prompt: str) -> str:
        assert "Context:" in prompt and "Question:" in prompt
        return "Python is a dynamically typed interpreted programming language."

    result = run_pipeline(
        query="What is Python?",
        retriever=retriever,
        reranker=reranker,
        parent_docs=parent_docs,
        faithfulness_checker=checker,
        generate_fn=fake_generate,
    )

    assert set(result.keys()) == {"answer", "context", "top_chunks", "faithful"}
    assert result["answer"] == "Python is a dynamically typed interpreted programming language."
    assert "python" in result["context"].lower()
    assert len(result["top_chunks"]) > 0
    assert isinstance(result["faithful"], (bool, np.bool_))
    # The generated answer is lifted verbatim from the Python context, so the
    # deterministic overlap-based NLI stub should judge it faithful.
    assert bool(result["faithful"]) is True

    print("block #3 (production RAG pipeline) OK:")
    print("  chunks:", len(child_chunks), " retrieved:", len(results), " reranked:", len(top))
    print("  faithful:", result["faithful"], " answer:", result["answer"])


if __name__ == "__main__":
    main()
