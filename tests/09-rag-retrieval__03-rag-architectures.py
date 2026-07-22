"""CI-tested extracts of runnable code blocks from
content/09-rag-retrieval/03-rag-architectures.md

Block inventory (per the chapter):
  #0 - non-python (SKIP)
  #1 - needs-gpu (SKIP)
  #2 - needs-gpu (SKIP)
  #3 - "Production-style RAG pipeline" (line ~467): hybrid BM25+dense
       retrieval, parent-child chunking, cross-encoder reranking, and an
       NLI-based faithfulness checker. TESTED below.

Block #3 imports three packages that are NOT in the guaranteed-CI set
(rank_bm25, sentence_transformers, faiss) plus `transformers`, whose
`pipeline(...)` constructor would otherwise download a model checkpoint over
the network. Per the harness rules we MOCK those boundaries with tiny,
deterministic, offline stand-ins injected into `sys.modules` / passed in as
fakes, so the book's OWN logic (RRF fusion, parent-child chunk lookup,
reranking, faithfulness scoring, full pipeline assembly) actually executes,
end to end, on a toy corpus -- nothing here swallows the logic under test.

BUG FOUND & FIXED (mirrored in the .md source):
  `FaithfulnessChecker.score()` called
      self.nli(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)
  with no `top_k` kwarg. HuggingFace's TextClassificationPipeline has a
  "legacy" mode: when `top_k` is absent from kwargs entirely, it returns only
  the single highest-scoring label wrapped in a list, not the full label
  distribution. So `label_score.get("entailment", 0.0)` silently returned 0.0
  every time "entailment" wasn't the single top-1 predicted label (i.e. most
  of the time), defeating the purpose of a soft 0..1 entailment probability
  threshold. Fixed by passing `top_k=None`, which asks the pipeline for
  scores across *all* labels. This is demonstrated directly below in
  `block_faithfulness_topk_bug_demo`.
"""
import sys
import types
from typing import List, Dict, Tuple

import numpy as np

# SKIP(non-python): block #0 is prose/pseudocode, not executable Python.
# SKIP(needs-gpu): block #1 uses a GPU-resident cross-encoder / large model.
# SKIP(needs-gpu): block #2 uses a GPU-resident cross-encoder / large model.


# ─────────────────────────────────────────────────────────────────────────
# Offline, deterministic stand-ins for optional third-party packages that
# are not guaranteed in CI (rank_bm25, sentence_transformers, faiss) and for
# the network-touching parts of `transformers`. These are installed into
# sys.modules *before* the book's own `import` statements run, so the book
# code below is copied verbatim and really executes against them.
# ─────────────────────────────────────────────────────────────────────────
def _install_fake_rank_bm25():
    mod = types.ModuleType("rank_bm25")

    class BM25Okapi:
        """Tiny term-overlap scorer standing in for the real BM25Okapi."""

        def __init__(self, tokenized_corpus):
            self.corpus = tokenized_corpus

        def get_scores(self, query_tokens):
            qset = set(query_tokens)
            return np.array(
                [float(sum(1 for w in doc if w in qset)) for doc in self.corpus]
            )

    mod.BM25Okapi = BM25Okapi
    sys.modules["rank_bm25"] = mod


def _install_fake_sentence_transformers():
    mod = types.ModuleType("sentence_transformers")
    DIM = 64

    def _hash_embed(text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in text.lower().split():
            v[hash(w) % DIM] += 1.0
        return v

    class SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
            vecs = np.stack([_hash_embed(t) for t in texts]).astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms
            return vecs

    class CrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            scores = []
            for q, d in pairs:
                qset, dset = set(q.lower().split()), set(d.lower().split())
                scores.append(float(len(qset & dset)))
            return np.array(scores)

    mod.SentenceTransformer = SentenceTransformer
    mod.CrossEncoder = CrossEncoder
    sys.modules["sentence_transformers"] = mod


def _install_fake_faiss():
    mod = types.ModuleType("faiss")

    class IndexFlatIP:
        def __init__(self, d):
            self.d = d
            self.vectors = None

        def add(self, embeddings):
            self.vectors = embeddings

        def search(self, query_vecs, top_n):
            sims = query_vecs @ self.vectors.T
            idx = np.argsort(-sims, axis=1)[:, :top_n]
            scores = np.take_along_axis(sims, idx, axis=1)
            return scores, idx

    mod.IndexFlatIP = IndexFlatIP
    sys.modules["faiss"] = mod


def _fake_hf_text_classification_pipeline(task, model=None, **_):
    """Stands in for transformers.pipeline("text-classification", model=...).

    Faithfully replicates the real pipeline's `top_k`-presence "legacy" quirk
    (see module docstring) so the bug we found/fixed is exercised for real:
    if `top_k` is absent from kwargs -> only the single top label is
    returned; if `top_k=None` is passed -> the full label distribution is
    returned.
    """

    def _classify(text, **kwargs):
        premise, _, hypothesis = text.partition(" [SEP] ")
        pset, hset = set(premise.lower().split()), set(hypothesis.lower().split())
        overlap = len(pset & hset) / max(len(hset), 1)
        entail = min(0.95, 0.2 + 0.8 * overlap)
        neutral = max(0.0, (1 - entail) * 0.6)
        contra = max(0.0, 1 - entail - neutral)
        scores = [
            {"label": "entailment", "score": entail},
            {"label": "neutral", "score": neutral},
            {"label": "contradiction", "score": contra},
        ]
        scores.sort(key=lambda x: -x["score"])

        legacy = "top_k" not in kwargs
        if legacy:
            return [scores[0]]  # HF's real "odd" single-item legacy behavior
        top_k = kwargs["top_k"]
        return scores if top_k is None else scores[:top_k]

    return _classify


_install_fake_rank_bm25()
_install_fake_sentence_transformers()
_install_fake_faiss()


# ─────────────────────────────────────────────────────────────────────────
# Block #3 (content/09-rag-retrieval/03-rag-architectures.md, line ~467)
# Copied verbatim (imports guarded per harness rules; the `top_k=None` fix
# below mirrors the fix applied to the .md source -- see module docstring).
# ─────────────────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss

    HAVE_RETRIEVAL_DEPS = True
except Exception:
    HAVE_RETRIEVAL_DEPS = False

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# Always route to the offline fake regardless of whether real `transformers`
# is installed -- the real `pipeline(...)` constructor would try to download
# a checkpoint from the network, which is forbidden in this test.
hf_pipeline = _fake_hf_text_classification_pipeline


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
class Chunk:
    def __init__(self, text: str, chunk_id: int, parent_id: int):
        self.text = text
        self.chunk_id = chunk_id
        self.parent_id = parent_id  # index into parent_docs list


class Document:
    def __init__(self, full_text: str, doc_id: int):
        self.full_text = full_text
        self.doc_id = doc_id


# ─────────────────────────────────────────────
# Build parent-child index
# ─────────────────────────────────────────────
def build_parent_child_index(
    documents: List[str],
    child_size: int = 128,  # words per child chunk
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


# ─────────────────────────────────────────────────────────────────────────
# Test harness: exercise every piece of block #3 end to end on a tiny,
# offline, three-document toy corpus.
# ─────────────────────────────────────────────────────────────────────────
def block_faithfulness_topk_bug_demo():
    """Demonstrates the bug we found and fixed in FaithfulnessChecker.score.

    Calling the pipeline WITHOUT `top_k` (the book's original code) only
    ever returns the single top-1 predicted label -- so `entailment` is
    silently reported as 0.0 whenever it isn't the top label, even when its
    true probability is high. Passing `top_k=None` (the fix) returns the
    full label distribution.
    """
    nli = hf_pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
    text = "the cat sat on the mat [SEP] a cat is on a mat"

    legacy_result = nli(text, truncation=True, max_length=512)  # buggy call (no top_k)
    fixed_result = nli(text, truncation=True, max_length=512, top_k=None)  # fixed call

    assert len(legacy_result) == 1, "legacy (buggy) call should return only the top-1 label"
    assert len(fixed_result) == 3, "fixed call should return all 3 NLI labels"

    legacy_labels = {r["label"] for r in legacy_result}
    fixed_labels = {r["label"] for r in fixed_result}
    assert "entailment" in fixed_labels
    # The buggy legacy path only *sometimes* has "entailment" in it, which is
    # exactly the silent-failure mode the fix addresses.
    print(f"legacy top-1 label only: {legacy_labels}")
    print(f"fixed full distribution: {fixed_labels}")


def block_rag_pipeline():
    # content lines ~467-669 (build_parent_child_index, HybridRetriever,
    # Reranker, FaithfulnessChecker, run_pipeline) exercised together.
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
        (
            "Stock market investing involves risk and diversification across a "
            "portfolio of assets. Investors buy stocks bonds and index funds to "
            "grow wealth over time. A diversified portfolio reduces investment "
            "risk by spreading capital across many different asset classes."
        ),
    ]

    # --- build_parent_child_index ---
    parent_docs, child_chunks = build_parent_child_index(
        documents, child_size=20, overlap=5
    )
    assert len(parent_docs) == 3
    assert len(child_chunks) > 3, "small child_size should split each doc into multiple chunks"
    assert all(isinstance(c, Chunk) for c in child_chunks)
    assert all(0 <= c.parent_id < 3 for c in child_chunks)

    # --- HybridRetriever (BM25 + dense FAISS, RRF-fused) ---
    retriever = HybridRetriever(child_chunks, embed_model="fake-embed-model")
    candidates = retriever.retrieve("Tell me about the Python programming language", top_n=6)
    assert 1 <= len(candidates) <= 6
    assert all(isinstance(c, Chunk) and isinstance(s, float) for c, s in candidates)
    # The lexically-overlapping (Python) chunks should be fused/ranked ahead
    # of the completely unrelated (dogs / stock market) chunks.
    top_chunk_text = candidates[0][0].text.lower()
    assert "python" in top_chunk_text

    # --- Reranker (cross-encoder) ---
    reranker = Reranker(model="fake-cross-encoder")
    reranked = reranker.rerank("Tell me about the Python programming language", candidates, top_k=3)
    assert len(reranked) == min(3, len(candidates))
    assert all(isinstance(c, Chunk) for c, _ in reranked)

    # --- FaithfulnessChecker (NLI-based, offline fake pipeline) ---
    checker = FaithfulnessChecker()
    grounded_score = checker.score(
        premise="Python is a dynamically typed interpreted programming language.",
        hypothesis="Python is a dynamically typed programming language.",
    )
    unrelated_score = checker.score(
        premise="Python is a dynamically typed interpreted programming language.",
        hypothesis="Golden retrievers love to play fetch in the yard.",
    )
    assert 0.0 <= grounded_score <= 1.0
    assert 0.0 <= unrelated_score <= 1.0
    assert grounded_score > unrelated_score, (
        "a paraphrase of the premise should score higher entailment than an "
        "unrelated sentence"
    )
    assert checker.check_answer(
        answer="Python is a dynamically typed programming language.",
        context="Python is a dynamically typed interpreted programming language.",
    ) is True
    assert checker.check_answer(
        answer="Golden retrievers love to play fetch in the yard.",
        context="Python is a dynamically typed interpreted programming language.",
    ) is False

    # --- Full pipeline: run_pipeline() ---
    def toy_generate_fn(prompt: str) -> str:
        # Deterministic, offline "generation": echo back the first sentence
        # of the retrieved context so the answer is genuinely grounded in it
        # (no real LLM call -- this stands in for `generate_fn`).
        ctx_start = prompt.find("Context:\n") + len("Context:\n")
        ctx_end = prompt.find("\n\nAnswer the question")
        context = prompt[ctx_start:ctx_end]
        first_sentence = context.split(".")[0].strip() + "."
        return first_sentence

    result = run_pipeline(
        query="Tell me about the Python programming language",
        retriever=retriever,
        reranker=reranker,
        parent_docs=parent_docs,
        faithfulness_checker=checker,
        generate_fn=toy_generate_fn,
    )
    assert set(result.keys()) == {"answer", "context", "top_chunks", "faithful"}
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    assert "python" in result["context"].lower()
    assert len(result["top_chunks"]) >= 1
    assert isinstance(result["faithful"], bool)
    # The answer is a direct echo of the top-ranked (Python) parent doc's
    # opening sentence, so it should be judged faithful.
    assert result["faithful"] is True, "an answer echoed verbatim from context should be faithful"

    print(f"retrieved top chunk: {candidates[0][0].text[:60]!r}...")
    print(f"pipeline answer: {result['answer']!r}")
    print(f"pipeline faithful: {result['faithful']}")


BLOCKS = [
    block_faithfulness_topk_bug_demo,
    block_rag_pipeline,
]


def main():
    for fn in BLOCKS:
        print(f"\n===== {fn.__name__} =====")
        fn()
    print(f"\nAll {len(BLOCKS)} code blocks executed and verified.")
    print(
        "\nSKIPPED (not executed): block #0 (non-python prose), "
        "#1 and #2 (need-gpu cross-encoder/large-model demos)."
    )


if __name__ == "__main__":
    main()
