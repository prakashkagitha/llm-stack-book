"""
Runs the CPU-runnable Python code blocks from:
    content/09-rag-retrieval/05-advanced-rag.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:  #1 (multihop_rag), #2 (corrective_rag), #3 (contextual
                 retrieval chunk builder), #4 (agentic_rag_loop), #5
                 (text_to_sql_rag), #6 (AST-aware code chunker), #7
                 (HippoRAG-style Personalized PageRank retrieval)

Skipped blocks: #0 (minimal_graphrag.py) -- SKIP(network): imports a real
                `openai.OpenAI()` client and `sentence_transformers.
                SentenceTransformer`, both of which perform network calls
                (API calls / model-weight download) merely to instantiate at
                module scope. This is exactly the "Requires: openai,
                networkx, python-louvain, sentence-transformers" block the
                chapter itself flags as needing external services, so it is
                left untested here rather than faked into meaninglessness.

All LLM/vector-DB/web-search dependencies used by the tested blocks are
injected as plain Python callables (`retrieve`, `llm`, `llm_chat`,
`web_search`, `embedder`) per the chapter's own "Both are injected for
testability" convention -- no network calls anywhere in this file.
"""

from __future__ import annotations

import ast
import re
import sqlite3
import textwrap
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Set, Tuple

import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


# ============================================================================
# Block #1 (line ~239) -- multihop_rag.py
# Iterative Chain-of-Thought retrieval with explicit decomposition.
# ============================================================================


def multihop_rag(
    original_question: str,
    retrieve,      # callable(query: str, k: int) -> List[str]
    llm,           # callable(prompt: str) -> str
    max_hops: int = 4,
    chunks_per_hop: int = 3,
) -> str:
    """
    Iterative retrieval: retrieve -> reason -> decide whether to retrieve again.
    Returns the final answer string.
    """
    accumulated_context: List[str] = []
    trajectory: List[Tuple[str, List[str]]] = []  # (sub-query, retrieved_chunks)

    DECOMPOSE_PROMPT = textwrap.dedent("""
        You are answering the question step by step.

        Original question: {question}

        Information gathered so far:
        {context}

        What is the SINGLE most important piece of information you still need?
        Write it as a short search query (one sentence).
        If you have enough information to answer, write "ANSWER:" followed by your final answer.

        Your response:
    """)

    current_query = original_question

    for hop in range(max_hops):
        # Retrieve chunks for this iteration's sub-query
        chunks = retrieve(current_query, chunks_per_hop)
        accumulated_context.extend(chunks)
        trajectory.append((current_query, chunks))

        # Ask the LLM whether we have enough information or need another hop
        context_str = "\n---\n".join(accumulated_context)
        prompt = DECOMPOSE_PROMPT.format(
            question=original_question,
            context=context_str[:6000],  # guard against context overflow
        )
        response = llm(prompt).strip()

        if response.startswith("ANSWER:"):
            # The model has enough to answer -- we're done
            return response[len("ANSWER:"):].strip()

        # Otherwise, the model's response IS the next sub-query
        current_query = response

    # Fallback: force an answer with whatever we have
    final_prompt = (
        f"Based on the following information, answer: {original_question}\n\n"
        + "\n---\n".join(accumulated_context[:5000])
    )
    return llm(final_prompt)


# ============================================================================
# Block #2 (line ~342) -- corrective_rag.py
# CRAG-style retrieval with relevance scoring and fallback.
# ============================================================================


class RetrievedDoc(NamedTuple):
    text: str
    score: float   # initial retrieval similarity


def evaluate_relevance(query: str, doc: str, llm) -> float:
    """
    Ask the LLM to score relevance 0.0-1.0.
    In production, use a cross-encoder reranker (faster, no API call).
    """
    prompt = (
        f"On a scale of 0.0 to 1.0, how relevant is the following document "
        f"to the query: '{query}'?\nDocument: {doc[:500]}\n"
        f"Return only a float like 0.85."
    )
    raw = llm(prompt).strip()
    try:
        return float(raw)
    except ValueError:
        return 0.5


def corrective_rag(
    query: str,
    initial_docs: List[RetrievedDoc],
    web_search,      # callable(query: str) -> List[str]
    llm,
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
) -> List[str]:
    """
    CRAG algorithm:
      - relevance >= high_threshold -> accept as-is
      - relevance <= low_threshold  -> discard, trigger web search
      - in between                  -> keep but also do web search
    Returns a final list of text passages for the generator.
    """
    final_passages: List[str] = []
    need_web = False

    for doc in initial_docs:
        rel = evaluate_relevance(query, doc.text, llm)

        if rel >= high_threshold:
            final_passages.append(doc.text)
        elif rel <= low_threshold:
            need_web = True  # discard this doc
        else:
            # Ambiguous: decompose into fine-grained sentences and keep good ones
            sentences = [s.strip() for s in doc.text.split(".") if s.strip()]
            for sent in sentences:
                sent_rel = evaluate_relevance(query, sent, llm)
                if sent_rel >= high_threshold:
                    final_passages.append(sent)
            need_web = True

    if need_web or not final_passages:
        web_results = web_search(query)
        final_passages.extend(web_results[:3])

    return final_passages


# ============================================================================
# Block #3 (line ~425) -- contextual_retrieval.py
# Prepend document-level context to each chunk before embedding.
# ============================================================================


CONTEXT_PROMPT = """\
Here is the chunk we want to situate within the full document.
<document>
{document}
</document>
<chunk>
{chunk}
</chunk>
Please give a short succinct context (1-2 sentences) to situate
this chunk within the overall document for improved search retrieval.
Answer only with the succinct context, nothing else.
"""


def build_contextual_chunks(
    document: str,
    chunks: List[str],
    llm,
    max_doc_chars: int = 8000,
) -> List[str]:
    """
    For each chunk, generate a context prefix using the full document,
    then prepend it. The result is stored in the embedding index instead
    of the raw chunk.
    """
    # Truncate the doc to fit in context (use a sliding reference window in prod)
    doc_excerpt = document[:max_doc_chars]
    contextual_chunks = []

    for chunk in chunks:
        prompt = CONTEXT_PROMPT.format(document=doc_excerpt, chunk=chunk)
        context_sentence = llm(prompt).strip()
        # The embedded text includes context, but the stored text can be the raw chunk
        embedded_text = f"{context_sentence}\n\n{chunk}"
        contextual_chunks.append(embedded_text)

    return contextual_chunks


# ============================================================================
# Block #4 (line ~482) -- agentic_rag.py
# RAG as an agent tool, using ReAct-style prompting.
# ============================================================================


SYSTEM_PROMPT = """\
You are a research assistant with access to a document retrieval tool.
Use the tool as many times as needed before giving a final answer.

Available tool:
  retrieve(query: str, k: int) -> list of text passages

Format your reasoning as:
  Thought: <what you're thinking>
  Action: retrieve(query="...", k=3)
  Observation: <tool result>
  ... (repeat as needed)
  Final Answer: <your answer>
"""


def parse_action(text: str) -> Dict[str, Any] | None:
    """Extract a retrieve() call from the LLM's output, if present."""
    match = re.search(
        r'Action:\s*retrieve\(query=["\'](.+?)["\'],\s*k=(\d+)\)',
        text,
        re.DOTALL,
    )
    if match:
        return {"query": match.group(1), "k": int(match.group(2))}
    return None


def agentic_rag_loop(
    question: str,
    retrieve,       # callable(query, k) -> List[str]
    llm_chat,       # callable(messages: List[Dict]) -> str  (chat format)
    max_steps: int = 8,
) -> str:
    """
    Run the ReAct-style agentic retrieval loop until the LLM produces
    a 'Final Answer:' or we exhaust max_steps.
    """
    messages: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]

    for step in range(max_steps):
        response = llm_chat(messages)
        messages.append({"role": "assistant", "content": response})

        # Check if the model has finished
        if "Final Answer:" in response:
            idx = response.index("Final Answer:")
            return response[idx + len("Final Answer:"):].strip()

        # Try to parse and execute a tool call
        action = parse_action(response)
        if action:
            passages = retrieve(action["query"], action["k"])
            observation = "\n---\n".join(passages) if passages else "No relevant documents found."
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })
        else:
            # Model didn't format a tool call -- nudge it
            messages.append({
                "role": "user",
                "content": "Please continue your reasoning or provide a Final Answer.",
            })

    # Exhausted steps -- force a conclusion
    messages.append({
        "role": "user",
        "content": "Please provide your Final Answer now based on what you have gathered.",
    })
    return llm_chat(messages)


# ============================================================================
# Block #5 (line ~579) -- text_to_sql_rag.py
# Minimal Text-to-SQL RAG with schema grounding.
# ============================================================================


TEXT_TO_SQL_PROMPT = """\
Given the following SQLite schema, write a SQL query to answer the question.
Return ONLY the SQL, no explanation.

Schema:
{schema}

Question: {question}

SQL:
"""

SCHEMA_REFLECT_PROMPT = """\
The query failed with: {error}
Original question: {question}
Schema: {schema}
Previous SQL attempt: {sql}

Write a corrected SQL query. Return ONLY the SQL.
"""


def get_schema(conn: sqlite3.Connection) -> str:
    """Extract CREATE TABLE statements from an SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    return "\n\n".join(row[0] for row in cursor.fetchall() if row[0])


def text_to_sql_rag(
    question: str,
    db_path: str,
    llm,
    max_retries: int = 3,
) -> Tuple[str, str]:
    """
    Convert question -> SQL -> execute -> generate final answer.
    Implements self-correction on SQL errors.
    Returns (answer, sql_used).
    """
    conn = sqlite3.connect(db_path)
    schema = get_schema(conn)

    sql = llm(TEXT_TO_SQL_PROMPT.format(schema=schema, question=question)).strip()
    # Strip markdown fences if the model wrapped it
    sql = sql.strip("`").lstrip("sql").strip()

    for attempt in range(max_retries):
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description or []]

            # Format results as a readable table
            if rows:
                header = " | ".join(col_names)
                body = "\n".join(" | ".join(str(v) for v in row) for row in rows[:50])
                result_text = f"{header}\n{body}"
            else:
                result_text = "(no rows returned)"

            # Generate a natural language answer from the SQL result
            answer_prompt = (
                f"The question was: {question}\n"
                f"SQL executed: {sql}\n"
                f"Results:\n{result_text}\n\n"
                f"Write a clear, concise answer."
            )
            return llm(answer_prompt), sql

        except Exception as e:
            if attempt < max_retries - 1:
                sql = llm(SCHEMA_REFLECT_PROMPT.format(
                    error=str(e), question=question, schema=schema, sql=sql
                )).strip().strip("`").lstrip("sql").strip()
            else:
                raise RuntimeError(f"SQL generation failed after {max_retries} attempts") from e

    conn.close()


# ============================================================================
# Block #6 (line ~674) -- code_rag_chunker.py
# AST-aware chunking for Python code repositories.
# ============================================================================


def extract_python_chunks(source: str, filepath: str) -> List[Dict]:
    """
    Parse Python source with AST and return one chunk per function/class.
    Each chunk includes: the full source text, a context string with
    the module docstring and all parent class names.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to whole-file chunking if parsing fails
        return [{"text": source, "context": filepath, "type": "file"}]

    module_doc = ast.get_docstring(tree) or ""
    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno  # Python 3.8+
            code_lines = source.splitlines()[start:end]
            code_text = "\n".join(code_lines)

            # Build a context string: module doc + qualified name
            qualname = node.name
            context = (
                f"File: {filepath}\n"
                f"Module context: {module_doc[:200]}\n"
                f"Definition: {qualname}"
            )
            chunks.append({
                "text": code_text,
                "context": context,
                "embedded_text": f"{context}\n\n{code_text}",  # what gets embedded
                "type": type(node).__name__,
                "lineno": node.lineno,
            })

    return chunks


def index_repository(repo_path: str, embedder) -> List[Dict]:
    """Walk a Python repository and produce embeddable chunks."""
    all_chunks = []
    for py_file in Path(repo_path).rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            chunks = extract_python_chunks(source, str(py_file))
            all_chunks.extend(chunks)
        except Exception:
            continue

    # Embed the chunks
    texts = [c["embedded_text"] for c in all_chunks]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return all_chunks


# ============================================================================
# Block #7 (line ~821) -- hippoppr.py
# Personalized PageRank over an entity graph for HippoRAG-style retrieval.
# ============================================================================


def personalized_pagerank(
    G: "nx.DiGraph",
    seed_nodes: Set[str],
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Power-iteration PPR.
    alpha: teleport probability back to seed nodes.
    Returns a dict of node -> PPR score.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # Personalisation vector: uniform over seed nodes
    p = np.zeros(n)
    for s in seed_nodes:
        if s in idx:
            p[idx[s]] = 1.0
    if p.sum() == 0:
        p = np.ones(n) / n  # fallback: uniform
    else:
        p /= p.sum()

    # Stochastic transition matrix (column-normalised)
    A = nx.to_numpy_array(G, nodelist=nodes)
    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1  # dangling nodes
    A = A / col_sums[np.newaxis, :]

    # Power iteration: r = alpha * p + (1 - alpha) * A^T r
    r = p.copy()
    for _ in range(max_iter):
        r_new = alpha * p + (1 - alpha) * A.T @ r
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    return {node: float(r[idx[node]]) for node in nodes}


def hippoppr_retrieve(
    query: str,
    G: "nx.DiGraph",
    entity_to_chunks: Dict[str, List[str]],
    entity_embeddings: Dict[str, np.ndarray],
    query_embedding: np.ndarray,
    top_entities: int = 5,
    top_chunks: int = 10,
    alpha: float = 0.15,
) -> List[str]:
    """
    Full HippoRAG-style retrieval:
      1. Find seed entities by embedding similarity to query.
      2. Run PPR to propagate relevance.
      3. Collect chunks associated with high-PPR entities.
    """
    # Step 1: identify seed entities
    sims = {
        eid: float(np.dot(query_embedding, emb) /
                   (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8))
        for eid, emb in entity_embeddings.items()
    }
    seed_nodes = set(
        sorted(sims, key=sims.get, reverse=True)[:top_entities]
    )

    # Step 2: PPR
    ppr_scores = personalized_pagerank(G, seed_nodes, alpha=alpha)

    # Step 3: rank entities by PPR, collect their source chunks
    ranked_entities = sorted(ppr_scores, key=ppr_scores.get, reverse=True)
    seen_chunks: Set[str] = set()
    result_chunks: List[str] = []

    for entity in ranked_entities:
        for chunk in entity_to_chunks.get(entity, []):
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                result_chunks.append(chunk)
                if len(result_chunks) >= top_chunks:
                    return result_chunks

    return result_chunks


# ============================================================================
# Test harness: exercise every block above with tiny CPU-only fixtures.
# ============================================================================


def test_multihop_rag():
    """Block #1: drive a 3-hop trajectory ending in an explicit ANSWER:."""
    corpus = {
        "author of PyTorch": ["PyTorch was created by Soumith Chintala at Facebook AI Research."],
        "company founded by Soumith Chintala": ["Soumith Chintala co-founded Extropic AI in 2023."],
        "programming language used at Extropic AI": ["Extropic AI's hardware control software is written primarily in Rust."],
    }

    def fake_retrieve(query: str, k: int) -> List[str]:
        for key, chunks in corpus.items():
            if key in query:
                return chunks[:k]
        return ["No matching chunk found."]

    responses = iter([
        "company founded by Soumith Chintala",
        "programming language used at Extropic AI",
        "ANSWER: Rust",
    ])

    def fake_llm(prompt: str) -> str:
        return next(responses)

    answer = multihop_rag(
        "What programming language does the company founded by the author of PyTorch use?",
        retrieve=fake_retrieve,
        llm=fake_llm,
        max_hops=4,
        chunks_per_hop=3,
    )
    assert answer == "Rust", f"expected Rust, got {answer!r}"

    # Also exercise the fallback path (never emits ANSWER: -> forced final call).
    def never_done_llm(prompt: str) -> str:
        return "some other sub-query"

    forced = multihop_rag(
        "unanswerable question",
        retrieve=lambda q, k: ["irrelevant chunk"],
        llm=never_done_llm,
        max_hops=2,
        chunks_per_hop=2,
    )
    # never_done_llm never returns "ANSWER:" so multihop_rag falls through
    # to the final forced-answer call which also invokes never_done_llm.
    assert forced == "some other sub-query"
    print("Block #1 (multihop_rag) OK ->", answer)


def test_corrective_rag():
    """Block #2: exercise accept / discard+web-search / ambiguous-decompose paths."""
    docs = [
        RetrievedDoc(text="ACME Corp reported revenue of $10M in 2023.", score=0.9),   # high -> accept
        RetrievedDoc(text="The weather in Paris was mild that week.", score=0.2),      # low -> discard, web search
        RetrievedDoc(text="ACME's R&D budget grew. Marketing spend also rose. Unrelated trivia followed.", score=0.5),  # ambiguous -> decompose
    ]

    scores_by_exact_doc = {
        "ACME Corp reported revenue of $10M in 2023.": "0.95",
        "The weather in Paris was mild that week.": "0.05",
        "ACME's R&D budget grew. Marketing spend also rose. Unrelated trivia followed.": "0.5",
        "ACME's R&D budget grew": "0.9",
        "Marketing spend also rose": "0.8",
        "Unrelated trivia followed": "0.1",
    }

    def fake_llm(prompt: str) -> str:
        # Extract the exact document/sentence text the block embedded in the
        # relevance-scoring prompt and look up its canned score. Using an
        # exact match (not substring) matters here because the ambiguous doc's
        # full text contains its own sentences as substrings.
        doc_text = prompt.split("Document: ", 1)[1].rsplit("\n", 1)[0]
        return scores_by_exact_doc[doc_text]

    def fake_web_search(query: str) -> List[str]:
        return [f"web result 1 for {query}", f"web result 2 for {query}"]

    passages = corrective_rag(
        "How is ACME Corp performing financially?",
        initial_docs=docs,
        web_search=fake_web_search,
        llm=fake_llm,
        high_threshold=0.7,
        low_threshold=0.3,
    )
    assert "ACME Corp reported revenue of $10M in 2023." in passages
    assert "ACME's R&D budget grew" in passages
    assert "Marketing spend also rose" in passages
    assert any(p.startswith("web result") for p in passages)  # low-score doc triggered web search
    print("Block #2 (corrective_rag) OK ->", passages)


def test_build_contextual_chunks():
    """Block #3: prepend an LLM-generated situating sentence to each chunk."""
    document = (
        "This is a 2019 employment discrimination ruling. "
        "The plaintiff, a former shift manager, argued constructive dismissal "
        "after being reassigned to a lower-paying role."
    )
    chunks = [
        "The plaintiff argued that the reassignment amounted to constructive dismissal.",
        "The court found in favor of the defendant on the retaliation claim.",
    ]

    def fake_llm(prompt: str) -> str:
        assert "<document>" in prompt and "<chunk>" in prompt
        return "This chunk is from a 2019 employment discrimination ruling."

    result = build_contextual_chunks(document, chunks, llm=fake_llm)
    assert len(result) == 2
    for original, contextual in zip(chunks, result):
        assert contextual.startswith("This chunk is from a 2019 employment discrimination ruling.")
        assert original in contextual
    print("Block #3 (build_contextual_chunks) OK ->", result[0][:60], "...")


def test_agentic_rag_loop():
    """Block #4: one retrieve-action step then a Final Answer."""
    def fake_retrieve(query: str, k: int) -> List[str]:
        assert query == "Extropic AI programming language"
        assert k == 3
        return ["Extropic AI's hardware control software is written primarily in Rust."]

    turns = iter([
        'Thought: I need to look this up.\nAction: retrieve(query="Extropic AI programming language", k=3)',
        "Final Answer: Rust",
    ])

    def fake_llm_chat(messages: List[Dict]) -> str:
        return next(turns)

    answer = agentic_rag_loop(
        "What language does Extropic AI use?",
        retrieve=fake_retrieve,
        llm_chat=fake_llm_chat,
        max_steps=8,
    )
    assert answer == "Rust"

    # Exhausted-steps fallback path: the model never emits a tool call or a
    # Final Answer within max_steps, forcing the "please conclude now" nudge.
    def stalling_llm_chat(messages: List[Dict]) -> str:
        if messages[-1]["content"].startswith("Please provide your Final Answer now"):
            return "Final Answer: forced conclusion"
        return "Thought: still thinking, no action yet."

    forced = agentic_rag_loop(
        "unanswerable",
        retrieve=lambda q, k: [],
        llm_chat=stalling_llm_chat,
        max_steps=2,
    )
    # Note: on the exhausted-steps path the block returns the raw llm_chat()
    # output verbatim (no "Final Answer:" stripping) -- that's the book's
    # actual code, so the raw prefix is expected here.
    assert forced == "Final Answer: forced conclusion"
    print("Block #4 (agentic_rag_loop) OK ->", answer)


def test_text_to_sql_rag():
    """Block #5: SQL generation + self-correction against a real sqlite db."""
    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "toy.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE revenue (year INTEGER, amount REAL)")
        conn.executemany(
            "INSERT INTO revenue (year, amount) VALUES (?, ?)",
            [(2021, 10.0), (2022, 12.5), (2023, 15.0)],
        )
        conn.commit()
        conn.close()

        # First SQL attempt is deliberately wrong (bad column name) to exercise
        # the self-correction retry path; second attempt is correct.
        sql_attempts = iter([
            "SELECT amonut FROM revenue",           # typo -> triggers OperationalError
            "SELECT year, amount FROM revenue",       # corrected
        ])
        answer_calls = []

        def fake_llm(prompt: str) -> str:
            if prompt.startswith("Given the following SQLite schema"):
                return next(sql_attempts)
            if prompt.startswith("The query failed with:"):
                return next(sql_attempts)
            # Final natural-language answer generation
            answer_calls.append(prompt)
            return "Revenue grew from $10.0M in 2021 to $15.0M in 2023."

        answer, sql_used = text_to_sql_rag(
            "What was the revenue trend?",
            db_path=db_path,
            llm=fake_llm,
            max_retries=3,
        )
        assert sql_used == "SELECT year, amount FROM revenue"
        assert "Revenue grew" in answer
        assert len(answer_calls) == 1
        print("Block #5 (text_to_sql_rag) OK ->", answer, "| sql:", sql_used)


def test_extract_python_chunks_and_index_repository():
    """Block #6: AST-aware chunking + a fake embedder over a tiny repo."""
    source = '''"""A tiny demo module."""

class Greeter:
    """Says hello."""

    def greet(self, name):
        return f"Hello, {name}!"


def add(a, b):
    return a + b
'''
    chunks = extract_python_chunks(source, "demo.py")
    # One chunk each for the class, its method, and the module-level function.
    assert len(chunks) == 3
    kinds = {c["type"] for c in chunks}
    assert kinds == {"ClassDef", "FunctionDef"}
    linenos = [c["lineno"] for c in chunks]
    assert len(set(linenos)) == len(chunks)
    for c in chunks:
        assert "embedded_text" in c and c["context"] in c["embedded_text"]

    # Syntax-error fallback path
    broken = extract_python_chunks("def broken(:\n  pass", "broken.py")
    assert broken == [{"text": "def broken(:\n  pass", "context": "broken.py", "type": "file"}]

    class FakeEmbedder:
        def encode(self, texts, batch_size=64, show_progress_bar=False):
            # Deterministic tiny embedding: length-based vector.
            return np.array([[float(len(t)), float(t.count("\n"))] for t in texts])

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        repo.mkdir()
        (repo / "demo.py").write_text(source)
        (repo / "notes.txt").write_text("not python, should be ignored by rglob('*.py')")

        indexed = index_repository(str(repo), embedder=FakeEmbedder())
        assert len(indexed) == 3
        for c in indexed:
            assert "embedding" in c and isinstance(c["embedding"], list)

    print("Block #6 (extract_python_chunks / index_repository) OK ->", [c["type"] for c in chunks])


def test_hippoppr_retrieve():
    """Block #7: build a tiny entity graph and run real PPR + retrieval."""
    if nx is None:
        print("Block #7 SKIPPED: networkx not importable in this environment")
        return

    # A -> B -> C chain plus a disconnected D, mirroring the multi-hop story
    # from the chapter ("PyTorch" -> "Soumith Chintala" -> "Extropic AI").
    G = nx.DiGraph()
    G.add_edge("pytorch", "soumith_chintala")
    G.add_edge("soumith_chintala", "extropic_ai")
    G.add_node("unrelated_entity")

    entity_to_chunks = {
        "pytorch": ["PyTorch was created by Soumith Chintala at Facebook AI Research."],
        "soumith_chintala": ["Soumith Chintala co-founded Extropic AI in 2023."],
        "extropic_ai": ["Extropic AI's hardware control software is written primarily in Rust."],
        "unrelated_entity": ["This chunk is about something else entirely."],
    }

    rng = np.random.default_rng(0)
    entity_embeddings = {
        "pytorch": np.array([1.0, 0.0, 0.0]),
        "soumith_chintala": np.array([0.9, 0.1, 0.0]),
        "extropic_ai": np.array([0.8, 0.2, 0.0]),
        "unrelated_entity": np.array([0.0, 0.0, 1.0]),
    }
    query_embedding = np.array([1.0, 0.0, 0.0])  # aligned with "pytorch"

    # Sanity-check personalized_pagerank directly: mass should concentrate on
    # the seed and decay along the reachable chain, and never reach the
    # disconnected node. Note: this power iteration has no dangling-node mass
    # redistribution (a node with no out-edges just absorbs/stops propagating
    # incoming mass rather than returning it), so total PPR mass is NOT
    # expected to sum to 1 here -- "extropic_ai" is a dangling sink. That
    # matches the closed-form fixed point for this simple chain graph.
    alpha = 0.15
    ppr = personalized_pagerank(G, seed_nodes={"pytorch"}, alpha=alpha)
    assert set(ppr.keys()) == set(G.nodes())
    assert all(v >= 0 for v in ppr.values())
    assert ppr["unrelated_entity"] == 0.0  # disconnected from the seed
    assert ppr["pytorch"] > ppr["soumith_chintala"] > ppr["extropic_ai"] > 0
    expected_pytorch = alpha
    expected_soumith = (1 - alpha) * alpha
    expected_extropic = (1 - alpha) ** 2 * alpha
    assert abs(ppr["pytorch"] - expected_pytorch) < 1e-4
    assert abs(ppr["soumith_chintala"] - expected_soumith) < 1e-4
    assert abs(ppr["extropic_ai"] - expected_extropic) < 1e-4

    result_chunks = hippoppr_retrieve(
        query="Who created PyTorch and what did they found next?",
        G=G,
        entity_to_chunks=entity_to_chunks,
        entity_embeddings=entity_embeddings,
        query_embedding=query_embedding,
        top_entities=2,
        top_chunks=10,
        alpha=0.15,
    )
    assert any("PyTorch was created by Soumith Chintala" in c for c in result_chunks)
    print("Block #7 (hippoppr_retrieve) OK ->", result_chunks)


if __name__ == "__main__":
    test_multihop_rag()
    test_corrective_rag()
    test_build_contextual_chunks()
    test_agentic_rag_loop()
    test_text_to_sql_rag()
    test_extract_python_chunks_and_index_repository()
    test_hippoppr_retrieve()
    print("\nAll blocks in 09-rag-retrieval/05-advanced-rag.md executed successfully.")
