# 9.5 Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG

Standard Retrieval-Augmented Generation (RAG) — embed a query, fetch the top-$k$ chunks, stuff them into the context — is a remarkable baseline. It is also frequently inadequate. Real corpora contain long-range dependencies that span documents, questions that require synthesizing information from many sources, and queries where a single retrieval step structurally cannot answer multi-hop reasoning chains. This chapter covers the frontier techniques that address those limitations: **GraphRAG**, which builds explicit entity and community graphs over the corpus; **agentic and iterative retrieval**, which lets a language model decide what to retrieve next and when to stop; **self-RAG and corrective RAG**, which teach the model to evaluate its own retrievals; **contextual retrieval**, which conditions chunk representations on their document; and the architectural question of whether to retrieve at all when context windows have grown to millions of tokens.

We assume familiarity with the material in [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html) and [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html). For embedding fundamentals see [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html); for vector search mechanics see [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

## Why Naive RAG Breaks

Before fixing something it helps to know exactly how it breaks. Consider a corpus of financial reports spanning a decade. The question "How has ACME Corp's R&D spending evolved relative to its revenue growth?" requires:

1. Identifying ACME's R&D and revenue figures across ten years of filings.
2. Computing or reasoning about a ratio that is never stated verbatim.
3. Comparing trends — a conclusion that spans many documents.

A top-$k$ semantic search will likely return a handful of relevant-sounding paragraphs but will miss the temporal continuity. There is also the **multi-hop problem**: "Which portfolio company of the VC firm that led ACME's Series B later went public?" requires resolving the VC firm first, then finding their portfolio, then finding an IPO — three sequential hops, each dependent on the previous answer. No single chunk can answer this; the retriever cannot know which chunks are relevant until after it has already partially answered the question.

More failure modes:

- **Fragmented context.** A single entity (a person, a product, a regulation) may be described across dozens of chunks. Fetching only a few will give an incomplete picture.
- **Contradictory retrievals.** Chunks from different time periods or sources may contradict each other. The LLM has no way to arbitrate without explicit provenance metadata.
- **Lost-in-the-middle.** Even when the right chunks are retrieved, LLMs notoriously under-attend to information in the middle of a long context (Liu et al., *Lost in the Middle*, 2023). Retrieval order matters.
- **Retrieval-generation mismatch.** The retrieved chunk may technically be relevant but not in the right form for the generation task — e.g., a table when the model needs a narrative, or vice versa.

Each technique in this chapter targets one or more of these failure modes.

## GraphRAG: Entity and Community Graphs

### From Flat Chunks to Knowledge Graphs

GraphRAG, introduced by Edge et al. (Microsoft Research, 2024), replaces the flat chunk index with a knowledge graph built by having an LLM extract entities and relationships from every document, then running community detection to cluster related entities into hierarchical "communities," and finally generating community-level summaries.

{{fig:advrag-graphrag-indexing-pipeline}}

At query time, two modes are offered:

- **Local search**: query → entity match in graph → expand neighborhood → fetch related source chunks and community reports → generate.
- **Global search**: query → fetch top community reports at the right granularity → generate a synthesis across many communities.

Global search is uniquely powerful for questions like "What are the major themes in this corpus?" that have no single-document answer and are completely intractable for flat retrieval.

### Building a Minimal GraphRAG Pipeline

```python
"""
minimal_graphrag.py — A stripped-down GraphRAG implementation.
Requires: openai, networkx, python-louvain (community), sentence-transformers
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import networkx as nx
import community as community_louvain  # pip install python-louvain
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

client = OpenAI()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ── 1. Entity + relation extraction ──────────────────────────────────────────

EXTRACT_PROMPT = """\
Extract entities and relationships from the text below.
Return JSON only, no commentary.

Format:
{
  "entities": [{"id": "E1", "name": "...", "type": "person|org|concept|event"}],
  "relations": [{"src": "E1", "dst": "E2", "label": "..."}]
}

Text:
{text}
"""


def extract_graph_elements(text: str) -> Dict:
    """Ask the LLM to extract entities and relations from one chunk."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


# ── 2. Build the knowledge graph ─────────────────────────────────────────────

def build_knowledge_graph(chunks: List[str]) -> Tuple[nx.Graph, Dict[str, List[str]]]:
    """
    Process every chunk, merge entities by name, build an undirected graph.
    Returns (graph, entity_to_chunks) so we can fetch source text later.
    """
    G = nx.Graph()
    entity_to_chunks: Dict[str, List[str]] = {}
    name_to_node: Dict[str, str] = {}  # canonical name → node id

    for chunk_idx, chunk in enumerate(chunks):
        elements = extract_graph_elements(chunk)
        local_id_to_name = {}

        for ent in elements.get("entities", []):
            name = ent["name"].strip().lower()
            if name not in name_to_node:
                node_id = f"node_{len(name_to_node)}"
                name_to_node[name] = node_id
                G.add_node(node_id, name=ent["name"], type=ent.get("type", "unknown"))
            nid = name_to_node[name]
            local_id_to_name[ent["id"]] = nid
            entity_to_chunks.setdefault(nid, []).append(chunk)

        for rel in elements.get("relations", []):
            src = local_id_to_name.get(rel["src"])
            dst = local_id_to_name.get(rel["dst"])
            if src and dst and src != dst:
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] += 1  # co-occurrence reinforcement
                else:
                    G.add_edge(src, dst, label=rel["label"], weight=1)

    return G, entity_to_chunks


# ── 3. Community detection + summarisation ───────────────────────────────────

def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Louvain community detection. Returns node → community_id map."""
    if len(G) == 0:
        return {}
    return community_louvain.best_partition(G)


COMMUNITY_SUMMARY_PROMPT = """\
You are summarising a community of related entities from a knowledge graph.
Entities in this community: {entities}
Key relationships: {relationships}

Write a concise 3-5 sentence summary of what this community represents,
its key members, and the most important connections.
"""


def summarise_community(G: nx.Graph, node_ids: List[str]) -> str:
    """Generate an LLM summary of a single community."""
    node_set = set(node_ids)
    entities = [G.nodes[n].get("name", n) for n in node_ids[:20]]  # cap to avoid huge prompts
    rels = [
        f"{G.nodes[u].get('name', u)} —{d.get('label', '?')}→ {G.nodes[v].get('name', v)}"
        for u, v, d in G.edges(data=True)
        if u in node_set and v in node_set
    ][:30]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": COMMUNITY_SUMMARY_PROMPT.format(
                entities=", ".join(entities),
                relationships="; ".join(rels) or "none",
            ),
        }],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def build_community_index(
    G: nx.Graph,
    partition: Dict[str, int],
) -> List[Dict]:
    """
    Build a list of community records with text summaries and embeddings
    suitable for storing in a vector DB.
    """
    from collections import defaultdict
    comm_nodes: Dict[int, List[str]] = defaultdict(list)
    for node, comm_id in partition.items():
        comm_nodes[comm_id].append(node)

    records = []
    for comm_id, nodes in comm_nodes.items():
        summary = summarise_community(G, nodes)
        embedding = embedder.encode(summary).tolist()
        records.append({
            "community_id": comm_id,
            "node_count": len(nodes),
            "summary": summary,
            "embedding": embedding,
        })
    return records


# ── 4. Query: global search ───────────────────────────────────────────────────

def global_search(
    query: str,
    community_records: List[Dict],
    top_k: int = 5,
) -> str:
    """Embed query, find nearest community summaries, synthesise."""
    q_emb = embedder.encode(query)
    # cosine similarity
    sims = [
        np.dot(q_emb, rec["embedding"]) /
        (np.linalg.norm(q_emb) * np.linalg.norm(rec["embedding"]) + 1e-8)
        for rec in community_records
    ]
    top_indices = np.argsort(sims)[::-1][:top_k]
    context = "\n\n".join(community_records[i]["summary"] for i in top_indices)

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based only on the provided community summaries."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
```

The key insight of GraphRAG is the **separation of indexing granularity from retrieval granularity**. A community summary might distill 500 chunks of text into two paragraphs, allowing global questions to be answered without fitting 500 chunks into a single context window.

## Multi-Hop Retrieval: Chaining Queries

Multi-hop retrieval solves the problem that the answer to a question may require information from documents that have no direct semantic overlap with the original query. The standard technique is **iterative query decomposition**:

$$
q_0 \xrightarrow{\text{LLM decompose}} \{q_1, q_2, \ldots\} \xrightarrow{\text{retrieve}} \{D_1, D_2, \ldots\} \xrightarrow{\text{LLM reason}} q_{1}^{(2)}, q_{2}^{(2)}, \ldots
$$

Each round of retrieval produces evidence that informs the next query. The depth of the chain is bounded by a maximum step count or by the LLM deciding it has enough information. This architecture was formalised in IRCoT (Trivedi et al., 2022) and later in BeamRAG and various implementations.

```python
"""
multihop_rag.py — Iterative Chain-of-Thought retrieval with explicit decomposition.
"""

from typing import List, Tuple
import textwrap

# Assume `retrieve(query, k)` calls your vector DB and returns a list of chunk strings.
# Assume `llm(prompt)` calls your LLM and returns a string.
# Both are injected for testability.


def multihop_rag(
    original_question: str,
    retrieve,      # callable(query: str, k: int) -> List[str]
    llm,           # callable(prompt: str) -> str
    max_hops: int = 4,
    chunks_per_hop: int = 3,
) -> str:
    """
    Iterative retrieval: retrieve → reason → decide whether to retrieve again.
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
            # The model has enough to answer — we're done
            return response[len("ANSWER:"):].strip()

        # Otherwise, the model's response IS the next sub-query
        current_query = response

    # Fallback: force an answer with whatever we have
    final_prompt = (
        f"Based on the following information, answer: {original_question}\n\n"
        + "\n---\n".join(accumulated_context[:5000])
    )
    return llm(final_prompt)
```

!!! example "Worked example — multi-hop traversal"
    Suppose the question is "What programming language does the company founded by the author of PyTorch use?"

    **Hop 1** — query: "author of PyTorch" → chunk: "PyTorch was created by Soumith Chintala at Facebook AI Research."

    **Hop 2** — query: "company founded by Soumith Chintala" → chunk: "Soumith Chintala co-founded Extropic AI in 2023."

    **Hop 3** — query: "programming language used at Extropic AI" → chunk: "Extropic AI's hardware control software is written primarily in Rust."

    **Answer**: Rust. Each hop is unretievable without the previous one because the final query ("programming language at Extropic AI") has no semantic overlap with the original question.

{{fig:advrag-multihop-vs-singleshot}}

## Self-RAG and Corrective RAG

### Self-RAG: Retrieval and Quality Tokens

Asai et al. (*Self-RAG*, 2023) fine-tune an LLM to generate four types of **reflection tokens** interleaved with its normal output:

| Token type | Meaning |
|---|---|
| `[Retrieve]` | "I need external information here" |
| `[Relevant]` / `[Irrelevant]` | "This retrieved passage is / is not useful" |
| `[Supported]` / `[Partially Supported]` / `[No Support]` | "My generation is factually grounded by the passage" |
| `[Utility]` 1–5 | "This overall response is how useful to the user" |

The model learns to insert these tokens at appropriate positions. During inference, if the model generates `[Retrieve]`, the system fetches passages and feeds them back. If it generates `[Irrelevant]`, it continues generating without that passage. This creates a feedback loop where retrieval is demand-driven rather than always-on.

A simpler variant is **Corrective RAG (CRAG)** (Yan et al., 2024): after the initial retrieval, a lightweight evaluator scores each retrieved document's relevance. Low-scoring documents trigger a web search to supplement or replace the original retrieval. Documents that score ambiguously are decomposed into individual factual claims and each claim is re-verified.

```python
"""
corrective_rag.py — CRAG-style retrieval with relevance scoring and fallback.
"""

from typing import List, NamedTuple


class RetrievedDoc(NamedTuple):
    text: str
    score: float   # initial retrieval similarity


def evaluate_relevance(query: str, doc: str, llm) -> float:
    """
    Ask the LLM to score relevance 0.0–1.0.
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
      - relevance >= high_threshold → accept as-is
      - relevance <= low_threshold  → discard, trigger web search
      - in between                  → keep but also do web search
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
```

### Key Design Choice: When to Retrieve

Llama-Index and LangChain both provide "router" components that decide whether retrieval is warranted at all. The routing decision can be made with:

- **A classifier** trained to distinguish factual questions (retrieve) from conversational or creative questions (skip).
- **LLM self-assessment**: prompt the LLM with "do you need external information to answer X?".
- **Uncertainty estimation**: if the model's top token probability is high, maybe it does not need retrieval; if it is low, retrieve. This is noisy but requires no extra calls.

## Contextual Retrieval and Dense Representations

A limitation of standard chunking is that a chunk often loses its context when it is embedded in isolation. Anthropic's **contextual retrieval** technique (2024) prepends a generated context sentence to each chunk before embedding:

```python
"""
contextual_retrieval.py — Prepend document-level context to each chunk before embedding.
"""

from typing import List, Tuple


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
```

The mechanism: the embedding of "the plaintiff argued..." is ambiguous without knowing this is an employment discrimination case from 2019. The prefix "This chunk is from a 2019 employment discrimination ruling in which the plaintiff argues constructive dismissal..." resolves that ambiguity and pushes the embedding toward the right neighbourhood.

{{fig:advrag-contextual-retrieval-embedding-shift}}

Empirically (Anthropic's own report), contextual retrieval combined with BM25 hybrid search reduced retrieval failure rates substantially on the tasks tested. The cost is one additional LLM call per chunk at indexing time — acceptable for corpora that do not change frequently, expensive for streaming ingestion.

## Agentic RAG: The Retrieval Loop as an Agent

Agentic RAG treats retrieval as a **tool in an agent's tool set** rather than a fixed preprocessing step. The agent from [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html) issues search tool calls, inspects results, decides whether they suffice, and may issue follow-up searches, reformulated queries, or queries to different sources.

```python
"""
agentic_rag.py — RAG as an agent tool, using ReAct-style prompting.
Each "thought" is followed by an "action" (retrieve, synthesize, answer).
"""

import json
import re
from typing import List, Dict, Any


SYSTEM_PROMPT = """\
You are a research assistant with access to a document retrieval tool.
Use the tool as many times as needed before giving a final answer.

Available tool:
  retrieve(query: str, k: int) → list of text passages

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
            # Model didn't format a tool call — nudge it
            messages.append({
                "role": "user",
                "content": "Please continue your reasoning or provide a Final Answer.",
            })

    # Exhausted steps — force a conclusion
    messages.append({
        "role": "user",
        "content": "Please provide your Final Answer now based on what you have gathered.",
    })
    return llm_chat(messages)
```

Agentic RAG has higher latency than single-shot RAG (multiple LLM round-trips), but it dramatically improves accuracy on complex questions. The agent can also invoke **multiple retrieval backends**: a code search index for code questions, a web search tool for current events, a SQL query tool for structured data, and a vector database for long-form prose — all in the same conversation.

The connection to [Multi-Agent Systems & Orchestration](../08-agents-harness/07-multi-agent-systems.html) is direct: in multi-agent architectures, individual sub-agents may each be a specialised RAG pipeline. A router agent dispatches to the right specialist.

## RAG Over Structured Data and Code

### SQL and Tabular Data

When the knowledge base is a relational database rather than free text, the retrieval problem becomes **Text-to-SQL**: translate the natural language question into a query, execute it, and feed the results to the generator.

```python
"""
text_to_sql_rag.py — Minimal Text-to-SQL RAG with schema grounding.
"""

import sqlite3
from typing import List, Tuple


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
    Convert question → SQL → execute → generate final answer.
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
```

### Code Retrieval

For RAG over code repositories, the chunking strategy must respect code structure. Chunk at the **function or class boundary**, not at fixed token counts. Include the function signature in every chunk's context (contextual retrieval applied to code).

```python
"""
code_rag_chunker.py — AST-aware chunking for Python code repositories.
"""

import ast
from pathlib import Path
from typing import List, Dict


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
```

The embedding should be the **comment-stripped, docstring-enriched** function signature plus body. For very large functions (>200 lines), split at the logical block level and use the function signature as the contextual prefix for each sub-chunk.

## Long Context vs. RAG: A Framework for the Choice

The most important architectural decision in 2025 is whether to retrieve at all. With context windows of 128k, 200k, and even 1M tokens, it is tempting to skip the retrieval pipeline entirely and simply stuff the entire corpus into the prompt. This section gives you a framework for making that tradeoff.

### Memory and Cost Analysis

Let $n$ be the number of documents in the corpus, $\bar{L}$ the average document length in tokens, and $d$ the model's context window size.

**Trivially fits in context:** if $n \cdot \bar{L} \ll d$, just put everything in the context. Retrieval adds engineering complexity for no gain. For a 200k-token window, this means corpora up to roughly 150 books worth of text at 1,000 words each.

**Attention cost at long contexts:** for a prefill of $L$ tokens, the attention computation is $O(L^2 d_{\text{model}} / d_k)$ FLOPs. Doubling the context quadruples the FLOPs (and the cost, for API-based systems).

!!! example "Cost comparison: retrieval vs. long context"
    Suppose you have a corpus of 1,000 documents, each 2,000 tokens — total 2 million tokens. The LLM charges USD 2.00 per million tokens input.

    **Long-context approach (put everything in):**
    Cost per query = 2,000,000 tokens × USD 2.00/1M = **USD 4.00 per query**.
    At 1,000 queries/day, that's USD 4,000/day.

    **RAG approach (retrieve top 10 chunks of 200 tokens each):**
    Retrieval overhead ≈ 1× embedding call (cheap) + query tokens.
    Context sent to LLM ≈ 2,000 tokens = **USD 0.004 per query**.
    At 1,000 queries/day, that's USD 4/day.

    The RAG approach is 1,000× cheaper here. The long-context approach is superior only if the question genuinely requires holistic synthesis across the entire corpus and RAG would miss key connections — which is the case GraphRAG's global search handles more cheaply than stuffing everything in.

### When Long Context Wins

| Scenario | Winner | Why |
|---|---|---|
| Few documents, holistic analysis | Long context | Retrieval might miss the right passages |
| "Summarize this 50-page contract" | Long context | All information is needed |
| Needle-in-a-haystack on small corpus | Long context | Simpler pipeline, high recall |
| Reading a codebase to answer one question | RAG | Most code is irrelevant |
| Question-answering over millions of docs | RAG | Cannot fit in context window |
| Multi-hop over structured relationships | GraphRAG | Semantic search alone cannot hop |
| Time-sensitive / frequently updated data | RAG | Reindexing is cheaper than re-prompting |

### Lost-in-the-Middle: The Catch

Even when documents fit in the context window, placement matters. LLMs reliably attend to information at the beginning and end of the context but under-attend to the middle (Liu et al., 2023). If you have 20 relevant chunks and the answer is in chunk 14, placing it 75% of the way through the context hurts performance more than just retrieving the right chunk alone.

Mitigation strategies:

- **Relevance-order placement**: put the most relevant retrieved chunk first, not interleaved at random.
- **Recency bias correction**: for time-stamped corpora, recent documents tend to be more relevant and should be placed near the query.
- **Chain-of-density reranking**: rerank retrieved chunks by predicted reading order, not retrieval score.

!!! interview "Interview Corner"
    **Q:** A candidate says "My corpus is only 50,000 tokens, so I'll just throw it all in the context window every time. RAG is unnecessary complexity." How do you evaluate this claim?

    **A:** The claim is reasonable for that corpus size but incomplete. Three considerations push back: (1) **Latency and cost**: even 50k tokens in prefill adds meaningful TTFT (time-to-first-token) delay and costs money at scale — at 1,000 queries/day, the bill adds up. (2) **Lost-in-the-middle**: if the answer is in a specific document, placing all 50k tokens in context may actually harm accuracy versus a targeted 2k-token retrieval. (3) **Freshness and privacy**: if the corpus updates frequently or contains sensitive documents the user shouldn't always see, selective retrieval is architecturally cleaner. The right answer is "it depends on query load, update frequency, and whether holistic reasoning across all documents is genuinely needed." A retrieval-free approach is valid when queries require synthesising the full corpus and the corpus is small enough not to trigger lost-in-the-middle issues.

## Frontier Techniques: HippoRAG, RAPTOR, and Beyond

### RAPTOR: Recursive Summarisation Trees

RAPTOR (Sarthi et al., 2024) addresses the multi-granularity problem. Instead of only indexing leaf chunks, it builds a tree by recursively clustering chunks, summarising each cluster with an LLM, and indexing both the summaries and the leaves.

{{fig:advrag-raptor-summary-tree}}

A query that requires high-level synthesis will match summary nodes; a query for a specific detail will match leaf chunks. The retrieval score propagates to the appropriate level automatically.

### HippoRAG: Personalized PageRank over Entity Graphs

HippoRAG (Gutierrez et al., 2024) combines the semantic richness of dense retrieval with the structure of knowledge graphs. After extracting a Hippocampus-inspired entity-centric graph, HippoRAG uses **Personalized PageRank** (PPR) seeded at query-matched entities to propagate relevance through the graph:

$$
\text{PPR}(v) = \alpha \cdot \frac{1}{|\text{seed}|}\sum_{s \in \text{seed}} \mathbf{1}[v = s] + (1 - \alpha) \sum_{u \in \text{in-neighbours}(v)} \frac{\text{PPR}(u)}{|\text{out-neighbours}(u)|}
$$

where $\alpha$ is the teleport probability (typically 0.15) and seed nodes are the entities that match the query. PPR propagates importance from the seed nodes through the graph's edges, meaning passages connected to multiple relevant entities get boosted even if they don't match the query directly. This handles the "integration across entities" problem that flat vector search misses.

{{fig:advrag-hipporag-ppr-propagation}}

```python
"""
hippoppr.py — Personalized PageRank over an entity graph for HippoRAG-style retrieval.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Set


def personalized_pagerank(
    G: nx.DiGraph,
    seed_nodes: Set[str],
    alpha: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Power-iteration PPR.
    alpha: teleport probability back to seed nodes.
    Returns a dict of node → PPR score.
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
    G: nx.DiGraph,
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
```

The power of PPR is that it naturally handles **transitive relevance**: an entity three hops away from the query seed can still accumulate high PPR score if it is densely connected to highly-scored intermediate entities. This is the multi-hop reasoning capability that flat vector search lacks, implemented without requiring the LLM to explicitly generate sub-queries.

!!! warning "Common pitfall — graph quality bottleneck"
    All graph-based RAG methods are only as good as the entity extraction step. If the LLM-based extractor misses aliases (ACME Corp vs. Acme Corporation vs. the company), the graph becomes disconnected and multi-hop reasoning fails. Always normalise entity names (lowercasing, fuzzy matching, co-reference resolution with a dedicated NER model) before adding nodes. Evaluate entity extraction recall separately from end-to-end RAG quality.

## Putting It All Together: Choosing Your RAG Architecture

{{fig:advrag-architecture-decision-tree}}

In practice, production systems combine multiple approaches: a **router** that classifies the query type, dispatches to the appropriate retrieval strategy, and optionally falls back to long-context if retrieval fails. This is exactly the [Context Engineering & Management](../08-agents-harness/04-context-engineering.html) problem addressed in Part VIII.

For indexing, the Anthropic contextual retrieval finding is almost universally applicable: the marginal cost of adding a one-sentence LLM-generated context prefix to each chunk at index time is small, and the retrieval recall improvement is consistent. Pair it with hybrid BM25 + dense retrieval (see [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)) and a cross-encoder reranker for a strong baseline before reaching for more complex graph-based methods.

!!! key "Key Takeaways"
    - **GraphRAG** builds entity/community graphs over the corpus and enables global synthesis questions that flat retrieval cannot answer; community summaries are the key primitive.
    - **Multi-hop retrieval** (IRCoT, agentic RAG) solves questions where no single chunk is sufficient by iteratively retrieving and reasoning, using each retrieval's result to form the next query.
    - **Self-RAG** and **Corrective RAG** teach the model to judge its own retrievals and trigger additional search when retrieved content is irrelevant or contradictory.
    - **Contextual retrieval** reduces chunk decontextualisation by prepending an LLM-generated context sentence before embedding; works especially well combined with hybrid BM25+dense search.
    - **Long context vs. RAG** is a cost/quality tradeoff: long context wins on holistic synthesis over small corpora; RAG wins on large corpora, high query volume, and cases where a targeted chunk suffices.
    - **Lost-in-the-middle** means that even when you use long context, the placement of relevant information matters — put the most relevant material at the beginning or end.
    - **HippoRAG's Personalized PageRank** propagates relevance through the entity graph without requiring explicit sub-query generation, handling transitive multi-hop paths naturally.
    - **RAG over structured data** requires Text-to-SQL with self-correction; RAG over code requires AST-aware chunking at function/class boundaries.
    - **Entity extraction quality** is the bottleneck for all graph-based methods — invest in alias normalisation and co-reference resolution before graph construction.

!!! sota "State of the Art & Resources (2026)"
    Advanced RAG has matured into a rich ecosystem: graph-based methods (GraphRAG, HippoRAG) handle multi-hop synthesis; agentic and iterative pipelines (IRCoT, Self-RAG, CRAG) adapt retrieval dynamically; and the long-context vs. RAG tradeoff is now a principled cost/quality decision rather than a guess.

    **Foundational work**

    - [Trivedi et al., *IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning* (2022)](https://arxiv.org/abs/2212.10509) — established iterative CoT-guided retrieval for multi-hop QA; the template for modern multi-hop pipelines.
    - [Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023)](https://arxiv.org/abs/2307.03172) — demonstrated that LLMs under-attend to middle-context information, motivating placement-aware retrieval strategies.

    **Recent advances (2023–2026)**

    - [Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (2023)](https://arxiv.org/abs/2310.11511) — introduced reflection tokens so models demand retrieval on-demand and self-grade relevance; ICLR 2024 oral.
    - [Yan et al., *Corrective Retrieval Augmented Generation* (2024)](https://arxiv.org/abs/2401.15884) — lightweight evaluator triggers web search fallback when retrieved docs score too low; plug-and-play on any RAG stack.
    - [Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (2024)](https://arxiv.org/abs/2404.16130) — Microsoft's GraphRAG using community detection and hierarchical summaries; uniquely addresses global sensemaking questions.
    - [Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (2024)](https://arxiv.org/abs/2401.18059) — recursive clustering+summarization builds multi-granularity trees, improving holistic synthesis; ICLR 2024.
    - [Gutiérrez et al., *HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs* (2024)](https://arxiv.org/abs/2405.14831) — Personalized PageRank over entity graphs propagates relevance for multi-hop retrieval without explicit sub-query generation; NeurIPS 2024.

    **Open-source & tools**

    - [microsoft/graphrag](https://github.com/microsoft/graphrag) — official Microsoft GraphRAG library; full pipeline from LLM entity extraction to community reports and local/global query modes.
    - [OSU-NLP-Group/HippoRAG](https://github.com/osu-nlp-group/hipporag) — reference implementation of HippoRAG with KG construction, PPR retrieval, and HippoRAG 2 updates.
    - [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) — official RAPTOR implementation for recursive tree-organized retrieval.

    **Go deeper**

    - [Anthropic, *Introducing Contextual Retrieval* (2024)](https://www.anthropic.com/news/contextual-retrieval) — Anthropic's report on prepending LLM-generated context to chunks; reduces retrieval failures by 49% and 67% with reranking.
    - [LlamaIndex, *Agentic RAG With LlamaIndex* (blog)](https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6) — practical walkthrough of hierarchical document-agent architectures for production agentic RAG.

## Further Reading

- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, Microsoft Research, 2024.
- Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, ICLR 2024.
- Yan et al., *Corrective Retrieval Augmented Generation (CRAG)*, 2024.
- Trivedi et al., *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)*, ACL 2023.
- Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, ICLR 2024.
- Gutierrez et al., *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*, NeurIPS 2024.
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2023.
- Anthropic, *Contextual Retrieval* (blog post), 2024.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020 — the original RAG paper.
- Microsoft GraphRAG open-source repository: `microsoft/graphrag` on GitHub.

## Exercises

**1.** Consider the chapter's multi-hop question: *"Which portfolio company of the VC firm that led ACME's Series B later went public?"* Explain why a single top-$k$ semantic search over a flat chunk index structurally cannot answer this, no matter how large $k$ is. Then describe, in terms of the chapter's iterative decomposition template, the minimum sequence of retrievals that *can* answer it.

??? note "Solution"
    The question is a chain of three dependent hops: (a) find the VC firm that led ACME's Series B, (b) find that firm's portfolio companies, (c) find which of those went public. The retriever ranks chunks by semantic similarity to the *query text*. But the chunk that actually contains the answer — a filing about some portfolio company's IPO — shares essentially no vocabulary or embedding-space proximity with the phrase "VC firm that led ACME's Series B." As the chapter puts it: "the retriever cannot know which chunks are relevant until after it has already partially answered the question." Increasing $k$ does not help, because the relevant chunk is not merely ranked low — it is not semantically close to the *original* query at all; it is only close to a query you can't write until hop (a) and (b) are resolved.

    The iterative template $q_0 \xrightarrow{\text{decompose}} q_1 \xrightarrow{\text{retrieve}} D_1 \xrightarrow{\text{reason}} q_2 \ldots$ resolves it in three retrievals:

    - Hop 1 — query "who led ACME's Series B" → retrieves the VC firm name, e.g. "Foobar Ventures."
    - Hop 2 — query "portfolio companies of Foobar Ventures" (only writable *after* hop 1) → retrieves the portfolio list.
    - Hop 3 — for the portfolio companies, query "which went public / IPO" → retrieves the IPO filing.

    Each query is constructed from the *evidence returned by the previous hop*, which is exactly what single-shot retrieval cannot do.

**2.** Anthropic's contextual retrieval prepends an LLM-generated context sentence to each chunk before embedding. (a) Using the chapter's own example ("the plaintiff argued..."), explain *mechanically* why this changes the chunk's position in embedding space and improves retrieval. (b) The chapter says this technique is "acceptable for corpora that do not change frequently, expensive for streaming ingestion." Quantify the indexing cost driver and explain the streaming-ingestion problem.

??? note "Solution"
    (a) An embedding model maps text to a vector based on the tokens present. The bare chunk "the plaintiff argued..." contains no tokens indicating *which case, what year, or what legal issue*, so its embedding lands in a generic "legal argument" region, far from a query like "2019 constructive dismissal employment case." Prepending "This chunk is from a 2019 employment discrimination ruling in which the plaintiff argues constructive dismissal..." injects the tokens *2019, employment, discrimination, constructive dismissal* into the text that is embedded. Those tokens shift the resulting vector toward the region occupied by such queries — the chapter's `advrag-contextual-retrieval-embedding-shift` figure. The stored/returned text can still be the raw chunk; only the *embedded* text carries the prefix (see `embedded_text` vs. raw chunk in the code).

    (b) The cost driver is **one additional LLM call per chunk at index time** (the `CONTEXT_PROMPT` call in `build_contextual_chunks`). For a static corpus of $N$ chunks this is a one-time cost of $N$ LLM calls, amortised over all future queries — cheap per query. For **streaming ingestion**, documents arrive continuously, so every new chunk incurs its LLM call *at ingest latency*, and the per-chunk LLM call sits on the write path adding both cost and latency to every insert. A corpus churning millions of chunks/day pays the full $N$-call cost repeatedly and continuously, which is why the technique is favored for slowly-changing corpora where the one-time index cost is dwarfed by query volume.

**3.** The chapter states that for a prefill of $L$ tokens, attention costs $O(L^2 d_{\text{model}} / d_k)$ FLOPs, so "doubling the context quadruples the FLOPs." A team is deciding between sending a 12,000-token retrieved context and a 48,000-token long-context prompt to the same model. Ignoring all non-attention costs, by what factor does the attention computation grow, and what does this imply about the cost framing in the chapter's "long context vs. RAG" comparison?

??? note "Solution"
    Attention scales as $L^2$. The ratio of the two prefill lengths is

    $$
    \frac{L_{\text{long}}}{L_{\text{RAG}}} = \frac{48{,}000}{12{,}000} = 4.
    $$

    Since attention FLOPs scale with $L^2$, the attention cost grows by

    $$
    \left(\frac{48{,}000}{12{,}000}\right)^2 = 4^2 = 16\times.
    $$

    So the long-context prompt requires roughly **16 times** the attention computation of the RAG prompt, even though it carries only 4 times the tokens. This is *super-linear*: the chapter's per-token API pricing (which is linear in tokens) actually *understates* the compute burden of long context, because the quadratic attention term means the marginal token near the end of a long context is far more expensive to process than a token in a short context. It reinforces the chapter's conclusion that RAG's targeted, short contexts win decisively on cost whenever a small retrieved set suffices.

**4.** Adapt the chapter's "Cost comparison" worked example to new numbers. A corpus has **500 documents of 4,000 tokens each**. The model charges **USD 3.00 per million input tokens**. Compute, for a single query: (a) the long-context cost (stuff everything in), (b) the RAG cost retrieving the **top 8 chunks of 250 tokens each**, and (c) the ratio between them. Then (d) at **2,000 queries/day**, give the daily cost of each approach.

??? note "Solution"
    (a) **Long context.** Total corpus tokens $= 500 \times 4{,}000 = 2{,}000{,}000$ tokens. Cost per query:

    $$
    2{,}000{,}000 \times \frac{3.00}{1{,}000{,}000} = \text{USD } 6.00 \text{ per query.}
    $$

    (b) **RAG.** Context sent $= 8 \times 250 = 2{,}000$ tokens (the query tokens and one cheap embedding call are negligible, per the chapter). Cost per query:

    $$
    2{,}000 \times \frac{3.00}{1{,}000{,}000} = \text{USD } 0.006 \text{ per query.}
    $$

    (c) **Ratio:**

    $$
    \frac{6.00}{0.006} = 1{,}000\times.
    $$

    RAG is 1,000x cheaper per query — the same order of magnitude the chapter found.

    (d) **At 2,000 queries/day:**

    - Long context: $6.00 \times 2{,}000 = \text{USD } 12{,}000\text{/day}$.
    - RAG: $0.006 \times 2{,}000 = \text{USD } 12\text{/day}$.

    Long context wins here *only* if the queries genuinely require holistic synthesis across all 500 documents (the case GraphRAG's global search handles more cheaply); otherwise RAG saves ~USD 11,988/day.

**5.** The chapter warns about **lost-in-the-middle** and recommends "relevance-order placement: put the most relevant retrieved chunk first." A stronger mitigation, since LLMs attend well to *both* the beginning and end of the context, is to place the highest-scoring chunks at the two ends and bury the weakest in the middle. Implement a function `edge_weighted_order(chunks_with_scores)` that takes a list of `(chunk_text, score)` pairs and returns the list of chunk texts reordered so that the highest-scoring chunk is first, the second-highest is last, the third-highest is second, the fourth-highest second-to-last, and so on — draining from the outside in. Keep it consistent with the chapter's plain-Python style.

??? note "Solution"
    Sort by score descending, then deal the sorted chunks alternately to the front and back of the output, so rank 1 → position 0, rank 2 → last position, rank 3 → position 1, rank 4 → second-to-last, etc. The weakest chunks end up in the middle, where the model under-attends.

    ```python
    """
    edge_placement.py — Mitigate lost-in-the-middle by placing the strongest
    retrieved chunks at both ends of the context and the weakest in the middle.
    """

    from typing import List, Tuple


    def edge_weighted_order(chunks_with_scores: List[Tuple[str, float]]) -> List[str]:
        """
        Reorder chunks so the highest-scoring go to the outer edges and the
        lowest-scoring collapse into the middle.

        rank 1 -> front, rank 2 -> back, rank 3 -> front, rank 4 -> back, ...
        """
        # 1. Sort by score, best first.
        ranked = [text for text, _ in
                  sorted(chunks_with_scores, key=lambda cs: cs[1], reverse=True)]

        n = len(ranked)
        result: List[str] = [None] * n
        front, back = 0, n - 1

        for i, text in enumerate(ranked):
            if i % 2 == 0:          # ranks 1, 3, 5, ... -> front
                result[front] = text
                front += 1
            else:                   # ranks 2, 4, 6, ... -> back
                result[back] = text
                back -= 1

        return result


    # ── quick check ───────────────────────────────────────────────────────────
    if __name__ == "__main__":
        example = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6), ("E", 0.5)]
        # sorted best->worst: A, B, C, D, E
        # A->front, B->back, C->front, D->back, E->front
        print(edge_weighted_order(example))
        # -> ['A', 'C', 'E', 'D', 'B']
    ```

    The strongest chunk `A` sits first and the second-strongest `B` sits last — both in high-attention positions — while the weakest chunk `E` lands dead center, exactly where the chapter says the model under-attends. This matches the chapter's guidance to "put the most relevant material at the beginning or end."

**6.** HippoRAG runs Personalized PageRank via the power iteration $r \leftarrow \alpha\, p + (1-\alpha)\, A^{\top} r$ from the chapter's `personalized_pagerank` code, where $A$ is the column-normalized transition matrix and $p$ is the seed personalization vector. Consider a tiny directed entity graph with edges $A \to B$, $B \to C$, $C \to A$ (a 3-node cycle). The query matches only entity $A$, so the seed set is $\{A\}$ and $p = [1, 0, 0]$ (order $A, B, C$). Using teleport probability $\alpha = 0.15$ and starting from $r_0 = p$, run **two** power iterations by hand. Which entity ends up with the highest PPR score, and what does that illustrate about the method?

??? note "Solution"
    **Set up the matrices.** With row = "from", column = "to", the adjacency matrix is

    $$
    A = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}
    \quad (A\to B,\; B\to C,\; C\to A).
    $$

    Each column already sums to 1 (every node has out-degree 1), so column-normalization leaves $A$ unchanged. The iteration uses $A^{\top}$:

    $$
    A^{\top} = \begin{bmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}.
    $$

    Seed / personalization: $p = [1, 0, 0]$, and $r_0 = p = [1, 0, 0]$.

    **Iteration 1.** First $A^{\top} r_0$:

    $$
    A^{\top}\,[1,0,0]^{\top} = [\,0,\;1,\;0\,]^{\top}.
    $$

    Then

    $$
    r_1 = 0.15\,[1,0,0] + 0.85\,[0,1,0] = [\,0.15,\;0.85,\;0\,].
    $$

    **Iteration 2.** First $A^{\top} r_1$:

    $$
    A^{\top}\,[0.15,\,0.85,\,0]^{\top} = [\,0,\;0.15,\;0.85\,]^{\top}
    $$

    (row $A$ picks component $C=0$; row $B$ picks component $A=0.15$; row $C$ picks component $B=0.85$). Then

    $$
    r_2 = 0.15\,[1,0,0] + 0.85\,[0,\,0.15,\,0.85]
        = [\,0.15,\;0.1275,\;0.7225\,].
    $$

    **Result.** After two iterations the scores are $A = 0.15$, $B = 0.1275$, $C = 0.7225$, so **entity $C$ has the highest PPR score** — even though the query matched only $A$, and $C$ is two hops away ($A \to B \to C$).

    **What it illustrates.** This is exactly the "transitive relevance" the chapter highlights: PPR propagates importance from the seed through the graph's edges, so an entity several hops from the query seed can accumulate high score without ever matching the query directly. Chunks attached to $C$ would be retrieved as relevant, giving multi-hop reasoning *without* the explicit LLM sub-query generation that IRCoT-style pipelines require.
