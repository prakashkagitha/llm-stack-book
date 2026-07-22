# 8.5 Memory Systems for Agents

Every human expert you have ever admired carries three things into a meeting: working notes they can reference in real time, a personal knowledge base built from years of experience, and an episodic record of what happened last time they worked with you. LLM agents need the same three things — and the context window alone cannot provide all of them. Without external memory, an agent reset to a blank context after 200,000 tokens is not a persistent assistant; it is a brilliant amnesiac.

This chapter maps the full memory stack for LLM agents: from the in-context scratch pad to durable vector stores, file systems, and knowledge graphs. We cover how to architect write, read, and reflect operations; the distinction between episodic and semantic memory; memory compaction strategies that fight context-window limits; and the file-as-memory pattern that many production agents rely on. We close with a fully runnable from-scratch implementation you can drop into your own harness.

Related chapters: this one focuses on the *memory layer* of agents. For the broader agentic loop, see [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html). For how retrieved text gets injected back into the model, see [Context Engineering & Management](../08-agents-harness/04-context-engineering.html). For the vector database machinery underneath, see [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

---

## 8.5.1 The Two Memory Horizons

We can draw a clean axis: memory that lives *inside* the context window versus memory that lives *outside* it. Everything inside is available on every token prediction; everything outside must be explicitly fetched.


{{fig:agentmem-two-horizons}}


**Short-term (in-context) memory** is the running conversation: system prompt, prior turns, tool call/result pairs, and any snippets the agent has fetched. Its capacity is bounded by the context length and its cost by the token budget. A 128 k-token model at on the order of \$3 per million output tokens will spend roughly \$0.04 per completely filled context — cheap per call, but it accumulates fast in long-running sessions. See [Context Engineering & Management](../08-agents-harness/04-context-engineering.html) for how to manage what goes into this budget.

**Long-term (external) memory** is any storage system the agent can query and update. It is not bounded by the context window, persists across sessions, and can be shared across many agent instances. Its cost is dominated by embedding latency (for vector retrieval) and I/O rather than per-token model pricing.

The fundamental design challenge: you must decide *what* to commit to external memory, *when*, *in what form*, and *how to retrieve it later accurately*. Getting any one of these wrong produces either a useless store (nothing relevant comes back on retrieval) or a poisoned store (stale or contradictory facts pollute the context).

---

## 8.5.2 Episodic vs Semantic Memory

Psychology distinguishes two kinds of long-term memory in humans; the same distinction is useful for agents.

**Episodic memory** records *events with their temporal context* — "on 2025-11-04, the user asked me to refactor the authentication module and I found the JWT library had a bug." Each entry is a timestamped, situated record of a specific interaction. Episodic stores are append-only by nature, grow monotonically, and are queried with temporal or situational cues ("what did we decide last Tuesday?").

**Semantic memory** records *general knowledge and facts* — "the production database uses PostgreSQL 15; the API gateway rate-limits at 1 000 req/s; the user prefers British English." Semantic entries are facts that should be true across sessions. They are updated (overwritten) when new information supersedes old, and queried with content-based cues ("what do I know about the database?").

In practice, most memory systems blend both types. A common architecture uses:

| Layer | Type | Store | Update rule |
|---|---|---|---|
| Session transcript | Episodic | Append-only log (file / DB) | Never delete, only append |
| Distilled facts | Semantic | Key-value or vector store | Upsert on new evidence |
| User preferences | Semantic | Structured record | Overwrite on explicit correction |
| Task outcomes | Episodic + semantic | Vector store | Append event; update fact on success |

The key insight: do not collapse both into a single vector store with no separation. Mixing a "2025-11-04: JWT bug" episodic entry with a "the JWT library version is 4.1.0" semantic fact produces a store where date-specific noise pollutes general-knowledge queries.

---

## 8.5.3 Memory Store Types

### 8.5.3.1 Vector Stores

A vector store encodes each memory as a dense embedding vector and supports approximate nearest-neighbour (ANN) retrieval. This is the most common substrate for agent memory because natural language is naturally suited to embedding-based similarity.

Given a query $q$ and a corpus of stored memories $\{m_1, m_2, \ldots, m_N\}$, the retrieval objective is:

$$
\text{retrieve}(q) = \arg\!\max_{m_i \in \mathcal{M}} \cos\!\left(\mathbf{e}(q),\, \mathbf{e}(m_i)\right)
$$

where $\mathbf{e}(\cdot)$ is an embedding model and $\cos$ is cosine similarity. In practice we retrieve the top-$k$ memories and re-rank them. See [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html) for embedding model choices and [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html) for re-ranking strategies.

**Strengths:** fuzzy matching, handles paraphrase, scales to millions of entries with HNSW or IVF indexes.

**Weaknesses:** exact-match failure ("what is the API key for service X?"), no structured filtering without metadata, embedding drift if you change embedding models.

### 8.5.3.2 Key-Value and Structured Stores

For structured facts — user preferences, configuration, task state — a simple key-value store or a relational database is superior. SQLite is an excellent single-agent choice: it is serverless, ACID-compliant, and has a negligible setup cost. A Redis or DynamoDB instance suits multi-agent setups.

Pattern: store semantic facts with a canonical key (for example, `db.engine`, `user.language`) and retrieve deterministically. No embedding required; latency is sub-millisecond.

### 8.5.3.3 Property Graphs

A property graph (Neo4j, Kuzu, or even NetworkX for small deployments) represents knowledge as nodes and labelled edges: `(User)--[PREFERS]-->(BritishEnglish)`, `(Service:AuthAPI)--[RATE_LIMITED_AT]-->(1000 req/s)`. Graph stores shine when relationships between entities matter — "which services depend on the authentication module?" is a graph traversal, not a similarity search.

Agents that build knowledge graphs over multiple sessions can answer multi-hop queries that purely flat memory architectures cannot, at the cost of a more complex write pipeline that must extract entities and relationships before storing.

### 8.5.3.4 File System as Memory

The simplest form of persistent memory is a structured file system directory. The agent writes markdown files, JSON blobs, or code artefacts; a lightweight indexer (even a plain `find` + `grep`) serves retrieval. This is the *file-as-memory* pattern: the working directory is the agent's notebook.


{{fig:agentmem-file-as-memory-tree}}


Several production agents (including early versions of Devin and Claude Code's memory feature) use precisely this pattern. It requires no external database, is human-readable and auditable, and integrates with version control. Retrieval is handled either by fuzzy search over the file tree or by maintaining a separate vector index of file contents.

---

## 8.5.4 The Write / Read / Reflect Triad

Every memory operation falls into one of three categories.

### 8.5.4.1 Write

A memory *write* commits information from the current context to external storage. Writes happen:

- **After a significant discovery** — the agent learned something new (a fact, a bug, a user preference).
- **At session boundaries** — before the context is cleared, the agent saves a summary.
- **On task completion** — the outcome of a sub-task is logged for future reference.
- **On explicit instruction** — the user says "remember this."

The write step must decide *what* to save, *how to format it*, and *what metadata to attach* (timestamp, session ID, confidence, topic tags). Writes that are too verbose pollute the store; writes that are too sparse lose signal.

### 8.5.4.2 Read

A memory *read* queries external storage and injects the result into the current context. Reads happen:

- **At session start** — bootstrap the context with relevant prior knowledge.
- **Before a tool call** — check whether the agent already knows the answer.
- **When the agent signals uncertainty** — "I don't remember whether..." triggers a retrieval.
- **Proactively, by the harness** — the orchestrating code detects topic shifts and retrieves matching memories.

The retrieved snippets must be ranked and filtered: if you inject the top-20 vector matches unfiltered, you fill the context with noise. A conservative top-$k$ of 3–5 with a relevance threshold works well for most agents.

### 8.5.4.3 Reflect

Reflection is the highest-value memory operation and the most commonly skipped. A *reflect* operation asks the agent to read a batch of recent episodes and distil new semantic facts or updated beliefs:

```text
"Given the last 10 interactions with this user, what are the three most important
 things to know about their preferences, working style, and current project state?"
```

Reflection produces compact, high-signal semantic memories from noisy episodic records. Generative Agents (Park et al., 2023) showed that reflection loops dramatically improve the coherence of long-running simulated agents. The reflect step should run: (a) periodically (every N episodes), (b) before a session ends, or (c) when a new episode contradicts an existing fact.

{{fig:agentmem-reflect-distillation}}

---

## 8.5.5 Memory Compaction

Even with external memory, the in-context portion — the conversation so far — grows unboundedly in a long session. *Memory compaction* is the process of replacing a long conversation prefix with a shorter, lossier representation while keeping the most important information.

### 8.5.5.1 Summarisation-Based Compaction

The simplest approach: when the conversation exceeds a threshold (say, 80% of the context window), ask the model to summarise the prefix into a bullet list, then discard the prefix and replace it with the summary.

$$
C_{\text{new}} = [\text{system prompt},\; \text{summary}(C_{1:t-k}),\; C_{t-k+1:t}]
$$

This preserves recent turns verbatim and compresses older turns. The rolling-summary pattern applies this recursively: the next time you compact, you summarise the previous summary plus the new window.

{{fig:agentmem-compaction-sliding-window}}

**Loss is guaranteed.** You must accept that some information will not survive compaction. This is the right trade-off for most tasks, but you should write high-value facts to the external store *before* compacting, not after.

### 8.5.5.2 Hierarchical Compaction

For very long sessions, a two-level hierarchy works well:

1. **Micro-summaries:** every 20 turns, summarise into a 200-word snippet and push to the episodic log.
2. **Macro-summaries:** every 10 micro-summaries, distil into a semantic fact update via a reflect call.

The context then always contains: system prompt + macro-summary + recent micro-summary + last N turns. This is bounded in size regardless of session length.

### 8.5.5.3 Structured Compaction

Rather than free-text summaries, some agents compact into structured formats:

```json
{
  "task_state": "In progress — refactoring auth module",
  "decisions_made": ["Use JWT HS256", "Add rate limiting at 500 req/s"],
  "open_questions": ["How to handle token refresh for mobile clients?"],
  "user_preferences": {"language": "TypeScript", "style_guide": "Google"},
  "next_steps": ["Implement refresh endpoint", "Write integration tests"]
}
```

This structured compaction is more retrievable and less prone to hallucination than free-text summaries because the agent is filling in a schema rather than generating freely.

---

## 8.5.6 Session Persistence: How Agents Survive Restarts

A naive agent starts each session with a blank context. A persistent agent carries its prior self into every new session. The architecture has four moving parts:

1. **Session ID:** a stable identifier for this user/project/thread. Every memory write tags itself with the session ID; every session-start retrieval filters by it.

2. **Bootstrap read:** the first thing the harness does is query external memory for entries relevant to this session, and prepend them to the system prompt. This gives the agent an immediate sense of continuity.

3. **Graceful shutdown write:** when a session ends (timeout, user exit, token limit), the harness triggers a compaction pass that writes a session summary to the episodic store and updates any semantic facts that changed.

4. **Cross-session reflection:** periodically (for example, at the start of a new week), a background job runs a reflection pass over all episodes, updates semantic facts, and prunes redundant episodic entries. This is the equivalent of sleeping — the agent consolidates its knowledge.


{{fig:agentmem-session-persistence-flow}}


The inter-session state that must be persisted is at minimum: the session summary, any newly learned facts, and any open task state (what was in progress when the session ended).

---

## 8.5.7 A From-Scratch Memory Store Implementation

The following implementation is a self-contained, runnable memory system using only Python standard library plus `numpy` and `sentence-transformers`. It implements all three memory types: episodic log (append-only), semantic facts (key-value), and a vector store for similarity search. It is designed to be readable and hackable rather than production-hardened.

```python
"""
agent_memory.py  —  A complete from-scratch memory store for LLM agents.

Requires:
    pip install numpy sentence-transformers

The design:
  - EpisodicLog:   append-only JSONL file, timestamped entries.
  - SemanticStore: JSON key-value for structured facts.
  - VectorMemory:  numpy-backed cosine similarity search over embedded entries.
  - AgentMemory:   unified facade that orchestrates all three.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

# We use sentence-transformers for embeddings; swap in any provider.
# lazy-import to avoid paying the cost when not using vector retrieval.
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        # 'all-MiniLM-L6-v2' produces 384-dim vectors; small but fast.
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ─────────────────────────── Data structures ───────────────────────────

@dataclass
class MemoryEntry:
    """A single memory item stored in the vector index."""
    id: str                    # UUID or hash
    text: str                  # The content to embed and return
    metadata: dict[str, Any]   # timestamp, session_id, topic, type, etc.
    # The vector is stored separately in the numpy array; not serialised here.


# ─────────────────────────── Episodic log ──────────────────────────────

class EpisodicLog:
    """
    Append-only JSONL log of interaction episodes.
    Each entry has: timestamp, session_id, role, content, metadata.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it does not exist.
        self.path.touch(exist_ok=True)

    def append(self, session_id: str, role: str, content: str,
               metadata: dict[str, Any] | None = None) -> None:
        """Write one episode entry. Thread-safe via append mode + file lock."""
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "role": role,           # 'user', 'assistant', 'tool', 'system'
            "content": content,
            "metadata": metadata or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def tail(self, n: int = 20, session_id: str | None = None) -> list[dict]:
        """
        Return the last n entries, optionally filtered by session_id.
        We read the whole file; for production, use a proper DB.
        """
        entries = [
            json.loads(line)
            for line in self.path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        if session_id:
            entries = [e for e in entries if e["session_id"] == session_id]
        return entries[-n:]

    def load_session(self, session_id: str) -> list[dict]:
        """Return all entries for a given session, in order."""
        return [
            json.loads(line)
            for line in self.path.read_text("utf-8").splitlines()
            if line.strip() and json.loads(line).get("session_id") == session_id
        ]


# ─────────────────────────── Semantic store ────────────────────────────

class SemanticStore:
    """
    A key-value store for structured semantic facts.
    Backed by a single JSON file. Keys are canonical strings like
    'user.language' or 'project.database.engine'.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._data: dict[str, Any] = json.loads(self.path.read_text("utf-8"))
        else:
            self._data = {}

    def set(self, key: str, value: Any, source: str = "agent",
            confidence: float = 1.0) -> None:
        """
        Upsert a semantic fact. Records the update time, source, and
        confidence score alongside the value.
        """
        self._data[key] = {
            "value": value,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source,
            "confidence": confidence,
        }
        self._flush()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for key, or default if not found."""
        entry = self._data.get(key)
        return entry["value"] if entry else default

    def get_all(self) -> dict[str, Any]:
        """Return all facts as {key: value} (stripping metadata)."""
        return {k: v["value"] for k, v in self._data.items()}

    def delete(self, key: str) -> None:
        """Remove a fact."""
        self._data.pop(key, None)
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), "utf-8")

    def as_context_string(self) -> str:
        """
        Format all facts as a human-readable block suitable for injection
        into a system prompt.
        """
        if not self._data:
            return "(no semantic facts stored)"
        lines = ["## Known facts"]
        for k, v in self._data.items():
            conf_note = f" (confidence: {v['confidence']:.0%})" \
                        if v["confidence"] < 0.9 else ""
            lines.append(f"- **{k}**: {v['value']}{conf_note}")
        return "\n".join(lines)


# ─────────────────────────── Vector memory ─────────────────────────────

class VectorMemory:
    """
    Cosine-similarity vector store backed by numpy arrays.
    For production, replace the numpy backend with FAISS, Qdrant, or Chroma.

    Storage format (one directory):
      vectors.npy    — float32 array, shape [N, D]
      entries.jsonl  — one JSON entry per line (id, text, metadata)
    """

    def __init__(self, directory: str | Path, dim: int = 384):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._vectors: np.ndarray   # shape [N, D]
        self._entries: list[MemoryEntry]
        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        vec_path = self.dir / "vectors.npy"
        ent_path = self.dir / "entries.jsonl"
        if vec_path.exists() and ent_path.exists():
            self._vectors = np.load(str(vec_path))
            self._entries = [
                MemoryEntry(**json.loads(line))
                for line in ent_path.read_text("utf-8").splitlines()
                if line.strip()
            ]
        else:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            self._entries = []

    def _save(self) -> None:
        np.save(str(self.dir / "vectors.npy"), self._vectors)
        lines = [json.dumps(asdict(e)) for e in self._entries]
        (self.dir / "entries.jsonl").write_text("\n".join(lines), "utf-8")

    # ── Write ─────────────────────────────────────────────────────────

    def add(self, text: str, metadata: dict[str, Any] | None = None,
            entry_id: str | None = None) -> str:
        """
        Embed text and add it to the store. Returns the entry ID.
        If an entry with the same ID already exists, it is replaced.
        """
        import hashlib
        metadata = metadata or {}
        entry_id = entry_id or hashlib.sha256(
            (text + str(time.time())).encode()
        ).hexdigest()[:16]

        # Embed the text (normalise to unit sphere for cosine = dot product).
        embedder = _get_embedder()
        vec = embedder.encode([text], normalize_embeddings=True)[0].astype(np.float32)

        # If ID already exists, overwrite.
        existing_idx = next(
            (i for i, e in enumerate(self._entries) if e.id == entry_id), None
        )
        if existing_idx is not None:
            self._vectors[existing_idx] = vec
            self._entries[existing_idx] = MemoryEntry(
                id=entry_id, text=text, metadata=metadata
            )
        else:
            self._vectors = np.vstack([self._vectors, vec[np.newaxis]])
            self._entries.append(
                MemoryEntry(id=entry_id, text=text, metadata=metadata)
            )

        self._save()
        return entry_id

    # ── Read ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.0,
               filter_metadata: dict[str, Any] | None = None
               ) -> list[tuple[MemoryEntry, float]]:
        """
        Return the top_k most similar entries to query, with their scores.
        Optionally filter by exact metadata key-value pairs.
        """
        if len(self._entries) == 0:
            return []

        embedder = _get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)

        # Dot product = cosine similarity when both vectors are L2-normalised.
        scores = self._vectors @ q_vec  # shape [N]

        # Apply metadata filter.
        if filter_metadata:
            mask = np.array([
                all(e.metadata.get(k) == v for k, v in filter_metadata.items())
                for e in self._entries
            ])
            scores = scores * mask + (mask - 1) * 1e9  # push filtered out to -inf

        # Rank and return.
        ranked_idx = np.argsort(-scores)
        results = []
        for idx in ranked_idx[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                break
            results.append((self._entries[idx], score))
        return results

    # ── Delete ─────────────────────────────────────────────────────────

    def delete(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found."""
        idx = next(
            (i for i, e in enumerate(self._entries) if e.id == entry_id), None
        )
        if idx is None:
            return False
        self._vectors = np.delete(self._vectors, idx, axis=0)
        self._entries.pop(idx)
        self._save()
        return True

    def __len__(self) -> int:
        return len(self._entries)


# ─────────────────────────── Unified facade ────────────────────────────

class AgentMemory:
    """
    Unified memory facade for an LLM agent.

    Usage pattern:
        mem = AgentMemory(base_dir=".agent/memory", session_id="sess-001")

        # At session start: bootstrap context
        context_str = mem.bootstrap_context()

        # During session: log episodes, store facts
        mem.log("user", "I prefer TypeScript")
        mem.remember("user.language", "TypeScript")

        # After an important finding: add to vector store
        mem.store("JWT library has a bug in token refresh endpoint",
                  metadata={"topic": "auth", "type": "bug"})

        # Retrieve relevant memories
        hits = mem.recall("token refresh", top_k=3)

        # At session end: write summary
        mem.close_session(summary="Refactored auth module; found JWT bug.")
    """

    def __init__(self, base_dir: str | Path, session_id: str):
        self.base_dir = Path(base_dir)
        self.session_id = session_id
        self.episodic = EpisodicLog(self.base_dir / "episodes.jsonl")
        self.semantic  = SemanticStore(self.base_dir / "facts.json")
        self.vectors   = VectorMemory(self.base_dir / "vectors")

    # ── Write ─────────────────────────────────────────────────────────

    def log(self, role: str, content: str,
            metadata: dict[str, Any] | None = None) -> None:
        """Append an episode to the episodic log."""
        meta = {"session_id": self.session_id, **(metadata or {})}
        self.episodic.append(
            session_id=self.session_id, role=role,
            content=content, metadata=meta
        )

    def remember(self, key: str, value: Any,
                 confidence: float = 1.0) -> None:
        """Upsert a semantic fact."""
        self.semantic.set(key, value, source=self.session_id,
                          confidence=confidence)

    def store(self, text: str,
              metadata: dict[str, Any] | None = None) -> str:
        """Add a free-text memory to the vector store."""
        meta = {"session_id": self.session_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **(metadata or {})}
        return self.vectors.add(text, metadata=meta)

    # ── Read ──────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 5,
               min_score: float = 0.3) -> list[tuple[str, float]]:
        """
        Query the vector store. Returns list of (text, score) tuples.
        Score is cosine similarity in [0, 1] (since vectors are normalised).
        """
        hits = self.vectors.search(query, top_k=top_k, min_score=min_score)
        return [(entry.text, score) for entry, score in hits]

    def bootstrap_context(self, top_k_memories: int = 5) -> str:
        """
        Build the context injection string for session start.
        Combines semantic facts + recent episodes + vector-recalled memories.
        """
        parts = []

        # 1. Semantic facts (always include all of them if small).
        facts = self.semantic.as_context_string()
        parts.append(facts)

        # 2. Recent episodic entries (last 5 from this session).
        recent = self.episodic.tail(n=5, session_id=self.session_id)
        if recent:
            lines = ["## Recent episode summary"]
            for e in recent:
                lines.append(f"[{e['iso_time']}] {e['role']}: {e['content'][:200]}")
            parts.append("\n".join(lines))

        # 3. Cross-session vector memories (not filtered to session).
        hits = self.vectors.search(
            "recent tasks and important findings",
            top_k=top_k_memories,
            min_score=0.25,
        )
        if hits:
            lines = ["## Recalled memories"]
            for entry, score in hits:
                lines.append(f"- [{score:.2f}] {entry.text}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def close_session(self, summary: str) -> None:
        """
        Write a session-end summary to both the episodic log and vector store.
        Call this before clearing the context window.
        """
        self.log("system", f"[SESSION SUMMARY] {summary}",
                 metadata={"type": "session_summary"})
        self.store(summary,
                   metadata={"type": "session_summary",
                             "session_id": self.session_id})
```

This gives you an immediately usable memory layer. The vector backend is pure numpy — swap in `faiss.IndexFlatIP` or `qdrant_client` for production with millions of entries.

---

## 8.5.8 Worked Example: Memory Budget and Retrieval Magnitudes

!!! example "Worked example: sizing an agent memory system"

    Suppose you are building an agent that handles long-running software projects.
    You want to understand the memory budget and retrieval latency at scale.

    **Configuration:**
    - Embedding model: `all-MiniLM-L6-v2`, output dimension $D = 384$, float32.
    - Target store size: $N = 100{,}000$ memories (about 3 years of daily use at ~100 entries/day).
    - Context window budget for injected memories: 2 000 tokens, roughly 1 500 words.

    **Storage cost:**
    $$
    \text{vector storage} = N \times D \times 4 \text{ bytes}
    = 100{,}000 \times 384 \times 4 = 153.6 \text{ MB}
    $$

    That is well within RAM for a desktop machine. A numpy brute-force search over
    100 k vectors (one dot-product scan) takes on the order of 5–15 ms on a modern
    CPU — acceptable for a human-in-the-loop agent but worth caching if the agent
    calls recall() in a tight loop. Switching to a FAISS HNSW index drops search to
    under 1 ms at the same recall@10 level.

    **Text storage:**
    If each memory entry averages 100 tokens, that is $100{,}000 \times 100 = 10 \text{ M tokens}$.
    At 4 bytes per token in UTF-8: roughly **40 MB** for all text. Well within a
    SQLite file or a directory of JSONL files.

    **Injection budget:**
    With 2 000 tokens injected and an average entry length of 100 tokens, we can
    inject at most 20 memories per call. In practice, top-$k = 5$ to $10$ is enough
    to avoid diluting the context with low-relevance entries. At a minimum
    cosine-similarity threshold of 0.4, typical agents inject 3–6 entries per turn.

    **Cost implication:**
    If the model charges on the order of \$1.50 per million input tokens, injecting
    5 entries × 100 tokens = 500 extra tokens per call costs $\approx$ \$0.00075 per
    call. Over 10 000 agent calls, that is **\$7.50** — negligible against the
    productivity gain from accurate memory recall.


---

## 8.5.9 The Reflect Step: Distilling Episodic to Semantic

The reflect step is a scheduled LLM call over recent episodic entries. Here is the pattern as a Python function that fits into the `AgentMemory` facade:

```python
def reflect(
    mem: AgentMemory,
    llm_call: callable,          # func(prompt: str) -> str
    n_recent_episodes: int = 30,
) -> list[tuple[str, str]]:
    """
    Run a reflection pass: read recent episodes, ask the LLM to distil
    facts, and write them back to the semantic store.

    Returns a list of (key, value) pairs that were upserted.
    """
    episodes = mem.episodic.tail(n=n_recent_episodes,
                                 session_id=mem.session_id)
    if not episodes:
        return []

    # Format episodes as a readable transcript.
    transcript = "\n".join(
        f"[{e['iso_time']}] {e['role']}: {e['content']}"
        for e in episodes
    )

    prompt = f"""You are reviewing the following conversation transcript.
Extract a list of lasting facts, preferences, or decisions that should
be remembered for future sessions. Format your response as JSON:
[
  {{"key": "canonical.key", "value": "concise fact", "confidence": 0.9}},
  ...
]
Only include facts that are stable across sessions, not one-off details.

TRANSCRIPT:
{transcript}

JSON output:"""

    raw = llm_call(prompt)

    # Robust parse: look for the JSON array in the response.
    import re, json as _json
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []

    facts = _json.loads(match.group())
    updated = []
    for fact in facts:
        key   = fact.get("key", "").strip()
        value = fact.get("value", "").strip()
        conf  = float(fact.get("confidence", 0.8))
        if key and value:
            mem.remember(key, value, confidence=conf)
            updated.append((key, value))

    return updated
```

This is the core of the Generative Agents reflection loop made concrete. A typical agent running on a personal software project might call `reflect()` at the end of each work session, discovering and persisting things like "user prefers composable functions over class hierarchies" from 30 episodes of code review feedback.

---

!!! interview "Interview Corner"

    **Q:** An LLM agent is running a multi-day research task. The user comes back after two days and the context window is empty. How do you design the memory system to give the agent continuity across sessions?

    **A:** The solution requires four pieces working together.

    First, an **episodic log**: every agent turn is appended to a durable append-only store (JSONL file or DB) tagged with a session ID. This gives us a ground-truth record.

    Second, a **semantic store**: at the end of each session, a reflect call distils the most important facts (task state, decisions made, user preferences) into a key-value or vector store. These are stable facts, not raw transcripts.

    Third, a **bootstrap read**: at the start of every new session, the harness queries both stores and prepends a compact summary to the system prompt — typically the last session summary plus top-$k$ relevant vector memories. This is bounded in size regardless of how long the project has been running.

    Fourth, a **session-end write**: before clearing context, the agent (or the harness) writes a structured session summary: what was in progress, what was decided, what the next steps are. This is the most important single artifact for continuity.

    The key insight is that you never rely on the model's weights to carry state between sessions — that is fine-tuning, which is too slow. Instead, you externalise state explicitly and inject it deterministically at session start.

---

!!! warning "Common pitfall: writing everything verbatim"

    Many teams start by logging every message verbatim to a vector store and
    retrieving the top-20 similar entries. This produces two failure modes.
    First, recent repetitive messages (boilerplate system prompt text, repeated
    clarification questions) crowd out genuinely informative memories because
    their text appears many times and therefore scores high on any query. Second,
    injecting 20 verbose entries at once consumes 2 000+ tokens of context budget.
    Fix: (a) filter what you write to the vector store — only write entries with
    genuine informational value; (b) cap injection at top-$k = 5$ with a
    minimum cosine-similarity threshold; (c) run a deduplication pass that
    merges near-identical memories.

---

!!! tip "Practitioner tip: use structured compaction before the context is full"

    Do not wait until the context window is 100% full to compact. Set a threshold
    at 70–80% of capacity and trigger a structured compaction pass that serialises
    the current task state to the semantic store before any information is lost.
    The compacted JSON blob is then prepended to the new context, giving the agent
    continuity with a deterministic (not lossy free-text) representation.

---

## Summary

Memory is the difference between a stateless chatbot and a persistent agent. We have covered the full stack:

- **Short-term vs long-term memory** — the context window is the working memory; external stores are the long-term memory.
- **Episodic vs semantic** — timestamped events vs stable facts; do not mix them indiscriminately.
- **Store types** — vector stores for fuzzy similarity, key-value for deterministic retrieval, graphs for relational knowledge, file systems for human-readable persistence.
- **Write / read / reflect** — three operations that every memory-capable agent must implement.
- **Compaction** — you must proactively manage context size or the window fills and information is lost.
- **Session persistence** — bootstrap read, in-session writes, graceful shutdown, and periodic reflection.

The code in this chapter gives you a concrete starting point. In the next chapter we look at how the [Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html) standardises the interface through which agents access both memory and tools. For retrieval-heavy applications, the detailed treatment of ANN indexes and hybrid search in [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html) and [Advanced RAG: GraphRAG, Agentic RAG & Long-Context vs RAG](../09-rag-retrieval/05-advanced-rag.html) will be essential.

---

!!! key "Key Takeaways"

    - The context window is short-term working memory; anything that must survive a context clear or a session boundary needs an external store.
    - Episodic memory records events with time context; semantic memory records stable facts. Keep them separate to avoid retrieval noise.
    - The three memory operations are **write** (commit to store), **read** (retrieve into context), and **reflect** (distil episodes into facts) — reflect is the highest-value operation and the most often omitted.
    - Memory compaction must happen *before* the context is full; a structured JSON schema compacts more reliably than free-text summaries.
    - Session persistence requires four steps: stable session ID, bootstrap read at start, graceful shutdown write at end, and periodic cross-session reflection.
    - Vector stores excel at fuzzy similarity retrieval but fail on exact-match queries; supplement with a structured key-value store for deterministic fact lookup.
    - File-as-memory (structured directories of markdown/JSON) is a production-proven pattern that requires no external database and is human-auditable.
    - Injection budget matters: at top-$k = 5$ and 100 tokens per entry, 500 extra input tokens per call costs under \$0.001 at typical API prices — memory retrieval is cheap relative to its value.
    - The reflect step should be triggered periodically (every $N$ episodes) and at session boundaries; it is the mechanism by which short-term experience becomes long-term knowledge.

---

!!! sota "State of the Art & Resources (2026)"
    Agent memory is an active research frontier: production systems now combine episodic logs, semantic key-value stores, and temporally-aware knowledge graphs, while recent surveys (2026) have formalised the write–manage–read loop and introduced multi-session agentic benchmarks that go well beyond static retrieval tests.

    **Foundational work**

    - [Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (2023)](https://arxiv.org/abs/2304.03442) — introduced the episodic-log + reflection + recency/importance/relevance retrieval architecture; the landmark paper that defined modern LLM agent memory.
    - [Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2023)](https://arxiv.org/abs/2310.08560) — framed memory management as OS-style virtual paging, introducing the virtual-context-manager pattern for infinite-length agent conversations.
    - [Sumers et al., *Cognitive Architectures for Language Agents* (CoALA, 2023)](https://arxiv.org/abs/2309.02427) — comprehensive taxonomy of agent memory across working/long-term stores and internal/external action spaces; useful conceptual framework for system design.

    **Recent advances (2023–2026)**

    - [Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* (2025)](https://arxiv.org/abs/2501.13956) — Graphiti-powered temporal KG that retains historical context across updates; outperforms MemGPT on Deep Memory Retrieval benchmarks.
    - [Xu et al., *A-MEM: Agentic Memory for LLM Agents* (2025)](https://arxiv.org/abs/2502.12110) — Zettelkasten-inspired dynamic indexing that creates interconnected memory networks; accepted NeurIPS 2025.
    - [Du, *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers* (2026)](https://arxiv.org/abs/2603.07670) — survey covering 2022–2026 work; formalises the write–manage–read loop and five mechanism families from context compression to policy-learned management.

    **Open-source & tools**

    - [mem0ai/mem0](https://github.com/mem0ai/mem0) — 57k-star "universal memory layer" combining vector, graph, and key-value stores; production-ready Python/TypeScript SDK.
    - [letta-ai/letta](https://github.com/letta-ai/letta) — formerly MemGPT; stateful-agent platform with built-in virtual context management and a REST API for persistent agent services.
    - [topoteretes/cognee](https://github.com/topoteretes/cognee) — open-source memory control plane with remember/recall/forget/improve operations backed by embeddings and knowledge graphs.

    **Go deeper**

    - [Lawson, *A Practical Guide to Memory for Autonomous LLM Agents* (Towards Data Science, 2026)](https://towardsdatascience.com/a-practical-guide-to-memory-for-autonomous-llm-agents/) — practitioner-oriented walkthrough of the four temporal memory scopes, five mechanism families, and common failure modes drawn from real distributed-agent deployments.

## Further Reading

- **Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," 2023** — introduced the episodic + reflection + retrieval architecture for persistent agent behaviour; the landmark paper on LLM memory systems.
- **Zep — the Memory Layer for AI** (getzep.com) — production system implementing temporal knowledge graphs and semantic extraction; useful reference architecture.
- **MemGPT: Towards LLMs as Operating Systems, Packer et al., 2023** — framed memory management as an OS paging problem; introduced the virtual-context-manager pattern for infinite-length conversations.
- **LangGraph (LangChain)** — open-source agent framework with first-class support for persistent state stores across agent steps; good reference implementation of session persistence.
- **Cognee** (topoteretes/cognee on GitHub) — open-source memory layer with knowledge-graph extraction from unstructured text; shows a practical graph-memory pipeline.
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, Lewis et al., 2020** — the foundational RAG paper; motivates why external retrieval beats storing everything in weights.

---

## Exercises

**1.** For each of the following memory items, decide whether it belongs in the **episodic** store or the **semantic** store, and give a one-line justification. Then explain, using the argument in section 8.5.2, why storing (a) and (c) together in a single undifferentiated vector store degrades retrieval quality.

  - (a) "2025-11-04 14:12: the user asked me to refactor the auth module and I found a bug in the JWT refresh endpoint."
  - (b) "The production database engine is PostgreSQL 15."
  - (c) "2025-11-05 09:30: reran the integration suite; 3 tests still failing on token refresh."
  - (d) "The user prefers British English spelling."

??? note "Solution"
    Classification:

    - (a) **Episodic.** It is a timestamped, situated record of a specific interaction ("on this date I did X"). It is append-only and queried with temporal/situational cues.
    - (b) **Semantic.** A stable fact that should hold across sessions; updated by overwrite when the engine changes. Canonical key e.g. `project.database.engine`.
    - (c) **Episodic.** Another timestamped event tied to a particular run.
    - (d) **Semantic.** A stable user preference; canonical key e.g. `user.language`, overwritten only on explicit correction.

    Why mixing (a) and (c) with semantic facts hurts retrieval: the chapter's key insight (8.5.2) is that episodic entries carry date-specific noise. Both (a) and (c) contain the token cluster "token refresh," so a *general-knowledge* query such as "what do I know about the auth module?" will surface these two dated event records with high cosine similarity, crowding out (or diluting) the stable fact you actually want. Episodic entries also grow monotonically, so over time date-specific noise increasingly dominates any content-based query. Keeping episodic events in an append-only log and distilling stable facts into a separate semantic store (via the reflect step) keeps general-knowledge queries clean.

**2.** You call `mem.recall(query, top_k=3, min_score=0.3)`. The query embeds (after L2 normalisation) to $q = [0.6,\ 0.8,\ 0.0]$. The vector store holds four already-normalised memory vectors:

  - $m_1 = [0.6,\ 0.8,\ 0.0]$
  - $m_2 = [0.0,\ 1.0,\ 0.0]$
  - $m_3 = [1.0,\ 0.0,\ 0.0]$
  - $m_4 = [0.8,\ -0.6,\ 0.0]$

Compute each cosine similarity by hand, then state exactly which entries `recall` returns and in what order. (Recall from the code that `search` sorts by descending score and *breaks* as soon as a score falls below `min_score`.)

??? note "Solution"
    Because every vector is unit-length, cosine similarity equals the plain dot product with $q$ (this is exactly why the chapter's `add`/`search` pass `normalize_embeddings=True`):

    $$
    \begin{aligned}
    \cos(q, m_1) &= 0.6\cdot0.6 + 0.8\cdot0.8 + 0 = 0.36 + 0.64 = 1.00 \\
    \cos(q, m_2) &= 0.6\cdot0.0 + 0.8\cdot1.0 + 0 = 0.80 \\
    \cos(q, m_3) &= 0.6\cdot1.0 + 0.8\cdot0.0 + 0 = 0.60 \\
    \cos(q, m_4) &= 0.6\cdot0.8 + 0.8\cdot(-0.6) + 0 = 0.48 - 0.48 = 0.00
    \end{aligned}
    $$

    Descending order: $m_1(1.00),\ m_2(0.80),\ m_3(0.60),\ m_4(0.00)$. With `top_k=3` we look at the first three; all three exceed `min_score=0.3`, so `recall` returns:

    ```text
    [(m1_text, 1.00), (m2_text, 0.80), (m3_text, 0.60)]
    ```

    $m_4$ is excluded on two counts: it is beyond `top_k=3`, and its score $0.00 < 0.3$ would trigger the `break` anyway. Note the ordering is by relevance, highest first.

**3.** You are sizing a memory store with a **1024-dimensional** float32 embedding model and a target of $N = 250{,}000$ memories, each averaging **80 tokens** of text.

  - (a) Compute the raw vector storage in bytes and MB/GB.
  - (b) Compute the total text token count and the UTF-8 text storage at 4 bytes/token.
  - (c) If the injection budget is 1500 tokens per call, what is the maximum number of 80-token entries you could inject, and how does that compare to the chapter's recommended `top_k`?
  - (d) At \$2.00 per million input tokens, what does injecting `top_k = 5` entries (80 tokens each) cost per call, and over 50 000 calls?

??? note "Solution"
    (a) Vector storage (formula from 8.5.8, $N \times D \times 4$ bytes):
    $$
    250{,}000 \times 1024 \times 4 = 250{,}000 \times 4096 = 1{,}024{,}000{,}000 \text{ bytes} = 1024 \text{ MB} \approx 1.02 \text{ GB}.
    $$
    Large enough that you would prefer an on-disk ANN index (FAISS/Qdrant) over holding one big numpy array in RAM.

    (b) Token count: $250{,}000 \times 80 = 20{,}000{,}000 = 20$ M tokens. At 4 bytes/token:
    $$
    20{,}000{,}000 \times 4 = 80{,}000{,}000 \text{ bytes} = 80 \text{ MB}.
    $$
    Comfortably a single SQLite file or a directory of JSONL.

    (c) Maximum entries $= \lfloor 1500 / 80 \rfloor = 18$. The chapter recommends a conservative `top_k` of 3-5 (up to ~10) with a relevance threshold, so you should inject far fewer than the 18 the budget technically allows — packing all 18 in would dilute the context with low-relevance entries (the "writing everything verbatim" pitfall).

    (d) Per call: $5 \times 80 = 400$ input tokens. Cost $= 400 / 1{,}000{,}000 \times \$2.00 = \$0.0008$ per call. Over 50 000 calls: $50{,}000 \times \$0.0008 = \$40$. Negligible against the productivity gain from accurate recall — the same conclusion the chapter reaches in 8.5.8.

**4.** The "writing everything verbatim" pitfall recommends fix (c): *run a deduplication pass that merges near-identical memories.* Implement a `deduplicate(self, threshold: float = 0.95) -> int` method on `VectorMemory` that removes near-duplicate entries (keeping the first member of each duplicate group) and returns the number removed. Exploit the fact that the stored vectors are already L2-normalised. Explain why that normalisation makes the check a single dot product.

??? note "Solution"
    Because `add` stores every vector with `normalize_embeddings=True`, each row of `self._vectors` is unit length, so the cosine similarity between any two stored vectors is just their dot product `v @ w`. We greedily keep entries whose vector is not within `threshold` cosine of any already-kept vector:

    ```python
    def deduplicate(self, threshold: float = 0.95) -> int:
        """
        Remove near-duplicate entries. Keeps the first member of each
        near-duplicate group; deletes the rest. Returns the number removed.
        Relies on stored vectors being L2-normalised (cosine == dot product).
        """
        if len(self._entries) < 2:
            return 0

        kept_entries: list[MemoryEntry] = []
        kept_vecs: list[np.ndarray] = []
        removed = 0

        for i, entry in enumerate(self._entries):
            vec = self._vectors[i]
            # cosine == dot product because both operands are unit vectors
            is_dup = any(float(vec @ kv) >= threshold for kv in kept_vecs)
            if is_dup:
                removed += 1
            else:
                kept_entries.append(entry)
                kept_vecs.append(vec)

        if removed:
            self._entries = kept_entries
            self._vectors = (
                np.vstack(kept_vecs) if kept_vecs
                else np.empty((0, self.dim), dtype=np.float32)
            )
            self._save()
        return removed
    ```

    Why a single dot product suffices: cosine similarity is $\cos(u,v) = \frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}$. When $\lVert u\rVert = \lVert v\rVert = 1$ the denominator is 1, so $\cos(u,v) = u\cdot v$ exactly — no per-pair norm computation is needed, which is the same shortcut `search` uses (`scores = self._vectors @ q_vec`). The pass is $O(N \cdot K)$ where $K$ is the number of kept (unique) entries; for large stores you would replace the inner loop with a batched matmul against the kept matrix.

**5.** Implement the *structured compaction* trigger described in the practitioner tip and section 8.5.5.3. Write a function `compact_if_needed(mem, transcript_tokens, llm_call, ctx_window=128_000, threshold=0.75)` that: (a) does nothing and returns `None` while the transcript is below `threshold` of the context window; (b) otherwise asks the LLM to serialise current state into the structured JSON schema from 8.5.5.3, parses it robustly, persists the fields to the semantic store, and returns the parsed dict. Why is serialising into this schema more reliable than a free-text summary?

??? note "Solution"
    The tip says: trigger at 70-80% of capacity (here `threshold=0.75`), before information is lost, and produce a deterministic structured blob rather than a lossy free-text summary. The function mirrors the robust-parse style of the chapter's `reflect`:

    ```python
    import re, json as _json

    def compact_if_needed(mem, transcript_tokens, llm_call,
                          ctx_window: int = 128_000,
                          threshold: float = 0.75):
        """
        Trigger a structured compaction pass once the in-context transcript
        crosses `threshold` of the context window. Persists the compacted
        state to the semantic store and returns the parsed dict (or None
        if compaction was not needed).
        """
        if transcript_tokens < threshold * ctx_window:
            return None  # still room; do not compact yet

        prompt = f"""Serialise the current session state into EXACTLY this JSON schema:
    {{
      "task_state": "one-line status",
      "decisions_made": ["..."],
      "open_questions": ["..."],
      "user_preferences": {{"key": "value"}},
      "next_steps": ["..."]
    }}
    Fill every field from the conversation so far. Output only the JSON object.

    JSON output:"""

        raw = llm_call(prompt)

        # Robust parse: grab the first {...} block.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        state = _json.loads(match.group())

        # Persist each field to the semantic store under canonical keys.
        mem.remember("compaction.task_state", state.get("task_state", ""))
        mem.remember("compaction.decisions_made", state.get("decisions_made", []))
        mem.remember("compaction.open_questions", state.get("open_questions", []))
        mem.remember("compaction.next_steps", state.get("next_steps", []))
        for k, v in (state.get("user_preferences") or {}).items():
            mem.remember(f"user.{k}", v)

        return state
    ```

    Usage: the harness prepends `mem.semantic.as_context_string()` (which now contains the compacted `compaction.*` and `user.*` keys) to the fresh context, giving deterministic continuity. Why the schema beats free text: as 8.5.5.3 notes, the model is *filling in a fixed set of slots* rather than generating prose freely, so it is far less prone to hallucination or to silently dropping a field; the result is machine-parseable, so downstream reads (`state["next_steps"]`) are deterministic instead of requiring the agent to re-read and re-interpret a paragraph. Because the state is written to the semantic store *before* the old prefix is discarded, no high-value fact is lost in the compaction — exactly the ordering the chapter insists on ("write high-value facts to the external store *before* compacting").
