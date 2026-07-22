"""
Runnable test for content/08-agents-harness/05-agent-memory.md

Tests block #2 (the full agent_memory.py implementation: EpisodicLog,
SemanticStore, VectorMemory, AgentMemory) and block #3 (the reflect()
function) by concatenating the chapter's code verbatim and exercising it
end-to-end on disk, in a temp directory, with no network access.

Block #0 and #1 are non-Python (prose/JSON) and are not tested here.

The only network-touching piece of block #2 is `_get_embedder()`, which
lazily imports and downloads `sentence-transformers`'s all-MiniLM-L6-v2
model. We replace that function with a deterministic, seeded, in-memory
fake embedder (unit-normalised random vectors keyed by a hash of the
input text) so that all the *book's own logic* — cosine search, top-k
ranking, metadata filtering, upsert-by-id, delete, persistence/reload —
actually executes, without ever importing sentence_transformers or
touching the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile

# =====================================================================
# Block #2 (verbatim from the chapter, ~line 200-608)
# =====================================================================

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


# =====================================================================
# Block #3 (verbatim from the chapter, ~line 662-718)
# =====================================================================

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


# =====================================================================
# Test glue (not from the book): deterministic offline embedder.
#
# The book's `_get_embedder()` lazily imports sentence-transformers and
# downloads 'all-MiniLM-L6-v2' from the network. We replace it with a
# seeded, deterministic fake so VectorMemory's own logic (cosine search,
# top-k, metadata filtering, upsert-by-id, delete, save/load) runs for
# real, with zero network access. Because VectorMemory calls the module-
# level name `_get_embedder()` at call time (Python late-binds globals),
# reassigning it here is sufficient — no monkeypatch library needed.
# =====================================================================

class _FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer.encode()."""

    def encode(self, texts, normalize_embeddings=True):
        dim = 384
        vecs = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            v = rng.randn(dim).astype(np.float32)
            if normalize_embeddings:
                v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


def _get_embedder():  # overrides the book's version for this test run
    global _embedder
    if _embedder is None:
        _embedder = _FakeEmbedder()
    return _embedder


# =====================================================================
# Exercise the assembled chapter code end-to-end.
# =====================================================================

def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="agent_memory_test_")
    try:
        mem_dir = os.path.join(tmpdir, "memory")
        mem = AgentMemory(base_dir=mem_dir, session_id="sess-test-1")

        # --- EpisodicLog via AgentMemory.log ---
        mem.log("user", "I prefer TypeScript over JavaScript for new projects.")
        mem.log("assistant", "Noted, I will use TypeScript going forward.")
        mem.log("tool", "ran `npm test`: 12 passed, 0 failed")

        # --- SemanticStore via AgentMemory.remember ---
        mem.remember("user.language", "TypeScript")
        mem.remember("project.database.engine", "PostgreSQL", confidence=0.7)

        facts = mem.semantic.get_all()
        assert facts["user.language"] == "TypeScript"
        assert facts["project.database.engine"] == "PostgreSQL"

        facts_str = mem.semantic.as_context_string()
        assert "## Known facts" in facts_str
        assert "user.language" in facts_str
        assert "confidence: 70%" in facts_str  # low-confidence fact gets annotated

        # --- VectorMemory via AgentMemory.store / recall ---
        id_bug = mem.store(
            "JWT library has a bug in token refresh endpoint",
            metadata={"topic": "auth", "type": "bug"},
        )
        id_infra = mem.store(
            "Migrated CI pipeline to GitHub Actions",
            metadata={"topic": "infra", "type": "note"},
        )
        assert len(mem.vectors) == 2

        # Exact-text query should retrieve itself with the top score (~1.0,
        # since our fake embedder is deterministic: same text -> same vector).
        hits = mem.recall(
            "JWT library has a bug in token refresh endpoint",
            top_k=2, min_score=0.0,
        )
        assert hits, "expected at least one recalled memory"
        top_text, top_score = hits[0]
        assert top_text == "JWT library has a bug in token refresh endpoint"
        assert top_score > 0.99, f"expected near-1.0 self-similarity, got {top_score}"

        # Metadata-filtered search on the underlying VectorMemory.
        filtered = mem.vectors.search("bug", top_k=5, filter_metadata={"topic": "auth"})
        assert len(filtered) == 1
        assert filtered[0][0].id == id_bug

        # Delete.
        assert mem.vectors.delete(id_infra) is True
        assert len(mem.vectors) == 1
        assert mem.vectors.delete("does-not-exist") is False

        # --- EpisodicLog.tail / load_session ---
        tail = mem.episodic.tail(n=2, session_id="sess-test-1")
        assert len(tail) == 2
        session_entries = mem.episodic.load_session("sess-test-1")
        assert len(session_entries) == 3

        # --- AgentMemory.bootstrap_context ---
        ctx = mem.bootstrap_context(top_k_memories=3)
        assert "## Known facts" in ctx
        assert "## Recent episode summary" in ctx

        # --- AgentMemory.close_session ---
        mem.close_session(summary="Investigated JWT bug; decided to use TypeScript.")
        session_entries_after = mem.episodic.load_session("sess-test-1")
        assert any(
            e["metadata"].get("type") == "session_summary"
            for e in session_entries_after
        )
        assert len(mem.vectors) == 2  # summary text got added back to the vector store

        # --- Persistence: reload from disk into a fresh AgentMemory ---
        mem2 = AgentMemory(base_dir=mem_dir, session_id="sess-test-1")
        assert len(mem2.vectors) == 2
        assert mem2.semantic.get("user.language") == "TypeScript"

        # --- Block #3: reflect() ---
        canned_llm_response = json.dumps([
            {"key": "user.language", "value": "TypeScript", "confidence": 0.95},
            {"key": "project.ci", "value": "GitHub Actions", "confidence": 0.8},
        ])

        def fake_llm_call(prompt: str) -> str:
            assert "TRANSCRIPT" in prompt
            assert "JSON output" in prompt
            return canned_llm_response

        updated = reflect(mem2, fake_llm_call, n_recent_episodes=10)
        assert ("user.language", "TypeScript") in updated
        assert ("project.ci", "GitHub Actions") in updated
        assert mem2.semantic.get("project.ci") == "GitHub Actions"

        # reflect() on a session with no episodes returns [] (no LLM call needed).
        mem3 = AgentMemory(base_dir=mem_dir, session_id="sess-empty")
        assert reflect(mem3, fake_llm_call) == []

        print("OK: block #2 (AgentMemory: EpisodicLog/SemanticStore/VectorMemory) "
              "and block #3 (reflect) both executed successfully.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
