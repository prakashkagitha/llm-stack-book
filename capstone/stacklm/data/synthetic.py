"""Offline synthetic corpus (Ch. 14.2). In production these entries stream from
HuggingFace (FineWeb-Edu, Cosmopedia, StarCoder, FineMath); for hermetic CI we
generate a tiny corpus in-process with no network. `stream_source` prefers the
real dataset when available but always falls back to `synthetic_corpus`.
"""
import hashlib
import random
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class DataMixEntry:
    name: str      # short id, e.g. "fineweb_edu"
    hf_path: str   # HuggingFace dataset id used in production
    weight: float  # fraction of the 20B-token budget
    domain: str    # "web" | "synthetic" | "code" | "math" -- routes the filter


STACK100M_MIX = [
    DataMixEntry("fineweb_edu",   "HuggingFaceFW/fineweb-edu", 0.70, "web"),
    DataMixEntry("cosmopedia_v2", "HuggingFaceTB/cosmopedia",  0.15, "synthetic"),
    DataMixEntry("starcoder",     "bigcode/starcoderdata",     0.10, "code"),
    DataMixEntry("finemath",      "HuggingFaceTB/finemath",    0.05, "math"),
]

TOTAL_TOKEN_BUDGET = 20_000_000_000  # ~20B tokens, ~200 tok/param (PLAN.md sec. 2)

_dup_cache: dict = {}

_VOCAB = {
    "web": ["photosynthesis", "converts", "sunlight", "into", "chemical", "energy",
            "plants", "use", "carbon", "dioxide", "and", "water", "to", "produce",
            "glucose", "the", "process", "occurs", "inside", "chloroplasts"],
    "synthetic": ["chapter", "one", "introduces", "the", "concept", "of", "gravity",
                  "as", "a", "force", "that", "attracts", "objects", "with", "mass",
                  "toward", "each", "other", "consider", "an", "example"],
    "code": ["def", "compute", "(", "x", ")", ":", "return", "x", "*", "x", "+", "1",
             "for", "i", "in", "range", "(", "10", ")", ":", "print", "(", "i", ")"],
    "math": ["let", "f", "(", "x", ")", "=", "x^2", "then", "the", "derivative",
             "is", "2x", "solve", "for", "x", "when", "3x", "+", "5", "=", "20"],
}


def synthetic_corpus(entry: DataMixEntry, n_docs: int = 2000) -> Iterator[dict]:
    seed = int(hashlib.blake2b(entry.name.encode(), digest_size=4).hexdigest(), 16)
    rng = random.Random(seed)
    vocab = _VOCAB[entry.domain]
    for i in range(n_docs):
        if entry.domain in _dup_cache and i % 97 == 0:
            text = _dup_cache[entry.domain]                        # exact duplicate
        elif entry.domain in _dup_cache and i % 53 == 0:
            base = _dup_cache[entry.domain].split()                # near-duplicate
            for _ in range(max(1, len(base) // 20)):
                base[rng.randrange(len(base))] = rng.choice(vocab)
            text = " ".join(base)
        else:
            n_words = rng.randint(60, 400)
            text = " ".join(rng.choice(vocab) for _ in range(n_words)) + "."
            _dup_cache[entry.domain] = text
        doc_id = hashlib.sha1(f"{entry.name}-{i}".encode()).hexdigest()[:12]
        yield {"text": text, "source": entry.name, "domain": entry.domain, "doc_id": doc_id}


def stream_source(entry: DataMixEntry, offline: bool = False) -> Iterator[dict]:
    if not offline:
        try:
            from datasets import load_dataset  # heavy optional dependency; not in CI
            ds = load_dataset(entry.hf_path, split="train", streaming=True)
            for row in ds:
                text = row.get("text", "")
                if text:
                    yield {"text": text, "source": entry.name, "domain": entry.domain}
            return
        except Exception:
            pass
    yield from synthetic_corpus(entry)


def synthetic_text_sample(n_docs_per_source: int = 300) -> str:
    """One big string sampled across all domains -- used to train the toy tokenizer."""
    parts = []
    for entry in STACK100M_MIX:
        for doc in synthetic_corpus(entry, n_docs=n_docs_per_source):
            parts.append(doc["text"])
    return "\n".join(parts)
