"""Source registry + offline synthetic corpus (Ch. 14.2).

In production every entry streams from the HuggingFace Hub (FineWeb-Edu,
Cosmopedia v2, StarCoder, FineMath). Each entry carries the *full* coordinates
needed to load it: repo id, config name, optional data_dir, and the column the
text actually lives in -- these differ per dataset, and getting them wrong is the
most common way a data pipeline silently produces zero documents.

For hermetic CI we generate a tiny corpus in-process with no network.
"""
import hashlib
import random
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class DataMixEntry:
    name: str                      # short id, e.g. "fineweb_edu"
    hf_path: str                   # HuggingFace repo id
    weight: float                  # fraction of the 20B-token budget
    domain: str                    # "web"|"synthetic"|"code"|"math" -- routes the filter
    hf_config: Optional[str] = None    # config/subset name (`name=` in load_dataset)
    hf_data_dir: Optional[str] = None  # directory-sharded repos (starcoderdata)
    text_column: str = "text"          # NOT always "text" (starcoderdata: "content")
    revision: str = "main"             # pin to a commit sha for a reproducible corpus
    gated: bool = False                # requires huggingface_hub.login()


# The exact mix fixed in capstone/PLAN.md sec. 2. Weights sum to 1.0.
STACK100M_MIX = [
    DataMixEntry("fineweb_edu", "HuggingFaceFW/fineweb-edu", 0.70, "web",
                 hf_config="sample-100BT"),
    DataMixEntry("cosmopedia_v2", "HuggingFaceTB/smollm-corpus", 0.15, "synthetic",
                 hf_config="cosmopedia-v2"),
    DataMixEntry("starcoder", "bigcode/starcoderdata", 0.10, "code",
                 hf_data_dir="python", text_column="content", gated=True),
    DataMixEntry("finemath", "HuggingFaceTB/finemath", 0.05, "math",
                 hf_config="finemath-4plus"),
]

TOTAL_TOKEN_BUDGET = 20_000_000_000  # ~20B tokens, ~200 tok/param (PLAN.md sec. 2)


def load_hf_stream(entry: DataMixEntry):
    """Open one source as a streaming `datasets.IterableDataset`.

    Raises rather than returning an empty stream: a mistyped config or column is
    a bug, and a pipeline that silently yields nothing is the worst possible
    failure mode -- you find out 20 GPU-hours later.
    """
    from datasets import load_dataset  # heavy optional dependency; not in CI

    ds = load_dataset(
        entry.hf_path,
        name=entry.hf_config,
        data_dir=entry.hf_data_dir,
        split="train",
        streaming=True,
        revision=entry.revision,
    )
    cols = getattr(ds, "column_names", None)
    if cols is not None and entry.text_column not in cols:
        raise KeyError(
            f"{entry.name}: text column {entry.text_column!r} not in {cols}. "
            "Most HF text corpora use 'text', but bigcode/starcoderdata "
            "uses 'content'."
        )
    return ds


def stream_hf(entry: DataMixEntry, probe: int = 8) -> Iterator[dict]:
    """Yield normalized {"text","source","domain"} docs from the real dataset.
    Asserts that the first `probe` rows are not all empty, so a misconfigured
    source fails loudly instead of contributing zero tokens."""
    ds = load_hf_stream(entry)
    n_seen = n_nonempty = 0
    for row in ds:
        text = row.get(entry.text_column) or ""
        if n_seen < probe:
            n_seen += 1
            n_nonempty += bool(text)
            if n_seen == probe and n_nonempty == 0:
                raise ValueError(
                    f"{entry.name}: first {probe} rows had an empty "
                    f"{entry.text_column!r} field -- wrong column or wrong config?"
                )
        if text:
            yield {"text": text, "source": entry.name, "domain": entry.domain}


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
    """Deterministic in-process corpus with injected exact (every 97th) and near
    (every 53rd) duplicates, so the dedup stages have something real to catch."""
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


def stream_source(entry: DataMixEntry, offline: bool = False,
                  n_docs: int = 2000) -> Iterator[dict]:
    """Real HF stream when `offline=False`, synthetic fallback otherwise.

    Only the *opening* of the stream is guarded: if `datasets` is missing or the
    Hub is unreachable we fall back to the synthetic corpus, but once the stream
    is open, errors propagate. Swallowing mid-stream exceptions would silently
    truncate the corpus (Ch. 14.2, "fail loudly, not quietly").
    """
    if not offline:
        try:
            gen = stream_hf(entry)
            first = next(gen)
        except (ImportError, OSError, ConnectionError):
            pass                                   # no `datasets` / no network
        else:
            yield first
            yield from gen
            return
    yield from synthetic_corpus(entry, n_docs=n_docs)


def synthetic_text_sample(n_docs_per_source: int = 300) -> str:
    """One big string sampled across all domains -- used to train the toy tokenizer."""
    parts = []
    for entry in STACK100M_MIX:
        for doc in synthetic_corpus(entry, n_docs=n_docs_per_source):
            parts.append(doc["text"])
    return "\n".join(parts)
