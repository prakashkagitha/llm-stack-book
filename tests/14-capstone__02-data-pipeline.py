"""
Runs the CPU-runnable Python code blocks from:
    content/14-capstone/02-data-pipeline.md

Blocks are copied faithfully from the chapter (verbatim logic) and concatenated
in document order. The ONLY edits are mechanical: the chapter's intra-package
relative imports (`from .dedup import ...`, `from .synthetic import ...`) are
dropped, because every symbol they would import is already defined earlier in
this single file. Each block is then actually exercised with tiny fixtures, so
every tested block EXECUTES rather than merely defining names.

Tested blocks:
    #1 (synthetic.py)     -- DataMixEntry / STACK100M_MIX / load_hf_stream (defined
                              only; its `datasets` import is lazy and never called)
    #2 (synthetic.py)     -- stream_source offline fallback
    #3 (synthetic.py)     -- synthetic_corpus with injected exact/near duplicates
    #4 (filters.py)       -- domain-routed quality gates + filter_config_hash
    #5 (dedup.py)         -- exact_dedup / MinHasher / LSHIndex / near_dedup_stream
    #6 (pack.py)          -- pack_documents / segments_from_bos /
                              build_intra_doc_causal_mask / segment_ids_from_positions
    #7 (shard.py)         -- ShardWriter / build_shards / manifest.json
    #8 (dataset.py)       -- PackedMemmapDataset (derives seq_ids from bos_id)
    #9 (build_corpus.py)  -- budgeted interleave, shuffle buffer, holdout split
    #10 (toy runner)      -- the chapter's ten-second end-to-end offline run
    #11 (Exercise 5)      -- dedup_all_sources cross-source generator pipeline

Skipped blocks:
    - `capstone/scripts/dedup_datatrove.py`: SKIP(optional-dep). Top-level
      `datatrove` imports; the block is a production multi-process, on-disk
      pipeline over S3 paths -- not installable or runnable in CI.
    - `datasets.interleave_datasets(...)` snippet: SKIP(network + optional-dep).
      Needs `datasets` AND live HuggingFace Hub access.

No network access and no optional third-party imports are exercised: only
numpy, torch, and the standard library, all in the guaranteed CI list.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import random
import re
import tempfile
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# =====================================================================
# Block #1-#3 (chapter: capstone/stacklm/data/synthetic.py)
# =====================================================================


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
    """Open one source as a streaming `datasets.IterableDataset`."""
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


def stream_source(entry: DataMixEntry, offline: bool = False,
                  n_docs: int = 2000) -> Iterator[dict]:
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


# --- exercise blocks #1-#3 -------------------------------------------------
assert abs(sum(e.weight for e in STACK100M_MIX) - 1.0) < 1e-12, "mix weights must sum to 1.0"
_by_name = {e.name: e for e in STACK100M_MIX}
# The chapter's headline correctness point: starcoderdata's text lives in "content".
assert _by_name["starcoder"].text_column == "content"
assert _by_name["starcoder"].hf_data_dir == "python" and _by_name["starcoder"].gated
# Cosmopedia *v2* lives in smollm-corpus, not the cosmopedia (v1) repo.
assert _by_name["cosmopedia_v2"].hf_path == "HuggingFaceTB/smollm-corpus"
assert _by_name["cosmopedia_v2"].hf_config == "cosmopedia-v2"
# Every multi-config repo names its config explicitly.
assert _by_name["fineweb_edu"].hf_config == "sample-100BT"
assert _by_name["finemath"].hf_config == "finemath-4plus"
# 70/15/10/5 of 20B is exactly 14.0B / 3.0B / 2.0B / 1.0B.
assert [round(e.weight * TOTAL_TOKEN_BUDGET / 1e9, 3) for e in STACK100M_MIX] == \
    [14.0, 3.0, 2.0, 1.0]

_web = _by_name["fineweb_edu"]
_docs = list(stream_source(_web, offline=True, n_docs=200))
assert len(_docs) == 200 and all(d["domain"] == "web" and d["text"] for d in _docs)
# `offline=False` with no network / no `datasets` must fall back, not explode.
assert len(list(stream_source(_web, offline=False, n_docs=5))) >= 1

print("\n[blocks #1-#3 OK] source registry, offline fallback, synthetic corpus.\n")


# =====================================================================
# Block #4 (chapter: capstone/stacklm/data/filters.py)
# =====================================================================

_WORD_RE = re.compile(r"\S+")

FILTER_CONFIG = {
    "web": {"min_words": 50, "max_words": 100_000, "min_mean_word_len": 3.0,
            "max_mean_word_len": 10.0, "min_alpha_frac": 0.60, "max_digit_frac": 0.20,
            "max_repeat_line_frac": 0.30},
    "code": {"min_chars": 20, "max_chars": 200_000, "max_char_frac": 0.30},
    "math": {"min_words": 20, "min_digit_frac": 0.03, "min_markers": 3},
}


def filter_config_hash() -> str:
    blob = json.dumps(FILTER_CONFIG, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=6).hexdigest()


def basic_stats(text: str) -> dict:
    words = _WORD_RE.findall(text)
    n_words = len(words) or 1
    n_chars = len(text) or 1
    alpha = sum(c.isalpha() for c in text)
    digit = sum(c.isdigit() for c in text)
    lines = text.splitlines() or [text]
    uniq = len(set(lines))
    return dict(
        n_words=n_words,
        alpha_frac=alpha / n_chars,
        digit_frac=digit / n_chars,
        mean_word_len=sum(len(w) for w in words) / n_words,
        dup_line_frac=1.0 - uniq / len(lines),   # boilerplate / nav-bar detector
    )


def passes_web_filter(text: str) -> bool:
    c = FILTER_CONFIG["web"]
    s = basic_stats(text)
    return (
        c["min_words"] <= s["n_words"] <= c["max_words"]
        and c["min_mean_word_len"] <= s["mean_word_len"] <= c["max_mean_word_len"]
        and s["alpha_frac"] >= c["min_alpha_frac"]
        and s["digit_frac"] <= c["max_digit_frac"]
        and s["dup_line_frac"] <= c["max_repeat_line_frac"]
    )


def passes_code_filter(text: str) -> bool:
    c = FILTER_CONFIG["code"]
    if not (c["min_chars"] <= len(text) <= c["max_chars"]):
        return False
    head = text[:2000]
    most_common_frac = max(head.count(ch) for ch in set(head)) / max(len(head), 1)
    return most_common_frac <= c["max_char_frac"]


def passes_math_filter(text: str) -> bool:
    c = FILTER_CONFIG["math"]
    s = basic_stats(text)
    markers = sum(text.count(m) for m in ("=", "\\frac", "$", "^", "\\sum"))
    return s["n_words"] >= c["min_words"] and (
        s["digit_frac"] >= c["min_digit_frac"] or markers >= c["min_markers"]
    )


_FILTERS = {
    "web": passes_web_filter,
    "synthetic": passes_web_filter,
    "code": passes_code_filter,
    "math": passes_math_filter,
}


def quality_filter(doc: dict) -> bool:
    fn = _FILTERS.get(doc.get("domain", "web"), passes_web_filter)
    return fn(doc["text"])


# --- exercise block #4 -----------------------------------------------------
_good_web = " ".join(["photosynthesis converts sunlight into chemical energy"] * 20)
assert passes_web_filter(_good_web), "clean prose must pass"
assert not passes_web_filter("too short"), "a 2-word doc must fail the length gate"
assert not passes_web_filter(" ".join(["1234567890"] * 200)), "digit spam must fail"
assert not passes_web_filter("\n".join(["subscribe to our newsletter today please"] * 60)), \
    "repeated-line boilerplate must fail"
# Code would be rejected by the prose gate but must pass the code gate -- the
# whole reason the filter is domain-routed.
_code = "def compute(x):\n    return x * x + 1\n\nfor i in range(10):\n    print(i)\n"
assert not passes_web_filter(_code) and passes_code_filter(_code)
assert not passes_code_filter("A" * 5000), "single-char (minified/binary) file must fail"
_math = "let f(x) = x^2 then the derivative is 2x; solve 3x + 5 = 20 for x " * 3
assert passes_math_filter(_math)
assert quality_filter({"text": _code, "domain": "code"}) is True
assert quality_filter({"text": _code, "domain": "web"}) is False
assert len(filter_config_hash()) == 12 and filter_config_hash() == filter_config_hash()

print("[block #4 OK] domain-routed quality gates + stable filter-config hash.\n")


# =====================================================================
# Block #5 (chapter: capstone/stacklm/data/dedup.py)
# =====================================================================

_WS_RE = re.compile(r"\s+")
_MERSENNE_31 = (1 << 31) - 1  # keeps a*h+b inside int64 for vectorized minhashing


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def exact_dedup(docs: Iterable[dict]) -> Iterator[dict]:
    seen: set = set()
    for doc in docs:
        h = hashlib.blake2b(normalize(doc["text"]).encode("utf-8"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        yield doc


def shingles(text: str, k: int = 5) -> List[str]:
    t = normalize(text)
    if len(t) < k:
        return [t]
    return list({t[i:i + k] for i in range(len(t) - k + 1)})


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1234):
        self.num_perm = num_perm
        self._p = _MERSENNE_31
        rng = random.Random(seed)
        self.a = np.array([rng.randrange(1, self._p) for _ in range(num_perm)], dtype=np.int64)
        self.b = np.array([rng.randrange(0, self._p) for _ in range(num_perm)], dtype=np.int64)

    def _shingle_hashes(self, shingle_list) -> np.ndarray:
        raw = b"".join(
            hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest() for s in shingle_list
        )
        h = np.frombuffer(raw, dtype=">u4").astype(np.int64)
        return h % self._p

    def signature(self, shingle_list) -> np.ndarray:
        if not shingle_list:
            return np.zeros(self.num_perm, dtype=np.uint32)
        h = self._shingle_hashes(shingle_list)                       # (n_shingles,)
        mixed = (self.a[:, None] * h[None, :] + self.b[:, None]) % self._p
        return mixed.min(axis=1).astype(np.uint32)                   # (num_perm,)


class SignatureStore:
    """Contiguous (n, num_perm) uint32 store, grown by doubling up to capacity."""

    def __init__(self, num_perm: int, capacity: int, initial: int = 4096):
        self.num_perm, self.capacity = num_perm, capacity
        self._buf = np.empty((max(1, min(initial, capacity)), num_perm), dtype=np.uint32)
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> np.ndarray:
        return self._buf[i]

    def append(self, sig: np.ndarray) -> None:
        assert self.n < self.capacity, "SignatureStore is full"
        if self.n == self._buf.shape[0]:
            grown = np.empty((min(2 * self.n, self.capacity), self.num_perm),
                             dtype=np.uint32)
            grown[: self.n] = self._buf
            self._buf = grown
        self._buf[self.n] = sig
        self.n += 1


class LSHIndex:
    def __init__(self, num_perm: int = 128, bands: int = 16):
        assert num_perm % bands == 0
        self.bands, self.rows = bands, num_perm // bands
        self.buckets: list = [dict() for _ in range(bands)]

    def _band_keys(self, sig) -> list:
        sig = np.asarray(sig, dtype=np.uint32)
        return [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def query_candidates(self, sig: tuple) -> set:
        cands: set = set()
        for b, key in enumerate(self._band_keys(sig)):
            cands.update(self.buckets[b].get(key, ()))
        return cands

    def insert(self, doc_idx: int, sig: tuple) -> None:
        for b, key in enumerate(self._band_keys(sig)):
            self.buckets[b].setdefault(key, []).append(doc_idx)


def estimate_jaccard(sig_a, sig_b) -> float:
    return float(np.mean(np.asarray(sig_a) == np.asarray(sig_b)))


def lsh_candidate_prob(jaccard: float, bands: int, rows: int) -> float:
    return 1.0 - (1.0 - jaccard ** rows) ** bands


def near_dedup_stream(docs: Iterable[dict], num_perm: int = 128, bands: int = 16,
                      threshold: float = 0.8, index_capacity: int = 500_000
                      ) -> Iterator[dict]:
    hasher = MinHasher(num_perm=num_perm)
    index = LSHIndex(num_perm=num_perm, bands=bands)
    store = SignatureStore(num_perm, index_capacity)
    warned = False
    for doc in docs:
        sig = hasher.signature(shingles(doc["text"]))
        if any(estimate_jaccard(sig, store[c]) >= threshold
               for c in index.query_candidates(sig)):
            continue
        if len(store) < index_capacity:
            index.insert(len(store), sig)
            store.append(sig)
        elif not warned:
            warned = True
            log.warning("near_dedup_stream: index_capacity=%d reached; near-dup "
                        "recall is now ZERO for the rest of this stream.",
                        index_capacity)
        yield doc


def near_dedup(docs, num_perm: int = 128, bands: int = 16,
               threshold: float = 0.8) -> list:
    return list(near_dedup_stream(docs, num_perm=num_perm, bands=bands,
                                  threshold=threshold))


# --- exercise block #5 -----------------------------------------------------
# MinHash signature agreement estimates the true Jaccard similarity.
_a = "the quick brown fox jumps over the lazy dog near the river bank at dawn"
_b = _a.replace("lazy dog", "sleepy hound")
_h = MinHasher(num_perm=256)
_sa, _sb = set(shingles(_a)), set(shingles(_b))
_true_j = len(_sa & _sb) / len(_sa | _sb)
_est = estimate_jaccard(_h.signature(list(_sa)), _h.signature(list(_sb)))
assert abs(_est - _true_j) < 0.10, f"MinHash estimate {_est:.3f} vs true {_true_j:.3f}"
assert estimate_jaccard(_h.signature(list(_sa)), _h.signature(list(_sa))) == 1.0

# The S-curve: more, thinner bands (16x8) are strictly more sensitive than
# fewer, fatter ones (8x16) at high similarity -- the chapter's Exercise 4.
_pa = lsh_candidate_prob(0.9, bands=16, rows=8)
_pb = lsh_candidate_prob(0.9, bands=8, rows=16)
assert _pa > 0.999 and 0.79 < _pb < 0.82, (_pa, _pb)
assert lsh_candidate_prob(0.5, 16, 8) < 0.1 < lsh_candidate_prob(0.8, 16, 8)

# End to end on the injected duplicates: every 97th doc is an exact repeat and
# every 53rd is a ~5% edit, so both stages must actually remove documents.
_dup_cache.clear()
_raw = list(synthetic_corpus(_by_name["fineweb_edu"], n_docs=300))
_after_exact = list(exact_dedup(_raw))
_after_near = list(near_dedup_stream(_after_exact, threshold=0.8))
assert len(_after_exact) < len(_raw), "exact dedup must drop the injected exact repeats"
assert len(_after_near) < len(_after_exact), "near dedup must drop the ~5%-edited copies"
# Exact dedup is idempotent; near dedup never resurrects a document.
assert len(list(exact_dedup(_after_exact))) == len(_after_exact)
assert len(near_dedup(_after_near, threshold=0.8)) == len(_after_near)

print(f"[block #5 OK] {len(_raw)} docs -> {len(_after_exact)} after exact "
      f"-> {len(_after_near)} after near dedup.\n")


# =====================================================================
# Block #6 (chapter: capstone/stacklm/data/pack.py)
# =====================================================================

SEQ_LEN = 2048  # Stack-100M pretraining max_seq_len (capstone/PLAN.md sec. 1)


class TokenizerProto(Protocol):
    bos_id: int
    eos_id: int
    pad_id: int
    def encode(self, text: str) -> list: ...


def pack_documents(docs: Iterable[dict], tokenizer, seq_len: int = SEQ_LEN) -> Iterator[tuple]:
    buf_ids: list = []
    buf_pos: list = []
    max_body = seq_len - 2  # room for <bos> and <eos> in every chunk

    for doc in docs:
        raw = doc["ids"] if "ids" in doc else tokenizer.encode(doc["text"])
        chunks = [raw[i:i + max_body] for i in range(0, len(raw), max_body)] or [[]]
        for chunk in chunks:
            toks = [tokenizer.bos_id, *chunk, tokenizer.eos_id]
            pos = list(range(len(toks)))  # this chunk's own position clock, from 0
            buf_ids.extend(toks)
            buf_pos.extend(pos)
            while len(buf_ids) >= seq_len:
                yield buf_ids[:seq_len], buf_pos[:seq_len]
                buf_ids, buf_pos = buf_ids[seq_len:], buf_pos[seq_len:]

    if buf_ids:  # flush a final, padded window
        pad_n = seq_len - len(buf_ids)
        buf_ids.extend([tokenizer.pad_id] * pad_n)
        buf_pos.extend([0] * pad_n)
        yield buf_ids, buf_pos


def segments_from_bos(input_ids: np.ndarray, bos_id: int) -> tuple:
    starts = input_ids == bos_id
    seq_ids = np.cumsum(starts) - 1                       # -1 for a leading tail
    idx = np.arange(input_ids.shape[0])
    seg_start = np.maximum.accumulate(np.where(starts, idx, -1))
    return seq_ids, idx - np.maximum(seg_start, 0)


def build_intra_doc_causal_mask(position_ids: np.ndarray) -> np.ndarray:
    seq_len = position_ids.shape[0]
    doc_id = np.cumsum(position_ids == 0)  # monotonically increasing per document
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    same_doc = doc_id[:, None] == doc_id[None, :]
    return causal & same_doc


def segment_ids_from_positions(position_ids: np.ndarray) -> np.ndarray:
    return np.cumsum(position_ids == 0) - 1


# =====================================================================
# Block #10a (chapter: the ten-second toy runner's ByteTokenizer)
# =====================================================================


class ByteTokenizer:
    """Dependency-free stand-in for Ch. 14.3's BPE tokenizer (vocab_size=32768)."""
    bos_id, eos_id, pad_id = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))


# --- exercise block #6 -----------------------------------------------------
_tok = ByteTokenizer()
_pack_docs = [{"text": "alpha beta"}, {"text": "gamma"}, {"text": "delta epsilon zeta"}]
_windows = list(pack_documents(_pack_docs, _tok, seq_len=16))
assert _windows, "packing must yield at least one window"
for _ids, _pos in _windows:
    assert len(_ids) == len(_pos) == 16, "every window is exactly seq_len long"
    assert 0 <= min(_pos) and max(_pos) < 16, "positions must lie in [0, seq_len)"

# The chapter's worked example, verbatim: seq_len=8, docs of 3/2/5 body tokens.
_w1_ids = np.array([256, 1, 2, 3, 257, 256, 11, 12])
_w1_pos = np.array([0, 1, 2, 3, 4, 0, 1, 2])
_mask = build_intra_doc_causal_mask(_w1_pos)
assert _mask[7, 5] and _mask[7, 6] and _mask[7, 7], "token 7 sees its own document"
assert not _mask[7, :5].any(), "token 7 must NOT see document A (tokens 0-4)"
assert not _mask[0, 1:].any(), "causality: token 0 sees nothing later"
# The redundancy claim of Exercise 3: bos alone recovers the same segmentation.
_seq_from_bos, _pos_from_bos = segments_from_bos(_w1_ids, bos_id=256)
assert (_seq_from_bos == segment_ids_from_positions(_w1_pos)).all()
assert (_pos_from_bos == _w1_pos).all()
# ... and a window starting mid-document isolates the carried-over tail as -1.
_w2_ids = np.array([257, 256, 21, 22, 23, 24, 25, 257])
_seq2, _pos2 = segments_from_bos(_w2_ids, bos_id=256)
assert _seq2[0] == -1 and (_seq2[1:] == 0).all(), _seq2
assert (_pos2 == np.array([0, 0, 1, 2, 3, 4, 5, 6])).all(), _pos2
# The mask built from derived positions equals the mask from stored positions.
assert (build_intra_doc_causal_mask(_pos_from_bos) == _mask).all()

# Exercise 2: cu_seqlens for FlashAttention's varlen API, derived from seq_ids,
# must reproduce exactly the dense mask built from position ids.
def cu_seqlens_from_seq_ids(seq_ids: np.ndarray) -> np.ndarray:
    bounds = np.flatnonzero(np.diff(seq_ids)) + 1
    return np.concatenate(([0], bounds, [seq_ids.shape[0]])).astype(np.int32)


def mask_from_cu_seqlens(cu: np.ndarray, seq_len: int) -> np.ndarray:
    seg = np.zeros(seq_len, dtype=np.int64)
    for k in range(len(cu) - 1):
        seg[cu[k]:cu[k + 1]] = k
    return np.tril(np.ones((seq_len, seq_len), dtype=bool)) & (seg[:, None] == seg[None, :])


_cu1 = cu_seqlens_from_seq_ids(_seq_from_bos)
assert _cu1.tolist() == [0, 5, 8], _cu1
assert int(_cu1[-1]) == _w1_ids.shape[0] and np.all(np.diff(_cu1) > 0)
assert (mask_from_cu_seqlens(_cu1, 8) == _mask).all()
_cu2 = cu_seqlens_from_seq_ids(_seq2)
assert _cu2.tolist() == [0, 1, 8], _cu2
assert (mask_from_cu_seqlens(_cu2, 8) ==
        build_intra_doc_causal_mask(_pos2)).all()

print("[block #6 OK] packing, position resets, block-diagonal mask, bos-derived "
      "segments, cu_seqlens round-trip.\n")


# =====================================================================
# Block #7 (chapter: capstone/stacklm/data/shard.py)
# =====================================================================

DTYPE = np.uint16  # vocab_size=32768 fits; see worked example


class ShardWriter:
    def __init__(self, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000,
                 store_positions: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.seqs_per_shard = max(1, tokens_per_shard // seq_len)
        self.store_positions = store_positions
        self._buf_ids: list = []
        self._buf_pos: list = []
        self._shard_idx = 0
        self.n_sequences = 0

    def add(self, input_ids, position_ids=None) -> None:
        self._buf_ids.append(np.asarray(input_ids, dtype=DTYPE))
        if self.store_positions:
            self._buf_pos.append(np.asarray(position_ids, dtype=DTYPE))
        self.n_sequences += 1
        if len(self._buf_ids) >= self.seqs_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_ids:
            return
        ids = np.stack(self._buf_ids)  # (n_seq, seq_len)
        stem = str(self.out_dir / f"shard_{self._shard_idx:05d}")
        ids.tofile(stem + ".tokens.bin")
        if self.store_positions:
            np.stack(self._buf_pos).tofile(stem + ".pos.bin")
        np.array([ids.shape[0], ids.shape[1]], dtype=np.int64).tofile(stem + ".meta.bin")
        self._shard_idx += 1
        self._buf_ids.clear()
        self._buf_pos.clear()

    def close(self) -> None:
        self._flush()  # flush the trailing partial shard

    def write_manifest(self, tokenizer=None, extra: dict = None) -> dict:
        man = {
            "seq_len": self.seq_len,
            "n_shards": self._shard_idx,
            "n_sequences": self.n_sequences,
            "n_tokens": self.n_sequences * self.seq_len,
            "store_positions": self.store_positions,
        }
        if tokenizer is not None:
            man.update(bos_id=int(tokenizer.bos_id), eos_id=int(tokenizer.eos_id),
                       pad_id=int(tokenizer.pad_id))
        if extra:
            man.update(extra)
        (self.out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        return man


def build_shards(docs, tokenizer, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000,
                 store_positions: bool = False, manifest_extra: dict = None) -> int:
    writer = ShardWriter(out_dir, seq_len=seq_len, tokens_per_shard=tokens_per_shard,
                         store_positions=store_positions)
    for input_ids, position_ids in pack_documents(docs, tokenizer, seq_len=seq_len):
        writer.add(input_ids, position_ids)
    writer.close()
    writer.write_manifest(tokenizer=tokenizer, extra=manifest_extra)
    return writer._shard_idx


# =====================================================================
# Block #8 (chapter: capstone/stacklm/data/dataset.py)
# =====================================================================


class PackedMemmapDataset(Dataset):
    def __init__(self, shard_dir: str, bos_id: int = None):
        self.shard_dir = Path(shard_dir)
        self._shards = []       # list of (tokens_memmap, pos_memmap_or_None)
        self._cum_seqs = [0]    # prefix sums of sequence counts, for indexing
        self.seq_len = None     # None, not a loop variable: an empty dir must not crash

        man_path = self.shard_dir / "manifest.json"
        self.manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
        self.bos_id = bos_id if bos_id is not None else self.manifest.get("bos_id")

        for meta_path in sorted(self.shard_dir.glob("shard_*.meta.bin")):
            n_seq, seq_len = (int(x) for x in np.fromfile(meta_path, dtype=np.int64))
            stem = str(meta_path)[: -len(".meta.bin")]
            tok_mm = np.memmap(stem + ".tokens.bin", dtype=np.uint16, mode="r",
                               shape=(n_seq, seq_len))
            pos_path = Path(stem + ".pos.bin")
            pos_mm = (np.memmap(pos_path, dtype=np.uint16, mode="r", shape=(n_seq, seq_len))
                      if pos_path.exists() else None)
            self._shards.append((tok_mm, pos_mm))
            self._cum_seqs.append(self._cum_seqs[-1] + n_seq)
            self.seq_len = seq_len

        if self._shards and self._shards[0][1] is None and self.bos_id is None:
            raise ValueError(
                f"{shard_dir}: no .pos.bin and no bos_id (manifest.json missing?). "
                "Pass PackedMemmapDataset(dir, bos_id=tok.bos_id)."
            )

    def __len__(self) -> int:
        return self._cum_seqs[-1]

    def _locate(self, idx: int) -> tuple:
        if not 0 <= idx < self._cum_seqs[-1]:
            raise IndexError(idx)
        s = bisect.bisect_right(self._cum_seqs, idx) - 1
        return s, idx - self._cum_seqs[s]

    def __getitem__(self, idx: int) -> dict:
        s, row = self._locate(idx)
        tok_mm, pos_mm = self._shards[s]
        ids_np = tok_mm[row].astype(np.int64)
        if pos_mm is not None:                        # legacy shards with .pos.bin
            pos_np = pos_mm[row].astype(np.int64)
            seq_np = np.cumsum(pos_np == 0) - 1
        else:                                         # derive from the tokens alone
            seq_np, pos_np = segments_from_bos(ids_np, self.bos_id)
        ids = torch.from_numpy(ids_np)
        return {
            "input_ids": ids[:-1],
            "position_ids": torch.from_numpy(np.ascontiguousarray(pos_np))[:-1],
            "seq_ids": torch.from_numpy(np.ascontiguousarray(seq_np))[:-1],
            "targets": ids[1:],
        }


# --- exercise blocks #7-#8 -------------------------------------------------
_tmp = tempfile.mkdtemp(prefix="ch142_shards_")
_dup_cache.clear()
_shard_docs = list(synthetic_corpus(_by_name["fineweb_edu"], n_docs=40))
_n_shards = build_shards(_shard_docs, _tok, _tmp, seq_len=64, tokens_per_shard=64 * 4)
assert _n_shards >= 2, "small tokens_per_shard must produce several shards"
assert not list(Path(_tmp).glob("*.pos.bin")), ".pos.bin must NOT be written by default"
_man = json.loads((Path(_tmp) / "manifest.json").read_text())
assert _man["bos_id"] == _tok.bos_id and _man["seq_len"] == 64

_ds = PackedMemmapDataset(_tmp)                 # bos_id recovered from the manifest
assert len(_ds) == _man["n_sequences"] > 0
_item = _ds[0]
assert sorted(_item) == ["input_ids", "position_ids", "seq_ids", "targets"]
assert _item["input_ids"].shape == _item["targets"].shape == (63,)
# Round-trip: the packed tokens survive the uint16 write/read unchanged.
_expected = list(pack_documents(_shard_docs, _tok, seq_len=64))
_flat_disk = np.concatenate([np.asarray(_ds[i]["input_ids"]) for i in range(len(_ds))])
_flat_mem = np.concatenate([np.asarray(w[0][:-1]) for w in _expected])
assert (_flat_disk == _flat_mem).all(), "uint16 round-trip must preserve every token"
# Targets are the inputs shifted by one.
assert (np.asarray(_item["targets"])[:-1] == np.asarray(_item["input_ids"])[1:]).all()
# Derived seq_ids agree with the stored-position derivation, window by window.
for _i in range(min(len(_ds), 8)):
    _ids_w, _pos_w = _expected[_i]
    assert (np.asarray(_ds[_i]["seq_ids"]) ==
            segment_ids_from_positions(np.asarray(_pos_w))[:-1]).all()
# An empty shard dir must not raise UnboundLocalError.
_empty = tempfile.mkdtemp(prefix="ch142_empty_")
assert len(PackedMemmapDataset(_empty, bos_id=256)) == 0

print(f"[blocks #7-#8 OK] {_n_shards} shards, tokens-only round trip, derived seq_ids.\n")


# =====================================================================
# Block #9 (chapter: capstone/stacklm/data/build_corpus.py)
# =====================================================================


def encode_batched(docs, tokenizer, batch: int = 1024):
    """Attach `ids` to each document, encoding in batches so the tokenizer's
    fast/parallel batch API (Ch. 14.3 `encode_corpus`, HF `encode_batch`,
    tiktoken `encode_ordinary_batch`) is reachable."""
    encode_batch = getattr(tokenizer, "encode_batch", None)
    buf = []

    def flush():
        texts = [d["text"] for d in buf]
        id_lists = (encode_batch(texts) if encode_batch is not None
                    else [tokenizer.encode(t) for t in texts])
        for d, ids in zip(buf, id_lists):
            yield {**d, "ids": list(ids)}
        buf.clear()

    for doc in docs:
        buf.append(doc)
        if len(buf) >= batch:
            yield from flush()
    if buf:
        yield from flush()


def _source_pipeline(entry, tokenizer, budget_tokens, offline, dedup_kwargs, stats,
                     encode_batch_size: int = 1024):
    docs = stream_source(entry, offline=offline)
    docs = (d for d in docs if quality_filter(d))
    docs = exact_dedup(docs)
    docs = near_dedup_stream(docs, **dedup_kwargs)
    used = 0
    for doc in encode_batched(docs, tokenizer, batch=encode_batch_size):
        if not doc["ids"]:
            continue
        used += len(doc["ids"]) + 2              # +2 for <bos>/<eos> added at pack time
        stats[entry.name] = used
        yield doc
        if used >= budget_tokens:
            return


def interleave_budgeted(entries, tokenizer, total_tokens, offline=True,
                        seed=1337, dedup_kwargs=None, stats=None):
    stats = {} if stats is None else stats
    dedup_kwargs = dedup_kwargs or {}
    rng = random.Random(seed)
    gens, weights = {}, {}
    for e in entries:
        budget = int(round(e.weight * total_tokens))
        gens[e.name] = _source_pipeline(e, tokenizer, budget, offline, dedup_kwargs, stats)
        weights[e.name] = e.weight
        stats.setdefault(e.name, 0)
    alive = list(gens)
    while alive:
        name = rng.choices(alive, weights=[weights[n] for n in alive], k=1)[0]
        try:
            yield next(gens[name])
        except StopIteration:
            alive.remove(name)


def shuffle_buffer(docs, size=100_000, seed=1337):
    rng = random.Random(seed)
    buf = []
    for doc in docs:
        if len(buf) < size:
            buf.append(doc)
            continue
        j = rng.randrange(size)
        yield buf[j]
        buf[j] = doc
    rng.shuffle(buf)
    yield from buf


def is_holdout(doc, per_mille: int = 1) -> bool:
    h = int(hashlib.blake2b(doc["text"].encode("utf-8"), digest_size=8).hexdigest(), 16)
    return (h % 1000) < per_mille


def build_corpus(out_dir, tokenizer, total_tokens=TOTAL_TOKEN_BUDGET, entries=None,
                 seq_len=2048, tokens_per_shard=100_000_000, offline=True,
                 holdout_per_mille=1, holdout_tokens=10_000_000,
                 shuffle_size=100_000, seed=1337, dedup_kwargs=None) -> dict:
    entries = entries if entries is not None else STACK100M_MIX
    assert abs(sum(e.weight for e in entries) - 1.0) < 1e-9, "mix weights must sum to 1"
    out = Path(out_dir)
    stats: dict = {}
    val_docs: list = []
    held = {"tokens": 0}

    docs = interleave_budgeted(entries, tokenizer, total_tokens, offline=offline,
                               seed=seed, dedup_kwargs=dedup_kwargs, stats=stats)
    docs = shuffle_buffer(docs, size=shuffle_size, seed=seed)

    def train_stream():
        for doc in docs:
            if is_holdout(doc, holdout_per_mille):
                if held["tokens"] < holdout_tokens:
                    val_docs.append(doc)
                    held["tokens"] += len(doc["ids"]) + 2
                continue                     # held-out docs NEVER enter training
            yield doc

    train_writer = ShardWriter(out / "train", seq_len=seq_len,
                               tokens_per_shard=tokens_per_shard)
    for ids, pos in pack_documents(train_stream(), tokenizer, seq_len=seq_len):
        train_writer.add(ids, pos)
    train_writer.close()

    val_writer = ShardWriter(out / "val", seq_len=seq_len,
                             tokens_per_shard=tokens_per_shard)
    for ids, pos in pack_documents(iter(val_docs), tokenizer, seq_len=seq_len):
        val_writer.add(ids, pos)
    val_writer.close()

    realized = sum(stats.values()) or 1
    provenance = {
        "seed": seed,
        "token_budget": total_tokens,
        "filter_config_hash": filter_config_hash(),
        "dedup": {"num_perm": 128, "bands": 16, "threshold": 0.8, **(dedup_kwargs or {})},
        "sources": [
            {"name": e.name, "hf_path": e.hf_path, "hf_config": e.hf_config,
             "hf_data_dir": e.hf_data_dir, "revision": e.revision,
             "target_weight": e.weight,
             "realized_tokens": stats.get(e.name, 0),
             "realized_weight": round(stats.get(e.name, 0) / realized, 4)}
            for e in entries
        ],
        "holdout": {"per_mille": holdout_per_mille, "tokens": held["tokens"]},
        "offline_synthetic": offline,
    }
    man = train_writer.write_manifest(tokenizer=tokenizer, extra=provenance)
    val_writer.write_manifest(tokenizer=tokenizer, extra=provenance)
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    return man


# --- exercise blocks #9-#10 ------------------------------------------------
# The chapter's ten-second toy runner, at the sizes it prescribes.
_dup_cache.clear()
_out = tempfile.mkdtemp(prefix="stack100m_toy_")
_manifest = build_corpus(_out, _tok,
                         total_tokens=200_000,      # 20B in the real run
                         seq_len=128,               # 2048 in the real run
                         tokens_per_shard=128 * 64,
                         offline=True,              # synthetic sources, hermetic
                         holdout_per_mille=20, holdout_tokens=20_000,
                         shuffle_size=256)
_train = PackedMemmapDataset(f"{_out}/train")
_val = PackedMemmapDataset(f"{_out}/val")
_batch = _train[0]
assert _batch["input_ids"].shape == _batch["targets"].shape == (_manifest["seq_len"] - 1,)
assert int(_batch["position_ids"].max()) < _manifest["seq_len"]
assert int(_batch["seq_ids"].max()) >= 0
assert len(_train) > 0 and len(_val) > 0, "both splits must be non-empty"

# The budget is enforced per source, and the realized mix tracks the target.
_realized = {s["name"]: s["realized_weight"] for s in _manifest["sources"]}
for _e in STACK100M_MIX:
    assert abs(_realized[_e.name] - _e.weight) < 0.03, (_e.name, _realized[_e.name])
assert abs(sum(_realized.values()) - 1.0) < 1e-3
assert sum(s["realized_tokens"] for s in _manifest["sources"]) <= 1.05 * 200_000

# Provenance the Ch. 14.12 reproducibility checklist needs.
assert _manifest["seed"] == 1337 and _manifest["filter_config_hash"] == filter_config_hash()
assert all(s["hf_path"] and s["revision"] for s in _manifest["sources"])
assert _manifest["holdout"]["tokens"] > 0

# Interleaving really is interleaved: code documents must appear early, not only
# after 85% of the stream (the "accidental curriculum" the chapter warns about).
_dup_cache.clear()
_stats: dict = {}
_mixed = list(interleave_budgeted(STACK100M_MIX, _tok, 100_000, offline=True,
                                  seed=1337, stats=_stats))
_first_code = next(i for i, d in enumerate(_mixed) if d["domain"] == "code")
assert _first_code < 0.25 * len(_mixed), f"first code doc at {_first_code}/{len(_mixed)}"
assert set(_stats) == {e.name for e in STACK100M_MIX}

# The held-out split is deterministic and atomic per document.
_probe = [{"text": f"document number {i}"} for i in range(2000)]
_held = [d for d in _probe if is_holdout(d, per_mille=50)]
assert 40 < len(_held) < 160, len(_held)          # ~5% of 2000, binomial slack
assert all(is_holdout(d, per_mille=50) for d in _held), "assignment must be stable"

# Shuffling actually reorders (and preserves) the stream.
_seq = [{"text": str(i), "ids": [i]} for i in range(500)]
_shuffled = list(shuffle_buffer(iter(_seq), size=64, seed=1337))
assert len(_shuffled) == 500 and {d["text"] for d in _shuffled} == {d["text"] for d in _seq}
assert [d["text"] for d in _shuffled] != [d["text"] for d in _seq]

print(f"[blocks #9-#10 OK] corpus built: {len(_train)} train / {len(_val)} val seqs, "
      f"realized mix {_realized}.\n")


# =====================================================================
# Block #11 (chapter Exercise 5: capstone/stacklm/data/dedup_all.py)
# =====================================================================


def dedup_all_sources(entries=None, offline: bool = True, num_perm: int = 128,
                      bands: int = 16, threshold: float = 0.8):
    """Concatenate every source's stream and deduplicate globally."""
    entries = entries if entries is not None else STACK100M_MIX
    combined = chain.from_iterable(
        stream_source(entry, offline=offline) for entry in entries
    )
    kept = (d for d in combined if quality_filter(d))   # 1. cheap per-doc gate
    kept = exact_dedup(kept)                            # 2. cheap streaming hash
    return near_dedup_stream(kept, num_perm=num_perm,   # 3. expensive MinHash
                             bands=bands, threshold=threshold)


# --- exercise block #11 ----------------------------------------------------
_dup_cache.clear()
_small_mix = [DataMixEntry(e.name, e.hf_path, e.weight, e.domain) for e in STACK100M_MIX]
_gen = dedup_all_sources(_small_mix, offline=True)
assert iter(_gen) is _gen, "must return a generator, not a materialized list"
_kept_docs = list(_gen)
_dup_cache.clear()
_all_raw = sum(len(list(stream_source(e, offline=True, n_docs=2000))) for e in _small_mix)
assert 0 < len(_kept_docs) < _all_raw, (len(_kept_docs), _all_raw)
assert {d["domain"] for d in _kept_docs} == {"web", "synthetic", "code", "math"}

print(f"[block #11 OK] cross-source dedup: {_all_raw} -> {len(_kept_docs)} documents.\n")


# =====================================================================
# SKIP notes (not executed -- see module docstring for rationale)
# =====================================================================
# capstone/scripts/dedup_datatrove.py -- SKIP(optional-dep): top-level
#   `datatrove` imports, multi-process executors, S3/scratch paths.
# `datasets.interleave_datasets(...)` snippet -- SKIP(network + optional-dep):
#   needs `datasets` and live Hub access. Its offline equivalent,
#   `interleave_budgeted`, IS exercised above.

print("=== All tested blocks (#1-#11) executed and verified successfully. ===")
