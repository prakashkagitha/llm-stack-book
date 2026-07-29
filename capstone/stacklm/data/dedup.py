"""Two-stage deduplication (Ch. 14.2).

  1. Exact dedup  -- blake2b over normalized text, streaming, O(16 bytes)/doc.
  2. Near dedup   -- MinHash (Broder, 1997) + LSH banding over character
                     5-shingles, streaming with a bounded index.

Implemented from scratch (stdlib + numpy) so the mechanism is visible. For a
real 20B-token corpus use `datatrove`'s Minhash* pipeline instead -- see the
"Production path" section of Ch. 14.2 for the throughput arithmetic.
"""
import hashlib
import logging
import random
import re
from typing import Iterable, Iterator, List

import numpy as np

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_MERSENNE_31 = (1 << 31) - 1  # keeps a*h+b inside int64 for vectorized minhashing


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace, for a stable exact-dup hash key."""
    return _WS_RE.sub(" ", text.lower()).strip()


def exact_dedup(docs: Iterable[dict]) -> Iterator[dict]:
    """Drop documents whose normalized-text digest has been seen before.

    The payload is 16 bytes per *unique* document, but a CPython `set` of
    `bytes` costs ~80-90 B/element once the hash-table slot and object header
    are counted: budget ~1.7 GB at 20M unique documents, not 320 MB.
    """
    seen: set = set()
    for doc in docs:
        h = hashlib.blake2b(normalize(doc["text"]).encode("utf-8"), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        yield doc


def shingles(text: str, k: int = 5) -> List[str]:
    """Character k-shingles: robust to the small word-level edits that
    near-duplicates are made of."""
    t = normalize(text)
    if len(t) < k:
        return [t]
    return list({t[i:i + k] for i in range(len(t) - k + 1)})


class MinHasher:
    """`num_perm` hash functions h_i(x) = (a_i*x + b_i) mod p give a signature of
    `num_perm` minima over a shingle set. Broder (1997):
    P(min_i(A) == min_i(B)) = Jaccard(A, B), so the fraction of matching
    signature positions is an unbiased estimator of the true Jaccard similarity.

    The permutation sweep is vectorized with numpy: the (num_perm, n_shingles)
    matrix is built once and reduced with `.min(axis=1)`, ~12x faster than the
    equivalent Python loop (measured ~8 ms vs ~104 ms on a 5.5 KB document).
    """

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
        """(num_perm,) uint32 signature. Deliberately NOT a tuple of Python
        ints: a 128-int tuple measures ~4.9 KB, this row is 128*4 = 512 B."""
        if not shingle_list:
            return np.zeros(self.num_perm, dtype=np.uint32)
        h = self._shingle_hashes(shingle_list)                       # (n_shingles,)
        mixed = (self.a[:, None] * h[None, :] + self.b[:, None]) % self._p
        return mixed.min(axis=1).astype(np.uint32)                   # (num_perm,)


class SignatureStore:
    """Contiguous (n, num_perm) uint32 store, grown by doubling up to `capacity`.

    Row cost is exactly num_perm*4 bytes -- 512 B at num_perm=128, versus ~4.9 KB
    for the tuple-of-ints it replaces.
    """

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
    """Banding: split the signature into `bands` bands of `rows` rows. Two docs
    are *candidates* if any band matches exactly, turning an O(n^2) all-pairs
    scan into near-linear bucket lookups at the cost of a probabilistic
    threshold governed by the (1/bands)^(1/rows) S-curve."""

    def __init__(self, num_perm: int = 128, bands: int = 16):
        assert num_perm % bands == 0
        self.bands, self.rows = bands, num_perm // bands
        self.buckets: list = [dict() for _ in range(bands)]

    def _band_keys(self, sig) -> list:
        """32-byte `bytes` keys, not tuples of Python ints: ~65 B vs ~340 B each,
        and byte-identical across processes and runs."""
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
    """Fraction of agreeing signature positions -- an unbiased estimator of the
    true Jaccard similarity (Broder, 1997)."""
    return float(np.mean(np.asarray(sig_a) == np.asarray(sig_b)))


def lsh_candidate_prob(jaccard: float, bands: int, rows: int) -> float:
    """P(at least one band collides) at true similarity J -- the LSH S-curve."""
    return 1.0 - (1.0 - jaccard ** rows) ** bands


def near_dedup_stream(docs: Iterable[dict], num_perm: int = 128, bands: int = 16,
                      threshold: float = 0.8, index_capacity: int = 500_000
                      ) -> Iterator[dict]:
    """Streaming near-dedup: yield documents that are not near-duplicates of an
    already-kept document. The corpus is never resident; memory is bounded by
    the index, measured at ~512 B/signature plus ~3.3 KB/document of LSH buckets
    (16 bands), i.e. ~1.9 GB at the 500k default.

    HARD CEILING: past `index_capacity` the index stops growing and near-dup
    recall for every later document silently drops to zero. We log loudly once.
    At 20B tokens (~20M documents) this ceiling is what forces the `datatrove`
    path -- do not just raise the number.
    """
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
                        "recall is now ZERO for the rest of this stream. Shard the "
                        "input or switch to datatrove's MinhashDedup* pipeline.",
                        index_capacity)
        yield doc


def near_dedup(docs, num_perm: int = 128, bands: int = 16,
               threshold: float = 0.8) -> list:
    """List-returning convenience wrapper around `near_dedup_stream`."""
    return list(near_dedup_stream(docs, num_perm=num_perm, bands=bands,
                                  threshold=threshold))
