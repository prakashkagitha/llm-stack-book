"""
Runs the CPU-runnable Python blocks from content/14-capstone/03-tokenizer.md,
concatenated in order so later blocks can rely on names defined by earlier
ones -- exactly as the chapter's own narrative does (block #5's train_bpe /
SPECIAL_TOKENS feed block #6's StackTokenizer, which blocks #8/#14/#15/#16/#20
all instantiate and use). Each block is copied verbatim from the chapter;
only the minimal glue needed to make it run standalone in one file (dropping
now-redundant cross-module imports, picking tiny fixture corpora/vocab sizes
in place of network-fetched or gigabyte-scale ones) is added and marked GLUE.

Blocks tested (of the 11 the task named):
  #2  (line ~97)   SPLIT_PATTERN_UNICODE regex literal
  #5  (line ~157)  bpe.py: SPECIAL_TOKENS, VOCAB_SIZE, SPLIT_PATTERN*, train_bpe()
  #7  (line ~573)  pack.py: the Tokenizer Protocol
  #8  (line ~615)  parallel.py: _init_worker / _encode_doc / encode_corpus
  #14 (line ~896)  train_tokenizer.py: stream_sample() + the __main__ training path
  #15 (line ~964)  the chapter's own CI/toy path (StackTokenizer round trip)
  #16 (line ~1143) round-trip verification against a freshly-trained tokenizer
  #20 (line ~1367) Exercise 8's failing-heap-bug regression test for train_bpe

Included as a hard dependency of the tested blocks above, though the task's
heuristic marks it "fragment" (it is a bare "(continued)" file fragment on
its own, not because its logic is unsafe or non-CPU):
  #6  (line ~348)  StackTokenizer class + load_tokenizer() -- every one of
                    #8/#14/#15/#16/#20 instantiates and calls this class; the
                    5 blocks the task actually asked for would be inert
                    without it. Copied verbatim.

SKIP(network / optional-dependency), not exercised, per the task's own
default-SKIP classification:
  #9  (line ~665)  export.py: to_tiktoken/to_hf_tokenizer/save_pretrained --
                    needs `tiktoken` and HuggingFace `tokenizers`/
                    `transformers`, all blocked in CI (needs-net).
  #10 (line ~787)  A `>>>` REPL transcript (not standalone code -- it reads
                    the REPL's implicit `_` sentinel) demonstrating that HF's
                    AddedVocabulary extracts `<|assistant|>` even with
                    add_special_tokens=False. Needs the `hf` object built by
                    block #9's `to_hf_tokenizer`, which needs `tokenizers`.
  #11 (line ~799)  Same REPL-transcript style; needs the `fast`
                    PreTrainedTokenizerFast object from block #9, which needs
                    `transformers`.
  #12 (line ~810)  test_tokenizer_export.py -- needs `jinja2`'s consumer
                    `stacklm.post.chat` (an un-shown chapter module),
                    `tokenizers`, and `transformers.AutoTokenizer`. needs-net
                    per the task's own classification.
  #13 (line ~876)  `assert llama_cpp_tokenize(gguf_path, PROBE) == ...` -- a
                    bare fragment referencing an undefined `llama_cpp_tokenize`
                    helper and a GGUF file that does not exist anywhere in
                    this hermetic environment; not standalone runnable code.

Every other block not in the task's list of 11 and not the hard dependency
#6 above (#0/#1 prose diagrams, #3/#4 regex/assert fragments already
subsumed by block #5's own copy of SPLIT_PATTERN_STDLIB, #17/#18/#19
Exercise-solution fragments) is intentionally left out, per the task's
default-SKIP instruction.
"""

# `from __future__ import annotations` must be the first statement after the
# module docstring in a Python file -- both block #5 and block #8 (parallel.py)
# open with this line in the book, since they are separate files there. GLUE:
# hoisted to the top once here; the duplicate occurrences below are commented
# out where they'd otherwise raise "must occur at the beginning of the file".
from __future__ import annotations

import glob
import heapq
import json
import multiprocessing as mp
import os
import tempfile
import warnings
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Protocol, Tuple, Union, runtime_checkable

import numpy as np


print("=" * 70)
print("Block #2 (line ~97): SPLIT_PATTERN_UNICODE, the cl100k-style regex")
print("=" * 70)

# --- verbatim from the chapter ---
SPLIT_PATTERN_UNICODE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"      # contractions, case-insensitive (GPT-2 was not)
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"          # a word, optionally preceded by one non-letter
    r"|\p{N}{1,3}"                        # <= 3 digits, NEVER absorbing a leading space
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"         # punctuation runs, trailing newlines attached
    r"|\s*[\r\n]+"                        # newline runs (indentation-friendly)
    r"|\s+(?!\S)"                         # trailing whitespace at end of a run
    r"|\s+"                               # any remaining whitespace
)
# --- end verbatim ---
assert isinstance(SPLIT_PATTERN_UNICODE, str) and len(SPLIT_PATTERN_UNICODE) > 0
print("SPLIT_PATTERN_UNICODE defined,", len(SPLIT_PATTERN_UNICODE), "chars")


print()
print("=" * 70)
print("Block #5 (line ~157): capstone/stacklm/tokenizer/bpe.py -- the trainer")
print("=" * 70)

# --- verbatim from the chapter (capstone/stacklm/tokenizer/bpe.py) ---
"""
Stack-100M's tokenizer: a from-scratch, byte-level BPE trainer + encoder/decoder.

This is the production version of the algorithm built from first principles in
../02-transformer/01-tokenization.html -- same "merge the most frequent adjacent
pair" idea, same byte-level guarantee that no input is ever unrepresentable, but
engineered to finish ~32.5k merges on a real multi-megabyte sample in seconds
instead of hours (measured numbers in "Training at scale" below).

Special tokens are reserved UP FRONT (see SPECIAL_TOKENS) even though most are
untouched until Ch. 14.9 (SFT/DPO) and Ch. 14.10 (the agent). Once this
tokenizer is trained and Stack-100M's embedding table is initialized against it,
the vocabulary is FROZEN: id 32763 means "<|user|>" forever, or every
checkpoint's embedding table becomes meaningless.
"""

# from __future__ import annotations   # GLUE: already imported at module top

try:                        # `regex` gives us \p{L} / \p{N}; see the aside above
    import regex as _re
    _HAVE_REGEX = True
except ImportError:         # hermetic CI: stdlib approximation, same digit cap
    import re as _re
    _HAVE_REGEX = False

# ---------------------------------------------------------------------------
# 1. Special tokens, reserved up front, in the exact order the rest of the
#    capstone relies on. Never reorder these once a tokenizer has been trained
#    and used to initialize an embedding table. Later chapters hardcode these
#    STRINGS (looked up via special_token_id), not raw integers -- but the ids
#    themselves become baked into every checkpoint's embedding rows.
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: Tuple[str, ...] = (
    "<|bos|>",           # beginning of sequence / beginning of a packed document
    "<|eos|>",           # end of sequence / end of document
    "<|pad|>",           # padding, always masked out of the loss
    "<|system|>",        # chat role marker (Ch. 14.9 SFT/DPO)
    "<|user|>",          # chat role marker
    "<|assistant|>",     # chat role marker -- loss starts right after this token
    "<|end|>",           # end-of-turn marker
    "<|tool_call|>",     # opens a tool invocation the model emits (Ch. 14.10)
    "<|tool_result|>",   # opens a tool's returned observation, masked from loss
)

VOCAB_SIZE = 32768                                          # PLAN Sec. 1 / Sec. 3
NUM_BYTES = 256                                             # every raw byte is id 0..255
NUM_MERGES = VOCAB_SIZE - NUM_BYTES - len(SPECIAL_TOKENS)   # = 32,503

SPLIT_PATTERN_UNICODE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
SPLIT_PATTERN_STDLIB = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|(?:[^\r\n\w]|_)?[^\W\d_]+"
    r"|\d{1,3}"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
SPLIT_PATTERN = SPLIT_PATTERN_UNICODE if _HAVE_REGEX else SPLIT_PATTERN_STDLIB
_SPLIT_RE = _re.compile(SPLIT_PATTERN)


# ---------------------------------------------------------------------------
# 2. The trainer. Operates on integer symbol ids throughout (0..255 for raw
#    bytes, 256+ for learned merges) instead of the string glyphs used in the
#    from-scratch chapter -- purely a speed choice, the algorithm is identical.
# ---------------------------------------------------------------------------
def train_bpe(word_freqs: Dict[Tuple[int, ...], int],
              num_merges: int) -> List[Tuple[int, int]]:
    """Learn `num_merges` BPE merges from a corpus reduced to
    (symbol-id-tuple -> frequency) counts.

    Naively, each merge rescans the ENTIRE corpus to recount every adjacent
    pair -- O(num_merges x corpus_size). Instead we maintain three pieces of
    running state so each merge touches only the (usually small) subset of
    words that actually contain the pair being merged:

      - `pair_counts`   : running frequency-weighted count of every adjacent pair
      - `pair_to_words` : inverted index, pair -> set of word indices containing it
      - `heap`          : a max-heap (via negated counts) of candidate pairs

    LAZY DELETION: we never remove stale entries, we discard them on pop when
    they no longer match the live count. This avoids an expensive decrease-key
    -- but it is only CORRECT if every count change re-pushes a fresh entry.
    Pushing only on increments loses decremented pairs forever (see the pitfall
    box above), so we collect `touched` and re-push all of it.
    """
    words: List[List[int]] = [list(w) for w in word_freqs.keys()]
    freqs: List[int] = list(word_freqs.values())

    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_to_words: Dict[Tuple[int, int], set] = defaultdict(set)
    for wi, (word, f) in enumerate(zip(words, freqs)):
        for a, b in zip(word, word[1:]):
            pair_counts[(a, b)] += f
            pair_to_words[(a, b)].add(wi)

    heap = [(-c, pair) for pair, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: List[Tuple[int, int]] = []
    next_id = NUM_BYTES  # the first learned merge becomes id 256

    while len(merges) < num_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        live_count = pair_counts.get(pair, 0)
        if live_count <= 0 or -neg_count != live_count:
            continue                       # stale heap entry -- skip it
        if live_count < 2:
            break                          # nothing left that repeats; stop early

        merges.append(pair)
        new_id = next_id
        next_id += 1

        touched: set = set()               # every pair whose count we change
        for wi in list(pair_to_words.get(pair, ())):
            word = words[wi]
            # `wi` can be a STALE member of pair_to_words[pair]: an earlier merge
            # may already have removed `pair` from this word. Verify membership
            # before mutating anything.
            if not any(word[i] == pair[0] and word[i + 1] == pair[1]
                       for i in range(len(word) - 1)):
                continue

            f = freqs[wi]
            # Remove this word's contribution to EVERY pair it currently forms
            # (not just `pair`) -- merging shifts adjacency across the word.
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] -= f
                touched.add((a, b))

            # Apply the merge greedily, left to right, non-overlapping.
            merged, i = [], 0
            while i < len(word):
                if (i < len(word) - 1
                        and word[i] == pair[0] and word[i + 1] == pair[1]):
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[wi] = merged

            # Re-add this word's contribution under its NEW symbol sequence.
            for a, b in zip(merged, merged[1:]):
                pair_counts[(a, b)] += f
                pair_to_words[(a, b)].add(wi)
                touched.add((a, b))

        pair_to_words.pop(pair, None)
        pair_counts.pop(pair, None)
        touched.discard(pair)              # this pair is consumed, never re-push

        # THE FIX: refresh the heap for every pair whose count moved -- up OR
        # down. Without the decrement half, pairs whose count fell are dropped
        # from the heap forever and the trainer silently under-produces merges.
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))

    return merges
# --- end verbatim (block #5) ---

assert NUM_MERGES == 32503
assert train_bpe({}, 5) == []                 # empty corpus -> no merges, no crash
print(f"train_bpe defined; NUM_MERGES == {NUM_MERGES} as the chapter's formula gives")


print()
print("=" * 70)
print("Block #6 (line ~348): StackTokenizer class (hard dependency of #8/#14/#15/#16/#20)")
print("=" * 70)

# --- verbatim from the chapter (capstone/stacklm/tokenizer/bpe.py, continued) ---
class StackTokenizer:
    """Byte-level BPE tokenizer for Stack-100M.

    ID layout (fixed for the whole project):
        0        .. 255           raw byte values (id == byte value)
        256      .. 256+M-1       learned merges, in the order they were trained
        256+M    .. vocab_size-1  filler <|unused_N|> (only if the sample was too
                                   small), then the 9 SPECIAL_TOKENS in order,
                                   ALWAYS occupying the final 9 ids.
    """

    def __init__(self, pattern: str = SPLIT_PATTERN) -> None:
        self.merges: List[Tuple[int, int]] = []             # learned merges, ordered
        self.merge_id: Dict[Tuple[int, int], int] = {}      # pair -> resulting id
        self.id_to_pair: Dict[int, Tuple[int, int]] = {}    # inverse, for decode
        self.special_to_id: Dict[str, int] = {}
        self.id_to_special: Dict[int, str] = {}
        self.pattern: str = pattern                          # frozen with the merges
        self._split_re = _re.compile(pattern)                # BOUND, not the global
        self._cache: Dict[Tuple[int, ...], List[int]] = {}   # pre-token -> ids

    @property
    def vocab_size(self) -> int:
        return NUM_BYTES + len(self.merges) + len(self.special_to_id)

    # -- the API surface every later chapter depends on -----------------------
    def special_token_id(self, token_str: str) -> int:
        """Ch. 14.9's chat template and Ch. 14.10's tool formatter call this."""
        return self.special_to_id[token_str]

    id = special_token_id                    # short alias used in 14.10/14.11

    @property
    def bos_id(self) -> int: return self.special_to_id["<|bos|>"]
    @property
    def eos_id(self) -> int: return self.special_to_id["<|eos|>"]
    @property
    def pad_id(self) -> int: return self.special_to_id["<|pad|>"]

    def _pretokenize(self, text: str) -> List[Tuple[int, ...]]:
        """Split `text` into pre-token chunks, each a tuple of raw UTF-8 byte
        VALUES (0..255) -- e.g. 'café' -> (99, 97, 102, 195, 169). Uses the
        INSTANCE's compiled pattern, so a loaded artifact tokenizes under the
        regex it was trained with, not whatever the module default is today."""
        return [tuple(chunk.encode("utf-8")) for chunk in self._split_re.findall(text)]

    # -- training -------------------------------------------------------------
    def _assign_ids(self, all_specials: Tuple[str, ...]) -> None:
        self.merge_id = {pair: NUM_BYTES + i for i, pair in enumerate(self.merges)}
        self.id_to_pair = {v: k for k, v in self.merge_id.items()}
        self.special_to_id, self.id_to_special = {}, {}
        nid = NUM_BYTES + len(self.merges)
        for s in all_specials:
            self.special_to_id[s] = nid
            self.id_to_special[nid] = s
            nid += 1
        self._cache = {}

    def train_from_iterable(self, docs: Iterable[str], vocab_size: int = VOCAB_SIZE,
                            special_tokens: Tuple[str, ...] = SPECIAL_TOKENS) -> int:
        """Train on a STREAM of documents. Returns the merge shortfall (0 is the
        healthy case). Memory is O(distinct pre-token chunks), not O(corpus)."""
        num_merges = vocab_size - NUM_BYTES - len(special_tokens)
        word_freqs: Counter = Counter()
        for doc in docs:
            word_freqs.update(self._pretokenize(doc))
        self.merges = train_bpe(word_freqs, num_merges)

        # Guard: a small/low-diversity sample runs out of repeated pairs. Pad
        # deterministically so vocab_size is EXACTLY what 14.4's embedding table
        # expects; put the fillers BEFORE the real specials so <|bos|> ..
        # <|tool_result|> keep the top 9 ids no matter what.
        shortfall = num_merges - len(self.merges)
        if shortfall > 0:
            warnings.warn(
                f"corpus exhausted after {len(self.merges)} merges "
                f"({shortfall} short of {num_merges}); padding with "
                f"{shortfall} <|unused_N|> tokens so vocab_size == {vocab_size}. "
                f"Use a larger sample for a real {vocab_size}-entry vocabulary.")
        fillers = tuple(f"<|unused_{i}|>" for i in range(shortfall))
        self._assign_ids(fillers + tuple(special_tokens))
        assert self.vocab_size == vocab_size
        return shortfall

    def train(self, text: str, vocab_size: int = VOCAB_SIZE,
              special_tokens: Tuple[str, ...] = SPECIAL_TOKENS) -> int:
        """Convenience wrapper for a single in-memory string (tests, demos)."""
        return self.train_from_iterable([text], vocab_size, special_tokens)

    # -- encode / decode ------------------------------------------------------
    def _apply_merges(self, symbols: List[int]) -> List[int]:
        """Repeatedly merge the pair with the LOWEST id (== earliest-learned ==
        highest priority) until no learned pair applies -- the standard
        rank-priority BPE encode loop."""
        while len(symbols) >= 2:
            best_pair, best_id = None, None
            for a, b in zip(symbols, symbols[1:]):
                mid = self.merge_id.get((a, b))
                if mid is not None and (best_id is None or mid < best_id):
                    best_pair, best_id = (a, b), mid
            if best_pair is None:
                break
            merged, i = [], 0
            while i < len(symbols):
                if (i < len(symbols) - 1
                        and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]):
                    merged.append(best_id)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return symbols

    def _encode_chunk(self, byte_ids: Tuple[int, ...]) -> List[int]:
        """Memoized per-pre-token encode. Pre-token frequencies are Zipfian, so
        this hits far more often than it misses (measured below)."""
        out = self._cache.get(byte_ids)
        if out is None:
            out = self._cache[byte_ids] = self._apply_merges(list(byte_ids))
        return out

    def encode(self, text: str,
               allowed_special: Union[str, frozenset] = frozenset(),
               add_special_tokens: bool = False) -> List[int]:
        """Special-token strings inside `text` are treated as ORDINARY BYTES by
        default. Opt in from trusted call sites only, via any of:
            allowed_special={"<|bos|>"}     -- an explicit set
            allowed_special="all"           -- tiktoken's sentinel
            add_special_tokens=True         -- the alias 14.9/14.10/14.11 use
        """
        if add_special_tokens or allowed_special == "all":
            allowed = frozenset(self.special_to_id)
        else:
            allowed = frozenset(allowed_special)

        if allowed:
            # longest-first so a specials set containing overlapping strings
            # cannot be split by a shorter alternative
            pattern = "(" + "|".join(_re.escape(s) for s in
                                     sorted(allowed, key=len, reverse=True)) + ")"
            segments = _re.split(pattern, text)     # keeps the special literals
        else:
            segments = [text]

        ids: List[int] = []
        for seg in segments:
            if seg in allowed:
                ids.append(self.special_to_id[seg])
                continue
            for byte_ids in self._pretokenize(seg):
                ids.extend(self._encode_chunk(byte_ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Expand every id back to raw bytes by walking the merge tree: a merged
        id's two children are themselves either smaller merge ids or raw byte
        ids, so we recurse (via an explicit stack) until only bytes 0..255
        remain, then decode UTF-8 with a safe fallback."""
        out = bytearray()
        for i in ids:
            if i in self.id_to_special:
                out += self.id_to_special[i].encode("utf-8")
                continue
            stack = [i]
            while stack:
                s = stack.pop()
                if s < NUM_BYTES:
                    out.append(s)
                else:
                    a, b = self.id_to_pair[s]
                    stack.extend([b, a])          # push reversed so `a` pops first
        # errors="replace": a lone invalid byte sequence never crashes decode,
        # it renders as the U+FFFD replacement character.
        return bytes(out).decode("utf-8", errors="replace")

    def token_bytes(self) -> List[bytes]:
        """The literal byte string of every non-special id, by walking the merge
        tree bottom-up. This is what the tiktoken / HF exporters need."""
        table = [bytes([i]) for i in range(NUM_BYTES)]
        for a, b in self.merges:
            table.append(table[a] + table[b])      # children always have lower ids
        return table

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "format": "stacklm-bpe-v1",
            "vocab_size": self.vocab_size,
            "pattern": self.pattern,          # the regex is PART of the artifact
            "merges": [list(p) for p in self.merges],
            "special_tokens": list(self.special_to_id.keys()),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "StackTokenizer":
        with open(path) as f:
            payload = json.load(f)
        pattern = payload.get("pattern", SPLIT_PATTERN)
        tok = cls(pattern=pattern)            # compile the SAVED regex, not ours
        if pattern != SPLIT_PATTERN:
            warnings.warn("artifact pre-tokenizer pattern differs from this "
                          "module's SPLIT_PATTERN; using the artifact's, because "
                          "the merges were trained against it.")
        tok.merges = [tuple(p) for p in payload["merges"]]
        tok._assign_ids(tuple(payload["special_tokens"]))
        assert tok.vocab_size == payload["vocab_size"]
        return tok


def load_tokenizer(path: str = "tokenizer/stack100m-32768.json") -> StackTokenizer:
    """The entry point Ch. 14.2's packer and Ch. 14.9's chat template call."""
    return StackTokenizer.load(path)
# --- end verbatim (block #6) ---

print("StackTokenizer class defined")


print()
print("=" * 70)
print("Block #7 (line ~573): capstone/stacklm/data/pack.py -- the Tokenizer Protocol")
print("=" * 70)

# --- verbatim from the chapter (capstone/stacklm/data/pack.py) ---
# from typing import List, Protocol, runtime_checkable   # GLUE: already imported at module top

@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int
    bos_id: int; eos_id: int; pad_id: int
    def encode(self, text: str, allowed_special=frozenset(),
               add_special_tokens: bool = False) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
    def special_token_id(self, token_str: str) -> int: ...
# --- end verbatim (block #7) ---

print("Tokenizer Protocol defined; instantiated/used below once a real "
      "StackTokenizer exists (see the isinstance check after block #15)")


print()
print("=" * 70)
print("Block #8 (line ~615): capstone/stacklm/tokenizer/parallel.py")
print("=" * 70)

# --- verbatim from the chapter (capstone/stacklm/tokenizer/parallel.py) ---
"""Parallel corpus encoding. Documents are independent, so this is a pure map:
one worker per core, each with its own tokenizer + its own chunk cache.

Split shards on <|eos|> boundaries, never mid-document -- a document cut in half
would be pre-tokenized differently on each side of the cut.
"""
# from __future__ import annotations                       # GLUE: already imported at module top
# import multiprocessing as mp                              # GLUE: already imported at module top
# from typing import Iterable, Iterator, List                # GLUE: already imported at module top
# import numpy as np                                          # GLUE: already imported at module top

# from stacklm.tokenizer.bpe import StackTokenizer, load_tokenizer
# GLUE: StackTokenizer/load_tokenizer are block #6's names, already in this
# module's global namespace -- the cross-module import above is dropped.

_TOK: "StackTokenizer | None" = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOK
    _TOK = load_tokenizer(tokenizer_path)      # each worker gets its own cache


def _encode_doc(text: str) -> List[int]:
    assert _TOK is not None
    # Untrusted corpus text: NEVER allow special strings through (see pitfall).
    # The <|bos|>/<|eos|> wrapper is added by code, in the packer.
    return _TOK.encode(text, allowed_special=frozenset())


def encode_corpus(docs: Iterable[str], tokenizer_path: str,
                  workers: int = 16, chunksize: int = 8) -> Iterator[np.ndarray]:
    """Yield one uint16 array per document, in input order."""
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(tokenizer_path,)) as pool:
        for ids in pool.imap(_encode_doc, docs, chunksize=chunksize):
            yield np.asarray(ids, dtype=np.uint16)   # vocab 32768 < 65536, fits
# --- end verbatim (block #8) ---

print("encode_corpus / _init_worker / _encode_doc defined; exercised below "
      "once a real tokenizer artifact exists on disk (block #15's toy tokenizer)")


print()
print("=" * 70)
print("Block #14 (line ~896): capstone/scripts/train_tokenizer.py")
print("=" * 70)

# --- verbatim from the chapter (capstone/scripts/train_tokenizer.py) ---
"""
Trains Stack-100M's tokenizer on a SAMPLE of the pretraining mix (Ch. 14.2) and
exports it into every format the rest of Part XIV needs.

Memory note: peak RSS is O(distinct pre-token chunks), not O(corpus bytes). On
this book's 8.34 MB manuscript the Counter holds 51,340 distinct chunks and the
whole process stays near 300 MB RSS; distinct-chunk count grows far slower than
corpus size for natural text. A few hundred MB of sample is comfortable on a
16 GB laptop; past a few GB, switch to the HF `tokenizers` trainer.
"""
# import glob                                                     # GLUE: already imported at module top
# from typing import Iterator                                     # GLUE: already imported at module top

# from stacklm.tokenizer.bpe import StackTokenizer, VOCAB_SIZE, SPECIAL_TOKENS
# from stacklm.tokenizer.export import save_pretrained
# GLUE: StackTokenizer/VOCAB_SIZE/SPECIAL_TOKENS are already in scope (block
# #5/#6). `save_pretrained` needs `transformers`+`tokenizers` (blocked in CI,
# see block #9's SKIP note above) -- the one call to it below is skipped, not
# faked; everything else in this block runs against the book's real logic.

CHUNK = 8 << 20     # read 8 MiB at a time; never f.read() a whole shard


def stream_sample(paths_glob: str, max_bytes: int = 500_000_000) -> Iterator[str]:
    """Yield bounded text chunks from raw-text shards, stopping at EXACTLY the
    byte budget (mid-file if necessary) rather than after whichever file
    happened to cross it."""
    total = 0
    for path in sorted(glob.glob(paths_glob)):
        with open(path, "r", encoding="utf-8") as f:
            while total < max_bytes:
                block = f.read(CHUNK)
                if not block:
                    break
                raw = block.encode("utf-8")
                if total + len(raw) <= max_bytes:
                    total += len(raw)
                    yield block
                    continue
                # Last block: slice to the exact REMAINING budget, back off to
                # the last whitespace so we don't cut a word, and decode with
                # errors="ignore" so a split UTF-8 sequence is dropped, not
                # turned into U+FFFD (which would pollute the merge table).
                keep = raw[: max_bytes - total]
                cut = keep.rfind(b" ")
                if cut > 0:
                    keep = keep[:cut]
                total = max_bytes
                yield keep.decode("utf-8", errors="ignore")
                break
        if total >= max_bytes:
            break


# GLUE: the book's `if __name__ == "__main__":` block below points at
# "data/mix_sample/*.txt" (doesn't exist here) and vocab_size=VOCAB_SIZE
# (32,768 -- needs a real multi-hundred-MB sample to fill without shortfall,
# per the chapter's own "Training at scale" section). We reproduce the EXACT
# same call sequence and assertions against a tiny generated fixture corpus
# and a proportionally small vocab_size instead, small enough to run in CI in
# well under a second yet still exercise `stream_sample` -> `train_from_iterable`
# -> the shortfall/vocab_size/tool_result-id assertions -> `tok.save`.
if True:
    import random as _random
    _random.seed(0)
    _WORDS_POOL_14 = ("the quick brown fox jumps over lazy dog near riverbank "
                       "clever cat watches quietly distance wondering strange "
                       "behavior animals forest morning light birds sing songs "
                       "spring rivers flow gently downstream past old stone "
                       "bridges wooden mills creak wind sun moon stars sky "
                       "cloud rain snow mountain valley ocean sea lake tree "
                       "leaf branch root flower garden path road house village "
                       "town city street market square").split()
    _fixture_text_14 = " ".join(_random.choice(_WORDS_POOL_14) for _ in range(4000))

    _fixture_dir_14 = tempfile.mkdtemp(prefix="mix_sample_")
    _half = len(_fixture_text_14) // 2
    with open(os.path.join(_fixture_dir_14, "shard_a.txt"), "w", encoding="utf-8") as f:
        f.write(_fixture_text_14[:_half])
    with open(os.path.join(_fixture_dir_14, "shard_b.txt"), "w", encoding="utf-8") as f:
        f.write(_fixture_text_14[_half:])

    # 150 merges is comfortably below this fixture corpus's ~216-merge
    # saturation point (measured), so shortfall == 0 holds deterministically,
    # exactly as the chapter's production run expects on a real sample.
    _TOY_VOCAB_SIZE_14 = 256 + 150 + len(SPECIAL_TOKENS)   # = 415

    tok = StackTokenizer()
    shortfall = tok.train_from_iterable(
        stream_sample(os.path.join(_fixture_dir_14, "*.txt"), max_bytes=500_000_000),
        vocab_size=_TOY_VOCAB_SIZE_14, special_tokens=SPECIAL_TOKENS)
    assert shortfall == 0, "sample too small to fill the fixture vocabulary -- enlarge it"
    assert tok.vocab_size == _TOY_VOCAB_SIZE_14     # 14.4 hardcodes nn.Embedding(V, 512)
    assert tok.special_to_id["<|tool_result|>"] == _TOY_VOCAB_SIZE_14 - 1

    _artifact_path_14 = os.path.join(_fixture_dir_14, "stack100m-toy.json")
    tok.save(_artifact_path_14)             # from-scratch artifact
    # save_pretrained(tok, ...)             # SKIP(optional-dependency): needs transformers
    print(f"trained {len(tok.merges)} merges, vocab_size={tok.vocab_size}")

# Exercise stream_sample's own core promise directly: it must never yield more
# bytes than the requested budget, and must cut mid-file rather than only
# mid-corpus, across MULTIPLE files.
_budget_dir_14 = tempfile.mkdtemp(prefix="budget_sample_")
with open(os.path.join(_budget_dir_14, "a.txt"), "w", encoding="utf-8") as f:
    f.write("alpha beta gamma delta epsilon " * 500)      # ~16.5 KB
with open(os.path.join(_budget_dir_14, "b.txt"), "w", encoding="utf-8") as f:
    f.write("zeta eta theta iota kappa " * 500)            # ~13 KB
_small_budget = 10_000     # smaller than either file alone -> must cut mid-file
_chunks = list(stream_sample(os.path.join(_budget_dir_14, "*.txt"), max_bytes=_small_budget))
_total_bytes = sum(len(c.encode("utf-8")) for c in _chunks)
assert _total_bytes <= _small_budget, "stream_sample exceeded its byte budget"
assert _total_bytes > _small_budget - CHUNK, "stream_sample stopped far short of its budget"
assert len(_chunks) >= 1
print(f"stream_sample respected a {_small_budget}-byte budget "
      f"(yielded {_total_bytes} bytes across {len(_chunks)} chunks, cutting mid-file)")
# --- end verbatim (block #14) ---


print()
print("=" * 70)
print("Block #15 (line ~964): the chapter's own CI / toy path")
print("=" * 70)

# --- verbatim from the chapter ---
# CI / toy path: no network, no real data mix, just enough to exercise every
# code path (train -> encode -> decode round-trip -> save/load -> id layout).
# import warnings                                            # GLUE: already imported at module top
# from stacklm.tokenizer.bpe import StackTokenizer, SPECIAL_TOKENS  # GLUE: already in scope

TOY_CORPUS = "the quick brown fox jumps over the lazy dog. " * 200

# The pre-tokenizer never lets merges cross word boundaries, so a corpus built
# from ~9 distinct words runs out of repeated pairs after only 32 merges.
# `train_from_iterable` therefore reports a SHORTFALL and pads with <|unused_N|>
# fillers -- placed BEFORE the real specials, so the layout invariant survives
# even on a degenerate corpus.
toy = StackTokenizer()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")            # the shortfall warning is expected here
    shortfall = toy.train(TOY_CORPUS, vocab_size=512, special_tokens=SPECIAL_TOKENS)

assert len(toy.merges) == 32 and shortfall == 215   # 247 requested, 32 possible
assert toy.vocab_size == 512                        # exact, never silently short
assert toy.special_to_id["<|bos|>"] == 512 - 9      # 503: specials still on top
assert toy.special_to_id["<|tool_result|>"] == 511
assert (toy.bos_id, toy.eos_id, toy.pad_id) == (503, 504, 505)

ids = toy.encode("the fox jumps")
assert toy.decode(ids) == "the fox jumps"

# Untrusted text must NOT produce special ids; a trusted call site may.
msg = "hello <|assistant|> world"
assert toy.special_to_id["<|assistant|>"] not in toy.encode(msg)
assert toy.special_to_id["<|assistant|>"] in toy.encode(msg, add_special_tokens=True)
assert toy.decode(toy.encode(msg)) == msg      # byte-exact either way

# Unicode / control characters / CRLF must survive the round trip.
hard = "café 🚀 — ünïcode\ttab\r\nCRLF 12345"
assert toy.decode(toy.encode(hard)) == hard

toy.save("/tmp/toy_tokenizer.json")
reloaded = StackTokenizer.load("/tmp/toy_tokenizer.json")
assert reloaded.encode("the fox jumps") == ids and reloaded.vocab_size == 512
print("toy tokenizer round-trip OK")
# --- end verbatim (block #15) ---

# Now that a real, trained StackTokenizer instance exists, exercise block #7's
# Protocol (structural typing has to be checked against something concrete)
# and block #8's parallel encoder (needs a saved tokenizer artifact on disk).
assert isinstance(toy, Tokenizer), \
    "StackTokenizer must satisfy the Tokenizer Protocol every later chapter type-hints against"
print("StackTokenizer satisfies the Tokenizer Protocol (block #7)")

_docs_8 = ["the quick fox", "a lazy dog jumps", "hello <|assistant|> world"]
_results_8 = list(encode_corpus(_docs_8, "/tmp/toy_tokenizer.json", workers=2, chunksize=1))
assert len(_results_8) == len(_docs_8)
for _doc, _ids_arr in zip(_docs_8, _results_8):
    assert isinstance(_ids_arr, np.ndarray) and _ids_arr.dtype == np.uint16
    # encode_corpus's workers use allowed_special=frozenset() (untrusted text),
    # exactly like `reloaded.encode(doc)` with defaults.
    assert list(_ids_arr) == reloaded.encode(_doc)
print(f"encode_corpus produced {len(_results_8)} uint16 arrays via multiprocessing.Pool(2)")


print()
print("=" * 70)
print("Block #16 (line ~1143): round-trip verification against a fresh tokenizer")
print("=" * 70)

# --- verbatim from the chapter ---
# Round-trip verification against a freshly-trained tokenizer.
# from stacklm.tokenizer.bpe import StackTokenizer, VOCAB_SIZE   # GLUE: already in scope

# GLUE: the book opens "corpus_sample.txt" by a bare relative path and trains
# at the real VOCAB_SIZE == 32768. We create that exact relative-path fixture
# file in a throwaway cwd (chdir'd back afterwards) and keep VOCAB_SIZE as
# the module constant -- the corpus is just too small to fill it, so
# train_from_iterable pads with fillers and warns, precisely as designed; the
# block itself never asserts shortfall == 0, so this is a faithful run.
_prev_cwd_16 = os.getcwd()
_work_dir_16 = tempfile.mkdtemp(prefix="corpus_sample_")
os.chdir(_work_dir_16)
try:
    _corpus_text_16 = (
        "# Tokenization Notes\n\n"
        "Tokenization is the most underestimated component of the stack. "
        "It is not part of the network, it is not trained by gradient descent, "
        "and it is frozen for the entire life of the model.\n\n"
        "café 🚀 — mixed ünïcode text, CRLF line endings\r\n and tabs\ttoo.\n\n"
        "```python\n"
        "def f(x):\n"
        "    return x ** 2  # 1234567 and 2026\n"
        "```\n\n"
    ) * 60  # comfortably over 6000 chars for the corpus[1000:6000] slice below
    with open("corpus_sample.txt", "w", encoding="utf-8") as f:
        f.write(_corpus_text_16)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # shortfall warning expected: tiny fixture corpus

        tok = StackTokenizer()
        tok.train_from_iterable(open("corpus_sample.txt", encoding="utf-8"),
                                vocab_size=VOCAB_SIZE)

        sample = ("Tokenization is the most underestimated component of the stack. "
                  "It is not part of the network, it is not trained by gradient descent, "
                  "and it is frozen for the entire life of the model.")
        assert tok.decode(tok.encode(sample)) == sample          # exact byte-for-byte match

        # 1. The pre-tokenizer must be LOSSLESS or nothing downstream can hold.
        corpus = open("corpus_sample.txt", encoding="utf-8").read()
        assert "".join(tok._split_re.findall(corpus)) == corpus

        # 2. Spot-check on a real corpus slice -- Unicode, markdown, code fences.
        assert tok.decode(tok.encode(corpus[1000:6000])) == corpus[1000:6000]

        # 3. Adversarial round trips the happy path misses.
        for hard in ["café 🚀 — ünïcode", "a\r\nb\tc", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "", " ",
                     "<|assistant|> literal in untrusted text", "0123456789" * 40]:
            assert tok.decode(tok.encode(hard)) == hard

    print("round trip OK on prose, corpus slice, and adversarial strings")
finally:
    os.chdir(_prev_cwd_16)
# --- end verbatim (block #16) ---


print()
print("=" * 70)
print("Block #20 (line ~1367): Exercise 8's lazy-heap-bug regression test")
print("=" * 70)

# --- verbatim from the chapter ---
# from stacklm.tokenizer.bpe import train_bpe   # GLUE: already in scope (block #5)

# symbols: a=97 b=98 c=99 d=100 (raw byte ids)
word_freqs = {
    (99, 97, 98): 6,        # "cab": contains (97,98) and (99,97)
    (100, 97, 98): 4,       # "dab": contains (97,98) and (100,97)
    (99, 97): 5,            # "ca":  boosts (99,97) so it merges FIRST
}
merges = train_bpe(word_freqs, num_merges=4)
# (99,97) has count 6+5 = 11 -> merged first, which destroys (97,98) inside
# "cab" and drops pair_counts[(97,98)] from 10 to 4 with no re-push.
assert (97, 98) in merges, "decremented pair was silently dropped from the heap"
# --- end verbatim (block #20) ---

print("lazy-heap fix verified: (97, 98) survives in", merges)


print()
print("=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)
