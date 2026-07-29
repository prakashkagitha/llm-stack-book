r"""Stack-100M's tokenizer (Ch. 14.3): from-scratch byte-level BPE trainer +
encoder/decoder, engineered to finish ~32.5k merges on a multi-megabyte sample
in seconds.

Pre-tokenizer: a cl100k/Llama-3-style pattern with digit runs capped at THREE
(GPT-2's unbounded ` ?\p{N}+` makes numeric segmentation content-dependent,
which sabotages the RLVR arithmetic task in Ch. 14.9). `\p{L}`/`\p{N}` need the
third-party `regex` package; the book's CI is hermetic and stdlib-only, so we
fall back to a documented stdlib approximation with the SAME digit cap. The
pattern actually used is written into the artifact and re-compiled on load, so
the two flavours can never be silently mixed.

Nine special tokens are reserved up front (order is load-bearing: it fixes ids):
bos/eos/pad, the chat roles, and the tool tokens used in Ch. 14.9 / 14.10. Once
this tokenizer initializes an embedding table the vocabulary is FROZEN.
"""
from __future__ import annotations

import heapq
import json
import warnings
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Union

try:                        # `regex` gives us \p{L} / \p{N}
    import regex as _re
    _HAVE_REGEX = True
except ImportError:         # hermetic CI: stdlib approximation, same digit cap
    import re as _re
    _HAVE_REGEX = False

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
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"       # contractions, case-insensitive
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"          # a word, optionally preceded by one non-letter
    r"|\p{N}{1,3}"                        # <= 3 digits, NEVER absorbing a leading space
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"         # punctuation runs, trailing newlines attached
    r"|\s*[\r\n]+"                        # newline runs (indentation-friendly)
    r"|\s+(?!\S)"                         # trailing whitespace at end of a run
    r"|\s+"                               # any remaining whitespace
)
# stdlib-`re` port: \p{L} ~ [^\W\d_], \p{N} ~ \d (differs only on Nl/No, e.g.
# Roman numerals and vulgar fractions, which land in the punctuation branch).
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


def _pretokenize_to_byte_ids(text: str, split_re=None) -> List[Tuple[int, ...]]:
    """Split `text` into pre-token chunks, each a tuple of raw UTF-8 byte VALUES
    (0..255) -- e.g. 'cafe' -> (99, 97, 102, 101)."""
    split_re = split_re if split_re is not None else _SPLIT_RE
    return [tuple(chunk.encode("utf-8")) for chunk in split_re.findall(text)]


def train_bpe(word_freqs: Dict[Tuple[int, ...], int],
              num_merges: int) -> List[Tuple[int, int]]:
    """Greedy BPE merge training with an incremental, lazy-deletion max-heap.

    Naively, each merge rescans the ENTIRE corpus -- O(num_merges x corpus).
    Instead we keep `pair_counts` (running counts), `pair_to_words` (inverted
    index), and a max-heap of candidates.

    LAZY DELETION is only CORRECT if every count change re-pushes a fresh entry.
    Pushing only on increments leaves decremented pairs with nothing but
    too-high (stale) entries, which the staleness check discards -- silently
    dropping those pairs forever. See Ch. 14.3's "lazy deletion" pitfall box.
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
            # may already have removed `pair` from this word.
            if not any(word[i] == pair[0] and word[i + 1] == pair[1]
                       for i in range(len(word) - 1)):
                continue

            f = freqs[wi]
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] -= f
                touched.add((a, b))

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

            for a, b in zip(merged, merged[1:]):
                pair_counts[(a, b)] += f
                pair_to_words[(a, b)].add(wi)
                touched.add((a, b))

        pair_to_words.pop(pair, None)
        pair_counts.pop(pair, None)
        touched.discard(pair)

        # THE FIX: refresh the heap for every pair whose count moved, up OR down.
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))

    return merges


def train_bpe_naive(word_freqs: Dict[Tuple[int, ...], int],
                    num_merges: int) -> List[Tuple[int, int]]:
    """O(merges x corpus) oracle used by the tests: recount every pair from
    scratch each round. No heap, so it cannot have the lazy-deletion bug. Ties
    are broken by the lexicographically smallest pair, matching what heapq does
    when it pops the smallest (-count, pair) tuple."""
    words = [list(w) for w in word_freqs.keys()]
    freqs = list(word_freqs.values())
    merges: List[Tuple[int, int]] = []
    next_id = NUM_BYTES
    while len(merges) < num_merges:
        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for word, f in zip(words, freqs):
            for a, b in zip(word, word[1:]):
                counts[(a, b)] += f
        if not counts:
            break
        best = max(counts.values())
        if best < 2:
            break
        pair = min(p for p, c in counts.items() if c == best)
        merges.append(pair)
        nid = next_id
        next_id += 1
        for wi, word in enumerate(words):
            merged, i = [], 0
            while i < len(word):
                if (i < len(word) - 1 and word[i] == pair[0]
                        and word[i + 1] == pair[1]):
                    merged.append(nid)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[wi] = merged
    return merges


class StackTokenizer:
    """Byte-level BPE tokenizer for Stack-100M.

    ID layout (fixed for the whole project):
        0        .. 255           raw byte values (id == byte value)
        256      .. 256+M-1       learned merges, in training order
        256+M    .. vocab_size-1  filler <|unused_N|> (only if the sample was too
                                  small), then the 9 SPECIAL_TOKENS in order,
                                  ALWAYS occupying the final 9 ids.
    """

    def __init__(self, pattern: str = SPLIT_PATTERN) -> None:
        self.merges: List[Tuple[int, int]] = []
        self.merge_id: Dict[Tuple[int, int], int] = {}
        self.id_to_pair: Dict[int, Tuple[int, int]] = {}
        self.special_to_id: Dict[str, int] = {}
        self.id_to_special: Dict[int, str] = {}
        self.pattern: str = pattern                          # frozen with the merges
        self._split_re = _re.compile(pattern)                # BOUND, not the global
        self._cache: Dict[Tuple[int, ...], List[int]] = {}

    @property
    def vocab_size(self) -> int:
        return NUM_BYTES + len(self.merges) + len(self.special_to_id)

    # --- the API surface every later chapter depends on (see Ch. 14.3) ---
    def special_token_id(self, token_str: str) -> int:
        return self.special_to_id[token_str]

    id = special_token_id                    # short alias used in 14.10 / 14.11

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    def _pretokenize(self, text: str) -> List[Tuple[int, ...]]:
        """Uses the INSTANCE's compiled pattern, so a loaded artifact tokenizes
        under the regex it was trained with, not the module default."""
        return [tuple(chunk.encode("utf-8")) for chunk in self._split_re.findall(text)]

    # --- training ---------------------------------------------------------
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
        # deterministically so vocab_size is EXACTLY what Ch. 14.4's embedding
        # table expects; fillers go BEFORE the real specials so <|bos|> ..
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

    # --- encode / decode --------------------------------------------------
    def _apply_merges(self, symbols: List[int]) -> List[int]:
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
        """Memoized per-pre-token encode; pre-token frequencies are Zipfian."""
        out = self._cache.get(byte_ids)
        if out is None:
            out = self._cache[byte_ids] = self._apply_merges(list(byte_ids))
        return out

    def encode(self, text: str,
               allowed_special: Union[str, frozenset] = frozenset(),
               add_special_tokens: bool = False) -> List[int]:
        """Special-token strings inside `text` are ORDINARY BYTES by default.
        Opt in from trusted call sites only, via any of:
            allowed_special={"<|bos|>"}   -- an explicit set
            allowed_special="all"         -- tiktoken's sentinel
            add_special_tokens=True       -- the alias 14.9/14.10/14.11 use
        """
        if add_special_tokens or allowed_special == "all":
            allowed = frozenset(self.special_to_id)
        else:
            allowed = frozenset(allowed_special)

        if allowed:
            # longest-first so overlapping specials cannot be split by a shorter
            # alternative
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
        return bytes(out).decode("utf-8", errors="replace")

    def token_bytes(self) -> List[bytes]:
        """Literal byte string of every non-special id, walking the merge tree
        bottom-up. This is what the tiktoken / HF exporters need."""
        table = [bytes([i]) for i in range(NUM_BYTES)]
        for a, b in self.merges:
            table.append(table[a] + table[b])      # children always have lower ids
        return table

    # --- persistence ------------------------------------------------------
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


def bytes_per_token(tok: StackTokenizer, text: str) -> float:
    """UTF-8 bytes per token: higher == better compression."""
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text))
    return n_bytes / max(1, n_tokens)
