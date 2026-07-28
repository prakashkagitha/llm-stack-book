r"""From-scratch byte-level BPE (Ch. 14.3), PURE stdlib.

The chapter uses the third-party `regex` module for the GPT-2 split pattern
(`\p{L}`, `\p{N}`). CI does not have `regex`, so we use stdlib `re` with
`[^\W\d_]` (letters) and `\d` (numbers) under Unicode matching -- behaviour is
close to GPT-2's for ordinary text.

Nine special tokens are reserved up front (order is load-bearing: it fixes ids):
bos/eos/pad, the chat roles, and the tool tokens used in Ch. 14.9 / 14.10.
"""
import json
import heapq
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

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

VOCAB_SIZE = 32768
NUM_BYTES = 256

# stdlib-`re` port of the GPT-2 split pattern (\p{L}->[^\W\d_], \p{N}->\d).
_SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _pretokenize_to_byte_ids(text: str) -> List[Tuple[int, ...]]:
    return [tuple(chunk.encode("utf-8")) for chunk in _SPLIT_PATTERN.findall(text)]


def train_bpe(word_freqs: Dict[Tuple[int, ...], int], num_merges: int
              ) -> List[Tuple[int, int]]:
    """Greedy BPE merge training with an incremental, lazy-deletion max-heap."""
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

        affected = list(pair_to_words.get(pair, ()))
        for wi in affected:
            word = words[wi]
            if not any(word[i] == pair[0] and word[i + 1] == pair[1]
                       for i in range(len(word) - 1)):
                continue

            f = freqs[wi]
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] -= f

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
                heapq.heappush(heap, (-pair_counts[(a, b)], (a, b)))

        pair_to_words.pop(pair, None)
        pair_counts.pop(pair, None)

    return merges


class StackTokenizer:
    def __init__(self) -> None:
        self.merges: List[Tuple[int, int]] = []
        self.merge_id: Dict[Tuple[int, int], int] = {}
        self.id_to_pair: Dict[int, Tuple[int, int]] = {}
        self.special_to_id: Dict[str, int] = {}
        self.id_to_special: Dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return NUM_BYTES + len(self.merges) + len(self.special_to_id)

    # --- special-token id accessors (needed by the data packer + chat template) ---
    def special_token_id(self, token_str: str) -> int:
        return self.special_to_id[token_str]

    def id(self, token_str: str) -> int:
        return self.special_to_id[token_str]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    def train(self, text: str, vocab_size: int = VOCAB_SIZE,
              special_tokens: Tuple[str, ...] = SPECIAL_TOKENS) -> None:
        num_merges = vocab_size - NUM_BYTES - len(special_tokens)
        word_freqs = Counter(_pretokenize_to_byte_ids(text))
        self.merges = train_bpe(word_freqs, num_merges)
        self.merge_id = {pair: NUM_BYTES + i for i, pair in enumerate(self.merges)}
        self.id_to_pair = {v: k for k, v in self.merge_id.items()}

        next_id = NUM_BYTES + len(self.merges)
        self.special_to_id, self.id_to_special = {}, {}
        for special_str in special_tokens:
            self.special_to_id[special_str] = next_id
            self.id_to_special[next_id] = special_str
            next_id += 1

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

    def encode(self, text: str, allowed_special: frozenset = frozenset(),
               add_special_tokens: bool = False) -> List[int]:
        """`add_special_tokens` is accepted for API compatibility with the chat
        template (which passes literal `<|...|>` strings). When True, every
        reserved special token is treated as atomic."""
        allowed = allowed_special
        if add_special_tokens:
            allowed = frozenset(self.special_to_id.keys())

        if allowed:
            pattern = "(" + "|".join(re.escape(s) for s in allowed) + ")"
            segments = re.split(pattern, text)     # keeps the special literals
        else:
            segments = [text]

        ids: List[int] = []
        for seg in segments:
            if seg in allowed:
                ids.append(self.special_to_id[seg])
                continue
            for byte_ids in _pretokenize_to_byte_ids(seg):
                ids.extend(self._apply_merges(list(byte_ids)))
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

    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,                        # list of [a, b] int pairs
            "special_tokens": list(self.special_to_id.keys()),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "StackTokenizer":
        with open(path) as f:
            payload = json.load(f)
        tok = cls()
        tok.merges = [tuple(p) for p in payload["merges"]]
        tok.merge_id = {pair: NUM_BYTES + i for i, pair in enumerate(tok.merges)}
        tok.id_to_pair = {v: k for k, v in tok.merge_id.items()}
        next_id = NUM_BYTES + len(tok.merges)
        for t in payload["special_tokens"]:
            tok.special_to_id[t] = next_id
            tok.id_to_special[next_id] = t
            next_id += 1
        return tok


def load_tokenizer(path: str) -> StackTokenizer:
    """Thin wrapper Ch. 14.2 refers to."""
    return StackTokenizer.load(path)


def bytes_per_token(tok: StackTokenizer, text: str) -> float:
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tok.encode(text))
    return n_bytes / max(1, n_tokens)
