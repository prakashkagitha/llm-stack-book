# 14.3 A Byte-Level BPE Tokenizer From Scratch (and Why Vocab Size Is a Design Lever at 100M)

Before Stack-100M sees a single training example, before we have written a line of the transformer block, before we know exactly how many layers fit in a 100M-parameter budget, we have to answer a question that sounds administrative but is actually one of the highest-leverage design decisions in the whole project: **what integers does the model see?**

That question is the tokenizer's job. [Chapter 14.2](../14-capstone/02-data-pipeline.html) streams and cleans the ~20B-token data mix; this chapter turns that raw text into the vocabulary Stack-100M will speak for the rest of its life, and [Chapter 14.4](../14-capstone/04-architecture.html) sizes the embedding table — and therefore how many parameters are left over for depth and width — against the number we fix here. Everything downstream depends on getting this chapter right first: the shard format in 14.2, the embedding table in 14.4, the chat template in [Chapter 14.9](../14-capstone/09-post-training.html), and the tool-call format in [Chapter 14.10](../14-capstone/10-agentic-narrow.html) all hardcode assumptions this chapter fixes.

We already built byte-level BPE from first principles in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) — the merge-the-most-frequent-adjacent-pair algorithm, the guarantee that byte-level fallback means no input is ever unrepresentable, the pre-tokenizer regex that keeps merges from crossing word boundaries. We will not re-derive any of that here. This chapter does two things that chapter does not: it engineers a trainer that actually finishes 32,503 merges on a real corpus in seconds rather than hours, and it makes the case — with real parameter arithmetic — that at 100M parameters, **vocabulary size is not a free hyperparameter**. It is a line item that competes directly with depth and width for the same fixed budget.

## Why Stack-100M Trains Its Own Tokenizer

{{tool:tokenizer-playground}}

You could, in principle, reuse an off-the-shelf tokenizer — GPT-2's 50,257-entry vocabulary, or Llama 3's 128k-entry one — and skip this chapter entirely. Three reasons we don't:

1. **Domain match.** Stack-100M's data mix ([Chapter 14.2](../14-capstone/02-data-pipeline.html)) is 70% FineWeb-Edu-style educational web text, 15% Cosmopedia-style synthetic textbooks, 10% code, and 5% math — a narrower, more structured distribution than "the open web" that GPT-2's tokenizer was fit to. A tokenizer trained on our own mix spends its merge budget on substrings that actually recur in *our* corpus, not GPT-2's.
2. **Full control over the ID layout.** We need nine special tokens — beginning/end/pad markers now, chat-role markers for [Chapter 14.9](../14-capstone/09-post-training.html), tool-call markers for [Chapter 14.10](../14-capstone/10-agentic-narrow.html) — reserved at specific, predictable positions before a single row of the embedding table is initialized. Borrowing someone else's tokenizer means inheriting (or awkwardly patching) their special-token layout instead.
3. **Vocabulary size is the whole point of this chapter.** GPT-2's tokenizer was fit for a family of models where the embedding table is a rounding error next to a 1.5B-parameter network. At 100M parameters that is no longer true, and we want a vocabulary size we chose deliberately, not one we inherited.

The good news, which we will demonstrate with real measurements rather than a hand-wave: training a from-scratch, 32,768-entry byte-level BPE tokenizer on a multi-megabyte corpus is a matter of **seconds**, not hours, if you engineer the trainer even a little. There is no excuse not to train your own.

**Where this fits in the pipeline:**

```text
raw text corpus (14.2)
        │
        ▼
  ┌─────────────────┐
  │  THIS CHAPTER    │   train byte-level BPE, vocab_size = 32768
  │  (14.3)          │   reserve 9 special tokens
  └────────┬─────────┘
           │  tokenizer/stack100m-32768.json  (frozen from here on)
           ▼
  ┌─────────────────┐        ┌──────────────────┐
  │ data packing      │       │ architecture (14.4)│
  │ (14.2, uint16     │       │ embedding table    │
  │ .bin shards)      │       │ sized to 32768×512 │
  └───────────────────┘       └──────────────────┘
```

## Reserving Special Tokens Up Front

A tokenizer is trained once by counting statistics, then frozen for the entire life of the model — a point [Chapter 2.1](../02-transformer/01-tokenization.html) makes and that bears repeating here with teeth: once Stack-100M's embedding table has a row for id 32763, that row *is* "the concept of `<|user|>`" for every checkpoint we will ever produce. If we discover three chapters from now, in [Chapter 14.10](../14-capstone/10-agentic-narrow.html), that we need a `<|tool_result|>` token and we forgot to reserve one, we have exactly two bad options: retrain the tokenizer (which invalidates every checkpoint's embedding table — id 4,521 no longer means what it meant yesterday) or grow `vocab_size` after the fact and bolt on new, randomly initialized rows that never received a single gradient step during the ~20B-token pretraining run. Both are avoidable. We reserve every special token Stack-100M will need for its entire lifecycle **now**, in this chapter, even though most of them sit unused for six more chapters.

| Token | id | Purpose | First used in |
|---|---|---|---|
| `<|bos|>` | 32759 | Beginning-of-sequence / start-of-packed-document marker | Data packing (14.2), pretraining (14.7) |
| `<|eos|>` | 32760 | End-of-sequence / end-of-document marker | Data packing (14.2) |
| `<|pad|>` | 32761 | Padding for short batches; always masked out of the loss | SFT ([14.9](../14-capstone/09-post-training.html)) |
| `<|system|>` | 32762 | Chat-role marker: opens a system message | SFT/DPO ([14.9](../14-capstone/09-post-training.html)), [chat templates](../05-posttraining-alignment/02-chat-templates-packing.html) |
| `<|user|>` | 32763 | Chat-role marker: opens a user turn | SFT/DPO ([14.9](../14-capstone/09-post-training.html)) |
| `<|assistant|>` | 32764 | Chat-role marker: opens an assistant turn — loss is computed only on tokens *after* this | SFT/DPO ([14.9](../14-capstone/09-post-training.html)) |
| `<|end|>` | 32765 | Closes any role's turn | SFT/DPO ([14.9](../14-capstone/09-post-training.html)) |
| `<|tool_call|>` | 32766 | Opens a tool invocation the model itself emits | Agent ([14.10](../14-capstone/10-agentic-narrow.html)), [tool use](../08-agents-harness/01-tool-use-function-calling.html) |
| `<|tool_result|>` | 32767 | Opens a tool's returned observation — masked from the loss, the model must never be trained to *predict* an observation it didn't generate | Agent ([14.10](../14-capstone/10-agentic-narrow.html)) |

The layout is simple by design: raw bytes get the first 256 ids, learned merges fill the middle, and the nine specials occupy the top of the range in the exact order listed above.

```text
id:        0 ─────────────── 255   256 ──────────────────── 32758   32759 ── 32767
contents:  raw UTF-8 bytes         32,503 learned BPE merges          9 reserved specials
           (fixed, universal)      (trained on the Stack-100M mix)    (fixed order, see table)
```

With `vocab_size = 32768`, `256` reserved bytes, and `9` special tokens, the number of merges the trainer must learn is fixed:

$$
M = V - 256 - S = 32{,}768 - 256 - 9 = 32{,}503
$$

{{fig:vocab-id-layout-frozen}}

!!! tip "Practitioner tip: reserved slots for the future"
    Some production tokenizers (Meta's Llama 3, for instance) pad their special-token block with dozens of unused `<|reserved_special_token_N|>` placeholders, so a future fine-tune can add a role or a tool format without touching `vocab_size` or reshuffling ids. We don't do that here — Stack-100M's special-token needs are fully enumerated by this table, and every one of the 32,768 rows should either be a real byte/merge or a token we know we will use, so none of the tight 100M-parameter budget is spent on speculative slots. If you extend this project past the capstone's scope, budgeting 8–16 reserved slots is cheap insurance.

## A From-Scratch, Efficient Byte-Level BPE Trainer

The algorithm is unchanged from [Chapter 2.1](../02-transformer/01-tokenization.html): pre-tokenize with the GPT-2 regex so merges never cross word or whitespace boundaries, then repeatedly merge the most frequent adjacent pair of symbols. What changes here is engineering. The naive trainer from the from-scratch chapter recomputes every pair count from scratch after every merge — `O(merges × corpus size)`. At `M = 32{,}503` merges over even a modest multi-megabyte sample, that quickly becomes a multi-hour job. We fix this with two standard data-structure tricks: an **inverted index** from each pair to the word indices that contain it (so a merge only touches the words it actually affects, not the whole corpus), and a **lazy-deleted max-heap** (so "which pair is most frequent right now" is an `O(log n)` heap pop instead of an `O(n)` linear scan).

{{fig:bpe-trainer-incremental-merge}}

```python
# capstone/stacklm/tokenizer.py
"""
Stack-100M's tokenizer: a from-scratch, byte-level BPE trainer + encoder/decoder.

This is the production version of the algorithm built from first principles in
../02-transformer/01-tokenization.html -- same "merge the most frequent adjacent
pair" idea, same byte-level guarantee that no input is ever unrepresentable, but
engineered to finish ~32.5k merges on a real multi-megabyte data sample in
seconds instead of hours (see the timing note in "Training at Scale" below).

Special tokens are reserved UP FRONT (see SPECIAL_TOKENS) even though most of
them are not touched until Ch. 14.9 (SFT/DPO) and Ch. 14.10 (the agent). Once
this tokenizer is trained and Stack-100M's embedding table is initialized
against it, the vocabulary is FROZEN for the rest of the project: id 32763
means "<|user|>" forever, or every checkpoint's embedding table becomes
meaningless.
"""

from __future__ import annotations

import json
import heapq
import regex as re                      # `regex`, not stdlib `re` -- needed for \p{L} / \p{N}
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. Special tokens, reserved up front, in the exact order the rest of the
#    capstone relies on. Never reorder these once a tokenizer has been trained
#    and used to initialize an embedding table -- every later chapter hardcodes
#    these STRINGS (looked up via special_to_id), not raw integer ids, but the
#    ids themselves become baked into every checkpoint's embedding rows.
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

VOCAB_SIZE = 32768                                            # Stack-100M's fixed vocab (PLAN Sec. 3)
NUM_BYTES = 256                                               # every raw byte value is always id 0..255
NUM_MERGES = VOCAB_SIZE - NUM_BYTES - len(SPECIAL_TOKENS)      # = 32,503

# GPT-2's pre-tokenizer regex (Radford et al., 2019 / tiktoken). Splitting on
# this pattern BEFORE any BPE merge runs is what keeps merges from crossing
# word/whitespace/punctuation boundaries -- see Ch. 2.1 for the full derivation.
_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _pretokenize_to_byte_ids(text: str) -> List[Tuple[int, ...]]:
    """Split `text` into GPT-2-style chunks, each returned as a tuple of raw
    UTF-8 byte VALUES (0..255) -- e.g. 'café' -> (99, 97, 102, 195, 169)."""
    return [tuple(chunk.encode("utf-8")) for chunk in _SPLIT_PATTERN.findall(text)]


# ---------------------------------------------------------------------------
# 2. The trainer. Operates on integer symbol ids throughout (0..255 for raw
#    bytes, 256+ for learned merges) instead of the string glyphs used in the
#    from-scratch chapter -- purely a speed choice, the algorithm is identical.
# ---------------------------------------------------------------------------
def train_bpe(word_freqs: Dict[Tuple[int, ...], int], num_merges: int
              ) -> List[Tuple[int, int]]:
    """Learn `num_merges` BPE merges from a corpus reduced to
    (symbol-id-tuple -> frequency) counts.

    Naively, each merge requires rescanning the ENTIRE corpus to recount every
    adjacent pair -- O(num_merges x corpus_size). For 32,503 merges over even a
    modest multi-MB sample this is a multi-hour job. Instead we maintain three
    pieces of running state so each merge touches only the (usually small)
    subset of words that actually contain the pair being merged:

      - `pair_counts`   : running frequency-weighted count of every adjacent pair
      - `pair_to_words` : inverted index, pair -> set of word indices containing it
      - `heap`          : a max-heap (via negated counts) of candidate pairs, so
                           "find the current best pair" is O(log n), not O(n)

    The heap uses LAZY DELETION: we never remove stale entries when a pair's
    count changes, we just leave old (count, pair) tuples in the heap and
    discard them on pop if they no longer match the live count in
    `pair_counts`. This avoids an expensive heap decrease-key operation.
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

        affected = list(pair_to_words.get(pair, ()))
        for wi in affected:
            word = words[wi]
            # `wi` can be a STALE member of pair_to_words[pair]: an earlier
            # merge processed in this same call may already have removed
            # `pair` from this word (we only prune pair_to_words for the pair
            # just merged, not for every pair a word's decrement touches).
            # Verify membership before mutating anything.
            if not any(word[i] == pair[0] and word[i + 1] == pair[1]
                       for i in range(len(word) - 1)):
                continue

            f = freqs[wi]
            # Remove this word's contribution to EVERY pair it currently forms
            # (not just `pair`) -- merging can shift adjacency everywhere in
            # the word, not only at the merge site.
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] -= f

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

            # Re-add this word's contribution under its NEW symbol sequence,
            # pushing a fresh heap entry for anything that changed.
            for a, b in zip(merged, merged[1:]):
                pair_counts[(a, b)] += f
                pair_to_words[(a, b)].add(wi)
                heapq.heappush(heap, (-pair_counts[(a, b)], (a, b)))

        pair_to_words.pop(pair, None)
        pair_counts.pop(pair, None)

    return merges
```

The `StackTokenizer` class wraps this trainer with the vocabulary bookkeeping, an encoder, a decoder, and JSON persistence so the artifact this chapter produces can be loaded by every later chapter without retraining:

```python
# capstone/stacklm/tokenizer.py (continued)

class StackTokenizer:
    """Byte-level BPE tokenizer for Stack-100M.

    ID layout (fixed for the whole project):
        0        .. 255           raw byte values (id == byte value)
        256      .. 256+M-1       learned merges, in the order they were trained
        256+M    .. vocab_size-1  the 9 SPECIAL_TOKENS, in SPECIAL_TOKENS order
    """

    def __init__(self) -> None:
        self.merges: List[Tuple[int, int]] = []             # learned merges, ordered
        self.merge_id: Dict[Tuple[int, int], int] = {}       # pair -> resulting id
        self.id_to_pair: Dict[int, Tuple[int, int]] = {}     # inverse, for decode
        self.special_to_id: Dict[str, int] = {}
        self.id_to_special: Dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return NUM_BYTES + len(self.merges) + len(self.special_to_id)

    # -- training -------------------------------------------------------------
    def train(self, text: str, vocab_size: int = VOCAB_SIZE,
              special_tokens: Tuple[str, ...] = SPECIAL_TOKENS) -> None:
        num_merges = vocab_size - NUM_BYTES - len(special_tokens)
        word_freqs = Counter(_pretokenize_to_byte_ids(text))
        self.merges = train_bpe(word_freqs, num_merges)
        self.merge_id = {pair: NUM_BYTES + i for i, pair in enumerate(self.merges)}
        self.id_to_pair = {v: k for k, v in self.merge_id.items()}

        next_id = NUM_BYTES + len(self.merges)
        for special_str in special_tokens:
            self.special_to_id[special_str] = next_id
            self.id_to_special[next_id] = special_str
            next_id += 1

    # -- encode / decode --------------------------------------------------------
    def _apply_merges(self, symbols: List[int]) -> List[int]:
        """Repeatedly merge the pair with the LOWEST id (== earliest-learned
        == highest priority) until no learned pair applies to what remains --
        the standard rank-priority BPE encode loop."""
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

    def encode(self, text: str, allowed_special: frozenset = frozenset()) -> List[int]:
        if allowed_special:
            pattern = "(" + "|".join(re.escape(s) for s in allowed_special) + ")"
            segments = re.split(pattern, text)     # keeps the special literals
        else:
            segments = [text]

        ids: List[int] = []
        for seg in segments:
            if seg in allowed_special:
                ids.append(self.special_to_id[seg])
                continue
            for byte_ids in _pretokenize_to_byte_ids(seg):
                ids.extend(self._apply_merges(list(byte_ids)))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Expand every id back to raw bytes by walking the merge tree: a
        merged id's two children are themselves either smaller merge ids or
        raw byte ids, so we recurse (via an explicit stack) until only bytes
        0..255 remain, then decode UTF-8 with a safe fallback."""
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

    # -- persistence --------------------------------------------------------------
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
```

Every important design decision from the earlier table shows up in code here: `SPECIAL_TOKENS` is an ordered tuple (not a set — order fixes ids), `train()` always assigns bytes → merges → specials in that sequence, and `decode()` treats special ids as opaque literals rather than trying to expand them through the merge tree.

Note the deliberate default on `encode`: `allowed_special` is an **empty** frozenset unless the caller opts in. That means a literal `<|assistant|>` sitting inside a user's message is *not* recognized as the special token — it is pre-tokenized and byte-encoded like any other text, producing a completely different id sequence than the reserved id 32764. Only trusted call sites (the chat-template formatter in [Chapter 14.9](../14-capstone/09-post-training.html), the packing code in [Chapter 14.2](../14-capstone/02-data-pipeline.html)) pass an explicit `allowed_special` set to turn recognition back on. This is the same design tiktoken adopted, and it exists for a security reason spelled out below.

!!! warning "Common pitfall: special-token injection"
    If `encode` recognized special-token *strings* everywhere by default, any user could type the literal text `<|assistant|>` into a chat box and have it tokenize to the real role-boundary id — letting them forge assistant turns, inject a fake `<|tool_result|>`, or otherwise escape the chat template. This is the tokenizer-level analogue of a prompt-injection attack (see [Security: Prompt Injection, Jailbreaks & Defenses](../12-production-mlops/06-security-prompt-injection.html)). The fix is exactly the default above: **untrusted text is encoded with `allowed_special=frozenset()`**, so special-token *ids* can only ever enter a sequence through code you control, never through content a user or a tool supplied. When you build the SFT loss mask in [Chapter 14.9](../14-capstone/09-post-training.html), this invariant is what lets you trust that every id 32764 in a training example is a real assistant boundary you placed, not a string an example's user turn happened to contain.

## Training Stack-100M's Tokenizer on the Data Mix

In production you point this at a representative sample of the mix from [Chapter 14.2](../14-capstone/02-data-pipeline.html) — you do **not** need the full ~20B-token corpus to fit a good vocabulary, because pair-frequency statistics converge long before that; a few hundred megabytes drawn from the same 70/15/10/5 FineWeb-Edu / Cosmopedia / code / math mix is enough to learn merges that generalize to the rest.

```python
# capstone/scripts/train_tokenizer.py
"""
Trains Stack-100M's tokenizer on a SAMPLE of the pretraining mix (Ch. 14.2).
A few hundred MB, drawn from the same mix the final model trains on, is
sufficient -- pair-frequency statistics saturate well before 20B tokens.
"""
import glob
from stacklm.tokenizer import StackTokenizer, VOCAB_SIZE, SPECIAL_TOKENS


def load_sample(paths_glob: str, max_bytes: int = 500_000_000) -> str:
    """Concatenate raw-text shards up to a byte budget."""
    chunks, total = [], 0
    for path in sorted(glob.glob(paths_glob)):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks.append(text)
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            break
    return "".join(chunks)


if __name__ == "__main__":
    sample = load_sample("data/mix_sample/*.txt", max_bytes=500_000_000)
    tok = StackTokenizer()
    tok.train(sample, vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    tok.save("tokenizer/stack100m-32768.json")
    print(f"trained {len(tok.merges)} merges, vocab_size={tok.vocab_size}")
```

For the hermetic, network-free CI smoke test that [Chapter 14.2](../14-capstone/02-data-pipeline.html) runs on every commit, we shrink both the corpus and the vocabulary so the whole test finishes in well under a second:

```python
# CI / toy path: no network, no real data mix, just enough to exercise every
# code path (train -> encode -> decode round-trip -> save/load).
from stacklm.tokenizer import StackTokenizer, SPECIAL_TOKENS

TOY_CORPUS = "the quick brown fox jumps over the lazy dog. " * 200

# NOTE on vocab_size here: the pre-tokenizer never lets merges cross word
# boundaries (by design -- see Ch. 2.1), so a corpus built from ~9 distinct
# words can only ever produce as many merges as it takes to collapse each
# word down to a single symbol -- after that, `train_bpe`'s early-stop
# (`if live_count < 2: break`) kicks in because nothing left repeats. Asking
# for an arbitrarily large vocab_size on a tiny, low-diversity toy corpus
# would silently stop short, so the toy path picks a target the corpus can
# actually fill; the real training run in the next section, over millions of
# unique words, has no such ceiling anywhere near 32,768.
toy_tok = StackTokenizer()
toy_tok.train(TOY_CORPUS, vocab_size=281, special_tokens=SPECIAL_TOKENS)
assert toy_tok.vocab_size == 281

ids = toy_tok.encode("the fox jumps", allowed_special=frozenset())
assert toy_tok.decode(ids) == "the fox jumps"

toy_tok.save("/tmp/toy_tokenizer.json")
reloaded = StackTokenizer.load("/tmp/toy_tokenizer.json")
assert reloaded.encode("the fox jumps") == ids
print("toy tokenizer round-trip OK")
```

### Training at scale: the real numbers

We ran the exact trainer above — no shortcuts, no pre-built libraries — against a real corpus: roughly 8.16 MB of English technical prose mixed with code and light math (this book's own manuscript text, a reasonable stand-in for the *flavor* of the FineWeb-Edu + Cosmopedia + code + math mix, though obviously not the real 20B-token corpus itself). The corpus pre-tokenized into 40,775 distinct chunks.

Training all `32,503` merges to reach `vocab_size = 32768` took **3.25 seconds**. Pre-tokenizing and counting the initial word frequencies took another 1.23 seconds. This is the payoff of the heap-plus-inverted-index design: a naive full-corpus rescan per merge, on a corpus with 40,775 unique chunks, run 32,503 times, is an entirely different asymptotic regime — comfortably pushed from single-digit seconds into tens of minutes or worse on the same hardware. (Production pipelines more often reach for the Rust-backed HuggingFace `tokenizers` library, which trains the same 32,768-entry vocabulary on the same corpus faster still — comfortably under a second — because it pushes the whole inner loop into compiled, parallel code. But the *algorithm* is identical to what is above, and having a correct from-scratch trainer we understand end to end, rather than a black box, is worth the modest engineering. Reach for the fast library once you understand exactly what it is doing.)

!!! note "Aside: why this is fast enough to just always do"
    A common piece of received wisdom is "training a tokenizer from scratch is expensive, just reuse GPT-2's." At the scale this capstone operates at, that wisdom is stale. A few seconds of CPU time buys full control over vocabulary size and special-token layout — control that matters far more at 100M parameters than it does at 100B, where a few million embedding parameters round to noise.

## Vocab Size Is a Design Lever at 100M Parameters

Here is the argument this chapter exists to make. [Chapter 14.4](../14-capstone/04-architecture.html) works out Stack-100M's full parameter accounting; we only need the headline numbers here. With `d_model = 512`, `n_kv_heads = 2`, `head_dim = 64`, and a SwiGLU MLP at `intermediate = 1408` (see [the modern architecture chapter](../02-transformer/10-modern-arch-improvements.html) and [SwiGLU/gated MLPs](../02-transformer/06-transformer-block.html) for the mechanisms), one transformer block costs:

$$
P_\text{block} = \underbrace{2d^2 + 2\,d\,d_\text{kv}}_{\text{attention (GQA)} \,=\, 655{,}360} + \underbrace{3\,d\,d_\text{ffn}}_{\text{SwiGLU} \,=\, 2{,}162{,}688} \approx 2.82\text{M params}
$$

Here the attention count is the two square projections $W_Q, W_O \in \mathbb{R}^{d\times d}$ plus the two GQA-shrunk projections $W_K, W_V \in \mathbb{R}^{d \times d_\text{kv}}$ with $d_\text{kv} = n_\text{kv} \cdot \text{head\_dim} = 2 \times 64 = 128$, so $2(512)^2 + 2(512)(128) = 524{,}288 + 131{,}072 = 655{,}360$, and the SwiGLU MLP is three $d \times d_\text{ffn}$ matrices, $3(512)(1408) = 2{,}162{,}688$. Thirty of those blocks cost `≈84.6M` params. That number is **independent of vocabulary size** — it is fixed by depth, width, and the GQA/SwiGLU shapes. The embedding table is not:

$$
P_\text{embed} = V \times d_\text{model}
$$

At `V = 32{,}768`, `d_model = 512`: `P_embed = 32{,}768 \times 512 = 16{,}777{,}216 ≈ 16.8`M params, giving Stack-100M's total of `≈84.6M + 16.8M ≈ 101.4`M.

### Tied embeddings: the first lever

Stack-100M **ties** the input embedding and output (unembedding / `lm_head`) weight matrices — the same `V × d_model` matrix both looks up a token's input vector and produces the logit distribution over the vocabulary at the output, following [Press & Wolf, *Using the Output Embedding to Improve Language Models*, 2017](../02-transformer/02-embeddings-input.html). Weight tying is a well-known regularizer (input and output representations are forced to share a coordinate system) and, at small model sizes, a serious parameter saver: without tying, `V × d_model` params are spent twice.

!!! example "Worked example: what tying is worth, in layers"
    Untied, a 32,768-vocabulary embedding pair costs `2 × 16.78M = 33.55M` params. Tied, it costs `16.78M`. The saving — `16.78M` params — is worth
    $$
    \frac{16.78\text{M}}{2.82\text{M / layer}} \approx 5.95 \approx 6 \text{ layers}
    $$
    of additional depth at fixed total parameter count. Put differently: tying the embeddings is worth roughly a fifth more depth than Stack-100M's 30 layers, for free. This is exactly the kind of trade this chapter is about, and it is why every capstone chapter after this one treats "tied embeddings" as non-negotiable, not a minor implementation detail.

{{fig:tied-embeddings-one-matrix-two-jobs}}

### The tradeoff table

Now hold the block architecture fixed (30 layers × 2.82M/layer, i.e. everything [Chapter 14.4](../14-capstone/04-architecture.html) fixes) and ask: against a nominal 100M-parameter budget, how much does the tied embedding table cost at different vocabulary sizes, and how many transformer layers could that difference buy instead?

| `vocab_size` | tied embed params | % of a 100M budget | layers affordable at fixed 100M | vs Stack-100M's 30 |
|---:|---:|---:|---:|---:|
| 8,192 | 4.19M | 4.2% | ~34 | +4 |
| 16,384 | 8.39M | 8.4% | ~33 | +3 |
| **32,768 (Stack-100M)** | **16.78M** | **16.8%** | **~30** | **baseline** |
| 50,257 (GPT-2) | 25.73M | 25.7% | ~26 | −4 |
| 65,536 | 33.55M | 33.6% | ~24 | −6 |
| 100,277 (cl100k_base-scale) | 51.34M | 51.3% | ~17 | −13 |

*(Methodology: `layers affordable` = `(100M − V·512) / 2.82M`, rounded — i.e., holding total parameters at a nominal 100M and letting depth absorb whatever the embedding table doesn't consume. Stack-100M's actual total lands at ≈101.4M, not exactly 100M, which is why the 32,768 row reads "~30" rather than exactly 30 — ordinary engineering rounding, not an error.)*

This is the number that motivates PLAN's headline claim: **a 50,257-entry vocabulary — GPT-2's, the "obvious" default — would eat about a quarter of a 100M-parameter budget**, on par with four entire transformer layers. That is not a subtle effect at this scale; it is the difference between a 26-layer and a 30-layer model, and depth is exactly the axis the "deep-and-thin" philosophy behind Stack-100M's architecture (MobileLLM, [Liu et al., 2024](../14-capstone/04-architecture.html); see also [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for why depth interacts with a model's effective capacity) is trying to protect.

{{fig:vocab-vs-depth-budget-lever}}

### Why 32,768 and not smaller

The table's left side shows the opposite failure mode. Shrinking further — to 8,192 or 16,384 — buys back a few more layers, but costs compression: a smaller vocabulary means fewer frequent substrings get their own merge, so more of every document falls back toward shorter, more numerous tokens. Every token that a bigger vocabulary would have compressed into one piece instead costs two or three, which means more forward passes per document at both training and inference time — directly eating into the "\~200 tokens/param" over-training budget this project is built around (see [Chapter 14.1](../14-capstone/01-overview-and-landscape.html) and [Chapter 14.2](../14-capstone/02-data-pipeline.html)). Byte-level fallback means a tiny vocabulary is never *incorrect* — nothing is ever unrepresentable — but it is *inefficient*: you pay for the missing merges in sequence length, every single training step, for the life of the run.

`32,768` is chosen as the point where the embedding table is a meaningful but not dominant slice of the 100M budget (≈17%, versus ≈26% for 50,257 or ≈4% for 8,192), while staying large enough that a byte-level BPE tokenizer trained on a code-and-math-heavy mix captures the multi-character substrings — common English morphemes, indentation-heavy code idioms, LaTeX-flavored math notation — that make compression efficient. It is also, not incidentally, a friendly number: `2^15`, which packs comfortably into a `uint16` (max 65,535) for the shard format [Chapter 14.2](../14-capstone/02-data-pipeline.html) writes to disk, with more than half the id space to spare.

!!! note "Aside: the softmax bottleneck"
    Tying pushes the *same* `d_model = 512`-dimensional space to serve two jobs: representing "what token is this" on the input side, and discriminating among 32,768 possible next tokens on the output side. Yang, Dai, Salakhutdinov & Cohen's *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model* (2018) shows that a softmax classifier's output distribution is fundamentally rank-limited by the hidden dimension — a narrow, tied model asks a 512-dimensional space to do double duty. In practice this cost is modest next to what tying saves at 100M scale, but it is a real tension, not a free lunch, and it is one more reason vocabulary size cannot be chosen in isolation from `d_model`.

!!! interview "Interview Corner"
    **Q:** You're building a 100M-parameter language model and someone suggests reusing a 100k-token vocabulary "because bigger vocabularies compress better and bigger models use them." What's wrong with that reasoning at this scale, and how would you actually decide the vocabulary size?

    **A:** The reasoning conflates two different regimes. At the scale of a 100B+ parameter frontier model, a 100k-entry tied embedding (≈51M params at `d_model=512`, or proportionally similar at larger `d_model`) is a rounding error against the rest of the network, so the compression benefit — fewer tokens per document, cheaper training and inference — dominates and bigger vocabularies are close to free. At 100M total parameters, that same embedding table is not a rounding error; it can be a third or more of the entire budget, directly trading off against depth and width, which is where a small model's actual capacity lives. The right approach is to fix a total parameter budget first, then treat `vocab_size` as one line item in that budget alongside `n_layers` and `d_model` — build the table in this chapter (embedding cost vs. layers affordable) for your own target size, and pick the vocabulary that leaves the block architecture (and hence effective capacity) as strong as possible while keeping compression efficient enough that you're not paying it back in wasted sequence length. There is no universal answer; the answer is a function of the parameter budget you actually have.

## Worked Example: Compression Ratio and Round-Trip Correctness

We measured the tokenizer trained in the "training at scale" experiment above — 32,768-entry vocabulary, trained on ≈8.16 MB of real text — on three concrete inputs.

!!! example "Worked example: measured compression, three ways"
    **Whole-corpus average.** Encoding the full 8,161,560-byte training corpus produced 2,088,565 tokens:
    $$
    \frac{8{,}161{,}560 \text{ bytes}}{2{,}088{,}565 \text{ tokens}} \approx 3.91 \text{ bytes/token}
    $$

    **Clean English prose** (a 184-byte sentence pulled from this book's own tokenization chapter — "Tokenization is the most underestimated component of the stack. It is not part of the network, it is not trained by gradient descent, and it is frozen for the entire life of the model.") encoded to 39 tokens:
    $$
    \frac{184}{39} \approx 4.72 \text{ bytes/token}
    $$
    higher than the corpus average, because clean natural-language prose (no markdown syntax, no code punctuation) is exactly what BPE compresses best.

    **A short Python snippet** (234 bytes of a `train_bpe`-style function body) encoded to 88 tokens:
    $$
    \frac{234}{88} \approx 2.66 \text{ bytes/token}
    $$
    markedly worse than prose — code's density of punctuation, brackets, and multi-level indentation gives a vocabulary trained on a prose-heavy mix fewer long, reusable substrings to exploit. This is exactly why Stack-100M's mix reserves a dedicated 10% code slice ([Chapter 14.2](../14-capstone/02-data-pipeline.html)): a tokenizer (and later, a model) that never sees code during training compresses and predicts it markedly worse than prose it was actually trained on.

    All three numbers are *measured*, from the exact `StackTokenizer` class defined above, not estimated — you can reproduce them by running the training script against your own sample. As a sanity floor: byte-level BPE at any vocabulary size can never do *worse* than 1.0 bytes/token (the un-merged byte fallback), and GPT-2's larger 50,257-entry vocabulary typically lands on the order of 4.0–4.3 bytes/token on general English prose (see [Chapter 2.1](../02-transformer/01-tokenization.html)) — so 32,768 landing a little below that on a similar corpus, as we measured, is exactly the expected shape of the tradeoff: a smaller vocabulary compresses somewhat less well, in exchange for a meaningfully smaller embedding table.

Round-trip correctness is non-negotiable — a tokenizer that cannot reconstruct its input byte-for-byte silently corrupts every downstream stage. We verified this on the same trained tokenizer, including a direct spot-check against a 5,000-character slice of the real training corpus (not just short hand-picked strings):

```python
# Round-trip verification against a freshly-trained tokenizer -- this is the
# exact StackTokenizer defined earlier in this chapter, trained on a real
# corpus sample (here: this book's own manuscript text, as in the "training
# at scale" experiment above; substitute your own corpus_sample.txt).
from stacklm.tokenizer import StackTokenizer, VOCAB_SIZE

tok = StackTokenizer()
tok.train(open("corpus_sample.txt", encoding="utf-8").read(), vocab_size=VOCAB_SIZE)

sample = ("Tokenization is the most underestimated component of the stack. "
          "It is not part of the network, it is not trained by gradient descent, "
          "and it is frozen for the entire life of the model.")

ids = tok.encode(sample)
assert tok.decode(ids) == sample                       # exact byte-for-byte match

# Spot-check against a slice of the real corpus, not a hand-picked string --
# this exercises Unicode, markdown syntax, and code fences the sample above
# does not.
corpus_slice = open("corpus_sample.txt", encoding="utf-8").read()[1000:6000]
assert tok.decode(tok.encode(corpus_slice)) == corpus_slice
print("round trip OK on both the sentence and the 5,000-char corpus slice")
```

Both assertions pass. This is the guarantee byte-level BPE buys you and character-level or word-level tokenizers cannot: because every string is, worst case, a sequence of raw UTF-8 bytes each with its own id (0–255), there is no input this tokenizer can fail to encode and later reconstruct exactly — not an emoji, not a mixed-script string, not a stray invalid byte.

## Wiring Into the Rest of the Capstone

Three concrete commitments this chapter locks in for the rest of Part XIV:

- **The shard format.** [Chapter 14.2](../14-capstone/02-data-pipeline.html) packs `StackTokenizer.encode()` output into `uint16` memmap `.bin` shards. `vocab_size = 32{,}768` fits comfortably inside `uint16`'s `0`–`65{,}535` range, with room to spare — a vocabulary above 65,536 would have forced `uint32` shards, doubling the data pipeline's disk footprint for no benefit.
- **The embedding table's shape.** [Chapter 14.4](../14-capstone/04-architecture.html) allocates `nn.Embedding(32768, 512)`, tied to the output projection, and every parameter-count claim in that chapter (and this one) depends on `vocab_size` never moving after this chapter ships `tokenizer/stack100m-32768.json`.
- **The loss mask.** [Chapter 14.9](../14-capstone/09-post-training.html)'s SFT loop masks the loss to tokens *after* `<|assistant|>` and *before* `<|end|>`, and separately masks out anything between `<|tool_result|>` and the next `<|assistant|>` in [Chapter 14.10](../14-capstone/10-agentic-narrow.html)'s agent traces — both rely on the exact ids this chapter assigned (32764 and 32767 respectively) being stable.

Get this chapter wrong and every one of those three commitments has to be redone from scratch, at the cost of every checkpoint trained in between. That is the real argument for spending a full chapter — and a few real CPU-seconds — getting the tokenizer right before writing a single line of the model itself.

## Key Takeaways

!!! key "Key Takeaways"
    - A tokenizer is trained once, by counting statistics, and then frozen for the life of the model — every design decision here (vocabulary size, special-token ids) is effectively permanent once training begins.
    - Byte-level BPE guarantees no input is ever unrepresentable: worst case, a string falls back to its raw UTF-8 bytes, each with its own id in 0–255.
    - A naive BPE trainer is `O(merges × corpus size)`; an inverted pair-index plus a lazy-deleted max-heap makes each merge touch only the words it affects, turning a multi-hour job into single-digit seconds — measured, not estimated, on a real ~8 MB corpus (3.25 s for all 32,503 merges).
    - Reserve every special token a project will ever need — chat roles, tool-call markers, padding — *before* training starts, in a fixed order, because adding one later means either retraining (invalidating every checkpoint) or bolting on untrained embedding rows.
    - At 100M parameters, vocabulary size is a real design lever, not a free hyperparameter: a tied embedding table at GPT-2's 50,257-entry vocabulary would cost roughly a quarter of the whole budget — on the order of four transformer layers' worth of parameters.
    - Tied input/output embeddings (Press & Wolf, 2017) are worth roughly 6 extra layers of depth at Stack-100M's scale, for a single line of code (`lm_head.weight = embed.weight`).
    - Bigger vocabularies compress better (fewer tokens per document) but cost more embedding parameters; smaller vocabularies protect depth but pay it back in longer sequences and more forward passes per document — 32,768 is Stack-100M's chosen point on that curve, not a universal default.
    - Measured, real numbers beat estimated ones: this chapter's compression ratios (≈3.9 bytes/token corpus-wide, ≈4.7 for clean prose, ≈2.7 for code) came from actually running the trainer above, and you should always measure your own tokenizer against your own data rather than trusting someone else's numbers.

!!! sota "State of the Art & Resources (2026)"
    Byte-level BPE (this chapter's algorithm) remains the dominant tokenization scheme for production LLMs, but 2024–2026 research has pushed hard on two fronts: making tokenizers themselves better at compression (superword tokenization), and asking whether an explicit tokenizer is needed at all (byte-level / patch-based models).

    **Foundational work**

    - [Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units* (2016)](https://arxiv.org/abs/1508.07909) — the paper that brought BPE from data compression into NLP as a subword tokenizer.
    - [Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — introduced byte-level BPE and the pre-tokenizer regex this chapter's trainer reuses directly.
    - [Kudo & Richardson, *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer* (2018)](https://arxiv.org/abs/1808.06226) — the language-agnostic, whitespace-as-symbol framing behind many production tokenizer pipelines.
    - [Press & Wolf, *Using the Output Embedding to Improve Language Models* (2017)](https://arxiv.org/abs/1608.05859) — the tied-embeddings result this chapter's parameter accounting (the "worth ~6 layers" argument) depends on.

    **Recent advances (2023–2026)**

    - [Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens* (2024)](https://arxiv.org/abs/2412.09871) — Meta's dynamic-entropy byte-patching architecture, the most credible recent attempt to match BPE-tokenized LLM quality without a fixed subword vocabulary at all.
    - [Liu, Hayase, Hofmann, Oh, Smith & Choi, *SuperBPE: Space Travel for Language Models* (2025)](https://arxiv.org/abs/2503.13423) — extends BPE to merge across whitespace into "superword" tokens, reporting meaningfully fewer tokens per document and lower inference compute at fixed vocabulary size — directly relevant to this chapter's compression-vs-embedding-budget tradeoff.

    **Open-source & tools**

    - [openai/tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokenizer; the `allowed_special` design this chapter's `encode()` follows originates here.
    - [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — the Rust-backed production trainer referenced in this chapter's "training at scale" timing comparison.
    - [karpathy/minbpe](https://github.com/karpathy/minbpe) — a minimal, from-scratch reference implementation of byte-level BPE train/encode/decode, good for cross-checking this chapter's trainer against an independently written one.

    **Go deeper**

    - [Hugging Face LLM Course — Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5) — a worked, step-by-step walkthrough of the BPE training and tokenization algorithm this chapter implements.

## Further reading

- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*, 2016 — the paper that introduced BPE to NLP.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 — introduced byte-level BPE and the pre-tokenizer regex this chapter reuses.
- Press & Wolf, *Using the Output Embedding to Improve Language Models*, 2017 — the tied-embeddings result this chapter's parameter accounting depends on.
- Yang, Dai, Salakhutdinov & Cohen, *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model*, 2018 — the rank-limitation argument behind the softmax-bottleneck aside above.
- Kudo & Richardson, *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing*, 2018 — the language-agnostic, whitespace-as-symbol framing many production tokenizers (including byte-level BPE variants) build on.
- HuggingFace `tokenizers` — the Rust-backed library referenced in the "training at scale" timing comparison; the production-speed counterpart to the from-scratch trainer in this chapter.
- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, 2024 — the deep-and-thin small-model philosophy this chapter's depth-vs-vocabulary tradeoff protects; developed further in [Chapter 14.4](../14-capstone/04-architecture.html).

## Exercises

**1.** By default, `StackTokenizer.encode` is called with `allowed_special=frozenset()`, so a literal `<|assistant|>` typed into a user message is byte-encoded like ordinary text instead of mapping to id 32764. Explain concretely what could go wrong if `encode` instead recognized special-token strings everywhere by default, and name the one place in the pipeline where this invariant is what makes a downstream guarantee trustworthy.

??? note "Solution"
    If `encode` recognized special-token *strings* everywhere, then any untrusted content — a user's chat message, or a tool's returned observation — that happened to contain the literal text `<|assistant|>` would tokenize to the real role-boundary id 32764. That lets an attacker forge turns: they could close the user turn early and open a fake assistant turn, or inject a fake `<|tool_result|>` (id 32767) to smuggle in an "observation" the model never actually retrieved. This is the tokenizer-level analogue of prompt injection.

    The fix is exactly the default: untrusted text is encoded with `allowed_special=frozenset()`, so special-token *ids* can only ever enter a sequence through code you control (the chat-template formatter, the packing code), never through content a user or tool supplied.

    The place this matters most is the **SFT loss mask** in [Chapter 14.9](../14-capstone/09-post-training.html). That loop computes loss only on tokens after `<|assistant|>` (id 32764). The mask is trustworthy only because the default guarantees that every id 32764 in a training example is a real assistant boundary the formatter placed — not a string that some example's user turn happened to contain.

**2.** The chapter fixes `vocab_size = 32768` with `256` reserved byte ids and `9` special tokens, giving `M = 32{,}503` merges. Suppose instead you targeted `vocab_size = 16384` with the *same* `9` special tokens in the *same* order. (a) How many merges `M` does the trainer learn? (b) What id does `<|bos|>` get? (c) What id does `<|user|>` get?

??? note "Solution"
    The layout is always bytes (ids $0..255$), then $M$ learned merges (ids $256..256+M-1$), then the $S$ specials at the top of the range in fixed order.

    (a) Number of merges:
    $$
    M = V - 256 - S = 16{,}384 - 256 - 9 = 16{,}119
    $$

    (b) The specials occupy the top $S = 9$ ids, so the first special `<|bos|>` sits at $V - S = 16{,}384 - 9 = 16{,}375$. (Equivalently $256 + M = 256 + 16{,}119 = 16{,}375$.)

    (c) In `SPECIAL_TOKENS` order, `<|bos|>` is index 0, `<|eos|>` 1, `<|pad|>` 2, `<|system|>` 3, `<|user|>` 4. So
    $$
    \text{id}(\texttt{<|user|>}) = 16{,}375 + 4 = 16{,}379.
    $$

    Note that unlike in the real `vocab_size = 32768` layout (where `<|user|>` is 32763), the id is *different* — which is exactly why the chapter insists the vocabulary is frozen: change `vocab_size` and every special-token id moves.

**3.** Using the chapter's parameter accounting (`d_model = 512`, one transformer block $\approx 2.82$M params), consider raising the vocabulary to `V = 65{,}536`. (a) What does the **tied** embedding table cost? (b) What would it cost **untied**? (c) Using the chapter's methodology `layers affordable = (100\text{M} - V\cdot 512)/2.82\text{M}`, how many layers can a nominal 100M-parameter budget afford, and how does that compare to Stack-100M's 30?

??? note "Solution"
    (a) Tied embedding is a single $V \times d_\text{model}$ matrix:
    $$
    P_\text{embed} = 65{,}536 \times 512 = 33{,}554{,}432 \approx 33.55\text{M params}.
    $$

    (b) Untied spends that matrix twice (input embedding + output `lm_head`):
    $$
    2 \times 33.55\text{M} = 67.11\text{M params}.
    $$

    (c) Layers affordable at a fixed 100M budget:
    $$
    \frac{100{,}000{,}000 - 65{,}536 \times 512}{2{,}820{,}000} = \frac{100{,}000{,}000 - 33{,}554{,}432}{2{,}820{,}000} = \frac{66{,}445{,}568}{2{,}820{,}000} \approx 23.56 \approx 24 \text{ layers}.
    $$

    That is $24 - 30 = -6$ layers versus Stack-100M's 30. Doubling the vocabulary from 32,768 to 65,536 costs roughly six transformer layers of depth at a fixed 100M budget — the same tradeoff the chapter's table reports for that row.

**4.** The CI toy path trains on `TOY_CORPUS = "the quick brown fox jumps over the lazy dog. " * 200` but asks for `vocab_size = 281`, not 32,768. Explain why asking for a large `vocab_size` on this corpus would silently "stop short," referring to the specific line in `train_bpe` that halts it.

??? note "Solution"
    The pre-tokenizer (the GPT-2 regex) splits text into chunks that never cross word or whitespace boundaries, and BPE only ever merges *adjacent* symbols *within* a chunk. The toy corpus is one sentence of ~9 distinct words repeated 200 times, so it pre-tokenizes into only a handful of distinct chunk types. Each distinct chunk can be collapsed, merge by merge, down to a single symbol — and once every chunk is a single symbol there are no adjacent pairs left that repeat.

    At that point `train_bpe` hits its early-stop guard:

    ```python
    if live_count < 2:
        break   # nothing left that repeats; stop early
    ```

    The most frequent remaining pair occurs fewer than 2 times (or no pairs remain at all), so the loop breaks before reaching `num_merges`. Asking for `vocab_size = 32768` would return far fewer than 32,503 merges, and `toy_tok.vocab_size` would silently be smaller than requested. The toy path picks 281 precisely because that is a target this tiny, low-diversity corpus can actually fill. The real training corpus, with millions of unique chunks, has no such ceiling anywhere near 32,768.

**5.** Implement a helper `bytes_per_token(tok, text)` that returns the compression ratio (UTF-8 bytes per token) for a trained `StackTokenizer`, encoding `text` as untrusted content. Then, using the chapter's measured numbers, state what ratio you'd expect for clean English prose versus a Python snippet, and why they differ.

??? note "Solution"
    ```python
    def bytes_per_token(tok: StackTokenizer, text: str) -> float:
        """UTF-8 bytes per token: higher == better compression.
        `text` is treated as untrusted, so no special-token strings are
        recognized (allowed_special defaults to the empty frozenset)."""
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(tok.encode(text))     # allowed_special=frozenset() by default
        return n_bytes / n_tokens
    ```

    The byte count must use `text.encode("utf-8")` (a multi-byte character like an emoji or an accented letter is more than one byte), and we must pass untrusted text through the default `allowed_special=frozenset()` so a stray `<|assistant|>` in the input is not miscounted as a single special id.

    Expected ratios, from the chapter's measured numbers on the 32,768-vocabulary tokenizer:

    - Clean English prose: about $184 / 39 \approx 4.72$ bytes/token — the best case, because prose is exactly the kind of frequent, long, reusable substring (common morphemes, whole words) that BPE dedicates merges to.
    - A Python snippet: about $234 / 88 \approx 2.66$ bytes/token — markedly worse, because code's dense punctuation, brackets, and multi-level indentation give a prose-heavy vocabulary far fewer long reusable substrings to compress.

    The floor is 1.0 bytes/token (pure byte fallback, no merges apply); byte-level BPE can never do worse than that.

**6.** A colleague wants to add `8` reserved special-token placeholders (like Llama 3's `<|reserved_special_token_N|>`) **but keep `vocab_size` fixed at 32768 and keep the 9 real special-token ids exactly where they are** (`<|bos|> = 32759` ... `<|tool_result|> = 32767`). (a) If they *append* the 8 reserved tokens after `<|tool_result|>`, what happens to the 9 real ids? (b) Show a placement that keeps all 9 real ids unchanged, and state what you pay for it. Back your answer with the id arithmetic.

??? note "Solution"
    With `vocab_size` fixed at 32768, the number of merges depends on the total number of specials $S$:
    $$
    M = V - 256 - S.
    $$
    Adding 8 reserved tokens makes $S = 9 + 8 = 17$, so $M = 32{,}768 - 256 - 17 = 32{,}495$ (eight fewer merges than the original 32,503). The special block therefore starts at $256 + M = 32{,}751$ instead of $32{,}759$ — it shifts *down* by 8.

    (a) **Appending** the reserved tokens (order = the 9 real ones, then 8 reserved) puts `<|bos|>` at the start of the block, id $32{,}751$, not $32{,}759$. Every one of the 9 real ids shifts down by 8: `<|tool_result|>` lands at $32{,}751 + 8 = 32{,}759$ instead of $32{,}767$. This breaks the frozen invariant and invalidates every checkpoint's embedding rows.

    (b) **Prepend** the 8 reserved tokens instead (order = 8 reserved, then the 9 real ones). The block still starts at $32{,}751$, but now the reserved tokens absorb ids $32{,}751 .. 32{,}758$, and the first real token `<|bos|>` lands at:
    $$
    32{,}751 + 8 = 32{,}759,
    $$
    with `<|eos|> = 32760`, ..., `<|tool_result|> = 32767` — all nine ids exactly where they were.

    ```python
    RESERVED = tuple(f"<|reserved_{i}|>" for i in range(8))
    SPECIAL_TOKENS_V2 = RESERVED + SPECIAL_TOKENS      # prepend: reserved first
    # S = 17, so M = 32768 - 256 - 17 = 32495
    # block starts at 256 + 32495 = 32751:
    #   reserved_0..reserved_7  -> 32751..32758
    #   <|bos|>                 -> 32759   (unchanged)
    #   ...
    #   <|tool_result|>         -> 32767   (unchanged)
    ```

    What you pay: 8 fewer learned merges (32,503 -> 32,495), a negligible compression cost. What you avoid: shifting any of the 9 real ids. The deeper lesson is the practitioner tip's point — the only *clean* way to have reserved slots is to have included them in the original `SPECIAL_TOKENS` before the tokenizer was ever trained; retrofitting them safely is possible only because we can control placement, and only prepending preserves the already-frozen ids.
