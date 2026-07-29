# 14.3 A Byte-Level BPE Tokenizer From Scratch (and Why Vocab Size Is a Design Lever at 100M)

Before Stack-100M sees a single training example, before we have written a line of the transformer block, before we know exactly how many layers fit in a 100M-parameter budget, we have to answer a question that sounds administrative but is actually one of the highest-leverage design decisions in the whole project: **what integers does the model see?**

That question is the tokenizer's job. [Chapter 14.2](../14-capstone/02-data-pipeline.html) streams and cleans the ~20B-token data mix; this chapter turns that raw text into the vocabulary Stack-100M will speak for the rest of its life, and [Chapter 14.4](../14-capstone/04-architecture.html) sizes the embedding table — and therefore how many parameters are left over for depth and width — against the number we fix here. Everything downstream depends on getting this chapter right first: the shard format in 14.2, the embedding table in 14.4, the chat template in [Chapter 14.9](../14-capstone/09-post-training.html), and the tool-call format in [Chapter 14.10](../14-capstone/10-agentic-narrow.html) all hardcode assumptions this chapter fixes.

We already built byte-level BPE from first principles in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) — the merge-the-most-frequent-adjacent-pair algorithm, the guarantee that byte-level fallback means no input is ever unrepresentable, the pre-tokenizer regex that keeps merges from crossing word boundaries. We will not re-derive any of that here. This chapter does four things that chapter does not:

1. It engineers a trainer that finishes all 32,503 merges on a real corpus in **seconds**, and gets the incremental bookkeeping *right* (there is a subtle, silent correctness bug in the obvious implementation — we will hit it head-on).
2. It makes encoding fast enough to actually tokenize ~84 GB of text, which is what the 20B-token budget means in bytes. A pure-Python `encode` that is 100× too slow is a black box between this chapter and 14.2, not a pipeline.
3. It **exports the result into the real ecosystem** — `tiktoken`, HuggingFace `tokenizers`, and `transformers.PreTrainedTokenizerFast` — because a bespoke JSON file that TRL, vLLM, and llama.cpp cannot load is not a shippable artifact.
4. It argues, with *measured* compression numbers and a FLOP budget rather than an assertion, that at 100M parameters **vocabulary size is not a free hyperparameter**. It is a line item that competes directly with depth and width for the same fixed budget.

## Why Stack-100M Trains Its Own Tokenizer

{{tool:tokenizer-playground}}

You could, in principle, reuse an off-the-shelf tokenizer — GPT-2's 50,257-entry vocabulary, or Llama 3's 128k-entry one — and skip this chapter entirely. Three reasons we don't:

1. **Domain match.** Stack-100M's data mix ([Chapter 14.2](../14-capstone/02-data-pipeline.html)) is 70% FineWeb-Edu-style educational web text, 15% Cosmopedia-style synthetic textbooks, 10% code, and 5% math — a narrower, more structured distribution than "the open web" that GPT-2's tokenizer was fit to. A tokenizer trained on our own mix spends its merge budget on substrings that actually recur in *our* corpus, not GPT-2's.
2. **Full control over the ID layout.** We need nine special tokens — beginning/end/pad markers now, chat-role markers for [Chapter 14.9](../14-capstone/09-post-training.html), tool-call markers for [Chapter 14.10](../14-capstone/10-agentic-narrow.html) — reserved at specific, predictable positions before a single row of the embedding table is initialized. Borrowing someone else's tokenizer means inheriting (or awkwardly patching) their special-token layout instead.
3. **Vocabulary size is the whole point of this chapter.** GPT-2's tokenizer was fit for a family of models where the embedding table is a rounding error next to a 1.5B-parameter network. At 100M parameters that is no longer true, and we want a vocabulary size we chose deliberately, not one we inherited.

The good news, which we demonstrate with real measurements rather than a hand-wave: training a from-scratch, 32,768-entry byte-level BPE tokenizer on a multi-megabyte corpus is a matter of **seconds**, not hours, if you engineer the trainer even a little. There is no excuse not to train your own.

**Where this fits in the pipeline:**

```text
raw text corpus (14.2)
        │
        ▼
  ┌──────────────────┐   train byte-level BPE, vocab_size = 32768
  │  THIS CHAPTER    │   reserve 9 special tokens
  │  (14.3)          │   export to the ecosystem
  └────────┬─────────┘
           │
           ├─► tokenizer/stack100m-32768.json        (from-scratch artifact, frozen)
           └─► tokenizer/stack100m-32768-hf/         (tokenizer.json + tokenizer_config.json)
                     │                                loaded by transformers / TRL / vLLM
           ┌─────────┴──────────┐
           ▼                    ▼
  ┌─────────────────┐   ┌────────────────────┐
  │ data packing    │   │ architecture (14.4) │
  │ (14.2, uint16   │   │ embedding table     │
  │ .bin shards)    │   │ sized to 32768×512  │
  └─────────────────┘   └─────────────────────┘
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
    Some production tokenizers (Meta's Llama 3, for instance) pad their special-token block with dozens of unused `<|reserved_special_token_N|>` placeholders, so a future fine-tune can add a role or a tool format without touching `vocab_size` or reshuffling ids. We don't do that here — Stack-100M's special-token needs are fully enumerated by this table, and every one of the 32,768 rows should either be a real byte/merge or a token we know we will use, so none of the tight 100M-parameter budget is spent on speculative slots. If you extend this project past the capstone's scope, budgeting 8–16 reserved slots is cheap insurance. (Exercise 6 works out exactly where to place them so the nine real ids do not move.)

## A From-Scratch, Efficient Byte-Level BPE Trainer

### The pre-tokenizer is a design decision too

BPE never merges across a pre-tokenizer boundary. That single fact means the split regex — which most tutorials copy from GPT-2 and never think about again — silently determines what kinds of tokens your model *can* have. It deserves the same scrutiny as `vocab_size`.

GPT-2's 2019 pattern (Radford et al.) includes ` ?\p{N}+`: an unbounded run of digits, merged greedily like any other substring. The consequence is that a big corpus produces idiosyncratic number tokens — a single token for `2020` because it was frequent in the crawl, but three tokens for `2031` because it wasn't. Arithmetic then becomes a task where the model's input segmentation is different for numerically adjacent inputs, which is a genuinely bad property. Every frontier tokenizer since GPT-4's `cl100k_base` — including Llama 3, Qwen, and Gemma — caps digit runs for exactly this reason. `capstone/PLAN.md` §8 makes **integer arithmetic** the RLVR task and §7 does math capability injection, so this is not a theoretical concern for us; it directly gates two later chapters.

Stack-100M therefore uses a `cl100k`-style pattern with digits capped at three:

```python
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"      # contractions, case-insensitive (GPT-2 was not)
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"          # a word, optionally preceded by one non-letter
    r"|\p{N}{1,3}"                        # <= 3 digits, NEVER absorbing a leading space
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"         # punctuation runs, trailing newlines attached
    r"|\s*[\r\n]+"                        # newline runs (indentation-friendly)
    r"|\s+(?!\S)"                         # trailing whitespace at end of a run
    r"|\s+"                               # any remaining whitespace
)
```

Three concrete differences from GPT-2's pattern, all measured on the tokenizer we train below:

| Input | GPT-2 pattern → chunks | Stack-100M pattern → tokens |
|---|---|---|
| `1234567` | `['1234567']` (one chunk; whatever merge exists) | `'123' '456' '7'` |
| `The year 2026` | `['The', ' year', ' 2026']` | `'The' ' year' ' ' '202' '6'` |
| `x = 100000` | `['x', ' =', ' 100000']` | `'x' ' =' ' ' '100' '000'` |

Numbers are now segmented by a *fixed, content-independent* rule (left-to-right groups of at most three digits, with the space never glued on), so `2026` and `2031` receive structurally identical treatment. `Case-insensitive` contractions fix a long-standing GPT-2 wart where `'S` and `'s` tokenize differently. And `\s*[\r\n]+` keeps runs of indentation attached to their newline, which is the code-side analogue of the same decision — StarCoder and the Llama-3 code tokenizers go further and give multi-space indent runs their own dedicated tokens; if your mix were majority code rather than 10% code, that is the next knob to turn.

!!! note "Aside: if arithmetic is your headline task, go further"
    Capping at three digits is the frontier default, but it is not the arithmetic-optimal choice. Llama 2 (via SentencePiece's `split_digits`) and Gemma split numbers into **single digits**, which makes column-wise addition and multiplication a positionally regular problem for the model at the cost of ~2× more tokens on numeric text. Changing our pattern to `\p{N}` (one digit) is a one-character edit and a legitimate ablation for the RLVR run in [Chapter 14.9](../14-capstone/09-post-training.html); we keep `{1,3}` because Stack-100M's mix is 95% non-math and the compression cost is paid on every token of every step.

!!! warning "Common pitfall: a lossy pre-tokenizer silently corrupts your corpus"
    `encode` concatenates the encodings of `pattern.findall(text)`. If the alternation does not cover *every* character, `findall` silently drops the uncovered ones and decode will never reproduce the input. Both patterns above are total (letters, digits, non-space-non-alphanumeric, and whitespace exhaust Unicode), but you must assert it whenever you touch the regex:

    ```python
    assert "".join(_SPLIT_RE.findall(sample)) == sample
    ```

    This one-line property test belongs in CI, and it is cheaper to run than to debug.

### The trainer

The algorithm is unchanged from [Chapter 2.1](../02-transformer/01-tokenization.html): pre-tokenize, then repeatedly merge the most frequent adjacent pair of symbols. What changes here is engineering. The naive trainer recomputes every pair count from scratch after every merge — `O(merges × corpus size)`. At `M = 32{,}503` merges over even a modest multi-megabyte sample, that is a multi-hour job. We fix it with two standard data structures: an **inverted index** from each pair to the word indices that contain it (so a merge only touches the words it actually affects), and a **lazy-deleted max-heap** (so "which pair is most frequent right now" is an `O(\log n)` heap pop instead of an `O(n)` linear scan).

{{fig:bpe-trainer-incremental-merge}}

!!! warning "Common pitfall: lazy deletion is only correct if you re-push on *every* change"
    The subtle bug in the obvious implementation: a merge both **increments** counts (for the new pairs it creates) and **decrements** them (for the pairs it destroys). It is natural to push a refreshed heap entry only on increments — the count went up, so the old entry is merely pessimistic and will be superseded. But a *decrement* leaves the heap holding only entries with counts that are now too **high**. When such an entry pops, the staleness check `-neg_count != live_count` discards it — and because nothing ever pushed an entry matching the new, lower count, that pair is **silently dropped from the heap forever**.

    Concretely: pair $P=(a,b)$ has count 10 (6 from word A, 4 from word B), and the heap holds only $(-10, P)$. A later merge destroys $P$ inside word A, so `pair_counts[P]` becomes 4 with no new push. When $(-10, P)$ pops it is discarded as stale, and $P$ is never reconsidered — even if 4 is the current maximum. The result is a vocabulary that differs from true BPE, a heap that drains early, and a `train_bpe` that quietly returns **fewer than 32,503 merges**. The fix is one `touched` set: collect every pair whose count changed, and re-push all of them after the word loop.

```python
# capstone/stacklm/tokenizer.py
"""
Stack-100M's tokenizer: a from-scratch, byte-level BPE trainer + encoder/decoder.

This is the production version of the algorithm built from first principles in
../02-transformer/01-tokenization.html -- same "merge the most frequent adjacent
pair" idea, same byte-level guarantee that no input is ever unrepresentable, but
engineered to finish ~32.5k merges on a real multi-megabyte data sample in
seconds instead of hours (measured numbers in "Training at scale" below).

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
import warnings
import regex as re                      # `regex`, not stdlib `re` -- needed for \p{L} / \p{N}
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Union

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

VOCAB_SIZE = 32768                                          # PLAN Sec. 1 / Sec. 3
NUM_BYTES = 256                                             # every raw byte value is id 0..255
NUM_MERGES = VOCAB_SIZE - NUM_BYTES - len(SPECIAL_TOKENS)   # = 32,503

# cl100k / Llama-3-style pre-tokenizer. Digits are capped at 3 and never absorb
# a leading space, so numeric segmentation is content-independent (see prose).
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_SPLIT_RE = re.compile(SPLIT_PATTERN)


def _pretokenize_to_byte_ids(text: str) -> List[Tuple[int, ...]]:
    """Split `text` into pre-token chunks, each returned as a tuple of raw
    UTF-8 byte VALUES (0..255) -- e.g. 'café' -> (99, 97, 102, 195, 169)."""
    return [tuple(chunk.encode("utf-8")) for chunk in _SPLIT_RE.findall(text)]


# ---------------------------------------------------------------------------
# 2. The trainer. Operates on integer symbol ids throughout (0..255 for raw
#    bytes, 256+ for learned merges) instead of the string glyphs used in the
#    from-scratch chapter -- purely a speed choice, the algorithm is identical.
# ---------------------------------------------------------------------------
def train_bpe(word_freqs: Dict[Tuple[int, ...], int],
              num_merges: int) -> List[Tuple[int, int]]:
    """Learn `num_merges` BPE merges from a corpus reduced to
    (symbol-id-tuple -> frequency) counts.

    Naively, each merge requires rescanning the ENTIRE corpus to recount every
    adjacent pair -- O(num_merges x corpus_size). Instead we maintain three
    pieces of running state so each merge touches only the (usually small)
    subset of words that actually contain the pair being merged:

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
            # `wi` can be a STALE member of pair_to_words[pair]: an earlier
            # merge may already have removed `pair` from this word. Verify
            # membership before mutating anything.
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
```

### The tokenizer class

`StackTokenizer` wraps the trainer with vocabulary bookkeeping, an encoder, a decoder, and JSON persistence, so the artifact this chapter produces can be loaded by every later chapter without retraining. Four details are load-bearing and easy to get wrong:

- **Streaming training.** `train_from_iterable` consumes documents one at a time and folds them into a `Counter`. It never materializes the corpus as one giant `str` — which is the difference between a script that runs on a laptop and one that OOMs.
- **The shortfall guard.** If the sample is too small to fill the requested vocabulary, `train_bpe` stops early. Silently returning `vocab_size < 32768` would break `nn.Embedding(32768, 512)` in 14.4. We instead **pad with `<|unused_N|>` fillers placed *before* the nine real specials**, so `vocab_size` is exactly as requested *and* the nine real ids stay pinned to the top of the range — and we warn loudly.
- **The pre-tokenizer pattern is part of the artifact.** A tokenizer file that records merges but not the regex that produced them is not reproducible. `save()` writes it; `load()` refuses a mismatch.
- **A per-chunk cache.** Pre-token chunks are Zipfian; caching their encodings is the single cheapest speedup available (measured below).

```python
# capstone/stacklm/tokenizer.py (continued)

class StackTokenizer:
    """Byte-level BPE tokenizer for Stack-100M.

    ID layout (fixed for the whole project):
        0        .. 255           raw byte values (id == byte value)
        256      .. 256+M-1       learned merges, in the order they were trained
        256+M    .. vocab_size-1  filler <|unused_N|> (only if the sample was too
                                   small), then the 9 SPECIAL_TOKENS in order,
                                   ALWAYS occupying the final 9 ids.
    """

    def __init__(self) -> None:
        self.merges: List[Tuple[int, int]] = []             # learned merges, ordered
        self.merge_id: Dict[Tuple[int, int], int] = {}      # pair -> resulting id
        self.id_to_pair: Dict[int, Tuple[int, int]] = {}    # inverse, for decode
        self.special_to_id: Dict[str, int] = {}
        self.id_to_special: Dict[int, str] = {}
        self.pattern: str = SPLIT_PATTERN                    # frozen with the merges
        self._cache: Dict[Tuple[int, ...], List[int]] = {}   # pre-token -> ids

    @property
    def vocab_size(self) -> int:
        return NUM_BYTES + len(self.merges) + len(self.special_to_id)

    # Ch. 14.2's `Tokenizer` Protocol requires these three by name.
    @property
    def bos_id(self) -> int: return self.special_to_id["<|bos|>"]
    @property
    def eos_id(self) -> int: return self.special_to_id["<|eos|>"]
    @property
    def pad_id(self) -> int: return self.special_to_id["<|pad|>"]

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
            word_freqs.update(_pretokenize_to_byte_ids(doc))
        self.merges = train_bpe(word_freqs, num_merges)

        # Guard: a small/low-diversity sample runs out of repeated pairs. Pad
        # deterministically so vocab_size is EXACTLY what 14.4's embedding table
        # expects; put the fillers BEFORE the real specials so <|bos|> .. and
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

    # -- encode / decode --------------------------------------------------------
    def _apply_merges(self, symbols: List[int]) -> List[int]:
        """Repeatedly merge the pair with the LOWEST id (== earliest-learned
        == highest priority) until no learned pair applies -- the standard
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
               allowed_special: Union[str, frozenset] = frozenset()) -> List[int]:
        """`allowed_special` defaults to EMPTY: special-token strings inside
        `text` are treated as ordinary bytes. Pass an explicit set, or the
        tiktoken-style sentinel "all", only from trusted call sites."""
        allowed = (frozenset(self.special_to_id) if allowed_special == "all"
                   else frozenset(allowed_special))
        if allowed:
            # longest-first so a specials set containing overlapping strings
            # cannot be split by a shorter alternative
            pattern = "(" + "|".join(re.escape(s) for s in
                                     sorted(allowed, key=len, reverse=True)) + ")"
            segments = re.split(pattern, text)     # keeps the special literals
        else:
            segments = [text]

        ids: List[int] = []
        for seg in segments:
            if seg in allowed:
                ids.append(self.special_to_id[seg])
                continue
            for byte_ids in _pretokenize_to_byte_ids(seg):
                ids.extend(self._encode_chunk(byte_ids))
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

    def token_bytes(self) -> List[bytes]:
        """The literal byte string of every non-special id, by walking the merge
        tree bottom-up. This is what the tiktoken / HF exporters need."""
        table = [bytes([i]) for i in range(NUM_BYTES)]
        for a, b in self.merges:
            table.append(table[a] + table[b])      # children always have lower ids
        return table

    # -- persistence --------------------------------------------------------------
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
        tok = cls()
        tok.pattern = payload["pattern"]
        if tok.pattern != SPLIT_PATTERN:
            raise ValueError("saved pre-tokenizer pattern differs from this module's "
                             "SPLIT_PATTERN -- the merges would be misapplied")
        tok.merges = [tuple(p) for p in payload["merges"]]
        tok._assign_ids(tuple(payload["special_tokens"]))
        assert tok.vocab_size == payload["vocab_size"]
        return tok


def load_tokenizer(path: str = "tokenizer/stack100m-32768.json") -> StackTokenizer:
    """The entry point Ch. 14.2's packer and Ch. 14.9's chat template call."""
    return StackTokenizer.load(path)
```

Every important design decision from the earlier table shows up in code here: `SPECIAL_TOKENS` is an ordered tuple (not a set — order fixes ids), ids are always assigned bytes → merges → fillers → specials in that sequence, and `decode()` treats special ids as opaque literals rather than expanding them through the merge tree.

Note the deliberate default on `encode`: `allowed_special` is an **empty** frozenset unless the caller opts in. A literal `<|assistant|>` sitting inside a user's message is *not* recognized as the special token — it is pre-tokenized and byte-encoded like any other text, producing a completely different id sequence than the reserved id 32764. Only trusted call sites (the chat-template formatter in [Chapter 14.9](../14-capstone/09-post-training.html), the packing code in [Chapter 14.2](../14-capstone/02-data-pipeline.html)) pass an explicit `allowed_special` set — or the tiktoken-style `"all"` sentinel — to turn recognition back on.

!!! warning "Common pitfall: special-token injection"
    If `encode` recognized special-token *strings* everywhere by default, any user could type the literal text `<|assistant|>` into a chat box and have it tokenize to the real role-boundary id — letting them forge assistant turns, inject a fake `<|tool_result|>`, or otherwise escape the chat template. This is the tokenizer-level analogue of a prompt-injection attack (see [Security: Prompt Injection, Jailbreaks & Defenses](../12-production-mlops/06-security-prompt-injection.html)). The fix is exactly the default above: **untrusted text is encoded with `allowed_special=frozenset()`**, so special-token *ids* can only ever enter a sequence through code you control, never through content a user or a tool supplied. When you build the SFT loss mask in [Chapter 14.9](../14-capstone/09-post-training.html), this invariant is what lets you trust that every id 32764 in a training example is a real assistant boundary you placed, not a string an example's user turn happened to contain. Note that HuggingFace `tokenizers` has the *opposite* default — `AddedToken`s are matched everywhere — so when you cross into the ecosystem (next section) you must pass `add_special_tokens=False` and keep untrusted content away from `apply_chat_template`'s raw string path.

## Making Encoding Fast Enough for 20B Tokens

Training the tokenizer takes seconds. *Using* it does not: 20B tokens at the ~4.2 bytes/token we measure below is roughly **84 GB of text** that has to pass through `encode` before [Chapter 14.2](../14-capstone/02-data-pipeline.html) can write a single `.bin` shard. If you skip this section your pipeline stalls here, and no amount of GPU budget helps.

All numbers below were measured on a single core of a commodity x86 server, with the tokenizer trained in the next section, encoding this book's own manuscript (8.63 MB across 154 markdown files).

| Path | Throughput | 84 GB corpus (extrapolated) |
|---|---:|---:|
| `_apply_merges` per chunk, no cache (cold) | ~1.2 MB/s | ~19 hours |
| `_encode_chunk` with the dict cache, warm | ~5.8 MB/s | ~4 hours |
| Same, `multiprocessing.Pool(16)` over documents | ~39.8 MB/s | ~35 minutes |
| HF `tokenizers` `encode_batch` (Rust, batched) | ~11.9 MB/s | ~2 hours |
| `tiktoken.Encoding` (Rust), single thread | ~17.9 MB/s | ~1.3 hours |

Three observations. First, the cache is nearly free and worth ~5×: on a 400 KB slice, 92,026 pre-token chunks reduced to 10,178 distinct ones — an **88.9% hit rate**, and it climbs as the cache warms over a real shard. Second, tokenization is embarrassingly parallel at document granularity, so `multiprocessing` gets the pure-Python path into "one coffee break" territory. Third, the compiled encoders are the right answer for production, and — crucially — you can have them *without abandoning the tokenizer you just trained*, because the merge table is portable (next section).

Here is the parallel encoder 14.2's shard builder actually calls:

```python
# capstone/stacklm/tokenizer_parallel.py
"""Parallel corpus encoding. Documents are independent, so this is a pure
map: one worker per core, each with its own tokenizer + its own chunk cache.

Split shards on <|eos|> boundaries, never mid-document -- a document cut in
half would be pre-tokenized differently on each side of the cut.
"""
from __future__ import annotations
import multiprocessing as mp
from typing import Iterable, Iterator, List
import numpy as np

from stacklm.tokenizer import StackTokenizer, load_tokenizer

_TOK: StackTokenizer | None = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOK
    _TOK = load_tokenizer(tokenizer_path)      # each worker gets its own cache


def _encode_doc(text: str) -> List[int]:
    assert _TOK is not None
    # Untrusted corpus text: NEVER allow special strings through (see pitfall).
    # The <|bos|>/<|eos|> wrapper is added by code, in the packer.
    return _TOK.encode(text, allowed_special=frozenset())


def encode_corpus(docs: Iterable[str], tokenizer_path: str,
                  workers: int = 16, chunksize: int = 32) -> Iterator[np.ndarray]:
    """Yield one uint16 array per document, in input order."""
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(tokenizer_path,)) as pool:
        for ids in pool.imap(_encode_doc, docs, chunksize=chunksize):
            yield np.asarray(ids, dtype=np.uint16)   # vocab 32768 < 65536, fits
```

!!! tip "Practitioner tip: budget the tokenization job before you rent the GPU"
    Tokenizing 84 GB is a *CPU* job, and on the flagship single-A100 tier you are paying for the GPU while it runs. Do it as a separate, cheap CPU-only pass that writes the `.bin` shards once, then start the GPU rental. The shard files are the handoff. This is also why 14.2 writes `uint16` memmaps rather than re-tokenizing on the fly: the pretraining loop should never call `encode` at all.

## Exporting to the Ecosystem: tiktoken, `tokenizers`, `transformers`

A tokenizer is only useful if the rest of your stack can load it. `tokenizer/stack100m-32768.json` — our bespoke `{pattern, merges, special_tokens}` file — is readable by exactly one program: ours. But [Chapter 14.9](../14-capstone/09-post-training.html) fine-tunes with **TRL**, and [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html) serves with **vLLM** and **llama.cpp**, and all three of those load a HuggingFace `tokenizer.json`. So the last step of this chapter is an exporter, and it is only about 40 lines, because a byte-level BPE vocabulary is just (a) a rank-ordered list of token byte strings and (b) a pre-tokenizer regex.

### The one gotcha: `bytes_to_unicode`

HuggingFace's `tokenizers` (and GPT-2 before it) cannot store raw bytes in a JSON vocabulary, so it maps each of the 256 byte values to a **printable Unicode codepoint** — byte `0x20` (space) becomes `Ġ`, byte `0x0A` (newline) becomes `Ċ`, and so on. This is the `bytes_to_unicode()` table from the original GPT-2 release. Every byte-level BPE exporter has to reproduce it exactly; getting it wrong produces a tokenizer that loads fine and encodes subtly differently, which is the worst possible failure mode.

```python
# capstone/stacklm/export.py
"""Export a trained StackTokenizer into the two artifacts the ecosystem reads:
a `tiktoken.Encoding` (fast, OpenAI-style) and a HuggingFace `tokenizer.json`
(what transformers / TRL / vLLM / llama.cpp actually load).

Both exporters are EXACT: an equivalence test in CI asserts they produce
byte-identical id sequences to the from-scratch encoder (see below).
"""
from __future__ import annotations
from typing import Dict

from stacklm.tokenizer import StackTokenizer, SPLIT_PATTERN


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's byte <-> printable-unicode table (Radford et al., 2019).
    Maps all 256 byte values to codepoints that survive a JSON round trip:
    the printable ASCII/Latin-1 ranges map to themselves, and the 68 remaining
    control/space bytes are shifted into U+0100.. -- so byte 32 (space) is 'Ġ'
    (U+0120) and byte 10 (newline) is 'Ċ' (U+010A)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_B2U = bytes_to_unicode()


def _as_hf_string(token: bytes) -> str:
    return "".join(_B2U[b] for b in token)


# --- 1. tiktoken -----------------------------------------------------------
def to_tiktoken(tok: StackTokenizer, name: str = "stack100m-32768"):
    """`mergeable_ranks` maps TOKEN BYTES -> rank. Rank order IS merge order,
    which is exactly the invariant our id layout already guarantees, so no
    explicit merge list is needed: tiktoken re-derives merges by rank lookup."""
    import tiktoken
    ranks = {b: i for i, b in enumerate(tok.token_bytes())}   # 0..255 then merges
    return tiktoken.Encoding(
        name=name,
        pat_str=tok.pattern,
        mergeable_ranks=ranks,
        special_tokens=dict(tok.special_to_id),               # 32759..32767
    )


# --- 2. HuggingFace tokenizers --------------------------------------------
def to_hf_tokenizer(tok: StackTokenizer):
    """Build the equivalent `tokenizers.Tokenizer`. The pre-tokenizer is a
    Sequence, exactly as in Llama 3's tokenizer.json: Split on OUR regex first,
    then ByteLevel with `use_regex=False` (it must NOT re-apply GPT-2's own
    pattern) and `add_prefix_space=False` (we never inject a leading space)."""
    from tokenizers import (AddedToken, Regex, Tokenizer, decoders,
                            models, pre_tokenizers)
    table = tok.token_bytes()
    vocab = {_as_hf_string(b): i for i, b in enumerate(table)}
    merges = [(_as_hf_string(table[a]), _as_hf_string(table[b]))
              for a, b in tok.merges]

    hf = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
    hf.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(tok.pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    hf.decoder = decoders.ByteLevel()
    # Added tokens get ids AFTER the model vocab, i.e. exactly 32759.. -- which
    # is why the shortfall padding above matters: len(vocab) must be 32759.
    hf.add_special_tokens([AddedToken(s, special=True, normalized=False)
                           for s in tok.special_to_id])

    assert hf.get_vocab_size() == tok.vocab_size
    for s, i in tok.special_to_id.items():
        assert hf.token_to_id(s) == i, f"{s} landed at {hf.token_to_id(s)}, want {i}"
    return hf


# --- 3. transformers: the file everything else loads ----------------------
# ChatML-style template over the reserved role tokens (Ch. 14.9 formats SFT
# data with exactly this; vLLM's /v1/chat/completions endpoint reads it too).
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|' + m['role'] + '|>' + m['content'] + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def save_pretrained(tok: StackTokenizer,
                    out_dir: str = "tokenizer/stack100m-32768-hf"):
    from transformers import PreTrainedTokenizerFast
    fast = PreTrainedTokenizerFast(
        tokenizer_object=to_hf_tokenizer(tok),
        bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>",
                                   "<|end|>", "<|tool_call|>", "<|tool_result|>"],
        chat_template=CHAT_TEMPLATE,
        model_max_length=2048,          # Stack-100M's pretrain context (PLAN Sec.1)
    )
    fast.save_pretrained(out_dir)       # -> tokenizer.json, tokenizer_config.json
    return fast
```

### The equivalence test that makes the export trustworthy

An exporter you have not tested is a liability. This test belongs in CI, and it is the single most valuable test in this chapter: it asserts that all three encoders — ours, tiktoken's, and HuggingFace's — produce **identical id sequences** on a non-trivial probe containing Unicode, emoji, code, and digits.

```python
# capstone/tests/test_tokenizer_export.py
from stacklm.tokenizer import StackTokenizer
from stacklm.export import to_tiktoken, to_hf_tokenizer, save_pretrained

PROBE = ("Tokenization is frozen for the life of the model.\n"
         "café 🚀 — mixed ünïcode\tTAB\r\nCRLF\n"
         "def f(x):\n    return x ** 2  # 1234567 and 2026\n")


def test_exports_are_bit_identical(trained_tok: StackTokenizer):
    ref = trained_tok.encode(PROBE)                       # untrusted default

    assert to_tiktoken(trained_tok).encode_ordinary(PROBE) == ref
    hf = to_hf_tokenizer(trained_tok)
    assert hf.encode(PROBE, add_special_tokens=False).ids == ref
    assert hf.decode(ref) == PROBE                        # ByteLevel decoder round-trip

    fast = save_pretrained(trained_tok, "/tmp/stack100m-hf")
    assert fast(PROBE, add_special_tokens=False)["input_ids"] == ref
    assert fast.bos_token_id == 32759 and fast.pad_token_id == 32761
    assert fast.convert_tokens_to_ids("<|assistant|>") == 32764
    assert fast.convert_tokens_to_ids("<|tool_result|>") == 32767

    # And it survives a round trip through disk, which is what vLLM does.
    from transformers import AutoTokenizer
    reloaded = AutoTokenizer.from_pretrained("/tmp/stack100m-hf")
    assert reloaded(PROBE, add_special_tokens=False)["input_ids"] == ref
    assert len(reloaded) == 32768
```

We ran exactly this against a real 32,768-entry tokenizer trained on the corpus described below: **all four assertions on identity hold exactly**, and `AutoTokenizer.from_pretrained` reproduces the from-scratch ids after a disk round trip. That is the property that makes the rest of Part XIV possible — `TRL`'s `SFTTrainer`, vLLM's `--tokenizer`, and llama.cpp's GGUF converter all read the same `tokenizer.json`.

!!! note "Aside: how close is our trainer to the production one?"
    A stronger question than "does the export work" is "did our from-scratch trainer learn the *same vocabulary* a battle-tested library would?" We trained HuggingFace's Rust `trainers.BpeTrainer` on the identical corpus with the identical pre-tokenizer and `vocab_size`, and compared merge tables. The first **100** merges are identical in rank order; **498 of the first 500** are; and over all 32,503 merges the learned token *sets* agree to within three tokens (>99.99% overlap). The residual disagreement is tie-breaking among equally-frequent pairs, which BPE does not specify. That is about as strong a cross-validation as this algorithm admits, and it is the kind of check worth running whenever you write a from-scratch implementation of something a library already does.

    (Speed, honestly: on this ~7.8 MB sample the Rust trainer took **7.4 s** end to end versus **3.4 s** for the pure-Python trainer above — the library's parallelism does not pay for its setup until the sample is much larger. The reason to reach for `tokenizers` is not trainer speed at this scale; it is encode throughput and the `tokenizer.json` artifact.)

## Training Stack-100M's Tokenizer on the Data Mix

In production you point the trainer at a representative sample of the mix from [Chapter 14.2](../14-capstone/02-data-pipeline.html) — you do **not** need the full ~20B-token corpus to fit a good vocabulary, because pair-frequency statistics converge long before that. A few hundred megabytes drawn from the same 70/15/10/5 FineWeb-Edu / Cosmopedia / code / math mix is enough to learn merges that generalize.

Two hard requirements on the sampling script: it must **stream** (never build one giant string), and it must **respect its byte budget mid-file** (a single 4 GB shard must not be able to blow the budget by 8×).

```python
# capstone/scripts/train_tokenizer.py
"""
Trains Stack-100M's tokenizer on a SAMPLE of the pretraining mix (Ch. 14.2)
and exports it into every format the rest of Part XIV needs.

Memory note: peak RSS is O(distinct pre-token chunks), not O(corpus bytes).
On this book's 7.8 MB manuscript the Counter holds 49,654 distinct chunks and
the whole process stays around 300 MB RSS; the trainer's `words`/`pair_counts`
state scales with distinct chunks, which grows roughly like the square root of
corpus size for natural text. A few hundred MB of sample is comfortable on a
16 GB laptop; past a few GB, switch to the HF `tokenizers` trainer.
"""
import glob
from typing import Iterator

from stacklm.tokenizer import StackTokenizer, VOCAB_SIZE, SPECIAL_TOKENS
from stacklm.export import save_pretrained

CHUNK = 8 << 20     # read 8 MiB at a time; never f.read() a whole shard


def stream_sample(paths_glob: str, max_bytes: int = 500_000_000) -> Iterator[str]:
    """Yield bounded text chunks from raw-text shards, stopping EXACTLY at the
    byte budget (mid-file if necessary) rather than after whichever file
    happened to cross it."""
    total = 0
    for path in sorted(glob.glob(paths_glob)):
        with open(path, "r", encoding="utf-8") as f:
            while total < max_bytes:
                block = f.read(CHUNK)
                if not block:
                    break
                nbytes = len(block.encode("utf-8"))
                if total + nbytes > max_bytes:
                    # truncate on a whitespace boundary so we don't cut a word
                    keep = block[: max(1, len(block) // 2)].rsplit(" ", 1)[0]
                    total = max_bytes
                    yield keep
                    break
                total += nbytes
                yield block
        if total >= max_bytes:
            break


if __name__ == "__main__":
    tok = StackTokenizer()
    shortfall = tok.train_from_iterable(
        stream_sample("data/mix_sample/*.txt", max_bytes=500_000_000),
        vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    assert shortfall == 0, "sample too small to fill 32,768 entries -- enlarge it"
    assert tok.vocab_size == VOCAB_SIZE            # 14.4 hardcodes nn.Embedding(32768, 512)
    assert tok.special_to_id["<|tool_result|>"] == VOCAB_SIZE - 1

    tok.save("tokenizer/stack100m-32768.json")     # from-scratch artifact
    save_pretrained(tok, "tokenizer/stack100m-32768-hf")   # ecosystem artifact
    print(f"trained {len(tok.merges)} merges, vocab_size={tok.vocab_size}")
```

For the hermetic, network-free CI smoke test that [Chapter 14.2](../14-capstone/02-data-pipeline.html) runs on every commit, we shrink both the corpus and the vocabulary so the whole test finishes in well under a second:

```python
# CI / toy path: no network, no real data mix, just enough to exercise every
# code path (train -> encode -> decode round-trip -> save/load -> id layout).
import warnings
from stacklm.tokenizer import StackTokenizer, SPECIAL_TOKENS

TOY_CORPUS = "the quick brown fox jumps over the lazy dog. " * 200

# The pre-tokenizer never lets merges cross word boundaries, so a corpus built
# from ~9 distinct words runs out of repeated pairs after only ~32 merges.
# `train_from_iterable` therefore reports a SHORTFALL and pads with
# <|unused_N|> fillers -- placed BEFORE the real specials, so the layout
# invariant survives even on a degenerate corpus.
toy = StackTokenizer()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")            # the shortfall warning is expected here
    shortfall = toy.train(TOY_CORPUS, vocab_size=512, special_tokens=SPECIAL_TOKENS)

assert toy.vocab_size == 512                   # exact, never silently short
assert shortfall > 0                           # a toy corpus cannot fill 512
assert toy.special_to_id["<|bos|>"] == 512 - 9         # 503: specials still on top
assert toy.special_to_id["<|tool_result|>"] == 511
assert (toy.bos_id, toy.eos_id, toy.pad_id) == (503, 504, 505)

ids = toy.encode("the fox jumps")
assert toy.decode(ids) == "the fox jumps"

# Untrusted text must NOT produce special ids; a trusted call site may.
msg = "hello <|assistant|> world"
assert toy.special_to_id["<|assistant|>"] not in toy.encode(msg)
assert toy.special_to_id["<|assistant|>"] in toy.encode(msg, allowed_special="all")
assert toy.decode(toy.encode(msg)) == msg      # byte-exact either way

# Unicode / control characters / CRLF must survive the round trip.
hard = "café 🚀 — ünïcode\ttab\r\nCRLF 12345"
assert toy.decode(toy.encode(hard)) == hard

toy.save("/tmp/toy_tokenizer.json")
reloaded = StackTokenizer.load("/tmp/toy_tokenizer.json")
assert reloaded.encode("the fox jumps") == ids and reloaded.vocab_size == 512
print("toy tokenizer round-trip OK")
```

### Training at scale: the real numbers

We ran the exact trainer above — no shortcuts, no pre-built libraries — against a real corpus: this book's own manuscript, 154 markdown files of English technical prose interleaved with Python code and LaTeX math. A reasonable stand-in for the *flavor* of the FineWeb-Edu + Cosmopedia + code + math mix, though obviously not the real 20B-token corpus. We split it **by file** into 138 training files (7,818,444 bytes) and 16 held-out files (810,623 bytes), so every compression number below is measured on text the tokenizer never saw.

The training corpus pre-tokenized into **49,654 distinct chunks**. Training all 32,503 merges to reach `vocab_size = 32768` took **3.4 seconds** end to end on one core: ~1.0 s to pre-tokenize and count, ~2.0 s in the merge loop. Peak RSS stayed near 300 MB.

This is the payoff of the heap-plus-inverted-index design. A naive full-corpus rescan per merge, over 49,654 unique chunks, 32,503 times, is an entirely different asymptotic regime — comfortably pushed from single-digit seconds into tens of minutes or worse on the same hardware.

!!! note "Aside: why this is fast enough to just always do"
    A common piece of received wisdom is "training a tokenizer from scratch is expensive, just reuse GPT-2's." At the scale this capstone operates at, that wisdom is stale — three and a half seconds of CPU time buys full control over vocabulary size and special-token layout, control that matters far more at 100M parameters than it does at 100B, where a few million embedding parameters round to noise. What *is* expensive is encoding the full corpus afterwards, which is why the previous section exists.

## Vocab Size Is a Design Lever at 100M Parameters

Here is the argument this chapter exists to make. [Chapter 14.4](../14-capstone/04-architecture.html) works out Stack-100M's full parameter accounting; we only need the headline numbers here. With `d_model = 512`, `n_kv_heads = 2`, `head_dim = 64`, and a SwiGLU MLP at `intermediate = 1408` (see [the modern architecture chapter](../02-transformer/10-modern-arch-improvements.html) and [SwiGLU/gated MLPs](../02-transformer/06-transformer-block.html) for the mechanisms), one transformer block costs:

$$
P_\text{block} = \underbrace{2d^2 + 2\,d\,d_\text{kv}}_{\text{attention (GQA)} \,=\, 655{,}360} + \underbrace{3\,d\,d_\text{ffn}}_{\text{SwiGLU} \,=\, 2{,}162{,}688} \approx 2.82\text{M params}
$$

Here the attention count is the two square projections $W_Q, W_O \in \mathbb{R}^{d\times d}$ plus the two GQA-shrunk projections $W_K, W_V \in \mathbb{R}^{d \times d_\text{kv}}$ with $d_\text{kv} = n_\text{kv} \cdot \text{head\_dim} = 2 \times 64 = 128$, so $2(512)^2 + 2(512)(128) = 524{,}288 + 131{,}072 = 655{,}360$, and the SwiGLU MLP is three $d \times d_\text{ffn}$ matrices, $3(512)(1408) = 2{,}162{,}688$. Thirty of those blocks cost `≈84.6M` params. That number is **independent of vocabulary size** — it is fixed by depth, width, and the GQA/SwiGLU shapes. The embedding table is not:

$$
P_\text{embed} = V \times d_\text{model}
$$

At `V = 32{,}768`, `d_model = 512`: `P_embed = 32{,}768 × 512 = 16{,}777{,}216 ≈ 16.8`M params, giving Stack-100M's total of `≈84.6M + 16.8M ≈ 101.4`M.

### Tied embeddings: the first lever

Stack-100M **ties** the input embedding and output (unembedding / `lm_head`) weight matrices — the same `V × d_model` matrix both looks up a token's input vector and produces the logit distribution over the vocabulary at the output, following [Press & Wolf, *Using the Output Embedding to Improve Language Models*, 2017](../02-transformer/02-embeddings-input.html). Weight tying is a well-known regularizer (input and output representations are forced to share a coordinate system) and, at small model sizes, a serious parameter saver: without tying, `V × d_model` params are spent twice.

!!! example "Worked example: what tying is worth, in layers"
    Untied, a 32,768-vocabulary embedding pair costs `2 × 16.78M = 33.55M` params. Tied, it costs `16.78M`. The saving — `16.78M` params — is worth
    $$
    \frac{16.78\text{M}}{2.82\text{M / layer}} \approx 5.95 \approx 6 \text{ layers}
    $$
    of additional depth at fixed total parameter count. Put differently: tying the embeddings is worth roughly a fifth more depth than Stack-100M's 30 layers, for free. This is exactly the kind of trade this chapter is about, and it is why every capstone chapter after this one treats "tied embeddings" as non-negotiable, not a minor implementation detail.

{{fig:tied-embeddings-one-matrix-two-jobs}}

### The tradeoff table: parameters, and measured compression

Now hold the block architecture fixed (30 layers × 2.82M/layer, i.e. everything [Chapter 14.4](../14-capstone/04-architecture.html) fixes) and ask: against a nominal 100M-parameter budget, how much does the tied embedding table cost at different vocabulary sizes — and what does the extra vocabulary actually *buy* in compression? The last column is not an estimate. We retrained the tokenizer at each size on the same 7.82 MB training split and measured bytes/token on the 810,623-byte held-out split.

| `vocab_size` | tied embed params | % of a 100M budget | layers affordable at 100M | vs 30 | held-out bytes/token |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 4.19M | 4.2% | ~34 | +4 | 3.673 |
| 16,384 | 8.39M | 8.4% | ~33 | +3 | 4.002 |
| **32,768 (Stack-100M)** | **16.78M** | **16.8%** | **~30** | **baseline** | **4.196** |
| 50,257 (GPT-2) | 25.73M | 25.7% | ~26 | −4 | 4.244 † |
| 65,536 | 33.55M | 33.6% | ~24 | −6 | 4.244 † |
| 100,277 (cl100k-scale) | 51.34M | 51.3% | ~17 | −13 | 4.244 † |

*(Methodology: `layers affordable` = `(100M − V·512) / 2.82M`, rounded — holding total parameters at a nominal 100M and letting depth absorb whatever the embedding table doesn't consume. Stack-100M's actual total lands at ≈101.4M, not exactly 100M, which is why the 32,768 row reads "~30" rather than exactly 30.)*

**† and this is the most instructive result in the table.** On a 7.82 MB sample, the trainer *ran out of repeated pairs* after 43,661 merges — a maximum reachable vocabulary of $256 + 43{,}661 + 9 = 43{,}926$. Every row at 50,257 and above is therefore the *same* tokenizer, padded with `<|unused_N|>` fillers, which is exactly the shortfall guard firing. Two lessons. First, a vocabulary is only as large as your *sample* can support: this run needed ~178 bytes of text per vocabulary entry merely to **fill** the table (7.82 MB bought 43,926 entries), and the deep merges you get at that ratio are artifacts of individual documents rather than statistics — which is why the production script above samples 500 MB, roughly 15 KB per entry, for 32,768. Second, without the shortfall guard the trainer would have silently handed 14.4 an embedding table of the wrong size.

This table is the number that motivates PLAN's headline claim: **a 50,257-entry vocabulary — GPT-2's, the "obvious" default — would eat about a quarter of a 100M-parameter budget**, on par with four entire transformer layers, while buying (on this corpus) about 1% more compression than 32,768. That is not a subtle effect at this scale; it is the difference between a 26-layer and a 30-layer model, and depth is exactly the axis the "deep-and-thin" philosophy behind Stack-100M's architecture (MobileLLM, [Liu et al., 2024](../14-capstone/04-architecture.html); see also [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)) is trying to protect.

{{fig:vocab-vs-depth-budget-lever}}

### Deriving the vocabulary instead of asserting it

The parameter table shows a cost. The compression column shows a benefit. Neither alone picks a number — but together, with the standard $C \approx 6ND$ FLOP rule ([Chapter 3.4](../03-pretraining/04-scaling-laws.html)), they do. Hold the **text budget** fixed (bytes of corpus are the resource you actually have) and note that changing $V$ moves *both* terms:

- $N(V) = 84.6\text{M} + 512\,V$ — bigger vocabulary, more parameters per forward pass.
- $D(V) = B / \text{bpt}(V)$ — bigger vocabulary, fewer tokens for the same $B$ bytes of text.

Stack-100M's 20B-token budget at the measured 4.196 bytes/token corresponds to $B \approx 83.9$ GB of text. Plugging the measured compression numbers in:

| `vocab_size` | bytes/token | $N$ | $D$ (tokens for 83.9 GB) | $C = 6ND$ | relative |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 3.673 | 88.79M | 22.85B | $1.22\times10^{19}$ | 1.001 |
| 16,384 | 4.002 | 92.99M | 20.97B | $1.17\times10^{19}$ | **0.962** |
| 32,768 | 4.196 | 101.38M | 20.00B | $1.22\times10^{19}$ | 1.000 |
| 50,257 | 4.244 † | 110.33M | 19.77B | $1.31\times10^{19}$ | 1.076 |

!!! example "Worked example: reading the compute curve honestly"
    The curve has a real minimum, and it is not at 32,768 — it is near **16,384**, about 4% cheaper in training FLOPs than our choice, with 8,192 and 32,768 essentially tied and 50,257 about 8% worse. Three things to take from that:

    **(1) The direction is right, the magnitude is small.** Doubling the vocabulary from 16k to 32k costs ~4% of the training compute; going to GPT-2's 50k costs ~11% relative to the optimum. Compression gains are logarithmic in $V$ (each doubling of the vocabulary buys steadily less) while embedding cost is *linear* in $V$ — so a minimum must exist, and at small $N$ it sits at a small $V$.

    **(2) Our measurement is biased against large vocabularies.** The 7.82 MB sample saturates at ~44k merges, so the bytes/token at 32,768 (and everything above) is pessimistic: the real 20B-token mix would fill those slots with genuinely useful merges and push compression higher, moving the minimum right. Re-run this table on *your* several-hundred-MB sample before trusting the exact location of the minimum. The methodology, not our numbers, is the deliverable.

    **(3) A subtlety in $6ND$ with tied embeddings.** The tied matrix is counted **once** in $N$, which is the right approximation here: the input side is a *gather* (no FLOPs), while the output `lm_head` is a real $d \times V$ matmul. So $N = 84.6\text{M} + 512V$ correctly charges the vocabulary exactly once, on the output side where the FLOPs are.

This is also precisely the question studied at scale by [Tao et al., *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies* (2024)](https://arxiv.org/abs/2407.13623). Their central result is that the compute-optimal vocabulary grows with the non-vocabulary parameter count but **sublinearly** — large models are typically under-vocabularied, and, read in the other direction, small models deserve small vocabularies. We do not import their fitted constants (they were fit on a different mix and a different tokenizer family, and extrapolating a power law two orders of magnitude below its fitting range is exactly the sin [Chapter 14.5](../14-capstone/05-mini-scaling-laws.html) warns against). The point is that the *shape* of their result and the shape of our measured curve agree: at $N_\text{non-vocab} \approx 84.6$M, the compute-optimal vocabulary is in the low tens of thousands, not the hundred-thousands that frontier models use.

### The vocabulary you pick is also an activation-memory decision

Parameters and FLOPs are not the binding constraint on a 24 GB GPU. **Logits are.** The `lm_head` output is the largest single activation in a small model, and it scales with $V$:

$$
\text{logits bytes} = (\text{micro-batch} \times \text{seq len}) \times V \times \text{bytes/element}
$$

[Chapter 14.7](../14-capstone/07-pretraining-run.html) uses `micro_batch_size = 32` at `seq_len = 2048`, i.e. 65,536 positions per forward pass:

!!! example "Worked example: the logits tensor is bigger than the model"
    At $V = 32{,}768$: $65{,}536 \times 32{,}768 = 2^{31}$ logits. In bf16 that is exactly **4.0 GiB** — and the naive `F.cross_entropy` path upcasts to fp32 for the softmax, adding **8.0 GiB**, with another 4 GiB for the gradient in the backward. Sixteen gigabytes of activations for a model whose *weights* are 0.2 GiB.

    At $V = 50{,}257$ the same micro-batch needs 6.1 GiB (bf16) plus 12.3 GiB (fp32) — on a 24 GB 4090 (PLAN's secondary tier) that alone is fatal. At $V = 8{,}192$ it is 1.0 GiB. **This is a far more visceral argument for a small vocabulary than the parameter count**, and it is why the "just use a 128k vocab" instinct breaks first at small scale.

The 2026 standard fixes, all of which apply directly and none of which change the math:

- **Chunked cross-entropy**: split the sequence dimension into chunks, materialize logits for one chunk at a time, accumulate the loss. Costs a little recompute, cuts peak logits memory by the chunk factor.
- **Fused linear + cross-entropy kernels**: Liger Kernel's `LigerFusedLinearCrossEntropy` fuses the `lm_head` matmul with the loss so the full logits tensor is never materialized in HBM at all — a Triton kernel of exactly the kind [Chapter 4.4](../04-kernels-efficiency/04-triton-kernels.html) teaches, and the single highest-leverage memory optimization for a small-model training loop.
- **Cut Cross-Entropy** (Wijmans et al., 2024): computes the loss with only the *correct-class* logit plus an online log-sum-exp, reducing the logits memory to effectively negligible.

See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html) and [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) for the general treatment.

!!! tip "Practitioner tip: always pad vocab_size to a multiple of 64 or 128"
    Tensor cores want matrix dimensions that are multiples of 8 (fp16/bf16) and prefer multiples of 64 or 128 so the `d × V` GEMM tiles evenly. Karpathy's well-known nanoGPT observation was that padding GPT-2's 50,257 to **50,304** ($= 128 \times 393$) — adding 47 tokens the model can never emit — measurably *speeds up* training, because the ragged tail tile disappears. `32,768 = 2^{15}` is a multiple of 128 for free, which is one concrete reason to prefer it over the superficially similar 32,000 (Llama 2's vocabulary, which is $250 \times 128$ — also fine — versus, say, 32,001, which would not be). If you pick an odd number, pad it and mask the extra logits to $-\infty$ at sampling time.

### Why 32,768 and not smaller

The table's left side shows the opposite failure mode. Shrinking to 8,192 or 16,384 buys back a few layers and (on our measurement) a hair of training compute, but costs compression: at 8,192 the same text needs 14% more tokens, which is 14% more forward passes per document at *inference* time, forever. Byte-level fallback means a tiny vocabulary is never *incorrect* — nothing is ever unrepresentable — but it is *inefficient*, and unlike training FLOPs, the inference cost is not a one-time payment. That asymmetry is the same deployment-economics argument that justifies over-training in [Chapter 14.2](../14-capstone/02-data-pipeline.html), pointing the other way.

So `32,768` is chosen where three curves are all still flat or favorable: the embedding table is a meaningful but not dominant slice of the budget (≈17%, versus ≈26% for 50,257 or ≈4% for 8,192); training compute is within ~4% of the measured optimum; and the vocabulary is large enough that a byte-level BPE tokenizer on a code-and-math-heavy mix captures the multi-character substrings — English morphemes, indentation-heavy code idioms, LaTeX-flavored math — that make compression efficient. It is also, not incidentally, a friendly number: $2^{15}$ is a multiple of 128 (tensor cores, above) and packs comfortably into a `uint16` (max 65,535) for the shard format [Chapter 14.2](../14-capstone/02-data-pipeline.html) writes to disk, with more than half the id space to spare.

!!! note "Aside: the softmax bottleneck"
    Tying pushes the *same* `d_model = 512`-dimensional space to serve two jobs: representing "what token is this" on the input side, and discriminating among 32,768 possible next tokens on the output side. Yang, Dai, Salakhutdinov & Cohen's *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model* (2018) shows that a softmax classifier's output distribution is fundamentally rank-limited by the hidden dimension — a narrow, tied model asks a 512-dimensional space to do double duty. In practice this cost is modest next to what tying saves at 100M scale, but it is a real tension, not a free lunch, and one more reason vocabulary size cannot be chosen in isolation from `d_model`.

!!! interview "Interview Corner"
    **Q:** You're building a 100M-parameter language model and someone suggests reusing a 100k-token vocabulary "because bigger vocabularies compress better and bigger models use them." What's wrong with that reasoning at this scale, and how would you actually decide the vocabulary size?

    **A:** The reasoning conflates two regimes. At 100B+ parameters, a 100k-entry tied embedding is a rounding error against the rest of the network, so the compression benefit dominates and bigger vocabularies are close to free — that is exactly Tao et al.'s (2024) finding that large models are usually *under*-vocabularied. At 100M total parameters the same table can be a third or more of the entire budget, trading directly against depth and width, which is where a small model's capacity lives. And there is a second constraint people forget: the logits tensor. At a 32-sequence × 2048-token micro-batch, a 32k vocabulary produces 4 GiB of bf16 logits; a 50k vocabulary produces 6.1 GiB, plus an fp32 softmax copy — on a 24 GB GPU that, not the parameter count, is what OOMs first.

    The right method is to *derive* it rather than inherit it. Fix a text budget, train the tokenizer at several vocabulary sizes (it takes seconds), measure bytes/token on a **held-out** slice, and compute $C = 6\,N(V)\,D(V)$ with $N(V) = N_\text{blocks} + V d_\text{model}$ and $D(V) = B/\text{bpt}(V)$. Compression gains are logarithmic in $V$ and embedding cost is linear, so there is a genuine minimum; at ~85M non-embedding parameters I'd expect it in the low tens of thousands. Then adjust for the things FLOPs don't capture: pad to a multiple of 128 for tensor cores, stay under 65,536 so token ids fit in `uint16`, and weight inference cost more heavily than training cost if you plan to serve the model. There is no universal answer — the answer is a function of the budget you actually have.

## Worked Example: Compression Ratio and Round-Trip Correctness

We measured the tokenizer trained above — 32,768-entry vocabulary, 32,503 merges, trained on the 7,818,444-byte training split — on three concrete inputs. Every number is from the held-out split or from strings the tokenizer never saw during training.

!!! example "Worked example: measured compression, three ways"
    **Held-out average.** Encoding the entire 810,623-byte held-out split produced 193,177 tokens:
    $$
    \frac{810{,}623 \text{ bytes}}{193{,}177 \text{ tokens}} \approx 4.196 \text{ bytes/token}
    $$
    (For reference, on the *training* split it reaches 4.259 bytes/token — a small, healthy train/held-out gap that tells you the merge table generalized rather than memorizing this corpus's idiosyncratic strings.)

    **Clean English prose** — a 184-byte sentence from this book's tokenization chapter ("Tokenization is the most underestimated component of the stack. It is not part of the network, it is not trained by gradient descent, and it is frozen for the entire life of the model.") — encoded to 38 tokens:
    $$
    \frac{184}{38} \approx 4.84 \text{ bytes/token}
    $$
    higher than the held-out average, because clean natural-language prose (no markdown syntax, no code punctuation) is exactly what BPE compresses best.

    **A short Python snippet** (214 bytes of a `train_bpe`-style function body) encoded to 55 tokens:
    $$
    \frac{214}{55} \approx 3.89 \text{ bytes/token}
    $$
    markedly worse than prose — code's density of punctuation, brackets, and multi-level indentation gives a vocabulary trained on a prose-heavy mix fewer long, reusable substrings to exploit. This is exactly why Stack-100M's mix reserves a dedicated 10% code slice ([Chapter 14.2](../14-capstone/02-data-pipeline.html)): a tokenizer (and later, a model) that never sees code compresses and predicts it markedly worse than prose it was actually trained on.

    All three numbers are *measured* from the exact `StackTokenizer` class defined above, not estimated — reproduce them by running `train_tokenizer.py` against your own sample. As a sanity floor: byte-level BPE at any vocabulary size can never do *worse* than 1.0 bytes/token (the un-merged byte fallback), and GPT-2's larger 50,257-entry vocabulary typically lands on the order of 4.0–4.3 bytes/token on general English prose (see [Chapter 2.1](../02-transformer/01-tokenization.html)) — so 32,768 landing right in that band on a similar corpus is exactly the expected shape of the tradeoff. Note also that 4.196 bytes/token is what turns PLAN's 20B-token budget into the ≈84 GB of raw text that [Chapter 14.2](../14-capstone/02-data-pipeline.html) must source and this chapter's parallel encoder must chew through.

Round-trip correctness is non-negotiable — a tokenizer that cannot reconstruct its input byte-for-byte silently corrupts every downstream stage:

```python
# Round-trip verification against a freshly-trained tokenizer.
from stacklm.tokenizer import StackTokenizer, VOCAB_SIZE, _SPLIT_RE

tok = StackTokenizer()
tok.train_from_iterable(open("corpus_sample.txt", encoding="utf-8"),
                        vocab_size=VOCAB_SIZE)

sample = ("Tokenization is the most underestimated component of the stack. "
          "It is not part of the network, it is not trained by gradient descent, "
          "and it is frozen for the entire life of the model.")
assert tok.decode(tok.encode(sample)) == sample          # exact byte-for-byte match

# 1. The pre-tokenizer must be LOSSLESS or nothing downstream can hold.
corpus = open("corpus_sample.txt", encoding="utf-8").read()
assert "".join(_SPLIT_RE.findall(corpus)) == corpus

# 2. Spot-check on a real corpus slice -- Unicode, markdown, code fences.
assert tok.decode(tok.encode(corpus[1000:6000])) == corpus[1000:6000]

# 3. Adversarial round trips the happy path misses.
for hard in ["café 🚀 — ünïcode", "a\r\nb\tc", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "", " ",
             "<|assistant|> literal in untrusted text", "0123456789" * 40]:
    assert tok.decode(tok.encode(hard)) == hard

print("round trip OK on prose, corpus slice, and adversarial strings")
```

Every assertion passes. This is the guarantee byte-level BPE buys you and character- or word-level tokenizers cannot: because every string is, worst case, a sequence of raw UTF-8 bytes each with its own id (0–255), there is no input this tokenizer can fail to encode and later reconstruct exactly — not an emoji, not a mixed-script string, not a stray invalid byte.

## Wiring Into the Rest of the Capstone

Four concrete commitments this chapter locks in for the rest of Part XIV:

- **The shard format.** [Chapter 14.2](../14-capstone/02-data-pipeline.html) packs `StackTokenizer.encode()` output into `uint16` memmap `.bin` shards. `vocab_size = 32{,}768` fits comfortably inside `uint16`'s `0`–`65{,}535` range — a vocabulary above 65,536 would have forced `uint32` shards, doubling the data pipeline's disk footprint for no benefit. Practically, 14.2 imports `encode_corpus` from this chapter and runs it as a CPU-only pass before any GPU is rented.
- **The embedding table's shape.** [Chapter 14.4](../14-capstone/04-architecture.html) allocates `nn.Embedding(32768, 512)`, tied to the output projection, and every parameter-count claim in that chapter (and this one) depends on `vocab_size` never moving after this chapter ships `tokenizer/stack100m-32768.json`. The shortfall guard is what makes that a *checked* invariant rather than a hope.
- **The loss mask.** [Chapter 14.9](../14-capstone/09-post-training.html)'s SFT loop masks the loss to tokens *after* `<|assistant|>` and *before* `<|end|>`, and separately masks out anything between `<|tool_result|>` and the next `<|assistant|>` in [Chapter 14.10](../14-capstone/10-agentic-narrow.html)'s agent traces — both rely on the exact ids this chapter assigned (32764 and 32767) being stable, and on `allowed_special` keeping forged boundaries out of training data.
- **The ecosystem handoff.** `tokenizer/stack100m-32768-hf/` is what TRL's `SFTTrainer` and `DPOTrainer` load in 14.9, what vLLM's `--tokenizer` flag points at in [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html), and what llama.cpp's GGUF converter reads for the int4 laptop build. The `chat_template` baked into `tokenizer_config.json` is the *same* ChatML-over-reserved-tokens format 14.9 trains on — one source of truth, so the served model cannot drift from the trained one.

Get this chapter wrong and every one of those four commitments has to be redone from scratch, at the cost of every checkpoint trained in between. That is the real argument for spending a full chapter — and a few real CPU-seconds — getting the tokenizer right before writing a single line of the model itself.

## Key Takeaways

!!! key "Key Takeaways"
    - A tokenizer is trained once, by counting statistics, and then frozen for the life of the model — vocabulary size, the pre-tokenizer regex, and special-token ids are all effectively permanent once training begins, so all three belong in the saved artifact.
    - The pre-tokenizer is a design decision, not boilerplate: GPT-2's unbounded ` ?\p{N}+` produces content-dependent number tokens, which is why every tokenizer since `cl100k_base` caps digit runs (we use `\p{N}{1,3}`) — and why our RLVR arithmetic task in 14.9 is not sabotaged before it starts.
    - Lazy heap deletion is only correct if you re-push on **every** count change. Pushing only on increments silently drops decremented pairs forever, producing a wrong vocabulary and fewer merges than requested — a `touched` set is the whole fix.
    - An inverted pair-index plus a correctly-refreshed lazy max-heap turns an `O(merges × corpus)` job into 3.4 seconds for all 32,503 merges on a 7.8 MB corpus (measured), including ~1.0 s of pre-tokenizing.
    - Training the tokenizer is seconds; *using* it on 84 GB is hours. A per-chunk dict cache (88.9% hit rate) is worth ~5×, `multiprocessing` another ~7×, and exporting to `tiktoken`/`tokenizers` gets you compiled speed without abandoning the vocabulary you trained.
    - Export or it didn't happen: walk the merge tree to token bytes, map through GPT-2's `bytes_to_unicode()`, and emit a `tokenizer.json` — then assert in CI that your encoder, `tiktoken`, and `PreTrainedTokenizerFast` produce **identical** ids. That file is what TRL, vLLM, and llama.cpp actually load.
    - Reserve every special token a project will ever need *before* training, in a fixed order; guard against a too-small sample by padding with `<|unused_N|>` fillers placed *before* the real specials, so `vocab_size` is exact and the nine real ids never move.
    - At 100M parameters vocabulary size is a real design lever: GPT-2's 50,257 would cost roughly a quarter of the budget (≈4 layers) to buy ~1% more compression, and tied embeddings (Press & Wolf, 2017) are worth ~6 layers of depth for one line of code.
    - Derive the vocabulary, don't inherit it: measure bytes/token on held-out text at several sizes and minimize $6\,N(V)\,D(V)$ at fixed text budget. Compression is logarithmic in $V$, embedding cost is linear, so a minimum exists — ours sat near 16k–32k, consistent with Tao et al. (2024)'s "small models deserve small vocabularies."
    - The logits tensor, not the parameter count, is usually what OOMs: 32k vocab × a 32×2048 micro-batch is 4 GiB in bf16 (plus an 8 GiB fp32 softmax copy). Fix it with chunked CE, Liger's fused linear-CE, or cut cross-entropy — and always pad `vocab_size` to a multiple of 128.

!!! sota "State of the Art & Resources (2026)"
    Byte-level BPE (this chapter's algorithm) remains the dominant tokenization scheme for production LLMs, but 2024–2026 research has pushed hard on three fronts: making tokenizers themselves better at compression (superword tokenization), asking whether an explicit tokenizer is needed at all (byte-level / patch-based models), and treating vocabulary size as a scaling-law variable rather than a constant.

    **Foundational work**

    - [Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units* (2016)](https://arxiv.org/abs/1508.07909) — the paper that brought BPE from data compression into NLP as a subword tokenizer.
    - [Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — introduced byte-level BPE, the pre-tokenizer regex, and the `bytes_to_unicode()` table every exporter still reproduces.
    - [Kudo & Richardson, *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer* (2018)](https://arxiv.org/abs/1808.06226) — the language-agnostic, whitespace-as-symbol framing behind many production tokenizer pipelines (and the `split_digits` option this chapter's arithmetic aside refers to).
    - [Press & Wolf, *Using the Output Embedding to Improve Language Models* (2017)](https://arxiv.org/abs/1608.05859) — the tied-embeddings result this chapter's parameter accounting (the "worth ~6 layers" argument) depends on.

    **Recent advances (2023–2026)**

    - [Tao, Liu, Dou, Muennighoff, Wan, Luo, Lin & Wong, *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies* (2024)](https://arxiv.org/abs/2407.13623) — makes vocabulary size a first-class scaling-law variable, finds compute-optimal $V$ grows sublinearly with non-vocabulary parameters, and argues most large models are under-vocabularied. Read in reverse, it is the argument for a small vocabulary at 100M.
    - [Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens* (2024)](https://arxiv.org/abs/2412.09871) — Meta's dynamic-entropy byte-patching architecture, the most credible recent attempt to match BPE-tokenized LLM quality without a fixed subword vocabulary at all.
    - [Liu, Hayase, Hofmann, Oh, Smith & Choi, *SuperBPE: Space Travel for Language Models* (2025)](https://arxiv.org/abs/2503.13423) — extends BPE to merge across whitespace into "superword" tokens, reporting meaningfully fewer tokens per document at fixed vocabulary size — directly relevant to this chapter's compression-vs-embedding-budget tradeoff.
    - [Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)](https://arxiv.org/abs/2411.09009) — Apple's cut cross-entropy, which removes the logits tensor from the memory budget entirely; the direct answer to this chapter's activation-memory section.

    **Open-source & tools**

    - [openai/tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokenizer. The `allowed_special` design this chapter's `encode()` follows originates here, and `tiktoken.Encoding(pat_str=..., mergeable_ranks=...)` is a 10-line drop-in for our merge table.
    - [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — the Rust trainer/encoder and, more importantly, the `tokenizer.json` format that `transformers`, TRL, vLLM, SGLang, and llama.cpp all consume.
    - [huggingface/transformers](https://github.com/huggingface/transformers) — `PreTrainedTokenizerFast` + `save_pretrained` is the packaging step that turns a merge table into an artifact the ecosystem can load.
    - [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — Triton kernels including `LigerFusedLinearCrossEntropy`, which fuses `lm_head` with the loss so the full logits tensor never lands in HBM.
    - [karpathy/minbpe](https://github.com/karpathy/minbpe) — a minimal, from-scratch reference implementation of byte-level BPE train/encode/decode, good for cross-checking this chapter's trainer against an independently written one.

    **Go deeper**

    - [Hugging Face LLM Course — Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5) — a worked, step-by-step walkthrough of the BPE training and tokenization algorithm this chapter implements.

## Further reading

- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*, 2016 — the paper that introduced BPE to NLP.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 — introduced byte-level BPE, the pre-tokenizer regex, and `bytes_to_unicode()`.
- Tao et al., *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*, 2024 — the reference for treating vocabulary size as a derived quantity rather than a default.
- Press & Wolf, *Using the Output Embedding to Improve Language Models*, 2017 — the tied-embeddings result this chapter's parameter accounting depends on.
- Yang, Dai, Salakhutdinov & Cohen, *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model*, 2018 — the rank-limitation argument behind the softmax-bottleneck aside above.
- Kudo & Richardson, *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing*, 2018 — the language-agnostic framing many production tokenizers build on, and the source of single-digit number splitting.
- Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, 2024 — the logits-memory fix named in the activation-memory section.
- HuggingFace `tokenizers` and `transformers` — the Rust trainer/encoder and the `tokenizer.json` packaging this chapter exports into.
- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, 2024 — the deep-and-thin small-model philosophy this chapter's depth-vs-vocabulary tradeoff protects; developed further in [Chapter 14.4](../14-capstone/04-architecture.html).

## Exercises

**1.** By default, `StackTokenizer.encode` is called with `allowed_special=frozenset()`, so a literal `<|assistant|>` typed into a user message is byte-encoded like ordinary text instead of mapping to id 32764. Explain concretely what could go wrong if `encode` instead recognized special-token strings everywhere by default, name the one place in the pipeline where this invariant is what makes a downstream guarantee trustworthy, and say what changes about this when you hand the tokenizer to HuggingFace `tokenizers`.

??? note "Solution"
    If `encode` recognized special-token *strings* everywhere, then any untrusted content — a user's chat message, or a tool's returned observation — that happened to contain the literal text `<|assistant|>` would tokenize to the real role-boundary id 32764. That lets an attacker forge turns: they could close the user turn early and open a fake assistant turn, or inject a fake `<|tool_result|>` (id 32767) to smuggle in an "observation" the model never actually retrieved. This is the tokenizer-level analogue of prompt injection.

    The fix is exactly the default: untrusted text is encoded with `allowed_special=frozenset()`, so special-token *ids* can only ever enter a sequence through code you control (the chat-template formatter, the packing code), never through content a user or tool supplied. Note the parallel encoder in `tokenizer_parallel.py` hardcodes `allowed_special=frozenset()` for exactly this reason — the pretraining corpus is untrusted scraped text.

    The place this matters most is the **SFT loss mask** in [Chapter 14.9](../14-capstone/09-post-training.html). That loop computes loss only on tokens after `<|assistant|>` (id 32764). The mask is trustworthy only because the default guarantees that every id 32764 in a training example is a real assistant boundary the formatter placed — not a string that some example's user turn happened to contain.

    **What changes in the ecosystem:** HuggingFace `tokenizers` takes the opposite default — an `AddedToken` is matched wherever it appears in the input string. So once you export, the invariant is no longer enforced by the tokenizer; you must enforce it at the boundary: pass `add_special_tokens=False`, and never concatenate untrusted content into a template string before tokenizing. (`AddedToken(..., special=True)` can be told to be skipped on decode, but that does not help on the encode side.) This is a good example of a security property that is *not* portable across an export and has to be re-established in the calling code.

**2.** The chapter fixes `vocab_size = 32768` with `256` reserved byte ids and `9` special tokens, giving `M = 32{,}503` merges. Suppose instead you targeted `vocab_size = 16384` with the *same* `9` special tokens in the *same* order. (a) How many merges `M` does the trainer learn? (b) What id does `<|bos|>` get? (c) What id does `<|user|>` get? (d) Would you have to change the `.bin` shard dtype in 14.2?

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

    (d) No — `uint16` holds $0..65{,}535$, so 16,384 fits with even more room to spare than 32,768. The dtype only has to change above 65,536, which is one of the reasons the chapter treats 65,536 as a hard practical ceiling rather than just an expensive option.

    Note that unlike in the real `vocab_size = 32768` layout (where `<|user|>` is 32763), the id is *different* — which is exactly why the chapter insists the vocabulary is frozen: change `vocab_size` and every special-token id moves.

**3.** Using the chapter's parameter accounting (`d_model = 512`, one transformer block $\approx 2.82$M params), consider raising the vocabulary to `V = 65{,}536`. (a) What does the **tied** embedding table cost? (b) What would it cost **untied**? (c) Using `layers affordable = (100\text{M} - V\cdot 512)/2.82\text{M}`, how many layers can a nominal 100M budget afford? (d) How large is the bf16 logits tensor for 14.7's `32 × 2048` micro-batch, and what does that imply on a 24 GB GPU?

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
    \frac{100{,}000{,}000 - 33{,}554{,}432}{2{,}820{,}000} = \frac{66{,}445{,}568}{2{,}820{,}000} \approx 23.56 \approx 24 \text{ layers}.
    $$
    That is $24 - 30 = -6$ layers versus Stack-100M's 30 — doubling the vocabulary costs roughly six transformer layers of depth.

    (d) The micro-batch is $32 \times 2048 = 65{,}536$ positions, so the logits tensor is
    $$
    65{,}536 \times 65{,}536 \times 2\text{ bytes} = 8{,}589{,}934{,}592 \text{ bytes} = 8\text{ GiB (bf16)},
    $$
    plus **16 GiB** if `F.cross_entropy` upcasts to fp32, plus another 8 GiB for the gradient. On a 24 GB card this OOMs on the very first forward pass, before any of the parameter arithmetic even matters. You would have to shrink the micro-batch (and raise `grad_accum` to compensate), or use a fused linear-cross-entropy kernel so the logits are never materialized. Note that this is a *sharper* constraint than the parameter count: 65,536 costs 6 layers of quality but 8–16 GiB of memory.

**4.** The CI toy path trains on `TOY_CORPUS = "the quick brown fox jumps over the lazy dog. " * 200` and asks for `vocab_size = 512`, yet asserts `toy.vocab_size == 512` and `toy.special_to_id["<|bos|>"] == 503`. Explain (a) why the trainer cannot produce 247 merges from this corpus, referring to the specific line in `train_bpe` that halts it, and (b) how the tokenizer nevertheless ends up with exactly 512 entries and `<|bos|>` at 503.

??? note "Solution"
    **(a)** The pre-tokenizer splits text into chunks that never cross word or whitespace boundaries, and BPE only ever merges *adjacent* symbols *within* a chunk. The toy corpus is one sentence of ~9 distinct words repeated 200 times, so it pre-tokenizes into only a handful of distinct chunk types. Each distinct chunk can be collapsed, merge by merge, down to a single symbol — and once every chunk is a single symbol there are no adjacent pairs left that repeat.

    At that point `train_bpe` hits its early-stop guard:

    ```python
    if live_count < 2:
        break   # nothing left that repeats; stop early
    ```

    The most frequent remaining pair occurs fewer than 2 times (or no pairs remain), so the loop breaks well before `num_merges = 512 - 256 - 9 = 247`. In practice this corpus yields on the order of 30 merges.

    **(b)** `train_from_iterable` computes `shortfall = num_merges - len(self.merges)`, warns, and then builds the special list as `fillers + SPECIAL_TOKENS` where `fillers = ("<|unused_0|>", ...)` has exactly `shortfall` entries. Ids are assigned bytes → merges → fillers → real specials, so the total is
    $$
    256 + \underbrace{M}_{\text{learned}} + \underbrace{(247 - M)}_{\text{fillers}} + 9 = 512
    $$
    for any $M$, and the nine real specials always occupy the final nine ids: `<|bos|>` at $512 - 9 = 503$, `<|tool_result|>` at 511. This is why the fillers are **prepended** to the special block rather than appended — appending would shift every real id down by `shortfall`, which is precisely the frozen-layout violation Exercise 6 explores.

    Without this guard the toy path would have to hand-pick a `vocab_size` the corpus can exactly fill, and — far worse — a slightly-too-small *production* sample would silently produce `vocab_size < 32768`, breaking `nn.Embedding(32768, 512)` in 14.4 with a shape error thousands of GPU-seconds later.

**5.** You have the trained 32,768-entry tokenizer and a 500 MB corpus sample. (a) Implement `bytes_per_token(tok, text)`. (b) Using the chapter's measured throughputs, estimate how long a single-process pure-Python `encode` pass over the full ~84 GB training corpus would take, and how you would get it under an hour. (c) Why must the byte count use `text.encode("utf-8")` and the encode call use the default `allowed_special`?

??? note "Solution"
    **(a)**
    ```python
    def bytes_per_token(tok: StackTokenizer, text: str) -> float:
        """UTF-8 bytes per token: higher == better compression.
        `text` is treated as untrusted, so no special-token strings are
        recognized (allowed_special defaults to the empty frozenset)."""
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(tok.encode(text))     # allowed_special=frozenset() by default
        return n_bytes / max(1, n_tokens)
    ```

    **(b)** The chapter measures ~5.8 MB/s for the warm, cached single-process path. At 20B tokens × 4.196 bytes/token ≈ 83.9 GB:
    $$
    \frac{83.9 \times 10^{9}}{5.8 \times 10^{6}} \approx 1.45 \times 10^{4}\text{ s} \approx 4.0 \text{ hours}.
    $$
    (Without the per-chunk cache, at ~1.2 MB/s, it would be ~19 hours.) Two ways under an hour, both in the chapter: `multiprocessing.Pool(16)` measured ~39.8 MB/s ⇒ ~35 minutes; or export to `tiktoken` (~17.9 MB/s single-threaded, ~1.3 h) and batch it across processes for a few minutes. Either way, run it as a **CPU-only** job *before* renting the A100 — the pretraining loop reads `uint16` shards and never calls `encode`.

    **(c)** The byte count must use `text.encode("utf-8")` because `len(str)` counts *codepoints*: an emoji is 1 codepoint but 4 bytes, and an accented Latin letter is 1 codepoint but 2 bytes, so `len(text)` would overstate compression on any non-ASCII corpus. And the encode call must use the default `allowed_special=frozenset()` so a stray `<|assistant|>` in the input is not collapsed to a single special id, which would flatter the ratio *and* silently violate the injection invariant.

    Expected ratios from the chapter's measurements: clean English prose ≈ $184/38 \approx 4.84$ bytes/token (best case — long, frequent, reusable substrings); a Python snippet ≈ $214/55 \approx 3.89$ (dense punctuation and indentation offer fewer long merges). The floor is 1.0 bytes/token (pure byte fallback).

**6.** A colleague wants to add `8` reserved special-token placeholders (like Llama 3's `<|reserved_special_token_N|>`) **but keep `vocab_size` fixed at 32768 and keep the 9 real special-token ids exactly where they are** (`<|bos|> = 32759` ... `<|tool_result|> = 32767`). (a) If they *append* the 8 reserved tokens after `<|tool_result|>`, what happens to the 9 real ids? (b) Show a placement that keeps all 9 real ids unchanged, and state what you pay for it. (c) Why does `to_hf_tokenizer` still produce the correct ids under your fix?

??? note "Solution"
    With `vocab_size` fixed at 32768, the number of merges depends on the total number of specials $S$:
    $$
    M = V - 256 - S.
    $$
    Adding 8 reserved tokens makes $S = 9 + 8 = 17$, so $M = 32{,}768 - 256 - 17 = 32{,}495$ (eight fewer merges than the original 32,503). The special block therefore starts at $256 + M = 32{,}751$ instead of $32{,}759$ — it shifts *down* by 8.

    (a) **Appending** the reserved tokens (order = the 9 real ones, then 8 reserved) puts `<|bos|>` at the start of the block, id $32{,}751$, not $32{,}759$. Every one of the 9 real ids shifts down by 8: `<|tool_result|>` lands at $32{,}751 + 8 = 32{,}759$ instead of $32{,}767$. This breaks the frozen invariant and invalidates every checkpoint's embedding rows.

    (b) **Prepend** the 8 reserved tokens instead (order = 8 reserved, then the 9 real ones). The block still starts at $32{,}751$, but the reserved tokens absorb ids $32{,}751 .. 32{,}758$, and `<|bos|>` lands at:
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

    What you pay: 8 fewer learned merges (32,503 → 32,495), a negligible compression cost. What you avoid: shifting any of the 9 real ids. This is the same mechanism the shortfall guard uses — and the general rule is that *anything* inserted into the special block must go **before** the frozen tokens, never after.

    (c) `to_hf_tokenizer` builds the BPE model vocabulary from `token_bytes()` (256 bytes + $M$ merges = 32,751 entries) and then calls `add_special_tokens` in `special_to_id` iteration order, which is insertion order: reserved first, then the nine real ones. HuggingFace assigns added-token ids sequentially starting at `len(vocab)`, so they land at $32{,}751 .. 32{,}767$ in exactly that order. The assertion loop at the end of `to_hf_tokenizer` (`hf.token_to_id(s) == i` for every special) is what turns this reasoning into a checked invariant — it would fail loudly if anyone appended instead of prepended.

**7.** The chapter switches the pre-tokenizer from GPT-2's ` ?\p{N}+` to `\p{N}{1,3}`. (a) Show how `1234567` and `2026` tokenize under each. (b) Explain why the unbounded version is bad for the RLVR arithmetic task in 14.9. (c) `\p{N}{1,3}` is not the arithmetic-optimal choice either — what is, and what does it cost? (d) What would you have to re-do if you changed the pattern *after* pretraining?

??? note "Solution"
    **(a)** Under GPT-2's pattern, `1234567` is a single pre-token chunk; whether it becomes one token, two, or seven depends entirely on which digit substrings happened to be frequent in the training corpus. `2026` is likewise a single chunk, and in a web-scraped corpus `2020` is far more frequent than `2031`, so the two get different token counts. Under `\p{N}{1,3}` the chunker is content-independent: `1234567` → `'123' '456' '7'`, and `2026` → `'202' '6'`. (Measured with the chapter's tokenizer; note also that the space is never absorbed, so `year 2026` yields `' '` as its own token.)

    **(b)** RLVR (Chapter 14.9) rewards exact-match correctness on integer arithmetic. If `47 + 58` and `48 + 58` are segmented differently — one operand a single token, the other two — the model must learn the algorithm separately for each segmentation pattern, from a training distribution where segmentations are distributed by *corpus frequency* rather than by numeric structure. That is a large, gratuitous increase in sample complexity for a 100M-parameter model that has very little capacity to spare, and it shows up as the classic failure mode where a model handles round numbers well and arbitrary ones badly. Capping the run makes the segmentation a deterministic function of digit position, so what the model learns on one number transfers to the next.

    **(c)** **Single-digit** splitting (`\p{N}`, i.e. SentencePiece's `split_digits`, as used by Llama 2 and Gemma) is better still: every digit is its own token, so column-wise addition is positionally regular and carries line up. The cost is roughly 2–3× more tokens on numeric text, paid on every training step and every inference forward pass over any document containing numbers — which for a 95%-non-math mix is a bad trade overall. `{1,3}` is the frontier compromise; single-digit is the right choice if arithmetic is your headline capability.

    **(d)** Everything. The merge table is defined *relative to* the chunk boundaries the pattern produces, so a different pattern applied to the same merges produces different — and wrong — tokenizations. You would have to retrain the tokenizer, which changes every id, which invalidates every embedding row, which means re-running the ~20B-token pretraining. This is exactly why `save()` writes `pattern` into the artifact and `load()` raises on a mismatch rather than quietly proceeding: a silent regex drift is a class of bug that would otherwise surface as mysteriously degraded loss thousands of GPU-seconds into a run.

**8.** Write the failing test for the lazy-heap bug. Specifically: construct a small `word_freqs` where the buggy trainer (re-pushing only on increments) drops a pair that should still be merged, and state what you would assert in CI to catch this class of bug in general.

??? note "Solution"
    The mechanism to reproduce: a pair $P$ must (i) have its count *reduced* by an earlier merge without being eliminated, and (ii) later become the maximum. Take two words with frequencies chosen so that destroying $P$ inside one of them still leaves $P$ as the eventual best candidate:

    ```python
    from stacklm.tokenizer import train_bpe

    # symbols: a=97 b=98 c=99 d=100 (raw byte ids)
    word_freqs = {
        (99, 97, 98): 6,        # "cab": contains (97,98) and (99,97)
        (100, 97, 98): 4,       # "dab": contains (97,98) and (100,97)
        (99, 97): 5,            # "ca":  boosts (99,97) so it merges FIRST
    }
    merges = train_bpe(word_freqs, num_merges=4)
    # (99,97) has count 6+5 = 11 -> merged first, which destroys (97,98)
    # inside "cab" and drops pair_counts[(97,98)] from 10 to 4 with no re-push.
    # A correct trainer still merges (97,98) afterwards; the buggy one never does.
    assert (97, 98) in merges, "decremented pair was silently dropped from the heap"
    ```

    More important than any single reproduction is the *class* of assertion that catches it. Three that belong in CI, in increasing order of strength:

    1. **No silent shortfall.** `assert len(train_bpe(word_freqs, k)) == k` whenever the corpus is provably rich enough (the fillers path in `train_from_iterable` exists precisely because this cannot always hold — so assert `shortfall == 0` in the production script).
    2. **Invariant check under a slow reference.** Re-implement the naive $O(\text{merges} \times \text{corpus})$ recount trainer in ~15 lines, run both on a small random corpus, and assert the merge lists are *identical*. The naive trainer has no heap, so it cannot have this bug; it is the oracle.
    3. **Cross-library agreement.** Train HuggingFace `trainers.BpeTrainer` on the same corpus with the same pattern and vocab size and assert the first 100 merges match in rank order and the token *sets* overlap above 99%. This chapter reports exactly those numbers; a heap bug would collapse both immediately.

    The general lesson is that lazy deletion trades a decrease-key operation for a staleness check, and the staleness check is only sound if *every* mutation publishes a fresh entry. Whenever you see "we skip stale entries on pop," look for the write path that forgot to push.
