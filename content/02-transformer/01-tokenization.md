# 2.1 Tokenization: BPE, WordPiece, Unigram & Byte-Level

A Large Language Model (LLM) does not read text. It reads **integers**. Before a single matrix multiply happens, before the first embedding lookup, a piece of software called the **tokenizer** has chopped your string into a sequence of discrete units and mapped each one to an index in a fixed vocabulary. That sequence of integers is the *only* thing the neural network ever sees. The model's entire universe — every word it can produce, every concept it can name, the way it counts digits, the languages it speaks fluently versus haltingly — is shaped, and often *limited*, by choices made in this preprocessing step.

Tokenization is the most underestimated component of the stack. It is not part of the network, it is not trained by gradient descent, and it is frozen for the entire life of the model. Yet it silently determines context-window cost (more tokens per document means fewer documents fit in the window and a larger inference bill), arithmetic ability, code quality, and the notorious gap in cost and performance between English and lower-resource languages. Famous failure modes — a model that cannot reliably spell "strawberry" or count its `r`s, the `SolidGoldMagikarp` "glitch tokens" that made GPT-2/3 misbehave, prompt-injection tricks that exploit Unicode normalization — are all tokenizer artifacts, not reasoning failures.

This chapter builds tokenization from first principles. We start with *why* subwords beat both words and characters, then implement **Byte-Pair Encoding (BPE)** from scratch — a complete, runnable trainer and encoder you could drop into a project. We cover the three algorithms that power essentially every production LLM: **BPE** (GPT family), **WordPiece** (BERT), and **Unigram** (the SentencePiece/T5 default). We explain **byte-level** BPE and why it guarantees no input is ever un-tokenizable. We dig into `tiktoken`, vocabulary-size tradeoffs with real arithmetic, special tokens, and the multilingual, code, and numeric pitfalls that bite practitioners daily. The tokens this chapter produces are the integers that feed directly into [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html).

## Why Subwords? The Granularity Problem

{{tool:tokenizer-playground}}

Imagine you must choose the atomic unit of text. There are three obvious candidates, and two of them are traps.

**Words.** Split on whitespace and punctuation, assign each distinct word an integer. This is how classic Natural Language Processing (NLP) worked. It fails badly for modern LLMs:

- **The vocabulary is unbounded.** English alone has hundreds of thousands of word forms; add morphology (`run`, `runs`, `running`, `runner`), compounds, names, URLs, hashtags, and code identifiers, and the vocabulary explodes. You must cap it, which forces an **out-of-vocabulary (OOV)** token `<unk>` for everything unseen. The model literally cannot represent — and therefore cannot generate — any word not in its training vocabulary. A new product name, a rare surname, a typo: all collapse to `<unk>` and the information is gone.
- **Morphology is wasted.** `running` and `runs` get unrelated integer IDs and unrelated embeddings; the model must relearn from scratch that they share a root. Word tokenization throws away the compositional structure of language.
- **The embedding matrix dominates parameters.** A 250,000-word vocabulary at $d_\text{model}=4096$ is $250{,}000 \times 4096 \approx 1.02$ billion parameters in the embedding table alone, before any Transformer layer exists.

**Characters (or bytes).** Go the other way: vocabulary is tiny (256 byte values, or ~150k Unicode code points), and there is *no OOV* — every string is representable. But:

- **Sequences become brutally long.** A 1,000-word document is maybe 1,300 word-tokens but roughly 6,000 characters. Since self-attention is $O(n^2)$ in sequence length $n$ (see [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html)), a 4–5× longer sequence is a 16–25× larger attention cost. Your effective context shrinks and your bill grows.
- **The model spends capacity learning spelling.** Each layer can only mix information a limited distance; forcing the network to assemble "information" from nine characters before it can reason about the *concept* wastes depth.

**Subwords** are the Goldilocks answer, and the central insight of modern tokenization:

> Keep frequent words whole (`the`, `running`, `import`), break rare words into reusable, meaningful pieces (`tokeni` + `zation`, `Anthrop` + `ic`), and fall back to bytes for anything truly novel. Common things are cheap (one token); rare things are still representable (several tokens); nothing is ever OOV.

This single idea — a **fixed-size vocabulary of variable-length pieces, optimized so that frequent substrings get their own token** — is what BPE, WordPiece, and Unigram all implement, differing only in the *objective* and *algorithm* used to choose the pieces. The result is a sweet spot: a vocabulary of typically 30k–256k entries, an average of roughly **0.75 words per token** for English (equivalently ~1.3 tokens per word), and graceful degradation to bytes for the long tail.

!!! note "Aside: the tokenizer is trained, but not by backprop"
    A tokenizer has a *training* phase — it learns its merges/vocabulary from a text corpus by counting statistics — but this is a **one-time, gradient-free** procedure that happens *before* model pretraining. Once frozen, the same tokenizer is used for the model's entire life. Changing it later means you cannot reuse the embedding table at all, because integer ID 4{,}521 now means something completely different. This is why tokenizers are chosen carefully and rarely changed.

{{fig:granularity-spectrum}}

## Byte-Pair Encoding From Scratch

BPE was originally a **data-compression** algorithm (Gage, 1994) and was adapted for NLP by Sennrich, Haddow & Birch (*Neural Machine Translation of Rare Words with Subword Units*, 2016). The idea is beautifully simple and greedy:

> Start with a vocabulary of individual characters. Repeatedly find the **most frequent adjacent pair** of symbols in the corpus, merge it into a single new symbol, and add that merge to your vocabulary. Stop when you reach the desired vocabulary size.

Each merge is a learned rule like "whenever you see `e` followed by `st`, glue them into `est`." After training you have an **ordered list of merge rules**; encoding new text means applying those rules greedily in the same order.

{{fig:bpe-merges}}

### The training algorithm, by hand

Let us work the canonical toy example with a tiny corpus. Each word carries a frequency, and we represent every word as a tuple of characters. (We will add an end-of-word marker shortly; for clarity this first pass omits it.)

| word | frequency | initial symbols |
|------|-----------|-----------------|
| `hug`  | 10 | `h u g` |
| `pug`  | 5  | `p u g` |
| `pun`  | 12 | `p u n` |
| `bun`  | 4  | `b u n` |
| `hugs` | 5  | `h u g s` |

**Iteration 0.** Count every adjacent pair, weighting by word frequency. The pair `(u, g)` appears in `hug` (10), `pug` (5), and `hugs` (5) for a total of **20** — the maximum. Merge it. New symbol `ug` enters the vocabulary; words become `h ug`, `p ug`, `p u n`, `b u n`, `h ug s`.

**Iteration 1.** Recount. `(u, n)` appears in `pun` (12) and `bun` (4) = **16**, the new max. Merge → `un`.

**Iteration 2.** `(h, ug)` appears in `hug` (10) and `hugs` (5) = **15**. Merge → `hug`.

**Iteration 3.** `(p, un)` = 12. Merge → `pun`.

**Iteration 4.** `(p, ug)` = 5. Merge → `pug`.

After five merges the learned, *ordered* merge list is `[(u,g), (u,n), (h,ug), (p,un), (p,ug)]`. To tokenize a new word like `hug`, we apply the rules in order: `h u g` → (apply `u g`) → `h ug` → (apply `h ug`) → `hug`. A single token. To tokenize `mug` (the `m` was never in our corpus, but pretend it was a base character) → `m u g` → `m ug`: two tokens, reusing the learned `ug`. Reuse is the whole point.

!!! example "Worked example: counting the first merge"
    Corpus (with frequencies): `hug`×10, `pug`×5, `pun`×12, `bun`×4, `hugs`×5. The candidate pairs and their frequency-weighted counts before the first merge:

    - `(h,u)`: from `hug`(10) + `hugs`(5) = **15**
    - `(u,g)`: from `hug`(10) + `pug`(5) + `hugs`(5) = **20**  ← winner
    - `(p,u)`: from `pug`(5) + `pun`(12) = **17**
    - `(u,n)`: from `pun`(12) + `bun`(4) = **16**
    - `(b,u)`: from `bun`(4) = **4**
    - `(g,s)`: from `hugs`(5) = **5**

    The greedy rule picks `(u,g)` with count 20. Notice it does *not* pick `(p,u)` at 17 even though `p` is common — BPE optimizes the immediate count, one pair at a time. This greediness is BPE's defining (and sometimes suboptimal) characteristic.

### A complete, runnable BPE trainer

Here is a from-scratch trainer and encoder. It is heavily commented and actually runs. Note the end-of-word marker `</w>`: without it, BPE cannot distinguish a substring that ends a word from the same substring mid-word (e.g. `est` in `newest` vs. inside `estimate`), and word boundaries get blurred.

```python
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. TRAINING
# ---------------------------------------------------------------------------

def get_pair_counts(word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
    """Count every adjacent symbol pair, weighted by the word's frequency.

    word_freqs maps a tuple of symbols (the current segmentation of a word)
    to how often that word occurs in the corpus.
    """
    pairs = Counter()
    for symbols, freq in word_freqs.items():
        for a, b in zip(symbols, symbols[1:]):   # all adjacent (a, b) pairs
            pairs[(a, b)] += freq
    return pairs


def merge_pair(pair: Tuple[str, str],
               word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    """Return a new word_freqs where every occurrence of `pair` is glued."""
    a, b = pair
    merged = a + b                    # the new symbol, e.g. ("u","g") -> "ug"
    new_word_freqs = {}
    for symbols, freq in word_freqs.items():
        out, i = [], 0
        while i < len(symbols):
            # if this position starts the target pair, emit the merged symbol
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_word_freqs[tuple(out)] = freq
    return new_word_freqs


def train_bpe(corpus: List[str], num_merges: int):
    """Learn an ordered list of BPE merges from a list of (whitespace) words.

    Returns:
        merges: ordered list of merged pairs  [(a,b), ...]
        vocab : set of all symbols (base chars + every merged symbol)
    """
    # Count raw word frequencies, then represent each word as char tuple + </w>.
    counts = Counter(corpus)
    word_freqs = {tuple(list(w) + ["</w>"]): f for w, f in counts.items()}

    vocab = set(ch for symbols in word_freqs for ch in symbols)  # base alphabet
    merges: List[Tuple[str, str]] = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break
        # Pick the most frequent pair. Tie-break on the pair itself for
        # *deterministic* output across runs/machines — crucial for reproducibility.
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best] < 2:     # nothing repeats; stop early
            break
        word_freqs = merge_pair(best, word_freqs)
        merges.append(best)
        vocab.add(best[0] + best[1])

    return merges, vocab


# ---------------------------------------------------------------------------
# 2. ENCODING a new word with the learned merges
# ---------------------------------------------------------------------------

def encode_word(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Greedily apply learned merges, in their learned order, to one word."""
    symbols = list(word) + ["</w>"]
    # Map each merge to its rank (priority). Lower rank = learned earlier = applied first.
    rank = {pair: i for i, pair in enumerate(merges)}

    while len(symbols) >= 2:
        # Find the adjacent pair present in `symbols` with the *lowest* rank.
        candidate, best_rank = None, float("inf")
        for a, b in zip(symbols, symbols[1:]):
            r = rank.get((a, b), float("inf"))
            if r < best_rank:
                best_rank, candidate = r, (a, b)
        if candidate is None:         # no learned pair applies; we are done
            break
        # Merge ALL non-overlapping occurrences of the chosen pair.
        a, b = candidate
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(symbols[i]); i += 1
        symbols = out
    return symbols


# ---------------------------------------------------------------------------
# 3. DEMO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    corpus = ("low low low low low lower lower "
              "newest newest newest newest newest newest "
              "widest widest widest").split()
    merges, vocab = train_bpe(corpus, num_merges=10)
    for i, m in enumerate(merges):
        print(f"merge {i}: {m}")
    print("encode('lowest') ->", encode_word("lowest", merges))
    print("encode('newer')  ->", encode_word("newer", merges))
```

Running this prints the learned merges (e.g. `('t','</w>')`, `('s','t</w>')`, `('e','st</w>')`, `('o','w')`, `('l','ow')`, …) and shows `lowest` tokenizing as `['low', 'est</w>']` — note it reuses `low` and `est</w>` learned from `lower` and `newest`/`widest`, even though the exact word `lowest` never appeared in training. That generalization is exactly what we want.

The key subtlety in `encode_word` is the **rank-based priority**: we do not just merge the most frequent pair at encode time (we no longer have corpus counts), we apply the merges in the *same order they were learned*. The earliest-learned merge has the highest priority. A naive implementation that scans left-to-right for the first applicable merge will produce *different, wrong* tokenizations. (Production tokenizers add tricks — a priority queue, or compiling the merges into a finite-state machine — to make this fast, but the semantics are exactly the above.)

!!! warning "Common pitfall: greedy BPE is not optimal"
    BPE's merge selection is locally greedy: it maximizes the count of the *single* most frequent pair at each step, never reconsidering. This can produce segmentations that are not the shortest possible token sequence for a given vocabulary, and it makes BPE sensitive to merge order. It is fast and "good enough," but if you want a *probabilistic* objective that scores whole segmentations, that is exactly what Unigram (below) provides.

!!! note "Aside: training BPE at scale"
    The trainer shown recounts every adjacent pair from scratch on every merge, so with $M$ merges over a corpus segmented into $P$ symbols it is $O(P)$ per merge, $O(P \cdot M)$ overall — fine for a toy corpus, hopelessly slow for 32k merges on gigabytes of text. Real trainers (HuggingFace `tokenizers`, Rust; `tiktoken`'s training path; the CS336 assignment-1 reference) apply three optimizations:

    1. **Unique-word compression.** Collapse the corpus to distinct pre-tokenized chunks with counts (the `Counter` step already does this), so work scales with the number of *distinct* words, not raw token occurrences. English is heavily Zipfian, so this alone is a large constant-factor win.
    2. **Incremental pair counts.** Maintain a `pair -> count` map plus, for each pair, the set of word positions where it occurs. When you merge a pair, only the pairs *adjacent to each merge site* change — the pair to the left and right of the merge are destroyed, and (at most) two new pairs are created — so you update a handful of counts instead of rescanning the whole corpus. Pull the top pair with a max-heap or bucket structure keyed by count, in roughly $O(\log P)$, instead of a full linear `max` scan.
    3. **Parallelize pre-tokenization** across corpus shards (embarrassingly parallel), then merge the per-shard word counters before the sequential merge loop.

    Rough wall-clock: the naive pure-Python trainer above (and educational trainers like `minbpe`) would take many hours to days for 32k merges on ~1 GB of text. The incremental Rust implementation in HuggingFace `tokenizers` trains 32k merges on ~1 GB in well under a minute — roughly ~20 s/GB. Takeaway: the *algorithm* above is exactly right for understanding BPE, but do not run it on real corpora — reach for HF `tokenizers` or `tiktoken` and read their incremental-update loop if you need to train a production tokenizer.

## Byte-Level BPE: Guaranteeing No `<unk>`

Character-level BPE has a quiet problem: what is your base alphabet? If you initialize from the characters seen in training, then a brand-new Unicode character at inference time — an emoji you never saw, a rare CJK glyph, a mathematical symbol — has no base token and becomes `<unk>`. For a frontier model that must ingest *anything*, that is unacceptable.

**Byte-level BPE** (introduced with GPT-2, Radford et al., 2019) solves this elegantly. Instead of starting from *characters*, start from the **256 possible byte values**. Every string, in any language, encodes to UTF-8 bytes, and every byte is one of 256 values — so the base alphabet is *complete and finite*. BPE merges then operate over bytes. There is **provably no OOV**: in the absolute worst case a novel character falls back to its individual UTF-8 bytes, each of which is a guaranteed token.

{{fig:byte-level-no-oov}}

There is one wrinkle. We want to run BPE over a *string-like* representation (the algorithm above manipulates symbols as text), but raw bytes include control characters and whitespace that break tooling. GPT-2's trick is a reversible **bytes-to-unicode** map: assign each of the 256 byte values a distinct, *printable* Unicode character. Printable ASCII bytes map to themselves; the rest (control chars, space, etc.) map to code points starting at U+0100. This is where the famous `Ġ` comes from — byte `0x20` (space) maps to `Ġ` (U+0120), so in GPT-2 output a leading space looks like `Ġthe`.

```python
def bytes_to_unicode():
    """Reversible map from the 256 byte values to printable Unicode chars.

    This is GPT-2's exact scheme. ASCII-printable bytes map to themselves;
    the remaining bytes (control chars, space, DEL, high bytes) get assigned
    unused code points starting at 256, so EVERY byte becomes a printable char
    that BPE can safely treat as an atomic symbol.
    """
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))      # the printable byte values
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:                  # an unprintable byte
            bs.append(b)
            cs.append(256 + n)           # give it a fresh, unused code point
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

b2u = bytes_to_unicode()
assert b2u[32] == "Ġ"      # space -> Ġ  (this is why GPT-2 tokens look like 'Ġthe')
assert b2u[10] == "Ċ"      # newline -> Ċ
assert len(set(b2u.values())) == 256   # all 256 bytes -> 256 distinct printable chars

# To tokenize "héllo": UTF-8-encode -> bytes -> map each byte through b2u ->
# run BPE over that printable string -> emit integer IDs. The 'é' (2 bytes in
# UTF-8) becomes two symbols, which BPE may or may not merge.
text = "héllo"
mapped = "".join(b2u[byte] for byte in text.encode("utf-8"))
print(mapped)   # 'héllo' rendered via the byte map; 'é' -> two glyphs
```

The payoff: a byte-level BPE tokenizer can encode *and decode* literally any byte sequence losslessly, including bytes that are not valid UTF-8 (corrupted data, binary blobs). Decoding maps the printable chars back to bytes, then UTF-8-decodes. This robustness is why GPT-2, GPT-3, GPT-4, and Llama-3 use **byte-level** BPE. Be precise about the lineage, though: Llama-1/2 and early Mistral (v0.x) do *not* use GPT-2-style byte-level BPE — they use **SentencePiece BPE** with byte-fallback, a 32k vocabulary, and a `▁` space marker (a related but distinct scheme, covered later). Llama-3 switched to a tiktoken-style byte-level BPE with a 128k vocabulary. So "byte-level BPE" specifically means the GPT-2/tiktoken family here, not every model in the BPE family.

!!! note "Aside: pre-tokenization with a regex"
    Before BPE runs, GPT-2/GPT-4 first split text with a hand-tuned regular expression (the "pre-tokenizer") that isolates words, leading spaces, numbers, and punctuation into chunks; BPE merges *only within* a chunk, never across. This prevents merges that span a space-plus-word boundary in unhelpful ways. GPT-2's pattern lets a run of digits merge freely, whereas `cl100k_base` adds a `\p{N}{1,3}` alternation that caps every digit chunk at three characters — which is why GPT-4 tokenizes numbers in groups of at most three. The exact patterns are printed and dissected in the next subsection. The regex matters more than people expect — it is part of why GPT-4 tokenizes numbers and code better than GPT-2.


### Putting it together: a complete byte-level BPE tokenizer

The char-level trainer above used an end-of-word marker `</w>` to keep word boundaries from blurring. Byte-level BPE drops `</w>` entirely: word boundaries are instead carried by the leading-space *byte* (`0x20`, which maps to the `Ġ` glyph) that the pre-tokenizer keeps attached to each word, so a merge can never accidentally span two words without a shared `Ġ`. One more thing to fix before writing code: integer IDs. This tokenizer uses the same convention `tiktoken` and `minbpe` use — IDs `0..255` are the 256 raw byte values (token = that byte's `b2u` glyph, id == byte value), then one ID per learned merge in the order it was learned (`256, 257, ...`), then special tokens on top.

**The exact GPT-2 pre-tokenization regex.** This is verified against `tiktoken`'s `gpt2` encoding — it is the literal pattern GPT-2 and GPT-3 use to chop text into chunks *before* BPE ever runs:

```python
import regex as re   # the third-party `regex` package (pip install regex),
                      # NOT stdlib `re` — stdlib `re` has no \p{L} support.

GPT2_SPLIT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
```

One line per alternation, left to right:

- `'s|'t|'re|'ve|'m|'ll|'d` — common English contraction suffixes are pulled out as their own chunks.
- ` ?\p{L}+` — an optional leading space plus a run of letters. This is why `' hello'` is one chunk and ends up as a single `' hello'` token, distinct from bare `'hello'`.
- ` ?\p{N}+` — optional leading space plus a run of digits.
- ` ?[^\s\p{L}\p{N}]+` — optional leading space plus a run of punctuation/symbols.
- `\s+(?!\S)` — trailing whitespace *not* followed by a non-space character, so a space that precedes a word attaches to that word rather than to the previous chunk.
- `\s+` — any remaining whitespace run (e.g. a whitespace-only string, or trailing whitespace at end of text).

`cl100k_base` (GPT-3.5/GPT-4) changes exactly two things in this pattern: it replaces ` ?\p{N}+` with `\p{N}{1,3}` — digits in groups of at most three — and it makes the contraction alternatives case-insensitive. The pre-tokenizer alone, before any BPE merge runs, is what forces GPT-4's 1–3-digit number grouping.

**The tokenizer class.** This reuses `bytes_to_unicode`, `Counter`, `get_pair_counts`, and `merge_pair` defined earlier in this chapter — training is exactly the same greedy loop, just over byte-glyph symbols instead of characters, with no `</w>`:

```python
class ByteLevelBPETokenizer:
    """A complete, from-scratch byte-level BPE tokenizer: train, encode, decode.

    ID convention:
      0 .. 255           -> the 256 raw byte values (id == byte value)
      256 .. 256+M-1     -> the M learned merges, in the order they were learned
      256+M ..           -> special tokens, in the order passed to train()
    """

    def __init__(self):
        self.b2u = bytes_to_unicode()                     # byte value -> printable glyph
        self.u2b = {v: k for k, v in self.b2u.items()}     # glyph -> byte value
        self.pat = GPT2_SPLIT
        self.merges: List[Tuple[str, str]] = []            # learned merges, in order
        self.ranks: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}                    # token string -> id
        self.inv_vocab: Dict[int, str] = {}                # id -> token string
        self.special: Dict[str, int] = {}                  # special text -> id
        self.inv_special: Dict[int, str] = {}               # id -> special text

    def _pretokenize(self, text: str) -> List[Tuple[str, ...]]:
        """Split into pre-tokenizer chunks, each as a tuple of byte-glyphs
        (one glyph per UTF-8 byte of the chunk)."""
        return [
            tuple(self.b2u[b] for b in chunk.encode("utf-8"))
            for chunk in self.pat.findall(text)
        ]

    def train(self, text: str, vocab_size: int, special_tokens: Tuple[str, ...] = ()):
        word_freqs = Counter(self._pretokenize(text))
        num_merges = vocab_size - 256 - len(special_tokens)

        for _ in range(num_merges):
            pair_counts = get_pair_counts(word_freqs)
            if not pair_counts:
                break
            # Same deterministic tie-break as the char-level trainer.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < 2:          # nothing repeats; stop early
                break
            word_freqs = merge_pair(best, word_freqs)
            self.merges.append(best)

        # Assign IDs: bytes first, then merges in learned order, then specials.
        next_id = 0
        for b in range(256):
            self.vocab[self.b2u[b]] = next_id
            next_id += 1
        for a, b in self.merges:
            self.vocab[a + b] = next_id
            next_id += 1
        for tok in special_tokens:
            self.special[tok] = next_id
            next_id += 1

        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.inv_special = {i: t for t, i in self.special.items()}

    def _apply_merges(self, symbols: Tuple[str, ...]) -> List[str]:
        """Rank-priority merge loop, identical to encode_word() above --
        except there is NO end-of-word marker (byte-level BPE has none)."""
        symbols = list(symbols)
        while len(symbols) >= 2:
            candidate, best_rank = None, float("inf")
            for a, b in zip(symbols, symbols[1:]):
                r = self.ranks.get((a, b), float("inf"))
                if r < best_rank:
                    best_rank, candidate = r, (a, b)
            if candidate is None:               # no learned pair applies
                break
            a, b = candidate
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(symbols[i]); i += 1
            symbols = out
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
                ids.append(self.special[seg])
                continue
            for chunk in self.pat.findall(seg):
                mapped = tuple(self.b2u[b] for b in chunk.encode("utf-8"))
                ids.extend(self.vocab[s] for s in self._apply_merges(mapped))
        return ids

    def decode(self, ids: List[int]) -> str:
        out = bytearray()
        for i in ids:
            if i in self.inv_special:
                out += self.inv_special[i].encode("utf-8")
            else:
                out += bytes(self.u2b[ch] for ch in self.inv_vocab[i])
        # errors="replace": invalid byte sequences never crash decode,
        # they render as the U+FFFD replacement character.
        return out.decode("utf-8", errors="replace")
```

**Adversarial round-trip test.** Train on a small mixed corpus, then verify `decode(encode(s)) == s` for strings that stress leading spaces, indentation, tabs/newlines, accented Latin, CJK, emoji, digits, and special-token literals:

```python
tok = ByteLevelBPETokenizer()
corpus_text = """
Tokenization turns text into integers. Byte-level BPE never fails on
unseen input because it falls back to raw bytes.

def encode(text):
    return tokenizer.encode(text)

café résumé naïve, 日本語のテキスト, emoji test 🤖🎉, numbers 1234567890.
""" * 20   # repeat so pairs actually recur enough to merge

tok.train(corpus_text, vocab_size=1000, special_tokens=["<|endoftext|>"])

adversarial = [
    "hello",
    " hello",
    "hello world",
    "  indent",
    "tab\tnl\n",
    "café résumé naïve",
    "日本語のテキスト",
    "emoji 🤖🎉 test",
    "1234567890",
    "<|endoftext|>done",
]
for s in adversarial:
    ids = tok.encode(s, allowed_special={"<|endoftext|>"})
    assert tok.decode(ids) == s, s
print("all round-trips OK")

# Byte-level BPE never crashes, even decoding a lone invalid-UTF-8 byte:
lone = tok.vocab[tok.b2u[0x80]]      # 0x80 is not valid UTF-8 on its own
print(repr(tok.decode([lone])))      # -> the U+FFFD replacement character
```

**Reference cross-check against `tiktoken`.** The same "merge the lowest-rank adjacent pair" logic, fed GPT-2's *real* merge table, reproduces `tiktoken` token-for-token — this is the same algorithm `tiktoken` runs internally, byte-level, merge the lowest-rank pair, so a from-scratch encoder that loads GPT-2's `vocab.json`/`merges.txt` (or, equivalently, `enc._mergeable_ranks`, where the rank *is* the token id) must match exactly:

```python
import tiktoken

def _bpe(piece: bytes, ranks: dict) -> List[int]:
    """Same algorithm as ByteLevelBPETokenizer._apply_merges, operating
    directly on raw bytes against tiktoken's rank table (rank == token id)."""
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        best_i, best_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_i, best_rank = i, r
        if best_i is None:
            break
        parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
    return [ranks[p] for p in parts]   # every single byte is in the table

def encode_ref(text: str, enc) -> List[int]:
    ids: List[int] = []
    for chunk in GPT2_SPLIT.findall(text):
        ids += _bpe(chunk.encode("utf-8"), enc._mergeable_ranks)
    return ids

enc = tiktoken.get_encoding("gpt2")
s = "Tokenization is sneaky, café 🤖 12345."
assert encode_ref(s, enc) == enc.encode(s)
print("matches tiktoken token-for-token:", enc.encode(s))
```

For a sense of scale: train ~50k merges on a few MB of English and you should see roughly **4.0–4.3 bytes per token** (about 1.3 tokens/word) — GPT-2's 50257-entry vocabulary lands right around there. That bytes-per-token number is your compression yardstick when comparing tokenizers or vocabulary sizes.

## WordPiece and Unigram: Two Other Objectives

BPE chooses merges by raw frequency. Two influential alternatives change the *objective function*.

### WordPiece (BERT)

WordPiece, used by BERT (Devlin et al., 2018) and originally from Google's voice-search work (Schuster & Nakajima, 2012), is "BPE with a likelihood-based merge criterion." Instead of merging the most *frequent* pair, it merges the pair that most increases the likelihood of the training corpus under a unigram language model over the current vocabulary. Concretely, for a candidate pair of symbols $(a, b)$ it considers the score

$$
\text{score}(a, b) = \frac{\operatorname{count}(ab)}{\operatorname{count}(a)\,\operatorname{count}(b)}
$$

and merges the pair with the highest score. Compare to BPE, which would just use $\operatorname{count}(ab)$. The denominator means WordPiece *prefers merging pieces that are individually rare but co-occur* — it down-weights pairs where both halves are already very common on their own. A pair like `(t, h)` has huge $\operatorname{count}(th)$ but also huge $\operatorname{count}(t)$ and $\operatorname{count}(h)$, so its WordPiece score is modest; BPE would merge it eagerly.

WordPiece's other visible difference is its **continuation marker**: pieces that do not start a word are prefixed with `##`. So `tokenization` might be `token`, `##ization`, and `playing` becomes `play`, `##ing`. The `##` tells you "glue this to the previous piece with no space." At encode time WordPiece uses **greedy longest-match-first**: from each position, take the longest piece in the vocabulary that matches, then continue. If no piece matches a character, the whole word becomes `[UNK]` (BERT's WordPiece is *not* byte-level, so it can still produce `[UNK]`).

```python
def wordpiece_encode(word, vocab, unk="[UNK]", max_len=100):
    """Greedy longest-match-first WordPiece tokenization of a single word.

    `vocab` is the set of known pieces; continuation pieces are stored with a
    leading '##'. This is the core of BERT's tokenizer.
    """
    if len(word) > max_len:
        return [unk]
    tokens, start = [], 0
    while start < len(word):
        end = len(word)
        cur = None
        # shrink the window from the right until we find a piece in the vocab
        while start < end:
            piece = word[start:end]
            if start > 0:
                piece = "##" + piece     # mark as a continuation piece
            if piece in vocab:
                cur = piece
                break
            end -= 1
        if cur is None:                  # no matching piece at all -> whole word UNK
            return [unk]
        tokens.append(cur)
        start = end
    return tokens

vocab = {"token", "##ization", "##s", "play", "##ing", "[UNK]"}
print(wordpiece_encode("tokenization", vocab))  # ['token', '##ization']
print(wordpiece_encode("playing", vocab))        # ['play', '##ing']
```

### Unigram / SentencePiece (T5, Llama, Gemma)

Unigram (Kudo, *Subword Regularization*, 2018), the default model in the **SentencePiece** library, takes the opposite philosophical approach from BPE. Where BPE *builds up* a vocabulary by merging, Unigram *prunes down*. It starts with a large seed vocabulary (e.g. all frequent substrings) and iteratively removes pieces, keeping the set that best explains the corpus under a probabilistic model.

{{fig:unigram-segmentation-lattice}}

The model is a **unigram language model over subwords**: each subword $x$ has a probability $p(x)$, and a particular segmentation $\mathbf{x} = (x_1, \dots, x_k)$ of a string has probability

$$
P(\mathbf{x}) = \prod_{i=1}^{k} p(x_i), \qquad \sum_{x \in \mathcal{V}} p(x) = 1.
$$

Because a string can be segmented many ways, the probability of the string is the sum (or, for encoding, the max) over segmentations. Unigram is trained with **Expectation–Maximization (EM)**:

1. **E-step:** fix the piece probabilities; for each word compute the expected counts of each piece over all its possible segmentations (efficiently, via the forward–backward / Viterbi dynamic program over the segmentation lattice).
2. **M-step:** re-estimate $p(x)$ from those expected counts.
3. **Prune:** compute, for each piece, the *loss in corpus log-likelihood* if it were removed, and drop the bottom fraction (e.g. 20%) of pieces with the smallest loss. Keep single characters so coverage is preserved.
4. Repeat E/M/prune until the vocabulary reaches the target size.

At encode time, Unigram runs **Viterbi** to find the single most probable segmentation:

$$
\mathbf{x}^* = \operatorname*{arg\,max}_{\mathbf{x} \in \mathcal{S}(\text{string})} \prod_i p(x_i)
= \operatorname*{arg\,max}_{\mathbf{x}} \sum_i \log p(x_i).
$$

```python
import math

def viterbi_segment(text, logp):
    """Most-probable Unigram segmentation via Viterbi over the char positions.

    logp: dict piece -> log probability. We find the segmentation maximizing
    the sum of piece log-probs (== product of probs). best[i] is the best
    log-score for text[:i]; back[i] records the start of the last piece.
    """
    n = len(text)
    NEG = float("-inf")
    best = [NEG] * (n + 1)
    back = [-1] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for j in range(i):                 # try piece text[j:i]
            piece = text[j:i]
            if piece in logp and best[j] > NEG:
                score = best[j] + logp[piece]
                if score > best[i]:
                    best[i] = score
                    back[i] = j
    # reconstruct
    pieces, i = [], n
    while i > 0:
        j = back[i]
        if j < 0:                          # unreachable: fall back to a char
            j = i - 1
        pieces.append(text[j:i])
        i = j
    return pieces[::-1]

# toy probabilities; in practice these come from EM training
logp = {c: math.log(0.01) for c in "helo wrd"}
for w, p in {"hello": 0.2, "he": 0.05, "llo": 0.05, "world": 0.2}.items():
    logp[w] = math.log(p)
print(viterbi_segment("hello", logp))   # -> ['hello'] (single high-prob piece)
```

Two properties make Unigram special. First, because it has an explicit probability model, it supports **subword regularization**: during *training* of the downstream model you can sample *different* segmentations of the same text (not always the Viterbi-best), which acts as data augmentation and improves robustness. BPE has an analogous trick called **BPE-dropout** (randomly skip merges). Second, SentencePiece treats the input as a **raw character stream including spaces**, encoding the space as a visible meta-symbol `▁` (U+2581). This makes tokenization fully reversible and **language-agnostic** — no assumption that words are whitespace-separated, which is essential for Chinese, Japanese, and Thai. Llama, Mistral (early), T5, Gemma, and many multilingual models use SentencePiece (some with Unigram, some with BPE backends).

!!! note "Aside: BPE vs WordPiece vs Unigram at a glance"
    | | BPE | WordPiece | Unigram |
    |---|---|---|---|
    | Direction | bottom-up (merge) | bottom-up (merge) | top-down (prune) |
    | Objective | most frequent pair | max likelihood gain | unigram LM likelihood (EM) |
    | Encode | apply merges in order | greedy longest-match | Viterbi best path |
    | Marker | `Ġ` (byte-level) | `##` continuation | `▁` (space) |
    | Used by | GPT-* (byte-level); Llama-3 (byte-level); Llama-1/2 & Mistral-v0.x (SentencePiece BPE) | BERT, DistilBERT | T5, Gemma, ALBERT, XLNet |
    | Sampling | BPE-dropout | — | subword regularization (native) |

## `tiktoken`, Vocabulary Size & The Cost Math

In practice you rarely train a tokenizer from scratch; you *use* one. For OpenAI models that means **`tiktoken`**, a fast Rust-backed byte-level BPE implementation. The encoding names map to model families: `gpt2`/`r50k_base` (~50k vocab), `cl100k_base` (GPT-3.5/GPT-4, ~100k), and `o200k_base` (GPT-4o, ~200k). Larger encodings pack more text per token.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")     # GPT-3.5 / GPT-4 family
ids = enc.encode("Tokenization is sneaky.")
print(ids)                                      # e.g. [3,2078,2065,374,83760,13]
print([enc.decode([i]) for i in ids])           # per-token text pieces
print(enc.decode(ids))                          # 'Tokenization is sneaky.'

# Why counting tokens matters: this is what you are billed for and what fills
# the context window. A quick budget check before sending a long prompt:
def fits(text, budget_tokens, encoding="cl100k_base"):
    n = len(tiktoken.get_encoding(encoding).encode(text))
    return n, n <= budget_tokens

# Note the leading-space asymmetry, a frequent source of bugs:
print(len(enc.encode("hello")))    # 'hello'
print(len(enc.encode(" hello")))   # ' hello' is a DIFFERENT single token
print(enc.encode("hello") == enc.encode(" hello"))   # False
```

The leading-space behavior bites everyone: in byte-level BPE, `" hello"` (with the space) is typically a *single* token distinct from `"hello"`. Concatenating model outputs naively, or stripping spaces before counting, gives wrong token counts and occasionally wrong continuations.

### The vocabulary-size tradeoff

Vocabulary size $V$ is the single biggest tokenizer knob. Increasing $V$ has competing effects:

**Fewer tokens per document (good).** A bigger vocabulary captures longer frequent substrings, so each document compresses into fewer tokens. Roughly, doubling $V$ from 50k to 100k might cut English token count by ~10–15% (illustrative, not a guarantee). Fewer tokens means cheaper inference and more text per context window — both compound at scale.

**Bigger embedding and output matrices (costly).** The embedding table and the final unembedding/softmax layer are each $V \times d_\text{model}$. The output softmax also costs $O(V)$ per generated token. These grow *linearly* in $V$.

**Rarer tokens are undertrained (subtle).** With a huge vocabulary, the long-tail tokens appear so seldom in pretraining that their embeddings barely move from initialization — this is the mechanism behind "glitch tokens" like `SolidGoldMagikarp`, byte sequences that got their own token from some scraped artifact but were essentially never trained, causing bizarre behavior when invoked.

!!! example "Worked example: embedding table cost vs. sequence savings"
    Take $d_\text{model} = 4096$ (a ~7B-class model). The embedding table has $V \times 4096$ parameters:

    - $V = 32{,}000$ (Llama-2): $32{,}000 \times 4096 \approx 1.31 \times 10^8 = 131$M params.
    - $V = 128{,}000$ (Llama-3): $128{,}000 \times 4096 \approx 5.24 \times 10^8 = 524$M params.

    So going from 32k to 128k adds ~393M embedding parameters (the output projection adds the same again if untied). On a 7B model that is a few percent of total parameters — *not* free, but affordable.

    Now the upside. Llama-3's 128k tokenizer compresses text noticeably better. Suppose a corpus needs 1.30 tokens/word at 32k but 1.15 tokens/word at 128k — about **12% fewer tokens**. For a fixed context window of 8{,}192 tokens, that is ~12% more *content* per request, ~12% lower inference cost per document, and ~12% fewer steps to pretrain over the same text. At the scale of trillions of training tokens and billions of inference calls, a 12% sequence-length reduction dwarfs the cost of 393M extra parameters. This is why the industry trend is **toward larger vocabularies** (32k → 100k → 256k).

A useful way to think about it: tokenization is **lossless compression**, and a good metric is **bytes-per-token** or **tokens-per-word** on your target distribution. Higher bytes-per-token = better compression = cheaper everything, up to the point where rare tokens become undertrained or the matrices dominate memory. The relationship to model loss is real: a tokenizer that compresses better lets a fixed compute budget see more *effective* text, intertwining with [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).

## Special Tokens, Multilingual, Code & Digit Pitfalls

### Special tokens

Beyond ordinary subwords, every tokenizer reserves a handful of **special tokens** with structural meaning. They occupy real vocabulary slots and have embeddings, but they never arise from merging text:

- `<|endoftext|>` / `</s>` — **end of sequence (EOS)**. Marks document boundaries during pretraining and tells the model when to *stop* generating at inference.
- `<bos>` / `<s>` — **beginning of sequence**, prepended so the model has a consistent left context.
- `<pad>` — **padding**, to make a batch's sequences equal length (masked out of the loss).
- `<unk>` — out-of-vocabulary fallback (absent in byte-level tokenizers, which cannot OOV).
- Chat/role markers like `<|im_start|>`, `<|im_end|>`, `<|user|>`, `<|assistant|>` — these structure multi-turn conversations and are central to [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html). Tool-use and reasoning tokens (e.g. for thinking traces) live here too.

!!! warning "Common pitfall: special tokens are an injection surface"
    Special tokens must be added by *trusted* code, never parsed out of user text. If your tokenizer treats the literal string `"<|endoftext|>"` typed by a user as the real EOS token, a user can prematurely end the prompt or impersonate the assistant role — a classic **prompt-injection / boundary-confusion** vector. Robust libraries require you to *explicitly opt in* to special-token parsing (`tiktoken`'s `encode` raises unless you pass `allowed_special`), and chat frameworks encode user content with special tokens *disabled*. See [Security: Prompt Injection, Jailbreaks & Defenses](../12-production-mlops/06-security-prompt-injection.html).

### Multilingual inequity

Because tokenizers are trained on a corpus that is overwhelmingly English (and English-like Latin-script text), they compress English far better than other languages. The same *meaning* costs many more tokens in, say, Hindi, Burmese, or Amharic — often **3–5×** more. This is not a rounding error; it has three concrete consequences:

1. **Cost.** Per-token billing means non-English users literally pay more for the same content.
2. **Effective context.** A document that fits in the window in English overflows it in a high-token-rate language.
3. **Quality.** Longer token sequences and undertrained pieces correlate with weaker performance.

Byte-level fallback guarantees *coverage* (nothing is unrepresentable) but not *efficiency*. The fixes are corpus rebalancing (oversample non-English text when training the tokenizer) and larger, more multilingual vocabularies — visible in the jump from Llama-2's 32k to Llama-3's 128k and Gemma's 256k.

### Code tokenization

Code is brutal on naive tokenizers. Two issues dominate:

- **Whitespace.** Python's significant indentation means long runs of spaces. GPT-2's tokenizer encoded each space (or pair) inefficiently, bloating code. Modern tokenizers (GPT-4's `cl100k_base`, Llama-3) add **dedicated multi-space tokens** (a token for 4 spaces, 8 spaces, etc.) and a tab token, dramatically improving code density and indentation fidelity.
- **Identifiers.** `getUserById` may split as `get`, `User`, `By`, `Id` — fine — but inconsistent splitting of similar identifiers makes it harder for the model to treat them uniformly. Tokenizers tuned on code (with the byte-level regex handling `camelCase` and `snake_case` boundaries) help.

### The digit and arithmetic problem

This is the most consequential and least obvious pitfall. How a tokenizer chunks numbers directly damages arithmetic.

{{fig:digit-place-value-scramble}}

Consider `12345`. Depending on the tokenizer it might be one token, or `123` + `45`, or `1234` + `5`, or `12` + `345` — and crucially, the chunking is **not consistent across numbers of the same length**, because it depends on which digit substrings happened to be frequent in the training corpus. A model trying to add `12345 + 6789` must first parse wildly inconsistent token boundaries; the *positional* meaning of a digit (units, tens, hundreds) is obscured. This is a major reason LLMs historically struggled with multi-digit arithmetic — not because they can't reason, but because the *input representation* scrambles place value.

Fixes that work:

- **Right-to-left digit grouping** so place value aligns to fixed-size chunks.
- **Digit splitting**: force every digit to be its *own* token (`1`,`2`,`3`,`4`,`5`). Llama-1 and Llama-2 (like PaLM) split every number into individual digits for exactly this reason. Llama-3 and GPT-4 take a middle path: their pre-tokenizer regex groups digits 1-3 at a time (the `\p{N}{1,3}` rule), which keeps place value consistent *within* each chunk while spending fewer tokens than full per-digit splitting. It costs more tokens but makes arithmetic *learnable* because place value becomes positional and consistent.
- The infamous **"how many `r`s in strawberry"** failure is the same disease: `strawberry` is ~2–3 tokens, so the model never "sees" individual letters and cannot count them without spelling-level information it was never given cleanly.

```python
# Why digit splitting helps: compare a per-digit scheme to whatever a BPE
# tokenizer does. With per-digit tokens, the model sees place value directly
# and consistently, which makes carrying learnable.
def per_digit(num_str):
    return list(num_str)        # '12345' -> ['1','2','3','4','5'] : 5 stable tokens

# A frequency-trained BPE might instead yield ['123','45'] for one number and
# ['12','3456'] for another of similar magnitude -- inconsistent boundaries that
# destroy the alignment between a digit's position and its value.
print(per_digit("12345"))       # ['1', '2', '3', '4', '5']
print(per_digit("6789"))        # ['6', '7', '8', '9']  -- same scheme every time
```

!!! tip "Practitioner tip: always count tokens with the *model's own* tokenizer"
    Token counts are not portable. The same string is a different number of tokens under `gpt2`, `cl100k_base`, `o200k_base`, Llama-3's tokenizer, and Gemma's. For budgeting context windows, estimating cost, truncating retrieved chunks (see [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)), or packing training sequences, always measure with the *exact* tokenizer the target model uses. A "4000-character" limit is meaningless; a "1000-token" limit measured with the right encoder is not.

!!! interview "Interview Corner"
    **Q:** A teammate reports that your LLM is great at reasoning but consistently fails at adding 6-digit numbers, and also can't reliably count the letters in a word. They suspect the model is "just bad at math." What is the more likely root cause, and how would you diagnose and address it?

    **A:** The likely root cause is **tokenization**, not reasoning. Most BPE tokenizers chunk multi-digit numbers into *inconsistent* subword pieces (e.g. `123|45` for one number, `12|345` for another), so the model cannot reliably recover **place value** — a digit's positional meaning is scrambled before the network ever sees it. The same mechanism explains letter-counting failures: a word like `strawberry` is 2–3 tokens, so individual characters are never exposed as discrete units to count. To diagnose, I'd dump the actual token IDs for a batch of numbers (`tiktoken`/the model's tokenizer) and check whether equal-length numbers tokenize consistently and whether digits are grouped. Fixes: (1) prefer or fine-tune a model whose tokenizer **splits numbers into individual digits** (Llama-style) so place value is positional and consistent; (2) for inference-only situations, format numbers with separators or spaces to nudge cleaner segmentation; (3) for letter-level tasks, prompt the model to spell the word out (one letter per token) first. The key interview signal is recognizing that the *input representation*, fixed before training, is the bottleneck — not the model's reasoning capacity.

## Key Takeaways

!!! key "Key Takeaways"
    - **Models read integers, not text.** The tokenizer maps strings to a fixed vocabulary of subword IDs; that mapping is frozen for the model's life and silently governs cost, context, and capability.
    - **Subwords are the Goldilocks unit.** Words cause OOV and huge vocabularies; characters cause crippling sequence lengths. Subwords keep frequent strings whole, split rare ones into reusable pieces, and (with bytes) never go OOV.
    - **BPE merges the most frequent adjacent pair, greedily and repeatedly**, producing an ordered merge list. Encoding applies those merges *in learned order* (rank-priority), not by left-to-right scanning.
    - **Byte-level BPE** starts from the 256 byte values, so any string in any language is representable with no `<unk>` — the reason GPT-2/3/4 and Llama-3 use it (Llama-1/2 and early Mistral instead use SentencePiece BPE with byte-fallback, a related but distinct scheme). The `Ġ` you see is the byte-to-unicode map for a space.
    - **WordPiece** (BERT) merges by likelihood gain and marks continuations with `##`; **Unigram/SentencePiece** (T5, Llama, Gemma) prunes a large vocabulary via EM, decodes with Viterbi, marks spaces with `▁`, and natively supports subword-regularization sampling.
    - **Vocabulary size is a tradeoff:** larger $V$ compresses text (fewer tokens → cheaper inference, more context, more effective pretraining data) but grows the embedding/softmax matrices linearly and risks undertrained "glitch" tokens. The industry trend is toward larger vocabularies (32k → 256k).
    - **Special tokens** (EOS, BOS, pad, chat/role markers) are a trusted-code-only injection surface — never parse them from user input.
    - **Digit and multilingual pitfalls are real:** inconsistent number chunking breaks arithmetic and letter-counting (fix with per-digit splitting), and English-centric tokenizers make other languages 3–5× more expensive. Always count tokens with the model's own tokenizer.

!!! sota "State of the Art & Resources (2026)"
    Subword tokenization (BPE, WordPiece, Unigram) remains the dominant paradigm for production LLMs in 2026, with vocabulary sizes scaling from 32k toward 256k. The frontier is moving toward tokenizer-free byte-patch architectures (e.g., Meta's BLT), but all major deployed models still use one of the three algorithms covered in this chapter.

    **Foundational work**

    - [Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016)](https://arxiv.org/abs/1508.07909) — the paper that adapted BPE for NLP and launched the subword era.
    - [Kudo, *Subword Regularization* (2018)](https://arxiv.org/abs/1804.10959) — introduces the Unigram language model for tokenization and stochastic segmentation sampling.
    - [Kudo & Richardson, *SentencePiece* (2018)](https://arxiv.org/abs/1808.06226) — the language-agnostic library (with `▁` space encoding) used by T5, Llama, Gemma, and others.

    **Recent advances (2023–2026)**

    - [Provilkov et al., *BPE-Dropout* (2020)](https://arxiv.org/abs/1910.13267) — randomly drops BPE merges during training for subword regularization; up to 2.3 BLEU gain over standard BPE.
    - [Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens* (2024)](https://arxiv.org/abs/2412.09871) — Meta's tokenizer-free architecture that groups bytes into entropy-based patches, matching Llama 3 quality with up to 50% fewer inference FLOPs.

    **Open-source & tools**

    - [openai/tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast Rust-backed BPE tokenizer; the canonical implementation for GPT-3.5/4/4o.
    - [google/sentencepiece](https://github.com/google/sentencepiece) — the reference C++/Python library for both BPE and Unigram tokenization used across most non-OpenAI models.
    - [karpathy/minbpe](https://github.com/karpathy/minbpe) — minimal, heavily commented Python BPE implementation built live on YouTube; ideal for understanding every line of the algorithm.
    - [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — Rust-powered tokenizers library supporting BPE, WordPiece, and Unigram; processes a GB of text in under 20 seconds.

    **Go deeper**

    - [Tiktokenizer](https://tiktokenizer.vercel.app/) — interactive browser tool for visualizing token boundaries across GPT-4o, Llama, Gemma, and other tokenizers side-by-side.

## Further reading

- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units* (2016) — the paper that brought BPE to NLP.
- Gage, *A New Algorithm for Data Compression* (1994) — the original byte-pair-encoding compression algorithm.
- Schuster & Nakajima, *Japanese and Korean Voice Search* (2012) and Devlin et al., *BERT* (2018) — the origin and use of WordPiece.
- Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates* (2018) — the Unigram language model.
- Kudo & Richardson, *SentencePiece: A Simple and Language Independent Subword Tokenizer* (2018) — the library and the `▁` space convention.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — byte-level BPE and the bytes-to-unicode trick.
- OpenAI's **`tiktoken`** repository and Andrej Karpathy's **`minbpe`** repository — clear, fast reference implementations to read and run.

## Exercises

**1.** Byte-level BPE is advertised as having *provably no* `<unk>`. Explain precisely *why* this holds, and give the worst-case number of tokens a single, never-before-seen character can cost. Contrast this with a character-level BPE tokenizer whose base alphabet was fixed from its training corpus.

??? note "Solution"
    Byte-level BPE initializes its base vocabulary from the **256 possible byte values**, not from characters seen in training. Every string in any language encodes to UTF-8, and every UTF-8 byte is by definition one of those 256 values — so each of the 256 base tokens is guaranteed to exist, and any byte sequence is representable. There is no way to encounter an "unseen" atomic symbol, because the atomic symbols are all 256 bytes and they are all in the vocabulary from the start.

    Worst case for a single novel character: it simply falls back to its individual UTF-8 bytes, each of which is a guaranteed single-byte token. UTF-8 encodes a code point in **1 to 4 bytes**, so a never-before-seen character (e.g. a new emoji, which is often 4 bytes) costs **at most 4 tokens** — never an `<unk>`, and never lost information.

    A character-level BPE tokenizer whose base alphabet was frozen from its training corpus has no such guarantee: a Unicode character it never saw (a rare CJK glyph, a new emoji, a math symbol) has *no* base token at all, so it collapses to `<unk>` and the information is destroyed. That is exactly the failure byte-level BPE was designed to remove.

**2.** Train BPE *by hand* (character level, with the `</w>` end-of-word marker) on this corpus:

    cat x5,  car x3,  bat x2

Find the first **two** merges (with their frequency-weighted counts), then use the resulting ordered merge list to tokenize the word `cart`. Apply merges using the chapter's rank-priority rule.

??? note "Solution"
    Represent each word as characters plus `</w>`, weighted by frequency:

    - `cat` x5 -> `c a t </w>`
    - `car` x3 -> `c a r </w>`
    - `bat` x2 -> `b a t </w>`

    **First merge.** Count adjacent pairs:

    - `(c,a)`: cat(5) + car(3) = **8**  <- winner
    - `(a,t)`: cat(5) + bat(2) = 7
    - `(t,</w>)`: cat(5) + bat(2) = 7
    - `(a,r)`: car(3) = 3
    - `(r,</w>)`: car(3) = 3
    - `(b,a)`: bat(2) = 2

    Merge `(c,a)` -> `ca`. Words become `ca t </w>`(5), `ca r </w>`(3), `b a t </w>`(2).

    **Second merge.** Recount:

    - `(t,</w>)`: cat(5) + bat(2) = **7**  <- winner
    - `(ca,t)`: 5
    - `(ca,r)`: 3
    - `(r,</w>)`: 3
    - `(b,a)`: 2
    - `(a,t)`: 2

    Merge `(t,</w>)` -> `t</w>`. Ordered merge list so far: `[(c,a), (t,</w>)]`, so rank 0 = `(c,a)`, rank 1 = `(t,</w>)`.

    **Tokenize `cart`.** Start: `c a r t </w>`.

    - Lowest-rank applicable pair present is `(c,a)` (rank 0) -> `ca r t </w>`.
    - Next, `(t,</w>)` (rank 1) is present -> `ca r t</w>`.
    - No learned pair remains. Result: `['ca', 'r', 't</w>']` — three tokens, reusing `ca` and `t</w>` even though `cart` never appeared in training.

**3.** Take $d_\text{model} = 2048$. Compare vocabulary sizes $V_1 = 50{,}000$ and $V_2 = 100{,}000$.

    (a) How many parameters does each embedding table hold, and how many are *added* by going from $V_1$ to $V_2$ (embedding table only)?
    (b) A 1{,}000{,}000-byte English document compresses at 4.0 bytes/token under the small tokenizer and 4.5 bytes/token under the large one. How many tokens each, and what is the percentage reduction? Relate the result to the vocabulary-size tradeoff.

??? note "Solution"
    **(a)** An embedding table is $V \times d_\text{model}$ parameters.

    - $V_1$: $50{,}000 \times 2048 = 1.024 \times 10^{8} = 102.4$M params.
    - $V_2$: $100{,}000 \times 2048 = 2.048 \times 10^{8} = 204.8$M params.
    - Added: $204.8\text{M} - 102.4\text{M} = 102.4$M params. (If the output/unembedding projection is *untied*, it is a second $V \times d_\text{model}$ matrix, so the true added cost is another 102.4M on top, ~204.8M total.)

    **(b)** Tokens = bytes / (bytes per token):

    - Small: $1{,}000{,}000 / 4.0 = 250{,}000$ tokens.
    - Large: $1{,}000{,}000 / 4.5 \approx 222{,}222$ tokens.
    - Reduction: $(250{,}000 - 222{,}222)/250{,}000 = 27{,}778/250{,}000 \approx 11.1\%$.

    So the larger vocabulary costs ~102M extra embedding parameters (a few percent of a 7B-class model) but buys ~11% fewer tokens per document — which compounds into ~11% cheaper inference, ~11% more content per context window, and ~11% fewer steps to pretrain over the same text. Because those savings multiply across trillions of training tokens and billions of inference calls, the sequence-length win typically dwarfs the fixed parameter cost — the exact reasoning behind the industry trend toward larger vocabularies.

**4.** The chapter's `GPT2_SPLIT` regex lets a run of digits merge freely (` ?\p{N}+`), whereas `cl100k_base` caps every digit chunk at three (`\p{N}{1,3}`). Modify the pre-tokenizer to the `cl100k`-style digit rule and write a function `digit_chunks(text)` that returns the pre-tokenizer chunks. Verify that `12345` splits into `['123', '45']` and `1234567` into `['123', '456', '7']`.

??? note "Solution"
    Only the digit alternation changes; everything else is identical to the chapter's pattern. Note that `\p{N}{1,3}` has *no* optional leading space, matching `cl100k_base`.

    ```python
    import regex as re

    CL100K_DIGIT_SPLIT = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    def digit_chunks(text):
        return CL100K_DIGIT_SPLIT.findall(text)

    assert digit_chunks("12345") == ["123", "45"]
    assert digit_chunks("1234567") == ["123", "456", "7"]
    print(digit_chunks("12345"))     # ['123', '45']
    print(digit_chunks("1234567"))   # ['123', '456', '7']
    ```

    Because `\p{N}{1,3}` is greedy and matches at most three digits at a time, a digit run is chopped left-to-right into groups of three with the remainder trailing: `12345` -> `123`,`45`; `1234567` -> `123`,`456`,`7`. This split happens in the pre-tokenizer, *before* any BPE merge runs, which is why GPT-4 groups numbers in chunks of at most three and thereby keeps place value consistent within each chunk.

**5.** BPE merges the pair with the largest raw count $\operatorname{count}(ab)$. WordPiece instead merges the pair maximizing $\operatorname{score}(a,b) = \dfrac{\operatorname{count}(ab)}{\operatorname{count}(a)\,\operatorname{count}(b)}$. Implement `best_wordpiece_pair(word_freqs)` (reusing the chapter's `word_freqs` representation and `get_pair_counts`) that returns the highest-scoring adjacent pair, and demonstrate on a corpus where it picks a *different* pair than BPE would.

??? note "Solution"
    We need frequency-weighted symbol counts (the denominator) as well as the pair counts (the numerator). A symbol's total count is its number of occurrences across all words, weighted by word frequency.

    ```python
    from collections import Counter

    def get_symbol_counts(word_freqs):
        """Frequency-weighted count of each individual symbol."""
        counts = Counter()
        for symbols, freq in word_freqs.items():
            for s in symbols:
                counts[s] += freq
        return counts

    def best_wordpiece_pair(word_freqs):
        """Return the adjacent pair maximizing count(ab)/(count(a)*count(b))."""
        pair_counts = get_pair_counts(word_freqs)      # from the chapter
        sym = get_symbol_counts(word_freqs)
        best, best_score = None, float("-inf")
        for (a, b), c_ab in pair_counts.items():
            score = c_ab / (sym[a] * sym[b])
            # deterministic tie-break on the pair itself, matching the chapter
            if (score, (a, b)) > (best_score, best if best else ("", "")):
                best, best_score = (a, b), score
        return best, best_score
    ```

    Demonstration. Take a corpus where one pair is very frequent but built from two individually ubiquitous symbols, while a rarer pair is built from symbols that occur *only* together:

    ```python
    # 'th' is frequent but t and h are everywhere; 'qz' is rarer but q,z occur
    # only as the pair qz.
    word_freqs = {
        ("t", "h", "e"): 10,   # t,h,e all common
        ("t", "h", "a", "t"): 8,
        ("h", "i", "t"): 5,
        ("q", "z"): 3,         # q and z appear ONLY here
    }

    # BPE would pick the raw-count winner:
    bpe_pick = max(get_pair_counts(word_freqs),
                   key=lambda p: get_pair_counts(word_freqs)[p])
    print("BPE picks:", bpe_pick)                 # ('t', 'h')  count 18
    print("WordPiece picks:", best_wordpiece_pair(word_freqs))
    ```

    Working the numbers: `(t,h)` has $\operatorname{count}=18$, but $\operatorname{count}(t)=10+8+8+5=31$ and $\operatorname{count}(h)=10+8+5=23$, giving score $18/(31\cdot23)\approx 0.0253$. The pair `(q,z)` has $\operatorname{count}=3$ with $\operatorname{count}(q)=3$, $\operatorname{count}(z)=3$, giving score $3/(3\cdot3)\approx 0.333$. So **BPE merges `(t,h)`** (highest raw count) while **WordPiece merges `(q,z)`** (highest likelihood-gain score). This is exactly the chapter's point: the denominator makes WordPiece *down-weight pairs whose halves are already common on their own* and prefer pieces that are individually rare but co-occur.
