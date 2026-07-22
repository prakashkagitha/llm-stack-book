"""
Runs the CPU-runnable Python blocks from content/02-transformer/01-tokenization.md,
concatenated in order so that later blocks can rely on names defined by earlier
ones (as they do in the chapter itself). Each block is copied verbatim from the
chapter; only the minimal glue needed to make blocks that only *define*
something actually execute has been added, and is clearly marked "GLUE". The
book's own `if __name__ == "__main__":` demo guards are unwrapped and executed
directly here, since this file itself plays the role of "__main__".

Blocks covered (the 7 heuristically CPU-runnable blocks the task named, plus
the three "fragment" blocks #1/#3/#6, which turn out to be trivially CPU-safe
and/or are a hard dependency of a tested block, so they are included too):

  #0 (line ~85)  - get_pair_counts / merge_pair / train_bpe / encode_word + demo
  #1 (line ~227) - bytes_to_unicode() [fragment per task; trivially CPU-safe,
                    included and exercised with the book's own asserts]
  #2 (line ~273) - GPT2_SPLIT regex (needs third-party `regex`, guarded)
  #3 (line ~295) - ByteLevelBPETokenizer class [fragment per task; it is a
                    hard dependency of tested block #4, so it must be defined
                    for #4 to run at all -- included]
  #4 (line ~408) - adversarial round-trip test of ByteLevelBPETokenizer
  #5 (line ~446) - tiktoken cross-check -- SKIP(network): see below
  #6 (line ~494) - wordpiece_encode() [fragment per task; trivially CPU-safe,
                    self-contained, no external deps -- included and exercised]
  #7 (line ~553) - viterbi_segment() Unigram decoding
  #8 (line ~609) - tiktoken cl100k_base demo -- SKIP(network): see below
  #9 (line ~700) - per_digit() digit-splitting demo

SKIP(network): blocks #5 and #8 both call `tiktoken.get_encoding(...)`. Under
the hood tiktoken does NOT ship the encoder tables in the pip package -- it
downloads `vocab.bpe`/`encoder.json` (or the cl100k_base rank file) from
`https://openaipublic.blob.core.windows.net/...` on first use and caches them
locally (see `tiktoken.load.read_file_cached` / `tiktoken_ext.openai_public`).
That is a network call, forbidden in this harness (CI has no network). Rather
than fabricate a fake rank table to "mock" the exact reference-matching
behaviour these blocks exist to demonstrate, they are left defined-not-called
where they contain no network-independent logic, and skipped outright where
they are nothing but a `get_encoding` call.

The `regex` package (used by GPT2_SPLIT / ByteLevelBPETokenizer) is a
third-party dependency not on the guaranteed-available list, so its import is
guarded; blocks #2/#3/#4 only run if it is importable.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

try:
    import regex as re  # the third-party `regex` package (pip install regex),
    _HAS_REGEX = True    # NOT stdlib `re` -- stdlib `re` has no \p{L} support.
except Exception:
    re = None
    _HAS_REGEX = False

try:
    import tiktoken
except Exception:
    tiktoken = None


print("=" * 70)
print("Block #0 (line ~85): BPE trainer + encoder from scratch")
print("=" * 70)

# --- verbatim from the chapter ---
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
        # *deterministic* output across runs/machines -- crucial for reproducibility.
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
# 3. DEMO (book's own __main__ block, executed directly here)
# ---------------------------------------------------------------------------
corpus = ("low low low low low lower lower "
          "newest newest newest newest newest newest "
          "widest widest widest").split()
merges, vocab = train_bpe(corpus, num_merges=10)
for i, m in enumerate(merges):
    print(f"merge {i}: {m}")
print("encode('lowest') ->", encode_word("lowest", merges))
print("encode('newer')  ->", encode_word("newer", merges))
# --- end verbatim ---

# matches the chapter's stated output exactly: ('t','</w>'), ('s','t</w>'),
# ('e','st</w>'), ('o','w'), ('l','ow'), ...
assert merges[:5] == [("t", "</w>"), ("s", "t</w>"), ("e", "st</w>"), ("o", "w"), ("l", "ow")]
assert encode_word("lowest", merges) == ["low", "est</w>"]
assert "low" in vocab and "est</w>" in vocab


print()
print("=" * 70)
print("Block #1 (line ~227): bytes_to_unicode() [fragment, trivially CPU-safe]")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---


if _HAS_REGEX:
    print()
    print("=" * 70)
    print("Block #2 (line ~273): the exact GPT-2 pre-tokenization regex")
    print("=" * 70)

    # --- verbatim from the chapter ---
    GPT2_SPLIT = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    # --- end verbatim ---

    assert GPT2_SPLIT.findall("hello world") == ["hello", " world"]
    assert GPT2_SPLIT.findall("I'll go") == ["I", "'ll", " go"]

    print()
    print("=" * 70)
    print("Block #3 (line ~295): ByteLevelBPETokenizer class")
    print("=" * 70)
    print("[fragment per task classification: it is a hard dependency of")
    print(" tested block #4 (tok = ByteLevelBPETokenizer()), so it is defined")
    print(" here rather than skipped -- otherwise block #4 could not run.]")

    # --- verbatim from the chapter ---
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
    # --- end verbatim ---

    print()
    print("=" * 70)
    print("Block #4 (line ~408): adversarial round-trip test")
    print("=" * 70)

    # --- verbatim from the chapter ---
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
    # --- end verbatim ---

    assert tok.decode([lone]) == "�"

    print()
    print("=" * 70)
    print("Block #5 (line ~446): tiktoken cross-check")
    print("=" * 70)
    print("SKIP(network): tiktoken.get_encoding('gpt2') downloads "
          "vocab.bpe/encoder.json from openaipublic.blob.core.windows.net on "
          "first use (see tiktoken.load.read_file_cached); CI has no network. "
          "The block's own point is comparing against the *real* tiktoken "
          "reference table, so faking that table would not honestly test "
          "anything -- skipped outright rather than mocked.")
else:
    print()
    print("SKIP(dependency): `regex` package not importable; skipping blocks "
          "#2 (GPT2_SPLIT), #3 (ByteLevelBPETokenizer), #4 (round-trip test), "
          "and #5 (tiktoken cross-check, itself SKIP(network) regardless).")


print()
print("=" * 70)
print("Block #6 (line ~494): wordpiece_encode() [fragment, trivially CPU-safe]")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

assert wordpiece_encode("tokenization", vocab) == ["token", "##ization"]
assert wordpiece_encode("playing", vocab) == ["play", "##ing"]
assert wordpiece_encode("xyz", vocab) == ["[UNK]"]


print()
print("=" * 70)
print("Block #7 (line ~553): viterbi_segment() Unigram decoding")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

assert viterbi_segment("hello", logp) == ["hello"]
assert viterbi_segment("world", logp) == ["world"]


print()
print("=" * 70)
print("Block #8 (line ~609): tiktoken cl100k_base demo")
print("=" * 70)
print("SKIP(network): tiktoken.get_encoding('cl100k_base') downloads its "
      "rank file from openaipublic.blob.core.windows.net on first use; CI "
      "has no network. This block is nothing but a get_encoding() call plus "
      "encode/decode/print -- there is no network-independent logic to "
      "extract, so it is skipped outright.")


print()
print("=" * 70)
print("Block #9 (line ~700): per_digit() digit-splitting demo")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

assert per_digit("12345") == ["1", "2", "3", "4", "5"]
assert per_digit("6789") == ["6", "7", "8", "9"]


print()
print("=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)
