"""
Trains Stack-100M's tokenizer (Ch. 14.3) on a SAMPLE of the pretraining mix
(Ch. 14.2) and exports it into every format the rest of Part XIV needs.

Memory note: peak RSS is O(distinct pre-token chunks), not O(corpus bytes). On
this book's 8.34 MB manuscript the Counter holds ~51k distinct chunks and the
whole process stays near 300 MB RSS; distinct-chunk count grows far slower than
corpus size for natural text. A few hundred MB of sample is comfortable on a
16 GB laptop; past a few GB, switch to the HF `tokenizers` trainer.

Usage:
    python -m scripts.train_tokenizer "data/mix_sample/*.txt"
"""
from __future__ import annotations

import glob
import sys
from typing import Iterator

from stacklm.tokenizer.bpe import StackTokenizer, VOCAB_SIZE, SPECIAL_TOKENS

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
                # errors="ignore" so a split UTF-8 sequence is dropped rather
                # than turned into U+FFFD (which would pollute the merge table).
                keep = raw[: max_bytes - total]
                cut = keep.rfind(b" ")
                if cut > 0:
                    keep = keep[:cut]
                total = max_bytes
                yield keep.decode("utf-8", errors="ignore")
                break
        if total >= max_bytes:
            break


def main(paths_glob: str = "data/mix_sample/*.txt",
         max_bytes: int = 500_000_000) -> StackTokenizer:
    tok = StackTokenizer()
    shortfall = tok.train_from_iterable(
        stream_sample(paths_glob, max_bytes=max_bytes),
        vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    assert shortfall == 0, "sample too small to fill 32,768 entries -- enlarge it"
    assert tok.vocab_size == VOCAB_SIZE      # 14.4 hardcodes nn.Embedding(32768, 512)
    assert tok.special_to_id["<|tool_result|>"] == VOCAB_SIZE - 1

    tok.save("tokenizer/stack100m-32768.json")             # from-scratch artifact
    try:
        from stacklm.tokenizer.export import save_pretrained
        save_pretrained(tok, "tokenizer/stack100m-32768-hf")   # ecosystem artifact
    except ImportError as e:                 # tokenizers/transformers not installed
        print(f"[warn] skipped HF export: {e}")
    print(f"trained {len(tok.merges)} merges, vocab_size={tok.vocab_size}")
    return tok


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
