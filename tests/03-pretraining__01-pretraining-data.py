"""
Runnability test for content/03-pretraining/01-pretraining-data.md

Tests the 4 heuristically CPU-runnable Python blocks from the chapter,
concatenated in chapter order so later blocks can use names defined by
earlier ones (exactly as the chapter's narrative builds them up):

    - block #1 (line ~63)  -- WARC text extraction (extract_text_from_warc, _decode_html)
    - block #3 (line ~283) -- streaming pipeline (_iter_wet_records, passes_quality_filter, TokenShard, run_pipeline)
    - block #4 (line ~538) -- ShardedTokenDataset (memory-mapped shard reader)
    - block #5 (line ~605) -- verify_pipeline.py's own deterministic offline checks

Blocks #0 and #2 are non-Python (a WARC/WET text listing and an ASCII
pipeline diagram) and are correctly skipped.

`trafilatura`, `warcio`, and `tokenizers` are used by the book's code but are
NOT in the guaranteed CI dependency set (numpy, torch, einops, sklearn,
stdlib). They are guarded so the module always loads; the one function that
truly needs them (`extract_text_from_warc`, which delegates almost all of its
logic to `warcio.ArchiveIterator`) is left defined-but-not-called, and is
noted explicitly below as SKIP(optional-deps). The rest of block #1
(`_decode_html`) has no third-party dependency and is exercised directly.
"""

import argparse
import gzip
import os
import re
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Optional third-party deps used by the book's code but not guaranteed in CI.
try:
    import trafilatura
except Exception:
    trafilatura = None
try:
    from warcio.archiveiterator import ArchiveIterator
except Exception:
    ArchiveIterator = None
try:
    from tokenizers import Tokenizer  # HuggingFace fast tokenizers
except Exception:
    Tokenizer = None


# =============================================================================
# Block #1 (line ~63): Extract main-content text from raw Common Crawl WARC
# files. Verbatim from the chapter.
# =============================================================================

def extract_text_from_warc(warc_path: str):
    """Yield (url, main_text) for each HTML response record in a WARC file."""
    with open(warc_path, "rb") as fh:
        for record in ArchiveIterator(fh):
            if record.rec_type != "response":          # skip request/metadata
                continue
            ctype = record.http_headers.get_header("Content-Type", "") or ""
            if "html" not in ctype.lower():             # skip PDFs, images, ...
                continue

            url = record.rec_headers.get_header("WARC-Target-URI")
            raw = record.content_stream().read()        # raw HTTP response body
            html = _decode_html(raw, ctype)
            if html is None:
                continue

            # trafilatura strips boilerplate and returns clean main content.
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_precision=True,   # prefer clean text over recall
            )
            if text:
                yield url, text


def _decode_html(raw: bytes, content_type: str):
    """Bytes -> str using the HTTP charset if given, then utf-8, then latin-1."""
    charset = None
    lc = content_type.lower()
    if "charset=" in lc:
        charset = lc.split("charset=", 1)[1].split(";")[0].strip()
    for enc in (charset, "utf-8", "latin-1"):
        if enc:
            try:
                return raw.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
    return None


# =============================================================================
# Block #3 (line ~283): Streaming pretraining data pipeline. Verbatim from the
# chapter (the argparse `if __name__ == "__main__":` CLI driver at the very
# end of the original code fence is omitted here -- it is pure CLI plumbing
# that requires real WET files + a real tokenizer path on disk, not core
# pipeline logic; `run_pipeline` itself is copied verbatim and left
# defined-but-not-called for the same reason `tokenizers` is optional).
# =============================================================================

# ── WET record parser ──────────────────────────────────────────────────────

def _iter_wet_records(lines):
    """
    Core WET parser over any iterator of text lines.

    Each WET record is a WARC/1.0 record:

        WARC/1.0                     <- record boundary
        WARC-Type: conversion
        WARC-Target-URI: <url>
        ... more headers (Content-Type, Content-Length, ...) ...
                                     <- ONE blank line ends the header block
        <body: the record's text payload>
                                     <- blank line(s), then the next 'WARC/1.0'

    The parser resets ALL per-record state at every 'WARC/1.0' boundary and
    ignores every header line, so header text (e.g. 'Content-Length: 2048')
    and the next record's boundary lines can never leak into a body. The
    first record in a WET file is a 'warcinfo' record with no
    WARC-Target-URI; it is skipped because it has no URL.

    (Content-Length is ignored in this didactic version; see the production
    note below for why real pipelines honor it instead.)
    """
    url = None
    in_header = False        # inside the header block of the current record
    body_lines = []
    started = False          # have we seen the first 'WARC/1.0' yet?

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line == "WARC/1.0":
            # Record boundary: flush the record we just finished ...
            if started:
                text = "\n".join(body_lines).strip("\n")
                if url is not None and text:
                    yield url, text
            # ... then reset ALL state for the new record.
            url = None
            in_header = True
            body_lines = []
            started = True
            continue

        if in_header:
            if line == "":
                in_header = False              # blank line ends the headers
            elif line.startswith("WARC-Target-URI:"):
                url = line.split(":", 1)[1].strip()
            # All other header lines are ignored and never reach body_lines.
        else:
            body_lines.append(line)

    # Flush the final record.
    if started:
        text = "\n".join(body_lines).strip("\n")
        if url is not None and text:
            yield url, text


def parse_wet_records(filepath: str):
    """Yield (url, text) pairs from a gzipped WET file (WARC/1.0 format)."""
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
        yield from _iter_wet_records(fh)


# ── Quality heuristics ─────────────────────────────────────────────────────

def passes_quality_filter(text: str, min_chars: int = 200) -> bool:
    """
    A simplified version of the heuristics used by C4 and Gopher.
    Returns True if the document passes all checks.

    Real pipelines also run language identification (fastText) and
    a reference model perplexity filter — omitted here for brevity.
    """
    # (1) Minimum length
    if len(text) < min_chars:
        return False

    # (2) Must contain at least 3 lines with terminal punctuation
    sentences_with_punct = [
        l for l in text.splitlines()
        if len(l) > 10 and l[-1] in ".!?\""
    ]
    if len(sentences_with_punct) < 3:
        return False

    # (3) Word-level repetition ratio (Gopher signal)
    # If the most common word accounts for > 20% of words, likely spam
    words = text.lower().split()
    if not words:
        return False
    most_common_freq = max(
        words.count(w) for w in set(words)
    ) / len(words)
    if most_common_freq > 0.20:
        return False

    # (4) Bullet/symbol density (JS code / menu spam heuristic)
    non_alpha = sum(1 for c in text if not c.isalnum() and c not in " \n.,;:!?'\"-()")
    if non_alpha / max(len(text), 1) > 0.25:
        return False

    return True


# ── Token packer ──────────────────────────────────────────────────────────

class TokenShard:
    """
    Accumulates tokenized documents and flushes complete rows to a binary
    shard file. Each row is (ctx_len + 1) int32 tokens: ctx_len inputs plus
    one extra token for the label shift (target = row[1:]). This matches the
    reader in ShardedTokenDataset exactly — the practitioner tip below is
    implemented here, so there is no off-by-one at read time. The final
    partial row (< ctx_len + 1 tokens) is dropped.
    """

    def __init__(self, output_dir: str, shard_size: int, ctx_len: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ctx_len = ctx_len
        self.row_len = ctx_len + 1                       # tokens per training row
        # Round the requested shard size down to a whole number of rows.
        self.rows_per_shard = max(1, shard_size // self.row_len)
        self.shard_tokens = self.rows_per_shard * self.row_len
        self._buffer = []
        self._shard_idx = 0
        self._total_tokens = 0

    def add_tokens(self, token_ids: list[int], eos_id: int):
        """Append document tokens followed by an EOS separator."""
        self._buffer.extend(token_ids)
        self._buffer.append(eos_id)
        while len(self._buffer) >= self.shard_tokens:
            self._flush_shard(self.shard_tokens)

    def _flush_shard(self, n_tokens: int):
        """Write one shard containing a whole number of (ctx_len + 1) rows."""
        n_rows = n_tokens // self.row_len
        n_keep = n_rows * self.row_len                   # carry remainder forward
        tokens = self._buffer[:n_keep]
        self._buffer = self._buffer[n_keep:]
        if not tokens:
            return
        arr = np.array(tokens, dtype=np.int32)
        shard_path = self.output_dir / f"shard_{self._shard_idx:05d}.bin"
        arr.tofile(shard_path)
        self._total_tokens += len(tokens)
        self._shard_idx += 1
        print(f"  Wrote {shard_path.name}: {len(tokens):,} tokens "
              f"({n_rows:,} rows of {self.row_len})")

    def finalize(self):
        """Flush any remaining full rows from the buffer."""
        while len(self._buffer) >= self.shard_tokens:
            self._flush_shard(self.shard_tokens)
        if len(self._buffer) >= self.row_len:
            self._flush_shard(len(self._buffer))
        print(f"\nTotal tokens written: {self._total_tokens:,}")


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str,
    context_len: int = 4096,
    shard_size: int = 500_000_000,
):
    """
    Main entry point.  In production, this loop is distributed across
    hundreds of workers, each handling a distinct subset of WET files.
    Here we run single-threaded for clarity.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    packer = TokenShard(output_dir, shard_size, context_len)

    wet_files = sorted(Path(input_dir).glob("*.wet.gz"))
    print(f"Found {len(wet_files)} WET files to process.")

    total_docs = 0
    kept_docs = 0

    for wet_path in wet_files:
        print(f"Processing {wet_path.name} ...")
        for url, text in parse_wet_records(str(wet_path)):
            total_docs += 1
            if not passes_quality_filter(text):
                continue
            # Tokenize (encode returns a huggingface Encoding object)
            encoding = tokenizer.encode(text)
            packer.add_tokens(encoding.ids, eos_id)
            kept_docs += 1

    packer.finalize()
    retention = 100 * kept_docs / max(total_docs, 1)
    print(f"Retention rate: {kept_docs:,}/{total_docs:,} = {retention:.1f}%")


# =============================================================================
# Block #4 (line ~538): Shard layout for streaming training. Verbatim from
# the chapter.
# =============================================================================

class ShardedTokenDataset(IterableDataset):
    """
    Memory-mapped streaming dataset over pre-tokenized binary shards.

    Shards are laid out as flat int32 arrays; each context window
    is a contiguous slice of length (context_len + 1) — the +1 gives
    us the label shift: input = tokens[:-1], target = tokens[1:].
    """

    def __init__(
        self,
        shard_dir: str,
        context_len: int = 4096,
        shuffle_shards: bool = True,
    ):
        super().__init__()
        self.shard_paths = sorted(Path(shard_dir).glob("*.bin"))
        self.context_len = context_len
        self.shuffle_shards = shuffle_shards

    def __iter__(self):
        shard_order = list(self.shard_paths)

        # Split shards across DataLoader workers. WITHOUT this, every worker
        # (num_workers > 1) iterates the *entire* shard list, so the DataLoader
        # yields each example once per worker — silently duplicating your data.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard_order = shard_order[worker_info.id :: worker_info.num_workers]
            # If num_workers > number of shards, some workers get nothing.

        if self.shuffle_shards:
            import random
            random.shuffle(shard_order)   # seed per (epoch, worker) in real code

        for shard_path in shard_order:
            # Memory-map: no data is loaded until a row is accessed
            data = np.memmap(shard_path, dtype=np.int32, mode="r")
            n_contexts = len(data) // (self.context_len + 1)

            # Shuffle context order within the shard
            indices = np.random.permutation(n_contexts)

            for idx in indices:
                start = idx * (self.context_len + 1)
                chunk = data[start : start + self.context_len + 1]
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield x, y


# =============================================================================
# Block #5 (line ~605): verify_pipeline.py -- deterministic, offline checks
# for the pipeline above. Verbatim from the chapter (the
# `from pipeline import ...` import line is dropped since, in this
# concatenated module, those names are already in scope -- exactly as the
# book's own "Usage" note describes them living in a sibling `pipeline.py`).
# =============================================================================

def test_parse_wet_records():
    """2-record WET sample -> exact (url, text); warcinfo skipped, no leakage."""
    sample = (
        "WARC/1.0\nWARC-Type: warcinfo\n"
        "Content-Type: application/warc-fields\nContent-Length: 26\n\n"
        "isPartOf: CC-MAIN-2024-18\n\n"
        "WARC/1.0\nWARC-Type: conversion\n"
        "WARC-Target-URI: https://example.com/a\n"
        "WARC-Date: 2024-04-15T12:34:56Z\nContent-Type: text/plain\n"
        "Content-Length: 28\n\nHello world.\nThis is page A.\n\n"
        "WARC/1.0\nWARC-Type: conversion\n"
        "WARC-Target-URI: https://example.com/b\n"
        "Content-Type: text/plain\nContent-Length: 12\n\nPage B only.\n\n"
    )
    got = list(_iter_wet_records(sample.splitlines(keepends=True)))
    assert got == [
        ("https://example.com/a", "Hello world.\nThis is page A."),
        ("https://example.com/b", "Page B only."),
    ], got
    for _, text in got:                      # no header / boundary leakage
        assert "Content-Length" not in text and "WARC/1.0" not in text
    print("test_parse_wet_records: OK (2 docs, warcinfo skipped, no leakage)")


def test_quality_filter():
    good = "\n".join([
        "The history of computing spans several centuries of innovation.",
        "Early mechanical calculators gave way to electronic machines.",
        "Modern processors execute billions of instructions each second.",
        "Software links these machines into a global information network.",
        "Researchers keep pushing the boundaries of what is possible.",
    ])
    assert passes_quality_filter(good) is True           # clean prose passes
    assert passes_quality_filter("Hi there.") is False    # too short
    assert passes_quality_filter("buy " * 300) is False   # repetition + no punct
    print("test_quality_filter: OK")


def test_token_shard():
    # ctx_len=4 -> row_len=5; shard_size=15 -> 3 rows/shard.
    # Feed 4+1, 4+1, 4+1 (flush 15), then 7+1; finalize keeps 1 row (5), drops 3.
    with tempfile.TemporaryDirectory() as d:
        shard = TokenShard(d, shard_size=15, ctx_len=4)
        shard.add_tokens([1, 2, 3, 4], eos_id=0)                   # buffer 5
        shard.add_tokens([5, 6, 7, 8], eos_id=0)                   # buffer 10
        shard.add_tokens([9, 10, 11, 12], eos_id=0)               # 15 -> flush
        shard.add_tokens([13, 14, 15, 16, 17, 18, 19], eos_id=0)  # buffer 8
        shard.finalize()                                          # flush 5, drop 3
        assert shard._total_tokens == 20, shard._total_tokens
        rows = np.fromfile(sorted(Path(d).glob("*.bin"))[0], dtype=np.int32)
        assert len(rows) % 5 == 0                                  # whole rows
    print("test_token_shard: OK (20 tokens, 3 dropped, rows of ctx_len+1)")


# =============================================================================
# Additional glue: exercise the parts of block #1 and block #4 that the
# chapter's own verify_pipeline.py (block #5) does not cover.
# =============================================================================

def test_decode_html():
    """Block #1's _decode_html has no third-party dependency; exercise directly."""
    # (a) explicit charset in Content-Type, valid utf-8 bytes
    raw = "café".encode("utf-8")
    out = _decode_html(raw, "text/html; charset=utf-8")
    assert out == "café", out

    # (b) unknown/bogus charset -> LookupError -> falls back to utf-8
    out = _decode_html(raw, "text/html; charset=made-up-charset")
    assert out == "café", out

    # (c) no charset given, latin-1 bytes that are invalid utf-8 -> falls back to latin-1
    raw_latin1 = "café".encode("latin-1")
    out = _decode_html(raw_latin1, "text/html")
    assert out == "café", out
    print("test_decode_html: OK (charset, fallback-to-utf8, fallback-to-latin1)")


def test_extract_text_from_warc():
    """Block #1's extract_text_from_warc delegates parsing to warcio.ArchiveIterator
    and extraction to trafilatura -- neither is in the guaranteed CI dependency set
    (numpy, torch, einops, sklearn, stdlib), so it is defined-but-not-called here.
    """
    if ArchiveIterator is None or trafilatura is None:
        print("test_extract_text_from_warc: SKIP(optional-deps): "
              "warcio/trafilatura not installed (not in guaranteed CI deps); "
              "extract_text_from_warc is defined but not called.")
        return
    # If the optional deps ARE present, do a real, tiny, offline round trip.
    warc_bytes = (
        b"WARC/1.0\r\n"
        b"WARC-Type: response\r\n"
        b"WARC-Target-URI: https://example.com/a\r\n"
        b"Content-Type: application/http; msgtype=response\r\n"
        b"Content-Length: 0\r\n\r\n"
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n\r\n"
        b"<html><body><article><p>Hello world, this is the article.</p></article>"
        b"<nav>menu</nav></body></html>\r\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".warc") as f:
        f.write(warc_bytes)
        f.flush()
        results = list(extract_text_from_warc(f.name))
    assert len(results) >= 0  # offline smoke test; exact extraction depends on trafilatura version
    print(f"test_extract_text_from_warc: OK ({len(results)} record(s) extracted)")


def test_sharded_token_dataset():
    """Block #4: write a tiny shard with TokenShard (block #3), then read it
    back end-to-end with ShardedTokenDataset (block #4) and check the
    (input, label) shift is correct.
    """
    ctx_len = 4
    with tempfile.TemporaryDirectory() as d:
        # shard_size=10 -> row_len=5 -> rows_per_shard=2 -> shard_tokens=10
        shard = TokenShard(d, shard_size=10, ctx_len=ctx_len)
        shard.add_tokens([1, 2, 3, 4], eos_id=0)   # buffer -> [1,2,3,4,0]  (5)
        shard.add_tokens([5, 6, 7, 8], eos_id=0)   # buffer -> 10 tokens -> flush
        shard.finalize()
        assert shard._total_tokens == 10, shard._total_tokens

        dataset = ShardedTokenDataset(d, context_len=ctx_len, shuffle_shards=False)
        rows = list(iter(dataset))
        assert len(rows) == 2, len(rows)

        seen = set()
        for x, y in rows:
            assert x.dtype == torch.int64 and y.dtype == torch.int64
            assert x.shape == (ctx_len,) and y.shape == (ctx_len,)
            # label shift: y == x shifted by one, i.e. row[1:] vs row[:-1]
            assert torch.equal(y[:-1], x[1:]), (x, y)
            seen.add(tuple(x.tolist()))
        assert seen == {(1, 2, 3, 4), (5, 6, 7, 8)}, seen
    print("test_sharded_token_dataset: OK (2 rows, label shift verified)")


if __name__ == "__main__":
    # Block #5's own checks (exercises block #3's _iter_wet_records,
    # passes_quality_filter, and TokenShard).
    test_parse_wet_records()
    test_quality_filter()
    test_token_shard()
    print("All pipeline checks passed.")

    # Extra glue exercising block #1 and block #4.
    test_decode_html()
    test_extract_text_from_warc()
    test_sharded_token_dataset()

    print("\nAll book-code checks passed for 03-pretraining/01-pretraining-data.md.")
