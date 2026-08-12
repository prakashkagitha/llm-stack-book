#!/usr/bin/env python3
"""Build the sub-phase-B shards: seq_len=8192, long documents only (Ch. 14.8).

Run once, between sub-phase A and sub-phase B:
    python capstone/scripts/repack_long.py --out data/mid --seq-len 8192

With per-document position resets (Ch. 14.2) the largest position id the model
ever sees equals the longest DOCUMENT, not the packed window -- so long-context
extension is a no-op unless the shards contain genuinely long documents.
`verify_positions` is the assertion that catches that before you burn 0.6B tokens.
"""
import argparse
from collections import defaultdict

import numpy as np

from stacklm.data import (DataMixEntry, PackedMemmapDataset, build_shards,
                          load_hf_stream, stream_source)              # Ch. 14.2
from stacklm.tokenizer import StackTokenizer                          # Ch. 14.3

MIN_DOC_TOKENS = 4096          # half the target window; see the assertion below


def length_filtered(docs, tok, min_tokens: int = MIN_DOC_TOKENS):
    """Keep only documents that can actually exercise positions past 2048.

    Tokenizing twice (here and in `pack_documents`) is wasteful; at 20B tokens you
    would instead carry a `n_tokens` field through the Ch. 14.2 pipeline, or use
    HuggingFace `datatrove`'s TokensCounter + a LambdaFilter to do it in one pass.
    """
    for doc in docs:
        if len(tok.encode(doc["text"])) >= min_tokens:
            yield doc


def repo_level_documents(files, sep: str = "\n\n# ==== file: {path} ====\n\n"):
    """Concatenate a repository's files into ONE document (StarCoder2 / DeepSeek-Coder).

    `files` is a stream of dicts with `repo_name`, `path`, `content`. Sorting by
    path makes the concatenation deterministic.
    """
    by_repo = defaultdict(list)
    for f in files:
        by_repo[f.get("repo_name", "unknown")].append(f)
    for repo, fs in by_repo.items():
        fs.sort(key=lambda f: f.get("path", ""))
        body = "".join(sep.format(path=f.get("path", "")) + f.get("content", f.get("text", ""))
                       for f in fs)
        yield {"text": body, "source": "starcoder_repo", "repo": repo}


# The sub-phase-B sources, as Ch. 14.2 `DataMixEntry` records -- FULL loading
# coordinates, not just repo ids: three of these repos are multi-config (no
# default) and starcoderdata stores text under `content` and is sharded by
# language, so an entry missing those fields raises or yields nothing (Ch. 14.2,
# "the dataset id is not enough"). The first three are genuinely long; the next is
# a length-FILTERED slice of a pretrain source; the last is the deliberately SHORT
# anti-drift anchor (ProLong; Llama 3).
LONG_SOURCES = [
    (DataMixEntry("starcoder_repo", "bigcode/starcoderdata", 0.35, "code",
                  hf_data_dir="python", text_column="content", gated=True), True),
    (DataMixEntry("books_pg19", "deepmind/pg19", 0.25, "web"), False),
    (DataMixEntry("arxiv_proofpile2", "EleutherAI/proof-pile-2", 0.15, "math",
                  hf_config="arxiv"), False),
    (DataMixEntry("fineweb_edu_long", "HuggingFaceFW/fineweb-edu", 0.15, "web",
                  hf_config="sample-100BT"), False),
    (DataMixEntry("cosmopedia_v2", "HuggingFaceTB/smollm-corpus", 0.10, "synthetic",
                  hf_config="cosmopedia-v2"), False),   # v2 is NOT in the `cosmopedia` repo
]


def verify_positions(shard_dir: str, seq_len: int, floor: int = 4096,
                     n_sample: int = 512, seed: int = 0):
    """The check that decides whether sub-phase B is real or theatre.

    Ch. 14.2 does NOT store position ids on disk (`store_positions=False` is the
    default; they are recomputed from `input_ids == bos_id` by
    `segments_from_bos`), so we read them back through the same dataset class the
    trainer uses -- which also proves the shards are readable and packed at the
    right length. Sampling a few hundred rows is enough: we only need ONE window
    whose document reaches past `floor`.
    """
    ds = PackedMemmapDataset(shard_dir)
    assert ds.seq_len == seq_len, f"{shard_dir} packed at {ds.seq_len}, not {seq_len}"
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(ds), size=min(n_sample, len(ds)), replace=False)
    hi = max(int(ds[int(i)]["position_ids"].max()) for i in rows)
    print(f"  max position id in {shard_dir}: {hi} (window {seq_len})")
    assert hi > floor, (
        f"{shard_dir} contains no document longer than {floor} tokens: RoPE "
        f"rescaling would train on positions the data never reaches.")


def main(out_root: str, seq_len: int, tokenizer_path: str):
    tok = StackTokenizer.load(tokenizer_path)     # Ch. 14.3, vocab 32768
    for entry, repo_level in LONG_SOURCES:
        # Repo-level grouping needs the RAW rows (`repo_name`, `path`, `content`);
        # `stream_source` normalizes every row to {"text","source","domain"} and
        # drops exactly the fields the grouping keys on.
        raw = load_hf_stream(entry) if repo_level else stream_source(entry)
        docs = repo_level_documents(raw) if repo_level else raw
        if entry.name != "cosmopedia_v2":         # the short-form anchor stays unfiltered
            docs = length_filtered(docs, tok)
        out = f"{out_root}/{entry.name}_{seq_len}"
        n = build_shards(docs, tok, out, seq_len=seq_len,
                         tokens_per_shard=100_000_000)
        verify_positions(out, seq_len,
                         floor=4096 if entry.name != "cosmopedia_v2" else 0)
        print(f"{entry.name}: {n} shard(s) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mid")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    a = ap.parse_args()
    main(a.out, a.seq_len, a.tokenizer)
