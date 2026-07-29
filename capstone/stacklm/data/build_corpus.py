"""The corpus driver (Ch. 14.2): turn the four sources into sharded, packed
`train/` + `val/` corpora that honour the 70/15/10/5 mix and a total token budget.

Stages, per source: stream -> quality filter -> exact dedup -> near dedup ->
tokenize (once) -> stop at that source's token budget. The four streams are then
interleaved by weight (so the model sees code from step 0, not only after 85% of
the run), pushed through a reservoir shuffle buffer, split into a deterministic
document-level held-out set, packed, and written as uint16 shards with a
`manifest.json` recording the realized mix.

In production the interleave step can be delegated to
`datasets.interleave_datasets(streams, probabilities=[...], seed=...,
stopping_strategy="all_exhausted")`; the hand-rolled version below is used so the
budget accounting is explicit and works on the offline synthetic corpus too.
"""
import hashlib
import json
import random
from pathlib import Path

from .dedup import exact_dedup, near_dedup_stream
from .filters import filter_config_hash, quality_filter
from .pack import pack_documents
from .shard import ShardWriter
from .synthetic import STACK100M_MIX, TOTAL_TOKEN_BUDGET, stream_source


def encode_batched(docs, tokenizer, batch: int = 1024):
    """Attach `ids` to each document, encoding in batches.

    Every fast encoder in the ecosystem is a BATCH API -- Ch. 14.3's
    `encode_corpus` (multiprocessing.Pool over documents), HF `tokenizers`'
    `encode_batch`, `tiktoken`'s `encode_ordinary_batch`. Calling `encode` one
    document at a time leaves the ~7x process-parallel speedup on the table, and
    at 84 GB of text that is ~4 core-hours instead of ~35 minutes.
    """
    encode_batch = getattr(tokenizer, "encode_batch", None)
    buf = []

    def flush():
        texts = [d["text"] for d in buf]
        id_lists = (encode_batch(texts) if encode_batch is not None
                    else [tokenizer.encode(t) for t in texts])
        for d, ids in zip(buf, id_lists):
            yield {**d, "ids": list(ids)}
        buf.clear()

    for doc in docs:
        buf.append(doc)
        if len(buf) >= batch:
            yield from flush()
    if buf:
        yield from flush()


def _source_pipeline(entry, tokenizer, budget_tokens, offline, dedup_kwargs, stats,
                     encode_batch_size: int = 1024):
    """Filtered, deduplicated, tokenized documents from one source, capped at
    `budget_tokens`. Tokenizing here (once) is what makes the budget exact."""
    docs = stream_source(entry, offline=offline)
    docs = (d for d in docs if quality_filter(d))
    docs = exact_dedup(docs)
    docs = near_dedup_stream(docs, **dedup_kwargs)
    used = 0
    for doc in encode_batched(docs, tokenizer, batch=encode_batch_size):
        if not doc["ids"]:
            continue
        used += len(doc["ids"]) + 2              # +2 for <bos>/<eos> added at pack time
        stats[entry.name] = used
        yield doc
        if used >= budget_tokens:
            return


def interleave_budgeted(entries, tokenizer, total_tokens, offline=True,
                        seed=1337, dedup_kwargs=None, stats=None):
    """Weighted round-robin over the per-source pipelines. Each source gets
    `weight * total_tokens`; a source that runs dry is dropped and the remaining
    weights renormalize -- deliberately UNLIKE
    `datasets.interleave_datasets(stopping_strategy=...)`, whose
    `"first_exhausted"` truncates the whole mix and whose `"all_exhausted"`
    oversamples the drained source. See the comparison at the end of this
    section."""
    stats = {} if stats is None else stats
    dedup_kwargs = dedup_kwargs or {}
    rng = random.Random(seed)
    gens, weights = {}, {}
    for e in entries:
        budget = int(round(e.weight * total_tokens))
        gens[e.name] = _source_pipeline(e, tokenizer, budget, offline, dedup_kwargs, stats)
        weights[e.name] = e.weight
        stats.setdefault(e.name, 0)
    alive = list(gens)
    while alive:
        name = rng.choices(alive, weights=[weights[n] for n in alive], k=1)[0]
        try:
            yield next(gens[name])
        except StopIteration:
            alive.remove(name)


def shuffle_buffer(docs, size=100_000, seed=1337):
    """Reservoir shuffle: without it, shard 0 would be ordered exactly as the
    sources were interleaved, and any resume-from-shard-k restart would see a
    biased slice of the mix."""
    rng = random.Random(seed)
    buf = []
    for doc in docs:
        if len(buf) < size:
            buf.append(doc)
            continue
        j = rng.randrange(size)
        yield buf[j]
        buf[j] = doc
    rng.shuffle(buf)
    yield from buf


def is_holdout(doc, per_mille: int = 1) -> bool:
    """Deterministic document-level held-out assignment. Hashing the *text*
    (not a counter) means the same document lands in the same split on every
    rebuild, and a document is never split across train and val -- the failure
    mode that silently contaminates a held-out perplexity."""
    h = int(hashlib.blake2b(doc["text"].encode("utf-8"), digest_size=8).hexdigest(), 16)
    return (h % 1000) < per_mille


def build_corpus(out_dir, tokenizer, total_tokens=TOTAL_TOKEN_BUDGET, entries=None,
                 seq_len=2048, tokens_per_shard=100_000_000, offline=True,
                 holdout_per_mille=1, holdout_tokens=10_000_000,
                 shuffle_size=100_000, seed=1337, dedup_kwargs=None) -> dict:
    """Build `out_dir/train` and `out_dir/val` shards. Returns the manifest."""
    entries = entries if entries is not None else STACK100M_MIX
    assert abs(sum(e.weight for e in entries) - 1.0) < 1e-9, "mix weights must sum to 1"
    out = Path(out_dir)
    stats: dict = {}
    val_docs: list = []
    held = {"tokens": 0}

    docs = interleave_budgeted(entries, tokenizer, total_tokens, offline=offline,
                               seed=seed, dedup_kwargs=dedup_kwargs, stats=stats)
    docs = shuffle_buffer(docs, size=shuffle_size, seed=seed)

    def train_stream():
        for doc in docs:
            if is_holdout(doc, holdout_per_mille):
                if held["tokens"] < holdout_tokens:
                    val_docs.append(doc)
                    held["tokens"] += len(doc["ids"]) + 2
                continue                     # held-out docs NEVER enter training
            yield doc

    train_writer = ShardWriter(out / "train", seq_len=seq_len,
                               tokens_per_shard=tokens_per_shard)
    for ids, pos in pack_documents(train_stream(), tokenizer, seq_len=seq_len):
        train_writer.add(ids, pos)
    train_writer.close()

    val_writer = ShardWriter(out / "val", seq_len=seq_len,
                             tokens_per_shard=tokens_per_shard)
    for ids, pos in pack_documents(iter(val_docs), tokenizer, seq_len=seq_len):
        val_writer.add(ids, pos)
    val_writer.close()

    realized = sum(stats.values()) or 1
    provenance = {
        "seed": seed,
        "token_budget": total_tokens,
        "filter_config_hash": filter_config_hash(),
        "dedup": {"num_perm": 128, "bands": 16, "threshold": 0.8, **(dedup_kwargs or {})},
        "sources": [
            {"name": e.name, "hf_path": e.hf_path, "hf_config": e.hf_config,
             "hf_data_dir": e.hf_data_dir, "revision": e.revision,
             "target_weight": e.weight,
             "realized_tokens": stats.get(e.name, 0),
             "realized_weight": round(stats.get(e.name, 0) / realized, 4)}
            for e in entries
        ],
        "holdout": {"per_mille": holdout_per_mille, "tokens": held["tokens"]},
        "offline_synthetic": offline,
    }
    man = train_writer.write_manifest(tokenizer=tokenizer, extra=provenance)
    val_writer.write_manifest(tokenizer=tokenizer, extra=provenance)
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    return man


if __name__ == "__main__":  # tiny offline demo: python -m stacklm.data.build_corpus
    import tempfile
    from ..tokenizer import StackTokenizer, SPECIAL_TOKENS
    from .synthetic import synthetic_text_sample

    tok = StackTokenizer()
    tok.train(synthetic_text_sample(60), vocab_size=384, special_tokens=SPECIAL_TOKENS)
    d = tempfile.mkdtemp(prefix="stacklm_corpus_")
    m = build_corpus(d, tok, total_tokens=200_000, seq_len=128,
                     tokens_per_shard=128 * 64, offline=True, holdout_per_mille=20,
                     holdout_tokens=20_000, shuffle_size=256)
    print(json.dumps(m, indent=2))
