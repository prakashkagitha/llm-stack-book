"""Write packed sequences to uint16 memmap shards (Ch. 14.2).

One shard = `shard_XXXXX.tokens.bin` (flat uint16, n_seq x seq_len) plus a tiny
`.meta.bin` holding [n_seq, seq_len]. Position ids are NOT stored: they are
derived on read from `input_ids == bos_id` (`pack.segments_from_bos`), which
halves the corpus on disk and in page cache. `store_positions=True` writes the
legacy `.pos.bin` companion for inspection/debugging.

A `manifest.json` beside the shards records everything the run needs to be
reproducible (Ch. 14.12): seq_len, special-token ids, token counts, and whatever
provenance the caller passes in.
"""
import json
import numpy as np
from pathlib import Path

DTYPE = np.uint16  # vocab_size=32768 and seq_len=2048 both fit


class ShardWriter:
    def __init__(self, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000,
                 store_positions: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.seqs_per_shard = max(1, tokens_per_shard // seq_len)
        self.store_positions = store_positions
        self._buf_ids: list = []
        self._buf_pos: list = []
        self._shard_idx = 0
        self.n_sequences = 0

    def add(self, input_ids, position_ids=None) -> None:
        self._buf_ids.append(np.asarray(input_ids, dtype=DTYPE))
        if self.store_positions:
            self._buf_pos.append(np.asarray(position_ids, dtype=DTYPE))
        self.n_sequences += 1
        if len(self._buf_ids) >= self.seqs_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_ids:
            return
        ids = np.stack(self._buf_ids)  # (n_seq, seq_len)
        stem = str(self.out_dir / f"shard_{self._shard_idx:05d}")
        ids.tofile(stem + ".tokens.bin")
        if self.store_positions:
            np.stack(self._buf_pos).tofile(stem + ".pos.bin")
        np.array([ids.shape[0], ids.shape[1]], dtype=np.int64).tofile(stem + ".meta.bin")
        self._shard_idx += 1
        self._buf_ids.clear()
        self._buf_pos.clear()

    def close(self) -> None:
        self._flush()  # flush the trailing partial shard

    def write_manifest(self, tokenizer=None, extra: dict = None) -> dict:
        """Write manifest.json beside the shards. `bos_id` is the load-bearing
        field: the dataset needs it to recover document boundaries on read."""
        man = {
            "seq_len": self.seq_len,
            "n_shards": self._shard_idx,
            "n_sequences": self.n_sequences,
            "n_tokens": self.n_sequences * self.seq_len,
            "store_positions": self.store_positions,
        }
        if tokenizer is not None:
            man.update(bos_id=int(tokenizer.bos_id), eos_id=int(tokenizer.eos_id),
                       pad_id=int(tokenizer.pad_id))
        if extra:
            man.update(extra)
        (self.out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        return man


def build_shards(docs, tokenizer, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000,
                 store_positions: bool = False, manifest_extra: dict = None) -> int:
    """End-to-end: pack a document stream, write shards + manifest. Returns the
    number of shards written."""
    from .pack import pack_documents
    writer = ShardWriter(out_dir, seq_len=seq_len, tokens_per_shard=tokens_per_shard,
                         store_positions=store_positions)
    for input_ids, position_ids in pack_documents(docs, tokenizer, seq_len=seq_len):
        writer.add(input_ids, position_ids)
    writer.close()
    writer.write_manifest(tokenizer=tokenizer, extra=manifest_extra)
    return writer._shard_idx
