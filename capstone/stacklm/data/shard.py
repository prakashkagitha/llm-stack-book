"""Write packed sequences to uint16 memmap shards (Ch. 14.2). Each shard is a
trio: `.tokens.bin`, `.pos.bin`, and a small `.meta.bin` holding [n_seq, seq_len].
"""
import numpy as np
from pathlib import Path

DTYPE = np.uint16  # vocab_size=32768 and seq_len=2048 both fit


class ShardWriter:
    def __init__(self, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.seqs_per_shard = max(1, tokens_per_shard // seq_len)
        self._buf_ids: list = []
        self._buf_pos: list = []
        self._shard_idx = 0

    def add(self, input_ids: list, position_ids: list) -> None:
        self._buf_ids.append(np.asarray(input_ids, dtype=DTYPE))
        self._buf_pos.append(np.asarray(position_ids, dtype=DTYPE))
        if len(self._buf_ids) >= self.seqs_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf_ids:
            return
        ids = np.stack(self._buf_ids)  # (n_seq, seq_len)
        pos = np.stack(self._buf_pos)  # (n_seq, seq_len)
        stem = str(self.out_dir / f"shard_{self._shard_idx:05d}")
        ids.tofile(stem + ".tokens.bin")
        pos.tofile(stem + ".pos.bin")
        np.array([ids.shape[0], ids.shape[1]], dtype=np.int64).tofile(stem + ".meta.bin")
        self._shard_idx += 1
        self._buf_ids.clear()
        self._buf_pos.clear()

    def close(self) -> None:
        self._flush()  # flush the trailing partial shard


def build_shards(docs, tokenizer, out_dir: str, seq_len: int = 2048,
                 tokens_per_shard: int = 100_000_000) -> int:
    from .pack import pack_documents
    writer = ShardWriter(out_dir, seq_len=seq_len, tokens_per_shard=tokens_per_shard)
    for input_ids, position_ids in pack_documents(docs, tokenizer, seq_len=seq_len):
        writer.add(input_ids, position_ids)
    writer.close()
    return writer._shard_idx
