"""Streaming memmap dataset over the packed shards (Ch. 14.2). Returns
`input_ids`, `position_ids`, `seq_ids` (segment id for document-aware masking) and
`targets` (the next-token labels). uint16 on disk, int64 in tensors.
"""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PackedMemmapDataset(Dataset):
    def __init__(self, shard_dir: str):
        self.shard_dir = Path(shard_dir)
        self._shards = []       # list of (tokens_memmap, pos_memmap)
        self._cum_seqs = [0]    # prefix sums of sequence counts, for indexing
        self.seq_len = None
        for meta_path in sorted(self.shard_dir.glob("shard_*.meta.bin")):
            n_seq, seq_len = (int(x) for x in np.fromfile(meta_path, dtype=np.int64))
            stem = str(meta_path)[: -len(".meta.bin")]
            tok_mm = np.memmap(stem + ".tokens.bin", dtype=np.uint16, mode="r",
                               shape=(n_seq, seq_len))
            pos_mm = np.memmap(stem + ".pos.bin", dtype=np.uint16, mode="r",
                               shape=(n_seq, seq_len))
            self._shards.append((tok_mm, pos_mm))
            self._cum_seqs.append(self._cum_seqs[-1] + n_seq)
            self.seq_len = seq_len

    def __len__(self) -> int:
        return self._cum_seqs[-1]

    def _locate(self, idx: int) -> tuple:
        for s, (start, end) in enumerate(zip(self._cum_seqs, self._cum_seqs[1:])):
            if start <= idx < end:
                return s, idx - start
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> dict:
        s, row = self._locate(idx)
        tok_mm, pos_mm = self._shards[s]
        ids = torch.from_numpy(tok_mm[row].astype(np.int64))
        pos = torch.from_numpy(pos_mm[row].astype(np.int64))
        seq_ids = torch.cumsum((pos == 0).long(), dim=0) - 1     # 0,1,2,... per document
        return {
            "input_ids": ids[:-1],
            "position_ids": pos[:-1],
            "seq_ids": seq_ids[:-1],
            "targets": ids[1:],
        }
