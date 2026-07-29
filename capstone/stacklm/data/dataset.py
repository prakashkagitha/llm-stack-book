"""Streaming memmap dataset over the packed shards (Ch. 14.2).

Only the token array is stored on disk. `seq_ids` (the segment id
`Stack100M.forward` consumes for document-aware masking) and `position_ids` are
derived per item from `input_ids == bos_id` -- two cheap numpy ops that save
40GB on a 20B-token corpus. uint16 on disk, int64 in tensors.
"""
import bisect
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .pack import segments_from_bos


class PackedMemmapDataset(Dataset):
    def __init__(self, shard_dir: str, bos_id: int = None):
        self.shard_dir = Path(shard_dir)
        self._shards = []       # list of (tokens_memmap, pos_memmap_or_None)
        self._cum_seqs = [0]    # prefix sums of sequence counts, for indexing
        self.seq_len = None

        man_path = self.shard_dir / "manifest.json"
        self.manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
        self.bos_id = bos_id if bos_id is not None else self.manifest.get("bos_id")

        for meta_path in sorted(self.shard_dir.glob("shard_*.meta.bin")):
            n_seq, seq_len = (int(x) for x in np.fromfile(meta_path, dtype=np.int64))
            stem = str(meta_path)[: -len(".meta.bin")]
            tok_mm = np.memmap(stem + ".tokens.bin", dtype=np.uint16, mode="r",
                               shape=(n_seq, seq_len))
            pos_path = Path(stem + ".pos.bin")
            pos_mm = (np.memmap(pos_path, dtype=np.uint16, mode="r", shape=(n_seq, seq_len))
                      if pos_path.exists() else None)
            self._shards.append((tok_mm, pos_mm))
            self._cum_seqs.append(self._cum_seqs[-1] + n_seq)
            self.seq_len = seq_len

        if self._shards and self._shards[0][1] is None and self.bos_id is None:
            raise ValueError(
                f"{shard_dir}: no .pos.bin and no bos_id (manifest.json missing?). "
                "Pass PackedMemmapDataset(dir, bos_id=tok.bos_id)."
            )

    def __len__(self) -> int:
        return self._cum_seqs[-1]

    def _locate(self, idx: int) -> tuple:
        """(shard, row) for a global sequence index. Binary search, not a linear
        scan: with 200 shards and ~10M samples/epoch the scan would cost billions
        of pointless Python comparisons per epoch."""
        if not 0 <= idx < self._cum_seqs[-1]:
            raise IndexError(idx)
        s = bisect.bisect_right(self._cum_seqs, idx) - 1
        return s, idx - self._cum_seqs[s]

    def __getitem__(self, idx: int) -> dict:
        s, row = self._locate(idx)
        tok_mm, pos_mm = self._shards[s]
        ids_np = tok_mm[row].astype(np.int64)
        if pos_mm is not None:                        # legacy shards with .pos.bin
            pos_np = pos_mm[row].astype(np.int64)
            seq_np = np.cumsum(pos_np == 0) - 1
        else:                                        # derive from the tokens alone
            seq_np, pos_np = segments_from_bos(ids_np, self.bos_id)
        ids = torch.from_numpy(ids_np)
        return {
            "input_ids": ids[:-1],
            "position_ids": torch.from_numpy(np.ascontiguousarray(pos_np))[:-1],
            "seq_ids": torch.from_numpy(np.ascontiguousarray(seq_np))[:-1],
            "targets": ids[1:],
        }
