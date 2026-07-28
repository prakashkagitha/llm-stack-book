"""Pack tokenized documents into fixed-length sequences with document-aware
position ids (Ch. 14.2). Every chunk restarts its position clock at 0, so a later
`cumsum(position_ids == 0)` recovers document boundaries without storing a
separate doc-id array. `build_intra_doc_causal_mask` turns those position ids into
a block-diagonal causal mask (no cross-document attention).
"""
from typing import Iterable, Iterator, Protocol
import numpy as np


class Tokenizer(Protocol):
    bos_id: int
    eos_id: int
    pad_id: int
    def encode(self, text: str) -> list: ...


def pack_documents(docs: Iterable[dict], tokenizer, seq_len: int) -> Iterator[tuple]:
    buf_ids: list = []
    buf_pos: list = []
    max_body = seq_len - 2  # room for <bos> and <eos> in every chunk

    for doc in docs:
        raw = tokenizer.encode(doc["text"])
        chunks = [raw[i:i + max_body] for i in range(0, len(raw), max_body)] or [[]]
        for chunk in chunks:
            toks = [tokenizer.bos_id, *chunk, tokenizer.eos_id]
            pos = list(range(len(toks)))  # this chunk's own position clock, from 0
            buf_ids.extend(toks)
            buf_pos.extend(pos)
            while len(buf_ids) >= seq_len:
                yield buf_ids[:seq_len], buf_pos[:seq_len]
                buf_ids, buf_pos = buf_ids[seq_len:], buf_pos[seq_len:]

    if buf_ids:  # flush a final, padded window
        pad_n = seq_len - len(buf_ids)
        buf_ids.extend([tokenizer.pad_id] * pad_n)
        buf_pos.extend([0] * pad_n)
        yield buf_ids, buf_pos


def build_intra_doc_causal_mask(position_ids: np.ndarray) -> np.ndarray:
    seq_len = position_ids.shape[0]
    doc_id = np.cumsum(position_ids == 0)  # monotonically increasing per document
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    same_doc = doc_id[:, None] == doc_id[None, :]
    return causal & same_doc


def segment_ids_from_positions(position_ids: np.ndarray) -> np.ndarray:
    """The per-token segment id (0,1,2,...) used by `Stack100M.forward(seq_ids=...)`."""
    return np.cumsum(position_ids == 0) - 1
