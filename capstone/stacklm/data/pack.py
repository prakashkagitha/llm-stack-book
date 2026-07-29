"""Pack tokenized documents into fixed-length sequences with document-aware
segmentation (Ch. 14.2).

Every packed chunk begins with `<bos>`, so the token array alone carries the
document boundaries: `cumsum(input_ids == bos_id)` recovers each token's segment
id and `build_intra_doc_causal_mask` turns that into a block-diagonal causal mask
(no cross-document attention). Position ids are *derived*, never stored -- see
`segments_from_bos`.
"""
from typing import Iterable, Iterator, Protocol
import numpy as np


class Tokenizer(Protocol):
    bos_id: int
    eos_id: int
    pad_id: int
    def encode(self, text: str) -> list: ...


def pack_documents(docs: Iterable[dict], tokenizer, seq_len: int) -> Iterator[tuple]:
    """Greedily concatenate `<bos> body <eos>` across documents into fixed-length
    windows. Yields (input_ids, position_ids) per window; the final partial window
    is padded so no tokens are silently dropped. A document dict may carry
    pre-computed `ids` (the corpus builder tokenizes once, for budgeting)."""
    buf_ids: list = []
    buf_pos: list = []
    max_body = seq_len - 2  # room for <bos> and <eos> in every chunk

    for doc in docs:
        raw = doc["ids"] if "ids" in doc else tokenizer.encode(doc["text"])
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


def segments_from_bos(input_ids: np.ndarray, bos_id: int) -> tuple:
    """Derive (seq_ids, position_ids) from a packed window's tokens alone.

    `seq_ids[i]` is the index of the document token i belongs to (-1 for a
    leading fragment continued from the previous window); `position_ids[i]` is
    the offset of token i inside its document. Storing these on disk would double
    the corpus for information the token array already contains.
    """
    starts = input_ids == bos_id
    seq_ids = np.cumsum(starts) - 1                       # -1 for a leading tail
    idx = np.arange(input_ids.shape[0])
    seg_start = np.maximum.accumulate(np.where(starts, idx, -1))
    return seq_ids, idx - np.maximum(seg_start, 0)


def build_intra_doc_causal_mask(position_ids: np.ndarray) -> np.ndarray:
    """Block-diagonal causal mask from position ids (a reset to 0 starts a new
    document). `True` means "token i may attend to token j"."""
    seq_len = position_ids.shape[0]
    doc_id = np.cumsum(position_ids == 0)  # monotonically increasing per document
    causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    same_doc = doc_id[:, None] == doc_id[None, :]
    return causal & same_doc


def segment_ids_from_positions(position_ids: np.ndarray) -> np.ndarray:
    """The per-token segment id (0,1,2,...) used by `Stack100M.forward(seq_ids=...)`."""
    return np.cumsum(position_ids == 0) - 1
