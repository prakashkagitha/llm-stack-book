"""Parallel corpus encoding (Ch. 14.3). Documents are independent, so this is a
pure map: one worker per core, each with its own tokenizer + its own chunk cache.

Split shards on <|eos|> boundaries, never mid-document -- a document cut in half
would be pre-tokenized differently on each side of the cut.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Iterable, Iterator, List, Optional

import numpy as np

from .bpe import StackTokenizer, load_tokenizer

_TOK: Optional[StackTokenizer] = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOK
    _TOK = load_tokenizer(tokenizer_path)      # each worker gets its own cache


def _encode_doc(text: str) -> List[int]:
    assert _TOK is not None
    # Untrusted corpus text: NEVER allow special strings through (Ch. 14.3's
    # injection pitfall). The <|bos|>/<|eos|> wrapper is added by the packer.
    return _TOK.encode(text, allowed_special=frozenset())


def encode_corpus(docs: Iterable[str], tokenizer_path: str,
                  workers: int = 16, chunksize: int = 8) -> Iterator[np.ndarray]:
    """Yield one uint16 array per document, in input order."""
    with mp.Pool(workers, initializer=_init_worker,
                 initargs=(tokenizer_path,)) as pool:
        for ids in pool.imap(_encode_doc, docs, chunksize=chunksize):
            yield np.asarray(ids, dtype=np.uint16)   # vocab 32768 < 65536, fits
