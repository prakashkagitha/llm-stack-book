"""Weighted source mixtures over the Ch. 14.2 packed shards (Ch. 14.8).

Each source lives in its own shard directory, packed at the sequence length the
sub-phase will train at (`data/mid/<source>_<seq_len>/`). Sampling is per
*sequence*, not per micro-batch, so a single forward pass mixes domains --
which decorrelates gradients across web / code / math within one micro-batch.
"""
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..data import PackedMemmapDataset

# Sub-phase A: the annealing mix (Ch. 14.8). Keys are shard-directory names under
# `data/mid/<name>_2048`. Ch. 14.2's `build_corpus` driver writes ONE interleaved
# train/val corpus, so a per-source mixture needs a per-source pass first: the same
# `build_shards` call at seq_len=2048, one output directory per source, no length
# filter. `instruct_flav` is the QA/how-to raw text of Cosmopedia v1's `wikihow` +
# `khanacademy` configs -- shaped like instructions, but NOT instruction/response
# pairs (that is Ch. 14.9).
ANNEAL_MIX = {
    "fineweb_edu":   0.40,
    "cosmopedia_v2": 0.30,
    "starcoder":     0.15,
    "finemath":      0.10,
    "instruct_flav": 0.05,
}

# Sub-phase B: long-context. Only genuinely long sources, plus a 10% short-form
# anchor so short-context quality does not drift (ProLong; Llama 3).
LONGCTX_MIX = {
    "starcoder_repo":   0.35,
    "books_pg19":       0.25,
    "arxiv_proofpile2": 0.15,
    "fineweb_edu_long": 0.15,
    "cosmopedia_v2":    0.10,
}

# Sub-phase C: capability injection at the LR floor. Every key must exist at
# seq_len=8192, i.e. must have been built by scripts/repack_long.py -- FineMath is
# short-form and lives only in the 2048 shards, so arXiv/proof-pile-2 carries math.
CAPABILITY_MIX = {
    "arxiv_proofpile2": 0.30,
    "starcoder_repo":   0.30,
    "cosmopedia_v2":    0.25,
    "fineweb_edu_long": 0.15,
}

for _m in (ANNEAL_MIX, LONGCTX_MIX, CAPABILITY_MIX):
    assert abs(sum(_m.values()) - 1.0) < 1e-9, "mixture weights must sum to 1"


def build_mixture_loader(mix: dict, seq_len: int, micro_bs: int, *,
                         root: str = "data/mid", seed: int = 1234,
                         num_workers: int = 4):
    """Return an INFINITE iterator of batch dicts drawn from `mix`.

    Each yielded dict has `input_ids`, `position_ids`, `seq_ids`, `targets`
    (shapes (micro_bs, seq_len - 1)) -- exactly what `Stack100M.forward` and the
    document-aware mask consume.
    """
    datasets, weights = [], []
    for name, w in mix.items():
        d = PackedMemmapDataset(f"{root}/{name}_{seq_len}")
        # Guard against the single most common mid-training bug: reading shards
        # packed at the PRETRAIN length while claiming to train long.
        assert d.seq_len == seq_len, (
            f"{name} shards are packed at {d.seq_len}, not {seq_len}; "
            f"re-run scripts/repack_long.py")
        datasets.append(d)
        weights.extend([w / len(d)] * len(d))   # per-sequence prob ∝ source weight

    concat = ConcatDataset(datasets)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(concat),
                                    replacement=True, generator=g)
    loader = DataLoader(concat, batch_size=micro_bs, sampler=sampler,
                        drop_last=True, num_workers=num_workers,
                        pin_memory=True, persistent_workers=num_workers > 0)

    def infinite():
        while True:
            yield from loader
    return infinite()
