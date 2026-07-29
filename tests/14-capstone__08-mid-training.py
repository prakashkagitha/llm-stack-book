"""
CI-sim test for content/14-capstone/08-mid-training.md

Blocks tested (in chapter order, concatenated so later blocks can use names
defined by earlier ones, exactly as the chapter's reading order implies):
  #0  (mixture.py)            -- build_mixture_loader / ANNEAL_MIX etc.
  #1  (schedule.py)           -- wsd_decay_multiplier
  #3  (rope.py)                -- build_rope_cache / ntk_rescaled_base
  #4  (continue_training.py)  -- extend_context / _unwrap
  #5  (repack_long.py)        -- length_filtered / repo_level_documents / verify_positions
  #6  (transformer.py _build_mask)
  #7  (doc_attention.py)      -- document_block_mask / doc_attention (FlexAttention)
  #11 (Exercise 6 solution)   -- bin_loss_by_position

Blocks SKIPPED:
  #2  -- fragment (loop over `muon`/`adamw`/`total_decay_steps` that are never
         defined in the surrounding text; it is a "here's the shape of the loop"
         illustration inside a warning box, not a standalone snippet).
  #8  -- needs-gpu: `pip install flash-attn` / `flash_attn_varlen_func`, a CUDA-only
         package not installed in CI (and not even importable on CPU).
  #9  -- needs-gpu: full mid-training loop launched under `torchrun`, real
         checkpoint files, real multi-GPU FSDP/DDP -- not CPU-runnable in a test.
  #10 -- needs-gpu: continuation of the same real training loop (checkpointing,
         eval loop against real shards).

The chapter's own code imports from `stacklm.*` (Ch. 14.2-14.6 modules that live
in the capstone package, not in this repo's test environment) and, in block #4,
uses a package-relative import (`from ..model.rope import ntk_rescaled_base`).
Since we're executing the blocks as one flat script rather than as the real
`capstone/stacklm` package, we:
  (a) register tiny stand-in modules in `sys.modules` for `stacklm.data` and
      `stacklm.tokenizer` so the chapter's import lines run UNCHANGED and the
      dataset/tokenizer objects they name have the right shape/interface, and
  (b) drop the relative import in block #4, since `ntk_rescaled_base` is already
      a global in this flat module (it's literally block #3, executed just above).
These are import-mechanics accommodations only -- no block's own logic is rewritten.
"""
import math
import sys
import types

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Glue: stand-in `stacklm.data` / `stacklm.tokenizer` modules (Ch. 14.2/14.3
# machinery this chapter re-uses but does not itself define). Shapes match what
# the chapter's code expects; no network, no disk.
# ----------------------------------------------------------------------------
import collections

DataMixEntry = collections.namedtuple("DataMixEntry", ["name", "hf_path", "weight", "domain"])


class PackedMemmapDataset:
    """Tiny stand-in for the Ch. 14.2 memmap-backed packed dataset. Deterministic,
    in-memory, and infers `seq_len` from the shard-directory naming convention
    the chapter itself uses (`<root>/<name>_<seq_len>`), so the chapter's own
    `assert d.seq_len == seq_len` guard is exercised for real."""

    def __init__(self, shard_dir: str, n_rows: int = 5):
        self.shard_dir = shard_dir
        self.seq_len = int(shard_dir.rsplit("_", 1)[-1])
        self.n_rows = n_rows

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        T = self.seq_len - 1
        g = torch.Generator().manual_seed(idx)
        return {
            "input_ids": torch.randint(0, 100, (T,), generator=g),
            "position_ids": torch.arange(T, dtype=torch.int64),
            "seq_ids": torch.zeros(T, dtype=torch.int64),
            "targets": torch.randint(0, 100, (T,), generator=g),
        }


def build_shards(*args, **kwargs):
    raise NotImplementedError("stub: real shard-building needs the Ch. 14.2 pipeline + network")


def stream_source(*args, **kwargs):
    raise NotImplementedError("stub: real streaming needs network access (HF datasets)")


class StackTokenizer:
    """Stand-in for the Ch. 14.3 BPE tokenizer -- whitespace-split is enough to
    exercise `length_filtered`'s token-count logic deterministically."""

    @classmethod
    def load(cls, path):
        return cls()

    def encode(self, text):
        return text.split()


_stacklm = types.ModuleType("stacklm")
_stacklm_data = types.ModuleType("stacklm.data")
_stacklm_data.PackedMemmapDataset = PackedMemmapDataset
_stacklm_data.DataMixEntry = DataMixEntry
_stacklm_data.build_shards = build_shards
_stacklm_data.stream_source = stream_source
_stacklm_tokenizer = types.ModuleType("stacklm.tokenizer")
_stacklm_tokenizer.StackTokenizer = StackTokenizer
sys.modules["stacklm"] = _stacklm
sys.modules["stacklm.data"] = _stacklm_data
sys.modules["stacklm.tokenizer"] = _stacklm_tokenizer


# ============================================================================
# Block #0 (line ~65) -- capstone/stacklm/mid/mixture.py
# ============================================================================
"""Weighted source mixtures over the Ch. 14.2 packed shards.

Each source lives in its own shard directory, packed at the sequence length the
sub-phase will train at (`data/mid/<source>_<seq_len>/`). Sampling is per
*sequence*, not per micro-batch, so a single forward pass mixes domains.
"""
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from stacklm.data import PackedMemmapDataset          # Ch. 14.2

# Sub-phase A: the annealing mix. Keys are shard-directory names written by the
# tokenize+pack pass (Ch. 14.2) and recorded in the data manifest.
ANNEAL_MIX = {
    "fineweb_edu":   0.40,
    "cosmopedia_v2": 0.30,
    "starcoder":     0.15,
    "finemath":      0.10,
    "instruct_flav": 0.05,
}

# Sub-phases B and C are defined in the same module; their sources are motivated
# in "Repacking for sub-phase B" and "Move 3" below.
LONGCTX_MIX = {
    "starcoder_repo":   0.35,
    "books_pg19":       0.25,
    "arxiv_proofpile2": 0.15,
    "fineweb_edu_long": 0.15,
    "cosmopedia_v2":    0.10,   # deliberately SHORT: the anti-drift anchor
}
CAPABILITY_MIX = {
    "finemath":         0.30,
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
    document-aware mask consume. All four are threaded into the model: dropping
    `position_ids` is a silent bug (see the training loop below).
    """
    datasets, weights = [], []
    for name, w in mix.items():
        d = PackedMemmapDataset(f"{root}/{name}_{seq_len}")
        # Guard against the single most common mid-training bug: reading shards
        # that were packed at the PRETRAIN length while claiming to train long.
        assert d.seq_len == seq_len, (
            f"{name} shards are packed at {d.seq_len}, not {seq_len}; "
            f"re-run the repack pass (see 'Repacking for sub-phase B')")
        datasets.append(d)
        # Per-sequence probability ∝ source weight, spread evenly inside a source.
        weights.extend([w / len(d)] * len(d))

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


# --- exercise block #0: actually pull a batch out of the mixture loader -----
# num_workers=0 (glue: avoids spawning subprocess workers in the test sandbox;
# the block's own sampling/collation logic is unchanged) and a tiny seq_len so
# the tiny stub shards are exercised end to end.
_gen = build_mixture_loader(ANNEAL_MIX, seq_len=8, micro_bs=2, num_workers=0)
_batch = next(_gen)
assert set(_batch.keys()) == {"input_ids", "position_ids", "seq_ids", "targets"}
assert _batch["input_ids"].shape == (2, 7)         # (micro_bs, seq_len - 1)
assert _batch["position_ids"].shape == (2, 7)
print("[block #0] mixture loader batch:", {k: tuple(v.shape) for k, v in _batch.items()})


# ============================================================================
# Block #1 (line ~160) -- capstone/stacklm/optim/schedule.py
# ============================================================================
import math

def wsd_decay_multiplier(mid_step: int, num_decay_steps: int,
                         final_frac: float = 0.0, shape: str = "1-sqrt") -> float:
    """LR multiplier for the WSD *decay* leg, indexed from the start of mid-training.

    mid_step        : steps taken *since* resuming from ckpt_stable (0-indexed).
    num_decay_steps : total mid-training steps (all three sub-phases together).
    final_frac      : LR floor as a fraction of peak (we use ~0.0; some use 0.1).
    shape           : "1-sqrt" (MiniCPM) or "linear" or "cosine".
    Returns a value in [final_frac, 1.0]; multiply by EACH GROUP's peak LR.
    """
    t = min(mid_step, num_decay_steps) / max(1, num_decay_steps)   # progress in [0, 1]
    if shape == "1-sqrt":
        decayed = 1.0 - math.sqrt(t)          # steep at first, long gentle tail
    elif shape == "linear":
        decayed = 1.0 - t
    elif shape == "cosine":
        decayed = 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        raise ValueError(shape)
    return final_frac + (1.0 - final_frac) * decayed


# --- exercise block #1: reproduce the chapter's own worked sanity check -----
N = 10_000
_expected = {0.0: 1.0000, 0.25: 0.5000, 0.5: 0.2929, 0.75: 0.1340, 1.0: 0.0000}
for frac, exp in _expected.items():
    s = int(frac * N)
    got = wsd_decay_multiplier(s, N)
    assert abs(got - exp) < 5e-4, f"progress {frac:.0%}: got {got:.4f}, expected {exp:.4f}"
print("[block #1] wsd_decay_multiplier matches the chapter's worked values")


# ============================================================================
# Block #3 (line ~257) -- capstone/stacklm/model/rope.py
# ============================================================================
def build_rope_cache(head_dim: int, max_seq: int, theta: float,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables of shape (max_seq, head_dim). Mid-training changes
    only `max_seq` and `theta`; the code path is identical to pretraining."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                                / head_dim))                  # (head_dim/2,)
    t = torch.arange(max_seq, device=device).float()           # positions
    freqs = torch.outer(t, inv_freq)                           # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                    # (max_seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def ntk_rescaled_base(base: float, head_dim: int,
                      old_len: int, new_len: int) -> float:
    """NTK-aware RoPE base rescaling: theta' = theta * s^(d/(d-2))."""
    s = new_len / old_len
    return base * (s ** (head_dim / (head_dim - 2)))


# --- exercise block #3: the chapter's own worked example (d=64, 2048->8192) --
cos, sin = build_rope_cache(head_dim=8, max_seq=16, theta=10000.0)
assert cos.shape == (16, 8) and sin.shape == (16, 8)

new_base = ntk_rescaled_base(10000.0, head_dim=64, old_len=2048, new_len=8192)
assert abs(new_base - 41830) < 50, f"expected ~41,830 per the worked example, got {new_base:.1f}"
print(f"[block #3] rope cache {tuple(cos.shape)}, ntk_rescaled_base -> {new_base:.1f}")


# ============================================================================
# Block #4 (line ~279) -- capstone/stacklm/mid/continue_training.py
# ============================================================================
import torch
# (glue) chapter has `from ..model.rope import ntk_rescaled_base` -- a
# package-relative import that only makes sense inside the real `capstone/`
# package layout. `ntk_rescaled_base` is already a global here (block #3,
# directly above), so the import line is dropped rather than rewritten.


def _unwrap(model):
    """`torch.compile` wraps the module; reach the real one for cfg/buffer surgery."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def extend_context(model, new_seq_len: int, device="cpu") -> float:
    """Rescale RoPE base for a longer context and rebuild the cache. Returns the
    new theta. NoPE layers are unaffected (they never consult the cache).

    Call this ONCE, at the start of sub-phase B. `rebuild_rope` swaps the model's
    single (rope_cos, rope_sin) buffer pair and updates cfg.max_seq_len /
    cfg.rope_theta, so the new geometry is recorded in every later checkpoint.
    """
    m = _unwrap(model)
    cfg = m.cfg
    new_base = ntk_rescaled_base(cfg.rope_theta, cfg.head_dim,
                                 old_len=cfg.max_seq_len, new_len=new_seq_len)
    m.rebuild_rope(new_seq_len, new_base, device=device)
    return new_base


# --- exercise block #4: a tiny model stand-in with cfg + rebuild_rope -------
class _TinyCfg:
    def __init__(self):
        self.rope_theta = 10000.0
        self.head_dim = 8
        self.max_seq_len = 16


class _TinyModel:
    """Minimal stand-in for `Stack100M`: only the surface `extend_context` touches
    (cfg fields + a `rebuild_rope` method that swaps the cos/sin buffers)."""

    def __init__(self):
        self.cfg = _TinyCfg()
        self.rope_cos, self.rope_sin = build_rope_cache(
            self.cfg.head_dim, self.cfg.max_seq_len, self.cfg.rope_theta)

    def rebuild_rope(self, new_seq_len, new_base, device="cpu"):
        self.rope_cos, self.rope_sin = build_rope_cache(
            self.cfg.head_dim, new_seq_len, new_base, device=device)
        self.cfg.max_seq_len = new_seq_len
        self.cfg.rope_theta = new_base


_tiny_model = _TinyModel()
_old_theta = _tiny_model.cfg.rope_theta
_returned_base = extend_context(_tiny_model, new_seq_len=64, device="cpu")
assert _tiny_model.cfg.max_seq_len == 64
assert _tiny_model.cfg.rope_theta == _returned_base != _old_theta
assert _tiny_model.rope_cos.shape == (64, 8)
# `_unwrap` on a plain (uncompiled) model is a no-op:
assert _unwrap(_tiny_model) is _tiny_model
print(f"[block #4] extend_context: theta {_old_theta} -> {_returned_base:.1f}, "
      f"rope_cos {tuple(_tiny_model.rope_cos.shape)}")


# ============================================================================
# Block #5 (line ~331) -- capstone/scripts/repack_long.py
# ============================================================================
"""Build the sub-phase-B shards: seq_len=8192, long documents only.

Run once, between sub-phase A and sub-phase B:
    python capstone/scripts/repack_long.py --out data/mid --seq-len 8192
"""
import argparse
from collections import defaultdict

import numpy as np

from stacklm.data import (DataMixEntry, PackedMemmapDataset, build_shards,
                          stream_source)                       # Ch. 14.2
from stacklm.tokenizer import StackTokenizer                   # Ch. 14.3

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
    path makes the concatenation deterministic (and puts headers near sources,
    which is what a human reading the repo would do).
    """
    by_repo = defaultdict(list)
    for f in files:
        by_repo[f["repo_name"]].append(f)
    for repo, fs in by_repo.items():
        fs.sort(key=lambda f: f["path"])
        body = "".join(sep.format(path=f["path"]) + f["content"] for f in fs)
        yield {"text": body, "source": "starcoder_repo", "repo": repo}


def verify_positions(shard_dir: str, seq_len: int, floor: int = 4096,
                     n_sample: int = 512, seed: int = 0):
    """The check that decides whether sub-phase B is real or theatre.

    Ch. 14.2 does NOT store position ids on disk (`store_positions=False` is the
    default; they are recomputed from `input_ids == bos_id` by
    `segments_from_bos`), so we read them back through the dataset that the
    trainer itself will use -- which also proves the shards are readable and
    packed at the right length. Sampling a few hundred rows is enough: we only
    need ONE window whose document reaches past `floor`.
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


# The sub-phase-B sources, as Ch. 14.2 `DataMixEntry` records (name, hf_path,
# weight, domain). The first three are genuinely long; the fourth is a
# length-FILTERED slice of a pretrain source; the last is the deliberately SHORT
# anti-drift anchor. `weight` is the sub-phase-B mixture weight consumed later
# by `build_mixture_loader`.
LONG_SOURCES = [
    (DataMixEntry("starcoder_repo",   "bigcode/starcoderdata",     0.35, "code"),  True),
    (DataMixEntry("books_pg19",       "deepmind/pg19",             0.25, "web"),   False),
    (DataMixEntry("arxiv_proofpile2", "EleutherAI/proof-pile-2",   0.15, "math"),  False),
    (DataMixEntry("fineweb_edu_long", "HuggingFaceFW/fineweb-edu", 0.15, "web"),   False),
    (DataMixEntry("cosmopedia_v2",    "HuggingFaceTB/cosmopedia",  0.10, "synthetic"), False),
]


def main(out_root: str, seq_len: int, tokenizer_path: str):
    tok = StackTokenizer.load(tokenizer_path)             # Ch. 14.3, vocab 32768
    for entry, repo_level in LONG_SOURCES:
        raw = stream_source(entry)                        # Ch. 14.2 streaming reader
        docs = repo_level_documents(raw) if repo_level else raw
        if entry.name != "cosmopedia_v2":     # the short-form anchor stays unfiltered
            docs = length_filtered(docs, tok)
        out = f"{out_root}/{entry.name}_{seq_len}"
        n = build_shards(docs, tok, out, seq_len=seq_len,
                         tokens_per_shard=100_000_000)
        verify_positions(out, seq_len,
                         floor=4096 if entry.name != "cosmopedia_v2" else 0)
        print(f"{entry.name}: {n} shard(s) -> {out}")


# --- exercise block #5 -------------------------------------------------------
# `main()` itself needs real network streaming (`stream_source`) and a real
# tokenize+pack pass (`build_shards`) -- both SKIP(network): our `stream_source`/
# `build_shards` stubs deliberately raise, so `main()` is defined but NOT called.
# The block's actual re-usable logic -- `length_filtered`, `repo_level_documents`,
# `verify_positions` -- is exercised directly against tiny fixtures instead.
_tok = StackTokenizer.load("artifacts/tokenizer.json")
_docs = [{"text": "a b c d e f g h"}, {"text": "only two"}]
_kept = list(length_filtered(_docs, _tok, min_tokens=5))
assert len(_kept) == 1 and _kept[0]["text"] == "a b c d e f g h"

_files = [
    {"repo_name": "repoA", "path": "b.py", "content": "print(2)\n"},
    {"repo_name": "repoA", "path": "a.py", "content": "print(1)\n"},
    {"repo_name": "repoB", "path": "x.py", "content": "print(3)\n"},
]
_repo_docs = list(repo_level_documents(_files))
assert len(_repo_docs) == 2
_repoA_doc = next(d for d in _repo_docs if d["repo"] == "repoA")
assert _repoA_doc["text"].index("a.py") < _repoA_doc["text"].index("b.py"), \
    "files within a repo must be concatenated in sorted path order"

# verify_positions: a shard "packed at seq_len=16" whose stub rows carry
# position ids up to 14 -- assert it passes with a low floor...
verify_positions("data/mid/starcoder_repo_16", seq_len=16, floor=5, n_sample=5)
# ...and assert it correctly TRIPS the guard when the floor is unreachable
# (the exact failure mode `verify_positions` exists to catch).
try:
    verify_positions("data/mid/starcoder_repo_16", seq_len=16, floor=4096, n_sample=5)
    raise AssertionError("verify_positions should have raised for an unreachable floor")
except AssertionError as e:
    assert "contains no document longer than" in str(e)
print("[block #5] length_filtered / repo_level_documents / verify_positions all exercised")


# ============================================================================
# Block #6 (line ~465) -- capstone/stacklm/model/transformer.py (_build_mask)
# ============================================================================
# capstone/stacklm/model/transformer.py  (Ch. 14.4 -- training path of `_build_mask`)
def _build_mask(self, seq_ids, T, kv_len, start_pos, device):
    """Bool mask (B, 1, T, kv_len); True = attend. None = plain-causal fast path."""
    if seq_ids is None and kv_len == T and start_pos == 0:
        return None                                     # SDPA's is_causal=True
    q_pos = torch.arange(start_pos, start_pos + T, device=device)
    kv_pos = torch.arange(kv_len, device=device)
    m = (q_pos[:, None] >= kv_pos[None, :])[None, None]          # (1, 1, T, kv_len)
    if seq_ids is not None:                                       # no cross-document
        same = seq_ids[:, -T:, None] == seq_ids[:, None, :kv_len] # (B, T, kv_len)
        m = m & same[:, None]
    return m


# --- exercise block #6 -------------------------------------------------------
# `_build_mask` is a method (`self` unused in the body) -- call it the same way
# `Stack100M.forward` does, just with a dummy `self`.
_fast_path = _build_mask(None, None, 4, 4, 0, "cpu")
assert _fast_path is None, "plain-causal fast path should return None (is_causal=True)"

_seq_ids = torch.tensor([[0, 0, 1, 1]])                # two documents, 2 tokens each
_mask = _build_mask(None, _seq_ids, T=4, kv_len=4, start_pos=0, device="cpu")
assert _mask.shape == (1, 1, 4, 4) and _mask.dtype == torch.bool
assert bool(_mask[0, 0, 0, 0]) is True                  # self-attend
assert bool(_mask[0, 0, 2, 0]) is False                 # doc 1 can't see doc 0
assert bool(_mask[0, 0, 2, 2]) is True                  # same doc, causal
assert bool(_mask[0, 0, 3, 2]) is True                  # same doc, causal, past position
print("[block #6] _build_mask: fast path + document-aware mask both correct")


# ============================================================================
# Block #7 (line ~486) -- capstone/stacklm/model/doc_attention.py (FlexAttention)
# ============================================================================
# capstone/stacklm/model/doc_attention.py
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Compile once at import; `dynamic=False` keeps one specialization per shape.
flex_attention = torch.compile(flex_attention, dynamic=False)
_create_block_mask = torch.compile(create_block_mask, dynamic=False)


def document_block_mask(seq_ids: torch.Tensor):
    """Block-sparse causal + intra-document mask from Ch. 14.2's `seq_ids`.

    seq_ids : (B, T) int tensor; tokens of the same packed document share an id.
    Returns a BlockMask consumable by flex_attention. Broadcast over heads (H=None).
    """
    B, T = seq_ids.shape

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx                       # no peeking ahead
        same_doc = seq_ids[b, q_idx] == seq_ids[b, kv_idx]
        return causal & same_doc

    return _create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T,
                              device=seq_ids.device)


def doc_attention(q, k, v, block_mask):
    """q, k, v: (B, n_heads, T, head_dim) -- GQA heads already expanded (Ch. 14.4).
    Scaling is 1/sqrt(head_dim) by default, matching SDPA."""
    return flex_attention(q, k, v, block_mask=block_mask)


# --- exercise block #7 -------------------------------------------------------
# Tiny shapes: FlexAttention's block-sparse kernel operates on 128-token blocks
# internally regardless of T, so this is genuinely the same code path as T=8192,
# just cheap enough to compile on CPU within the test budget.
torch.manual_seed(0)
_B, _T, _n_heads, _head_dim = 1, 32, 2, 8
_seq_ids2 = torch.zeros(_B, _T, dtype=torch.int64)
_seq_ids2[:, 16:] = 1                                   # two documents inside the batch
_block_mask = document_block_mask(_seq_ids2)
_q = torch.randn(_B, _n_heads, _T, _head_dim)
_k = torch.randn(_B, _n_heads, _T, _head_dim)
_v = torch.randn(_B, _n_heads, _T, _head_dim)
_out = doc_attention(_q, _k, _v, _block_mask)
assert _out.shape == (_B, _n_heads, _T, _head_dim)
assert torch.isfinite(_out).all()
print(f"[block #7] FlexAttention document mask -> output {tuple(_out.shape)}")


# ============================================================================
# Block #11 (line ~1040) -- Exercise 6 solution: bin_loss_by_position
# ============================================================================
import torch

def bin_loss_by_position(per_token_loss: torch.Tensor, num_bins: int = 16) -> torch.Tensor:
    """Mean next-token loss binned by position within the sequence.

    per_token_loss : (B, T) tensor of per-position NLL (targets already shifted),
                     e.g. F.cross_entropy(logits, y, reduction="none") reshaped to (B, T).
    Returns         : (num_bins,) tensor; entry j = mean loss over positions in bin j.
    Bin j spans positions [j*T/num_bins, (j+1)*T/num_bins). Average over the batch
    first, then scatter-add into bins so every position contributes equally.
    """
    B, T = per_token_loss.shape
    per_pos = per_token_loss.mean(dim=0)                       # (T,) average over batch
    pos = torch.arange(T, device=per_token_loss.device)
    bin_idx = (pos * num_bins) // T                            # 0 .. num_bins-1
    sums = torch.zeros(num_bins, device=per_token_loss.device)
    counts = torch.zeros(num_bins, device=per_token_loss.device)
    sums.index_add_(0, bin_idx, per_pos)
    counts.index_add_(0, bin_idx, torch.ones_like(per_pos))
    return sums / counts.clamp(min=1)                          # (num_bins,)


# --- exercise block #11 ------------------------------------------------------
torch.manual_seed(0)
_B2, _T2, _num_bins = 4, 32, 4
# Construct per-token loss that rises linearly with position so the binned
# output must be monotonically non-decreasing -- a real, checkable property.
_per_pos_loss = torch.linspace(0.5, 2.5, _T2)
_per_token_loss = _per_pos_loss.unsqueeze(0).expand(_B2, _T2).contiguous()
_binned = bin_loss_by_position(_per_token_loss, num_bins=_num_bins)
assert _binned.shape == (_num_bins,)
assert torch.all(_binned[1:] >= _binned[:-1] - 1e-6), "binned loss should be non-decreasing"
assert abs(float(_binned[0]) - float(_per_pos_loss[:8].mean())) < 1e-5
print(f"[block #11] bin_loss_by_position -> {_binned.tolist()}")

print("\nAll tested blocks executed successfully.")
