# %% [markdown]
# # Sharding a Model with FSDP (ZeRO) Across 2 GPUs
#
# > **Hardware:** 2x A100. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will wrap a real transformer in PyTorch FSDP, run it under three sharding
# strategies that map directly onto DeepSpeed's ZeRO-1/2/3 stages, and measure
# the per-GPU memory each one actually uses.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/07-megatron-deepspeed.html) for the full explanation.

# %%
# No extra packages needed: FSDP, mixed precision, and distributed checkpointing
# all live in the torch.distributed / torch.distributed.fsdp / torch.distributed.checkpoint
# modules that ship inside torch itself (torch is preinstalled on the target image).
# Uncomment only if your image ships an older torch than this notebook needs
# (torch>=2.2, for the current `dcp.save` signature and FSDP StateDictType API):
# %pip install -q -U "torch>=2.2"

import os
import subprocess

import torch

print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
n_gpus = torch.cuda.device_count()
print(f"visible GPUs: {n_gpus}")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {props.name}  ({props.total_memory / 1e9:.1f} GB)")

assert n_gpus >= 2, (
    "This notebook needs 2 GPUs. If you have 2 but only 1 shows up, check "
    "CUDA_VISIBLE_DEVICES."
)
assert torch.cuda.is_bf16_supported(), (
    "bf16 mixed precision needs an Ampere-or-newer GPU; A100 qualifies."
)

torch.manual_seed(0)

# %% [markdown]
# ## Plan
#
# A notebook process can't `torch.distributed.init_process_group` a 2-rank
# NCCL job inline the way it can spin up a single CUDA context — `torchrun`
# has to be the thing that forks the 2 ranks, sets `RANK`/`LOCAL_RANK`/
# `WORLD_SIZE`, and gives each rank its own GPU. So the pattern for the rest
# of this notebook is: **write a real `.py` script with `%%writefile`, then
# launch it with `!torchrun --nproc_per_node=2 <script>.py`** — exactly what
# you'd do outside a notebook, just driven from a cell.
#
# We build one toy but real GPT-style decoder (~0.7B parameters — big enough
# that its optimizer state is a few GB, small enough to comfortably fit
# *unsharded* on a single A100 so we can measure that baseline directly) and
# run the identical training step under three `ShardingStrategy` settings,
# **all launched with the same `--nproc_per_node=2`** so the only thing that
# changes between runs is how much of the model state each of the 2 GPUs
# actually holds:
#
# | FSDP `ShardingStrategy` | What's sharded across the 2 ranks | ZeRO equivalent |
# |---|---|---|
# | `NO_SHARD` | nothing — full params+grads+optimizer state replicated on every GPU | no ZeRO (plain DDP) |
# | `SHARD_GRAD_OP` | gradients + optimizer state; full params kept resident during compute | ZeRO-2 |
# | `FULL_SHARD` | params + gradients + optimizer state, all reconstructed just-in-time | ZeRO-3 |
#
# `NO_SHARD` is our baseline: it's what plain DDP does, and it's exactly the
# case FSDP/ZeRO was built to fix once a model's optimizer state no longer
# fits on one GPU.

# %%
%%writefile fsdp_model.py
"""Toy GPT-style decoder-only transformer (~0.7B params with the defaults
used below), sized to make the FSDP sharding effect visible without needing
a model that's actually too big for one A100. Uses
torch.nn.functional.scaled_dot_product_attention, which dispatches to a
fused, flash-attention-style kernel on Ampere+ GPUs -- no separate
flash-attn package required.

Absolute position embeddings + full (non-grouped) attention are used purely
for simplicity; nothing here depends on that choice -- FSDP wraps whatever
`nn.Module` structure you give it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # is_causal=True routes this through a fused causal-attention kernel.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int = 4):
        super().__init__()
        inner = mlp_ratio * hidden_size
        self.fc_in = nn.Linear(hidden_size, inner, bias=False)
        self.fc_out = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x)))


class TransformerBlock(nn.Module):
    """One decoder block. This is the *unit* FSDP's auto-wrap policy wraps:
    each instance becomes its own FSDP unit, so this block's params/grads are
    all-gathered / reduce-scattered independently of every other block. That
    per-block granularity is what lets FSDP prefetch the *next* block's
    parameter all-gather while the *current* block is still computing.
    """

    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, n_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ToyGPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_size, n_heads) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# %% [markdown]
# ## The distributed training + memory-measurement script
#
# This script is launched fresh (via `torchrun`) once per sharding strategy.
# Every rank:
#
# 1. Joins the NCCL process group and pins itself to its local GPU.
# 2. Builds the identical model (same seed, so both ranks start with the
#    same weights *before* FSDP shards them).
# 3. Wraps it in FSDP with an **auto-wrap policy on `TransformerBlock`**, a
#    **`MixedPrecision` policy** (bf16 compute, fp32-owned parameter shard —
#    the standard FSDP mixed-precision recipe), and the sharding strategy
#    under test.
# 4. Runs a few warmup steps (let cuDNN/SDPA autotune and let FSDP allocate
#    its communication buffers), resets the peak-memory counter, then times
#    the *measured* steps with `torch.cuda.Event` and reads
#    `torch.cuda.max_memory_allocated`.
# 5. Optionally saves a **sharded checkpoint** — each rank writes only the
#    parameter shard it owns, so checkpointing stays cheap even when the
#    full model doesn't fit in one rank's memory.

# %%
%%writefile fsdp_zero_demo.py
"""Run the identical training step under NO_SHARD / SHARD_GRAD_OP / FULL_SHARD
on 2 GPUs, and report per-rank peak memory + step time.

    torchrun --standalone --nproc_per_node=2 fsdp_zero_demo.py --strategy full_shard
"""
import argparse
import functools
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)

try:
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except ImportError:  # pragma: no cover - path moved across some torch releases
    from torch.distributed.fsdp import wrap as _wrap
    transformer_auto_wrap_policy = _wrap.transformer_auto_wrap_policy

import torch.distributed.checkpoint as dcp

from fsdp_model import ToyGPT, TransformerBlock, num_params

STRATEGY_MAP = {
    "no_shard": ShardingStrategy.NO_SHARD,          # no ZeRO / plain-DDP-equivalent
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ~ ZeRO-2
    "full_shard": ShardingStrategy.FULL_SHARD,        # ~ ZeRO-3
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=list(STRATEGY_MAP), default="full_shard")
    p.add_argument("--hidden-size", type=int, default=1536)
    p.add_argument("--n-layers", type=int, default=24)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--measured-steps", type=int, default=5)
    p.add_argument("--save-ckpt", action="store_true")
    args = p.parse_args()

    # torchrun sets these env vars for every rank it launches.
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    torch.manual_seed(0)  # identical init on every rank before FSDP shards it
    model = ToyGPT(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    )
    n_params = num_params(model)

    # bf16 for the all-gathered compute copy; FSDP keeps each rank's *owned*
    # parameter shard in fp32 underneath for a numerically stable optimizer
    # step -- this is the standard FSDP mixed-precision recipe.
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap every TransformerBlock as its own FSDP unit (see markdown above).
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=STRATEGY_MAP[args.strategy],
        device_id=torch.cuda.current_device(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Dummy next-token-prediction batch: random ids are enough to exercise
    # real forward/backward/optimizer-step memory and communication; we only
    # care about system behavior here, not what the model learns.
    input_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.seq_len), device=local_rank
    )
    targets = torch.randint(
        0, args.vocab_size, (args.batch_size, args.seq_len), device=local_rank
    )

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, args.vocab_size), targets.view(-1)
        )
        loss.backward()  # triggers reduce-scatter of grads (FULL_SHARD/SHARD_GRAD_OP)
        optimizer.step()  # each rank updates only the shard of state it owns
        return loss

    # Warmup: lets SDPA/cuDNN autotune kernels and lets FSDP allocate its
    # persistent communication buffers, so the *measured* steps reflect
    # steady-state memory/time, not one-time setup cost.
    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(local_rank)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(args.measured_steps):
        loss = train_step()
    end_evt.record()
    torch.cuda.synchronize()

    ms_per_step = start_evt.elapsed_time(end_evt) / args.measured_steps
    peak_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9

    if rank == 0:
        print(f"[params] {n_params / 1e9:.3f}B total (tied embedding/head)")
    print(
        f"[rank {rank}] strategy={args.strategy:<14} "
        f"peak_mem={peak_gb:6.2f} GB  step_time={ms_per_step:7.1f} ms  "
        f"loss={loss.item():.3f}"
    )

    if args.save_ckpt:
        ckpt_dir = f"fsdp_ckpt_{args.strategy}"
        # SHARDED_STATE_DICT: each rank writes only the parameter shard it
        # owns to `ckpt_dir` -- no all-gather to rank 0 first, so saving
        # stays cheap even for models much larger than one GPU's memory.
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=False),
        ):
            sharded_state = {"model": model.state_dict()}
            # storage_writer= is supported on every torch>=2.2; the newer
            # checkpoint_id= shorthand only landed later, so we use the writer
            # form to stay portable across images.
            dcp.save(sharded_state, storage_writer=dcp.FileSystemWriter(ckpt_dir))
        if rank == 0:
            print(f"[ckpt] sharded checkpoint written to {ckpt_dir}/")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Baseline: `NO_SHARD` (no ZeRO — what plain DDP gives you)
#
# Every rank keeps the *entire* model's parameters, gradients, and Adam
# optimizer state (first + second moment) resident, all in fp32 under the
# hood. With ~0.7B parameters that's on the order of 16 bytes/parameter for
# model state alone (4 fp32 params + 4 fp32 grads + 8 fp32 Adam moments) —
# roughly a low-double-digit number of GB before activations, comfortably
# inside 80 GB, which is exactly why we picked this model size: it lets us
# measure the *unsharded* case directly instead of just asserting it would
# OOM.

# %%
result_no_shard = subprocess.run(
    [
        "torchrun", "--standalone", "--nproc_per_node=2",
        "fsdp_zero_demo.py", "--strategy", "no_shard",
    ],
    capture_output=True, text=True,
)
print(result_no_shard.stdout)
print(result_no_shard.stderr[-2000:])  # torchrun/NCCL logs go to stderr
if result_no_shard.returncode != 0:
    print(f"[warning] torchrun exited with code {result_no_shard.returncode}")

# %% [markdown]
# ## `FULL_SHARD` (ZeRO-3): parameters, gradients, and optimizer state all sharded
#
# Every forward pass, each `TransformerBlock`'s FSDP unit **all-gathers**
# its own full bf16 parameters just before it's used, runs the block, then
# **frees** the shard it doesn't own — so at any instant a rank only holds
# one block's full parameters, not the whole model's. In the backward pass,
# gradients are **reduce-scattered**: each rank ends up owning the (already
# reduced) gradient for only its 1/world_size slice of every parameter, and
# the optimizer step then only touches that owned slice.
#
# With `world_size=2`, expect the *model-state* portion of peak memory
# (params + grads + optimizer state) to drop to roughly half of the
# `NO_SHARD` run's model-state memory — activation memory isn't sharded by
# data-parallel FSDP, so don't expect a clean 2x on *total* peak memory, just
# on the model-state component. At real training scale (a large data-parallel
# degree — many more than 2 ranks; see Rajbhandari et al., "ZeRO: Memory
# Optimizations Toward Training Trillion Parameter Models") that same 1/N
# scaling is what produces the much larger, order-of-magnitude reductions
# associated with ZeRO-3.

# %%
result_full_shard = subprocess.run(
    [
        "torchrun", "--standalone", "--nproc_per_node=2",
        "fsdp_zero_demo.py", "--strategy", "full_shard", "--save-ckpt",
    ],
    capture_output=True, text=True,
)
print(result_full_shard.stdout)
print(result_full_shard.stderr[-2000:])
if result_full_shard.returncode != 0:
    print(f"[warning] torchrun exited with code {result_full_shard.returncode}")

# %% [markdown]
# ## `SHARD_GRAD_OP` (ZeRO-2): shard grads + optimizer state, keep full params
#
# This sits between the two runs above: gradients and optimizer state are
# sharded the same way as `FULL_SHARD`, but parameters are kept fully
# materialized on every rank for the whole forward+backward (no per-block
# all-gather/free of *parameters*). Expect its peak memory to land between
# the `NO_SHARD` and `FULL_SHARD` numbers, and — since it skips the
# parameter all-gather on the forward pass — its step time should be at
# least as fast as `FULL_SHARD`, likely closer to `NO_SHARD`.

# %%
result_shard_grad_op = subprocess.run(
    [
        "torchrun", "--standalone", "--nproc_per_node=2",
        "fsdp_zero_demo.py", "--strategy", "shard_grad_op",
    ],
    capture_output=True, text=True,
)
print(result_shard_grad_op.stdout)
print(result_shard_grad_op.stderr[-2000:])
if result_shard_grad_op.returncode != 0:
    print(f"[warning] torchrun exited with code {result_shard_grad_op.returncode}")

# %% [markdown]
# ## Inspecting the sharded checkpoint
#
# The `--save-ckpt` flag on the `full_shard` run wrote a **distributed
# checkpoint**, not a single rank-0 file: each rank contributed only the
# shard it owned, plus a shared `.metadata` file that records how to
# reassemble the full tensors (e.g. for resharding onto a different GPU
# count later). List the directory to confirm there's no single giant
# rank-0 file — that's the point of sharded (vs. `FULL_STATE_DICT`)
# checkpointing at scale.

# %%
ckpt_dir = "fsdp_ckpt_full_shard"
if os.path.isdir(ckpt_dir):
    for name in sorted(os.listdir(ckpt_dir)):
        path = os.path.join(ckpt_dir, name)
        print(f"{os.path.getsize(path):>10} bytes  {name}")
else:
    print(f"{ckpt_dir}/ not found — did the full_shard run above complete?")

# %% [markdown]
# ## What you should see
#
# - **`no_shard`**: both ranks report roughly the same peak memory, on the
#   order of several GB to a low double-digit number of GB — dominated by
#   the full fp32 params + grads + Adam moments for the ~0.7B-parameter
#   model, comfortably under 80 GB.
# - **`full_shard`**: both ranks' peak memory should be noticeably lower
#   than the `no_shard` run — expect the model-state portion to be roughly
#   halved (world_size=2), with a smaller relative drop in *total* peak
#   memory once (unsharded) activations are counted in.
# - **`shard_grad_op`**: peak memory between the two above; step time close
#   to (often faster than) `full_shard` since it skips the per-block
#   parameter all-gather in the forward pass.
# - The checkpoint directory should contain multiple shard files plus a
#   `.metadata` file, not one large rank-0-only file.
#
# Exact numbers depend on your driver/torch/NCCL versions and are for you to
# read off the printed output — don't expect them to match any number
# printed here, because none is fabricated in advance.
#
# **Key takeaways**
#
# 1. FSDP's `ShardingStrategy` is literally the ZeRO stage dial: `NO_SHARD`
#    ≈ no ZeRO (plain DDP), `SHARD_GRAD_OP` ≈ ZeRO-2, `FULL_SHARD` ≈ ZeRO-3.
#    Per-GPU model-state memory scales roughly as `1/world_size` once you
#    shard parameters (`FULL_SHARD`/ZeRO-3).
# 2. The communication pattern that buys that memory back is: **all-gather
#    parameters** just-in-time in the forward pass (and again for the
#    recomputed backward), **reduce-scatter gradients** in the backward
#    pass. More sharding (`FULL_SHARD`) means more communication than less
#    sharding (`SHARD_GRAD_OP`/`NO_SHARD`) — memory and communication trade
#    off against each other.
# 3. `auto_wrap_policy` controls FSDP's *unit* granularity. Wrapping at the
#    transformer-block level (rather than the whole model as one unit) is
#    what lets FSDP prefetch the next block's all-gather while the current
#    block still computes.
# 4. FSDP/ZeRO-3 earns its keep when a model's parameters + gradients +
#    optimizer state don't fit on one GPU even in mixed precision — at that
#    point `NO_SHARD`/DDP simply can't allocate, and sharding across GPUs is
#    the only way to train at all, not just a memory optimization on top of
#    something that already worked.
#
# **Next step:** see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/05-distributed-data-parallel.html)
# for the underlying theory, and [Megatron-LM & DeepSpeed in Practice](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/07-megatron-deepspeed.html)
# for how this composes with tensor/pipeline parallelism and DeepSpeed's
# ZeRO-Offload/ZeRO-Infinity at real cluster scale.
