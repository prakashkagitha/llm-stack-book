# %% [markdown]
# # Data-Parallel Training Across 2 GPUs (DDP) & Scaling Efficiency
#
# > **Hardware:** 2x A100. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will train a small GPT-style transformer with real `DistributedDataParallel` on 2 GPUs via `torchrun`, measure single-GPU vs. 2-GPU tokens/sec with `torch.cuda.Event` timing, and see why the observed speedup lands below 2x.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/05-distributed-data-parallel.html) for the full explanation.

# %% [markdown]
# ## Dependencies
#
# There is nothing to `pip install` for this notebook: `torch.distributed`,
# `torch.nn`, and `torch.utils.data` all ship with PyTorch itself, and
# attention uses PyTorch's built-in fused SDPA kernel (no separate `flash-attn`
# wheel needed). You only need a recent CUDA-enabled PyTorch build. Uncomment
# the line below if you are starting from a bare environment.

# %%
# %pip install --quiet "torch>=2.1"  # only if torch is not already installed

# %% [markdown]
# ## GPU sanity check and global setup

# %%
import os

import torch

assert torch.cuda.is_available(), "This notebook needs at least 1 (ideally 2) CUDA GPUs."
n_gpus = torch.cuda.device_count()
print(f"CUDA devices visible: {n_gpus}")
for i in range(n_gpus):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}, "
          f"bf16 supported: {torch.cuda.is_bf16_supported()}")
if n_gpus < 2:
    print("WARNING: fewer than 2 GPUs visible. The torchrun cell below is written "
          "for --nproc_per_node=2 (2x A100); change that flag to match what you have.")

# -- Global setup: seed, TF32 matmuls (A100 tensor cores), and the device we'll
#    use for the single-GPU baseline later in this notebook. --------------------
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda:0")
BF16 = torch.bfloat16

# %% [markdown]
# ## The model and the synthetic benchmark data
#
# We need a model large enough to be **compute-bound** (so throughput actually
# reflects GPU utilization, not Python/launch overhead) but small enough to
# train in a few minutes. We build a compact decoder-only transformer
# (~8 layers, 768-wide, nanoGPT-style: fused SDPA attention, weight-tied
# embedding/head) and a **synthetic** random-token dataset — this is a
# throughput benchmark, not a quality run, so the labels being random is fine
# and is standard practice for scaling measurements (several training stacks
# ship a "mock data" mode for exactly this purpose).
#
# We write this shared model/data code to `model_def.py` on disk with
# `%%writefile` because the DDP script launched by `torchrun` below runs in a
# **separate process** and needs to `import` it — a notebook cell cannot be
# imported directly by a subprocess.

# %%
%%writefile model_def.py
"""
model_def.py -- a compact decoder-only transformer (GPT-style) and a synthetic
token dataset, shared by the single-GPU baseline (run inline in the notebook)
and the DDP training script (run via torchrun as a separate process). Kept
small (tens of millions of parameters) so the full experiment finishes in a
few minutes on 2x A100, while still being large enough to be compute-bound.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class GPTConfig:
    vocab_size: int = 50304   # GPT-2 vocab, padded to a multiple of 64 for tensor-core efficiency
    block_size: int = 512     # sequence length (context window)
    n_layer: int = 8
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (nn.LayerNorm doesn't let you cleanly
    drop the bias term on every torch version, so we do it by hand)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch's fused
    scaled_dot_product_attention, which dispatches to the Flash-Attention
    (or memory-efficient) backend automatically on A100 -- no separate
    flash-attn install required."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # applies the causal mask inside the fused kernel
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Weight tying AFTER init so the (identical std=0.02) init isn't applied
        # twice to the same shared tensor; both modules now point at head.weight.
        self.tok_emb.weight = self.head.weight  # weight tying (standard GPT-2 trick)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self):
        # Deduplicate the tied embedding/head weight so we don't double-count it.
        seen = {}
        for p in self.parameters():
            seen[id(p)] = p.numel()
        return sum(seen.values())


class RandomTokenDataset(Dataset):
    """Synthetic next-token-prediction dataset: fixed, seeded random token ids.

    This is a THROUGHPUT benchmark, not a quality run -- labels are random, so
    the loss will hover near ln(vocab_size) and won't meaningfully decrease.
    Swap in a real tokenized corpus (see ch. 3.1-3.2) to turn this into an
    actual pretraining job; the DDP mechanics below are identical either way.
    """

    def __init__(self, n_samples: int, block_size: int, vocab_size: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        # +1 token so each row can be sliced into input = tok[:-1], target = tok[1:].
        self.data = torch.randint(0, vocab_size, (n_samples, block_size + 1), generator=g)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

# %% [markdown]
# ## Single-GPU baseline: measuring tokens/sec correctly
#
# GPU work is asynchronous, so `time.time()` around a training loop mostly
# measures Python overhead and kernel-launch latency, not GPU compute. The
# correct tool is `torch.cuda.Event(enable_timing=True)`, bracketing the
# region with a `torch.cuda.synchronize()` before we start the clock and
# another before we read it back. We also run an untimed **warmup** (lets
# `scaled_dot_product_attention`/cuDNN pick kernels and the caching allocator
# settle) before the timed region, and reset peak-memory stats so
# `torch.cuda.max_memory_allocated()` reflects only the steady-state loop.
#
# **Expected result:** this single-GPU number is your scaling denominator —
# everything below compares the 2-GPU aggregate throughput against
# `2 x this number`.

# %%
from model_def import GPT, GPTConfig, RandomTokenDataset  # noqa: E402  (written just above)


def run_single_gpu_baseline(steps: int = 50, warmup: int = 10,
                            batch_size: int = 16, block_size: int = 512):
    """Train GPT on one GPU (cuda:0) and report steady-state tokens/sec."""
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=block_size)
    model = GPT(cfg).to(DEVICE)
    print(f"model parameters: {model.num_parameters() / 1e6:.1f}M")

    ds = RandomTokenDataset(n_samples=4096, block_size=block_size, vocab_size=cfg.vocab_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    def cycle():
        while True:
            for x, y in loader:
                yield x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

    data_iter = cycle()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    # -- Warmup (not timed): autotune kernels, warm the allocator. -----------
    for _ in range(warmup):
        x, y = next(data_iter)
        with torch.autocast(device_type="cuda", dtype=BF16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # -- Timed region: torch.cuda.Event, not time.time(). --------------------
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(steps):
        x, y = next(data_iter)
        with torch.autocast(device_type="cuda", dtype=BF16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    end_evt.record()
    torch.cuda.synchronize()  # must sync before reading elapsed_time

    elapsed_s = start_evt.elapsed_time(end_evt) / 1000.0  # ms -> s
    tokens = steps * batch_size * block_size
    tokens_per_sec = tokens / elapsed_s
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"[1-GPU baseline] {elapsed_s:.2f}s for {steps} steps | "
          f"~{tokens_per_sec:,.0f} tokens/s | peak mem {peak_mem_gb:.2f} GB")
    return tokens_per_sec


BATCH_SIZE, BLOCK_SIZE, STEPS, WARMUP = 16, 512, 50, 10
baseline_tokens_per_sec = run_single_gpu_baseline(
    steps=STEPS, warmup=WARMUP, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE
)

# The baseline model is now out of scope, but PyTorch's caching allocator still
# holds its blocks reserved on cuda:0 in THIS process. Release them so the
# torchrun subprocess below (whose rank 0 also runs on cuda:0) gets the full GPU.
torch.cuda.empty_cache()

# %% [markdown]
# ## The DDP training script
#
# A notebook process cannot call `dist.init_process_group` and spawn a
# 2-GPU NCCL job inline — `torchrun` needs to fork **separate OS processes**,
# one per GPU, each with its own `RANK`/`LOCAL_RANK`/`WORLD_SIZE` environment
# variables. The standard idiom (used throughout the chapter) is: write a
# self-contained script with `%%writefile`, then launch it with
# `!torchrun --nproc_per_node=N script.py`.
#
# This script implements everything the chapter describes for the
# happy-path DDP loop:
#
# - `dist.init_process_group(backend="nccl")` — NCCL is the GPU-to-GPU collective backend.
# - **Identical initialization**: the same seed on every rank before constructing the model (and DDP additionally broadcasts rank-0's parameters at wrap time, so replicas start bit-identical either way).
# - `DistributedSampler` — shards the dataset so each rank sees a disjoint slice; `set_epoch()` reshuffles deterministically per epoch across ranks.
# - `DDP(model, device_ids=[local_rank], bucket_cap_mb=...)` — wraps the model; internally this registers the autograd hooks that bucket gradients and all-reduce each bucket asynchronously as soon as it's full, overlapping communication with the rest of backward (see the chapter's from-scratch `TinyDDP` for exactly how).
# - **Per-rank logging** plus a **loss `all_reduce`** — DDP already all-reduces every *gradient* inside `loss.backward()`; the extra `all_reduce` on the loss *value* here is purely for a globally-meaningful printed number, not part of the optimization.
# - Global vs. per-GPU batch size: `--batch-size` is the **per-GPU (local)** batch; the **global** batch is `batch_size * world_size` — going from 1 to 2 GPUs at fixed local batch doubles the global batch and the effective tokens/step, which is exactly the throughput-vs-optimization tradeoff the chapter flags (you may need to retune LR/warmup at larger world sizes).

# %%
%%writefile ddp_train.py
"""
ddp_train.py -- real multi-GPU DDP training.
Run with:
    torchrun --standalone --nproc_per_node=2 ddp_train.py

Measures steady-state tokens/sec across the DDP group and prints both a
per-rank number and the group-aggregate number.
"""
import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model_def import GPT, GPTConfig, RandomTokenDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16,
                        help="PER-GPU (local) batch size; global batch = this * world_size")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50, help="measured steps, after warmup")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--bucket-cap-mb", type=int, default=25,
                        help="DDP gradient bucket size in MB (PyTorch default is 25)")
    args = parser.parse_args()

    # -- Process group + device setup. torchrun sets these env vars for us. --
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -- Invariant 1 (identical initialization): same seed on every rank BEFORE
    #    constructing the model, so every replica starts bit-identical. -------
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=args.block_size)
    model = GPT(cfg).to(device)

    # -- DDP wrap: registers per-parameter autograd hooks that bucket
    #    gradients (bucket_cap_mb) and all-reduce+average each full bucket
    #    asynchronously, overlapping communication with backward compute. ----
    ddp_model = DDP(model, device_ids=[local_rank], bucket_cap_mb=args.bucket_cap_mb)

    # -- Data: DistributedSampler gives each rank a disjoint shard. Every rank
    #    builds the SAME dataset (seed 0), and the sampler splits it by rank. -
    ds = RandomTokenDataset(n_samples=8192, block_size=args.block_size, vocab_size=cfg.vocab_size)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        drop_last=True, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    def batches():
        epoch = 0
        while True:
            sampler.set_epoch(epoch)  # reshuffle deterministically, differently per epoch
            for x, y in loader:
                yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            epoch += 1

    data_iter = batches()

    if rank == 0:
        n_params = model.num_parameters()
        print(f"[rank0] model parameters: {n_params / 1e6:.1f}M | world_size={world_size} | "
              f"per-GPU batch={args.batch_size} | GLOBAL batch={args.batch_size * world_size}")

    # -- Warmup (not timed): autotune kernels, let NCCL rings spin up. -------
    for _ in range(args.warmup):
        x, y = next(data_iter)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = ddp_model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # DDP's bucket hooks fire here; grad all-reduce overlaps backward compute
        opt.step()
    torch.cuda.synchronize()
    dist.barrier()  # make sure every rank starts the timed region together
    torch.cuda.reset_peak_memory_stats()

    # -- Timed region: torch.cuda.Event, not time.time(). --------------------
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    last_loss = None
    for _ in range(args.steps):
        x, y = next(data_iter)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = ddp_model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = loss.detach()
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_s = start_evt.elapsed_time(end_evt) / 1000.0

    # -- All-reduce the last loss VALUE for a globally-meaningful log line.
    #    (Distinct from the per-parameter gradient all-reduce DDP already did
    #    inside loss.backward() above -- this one is for humans, not the optimizer.)
    loss_for_log = last_loss.clone()
    dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)

    local_tokens = args.steps * args.batch_size * args.block_size
    local_tokens_per_sec = local_tokens / elapsed_s
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"[rank{rank}] {elapsed_s:.2f}s for {args.steps} steps | "
          f"local ~{local_tokens_per_sec:,.0f} tok/s | peak mem {peak_mem_gb:.2f} GB | "
          f"avg loss (all-reduced) = {loss_for_log.item():.4f}")

    # -- Group-aggregate throughput: sum tokens processed by every rank,
    #    divide by the SLOWEST rank's wall time (max, not mean) -- the job
    #    isn't done until every rank finishes its share. -----------------
    tokens_t = torch.tensor([float(local_tokens)], device=device)
    time_t = torch.tensor([elapsed_s], device=device)
    dist.all_reduce(tokens_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(time_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        agg_tokens_per_sec = tokens_t.item() / time_t.item()
        print(f"[rank0] AGGREGATE {world_size}-GPU throughput: ~{agg_tokens_per_sec:,.0f} tokens/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Launching the 2-GPU job
#
# `--standalone` tells `torchrun` to manage rendezvous itself for a
# single-node job (no need to set `MASTER_ADDR`/`MASTER_PORT` by hand).
# `--nproc_per_node=2` spawns one process per GPU. Watch for one line per
# rank plus a final `AGGREGATE` line from rank 0. The `{BATCH_SIZE}` etc. below
# are IPython shell-variable expansions of the Python values set earlier.

# %%
!torchrun --standalone --nproc_per_node=2 ddp_train.py \
    --batch-size {BATCH_SIZE} --block-size {BLOCK_SIZE} --steps {STEPS} --warmup {WARMUP}

# %% [markdown]
# ## Computing scaling efficiency
#
# **Speedup** = (2-GPU aggregate tokens/s) / (1-GPU tokens/s). **Perfect**
# linear scaling would give speedup = 2.0 (efficiency = 100%). Real DDP lands
# below that because of exactly the mechanism the chapter walks through: the
# gradient all-reduce is *mostly* overlapped with backward compute via
# bucketing, but the **last bucket's** all-reduce has no remaining backward
# compute left to hide behind — that exposed communication, plus
# `DistributedSampler`/`DataLoader` overhead the single-process run doesn't
# pay, is where the gap from 2.0x comes from.

# %%
def scaling_efficiency(baseline_1gpu_tokens_per_sec: float,
                       aggregate_ngpu_tokens_per_sec: float,
                       n_gpus: int):
    speedup = aggregate_ngpu_tokens_per_sec / baseline_1gpu_tokens_per_sec
    efficiency = speedup / n_gpus
    print(f"1-GPU:  ~{baseline_1gpu_tokens_per_sec:,.0f} tok/s")
    print(f"{n_gpus}-GPU: ~{aggregate_ngpu_tokens_per_sec:,.0f} tok/s (aggregate)")
    print(f"speedup = {speedup:.2f}x  |  scaling efficiency = {efficiency * 100:.0f}%")
    return speedup, efficiency


# Paste the "[rank0] AGGREGATE 2-GPU throughput: ~... tokens/s" number printed
# by the torchrun cell above:
ddp_aggregate_tokens_per_sec = None  # <- replace None with that number, e.g. 3.4e4

if ddp_aggregate_tokens_per_sec is not None:
    scaling_efficiency(baseline_tokens_per_sec, ddp_aggregate_tokens_per_sec, n_gpus=2)
else:
    print("Run the torchrun cell above first, then paste its printed AGGREGATE "
          "tokens/s value into ddp_aggregate_tokens_per_sec and re-run this cell.")

# %% [markdown]
# ## What you should see
#
# - The 1-GPU baseline should print a steady-state tokens/sec figure once
#   warmup has settled the allocator and SDPA kernel selection.
# - The 2-GPU `torchrun` run should print one `[rank0]`/`[rank1]` line each
#   (their local tokens/s and peak memory should be close to each other and
#   close to the 1-GPU number, since each rank does the same amount of local
#   work) plus one `AGGREGATE` line from rank 0.
# - On 2x A100 with NVLink (or a fast PCIe/NVSwitch interconnect), expect the
#   aggregate throughput to land on the order of **roughly 1.7-1.9x** the
#   1-GPU baseline, not 2.0x — the gap is the exposed final-bucket all-reduce
#   plus data-loading/sampler overhead, not a bug. A much lower ratio
#   (say, under 1.3x) usually points to a slow interconnect, buckets sized
#   too small (too many tiny all-reduces) or too large (poor overlap), or a
#   CPU/data-loading bottleneck rather than a communication one. Treat these
#   as ballpark expectations, not guarantees — your exact numbers depend on
#   the interconnect, driver, and PyTorch/NCCL versions.
#
# **Key takeaways:**
# 1. Multi-GPU training in a notebook always goes through `%%writefile` + `!torchrun` — a notebook process cannot itself hold multiple ranks of a NCCL process group.
# 2. Measure GPU throughput with `torch.cuda.Event` + `synchronize()`, always after a warmup phase and with peak memory read via `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()` — wall-clock `time.time()` around async CUDA calls is not trustworthy.
# 3. DDP's gradient bucketing overlaps all but the last bucket's all-reduce with backward compute, which is exactly why observed scaling is high but not perfect.
# 4. Distinguish **per-GPU (local)** batch size from **global** batch size (`local * world_size`) — this notebook holds local batch fixed across the 1-GPU and 2-GPU runs, so the 2-GPU run's global batch (and therefore its optimization dynamics, not just its throughput) is doubled.
#
# **Next step:** once DDP's replicate-everything memory cost becomes the bottleneck (bigger models, not more throughput), move to sharded data parallelism — see [ZeRO & FSDP in the same chapter](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/05-distributed-data-parallel.html#zero-sharding-the-redundancy-away) for the from-scratch ZeRO-1 optimizer-state sharding and the full FSDP2 training script.
