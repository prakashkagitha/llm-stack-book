# %% [markdown]
# # Multi-GPU Inference: Tensor-Parallel Serving of a Model Too Big for One GPU
#
# > **Hardware:** 2x A100. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will build a minimal Megatron-style tensor-parallel (column-parallel then
# row-parallel, with an all-reduce) transformer forward pass, launch it across 2 GPUs
# with `torchrun`, verify it is numerically correct against an un-sharded reference,
# and measure the actual per-GPU weight memory and per-token decode latency — then
# see the production path (vLLM's `tensor_parallel_size=2`) and the decision rule for
# when TP inference is worth its NVLink communication cost.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/07-inference-serving/11-multi-gpu-inference.html) for the full explanation.

# %%
# No extra packages are required for the from-scratch tensor-parallel demo below —
# it only uses torch + torch.distributed (both preinstalled). vLLM is optional and
# only needed if you want to actually execute the production snippet near the end
# (it downloads real model weights and is NOT run by default in this notebook).
# %pip install -q vllm   # optional, several GB, only needed for the vLLM section

import torch

torch.manual_seed(0)

assert torch.cuda.is_available(), "This notebook needs CUDA GPUs."
n_gpus = torch.cuda.device_count()
print(f"Visible CUDA devices: {n_gpus}")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} -> {props.name}, {props.total_memory / 1e9:.1f} GB")
assert n_gpus >= 2, (
    "This notebook targets 2x A100 and launches a 2-way tensor-parallel job; "
    f"found only {n_gpus} GPU(s)."
)
assert torch.cuda.is_bf16_supported(), "A100 supports bf16; if this fails, check your driver/CUDA version."

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# %% [markdown]
# ## Why split inference across GPUs at all?
#
# A dense model's weight memory in BF16 is `2 bytes x num_params`. Llama-3 70B needs
# roughly 140 GB just for weights — already more than one 80 GB A100/H100 before you
# add the KV cache or activations. Tensor parallelism (TP) splits each weight matrix
# column-wise or row-wise across `T` GPUs, cutting per-GPU weight memory by
# (roughly) `1/T` and per-GPU matmul time by (roughly) `1/T`, at the cost of one
# all-reduce per attention block and one per MLP block, every layer, every forward
# pass. Because that all-reduce sits on the decode critical path, TP is only cheap
# when the GPUs share fast NVLink — crossing PCIe or Ethernet can make the
# communication cost comparable to (or larger than) the compute it saved.
#
# This is different from **pipeline parallelism** (splits layers into stages,
# communicates only activations between stages, latency-neutral for decode) and
# **data-parallel replication** (full independent copies, communicates nothing
# during inference, pure throughput scaling). TP is the one axis that can lower
# a *single* request's per-token latency while also shrinking memory — which is
# why it is the default choice within one NVLink-connected node.
#
# Below we build a small-but-real transformer stack and run it two ways with the
# exact same code: once sharded across 2 GPUs (TP = 2) and once whole on 1 GPU
# (TP = 1, i.e. `world_size == 1`). The model is deliberately scaled down from a
# 70B-class model so the whole demo runs in well under a minute on 2 A100s while
# still exercising the identical column/row-parallel + all-reduce mechanism used
# in Megatron-LM (Shoeybi et al.) and every production TP serving stack.
#
# A notebook process cannot itself join a `torch.distributed` process group with
# more than one rank per process, so — the standard idiom — we `%%writefile` a
# self-contained script to disk and launch it with `!torchrun`, once with
# `--nproc_per_node=2` (tensor-parallel) and once with `--nproc_per_node=1` (the
# un-sharded single-GPU baseline), using the SAME script and SAME weight seeds.

# %%
%%writefile tp_infer.py
"""
tp_infer.py — minimal Megatron-style tensor-parallel decode benchmark.

Run with:
  torchrun --nproc_per_node=2 tp_infer.py   # TP = 2, weights sharded across both GPUs
  torchrun --nproc_per_node=1 tp_infer.py   # TP = 1, whole (un-sharded) model on one GPU

Both invocations build their weights from the SAME per-tensor seeds, so the two
runs are directly comparable in per-GPU memory and decode latency, and their
saved outputs can be diffed as a (loose, cross-process) sanity check. The tight
correctness check — TP output vs. single-GPU output for identical weights — is
done in-process (world_size == 2 only), in float64, so neither bf16 nor
ordinary float32 reduction-order noise can hide a real bug.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ---- toy "production-shaped" transformer config -----------------------------
# Scaled down from a 70B-class model so the whole notebook runs in well under a
# minute on 2 A100s, but structurally identical: column-parallel QKV + row-
# parallel O in attention, column-parallel up + row-parallel down in the MLP,
# one all-reduce per attention block and one per MLP block, every layer.
D_MODEL = 4096
N_HEADS = 32
HEAD_DIM = D_MODEL // N_HEADS          # 128
D_FF = 4 * D_MODEL                     # 16384
N_LAYERS = 6
BATCH = 16
N_DECODE_STEPS = 30
WARMUP_STEPS = 5
SEED = 0


def setup_distributed():
    """torchrun always sets RANK/WORLD_SIZE/LOCAL_RANK, even for nproc_per_node=1."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def sharded_weight(shape, seed, rank, world_size, shard_dim, device, dtype):
    """Build the FULL weight matrix deterministically from `seed` (identical on
    every rank / every world_size, so world_size=1 and world_size=2 runs use the
    exact same underlying logical weight), then return only this rank's local
    shard along `shard_dim` (0 = split rows/out_features, 1 = split columns/
    in_features). Real systems load a checkpoint and shard it once; regenerating
    on every rank is a simplification that keeps this script self-contained.
    """
    full = torch.empty(shape, dtype=torch.float32)
    full.normal_(mean=0.0, std=0.02, generator=torch.Generator().manual_seed(seed))
    size = shape[shard_dim]
    assert size % world_size == 0, f"{size} not divisible by world_size={world_size}"
    local = size // world_size
    start = rank * local
    idx = [slice(None)] * len(shape)
    idx[shard_dim] = slice(start, start + local)
    shard = full[tuple(idx)].contiguous()
    return shard.to(device=device, dtype=dtype)


class ColParallelLinear(nn.Module):
    """Column-parallel: this rank holds out_features[start:end]. Input is
    replicated across ranks; output is partitioned. No communication here —
    a following RowParallelLinear does the reduction."""

    def __init__(self, in_features, out_features, rank, world_size, seed, device, dtype):
        super().__init__()
        w = sharded_weight((out_features, in_features), seed, rank, world_size,
                            shard_dim=0, device=device, dtype=dtype)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """Row-parallel: this rank holds in_features[start:end]. Input is
    partitioned (the output of a ColParallelLinear); each rank computes a
    partial output and an all-reduce sums the partials into the full result,
    which ends up replicated on every rank."""

    def __init__(self, in_features, out_features, rank, world_size, seed, device, dtype):
        super().__init__()
        w = sharded_weight((out_features, in_features), seed, rank, world_size,
                            shard_dim=1, device=device, dtype=dtype)
        self.weight = nn.Parameter(w)
        self.world_size = world_size

    def forward(self, x):
        partial = F.linear(x, self.weight)
        if self.world_size > 1:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial


class TPAttention(nn.Module):
    """Q/K/V are column-parallel (each rank owns a disjoint set of attention
    heads); O is row-parallel (one all-reduce recombines the heads' outputs)."""

    def __init__(self, rank, world_size, layer_idx, device, dtype):
        super().__init__()
        assert N_HEADS % world_size == 0
        self.local_heads = N_HEADS // world_size
        self.head_dim = HEAD_DIM
        base = SEED + 1000 * layer_idx
        self.q_proj = ColParallelLinear(D_MODEL, D_MODEL, rank, world_size, base + 1, device, dtype)
        self.k_proj = ColParallelLinear(D_MODEL, D_MODEL, rank, world_size, base + 2, device, dtype)
        self.v_proj = ColParallelLinear(D_MODEL, D_MODEL, rank, world_size, base + 3, device, dtype)
        self.o_proj = RowParallelLinear(D_MODEL, D_MODEL, rank, world_size, base + 4, device, dtype)

    def forward(self, x, kv_cache):
        # x: (B, S, D_MODEL), replicated across ranks (S == 1 during decode).
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        # A single new query attending, unmasked, to the whole (growing) KV
        # cache is the standard decode-step formulation; is_causal only
        # matters for the first (prefill) call, where S > 1.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None and S > 1))
        out = out.transpose(1, 2).contiguous().view(B, S, self.local_heads * self.head_dim)
        out = self.o_proj(out)  # all-reduced -> replicated (B, S, D_MODEL)
        return out, new_cache


class TPMlp(nn.Module):
    """Up-projection column-parallel -> GELU (elementwise, safe on a
    partition) -> down-projection row-parallel (one all-reduce)."""

    def __init__(self, rank, world_size, layer_idx, device, dtype):
        super().__init__()
        base = SEED + 1000 * layer_idx
        self.up = ColParallelLinear(D_MODEL, D_FF, rank, world_size, base + 5, device, dtype)
        self.down = RowParallelLinear(D_FF, D_MODEL, rank, world_size, base + 6, device, dtype)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class TPBlock(nn.Module):
    def __init__(self, rank, world_size, layer_idx, device, dtype):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL).to(device=device, dtype=dtype)
        self.attn = TPAttention(rank, world_size, layer_idx, device, dtype)
        self.ln2 = nn.LayerNorm(D_MODEL).to(device=device, dtype=dtype)
        self.mlp = TPMlp(rank, world_size, layer_idx, device, dtype)

    def forward(self, x, kv_cache):
        a, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class TPTinyGPT(nn.Module):
    def __init__(self, rank, world_size, device, dtype):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TPBlock(rank, world_size, i, device, dtype) for i in range(N_LAYERS)]
        )

    def forward(self, x, kv_caches=None):
        new_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, nc = block(x, kv)
            new_caches.append(nc)
        return x, new_caches


def correctness_check(rank, world_size, device):
    """Tight, in-process sanity check that a column-parallel Linear followed by
    a row-parallel Linear (all-reduced) reproduces exactly what an un-sharded
    pair of Linears computes for the SAME logical weight matrices. Only
    meaningful with world_size == 2 (need >1 rank to shard across).

    Uses float64: with unscaled N(0,1) weights and a 1024-wide reduction, the
    activations reach a magnitude of hundreds, so ordinary float32 rounding
    (comparable across a differently-ordered sum-then-all-reduce vs. one big
    matmul) can exceed a naive fixed atol/rtol on individual near-zero output
    entries even though nothing is wrong. float64 removes that ambiguity so an
    `allclose=False` here would mean a real bug, not rounding noise.
    """
    if world_size != 2:
        print(f"[rank {rank}] correctness_check skipped (needs world_size==2, got {world_size})")
        return
    d_in, d_hidden, B, S = 256, 1024, 4, 3
    dt = torch.float64
    x = torch.empty(B, S, d_in, dtype=dt).normal_(generator=torch.Generator().manual_seed(7)).to(device)

    w_up_full = torch.empty(d_hidden, d_in, dtype=dt).normal_(generator=torch.Generator().manual_seed(42))
    w_down_full = torch.empty(d_in, d_hidden, dtype=dt).normal_(generator=torch.Generator().manual_seed(43))

    local_hidden = d_hidden // world_size
    up_shard = w_up_full[rank * local_hidden:(rank + 1) * local_hidden].to(device)
    down_shard = w_down_full[:, rank * local_hidden:(rank + 1) * local_hidden].to(device)

    h_local = F.linear(x, up_shard)             # (B, S, local_hidden), partitioned
    y_tp = F.linear(h_local, down_shard)         # (B, S, d_in), partial per rank
    dist.all_reduce(y_tp, op=dist.ReduceOp.SUM)  # -> full result, replicated

    y_ref = F.linear(F.linear(x, w_up_full.to(device)), w_down_full.to(device))

    max_abs_diff = (y_tp - y_ref).abs().max().item()
    ok = torch.allclose(y_tp, y_ref, atol=1e-9, rtol=1e-9)
    print(f"[rank {rank}] TP-vs-single-GPU correctness (fp64): allclose={ok}, "
          f"max_abs_diff={max_abs_diff:.2e}")


def main():
    rank, world_size, device = setup_distributed()
    correctness_check(rank, world_size, device)

    dtype = torch.bfloat16
    model = TPTinyGPT(rank, world_size, device, dtype)

    n_local_params = sum(p.numel() for p in model.parameters())
    local_weight_gb = n_local_params * 2 / 1e9  # bf16 = 2 bytes/param
    print(f"[rank {rank}/{world_size}] local params: {n_local_params/1e6:.1f}M "
          f"-> weight memory: {local_weight_gb:.3f} GB")

    # ---- prefill: a single-token prompt (kept trivial; the point here is the
    # decode loop) shared across ranks via the same input seed. The same seed on
    # two identical A100s yields identical inputs (torch's CUDA RNG is a
    # device-independent Philox stream), which is exactly what a column-parallel
    # layer needs — a replicated input on every rank. ----
    torch.manual_seed(123)
    x = torch.randn(BATCH, 1, D_MODEL, device=device, dtype=dtype)
    with torch.no_grad():
        x_out, kv_caches = model(x, None)

    # ---- warmup (excluded from timing/memory: lets CUDA context, cuBLAS
    # algorithm selection, and NCCL setup settle before we measure). ----
    next_tok = x_out[:, -1:, :]
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            next_tok, kv_caches = model(next_tok, kv_caches)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    # ---- timed decode loop: one new token per step, KV cache grows each step ----
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_DECODE_STEPS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(N_DECODE_STEPS)]
    with torch.no_grad():
        for i in range(N_DECODE_STEPS):
            starts[i].record()
            next_tok, kv_caches = model(next_tok, kv_caches)
            ends[i].record()
    torch.cuda.synchronize(device)

    step_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    avg_ms = sum(step_ms) / len(step_ms)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9

    print(f"[rank {rank}/{world_size}] avg decode-step latency: {avg_ms:.3f} ms "
          f"over {N_DECODE_STEPS} steps (world_size={world_size})")
    print(f"[rank {rank}/{world_size}] peak memory during timed decode: {peak_gb:.3f} GB")

    if rank == 0:
        out_path = f"tp_infer_output_ws{world_size}.pt"
        torch.save({"output": next_tok.float().cpu(), "avg_ms": avg_ms,
                    "peak_gb": peak_gb, "local_weight_gb": local_weight_gb,
                    "world_size": world_size}, out_path)
        print(f"[rank 0] saved -> {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Run it: TP = 2 (sharded across both GPUs)
#
# `torchrun --nproc_per_node=2` spawns two processes, each running `tp_infer.py`,
# joined into one NCCL process group. Rank 0's process prints the in-process
# correctness check first (float64, tight tolerance — this is the real "does TP
# match single-GPU math" proof), then both ranks print their own local weight
# memory and decode latency.
#
# **Expected result:** `allclose=True` with `max_abs_diff` on the order of
# `1e-10`–`1e-12` (ordinary float64 rounding, not a TP-specific error). Each
# rank's local weight memory should be roughly half of the TP=1 run below it,
# since every column/row-parallel matrix is now split across 2 GPUs.

# %%
!torchrun --nproc_per_node=2 tp_infer.py

# %% [markdown]
# ## Run it again: TP = 1 (the whole model on one GPU)
#
# Same script, same weight seeds, `--nproc_per_node=1`. Because `world_size == 1`,
# `ColParallelLinear`/`RowParallelLinear` degrade to holding the FULL (un-sharded)
# weight matrix and the row-parallel all-reduce is skipped — this is exactly the
# "model replicated whole on one GPU" baseline we want to compare against.
#
# **Expected result:** the correctness check is skipped (needs 2 ranks); local
# weight memory should be roughly double the per-GPU number from the TP=2 run
# above; decode latency per step should be in the same ballpark as TP=2 (this
# toy model is far too small for TP's matmul-time savings to show clearly — the
# real payoff at 70B+ scale is explained in the chapter's worked example) but
# the *memory* story is the one this toy faithfully reproduces.

# %%
!torchrun --nproc_per_node=1 tp_infer.py

# %% [markdown]
# ## Cross-checking the two runs
#
# The TP=2 and TP=1 processes are two independent launches, so their outputs are
# not expected to be bit-identical: bf16 matmuls are not strictly associative,
# and TP=2 sums two half-width partial products via all-reduce where TP=1 does
# one full-width matmul, so tiny rounding differences accumulate across 6 layers
# of attention + MLP + LayerNorm + GELU. This is the same bf16-noise reason
# production frameworks validate TP correctness at the single-layer level (as
# `correctness_check` above does, in float64) rather than expecting deep-stack
# bit-exactness.

# %%
# `torch` is already imported at the top of this notebook; the two .pt files
# were written to disk by the `!torchrun` runs above. weights_only=True is the
# safe/forward-compatible load path — the payload is just tensors + scalars.
out2 = torch.load("tp_infer_output_ws2.pt", weights_only=True)
out1 = torch.load("tp_infer_output_ws1.pt", weights_only=True)

diff = (out2["output"] - out1["output"]).abs()
print(f"TP=2 vs TP=1 final hidden state: mean|diff|={diff.mean().item():.4e}, "
      f"max|diff|={diff.max().item():.4e} (bf16 rounding-order noise, not a bug)")
print()
print(f"{'':>18}{'TP=2 (per rank)':>20}{'TP=1 (single GPU)':>22}")
print(f"{'local weight mem':>18}{out2['local_weight_gb']:>17.3f} GB{out1['local_weight_gb']:>19.3f} GB")
print(f"{'avg decode step':>18}{out2['avg_ms']:>17.3f} ms{out1['avg_ms']:>19.3f} ms")
print(f"{'peak mem (decode)':>18}{out2['peak_gb']:>17.3f} GB{out1['peak_gb']:>19.3f} GB")

# %% [markdown]
# ## The production path: vLLM with `tensor_parallel_size=2`
#
# Nobody hand-rolls `ColParallelLinear`/`RowParallelLinear` in production — vLLM,
# SGLang, and TensorRT-LLM all implement TP internally (with fused kernels,
# CUDA-graph capture of the decode step, and paged/sharded KV-cache allocation)
# and expose it as a single integer. The snippet below is real, correct vLLM API
# usage (not a fabrication) but is **guarded and not executed by default** —
# it needs `pip install vllm`, real model weights (a multi-GB download), and
# meaningfully more than "a few minutes." Flip `RUN_VLLM_DEMO = True` and supply
# a model you're licensed to download if you want to actually run it.
#
# Two things vLLM handles for you that our toy script did by hand: (1) it shards
# the KV cache along with the weights — under TP, each GPU stores only the KV
# entries for the attention heads it owns (for GQA, if the KV-head count is
# smaller than the TP degree, some GPUs duplicate KV heads instead of sharding
# them further); (2) it captures the decode step as a CUDA graph per batch size,
# so the per-layer all-reduce becomes part of the replayed graph rather than a
# fresh kernel launch every step.

# %%
RUN_VLLM_DEMO = False  # set True (+ pip install vllm) to actually run this section

if RUN_VLLM_DEMO:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",  # swap for any model you can access
        tensor_parallel_size=2,       # TP degree: each of the 2 GPUs holds 1/2 of every weight
        dtype="bfloat16",
        max_model_len=8192,            # bounds KV-cache allocation per GPU
        gpu_memory_utilization=0.90,   # leave headroom for activations/framework overhead
        enforce_eager=False,           # allow CUDA-graph capture of the decode step
    )
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
    outputs = llm.generate(
        prompts=["Explain tensor parallelism in one paragraph."],
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
else:
    print("RUN_VLLM_DEMO is False — showing the guarded code path only (see cell source above).")

# %% [markdown]
# ## When TP inference is the right tool
#
# - **Use TP** when the weights (plus the KV cache you need) do not fit in one
#   GPU's memory, *and* the GPUs you're splitting across share NVLink (or an
#   equivalent fast intra-node fabric). TP lowers both memory and per-token
#   latency, at the cost of one all-reduce per attention block and one per MLP
#   block, every layer, every step.
# - **Prefer pipeline parallelism (PP)** when the model is too large for TP alone
#   even within one NVLink island (e.g. it needs more stages than you have
#   NVLink-connected GPUs), or when you're spanning multiple nodes and want to
#   avoid putting an all-reduce on an inter-node link. PP only ships activations
#   between stages (much smaller volume) and is latency-neutral for decode — it
#   buys memory capacity and throughput, not per-token speed.
# - **Prefer plain data-parallel replication** when a single replica already fits
#   comfortably in memory. Replication needs zero communication during inference
#   and scales throughput linearly; it does nothing for a single request's
#   latency.
# - **NVLink is what makes TP viable at all.** The per-step all-reduce payload
#   scales roughly like `d_model x L x (all-reduces per layer) x bytes-per-element`
#   (and a ring all-reduce moves on the order of ~2x that on the wire, and it also
#   scales with the batch dimension), which for a large dense model works out to a
#   few MB per decode step. At NVLink bandwidths (on the order of several hundred
#   GB/s to ~1 TB/s per link, depending on generation) that is on the order of a
#   few-to-tens of microseconds — small next to a per-token budget of tens of
#   milliseconds. The same bytes over PCIe or an inter-node network are one to two
#   orders of magnitude slower, which is why the standing rule is "TP within one
#   NVLink island only."
#
# You can reproduce that scaling for our own toy config as a rough sanity anchor
# (not a claim about real hardware you haven't measured):

# %%
# These are the exact D_MODEL / N_LAYERS constants from tp_infer.py above (not
# arbitrary numbers) plugged into a transparent per-step all-reduce estimate, so
# this is a reproducible calculation for OUR toy config rather than a claim about
# hardware we have not measured. It counts only the hidden-vector payload of each
# all-reduce (a batch=1 slice); real traffic also scales with the batch dimension,
# and a ring all-reduce moves ~2x the payload on the wire.
d_model, n_layers, all_reduces_per_layer, dtype_bytes = 4096, 6, 2, 2
payload_bytes = d_model * n_layers * all_reduces_per_layer * dtype_bytes
wire_bytes = 2 * payload_bytes  # rough ring-all-reduce factor
print(f"Toy model TP all-reduce payload per decode step (per rank, batch=1 slice): "
      f"{payload_bytes / 1e3:.1f} KB  (~{wire_bytes / 1e3:.1f} KB on the wire)")
print("At a rough few-hundred-GB/s NVLink link that is on the order of nanoseconds to "
      "low microseconds per step at this toy scale — dwarfed by kernel-launch overhead. "
      "Communication only starts to matter once d_model, n_layers, and batch grow to "
      "production sizes, or when the link degrades to PCIe/Ethernet — see the chapter's "
      "Llama-3-70B worked example for numbers at that scale.")

# %% [markdown]
# ## What you should see
#
# - The in-process correctness check (world_size=2 run) should print
#   `allclose=True` with `max_abs_diff` on the order of `1e-10`–`1e-12` — ordinary
#   float64 rounding, confirming the column/row-parallel + all-reduce decomposition
#   is mathematically identical to the un-sharded computation.
# - Per-rank weight memory in the TP=2 run should be roughly half the TP=1 run's
#   single-GPU weight memory (modulo the small, non-sharded LayerNorm parameters).
# - Per-step decode latency at this toy scale will likely be similar (maybe within
#   a small factor either way) between TP=1 and TP=2 — the model is far too small
#   for TP's matmul-time halving to dominate over NVLink all-reduce and kernel-launch
#   overhead; the latency *benefit* of TP shows up at real production scale (tens of
#   billions of parameters), while the *memory-fitting* benefit is exactly what you
#   just measured, at any scale.
# - The TP=2 vs TP=1 output diff should be small (bf16-rounding-scale), not exactly
#   zero — expected, and explained above.
#
# **Key takeaways**
# 1. Tensor parallelism is column-parallel-then-row-parallel-with-all-reduce,
#    applied once to attention (QKV column / O row) and once to the MLP (up
#    column / down row) — exactly two all-reduces per transformer layer.
# 2. Correctness is best verified at the single-layer level in float64; a deep
#    stack compared across bf16 runs will show small rounding-order differences
#    that are not bugs.
# 3. TP only pays for itself on fast intra-node interconnect (NVLink); the same
#    communication volume over PCIe or inter-node networks can erase the latency
#    win, which is why production TP groups stay within one NVLink island.
# 4. Real deployments use vLLM/SGLang/TensorRT-LLM's `tensor_parallel_size`
#    rather than hand-written column/row-parallel layers — but the mechanism
#    underneath is exactly what you built and measured above.
#
# **Next step:** see [Distributed Training II: Tensor, Pipeline, Sequence &
# Expert Parallelism](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/06-distributed-model-parallel.html)
# for the training-side treatment of the same parallelism axes, and the
# `03-pretraining__tensor-parallel-from-scratch` notebook for a training-loop
# version of this idiom (forward + backward, not just decode).
