# %% [markdown]
# # Tensor Parallelism From Scratch: Splitting a Layer Across 2 GPUs
#
# > **Hardware:** 2x A100. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will build a Megatron-style column-parallel -> row-parallel MLP block split across
# 2 GPUs, verify its output matches a single-GPU reference (up to bf16 rounding),
# and measure the forward-pass communication cost of the one all-reduce that ties the shards together.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/06-distributed-model-parallel.html) for the full explanation.

# %%
# Dependencies: only PyTorch with a CUDA build (which bundles the NCCL collective backend). This is
# already preinstalled on the GPU image, so there is nothing to install here. On a fresh box you would
# install a CUDA build of torch, e.g. (uncomment and match your CUDA version):
# !pip install torch --index-url https://download.pytorch.org/whl/cu121
#
# We do NOT use Megatron-LM's package: we hand-roll its two core primitives
# (ColumnParallelLinear, RowParallelLinear) to see exactly what they do, then reference the real
# implementation (Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models
# Using Model Parallelism") for the production version.

# %%
import os
import torch

torch.manual_seed(0)

assert torch.cuda.is_available(), "This notebook requires CUDA GPUs."
n_gpus = torch.cuda.device_count()
print(f"Visible GPUs: {n_gpus}")
for i in range(n_gpus):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}, bf16 supported: {torch.cuda.is_bf16_supported()}")

assert n_gpus >= 2, "This notebook targets 2x A100 -- adjust --nproc_per_node below if your box differs."

DTYPE = torch.bfloat16
print(f"dtype for the distributed cells below: {DTYPE}")
print("A notebook process cannot itself join a torch.distributed NCCL process group with >1 rank per "
      "process, so every multi-GPU cell below follows the %%writefile + `!torchrun` idiom: we write a "
      "standalone script to disk, then launch it as N independent processes (one per GPU).")

# %% [markdown]
# ## The pattern: column-parallel -> (elementwise nonlinearity) -> row-parallel -> all-reduce
#
# This is the classic Megatron-LM MLP block. A weight `A` of shape `[H, 4H]` is split along its
# **output columns** across the `t` GPUs in the tensor-parallel group: GPU `i` holds `A_i` of shape
# `[H, 4H/t]` and computes `Y_i = X @ A_i` with **no communication** -- the input `X` is replicated
# (identical on every GPU), the output `Y_i` is left sharded along the feature dimension.
#
# GeLU is elementwise, so it can be applied to each `Y_i` shard independently -- still zero
# communication.
#
# The second weight `B` of shape `[4H, H]` is split along its **input rows**: GPU `i` holds `B_i` of
# shape `[4H/t, H]` and computes a **partial sum** `Z_i = Y_i @ B_i` of shape `[S, H]`. Summing the
# partial sums across all `t` GPUs with a single `all_reduce(SUM)` recovers the full result -- this is
# the *one* communication point in the whole block.
#
# Expected result: the tensor-parallel forward pass should match a single-GPU reference computed
# with the same (unsplit) weights, up to the rounding you'd expect from bf16 matmuls and a different
# summation order.

# %%
%%writefile tp_mlp.py
"""
Megatron-style tensor-parallel MLP block (column-parallel -> GeLU -> row-parallel -> all-reduce),
run as one process per GPU under torchrun. Verifies correctness against a single-GPU reference
and times the forward pass with torch.cuda.Event.
"""
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    # torchrun sets these env vars for us; RANK/WORLD_SIZE identify this process in the TP group,
    # LOCAL_RANK identifies which GPU on this node it should use.
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # NCCL is the collective backend for GPU-GPU comm
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    # --- toy MLP block sizes (Llama-style 4x expansion) ---
    H = 4096            # hidden size
    F_ = 4 * H          # ffn size = 16384
    S = 8192            # "tokens" = batch * seq_len, flattened
    assert F_ % world_size == 0, "ffn size must be divisible by the tensor-parallel world size"
    F_shard = F_ // world_size

    # --- build IDENTICAL full-size input + weights on every rank ---
    # A CPU generator with a fixed seed produces the exact same tensor values regardless of which
    # process runs this code, so every rank can independently reconstruct the same "ground truth"
    # weights and slice out its own shard -- no need to broadcast anything.
    g = torch.Generator(device="cpu").manual_seed(1234)
    X_full = torch.randn(S, H, generator=g, dtype=torch.float32)
    A_full = torch.randn(H, F_, generator=g, dtype=torch.float32) * (H ** -0.5)   # column-parallel weight
    B_full = torch.randn(F_, H, generator=g, dtype=torch.float32) * (F_ ** -0.5)  # row-parallel weight

    # --- single-GPU reference: full (unsplit) computation, replicated identically on every rank ---
    X_ref = X_full.to(device=device, dtype=dtype)
    A_ref = A_full.to(device=device, dtype=dtype)
    B_ref = B_full.to(device=device, dtype=dtype)
    with torch.no_grad():
        Y_ref = F.gelu(X_ref @ A_ref) @ B_ref  # [S, H], exactly what a single GPU would compute

    # --- tensor-parallel shards ---
    col_lo, col_hi = rank * F_shard, (rank + 1) * F_shard
    A_shard = A_full[:, col_lo:col_hi].to(device=device, dtype=dtype).contiguous()  # [H, F_shard]
    B_shard = B_full[col_lo:col_hi, :].to(device=device, dtype=dtype).contiguous()  # [F_shard, H]
    X = X_full.to(device=device, dtype=dtype)  # replicated: identical activations on every GPU

    @torch.no_grad()
    def tp_forward():
        Y_local = F.gelu(X @ A_shard)              # column-parallel: sharded output, NO communication
        Z_partial = Y_local @ B_shard              # row-parallel: partial sum, shape [S, H]
        dist.all_reduce(Z_partial, op=dist.ReduceOp.SUM)  # <-- the ONE all-reduce for this block
        return Z_partial

    # --- warmup (first calls pay CUDA context / kernel autotune / NCCL setup costs) ---
    for _ in range(3):
        _ = tp_forward()
    torch.cuda.synchronize()

    # --- correctness check ---
    Y_tp = tp_forward()
    torch.cuda.synchronize()
    max_abs_diff = (Y_tp.float() - Y_ref.float()).abs().max().item()
    # Loose bf16 tolerance: the row-parallel path rounds each partial sum to bf16 before the all-reduce,
    # so it accumulates in a different order than the reference's single big matmul -- not bit-exact.
    ok = torch.allclose(Y_tp, Y_ref, atol=5e-2, rtol=5e-2)

    # --- timed forward pass: torch.cuda.Event, not wall-clock time.time() ---
    torch.cuda.reset_peak_memory_stats(device)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    n_iters = 20
    start_evt.record()
    for _ in range(n_iters):
        _ = tp_forward()
    end_evt.record()
    torch.cuda.synchronize()
    ms_per_iter = start_evt.elapsed_time(end_evt) / n_iters
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        print(f"world_size={world_size}  A_shard={tuple(A_shard.shape)}  B_shard={tuple(B_shard.shape)}")
        print(f"allclose(Y_tp, Y_ref) = {ok}   max_abs_diff = {max_abs_diff:.3e}")
        print(f"forward time  = {ms_per_iter:.3f} ms/iter (avg over {n_iters} iters, after warmup)")
        print(f"peak GPU mem  = {peak_mem_gb:.3f} GB on rank 0 (holds one shard, not the full weights)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Launch it: one process per GPU via `torchrun`
#
# `torchrun --nproc_per_node=2` spawns 2 independent Python processes on this node, each bound to
# one GPU, and sets `RANK` / `WORLD_SIZE` / `LOCAL_RANK` env vars so `dist.init_process_group`
# knows how to find its peers (the `--standalone` flag uses a local TCP rendezvous, which is all we
# need for a single node).
#
# Expected result: `allclose(Y_tp, Y_ref) = True`, with `max_abs_diff` on the order of `1e-2` --
# bf16 has roughly 2-3 decimal digits of precision, and the row-parallel partial-sum-then-all-reduce
# accumulates in a different order than the reference's single big matmul, so don't expect bit-exact
# equality. The forward time should be small -- this is a single matmul pair on ~8K tokens, so you're
# likely looking at low single-digit milliseconds, dominated as much by kernel launch overhead as by
# the all-reduce at this toy size.

# %%
!torchrun --standalone --nproc_per_node=2 tp_mlp.py

# %% [markdown]
# ## Why the communication volume matters: NVLink vs. cross-node InfiniBand
#
# Per the chapter, a ring all-reduce over `t` GPUs moves roughly
# `2 * (t-1)/t * (message_bytes)` bytes in and out of each GPU, where `message_bytes` is the size
# of the tensor being reduced (here, the row-parallel output `Z_partial` of shape `[S, H]` in bf16,
# i.e. `S * H * 2` bytes).
#
# Expected result: at the toy sizes above (`S=8192`, `H=4096`), the reduced tensor is on the order of
# tens of MB -- comfortably inside what NVLink moves in well under a millisecond, but if this all-reduce
# had to cross node boundaries over InfiniBand instead of NVLink, the same bytes would take roughly an
# order of magnitude (or more) longer, because aggregate NVLink bandwidth on an A100 is on the order of
# hundreds of GB/s while a typical cross-node InfiniBand NIC is on the order of tens of GB/s. That gap
# is exactly why the chapter's rule is "keep the TP group inside one node."

# %%
def ring_allreduce_bytes_per_gpu(t: int, message_bytes: int) -> float:
    """Bytes each GPU sends+receives for one ring all-reduce of a `message_bytes`-sized tensor."""
    return 2 * (t - 1) / t * message_bytes

S, H, t = 8192, 4096, 2
message_bytes = S * H * 2  # bf16 = 2 bytes/element
bytes_per_gpu = ring_allreduce_bytes_per_gpu(t, message_bytes)
print(f"Tensor being reduced: [{S}, {H}] bf16 = {message_bytes / 1e6:.1f} MB")
print(f"Ring all-reduce moves ~{bytes_per_gpu / 1e6:.1f} MB in+out per GPU (t={t})")

# Rough, order-of-magnitude estimates from published hardware bandwidth specs -- NOT measured
# benchmarks. Treat these only as "why NVLink vs. InfiniBand matters", not as predictions: real
# collectives never hit peak bandwidth, and small messages are latency-bound.
nvlink_gbps = 600e9     # A100 aggregate NVLink: on the order of hundreds of GB/s per GPU
infiniband_gbps = 25e9  # typical cross-node IB NIC: on the order of tens of GB/s
print(f"~{bytes_per_gpu / nvlink_gbps * 1e6:.1f} us over NVLink (order-of-magnitude, ignores overheads)")
print(f"~{bytes_per_gpu / infiniband_gbps * 1e6:.1f} us over cross-node InfiniBand (order-of-magnitude)")
print("-> roughly a 10-25x gap at this message size, and it only gets worse as H grows or you cross more nodes.")

# %% [markdown]
# ## Tensor parallelism vs. data parallelism vs. FSDP: what gets communicated, and when
#
# All three are ways of spreading a model across GPUs, but they cut along different axes and pay for
# communication in very different places:
#
# - **DDP (data parallelism):** every GPU holds a **full replica** of the model and processes a
#   different shard of the *batch*. Communication is **one gradient all-reduce per step**
#   (typically overlapped with backward), so it tolerates a slower interconnect fine -- it only has
#   to happen once per step, not once per layer.
# - **FSDP (sharded data parallelism):** parameters/gradients/optimizer state are sharded across
#   GPUs; each layer's full weight is reassembled via an **all-gather** just before it's used and
#   its gradient is **reduce-scattered** right after. More communication than DDP, still once per
#   layer per step, and still batch-sharded.
# - **Tensor parallelism (this notebook):** the **computation graph itself** is split -- a single
#   matmul's output or input is sharded across GPUs. That means an all-reduce (or all-gather) is
#   needed **inside the forward and backward pass of every TP-split layer**: a full transformer
#   block with both attention and MLP sublayers pays two all-reduces forward (attention's output
#   projection, MLP's down-projection) and two more backward -- four per layer per step. This is
#   why the chapter's rule of thumb is to keep the TP group inside one node (`t <= 8`) on the
#   fastest interconnect available (NVLink/NVSwitch): a per-layer collective serializing behind a
#   slow cross-node link collapses your throughput in a way a once-per-step DDP all-reduce never
#   would.
#
# DP/FSDP and TP are **orthogonal** and combine multiplicatively in real training: you pick a TP
# degree that fits inside one node's NVLink domain, then wrap TP groups in DP or FSDP across nodes.
# The production version of exactly the primitives built above -- `ColumnParallelLinear`,
# `RowParallelLinear`, plus vocab-parallel embeddings and cross-entropy -- lives in Megatron-LM
# (Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
# Parallelism"), under `megatron/core/tensor_parallel/`.

# %% [markdown]
# ## What you should see
#
# - `allclose(Y_tp, Y_ref) = True`, with `max_abs_diff` on the order of `1e-2` or smaller -- the
#   tensor-parallel and single-GPU computations agree up to bf16 rounding and summation-order
#   differences, not bit-exact equality.
# - A forward pass time on the order of low single-digit milliseconds for this toy block size,
#   with peak memory on each GPU reflecting only *its shard* of the weights (`A_shard`/`B_shard`),
#   not the full `[H, 4H]` and `[4H, H]` matrices -- this is the whole point of tensor parallelism:
#   no single GPU ever materializes the full weight.
# - The communication-volume estimate should make the "NVLink-bound, keep it intra-node" rule
#   concrete: the same bytes take roughly an order of magnitude longer over cross-node InfiniBand
#   than over NVLink, and this all-reduce happens on every TP-split layer, every forward AND
#   backward pass -- unlike DDP's once-per-step gradient sync.
#
# **Key takeaways:**
# 1. Column-parallel then row-parallel is the whole trick: split so the intermediate activation
#    never needs to be gathered, and pay for exactly one all-reduce at the end of the pair.
# 2. TP splits the *computation graph*; DP/FSDP split the *batch* (and, for FSDP, the *parameters*).
#    They compose multiplicatively, not by substitution.
# 3. TP's per-layer, per-forward-and-backward all-reduce is why it belongs on NVLink inside one
#    node (`t <= 8`), while DDP's once-per-step all-reduce tolerates much slower fabrics.
# 4. A correct toy implementation is short -- two sharded matmuls and one `all_reduce` -- which is
#    exactly why it's worth building from scratch once before trusting Megatron-LM's production
#    version.
#
# **Next step:** see the chapter's sections on `VocabParallelEmbedding` and
# `vocab_parallel_cross_entropy` for how the same column/row-parallel + all-reduce pattern extends
# to the embedding and loss layers, and combine this TP group with FSDP/DDP across nodes for a full
# 3D-parallel training setup.
