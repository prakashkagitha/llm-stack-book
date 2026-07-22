# 4.3 FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8

FlashAttention I (covered in [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)) solved the *memory* problem: by tiling the attention computation and keeping the intermediate softmax statistics in on-chip SRAM, it computes exact attention without ever materializing the $N \times N$ score matrix in high-bandwidth memory (HBM). That single idea — fuse the whole attention into one kernel, never write the scores out — is the foundation everything in this chapter builds on.

But FlashAttention-1 (FA1) left a lot of GPU performance on the table. Measured against the device's peak matmul throughput, the original kernel ran at roughly 25–40% of the GPU's FLOP ceiling on an A100, while a well-tuned dense GEMM hits 80–90%. The gap is not memory traffic — FA1 already minimized that. The gap is *occupancy and instruction mix*: the kernel was leaving compute units idle and burning too many cycles on non-matmul arithmetic. FlashAttention-2 (FA2, Dao 2023) closes most of that gap on Ampere with three changes to *how the work is partitioned*. FlashAttention-3 (FA3, Shah et al. 2024) then exploits hardware features specific to NVIDIA's Hopper architecture — asynchronous tensor-core instructions, the Tensor Memory Accelerator (TMA), and FP8 tensor cores — to push utilization to roughly 75% of bf16 peak and into the petaFLOP range with FP8.

This chapter is about the *systems engineering* of attention kernels. We assume you already understand the online-softmax recurrence; here we focus on **work partitioning** (which thread does what), **warp specialization** (producer/consumer pipelines), and **low-precision attention** (FP8 with incoherent processing). To follow the hardware reasoning, keep [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) and [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) nearby — we lean heavily on the concepts of warps, warp-groups, shared memory, register pressure, and arithmetic intensity.

## What FlashAttention-1 left on the table

Let us restate the FA1 inner loop precisely, because the FA2 improvements are best understood as edits to it. We tile the query into blocks of $B_r$ rows and the keys/values into blocks of $B_c$ columns. For a fixed query block $Q_i$ we loop over all key blocks $K_j, V_j$, maintaining a running max $m$, running denominator $\ell$, and a running unnormalized output accumulator $O$:

$$
S_{ij} = Q_i K_j^\top, \quad m^{\text{new}} = \max(m, \operatorname{rowmax}(S_{ij})), \quad P_{ij} = \exp(S_{ij} - m^{\text{new}})
$$

$$
\ell^{\text{new}} = e^{m - m^{\text{new}}}\,\ell + \operatorname{rowsum}(P_{ij}), \qquad O^{\text{new}} = e^{m - m^{\text{new}}}\,O + P_{ij} V_j
$$

After the last key block, the final output is $O / \ell$. This is exact attention — the online recurrence reproduces the global softmax. Three inefficiencies hide in this loop.

**1. The rescaling does too much non-matmul work.** Every inner iteration multiplies the accumulator $O$ by the correction factor $e^{m - m^{\text{new}}}$. On a GPU, FLOPs are *not* fungible: a tensor-core matmul instruction on an A100 runs at ~312 TFLOP/s in bf16, but the same hardware executes element-wise transcendental and multiply operations on the much slower CUDA cores at perhaps 1/16th that rate. So the $\exp$, the rowmax, and the per-iteration rescale — though they are a small *fraction* of total FLOPs — consume a disproportionate share of *time*. FA1 rescales $O$ once per inner iteration; that is more non-matmul work than necessary.

**2. The parallelization wastes blocks.** FA1 parallelized over the batch and head dimensions and over query blocks, assigning one thread block per $(\text{batch},\text{head},\text{query-block})$. The number of these is `batch * heads * (seqlen / B_r)`. For training large models with long sequences, batch-times-heads can be small (you trade batch for sequence length), so the grid may not have enough thread blocks to fill all the streaming multiprocessors (SMs). An A100 has 108 SMs; if your grid launches only 60 thread blocks, 44% of the chip is idle.

**3. Work is split badly across warps inside a block.** A thread block on Ampere is typically 4 or 8 warps (128–256 threads). FA1 split the *key/value* block across warps: each warp computed $Q_i K_j^\top$ for a slice of columns. But then every warp needs the full $Q_i$ in registers, and after computing its partial $P V$ the warps must *communicate through shared memory* to combine results, because each warp owns a column-slice of the scores but the output reduction is along columns. This shared-memory round trip — the so-called "split-K" pattern — stalls the warps and adds synchronization.

FA2 fixes all three. None of these changes touch the math; they change *who computes what, in what order*.

## FlashAttention-2: three edits to the inner loop

### Edit 1 — defer the rescaling

The first FA2 trick removes the per-iteration accumulator rescale. Instead of keeping $O$ correctly normalized at every step, we keep an **unnormalized** accumulator and only divide by $\ell$ once at the very end.

Look again at the recurrence. The accumulator update is $O^{\text{new}} = e^{m - m^{\text{new}}} O + P_{ij} V_j$. The factor $e^{m - m^{\text{new}}}$ must still be applied — it corrects for the change in the running max — so we cannot drop it. But the *normalization by $\ell$* can be deferred. In FA1's framing the normalization and the max-correction were tangled together; FA2 separates them. The accumulator is held as $\tilde{O}$ (unnormalized), and we only compute $O_i = \tilde{O}_i / \ell_i$ after the key loop finishes. This removes one full division per element per inner iteration and replaces it with a single division at the end.

More importantly, FA2 reorganizes which operations are on the critical path so that the bulk of the inner loop is two back-to-back matmuls ($Q K^\top$ then $P V$) with the minimum possible scalar work between them. The rule of thumb: **keep the tensor cores fed; do everything else as rarely as possible.**

Here is a from-scratch, runnable reference implementation in pure PyTorch that mirrors the FA2 dataflow. It is not fast — it is meant to make the bookkeeping unambiguous so you can map it onto the CUDA kernel.

```python
import torch

def flash_attention_2_reference(Q, K, V, Br=64, Bc=64, causal=False):
    """
    Exact attention via the FlashAttention-2 recurrence.
    Q, K, V: (N, d). Returns O: (N, d).

    Mirrors the FA2 dataflow:
      - outer loop over QUERY blocks (this is the parallel axis on the GPU)
      - inner loop over KEY/VALUE blocks
      - the output accumulator is kept UNNORMALIZED and divided by l only at the end
    """
    N, d = Q.shape
    O = torch.zeros_like(Q)
    # running statistics, one per query row
    for i in range(0, N, Br):
        qi = Q[i:i+Br]                          # (Br, d)
        Oi = torch.zeros(qi.shape[0], d)        # UNNORMALIZED accumulator
        mi = torch.full((qi.shape[0],), -float('inf'))   # running max
        li = torch.zeros(qi.shape[0])           # running denominator (sum of exp)

        for j in range(0, N, Bc):
            if causal and j > i + Br - 1:
                break                            # whole key block is in the future
            kj = K[j:j+Bc]                       # (Bc, d)
            vj = V[j:j+Bc]                       # (Bc, d)

            # ---- matmul 1: scores ----
            Sij = (qi @ kj.T) / (d ** 0.5)       # (Br, Bc)
            if causal:
                # mask future positions inside the diagonal block
                qpos = torch.arange(i, i+qi.shape[0]).unsqueeze(1)
                kpos = torch.arange(j, j+kj.shape[0]).unsqueeze(0)
                Sij = Sij.masked_fill(kpos > qpos, -float('inf'))

            # ---- online softmax statistics ----
            mij = Sij.max(dim=1).values                  # block rowmax
            mi_new = torch.maximum(mi, mij)
            Pij = torch.exp(Sij - mi_new.unsqueeze(1))   # (Br, Bc)
            alpha = torch.exp(mi - mi_new)               # correction factor

            # rescale running denom and accumulator by alpha
            li = alpha * li + Pij.sum(dim=1)

            # ---- matmul 2: weighted values, accumulated UNNORMALIZED ----
            Oi = alpha.unsqueeze(1) * Oi + Pij @ vj      # (Br, d)

            mi = mi_new

        # single normalization at the very end (FA2's deferred division)
        O[i:i+qi.shape[0]] = Oi / li.unsqueeze(1)
    return O


if __name__ == "__main__":
    torch.manual_seed(0)
    N, d = 256, 64
    Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
    ref = torch.softmax(Q @ K.T / d**0.5, dim=-1) @ V
    out = flash_attention_2_reference(Q, K, V)
    print("max abs error vs dense softmax:", (ref - out).abs().max().item())
    # prints something like 1e-6 — exact attention up to floating point
```

The single line `Oi = alpha.unsqueeze(1) * Oi + Pij @ vj` plus the final `Oi / li` is the heart of the deferred-normalization scheme. Compare it to FA1, which would divide inside the loop. The savings are not the division per se but the *register and instruction budget* freed up for the tensor cores.

### Edit 2 — parallelize over the sequence dimension

FA2's second change is at the *grid* level, not inside the kernel. FA1 assigned one thread block per query block but did not use sequence-length parallelism beyond that. FA2 makes the **outer loop over query blocks the parallel axis** and additionally allows splitting along sequence length so that the grid has enough thread blocks to saturate every SM even when batch and heads are small.

Concretely, the FA2 launch grid is three-dimensional:

```text
gridDim = (ceil(seqlen_q / Br),   num_heads,   batch_size)
```

Each thread block owns one query block for one (head, batch) and loops over *all* key blocks itself. Because the query-block axis is now first and independent, long sequences directly produce more thread blocks. For a sequence of 8192 tokens with `Br = 64`, that is 128 query blocks *per head per batch element* — easily enough to fill 108 SMs even at batch=1, head=1.

For the *backward* pass the story is subtler. The backward pass needs to accumulate gradients $dK$ and $dV$, which are reductions over query blocks. FA2 parallelizes the backward over *key/value* blocks (so each thread block owns a $K_j, V_j$ and loops over query blocks), avoiding atomic adds into $dK, dV$. The asymmetry — forward parallel over queries, backward parallel over keys — is a recurring theme in attention kernels: pick the parallel axis that makes the reduction local.

!!! note "Aside: why causal masking is a load-balancing problem"
    With a causal mask, query block $i$ only attends to key blocks $j \le i$. So the *amount of work per query block grows linearly with $i$*: the first query block does one inner iteration, the last does many. If you naively map query blocks to SMs, the SMs handling early query blocks finish quickly and sit idle while late ones grind on — a classic load imbalance. FA2 mitigates this by skipping fully-masked key blocks entirely (the `break` in the reference code) and by scheduling so that long and short query blocks are interleaved across SMs. This is why a causal FlashAttention forward is roughly *half* the FLOPs of a full one, but only if the scheduler actually exploits the triangular structure.

### Edit 3 — partition warps over queries, not keys

The third FA2 change is the most hardware-specific and the one that gives the kernel its "warp partitioning" name. Recall a thread block on Ampere has, say, 4 warps. The question is how to divide the $B_r \times B_c$ tile of work among them.

FA1 used the **split-K** layout: each warp computes $Q_i K_j^\top$ for *all* $B_r$ query rows but only a slice of the $B_c$ key columns. The problem: after the second matmul $P_{ij} V_j$, the partial outputs from different warps must be summed (they hold partial sums over disjoint key columns), and that reduction goes through shared memory with a synchronization barrier. Every inner iteration pays for a shared-memory write, a `__syncthreads()`, and a read.

FA2 flips this to the **split-Q** layout (sometimes called "warp partitioning over rows"): each warp owns a slice of the $B_r$ query rows and computes the *full* row of scores and output for those rows. Now warp 0 produces the final output for query rows 0–15, warp 1 for rows 16–31, and so on — *no inter-warp reduction is needed*, because each warp's rows are independent. $K_j$ and $V_j$ are shared across warps (read from shared memory), but there is no write-back-and-sum.

```text
SPLIT-K (FA1)                           SPLIT-Q (FA2)
each warp: all Br rows, slice of cols    each warp: slice of Br rows, all cols

  Q_i (all rows)                          Q_i rows 0-15  -> warp 0  (full output)
   |                                      Q_i rows 16-31 -> warp 1  (full output)
   v                                      Q_i rows 32-47 -> warp 2  (full output)
  K_j cols 0-15  -> warp 0  (partial)     Q_i rows 48-63 -> warp 3  (full output)
  K_j cols 16-31 -> warp 1  (partial)
   ...                                    K_j / V_j: SHARED across warps (read-only)
  REDUCE across warps via shared mem  <-- no reduction needed
  + __syncthreads()                       no extra __syncthreads in the hot path
```

{{fig:fa2-split-k-vs-split-q}}

Removing that per-iteration synchronization and shared-memory traffic is worth a large fraction of FA2's speedup. The cost is that each warp needs the $K_j, V_j$ tile available; FA2 keeps those in shared memory and lets all warps read them, which is exactly what shared memory is good at (broadcast reads).

Putting the three edits together, FA2 roughly doubles FA1's throughput on A100, reaching on the order of 50–73% of theoretical bf16 matmul peak depending on head dimension and sequence length. In practice you never write this yourself — you call the library:

```python
import torch
import torch.nn.functional as F

# PyTorch's scaled_dot_product_attention dispatches to a FlashAttention-2
# backend on Ampere+/CUDA when the inputs qualify (fp16/bf16, head_dim <= 256,
# contiguous, etc.). This is the FA2 kernel in production form.
q = torch.randn(2, 16, 8192, 128, device="cuda", dtype=torch.bfloat16)  # (B,H,N,d)
k = torch.randn_like(q)
v = torch.randn_like(q)

with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

print(out.shape)  # (2, 16, 8192, 128)
```

!!! example "Worked example: how much does the non-matmul work matter?"
    Take one attention head with sequence length $N = 8192$ and head dimension $d = 128$, full (non-causal) attention. The two matmuls $QK^\top$ and $PV$ cost about $2 \cdot 2 N^2 d = 4 N^2 d$ FLOPs (a factor 2 per matmul for multiply-add). Plug in:

    $$
    4 N^2 d = 4 \cdot (8192)^2 \cdot 128 \approx 3.4 \times 10^{10}\ \text{matmul FLOPs}.
    $$

    The softmax non-matmul work — one $\exp$, one max, one sum, one rescale per score — is on the order of a handful of FLOPs per score, so roughly $c \cdot N^2$ with $c \approx 5$:

    $$
    5 N^2 = 5 \cdot (8192)^2 \approx 3.4 \times 10^8\ \text{non-matmul FLOPs}.
    $$

    So non-matmul work is only ~1% of the FLOP *count*. But on an A100, suppose tensor cores run at 312 TFLOP/s and the special-function unit handling $\exp$ runs at ~20 TFLOP/s. The matmul takes $3.4\times10^{10} / 3.12\times10^{14} \approx 109\ \mu s$; the non-matmul takes $3.4\times10^{8} / 2.0\times10^{13} \approx 17\ \mu s$. That "1% of FLOPs" is suddenly **~14% of the runtime** if it is on the critical path. This is exactly why FA2 fights to minimize and overlap the softmax work, and why FA3's overlapping of matmul and softmax (below) matters so much.

{{fig:fa2-flops-not-fungible}}

## FlashAttention-3: exploiting Hopper

FA2 is close to optimal *for Ampere's programming model*, where the tensor-core instruction (`mma`) is synchronous: a warp issues it and waits for the result. NVIDIA's Hopper architecture (H100) added new hardware that breaks that synchronous assumption, and FA3 is essentially "FA2 rewritten to use everything Hopper offers." Three Hopper features matter.

**Warpgroup-wide asynchronous matmul (WGMMA).** Hopper's `wgmma` instruction is issued by a *warpgroup* (4 warps = 128 threads acting together) and runs *asynchronously*: the warpgroup issues it, then continues executing other instructions while the tensor cores grind, and later waits on the result. This is the key that lets us overlap matmul with softmax.

**The Tensor Memory Accelerator (TMA).** TMA is a dedicated hardware unit that performs bulk asynchronous copies between global memory and shared memory, with the address generation done in hardware. Instead of every thread computing addresses and issuing loads, a single thread kicks off a TMA copy of a whole tile and the unit streams it in the background. This frees the warps from address arithmetic and overlaps data movement with compute.

**FP8 tensor cores.** Hopper tensor cores can do matmuls in 8-bit floating point (`e4m3`/`e5m2`), roughly doubling throughput over bf16 — on the order of ~2 PFLOP/s for FP8 on an H100 versus ~1 PFLOP/s bf16 (dense). For attention, where the matmuls dominate, FP8 is a huge prize *if* you can manage the accuracy.

FA3's design is built around two ideas: **warp specialization** (a producer/consumer pipeline) and **overlapping the two matmuls with the softmax**.

### Warp specialization: producer/consumer with TMA

In FA2, every warp does the same thing: load, matmul, softmax, matmul. FA3 instead **specializes** warps into roles, turning the thread block into a software pipeline.

- **Producer warps** do almost no compute. Their job is to issue TMA loads that stream the next $K_j, V_j$ tiles from HBM into a circular buffer in shared memory, staying one or two stages ahead of consumption.
- **Consumer warpgroups** do the math. They wait until a tile is present in the shared-memory buffer (signaled by the producer via an asynchronous barrier), then run `wgmma` for $QK^\top$, the softmax, and `wgmma` for $PV$.

This is the classic **producer/consumer** decoupling. Because the producer uses TMA (which is asynchronous and doesn't occupy the tensor cores or the math pipes), data movement for iteration $j+1$ happens *while* the consumer computes iteration $j$. The shared-memory buffer is a small ring (often 2–3 stages deep); synchronization uses Hopper's `mbarrier` asynchronous transaction barriers rather than `__syncthreads()`.

```text
                         shared-memory ring buffer (stages 0,1,2)
   PRODUCER warps   --TMA-->  [ K0 V0 | K1 V1 | K2 V2 ]  --read-->  CONSUMER warpgroups
   (issue async copies,                                            (wgmma QK^T, softmax,
    stay ahead)         <-- "buffer empty" signal (mbarrier) --     wgmma PV, advance)

   time --->
   prod:  load0  load1  load2  load3  load4 ...
   cons:         gemm0  gemm1  gemm2  gemm3 ...   (always one step behind, never starved)
```

{{fig:fa3-producer-consumer-pingpong}}

The payoff: the tensor cores rarely stall waiting for data, and the warps doing math never spend cycles on address generation. Register usage is also better balanced — Hopper lets you *reallocate* registers between warpgroups (`setmaxnreg`), giving the compute warpgroups more registers and the producer warps fewer.

### Overlapping the two matmuls with softmax (ping-pong & pipelined)

The deeper FA3 trick attacks the worked-example problem above: the softmax (an $\exp$-heavy non-matmul op running on the slow special-function units) sits *between* the two matmuls and stalls the tensor cores. FA3 overlaps them in two complementary ways.

**Inter-warpgroup "ping-pong" scheduling.** With two consumer warpgroups, FA3 staggers them so that while warpgroup 1 is doing its softmax (on the special-function units / CUDA cores), warpgroup 2 is doing its `wgmma` (on the tensor cores), and then they swap. Because softmax uses different execution units than the matmul, the two warpgroups interleave to keep *both* unit types busy. The scheduler uses synchronization barriers to force this alternation, so the tensor cores are almost never idle waiting on an $\exp$.

```text
PING-PONG across two consumer warpgroups (WG):

WG1:  GEMM(QK)  | softmax  | GEMM(PV) | GEMM(QK)  | softmax  | ...
WG2:  ........  | GEMM(QK) | softmax  | GEMM(PV)  | GEMM(QK) | ...
            ^ while WG1 does softmax (SFU), WG2 does GEMM (tensor cores)
              -> tensor cores and SFUs both stay busy
```

**Intra-warpgroup pipelining.** Within a single warpgroup, FA3 also software-pipelines across inner-loop iterations: the $QK^\top$ matmul of iteration $j+1$ is issued (asynchronously, via `wgmma`) *before* the softmax of iteration $j$ finishes, so the score matmul for the next block overlaps the current block's exponentiation. Because `wgmma` is asynchronous, you issue it and move on; the dependency is resolved later by a wait. This is exactly the kind of overlap that is *impossible* on Ampere, where `mma` is synchronous.

The combination — TMA-fed producer/consumer plus matmul/softmax overlap — is why FA3 reaches roughly 75% of H100 bf16 peak (on the order of 700+ TFLOP/s) where FA2 on the same hardware sits closer to 35–45%. FA2 simply cannot express the overlap; it was designed for a synchronous instruction set.

!!! warning "Common pitfall: FA2 on Hopper is leaving half the chip idle"
    A frequent surprise: you upgrade from A100 to H100, keep using the FA2 kernel, and see far less than the expected speedup. The reason is that FA2's synchronous structure cannot use `wgmma` overlap or TMA, so on Hopper it under-utilizes the tensor cores badly. The fix is to use a kernel built for Hopper (the FA3 kernel, or cuDNN's Hopper attention, or a Triton/CUTLASS kernel that targets `wgmma`). Check which backend your framework actually dispatches to — "FlashAttention is enabled" does not tell you whether it is the Hopper-optimized path.

### A from-scratch model of the producer/consumer pipeline

You cannot write `wgmma` in Python, but you *can* simulate the dataflow to internalize the dependency structure. The following models the ring buffer and the producer/consumer roles, computing exact attention while showing how a consumer is "always one step behind" the producer.

```python
import torch
from collections import deque

def fa3_pipeline_model(Q, K, V, Bc=64, n_stages=2):
    """
    A *dataflow model* of FA3's producer/consumer pipeline for one query block.
    It does NOT use real async hardware; it shows the ordering:
      - producer 'prefetches' up to n_stages KV tiles into a ring buffer
      - consumer pops a tile only after it is present, then does the GEMMs/softmax
    Result is exact attention for the single query block Q.
    """
    N, d = K.shape
    ring = deque()                      # the shared-memory ring buffer of (Kj, Vj)
    Oi = torch.zeros(Q.shape[0], d)     # unnormalized accumulator
    mi = torch.full((Q.shape[0],), -float('inf'))
    li = torch.zeros(Q.shape[0])

    key_blocks = [(j, K[j:j+Bc], V[j:j+Bc]) for j in range(0, N, Bc)]
    it = iter(key_blocks)

    def produce():                      # PRODUCER: issue one "TMA load"
        try:
            ring.append(next(it))       # copy a KV tile into the ring
            return True
        except StopIteration:
            return False

    # prime the pipeline: producer runs ahead by n_stages
    for _ in range(n_stages):
        produce()

    while ring:                         # CONSUMER drains the ring
        j, kj, vj = ring.popleft()      # wait-for-tile then take it
        produce()                       # producer refills behind the consumer

        Sij = (Q @ kj.T) / (d ** 0.5)          # wgmma #1 (async on real HW)
        mij = Sij.max(dim=1).values
        mi_new = torch.maximum(mi, mij)
        Pij = torch.exp(Sij - mi_new.unsqueeze(1))   # softmax on SFU (overlapped)
        alpha = torch.exp(mi - mi_new)
        li = alpha * li + Pij.sum(dim=1)
        Oi = alpha.unsqueeze(1) * Oi + Pij @ vj      # wgmma #2 (async on real HW)
        mi = mi_new

    return Oi / li.unsqueeze(1)


if __name__ == "__main__":
    torch.manual_seed(1)
    N, d = 512, 64
    Q = torch.randn(128, d); K = torch.randn(N, d); V = torch.randn(N, d)
    ref = torch.softmax(Q @ K.T / d**0.5, dim=-1) @ V
    out = fa3_pipeline_model(Q, K, V)
    print("max abs error:", (ref - out).abs().max().item())  # ~1e-6
```

The takeaway is the ordering: `produce()` is called *before* the consumer touches the tile, and again *after* it pops one, so the buffer stays full and the consumer never waits. On real Hopper hardware the `produce()` is a TMA instruction that the GPU executes concurrently with the consumer's `wgmma`s.

## FP8 attention and incoherent processing

The biggest single lever in FA3 is FP8. Hopper's FP8 tensor cores roughly double matmul throughput, and since attention is matmul-bound, FP8 attention can nearly double the kernel's speed. But naive FP8 attention is *inaccurate*, and FA3 introduces a specific technique — borrowed from quantization literature — to make it usable.

### Why FP8 attention is hard

FP8 `e4m3` has 1 sign bit, 4 exponent bits, 3 mantissa bits — about 3–4 bits of precision, representing a small dynamic range with coarse steps. Two problems arise.

**Outliers blow up the scale.** To quantize a tensor to FP8 you pick a scale $s$ so that values map into the representable range. If a few entries of $Q$ or $K$ are large outliers (and attention activations are notoriously outlier-prone — see [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html)), the scale must stretch to cover them, crushing all the *typical* values into just a handful of FP8 levels. The quantization error on the bulk of the data explodes.

**Layout mismatch for the second matmul.** FP8 `wgmma` on Hopper requires specific operand layouts; the probability matrix $P$ produced by softmax and the value matrix $V$ may not be in the right layout, forcing in-kernel transposes/shuffles. FA3 handles this with byte-permute tricks and careful layout choices, but it is real engineering, not free.

### Incoherent processing: rotate away the outliers

FA3's accuracy fix for the outlier problem is **incoherent processing**, an idea from the QuIP quantization work (Chee et al.). The insight: multiply $Q$ and $K$ by a random orthogonal matrix $M$ before quantizing. Because $M$ is orthogonal, $M^\top M = I$, so

$$
(Q M)(K M)^\top = Q M M^\top K^\top = Q K^\top,
$$

the attention scores are *unchanged*. But the rotation **spreads each outlier across many coordinates**: a single huge entry becomes many moderate entries, so no coordinate dominates and the FP8 scale no longer has to stretch to cover an outlier. The quantization becomes far more uniform and the error drops.

In practice FA3 uses a structured orthogonal matrix that is cheap to apply — a random-sign **Hadamard transform**, $M = \frac{1}{\sqrt{d}} H D$, where $H$ is the Hadamard matrix (applied in $O(d \log d)$ via the fast Hadamard transform) and $D$ is a random diagonal of $\pm 1$. The cost is a small pre-rotation of $Q$ and $K$; the benefit is that FP8 attention error approaches that of a properly-scaled quantization without outlier contamination.

{{fig:fa3-incoherent-processing}}

```python
import torch

def hadamard(n):
    """Build the n x n (Sylvester) Hadamard matrix; n must be a power of 2."""
    assert n & (n - 1) == 0, "n must be a power of two"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H,  H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / (H.shape[0] ** 0.5)            # orthonormal: H^T H = I

def incoherent_rotation(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    H = hadamard(d)
    D = torch.where(torch.rand(d, generator=g) > 0.5, 1.0, -1.0)  # random +-1 signs
    return H @ torch.diag(D)                   # M = H D, orthogonal

def fake_fp8_quant(x):
    """Crude e4m3-like quantization: per-tensor scale, ~3 mantissa bits."""
    s = x.abs().max() / 448.0                  # 448 = max finite e4m3 magnitude
    if s == 0:
        s = torch.tensor(1.0)
    xs = x / s                                 # bring values into FP8 range
    # simulate ~3 mantissa bits by rounding to 2^-3 relative steps
    # (illustrative, not a bit-exact e4m3 emulator)
    mant = torch.round(xs * 8) / 8
    return mant * s

if __name__ == "__main__":
    torch.manual_seed(0)
    d = 128
    Q = torch.randn(64, d); K = torch.randn(64, d)
    Q[:, 7] *= 25.0; K[:, 7] *= 25.0           # inject an outlier feature
    ref = Q @ K.T

    # --- FP8 WITHOUT incoherent processing ---
    err_plain = (fake_fp8_quant(Q) @ fake_fp8_quant(K).T - ref).abs().mean()

    # --- FP8 WITH incoherent processing (rotate, quantize, scores invariant) ---
    M = incoherent_rotation(d)
    Qr, Kr = Q @ M, K @ M                       # (QM)(KM)^T == QK^T exactly
    err_rot = (fake_fp8_quant(Qr) @ fake_fp8_quant(Kr).T - ref).abs().mean()

    print(f"mean score error  plain FP8 : {err_plain:.4f}")
    print(f"mean score error  rotated   : {err_rot:.4f}")
    # the rotated version has substantially lower error because the
    # outlier in feature 7 is spread across all d coordinates
```

Run it and the rotated version shows markedly lower error: the outlier in feature 7, which forced a huge scale and wrecked the plain FP8 matmul, is smeared across all $d$ coordinates after the Hadamard rotation. This is the same principle that powers Hadamard-based quantization schemes elsewhere in the stack (QuaRot, SpinQuant); see [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html) and the FP8 training discussion in [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

### What stays in higher precision

A crucial detail: FA3 does **not** do everything in FP8. The matmul *inputs* ($Q$, $K$, $P$, $V$) are FP8, but the tensor cores **accumulate in FP32**, and the softmax statistics ($m$, $\ell$) and the running output accumulator are kept in higher precision. You quantize the operands going *into* the matmul, not the running state. This mixed strategy — low precision for the bandwidth- and throughput-bound multiply, high precision for the sensitive accumulation and the softmax — is the same pattern as bf16 mixed-precision training and is what keeps end-to-end error tolerable.

## Comparison, trade-offs, and when each wins

It is worth stepping back to place these kernels against each other and against alternatives.

| Kernel | Target HW | Key mechanism | Precision | Rough peak util. |
|---|---|---|---|---|
| FlashAttention-1 | Ampere (A100) | IO-aware tiling, online softmax | fp16/bf16 | ~25–40% |
| FlashAttention-2 | Ampere (A100) | deferred rescale, seqlen parallel, split-Q warps | fp16/bf16 | ~50–73% |
| FlashAttention-3 | Hopper (H100) | warp-specialized producer/consumer (TMA), wgmma overlap | bf16 / FP8 | ~75% (bf16); higher FP8 |
| xFormers mem-efficient | Ampere+ | tiled attention (similar idea) | fp16/bf16 | comparable to FA1/FA2 |
| cuDNN fused attention | Ampere/Hopper | vendor kernels, Hopper-aware | bf16/FP8 | competitive on Hopper |
| FlashDecoding | inference decode | split-KV parallelism for tiny query | fp16/bf16 | optimizes decode latency |

A few important clarifications:

**xFormers and the lineage of `memory_efficient_attention`.** xFormers is Meta's library of composable attention/kernel building blocks, and its `memory_efficient_attention` was one of the first widely-used fused, tiled attention kernels: it never materializes the full $N \times N$ score matrix, using the same online-softmax-plus-tiling idea as FlashAttention. Its lineage traces to Rabe & Staats's "Self-Attention Does Not Need $O(N^2)$ Memory" (2021) combined with FlashAttention-style IO-aware tiling, and its fast path is a set of CUTLASS-based CUDA C++ kernels (with a fallback Triton/flash path). The API differs from SDPA's layout — `q`/`k`/`v` are shaped `(batch, seqlen, num_heads, head_dim)` rather than `(B, H, N, d)`, and causal masking is passed as a bias object rather than a boolean flag:

```python
from xformers.ops import memory_efficient_attention, LowerTriangularMask

# q, k, v: (batch, seqlen, num_heads, head_dim)
out = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
```

The library-mapping point that matters most: PyTorch SDPA's `EFFICIENT_ATTENTION` backend (`torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION`) descends directly from these xFormers CUTLASS kernels, so when SDPA does not select the `FLASH_ATTENTION` backend, this is often the path it falls back to. Today `memory_efficient_attention` is mostly worth reaching for directly in two cases: on older GPUs where prebuilt `flash-attn` wheels are unavailable or unsupported — e.g., Turing/Volta (sm < 80) — since it has broader architecture/head-dim coverage, and in diffusion stacks, where Hugging Face `diffusers` exposes `pipe.enable_xformers_memory_efficient_attention()` and it long predated SDPA becoming the default. On modern Ampere/Hopper training and LLM inference, prefer `scaled_dot_product_attention` or `flash-attn` directly, which dispatch to the FA2/FA3 kernels documented above.

**Forward vs backward.** All the FA2/FA3 ideas apply to the *forward* pass cleanly. The backward pass is harder: it has more matmuls (five vs two), more intermediate quantities, and a different optimal parallelization (over keys, as noted). FA3's FP8 backward in particular is delicate; many production setups run the forward in FP8 but keep the backward in bf16.

**Decode is a different regime.** During autoregressive *inference decode* the query is a single token ($B_r = 1$), so the $QK^\top$ matmul is a matrix-vector product — memory-bound, not compute-bound. FA2's seqlen parallelism over queries does nothing when there is one query. The right tool is **FlashDecoding**, which parallelizes over the *KV* sequence (split the long KV cache across SMs, each computes a partial softmax, then combine) to keep all SMs busy. This is the same online-softmax combine logic, applied along a different axis. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) and [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

**Head dimension matters.** These kernels are tuned for head dimensions up to 128 or 256. Very large head dims, or unusual ones, may fall off the fast path and dispatch to a slower fallback. Variants like GQA/MQA/MLA change the effective shapes ([Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)); modern kernels have GQA-aware variants that avoid replicating KV.

**Alternatives to attention.** None of this helps the *asymptotic* $O(N^2)$ scaling — FlashAttention makes the constant factor and memory tiny, but the work still grows quadratically in sequence length. Sub-quadratic architectures (state-space models, linear attention) attack the asymptotics directly and are covered in [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html). In practice, FlashAttention-style exact attention remains dominant because its constant is so small that quadratic-but-fast beats subquadratic-but-clunky up to quite long contexts.

!!! tip "Practitioner tip: you almost never write these kernels"
    The right move is to *use* the kernel through `torch.nn.functional.scaled_dot_product_attention`, the `flash-attn` package, vLLM/SGLang's attention backends, or cuDNN — and to make sure your tensors satisfy the fast-path conditions (supported dtype, contiguous, head_dim in range, correct mask type). When you profile and find attention is the bottleneck, first check *which backend dispatched* before reaching for a custom kernel. Writing a competitive Hopper attention kernel is a multi-month CUTLASS/`wgmma` project; see [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html) and [CUDA Programming Essentials for ML Engineers](../04-kernels-efficiency/05-cuda-essentials.html) for the tools and the ceiling on what hand-written kernels can reach.

!!! interview "Interview Corner"
    **Q:** FlashAttention-1 already minimized HBM traffic. So why is FlashAttention-2 roughly 2× faster on the *same* GPU, and why does it take a *third* kernel (FA3) to get most of the remaining speedup on H100?

    **A:** FA1 was memory-optimal but compute-*under*-utilized — it ran at ~25–40% of tensor-core peak. FA2 raises utilization without touching memory traffic via three work-partitioning changes: (1) it defers the softmax normalization so the inner loop spends fewer cycles on slow non-matmul (transcendental) ops, (2) it parallelizes the forward over query blocks along the sequence dimension so the grid has enough thread blocks to fill all SMs even at small batch/heads, and (3) it switches warp partitioning from split-K to split-Q, so warps own disjoint query rows and need no inter-warp reduction through shared memory each iteration. Those are *occupancy and instruction-mix* wins, not bandwidth wins.

    FA2 still assumes Ampere's *synchronous* tensor-core instruction (`mma`): issue and wait. Hopper adds asynchronous `wgmma`, the TMA copy engine, and FP8 tensor cores — none of which FA2's structure can exploit. FA3 is a rewrite that (a) splits warps into TMA *producers* and compute *consumers* (a software pipeline so data movement overlaps compute), (b) overlaps the two matmuls with the softmax via ping-pong scheduling across warpgroups and intra-warpgroup pipelining, hiding the slow $\exp$ behind tensor-core work, and (c) runs the matmuls in FP8 — using incoherent processing (a Hadamard rotation that leaves $QK^\top$ invariant but spreads outliers) to keep accuracy. So: FA2 fixes Ampere occupancy; FA3 unlocks Hopper-specific async hardware and low precision.

!!! key "Key Takeaways"
    - FlashAttention-1 was memory-optimal but only ~25–40% of compute peak; FA2 and FA3 are about **compute utilization**, not memory traffic.
    - FA2's three edits: **defer normalization** (less non-matmul work on the critical path), **parallelize the forward over query blocks / sequence length** (fill all SMs at small batch×heads), and **split-Q warp partitioning** (each warp owns query rows → no per-iteration inter-warp reduction). Backward parallelizes over keys instead, to keep $dK,dV$ reductions local.
    - On a GPU, FLOPs are not fungible: the ~1% of FLOPs that are softmax/$\exp$ can be ~15% of *runtime* because they run on slow special-function units — which is why minimizing and overlapping them matters.
    - FA3 targets Hopper with **warp specialization**: TMA-driven *producer* warps prefetch KV tiles into a shared-memory ring buffer while *consumer* warpgroups run `wgmma`, so data movement overlaps compute.
    - FA3 overlaps the two matmuls with softmax via **ping-pong scheduling** (one warpgroup does softmax on SFUs while another does GEMM on tensor cores) and intra-warpgroup pipelining — possible only because Hopper's `wgmma` is asynchronous.
    - **FP8 attention** roughly doubles matmul throughput; **incoherent processing** (a random-sign Hadamard rotation $M$ with $M^\top M = I$, so $QK^\top$ is unchanged) spreads outliers across coordinates so the FP8 scale stays tight. Inputs are FP8 but accumulation and softmax stats stay in FP32.
    - These kernels keep attention at $O(N^2)$ work but with a tiny constant; **decode** (single-token query) is memory-bound and wants **FlashDecoding** (split-KV) instead, and subquadratic alternatives are a separate design axis.
    - In practice you call the kernel (`scaled_dot_product_attention`, `flash-attn`, vLLM/cuDNN) — the engineering lesson is to verify *which backend dispatched* on your hardware before assuming you got the fast path.

!!! sota "State of the Art & Resources (2026)"
    FlashAttention has evolved through four generations — from IO-aware tiling (FA1) through work-partitioning on Ampere (FA2), Hopper warp specialization and FP8 (FA3), to algorithm/kernel co-design for Blackwell's asymmetric hardware (FA4) — and remains the dominant exact-attention implementation in production training and inference stacks.

    **Foundational work**

    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — the original IO-aware tiled kernel that eliminated HBM materialization of the $N\times N$ score matrix.
    - [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691) — deferred rescaling, sequence-length parallelism, and split-Q warp partitioning that lift A100 utilization to 50–73% of bf16 peak.

    **Recent advances (2023–2026)**

    - [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608) — TMA producer/consumer warp specialization, wgmma overlap of matmul and softmax, and FP8 with incoherent processing reaching 740 TFLOP/s on H100.
    - [Zadouri et al., *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling* (2026)](https://arxiv.org/abs/2603.05451) — fully asynchronous MMA pipelines and software-emulated exponentials for NVIDIA Blackwell, achieving 1613 TFLOP/s (71% utilization) on B200.
    - [Chee et al., *QuIP: 2-Bit Quantization of Large Language Models With Guarantees* (2023)](https://arxiv.org/abs/2307.13304) — origin of incoherent processing via random orthogonal/Hadamard rotation that FA3 borrows to tame FP8 outliers.

    **Open-source & tools**

    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — official FA2/FA3/FA4 kernels with GQA, causal, variable-length, and ROCm support; the implementation virtually every framework calls.
    - [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — CUDA C++ templates and CuTe DSL providing the wgmma, TMA, and FP8 abstractions that FA3/FA4 are built on; essential reading for writing Hopper/Blackwell kernels.

    **Go deeper**

    - [Dao, Haziza, Massa, Sizov — *Flash-Decoding for long-context inference* (2023)](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) — split-KV parallelism that makes decode-time attention fast when the query is a single token and the KV cache is long.
    - [PyTorch Blog — *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://pytorch.org/blog/flashattention-3/) — accessible technical walkthrough of FA3's three key techniques with context on PyTorch's SDPA dispatch.
    - [ICLR Blogposts 2026 — *The Evolution of FlashAttention*](https://iclr-blogposts.github.io/2026/blog/2026/the-evolution-of-flashattention/) — end-to-end survey from FA1 through FA4 with roofline analysis, backward-pass derivations, and coverage of Ring Attention and block-sparse extensions.

## Further reading

- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023) — the definitive source for the deferred-rescale, sequence-parallel, and split-Q warp-partitioning ideas.
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao, *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024) — warp specialization, TMA producer/consumer, `wgmma` overlap, and FP8 with incoherent processing.
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) — the original; read alongside [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html).
- Tri Dao et al., *Flash-Decoding for long-context inference* (2023) — the split-KV technique for the decode regime.
- Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa, *QuIP: 2-Bit Quantization of Large Language Models With Guarantees* (2023) — origin of incoherent processing via random orthogonal/Hadamard rotations.
- NVIDIA, *Hopper Architecture Whitepaper* and the CUTLASS library / CUDA programming guide — authoritative references for `wgmma`, TMA, asynchronous barriers, and FP8 tensor cores.
- The `Dao-AILab/flash-attention` repository — production kernels and the GQA/causal/varlen variants you will actually call.

## Exercises

**1.** FlashAttention-1 already minimized HBM traffic to the theoretical minimum for exact attention. Yet FA2 is roughly $2\times$ faster on the *same* A100. Explain in one or two sentences why the two facts are not contradictory, and name the bottleneck FA2 actually attacks.

??? note "Solution"
    They are not contradictory because FA1's bottleneck was never memory bandwidth — it was **compute utilization**. FA1 ran at only ~25-40% of the A100's tensor-core FLOP peak while a well-tuned dense GEMM hits 80-90%. The idle time came from **occupancy and instruction mix**: SMs sitting empty (too few thread blocks) and cycles spent on slow non-matmul work (rescales, `exp`, inter-warp reductions through shared memory) rather than on the tensor cores. FA2's three edits — deferred normalization, sequence-length parallelism, and split-Q warp partitioning — raise utilization without changing the HBM traffic at all. Minimizing memory traffic and maximizing compute utilization are two orthogonal axes; FA1 nailed the first and left the second on the table.

**2.** In the FA2 reference recurrence, the per-iteration correction factor $\alpha = e^{m - m^{\text{new}}}$ is applied to *both* the running denominator $\ell$ and the unnormalized accumulator $\tilde{O}$, but the division by $\ell$ is deferred to the end. Suppose a well-meaning "optimizer" also tries to defer the $\alpha$ multiplication on $\tilde{O}$ (i.e., drops the `alpha.unsqueeze(1) * Oi` term and just does `Oi = Oi + Pij @ vj`). Does the kernel still compute exact attention? Explain precisely what breaks.

??? note "Solution"
    No — dropping the $\alpha$ multiplication on $\tilde{O}$ produces **wrong** results. The two factors play different roles and only one can be deferred.

    - The **division by $\ell$** can be deferred because it is a single global scaling applied identically to every term of the sum $\sum_j P_{ij} V_j$. Multiplying the final accumulator by $1/\ell$ is algebraically identical to dividing each contribution as it arrives, so you can safely wait until the end.
    - The **$\alpha$ correction cannot be deferred**, because each inner iteration recomputes $P_{ij} = \exp(S_{ij} - m^{\text{new}})$ against the *current* running max $m^{\text{new}}$, which changes from block to block. Contributions accumulated under an *older, smaller* max are exponentiated with the wrong offset; $\alpha = e^{m^{\text{old}} - m^{\text{new}}}$ retroactively rescales every already-accumulated term so it is consistent with the new max. If you drop it, early blocks' contributions are left multiplied by $e^{m^{\text{old}}}$ while later blocks carry $e^{m^{\text{new}}}$, and the sum mixes incompatible scales.

    Concretely: after processing block 1 with max $m_1$, $\tilde{O} = \sum P^{(1)} V_1$ where $P^{(1)} = e^{S_1 - m_1}$. If block 2 raises the max to $m_2 > m_1$, the correct running accumulator is $e^{m_1 - m_2}\sum P^{(1)} V_1 + \sum e^{S_2 - m_2} V_2$. Without $\alpha$ you would compute $\sum e^{S_1 - m_1} V_1 + \sum e^{S_2 - m_2} V_2$, whose two halves are normalized against different maxima — not any valid softmax. The denominator $\ell$ would also then be inconsistent with the numerator. So $\alpha$ on $\tilde{O}$ is load-bearing; only the $\ell$ division is deferrable.

**3.** *(Quantitative.)* Reproduce and extend the chapter's "FLOPs are not fungible" calculation for a smaller head. Take one attention head, full (non-causal) attention, sequence length $N = 4096$, head dimension $d = 64$. Assume the two matmuls cost $4N^2 d$ FLOPs total, the non-matmul softmax work is $\approx 5N^2$ FLOPs, tensor cores run at $312$ TFLOP/s, and the special-function unit doing `exp`/max runs at $20$ TFLOP/s. Compute (a) the matmul time, (b) the non-matmul time, and (c) the non-matmul share of total runtime *assuming no overlap*. Then (d) state what the number becomes if FA3's ping-pong scheduling perfectly overlaps the softmax with matmul.

??? note "Solution"
    (a) Matmul FLOPs and time:

    $$
    4 N^2 d = 4 \cdot (4096)^2 \cdot 64 = 4 \cdot 1.6777 \times 10^7 \cdot 64 \approx 4.295 \times 10^{9}\ \text{FLOPs}.
    $$

    $$
    t_{\text{mm}} = \frac{4.295 \times 10^{9}}{3.12 \times 10^{14}\ \text{FLOP/s}} \approx 1.377 \times 10^{-5}\ \text{s} \approx 13.8\ \mu s.
    $$

    (b) Non-matmul FLOPs and time:

    $$
    5 N^2 = 5 \cdot (4096)^2 \approx 8.389 \times 10^{7}\ \text{FLOPs}.
    $$

    $$
    t_{\text{sfu}} = \frac{8.389 \times 10^{7}}{2.0 \times 10^{13}\ \text{FLOP/s}} \approx 4.19 \times 10^{-6}\ \text{s} \approx 4.2\ \mu s.
    $$

    (c) With no overlap the total is $t_{\text{mm}} + t_{\text{sfu}} \approx 13.8 + 4.2 = 18.0\ \mu s$, so the non-matmul share is

    $$
    \frac{4.2}{18.0} \approx 0.23 = 23\%.
    $$

    Note the softmax is still only ~1.9% of the FLOP *count* ($8.39\times10^7$ of $4.38\times10^9$), yet ~23% of the *time* — and the share is *larger* than the chapter's $d=128$, $N=8192$ example (~14%) because softmax work scales as $N^2$ while matmul work scales as $N^2 d$, so shrinking $d$ makes the non-matmul fraction of time grow.

    (d) If ping-pong scheduling perfectly overlaps the softmax behind matmul work, the runtime becomes $\max(t_{\text{mm}}, t_{\text{sfu}}) = 13.8\ \mu s$ (the tensor cores are the long pole and the SFU work hides completely underneath them). The softmax's contribution to wall-clock time drops from 23% to effectively 0%, which is exactly the win FA3 chases. (In practice overlap is imperfect, but this is the ceiling.)

**4.** In FA2's split-Q warp layout each warp owns a slice of the $B_r$ query rows and needs no inter-warp reduction, whereas FA1's split-K layout forced a shared-memory reduction plus a `__syncthreads()` every inner iteration. (a) Why does split-Q eliminate the reduction — what property of the query-row partition makes the warps independent? (b) What does split-Q now *require* of the $K_j, V_j$ tiles, and why is shared memory well suited to satisfying it? (c) The chapter says the backward pass instead parallelizes over *key* blocks. Give the one-sentence reason the forward and backward pick opposite axes.

??? note "Solution"
    (a) The softmax and the output for a given query row are computed entirely from that row's score vector against *all* keys; different query rows never share a partial sum. So partitioning the tile by query rows gives each warp a set of **completely independent output rows** — warp 0 produces the final output for rows 0-15, warp 1 for rows 16-31, etc. There is nothing to combine across warps. Split-K, by contrast, gave each warp a slice of key *columns*, so each warp held only a *partial* sum over its column slice; the final output for any row is a sum across all warps' partials, which forces the shared-memory reduction and barrier.

    (b) Split-Q requires that every warp have access to the *full* $K_j, V_j$ tile (each warp computes full-width scores $Q_{\text{rows}} K_j^\top$ and the full $P_{ij} V_j$ for its rows). Shared memory is ideal because this is a **broadcast read**: all warps read the same $K_j/V_j$ bytes, read-only, with no write-back — exactly the access pattern shared memory handles efficiently (no bank-conflict-inducing writes, no synchronization needed among readers).

    (c) You parallelize over the axis that makes the reduction *local*: the forward reduces over keys (so make queries the independent/parallel axis), while the backward must accumulate $dK$ and $dV$, which are reductions *over query blocks* — so making key/value blocks the parallel axis keeps each thread block's $dK_j, dV_j$ accumulation local and avoids atomic adds.

**5.** *(Implementation.)* The reference `flash_attention_2_reference` in the chapter handles a full and a causal mask. Extend it to support a **sliding-window** (local) attention mask of width $w$: query position $q$ may attend only to key positions $k$ with $q - w < k \le q$ (a causal window of the last $w$ keys, including itself). Your implementation must (a) still skip entire key blocks that fall completely outside the window (the load-balancing point from the chapter's causal aside), and (b) mask the partial-overlap blocks correctly. Verify against a dense reference.

??? note "Solution"
    The key ideas: a key block $[j, j+B_c)$ is entirely outside the window for query block $[i, i+B_r)$ if it is all in the future ($j > i + B_r - 1$, same as causal) *or* entirely too far in the past — the newest query row is $i + B_r - 1$ and it can see back only to $k > (i+B_r-1) - w$, so if the block's last key $j + B_c - 1 \le (i) - w$... more carefully, the *oldest* query row $i$ sees back to $k > i - w$, so a block is fully in the past when its newest key $j + B_c - 1 < i - w + 1$, i.e. $j + B_c - 1 \le i - w$. Within a surviving block we apply both the causal bound $k \le q$ and the window bound $k > q - w$.

    ```python
    import torch

    def flash_attention_2_sliding_window(Q, K, V, w, Br=64, Bc=64):
        """
        Exact FA2 recurrence with a sliding-window causal mask of width w:
        query q attends to keys k with (q - w) < k <= q.
        Skips key blocks fully outside the window; masks partial-overlap blocks.
        """
        N, d = Q.shape
        O = torch.zeros_like(Q)
        for i in range(0, N, Br):
            qi = Q[i:i+Br]
            br = qi.shape[0]
            Oi = torch.zeros(br, d)
            mi = torch.full((br,), -float('inf'))
            li = torch.zeros(br)

            q_lo, q_hi = i, i + br - 1                     # query row range in this block
            for j in range(0, N, Bc):
                kj = K[j:j+Bc]; vj = V[j:j+Bc]
                bc = kj.shape[0]
                k_lo, k_hi = j, j + bc - 1                 # key range in this block

                # ---- block-level skip (load balancing) ----
                # entirely in the future: even the last query can't reach k_lo
                if k_lo > q_hi:
                    break                                 # all later blocks are future too
                # entirely past the window: even the oldest query (q_lo) can't reach k_hi
                # q_lo attends to k > q_lo - w, so block is dead if k_hi <= q_lo - w
                if k_hi <= q_lo - w:
                    continue

                Sij = (qi @ kj.T) / (d ** 0.5)            # (br, bc)
                qpos = torch.arange(q_lo, q_lo + br).unsqueeze(1)   # (br,1)
                kpos = torch.arange(k_lo, k_lo + bc).unsqueeze(0)   # (1,bc)
                # causal upper bound AND window lower bound
                mask = (kpos > qpos) | (kpos <= qpos - w)
                Sij = Sij.masked_fill(mask, -float('inf'))

                mij = Sij.max(dim=1).values
                mi_new = torch.maximum(mi, mij)
                # A block can survive the block-level skip yet still leave SOME
                # query rows fully masked (e.g. the newest rows of this query
                # block vs. an old key block). Those rows have mi_new = -inf and
                # exp(-inf - (-inf)) = nan. Guard them: they contribute nothing,
                # so force Pij -> 0 and alpha -> 1 (their Oi, li stay 0).
                dead = torch.isinf(mi_new)
                safe = torch.where(dead, torch.zeros_like(mi_new), mi_new)
                Pij = torch.exp(Sij - safe.unsqueeze(1)).masked_fill(dead.unsqueeze(1), 0.0)
                alpha = torch.where(dead, torch.ones_like(mi), torch.exp(mi - safe))
                li = alpha * li + Pij.sum(dim=1)
                Oi = alpha.unsqueeze(1) * Oi + Pij @ vj
                mi = mi_new

            O[i:i+br] = Oi / li.unsqueeze(1)
        return O


    if __name__ == "__main__":
        torch.manual_seed(0)
        N, d, w = 256, 64, 48
        Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)

        # dense reference with the same sliding-window mask
        S = Q @ K.T / d**0.5
        q = torch.arange(N).unsqueeze(1); k = torch.arange(N).unsqueeze(0)
        allowed = (k <= q) & (k > q - w)
        S = S.masked_fill(~allowed, -float('inf'))
        ref = torch.softmax(S, dim=-1) @ V

        out = flash_attention_2_sliding_window(Q, K, V, w)
        print("max abs error:", (ref - out).abs().max().item())   # ~1e-6
    ```

    Notes on correctness of the skip logic. The future test `k_lo > q_hi` lets us `break` (all subsequent blocks have larger `k_lo`, so they are future too), exactly the causal load-balancing shortcut from the chapter. The past test `k_hi <= q_lo - w` uses `continue` (not `break`) because later blocks move *forward* in time and re-enter the window. The `dead`-row guard is the one subtlety people miss: the block-level skips are decided from the *extreme* query rows ($q_{\text{lo}}, q_{\text{hi}}$), so a block can survive yet still leave *some* rows in the middle of the query block with no in-window key inside *that particular block* — e.g. the newest rows of query block $[64,128)$ have no window overlap with key block $[0,64)$ even though the oldest rows do. Those rows get `mi_new = -inf` and would compute `exp(-inf - (-inf)) = nan` without the guard. Globally every query $q$ still attends to at least the diagonal $k = q$ (for $w \ge 1$), so $\ell > 0$ and the final division is safe; the guard only handles the per-block gap. The result matches the dense reference to floating-point error (~$10^{-6}$), confirming exact windowed attention.
