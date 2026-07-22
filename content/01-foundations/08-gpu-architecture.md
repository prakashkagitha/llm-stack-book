# 1.8 GPU Architecture & The Memory Hierarchy

Every claim you will ever read about LLM training and inference — "this kernel is memory-bound," "decode is bandwidth-limited," "we need to keep the tensor cores fed," "the KV cache spilled to a slower tier" — is downstream of the hardware described in this chapter. The GPU is not a magic box that does matrix multiplication fast. It is a very specific machine with a very specific shape: thousands of arithmetic units starved by a comparatively thin pipe to memory, organized into a strict hierarchy of caches and scratchpads, executing instructions in lockstep groups of 32. If you understand that shape, the entire performance-engineering half of this book becomes mechanical. If you do not, you will spend your career confused about why your "compute-heavy" workload runs at 5% of peak FLOPs.

This is the single most important systems chapter in the book. Almost everything in [Part IV — Kernels, Efficiency & Quantization](../04-kernels-efficiency/01-roofline-performance.html) and [Part VII — Inference & Serving](../07-inference-serving/01-anatomy-inference.html) is an exercise in respecting the constraints we lay out here. We will build up the GPU from threads to warps to streaming multiprocessors (SMs); walk down the memory hierarchy from registers to high-bandwidth memory (HBM); develop the central concept of **arithmetic intensity** that decides whether you are compute-bound or memory-bound; and ground all of it in the concrete specifications of the NVIDIA A100, H100, H200, and B200 datacenter GPUs. We assume you know Python and the linear algebra of [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html); we explain everything GPU-specific.

Cross-references: [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) explains the FP16/BF16/FP8 number formats whose throughput we quote here; [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html) takes the NVLink and multi-GPU story to the cluster scale; [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) formalizes arithmetic intensity into the roofline; [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) is the canonical case study of trading FLOPs for memory traffic; and [CUDA Programming Essentials for ML Engineers](../04-kernels-efficiency/05-cuda-essentials.html) turns the execution model below into real kernels.

---

## Why a GPU Looks the Way It Does

A CPU is optimized for **latency**: finish one thread's instruction stream as fast as possible. It spends most of its transistor budget on machinery to make a single thread fast — large caches, branch predictors, out-of-order execution, deep speculation. A modern CPU core might have a few hundred kilobytes of private cache and devote enormous silicon to *avoiding* memory stalls for one thread.

A GPU is optimized for **throughput**: finish an enormous number of independent threads per second, and tolerate each individual thread being slow. It spends its transistor budget on arithmetic units and on the ability to keep thousands of threads in flight simultaneously. When one thread stalls waiting on memory, the hardware instantly switches to another ready thread. This is **latency hiding through massive parallelism**, and it is the single design decision from which everything else follows.

The consequence: a GPU only goes fast when you give it tens of thousands of independent units of work. A matrix multiply of two large matrices is perfect — billions of independent multiply-accumulates. Autoregressive decode of one token at a time for a single request is nearly the worst case — almost no parallelism, so the machine sits idle hiding latency it cannot hide. Half of inference engineering ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)) is manufacturing parallelism the workload does not naturally have.

```text
        CPU                                  GPU
  +---------------+              +-------------------------------+
  | big control   |              | tiny control, replicated many |
  | huge caches   |              | small caches, many ALUs        |
  | few fat cores |              | thousands of thin "cores"      |
  +---------------+              +-------------------------------+
  goal: 1 thread fast           goal: 10,000s of threads in flight
  hides latency by              hides latency by SWITCHING to
  predicting / caching          another ready warp every cycle
```

A GPU is also a **co-processor**. It hangs off the CPU (the "host") over PCIe or, on Grace-Hopper-class systems, a fast chip-to-chip link. The host launches **kernels** (GPU functions) and shuffles data to and from GPU memory. A recurring performance bug is a workload that is secretly host-bound or PCIe-bound: the GPU is fast but it spends its life waiting for the CPU to feed it.

---

## The Execution Model: Threads, Warps, Blocks, and SMs

NVIDIA's programming model (CUDA — Compute Unified Device Architecture) exposes a hierarchy of parallel work. Understanding it is non-negotiable.

### Threads and the SIMT model

The smallest unit is a **thread**. A thread runs the kernel body once, with its own registers and its own index, and typically computes one element (or a few) of the output. You launch many thousands of them.

Threads are grouped into **warps** of exactly **32 threads**. This number is a hardware constant on every NVIDIA GPU to date and it dominates everything. The 32 threads in a warp execute **in lockstep**: they share one instruction fetch/decode unit and one program counter, and on each cycle all 32 lanes execute the *same* instruction on *different* data. NVIDIA calls this **SIMT** — Single Instruction, Multiple Threads. It is SIMD (Single Instruction, Multiple Data) with a thread-shaped programming abstraction layered on top.

Two consequences of the warp being the real unit of execution:

- **Warp divergence.** If threads in a warp take different sides of an `if`, the warp executes *both* paths serially, masking off the inactive lanes on each path. A branch that splits a warp 50/50 can halve throughput. Branchy, data-dependent code is GPU-hostile.
- **Coalesced memory access.** When a warp issues a load, the hardware tries to service all 32 lanes with as few memory transactions as possible. If lane $i$ reads address `base + i*4` (32 consecutive 4-byte words), the whole warp is satisfied by one or two 128-byte transactions — **coalesced**, near-peak bandwidth. If the 32 lanes read scattered addresses, you get up to 32 separate transactions and a 32× bandwidth penalty.

### Blocks (CTAs) and the SM

Warps are grouped into a **thread block**, also called a **cooperative thread array (CTA)**. A block is the unit of scheduling and the unit of *cooperation*: threads within a block can synchronize with a barrier (`__syncthreads()`) and share data through a fast on-chip scratchpad called **shared memory (SMEM)**. Threads in *different* blocks cannot cheaply synchronize or share scratchpad; this isolation is what lets the GPU run blocks independently and scale.

Each block is assigned to exactly one **streaming multiprocessor (SM)** for its entire life. The SM is the fundamental compute building block of the GPU — think of it as a core, but a wide, throughput-oriented one. An H100 has 132 SMs; an A100 has 108. Inside one SM you find:

- A register file (e.g. 65,536 32-bit registers — 256 KB — per SM), partitioned among all resident threads.
- A pool of **CUDA cores**: scalar ALUs for FP32/INT32 arithmetic.
- **Tensor Cores**: dedicated matrix-multiply-accumulate units (more below).
- A block of on-chip SRAM split between **shared memory** and **L1 cache** (configurable, e.g. up to 228 KB total per SM on H100).
- **Warp schedulers** (typically 4 per SM) that pick a ready warp each cycle and issue its next instruction.

```text
              GPU (e.g. H100: 132 SMs)
   +-----------------------------------------------+
   |  SM 0   SM 1   SM 2   ...            SM 131    |
   |   ||      ||                                   |
   |   ||  shared by all SMs:  L2 cache (~50 MB)    |
   |   ||              |                            |
   +-----------------------------------------------+
                       |
                  HBM (off-chip DRAM, 80 GB)

   Inside ONE SM:
   +-----------------------------------------------+
   |  4 warp schedulers                            |
   |  Register file (256 KB)                       |
   |  CUDA cores (FP32/INT)   Tensor Cores         |
   |  Shared memory / L1 (configurable SRAM)       |
   +-----------------------------------------------+

   Software hierarchy:    Grid  >  Block(CTA)  >  Warp(32)  >  Thread
   Hardware mapping:      GPU   >  SM          >  scheduler >  lane
```

### The grid and the launch

A kernel launch creates a **grid** of blocks. You choose the grid dimensions and block dimensions. The GPU's global scheduler hands blocks out to SMs as they free up. Because there are usually far more blocks than SMs, this gives automatic load balancing and forward scalability: the same kernel runs on a small GPU (few SMs) or a large one (many SMs) with no code change — bigger GPUs simply chew through more blocks concurrently.

{{fig:execution-model-warp-simt}}

```python
# A from-scratch mental model of the GPU execution hierarchy in pure Python.
# This is NOT how you run on a GPU (you'd use CUDA/Triton) -- it is a faithful
# simulation of the indexing and lockstep semantics so the model is concrete.

WARP_SIZE = 32

def launch_kernel(grid_dim, block_dim, kernel, *args):
    """Emulate a CUDA grid launch: grid_dim blocks, each of block_dim threads."""
    for block_id in range(grid_dim):
        # Each block is independent -> could run on any SM, in any order.
        block_threads = list(range(block_dim))
        # Threads execute in warps of 32, in lockstep within a warp.
        for warp_start in range(0, block_dim, WARP_SIZE):
            warp = block_threads[warp_start: warp_start + WARP_SIZE]
            # All lanes in this warp run the SAME instruction stream together.
            for thread_id in warp:               # in HW: simultaneous, not serial
                global_tid = block_id * block_dim + thread_id
                kernel(global_tid, block_id, thread_id, *args)

def saxpy(global_tid, block_id, thread_id, a, x, y, out, n):
    """out[i] = a * x[i] + y[i], one element per thread (the classic 'SAXPY')."""
    i = global_tid
    if i < n:                                    # guard: grid may overshoot n
        out[i] = a * x[i] + y[i]

n = 100
x = [float(i) for i in range(n)]
y = [1.0] * n
out = [0.0] * n
block_dim = 32                                   # one warp per block here
grid_dim = (n + block_dim - 1) // block_dim      # ceil-div = 4 blocks for n=100
launch_kernel(grid_dim, block_dim, saxpy, 2.0, x, y, out, n)
print(out[:5])     # [1.0, 3.0, 5.0, 7.0, 9.0]  == 2*x + 1
```

This `saxpy` is the canonical *memory-bound* kernel: for every element it does 2 FLOPs (one multiply, one add) but reads 8 bytes (`x[i]`, `y[i]`) and writes 4 bytes. We will quantify exactly why that is bad shortly.

---

## CUDA Cores vs Tensor Cores

There are two fundamentally different arithmetic engines on a modern datacenter GPU, and conflating them is the most common beginner error in performance reasoning.

### CUDA cores: scalar, general-purpose

A **CUDA core** is a scalar ALU. Each one does one fused multiply-add (FMA) per cycle on FP32 (or FP64 on the smaller FP64 units, or INT32). When you write `c = a * b + d` in a kernel on plain floats, that runs on CUDA cores. The FP32 throughput of a GPU is roughly (number of FP32 lanes) × (clock) × 2 (the FMA counts as 2 FLOPs). On an A100 this is on the order of 19.5 TFLOP/s of FP32. Respectable, but not where the headline numbers come from.

### Tensor Cores: matrix-multiply-accumulate engines

A **Tensor Core** is a small systolic-array-like unit that computes an entire small matrix multiply-accumulate per operation: $D = A \times B + C$ where $A$, $B$, $C$, $D$ are small tiles (e.g. $16\times16$). Instead of issuing $16^3$ scalar FMAs, one Tensor Core instruction consumes whole tiles and produces a whole tile. This is why the BF16/FP16 Tensor Core throughput of an A100 (on the order of 312 TFLOP/s) is **16×** its FP32 CUDA-core throughput. The H100 pushes this further with FP8 Tensor Cores; the Blackwell B200 adds FP4. Every modern training and inference workload lives or dies on Tensor Core utilization. The Tensor Core is a bolt-on matrix-multiply unit dropped into an otherwise general-purpose, programmable SIMT core; the opposite design point — a chip that is almost entirely systolic array, driven by a compiler instead of a warp/SMEM hierarchy — is Google's TPU (and AWS Trainium), and that systolic-array-versus-SIMT contrast is developed in [The Accelerator Landscape: TPUs, Trainium, AMD/ROCm & Gaudi](../01-foundations/10-accelerator-landscape.html).

The catch: Tensor Cores only accelerate **matrix multiplication** (and operations you can phrase as one), and only in **reduced precision** (BF16/FP16/FP8/FP4 inputs, usually with FP32 accumulation). They want tile-shaped, contiguous data with dimensions that are multiples of 8 or 16. A pointwise operation (add, GELU, layernorm) gets *zero* benefit from Tensor Cores — it runs on CUDA cores and is almost always memory-bound. This split — matmuls on Tensor Cores, everything else on CUDA cores and bandwidth-limited — is the structural reason kernel fusion ([Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)) matters so much.

{{fig:cuda-core-vs-tensor-core}}

!!! note "Aside: how the throughput numbers multiply up"
    Vendors quote "sparse" Tensor Core numbers (assuming 2:4 structured sparsity) that double the dense figure. Always check whether a headline TFLOP/s is dense or sparse, and which precision. A "1979 TFLOP/s" H100 number is FP16 *with sparsity*; the dense FP16 figure is about half. For honest roofline math, use dense numbers in the precision you actually run.

```python
import torch

# Demonstrate the CUDA-core vs Tensor-core gap empirically (run on a GPU).
# FP32 matmul uses CUDA cores; BF16 matmul uses Tensor Cores.
def benchmark_matmul(dtype, n=8192, iters=50):
    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    _ = a @ b                                  # warm up / autotune
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    flops = 2 * n**3                           # 2*n^3 FLOPs for n x n matmul
    tflops = flops / (ms * 1e-3) / 1e12
    return ms, tflops

# On an A100 you would typically see something on the order of:
#   FP32 : ~19  TFLOP/s   (CUDA cores)
#   BF16 : ~280 TFLOP/s   (Tensor Cores) -- over 10x faster on the SAME hardware
for dt in (torch.float32, torch.bfloat16):
    res = benchmark_matmul(dt)
    if res:
        ms, tflops = res
        print(f"{str(dt):16s}  {ms:7.2f} ms  {tflops:7.1f} TFLOP/s")
```

The reason `torch.set_float32_matmul_precision("high")` and BF16 autocast exist is precisely to route FP32 *logical* math onto the Tensor Cores (via TF32 or BF16) and capture that order-of-magnitude speedup. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

---

## The Memory Hierarchy: Registers, SMEM, L2, and HBM

Compute is cheap; moving data is expensive. The GPU memory hierarchy is a series of tiers, each roughly an order of magnitude larger and an order of magnitude slower than the one above it. The art of GPU kernel writing is keeping data in the fast tiers and minimizing trips to the slow ones.

{{fig:memory-hierarchy}}

| Tier | Scope | Approx. size (H100-class) | Approx. bandwidth | Approx. latency |
|------|-------|---------------------------|-------------------|-----------------|
| Registers | per-thread | 256 KB / SM (65,536 × 32-bit) | ~tens of TB/s (per SM) | ~1 cycle |
| Shared memory / L1 | per-block (per-SM SRAM) | up to ~228 KB / SM | ~tens of TB/s (per SM) | ~20–30 cycles |
| L2 cache | whole GPU | ~50 MB | ~tens of TB/s aggregate | ~200 cycles |
| HBM (global/DRAM) | whole GPU | 80 GB | ~3.35 TB/s | ~400–800 cycles |
| NVLink (to peer GPU) | node | — | ~450–900 GB/s/dir | microseconds |
| PCIe (to host) | node | — | ~32–64 GB/s/dir (Gen4–Gen5 x16; ~128 GB/s bidir on Gen5) | microseconds |

The numbers are illustrative and vary by exact SKU and clock, but the *ratios* are the point: SMEM is roughly an order of magnitude faster than L2, which is faster than HBM, which dwarfs PCIe. Each step down is a cliff.

### Registers

Each thread has private registers. These are the only operands the ALUs read directly, and accessing them is effectively free. The register file is a fixed resource per SM (e.g. 65,536 32-bit registers). If each thread uses 64 registers, an SM can host at most $65536 / 64 = 1024$ threads. Use too many registers per thread and you reduce **occupancy** (next section). "Register spilling" — when the compiler runs out of registers and pushes variables to slow "local memory" (which actually lives in L1/L2/HBM) — is a silent performance killer.

### Shared memory (SMEM): the programmer-managed scratchpad

**Shared memory** is fast on-chip SRAM, private to a thread block, and — crucially — **explicitly managed by you**, the kernel author. It is not a cache that fills automatically; you stage data into it on purpose. The canonical use is **tiling** a matrix multiply: cooperatively load a tile of $A$ and a tile of $B$ from HBM into SMEM once, then have all threads in the block reuse those tiles many times from the fast scratchpad before fetching the next tile. This converts a flood of redundant HBM reads into a trickle. SMEM is physically banked (32 banks); if two lanes of a warp hit the same bank with different addresses you get a **bank conflict** and serialized access — the SMEM analogue of uncoalesced HBM access.

This staged reuse is the entire mechanism behind FlashAttention: keep the attention computation's working set in SMEM and registers, never materialize the full $S \times S$ attention matrix in HBM. See [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html).

### L2 cache

The **L2 cache** is shared by all SMs and caches HBM transparently. You do not manage it directly, but you exploit it by structuring access patterns so that data fetched by one SM is reused by another while still resident. On Hopper, the **L2 cache residency controls** and the new **thread block clusters / distributed shared memory** let blocks on nearby SMs share SMEM, effectively a programmable extension of the SMEM tier.

### HBM: the headline "GPU memory"

When someone says "an 80 GB H100," they mean its **HBM** — High Bandwidth Memory, stacked DRAM sitting on the same package as the GPU die, connected by an extremely wide bus. HBM is where your model weights, activations, optimizer states, and KV cache live. Its capacity bounds how big a model you can hold; its **bandwidth** bounds how fast you can stream those weights through the compute units. For LLM *decode*, HBM bandwidth is almost always the binding constraint, because each generated token must read the entire model's weights from HBM at least once and does very little arithmetic with them. This is the central fact of inference economics ([The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html)).

```text
   fast & tiny  ----------------------------------------->  slow & huge
   +-----------+   +------------+   +--------+   +-----------------+
   | registers | < | SMEM / L1  | < |   L2   | < |   HBM (DRAM)    |
   | 256 KB/SM |   | ~228 KB/SM |   | ~50 MB |   |     80 GB       |
   |  ~1 cyc   |   |  ~30 cyc   |   |~200 cyc|   |   ~500 cyc      |
   +-----------+   +------------+   +--------+   +-----------------+
        ^                ^                            |
        |   you stage    |   you cache-friendly       |  you minimize
        |   explicitly   |   structure access         |  total bytes moved
```

---

## Occupancy: Keeping the Machine Busy

The GPU hides memory latency by having many warps resident on each SM: when one warp stalls on a load, a warp scheduler instantly issues from another ready warp. **Occupancy** is the ratio of resident warps to the hardware maximum per SM. Higher occupancy generally means more latency-hiding headroom.

Occupancy is bounded by whichever per-SM resource runs out first:

- **Registers.** Resident threads × registers/thread ≤ register file size.
- **Shared memory.** Resident blocks × SMEM/block ≤ SMEM size per SM.
- **Block/warp slots.** A hardware cap on resident blocks and warps per SM.

$$
\text{occupancy} = \frac{\text{active warps per SM}}{\text{max warps per SM}}, \qquad \text{max warps per SM} = \frac{\text{max threads per SM}}{32}
$$

A worked occupancy calculation makes the trade-offs concrete.

!!! example "Worked example: occupancy as a resource-packing problem"
    Consider an SM with these (representative) limits: 65,536 32-bit registers, 100 KB of shared memory available, a cap of 2,048 resident threads (= 64 warps), and a cap of 32 resident blocks.

    You launch a kernel with **256 threads per block**, using **48 registers per thread** and **24 KB of shared memory per block**. How many blocks fit?

    - Register limit: each block needs $256 \times 48 = 12{,}288$ registers. The file holds $65{,}536$, so at most $\lfloor 65536 / 12288 \rfloor = 5$ blocks.
    - Shared-memory limit: $\lfloor 100{,}000 / 24{,}000 \rfloor = 4$ blocks.
    - Thread/block-count limit: $\lfloor 2048 / 256 \rfloor = 8$ blocks, well under the 32-block cap.

    The binding constraint is **shared memory**: 4 blocks fit. That is $4 \times 256 = 1{,}024$ threads $= 32$ warps resident, out of a maximum of 64. So **occupancy = 32 / 64 = 50%**.

    Want higher occupancy? Cut shared memory per block to 16 KB and now $\lfloor 100000/16000 \rfloor = 6$ blocks fit, but registers now bind at 5 blocks → $5\times256=1280$ threads = 40 warps = **62.5% occupancy** (5 blocks $\times$ 256 threads $= 1{,}280$ threads $= 40$ warps, out of 64). Drop registers to 32/thread *and* shared memory to 8 KB, and both soft limits lift: registers allow $\lfloor 65536/(256\times32)\rfloor = 8$ blocks and shared memory allows $\lfloor 100000/8000\rfloor = 12$ blocks, so the thread-count cap binds first at 8 blocks = 2,048 threads = **100% occupancy**. (Holding shared memory at 16 KB instead would leave *it* binding at 6 blocks = 75% — you must relieve **both** limits.) This is the daily reality of kernel tuning: you trade registers and shared memory against occupancy.

The crucial subtlety, established by Vasily Volkov's work, is that **maximum occupancy is not the goal — performance is**. A kernel that uses lots of registers per thread to hold a big accumulator tile (low occupancy) can dramatically outperform a high-occupancy kernel, because it gets more reuse per byte loaded. FlashAttention and high-performance GEMM kernels deliberately run at modest occupancy. Occupancy buys you *latency hiding*; if you already hide latency another way (instruction-level parallelism, big register tiles), you do not need much of it. Treat occupancy as one lever, not the objective.

```python
def max_blocks_per_sm(threads_per_block, regs_per_thread, smem_per_block_bytes,
                      regfile=65536, smem_per_sm=101376,
                      max_threads=2048, max_blocks=32, warp=32):
    """Replicate NVIDIA's occupancy calculator logic: the min over all limits."""
    regs_per_block = threads_per_block * regs_per_thread
    by_regs   = regfile // regs_per_block if regs_per_block else max_blocks
    by_smem   = smem_per_sm // smem_per_block_bytes if smem_per_block_bytes else max_blocks
    by_threads = max_threads // threads_per_block
    blocks = min(by_regs, by_smem, by_threads, max_blocks)
    warps  = blocks * threads_per_block // warp
    occupancy = warps / (max_threads // warp)
    return blocks, warps, occupancy

for regs, smem in [(48, 24*1024), (32, 16*1024), (32, 8*1024)]:
    b, w, occ = max_blocks_per_sm(256, regs, smem)
    print(f"regs/thread={regs:2d} smem={smem//1024:2d}KB -> "
          f"{b} blocks, {w} warps, occupancy={occ:.0%}")
```

---

## Arithmetic Intensity: The Master Concept

Here is the idea that unifies this entire chapter and most of Part IV. Every kernel does some amount of arithmetic and moves some number of bytes. The ratio is its **arithmetic intensity** (also called operational intensity):

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved to/from HBM}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]
$$

The hardware has a fixed peak compute $\pi$ (FLOP/s) and a fixed peak bandwidth $\beta$ (byte/s). Their ratio defines the **machine balance** or **ridge point**:

$$
I_{\text{ridge}} = \frac{\pi}{\beta} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]
$$

The achievable performance of a kernel is bounded by:

$$
P \le \min\!\big(\pi,\; I \cdot \beta\big)
$$

This is the **roofline** ([The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)). Read it as a simple dichotomy:

- If $I < I_{\text{ridge}}$, the kernel is **memory-bound**: it would finish faster if HBM were faster, and adding more FLOPs (e.g. recomputation) is "free." You are bandwidth-limited.
- If $I > I_{\text{ridge}}$, the kernel is **compute-bound**: it is limited by the arithmetic units. You are FLOP-limited.

{{fig:roofline-intensity-decode-vs-matmul}}

For an A100, $\pi \approx 312\,\text{TFLOP/s}$ (BF16) and $\beta \approx 2.0\,\text{TB/s}$, so $I_{\text{ridge}} \approx 156$ FLOP/byte. For an H100 with $\pi \approx 990\,\text{TFLOP/s}$ (BF16, dense) and $\beta \approx 3.35\,\text{TB/s}$, $I_{\text{ridge}} \approx 295$ FLOP/byte. **The ridge point has been climbing every GPU generation** — compute grows faster than bandwidth — which means more and more kernels fall on the memory-bound side over time. This is *the* secular trend in ML systems: we are increasingly bandwidth-starved, and techniques that trade FLOPs for bytes keep winning.

!!! example "Worked example: why decode is memory-bound and matmul is not"
    **Large square matmul.** Multiply two $N \times N$ BF16 matrices, $N = 8192$. FLOPs $= 2N^3 = 2 \cdot 8192^3 \approx 1.10 \times 10^{12}$. Bytes moved (read both inputs, write output, ignoring reuse caching for a lower bound) $= 3 N^2 \cdot 2 = 3 \cdot 8192^2 \cdot 2 \approx 4.0 \times 10^{8}$ bytes. Arithmetic intensity $I = 1.10\times10^{12} / 4.0\times10^{8} \approx 2{,}700$ FLOP/byte. That is *far* above the A100 ridge of 156, so a big matmul is solidly **compute-bound** — exactly why it can approach peak Tensor Core throughput.

    **Autoregressive decode (one token, one request).** A dense 7B model in BF16 holds about $7\times10^9 \times 2 = 1.4\times10^{10}$ bytes of weights. To generate one token you must read essentially all of them from HBM once: bytes $\approx 1.4\times10^{10}$. The arithmetic is roughly $2 \times \text{params} = 2 \times 7\times10^9 = 1.4\times10^{10}$ FLOPs (two FLOPs per parameter for the matrix-vector products). So $I = 1.4\times10^{10} / 1.4\times10^{10} = 1$ FLOP/byte. That is *vastly* below the ridge — decode is profoundly **memory-bound**.

    The consequence is brutal and beautiful: per-token decode latency for a 7B model on an A100 is roughly $\frac{1.4\times10^{10}\,\text{bytes}}{2.0\times10^{12}\,\text{byte/s}} \approx 7\,\text{ms}$ — set entirely by bandwidth, with the Tensor Cores 99% idle. To use those idle Tensor Cores you must **batch**: process many requests' tokens together so the same weight bytes, read once, do work for many tokens. Batching raises $I$ until decode becomes compute-bound. This is the whole reason continuous batching exists, and why throughput-per-dollar and per-request latency are in tension. Speculative decoding ([Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)) attacks the same wall from a different angle: verify several tokens per weight-load.

```python
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved

def roofline_perf(intensity, peak_flops, peak_bw):
    """Attainable FLOP/s under the roofline model."""
    return min(peak_flops, intensity * peak_bw)

# A100-class numbers (BF16 Tensor Core peak, HBM2e bandwidth).
PEAK_FLOPS = 312e12      # 312 TFLOP/s
PEAK_BW    = 2.0e12      # 2.0 TB/s
ridge = PEAK_FLOPS / PEAK_BW
print(f"A100 ridge point: {ridge:.0f} FLOP/byte")

# Case 1: 8192^3 BF16 matmul
N = 8192
mm_flops = 2 * N**3
mm_bytes = 3 * N**2 * 2          # 2 bytes/elem, read A,B write C
I_mm = arithmetic_intensity(mm_flops, mm_bytes)
print(f"matmul  I={I_mm:8.0f} FLOP/byte -> "
      f"{'COMPUTE' if I_mm > ridge else 'MEMORY'}-bound, "
      f"{roofline_perf(I_mm, PEAK_FLOPS, PEAK_BW)/1e12:.0f} TFLOP/s attainable")

# Case 2: 7B decode, batch=1
params = 7e9
dec_bytes = params * 2           # BF16 weights, read once
dec_flops = 2 * params           # ~2 FLOPs/param for the matrix-vector products
I_dec = arithmetic_intensity(dec_flops, dec_bytes)
print(f"decode  I={I_dec:8.1f} FLOP/byte -> "
      f"{'COMPUTE' if I_dec > ridge else 'MEMORY'}-bound, "
      f"per-token >= {dec_bytes/PEAK_BW*1e3:.1f} ms (bandwidth-bound)")

# Case 3: decode at batch B reuses the SAME weight bytes for B tokens
for B in (1, 8, 32, 128, 256):
    I_b = (2 * params * B) / (params * 2)   # bytes constant, flops scale with B
    bound = 'COMPUTE' if I_b > ridge else 'MEMORY'
    print(f"  batch={B:4d}: I={I_b:6.0f} FLOP/byte -> {bound}-bound")
```

The third loop above shows the magic of batching in one line: weight bytes read are constant in batch size, but useful FLOPs scale linearly with it, so arithmetic intensity scales linearly with batch size until you cross the ridge and finally light up the Tensor Cores.

---

## The Datacenter GPUs: A100, H100, H200, B200, and NVLink

You will reason about these chips constantly. Here are the load-bearing specs and what each generation changed. Treat the figures as representative of the SXM datacenter variants; exact numbers vary by SKU, clock, and whether sparsity is assumed (we quote *dense* compute unless noted).

| Spec | A100 (Ampere) | H100 (Hopper) | H200 (Hopper) | B200 (Blackwell) |
|------|---------------|---------------|---------------|------------------|
| SMs | 108 | 132 | 132 | (two dies, ~2×) |
| HBM type | HBM2e | HBM3 | HBM3e | HBM3e |
| HBM capacity | 40 / 80 GB | 80 GB | 141 GB | ~192 GB |
| HBM bandwidth | ~1.6 / 2.0 TB/s | ~3.35 TB/s | ~4.8 TB/s | ~8 TB/s |
| L2 cache | 40 MB | 50 MB | 50 MB | larger |
| BF16/FP16 dense | ~312 TFLOP/s | ~990 TFLOP/s | ~990 TFLOP/s | higher |
| FP8 dense | — | ~1979 TFLOP/s | ~1979 TFLOP/s | much higher |
| FP4 | — | — | — | yes (new) |
| NVLink BW/GPU | ~600 GB/s | ~900 GB/s | ~900 GB/s | ~1.8 TB/s |

### What each generation actually changed

- **A100 (Ampere, 2020).** Established the modern recipe: BF16/TF32 Tensor Cores, 2:4 structured sparsity, and large HBM. The reference point for almost all published scaling-law and kernel work. Ridge point ≈ 156 FLOP/byte.
- **H100 (Hopper, 2022).** Roughly tripled BF16 Tensor Core throughput and added **FP8** Tensor Cores, doubling effective matmul throughput again for models that tolerate FP8 ([Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)). Introduced the **Transformer Engine** (automatic per-tensor FP8 scaling), **thread block clusters** with **distributed shared memory** (blocks on neighboring SMs share SMEM), and the **Tensor Memory Accelerator (TMA)** for asynchronous bulk copies between HBM and SMEM that free the warps from address arithmetic. FlashAttention-3 ([FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html)) is built specifically to exploit TMA, warp specialization, and FP8 on Hopper. But bandwidth only went from 2.0 to 3.35 TB/s while compute tripled — so the ridge point jumped to ≈295, and *more* workloads became memory-bound.
- **H200 (Hopper refresh, 2024).** Same compute die as H100 but with **HBM3e**: 141 GB at ~4.8 TB/s. The extra capacity and bandwidth are aimed squarely at LLM inference, where you are bandwidth- and KV-cache-capacity-bound. More memory means longer contexts and bigger batches per GPU; more bandwidth directly speeds up decode (which is bandwidth-bound, as we computed). Same FLOPs, but materially faster and roomier serving.
- **B200 (Blackwell, 2024).** A **two-die** package presented as one logical GPU, with much higher HBM3e capacity (~192 GB) and bandwidth (~8 TB/s), a second-generation Transformer Engine, and new low-precision formats including **FP4** for inference. NVLink jumps to ~1.8 TB/s per GPU. The design philosophy is unchanged but the dials are turned up, and FP4 pushes effective inference throughput much higher for models that survive 4-bit ([Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html)).

### NVLink and NVSwitch: the intra-node fabric

A single GPU cannot hold a frontier model, and even when it can, you want many GPUs cooperating on one forward/backward pass. The link between GPUs *inside a server* is **NVLink** — a dedicated high-bandwidth, low-latency interconnect far faster than PCIe. NVLink generations scale from ~600 GB/s per GPU on A100 to ~900 GB/s on H100/H200 to ~1.8 TB/s on B200 (bidirectional aggregate). An **NVSwitch** is a crossbar that connects all 8 GPUs in a node so any pair gets full NVLink bandwidth — an **all-to-all** topology rather than a ring.

Why this matters: **tensor parallelism** ([Distributed Training II](../03-pretraining/06-distributed-model-parallel.html)) splits each weight matrix across GPUs and must do an all-reduce on activations *every layer*, which is only viable over NVLink, not PCIe — hence tensor parallelism stays within an NVLink domain (typically 8 GPUs), while pipeline and data parallelism span the slower inter-node network ([Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)). The Grace-Hopper and Blackwell systems extend this with **NVLink-C2C** (a coherent CPU–GPU link) and **NVLink Switch** fabrics that let dozens of GPUs share memory at NVLink-class bandwidth, blurring the line between "one big GPU" and "a node."

```python
# Where does your 70B model physically fit? A capacity + bandwidth sanity check.
def model_footprint_gb(params_b, bytes_per_param):
    return params_b * 1e9 * bytes_per_param / 1e9   # -> GB

def kv_cache_gb(layers, kv_heads, head_dim, seq, batch, bytes_per_elem=2):
    # K and V, both, per layer, per (kv) head: factor of 2 for K&V.
    elems = 2 * layers * kv_heads * head_dim * seq * batch
    return elems * bytes_per_elem / 1e9

# Llama-70B-ish: 70B params, 80 layers, 8 KV heads (GQA), head_dim 128.
for dtype, bpp in [("bf16", 2), ("fp8", 1), ("int4", 0.5)]:
    w = model_footprint_gb(70, bpp)
    print(f"70B weights in {dtype:5s}: {w:6.1f} GB")

kv = kv_cache_gb(layers=80, kv_heads=8, head_dim=128,
                 seq=8192, batch=32, bytes_per_elem=2)
print(f"KV cache (8k ctx, batch 32, GQA-8, bf16): {kv:.1f} GB")
# 70B bf16 weights ~140 GB -> does NOT fit one 80GB H100; fits one 192GB B200,
# or shards across 2x H100 / 1x via int4. Bandwidth then sets decode speed.
```

This snippet captures the two questions you ask of every (model, GPU) pair: **does it fit** (HBM capacity, including the KV cache that grows with batch × context — see [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)), and **how fast does it decode** (HBM bandwidth). H200's jump to 141 GB and B200's to 192 GB are aimed directly at the first question; their bandwidth bumps at the second.

---

## Putting It Together: Reading a Workload Like the Hardware Does

When you profile or design a kernel, walk the hierarchy top-down and ask the questions the hardware forces on you:

1. **Is this op a matmul?** If yes, it can use Tensor Cores — aim for peak by tiling through SMEM and feeding contiguous, well-shaped BF16/FP8 tiles. If no (pointwise, normalization, softmax, reductions), it runs on CUDA cores and is almost certainly memory-bound; your only lever is to **fuse** it with neighbors so intermediate tensors never round-trip to HBM.
2. **What is its arithmetic intensity?** Estimate FLOPs and HBM bytes. Compare to the chip's ridge point. This single ratio tells you whether to optimize for compute (better tiling, higher Tensor Core utilization) or for memory (fewer bytes moved: fusion, recomputation, lower precision, better reuse).
3. **Is the working set staying in fast tiers?** A memory-bound kernel that re-reads the same HBM data many times is leaving enormous performance on the table. Stage reused data into SMEM/registers (the FlashAttention insight).
4. **Are the warps full and busy?** Check for warp divergence, uncoalesced loads, and shared-memory bank conflicts. Check occupancy — but only to confirm you have *enough* latency-hiding, not to maximize it blindly.
5. **Is the GPU even the bottleneck?** Profile end-to-end. Host-side Python overhead, tiny serialized kernels (launch latency), and PCIe transfers routinely cap throughput well below the GPU's capability. CUDA graphs and `torch.compile` exist to kill launch overhead ([Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)).

```python
import torch

def classify_op(flops, hbm_bytes, peak_flops=990e12, peak_bw=3.35e12):
    """A 30-second back-of-envelope to decide where to spend optimization effort."""
    intensity = flops / hbm_bytes
    ridge = peak_flops / peak_bw
    attainable = min(peak_flops, intensity * peak_bw)
    bound = "compute" if intensity > ridge else "memory"
    return dict(intensity=round(intensity, 2),
                ridge=round(ridge, 1),
                bound=bound,
                attainable_TFLOPs=round(attainable / 1e12, 1),
                advice=("tile/fuse-for-TC; you can chase peak FLOPs"
                        if bound == "compute"
                        else "cut bytes: fuse, recompute, lower precision, reuse"))

# Example: a fused LayerNorm over a (batch*seq, hidden) tensor, bf16.
rows, hidden = 4096 * 2048, 8192       # tokens x hidden
elems = rows * hidden
# LayerNorm: ~a handful of FLOPs/elem; reads + writes the tensor once each.
ln = classify_op(flops=8 * elems, hbm_bytes=2 * elems * 2)   # read+write, 2B/elem
print("LayerNorm:", ln)   # -> memory-bound: the reason norms get fused into matmuls
```

!!! warning "Common pitfall: 'my model has lots of FLOPs so it must be compute-bound'"
    Total FLOPs tells you almost nothing about whether you are compute- or memory-bound; **arithmetic intensity** does. A transformer forward pass is a mix: the big QKV/MLP matmuls are compute-bound, but the attention softmax, the residual adds, the norms, the activation functions, the embedding lookups, and (especially) single-token decode are all memory-bound. A "compute-heavy" model can spend the majority of its wall-clock time in memory-bound glue. Always profile, always compute intensity per op, and remember the ridge point keeps rising — so assume memory-bound until proven otherwise.

!!! tip "Practitioner tip: the three knobs that actually move the needle"
    When a real LLM kernel is slow, in order of frequency the fix is: (1) **use Tensor Cores** — make sure your matmuls run in BF16/FP8 with shapes that are multiples of 8/16, not FP32 on CUDA cores; (2) **stop round-tripping HBM** — fuse pointwise ops into the producing matmul, and stage reused data in SMEM (use a compiler/`torch.compile` or a Triton kernel rather than hand-CUDA when you can); (3) **batch to raise arithmetic intensity** for decode. Reach for exotic occupancy tuning only after these three are exhausted.

!!! interview "Interview Corner"
    **Q:** You serve a 13B-parameter model on a single H100 (80 GB HBM, ~3.35 TB/s bandwidth, ~990 TFLOP/s BF16). At batch size 1, decode throughput is far below the GPU's FLOP capacity, and the Tensor Cores show near-zero utilization. Explain precisely why, and give two distinct ways to fix it. What is the fundamental limit each one runs into?

    **A:** Single-request decode is **memory-bandwidth-bound**, not compute-bound. Generating one token requires reading essentially all 13B weights from HBM once — about $13\times10^9 \times 2 = 2.6\times10^{10}$ bytes in BF16 — while doing only ~$2\times13\times10^9 = 2.6\times10^{10}$ FLOPs of matrix-vector work. That is an arithmetic intensity of ~1 FLOP/byte, versus the H100's ridge point of ~$990/3.35 \approx 295$ FLOP/byte. We are roughly 300× below the ridge, so the Tensor Cores sit idle while we wait on HBM. The bandwidth floor on per-token latency is $2.6\times10^{10}\,\text{B} / 3.35\times10^{12}\,\text{B/s} \approx 7.8\,\text{ms}$, independent of how fast the math units are.

    Two fixes: **(1) Increase batch size** (continuous batching). The weight bytes are read once but now do work for $B$ tokens, so arithmetic intensity scales linearly with $B$. Once $B$ pushes intensity past the ridge (~hundreds), decode becomes compute-bound and the Tensor Cores light up — throughput rises with no extra weight traffic. The fundamental limit: **HBM capacity for the KV cache**, which grows with batch × context length and eventually fills the 80 GB, and **per-request latency**, since large batches increase queuing/step time. **(2) Reduce bytes moved per token** via quantization — INT4/FP8 weights cut weight traffic 2–4×, directly cutting decode latency since it is bandwidth-bound. The fundamental limit: **accuracy degradation** at low precision, and that activations/KV may still dominate traffic. A third, orthogonal answer interviewers love: **speculative decoding**, which verifies several draft tokens per single weight-load, amortizing the bandwidth cost across multiple accepted tokens.

---

## Key Takeaways

!!! key "Key Takeaways"
    - The GPU is a **throughput** machine that hides memory latency with massive parallelism. It runs fast only when fed tens of thousands of independent units of work; single-stream, serialized, or branchy workloads waste it.
    - The execution hierarchy is **thread → warp (32, lockstep SIMT) → block/CTA (one SM, shares SMEM) → grid**. Warp divergence and uncoalesced memory access are the two most common silent throughput killers.
    - **CUDA cores** do scalar FP32/INT math; **Tensor Cores** do tile-shaped reduced-precision matmul-accumulate and are ~16× faster — but only for matmuls in BF16/FP16/FP8/FP4. Everything else (norms, activations, softmax) runs on CUDA cores and is usually memory-bound.
    - The memory hierarchy — **registers → SMEM/L1 → L2 → HBM** — drops roughly an order of magnitude in speed and rises an order of magnitude in size at each step. Kernel performance is largely the art of keeping reused data in the fast tiers (SMEM tiling, the FlashAttention insight).
    - **Occupancy** is the ratio of resident warps to the SM maximum, bounded by registers, shared memory, and slot limits. More occupancy buys latency-hiding, but **maximum occupancy is not the goal** — high-register, low-occupancy GEMM/attention kernels often win.
    - **Arithmetic intensity** $I = \text{FLOPs}/\text{HBM bytes}$ versus the **ridge point** $\pi/\beta$ decides compute- vs memory-bound. Large matmuls are compute-bound ($I \sim 10^3$); single-token decode is memory-bound ($I \sim 1$). Total FLOPs alone tells you nothing.
    - Across **A100 → H100 → H200 → B200**, compute (and low precision: FP8, then FP4) grew faster than bandwidth, so the **ridge point keeps rising** and more workloads become memory-bound. H200/B200's big HBM3e capacity and bandwidth target inference's two limits: fitting the model+KV cache, and bandwidth-bound decode.
    - **NVLink/NVSwitch** give all-to-all, multi-hundred-GB/s intra-node bandwidth that makes tensor parallelism viable inside an 8-GPU domain; PCIe and inter-node networks are an order of magnitude slower, which is why parallelism strategies map onto the interconnect tiers.

---

!!! sota "State of the Art & Resources (2026)"
    GPU architecture for LLM workloads is defined today by the Hopper/Blackwell lineage: widening HBM3e bandwidth, FP8/FP4 Tensor Cores, and hardware-managed async copies (TMA) that let kernels overlap compute and data movement. The central tension — compute growing faster than bandwidth, raising the ridge point each generation — continues to make memory-hierarchy awareness the dominant skill in ML systems engineering.

    **Foundational work**

    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — the canonical proof that SMEM-tiling to avoid HBM round-trips is worth more than extra FLOPs.
    - [Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (2009)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) — the original arithmetic-intensity / ridge-point framework used throughout this chapter.

    **Recent advances (2023–2026)**

    - [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608) — exploits Hopper's TMA and warp-specialization to reach 75 % of H100 FP16 peak; the state-of-the-art attention kernel.
    - [Luo et al., *Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis* (2025)](https://arxiv.org/abs/2501.12084) — quantitative measurements of H100 L2, HBM, and Tensor Core pipelines that put concrete numbers on the specs in this chapter.
    - [Jarmusch & Chandrasekaran, *Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis* (2025)](https://arxiv.org/abs/2512.02189) — first detailed characterization of B200 TMEM, FP4 tensor cores, and the decompression engine.

    **Open-source & tools**

    - [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — production C++ template library and CuTe DSL for tiled GEMM across Ampere through Blackwell; the reference for hand-written Tensor Core kernels.
    - [triton-lang/triton](https://github.com/triton-lang/triton) — Python-level GPU kernel authoring language that compiles tile programs to Tensor Core PTX; lower barrier than CUDA for experimenting with the tiling ideas in this chapter.
    - [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) — the official interactive profiler for CUDA kernels; surfaces occupancy, memory throughput, and Tensor Core utilization to verify roofline predictions.

    **Go deeper**

    - [NVIDIA, *Hopper Architecture In-Depth* (2022)](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — official deep-dive on H100's Transformer Engine, TMA, thread-block clusters, and FP8; required reading before writing Hopper-specific kernels.
    - [NVIDIA, *Blackwell Architecture Technical Overview* (2024)](https://resources.nvidia.com/en-us-blackwell-architecture) — covers B200's dual-die design, FP4 Tensor Cores, and NVLink 5 specs.
    - [Horace He, *Making Deep Learning Go Brrrr From First Principles* (2022)](https://horace.io/brrr_intro.html) — concise engineer's guide to the compute / memory-bandwidth / overhead trifecta, with worked PyTorch examples.
    - [NVIDIA, *CUDA Programming Guide* (continuously updated)](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) — authoritative reference for the execution model, memory hierarchy, and warp semantics described in this chapter.

---

## Further Reading

- **NVIDIA, *CUDA C++ Programming Guide* and the per-architecture whitepapers (Ampere GA100, Hopper GH100, Blackwell)** — The authoritative source for the execution model, memory hierarchy, and the per-generation specs summarized in this chapter. Read the architecture whitepapers for the "what changed" story.
- **Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (2009)** — The original roofline paper that formalizes arithmetic intensity, the ridge point, and the compute- vs memory-bound dichotomy.
- **Vasily Volkov, *Better Performance at Lower Occupancy* (GTC 2010)** — The classic demonstration that maximizing occupancy is not the same as maximizing performance; essential context for the occupancy section.
- **Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)** and **Dao, *FlashAttention-2* (2023)** — The canonical case studies in trading FLOPs for HBM traffic by keeping the working set in SMEM/registers; the practical payoff of everything in this chapter.
- **Horace He, *Making Deep Learning Go Brrrr From First Principles* (2022)** — A widely-cited engineer's walkthrough of compute-bound vs memory-bound vs overhead-bound regimes in PyTorch, with the same intensity-first mindset.
- **NVIDIA *CUTLASS* and OpenAI *Triton* repositories** — Production and pedagogical implementations of tiled, Tensor-Core GEMM and fused kernels; the best way to see the SMEM-tiling and warp-level mechanics described here turned into real code (see [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html)).

---

## Exercises

**1.** A kernel contains the branch `if (x[i] > 0) { y[i] = expensive_A(x[i]); } else { y[i] = expensive_B(x[i]); }`, where `expensive_A` and `expensive_B` each take about the same time $t$ and neither is trivial. For a warp in which 16 lanes take the `if` side and 16 take the `else` side, how long does the warp take relative to the best case where all 32 lanes take the same side? Explain the mechanism, and describe one way to restructure the data so the warp stops paying this cost.

??? note "Solution"
    Recall that the 32 lanes of a warp share one instruction fetch/decode unit and one program counter and execute **in lockstep** (SIMT). When lanes disagree on a data-dependent branch, the warp cannot run the two sides simultaneously — it **serializes** them: it executes the `if` path with the 16 `else`-lanes masked off (idle), then executes the `else` path with the 16 `if`-lanes masked off. This is **warp divergence**.

    So the divergent warp takes $t_A + t_B \approx 2t$: the full cost of both paths, back to back. The convergent best case (all 32 lanes on one side) takes only $t$, with all lanes doing useful work. The divergent warp is therefore about **2x slower**, and during each path only 16 of 32 lanes are active, so hardware utilization on each path is 50%. A branch that splits a warp $k$ ways in general costs the sum of all taken paths' times.

    Restructuring: **sort or partition the data so that lanes within a warp mostly agree.** If you reorder `x` so that all positive values are contiguous, then most warps become homogeneous (all-`if` or all-`else`) and run each at full speed $t$ with no masking. This is the same idea as "make branchy work coherent" — group like with like so the branch is decided the same way for all 32 lanes of a warp. Only warps straddling the boundary between the positive and negative regions still diverge, and there are very few of them.

**2.** Consider an SM with these limits: 65,536 32-bit registers, 100 KB (take as 100,000 bytes) of shared memory, a cap of 2,048 resident threads, and a cap of 32 resident blocks. You launch a kernel with **128 threads per block**, using **40 registers per thread** and **8 KB (8,192 bytes) of shared memory per block**. How many blocks fit on the SM, and what is the resulting occupancy?

??? note "Solution"
    Occupancy is set by whichever per-SM resource runs out first; compute the block count each limit allows and take the minimum.

    - **Registers.** Each block needs $128 \times 40 = 5{,}120$ registers. The file holds $65{,}536$, so $\lfloor 65536 / 5120 \rfloor = \lfloor 12.8 \rfloor = 12$ blocks.
    - **Shared memory.** $\lfloor 100{,}000 / 8{,}192 \rfloor = \lfloor 12.2 \rfloor = 12$ blocks.
    - **Thread cap.** $\lfloor 2048 / 128 \rfloor = 16$ blocks.
    - **Block cap.** 32 blocks.

    The minimum is $\min(12, 12, 16, 32) = 12$ blocks (registers and shared memory tie as the binding constraints). That is $12 \times 128 = 1{,}536$ resident threads $= 1536 / 32 = 48$ warps.

    The maximum warps per SM is $2048 / 32 = 64$. So

    $$
    \text{occupancy} = \frac{48}{64} = 0.75 = \textbf{75\%}.
    $$

    You can confirm this with the chapter's `max_blocks_per_sm` helper: `max_blocks_per_sm(128, 40, 8*1024, smem_per_sm=100000, max_threads=2048)` returns `(12, 48, 0.75)`.

**3.** A dense **34B**-parameter model is served on a single H100 (peak $\approx 990$ TFLOP/s BF16 dense, HBM bandwidth $\beta \approx 3.35$ TB/s), using **FP8** weights (1 byte per parameter). For single-request (batch-1) autoregressive decode: (a) how many bytes must be read from HBM to generate one token, and what is the bandwidth-floor per-token latency? (b) What is the arithmetic intensity, and is decode compute- or memory-bound? (c) The H100 ridge point in FP8 is about $1979 / 3.35 \approx 590$ FLOP/byte. Roughly what minimum batch size would push batch-scaled decode past that ridge?

??? note "Solution"
    **(a) Bytes and latency.** Generating one token in batch-1 decode reads essentially every weight from HBM once. At 1 byte/param (FP8):

    $$
    \text{bytes} \approx 34\times10^9 \times 1 = 3.4\times10^{10}\ \text{bytes}.
    $$

    The bandwidth floor on per-token latency is bytes divided by bandwidth:

    $$
    t \ge \frac{3.4\times10^{10}}{3.35\times10^{12}} \approx 1.01\times10^{-2}\ \text{s} \approx \textbf{10.1 ms}.
    $$

    This floor is set purely by HBM bandwidth; making the Tensor Cores faster does not help.

    **(b) Intensity.** The matrix-vector products do about $2\times\text{params}$ FLOPs per token: $2 \times 34\times10^9 = 6.8\times10^{10}$ FLOPs. So

    $$
    I = \frac{6.8\times10^{10}}{3.4\times10^{10}} = 2\ \frac{\text{FLOP}}{\text{byte}}.
    $$

    (In general, with $b$ bytes per param, $I = 2\text{params} / (b\cdot\text{params}) = 2/b$; here $b=1$ so $I=2$.) That is roughly 300x below the FP8 ridge of ~590, so batch-1 decode is profoundly **memory-bound** — the Tensor Cores are almost entirely idle.

    **(c) Minimum batch to cross the ridge.** In decode, the weight bytes read are **constant** in batch size (you read each weight once and reuse it for all requests in the batch), while useful FLOPs scale linearly with batch $B$. So intensity scales as $I(B) = 2B$ (FP8: $2/b \cdot B$ with $b=1$). Setting $I(B) > 590$:

    $$
    2B > 590 \ \Rightarrow\ B > 295 \ \Rightarrow\ B_{\min} = \textbf{296}.
    $$

    So you need roughly a batch of ~300 concurrent tokens before FP8 decode becomes compute-bound and the Tensor Cores light up. (Whether that batch fits is a separate question, bounded by HBM capacity for the KV cache.)

**4.** Using the chapter's A100 numbers ($\pi \approx 312$ TFLOP/s BF16, $\beta \approx 2.0$ TB/s, ridge $\approx 156$ FLOP/byte), consider a 7B model in BF16. (a) At batch 1, the arithmetic intensity of decode is $I = 1$ FLOP/byte. What is the smallest integer batch size at which decode becomes compute-bound? (b) At that batch size, is per-token latency still floored by the same 7 ms bandwidth number as batch 1? Explain what has (and has not) changed.

??? note "Solution"
    **(a) Crossover batch.** For BF16 (2 bytes/param), batch-1 intensity is $I = 2\text{params}/(2\text{params}) = 1$ FLOP/byte. Because weight bytes are read once and reused across the batch, intensity scales linearly with batch size: $I(B) = B$. Decode becomes compute-bound when $I(B) > I_\text{ridge} = 156$:

    $$
    B > 156 \ \Rightarrow\ B_{\min} = \textbf{157}.
    $$

    So batch 157 is the first integer batch where the 7B model's decode crosses the A100 ridge (at exactly $B=156$ it sits on the ridge; $B=157$ is the first strictly compute-bound point).

    **(b) What changed.** At batch 1 the per-*token* latency is floored by weight traffic: $1.4\times10^{10}\ \text{bytes} / 2.0\times10^{12}\ \text{byte/s} \approx 7$ ms to produce **one** token. At batch $\approx157$ you read the same $1.4\times10^{10}$ bytes of weights **once per step**, but that single weight-load now produces **157 tokens** (one per request). So:

    - The **per-step** weight traffic and the ~7 ms bandwidth cost of streaming the weights are essentially unchanged.
    - But the ~7 ms is now amortized over 157 tokens, so **per-token throughput** rises by ~157x, and beyond the ridge the step time starts being set by the Tensor Cores (compute) rather than by HBM.

    In short: batching does not make any single token cheaper to *produce in isolation* — it makes each expensive weight-load pay for many tokens at once, which is exactly why continuous batching converts idle-Tensor-Core, bandwidth-bound decode into efficient, compute-bound work.

**5.** Implement a small helper, in the style of the chapter's roofline code, that answers the two questions of batched decode at once: `decode_analysis(params, bytes_per_param, batch, peak_flops, peak_bw)` should return the arithmetic intensity, whether decode is compute- or memory-bound, and the **per-step** time under the roofline (the max of the bandwidth time and the compute time). Then add `min_batch_for_compute_bound(bytes_per_param, peak_flops, peak_bw)` returning the smallest integer batch that crosses the ridge. Run both for a 7B BF16 model on the A100 numbers from Exercise 4 and confirm the crossover matches your hand calculation.

??? note "Solution"
    The key facts to encode: in decode the weight bytes are read **once per step** regardless of batch (`bytes = params * bytes_per_param`), while FLOPs scale with batch (`flops = 2 * params * batch`). Intensity is `flops / bytes`, which reduces to `(2/bytes_per_param) * batch`. The per-step time under the roofline is the larger of the time to stream the bytes and the time to do the FLOPs.

    ```python
    import math

    def decode_analysis(params, bytes_per_param, batch, peak_flops, peak_bw):
        bytes_moved = params * bytes_per_param        # weights read once per step
        flops       = 2 * params * batch              # ~2 FLOPs/param/token
        intensity   = flops / bytes_moved
        ridge       = peak_flops / peak_bw
        bw_time      = bytes_moved / peak_bw           # seconds to stream weights
        compute_time = flops / peak_flops              # seconds to do the math
        step_time    = max(bw_time, compute_time)      # roofline: the binding one
        return dict(
            intensity=round(intensity, 2),
            ridge=round(ridge, 1),
            bound=("compute" if intensity > ridge else "memory"),
            step_ms=round(step_time * 1e3, 3),
            per_token_ms=round(step_time * 1e3 / batch, 4),
        )

    def min_batch_for_compute_bound(bytes_per_param, peak_flops, peak_bw):
        ridge = peak_flops / peak_bw
        # intensity(B) = (2 / bytes_per_param) * B ; want intensity > ridge
        b = ridge * bytes_per_param / 2.0
        return math.floor(b) + 1                       # first strictly-above integer

    # 7B BF16 on A100 numbers from Exercise 4.
    PARAMS   = 7e9
    BPP      = 2                 # BF16
    PEAK_FLOPS = 312e12          # 312 TFLOP/s
    PEAK_BW    = 2.0e12          # 2.0 TB/s

    bmin = min_batch_for_compute_bound(BPP, PEAK_FLOPS, PEAK_BW)
    print("min batch for compute-bound:", bmin)        # -> 157

    for B in (1, bmin - 1, bmin):
        print(B, decode_analysis(PARAMS, BPP, B, PEAK_FLOPS, PEAK_BW))
    ```

    Expected output (matching Exercise 4):

    ```text
    min batch for compute-bound: 157
    1   {'intensity': 1.0,   'ridge': 156.0, 'bound': 'memory',  'step_ms': 7.0,    'per_token_ms': 7.0}
    156 {'intensity': 156.0, 'ridge': 156.0, 'bound': 'memory',  'step_ms': 7.0,    'per_token_ms': 0.0449}
    157 {'intensity': 157.0, 'ridge': 156.0, 'bound': 'compute', 'step_ms': 7.045,  'per_token_ms': 0.0449}
    ```

    Reading the output: at batch 1 the step is memory-bound and takes ~7 ms for a single token (`step_ms == per_token_ms`). At batch 156 the intensity has climbed to exactly the ridge; the step still costs ~7 ms of bandwidth time but now yields 156 tokens, so per-token time has collapsed by ~156x. At batch 157 the compute time finally exceeds the bandwidth time, `bound` flips to `compute`, and from here on the Tensor Cores — not HBM — set the step time. The crossover at **157** matches the hand calculation from Exercise 4.
