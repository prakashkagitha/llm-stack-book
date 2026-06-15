# 4.5 CUDA Programming Essentials for ML Engineers

Modern deep learning lives or dies on the GPU. Yet most ML engineers treat the GPU as an opaque box: they call PyTorch, cuBLAS does something fast, and tokens appear. That abstraction breaks the moment you need a custom operation — a fused activation, a new attention variant, a quantized matmul — and there is no existing kernel that fits. At that point you must either drop to CUDA C++ or reach for Triton. Understanding CUDA is not optional for a serious ML engineer; it is the shared vocabulary of every high-performance LLM paper you will read.

This chapter teaches you CUDA from the ground up, with enough depth to understand FlashAttention, write a tiled matrix multiplication, diagnose performance bottlenecks, and make principled decisions between CUDA, Triton, and PyTorch custom ops. We assume you have seen the GPU memory hierarchy — if you need a refresher, read [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) first. For the broader performance model connecting compute, bandwidth, and the arithmetic intensity axis, see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).

## The CUDA Execution Model

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel programming framework. A CUDA program consists of *host code* running on the CPU and *device code* (kernels) running on the GPU. The central insight is that a GPU can schedule tens of thousands of lightweight threads simultaneously, hiding memory latency by switching to other threads while one stalls on a load.

### Grid, Block, and Thread Hierarchy

Every kernel is launched with a **grid** of **blocks**, each block containing a fixed number of **threads**. This three-level hierarchy maps onto the physical GPU hierarchy.

{{fig:cuda-grid-block-thread-hierarchy}}

Each block executes on a single **Streaming Multiprocessor (SM)**. An A100 has 108 SMs; an H100 has 132. Threads within a block can share on-chip **shared memory** and can synchronize with `__syncthreads()`. Threads in *different* blocks cannot directly communicate — they must go through global (DRAM) memory.

```cpp
// CUDA kernel: each thread computes one element of C = A + B
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // Thread's flat index in the 1D grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host-side launch
int N = 1 << 24;  // 16M elements
int threads_per_block = 256;
int blocks = (N + threads_per_block - 1) / threads_per_block;
vector_add<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);
```

Built-in variables available inside every kernel:

| Variable | Meaning |
|---|---|
| `threadIdx.{x,y,z}` | Thread index within its block |
| `blockIdx.{x,y,z}` | Block index within the grid |
| `blockDim.{x,y,z}` | Number of threads per block |
| `gridDim.{x,y,z}` | Number of blocks in the grid |

Dimensions can be 1D, 2D, or 3D; you choose the shape that maps naturally to your data (e.g., 2D blocks for matrix tiles).

### Warps: The Unit of Execution

A **warp** is 32 threads that execute in lockstep on a single set of functional units (the SIMT — Single Instruction Multiple Threads — model). This is the most important micro-architectural fact for performance.

- A block is partitioned into warps: block of 256 threads → 8 warps.
- All threads in a warp execute the *same* instruction each cycle.
- **Warp divergence**: when threads in the same warp take different branches (`if/else`), both paths are executed sequentially, with inactive threads masked. This halves throughput for a 50/50 split.
- The SM schedules many warps concurrently. On an A100, each SM can hold up to 64 warps. When one warp stalls on a memory load, the SM issues instructions for a ready warp with zero overhead — this is *latency hiding*.

**Occupancy** is the ratio of active warps per SM to the maximum. High occupancy is a proxy for effective latency hiding, though it is not the only factor — kernel-bound workloads may prefer fewer, more register-rich warps.

## The GPU Memory Hierarchy

Getting memory access right is the single most important performance lever in CUDA. The full hierarchy for an A100, with approximate bandwidths and latencies:

{{fig:cuda-memory-hierarchy-bandwidth-latency}}

Bandwidth numbers are order-of-magnitude illustrations; see NVIDIA's official architecture whitepapers for precise figures. The key message: HBM is 10× slower than L2, and L2 is 10× slower than shared memory.

### Global Memory Coalescing

When threads in a warp access global memory, the hardware tries to *coalesce* the accesses into as few 128-byte cache-line transactions as possible. If warp lane $i$ reads address $A + i$, one transaction serves all 32 threads — perfect coalescing. If lane $i$ reads address $A + i \cdot 64$, you get 32 separate transactions — a 32× bandwidth penalty.

**Pattern to prefer**: threads in a warp should access consecutive (strided-by-1) memory addresses.

```cpp
// GOOD: coalesced — thread i reads row-major element (row, i)
float val = A[row * N + threadIdx.x];   // threads 0..31 read consecutive floats

// BAD: strided — thread i reads column-major element (i, col)
float val = A[threadIdx.x * N + col];   // threads 0..31 are N floats apart
```

### Shared Memory and Bank Conflicts

Shared memory is organized into 32 **banks** (on modern GPUs), each 4 bytes wide. Bank $b$ holds bytes $4b, 4b+128, 4b+256, \ldots$. Accesses from the same warp to different addresses in the *same* bank are **serialized** — a bank conflict.

The golden rule: if warp lane $i$ accesses shared memory address $s_i$, there is no conflict if all $s_i$ map to distinct banks, i.e., $(s_i \bmod 32)$ are all distinct.

A common source of bank conflicts is the naive tiled matmul transpose: when you load a tile column-by-column into a $32 \times 32$ shared-memory array, all threads in a warp hit the same bank. The fix is to add a **padding column**:

```cpp
// Without padding: 32-way bank conflict when accessing column j
__shared__ float tile[32][32];

// With +1 padding: each row starts on a different bank offset
__shared__ float tile[32][33];  // 33 = 32 + 1 padding column
```

The padding wastes 32 floats (128 bytes) per tile but eliminates the conflict entirely.

!!! example "Worked Example: Shared Memory Bandwidth"

    A kernel uses a $32 \times 32$ shared-memory tile. Each thread in a 32-thread warp reads one element per column from the tile.
    
    Without padding:
    - All 32 threads in the warp access column $j$.
    - Element $(i, j)$ lives at offset $i \cdot 32 + j$ bytes/4 = offset $i \cdot 32 + j$ in 4-byte words.
    - Bank for element $(i,j)$ = $(i \cdot 32 + j) \bmod 32 = j \bmod 32$.
    - All 32 threads access bank $j \bmod 32$ — a 32-way conflict! Shared memory throughput drops from ~19 TB/s to ~0.6 TB/s.
    
    With padding (`tile[32][33]`):
    - Element $(i, j)$ lives at offset $i \cdot 33 + j$ words.
    - Bank = $(i \cdot 33 + j) \bmod 32 = (i + j) \bmod 32$.
    - Thread $i$ in the warp accesses bank $(i + j) \bmod 32$, which cycles through all 32 banks — zero conflicts.

## Warp Primitives and Shuffle Instructions

CUDA exposes primitives for threads within a warp to communicate directly, without touching shared memory. These **warp shuffle** intrinsics are the building blocks of efficient reductions, prefix sums, and softmax.

```cpp
// __shfl_sync: broadcast lane src's value to all lanes in mask
float val = __shfl_sync(0xFFFFFFFF, x, src_lane);

// __shfl_down_sync: lane i gets lane i+delta's value
float val = __shfl_down_sync(0xFFFFFFFF, x, delta);

// __shfl_xor_sync: butterfly exchange for tree reductions
float val = __shfl_xor_sync(0xFFFFFFFF, x, mask);
```

The first argument `0xFFFFFFFF` is the *active mask* — all 32 lanes participate. Here is a complete warp-level reduction that sums 32 values into lane 0:

```cpp
// Warp reduction: sum x across all 32 lanes, result in lane 0
__device__ float warp_reduce_sum(float val) {
    // Tree reduction: delta = 16, 8, 4, 2, 1
    // Each step: lane i adds lane i+delta's value
    for (int delta = 16; delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, delta);
    }
    return val;  // Only lane 0 holds the correct sum
}

// Block-level reduction using warp reductions + shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];  // At most 32 warps per block
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp reduces to its lane 0
    val = warp_reduce_sum(val);

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // First warp reduces the per-warp sums
    val = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    return val;  // Lane 0 of warp 0 holds the block sum
}
```

Warp shuffles are significantly faster than shared memory reductions because they avoid `__syncthreads()` barriers and do not consume shared memory bandwidth. This pattern is used inside FlashAttention's online softmax — see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) for the full derivation.

## Tiled Matrix Multiplication: A Complete Kernel

Matrix multiplication is the dominant operation in every LLM layer — it is the attention projection, the FFN weight multiply, the embedding lookup. A naive CUDA matmul reads each element of A and B $N$ times from global memory; a tiled matmul cuts that to $N/T$ reads (where $T$ is the tile size) by reusing data from shared memory. This is the single most important kernel to understand.

### Naive Matmul (Baseline)

For $C = A \cdot B$ where $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, the operation is:

$$
C[i,j] = \sum_{k=0}^{K-1} A[i,k] \cdot B[k,j]
$$

Total FLOPs: $2 \cdot M \cdot N \cdot K$ (multiply + add per pair).

```cpp
// Naive: each thread computes one output element by iterating over K
__global__ void matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
```

The bottleneck: `A[row * K + k]` is the same for all threads in the column direction; `B[k * N + col]` is the same for all threads in the row direction. Both are re-read from global memory on every iteration — wasting bandwidth.

### Tiled Matmul (SGEMM Quality)

The idea: divide A and B into $T \times T$ tiles. Each block cooperatively loads one tile of A and one tile of B into shared memory, computes the partial dot products, and advances to the next tile.

```cpp
// Tiled matrix multiplication with shared memory
// Tile size T must divide blockDim.x == blockDim.y (set T = BLOCK_SIZE)
#define BLOCK_SIZE 32

__global__ void matmul_tiled(
    const float* __restrict__ A,   // [M, K], row-major
    const float* __restrict__ B,   // [K, N], row-major
    float*       __restrict__ C,   // [M, N], row-major
    int M, int N, int K
) {
    // Identify this thread's output position
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;   // global row in C
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;   // global col in C

    // Shared memory tiles — +1 padding to avoid bank conflicts on B's transpose
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1 avoids bank conflicts

    float acc = 0.0f;  // Accumulator lives in a register

    // Loop over K-dimension in tiles of size BLOCK_SIZE
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // ---- Cooperative load of tile t ----
        // Each thread loads one element of A and one element of B

        int a_col = t * BLOCK_SIZE + threadIdx.x;   // column in A for this tile
        int b_row = t * BLOCK_SIZE + threadIdx.y;   // row in B for this tile

        // Guard: handle matrices whose dimensions aren't multiples of BLOCK_SIZE
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // ---- Synchronize before compute ----
        // All threads in the block must have written to shared memory
        __syncthreads();

        // ---- Compute partial dot product for this tile ----
        // Unroll hint: compiler may auto-unroll; explicit #pragma unroll helps
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // ---- Synchronize before next load ----
        // Ensure all threads are done reading before anyone overwrites the tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
```

**Launch configuration:**

```cpp
// Host-side: launch the kernel
void launch_matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);              // 32×32 = 1024 threads/block
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,          // ceil(N/32) blocks in x
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE           // ceil(M/32) blocks in y
    );
    matmul_tiled<<<grid, block>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();  // Wait and check for errors in development
}
```

!!! example "Worked Example: Memory Traffic Reduction"

    Consider $M = N = K = 4096$ (a typical attention projection for a 7B model, hidden dimension 4096).

    **Naive kernel:**
    - Each output element $C[i,j]$ requires reading the full row $i$ of $A$ ($K = 4096$ floats) and the full column $j$ of $B$ ($K = 4096$ floats) from global memory — every time, for every thread.
    - Total global memory reads: $M \cdot N \cdot 2K$ floats = $4096 \times 4096 \times 8192 \approx 137 \times 10^9$ floats = ~549 GB (at FP32).

    **Tiled kernel with $T = 32$:**
    - Each tile is loaded once and reused by 32 threads along the row/column.
    - Each element of $A$ is loaded $N/T = 128$ times (once per block covering that row).
    - Each element of $B$ is loaded $M/T = 128$ times.
    - Total reads: $(M \cdot K + K \cdot N) \times 1$ (each element loaded once per tile pass) = $2 \times K^2$ floats for the square case.
    - That's $2 \times 4096^2 \approx 33 \times 10^6$ floats = ~134 MB — a ~4000× reduction in global memory traffic.
    - In practice the L2 cache captures some reuse even in the naive case, but shared memory provides a guaranteed, programmer-managed cache at much higher bandwidth.

    **Arithmetic intensity:**
    - FLOPs: $2 \times 4096^3 \approx 137 \times 10^9$
    - Bytes read (tiled): ~134 MB
    - Arithmetic intensity: $137 \times 10^9 / (134 \times 10^6) \approx 1024$ FLOP/byte
    - A100 peak: ~312 TFLOP/s FP32, HBM bandwidth ~2 TB/s → roofline crossover at ~156 FLOP/byte.
    - At 1024 FLOP/byte we are firmly compute-bound, meaning the kernel is bottlenecked by FMA throughput, not memory. That is the correct regime for a matmul.

### Register Blocking and Double Buffering

Production SGEMM kernels go further:

1. **Register blocking**: each thread accumulates a $4 \times 4$ or $8 \times 8$ sub-tile in registers instead of one element, increasing the ratio of compute to shared-memory loads.
2. **Double buffering**: while computing on tile $t$, asynchronously prefetch tile $t+1$ into a second shared-memory buffer using `__pipeline_memcpy_async()` (CUDA 11+), hiding the global-memory load latency completely.
3. **Tensor Core instructions**: `wmma::mma_sync` or PTX `mma` instructions dispatch 4×8×16 mixed-precision fused operations directly to Tensor Cores, achieving peak TFLOP/s. cuBLAS and CUTLASS both use this path; you should too for any production matmul.

Understanding these principles is what makes FlashAttention's tiled, IO-aware attention comprehensible — see [FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8](../04-kernels-efficiency/03-flash-attention-2-3.html) for how they push these ideas further with warp specialization and pipeline stages.

## Synchronization and Atomic Operations

### `__syncthreads()`

`__syncthreads()` is a **block-level barrier**: execution of any thread in the block does not proceed past this point until *all* threads in the block have reached it. The canonical double-barrier pattern in tiled kernels (sync after load, sync after compute) prevents two hazards:

- **Read-after-write**: a thread reading shared memory before another thread has finished writing.
- **Write-after-read**: a thread overwriting shared memory for the next tile before another thread has finished reading the current tile.

!!! warning "Divergent __syncthreads() is Undefined Behavior"

    Never call `__syncthreads()` inside a conditional branch where some threads in the block might not reach it. The GPU does not automatically wait for divergent threads; the hardware deadlocks or produces incorrect results. If you need conditional synchronization, use `__syncwarp()` (warp-level) or restructure so all threads reach the barrier.

### Atomic Operations

Atomic operations (`atomicAdd`, `atomicMax`, `atomicCAS`) provide thread-safe read-modify-write on global or shared memory. They are essential for histogram building, scatter-add operations (used in sparse attention), and lock-free algorithms.

```cpp
// Parallel histogram: each thread atomically increments a bin
__global__ void histogram(const int* data, int* hist, int N, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = data[idx] % num_bins;
        atomicAdd(&hist[bin], 1);  // Thread-safe increment
    }
}

// Optimization: first reduce into shared memory, then one atomic per block
__global__ void histogram_shared(const int* data, int* hist, int N, int num_bins) {
    __shared__ int local_hist[256];  // Assumes num_bins <= 256
    // Initialize shared histogram
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        local_hist[b] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int bin = data[idx] % num_bins;
        atomicAdd(&local_hist[bin], 1);   // Fast: shared memory atomic
    }
    __syncthreads();

    // One global atomic per bin per block
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        atomicAdd(&hist[b], local_hist[b]);
}
```

Global atomics on older hardware were slow; on A100+ they are heavily optimized and the shared-memory staging pattern is often unnecessary for sparsely contested bins.

## Calling CUDA Kernels from PyTorch

For production use, you compile a CUDA kernel and expose it to Python via a PyTorch extension. This lets you call your kernel exactly like any PyTorch operation, with automatic gradient support if you register a `torch.autograd.Function`.

```cpp
// matmul_ext.cu — save as a .cu file
#include <torch/extension.h>  // PyTorch C++ frontend
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
// (matmul_tiled kernel definition from above goes here)

// Wrapper called from Python via pybind11
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());  // Allocate output on GPU

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_tiled<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    return C;
}

// Expose to Python with PYBIND11_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_cuda, "Tiled CUDA matrix multiplication");
}
```

```python
# setup.py — build and install the extension
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="matmul_ext",
    ext_modules=[
        CUDAExtension(
            name="matmul_ext",
            sources=["matmul_ext.cu"],
            extra_compile_args={"nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```bash
# Install and test
pip install -e .
python -c "
import torch, matmul_ext
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
C = matmul_ext.matmul(A, B)
print('Max error vs torch.mm:', (C - torch.mm(A, B)).abs().max().item())
"
```

Alternatively, `torch.utils.cpp_extension.load()` compiles and loads JIT at runtime — convenient for rapid iteration:

```python
import torch
from torch.utils.cpp_extension import load

matmul_ext = load(
    name="matmul_ext",
    sources=["matmul_ext.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)
```

For gradient support, wrap the function in `torch.autograd.Function` with a custom `backward` that calls the transpose matmuls.

## CUDA vs Triton: When to Use Each

[Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html) covers Triton in depth, but it is worth putting both tools on the same axis so you can make the right choice.

| Dimension | CUDA C++ | Triton |
|---|---|---|
| **Abstraction level** | Threads + warps (manual) | Blocks of tiles (automatic) |
| **Bank conflict handling** | Manual padding required | Compiler handles automatically |
| **Warp scheduling** | Full control | Hidden (implicit warp tiling) |
| **Tensor Core access** | Via WMMA/CuTe/PTX | Automatic for fp16/bf16 matmul |
| **Register pressure** | Manual (`#pragma unroll`) | Managed by compiler |
| **Python interop** | pybind11 / CUDAExtension | Native (kernel is Python) |
| **Debugging** | `cuda-gdb`, `compute-sanitizer` | More accessible; Python errors |
| **Portability** | NVIDIA only | NVIDIA + AMD (ROCm) + future |
| **Peak performance** | Highest possible (CUTLASS level) | 80–95% of expert CUDA |

**When to write CUDA:**

- You need Tensor Core access with custom memory layouts (e.g., FP8 mixed-precision not yet supported by Triton).
- The operation has irregular memory access patterns (e.g., ragged batches, variable-length sequences) where Triton's tiled model is awkward.
- You are writing a kernel that requires warp-level synchronization patterns not expressible in Triton (e.g., producer-consumer pipelines with warp specialization, as in FlashAttention 3).
- Maximum performance for a widely deployed operation (cuBLAS, CUTLASS-level GEMM).
- You need persistent kernels or grid-level synchronization.

**When to use Triton:**

- You are writing a fused activation, layer norm, softmax, or custom attention variant — the productivity gain is enormous.
- You want portability across GPU vendors.
- The 5–10% performance gap compared to expert CUDA is acceptable (it usually is).
- You are prototyping quickly and may iterate on the algorithm; Triton's Python syntax shortens the iteration loop dramatically.

**When to use neither (torch.compile + PyTorch):**

- `torch.compile` with `inductor` backend will auto-generate Triton kernels for most PyTorch operations. For standard ops — matmul, LayerNorm, ReLU — this is often within 5% of hand-written Triton and requires no custom kernel code. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) for how this works.

!!! tip "Practitioner Tip: Start with Triton, Profile, then Descend to CUDA"

    Begin with a Triton kernel or `torch.compile`. Profile with `nsys` (Nsight Systems) or `ncu` (Nsight Compute). If you are below ~80% of the theoretical roofline and the bottleneck is something Triton cannot fix (e.g., shared-memory padding, pipeline depth, warp scheduling), then write CUDA. This staged approach saves weeks of development time and keeps code maintainable.

## Profiling and Debugging CUDA Kernels

You cannot optimize what you cannot measure. NVIDIA's toolchain gives you two main profilers:

**Nsight Systems (`nsys`)** — timeline-level, low overhead:
```bash
# Profile training step; view in nsys-ui
nsys profile --trace=cuda,nvtx -o profile_output \
    python train.py --steps 100
```

**Nsight Compute (`ncu`)** — roofline analysis, instruction-level, high overhead:
```bash
# Profile a specific kernel with full metrics
ncu --set full --kernel-name matmul_tiled \
    --launch-count 1 \
    python run_matmul.py
```

Key metrics to check in Nsight Compute:

| Metric | What it tells you |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Overall SM utilization |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | Global load bytes |
| `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` | FP32 add instructions |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | Shared memory bank conflicts |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |

**`cuda-memcheck` / `compute-sanitizer`** catches race conditions and out-of-bounds accesses at the cost of ~20× slowdown:
```bash
compute-sanitizer --tool memcheck python my_kernel_test.py
compute-sanitizer --tool racecheck python my_kernel_test.py
```

A fast development workflow: write the kernel, run correctness checks against PyTorch reference outputs, profile with `ncu`, iterate. The correctness check is trivial to automate:

```python
import torch

def check_correctness(fn_cuda, fn_ref, *args, rtol=1e-3, atol=1e-4):
    """Compare CUDA kernel output against a reference implementation."""
    out_cuda = fn_cuda(*[a.clone() for a in args])
    out_ref  = fn_ref(*[a.clone() for a in args])
    max_err  = (out_cuda - out_ref).abs().max().item()
    rel_err  = (out_cuda - out_ref).abs() / (out_ref.abs() + 1e-8)
    print(f"Max absolute error: {max_err:.2e}")
    print(f"Max relative error: {rel_err.max().item():.2e}")
    assert torch.allclose(out_cuda, out_ref, rtol=rtol, atol=atol), \
        f"MISMATCH: max error = {max_err:.2e}"
    print("PASS")

# Example usage
A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
check_correctness(matmul_ext.matmul, torch.mm, A, B)
```

## Practical Patterns: Fused Kernels and the ML Workload

The reason CUDA matters so much for LLMs is that naive PyTorch chains many small kernel launches, each reading and writing through HBM. A *fused kernel* combines multiple operations into one — loading data once, doing all operations in registers/shared memory, then writing back. This is the core idea behind FlashAttention.

Here is a simple fused ReLU + bias + scale kernel that illustrates the principle:

```cpp
// Fused bias-add + ReLU + scale: C = max(0, A + bias) * scale
// All in one pass — no intermediate materialization in HBM
__global__ void fused_bias_relu_scale(
    const float* __restrict__ A,     // [N]
    const float* __restrict__ bias,  // [N]
    float* __restrict__ C,           // [N]
    float scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = A[idx] + bias[idx];   // bias-add
        val = val > 0.0f ? val : 0.0f;   // ReLU (branch-free alternative: fmaxf)
        C[idx] = val * scale;             // scale
    }
}

// Vectorized version: process 4 floats per thread using float4
__global__ void fused_bias_relu_scale_vec4(
    const float4* __restrict__ A,
    const float4* __restrict__ bias,
    float4* __restrict__ C,
    float scale,
    int N4  // N / 4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        float4 a    = A[idx];
        float4 b    = bias[idx];
        float4 res;
        // Process 4 elements per thread — increases memory throughput
        res.x = fmaxf(a.x + b.x, 0.0f) * scale;
        res.y = fmaxf(a.y + b.y, 0.0f) * scale;
        res.z = fmaxf(a.z + b.z, 0.0f) * scale;
        res.w = fmaxf(a.w + b.w, 0.0f) * scale;
        C[idx] = res;
    }
}
```

The `float4` version loads 16 bytes per thread per memory transaction rather than 4, improving the effective memory bandwidth utilization toward the hardware maximum. This vectorization pattern applies to any memory-bandwidth-bound (memory-roofline-limited) kernel.

Connection to quantization: fused kernels are essential for INT8/FP8 inference because the dequantization, matmul, and requantization must happen in one pass to avoid materializing full-precision intermediates. See [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html) and [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html) for how quantized kernels are structured.

!!! interview "Interview Corner"

    **Q:** You write a CUDA kernel where each thread in a 32-thread warp accesses a different element of a float array stored in shared memory, but they are all in the same column of a 2D array with 32 columns. No thread accesses the same address. Why is this slow, and how do you fix it?

    **A:** The shared memory is organized into 32 banks, each 4 bytes wide. In a 32-column float array, elements in the same column are separated by `32 * sizeof(float) = 128 bytes = 32 banks`. But since the array is 32 columns wide, column $j$ always maps to bank $j \bmod 32 = j$ for any row. So all 32 threads in the warp access the same bank (bank $j$), causing a 32-way bank conflict. The hardware serializes all 32 accesses, dropping throughput by 32×. The fix is to add one padding element per row: declare the array as `float tile[HEIGHT][33]` instead of `[HEIGHT][32]`. This shifts each row's base address by one bank, so column $j$ in row $i$ now maps to bank $(i \cdot 33 + j) \bmod 32$, which varies across rows and eliminates conflicts.

## Key Takeaways

!!! key "Key Takeaways"

    - The GPU execution model is a three-level hierarchy: grid → blocks → threads. Warps (32 threads) execute in lockstep; the SM hides memory latency by switching between warps.
    - **Memory coalescing** is the single most impactful access pattern optimization: threads in a warp should read/write consecutive addresses to minimize HBM transactions.
    - **Shared memory** is programmer-managed L1 cache (~19 TB/s). Use it to reuse data loaded from HBM — the tiled matmul reduces global reads by a factor of $T$ (tile size), converting a bandwidth-bound kernel into a compute-bound one.
    - **Bank conflicts** occur when multiple threads in a warp access different addresses in the same shared-memory bank. Fix them by padding shared arrays by one element per row.
    - **Warp shuffle intrinsics** (`__shfl_sync`, `__shfl_down_sync`) enable intra-warp reductions and broadcasts faster than shared memory, without `__syncthreads()` overhead.
    - A tiled 32×32 SGEMM at $M=N=K=4096$ achieves arithmetic intensity ~1024 FLOP/byte, firmly in the compute-bound regime — this is the goal for any large matmul kernel.
    - Choose Triton for new fused operators (80–95% of CUDA peak, Python syntax, portable), CUDA for maximum performance or irregular access patterns, and `torch.compile` for standard PyTorch graphs.
    - Always validate kernel outputs numerically against a PyTorch reference before profiling; use `ncu` for roofline analysis and bank-conflict detection.
    - Fused kernels (bias+activation, online softmax, dequant+matmul) reduce HBM traffic by eliminating intermediate writes — this is the design philosophy behind FlashAttention and quantized inference.

!!! sota "State of the Art & Resources (2026)"
    CUDA kernel development for ML has matured rapidly: hand-fused kernels written in CUDA C++ or Triton now underpin virtually every high-performance LLM serving stack, and NVIDIA's Hopper (H100) and Blackwell GPU architectures have pushed FP8 and asynchronous pipelining to the forefront of kernel design. The field is moving from per-operation tuning toward compiler-driven kernel generation (`torch.compile` / Inductor) while still requiring deep CUDA fluency for frontier work.

    **Foundational work**

    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — definitive example of IO-aware, tiled, fused CUDA kernel design applied to attention.
    - [Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019)](https://www.researchgate.net/publication/366963691_Triton_an_intermediate_language_and_compiler_for_tiled_neural_network_computations) — original Triton paper introducing tile-level IR as an alternative to raw CUDA for ML kernels.

    **Recent advances (2023–2026)**

    - [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691) — improved work partitioning across warps; 2× speedup over FA1; the implementation most production systems use.
    - [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608) — exploits H100 warp specialization and FP8 Tensor Cores; reaches ~740 TFLOP/s.
    - [Hsu et al., *Liger Kernel: Efficient Triton Kernels for LLM Training* (2024)](https://arxiv.org/abs/2410.10989) — fused Triton kernels (RMSNorm, RoPE, SwiGLU, CrossEntropy) that cut LLM training memory by ~60% with minimal code changes.

    **Open-source & tools**

    - [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — production-quality CUDA C++ templates for GEMM, including Tensor Core paths, pipeline stages, and Blackwell FP4/FP8 support; the reference for expert-level matmul kernels.
    - [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — drop-in Triton kernel replacements for Hugging Face model components; shows the practical pattern for kernel-level LLM training optimization.

    **Go deeper**

    - [NVIDIA CUDA Programming Guide (v13.3)](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) — authoritative reference for the execution model, memory hierarchy, warp primitives, and CUDA Tile C++ (new in CUDA 13).
    - [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — practical optimization checklist: coalescing, occupancy, profiling methodology, and memory transfer strategies.
    - [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — deep-dive on H100 SM design, fourth-generation Tensor Cores, FP8 Transformer Engine, and TMA async copy; required reading before writing H100-specific kernels.

## Further Reading

- **NVIDIA CUDA C++ Programming Guide** — the definitive reference for the execution model, memory hierarchy, warp primitives, and synchronization.
- **CUTLASS** (NVIDIA, GitHub) — production-quality CUDA templates for GEMM with register blocking, pipeline stages, and Tensor Core support; the best real-world CUDA code to study.
- **Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)** — applies tiled, fused-kernel thinking to the attention computation; see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html).
- **Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (2019)** — introduces Triton; see the companion chapter [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html).
- **"Programming Massively Parallel Processors" by Kirk & Hwu** — a thorough textbook treatment of CUDA including shared memory, bank conflicts, and performance optimization.
- **Luo et al., "A Survey of GPU Architectures and Optimization Techniques"** — for a historical perspective on how SM design has evolved across Volta, Ampere, and Hopper.
- **NVIDIA Nsight Compute Documentation** — for the full list of hardware performance counters and roofline methodology used to diagnose kernel bottlenecks.
