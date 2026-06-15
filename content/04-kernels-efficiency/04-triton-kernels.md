# 4.4 Writing GPU Kernels with Triton

There is a moment in every LLM engineer's life when `torch.compile` is not enough, when fusing the operation you want by hand in CUDA C++ would take a week, and when the kernel you need does not yet exist in any library. That is the moment Triton was built for. Triton, originally created by Philippe Tillet and now an OpenAI project, is a Python-embedded language and compiler for writing GPU kernels. You write something that *looks* like NumPy operating on blocks of data; Triton's compiler turns it into PTX (NVIDIA's assembly) that, for many bandwidth-bound and moderately compute-bound kernels, lands within a few percent of expertly hand-tuned CUDA — at a tenth of the development cost.

This is the most hands-on chapter in Part IV. We will build, from nothing, four kernels of escalating difficulty: a **vector add** (to learn the programming model), a **fused softmax** (to learn reductions and row-wise work), a **matmul** (to learn tiling, accumulation, and the L2-cache `swizzle`), and a **simplified FlashAttention** (to tie it all together with the online softmax). Every kernel here is runnable. By the end you should be able to read the real FlashAttention and `fused_moe` Triton kernels in vLLM and understand every line.

This chapter assumes you have internalized [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) (SMs, warps, shared memory, HBM vs SRAM) and [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) (arithmetic intensity, memory- vs compute-bound). The softmax and attention kernels here are the *implementation* of the IO-aware ideas from [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html).

## Why Triton Exists: The Abstraction Gap

To appreciate Triton you have to feel the pain it removes. In raw CUDA, *you* are responsible for the full hierarchy of parallelism: you decide the grid of thread blocks, the threads within a block, how those threads cooperate through shared memory, how to coalesce global-memory loads so that the 32 threads of a warp touch contiguous addresses, how to avoid shared-memory bank conflicts, and how to feed the Tensor Cores with the exact fragment layout `mma` instructions demand. A competent CUDA matmul is several hundred lines; a competitive one is thousands.

Triton raises the abstraction by one crucial level: **the program operates on blocks (tiles), not scalars.** A Triton "program" (the equivalent of a CUDA thread block) loads a *block* of data, does *block*-level arithmetic, and stores a *block* of results. Inside that block, the compiler decides the thread-to-data mapping, the shared-memory staging, the vectorization, and the Tensor Core fragment layout. You think about *which tile of the output this program computes*; the compiler thinks about *how 1024 threads cooperate to compute it*.


{{fig:triton-cuda-vs-triton-mental-model}}


The trade is control for productivity. You give up the ability to micro-place every byte (which is why the very best vendor kernels — cuBLAS, CUTLASS, cuDNN, and the FP8 FlashAttention-3 kernels of [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html) — are still hand-written), but you gain the ability to write a *fused*, *correct*, *autotuned* kernel for your exact problem in an afternoon. For the long tail of custom operations in LLM training and serving — fused RMSNorm, custom RoPE, MoE routing, dequantize-and-matmul — Triton is the default tool. We compare it to writing raw CUDA in [CUDA Programming Essentials](../04-kernels-efficiency/05-cuda-essentials.html) and to the compiler-driven path in [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) (indeed, `torch.compile`'s GPU backend, TorchInductor, *generates Triton*).

!!! note "Aside: SIMT vs the block abstraction"
    A GPU executes in **SIMT** (Single Instruction, Multiple Thread) fashion: a warp of 32 threads marches in lockstep through the same instruction stream. Triton hides the warp from you almost entirely — you never write `threadIdx.x`. The compiler's job is to take your block-level program and "lower" it to a SIMT schedule. The one place the warp leaks through is performance tuning (`num_warps`), which controls how many warps cooperate on one program's tile.

## The Triton Programming Model

Five concepts carry the entire language. Learn these and the rest is detail.

1. **`@triton.jit`** — the decorator that marks a Python function as a kernel to be JIT-compiled to GPU code. Inside it, you may only use Triton operations and a restricted Python subset (no arbitrary objects, no lists of tensors; control flow and arithmetic on `tl` values).
2. **The launch grid** — when you call `kernel[grid](...)`, `grid` is a tuple (or a function returning one) giving how many independent program instances to launch. This is the CUDA grid.
3. **`tl.program_id(axis)`** — inside the kernel, this returns *which* program instance you are, along a given axis (0, 1, or 2). It is how each program figures out which slice of data it owns. This is the analog of `blockIdx`.
4. **Pointers and `tl.arange`** — Triton works with raw pointers into tensors. You compute a *block of pointers* (a vector/tensor of addresses) by adding offsets to a base pointer, then `tl.load` / `tl.store` that whole block at once.
5. **Masks** — because tensor dimensions are rarely exact multiples of your block size, you pass a boolean `mask` to `tl.load`/`tl.store` so that out-of-bounds lanes are skipped (and optionally given an `other` fill value). Masks are how Triton stays correct at the ragged edges.

### The Anatomy of a Pointer Computation

The single most important skill in Triton is turning "the elements I want" into "a block of pointers to them." A PyTorch tensor handed to a Triton kernel decays to a pointer to its first element (plus you usually pass its `.stride()` values explicitly). Suppose `x_ptr` points at a 1-D tensor and this program is responsible for the contiguous chunk starting at element `block_start`. Then:

```python
# offsets is a *vector* of BLOCK_SIZE element indices owned by this program
offsets = block_start + tl.arange(0, BLOCK_SIZE)   # e.g. [1024, 1025, ..., 2047]
# ptrs is a *vector* of BLOCK_SIZE addresses (pointer arithmetic broadcasts)
ptrs = x_ptr + offsets
# mask keeps lanes that are still inside the tensor of length n_elements
mask = offsets < n_elements
# one vectorized, coalesced load of up to BLOCK_SIZE elements
block = tl.load(ptrs, mask=mask, other=0.0)
```

For 2-D tensors you build a 2-D block of pointers by broadcasting two `arange` vectors against the row and column strides:

```python
# A BLOCK_M x BLOCK_N tile of pointers into a matrix with strides (stride_m, stride_n)
row = tl.arange(0, BLOCK_M)[:, None]   # shape (BLOCK_M, 1)
col = tl.arange(0, BLOCK_N)[None, :]   # shape (1, BLOCK_N)
ptrs = base_ptr + row * stride_m + col * stride_n   # shape (BLOCK_M, BLOCK_N)
```

This `[:, None]` / `[None, :]` broadcasting (identical to NumPy) is the workhorse. Internalize it: **rows down, columns across, strides convert logical indices to memory offsets.** `BLOCK_M`, `BLOCK_N`, and `BLOCK_SIZE` are *compile-time constants* (declared `tl.constexpr`), which lets the compiler unroll loops, size registers, and pick vector widths. Newer Triton also offers `tl.make_block_ptr`, a structured block-pointer object that tracks shape, strides, and offsets for you; we use explicit pointer math here because it makes the mechanism visible.

## Kernel 1: Vector Add — The "Hello World"

The whole machine in one screen. We add two length-`n` vectors. Each program handles `BLOCK_SIZE` consecutive elements; the grid has `ceil(n / BLOCK_SIZE)` programs.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,            # *Pointer* to first input vector
    y_ptr,            # *Pointer* to second input vector
    out_ptr,          # *Pointer* to output vector
    n_elements,       # Size of the vectors (a runtime int)
    BLOCK_SIZE: tl.constexpr,   # Elements per program — compile-time constant
):
    # 1. Which program am I? There is a 1-D grid, so we read axis 0.
    pid = tl.program_id(axis=0)

    # 2. Compute the slice of the output this program owns.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # vector of indices

    # 3. Guard against the tail block (n may not divide BLOCK_SIZE).
    mask = offsets < n_elements

    # 4. Load BLOCK_SIZE elements of x and y from HBM (masked, coalesced).
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. The actual compute — elementwise add, entirely in registers.
    out = x + y

    # 6. Write the result back to HBM.
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and out.is_cuda
    n_elements = out.numel()

    # The grid is a function of META so autotuning can change BLOCK_SIZE.
    # triton.cdiv(a, b) == ceil(a / b).
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch. The [grid] indexing syntax enqueues the kernel on the GPU.
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(98_432, device="cuda")
    y = torch.randn(98_432, device="cuda")
    out_triton = triton_add(x, y)
    out_torch = x + y
    max_err = (out_triton - out_torch).abs().max().item()
    print(f"max abs error vs torch: {max_err:.3e}")   # ~0.0
```

Note that `98_432` is not a multiple of `1024` (it is `96 * 1024 + 128`), so the last of the 97 programs has 896 masked-off lanes. The kernel is still correct because `tl.store`'s `mask` prevents those lanes from writing past the end of `out`. This is bandwidth-bound: we move 3 floats per element (read `x`, read `y`, write `out`) and do one add, an arithmetic intensity of $1/12$ FLOP/byte in fp32 — far to the left of the roofline ridge, so the kernel runs at HBM bandwidth and there is nothing clever to do. Triton's whole value here is that it took 30 lines.

!!! tip "Practitioner tip: launch overhead and `cdiv`"
    Always size the grid with `triton.cdiv(n, BLOCK)` — never integer-divide, or you silently drop the tail. And remember every `kernel[grid](...)` call has launch overhead on the order of microseconds; for tiny tensors a Triton kernel can be *slower* than eager PyTorch purely because of launch cost. Triton wins when each program has real work to do. To remove launch overhead in steady-state serving, capture the kernel in a CUDA Graph (see [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)).

## Kernel 2: Fused Softmax — Reductions and the Memory Win

Softmax over the last dimension of an `M x N` matrix is the first kernel where *fusion* pays off. The naive PyTorch path materializes several intermediate `M x N` tensors in HBM:

$$
\text{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_k e^{x_k - \max_j x_j}}
$$

A library implementation reads `x` (to find the max), reads `x` again (to exponentiate), writes the exponentials, reads them again (to sum), and reads once more to divide — roughly **5 passes over HBM**. The max-subtraction is not optional: it is the standard numerically stable softmax that prevents `exp` from overflowing for large logits (see [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html)).

The fused Triton kernel assigns **one program per row**. The program loads the entire row into registers/SRAM *once*, computes the max, the exponentials, and the sum without ever round-tripping the intermediates to HBM, and writes the row *once*. That is 1 read + 1 write — a 2.5x reduction in memory traffic, and since softmax is memory-bound, roughly a 2.5x speedup.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,   # how many elements to step to go down one row
    n_cols,                          # number of columns (runtime)
    BLOCK_SIZE: tl.constexpr,        # padded power-of-two >= n_cols
):
    # One program per row. program_id(0) is the row index.
    row_idx = tl.program_id(axis=0)

    # Pointer to the start of this row, then a vector covering all columns.
    row_start = in_ptr + row_idx * in_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_ptrs = row_start + col_offsets

    # Load the whole row. For padding lanes (col >= n_cols) we load -inf so
    # they never become the max and contribute exp(-inf) = 0 to the sum.
    mask = col_offsets < n_cols
    row = tl.load(in_ptrs, mask=mask, other=-float("inf"))

    # --- Numerically stable softmax, entirely on-chip ---
    row_max = tl.max(row, axis=0)            # block-level reduction -> scalar
    row = row - row_max                      # subtract max for stability
    numerator = tl.exp(row)                  # exp of every (real) element
    denominator = tl.sum(numerator, axis=0)  # reduction -> scalar
    out = numerator / denominator

    # Write the row back once.
    out_row_start = out_ptr + row_idx * out_row_stride
    tl.store(out_row_start + col_offsets, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2 and x.is_cuda
    M, N = x.shape
    # BLOCK_SIZE must be a power of two and cover a full row, so the row
    # fits in one program. (This simple version requires N <= ~64K.)
    BLOCK_SIZE = triton.next_power_of_2(N)
    # More warps for wider rows => more parallel reduction throughput.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    out = torch.empty_like(x)
    # One program per row => grid is (M,).
    softmax_kernel[(M,)](
        out, x,
        x.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


if __name__ == "__main__":
    x = torch.randn(1823, 781, device="cuda")
    ours = triton_softmax(x)
    ref = torch.softmax(x, dim=1)
    print("max abs error:", (ours - ref).abs().max().item())   # ~1e-7
```

Two subtleties worth dwelling on. First, `tl.max` and `tl.sum` with `axis=0` are **block reductions**: the compiler lowers them to a tree reduction across the warps and threads cooperating on the row, including a shared-memory shuffle stage. You write one line; the compiler emits the log-depth reduction. Second, the `other=-inf` fill is the elegant trick that lets the mask handle ragged column counts *and* the numerical stability simultaneously — padded lanes are `-inf`, so they lose the max comparison and their `exp` is exactly `0`, contributing nothing to the denominator.

This "load row, reduce on chip, write row" pattern is the exact same idea that, scaled up to *2-D tiles that don't fit in SRAM*, becomes FlashAttention. Keep it in mind; we return to it in Kernel 4.

!!! warning "Common pitfall: the whole row must fit"
    This kernel assumes one full row fits in a single program's tile (`BLOCK_SIZE >= N`). For an attention score row of length 128K that is false. The real fix is **tiling the reduction** with a running max and running sum — the *online softmax*, derived in detail in [FlashAttention I](../04-kernels-efficiency/02-flash-attention-1.html). Kernel 4 below implements exactly that streaming form.

## Kernel 3: Matrix Multiplication — Tiling, Accumulation, and Swizzling

Matmul is where Triton shows its compute-bound chops, because it is the one operation that *should* live on the right side of the roofline. We compute $C = A B$ where $A$ is $M\times K$, $B$ is $K\times N$, and $C$ is $M\times N$. The arithmetic is $2MNK$ FLOPs against, in the tiled scheme, far fewer bytes — high arithmetic intensity, so a good matmul saturates the Tensor Cores.

The strategy is classic **tiling**: each program computes one `BLOCK_M x BLOCK_N` tile of `C`. To do so it walks the shared `K` dimension in steps of `BLOCK_K`, at each step loading a `BLOCK_M x BLOCK_K` tile of `A` and a `BLOCK_K x BLOCK_N` tile of `B`, multiplying them with `tl.dot` (which targets the Tensor Cores), and **accumulating** into an fp32 register tile. Only after the full `K` loop does it write the tile to HBM. Each element of `A` and `B` is thus loaded from HBM once per output tile but *reused* `BLOCK_N` and `BLOCK_M` times respectively from SRAM — that reuse is the entire point.

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    # Triton benchmarks each config on the first call for a given problem
    # shape and caches the winner. These are reasonable A100/H100 starts.
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
                      num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],   # re-autotune when these change
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,    # A strides: row, col
    stride_bk, stride_bn,    # B strides: row, col
    stride_cm, stride_cn,    # C strides: row, col
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- 1. Which output tile do I compute? (with L2-cache swizzle) ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Group rows of tiles so that programs running concurrently reuse the
    # same B columns / A rows in L2. This "swizzle" is the single biggest
    # perf lever after tiling. Without it, tiles are visited row-major and
    # L2 reuse is poor.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. Build the initial block pointers for A and B tiles ----
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M   # row indices of C tile
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N   # col indices of C tile
    offs_k = tl.arange(0, BLOCK_K)
    # A tile: (BLOCK_M, BLOCK_K). B tile: (BLOCK_K, BLOCK_N).
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # ---- 3. The K-loop: load, multiply, accumulate in fp32 ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Mask the K tail so we don't read past column K.
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # tl.dot issues Tensor Core (mma) instructions; accumulate in fp32.
        acc += tl.dot(a, b)
        # Advance the pointers to the next K-tile.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)   # cast back to the output dtype

    # ---- 4. Write the tile, masking the M and N edges ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "incompatible dims"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1-D grid of (num_m_tiles * num_n_tiles) programs; swizzle maps pid->(m,n).
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((512, 768), device="cuda", dtype=torch.float16)
    b = torch.randn((768, 1024), device="cuda", dtype=torch.float16)
    ours = triton_matmul(a, b)
    ref = torch.matmul(a, b)
    # fp16 accumulation differences => compare with a tolerance.
    print("max abs error:", (ours - ref).abs().max().item())
    print("allclose:", torch.allclose(ours, ref, atol=1e-1, rtol=0))
```

Three pieces deserve emphasis.

**`tl.dot` and fp32 accumulation.** `tl.dot(a, b)` is the line that lights up the Tensor Cores. The inputs are fp16/bf16 tiles; the accumulator `acc` is fp32. This mixed-precision accumulate — multiply in low precision, sum in high precision — is exactly the pattern from [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html), and it is non-negotiable: accumulating a 768-long dot product in fp16 would lose catastrophic precision because each partial sum rounds to ~3 decimal digits.

**The group swizzle.** The block of code computing `pid_m`/`pid_n` from a flat `pid` reorders which tiles run together. The natural row-major order has many concurrently-running programs hammering different `B` columns, blowing past L2. Grouping `GROUP_M` tile-rows so that nearby programs share the same `A` rows and `B` columns dramatically improves L2 hit rate — frequently a 10–30% win at zero arithmetic cost. This is the kind of memory-traffic reasoning the roofline model trains you to do.

**`num_stages` and software pipelining.** The `num_stages` knob tells Triton how deep to *software-pipeline* the K-loop: while the Tensor Cores chew on the current `BLOCK_K` tile, the loads for the *next* tile are already in flight (issued via the GPU's async copy units). More stages hide more memory latency at the cost of more shared memory. Autotuning searches over it because the sweet spot depends on the GPU and the tile shape.

!!! example "Worked example: how much SRAM does a matmul tile need?"
    Take `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 64`, fp16 inputs (2 bytes), `num_stages = 3`. Each pipeline stage stores one A tile and one B tile in shared memory:

    - A tile: $128 \times 64 \times 2 = 16{,}384$ bytes $= 16$ KiB
    - B tile: $64 \times 128 \times 2 = 16{,}384$ bytes $= 16$ KiB
    - Per stage: $32$ KiB; with $3$ stages: $96$ KiB.

    An A100 SM has up to 164 KiB of shared memory and an H100 up to 228 KiB, so 96 KiB fits — but pushing to `num_stages = 4` ($128$ KiB) leaves little room for anything else and may cut **occupancy** (how many tiles run concurrently per SM). Meanwhile the fp32 accumulator is $128 \times 128 \times 4 = 65{,}536$ bytes spread across registers, not shared memory. This is precisely the resource budgeting Triton's autotuner explores for you — and why a "bigger tile" is not always faster. (See [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) for occupancy.)

A handwritten Triton matmul like this typically reaches a large fraction of cuBLAS on dense GEMMs. You usually would *not* ship it to replace cuBLAS; you write it so you can **fuse** something cuBLAS won't — a dequantize-then-matmul for INT4 weights ([Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html)), a matmul-plus-activation, or the per-expert GEMMs of an MoE ([Mixture-of-Experts](../02-transformer/09-mixture-of-experts.html)).

## Kernel 4: A Simplified FlashAttention

Now we assemble everything. Attention computes, for queries $Q$, keys $K$, values $V$ (each $N \times d$ for one head):

$$
\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

The naive route materializes the full $N \times N$ score matrix $S = QK^\top/\sqrt{d}$ in HBM. For a 16K-token sequence that is $16384^2 = 268$M entries *per head* — quadratic memory that dominates the runtime and caps context length. FlashAttention's insight (Dao et al.) is that you never need the full $S$ in HBM: you can stream over the keys/values in blocks, maintaining a **running softmax** so the output of each query block is built incrementally. This is the online softmax from Kernel 2, generalized to a reduction that does not fit in SRAM. Its derivation lives in [FlashAttention I](../04-kernels-efficiency/02-flash-attention-1.html); here we *implement* it.

### The Online Softmax Recurrence

Process key/value blocks $j = 1, 2, \dots$ For each query block we keep three running statistics: $m$ (running max of scores), $\ell$ (running sum of exponentials), and the unnormalized output accumulator $O$. On seeing a new score block $S^{(j)}$ with block max $m^{(j)} = \max S^{(j)}$:

$$
m_{\text{new}} = \max(m, m^{(j)}), \qquad
\alpha = e^{\,m - m_{\text{new}}}
$$

$\alpha$ is the **correction factor**: because the max changed, every previously accumulated term was exponentiated against the *old* max and must be rescaled by $\alpha$. We then update:

$$
\ell_{\text{new}} = \alpha\,\ell + \textstyle\sum e^{\,S^{(j)} - m_{\text{new}}}, \qquad
O_{\text{new}} = \alpha\,O + e^{\,S^{(j)} - m_{\text{new}}}\,V^{(j)}
$$

After the last block, divide once: $O \leftarrow O / \ell$. The output is bit-for-bit the same as a full softmax-then-matmul, but no $N\times N$ tensor ever touched HBM. The kernel below implements the forward pass for one `(batch, head)` per program row-block.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,                       # 1/sqrt(d), folded with log2e below
    stride_qb, stride_qh, stride_qm, stride_qd,   # Q strides (B,H,seq,dim)
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, N_CTX,
    BLOCK_M: tl.constexpr,          # query block (rows handled per program)
    BLOCK_N: tl.constexpr,          # key/value block (streamed)
    HEAD_DIM: tl.constexpr,         # d, the per-head dimension
):
    # ---- Which query block, and which (batch, head)? 2-D grid. ----
    start_m = tl.program_id(0)      # query-block index along the sequence
    off_bh = tl.program_id(1)       # flattened (batch * head) index
    off_b = off_bh // H
    off_h = off_bh % H

    # Base pointers into this (batch, head)'s matrices.
    q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = V_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = O_ptr + off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # this block's query rows
    offs_d = tl.arange(0, HEAD_DIM)                       # the head dimension

    # Load THIS query block once; it stays resident the whole kernel.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # ---- Online-softmax running state ----
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                 # running sum
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)        # running output

    # Fold 1/sqrt(d) and log2(e) so we can use the faster base-2 exp2.
    qk_scale = sm_scale * 1.44269504   # 1/ln(2)

    # ---- Stream over key/value blocks ----
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_CTX

        # Load a K block (HEAD_DIM x BLOCK_N) and a V block (BLOCK_N x HEAD_DIM).
        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Scores for this block: (BLOCK_M, BLOCK_N). tl.dot -> Tensor Cores.
        s = tl.dot(q, k) * qk_scale
        # Mask out padded keys so they never win the max / contribute to sum.
        s = tl.where(n_mask[None, :], s, -float("inf"))

        # ----- online softmax update -----
        m_ij = tl.max(s, axis=1)                 # per-row block max
        m_new = tl.maximum(m_i, m_ij)            # new running max
        alpha = tl.exp2(m_i - m_new)             # correction for prior state
        p = tl.exp2(s - m_new[:, None])          # exp of this block's scores
        l_i = l_i * alpha + tl.sum(p, axis=1)    # rescale + add new mass
        acc = acc * alpha[:, None]               # rescale accumulated output
        acc += tl.dot(p.to(v.dtype), v)          # add this block's contribution
        m_i = m_new                              # commit new max

    # ---- Final normalization (one division at the end) ----
    acc = acc / l_i[:, None]

    # Write the output block.
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention(q, k, v, sm_scale=None):
    # q,k,v: (B, H, N_CTX, HEAD_DIM)
    B, H, N_CTX, HEAD_DIM = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)   # (query blocks, batch*head)
    flash_attn_kernel[grid](
        q, k, v, o, sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
    )
    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 256, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    ours = flash_attention(q, k, v)

    # Reference: plain (non-causal) attention in fp32 for accuracy.
    scale = 1.0 / (D ** 0.5)
    ref = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * scale, dim=-1) @ v.float()
    print("max abs error:", (ours.float() - ref).abs().max().item())   # ~1e-2 (fp16)
```

A few notes that connect this to production kernels.

**Why `exp2` instead of `exp`.** GPUs have a fast hardware approximation for base-2 exponentiation (`exp2` / `ex2.approx`). FlashAttention folds the $1/\ln 2$ factor into the QK scale (`qk_scale = sm_scale * 1.44269...`) so that $e^{x} = 2^{x \log_2 e}$ becomes a single `exp2`. It is a small constant-factor win that the real kernels all use.

**This is the non-causal version.** A causal (decoder) attention adds a mask so query $i$ only attends to keys $j \le i$, which lets the kernel *skip* key blocks entirely beyond the diagonal — a roughly 2x saving. You implement it by (a) breaking the loop at the block containing the diagonal and (b) applying a triangular mask in the diagonal block via `tl.where(offs_m[:, None] >= offs_n[None, :], s, -inf)`. We omit it for clarity; the real vLLM/FlashAttention Triton kernels include it.

**The memory win, quantified.** The naive path reads/writes the $N\times N$ score matrix. Our kernel keeps $Q$, the running $O$, $m$, and $\ell$ in registers/SRAM and streams $K$,$V$ tiles — total HBM traffic is $O(N d)$ per head, not $O(N^2)$. This is *the* reason long-context attention is feasible at all. The backward pass needs the same statistics recomputed (FlashAttention recomputes $S$ in the backward rather than storing it — trading FLOPs for memory), which is covered in the FlashAttention chapters.

!!! warning "Common pitfall: forgetting to rescale the accumulator"
    The most common bug when writing your first online-softmax kernel is rescaling `l_i` by `alpha` but forgetting to rescale `acc` by `alpha` (or vice versa). Both the running sum *and* the running output accumulator were computed against the old max, so **both** must be multiplied by `alpha = exp(m_old - m_new)` before adding the new block's contribution. A unit test against a dense fp32 reference (as above) catches this instantly — always write that test first.

## Autotuning, Debugging, and Performance Practice

Writing a correct kernel is half the job; making it fast and trusting it is the other half. A few tools and habits.

**Autotuning.** As shown in the matmul, `@triton.autotune` takes a list of `triton.Config`s (each a set of `constexpr` meta-parameters plus `num_warps` and `num_stages`) and a `key` of argument names. On the first launch for each distinct key, Triton benchmarks every config and caches the winner. The knobs that matter most:

| Knob | What it controls | Typical effect |
|---|---|---|
| `BLOCK_M/N/K` | Tile sizes | Bigger tiles → more reuse, more SRAM/registers, less occupancy |
| `num_warps` | Warps cooperating per program | More warps → more parallel reduction, finer pipelining |
| `num_stages` | Software-pipeline depth on loops | Hides memory latency, costs shared memory |
| `GROUP_M` | L2 swizzle group size | Improves L2 reuse for GEMM-like kernels |

**Debugging.** Three indispensable tools:

```python
# 1. Run the kernel on the CPU in pure Python, element by element, so you can
#    use print() and a debugger. Slow, but exact — your first line of defense.
import os
os.environ["TRITON_INTERPRET"] = "1"   # set BEFORE importing triton

# 2. Inside an interpreted kernel you can even print tiles:
#    tl.device_print("scores", s)

# 3. Sanity-check shapes/strides — most bugs are pointer-arithmetic bugs.
#    Always diff against an eager-PyTorch reference with torch.allclose.
```

The single most valuable habit: **write the eager-PyTorch reference first, then make the kernel match it** under `torch.allclose` with an appropriate tolerance (looser for fp16/bf16 because of accumulation order differences). Almost every Triton bug is a pointer/stride mistake or a missing mask, and a reference test localizes it immediately.

**Benchmarking.** Use `triton.testing.do_bench`, which handles GPU warmup, CUDA-stream synchronization, and the L2-cache flush between runs that naive `time.time()` benchmarks get wrong:

```python
import triton

ms = triton.testing.do_bench(lambda: triton_matmul(a, b))
flops = 2 * M * N * K
print(f"{ms:.3f} ms,  {flops / (ms * 1e-3) / 1e12:.1f} TFLOP/s")
```

**Where Triton fits in the stack.** TorchInductor — the backend of `torch.compile` — *generates Triton code* for fused pointwise and reduction kernels automatically. So even if you never write a `@triton.jit` function, you are running Triton when you `torch.compile` a model on an NVIDIA GPU. Writing kernels by hand is for the cases the compiler can't fuse well: novel attention variants, quantized matmuls, MoE dispatch, custom losses. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html).

!!! interview "Interview Corner"
    **Q:** A candidate proposes rewriting softmax in Triton to make it faster. Walk me through *why* it's faster than calling several PyTorch ops, and what the speedup is fundamentally limited by.

    **A:** Softmax is **memory-bound**: its arithmetic intensity is low, so runtime is set by HBM traffic, not FLOPs. The multi-op PyTorch path materializes intermediates (the max, the exponentials, the sum) to HBM and re-reads them — on the order of five passes over the data. A fused Triton kernel assigns one program per row, loads the row into SRAM/registers **once**, computes the max, exp, and sum on-chip, and writes the result **once** — about two passes. So the speedup is roughly the ratio of HBM bytes moved, around 2–2.5x, and it's fundamentally **capped by memory bandwidth**: you cannot beat one read plus one write, so no amount of cleverness gets you past ~2.5x over the naive path. The same fusion logic, generalized to a reduction too large for SRAM via the online (running-max) softmax, is exactly what makes FlashAttention an IO-aware win. I'd confirm the bound by computing arithmetic intensity and checking it against the roofline ridge point.

## Key Takeaways

!!! key "Key Takeaways"
    - Triton raises the GPU abstraction to **blocks/tiles**: you write NumPy-like code on tiles and `program_id` selects which tile you own; the compiler handles thread mapping, coalescing, shared memory, and Tensor Core layout.
    - The five load-bearing concepts are `@triton.jit`, the **launch grid**, `tl.program_id`, **pointer-block arithmetic** (`base + offsets`, broadcast with `[:,None]`/`[None,:]`), and **masks** for ragged edges and numerical fills (`other=-inf`).
    - **Fusion is the win for memory-bound ops:** fused softmax loads a row once and writes once (~2 HBM passes vs ~5), giving ~2.5x — and that ceiling is set by bandwidth, not compute.
    - **Matmul is about reuse:** tile the `M`,`N`,`K` dims, accumulate `tl.dot` results in **fp32**, and use the **GROUP_M swizzle** for L2 locality; `num_stages` software-pipelines the K-loop to hide load latency.
    - **FlashAttention = online softmax over streamed K/V tiles:** keep running `m`, `ℓ`, and `O`; rescale **both** `ℓ` and `O` by `α = exp(m_old − m_new)` on every block; normalize once at the end. HBM traffic drops from $O(N^2)$ to $O(Nd)$.
    - Always **write an eager-PyTorch reference first** and diff with `torch.allclose`; debug with `TRITON_INTERPRET=1`; benchmark with `triton.testing.do_bench`.
    - Let `@triton.autotune` search `BLOCK_*`, `num_warps`, `num_stages`, `GROUP_M`; the best config depends on GPU, dtype, and problem shape — and bigger tiles can *lower* occupancy.
    - You usually don't beat cuBLAS/cuDNN with hand Triton; you write Triton to **fuse** what they can't (quantized GEMMs, custom attention, MoE), and TorchInductor already emits Triton under `torch.compile`.

!!! sota "State of the Art & Resources (2026)"
    Triton is now the default GPU kernel language for the LLM stack: TorchInductor emits it under `torch.compile`, and virtually every major inference engine (vLLM, Unsloth, SGLang) ships hand-written Triton kernels for fused attention, RMSNorm, RoPE, and MoE routing. The compiler itself is actively evolving — gaining distributed-memory primitives, AMD/Intel backends, and LLM-assisted autotuning — while the FlashAttention line (v1/v2/v3) remains the canonical showcase of what expert Triton kernels can achieve.

    **Foundational work**

    - [Tillet, Kung & Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (2019)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — the original MAPL paper that introduced the block/tile abstraction and the Triton compiler.
    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — the paper that defined IO-aware kernel design and the online softmax; Kernel 4 in this chapter is its direct implementation.

    **Recent advances (2023–2026)**

    - [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691) — improved work distribution across warps and thread blocks, ~2× speedup over FA-1 and 50–73% of peak A100 FLOPs.
    - [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608) — exploits H100 WGMMA and TMA instructions for async pipelining; reaches 75% of peak H100 FLOPs.
    - [Hsu et al., *Liger Kernel: Efficient Triton Kernels for LLM Training* (2024)](https://arxiv.org/abs/2410.10989) — drop-in fused kernels for RMSNorm, RoPE, SwiGLU, and cross-entropy; 20% throughput gain and 60% memory reduction over HuggingFace defaults.
    - [Ringlein et al., *The Anatomy of a Triton Attention Kernel* (2025)](https://arxiv.org/abs/2511.11581) — step-by-step walkthrough of building a production paged-attention kernel in Triton that achieves cross-platform SOTA on NVIDIA and AMD.

    **Open-source & tools**

    - [triton-lang/triton](https://github.com/triton-lang/triton) — the official Triton language and compiler repository; includes the canonical vector-add, softmax, matmul, and fused-attention tutorial kernels.
    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference implementations of FlashAttention v1/v2/v3 in both CUDA and Triton; the benchmark to beat for any attention kernel.
    - [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — production-ready Triton kernels for LLM training (RMSNorm, RoPE, SwiGLU, cross-entropy) compatible with HuggingFace Transformers and FSDP.
    - [unslothai/unsloth](https://github.com/unslothai/unsloth) — fine-tuning library whose entire backward pass is rewritten as hand-crafted Triton kernels, delivering 2× faster training with 70% less VRAM.

    **Go deeper**

    - [Official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — the maintained, runnable reference for every kernel type covered in this chapter: vector add, fused softmax, matmul, layer norm, and fused attention (FA-2).
    - [PyTorch 2.x — torch.compile and TorchInductor](https://pytorch.org/get-started/pytorch-2-x/) — explains how TorchInductor uses Triton as its GPU codegen backend, with benchmarks across 163 open-source models.

## Further reading

- Philippe Tillet, H. T. Kung, David Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (2019) — the original paper.
- The official **OpenAI Triton** tutorials (vector-add, fused softmax, matrix multiplication, and the FlashAttention example) — the canonical, maintained reference implementations these kernels follow.
- Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022), and *FlashAttention-2* (2023) — the online-softmax and work-partitioning ideas behind Kernel 4.
- NVIDIA **CUTLASS** documentation — for the layout/tiling/pipelining concepts that Triton automates, seen from the hand-written CUDA side.
- The **vLLM** and **Unsloth** repositories — production Triton kernels (paged/flash attention, fused MoE, RMSNorm, RoPE) worth reading once you can follow this chapter's code.
