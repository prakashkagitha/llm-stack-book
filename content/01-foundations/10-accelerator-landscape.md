# 1.10 The Accelerator Landscape: TPUs, Trainium, AMD/ROCm & Gaudi

For most of the last decade, "running an LLM" silently meant "running it on an NVIDIA GPU with CUDA." That assumption is no longer safe, and a working LLM engineer who only knows CUDA is now under-equipped. The largest production language models on earth — Google's Gemini family — never touch an NVIDIA chip; they train and serve on **TPUs**. Anthropic and many AWS customers train and serve on **Trainium** and **Inferentia**. AMD's **Instinct MI300X**, with 192 GB of HBM3 per package, has become a credible serving target precisely because its memory capacity lets a 70B-class model fit on fewer devices. Intel's **Gaudi** is the budget challenger. Procurement reality — supply, price, and the simple fact that you cannot buy as many H100s as you want — means you *will* eventually be handed a non-NVIDIA fleet and asked to make it sing.

This chapter is a vendor-by-vendor field guide to that landscape. The goal is not to memorize spec sheets that will be stale in a year, but to internalize the *mental models* that let you read any new spec sheet and any new SDK and reason about it: the difference between a **systolic array** and a **SIMT** machine; what the compiler does on each platform; what ports cleanly from CUDA and what does not; and how to look at HBM capacity, HBM bandwidth, and FP8/FP4 TFLOPS and decide whether a given chip is right for *your* training or serving workload.

We build directly on [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) (the SIMT baseline we are contrasting against), [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) (the FP8/FP4/bf16 formats whose throughput we compare), and [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html) (the interconnects — NVLink vs ICI vs NeuronLink vs Infinity Fabric — that decide how chips scale into pods). When the discussion turns to *programming* these chips, we lean on [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html) and [CUDA Programming Essentials for ML Engineers](../04-kernels-efficiency/05-cuda-essentials.html). The roofline reasoning we use to pick hardware is developed fully in [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).

---

## Two Silicon Philosophies: Systolic Arrays vs SIMT

Before any vendor, you need the one architectural distinction that organizes the whole chapter. There are two fundamentally different ways to build a chip that multiplies matrices fast, and almost every accelerator is a point on the spectrum between them.

### The SIMT GPU (NVIDIA, AMD)

You met this in [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html). A GPU is a sea of small programmable cores (NVIDIA: **SMs**; AMD: **CUs**, compute units) running threads in lockstep groups (NVIDIA **warp** = 32 lanes; AMD **wavefront** = 64 lanes). Matrix multiply throughput comes from bolting on dedicated MMA units — NVIDIA **Tensor Cores**, AMD **Matrix Cores** — but the chip remains *general-purpose and programmable*. You write a kernel, the threads do whatever you tell them, and the same silicon can run a softmax, a sort, a physics simulation, or a ray tracer. Flexibility is the defining feature; the cost is that you, the programmer, must orchestrate the memory hierarchy (registers → shared memory → L2 → HBM) by hand to feed the math units.

### The systolic array (TPU, Trainium, Gaudi's MME)

A **systolic array** is the opposite bet. Imagine a 2D grid — say $128 \times 128$ — of tiny multiply-accumulate (MAC) cells. Data is *pumped* through the grid like blood through a heart (Greek *systolē*, contraction — hence "systolic"). Weights are pre-loaded into the cells; activations flow in from the left edge, one column per clock; partial sums flow down. Each cell, every cycle, does one fused multiply-add and passes its operands to its neighbors. After the pipeline fills, the array retires a full $128 \times 128$ matrix-multiply tile's worth of MACs *every single clock cycle*, with almost no instruction-fetch overhead and — critically — **almost no register-file or SRAM traffic for the intermediate partial sums**, because they never leave the array; they march cell-to-cell.


{{fig:accel-systolic-array-dataflow}}


This is spectacularly efficient *for dense matrix multiply* — far higher FLOPs per watt and per transistor than a general SIMT core, because almost all the silicon is MAC units and almost none is control logic or caches. The price is rigidity. A systolic array does one thing — large dense matmuls — superbly, and *everything else* (the elementwise ops, the softmax, the layernorm, the gather/scatter, the dynamic shapes) must be handled by a smaller companion vector unit and, crucially, must be arranged ahead of time by a **compiler** that knows the static shape of every tensor. There is no `__syncthreads()`, no hand-written warp shuffle, no in-kernel branching on data. You do not program a TPU thread-by-thread; you describe a *dataflow graph* and a compiler maps it onto the array.

Hold onto this single sentence, because it explains 90% of the porting pain in this chapter: **GPUs give you a programmable machine and make you manage memory; systolic accelerators give you a compiler and make you accept its abstractions.**

{{fig:accel-simt-vs-systolic-silicon-budget}}

The arithmetic-intensity reasoning from [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) still governs both. A chip's matmul units can only run at peak if the operands arrive fast enough. We formalize this with the ridge point of the roofline:

$$
I^{*} = \frac{\text{peak compute (FLOP/s)}}{\text{peak memory bandwidth (byte/s)}}
$$

Any operation whose arithmetic intensity (FLOPs per byte of HBM traffic) is below $I^{*}$ is **memory-bound** no matter how many MAC cells the chip has. For an H100 at roughly $\sim 990$ bf16 TFLOP/s over $\sim 3.35$ TB/s of HBM, $I^{*} \approx 295$ FLOP/byte. For a TPU or MI300X the numbers differ, but the logic is identical — and it is why LLM *decode*, which is fundamentally bandwidth-bound, often runs *better* on the memory-rich MI300X than its raw FLOPS would suggest.

---

## Google TPU: The Original Systolic Accelerator

The Tensor Processing Unit is the most mature non-NVIDIA accelerator and the one that has trained the most frontier-scale models. Its lineage runs from the inference-only TPU v1 (2015) through training-capable v2/v3, the v4/v5 generation (v5e for cost-efficiency, v5p for peak training), the v6 ("Trillium"), and the v7 ("Ironwood") generation aimed squarely at large-scale inference and reasoning workloads.

### The chip: MXU + VPU + HBM

A modern TPU chip contains a small number of large **MXUs** (Matrix Multiply Units) — the systolic arrays, classically $128 \times 128$ MAC grids — paired with a **VPU** (Vector Processing Unit) for the elementwise/reduction work the MXU cannot do, plus **scalar units** and a large slab of HBM. A v5p chip, for example, carries on the order of 95 GB of HBM; v6/v7 push capacity and bandwidth substantially higher. The defining feature versus a GPU is the *ratio*: enormous dense-matmul throughput concentrated in a few big arrays, fed by a comparatively simple, compiler-scheduled memory system, with no programmer-visible warp/SM hierarchy and no shared-memory scratchpad to manage by hand.

### Pods and ICI: the interconnect is the product

A single TPU chip is unremarkable; the **pod** is the point. TPUs are wired together with a dedicated **ICI** (Inter-Chip Interconnect) into a **2D or 3D torus** topology. In a torus, each chip talks directly to its neighbors and the mesh wraps around at the edges, so collective operations — the all-reduce and all-gather of [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html) — flow around rings with no expensive central switch. A full v5p pod connects thousands of chips (v5p: up to 8,960) into one tightly-coupled machine with a flat, predictable bandwidth model. This torus-with-good-collectives design is *why* TPUs scale to enormous training jobs gracefully: the hardware topology and the parallelism strategy are co-designed.

### The programming model: XLA, JAX, and "the compiler is the API"

You do not write TPU kernels by hand in the CUDA sense. You write high-level array code — almost always **JAX**, sometimes TensorFlow or PyTorch/XLA — and the **XLA** (Accelerated Linear Algebra) compiler lowers your whole-program computation graph onto the MXU/VPU/ICI. XLA does the heavy lifting that a CUDA programmer does manually: operator fusion, memory layout assignment, tiling for the systolic array, and — through **GSPMD/`shard_map`** — inserting the cross-chip collectives implied by your sharding annotations.

```python
# JAX on TPU: data-parallel + tensor-parallel matmul across a pod slice.
# The KEY idea: you annotate HOW arrays are sharded across the physical
# chip mesh; XLA inserts every all-gather / reduce-scatter for you.
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# 1) Discover the physical TPU chips and arrange them into a logical 2D mesh.
#    Axis "data" = data parallelism; axis "model" = tensor parallelism.
devices = mesh_utils.create_device_mesh((4, 2))   # e.g. an 8-chip slice
mesh = Mesh(devices, axis_names=("data", "model"))

def shard(x, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))

# 2) Shard a weight matrix's columns across the "model" axis (tensor parallel),
#    and the activation batch across the "data" axis (data parallel).
W = shard(jnp.ones((8192, 8192)),  P(None, "model"))   # [in, out/TP]
x = shard(jnp.ones((1024, 8192)),  P("data", None))    # [batch/DP, in]

@jax.jit                      # <-- this single decorator invokes XLA.
def layer(x, W):
    # You write a plain matmul. XLA sees the shardings on x and W and
    # automatically emits the all-gather over "model" needed to form the
    # full output, fused with the matmul, tiled for the 128x128 MXU.
    return jnp.tanh(x @ W)

y = layer(x, W)               # runs across all 8 chips, collectives inserted.
print(y.shape, y.sharding)    # (1024, 8192), sharded as XLA decided.
```

The mental shift for a GPU person: there is no kernel to profile in the Nsight sense, no occupancy to tune, no shared-memory bank conflict to chase. Your performance levers are (a) the **sharding annotations** — get the parallelism wrong and XLA inserts catastrophic collectives — (b) keeping shapes **static** so XLA can compile once, and (c) avoiding ops XLA cannot fuse well. When you *do* need a custom kernel — a fused FlashAttention variant, a block-sparse MoE op — you reach for **Pallas**, JAX's kernel language (spiritually a Triton for TPU/GPU) that lets you write tiled programs against the MXU/VPU directly. This is the TPU answer to [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html).

!!! note "Aside: why TPUs love bf16 and big batches"
    TPUs were co-designed with **bfloat16** (Google invented the format). bf16 keeps FP32's 8 exponent bits — same dynamic range — and throws away mantissa bits, which is exactly the trade a systolic MAC array wants: wide range so you rarely need loss scaling, narrow mantissa so each MAC cell is cheap. Because the MXU retires a full tile every cycle once filled, TPUs are happiest with *large, statically-shaped* matmuls — big batch, big hidden dim. Tiny, dynamically-shaped, branchy workloads (the kind that plague a SIMT GPU less) are where the systolic model struggles, since the array drains and refills.

---

## AWS Trainium & Inferentia: Systolic Arrays Behind the Neuron SDK

Amazon designed two custom chips through its Annapurna Labs group: **Inferentia** (inference) and **Trainium** (training, also strong at inference). Trainium2 powers large training clusters ("UltraClusters") and the UltraServer configurations that connect many chips with the **NeuronLink** interconnect; Trainium3 is the next step up. Anthropic's training and serving at scale on Trainium make this a first-class target, not a curiosity.

### Architecture: NeuronCores

Each Trainium chip contains multiple **NeuronCores**. A NeuronCore is itself heterogeneous: a systolic-array **TensorEngine** for matmuls, a **VectorEngine** and **ScalarEngine** for the elementwise/reduction/activation work, and a **GPSIMD** engine for more general operations, all fed from on-chip SBUF/PSUM scratchpad memory and HBM. The shape is the same family as the TPU — big systolic matmul engine plus vector helpers plus a compiler — but with AWS's own scratchpad and interconnect design.

### The programming model: Neuron SDK, compile-ahead, and `xla`

You target Trainium through the **AWS Neuron SDK**. The default path is **PyTorch/XLA**: you write ordinary PyTorch, but execution goes through the `torch_xla` bridge, which traces your model into an XLA graph and the **Neuron Compiler (`neuronx-cc`)** lowers it onto the NeuronCores. JAX is also supported via XLA. The dominant practical fact, just like TPU, is that this is a **trace-and-compile** model, not eager execution.

```python
# Trainium via PyTorch/XLA. The code looks like normal PyTorch, but the
# device is 'xla' and execution is LAZY: ops are recorded into a graph and
# only compiled+run when you force a materialization (xm.mark_step()).
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()                 # an 'xla:0' NeuronCore, not 'cuda'.

model = MyTransformer().to(device)       # weights live on the NeuronCore.
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step, (xb, yb) in enumerate(loader):
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    loss = model(xb, yb)                 # NOTHING executes yet -- lazy trace.
    loss.backward()                      # still just building the graph.
    opt.step()
    # mark_step() cuts the graph, hands it to neuronx-cc, and runs it.
    # The FIRST occurrence of a given graph shape triggers a (slow) compile;
    # afterwards the compiled binary is cached and reused.
    xm.mark_step()
```

The big operational gotchas for a GPU engineer moving to Trainium are exactly the gotchas of any lazy/compiled stack: (1) **dynamic shapes trigger recompilation** — if your sequence length or batch size varies every step, you thrash the compiler, so you bucket/pad to a fixed set of shapes; (2) **data-dependent control flow** (Python `if` on a tensor value) forces graph breaks and is slow; (3) the **first step is dominated by compilation**, so warm-up and a persistent compile cache matter. When you need a true custom kernel, the **Neuron Kernel Interface (NKI)** is the Trainium analog of Triton/Pallas — a Python-embedded language for writing tiled kernels directly against the TensorEngine/VectorEngine.

!!! tip "Practitioner tip: bucket your shapes on any XLA backend"
    On TPU *and* Trainium, the single highest-leverage habit is to **enumerate and fix your tensor shapes**. Pad sequences to a small set of length buckets (e.g. 512/1024/2048/4096), fix the batch size, and pad the vocabulary to a friendly multiple. You pay a one-time compile per unique shape and then run compiled binaries forever. The most common "TPU/Trainium is mysteriously slow" bug is silent recompilation from shapes that wobble step to step.

---

## AMD Instinct & ROCm: The Closest Thing to a CUDA Drop-In

AMD's **Instinct** GPUs are the strategic alternative to NVIDIA because they are *also SIMT GPUs* — same philosophy, so the porting story is fundamentally easier than for any systolic chip. The **MI300X** pairs a CDNA-architecture GPU with **192 GB of HBM3** (versus 80 GB on an H100), and the **MI325X**/**MI350** generation pushes capacity and bandwidth further, with MI350 adding native FP4/FP6 support aimed at low-precision inference. That memory capacity is the headline feature: a model that needs two or three H100s to hold weights plus KV cache may fit on a single MI300X, collapsing tensor-parallel communication and simplifying serving.

### CDNA: CUs, wavefronts, and Matrix Cores

The MI300X is built from **XCDs** (accelerator complex dies) carrying **Compute Units (CUs)**, AMD's equivalent of SMs. Threads execute in **wavefronts of 64** (versus NVIDIA's 32-lane warps) — a real difference that affects how you write and tune kernels. Each CU has **Matrix Cores**, AMD's MMA units analogous to Tensor Cores, executing **MFMA** (Matrix Fused Multiply-Add) instructions. The whole package is glued together with AMD's **Infinity Fabric** for chiplet-to-chiplet and GPU-to-GPU links. If you understood [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html), you already understand the MI300X — only the names and the wavefront width change.

### ROCm and HIP: CUDA with the serial numbers filed off

AMD's software stack is **ROCm** (Radeon Open Compute). The programming language is **HIP** (Heterogeneous-Compute Interface for Portability), which is deliberately a near-clone of CUDA: the API calls are renamed `cuda*` → `hip*`, the kernel-launch syntax is the same, and the device-code intrinsics map closely. The promise is *source portability*: one HIP source compiles to AMD (via the ROCm compiler) or to NVIDIA (HIP-over-CUDA).

```cpp
// A HIP kernel. Compare this to CUDA -- it is nearly identical.
// Compile for AMD with:   hipcc saxpy.cpp -o saxpy
#include <hip/hip_runtime.h>

// y = a*x + y   (the classic SAXPY), one thread per element.
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // same index math as CUDA
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    int n = 1 << 20;
    float *dx, *dy;
    hipMalloc(&dx, n * sizeof(float));   // cudaMalloc      -> hipMalloc
    hipMalloc(&dy, n * sizeof(float));
    // ... fill dx, dy ...
    int threads = 256, blocks = (n + threads - 1) / threads;
    // CUDA's <<<blocks,threads>>> triple-chevron is supported by hipcc too:
    hipLaunchKernelGGL(saxpy, dim3(blocks), dim3(threads), 0, 0,
                       n, 2.0f, dx, dy);
    hipDeviceSynchronize();              // cudaDeviceSynchronize -> hipDevice...
    hipFree(dx); hipFree(dy);            // cudaFree              -> hipFree
}
```

For an existing CUDA codebase, AMD ships **`hipify`** tools (`hipify-perl`, `hipify-clang`) that mechanically rewrite `cuda*` calls to `hip*`. Plumbing — `cudaMalloc`, `cudaMemcpy`, stream and event management, most of the C++ — translates almost mechanically. The Python ML stack is even smoother: there is a **ROCm build of PyTorch** in which `torch.cuda` is *retargeted* to the AMD device, so a remarkable amount of model code runs unchanged — you still call `tensor.cuda()` and `device="cuda"`, and it lands on the Instinct GPU.

### What ports cleanly — and what does not

Be precise here, because "PyTorch just works on AMD" oversells it and "nothing works" undersells it.

- **Ports cleanly:** high-level PyTorch/JAX models on the ROCm builds; straightforward HIP kernels; the overall structure of any CUDA program.
- **Ports with friction:** anything **Triton**. Triton has an AMD backend, so many kernels recompile and run — but performance tuning (block sizes, `num_warps`, pipelining) was done for NVIDIA and must be re-tuned for the 64-wide wavefront and CDNA memory system. FlashAttention and the fused kernels inside vLLM/SGLang have AMD ports, but they tend to lag the NVIDIA versions and need version-matched ROCm.
- **Ports painfully or not at all:** hand-written **PTX/SASS** inline assembly (NVIDIA-specific ISA), kernels using NVIDIA-only hardware features (e.g. `wgmma`/TMA-style Hopper instructions, certain `cp.async` patterns), and anything depending on **CUDA-only libraries** without a ROCm analog. The ROCm analogs exist — **rocBLAS/hipBLASLt** for cuBLAS, **MIOpen** for cuDNN, **RCCL** for NCCL, **CK** (Composable Kernel) for CUTLASS — but the surface area is not 100% and the bleeding edge of NVIDIA appears first.

!!! warning "Common pitfall: assuming `warpSize == 32` on AMD"
    Mountains of CUDA code bake in the constant 32 — warp-shuffle reductions, masks, `__ballot` patterns, shared-memory tile sizes chosen for 32 lanes. On a CDNA GPU a wavefront is **64** lanes. `hipify` will not fix the *logic*; it only renames API calls. A reduction that loops `for (offset = 16; offset > 0; offset >>= 1)` assumes 32-wide and will silently drop half the lanes on AMD. Always read `warpSize` (a runtime value on AMD) rather than hard-coding 32, and re-derive tile sizes from it.

---

## Intel Gaudi: The Budget Challenger

Intel's **Gaudi** line (from the Habana Labs acquisition) — Gaudi2 and **Gaudi3** — is positioned as a cost-per-token challenger. Architecturally it is another heterogeneous design: a cluster of fully-programmable **TPCs** (Tensor Processor Cores, VLIW SIMD vector units) plus configurable **MME** (Matrix Multiplication Engine) systolic blocks for the dense matmuls, with HBM and — its most distinctive hardware choice — **on-die integrated RoCE** (RDMA over Converged Ethernet) ports. Instead of a proprietary fabric, every Gaudi card has many 100+ GbE RoCE links, so chips scale out over *standard Ethernet*, which is attractive for datacenters that would rather not build a proprietary interconnect island.

The software stack is **SynapseAI**, exposed to PyTorch through the **`habana_frameworks`** bridge. Like Trainium and TPU it is fundamentally a **graph-compile** model (lazy by default, with an eager mode), and like them it rewards static shapes and punishes recompilation.

```python
# Gaudi via the Habana PyTorch bridge. Note the 'hpu' device.
import torch
import habana_frameworks.torch.core as htcore   # registers the 'hpu' backend

device = torch.device("hpu")                     # not 'cuda', not 'xla'
model = MyTransformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    loss = model(xb, yb)
    loss.backward()
    opt.step()
    htcore.mark_step()   # like xm.mark_step(): cut & dispatch the graph.
```

The takeaway: Gaudi sits in the same "systolic-MME + programmable-vector + graph-compiler" family as TPU and Trainium, with the twist that its scale-out is plain Ethernet/RoCE. The porting mental model is the XLA-family one (bucket shapes, avoid graph breaks, warm the compile cache), not the CUDA one.

---

## The Portability Spectrum: A Unified Map

Step back and the whole landscape collapses onto one axis — *how much of the NVIDIA programming model survives*:


{{fig:accel-portability-spectrum}}


A few organizing principles fall out of this map and are worth stating as rules of thumb:

1. **SIMT ports to SIMT.** Moving CUDA → AMD is a *translation* problem (rename APIs, re-tune for 64-wide wavefronts, swap libraries). Moving CUDA → TPU/Trainium/Gaudi is a *paradigm* problem (rewrite to graph-level array code and let a compiler own the kernels).
2. **The higher you wrote your code, the more portable it is.** Pure PyTorch/JAX module code is the most portable artifact in ML — it runs on all four families with minimal change. Triton is the next tier (recompiles, re-tunes). Hand-written CUDA C++ with PTX is the least portable.
3. **On compiler-first chips, your kernel is the compiler's problem, so your job is shapes and sharding.** The performance work moves from "tune occupancy and shared memory" to "keep shapes static and annotate sharding correctly."
4. **The interconnect is part of the chip.** TPU ICI torus, Trainium NeuronLink, AMD Infinity Fabric, Gaudi RoCE-over-Ethernet — these determine pod-scale efficiency as much as the per-chip FLOPS (see [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)).

---

## Reading a Spec Sheet to Pick Hardware

Now the practical skill: given a workload, choose a chip by reading three numbers — **HBM capacity (GB)**, **HBM bandwidth (TB/s)**, and **low-precision matmul throughput (FP8/FP4 TFLOP/s)** — and one fourth, **interconnect bandwidth (GB/s per link)**, for multi-chip jobs. Which number dominates depends entirely on whether you are training or serving, and within serving, whether you are *prefill*-bound or *decode*-bound (see [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html)).

### Step 1: Does the model even fit? (capacity)

Weights memory is straightforward:

$$
M_{\text{weights}} = N_{\text{params}} \times b_{\text{bytes/param}}
$$

For serving you must add the **KV cache**, whose per-token size for a model with $L$ layers, $H_{kv}$ key/value heads, head dimension $d_h$, in $b$ bytes per element (2 for the K *and* V tensors), is:

$$
M_{\text{KV/token}} = 2 \times L \times H_{kv} \times d_h \times b
$$

Capacity is where the MI300X's 192 GB and the TPU/Gaudi HBM slabs earn their keep: fit the model on fewer devices and you cut the tensor-parallel all-reduce traffic that otherwise caps throughput.

### Step 2: Is it compute-bound or bandwidth-bound? (roofline)

Use the ridge point $I^{*}$ from above. **Training** and **prefill** are dense GEMMs with high arithmetic intensity → **compute-bound** → you care about FP8/bf16 TFLOPS. Autoregressive **decode** streams the entire weight matrix from HBM to generate each token with tiny per-token compute → **bandwidth-bound** → you care about HBM TB/s, almost regardless of peak FLOPS. This is the single most important distinction when picking a serving chip.

{{fig:accel-roofline-decode-vs-prefill}}

### Step 3: A back-of-envelope decision function

```python
# A toy hardware picker. Numbers are ILLUSTRATIVE order-of-magnitude specs
# meant to teach the REASONING; always re-check the current datasheet.
from dataclasses import dataclass

@dataclass
class Chip:
    name: str
    hbm_gb: float          # capacity
    hbm_tbs: float         # bandwidth, TB/s
    fp8_tflops: float      # peak FP8 matmul throughput
    link_gbs: float        # per-chip interconnect bandwidth, GB/s

# Illustrative figures -- teach the method, not the exact values.
CHIPS = [
    Chip("H100-SXM",   80,  3.35,  1979,  450),   # NVLink
    Chip("MI300X",    192,  5.3,   2615,  448),   # Infinity Fabric
    Chip("TPU v5p",    95,  2.76,   918,  600),   # ICI (bf16-class peak)
    Chip("Gaudi3",    128,  3.7,   1835,  300),   # RoCE/Ethernet (aggregate)
]

def kv_bytes_per_token(L, H_kv, d_h, bytes_per_elt=2):
    return 2 * L * H_kv * d_h * bytes_per_elt          # K and V

def pick(workload, params_b, dtype_bytes, L, H_kv, d_h,
         ctx_len, batch, target="decode"):
    need_weights = params_b * 1e9 * dtype_bytes
    kv_per_tok   = kv_bytes_per_token(L, H_kv, d_h)
    need_kv      = kv_per_tok * ctx_len * batch
    need_total   = (need_weights + need_kv) / 1e9       # GB
    print(f"[{workload}] need ~{need_total:.0f} GB "
          f"(weights {need_weights/1e9:.0f} + KV {need_kv/1e9:.0f})")
    for c in CHIPS:
        n_dev = -(-need_total // c.hbm_gb)              # ceil-divide to fit
        # decode is bandwidth-bound -> rank by aggregate HBM TB/s;
        # prefill/training is compute-bound -> rank by aggregate FP8 TFLOPS.
        score = (c.hbm_tbs if target == "decode" else c.fp8_tflops) * n_dev
        print(f"  {c.name:10s}: fits on {int(n_dev)} dev, "
              f"score={score:.0f} ({'TB/s' if target=='decode' else 'TFLOPS'} agg)")

# Llama-70B-ish: 70B params, fp8 weights, 80 layers, 8 KV heads (GQA),
# head_dim 128, 8k context, batch 32, decode-bound serving.
pick("serve-70B-decode", 70, 1, 80, 8, 128, ctx_len=8192, batch=32)
```

The function encodes the doctrine: for **decode** serving, rank by *aggregate HBM bandwidth across the fewest devices that fit*; for **training/prefill**, rank by *aggregate low-precision TFLOPS*; and always gate on capacity first because a chip that cannot hold the model is disqualified before any throughput comparison.

!!! example "Worked example: serving a 70B model — capacity changes everything"
    Take a 70B-parameter model served in **FP8** (1 byte/param), so weights are $70 \times 10^{9} \times 1 = 70\ \text{GB}$. It uses GQA with $L=80$ layers, $H_{kv}=8$ KV heads, $d_h=128$, and we serve it in **bf16 KV** (2 bytes). Per-token KV cache:

    $$
    M_{\text{KV/token}} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\ \text{bytes} \approx 0.33\ \text{MB}
    $$

    At 8,192-token context and batch 32 that is $0.33\ \text{MB} \times 8192 \times 32 \approx 86\ \text{GB}$ of KV cache. Total live footprint $\approx 70 + 86 = 156\ \text{GB}$.

    - On **80 GB H100s** this needs $\lceil 156/80 \rceil = 2$ GPUs minimum just to fit — and in practice more, since you want headroom and the tensor-parallel split fragments memory. Decode now pays an NVLink all-reduce per layer.
    - On a **192 GB MI300X**, the whole 156 GB fits on **one** device. No tensor-parallel collective in the decode path at all. Because decode is **bandwidth-bound**, that single MI300X streaming weights at $\sim 5.3\ \text{TB/s}$ from one HBM stack — with zero inter-GPU communication — is a genuinely strong serving configuration despite the MI300X's raw FLOPS being a serving afterthought.

    The lesson is the chapter's thesis in miniature: **for decode serving you often pick the chip by HBM capacity and bandwidth, not by peak TFLOPS.** Flip the workload to *training* the same model and the calculus inverts — now it is dense, compute-bound GEMMs over a huge cluster, you care about aggregate FP8/bf16 TFLOPS and the all-reduce efficiency of the interconnect, and the TPU pod's torus + XLA collectives or a large NVLink/Infinity-Fabric domain become the deciding factors.

{{fig:accel-capacity-decision-70b}}

### A comparison table to anchor the families

| Vendor / chip | Core architecture | Lanes/group | Matmul unit | SDK / language | Compile model | Interconnect | Standout trait |
|---|---|---|---|---|---|---|---|
| NVIDIA H100/B200 | SIMT GPU | warp = 32 | Tensor Core | CUDA / Triton | eager + JIT | NVLink/NVSwitch | ecosystem, kernels first |
| AMD MI300X/MI350 | SIMT GPU (CDNA) | wavefront = 64 | Matrix Core (MFMA) | ROCm / HIP | eager + JIT | Infinity Fabric | huge HBM (192 GB) |
| Google TPU v5/v6/v7 | systolic | n/a (compiler) | MXU (128×128) | JAX / XLA, Pallas | ahead-of-time | ICI torus | pod scale, mature |
| AWS Trainium2/3 | systolic | n/a (compiler) | TensorEngine | Neuron / PyTorch-XLA, NKI | ahead-of-time | NeuronLink | AWS price/scale |
| Intel Gaudi3 | MME + TPC | TPC SIMD | MME | SynapseAI / PyTorch | graph compile | RoCE/Ethernet | Ethernet scale-out |

*(Figures and feature sets evolve every product cycle; treat this as the shape of the landscape, not a current datasheet.)*

---

## Porting in Practice: A Realistic Checklist

When someone hands you a non-NVIDIA fleet, the work is predictable. The order below is roughly highest-leverage first.

1. **Get the high-level model running first.** Install the vendor's PyTorch/JAX build (ROCm PyTorch, `torch_xla`+Neuron, `habana_frameworks`, or JAX-on-TPU). Most module code runs with only a device-string change (`cuda`→`hpu`, `cuda`→`xla`, or unchanged on ROCm). Confirm a forward/backward pass matches the NVIDIA reference loss within numerical tolerance.
2. **Freeze your shapes.** On any compiler-first backend (TPU/Trainium/Gaudi) bucket sequence lengths, fix batch size, pad vocab, and warm the compile cache. This step alone often turns a "10× slower" port into a competitive one. (Irrelevant on AMD, which is eager like NVIDIA.)
3. **Swap the kernel libraries.** Replace NCCL→RCCL (AMD) or use the vendor collective; ensure FlashAttention has a backend on your target (AMD CK / Triton-AMD; Pallas/NKI flash kernels on systolic chips). For serving, check that **vLLM/SGLang** have an upstream backend for your chip — they increasingly do — rather than porting kernels yourself.
4. **Re-tune, do not re-translate, the hot kernels.** If you own Triton kernels, recompile for the target and re-sweep block sizes / `num_warps` for the 64-wide wavefront (AMD) or the MXU tile (Pallas/NKI). Do not assume NVIDIA-tuned constants transfer.
5. **Audit for NVIDIA-only assumptions.** Grep for hard-coded `32` (warp size), inline PTX, Hopper-specific intrinsics, and `cuda`-only library calls with no portable analog. These are the items that genuinely do not port and must be rewritten.

!!! interview "Interview Corner"
    **Q:** A team wants to serve a 70B model and is choosing between two H100s (80 GB each) and one AMD MI300X (192 GB). The MI300X has *lower* peak FP8 TFLOPS than two H100s combined. Why might the single MI300X still be the better serving choice, and when would you reverse the decision?

    **A:** Because autoregressive **decode is memory-bandwidth-bound, not compute-bound.** Each generated token requires streaming the full set of weights (and the KV cache) from HBM while doing only a tiny amount of arithmetic, so the binding constraint is HBM bandwidth and *capacity*, not peak matmul FLOPS. Fitting the entire model plus KV cache in the MI300X's 192 GB on a **single device** eliminates the per-layer tensor-parallel all-reduce that two H100s would pay over NVLink, and it streams weights from one HBM stack at very high bandwidth. So for batched decode-dominated serving, the MI300X's surplus capacity and bandwidth can beat the two-H100 setup despite lower aggregate FLOPS. I would **reverse** the decision when the workload becomes **compute-bound**: long-context prefill on large batches, or training/fine-tuning, are dense high-arithmetic-intensity GEMMs where aggregate FP8/bf16 TFLOPS and interconnect all-reduce efficiency dominate — there the two-H100 (or a larger NVLink/TPU-pod) configuration with more compute and a mature kernel stack typically wins.

---

!!! sota "State of the Art & Resources (2026)"
    The non-NVIDIA accelerator landscape has matured rapidly: Google's TPU v7 Ironwood (2025) targets large-scale inference with 192 GB HBM per chip and 7.37 TB/s bandwidth; AMD's MI350 now ships with 288 GB HBM3e and FP4 support; and AWS Trainium3 anchors large training clusters — making fluency with at least one non-NVIDIA stack a practical job requirement.

    **Foundational work**

    - [Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit* (2017)](https://arxiv.org/abs/1704.04760) — the paper that introduced systolic-array TPUs and benchmarked them against contemporary CPUs/GPUs; still the canonical reference for the MXU design.

    **Recent advances (2023–2026)**

    - [Ambati & Diep, *AMD MI300X GPU Performance Analysis* (2024)](https://arxiv.org/abs/2510.27583) — systematic microbenchmark study of MI300X compute, memory bandwidth (~81% of 5.3 TB/s peak), and Infinity Fabric interconnect, directly informing serving decisions.
    - [Google Cloud, *Introducing Trillium, sixth-generation TPUs* (2024)](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus) — 4.7× compute-per-chip improvement over v5e, doubled HBM and ICI bandwidth, 67% better energy efficiency.
    - [Google, *Ironwood: The first Google TPU for the age of inference* (2025)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) — TPU v7 announcement; 192 GB HBM and 7.37 TB/s per chip, 4× throughput gain over Trillium, available in 9,216-chip pods.

    **Open-source & tools**

    - [ROCm/ROCm](https://github.com/ROCm/ROCm) — AMD's open-source GPU compute stack; the entry point for HIP, rocBLAS, RCCL, and the full AMD software ecosystem.
    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the de facto multi-backend LLM serving engine with upstream support for NVIDIA, AMD ROCm, Google TPU, and Intel Gaudi, removing the need to port kernels by hand.

    **Go deeper**

    - [Google Cloud TPU v6e Architecture Docs](https://docs.cloud.google.com/tpu/docs/v6e) — official spec and topology reference for Trillium (v6e); good companion when reading the roofline numbers.
    - [AWS Trainium product page](https://aws.amazon.com/ai/machine-learning/trainium/) — covers Trainium1/2/3 generations, UltraServer configurations, and NeuronLink interconnect specs.
    - [About Neuron Kernel Interface (NKI) — AWS Neuron Docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/index.html) — the Triton/Pallas analog for Trainium; explains the TensorEngine/VectorEngine programming model.
    - [AMD ROCm MI300 Series Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) — official AMD guide covering vLLM tuning, attention backends, and quantization on MI300X/MI350 for LLM inference.
    - [JAX Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html) — the TPU/GPU custom-kernel layer above XLA; the right tool when you need a fused FlashAttention or MoE gate on TPU.
    - [vLLM Blog: Serving LLMs on AMD MI300X — Best Practices (2024)](https://vllm.ai/blog/2024-10-23-vllm-serving-amd) — concrete benchmark showing 1.5× higher throughput than TGI on Llama 3.1 405B; illustrates the real-world serving advantage of the MI300X's 192 GB HBM.

## Key Takeaways & Further Reading

!!! key "Key Takeaways"
    - There are two silicon philosophies. **SIMT GPUs** (NVIDIA, AMD) are programmable seas of cores where *you* manage the memory hierarchy; **systolic accelerators** (TPU, Trainium, Gaudi's MME) pump data through fixed MAC arrays and make *a compiler* own the kernels and the sharding.
    - **Portability tracks philosophy.** CUDA→AMD is a translation problem (HIP ≈ CUDA, `hipify`, ROCm PyTorch) needing re-tuning for 64-wide wavefronts; CUDA→TPU/Trainium/Gaudi is a paradigm shift to graph-level array code under XLA/Neuron/SynapseAI.
    - On **compiler-first chips**, your performance levers are **static shapes** and **sharding annotations**, not occupancy and shared memory. The #1 mystery slowdown is silent recompilation from wobbling shapes — bucket and pad everything.
    - **The higher-level your code, the more portable it is.** Pure PyTorch/JAX runs on all four families; Triton recompiles but must be re-tuned; hand-written PTX/SASS and Hopper-only intrinsics do not port.
    - **Read three numbers off the spec sheet:** HBM capacity (does it fit?), HBM bandwidth (decode throughput), and FP8/FP4 TFLOPS (training/prefill throughput) — plus interconnect bandwidth for pods.
    - **Decode serving is bandwidth-bound; training/prefill is compute-bound.** This single distinction flips which chip wins — which is why a 192 GB MI300X can out-serve two H100s on decode while losing on training.
    - **The interconnect is part of the chip.** TPU's ICI torus, Trainium's NeuronLink, AMD's Infinity Fabric, and Gaudi's RoCE-over-Ethernet decide pod-scale efficiency as much as per-chip FLOPS.
    - vLLM/SGLang/PyTorch/JAX increasingly ship upstream backends for all four families — the portable path is to ride those, not to port kernels by hand.

**Further reading.**

- Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit* (ISCA 2017) — the foundational TPU systolic-array paper.
- Norrie et al., *The Design Process for Google's Training Chips: TPUv2 and TPUv3*, and Google's subsequent TPU pod / optical-circuit-switch papers — how training-scale pods are built.
- Bradbury et al., *JAX: composable transformations of Python+NumPy programs*, and the **XLA** / **GSPMD** documentation and papers — the compiler-first programming model.
- AMD, *ROCm* and *HIP Programming Guide*, and the AMD *Composable Kernel (CK)* and *hipBLASLt* projects — the CUDA-portability stack.
- AWS, *AWS Neuron SDK* documentation and the *Neuron Kernel Interface (NKI)* guide — Trainium/Inferentia programming.
- Intel/Habana, *Gaudi / SynapseAI* documentation — the MME + TPC + RoCE design.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM) — the serving engine whose multi-backend support makes cross-vendor inference practical.

## Exercises

**1.** (Conceptual) The chapter says a systolic array is "spectacularly efficient *for dense matrix multiply*" yet "everything else ... must be handled by a smaller companion vector unit." Using the description of the systolic array's dataflow, explain *why* the array is so efficient for a large dense matmul specifically — name the two kinds of on-chip traffic it avoids — and then explain why a workload of many tiny, dynamically-shaped, branchy matmuls fails to realize that efficiency.

??? note "Solution"
    The efficiency comes from how the array handles the *partial sums* and the *control*. In a $128 \times 128$ MAC grid, weights are pre-loaded into the cells; activations flow in from the left one column per clock and partial sums flow down cell-to-cell. Once the pipeline fills, the array retires a full $128 \times 128$ tile of MACs *every clock cycle*. Two kinds of traffic that a general SIMT core pays are avoided:

    - **Register-file / SRAM traffic for the intermediate partial sums.** The partials never leave the array — they march from one cell to its neighbor — so they are never written back to a register file or shared-memory scratchpad.
    - **Instruction-fetch / control overhead.** Each cell does the same fused multiply-add every cycle with no per-operation instruction fetch, so almost all the silicon is MAC units and almost none is control logic or caches. That is why FLOPs per watt and per transistor beat a general SIMT core.

    Both savings are *amortized over the fill/drain of a large tile*. For a big dense matmul the array stays full for many cycles, so the one-time cost of filling the pipeline is negligible against thousands of full-throughput cycles. For **many tiny, dynamically-shaped, branchy** matmuls the array **drains and refills** constantly: each tiny op barely fills the pipeline before it ends, so you pay fill/drain overhead repeatedly and rarely reach the steady-state one-tile-per-cycle regime. Branching and dynamic shapes also defeat the ahead-of-time compiler that must know static shapes to schedule the dataflow. This is exactly the case the chapter's aside flags: "Tiny, dynamically-shaped, branchy workloads ... are where the systolic model struggles."

**2.** (Conceptual) You `hipify` a CUDA warp-reduction kernel that ends with the loop `for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(mask, v, offset);` and run it on an MI300X. The build succeeds and the kernel runs, but the reduction result is wrong. Explain the bug, why `hipify` did not catch it, and the correct fix.

??? note "Solution"
    The loop assumes a warp of **32 lanes**: starting `offset` at 16 (half of 32) and halving down to 1 reduces exactly 32 lanes. On a CDNA GPU (MI300X) the lockstep group is a **wavefront of 64 lanes**, not 32. Starting at `offset = 16` only combines lanes within each 32-lane half of the 64-wide wavefront; the upper and lower halves are never summed together, so the reduction **silently drops half the lanes** and produces a wrong (roughly half) result.

    `hipify` did not catch it because `hipify` only **renames API calls** (`cuda*` -> `hip*`); it does not rewrite kernel *logic* or the hard-coded constant `32`/`16`. As the chapter's warning states, "`hipify` will not fix the *logic*; it only renames API calls."

    The fix is to derive the starting offset from the actual wavefront width instead of hard-coding it. On AMD `warpSize` is a runtime value, so start the reduction at `warpSize/2`:

    ```cpp
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset);   // 32 on AMD -> reduces all 64 lanes
    ```

    More generally, re-derive any shared-memory tile sizes and lane masks from `warpSize` rather than baking in 32.

**3.** (Quantitative) A model has $L = 96$ layers, uses multi-head attention (MHA, so $H_{kv}$ = number of query heads) with 96 heads of head dimension $d_h = 128$, and you serve the KV cache in bf16 (2 bytes). (a) Compute the per-token KV-cache size in bytes. (b) At a 32,768-token context and batch 16, how many GB of KV cache is that? (c) The same model architecture is then redesigned to use GQA with only $H_{kv} = 8$ KV heads (everything else unchanged). Recompute (a) and (b), and state the reduction factor. Use $1\ \text{GB} = 10^9$ bytes, consistent with the chapter's code.

??? note "Solution"
    Use the chapter's formula $M_{\text{KV/token}} = 2 \times L \times H_{kv} \times d_h \times b$ (the leading 2 is for the K *and* V tensors; $b = 2$ bytes for bf16).

    (a) MHA, $H_{kv} = 96$:

    $$
    M_{\text{KV/token}} = 2 \times 96 \times 96 \times 128 \times 2 = 4{,}718{,}592\ \text{bytes} \approx 4.72\ \text{MB/token}.
    $$

    (b) Context 32,768, batch 16:

    $$
    4{,}718{,}592 \times 32768 \times 16 = 2.474 \times 10^{12}\ \text{bytes} \approx 2474\ \text{GB} \approx 2.47\ \text{TB}.
    $$

    (This is enormous — no single accelerator in the chapter's table holds it; it forces many devices or a shorter context/batch.)

    (c) GQA, $H_{kv} = 8$:

    $$
    M_{\text{KV/token}} = 2 \times 96 \times 8 \times 128 \times 2 = 393{,}216\ \text{bytes} \approx 0.39\ \text{MB/token}.
    $$

    $$
    393{,}216 \times 32768 \times 16 = 2.061 \times 10^{11}\ \text{bytes} \approx 206\ \text{GB}.
    $$

    Reduction factor $= 96 / 8 = 12\times$ (KV cache scales linearly in $H_{kv}$). GQA cuts the per-token KV from 4.72 MB to 0.39 MB and the total from ~2474 GB to ~206 GB — the same 12x — which is exactly why frontier serving models use GQA: it is what makes the KV cache fit in HBM.

**4.** (Quantitative) Using the chapter's illustrative specs — H100 at 1979 FP8 TFLOP/s over 3.35 TB/s HBM, and MI300X at 2615 FP8 TFLOP/s over 5.3 TB/s HBM — compute each chip's roofline ridge point $I^{*}$ in FLOP/byte (treat the FP8 TFLOPS as the peak compute number). A decode step for one token in an FP8-weight linear layer of a 70B model does roughly $2 \times 70\times10^{9} = 1.4\times10^{11}$ FLOPs while streaming $70\times10^{9}$ bytes of weights, giving an arithmetic intensity of $\approx 2$ FLOP/byte. Confirm decode is memory-bound on both chips, and explain in one sentence why the MI300X's *higher* ridge point does not make it worse for decode.

??? note "Solution"
    Ridge point $I^{*} = \dfrac{\text{peak compute (FLOP/s)}}{\text{peak bandwidth (byte/s)}}$. Convert TFLOP/s to FLOP/s ($\times 10^{12}$) and TB/s to byte/s ($\times 10^{12}$), so the $10^{12}$ factors cancel and $I^{*} = \text{TFLOPS} / \text{TB/s}$:

    - **H100:** $I^{*} = 1979 / 3.35 \approx 591$ FLOP/byte.
    - **MI300X:** $I^{*} = 2615 / 5.3 \approx 493$ FLOP/byte.

    (Sanity check: the chapter's bf16 H100 ridge of ~295 uses ~990 bf16 TFLOP/s; FP8 roughly doubles the compute number, so ~591 for FP8 is consistent.)

    Decode arithmetic intensity $\approx 2$ FLOP/byte is *vastly* below both ridge points ($2 \ll 493 < 591$), so decode is firmly **memory-bound on both chips** — it cannot come anywhere near peak FLOPS and its speed is set by HBM bandwidth.

    The MI300X's higher ridge point does not hurt decode because a higher $I^{*}$ just means the chip needs more arithmetic intensity to *become* compute-bound; for a bandwidth-bound op the only thing that matters is the denominator — HBM bandwidth — and the MI300X's 5.3 TB/s vs the H100's 3.35 TB/s means it streams weights (and thus generates tokens) faster.

**5.** (Implementation) The chapter's `pick()` function only *prints* a score and never returns a winner, and its `score = per_device_figure * n_dev` rewards using *more* devices — which is backwards for **decode**, where splitting a model across devices adds the per-layer all-reduce the chapter tells you to avoid. Rewrite the picker so it *returns* the winning `Chip` and its device count, using a **target-aware** rule: for **decode**, pick the chip that fits on the **fewest devices** (minimizing cross-device collectives), breaking ties by higher aggregate HBM bandwidth; for **prefill/training**, pick the chip with the highest **aggregate FP8 TFLOPS** across the minimum devices that fit (more devices are welcome when you are compute-bound). Raise if `need_total` is non-positive. Keep the `Chip` dataclass and style, and confirm that decode returns the MI300X and prefill returns the H100 for the chapter's 70B footprint.

??? note "Solution"
    The subtlety is that `score = per_device * n_dev` is **aggregate** bandwidth, and aggregate bandwidth *grows with device count* — so for the 70B decode footprint (~156 GB) it would actually rank Gaudi3 (2 devices $\times$ 3.7 = **7.4** TB/s aggregate) *above* the single MI300X (1 device $\times$ 5.3 = **5.3** TB/s). But that number silently ignores the per-layer all-reduce every 2-device config pays over its interconnect — exactly the cost the chapter says to avoid. The chapter's decode doctrine is therefore not "maximize aggregate bandwidth" but "**fit on the fewest devices**, then look at bandwidth." So we sort by device count first for decode, and only fall back to aggregate bandwidth to break ties. (For **prefill/training** you *are* compute-bound and happy to scale out, so there the right key is genuinely aggregate FP8 TFLOPS.)

    ```python
    from dataclasses import dataclass
    from math import ceil

    @dataclass
    class Chip:
        name: str
        hbm_gb: float
        hbm_tbs: float
        fp8_tflops: float
        link_gbs: float

    CHIPS = [
        Chip("H100-SXM",   80,  3.35,  1979,  450),
        Chip("MI300X",    192,  5.3,   2615,  448),
        Chip("TPU v5p",    95,  2.76,   918,  600),
        Chip("Gaudi3",    128,  3.7,   1835,  300),
    ]

    def kv_bytes_per_token(L, H_kv, d_h, bytes_per_elt=2):
        return 2 * L * H_kv * d_h * bytes_per_elt

    def pick(workload, params_b, dtype_bytes, L, H_kv, d_h,
             ctx_len, batch, target="decode"):
        need_weights = params_b * 1e9 * dtype_bytes
        kv_per_tok   = kv_bytes_per_token(L, H_kv, d_h)
        need_kv      = kv_per_tok * ctx_len * batch
        need_total   = (need_weights + need_kv) / 1e9      # GB
        if need_total <= 0:
            raise ValueError("footprint must be positive")
        print(f"[{workload}] need ~{need_total:.0f} GB "
              f"(weights {need_weights/1e9:.0f} + KV {need_kv/1e9:.0f})")

        best, best_key, best_ndev = None, None, 0
        for c in CHIPS:
            n_dev   = ceil(need_total / c.hbm_gb)          # fewest devices that fit
            per_dev = c.hbm_tbs if target == "decode" else c.fp8_tflops
            agg     = per_dev * n_dev                       # aggregate over the fit
            unit    = "TB/s" if target == "decode" else "TFLOPS"
            print(f"  {c.name:10s}: fits on {n_dev} dev, agg={agg:.0f} {unit}")
            # Tuples compare left-to-right; SMALLER is better.
            # decode  -> (fewest devices, then most aggregate BW)
            # prefill -> (most aggregate TFLOPS); more devices welcome
            key = (n_dev, -agg) if target == "decode" else (-agg,)
            if best_key is None or key < best_key:
                best, best_key, best_ndev = c, key, n_dev

        print(f"  -> winner: {best.name} on {best_ndev} device(s)")
        return best, best_ndev

    # Llama-70B-ish decode, as in the chapter.
    winner, ndev = pick("serve-70B-decode", 70, 1, 80, 8, 128,
                        ctx_len=8192, batch=32)
    ```

    On the chapter's 70B footprint (~156 GB), the **MI300X is the only chip that fits on 1 device** (192 GB); every other chip needs 2. For `target="decode"` the fewest-devices key makes the single MI300X win outright — no per-layer collective — which is precisely the chapter's worked-example conclusion. Note the behavioral change from the original: the function now *returns a decision* (`best`, `best_ndev`) instead of only printing, so it can drive downstream provisioning. Flip to `target="prefill"` and the key switches to aggregate FP8 TFLOPS: now the **H100** wins (2 devices $\times$ 1979 = 3958 TFLOPS aggregate, beating the single MI300X's 2615), because prefill/training is compute-bound and rewards scaling out — exactly the compute-vs-bandwidth flip the chapter's doctrine predicts.
