# 4.1 The Roofline Model & Performance Engineering

Every performance optimization you will ever apply to a large language model — FlashAttention, quantization, kernel fusion, continuous batching, paged KV caches — is a move in one of two games: *do less work*, or *move less data*. The roofline model is the single mental picture that tells you, for any given kernel on any given GPU, which of those two games you are actually playing. Get this right and the rest of Part IV becomes a sequence of "obvious in hindsight" optimizations. Get it wrong and you will spend a week shaving FLOPs off a kernel whose runtime is dominated entirely by memory traffic, and watch your "improvement" change nothing.

This chapter builds that mental model from first principles. We define arithmetic intensity, draw the roofline, derive where attention and the feed-forward network (FFN) sit on it, estimate the FLOPs of a full transformer, compute Model FLOPs Utilization (MFU) for a real training run, and finally profile a kernel with both the PyTorch profiler and NVIDIA Nsight to confirm the theory against silicon. Everything here rests on the hardware described in [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html); if the terms *HBM*, *SM*, and *L2* are unfamiliar, read that first.

## Two Speed Limits: Compute and Bandwidth

A GPU has two fundamental resources that bound how fast a computation can run. The first is **arithmetic throughput**: how many floating-point operations per second (FLOP/s) the arithmetic units can retire. The second is **memory bandwidth**: how many bytes per second can be moved between high-bandwidth memory (HBM) — the multi-gigabyte DRAM stacked next to the die — and the on-chip registers and SRAM where arithmetic actually happens.

These two numbers are wildly mismatched, and that mismatch *is* the entire story. Consider an NVIDIA A100 (80 GB, SXM). Its peak HBM bandwidth is on the order of 2.0 TB/s, and its peak bf16/fp16 tensor-core throughput is on the order of 312 TFLOP/s. Take the ratio:

$$
\frac{312 \times 10^{12}\ \text{FLOP/s}}{2.0 \times 10^{12}\ \text{B/s}} \approx 156\ \frac{\text{FLOP}}{\text{B}}.
$$

This number — call it the machine's **ridge point** $\pi/\beta$, where $\pi$ is peak FLOP/s and $\beta$ is peak bytes/s — says: *the hardware can perform about 156 floating-point operations in the time it takes to read one byte from HBM.* If your kernel does fewer than ~156 FLOP per byte it touches, it will finish its arithmetic and then sit idle waiting for the next bytes to arrive. It is **memory-bound**. If it does more, the memory system keeps up and the arithmetic units are the bottleneck; it is **compute-bound**.

On an H100 (SXM) the imbalance is even more extreme: roughly 3.35 TB/s of HBM3 bandwidth against close to 990 TFLOP/s of dense bf16 tensor-core throughput, for a ridge of roughly 295 FLOP/B. Each GPU generation has pushed FLOP/s up faster than bandwidth, so the ridge point keeps climbing and **more and more kernels fall into the memory-bound regime**. This is why so much of modern LLM systems work is about data movement, not arithmetic.

!!! note "Aside: why the asymmetry exists"

    Adding more multiply-accumulate units to a die is comparatively cheap — they are small and regular. Increasing DRAM bandwidth means more pins, more power for the I/O, exotic packaging (HBM is stacked DRAM bonded with through-silicon vias), and it bumps against fundamental physical limits. So vendors add FLOPs faster than bytes/s, generation over generation. The roofline's ridge point drifts right, and the engineer's job drifts toward minimizing bytes moved.

### Arithmetic intensity

The property of *your kernel* that we compare against the machine ridge is its **arithmetic intensity** (AI), also called *operational intensity*:

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved to/from HBM}}\quad \left[\frac{\text{FLOP}}{\text{B}}\right].
$$

Crucially, "bytes moved" means traffic to the slow, large memory (HBM / DRAM), *not* traffic within registers or shared memory. Data you keep resident on-chip and reuse is free in this accounting — which is precisely the lever FlashAttention pulls (see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)).

A worked feel for the magnitudes:

- **Element-wise add** ($z = x + y$ over $N$ fp16 elements): $N$ FLOPs, but $3N \times 2$ bytes moved (read $x$, read $y$, write $z$). $I = N / 6N = 0.17$ FLOP/B. Hopelessly memory-bound.
- **A GEMM** (general matrix-matrix multiply) $C_{m\times n} = A_{m\times k} B_{k\times n}$: about $2mnk$ FLOPs, and (read $A$, read $B$, write $C$) $\approx 2(mk + kn + mn)$ bytes. For large square matrices $I \approx 2m^3 / (6m^2) = m/3$, which grows with size — big GEMMs are compute-bound.

The contrast is the whole point: matrix multiplication has *reuse* (each loaded element participates in many FLOPs), while element-wise ops have none. The transformer is a tug-of-war between big reuse-heavy GEMMs and a swarm of tiny reuse-free element-wise and normalization ops.

{{fig:intensity-reuse-ladder}}

## The Roofline Model

Now we combine the two machine limits and the kernel's intensity into one diagram. Plot achievable performance $P$ (FLOP/s, $y$-axis, log scale) against arithmetic intensity $I$ (FLOP/B, $x$-axis, log scale). The peak attainable performance is:

$$
P(I) = \min\bigl(\pi,\; \beta \cdot I\bigr),
$$

where $\pi$ is peak compute (FLOP/s) and $\beta$ is peak bandwidth (B/s). The $\beta \cdot I$ term is the **memory roof** — a diagonal line of slope $\beta$ in log-log space (bandwidth times intensity = achievable FLOP/s). The flat $\pi$ term is the **compute roof**. They meet at the ridge point $I^\* = \pi/\beta$.

{{fig:roofline}}

```text
  Performance (FLOP/s, log)
   ^
 π |..................______________________  compute roof (π)
   |                 /
   |                /  <- you want kernels here (compute-bound)
   |  memory roof  /
   |   (slope β)  /
   |            ./  <- ridge point  I* = π/β
   |          ./
   |        ./   kernels here are memory-bound:
   |      ./     P = β·I  (raise I to go faster)
   |    ./
   +--/-------------------------------------> Arithmetic intensity (FLOP/B, log)
        I*
```

Reading the diagram is a three-step ritual you will repeat constantly:

1. Compute the kernel's intensity $I$.
2. Compare to the machine ridge $I^\* = \pi/\beta$. If $I < I^\*$, you are **memory-bound** and your performance ceiling is $\beta I$ — far below peak FLOP/s. If $I > I^\*$, you are **compute-bound**, ceiling $\pi$.
3. Choose the optimization that *moves you up the roof*. For memory-bound kernels: raise $I$ (fuse ops, keep data on-chip, recompute instead of reload, quantize to move fewer bytes). For compute-bound kernels: use faster math (tensor cores, lower precision) or simply do fewer FLOPs.

!!! warning "Common pitfall: optimizing the wrong axis"

    The single most common performance mistake is reducing FLOPs on a memory-bound kernel and expecting a speedup. If a LayerNorm is bandwidth-limited (it is — intensity well under 1 FLOP/B), then rewriting it to do 20% fewer arithmetic ops changes its runtime by ~0%, because it was never waiting on arithmetic. The roofline tells you *which* resource is saturated; optimize that one. Conversely, fusing a memory-bound op into an adjacent kernel (so its inputs are already in registers) can give a large speedup with zero FLOP reduction.

### A naive vs. real roofline

The simple roofline above assumes a kernel can hit either roof. Real kernels rarely touch peak because of additional ceilings: insufficient occupancy to hide latency, no use of tensor cores (relegating you to the much lower CUDA-core FP32 roof), bank conflicts in shared memory, or imperfect overlap of load and compute. A *hierarchical roofline* draws several compute roofs (tensor-core peak, FMA peak, no-vectorization peak) and several memory roofs (HBM, L2, L1/shared). Your kernel's measured point sitting under the HBM roof but far below the tensor-core ceiling is a diagnosis: "I am compute-bound but not using tensor cores," or "I am bandwidth-bound on L2, not HBM." Nsight Compute can plot exactly this, which we use in the profiling section.

## Counting FLOPs: The Transformer From the Top

To place transformer kernels on the roofline we need their FLOP and byte costs. We adopt the standard convention that one multiply-accumulate (MAC) is **2 FLOPs** (a multiply and an add). A matrix multiply $A_{m\times k}B_{k\times n}$ therefore costs $2mnk$ FLOPs.

Consider a decoder-only transformer with hidden size $d$ (a.k.a. $d_{\text{model}}$), FFN inner size $d_{\text{ff}}$ (commonly $4d$), $L$ layers, vocabulary $V$, sequence length $s$, and batch $B$. We count the forward pass for one token position and then scale.

### The FFN (the FLOP hog)

Each transformer block has an FFN: an up-projection $d \to d_{\text{ff}}$, a nonlinearity, and a down-projection $d_{\text{ff}} \to d$. Per token, the two GEMMs cost:

$$
\text{FLOPs}_{\text{FFN}} = 2 \cdot (d \cdot d_{\text{ff}}) + 2 \cdot (d_{\text{ff}} \cdot d) = 4\, d\, d_{\text{ff}} = 16\, d^2 \quad (\text{when } d_{\text{ff}}=4d).
$$

(A gated FFN like SwiGLU has three matrices — gate, up, down — so it is $6\,d\,d_{\text{ff}}$; see [The Transformer Block](../02-transformer/06-transformer-block.html).)

### Attention projections vs. the attention score

Attention has two FLOP populations that behave very differently on the roofline. First, the **projections**: computing $Q,K,V$ from the input ($3$ matmuls of $d\to d$) and the output projection ($d\to d$). Per token:

$$
\text{FLOPs}_{\text{QKVO}} = 4 \cdot 2 d^2 = 8\, d^2.
$$

Second, the **score-and-value** computation: $QK^\top$ and the $\text{softmax}\cdot V$. For a single query attending over $s$ keys with head dimension summing to $d$, this costs about $2 \cdot 2\, s\, d = 4\, s\, d$ FLOPs per query token. Summed over a sequence of length $s$ (each of $s$ queries attends to up to $s$ keys), the score computation for the whole sequence is $\propto s^2 d$ — the famous quadratic. Per *token*, it is $\approx 4 s d$.

Putting the per-token, per-layer forward cost together:

$$
\text{FLOPs/token/layer} \approx \underbrace{8 d^2}_{\text{QKVO}} + \underbrace{4 s d}_{\text{scores}} + \underbrace{16 d^2}_{\text{FFN}} = 24 d^2 + 4 s d.
$$

The headline observation: **for typical $d$ and moderate $s$, the $d^2$ terms dominate.** With $d = 4096$ and $s = 2048$, $24 d^2 \approx 4.0\times 10^8$ while $4 s d \approx 3.4\times 10^7$ — the dense projections and FFN are ~12× the attention-score FLOPs. The quadratic attention score only *dominates the FLOP budget* once $s$ becomes comparable to $6d$. But — and this is the subtlety that motivates FlashAttention — even when attention is a minority of FLOPs, it can dominate *runtime*, because its score matrix is enormous and memory-bound. FLOPs and time are not the same thing; that gap is exactly what the roofline exists to expose.

{{fig:transformer-flop-anatomy}}

### The famous $6N$ rule for training

{{tool:param-flop-counter}}

Summing layers and adding the embedding/unembedding, the forward pass of a model with $N$ non-embedding parameters costs about $2N$ FLOPs per token (every parameter participates in one MAC = 2 FLOPs). The backward pass costs about twice the forward — you compute gradients with respect to both inputs and weights — giving roughly $4N$. Hence the rule every LLM engineer memorizes:

$$
C_{\text{train}} \approx 6N \cdot D \quad\text{FLOPs},
$$

for a full training run over $D$ tokens with $N$ parameters. This is the same $6N$ that underlies Chinchilla-style scaling analysis in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html). For inference (forward only), it is $\approx 2N$ FLOPs per generated token. These two constants — $6$ for training, $2$ for inference — are the back-of-the-envelope tools you will reach for in any capacity-planning or interview setting.

```python
def transformer_flops_per_token(d_model, d_ff, n_layers, seq_len,
                                vocab, gated=False):
    """Approximate forward-pass FLOPs for ONE token, training-style accounting.
    Returns a dict so we can see where the FLOPs live. 1 MAC = 2 FLOPs."""
    # --- per layer ---
    qkvo = 4 * 2 * d_model**2                 # Q,K,V,O projections (4 GEMMs d->d)
    scores = 2 * (2 * seq_len * d_model)      # QK^T  and  softmax@V  ~ 4*s*d
    ffn_mul = 6 if gated else 4               # SwiGLU has 3 mats; vanilla has 2
    ffn = ffn_mul * d_model * d_ff            # up (+gate) and down projections
    per_layer = qkvo + scores + ffn

    # --- whole stack ---
    layers = n_layers * per_layer
    unembed = 2 * d_model * vocab             # final logits projection (per token)
    fwd = layers + unembed
    return {
        "fwd_per_token": fwd,
        "qkvo_total":  n_layers * qkvo,
        "scores_total": n_layers * scores,
        "ffn_total":   n_layers * ffn,
        "unembed":     unembed,
    }

# Llama-2-7B-ish config
cfg = dict(d_model=4096, d_ff=11008, n_layers=32, seq_len=4096,
           vocab=32000, gated=True)
f = transformer_flops_per_token(**cfg)
fwd = f["fwd_per_token"]
print(f"forward FLOPs / token : {fwd/1e9:8.2f} GFLOP")
print(f"  in FFN              : {f['ffn_total']/1e9:8.2f} GFLOP "
      f"({100*f['ffn_total']/fwd:.0f}%)")
print(f"  in QKVO proj        : {f['qkvo_total']/1e9:8.2f} GFLOP "
      f"({100*f['qkvo_total']/fwd:.0f}%)")
print(f"  in attn scores      : {f['scores_total']/1e9:8.2f} GFLOP "
      f"({100*f['scores_total']/fwd:.0f}%)")
print(f"training FLOPs / token (~3x fwd): {3*fwd/1e9:8.2f} GFLOP")

# Cross-check against the 6N rule using true param count (~6.7B):
N = 6.7e9
print(f"6N per token (N=6.7B): {6*N/1e9:8.2f} GFLOP")
```

Running this prints a forward cost on the order of a few GFLOP/token, with the FFN taking the largest slice, the projections second, and attention scores a small minority at this sequence length — and the $\approx 3\times$ forward estimate landing close to the $6N$ figure. Two independent methods agreeing within ~10–20% is exactly the confidence you want from a FLOP estimate.

!!! note "Aside: why $3\times$ forward, not $3\times$ everything"

    The $6N$ rule bundles forward ($2N$) and backward ($4N$). Our per-token function counts forward only, so we multiply by $3$ to approximate forward+backward. The attention-score term technically scales differently in the backward pass (and recomputation in activation checkpointing adds an extra forward), so treat $3\times$ as a planning estimate, not a guarantee. For exact accounting, libraries like `torch.utils.flop_counter` or DeepSpeed's FLOPs profiler instrument the real graph.

## Bytes, Intensity, and the Two Regimes of Inference

FLOPs alone don't tell you where a kernel lands on the roofline — you need bytes. The cleanest place to see this is LLM inference, which splits into two phases with opposite roofline character (covered operationally in [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html)).

**Prefill** processes the whole prompt at once. With a prompt of $s$ tokens, every weight matrix is reused across all $s$ tokens — a GEMM with a big $s$ dimension. High reuse means high arithmetic intensity, so prefill is **compute-bound**. This is where the GPU's FLOP/s actually matter.

**Decode** generates one token at a time. Each step multiplies a single activation vector ($1\times d$) against every weight matrix. That is a matrix-*vector* product (GEMV): you stream the entire weight matrix from HBM to produce one output vector, with essentially no reuse. For a weight matrix of $d\times k$ in bf16, you do $2dk$ FLOPs but move $2dk$ bytes — intensity $\approx 1$ FLOP/B. Far below any modern ridge point, so single-stream decode is brutally **memory-bound**.

This is why decode throughput is governed by *how fast you can read the weights*, and why the only way to make decode efficient is to **batch** many requests so the same loaded weights serve many tokens at once — raising the effective intensity. It is also why the KV cache, not the weights, often becomes the bandwidth bottleneck at long context; see [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

{{fig:decode-batching-amortization}}

!!! example "Worked example: how many tokens/sec can a single decode stream produce?"

    Take Llama-2-7B in bf16 on one A100-80GB. Decode reads (approximately) every weight once per generated token. Model weights $\approx 6.7\text{B} \times 2\,\text{bytes} = 13.4\,\text{GB}$.

    At peak HBM bandwidth $\beta = 2.0\,\text{TB/s}$, the time to stream the weights once is

    $$
    t_{\text{token}} \ge \frac{13.4\times 10^9\,\text{B}}{2.0\times 10^{12}\,\text{B/s}} \approx 6.7\,\text{ms}.
    $$

    So the *hardware ceiling* for single-stream decode is about $1/0.0067 \approx 150$ tokens/s, and real systems hit perhaps 60–70% of that. Notice what this says: the GPU's 312 TFLOP/s is almost irrelevant here. The decode FLOPs are $2N \approx 1.34\times 10^{10}$ per token; at 150 tok/s that's $\approx 2\,\text{TFLOP/s}$ — under **1%** of peak compute. The chip is starved on bandwidth.

    Now batch 64 requests. The weights are read once and reused across all 64 tokens, so per-token weight traffic drops ~64×, intensity rises above the ridge, and the workload becomes compute-bound — *until* the per-request KV cache traffic and capacity start to dominate. This single example is the entire economic argument for continuous batching ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)).

```python
def decode_roofline(num_params, bytes_per_param=2,
                    hbm_bw_TBs=2.0, peak_tflops=312):
    """Single-stream decode is GEMV-like: ~1 weight read per token.
    Returns the bandwidth-bound token rate and the implied compute use."""
    weight_bytes = num_params * bytes_per_param
    t_token = weight_bytes / (hbm_bw_TBs * 1e12)      # seconds, bw-bound
    tok_per_s = 1.0 / t_token
    flops_per_token = 2 * num_params                   # 2N inference rule
    achieved_tflops = flops_per_token * tok_per_s / 1e12
    return {
        "weight_GB": weight_bytes / 1e9,
        "ms_per_token": t_token * 1e3,
        "tok_per_s_ceiling": tok_per_s,
        "achieved_TFLOPs": achieved_tflops,
        "pct_of_peak_compute": 100 * achieved_tflops / peak_tflops,
    }

for n in (6.7e9, 13e9, 70e9):
    r = decode_roofline(n)
    print(f"{n/1e9:5.1f}B params: {r['tok_per_s_ceiling']:6.1f} tok/s ceiling, "
          f"using {r['pct_of_peak_compute']:.2f}% of peak FLOP/s "
          f"({r['weight_GB']:.1f} GB weights)")
```

## Model FLOPs Utilization (MFU)

{{tool:train-compute-estimator}}

For *training* we want a single dimensionless number that says how well a run uses the hardware. **Model FLOPs Utilization** (MFU), introduced in the PaLM paper, is:

$$
\text{MFU} = \frac{\text{useful model FLOP/s achieved}}{\text{peak FLOP/s of the hardware}} = \frac{6 N \cdot (\text{tokens/s})}{\text{GPUs} \times \pi}.
$$

The numerator uses the $6N$ rule on the *model's* FLOPs (the math the model defines), deliberately *excluding* any extra work like activation-recomputation. A related metric, **HFU** (Hardware FLOPs Utilization), *includes* recomputation FLOPs and so is always $\ge$ MFU; the gap tells you how much you're paying for activation checkpointing ([Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html)). MFU is the honest "how close to the speed of light are we" number; large, well-tuned LLM training runs land in roughly the 40–55% MFU range, and anything below ~30% usually signals a fixable bottleneck (small batch, comms-bound, bad kernels, or memory-bound element-wise ops).

```python
def mfu(num_params, tokens_per_sec, num_gpus, peak_tflops_per_gpu):
    """Model FLOPs Utilization for a training run (6N convention)."""
    model_flops_per_s = 6 * num_params * tokens_per_sec
    peak = num_gpus * peak_tflops_per_gpu * 1e12
    return model_flops_per_s / peak

# Example: 7B model, 256 A100s (312 TFLOP/s bf16 peak each).
# Suppose we measure global throughput of 1.5M tokens/sec.
util = mfu(num_params=6.7e9, tokens_per_sec=1.5e6,
           num_gpus=256, peak_tflops_per_gpu=312)
print(f"MFU = {util*100:.1f}%")

# Inverting: at a target MFU, what throughput do we expect?
def expected_tokens_per_sec(num_params, num_gpus, peak_tflops, target_mfu):
    return target_mfu * num_gpus * peak_tflops * 1e12 / (6 * num_params)
print(f"At 50% MFU we'd expect "
      f"{expected_tokens_per_sec(6.7e9, 256, 312, 0.50)/1e6:.2f} M tok/s")
```

!!! tip "Practitioner tip: MFU is your north-star regression test"

    Track MFU on every training run, not just loss. A code change that drops MFU from 48% to 31% with no architectural reason is a performance regression — maybe a kernel stopped fusing, a collective started blocking, or a tensor fell off the tensor-core path (wrong dtype, a non-multiple-of-8 dimension). MFU catches these where wall-clock-per-step might be noisy. Compute it from first principles ($6N \cdot \text{tok/s}$) so it's independent of any framework's possibly-wrong internal counter.

### Why low precision raises the roof

Tensor cores run dramatically faster at lower precision: an H100's FP8 tensor-core throughput is roughly double its bf16 throughput, which is itself far above FP32. Dropping precision does two things on the roofline simultaneously — it raises the **compute roof** $\pi$ (faster math) *and* lowers bytes moved (smaller dtype), which raises **intensity** for memory-bound kernels. That double win is why FP8 training and INT4/INT8 inference are central to Part IV; the numerics and failure modes are in [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) and [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html). The roofline is precisely the tool that quantifies the payoff: a kernel sitting on the memory roof at $I=1$ FLOP/B doubles its ceiling if you halve its byte width.

## Profiling: Confronting the Model With Reality

Theory places kernels on the roofline; profiling tells you where they *actually* land and why. Two tools cover almost everything an LLM engineer needs: the **PyTorch profiler** (operator-level, framework-aware) and **NVIDIA Nsight** (Systems for timeline/overlap, Compute for per-kernel roofline).

### The PyTorch profiler

Start here. It attributes time to PyTorch ops and CUDA kernels, captures shapes, and exports a Chrome trace you can open in `chrome://tracing` or Perfetto.

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

model = build_model().cuda().eval()          # your nn.Module
x = torch.randint(0, 32000, (8, 2048), device="cuda")

# Warm up: trigger cuDNN/cuBLAS autotuning, allocator, and torch.compile
for _ in range(3):
    with torch.no_grad():
        model(x)
torch.cuda.synchronize()

# schedule: skip 1 step, warm up 1, then record 3 active steps, repeat once
prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    record_shapes=True,        # capture input shapes per op
    profile_memory=True,       # track allocator activity
    with_stack=False,          # set True for source attribution (slower)
) as prof:
    for _ in range(6):
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()   # accurate per-step boundaries
        prof.step()

# Rank kernels by total CUDA time — this is your hot list.
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=15))

# Export a timeline you can inspect visually:
prof.export_chrome_trace("trace.json")
```

What to look for in the output table:

- **The top few CUDA kernels by total time** — that's where optimization pays. Usually some flavor of GEMM (`ampere_..._gemm`, `cutlass`) and the attention kernel.
- **A long tail of tiny element-wise/normalization kernels.** If their *summed* time is large, you have a fusion opportunity (`torch.compile`, see [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)). Many small memory-bound kernels back-to-back = bandwidth-bound program.
- **Gaps where the GPU is idle** in the Chrome trace — that's CPU launch overhead or a synchronizing `.item()`/`.cpu()` call stalling the pipeline. CUDA graphs or removing host syncs fix these.

### Turning a profile into a roofline verdict

A measured kernel time plus its FLOP and byte counts gives an *achieved* intensity and *achieved* FLOP/s, which you drop onto the roofline.

```python
def kernel_verdict(flops, bytes_moved, seconds,
                   peak_tflops=312, hbm_TBs=2.0):
    """Given measured time + analytic FLOPs/bytes, classify the kernel."""
    intensity = flops / bytes_moved                      # FLOP/B
    ridge = (peak_tflops * 1e12) / (hbm_TBs * 1e12)      # machine ridge
    achieved = flops / seconds / 1e12                    # TFLOP/s achieved
    roof = min(peak_tflops, hbm_TBs * 1e12 * intensity / 1e12)  # the ceiling
    return {
        "intensity_FLOP_per_B": round(intensity, 2),
        "machine_ridge": round(ridge, 1),
        "bound": "compute" if intensity > ridge else "memory",
        "achieved_TFLOPs": round(achieved, 1),
        "roofline_ceiling_TFLOPs": round(roof, 1),
        "efficiency_vs_ceiling_%": round(100 * achieved / roof, 1),
    }

# A LayerNorm over (8, 2048, 4096) bf16, measured at, say, 90 microseconds.
n = 8 * 2048 * 4096
ln_flops = 8 * n            # rough: a handful of ops per element
ln_bytes = 2 * (2 * n)      # read x, write y in bf16 (ignoring the tiny γ,β)
print("LayerNorm:", kernel_verdict(ln_flops, ln_bytes, 90e-6))

# A big FFN GEMM (8*2048 tokens) x (4096 -> 11008), bf16, measured 1.1 ms.
m, k, nn = 8*2048, 4096, 11008
gemm_flops = 2 * m * k * nn
gemm_bytes = 2 * (m*k + k*nn + m*nn)
print("FFN GEMM :", kernel_verdict(gemm_flops, gemm_bytes, 1.1e-3))
```

The LayerNorm comes back with intensity well under 1 FLOP/B and "memory"-bound — confirming it is hopeless to optimize its arithmetic and ripe for fusion. The GEMM comes back compute-bound with an efficiency you can compare against peak; if it's at 70% of the compute roof, that's healthy, and chasing the last 30% means kernel-level work (CUTLASS, better tiling) rather than algorithmic change.

### NVIDIA Nsight: Systems and Compute

When the PyTorch profiler isn't enough — you need to see multi-GPU overlap, NVLink/NCCL traffic, or a true hardware roofline — drop to Nsight.

**Nsight Systems** (`nsys`) captures a full timeline: CUDA kernels, memory copies, NCCL collectives, CPU threads, and the gaps between them. It's the tool for "why is my GPU only 60% busy" — usually the answer is a comms bubble or a host-side stall.

```bash
# Capture a timeline of a training step (limit duration to keep the file small).
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --output=run_timeline \
    python train.py --max-steps 20

# Open run_timeline.nsys-rep in the Nsight Systems GUI, or get a CLI summary:
nsys stats run_timeline.nsys-rep
```

Annotate regions with NVTX so the timeline is readable:

```python
import torch.cuda.nvtx as nvtx
with nvtx.range("attention"):
    out = attn(x)
with nvtx.range("ffn"):
    out = ffn(out)
# These ranges appear as labeled blocks on the Nsight timeline.
```

**Nsight Compute** (`ncu`) profiles a *single kernel* in microscopic detail — and it can draw the hierarchical roofline directly, placing your kernel's point under the tensor-core, FMA, and memory roofs. It is heavyweight (replays the kernel many times to collect every counter), so target one kernel:

```bash
# Profile just the kernels whose mangled name matches 'gemm', collect the
# full set including the roofline section, write a report file.
ncu --set full \
    --kernel-name-base demangled \
    --kernel-name regex:".*gemm.*" \
    --launch-count 5 \
    --export gemm_report \
    python forward_once.py
```

In the resulting report, the key panels are **Compute (SM) Throughput** and **Memory Throughput** (each as % of peak), the **Roofline** chart (your dot vs. the roofs), and the **Warp State** / **Occupancy** sections that explain *why* a compute-bound kernel still misses peak (stalls on memory dependency, low occupancy, etc.). If SM throughput is ~90% you're compute-bound and near the roof — done. If memory throughput is ~90% but SM is low, you're bandwidth-bound and the fix is data movement, not math. This is the roofline, measured on silicon.

!!! warning "Common pitfall: measuring without warmup or synchronization"

    CUDA kernel launches are asynchronous. If you time with `time.time()` around a forward pass and forget `torch.cuda.synchronize()`, you measure *launch* time, not *execution* time — often 100× too fast. And the first few iterations include one-time costs: cuBLAS/cuDNN autotuning, `torch.compile` JIT, allocator growth, and clock spin-up. Always warm up (3–10 iters) and synchronize at every measurement boundary. For micro-benchmarks prefer `torch.cuda.Event` timing or `torch.utils.benchmark.Timer`, which handle this correctly.

## Putting It Together: A Performance-Debugging Playbook

The roofline turns "my model is slow" into a directed search. Here is the loop, in order:

```text
1. MEASURE end-to-end throughput (tokens/s) and compute MFU = 6N·tok_s / (GPUs·π).
        |
        v
2. Is MFU healthy (>~40%)?  --yes--> ship it; revisit only if cost demands.
        | no
        v
3. PROFILE one step (torch.profiler / nsys). Find the top time sinks.
        |
        +--> Big GEMMs dominate, GPU ~saturated? => compute-bound.
        |       Fix: lower precision (bf16/fp8), better GEMM (cuBLAS/CUTLASS),
        |            fewer FLOPs (MoE, smaller d_ff), tensor-core-friendly shapes.
        |
        +--> Many tiny element-wise/norm kernels sum large? => memory-bound.
        |       Fix: fuse (torch.compile), keep data on-chip, recompute>reload,
        |            quantize activations to move fewer bytes.
        |
        +--> GPU idle gaps on the timeline? => overhead/comms-bound.
                Fix: CUDA graphs, remove host syncs (.item()/.cpu()),
                     overlap comms with compute, bigger batch, better parallelism.
        |
        v
4. ncu the single worst kernel: read SM% vs Memory%; place it on the roofline;
   apply the matching fix; remeasure MFU. Repeat.
```

Notice that *every* branch maps to a roofline regime: compute-bound (raise the compute roof or do fewer FLOPs), memory-bound (raise intensity / move fewer bytes), or overhead-bound (a third axis the simple roofline doesn't draw — launch/sync/comms latency — fixed by overlap and batching). Internalizing this triage is what separates "I tried `torch.compile` and it didn't help" from "the timeline showed an NCCL bubble, so I overlapped the all-reduce."

!!! interview "Interview Corner"

    **Q:** You're serving a 13B-parameter model in bf16 on a single A100-80GB and getting only ~50 tokens/second on single-stream decode. Your colleague proposes pruning 20% of the FLOPs by removing attention heads. Will that help, and what would you do instead?

    **A:** Pruning FLOPs won't meaningfully help, because single-stream decode is *memory-bound*, not compute-bound. Each generated token requires reading essentially all 26 GB of weights ($13\text{B}\times 2$ bytes) from HBM. At ~2 TB/s that's a ~13 ms floor, i.e. a ceiling around 75 tok/s before overhead — exactly the regime we're in. Decode does only $2N \approx 2.6\times10^{10}$ FLOPs/token, well under 1% of the A100's 312 TFLOP/s, so the arithmetic units are already idle; removing 20% of FLOPs leaves runtime essentially unchanged. The right fixes attack *bytes moved*, not FLOPs: (1) **batch** requests with continuous batching so one weight read serves many tokens, pushing intensity above the ridge; (2) **quantize** weights to INT8/INT4 to halve or quarter the bytes streamed per token — this directly raises the bandwidth-bound token rate; (3) **speculative decoding** to verify several tokens per weight-read pass. The general principle: identify which roofline regime you're in *first*; decode is bandwidth-bound, so optimize data movement.

!!! example "Worked example: MFU of a concrete training step"

    A 7B model ($N = 6.7\times10^9$) trains on 64 H100s (bf16 peak $\approx 990$ TFLOP/s each). We measure one global step processing a batch of $4{,}096$ sequences $\times\,4{,}096$ tokens $= 1.68\times10^{7}$ tokens in $24$ seconds.

    Throughput: $1.68\times10^{7} / 24 \approx 7.0\times10^{5}$ tokens/s.

    Model FLOP/s: $6N \times \text{tok/s} = 6 \times 6.7\times10^{9} \times 7.0\times10^{5} \approx 2.8\times10^{16}$ FLOP/s $= 28$ PFLOP/s.

    Peak: $64 \times 990$ TFLOP/s $= 63{,}360$ TFLOP/s $= 6.34\times10^{16}$ FLOP/s $= 63.4$ PFLOP/s.

    $$
    \text{MFU} = \frac{28\ \text{PFLOP/s}}{63.4\ \text{PFLOP/s}} \approx 0.44 = 44\%.
    $$

    A 44% MFU is solidly in the healthy band for a large bf16 run. If we'd measured 22%, the playbook says profile: likely a comms bubble (try better overlap or a different parallelism split, see [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html)) or a memory-bound activation path begging for fusion.

## Limits and Refinements of the Roofline

The roofline is a *first-order* model, and knowing its blind spots keeps you honest. It assumes (1) a single dominant memory level — but real kernels also hit L2 and shared-memory roofs, which is why hierarchical rooflines exist; (2) perfect overlap of compute and memory — real kernels lose time to latency that isn't hidden by enough in-flight work (low occupancy), which the basic model can't see; (3) that latency-bound, low-parallelism regimes don't exist — but a tiny kernel with one CTA touches neither roof, bottlenecked purely on launch and latency. For LLM serving specifically, the simple two-resource picture omits a critical third axis — **comms** (NVLink/InfiniBand bandwidth and latency for tensor/pipeline/data parallelism), which deserves its own roofline-like analysis ([Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)).

Despite these caveats, the roofline remains the most valuable single tool in performance engineering because it forces the right *first* question — "am I compute-bound or memory-bound?" — and that question, answered correctly, eliminates the majority of wasted optimization effort. Everything in the rest of Part IV is, in one way or another, a technique for moving a transformer kernel up and to the right on this diagram.

!!! key "Key Takeaways"

    - The roofline is $P(I) = \min(\pi, \beta I)$: a flat **compute roof** $\pi$ (peak FLOP/s) and a sloped **memory roof** $\beta I$ (bandwidth $\times$ arithmetic intensity), meeting at the **ridge** $I^\* = \pi/\beta$ (~156 FLOP/B on A100, ~295 on H100).
    - **Arithmetic intensity** $I = \text{FLOPs}/\text{HBM bytes}$ decides everything: $I < I^\*$ is memory-bound (raise intensity, move fewer bytes), $I > I^\*$ is compute-bound (faster math, fewer FLOPs).
    - Transformer FLOPs are dominated by the $d^2$ terms (FFN + QKVO projections); attention scores are $\propto s^2$ and only dominate the FLOP budget at very long context — yet can dominate *runtime* because the score matrix is memory-bound.
    - Memorize the constants: training $\approx 6N\!\cdot\!D$ FLOPs, inference $\approx 2N$ FLOPs/token. They power every capacity and cost estimate.
    - **Decode is memory-bound** (GEMV, $I\approx1$): single-stream speed is set by weight-read bandwidth, not FLOP/s. Batching and quantization, not FLOP pruning, are the fixes.
    - **MFU** $= 6N\!\cdot\!\text{tok/s} / (\text{GPUs}\times\pi)$ is your north-star efficiency metric; healthy large runs sit ~40–55%, and a drop with no architectural cause is a regression.
    - Profile with the **PyTorch profiler** for operator-level hot lists and **Nsight Systems/Compute** for timelines and hardware rooflines — and always **warm up + synchronize** or your measurements are fiction.
    - Lower precision raises the compute roof *and* the effective intensity simultaneously — the double win behind FP8/INT4.

!!! sota "State of the Art & Resources (2026)"
    The roofline model remains the dominant first-order framework for GPU performance analysis in 2026, and recent work has extended it to LLM-specific settings — characterizing prefill vs. decode regimes, on-device SLMs, and multi-level memory hierarchies. Automated roofline tools now ship inside NVIDIA's official profilers, making the methodology accessible end-to-end without manual FLOP counting.

    **Foundational work**

    - [Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (2009)](https://dl.acm.org/doi/10.1145/1498765.1498785) — the original paper that defined arithmetic intensity, the memory roof, and the compute roof; every GPU performance conversation still references it.
    - [Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022)](https://arxiv.org/abs/2204.02311) — introduces Model FLOPs Utilization (MFU) and Hardware FLOPs Utilization (HFU) as standard training-efficiency metrics for large-scale LLM runs.
    - [Pope et al., *Efficiently Scaling Transformer Inference* (2022)](https://arxiv.org/abs/2211.05102) — derives analytical models for LLM inference efficiency, showing how the prefill/decode split maps onto compute-bound vs. memory-bound roofline regimes on TPU slices.

    **Recent advances (2023–2026)**

    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — the canonical roofline-driven kernel redesign: avoids materializing the full attention matrix in HBM to cross the memory roof.
    - [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608) — extends IO-aware attention to H100 hardware, exploiting async WGMMA and TMA instructions to push closer to the H100 compute roof.
    - [Yuan et al., *LLM Inference Unveiled: Survey and Roofline Model Insights* (2024)](https://arxiv.org/abs/2402.16363) — systematic survey that places every major LLM inference optimization (quantization, speculative decoding, sparsity) on a shared roofline framework with worked hardware numbers.
    - [Bi et al., *RooflineBench: A Benchmarking Framework for On-Device LLMs via Roofline Analysis* (2026)](https://arxiv.org/abs/2602.11506) — extends roofline analysis to edge/mobile hardware, identifies the operational-intensity regression as model depth grows, and evaluates MLA as a fix.

    **Open-source & tools**

    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — official FlashAttention repo (v1–v4); the reference IO-aware CUDA kernel used in virtually every production LLM training stack.
    - [hahnyuan/LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) — open-source tool that computes per-layer arithmetic intensity, roofline position, and memory traffic for arbitrary LLM configs across hardware targets; companion to the Yuan et al. 2024 survey.

    **Go deeper**

    - [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) — official guide to the GPU kernel profiler that renders hierarchical roofline charts directly from hardware counters; covers the `--set full` roofline section used throughout this chapter.
    - [NERSC Roofline Performance Model Guide](https://docs.nersc.gov/tools/performance/roofline/) — practical tutorial on the Empirical Roofline Toolkit (ERT), hierarchical rooflines, and measuring arithmetic intensity with Nsight Compute on real GPU workloads.

## Further reading

- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (2009) — the original roofline paper.
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022) — defines and popularizes Model FLOPs Utilization (MFU) and Hardware FLOPs Utilization (HFU).
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020) and Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022) — origin of the $6N$ training-FLOP convention used throughout.
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) — the canonical example of an IO-aware, roofline-driven kernel redesign.
- NVIDIA *Nsight Systems* and *Nsight Compute* documentation, and the *CUDA C++ Programming Guide* — the practical profiling tools, including Nsight Compute's built-in roofline analysis.
- PyTorch documentation for `torch.profiler` and `torch.utils.flop_counter` — operator-level profiling and exact FLOP accounting on a real graph.
