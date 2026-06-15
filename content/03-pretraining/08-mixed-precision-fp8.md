# 3.8 Mixed Precision, bf16 & FP8 Training

Every frontier model you have heard of was trained in low precision. Not because anyone wanted to throw away bits, but because the alternative — full FP32 — is roughly twice the memory, half the arithmetic throughput, and double the network traffic. Modern accelerators are built around low-precision matrix engines: an NVIDIA H100 delivers on the order of ~1,000 TFLOP/s of bf16 tensor-core throughput and roughly twice that in FP8, while FP32 on the CUDA cores is an order of magnitude slower. If you train in FP32 you are using a fraction of the chip you paid for.

But you cannot just cast everything to 16 or 8 bits and press go. Floating-point numbers have *finite range* and *finite precision*, and the gradients, activations, and weight updates of a deep network span many orders of magnitude. Cast naively and you get `NaN` within a hundred steps, or — more insidiously — a model that trains but silently converges to a worse loss because small gradient contributions vanished into zero.

This chapter is about the discipline of **mixed precision**: keeping the bits that matter in a wide format and pushing everything else into a narrow one. We will build the theory from the floating-point representation up, write a correct AMP training loop from scratch, and then go all the way to **FP8** — the 8-bit regime that GPT-class models now use in production. We assume you have read [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html); we lean on it heavily but re-derive the parts that matter here.

## The floating-point formats: fp32, fp16, bf16, fp8

A floating-point number is `sign × mantissa × 2^exponent`. The split between exponent bits (which set the *dynamic range* — how big and how small a number can be) and mantissa bits (which set the *relative precision* — how finely you can distinguish nearby numbers) is the entire story of mixed precision. Here are the four formats that matter, with their IEEE-style `(exponent, mantissa)` bit budgets:

| Format | Bits | Exp / Mant | Max finite | Smallest normal | Rel. precision (ulp) |
|---|---|---|---|---|---|
| FP32 | 32 | 8 / 23 | ~3.4e38 | ~1.2e-38 | ~1.2e-7 |
| FP16 (IEEE half) | 16 | 5 / 10 | 65504 | ~6.1e-5 | ~9.8e-4 |
| BF16 (bfloat16) | 16 | 8 / 7 | ~3.4e38 | ~1.2e-38 | ~7.8e-3 |
| FP8 E4M3 | 8 | 4 / 3 | 448 | ~1.5e-2 (subnormal smaller) | ~0.125 |
| FP8 E5M2 | 8 | 5 / 2 | 57344 | ~6.1e-5 | ~0.25 |

Read this table as the key to everything that follows. The two 16-bit formats have the *same size* but make opposite trades:

- **FP16** keeps 10 mantissa bits (good precision) but only 5 exponent bits, so its maximum value is **65504** and its smallest normal is ~6e-5. Gradients in a transformer routinely live below 6e-5, so they **underflow to zero in fp16**. This is why fp16 *requires loss scaling* (we cover it below).
- **BF16** is simply "FP32 with the bottom 16 mantissa bits chopped off." It keeps all 8 exponent bits, so it has the **same dynamic range as fp32** — it will never overflow or underflow where fp32 wouldn't. The price is only 7 mantissa bits, i.e. ~2-3 decimal digits of precision. For training, *range matters more than precision*, which is why bf16 has become the default and **bf16 needs no loss scaling.**

The conversion between bf16 and fp32 is so simple it is worth seeing explicitly — it is just a truncation (or round-to-nearest) of the low 16 bits:

```python
import torch
import struct

def fp32_to_bf16_bits(x: float) -> int:
    """Show that bf16 is literally the top 16 bits of fp32.
    Round-to-nearest-even on the 16 discarded mantissa bits."""
    [bits] = struct.unpack("<I", struct.pack("<f", x))   # 32-bit pattern
    # round to nearest even: add the rounding bias before truncating
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    bits = (bits + rounding_bias) >> 16
    return bits & 0xFFFF

x = 3.1415927
print(f"fp32 {x}  ->  bf16 bits 0x{fp32_to_bf16_bits(x):04x}")
# Compare against PyTorch's own conversion:
t = torch.tensor(x, dtype=torch.float32)
print("torch bf16:", t.to(torch.bfloat16).item())   # ~3.140625  (7-bit mantissa)
```

Notice the bf16 value `3.140625` differs from π in the third decimal — that is the 7-bit mantissa biting. For activations and weights this rounding error is noise the optimizer happily absorbs; for the *accumulation* inside a matmul it would be catastrophic, which is the next idea.

### Tensor cores accumulate in fp32

A critical, often-missed detail: when a tensor core multiplies two bf16 (or fp16, or fp8) matrices, it does **not** accumulate the dot product in the input precision. It multiplies pairs in low precision and **accumulates the partial sums in fp32** inside the hardware. So a `[4096 × 4096] @ [4096 × 4096]` bf16 matmul sums 4096 products in fp32 and only rounds the final result back to bf16. This is why low-precision matmul is numerically tolerable at all: the long error-accumulating reduction happens in 32 bits. Keep this picture — *narrow inputs, wide accumulator* — in mind; it is the same trick FP8 uses, just more aggressively.

{{fig:fp8mp-tensorcore-accumulate}}

## Why naive fp16 breaks: range, underflow, and the update problem

Let's make the failure concrete. Consider a single weight $w = 1.0$ and a tiny gradient times learning rate, $\eta g = 2 \times 10^{-4}$. We want $w \leftarrow w - \eta g = 0.9998$.

In fp16, the representable numbers near $1.0$ are spaced $2^{-10} \approx 9.77 \times 10^{-4}$ apart (that is the ulp — unit in the last place). Our desired update $2 \times 10^{-4}$ is **smaller than half an ulp**, so when we round $1.0 - 0.0002$ to the nearest fp16 value we get... exactly $1.0$. **The update is silently lost.** Worse, this happens at *every* step late in training when gradients shrink, so the model stops learning even though loss looks vaguely fine.

This is the **swamping** or **stagnation** problem, and it has nothing to do with overflow — it is pure precision loss when you add a small number to a large one in the same low-precision format. There are two complementary fixes, and you need both for fp16:

1. **Master weights in fp32.** Keep the *authoritative* copy of every weight in fp32. Do the matmuls and activations in fp16 (fast), but apply the optimizer update to the fp32 master copy (precise), then cast a fresh fp16 copy for the next forward. Now $1.0 - 0.0002 = 0.9998$ is representable, the update sticks, and the tiny updates accumulate over many steps.
2. **Loss scaling** to fight *underflow* of the gradients themselves before they ever reach the optimizer (next section).

The second problem is **range**. The fp16 max is 65504. Attention logits, the output of a large matmul, or a loss spike can exceed that and produce `inf`, which propagates to `NaN` through the backward pass. bf16, with its fp32-sized exponent, essentially never hits this.

!!! note "Aside: this is the same idea as Kahan summation"

    Master weights are a form of compensated summation. You are keeping the running total (the weight) at higher precision than the increments (the scaled gradients) so that many small increments are not swamped by one large running value. The connection to the classic Kahan summation algorithm from [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) is exact in spirit: protect the accumulator.

## Automatic Mixed Precision (AMP) and loss scaling

**Automatic Mixed Precision (AMP)** is the framework that automates "wide where it matters, narrow where it's safe." In PyTorch it has two halves:

- `torch.autocast`: a context manager that, for ops inside it, automatically chooses a precision. Matmuls, convolutions, and linear layers run in the low precision (bf16/fp16); reductions that need range — softmax, layer norm, loss, `exp`, `sum` — are kept in fp32. It maintains an internal **op allow/deny list** so you don't have to annotate every layer.
- `torch.cuda.amp.GradScaler` (or `torch.amp.GradScaler`): implements **loss scaling**, needed only for fp16.

### The loss-scaling trick

{{fig:loss-scaling}}

The fix for gradient underflow is beautifully simple. Gradients are too small to represent in fp16, so before the backward pass we multiply the loss by a large constant $S$ (the **loss scale**, e.g. $S = 2^{16} = 65536$). By the chain rule, *every* gradient in the network is then multiplied by $S$ too:

$$
\frac{\partial (S \cdot \mathcal{L})}{\partial w} = S \cdot \frac{\partial \mathcal{L}}{\partial w}
$$

This shifts the whole gradient distribution up by $S$, out of the fp16 underflow zone and into the representable range. Then, *after* the backward pass but *before* the optimizer step, we divide the gradients by $S$ ("unscale") to recover the true gradient, and update the fp32 master weights. The scaling is mathematically a no-op on the final update; it only buys representable precision during backprop.

```text
loss ──×S──► backward ──► grads (×S, now representable) ──÷S──► unscaled grads ──► clip ──► optimizer.step()  (fp32 master)
```

### Static vs dynamic loss scaling

What value of $S$? Too small and gradients still underflow; too large and the *scaled* gradients overflow to `inf`. The sweet spot drifts during training as gradient magnitudes change. Two strategies:

- **Static loss scaling:** pick one constant (say $2^{15}$) and hope. Simple, but fragile — wrong choice wastes range or causes overflow.
- **Dynamic loss scaling** (what `GradScaler` does): start high (e.g. $2^{16}$), and adapt. After each backward, **check the gradients for `inf`/`NaN`**. If any are found, the scale was too big: **skip the optimizer step** (don't corrupt the weights with garbage) and **halve** $S$. If many steps pass with no overflow (e.g. 2000 steps), the scale may be too conservative: **double** $S$ to claw back precision. This is an AIMD (additive-increase / multiplicative-decrease)-style controller that automatically tracks the gradient distribution.

```python
# Conceptual core of a dynamic GradScaler (PyTorch implements this in C++).
class DynamicLossScaler:
    def __init__(self, init_scale=2.0**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor      # multiply by this on success
        self.backoff_factor = backoff_factor    # multiply by this on overflow
        self.growth_interval = growth_interval  # steps of success before growing
        self._good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def update(self, found_inf: bool):
        """Call after inspecting unscaled grads for inf/nan."""
        if found_inf:
            self.scale *= self.backoff_factor   # too big -> back off, skip step
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale *= self.growth_factor  # been safe a while -> grow
                self._good_steps = 0
```

### bf16 needs no loss scaling — here is exactly why

This is a favorite interview question, so be precise. Loss scaling exists to combat gradient **underflow**, which is a *range* problem: fp16's smallest normal is ~6e-5, and gradients live below that. bf16 has the **same exponent width as fp32**, so its smallest normal is ~1.2e-38 — gradients simply never underflow there. There is nothing to rescue, so loss scaling adds complexity for zero benefit. You drop the `GradScaler` entirely. (You still keep fp32 master weights inside the optimizer if you want the most precise updates, though with bf16 + a state-fp32 optimizer like Adam this is often handled implicitly — see below.) The trade you accept is bf16's coarser 7-bit mantissa, but the network tolerates that rounding noise.

!!! warning "Common pitfall: using a GradScaler with bf16"

    If you wrap a bf16 run in a `GradScaler`, at best it is a no-op that wastes a little time, and at worst the inf-checking logic interacts badly and skips steps it shouldn't. Rule: **fp16 ⇒ GradScaler; bf16 ⇒ no GradScaler.** In `torch.autocast(dtype=torch.bfloat16)` you should not scale.

## A correct AMP training loop, from scratch

Here is a complete, heavily commented training step that works for both regimes. It shows master weights, autocast, scaling, unscaling-before-clipping (the order matters!), and skipped steps. This is the loop you would actually ship.

```python
import torch
import torch.nn as nn

# --- choose your regime ----------------------------------------------------
USE_BF16 = True                       # bf16: no scaler. fp16: needs scaler.
amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
device = "cuda"

model = build_transformer().to(device)            # weights in fp32 (the masters)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# GradScaler is a no-op when disabled, so this one line handles both regimes.
scaler = torch.amp.GradScaler(device, enabled=not USE_BF16)

def train_step(batch):
    optimizer.zero_grad(set_to_none=True)

    # 1) FORWARD under autocast: matmuls run in amp_dtype on tensor cores,
    #    softmax / layernorm / loss are auto-kept in fp32 by the op allow-list.
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(batch["input_ids"])
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),   # cast logits to fp32 for the loss
            batch["labels"].view(-1),
        )

    # 2) BACKWARD on the *scaled* loss. With bf16 the scale is 1.0 (no-op).
    #    With fp16 this lifts grads out of the underflow zone.
    scaler.scale(loss).backward()

    # 3) UNSCALE the grads in-place so we can clip on TRUE gradient magnitudes.
    #    (Clipping a *scaled* grad would clip to the wrong threshold!)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 4) STEP. scaler.step() internally checks for inf/nan: if found, it SKIPS
    #    the optimizer.step() (so garbage never touches the fp32 masters).
    scaler.step(optimizer)

    # 5) UPDATE the dynamic loss scale for next iteration (no-op in bf16).
    scaler.update()

    return loss.item()
```

The five non-obvious correctness points, in order:

1. **Cast logits to fp32 before cross-entropy.** The log-softmax over a 50k-vocab needs the fp32 range/precision; doing it in fp16 risks `inf` from `exp` of a large logit.
2. **Backward on the scaled loss**, not the raw loss.
3. **Unscale before clipping.** `clip_grad_norm_` compares the gradient norm to `max_norm=1.0`. If the grads are still scaled by $2^{16}$, every batch looks like it has a norm of ~65,000 and you clip everything to noise. Unscale first.
4. **`scaler.step` does the inf check and the skip.** You never call `optimizer.step()` directly in fp16 AMP.
5. **`scaler.update()`** runs the AIMD controller.

### Where do master weights live?

In the loop above the model's own `nn.Parameter`s are fp32 — they *are* the master copy. `autocast` casts them to bf16/fp16 *on the fly* for each matmul and discards the cast; the fp32 originals are what AdamW updates. This is the standard PyTorch AMP pattern and the simplest mental model: **parameters fp32, compute low-precision, optimizer touches fp32.**

A second pattern, common in large-scale frameworks (DeepSpeed, Megatron, FSDP with `MixedPrecision`), stores the *parameters themselves* in bf16 to halve parameter memory and communication, and keeps a **separate fp32 master copy plus fp32 optimizer state** (momentum, variance). The optimizer steps in fp32, then copies the result back into the bf16 parameter. This is what people mean by the canonical "mixed precision" recipe from Micikevicius et al.'s *Mixed Precision Training* (2017). The memory accounting (per parameter): 2 bytes bf16 weight + 4 bytes fp32 master + 4 + 4 bytes Adam states = **14 bytes/param**, versus 16 bytes for the all-fp32 recipe — and crucially the *communication* (all-reduce of gradients, all-gather of weights) moves in 2-byte bf16, halving network traffic. See [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) for how ZeRO shards exactly these fp32 states.

!!! example "Worked example: memory and the cost of master weights"

    Take a 7B-parameter model trained with bf16 weights + fp32 master + AdamW (fp32 momentum and variance). Per parameter we store:

    - bf16 weight: 2 bytes
    - fp32 master weight: 4 bytes
    - fp32 Adam first moment $m$: 4 bytes
    - fp32 Adam second moment $v$: 4 bytes

    Total = **14 bytes/param**. For 7e9 params: $7\times10^9 \times 14 = 9.8\times10^{10}$ bytes $\approx$ **98 GB** just for the optimizer + weights, before activations and gradients. Gradients add another 2 bytes/param (bf16) = 14 GB. This is why a "small" 7B model does not fit the optimizer state on a single 80 GB H100 and you reach for ZeRO/FSDP sharding. Notice the master weights (4 bytes) are the second-biggest line item — people sometimes ask "can we drop them?" The answer is the stagnation problem above: without fp32 masters, small bf16 updates get swamped and late-training progress stalls.

## FP8 training: pushing the matmuls to 8 bits

bf16 is now table stakes. The 2024-2026 frontier is **FP8 training**, where the heavy matmuls run with 8-bit inputs, doubling tensor-core throughput again and halving the bytes moved through the matmul. NVIDIA's Hopper (H100) and Blackwell GPUs have native FP8 tensor cores; DeepSeek-V3 famously trained a 671B-parameter MoE largely in FP8 and documented the recipe.

But 8 bits is *brutally* few. Recall the two FP8 formats and their jobs:

- **E4M3** (4 exponent, 3 mantissa): more mantissa, less range. Max ~448. Used for the **forward pass** tensors — weights and activations — where we want a bit more precision and the magnitudes are bounded.
- **E5M2** (5 exponent, 2 mantissa): more range, less precision. Max ~57344. Used for **gradients** in the backward pass, which span a wider dynamic range and benefit from the extra exponent bit (this echoes the bf16-vs-fp16 logic, one level down).

With only 3 mantissa bits, the relative spacing of E4M3 numbers is ~12.5% — *one part in eight*. You cannot just cast a tensor whose values span four orders of magnitude into E4M3 and keep any signal. The entire art of FP8 training is **scaling**: choosing a per-tensor (or finer) multiplier that maps each tensor's actual distribution into the narrow E4M3/E5M2 representable window, just like loss scaling but applied tensor-by-tensor, continuously.

### Per-tensor scaling and the delayed-scaling recipe

For each FP8 matmul operand $X$ (an activation, weight, or gradient) we keep a scale factor $s_X$. We store $X_{\text{fp8}} = \operatorname{cast}_{\text{fp8}}(s_X \cdot X)$ and remember $s_X$. To use it, the matmul computes in FP8 and the result is de-scaled by $1/s_X$. The scale is chosen so that the **maximum absolute value** (the *amax*) of the tensor lands near the top of the FP8 range without overflowing:

$$
s_X = \frac{\text{fp8\_max}}{\operatorname{amax}(X)} \cdot \alpha, \qquad \alpha \lesssim 1 \text{ (safety margin)}
$$

Computing `amax(X)` requires a full pass over the tensor *before* you can cast it — an extra reduction on the critical path. The clever production trick, used by NVIDIA's **Transformer Engine**, is **delayed scaling**: keep a short rolling history (e.g. the last 16 steps) of each tensor's amax, and use the **max over that history** to pick this step's scale. Then casting and the matmul can be fused — you don't stall waiting for the current amax; you compute it *while* you cast and stash it in the history buffer for next time. It is a small bet that the amax does not jump wildly between consecutive steps, which holds in practice once training is stable.

```python
# Sketch of FP8 per-tensor cast with delayed scaling (the Transformer Engine idea).
import torch

FP8_E4M3_MAX = 448.0

class DelayedScale:
    def __init__(self, history_len=16, margin=1.0):
        self.amax_history = torch.zeros(history_len)
        self.ptr = 0
        self.margin = margin

    def compute_scale(self):
        amax = self.amax_history.max().clamp_min(1e-12)   # max over recent history
        # scale maps amax -> FP8 max, with a safety margin < 1
        return (FP8_E4M3_MAX / amax) * self.margin

    def cast_to_fp8(self, x: torch.Tensor):
        scale = self.compute_scale()                 # uses PAST amax (delayed)
        x_scaled = x * scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)     # native FP8 dtype
        # record THIS tensor's amax for future steps (off the critical path)
        self.amax_history[self.ptr] = x.abs().amax()
        self.ptr = (self.ptr + 1) % self.amax_history.numel()
        # return both: the matmul de-scales its output by 1/scale
        return x_fp8, scale
```

A full FP8 linear layer then does: cast `X` and `W` to E4M3 with their scales $s_X, s_W$; run the FP8 tensor-core matmul (which accumulates in fp32 internally); and de-scale the fp32 output by $1/(s_X s_W)$. Three matmuls per linear layer get FP8'd — the forward ($Y = XW$), and the two backward matmuls ($\nabla X = \nabla Y\, W^\top$ and $\nabla W = X^\top \nabla Y$), the latter two using E5M2 for the gradient operand.

### Blockwise / fine-grained scaling: the DeepSeek refinement

Per-tensor scaling has a weakness: a single **outlier** value blows up the amax, forcing a small scale that crushes all the *normal* values into the bottom few FP8 codes, where the 3-bit mantissa quantizes them coarsely. Transformers are notorious for activation outliers in specific channels (the same phenomenon that motivates SmoothQuant in [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html)).

The fix is **finer-grained scaling**: instead of one scale per tensor, use one scale per **block**. DeepSeek-V3's recipe uses **per-token-group (1×128) tile scaling for activations and 128×128 block scaling for weights**. Each block gets its own amax and scale, so an outlier in one block no longer poisons the quantization of every other block. The cost is bookkeeping — you carry many small scale factors and must apply them correctly through the matmul — but the numerical robustness is what made full-FP8 training of a 671B model feasible. DeepSeek also kept certain sensitive components (embeddings, the output head, normalization, and the attention softmax) in higher precision (bf16/fp32), and crucially **accumulated the FP8 matmul partial sums into fp32 with periodic promotion** rather than trusting the tensor core's native accumulation alone, because at FP8 input scale even the accumulator precision starts to matter.


{{fig:fp8mp-pertensor-vs-blockwise-scaling}}


### Using FP8 in practice: Transformer Engine

You rarely hand-roll the casts. NVIDIA's **Transformer Engine (TE)** provides FP8-aware layers and an `fp8_autocast` context that manages scales, history, and the E4M3/E5M2 split for you.

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

# A TE Linear behaves like nn.Linear but can run its matmul in FP8.
layer = te.Linear(4096, 4096, bias=True)

# HYBRID = E4M3 for forward tensors, E5M2 for the gradient (backward) tensors.
fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,        # rolling window for delayed scaling
    amax_compute_algo="max",    # use the max over the window
)

x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    y = layer(x)         # the GEMM runs in FP8; accumulation is fp32
# Everything outside the context (norms, residual adds, softmax) stays bf16/fp32.
```

The mental model: **FP8 is for the GEMMs only.** The element-wise glue of a transformer — residual adds, the nonlinearity inside softmax and the RMSNorm/LayerNorm statistics — stays in bf16 or fp32. You FP8 the three big matmuls per linear layer (forward + two backward) and per attention projection, because that is where >90% of the FLOPs and a large share of the bytes are. See [FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8](../04-kernels-efficiency/03-flash-attention-2-3.html) for FP8 *inside* the attention kernel itself, which is a harder problem because the softmax sits between two matmuls.

!!! tip "Practitioner tip: keep the head and embeddings out of FP8"

    Empirically the input embedding, the final LM head (the big $d_\text{model} \times \text{vocab}$ projection), the LayerNorm/RMSNorm, and the attention softmax are the most precision-sensitive parts of a transformer. Almost every successful FP8 recipe (DeepSeek-V3, NVIDIA's) keeps these in bf16/fp32 and FP8s only the bulk feed-forward and projection GEMMs. The throughput you give up is small; the stability you buy is large. Start conservative, then expand the FP8 surface as you confirm the loss curve matches a bf16 baseline.

## Numerics, stability, and debugging low-precision runs

Low precision interacts with everything else in the training stack. A few mechanisms worth internalizing:

**Stochastic rounding.** When you repeatedly add small bf16 increments to a bf16 value, round-to-nearest can systematically lose every increment smaller than half an ulp (the stagnation problem). **Stochastic rounding** rounds up or down *with probability proportional to the distance* to each neighbor, so in expectation the increments are preserved even when each individual one is below the ulp. This is why some bf16-only optimizers (and the bf16 master-weight-free recipes) use stochastic rounding on the weight update — it lets you skip the fp32 master copy and still make progress. It is an *unbiased* rounding scheme; round-to-nearest is biased toward zero for sub-ulp updates.

$$
\operatorname{SR}(x) = \begin{cases} \lceil x \rceil & \text{with prob. } (x - \lfloor x \rfloor) \\ \lfloor x \rfloor & \text{with prob. } (\lceil x \rceil - x) \end{cases}
\qquad \mathbb{E}[\operatorname{SR}(x)] = x
$$

**Loss spikes and precision.** Many large-run loss spikes ([Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)) are precision-mediated: an activation overflows fp16 → `inf` → `NaN` gradient → corrupted weights → divergence. bf16 removes most of these by construction. In FP8, spikes can come from a sudden amax jump that the delayed-scaling history hadn't anticipated, momentarily overflowing E4M3; a per-block scaling scheme with a conservative margin mitigates it.

**What to keep in fp32, always.** A reliable checklist of "never FP8, often not even bf16" components: the softmax normalization, LayerNorm/RMSNorm running statistics, the loss and its log-softmax, the optimizer state and the master weights, and any `1/x`, `exp`, `log`, or large reduction. The unifying principle: **anything that involves a large-range reduction or a division by a small number wants fp32.**

**Debugging checklist when a low-precision run misbehaves:**

```text
Symptom                         Likely cause                         Fix
─────────────────────────────   ──────────────────────────────────   ─────────────────────────────
NaN within ~100 steps (fp16)    loss-scale too high -> grad inf       lower init scale / trust GradScaler
loss flat after early progress  update swamping (no fp32 master)      add fp32 master / stochastic round
loss matches bf16 then diverges FP8 amax spike overflows E4M3         shorter history / bigger margin / blockwise
slightly worse final loss vs    head/embeddings/softmax in FP8        exclude sensitive layers from FP8
   bf16 baseline                                                      
grad norm ~65000 every step     clipping a *scaled* gradient          unscale_ before clip_grad_norm_
```

**A note on determinism.** Low-precision tensor-core matmuls are not bit-for-bit deterministic across runs unless you force it, because fp32 accumulation order can vary with the kernel's tiling. This rarely matters for training quality but matters for debugging "did my change move the loss?" — pin seeds and accept small nondeterminism, or use deterministic kernels for an A/B and pay the speed cost.

!!! interview "Interview Corner"

    **Q:** Your colleague switches a training run from bf16 to fp16 to "get more precision" and it starts producing `NaN`s after a few hundred steps. What's going on, and what would you change?

    **A:** fp16 has *more mantissa* (10 vs 7 bits) but *much less dynamic range* — only 5 exponent bits, max value 65504, smallest normal ~6e-5. Two failures follow. (1) **Overflow:** an activation or attention logit exceeds 65504 → `inf` → `NaN`. (2) **Gradient underflow:** small gradients fall below ~6e-5 and round to zero. The `NaN`s are usually the overflow path. The fix is *not* to add precision but to manage range: enable a **dynamic GradScaler** so the loss (and hence all gradients) is multiplied up out of the underflow zone, with inf-checking that skips corrupted steps and backs off the scale; and ensure **fp32 master weights** so the recovered updates actually stick. But the cleaner answer is: on any Ampere/Hopper GPU, **just use bf16** — same 16 bits, fp32-equal range, no loss scaling, no overflow, marginally coarser mantissa that the optimizer absorbs. fp16 is essentially legacy for pre-Ampere hardware. The "more precision" intuition is a trap: for training, *range beats precision*.

!!! key "Key Takeaways"

    - **Range vs precision is the whole game.** Exponent bits set range; mantissa bits set precision. bf16 trades fp16's precision for fp32's range — and for training, range wins.
    - **bf16 needs no loss scaling** because its exponent matches fp32, so gradients never underflow. fp16 needs a (dynamic) **GradScaler** to lift gradients out of the underflow zone and inf-checking to skip corrupted steps.
    - **Master weights in fp32** solve the update-swamping (stagnation) problem: small low-precision updates are otherwise lost when added to a large weight. The optimizer steps in fp32; compute runs low-precision.
    - Tensor cores take **narrow inputs but accumulate in fp32** — this is what makes any low-precision matmul numerically survivable.
    - **AMP loop order matters:** scale the loss → backward → **unscale before clipping** → step (with inf-skip) → update scale. Cast logits to fp32 before cross-entropy.
    - **FP8 doubles throughput again** using E4M3 (forward) and E5M2 (gradients), but its 3-bit mantissa demands **per-tensor or blockwise scaling** to map each tensor into the tiny representable window; Transformer Engine's **delayed scaling** hides the amax reduction.
    - **DeepSeek-V3-style fine-grained (128×128) blockwise scaling** tames activation outliers that would wreck per-tensor scaling, and keeps embeddings/head/softmax/norms out of FP8.
    - Keep **softmax, norms, loss, and optimizer state in fp32**; the rule of thumb is "any large-range reduction or division by a small number wants fp32."

!!! sota "State of the Art & Resources (2026)"
    Mixed-precision training is now standard practice for all large-scale LLM runs: bf16 with fp32 optimizer states is the default baseline, and FP8 (E4M3/E5M2 with per-tensor or blockwise delayed scaling) is the production frontier on Hopper and Blackwell GPUs. The key open challenges are extending fine-grained FP8 scaling to attention and MoE routing layers while preserving training stability at multi-trillion-token scale.

    **Foundational work**

    - [Micikevicius et al., *Mixed Precision Training* (2018)](https://arxiv.org/abs/1710.03740) — introduced fp16 + loss scaling + fp32 master weights, the template every AMP library follows.
    - [Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* (2019)](https://arxiv.org/abs/1905.12322) — first comprehensive study showing bf16 matches fp32 accuracy across domains with no loss scaling needed.
    - [Micikevicius et al., *FP8 Formats for Deep Learning* (2022)](https://arxiv.org/abs/2209.05433) — defines the E4M3/E5M2 split and rationale; the specification all hardware vendors implemented.

    **Recent advances (2023–2026)**

    - [Peng et al., *FP8-LM: Training FP8 Large Language Models* (2023)](https://arxiv.org/abs/2310.18313) — extends FP8 to gradients and optimizer states, achieving 75% faster training and 39% memory reduction vs. BF16 on GPT-175B.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — first public account of fine-grained (per-token-group and 128×128 tile) FP8 training at 671B scale; details which components stay in bf16/fp32.
    - [Xi et al., *COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training* (2024)](https://arxiv.org/abs/2410.19313) — ICLR 2025; reduces end-to-end training memory 1.54× vs. BF16 by quantizing optimizer states and activations into FP8.

    **Open-source & tools**

    - [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) — the reference library for FP8-aware layers, `fp8_autocast`, delayed scaling, and the E4M3/E5M2 HYBRID recipe on Hopper/Ada/Blackwell.

    **Go deeper**

    - [PyTorch AMP Tutorial](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — official step-by-step guide to `torch.autocast` and `GradScaler` with timing benchmarks.
    - [NVIDIA Developer Blog, *Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training*](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) — accessible explainer of E4M3/E5M2, scaling strategies, and throughput gains on H100.

## Further reading

- Micikevicius et al., *Mixed Precision Training* (2017) — the original fp16 + loss-scaling + master-weights recipe.
- Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* (2019) — why bf16's range removes the need for loss scaling.
- Micikevicius et al., *FP8 Formats for Deep Learning* (2022) — the E4M3 / E5M2 definitions and rationale.
- NVIDIA, *Transformer Engine* documentation and repository — delayed scaling, `fp8_autocast`, and the HYBRID recipe.
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024) — fine-grained (tile/block) FP8 scaling at 671B scale, with the components kept in higher precision.
- PyTorch AMP documentation (`torch.autocast`, `torch.amp.GradScaler`) — the canonical reference for the loop in this chapter.
- Wang et al., *Training Deep Neural Networks with 8-bit Floating Point Numbers* (2018) — early FP8 training with chunk-based accumulation and stochastic rounding.
