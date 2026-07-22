# 4.7 Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)

A 70-billion-parameter model in 16-bit floating point weighs 140 GB. That does not fit on a single 80 GB A100 or H100, and even if it did, you would have nothing left for the KV cache or activations. Now store the same weights in 4 bits: 35 GB. Suddenly the model fits on one GPU, the weight-loading traffic from HBM (high-bandwidth memory) drops by 4x, and — because LLM *decoding* is memory-bandwidth bound, not compute bound (see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)) — your tokens-per-second can nearly quadruple. The price you pay is a tiny, often imperceptible, degradation in quality.

This is the promise of **quantization**: represent weights (and sometimes activations) with fewer bits per number. This chapter covers *post-training quantization* (PTQ) — methods that take an already-trained FP16 model and squeeze it down **without any gradient-based retraining**, using at most a few hundred calibration examples and a few GPU-minutes. We build the arithmetic from scratch, confront the central villain of LLM quantization — **activation outliers** — and then dissect the three algorithms every LLM engineer must know: **GPTQ**, **AWQ**, and **SmoothQuant**, plus the mixed-precision trick **LLM.int8()**. The companion chapter [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html) covers number formats, file formats, and quantization-*aware* training.

## The Arithmetic of Quantization

Quantization maps a continuous (or high-precision) value $x$ to one of a small finite set of integers, then maps it back. The forward map is **quantize**, the inverse is **dequantize**. For a $b$-bit integer grid we have $2^b$ levels.

### Affine (asymmetric) quantization

The general **affine** scheme uses two parameters: a positive **scale** $s$ (a floating-point number, the size of one quantization step) and an integer **zero-point** $z$ (which integer code maps to the real value 0).

{{fig:quant-mapping}}

To quantize a real value $x$:

$$
q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right) + z,\; q_{\min},\; q_{\max}\right)
$$

and to dequantize back to an approximate real value $\hat{x}$:

$$
\hat{x} = s \cdot (q - z).
$$

For unsigned $b$-bit codes, $q_{\min}=0$ and $q_{\max}=2^b-1$ (so $0\ldots15$ for INT4, $0\ldots255$ for INT8). Given the observed range $[\,x_{\min}, x_{\max}\,]$ of the tensor, we choose

$$
s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}}, \qquad z = \operatorname{round}\!\left(q_{\min} - \frac{x_{\min}}{s}\right).
$$

The zero-point lets an asymmetric range like $[-0.2, 1.8]$ use the *full* integer grid. This matters for **activations** after ReLU/GELU, which are skewed and non-negative-ish.

### Symmetric quantization

If the distribution is roughly symmetric about zero — which is true of most **weight** matrices — we drop the zero-point ($z=0$) and use a signed grid. For INT4 that is $[-8, 7]$; for INT8, $[-128, 127]$. The scale is set from the maximum absolute value:

$$
s = \frac{\max_i |x_i|}{2^{b-1}-1}, \qquad q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right),\, -(2^{b-1}-1),\, 2^{b-1}-1\right).
$$

Symmetric quantization is cheaper at inference time: dequantization is a single multiply $\hat{x}=s\cdot q$ with no add, and the integer matmul does not need a zero-point correction term. **Almost all modern LLM weight-quantization (GPTQ, AWQ) uses symmetric or near-symmetric per-group schemes.**

Here is the whole idea in runnable code:

```python
import torch

def quantize_symmetric(x: torch.Tensor, n_bits: int = 4):
    """Symmetric per-tensor quantization. Returns (codes, scale)."""
    qmax = 2 ** (n_bits - 1) - 1          # e.g. 7 for INT4, 127 for INT8
    qmin = -qmax                           # symmetric grid, drop the -8 slot for simplicity
    s = x.abs().max() / qmax               # one scalar scale for the whole tensor
    s = s.clamp(min=1e-8)                  # avoid divide-by-zero on all-zero tensors
    q = torch.clamp(torch.round(x / s), qmin, qmax)
    return q.to(torch.int8), s

def dequantize_symmetric(q: torch.Tensor, s: torch.Tensor):
    return q.to(torch.float32) * s

def quantize_affine(x: torch.Tensor, n_bits: int = 4):
    """Asymmetric (affine) quantization with a zero-point. Good for activations."""
    qmin, qmax = 0, 2 ** n_bits - 1        # unsigned grid: 0..15 for INT4
    xmin, xmax = x.min(), x.max()
    s = (xmax - xmin) / (qmax - qmin)
    s = s.clamp(min=1e-8)
    z = torch.round(qmin - xmin / s)        # the integer code that represents real 0.0
    q = torch.clamp(torch.round(x / s) + z, qmin, qmax)
    return q.to(torch.uint8), s, z

def dequantize_affine(q, s, z):
    return (q.to(torch.float32) - z) * s

# --- sanity check ---
w = torch.randn(4096) * 0.05               # a slice of a typical weight column
q, s = quantize_symmetric(w, n_bits=4)
err = (dequantize_symmetric(q, s) - w).abs().mean()
print(f"INT4 mean abs error: {err:.4e}  (scale={s:.4e})")
```

### Why rounding is the enemy

The error introduced by `round` is the **quantization error**. For a uniform grid of step $s$, if the rounding error were uniformly distributed on $[-s/2, s/2]$, its variance would be $s^2/12$ — the classic quantization-noise formula. The take-away: **error is proportional to the step size $s$, which is proportional to the *range* of the tensor.** One giant value in the tensor inflates $\max|x|$, inflates $s$, and corrupts the precision of *every other* value. This single observation drives the entire chapter.

## Granularity: Per-Tensor, Per-Channel, Per-Group

We do not have to use one scale for an entire matrix. Finer **granularity** = more scales = better fidelity (each scale is fit to a smaller, more homogeneous slice) at the cost of a little more memory and bookkeeping.

{{fig:quant-ptq-granularity}}

- **Per-tensor**: a single $(s, z)$ for the matrix. Fastest kernels, but one outlier ruins everything. Common for *activations* where you want a cheap dynamic scale.
- **Per-channel** (per-row for weights): one scale per output channel. The matmul $y = Wx$ produces output $y_j = \sum_i W_{ji} x_i$; a per-row scale $s_j$ factors out cleanly because the whole row shares it: $y_j \approx s_j \sum_i q^{W}_{ji} x_i$.
- **Per-group**: split each row into contiguous groups of $G$ weights (typically $G=128$) and give each group its own scale. This is the dominant scheme for 4-bit LLM weights (GPTQ, AWQ, GGUF all default to group size 128). It costs $\frac{16}{G} = \frac{16}{128} = 0.125$ extra bits per weight to store the FP16 scales — negligible.

```python
def quantize_per_group(W: torch.Tensor, n_bits=4, group_size=128):
    """Per-group symmetric quantization of a weight matrix [out, in]."""
    out_f, in_f = W.shape
    assert in_f % group_size == 0
    Wg = W.reshape(out_f, in_f // group_size, group_size)   # [out, n_groups, G]
    qmax = 2 ** (n_bits - 1) - 1
    s = Wg.abs().amax(dim=-1, keepdim=True) / qmax           # one scale per (row, group)
    s = s.clamp(min=1e-8)
    q = torch.clamp(torch.round(Wg / s), -qmax, qmax)
    return q.to(torch.int8), s, (out_f, in_f, group_size)

def dequantize_per_group(q, s, shape):
    out_f, in_f, G = shape
    W = (q.to(torch.float32) * s).reshape(out_f, in_f)
    return W
```

!!! tip "Practitioner tip"

    Why is **per-channel quantization of activations along the *token* dimension** awkward? Because the number of tokens (the sequence length) changes every forward pass and the scale would have to be recomputed dynamically. The clean axis to quantize activations along is the *channel/feature* dimension (which is fixed), and the clean axis for weights is the *output-channel* dimension. SmoothQuant, below, is essentially a trick to make the activation problem fit a per-channel-of-weights framing.

## The Outlier Problem

Here is the fact that breaks naive quantization of large transformers. As models scale past roughly 6–7B parameters, a small number of **feature dimensions** in the activations develop *enormous* magnitudes — often 10x to 100x larger than the typical activation. These are **activation outliers**. They are not noise: they are systematic, they live in the same handful of channels across many tokens, and ablating them destroys the model's accuracy. This phenomenon was documented carefully in the **LLM.int8()** paper (Dettmers et al., 2022).

Why are outliers fatal? Recall that per-tensor quantization error scales with $\max|x|$. If one channel is 100x larger than the rest, the per-tensor scale $s$ becomes 100x too coarse for the other 99% of values, and they all collapse toward zero. Consider quantizing the vector $[0.1, -0.2, 0.15, 70.0]$ to INT8 per-tensor:

$$
s = \frac{70.0}{127} \approx 0.551, \quad \text{so } 0.1 \mapsto \operatorname{round}(0.18) = 0,\;\; -0.2 \mapsto 0,\;\; 0.15 \mapsto 0.
$$

The three normal values all round to **zero**. The single outlier ate the entire dynamic range.

{{fig:quant-ptq-activation-outliers}}

Crucially, the outliers are predominantly in **activations**, not weights. Weight distributions are well-behaved and quantize beautifully to 4 bits with per-group scales. **Activations** are the hard part. This asymmetry explains the design space:

- **Weight-only quantization** (GPTQ, AWQ): keep activations in FP16, quantize only weights (to INT4/INT3). The matmul dequantizes weights on the fly. Great for *memory-bound decoding*, which is where LLM inference spends most of its time.
- **Weight + activation quantization** (SmoothQuant, LLM.int8()): quantize both so the matmul itself runs in INT8 on tensor cores. Needed for *compute-bound prefill* and maximum throughput, but you must tame the activation outliers first.

!!! note "Aside"

    Why do outliers emerge? A leading hypothesis is that attention needs a "no-op" — a way to attend to nothing — and the network learns to push certain residual-stream channels to extreme values to drive specific softmax/normalization behaviors. Whatever the cause, by the time you receive a pretrained checkpoint, the outliers are baked in and PTQ must cope with them.

## Weight-Only Quantization I: GPTQ

The simplest weight quantization is **round-to-nearest (RTN)**: independently round each weight to its grid. RTN is shockingly good for INT8 and decent for INT4 with group size 128, but it leaves accuracy on the table because it ignores the *interactions* between weights. **GPTQ** (Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022) does better by asking a sharper question.

### The objective: minimize the *output* error, not the weight error

We do not actually care whether each quantized weight is close to the original. We care that the **layer's output** is close. For a linear layer with weight row $\mathbf{w}$ and a batch of calibration inputs $X$ (shape $[\,d_{\text{in}}, N\,]$ of $N$ tokens), we want a quantized $\hat{\mathbf{w}}$ minimizing the reconstruction error:

$$
\hat{\mathbf{w}} = \arg\min_{\hat{\mathbf{w}} \in \text{grid}} \; \big\| \mathbf{w} X - \hat{\mathbf{w}} X \big\|_2^2.
$$

This is the **Optimal Brain Quantization** formulation, descended from the classic *Optimal Brain Surgeon* pruning work. The key object is the **Hessian** of this least-squares loss:

$$
H = 2\, X X^\top \quad (\text{shape } d_{\text{in}} \times d_{\text{in}}).
$$

$H$ captures the correlations between input features. The crucial insight: when we quantize one weight and incur an error, we can **compensate** by nudging the *not-yet-quantized* weights to absorb that error, and the optimal nudge is dictated by $H^{-1}$.

### The GPTQ algorithm

OBS/OBQ says: quantize weights one at a time; after fixing weight $i$ to its grid value, update all remaining weights to compensate. The optimal update when quantizing column $i$ is

$$
\delta_{\text{remaining}} = -\frac{w_i - q_i}{[H^{-1}]_{ii}} \cdot H^{-1}_{:,i},
$$

where $w_i - q_i$ is the rounding error you just committed. GPTQ makes this practical at LLM scale with three engineering moves:

1. **Same quantization order for all rows.** OBQ picks a different greedy order per row; GPTQ proves that quantizing columns in a *fixed* left-to-right order is nearly as good, which lets all rows share one $H^{-1}$ and turns the algorithm into batched matrix operations.
2. **Cholesky reformulation.** Repeated $H^{-1}$ updates are numerically unstable. GPTQ pre-computes a Cholesky factorization of $H^{-1}$ once and reads off the needed columns, which is both stable and fast.
3. **Lazy block updates.** Update remaining weights in blocks of ~128 columns to keep memory traffic cache-friendly.

A faithful-but-readable core of GPTQ:

```python
import torch

def gptq_quantize_layer(W, H, n_bits=4, group_size=128, percdamp=0.01):
    """
    GPTQ for one linear layer.
      W : [out_features, in_features]  original weights (float)
      H : [in_features, in_features]   Hessian  = X X^T  over calibration tokens
    Returns the quantized-then-dequantized weight (ready to use in FP matmul,
    or to be packed into INT4). This mirrors the real algorithm's structure.
    """
    W = W.clone().float()
    out_f, in_f = W.shape
    qmax = 2 ** (n_bits - 1) - 1

    # 1) Dampen the Hessian diagonal for invertibility (some inputs are dead).
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H))
    H[range(in_f), range(in_f)] += damp

    # 2) Cholesky of the INVERSE Hessian -> upper triangular factor we walk along.
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)   # upper-tri Cholesky of H^{-1}

    Q = torch.zeros_like(W)
    Err = torch.zeros_like(W)

    # 3) Walk columns left-to-right; quantize, then propagate the error forward.
    for i in range(in_f):
        w = W[:, i].clone()                # current column across all output rows
        d = Hinv[i, i]                     # diagonal entry => error-scaling factor

        # per-group symmetric scale (recomputed at each group boundary)
        if i % group_size == 0:
            g = W[:, i:i + group_size]
            scale = g.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax

        q = torch.clamp(torch.round(w / scale.squeeze(1)),
                        -qmax, qmax) * scale.squeeze(1)
        Q[:, i] = q

        err = (w - q) / d                  # the OBS-optimal scaled error
        # compensate every not-yet-quantized column j>i using H^{-1} row i.
        W[:, i:] -= err.unsqueeze(1) * Hinv[i, i:].unsqueeze(0)
        Err[:, i] = err

    return Q
```

The Hessian itself is accumulated during a forward pass over a calibration set (e.g. 128 sequences of 2048 tokens from C4 or WikiText):

```python
def accumulate_hessian(layer, calibration_inputs):
    """H = sum over tokens of x x^T, normalized. x is the layer's input."""
    in_f = layer.weight.shape[1]
    H = torch.zeros(in_f, in_f)
    n = 0
    for X in calibration_inputs:           # X: [n_tokens, in_f]
        X = X.float()
        H += X.t() @ X                     # accumulate outer products
        n += X.shape[0]
    return (2.0 / n) * H
```

GPTQ runs in **minutes to a couple of hours** for a 7B–70B model on a single GPU and reliably hits INT4 (and often INT3) with small perplexity loss. Its weakness: it is sequential per layer and Hessian-heavy, and it can struggle on the very narrowest bit-widths without a group size.

{{fig:gptq-error-compensation}}

## Weight-Only Quantization II: AWQ

**AWQ** — Activation-aware Weight Quantization (Lin et al., 2023) — starts from a different observation: **not all weights are equally important, and the important ones are revealed by the activations.** If a weight column multiplies an input channel that frequently carries large activations, errors in that column get amplified into the output. AWQ identifies these **salient weight channels** (typically the ~1% aligned with activation outliers) and protects them.

### The scaling trick

The naive way to protect salient weights is mixed precision (keep 1% in FP16), but mixed-precision INT4/FP16 matmuls are kernel-unfriendly. AWQ's elegant alternative: **keep everything INT4, but rescale.** Consider a single weight $w$ multiplied by input $x$: $y = w \cdot x$. We can scale the weight up by $\alpha > 1$ and the input down by the same factor without changing the product:

$$
y = w \cdot x = (w \cdot \alpha)\cdot\!\left(\frac{x}{\alpha}\right).
$$

Quantizing $w\cdot\alpha$ instead of $w$ shrinks its **relative** rounding error by roughly $\alpha$, because the error is $s/2$ in absolute terms but the value is now $\alpha$ times bigger. The cost is that the input $x/\alpha$ must be rescaled — but $x$ stays in FP16 (weight-only), so we can fold $1/\alpha$ into the previous layer's normalization weights *for free*. AWQ searches for a **per-input-channel** scale vector $\mathbf{s}$ that minimizes the layer output error, using a simple grid search over a hyperparameter that interpolates between "scale by activation magnitude" and "scale by weight magnitude":

$$
\mathbf{s} = \big(\,\text{mean}_t |x|_{:,c}\,\big)^{\beta}, \qquad \beta \in [0, 1] \text{ chosen by grid search}.
$$

```python
import torch

def awq_search_scales(W, X, n_bits=4, group_size=128, grid=20):
    """
    AWQ per-input-channel scale search for one linear layer.
      W : [out, in]   weights
      X : [n_tokens, in]   calibration activations into this layer
    Returns scale vector s of shape [in]; you then quantize (W * s) and fold
    (1/s) into the upstream layernorm / previous linear.
    """
    W = W.float()
    x_mag = X.abs().float().mean(dim=0)          # [in]  per-channel activation scale
    w_mag = W.abs().mean(dim=0).clamp(min=1e-6)  # [in]  per-channel weight scale

    def quant_dequant(w):                         # group-wise INT4 RTN
        out, inf = w.shape
        wg = w.reshape(out, inf // group_size, group_size)
        qmax = 2 ** (n_bits - 1) - 1
        s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(wg / s), -qmax, qmax) * s
        return q.reshape(out, inf)

    ref = X.float() @ W.t()                       # FP16 reference output
    best_err, best_s = float("inf"), torch.ones_like(x_mag)
    for i in range(grid):
        beta = i / grid                           # interpolate importance exponent
        s = (x_mag ** beta) / (w_mag ** (1 - beta))
        s = s.clamp(min=1e-4)
        s = s / s.max()                           # normalize to avoid blowup
        Wq = quant_dequant(W * s)                 # quantize the *scaled* weights
        out = (X.float() / s) @ Wq.t()            # input rescaled by 1/s
        err = (out - ref).pow(2).mean().item()
        if err < best_err:
            best_err, best_s = err, s
    return best_s
```

AWQ is **faster than GPTQ** (no Hessian inverse, no sequential per-column loop — just a grid search over a scale vector), has **no overfitting to the calibration set** (it never solves a least-squares problem against specific tokens, only a magnitude statistic), and tends to **generalize better across domains**. In practice GPTQ and AWQ are both excellent at INT4-g128; AWQ is often the default in `vllm`/`sglang` serving stacks because its kernels are simple and it is robust.

!!! warning "Common pitfall"

    Both GPTQ and AWQ are **calibration-data sensitive in different ways**. GPTQ fits a least-squares problem to your calibration tokens — if you calibrate a code model on Wikipedia text, you can damage coding ability. AWQ only uses per-channel *magnitude* statistics, so it is more forgiving, but still: calibrate on data that resembles your deployment distribution, use at least ~128 sequences, and always re-measure perplexity *and* a downstream eval, not just perplexity. Perplexity can look fine while a benchmark like GSM8K silently drops.

## Activation Quantization: SmoothQuant & LLM.int8()

Weight-only quantization halves or quarters memory but the matmul still runs in FP16. To use **INT8 tensor cores** (2x–4x the FLOPs of FP16 on many GPUs) we must quantize activations too — and that means facing the outliers head-on. Two complementary answers:

### SmoothQuant: migrate the difficulty

**SmoothQuant** (Xiao et al., 2022) makes a beautiful observation: activations have outliers and are hard to quantize; weights are smooth and easy. So **move some of the difficulty from activations into weights** using the same scale-invariance AWQ exploits. For $y = X W$, insert a per-channel diagonal scale $\operatorname{diag}(\mathbf{s})$:

$$
Y = (X \operatorname{diag}(\mathbf{s})^{-1})\,(\operatorname{diag}(\mathbf{s}) W) = \hat{X}\hat{W}.
$$

Choose $\mathbf{s}$ so the *smoothed* activation $\hat{X}$ has no outliers and the *smoothed* weight $\hat{W}$ is still quantizable. The per-channel scale that balances the two difficulties is:

$$
s_c = \frac{\big(\max_t |X_{t,c}|\big)^{\alpha}}{\big(\max_j |W_{c,j}|\big)^{1-\alpha}}, \qquad \alpha \in [0,1]\ (\text{often } 0.5).
$$

The **migration strength** $\alpha$ trades off: $\alpha=1$ pushes all difficulty into weights, $\alpha=0$ leaves it in activations, $\alpha=0.5$ splits it. Because $\operatorname{diag}(\mathbf{s})^{-1}$ multiplies $X$ and $X$ is the output of a LayerNorm/RMSNorm, the scale **folds into the norm's affine weights at zero runtime cost** — exactly like AWQ. After smoothing, *both* $\hat{X}$ and $\hat{W}$ quantize cleanly to INT8, and the matmul runs entirely on INT8 tensor cores with per-tensor or per-channel scales.

```python
import torch

@torch.no_grad()
def smoothquant_scales(act_abs_max, weight, alpha=0.5):
    """
    Compute the SmoothQuant per-channel migration scale.
      act_abs_max : [in]   per-channel max |activation| from calibration
      weight      : [out, in]
    Returns s of shape [in]: divide activations by s, multiply weight cols by s.
    """
    w_abs_max = weight.abs().amax(dim=0).clamp(min=1e-5)   # [in]
    a_abs_max = act_abs_max.clamp(min=1e-5)
    s = (a_abs_max.pow(alpha) / w_abs_max.pow(1 - alpha)).clamp(min=1e-5)
    return s

@torch.no_grad()
def apply_smoothing(weight, ln_weight, s):
    """Fold 1/s into the upstream norm, s into this linear's weight columns."""
    ln_weight = ln_weight / s          # activations get divided by s for free
    weight = weight * s.unsqueeze(0)   # weight columns multiplied by s
    return weight, ln_weight
```

{{fig:smoothquant-migration}}

### LLM.int8(): keep the outliers in FP16

**LLM.int8()** (Dettmers et al., 2022) takes the bluntest possible approach and it works: **decompose the matmul.** Detect the outlier feature dimensions at runtime (channels whose magnitude exceeds a threshold, e.g. 6.0). Run the ~99.9% of "normal" dimensions through a fast **INT8 matmul**, and run the handful of outlier columns through a small **FP16 matmul**. Sum the two. Mathematically, splitting the input feature set into outlier indices $O$ and the rest $R$:

{{fig:llm-int8-decomposition}}

$$
X W = X_{:,R}\,W_{R,:} + X_{:,O}\,W_{O,:},
$$

where the first term is INT8 and the second is FP16. Because $|O|$ is tiny (often a few dozen channels out of thousands), the FP16 part is cheap, and the INT8 part captures the bulk of the FLOPs with no accuracy loss on the well-behaved channels.

```python
import torch

@torch.no_grad()
def llm_int8_matmul(X, W, threshold=6.0):
    """
    Mixed INT8 / FP16 matmul, the LLM.int8() decomposition.
      X : [tokens, in]   activations (fp16)
      W : [in, out]      weights (fp16)
    """
    # 1) find outlier feature columns by magnitude across the batch
    col_max = X.abs().amax(dim=0)                # [in]
    outlier = col_max > threshold               # boolean mask
    reg = ~outlier

    # 2) FP16 path for the few outlier dimensions (exact, no quant error)
    y_fp16 = X[:, outlier] @ W[outlier, :]

    # 3) INT8 path for the regular dimensions
    Xr, Wr = X[:, reg], W[reg, :]
    sx = Xr.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127   # per-row act scale
    sw = Wr.abs().amax(dim=0, keepdim=True).clamp(min=1e-8) / 127   # per-col wgt scale
    Xq = torch.clamp(torch.round(Xr / sx), -127, 127).to(torch.int8)
    Wq = torch.clamp(torch.round(Wr / sw), -127, 127).to(torch.int8)
    acc = (Xq.float() @ Wq.float())             # stands in for an INT8 tensor-core matmul
    y_int8 = acc * sx * sw                       # dequantize with outer-product of scales

    return y_int8 + y_fp16
```

LLM.int8() is *zero-tuning* — no calibration, no Hessian, no scale search — and is the default behind `load_in_8bit=True` in `bitsandbytes`. Its downside is **latency**: the dynamic outlier detection and the split matmul add overhead, so it is better for fitting a model in memory than for maximum throughput. SmoothQuant, by contrast, is a static transform that yields a clean INT8-everywhere model with fast kernels.

!!! example "Worked example: quantizing a 13B model to INT4"

    Take a 13B-parameter decoder, FP16 weights.

    **Memory.** FP16: $13\times10^{9}\times 2\,\text{B} = 26\ \text{GB}$ just for weights — already tight on a 24 GB consumer GPU (RTX 4090). INT4-g128: each weight is 4 bits = $13\times10^{9}\times 0.5\,\text{B} = 6.5\ \text{GB}$, **plus** the per-group scales. With group size 128, there is one FP16 scale per 128 weights: $\frac{13\times10^9}{128}\times 2\,\text{B} \approx 0.20\ \text{GB}$. Total ≈ **6.7 GB**. A 4x reduction; the model now fits with plenty of room for KV cache.

    **Decode speedup.** Decoding one token requires streaming *all* weights from HBM once (it is bandwidth-bound — see [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html)). Moving 6.7 GB instead of 26 GB at, say, 2 TB/s of HBM bandwidth: $26/2000 = 13\ \text{ms}$ vs $6.7/2000 \approx 3.4\ \text{ms}$ of weight-load time per token. Close to 4x fewer milliseconds spent moving weights — the headline reason INT4 weight-only quantization speeds up *decoding*.

    **Quantization error budget.** A typical weight column has values around $|w|\sim 0.02$. Per-group INT4 with $\max|w|\approx0.08$ gives step $s = 0.08/7 \approx 0.0114$. Worst-case rounding error per weight is $s/2 \approx 0.0057$ — about 7% of a typical weight, but **uncorrelated across the 5120 weights in a row**, so the error on the *output* sum shrinks like $1/\sqrt{5120}\approx 1.4\%$. That averaging is why INT4 weight-only quantization barely moves perplexity, and why GPTQ/AWQ — which actively *correlate* and *protect* against the worst errors — close most of the remaining gap.

## Putting It Together: A Decision Guide

{{fig:quant-ptq-decision-guide}}

The shorthand **W4A16** means 4-bit weights, 16-bit activations (GPTQ/AWQ territory); **W8A8** means 8-bit weights and activations (SmoothQuant territory). A practical pipeline:

1. Pick a target: W4A16 for single-GPU decoding, W8A8 for throughput serving.
2. Gather ~128 calibration sequences from in-domain data.
3. Run GPTQ or AWQ (weight-only) or SmoothQuant+per-tensor INT8 (W8A8).
4. **Evaluate**: perplexity on held-out text *and* at least one task eval. A 0.1–0.3 perplexity rise is normal and usually invisible downstream; a >1.0 rise or a benchmark cliff means revisit group size, calibration data, or keep more layers in higher precision (the first and last layers, and `lm_head`, are often kept in FP16).

For the actual file formats, packing, and how these integrate with `bitsandbytes`, GGUF, and quantization-aware training, continue to [Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html). For how quantized models slot into serving systems, see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html); the floating-point background lives in [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html), and quantization for *training* in [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

!!! interview "Interview Corner"

    **Q:** You quantize a 7B model to INT4 and decoding gets ~3.5x faster, but you also quantize the *activations* to INT8 and prefill barely speeds up while accuracy drops. Explain both observations and how you'd fix the accuracy.

    **A:** Decoding is **memory-bandwidth bound**: each token streams the full weight matrix from HBM once, so cutting weights 16→4 bits cuts the dominant cost ~4x — hence the speedup, and weight-only INT4 introduces little error because rounding noise averages out over thousands of summed terms. Prefill is **compute-bound** (large matmuls over many tokens), so to speed it up you need faster *math*, i.e. INT8 tensor cores, which requires quantizing activations too. Activations contain **systematic outlier channels** (10–100x the typical magnitude) that blow up a per-tensor INT8 scale and crush the normal values to zero, so naive activation quantization tanks accuracy. The fix is **SmoothQuant**: migrate per-channel difficulty from activations into weights via a diagonal scale $\operatorname{diag}(\mathbf{s})$, folding $1/\mathbf{s}$ into the upstream RMSNorm at zero cost, so both tensors become quantizable. Alternatively **LLM.int8()** keeps the outlier dimensions in FP16 and runs the rest in INT8, trading some kernel overhead for zero tuning.

!!! interview "Interview Corner"

    **Q:** Why does GPTQ need a Hessian but AWQ does not, and when would you prefer one over the other?

    **A:** GPTQ solves a per-layer least-squares problem — minimize $\|WX - \hat{W}X\|^2$ — whose curvature is the Hessian $H = 2XX^\top$. The Hessian tells GPTQ how to **compensate** for each rounding error by adjusting the not-yet-quantized weights ($\delta \propto H^{-1}$), which is what makes it accurate at low bit-widths. AWQ never solves that optimization; it only computes per-channel **activation magnitude** statistics to find salient weight channels and protects them via a scale, then does plain round-to-nearest. So AWQ is cheaper, has no $H^{-1}$ stability issues, and overfits the calibration set less — prefer it for robustness and speed, especially across mixed domains. Prefer GPTQ when you need to push to INT3 or want the last bit of accuracy and can afford the Hessian compute; in practice both are excellent at INT4-g128 and the choice often comes down to which has better kernels in your serving stack.

!!! key "Key Takeaways"

    - **Quantization maps floats to a small integer grid** via a scale $s$ and (optionally) a zero-point $z$; error is proportional to $s$, which is proportional to the tensor's dynamic range — so outliers are poison.
    - **Symmetric per-group (G=128) is the LLM weight default**: cheap dequant, full-grid usage, ~0.125 extra bits/weight for scales. Asymmetric (with zero-point) suits skewed activations.
    - **The central problem is activation outliers**: a few channels run 10–100x larger past ~6B params; they wreck per-tensor activation scales but live mostly in activations, not weights — which is why weight-only INT4 is nearly free.
    - **GPTQ** minimizes per-layer output error using the input Hessian $H=2XX^\top$, quantizing column-by-column and compensating remaining weights via $H^{-1}$ (Cholesky, fixed order, block updates).
    - **AWQ** protects salient weight channels (revealed by activation magnitude) with a per-channel scale, no Hessian — faster, more robust to calibration domain.
    - **SmoothQuant** migrates per-channel difficulty from activations into weights via $\operatorname{diag}(\mathbf{s})$ folded into the upstream norm, enabling clean **W8A8** INT8-tensor-core matmuls.
    - **LLM.int8()** splits the matmul: outlier dims in FP16, the rest in INT8 — zero tuning, default for `load_in_8bit`, but with kernel overhead.
    - **W4A16 for memory-bound decoding, W8A8 for compute-bound prefill/throughput.** Always evaluate perplexity *and* a downstream task; keep `lm_head` and edge layers high-precision if accuracy slips.

!!! sota "State of the Art & Resources (2026)"
    Post-training quantization for LLMs is now a mature, production-grade discipline: INT4 weight-only (GPTQ/AWQ) and INT8 weight+activation (SmoothQuant) are standard in every major serving stack, and 2024–2025 research has pushed further into sub-4-bit regimes with incoherence-based and vector-quantization methods.

    **Foundational work**

    - [Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022)](https://arxiv.org/abs/2210.17323) — Hessian-compensated column-wise rounding that made INT4 LLMs practical; ICLR 2023.
    - [Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022)](https://arxiv.org/abs/2208.07339) — Documents activation outliers and introduces the mixed INT8/FP16 decomposition; NeurIPS 2022.
    - [Nagel et al., *A White Paper on Neural Network Quantization* (2021)](https://arxiv.org/abs/2106.08295) — Rigorous primer on scales, zero-points, granularity, PTQ vs QAT from Qualcomm AI Research.

    **Recent advances (2023–2026)**

    - [Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023)](https://arxiv.org/abs/2306.00978) — Salience-guided per-channel scaling with no Hessian; MLSys 2024 Best Paper.
    - [Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (2022)](https://arxiv.org/abs/2211.10438) — Migrates activation difficulty into weights via a diagonal scale, enabling W8A8 on tensor cores; ICML 2023.
    - [Tseng et al., *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks* (2024)](https://arxiv.org/abs/2402.04396) — Pushes sub-4-bit quality with randomized Hadamard incoherence and E₈ lattice codebooks; ICML 2024.

    **Open-source & tools**

    - [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) — Original GPTQ reference implementation (ICLR 2023).
    - [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) — Official AWQ repo with CUDA kernels, model zoo, and TinyChat edge inference.
    - [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant) — Official SmoothQuant implementation enabling W8A8 for OPT, LLaMA, Mistral, and more.
    - [bitsandbytes-foundation/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — The `load_in_8bit` / `load_in_4bit` library powering LLM.int8() and QLoRA in Hugging Face Transformers.
    - [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) — Actively maintained successor to AutoGPTQ, supporting GPTQ, AWQ, FP8, GGUF and multi-backend inference.

    **Go deeper**

    - [Overview of natively supported quantization schemes in Transformers](https://huggingface.co/blog/overview-quantization-transformers) — Hugging Face blog comparing bitsandbytes vs GPTQ on speed, memory, and fine-tuning.

## Further reading

- Frantar, Ashkboos, Hoefler, Alistarh — *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022). The Hessian-based, Cholesky-stabilized method; see also the `gptq` and `AutoGPTQ` repositories.
- Lin, Tang, Tang, et al. — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023). The salience/scaling idea; the `llm-awq` repository.
- Xiao, Lin, Seznec, Wu, Demouth, Han — *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (2022). Migrating difficulty for W8A8.
- Dettmers, Lewis, Belkada, Zettlemoyer — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022). Outlier features and mixed-precision decomposition; implemented in `bitsandbytes`.
- Frantar & Alistarh — *Optimal Brain Compression* / *Optimal Brain Quantization* (the OBS/OBQ lineage GPTQ builds on).
- Nagel et al. — *A White Paper on Neural Network Quantization* (Qualcomm AI Research). An excellent, rigorous primer on scales, zero-points, granularity, and PTQ vs QAT.

## Exercises

**1.** *(Quantitative — by hand.)* Quantize the weight vector $\mathbf{w} = [\,0.02,\ -0.05,\ 0.08,\ -0.03\,]$ to **symmetric INT4** using the chapter's convention $q_{\max}=2^{b-1}-1=7$, $q_{\min}=-7$. Compute the scale $s$, the integer codes, the dequantized values $\hat{w}$, and the mean absolute error. Which value is quantized *exactly*, and why?

??? note "Solution"
    The symmetric scale is set from the maximum absolute value:

    $$
    s = \frac{\max_i |w_i|}{q_{\max}} = \frac{0.08}{7} \approx 0.0114286.
    $$

    Codes are $q_i = \operatorname{clip}(\operatorname{round}(w_i/s),\,-7,\,7)$:

    - $0.02/s = 1.750 \Rightarrow \operatorname{round} = 2$
    - $-0.05/s = -4.375 \Rightarrow \operatorname{round} = -4$
    - $0.08/s = 7.000 \Rightarrow \operatorname{round} = 7$
    - $-0.03/s = -2.625 \Rightarrow \operatorname{round} = -3$

    So $\mathbf{q} = [\,2,\,-4,\,7,\,-3\,]$. Dequantize with $\hat{w}_i = s\,q_i$:

    - $2s \approx 0.022857$
    - $-4s \approx -0.045714$
    - $7s = 0.080000$
    - $-3s \approx -0.034286$

    Absolute errors $|\hat{w}_i - w_i|$: $0.002857,\ 0.004286,\ 0.000000,\ 0.004286$. The mean absolute error is

    $$
    \frac{0.002857 + 0.004286 + 0 + 0.004286}{4} \approx 2.86\times10^{-3}.
    $$

    The value $0.08$ is quantized **exactly**: it is the tensor's max-absolute value, and symmetric quantization defines $s$ precisely so that $\max|w|$ lands on the top code $q_{\max}=7$ with zero rounding error ($0.08/s = 7.000$). Every other value carries error up to $s/2 \approx 5.7\times10^{-3}$.

**2.** *(Conceptual + small calc.)* The chapter shows that quantizing $[\,0.1,\,-0.2,\,0.15,\,70.0\,]$ per-tensor to INT8 collapses the three small values to zero. (a) Explain in one or two sentences *why* a single outlier does this. (b) Now split the vector into two per-group groups of size 2, $[\,0.1,\,-0.2\,]$ and $[\,0.15,\,70.0\,]$, and quantize each group symmetrically to INT8 ($q_{\max}=127$). Show that the small values survive in the first group. (c) What does this tell you about why weights quantize well to INT4-g128 but per-tensor *activation* quantization does not?

??? note "Solution"
    **(a)** Quantization error is proportional to the step $s$, and for a symmetric per-tensor scale $s = \max|x|/q_{\max}$. One value 350x larger than the rest ($70.0$ vs $\sim 0.2$) inflates $\max|x|$, hence $s$, so the step becomes far coarser than the small values themselves — they round to the nearest grid point, which is $0$.

    **(b)** Group 1 $=[\,0.1,\,-0.2\,]$: $s_1 = 0.2/127 \approx 1.575\times10^{-3}$.

    - $0.1/s_1 \approx 63.5 \Rightarrow \operatorname{round}=64$, dequant $\approx 0.1008$
    - $-0.2/s_1 = -127 \Rightarrow -127$, dequant $=-0.2$

    Both survive with tiny error. Group 2 $=[\,0.15,\,70.0\,]$: $s_2 = 70.0/127 \approx 0.551$, so $0.15 \to 0$ still — but the damage is now **confined to the group that actually contains the outlier**.

    **(c)** Fine granularity quarantines an outlier's damage to its own group instead of letting it poison the entire tensor. Weight matrices are well-behaved (no 100x outliers), so per-group INT4 fits each 128-wide slice tightly and averages out. Activations *do* contain systematic outlier channels; per-tensor activation scales are exactly the failure mode in (a), which is why W4A16 (weights only) is nearly free while activation quantization needs SmoothQuant/LLM.int8() to first tame or isolate the outliers.

**3.** *(Quantitative.)* You have a 7B-parameter model in FP16. (a) Compute the weight memory in FP16. (b) Compute the weight memory after INT4 with group size 128, *including* the FP16 group scales. (c) What is the compression ratio? (d) If HBM bandwidth is 2 TB/s and decoding is bandwidth-bound (stream all weights once per token), estimate the per-token weight-load time before and after, and the speedup.

??? note "Solution"
    **(a)** FP16 = 2 bytes/weight: $7\times10^{9}\times 2\,\text{B} = 14\ \text{GB}$.

    **(b)** INT4 payload = 0.5 byte/weight: $7\times10^{9}\times 0.5\,\text{B} = 3.5\ \text{GB}$. Group scales: one FP16 scale per 128 weights, so $\frac{7\times10^{9}}{128}\times 2\,\text{B} \approx 1.09\times10^{8}\,\text{B} \approx 0.11\ \text{GB}$ (equivalently $16/128 = 0.125$ extra bits/weight). Total $\approx 3.5 + 0.11 = 3.61\ \text{GB}$.

    **(c)** Compression ratio $= 14 / 3.61 \approx 3.9\times$ (slightly under the ideal $4\times$ because of the scale overhead).

    **(d)** Time $=$ bytes $/$ bandwidth. FP16: $14\ \text{GB} / 2\ \text{TB/s} = 7.0\ \text{ms}$ per token. INT4: $3.61\ \text{GB} / 2\ \text{TB/s} \approx 1.8\ \text{ms}$ per token. Speedup $\approx 7.0/1.8 \approx 3.9\times$ — matching the compression ratio, because bandwidth-bound decode time is dominated by how many weight bytes you move.

**4.** *(Conceptual + calc — the AWQ scaling trick.)* A salient input channel carries a weight $w = 0.02$ that sits inside a group whose max-absolute weight is $0.08$ (dominated by *other* channels), so the group's INT4 step is $s = 0.08/7 \approx 0.0114$. (a) What is the worst-case *relative* rounding error on $w$? (b) AWQ multiplies this channel's weight by $\alpha = 3$ (and divides the corresponding activation by 3, folded upstream). Assuming the scaled weight $0.06$ is still below the group max $0.08$ so $s$ is unchanged, what is the new worst-case relative error? (c) State the general rule and the one condition under which it can break.

??? note "Solution"
    **(a)** Worst-case absolute rounding error is $s/2 \approx 0.00571$. Relative to $w = 0.02$: $0.00571/0.02 \approx 28.6\%$.

    **(b)** After scaling, the weight is $0.06$ and (by assumption) $s$ is unchanged, so the absolute error is still $s/2 \approx 0.00571$, but the value it sits on is 3x larger: $0.00571/0.06 \approx 9.5\%$. The relative error dropped by a factor of $\approx 3 = \alpha$.

    **(c)** General rule: multiplying a salient weight by $\alpha$ leaves the absolute step $s/2$ (roughly) unchanged while making the stored value $\alpha$x larger, so its **relative** rounding error shrinks by $\approx \alpha$; the compensating $1/\alpha$ on the activation is free because it folds into the upstream LayerNorm/RMSNorm. The condition: this holds only while the scaled weight does **not** become the group's new max-absolute value. If $w\cdot\alpha$ exceeds the group max, $s$ itself grows (proportionally to $\alpha$ in the extreme), and the relative-error gain evaporates. That is exactly why AWQ does not blindly scale up salient channels — it grid-searches a per-channel scale $\mathbf{s}=(\text{mean}_t|x|)^\beta$ that balances protecting salient channels against inflating the group scale.

**5.** *(Implementation.)* Implement SmoothQuant's per-channel migration and empirically verify its two defining properties on a synthetic layer with an injected activation outlier: (i) the transform is *mathematically exact* — $\hat{X}\hat{W}^\top = XW^\top$ — and (ii) it *reduces* the activation's max-absolute value. Use $\alpha = 0.5$.

??? note "Solution"
    ```python
    import torch
    torch.manual_seed(0)

    X = torch.randn(32, 8)            # [tokens, in]  activations
    X[:, 3] *= 40.0                   # inject one outlier channel
    W = torch.randn(16, 8) * 0.05     # [out, in]  weights

    @torch.no_grad()
    def smoothquant_scales(act_abs_max, weight, alpha=0.5):
        w_abs_max = weight.abs().amax(dim=0).clamp(min=1e-5)   # [in]
        a_abs_max = act_abs_max.clamp(min=1e-5)
        return (a_abs_max.pow(alpha) / w_abs_max.pow(1 - alpha)).clamp(min=1e-5)

    a_max = X.abs().amax(dim=0)              # [in]  per-channel act max
    s = smoothquant_scales(a_max, W, alpha=0.5)

    Xhat = X / s                             # activations divided by s
    What = W * s                             # weight columns multiplied by s

    # (i) exact invariance: Xhat @ What.t() == X @ W.t()
    err = (Xhat @ What.t() - X @ W.t()).abs().max().item()
    print(f"max reconstruction error: {err:.2e}")   # ~1e-6 (float noise)

    # (ii) outlier tamed
    print(f"activation max before: {X.abs().amax(0).max().item():8.3f}")
    print(f"activation max after : {Xhat.abs().amax(0).max().item():8.3f}")
    ```

    Why it works: writing $Y = XW^\top$ with a diagonal scale $\operatorname{diag}(\mathbf{s})$,

    $$
    \hat{X}\hat{W}^\top = (X\operatorname{diag}(\mathbf{s})^{-1})(W\operatorname{diag}(\mathbf{s}))^\top = X\operatorname{diag}(\mathbf{s})^{-1}\operatorname{diag}(\mathbf{s})W^\top = XW^\top,
    $$

    so property (i) holds up to floating-point noise ($\sim 10^{-6}$). For (ii), the outlier channel had a large $a_{\max}$, so its $s_c$ is large and dividing by it shrinks that channel's activation magnitude — the printed "after" max is far smaller than the "before" max ($\approx 40\times$ the typical value). The migrated difficulty lands in $\hat{W}$, which was smooth and has room to absorb it; and because $\operatorname{diag}(\mathbf{s})^{-1}$ multiplies the output of the upstream norm, it folds into that norm's affine weights at zero runtime cost.

**6.** *(Implementation — hard.)* Empirically demonstrate GPTQ's core claim: that Hessian-based error compensation yields *lower layer-output error* than plain round-to-nearest (RTN) when the calibration inputs are **correlated** (a non-diagonal Hessian). Build a synthetic layer with correlated inputs, quantize it both ways, and compare $\|WX^\top - \hat{W}X^\top\|_2^2$.

??? note "Solution"
    Correlation is what gives GPTQ something to exploit: $H = 2XX^\top$ is non-diagonal, so an error committed on column $i$ can be *partly cancelled* by nudging correlated columns $j>i$ via $H^{-1}$. With uncorrelated (white) inputs $H$ is nearly diagonal and GPTQ collapses toward RTN.

    ```python
    import torch
    torch.manual_seed(0)

    in_f, out_f, N = 256, 128, 512
    A = torch.randn(in_f, in_f)              # mixing matrix -> correlated features
    X = torch.randn(N, in_f) @ A.t()         # [tokens, in]  correlated calibration
    W = torch.randn(out_f, in_f) * 0.05      # [out, in]

    H = (2.0 / N) * (X.t() @ X)              # input Hessian, non-diagonal

    def rtn(W, n_bits=4, group_size=128):
        out, inf = W.shape
        wg = W.reshape(out, inf // group_size, group_size)
        qmax = 2 ** (n_bits - 1) - 1
        s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        return (torch.clamp(torch.round(wg / s), -qmax, qmax) * s).reshape(out, inf)

    # gptq_quantize_layer(W, H, ...) is the function defined earlier in this chapter.
    Wq_rtn  = rtn(W.clone())
    Wq_gptq = gptq_quantize_layer(W.clone(), H.clone(), n_bits=4, group_size=128)

    ref = W @ X.t()                          # FP16 reference output [out, tokens]
    e_rtn  = (Wq_rtn  @ X.t() - ref).pow(2).mean().item()
    e_gptq = (Wq_gptq @ X.t() - ref).pow(2).mean().item()
    print(f"RTN  output MSE: {e_rtn:.4e}")
    print(f"GPTQ output MSE: {e_gptq:.4e}")
    print(f"GPTQ is {e_rtn / e_gptq:.2f}x lower")
    ```

    Both methods place weights on the *same* INT4-g128 grid, so they incur nearly identical *weight*-space error. The difference is entirely in the objective: RTN minimizes per-weight error and ignores $X$, while GPTQ minimizes the layer *output* error $\|WX^\top-\hat WX^\top\|_2^2$. After quantizing column $i$ it propagates the scaled residual $(w_i-q_i)/[H^{-1}]_{ii}$ into the not-yet-quantized columns (the `W[:, i:] -= ...` step), cancelling part of the output error that RTN leaves on the table. With correlated inputs you should see GPTQ's output MSE clearly below RTN's. If you replace `X` with white noise (`X = torch.randn(N, in_f)`), $H$ becomes near-diagonal, the compensation term vanishes, and the two errors converge — confirming that GPTQ's advantage comes specifically from *input correlations* captured by the Hessian.
