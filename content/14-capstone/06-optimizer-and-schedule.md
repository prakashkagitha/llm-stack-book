# 14.6 Optimizer & Schedule: Muon + MuonClip and Warmup-Stable-Decay

Every choice we have made so far — the deep-and-thin `Stack-100M` architecture of [Chapter 14.4](../14-capstone/04-architecture.html), the ~20B-token over-training budget, the FineWeb-Edu/Cosmopedia data mix — only pays off if the optimizer can actually walk the loss down cleanly across **38,147 steps** without spiking, and if the learning-rate schedule leaves a hook for the mid-training annealing phase to come. This chapter fixes both, and it fixes them *numerically*: every constant here is the one that appears in `stacklm/config.py` and in the [pretraining run](../14-capstone/07-pretraining-run.html).

We adopt the optimizer stack the capstone spec pins in §5: **Muon** (Jordan et al., 2024) for the 2D hidden weight matrices, **AdamW** for embeddings, norms, and every 1D parameter, a **QK-clip** in the spirit of **MuonClip** (Kimi K2, Moonshot 2025) for attention-logit stability, and the **Warmup-Stable-Decay (WSD)** schedule (MiniCPM, Hu et al., 2024; used by DeepSeek).

This is the applied, capstone-specific companion to two deeper reference chapters you should have open in another tab: [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) derives Adam and Muon from first principles, and [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html) covers warmup, cosine, batch/LR coupling, and muP. We will not re-derive those; we will *use* them, and explain the three or four things that are specific to training a 100M model on a single A100 to 200 tokens per parameter. When something threatens to blow the run up, we lean on [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

!!! note "Aside: the API contract this chapter fixes"
    Three functions cross the boundary into [Chapter 14.7](../14-capstone/07-pretraining-run.html) and the `capstone/stacklm` package. Their signatures are frozen here:

    ```text
    build_optimizers(model, muon_lr, adamw_lr, weight_decay, betas) -> (muon, adamw)
    wsd_lr(step, *, peak_lr, warmup_steps, total_steps,
           decay_steps=None, decay_frac=0.2, final_frac=0.0) -> float
    qk_clip_(model, max_logits: dict[int, Tensor], tau) -> int   # layers that fired
    ```

    Two consequences worth stating up front, because they are easy to get wrong. **(a)** Pretraining uses *two* optimizer objects, not one; where Ch. 14.7's prose says "`optimizer.step()`", read it as a loop over the `(muon, adamw)` pair — the training loop in `stacklm/train/loop.py` literally does `for opt in optimizers: opt.step()`. **(b)** `qk_clip_` is called from the **training loop, after** the optimizer steps, on a schedule (`qk_clip_every`) — it is *not* hidden inside `Muon.step()`, because it needs a quantity harvested from the forward pass that the optimizer never sees.

## Why Muon, and Why a Hybrid

Start with the memory arithmetic, because at 100M parameters it is not the binding constraint — and that itself is the lesson. AdamW keeps two full-precision state tensors per parameter, the first moment $m$ and second moment $v$. For `Stack-100M`'s ~101M parameters that is `101M × 2 × 4 bytes ≈ 0.8 GB` of optimizer state, trivial on an 80GB A100. So unlike a 70B run, we are **not** choosing Muon to save memory. We are choosing it because, at fixed compute, it walks the loss down measurably faster on the 2D weight matrices that dominate a transformer — the attention projections and the SwiGLU MLP. Muon has driven a series of speedrun records on nanoGPT-scale models (KellerJordan/modded-nanogpt), is the optimizer in Karpathy's `nanochat` (2025), and was used at frontier scale by **Kimi K2** (Moonshot, 2025). For a project whose whole thesis is "spend training compute once, serve forever," a faster optimizer is free perplexity.

The core idea, developed carefully in [Chapter 3.9](../03-pretraining/09-optimizers.html), is **orthogonalization of the momentum update**. Adam rescales each coordinate independently by its running gradient magnitude; it never looks at the *matrix structure* of a weight tensor. Muon does. It takes the momentum buffer $B_t$ for a weight matrix $W \in \mathbb{R}^{m\times n}$ and replaces it with the nearest **semi-orthogonal** matrix — the $UV^\top$ from its singular value decomposition $B_t = U\Sigma V^\top$, which is exactly the orthogonal factor in the polar decomposition. Intuitively, the raw momentum update is often dominated by a few large singular directions; the network learns fast along those and starves the rest. Orthogonalizing sets **every singular value to 1**, so the update pushes equally hard along all directions of the matrix. This is a spectral condition-number fix, and it is why Muon behaves like a cheap approximation to a full matrix preconditioner (Shampoo, Gupta et al., 2018) without the cost of forming and inverting the preconditioner.

{{fig:muon-orthogonalization-spectrum}}

Computing an SVD every step of every layer would be far too slow. Muon's key engineering move is to approximate the orthogonal factor with a fixed number (typically 5) of **Newton–Schulz iterations** — a matrix polynomial recurrence that converges to $(B B^\top)^{-1/2} B \approx U V^\top$ using only matrix multiplies, which are exactly what GPUs are fastest at. We implement it below.

### Why not Muon on everything?

Muon's orthogonalization only makes sense for a matrix whose two dimensions are both "feature" axes that mix. Three parameter groups in `Stack-100M` fail that test and stay on **AdamW**:

- **The token embedding** (`32768 × 512`, tied to the output head per Press & Wolf 2017). Its rows are per-token and extremely sparse in the gradient — most tokens in a batch touch a tiny fraction of rows. Orthogonalizing across the 32,768-token axis is meaningless; you want per-row adaptive rates, which is exactly Adam.
- **RMSNorm gains and any 1D vector.** A 1D tensor has no matrix structure to orthogonalize. (This group includes the QK-norm gains — which, as we will see in the stability section, are also the parameters the QK-clip has to reach for.)
- **Biases** — `Stack-100M` has essentially none (SwiGLU and RMSNorm are bias-free), but if present they go to AdamW.

This split — **Muon for 2D hidden matrices, AdamW for embeddings/norms/1D** — is not a compromise; it is the standard hybrid used at scale, including Kimi K2 and Moonlight. The capstone commits to it.

```text
Stack-100M parameter routing
┌─────────────────────────────────────────────┬───────────┬─────────┐
│ parameter group                              │ optimizer │  count  │
├─────────────────────────────────────────────┼───────────┼─────────┤
│ tok_emb.weight  (32768×512, tied)            │  AdamW    │    1    │
│ blocks.*.attn.wq / wo   (512×512)            │  Muon     │   60    │
│ blocks.*.attn.wk / wv   (128×512)            │  Muon     │   60    │
│ blocks.*.mlp.w_gate / w_up (1408×512)        │  Muon     │   60    │
│ blocks.*.mlp.w_down     (512×1408)           │  Muon     │   30    │
│ blocks.*.norm*.weight   (1D RMSNorm gains)   │  AdamW    │   60    │
│ blocks.*.attn.q_norm / k_norm.weight (1D)    │  AdamW    │   60    │
│ final_norm.weight       (1D)                 │  AdamW    │    1    │
└─────────────────────────────────────────────┴───────────┴─────────┘
                                       Muon total: 210 matrices
```

That is **7 two-dimensional matrices per block** (four attention projections + three SwiGLU matrices) × 30 blocks = **210 matrices** routed to Muon; everything else goes to AdamW. Keep that 210 in mind — it reappears twice, once when we estimate Muon's FLOP overhead and once when we fix its *wall-clock* overhead, which is a different and more interesting problem.

## Muon: The Newton–Schulz Orthogonalizer

Let us build the numerical core first, because everything else hangs off it. Given the momentum buffer $B \in \mathbb{R}^{m\times n}$, we want an orthogonal factor $O = U V^\top$ where $B = U\Sigma V^\top$. The quintic Newton–Schulz iteration Keller Jordan uses starts from a spectrally-normalized $X_0 = B / \lVert B \rVert_F$ (so all singular values land in $[0,1]$) and applies

$$
X_{k+1} = a\,X_k + b\,(X_k X_k^\top) X_k + c\,(X_k X_k^\top)^2 X_k,
$$

with the tuned coefficients $(a,b,c) = (3.4445,\,-4.7750,\,2.0315)$. This is a degree-5 odd polynomial $p(\sigma)$ applied to each singular value $\sigma$; the coefficients are chosen so that $p$ pushes every $\sigma \in (0,1]$ rapidly toward 1 while staying stable near 0. Five iterations get every singular value close enough to 1 for optimization purposes — we do **not** need machine-precision orthogonality, just a well-conditioned direction.

```python
# stacklm/optim/muon.py
import torch

@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal (polar) factor UV^T of a 2D matrix G via a
    quintic Newton-Schulz iteration. Returns a matrix with singular values ~1.

    This is the numerical heart of Muon (Jordan et al., 2024). It uses only
    matmuls, so it runs fast on tensor cores. We deliberately do NOT aim for
    exact orthogonality -- 5 steps is enough to give a well-conditioned update
    direction. The reference implementation runs this in bf16; the capstone repo
    runs it in fp32 so CPU and GPU results agree bit-for-bit in the smoke test.
    """
    assert G.ndim == 2, "Newton-Schulz orthogonalization is only for 2D matrices"
    a, b, c = 3.4445, -4.7750, 2.0315          # tuned quintic coefficients
    X = G.float()
    X = X / (X.norm() + 1e-7)                    # normalize so all sigma <= 1
    transposed = G.size(0) > G.size(1)          # keep the short side as rows
    if transposed:
        X = X.T                                  # iterate on the smaller Gram matrix
    for _ in range(steps):
        A = X @ X.T                              # A = X X^T  (small: min(m,n)^2)
        P = b * A + c * (A @ A)                  # polynomial in the Gram matrix
        X = a * X + P @ X                        # quintic Newton-Schulz update
    if transposed:
        X = X.T
    return X
```

Note the `transposed` trick: the Gram matrix $XX^\top$ has size $\min(m,n)^2$, so we always iterate on the smaller of the two dimensions. For a SwiGLU weight of shape `1408 × 512` that means the inner products live in $512^2$, not $1408^2$ — a real speedup. (We name the polynomial term `P`, not `b`, to avoid shadowing the coefficient; a subtle way to break this iteration is to reuse a coefficient name inside the loop.)

### The full Muon step and RMS matching

Two scale factors turn the orthogonal direction into a usable update.

**First**, the orthogonalized update has a root-mean-square element magnitude of roughly $1/\sqrt{\max(m,n)}$: a semi-orthogonal $m\times n$ matrix has $\min(m,n)$ singular values equal to 1, hence Frobenius norm $\sqrt{\min(m,n)}$, spread over $mn$ entries, so RMS $=\sqrt{\min(m,n)/(mn)} = 1/\sqrt{\max(m,n)}$. That is *shape-dependent*: the same learning rate would move a `1408×512` matrix and a `128×512` matrix by different relative amounts.

**Second**, to put Muon's step magnitude in the same regime as AdamW's, the Moonshot "Muon is Scalable" report (Liu et al., 2025) multiplies the Muon update by $0.2\cdot\sqrt{\max(m,n)}$, which cancels the shape dependence and brings the RMS to exactly $0.2$. Where does $0.2$ come from? Not from the idealized AdamW bound. AdamW's update is $m/(\sqrt{v}+\epsilon)$, whose *per-element* magnitude is $\approx 1$ only if $m$ and $v$ are estimated over the same window on a stationary gradient. In real LLM training the two EMAs disagree ($\beta_1 = 0.9$ vs $\beta_2 = 0.95$) and the gradient is noisy, so the **measured** update RMS across a transformer's weight matrices sits well below 1 — on the order of $0.2$–$0.4$. Moonshot picked $0.2$ to land inside that measured band. That is the honest statement; the "AdamW step has RMS 1" folklore is the theoretical ceiling, not the observed value.

{{fig:muon-rms-matching}}

RMS matching does **not** mean one literal learning rate for both groups. It means the two groups now live in the same *order of magnitude*, so you tune one number and derive the other with a small fixed ratio instead of searching a 2D grid. The capstone uses **`muon_lr = 6e-3`** and **`adamw_lr = muon_lr / 2 = 3e-3`**. The factor of two is not RMS-related at all: it is there because the AdamW group is dominated by the tied embedding, whose gradient is row-sparse (a rare token's row gets a full-magnitude update from the handful of batches that contain it), and a slightly smaller step on that group is the cheap insurance.

```python
# stacklm/optim/muon.py  (continued)
from torch.optim.optimizer import Optimizer

class Muon(Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalization, for 2D matrices ONLY.
    Route embeddings/norms/1D params to AdamW instead (see build_optimizers()).

    Args mirror the Jordan et al. reference plus the Moonshot RMS-matching scale
    and decoupled weight decay (Liu et al., 2025)."""

    def __init__(self, params, lr=6e-3, momentum=0.95, nesterov=True,
                 weight_decay=0.1, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            nesterov, wd, ns = group["nesterov"], group["weight_decay"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon received a non-2D param; check routing"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)                    # B <- mu*B + g
                # Nesterov look-ahead: use g + mu*B as the direction to orthogonalize
                d = g.add(buf, alpha=mu) if nesterov else buf
                o = zeropower_via_newtonschulz5(d, steps=ns).to(p.dtype)
                # RMS-match to AdamW so the two groups share one LR scale (Moonshot 2025)
                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5
                # Decoupled weight decay (AdamW-style): shrink the weight itself
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(o, alpha=-lr * scale)            # W <- W - lr*scale*O
        return loss
```

Two design points worth pausing on. We orthogonalize the **Nesterov look-ahead** direction $g + \mu B$ rather than the bare buffer — this is the reference Muon default and gives a slightly more responsive update. And weight decay is **decoupled** exactly as in AdamW ([Chapter 3.9](../03-pretraining/09-optimizers.html)): we shrink the weight by `(1 - lr*wd)` *before* applying the orthogonalized step, so the decay is a clean L2 pull toward zero and not entangled with the (unit-scale) orthogonal update. The capstone uses `weight_decay = 0.1` for both optimizers.

!!! note "Aside: Muon needs 2D, so reshape or route"
    Convolutional or fused QKV weights sometimes arrive as 3D/4D tensors. Muon requires 2D. In `Stack-100M` we keep the attention projections as separate 2D `wq/wk/wv/wo` matrices precisely so routing is unambiguous. If you fuse them, reshape to `(out, in)` before the Newton–Schulz call and reshape the result back — never hand Muon a raw >2D tensor.

### FLOPs are cheap; kernel launches are not

The FLOP overhead of Muon is genuinely negligible, and the worked example below computes it exactly: **~1.04 TFLOP per optimizer step against the model's ~319 TFLOP**, i.e. **0.33%**. But a FLOP count is not a wall-clock claim, and this is the single most common way Muon disappoints in practice.

Count the *launches* instead. The loop above visits 210 matrices; each runs 5 iterations; each iteration issues roughly four GPU kernels (`X @ X.T`, `A @ A`, the fused `b*A + c*(A@A)` elementwise, `P @ X`), plus the momentum and weight-decay ops. That is on the order of **4,200 kernel launches per optimizer step**, on matrices whose inner dimension is 128–1408. At those sizes an A100 finishes each matmul in single-digit microseconds while a kernel launch costs several microseconds — this regime is **launch-latency bound, not FLOP bound**. Measured naively, Muon's overhead at 100M scale is typically several percent of step time, not 0.3%.

Two fixes, both standard, both worth doing:

1. **Compile the iteration.** Wrapping `zeropower_via_newtonschulz5` in `torch.compile` fuses the elementwise work and can capture the whole 5-step loop into a CUDA graph, collapsing most of the launch overhead ([Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)).
2. **Batch same-shaped matrices.** After the orientation-normalizing transpose, `Stack-100M`'s 210 matrices collapse into just **three** shape classes: 60 of `512×512` (`wq`, `wo`), 60 of `128×512` (`wk`, `wv`), and 90 of `512×1408` (`w_gate`, `w_up`, and `w_down` transposed). Stack each class and run the iteration as a single `bmm` per class: **3 classes × 5 iterations × 4 kernels ≈ 60 launches**, a ~70× reduction.

```python
# stacklm/optim/muon.py  (optional fast path)
@torch.no_grad()
def zeropower_batched(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Same quintic, but on a STACK of identically-shaped matrices: G is
    (K, m, n) and we return (K, m, n). Collapses K*5*4 kernel launches into 5*4
    batched ones. Numerically identical to looping zeropower_via_newtonschulz5.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)   # per-matrix Frobenius
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT                                           # batched transpose
    for _ in range(steps):
        A = X @ X.mT                                       # (K, r, r) via bmm
        P = b * A + c * (A @ A)
        X = a * X + P @ X
    if transposed:
        X = X.mT
    return X
```

!!! tip "Practitioner tip: measure `optimizer.step()`, don't trust the FLOP ratio"
    Time it, because the answer is hardware- and shape-dependent:

    ```python
    torch.cuda.synchronize(); t0 = time.perf_counter()
    muon.step(); adamw.step()
    torch.cuda.synchronize(); print("opt step ms:", 1e3 * (time.perf_counter() - t0))
    ```

    Compare against the full step time. If the optimizer is more than ~2% of it, you are launch-bound: turn on `torch.compile` for the Newton–Schulz call, or switch to the batched path. Also use PyTorch's `torch._foreach_mul_` / `torch._foreach_add_` for the momentum bookkeeping — the same "one kernel for many tensors" trick that `fused=True` AdamW already applies internally.

## Setting the Peak Learning Rate (A Protocol, Not a Guess)

The peak LR is the most consequential number in this chapter, and "0.02-ish, like the reference recipes" is not an answer a reproducible book is allowed to give. [Chapter 14.5](../14-capstone/05-mini-scaling-laws.html) already built a **ladder** of tiny models (`{4M, 9M, 19M, 43M}` params) under a shared recipe precisely so we can measure hyperparameters instead of guessing them. Here is the protocol, and it is cheap: the whole sweep costs well under an hour on the same A100.

**Step 1 — sweep on the top ladder rung.** Take the 43M rung. Train it for 500–1000 optimizer steps at a *reduced* effective batch of one micro-batch, `32 × 2048 = 65,536` tokens/step, with the same WSD warmup shape (scale warmup proportionally, e.g. 100 steps). Sweep the Muon peak over a ×2 grid, `{1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2}`, keeping `adamw_lr = muon_lr / 2` throughout so you are searching a *line*, not a plane.

**Step 2 — read the curve, then back off one grid point.** Plot final held-out loss vs peak LR. You get a U: too low and the model is under-trained at this step count; too high and the loss either spikes or plateaus above the minimum. Pick the argmin, then step *one grid point down*. The full run is 40× longer than the probe, and the highest LR that survives 1,000 steps is not always the highest LR that survives 38,147.

**Step 3 — transfer across batch size.** The probe ran at 65,536 tokens/step; the real run is 524,288, an 8× increase. Below the critical batch size, the standard rule of thumb for adaptive/normalized updates is $\eta \propto \sqrt{B}$ ([Chapter 3.10](../03-pretraining/10-lr-schedules-hparams.html)), so multiply by $\sqrt{8} \approx 2.83$. A probe optimum in the low-`1e-3`s therefore lands near **`6e-3`** at full batch — which is exactly the `peak_lr` in `StackConfig`.

**Step 4 — transfer across width.** The 43M rung is narrower than `d_model = 512`. Two defensible options: (a) **muP** width scaling, which says matrix-like parameters want $\eta \propto 1/\text{fan\_in}$ ([Chapter 3.10](../03-pretraining/10-lr-schedules-hparams.html)); or (b) rely on the fact that Muon's RMS-matched update has a *fixed* per-element magnitude by construction, which makes its LR far closer to width-invariant than Adam's — this is one of Muon's practical selling points. We take (b) and then *verify* with a 200-step confirmation run at full width and full batch: if the loss curve at step 200 is not monotone-decreasing and the grad-norm log is not flat, drop the LR by 2× and repeat.

!!! warning "Common pitfall: what to do when it diverges anyway"
    You typed `6e-3` and the loss NaN'd at step 900. Work the list **in this order**, changing one thing at a time:

    1. **Look at the pre-clip grad norm** (`clip_grad_norm_` returns it — log it). A multi-sigma jump *before* the loss moves is the real timestamp of the failure.
    2. **Lengthen warmup** before lowering the peak. 500 steps is short; 1,000–2,000 costs almost nothing and fixes a large fraction of early-run divergence.
    3. **Halve the peak LR** (`6e-3` → `3e-3`). Re-run 500 steps and compare.
    4. **Check the QK-clip trigger log** (next section). If it is firing constantly, the instability is attention-logit growth and the LR is the cause, not the cure.
    5. **Safe fallback: drop Muon entirely.** Route everything to AdamW at `peak_lr = 6e-4` — the nanoGPT-124M-calibrated value, a known-good setting for a model of this size. You lose some convergence speed and you will finish the run.

## Attention-Logit Stability: QK-Norm, QK-Clip, and Soft-Caps

Muon's aggressiveness has a failure mode that AdamW largely hides: because it pushes equally along every singular direction, the **query and key projections can grow until attention logits explode**. The pre-softmax score is $s_{ij} = q_i^\top k_j / \sqrt{d_h}$; if $\lVert W_Q\rVert$ and $\lVert W_K\rVert$ drift upward together, $\max_{ij} s_{ij}$ climbs into the hundreds, the softmax saturates to a one-hot, gradients through it vanish, and — in bf16 — you get a loss spike or an outright NaN ([Chapter 3.11](../03-pretraining/11-training-stability.html)). Kimi K2 hit exactly this at trillion-parameter scale and introduced **MuonClip**, whose active ingredient is **QK-clip**: a per-head, post-step rescaling that caps the maximum attention logit.

Now the part most treatments get wrong. **Kimi K2 needed QK-clip on $W_Q$/$W_K$ precisely because it does not QK-norm.** `Stack-100M` *does* ([Chapter 14.4](../14-capstone/04-architecture.html) sets `qk_norm = True`), and under QK-norm, rescaling $W_Q$ or $W_K$ changes nothing at all. The two mechanisms are alternatives operating on the same knob, not layers that stack.

### The invariance, and the bound QK-norm actually buys

`Stack-100M`'s attention computes, per head,

$$
q = \gamma_q \odot \frac{W_Q x}{\operatorname{rms}(W_Q x)}, \qquad
k = \gamma_k \odot \frac{W_K x}{\operatorname{rms}(W_K x)},
$$

with $\gamma_q,\gamma_k \in \mathbb{R}^{d_h}$ the learned RMSNorm gains. RMSNorm is **positively homogeneous of degree 0**: for any $\eta > 0$, $\operatorname{RMSNorm}(\eta z) = \operatorname{RMSNorm}(z)$ up to the $\epsilon$ in the denominator. Therefore

$$
W_Q \mapsto \eta\,W_Q \;\Longrightarrow\; q \mapsto q, \qquad s_{ij} \mapsto s_{ij}.
$$

Scaling the projections is a **no-op**. RoPE does not rescue it either: RoPE is a rotation applied *after* the norm, and rotations preserve norms.

What *does* set the logit scale is the pair of gains. Since $u = z/\operatorname{rms}(z)$ has unit RMS, $\lVert u \rVert_2 = \sqrt{d_h}$, so $\lVert q \rVert_2 \le \lVert \gamma_q \rVert_\infty \sqrt{d_h}$, and by Cauchy–Schwarz

$$
\boxed{\;|s_{ij}| \;=\; \frac{|q_i^\top k_j|}{\sqrt{d_h}} \;\le\; \sqrt{d_h}\;\lVert\gamma_q\rVert_\infty \lVert\gamma_k\rVert_\infty\;}
$$

For `Stack-100M`, $d_h = 64$ and both gains initialize to 1, so **the max attention logit is bounded by 8 at initialization**, with no dependence on $W_Q, W_K$ whatsoever. Blow-up is only possible if the *gains* grow. That is a much narrower failure surface — and it is where the clip must act.

You can verify all of this in about ten seconds against the capstone package:

```python
# verify_qk_invariance.py -- run from capstone/
import torch
from stacklm.config import toy_config
from stacklm.model.transformer import Stack100M

torch.manual_seed(0)
cfg = toy_config()                       # head_dim=16 -> a-priori bound sqrt(16)=4
model = Stack100M(cfg)
x = torch.randint(0, cfg.vocab_size, (2, 32))

def max_logit(m):
    rec = {}
    for b in m.blocks:                   # exact per-head max, not the cheap bound
        b.attn.record_exact = True
    with torch.no_grad():
        m(x, record=rec)
    return max(float(v.max()) for v in rec.values())

a = model.blocks[0].attn
gq, gk = a.q_norm.weight.detach(), a.k_norm.weight.detach()
bound = cfg.head_dim ** 0.5 * gq.abs().max() * gk.abs().max()
print(f"a-priori bound        {float(bound):.4f}")
print(f"observed max logit    {max_logit(model):.4f}")

with torch.no_grad():                    # (1) rescale the PROJECTIONS
    for b in model.blocks:
        b.attn.wq.weight.mul_(0.3); b.attn.wk.weight.mul_(0.3)
print(f"after W_Q,W_K x0.3    {max_logit(model):.4f}   <- unchanged (no-op)")

with torch.no_grad():                    # (2) rescale the GAINS
    for b in model.blocks:
        b.attn.q_norm.weight.mul_(0.5); b.attn.k_norm.weight.mul_(0.5)
print(f"after gains x0.5 each {max_logit(model):.4f}   <- exactly 0.25x")
```

```text
a-priori bound        4.0000
observed max logit    3.3529
after W_Q,W_K x0.3    3.3388   <- unchanged (no-op)
after gains x0.5 each 0.8347   <- exactly 0.25x
```

The projection rescale moved the max logit by 0.4% — that residue is entirely the RMSNorm $\epsilon$, not a real effect. The gain rescale multiplied it by $0.5\times0.5 = 0.25$ **exactly**, as the algebra promises. And the a-priori bound of 4 (here $d_h = 16$) is tight to within 20% of the observed maximum.

### Harvesting $S_{\max}$ without giving up FlashAttention

Before you can clip, you need the per-head maximum logit — and this is *not* free, contrary to what a casual reading of the Kimi recipe suggests. `Stack-100M`'s fast path is `F.scaled_dot_product_attention`, which dispatches to a FlashAttention-style kernel ([FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)). That kernel's entire point is that it **never materializes** the $(B, H, T, T)$ score matrix. There is no `.amax()` to take, because there is nothing to take it of.

If you want the exact maximum you must fall back to eager attention and build the score tensor. At the capstone's `micro_batch_size = 32`, `n_heads = 8`, `T = 2048`, in fp32:

$$
32 \times 8 \times 2048 \times 2048 \times 4\ \text{bytes} \;=\; 4.29\ \text{GB} \quad\textbf{per layer},
$$

2.15 GB in bf16, plus $O(T^2)$ FLOPs and the loss of FlashAttention's memory win. Doing that every step, on every micro-batch, of all 30 layers is not a rounding error; it is a different training run.

The capstone therefore offers **three tiers**, and defaults to the cheap one:

| Tier | What it computes | Cost | Keeps SDPA? |
|---|---|---|---|
| 0 — weight-only | $\sqrt{d_h}\lVert\gamma_q\rVert_\infty\lVert\gamma_k\rVert_\infty$ | free, no forward pass | yes |
| 1 — **default** | $\max_i\lVert q_i\rVert \cdot \max_j\lVert k_j\rVert / \sqrt{d_h}$ | $O(BHTd_h)$, no $T^2$ tensor | yes |
| 2 — exact | $\max_{ij} s_{ij}$ over the causal mask | $O(BHT^2)$, 4.3 GB/layer | **no** |

Tiers 0 and 1 are Cauchy–Schwarz *upper* bounds, so they fire the clip slightly early — a conservative error, which is what you want in a safety net. Tier 1 is tight in practice because attention keys and queries are not adversarially aligned. Here is the real plumbing in `stacklm/model/attention.py`:

```python
# stacklm/model/attention.py  (inside Attention.forward, after QK-norm + RoPE,
#  after k/v have been repeat_interleave'd up to n_heads)
scale = 1.0 / (self.head_dim ** 0.5)

if record is not None:
    with torch.no_grad():
        if self.record_exact:
            # Tier 2: exact, but materializes (B, n_heads, T, T) and gives up SDPA.
            att = (q.float() @ k.float().transpose(-2, -1)) * scale
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~causal, float("-inf"))
            record[self.layer_idx] = att.amax(dim=(0, 2, 3))     # (n_heads,)
        else:
            # Tier 1 (default): Cauchy-Schwarz bound. O(B*H*T*d_h) work,
            # no T^2 tensor, FlashAttention/SDPA fast path preserved below.
            qn = q.float().norm(dim=-1).amax(dim=(0, 2))         # (n_heads,)
            kn = k.float().norm(dim=-1).amax(dim=(0, 2))         # (n_heads,)
            record[self.layer_idx] = qn * kn * scale
```

And the second half of the answer is **frequency**. Attention-logit drift is *slow* — the gains move by a decayed learning rate per step. There is no reason to measure it 38,147 times. The training loop gates the whole thing behind `qk_clip_every` (we use **200**, i.e. ~190 measurements over the run) and takes the reading on a single small **probe batch** with post-step weights:

```python
# stacklm/train/loop.py  (excerpt: after the optimizer steps)
qk_fired = 0
if qk_clip_every and step % qk_clip_every == 0:
    rec = {}
    with torch.no_grad():
        b = _to_device(next(it), device)      # one probe micro-batch, no grads
        model(b["input_ids"], seq_ids=b["seq_ids"], record=rec)
    qk_fired = qk_clip_(model, rec, tau=qk_tau)
```

Amortized over 200 steps, even a Tier-2 exact reading costs well under 0.1% of the run. Log `qk_fired`: it is the single most informative stability signal this loop produces.

### The clip that actually works here

Since the gains are the knob, the clip scales the gains. Scaling $\gamma_q$ and $\gamma_k$ each by $\sqrt{\tau/S_{\max}}$ multiplies every logit in the layer by exactly $\tau/S_{\max}$ — this is *exactly* the bilinearity argument Kimi uses, just applied to the parameters that are actually free.

One structural consequence you must not paper over: `Stack-100M` allocates **one** gain vector of length `head_dim` per layer, *shared by all heads* (`RMSNorm(cfg.head_dim)` in Ch. 14.4's `Attention.__init__`). So the clip's granularity is the **layer**, driven by the layer's worst head — not the head. That is the price of the cheaper parameterization, and it is fine: over-shrinking the well-behaved heads in a layer by the same factor is conservative and reversible (they will regrow their gains). If you want Kimi's per-head granularity, widen the gains to `(n_heads, 1, head_dim)` and `(n_kv_heads, 1, head_dim)`, which costs `8*64 + 2*64 = 640` parameters per layer instead of 128 — and then the GQA group logic below transfers verbatim (Exercise 7 works this through).

{{fig:qk-clip-gqa}}

```python
# stacklm/optim/qk_clip.py
import torch

@torch.no_grad()
def qk_clip_(model, max_logits, tau: float = 30.0) -> int:
    """QK-clip (the core of MuonClip, Kimi K2 / Moonshot 2025), architecture-aware.

    `max_logits[layer_idx]` is an (n_heads,) tensor recorded by the attention
    module this forward pass. Returns the number of layers that fired -- log it.

    The clip acts on whatever parameter actually controls the logit scale:
      * QK-norm ON  -> the RMSNorm gains (rescaling W_Q/W_K is provably a no-op).
      * QK-norm OFF -> W_Q / W_K, per Kimi K2, GQA-aware.
    """
    fired = 0
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx not in max_logits:
            continue
        s_max = max_logits[layer_idx].float()          # (n_heads,)
        attn = block.attn
        if isinstance(attn.q_norm, torch.nn.Identity):
            fired += _clip_projections_(attn, s_max, tau)    # Kimi K2 config
        else:
            fired += _clip_qk_norm_gains_(attn, s_max, tau)  # Stack-100M
    return fired


@torch.no_grad()
def _clip_qk_norm_gains_(attn, s_max, tau: float) -> int:
    """QK-norm path: the gains ARE the temperature, so clip them.

    Scaling gamma_q and gamma_k each by sqrt(tau/S) multiplies every logit in the
    layer by exactly tau/S (RoPE is a rotation and commutes with the scale).
    Stack-100M shares one gain vector across heads, so this is a per-LAYER clip
    driven by the layer's worst head.
    """
    s = float(s_max.max())
    if s <= tau:
        return 0                                  # this layer is fine
    eta = (tau / s) ** 0.5                        # sqrt so q AND k share it
    attn.q_norm.weight.mul_(eta)
    attn.k_norm.weight.mul_(eta)
    return 1
```

**Choosing $\tau$.** Kimi K2 used $\tau = 100$. Copying that number here would give you a **dead safety net**: under QK-norm the logits start bounded by 8, so reaching 100 requires $\lVert\gamma_q\rVert_\infty\lVert\gamma_k\rVert_\infty > 12.5$ — a state so pathological the run is already lost. Pin $\tau$ relative to *your* architecture's natural scale. We use **$\tau = 30$**, roughly $4\times$ the initialization bound: comfortably above normal gain drift, far below the softmax-saturation regime, and low enough that the clip is a genuine early-warning system rather than a last rite.

### If you drop QK-norm: Kimi K2's per-head, GQA-aware clip

Run `Stack-100M` with `qk_norm = False` (the ablation Ch. 14.4 recommends you actually try) and $W_Q, W_K$ become the knob again. Now the original MuonClip applies — and here GQA introduces a wrinkle worth its own code path.

In `Stack-100M`, `n_heads = 8` query heads share `n_kv_heads = 2` key/value heads (a 4:1 ratio; see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)), so four query heads read the *same* $W_K$ slice. We must scale each $W_Q$ slice **per query head**, but each shared $W_K$ slice only **once** — otherwise a KV slice read by four tripping heads gets multiplied four times over, badly over-shrinking it. Two separate loops:

```python
# stacklm/optim/qk_clip.py  (continued)
@torch.no_grad()
def _clip_projections_(attn, s_max, tau: float) -> int:
    """No-QK-norm path (Kimi K2): per-query-head W_Q, per-KV-head shared W_K.

    Attribute names (n_heads, n_kv_heads, head_dim, groups) are exactly those of
    stacklm.model.attention.Attention -- keep them stable across chapters.
    """
    hd = attn.head_dim
    group = attn.groups                          # q-heads per kv-head (= 4)
    fired = 0

    # (1) Per-query-head scale on W_Q.
    for h in range(attn.n_heads):
        if float(s_max[h]) <= tau:
            continue                             # this head is fine
        eta = (tau / float(s_max[h])) ** 0.5     # sqrt so q AND k share it
        attn.wq.weight[h * hd:(h + 1) * hd].mul_(eta)
        fired += 1

    # (2) Per-KV-head scale on the SHARED W_K, using the group's worst logit.
    for kv in range(attn.n_kv_heads):
        s_grp = float(s_max[kv * group:(kv + 1) * group].max())
        if s_grp <= tau:
            continue
        eta = (tau / s_grp) ** 0.5
        attn.wk.weight[kv * hd:(kv + 1) * hd].mul_(eta)
    return fired
```

The head that *set* the group maximum lands exactly at $\tau$; the other three query heads in its group get their shared key scaled by the same amount and so are pulled slightly further below $\tau$ — a conservative, safe outcome. Exercise 4 walks the arithmetic. In this configuration $\tau = 100$ is the right order of magnitude, because without QK-norm there is no a-priori bound at all.

!!! warning "Common pitfall: attribute names must match across chapters"
    `qk_clip_` reaches *into* the attention module, so it is coupled to that module's attribute names. The canonical names — the ones `capstone/stacklm/model/attention.py` defines and this chapter uses — are **`n_heads`, `n_kv_heads`, `head_dim`, `groups`, `wq`, `wk`, `q_norm`, `k_norm`**. If you transcribe an `Attention` class that abbreviates them (`d_h`, `n_kv`), the clip dies with an `AttributeError` on its first firing — which, since it fires rarely, may be thousands of steps into the run. Either keep the canonical names or add a two-line adapter. A `getattr(attn, "head_dim", None) or attn.d_h` in library code is a smell; fixing the model class is the right move.

### The third lever: attention soft-capping

`StackConfig` already carries `attn_soft_cap` (default `0.0` = off). Set it to `50.0` and the attention scores pass through Gemma-2's $c\tanh(s/c)$ before the softmax: a smooth, differentiable ceiling that makes blow-up *impossible* rather than merely corrected-after-the-fact. Its cost is that the naive implementation materializes the score matrix and abandons SDPA — the same $T^2$ tax as Tier-2 recording. The modern answer is PyTorch's **FlexAttention** (`torch.nn.attention.flex_attention`), which lets you express the cap as a `score_mod` function that the compiler fuses *into* a FlashAttention-style kernel, keeping the memory win ([Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)).

The three levers form a clean hierarchy, and you should be able to say which is which in an interview:

| Lever | Acts on | When | Cost |
|---|---|---|---|
| **QK-norm** | forward pass, every token | always (default ON) | 128 params/layer; a norm |
| **Soft-cap** | forward pass, every score | if you need a hard ceiling | fused via FlexAttention, else $T^2$ |
| **QK-clip** | parameters, post-step | every `qk_clip_every` steps | one probe forward per 200 steps |

QK-norm removes the failure mode structurally. Soft-cap bounds it analytically. QK-clip is the *monitor plus correction* — its real value at 100M scale is not that it fires, but that its trigger log tells you your LR is too high days before the loss does.

## The WSD Schedule: Warmup–Stable–Decay

Now the schedule. The dominant pretraining schedule of 2023–2024 was **cosine annealing** ([Chapter 3.10](../03-pretraining/10-lr-schedules-hparams.html)): warm up, then decay the LR along a cosine from peak to a small floor over the *entire* run. Cosine has one fatal inconvenience for a project like ours: **you must know the total step count in advance**, because the cosine is stretched to fit it. Decide to train longer and the whole curve is wrong; the model trained to step $T$ was on a schedule optimized for $T$, not $2T$. And cosine gives you no natural place to swap the data distribution.

**Warmup–Stable–Decay** (WSD; MiniCPM, Hu et al., 2024; adopted by DeepSeek) fixes both. It has three phases:

$$
\eta(t) = \begin{cases}
\eta_{\max}\cdot \dfrac{t}{T_w} & 0 \le t < T_w \quad(\textbf{warmup})\\[8pt]
\eta_{\max} & T_w \le t < T_s \quad(\textbf{stable})\\[8pt]
\eta_{\max}\cdot f\!\left(\dfrac{t - T_s}{T - T_s}\right) & T_s \le t \le T \quad(\textbf{decay})
\end{cases}
$$

{{fig:wsd-schedule-anneal}}

where $f$ decays from 1 to (near) 0 over the final phase. MiniCPM found a **$1-\sqrt{\cdot}$** or exponential-style decay works better than linear; we use the $1-\sqrt{\cdot}$ form. Two properties make WSD ideal for the capstone:

1. **The stable phase is schedule-agnostic in length.** You hold $\eta_{\max}$ constant for as long as your token budget allows. Because there is no baked-in endpoint, you can *decide to keep going* — extend the stable phase, or branch off multiple decay runs from the same stable checkpoint to compare data mixes. This is the "continuable pretraining" property MiniCPM highlighted.
2. **The decay phase is where the loss drops fastest — and that is precisely where we anneal on premium data.** Empirically WSD shows a characteristic sharp loss drop once decay begins. The capstone exploits this: the WSD **decay phase *is* the mid-training annealing phase** of [Chapter 14.8](../14-capstone/08-mid-training.html). We spend the constant-LR stable phase on the bulk FineWeb-Edu mix, then, as the LR decays, we shift the data toward higher-quality Cosmopedia, math, and code. Low LR + high-quality data = the model "polishes" on the best tokens without the high-LR thrash that would otherwise wash them out. Cosine cannot cleanly do this because its LR is already low through most of the middle of training.

```python
# stacklm/optim/schedule.py
import math

def wsd_lr(step: int, *, peak_lr: float, warmup_steps: int, total_steps: int,
           decay_steps: int | None = None, decay_frac: float = 0.2,
           final_frac: float = 0.0) -> float:
    """Warmup-Stable-Decay learning rate (MiniCPM, Hu et al. 2024).

    - Linear warmup for `warmup_steps`.
    - Constant `peak_lr` through the stable phase.
    - 1 - sqrt() decay over the final phase, down to `final_frac * peak_lr`
      (we use 0.0, i.e. anneal fully to ~0).

    Give EITHER `decay_steps` (absolute -- what Ch. 14.7's StackConfig passes,
    6000) or `decay_frac` (a fraction of total_steps). `decay_steps` wins.

    Returns the *actual* LR (not a multiplier), so you can set it directly.
    """
    if decay_steps is None:
        decay_steps = int(decay_frac * total_steps)
    stable_end = total_steps - decay_steps               # first decay step
    if step < warmup_steps:                              # --- warmup ---
        return peak_lr * (step + 1) / warmup_steps
    if step < stable_end:                                # --- stable ---
        return peak_lr
    # --- decay: 1 - sqrt(progress) ---
    progress = (step - stable_end) / max(1, decay_steps)
    # CLAMP: without min(...,1.0) any step past total_steps gives sqrt(p) > 1 and
    # a NEGATIVE learning rate -- live risk, since Ch. 14.8 continues past the
    # pretrain horizon. Past the end we sit at the floor.
    decay_mult = 1.0 - math.sqrt(min(progress, 1.0))     # 1 -> 0
    floor = final_frac * peak_lr
    return floor + (peak_lr - floor) * decay_mult
```

We apply the *same* schedule shape to both optimizers, each at its own peak. The step-by-step wiring, using the frozen `StackConfig` numbers:

```python
# stacklm/train/loop.py  (once per optimizer step)
PEAK_LR      = 6e-3       # Muon group
WARMUP_STEPS = 500
DECAY_STEPS  = 6_000
TOTAL_STEPS  = 38_147     # ceil(20e9 tokens / 524,288 tokens per step)

lr = wsd_lr(step, peak_lr=PEAK_LR, warmup_steps=WARMUP_STEPS,
            total_steps=TOTAL_STEPS, decay_steps=DECAY_STEPS)
for g in muon.param_groups:  g["lr"] = lr          # Muon peak = 6e-3
for g in adamw.param_groups: g["lr"] = lr * 0.5    # AdamW peak = 3e-3
```

!!! note "Aside: why the decay must reach ~0"
    The loss drop in WSD's decay phase comes from the LR going genuinely small, letting the optimizer settle into a sharper minimum on the (now higher-quality) data. If you floor the LR at, say, $\eta_{\max}/10$ the way a cosine schedule often does, you leave that gain on the table. We anneal to essentially zero. The cost is that a decayed checkpoint is "spent" — you cannot productively resume high-LR training from it — which is exactly why we branch decay runs off the *stable* checkpoint, never off a decayed one.

!!! tip "Practitioner tip: WSD in the standard libraries"
    You do not have to hand-roll this in every project. Recent versions of HuggingFace `transformers` ship a `get_wsd_schedule` alongside the classic `get_cosine_schedule_with_warmup`, and PyTorch's `torchtitan` reference training stack exposes warmup/stable/decay phase lengths as config. We write ours from scratch because the capstone's decay phase is not just a schedule — [Chapter 14.8](../14-capstone/08-mid-training.html) swaps the *data distribution* at exactly the same boundary, and it is much clearer when the schedule is 20 lines you own.

## Putting It Together: Building the Optimizers

Here is the routing function the training loop calls once at startup. It walks the model's named parameters, sends every 2D hidden matrix to Muon and everything else to AdamW, and returns both. The dimensionality test (`p.ndim == 2` and "not the embedding") is the whole trick.

```python
# stacklm/optim/build.py
import torch
from stacklm.optim.muon import Muon

def build_optimizers(model, muon_lr=6e-3, adamw_lr=3e-3,
                     weight_decay=0.1, betas=(0.9, 0.95)):
    """Route Stack-100M params: 2D hidden matrices -> Muon, everything else
    (tied embedding, RMSNorm/QK-norm gains, any 1D/bias) -> AdamW.
    Returns (muon, adamw). Capstone defaults: 6e-3 and 3e-3 (a 2:1 ratio).
    """
    muon_params, adamw_params = [], []
    # The tied embedding is exposed as model.tok_emb.weight (output head shares it).
    embed_ids = {id(model.tok_emb.weight)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in embed_ids:
            adamw_params.append(p)                 # embedding -> AdamW (sparse rows)
        elif p.ndim == 2:
            muon_params.append(p)                  # hidden matrix -> Muon
        else:
            adamw_params.append(p)                 # 1D norms/biases -> AdamW
    muon = Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True,
                weight_decay=weight_decay, ns_steps=5)
    # Weight-decay split for AdamW: decay the 2D embedding lightly, leave 1D norm
    # gains un-decayed (shrinking a RMSNorm gain toward 0 fights the network --
    # and would fight the QK-clip, which uses those same gains as its knob).
    decay, no_decay = [], []
    for p in adamw_params:
        (decay if p.ndim >= 2 else no_decay).append(p)
    adamw = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=adamw_lr, betas=betas, eps=1e-8, fused=torch.cuda.is_available())
    return muon, adamw
```

The AdamW **betas are `(0.9, 0.95)`**, the LLM-standard choice ([Chapter 3.9](../03-pretraining/09-optimizers.html)): the lower $\beta_2=0.95$ (vs the vision default 0.999) makes the second-moment estimate more responsive, which matters because language gradients are noisier and more non-stationary than image gradients. We enable `fused=True` on CUDA for a single-kernel AdamW step. Note the param-group split so that 1D norm gains get **no weight decay** — and notice this is now doubly important: decaying $\gamma_q,\gamma_k$ toward zero would slowly starve attention of temperature while the QK-clip is trying to manage exactly that quantity.

### The full step: grad accumulation, clipping, bf16, periodic QK-clip

The capstone's effective batch is **524,288 tokens** — `micro_batch_size 32 × seq_len 2048 × grad_accum_steps 8`, the number PLAN §5 rounds to "~0.5M". Everything runs under **bf16 autocast** ([Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)); bf16's wide exponent means we need **no loss scaler**, unlike fp16.

```python
# stacklm/train/loop.py  (the inner optimizer step, abbreviated)
import torch
from stacklm.optim import build_optimizers, qk_clip_, wsd_lr

MICRO_BSZ      = 32       # sequences per micro-step
GRAD_ACCUM     = 8        # micro-steps per optimizer step  -> 524,288 tokens
GRAD_CLIP      = 1.0
QK_CLIP_EVERY  = 200      # measure + clip every 200 optimizer steps
QK_TAU         = 30.0     # ~4x the a-priori QK-norm bound of sqrt(d_h) = 8

def optimizer_step(model, batches, muon, adamw, step, cfg):
    lr = wsd_lr(step, peak_lr=cfg.peak_lr, warmup_steps=cfg.warmup_steps,
                total_steps=cfg.total_steps, decay_steps=cfg.decay_steps)
    for g in muon.param_groups:  g["lr"] = lr
    for g in adamw.param_groups: g["lr"] = lr * 0.5

    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    for _ in range(GRAD_ACCUM):
        x, y, seq_ids = next(batches)                # (MICRO_BSZ, 2048)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y, seq_ids=seq_ids)
            loss = loss / GRAD_ACCUM                 # average, not sum, over accum
        loss.backward()                              # accumulate grads

    # Clip the GLOBAL grad norm across ALL params (both optimizers) to 1.0.
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    muon.step()
    adamw.step()

    # MuonClip / QK-clip: PERIODIC, on a fresh probe batch with post-step weights.
    qk_fired = 0
    if step % QK_CLIP_EVERY == 0:
        rec = {}                                     # dict[layer_idx] -> (n_heads,)
        with torch.no_grad():
            xp, _, sp = next(batches)
            model(xp, seq_ids=sp, record=rec)        # Tier-1 cheap bound by default
        qk_fired = qk_clip_(model, rec, tau=QK_TAU)
    return float(loss) * GRAD_ACCUM, float(gnorm), lr, qk_fired
```

Four details that bite people:

1. **Average, don't sum.** We divide the loss by `GRAD_ACCUM` *inside* the loop so the gradient magnitude — and thus the effective LR — is independent of how many micro-steps we chose; change the accumulation count and the run behaves identically.
2. **Clip globally.** The grad-norm clip runs over *all* parameters at once, `max_norm = 1.0`, so a spike in any one layer is tamed relative to the whole update. Clipping each optimizer's group separately would let one group's blow-up hide behind the other's normal-sized gradients.
3. **`record` is a `dict`, not a tensor.** It maps `layer_idx -> (n_heads,)`. It is created fresh each time it is used and thrown away; there is no `.zero_()` and no persistent buffer to forget to reset.
4. **QK-clip is periodic and post-step.** It reads weights *after* `.step()`, on a probe batch, every `QK_CLIP_EVERY` steps. Doing it every step on every micro-batch is the mistake that turns a 0.05% safety net into a 30% tax.

### Why 524,288 tokens, and the critical batch size

The batch is not arbitrary. [Chapter 3.10](../03-pretraining/10-lr-schedules-hparams.html) develops the **critical batch size** (McCandlish et al., 2018): below it, doubling the batch nearly halves the step count for the same loss (compute is "wasted" on gradient noise you could average away); above it, extra batch buys diminishing returns and mostly costs memory. For a 100M model the critical batch sits on the order of a few hundred thousand to ~1M tokens, so ~0.5M is a sweet spot — large enough that the gradient estimate is clean (which Muon likes, since orthogonalizing a noisy buffer wastes the preconditioner) and small enough that we still take 38,147 update steps to spend the budget. Push the batch to 4M tokens and you would take only ~4,800 steps, each barely better than at 0.5M — a worse use of the same tokens.

Micro-batch sizing is then a pure memory game on the A100. Activation memory scales with `MICRO_BSZ × seq_len × d_model × n_layers`; at `MICRO_BSZ = 32` and `seq_len = 2048`, `Stack-100M` fits comfortably in 80GB *without* activation checkpointing (Ch. 14.7 does the arithmetic). On a 24GB 4090 you drop to `MICRO_BSZ = 8` and raise `GRAD_ACCUM` to 32 — same effective batch, same recipe, more wall-clock, exactly as the spec's secondary tier promises. The invariant to preserve is the product:

$$
\texttt{micro\_batch\_size} \times \texttt{seq\_len} \times \texttt{grad\_accum\_steps} = 524{,}288 .
$$

### One table for the shared knobs

| Knob | Value | Why |
|---|---|---|
| Muon peak LR | **6e-3** | swept on the 43M ladder rung, $\times\sqrt{8}$ batch transfer |
| AdamW peak LR | **3e-3** (= peak/2) | RMS matching puts them in one decade; ÷2 for the row-sparse embedding |
| AdamW betas | (0.9, 0.95) | $\beta_2$ lowered from 0.999 for noisy LM gradients |
| Weight decay | 0.1 | decoupled; **0.0** on all 1D gains |
| Grad clip (global) | 1.0 | one clip over every parameter |
| Momentum (Muon) | 0.95, Nesterov | orthogonalize the look-ahead direction |
| Newton–Schulz steps | 5 | enough to condition the direction; not exact |
| Precision | bf16 autocast | wide exponent ⇒ no loss scaler |
| Effective batch | 524,288 tokens | 32 × 2048 × 8; near critical batch for 100M |
| Total steps | 38,147 | $\lceil 20\text{e}9 / 524{,}288 \rceil$ |
| Warmup / stable / decay | 500 / 31,647 / 6,000 | decay = the mid-training window |
| QK-clip $\tau$ | **30** (QK-norm on) | ~4× the a-priori bound $\sqrt{d_h}=8$; use 100 only without QK-norm |
| `qk_clip_every` | 200 | logit drift is slow; measuring is not free |

## Worked Example: Step Budget, Overhead, and a Newton–Schulz Trace

!!! example "The Stack-100M schedule, in real numbers"
    **Tokens per step.** $32 \times 2048 \times 8 = 524{,}288$ tokens.

    **Total steps.** Budget = 20B tokens:
    $$
    T = \left\lceil \frac{20\times10^{9}}{524{,}288} \right\rceil = 38{,}147 \text{ optimizer steps.}
    $$

    **Phase split.** $T_w = 500$ (warmup, ~262M tokens, 1.3% of the run), decay $= 6{,}000$ steps (~3.15B tokens), stable $= 38{,}147 - 500 - 6{,}000 = 31{,}647$ steps (~16.6B tokens on the bulk mix). The **6,000-step decay phase is the mid-training window** where we anneal on premium data ([Chapter 14.8](../14-capstone/08-mid-training.html)).

    **Muon LR trace.** At step 250 (mid-warmup): $6\text{e-}3 \times 251/500 \approx 3.0\text{e-}3$. Through the stable phase: $6\text{e-}3$ flat. Halfway through decay ($\text{progress}=0.5$): $1-\sqrt{0.5}=0.293$, so $\eta = 6\text{e-}3\times0.293 \approx 1.8\text{e-}3$. At progress $0.9$: $1-\sqrt{0.9}=0.051$, $\eta\approx 3.1\text{e-}4$. The LR falls off fast early in decay — matching the "sharp loss drop" WSD is known for. The AdamW group tracks the same curve at half the magnitude.

    **Newton–Schulz FLOPs, exactly.** For $X$ of shape $(r, c)$ with $r=\min(m,n)$, one iteration costs $2r^2c$ (for $XX^\top$) $+\;2r^3$ (for $A^2$) $+\;2r^2c$ (for $PX$). Over 5 iterations:

    - `w_gate`/`w_up`/`w_down` → $r=512, c=1408$: $8.72$ GFLOP each × 90 matrices = **785 GFLOP**
    - `wq`/`wo` → $r=c=512$: $4.03$ GFLOP each × 60 = **242 GFLOP**
    - `wk`/`wv` → $r=128, c=512$: $0.19$ GFLOP each × 60 = **11 GFLOP**

    Total $\approx 1.04$ **TFLOP per optimizer step**. Against the model's forward+backward of $6ND = 6 \times 101.4\text{e}6 \times 524{,}288 \approx 3.19\times10^{14}$ FLOP/step (≈ 319 TFLOP), Muon's orthogonalization overhead is
    $$
    \frac{1.04\times10^{12}}{3.19\times10^{14}} = \mathbf{0.33\%}.
    $$

    **But the launch count.** $210 \text{ matrices} \times 5 \text{ iters} \times 4 \text{ kernels} \approx 4{,}200$ launches per step. Batched by the three shape classes: $3 \times 5 \times 4 = 60$. That is the difference between "0.33% as predicted" and "several percent, why is Muon slow."

    **QK-clip amortized cost.** A Tier-1 (cheap-bound) probe forward is one extra forward pass every 200 steps: $\approx \frac{1}{3}\times\frac{1}{200} \approx 0.17\%$ of training compute (a forward is ~1/3 of forward+backward). A Tier-2 exact probe adds $O(BHT^2)$ and ~4.3 GB of transient memory per layer, still amortized to well under 1%.

The takeaway: Muon buys faster convergence for essentially free *compute* at this scale — the Newton–Schulz iterations are dwarfed by the transformer's own matmuls — provided you do not squander it on kernel launches.

!!! warning "Common pitfall: writing the per-step FLOPs as ~1e17"
    A frequent slip is to compute $6ND$ with $D$ = the *whole* 20B-token budget and call the result "per step." $6 \times 101.4\text{e}6 \times 20\text{e}9 \approx 1.22\times10^{19}$ is the **total** training FLOPs for the run — a useful number in its own right. At 40% MFU on an A100's ~312 TFLOP/s bf16 peak it predicts $1.22\text{e}19 / (0.4\times312\text{e}12) \approx 9.8\times10^{4}$ s ≈ **27 GPU-hours**, a little above PLAN's 15–25 hour estimate; closing that gap is exactly what `torch.compile`, a well-fed micro-batch, and the batched Muon path buy you. Per *step*, $D$ is the 524,288 tokens in the batch, giving $\sim3.2\times10^{14}$.

## Contrasting the Choices, and What Would Break

It is worth stating plainly why each alternative was rejected, because an interviewer will ask.

- **Pure AdamW (no Muon).** Perfectly safe, and what nanoGPT-2024 used. It simply converges slower per token on the 2D matrices; at a fixed 20B-token budget you leave perplexity on the table. If you distrust Muon's stability, this is the fallback — route all 2D matrices to AdamW too and drop the peak to `6e-4`.
- **Pure Muon (no AdamW).** Breaks on the embedding: orthogonalizing a `32768 × 512` matrix with sparse per-row gradients is both wrong and slow, and 1D norms cannot be orthogonalized at all. The hybrid is not optional.
- **Shampoo / full-matrix preconditioning.** Strictly more expressive than Muon and strictly more expensive: you must form and invert (or root) the preconditioner, and keep its state. Muon is the "just the orthogonal factor, via matmuls only" corner of that design space, and at 100M the extra expressiveness does not pay for its wall-clock.
- **Cosine instead of WSD.** Works, but forces you to commit to a step count and gives no clean data-annealing hook. Since the capstone's entire mid-training story ([Chapter 14.8](../14-capstone/08-mid-training.html)) rides on the decay phase, WSD is the natural fit. A useful mental model: **WSD is cosine with the middle stretched into a flat plateau you can extend at will.**
- **No QK-clip.** Defensible here, and worth saying out loud: with QK-norm on, the logits are bounded a priori by $\sqrt{d_h}\lVert\gamma_q\rVert_\infty\lVert\gamma_k\rVert_\infty$, so the run is already structurally safe. We keep the clip because its trigger log is a *free instability sensor* — a rising `qk_fired` rate is the earliest warning that the LR is too high. Without QK-norm, it is not optional; it is the thing standing between you and a NaN at hour 14.
- **Scaling out.** If you move past one GPU, note that Muon's orthogonalization needs the **whole** matrix, which fights naive optimizer-state sharding. Moonshot's Moonlight release shows the fix: a ZeRO-1-style distributed Muon that gathers each matrix, orthogonalizes, and re-shards ([Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)). At 100M you will never need it — but it is the reason "just wrap it in FSDP" is not a complete answer.

!!! warning "Common pitfall: mismatched Muon/AdamW learning rates"
    The single most common way to get a bad Muon run is to forget the RMS-matching scale ($0.2\sqrt{\max(m,n)}$) and then reuse an AdamW-tuned learning rate directly. Without the scale, Muon's raw orthogonal update has RMS $\sim 1/\sqrt{\max(m,n)}$ — $0.027$ for a `1408×512` matrix — so a "normal" LR of `3e-3` barely moves the weights and the run looks dead. With the scale, `6e-3` on Muon and `3e-3` on AdamW live in comparable regimes. If Muon training stalls, check this scale *first*, before touching the LR.

!!! interview "Interview Corner"
    **Q:** You're training a 100M model with Muon on the weight matrices and AdamW on embeddings and norms. Why the split, and what stability problem does Muon introduce that you must mitigate?

    **A:** Muon orthogonalizes the momentum update of a 2D matrix — it replaces the update with the $UV^\top$ polar factor via ~5 Newton–Schulz iterations, so every singular direction gets an equal-magnitude push. That's a cheap spectral preconditioner that converges faster than Adam's per-coordinate rescaling on the dense attention and MLP matrices. It only makes sense for 2D "feature-mixing" matrices, though: the token embedding has row-sparse gradients (most rows untouched each batch) and 1D RMSNorm gains have no matrix structure, so both stay on AdamW for per-coordinate adaptive rates. I'd also mention the RMS-matching scale $0.2\sqrt{\max(m,n)}$ — it cancels the shape dependence of the orthogonal update and lands its per-element RMS at 0.2, which is where AdamW's *measured* update RMS actually sits, so the two groups can share one tuned LR up to a small fixed ratio.

    The stability problem is **attention-logit blow-up**: Muon's equal-direction updates let the query/key path grow until $\max q^\top k/\sqrt{d_h}$ saturates the softmax and NaNs the loss in bf16. The fix depends on your architecture, and this is the part people get wrong. Kimi K2's **MuonClip** rescales $W_Q$ and $W_K$ by $\sqrt{\tau/S_{\max}}$ after any step where a head's max logit exceeded $\tau$ — with GQA, the shared key slice is scaled once by its group's worst logit, not once per query head, or you'd shrink it four times over. But that only works if you *don't* have QK-norm. With QK-norm, RMSNorm is scale-invariant, so rescaling $W_Q$ is provably a no-op; the logit is bounded a priori by $\sqrt{d_h}\lVert\gamma_q\rVert_\infty\lVert\gamma_k\rVert_\infty$ and the only free scale is the learned gains — so you clip *those*. I'd close by noting that harvesting $S_{\max}$ isn't free either: the exact max forces you off FlashAttention and materializes a $(B,H,T,T)$ tensor, so in practice you either use the Cauchy–Schwarz bound $\max\lVert q\rVert\max\lVert k\rVert/\sqrt{d_h}$, which keeps the fused kernel, or you measure exactly but only every few hundred steps.

!!! key "Key Takeaways"
    - **Muon orthogonalizes the momentum update** of 2D weight matrices via ~5 Newton–Schulz iterations (a matmul-only approximation to the $UV^\top$ polar factor), setting all singular values to ~1 so the update pushes equally in every direction — faster per-token convergence than AdamW on attention/MLP matrices.
    - The standard **hybrid is mandatory**: Muon for the 210 2D hidden matrices, **AdamW for the tied embedding, all RMSNorm/QK-norm gains, and every 1D param**. Route by `p.ndim == 2` (and "not the embedding").
    - A **RMS-matching scale** of $0.2\sqrt{\max(m,n)}$ cancels the shape dependence and lands the update RMS at 0.2 — where AdamW's *measured* update RMS actually sits. Forget it and Muon appears "dead." Capstone peaks: Muon **6e-3**, AdamW **3e-3**.
    - **FLOP overhead ≠ wall-clock overhead.** Newton–Schulz is 0.33% of the step's FLOPs but ~4,200 kernel launches; `torch.compile` it or batch the three shape classes into `bmm`s (~60 launches), then *measure* `optimizer.step()`.
    - **Under QK-norm, rescaling $W_Q$/$W_K$ is a no-op** — RMSNorm is scale-invariant. The logit obeys $|s| \le \sqrt{d_h}\lVert\gamma_q\rVert_\infty\lVert\gamma_k\rVert_\infty$ (= 8 at init for `Stack-100M`), so the **QK-clip scales the learned gains** by $\sqrt{\tau/S_{\max}}$. Kimi K2's per-head, GQA-aware $W_Q/W_K$ clip is the *alternative* you use when QK-norm is off.
    - **Measuring $S_{\max}$ costs real money**: the exact max forces eager attention and a $(B,H,T,T)$ tensor (~4.3 GB/layer at micro-batch 32). Default to the Cauchy–Schwarz bound (SDPA-safe) and gate the whole thing behind `qk_clip_every = 200`. Set $\tau = 30$, not Kimi's 100, because QK-norm shrinks the natural logit scale.
    - **WSD (Warmup–Stable–Decay)** replaces cosine: linear warmup, a long **constant-LR stable phase you can extend at will**, then a $1-\sqrt{}$ **decay to ~0** — and clamp `progress` to 1.0 or you will return a negative LR past the horizon.
    - The **decay phase is the mid-training annealing phase** ([Chapter 14.8](../14-capstone/08-mid-training.html)): low LR + higher-quality data yields a sharp loss drop for little compute — a hook cosine can't cleanly provide.
    - Frozen `Stack-100M` numbers: **524,288-token batch** (32 × 2048 × 8), **38,147 steps**, **500 warmup / 31,647 stable / 6,000 decay**, weight decay 0.1 (0.0 on 1D), global grad-clip 1.0, betas (0.9, 0.95), bf16 with no loss scaler.

!!! sota "State of the Art & Resources (2026)"
    Muon (with the Moonshot RMS-matching fix) and QK-clip have moved from nanoGPT speedrun tricks to production ingredients at frontier scale, and WSD is now a common default for schedule-agnostic pretraining — the capstone's stack mirrors what Kimi K2, Moonlight, and DeepSeek actually shipped. The live research frontier is making orthogonalized updates *distributed-friendly* (Moonlight's ZeRO-1 Muon; follow-ups such as Microsoft's **Dion**, 2025, which explores communication-efficient orthonormalized updates) and pinning down how Muon's optimal LR transfers across width and batch.

    **Foundational work**

    - [Gupta, Koren & Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018)](https://arxiv.org/abs/1802.09568) — the full-matrix preconditioner Muon cheaply approximates via Newton–Schulz.
    - [McCandlish, Kaplan & Amodei, *An Empirical Model of Large-Batch Training* (2018)](https://arxiv.org/abs/1812.06162) — the gradient-noise-scale / critical-batch-size argument behind the ~0.5M-token effective batch.
    - [Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2019)](https://arxiv.org/abs/1711.05101) — the AdamW paper; Muon's decoupled weight decay follows the same recipe.

    **Recent advances (2023–2026)**

    - [Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024)](https://arxiv.org/abs/2404.06395) — introduces the Warmup–Stable–Decay schedule and its continuable-pretraining property.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — a frontier-scale run using a WSD-style multi-stage schedule.
    - [Liu et al. (Moonshot AI), *Muon is Scalable for LLM Training* (2025)](https://arxiv.org/abs/2502.16982) — the Moonlight report: decoupled weight decay for Muon, the update-RMS matching constant, and distributed (ZeRO-1) Muon.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — introduces MuonClip and the per-head QK-clip that tamed attention-logit blow-up under Muon at trillion-parameter scale.

    **Open-source & tools**

    - [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — the reference PyTorch `Optimizer` implementation this chapter's `muon.py` follows.
    - [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — the speedrun repo where Muon, QK-norm, and related tricks were stress-tested step by step.
    - [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) — open-source distributed Muon plus checkpoints from the "Muon is Scalable" report.
    - [karpathy/nanochat](https://github.com/karpathy/nanochat) — a small full-stack LLM repo (2025) that uses the same Muon-for-matrices / AdamW-for-embeddings hybrid, a useful second reference implementation.
    - `torch.optim.AdamW(fused=True)`, `torch._foreach_*`, `torch.compile`, and `torch.nn.attention.flex_attention` — the PyTorch primitives this chapter leans on for a fast optimizer step and a fused soft-cap.

    **Go deeper**

    - [Keller Jordan, *Muon: An optimizer for hidden layers in neural networks* (blog, 2024)](https://kellerjordan.github.io/posts/muon/) — the original writeup deriving Newton–Schulz orthogonalization and the nanoGPT speedrun results.

## Further Reading

- **Keller Jordan et al.**, *Muon: An optimizer for the hidden layers of neural networks* (2024) — the original Muon and Newton–Schulz orthogonalization; the nanoGPT speedrun context.
- **Liu et al. (Moonshot AI)**, *Muon is Scalable for LLM Training* (2025) — decoupled weight decay for Muon, the update-RMS matching constant, and ZeRO-1 distributed Muon.
- **Kimi K2 Technical Report** (Moonshot AI, 2025) — MuonClip and the per-head QK-clip mechanism for attention-logit stability at scale.
- **Henry et al.**, *Query-Key Normalization for Transformers* (2020) — the QK-norm whose scale-invariance is the reason this chapter's clip targets the gains, not the projections.
- **Hu et al.**, *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024) — the Warmup–Stable–Decay schedule and its continuable-pretraining / annealing properties.
- **DeepSeek-V2 / V3 technical reports** (DeepSeek-AI, 2024) — WSD-style multi-step schedules used at frontier scale.
- **McCandlish, Kaplan, Amodei et al.**, *An Empirical Model of Large-Batch Training* (2018) — the critical batch size that justifies the ~0.5M-token effective batch.
- **Kingma & Ba**, *Adam: A Method for Stochastic Optimization* (2015) and **Loshchilov & Hutter**, *Decoupled Weight Decay Regularization (AdamW)* (2019) — the baseline this whole stack extends.
- **Gupta, Koren & Singer**, *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018) — the full-preconditioner method Muon cheaply approximates.
- **Yang & Hu et al.**, *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (muP, 2022) — the principled alternative to this chapter's empirical LR-transfer protocol.

## Exercises

**1.** In `build_optimizers()` the tied token embedding (`32768 x 512`) is routed to AdamW even though it is a 2D tensor and would pass a naive `p.ndim == 2` test. Explain why orthogonalizing this matrix with Muon is both *wrong* and *wasteful*, and name the property of embedding gradients that makes AdamW the right choice.

??? note "Solution"
    Muon's orthogonalization only makes sense when both axes of a matrix are "feature" axes that mix together, because setting every singular value to 1 spreads the update equally across all directions of the matrix. The embedding's row axis is the **token axis** (32,768 entries), not a feature axis. In any given batch only a tiny fraction of the 32,768 rows are touched at all — most tokens do not appear — so the gradient is **extremely sparse per row**. Orthogonalizing across the token axis mixes an untouched row's (zero) gradient direction with the handful of active rows, which is semantically meaningless: there is no shared low-dimensional structure across arbitrary vocabulary rows to precondition.

    It is also wasteful: the Newton–Schulz iteration would run on a `32768 x 512` matrix. Iterating on the 512 side, each of the 5 iterations costs $2\cdot512^2\cdot32768 + 2\cdot512^3 + 2\cdot512^2\cdot32768 \approx 3.5\times10^{10}$ FLOP, so about **173 GFLOP per optimizer step** for this single tensor — a 17% surcharge on top of the 1.04 TFLOP that *all 210* hidden matrices cost combined, spent to produce a direction that is wrong anyway.

    What the embedding actually wants is a **per-row adaptive learning rate** so that a rarely-seen token's row is not swamped by the scale of frequent tokens. That is exactly what AdamW's per-coordinate second-moment rescaling gives. Hence the hybrid: 2D *hidden* feature-mixing matrices to Muon, embedding/norms/1D to AdamW.

**2.** The RMS-matching scale is $0.2\sqrt{\max(m,n)}$. Take the SwiGLU gate weight `w_gate` of shape `1408 x 512`. (a) What is the root-mean-square element magnitude of the raw orthogonalized update $O = UV^\top$ *before* scaling? (b) Compute the scale factor. (c) Confirm the post-scale RMS. (d) The constant is 0.2, not 1.0 — what empirical fact about AdamW does that number encode, and why does it matter for the learning rate?

??? note "Solution"
    Let $m = 1408$, $n = 512$, so $\min(m,n) = 512$ and $\max(m,n) = 1408$.

    **(a)** A semi-orthogonal $m\times n$ matrix has $\min(m,n) = 512$ singular values equal to 1, so its Frobenius norm is $\sqrt{512}$. That energy is spread over $mn = 1408 \times 512 = 720{,}896$ entries, so
    $$
    \text{RMS}(O) = \sqrt{\frac{\min(m,n)}{mn}} = \sqrt{\frac{512}{720{,}896}} = \frac{1}{\sqrt{1408}} \approx 0.02665.
    $$

    **(b)** The scale is
    $$
    0.2\sqrt{\max(m,n)} = 0.2\sqrt{1408} = 0.2 \times 37.523 \approx 7.505.
    $$

    **(c)** Post-scale RMS $= 0.02665 \times 7.505 \approx 0.200$. (Cleanly: $\tfrac{1}{\sqrt{1408}} \cdot 0.2\sqrt{1408} = 0.2$ exactly.)

    **(d)** The idealized AdamW update $m/(\sqrt{v}+\epsilon)$ has per-element magnitude $\approx 1$ *only* when $m$ and $v$ are estimated over the same window on a stationary gradient. Real LLM training violates both assumptions ($\beta_1 = 0.9$ vs $\beta_2 = 0.95$, and a highly non-stationary gradient), and the **measured** update RMS across a transformer's matrices comes out substantially smaller — on the order of 0.2–0.4, which is where Moonshot's constant of 0.2 comes from. So the constant is empirical, not theoretical.

    Why it matters: the scale first **cancels the shape dependence** (without it, `1408×512` and `128×512` matrices would move by different relative amounts under the same LR), and then **puts Muon's step in AdamW's measured band**, so you tune one number and derive the other with a small fixed ratio (here 2:1) instead of searching a 2D grid. Forget the scale entirely and Muon's raw update has RMS $\approx 0.027$ — a normal LR barely moves the weights and training looks dead.

**3.** Suppose you re-budget the run to **30B tokens** at the same **524,288-token** effective batch, keep `warmup_steps = 500`, and want the decay phase to cover the same *fraction* of the run as the flagship (6,000 / 38,147). (a) How many total optimizer steps? (b) Give the warmup / stable / decay step counts. (c) Using the $1-\sqrt{\cdot}$ decay with `peak_lr = 6e-3`, what Muon LR is in effect at **one quarter** of the way through the decay phase, and what is the corresponding AdamW LR?

??? note "Solution"
    **(a)** Total steps:
    $$
    T = \left\lceil\frac{30\times10^{9}}{524{,}288}\right\rceil = \left\lceil 57{,}220.5 \right\rceil = 57{,}221 \text{ steps.}
    $$

    **(b)** The flagship decay fraction is $6{,}000/38{,}147 = 0.1573$. Applied to 57,221: decay $= 0.1573 \times 57{,}221 \approx 9{,}000$ steps. Warmup stays at 500. Stable $= 57{,}221 - 500 - 9{,}000 = 47{,}721$ steps.

    (Equivalently, call `wsd_lr(..., total_steps=57_220, decay_steps=9_000)` — the `decay_steps` argument exists precisely so you can state this directly instead of re-deriving a fraction.)

    **(c)** At progress $= 0.25$ through decay:
    $$
    \text{decay\_mult} = 1 - \sqrt{0.25} = 1 - 0.5 = 0.5,
    $$
    so (with `final_frac = 0`) $\eta_{\text{Muon}} = 6\text{e-}3 \times 0.5 = 3\text{e-}3$, and $\eta_{\text{AdamW}} = 0.5 \times \eta_{\text{Muon}} = 1.5\text{e-}3$. Note the LR is already halved only a quarter of the way in — the $1-\sqrt{\cdot}$ shape drops fast early, which is what produces WSD's characteristic sharp loss drop right as decay (and premium-data annealing) begins.

**4.** *(The no-QK-norm configuration.)* You set `qk_norm = False` to reproduce Kimi K2's setup, so `qk_clip_` takes the `_clip_projections_` path. A block has `n_heads = 8` query heads and `n_kv_heads = 2` (GQA group size 4), with $\tau = 100$. The per-head max pre-softmax logits recorded this step are

    s_max = [120, 90, 100, 80, 150, 60, 70, 200]   # heads 0..7

(a) Which query heads get their $W_Q$ slice scaled, and by what factor? (b) What single factor scales each shared $W_K$ group? (c) Verify that the head that *set* group 1's maximum lands exactly at $\tau$, and show that head 4 (also in group 1) ends up *below* $\tau$. (d) What would go wrong if you instead scaled $W_K$ once per *query* head?

??? note "Solution"
    Group 0 = heads 0–3, group 1 = heads 4–7. The per-weight scale is $\eta = \sqrt{\tau / S_{\max}}$ (the square root, because the logit is bilinear in $W_Q$ and $W_K$, so scaling *both* multiplies the logit by the full ratio $\tau/S_{\max}$).

    **(a) $W_Q$, per query head** (only heads with $S_{\max} > \tau$; head 2 at exactly 100 is *not* $> 100$, so it is untouched):

    - head 0: $\sqrt{100/120} = 0.9129$
    - head 4: $\sqrt{100/150} = 0.8165$
    - head 7: $\sqrt{100/200} = 0.7071$
    - heads 1, 2, 3, 5, 6: untouched.

    **(b) $W_K$, once per group, using the group's worst logit:**

    - group 0: $\max(120,90,100,80) = 120 \Rightarrow \eta = \sqrt{100/120} = 0.9129$
    - group 1: $\max(150,60,70,200) = 200 \Rightarrow \eta = \sqrt{100/200} = 0.7071$

    **(c)** Head 7 set group 1's max. Its logit is scaled by $\eta_{W_Q} \cdot \eta_{W_K} = 0.7071 \times 0.7071 = 0.5$, giving $200 \times 0.5 = 100 = \tau$ exactly. Head 4 shares group 1's $W_K$ ($0.7071$) but keeps its own $W_Q$ scale ($0.8165$): its logit becomes $150 \times 0.8165 \times 0.7071 = 150 \times 0.5774 \approx 86.6 < \tau$. So the group-setting head lands on the cap and the other members are pulled conservatively below it.

    **(d)** Group 1 contains two tripping heads (4 and 7). Scaling $W_K$ once per tripping query head would multiply that KV slice by $0.8165 \times 0.7071 = 0.5774$ instead of $0.7071$ — a 18% over-shrink. In the worst case (all four query heads in a group tripping) the shared slice would be multiplied four times, e.g. $0.7071^4 = 0.25$, gutting a key projection that four heads depend on and taking several hundred steps to recover. That is the whole reason for the two-loop structure.

**5.** The chapter stresses WSD's *continuable* property: you branch multiple decay runs off the same **stable** checkpoint, and the decay length is not tied to any pre-committed total. `wsd_lr()` now accepts `decay_steps` directly — but it still computes `stable_end = total_steps - decay_steps`, so a horizon is baked in. Implement `wsd_lr_branch()` that takes `stable_steps` and `decay_steps` **directly**, with no `total_steps` at all, so you can spawn a decay of any length from a checkpoint saved at the end of the stable phase. Keep the linear warmup, the constant stable phase, and the $1-\sqrt{\cdot}$ decay to `final_frac * peak_lr`.

??? note "Solution"
    The fix is to stop referencing a global horizon and instead pass the three phase lengths independently. This makes the stable checkpoint a genuine fork point: save at `warmup_steps + stable_steps`, then launch any number of decay runs, each with its own `decay_steps`, all reading the same weights — which is exactly the experiment [Chapter 14.8](../14-capstone/08-mid-training.html) runs when comparing annealing mixes.

    ```python
    # stacklm/optim/schedule.py  (continued)
    import math

    def wsd_lr_branch(step: int, *, peak_lr: float, warmup_steps: int,
                      stable_steps: int, decay_steps: int,
                      final_frac: float = 0.0) -> float:
        """WSD LR with phase lengths given DIRECTLY (no baked-in total horizon).

        Decouples the decay from any pre-committed step count, so multiple decay
        runs of different lengths can branch off one stable-phase checkpoint
        (MiniCPM's continuable-pretraining property).
        """
        stable_end = warmup_steps + stable_steps          # first decay step
        if step < warmup_steps:                           # --- warmup ---
            return peak_lr * (step + 1) / warmup_steps
        if step < stable_end:                             # --- stable ---
            return peak_lr
        # --- decay: 1 - sqrt(progress), clamped to [0, 1] ---
        progress = (step - stable_end) / max(1, decay_steps)
        progress = min(progress, 1.0)                     # stay at the floor past the end
        decay_mult = 1.0 - math.sqrt(progress)            # 1 -> 0
        floor = final_frac * peak_lr
        return floor + (peak_lr - floor) * decay_mult
    ```

    Two things to notice. First, `total_steps` never appears — the schedule is fully determined by the three phase lengths, so extending the stable phase or trying a longer/shorter decay is a config change, not a re-derivation. Second, the `min(progress, 1.0)` clamp: without it, any step past the end of decay gives $\sqrt{\text{progress}} > 1$, `decay_mult` goes negative, and you return a **negative learning rate** — which does not error, it silently performs *gradient ascent*. (This is why the main-text `wsd_lr` carries the same clamp.) To reproduce the flagship recipe, call it with `warmup_steps = 500`, `stable_steps = 31_647`, `decay_steps = 6_000`.

**6.** You want a sanity check that `zeropower_via_newtonschulz5` really conditions the update. Write a short script that builds an ill-conditioned random `256 x 128` matrix (singular values spanning a couple orders of magnitude, condition number $\sim100$), runs the orthogonalizer, and prints the singular-value spread of the result. Explain what you expect to see and why the chapter says we do *not* need machine-precision orthogonality.

??? note "Solution"
    Build a matrix with a deliberately bad spectrum via an SVD with geometrically spaced singular values, orthogonalize, then read back the singular values of the output with `torch.linalg.svdvals`.

    ```python
    import torch
    from stacklm.optim.muon import zeropower_via_newtonschulz5

    torch.manual_seed(0)
    m, n = 256, 128
    # Random orthonormal U, V and a bad spectrum from 1e-1 to 1e1 (cond ~1e2).
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    s = torch.logspace(-1, 1, n)                      # 1e-1 ... 1e1, span 1e2
    G = (U[:, :n] * s) @ V.T                          # ill-conditioned (m x n)

    print("input  sigma: min %.3e  max %.3e  cond %.3e"
          % (s.min(), s.max(), s.max() / s.min()))

    O = zeropower_via_newtonschulz5(G, steps=5).float()
    sv = torch.linalg.svdvals(O)
    print("output sigma: min %.4f  max %.4f  cond %.4f"
          % (sv.min(), sv.max(), sv.max() / sv.min()))
    ```

    **Output (fp32 iteration, as in `capstone/stacklm/optim/muon.py`):**

    ```text
    input  sigma: min 1.000e-01  max 1.000e+01  cond 1.000e+02
    output sigma: min 0.6818     max 1.2023     cond 1.7634
    ```

    The input condition number is $\sim 10^2$; after 5 Newton–Schulz steps every singular value has been pulled toward 1, so the output's spread collapses to roughly $[0.68, 1.20]$ and its condition number drops from $\sim100$ to under 2. That is the whole point of orthogonalization — turn a spectrum dominated by a few large directions into a nearly isotropic one so the update pushes equally along all directions of the matrix. Notice the largest singular values slightly **overshoot** 1 (to $\approx1.2$): the tuned quintic is designed to converge *fast*, not monotonically, so it rings a little around 1 rather than approaching it from below. (In bf16 the last digits move; the shape does not.)

    Why not more steps for exact orthogonality? Because we only need a **well-conditioned direction**, not $\sigma = 1$ to machine precision. The optimizer immediately multiplies the result by $0.2\sqrt{\max(m,n)}$ and the learning rate; the residual $\pm20\%$ deviation from perfect orthogonality is utterly washed out by that. Five matmul-only iterations are far cheaper than an exact SVD and give a direction as good as the optimization needs. (Push the input condition number to $10^5$ and 5 steps would *not* fully recover the tiniest singular values — Newton–Schulz converges only linearly near 0 — but real momentum buffers are nowhere near that ill-conditioned.)

**7.** *(The invariance, from scratch.)* Prove that with QK-norm enabled, replacing $W_Q \leftarrow \eta W_Q$ leaves every attention logit unchanged (ignore $\epsilon$), and that replacing $\gamma_q \leftarrow \eta\gamma_q$ multiplies every logit by exactly $\eta$. Then: `Stack-100M` has $d_h = 64$ and shares one gain vector per layer across all 8 query heads. (a) State the a-priori bound on $|s_{ij}|$ at initialization. (b) A layer's worst head reports $S_{\max} = 45$ with $\tau = 30$. What factor does `_clip_qk_norm_gains_` apply to $\gamma_q$ and to $\gamma_k$, and what happens to a head in the same layer that was sitting at $S_{\max} = 12$? (c) Why is that acceptable, and what one-line architecture change would make the clip per-head instead?

??? note "Solution"
    **Proof.** With $z = W_Q x$, the query is $q = \gamma_q \odot z/\operatorname{rms}(z)$ where $\operatorname{rms}(z) = \sqrt{\frac{1}{d_h}\sum_i z_i^2}$. Replace $W_Q$ by $\eta W_Q$: then $z \mapsto \eta z$ and $\operatorname{rms}(z) \mapsto \eta\operatorname{rms}(z)$, so the ratio $z/\operatorname{rms}(z)$ is unchanged and $q \mapsto q$. RoPE is applied afterwards and is a rotation, hence linear and norm-preserving, so it cannot reintroduce the scale. Therefore $s_{ij} = q_i^\top k_j/\sqrt{d_h}$ is unchanged — **the projection rescale is a no-op**. By contrast $\gamma_q \mapsto \eta\gamma_q$ gives $q \mapsto \eta q$ directly (the gain multiplies *after* normalization), so $s_{ij} \mapsto \eta s_{ij}$ exactly. Scaling both gains by $\sqrt{\tau/S}$ therefore multiplies every logit by exactly $\tau/S$.

    **(a)** Since $u = z/\operatorname{rms}(z)$ has unit RMS, $\lVert u\rVert_2 = \sqrt{d_h}$, so $\lVert q\rVert_2 \le \lVert\gamma_q\rVert_\infty\sqrt{d_h}$ and likewise for $k$. Cauchy–Schwarz gives
    $$
    |s_{ij}| \le \frac{\lVert q\rVert\lVert k\rVert}{\sqrt{d_h}} \le \sqrt{d_h}\,\lVert\gamma_q\rVert_\infty\lVert\gamma_k\rVert_\infty .
    $$
    At initialization $\gamma = \mathbf{1}$, so the bound is $\sqrt{64} = \mathbf{8}$.

    **(b)** $\eta = \sqrt{\tau/S_{\max}} = \sqrt{30/45} = \sqrt{2/3} \approx 0.8165$, applied to **both** $\gamma_q$ and $\gamma_k$. The worst head's logit becomes $45 \times 0.8165^2 = 45 \times 2/3 = 30 = \tau$ exactly. The quiet head at 12 shares the same layer-wide gains, so it drops to $12 \times 2/3 = 8$ — well below the cap, through no fault of its own.

    **(c)** It is acceptable because the intervention is (i) *conservative* — shrinking a healthy head's temperature reduces sharpness, it never destabilizes; (ii) *rare* — with $\tau = 30$ against an init bound of 8, it fires only when gains have genuinely drifted; and (iii) *reversible* — the gain is a trained parameter on AdamW with zero weight decay, so a head that wants more temperature simply regrows it over the next few hundred steps. The one-line change for per-head granularity is to allocate the gains per head:

    ```python
    # Ch. 14.4 Attention.__init__, per-head QK-norm gains.
    # q, k are (B, n_heads, T, head_dim) at this point, so the gain must carry a
    # singleton time axis to broadcast correctly: (n_heads, 1, head_dim).
    self.q_norm = RMSNorm((self.n_heads,    1, self.head_dim), cfg.norm_eps)
    self.k_norm = RMSNorm((self.n_kv_heads, 1, self.head_dim), cfg.norm_eps)
    ```

    (`RMSNorm` still reduces over the last axis; only the shape of `self.weight` changes, from `torch.ones(dim)` to `torch.ones(*shape)`.) This costs $8{\cdot}64 + 2{\cdot}64 = 640$ parameters per layer instead of 128 — 15,360 extra across 30 layers, 0.015% of the model. With per-head gains, the GQA logic from `_clip_projections_` transfers verbatim: scale $\gamma_q$ per query head, and each shared $\gamma_k$ once by its group's worst logit. Note that `k_norm` is applied *before* `repeat_interleave`, so its gain is indexed by KV head — exactly the granularity the GQA argument needs.

**8.** *(Cost accounting.)* Your training loop records the exact per-head max logit (Tier 2) on **every** micro-batch of **every** step, with `micro_batch_size = 32`, `seq_len = 2048`, `n_heads = 8`, `n_layers = 30`, `grad_accum_steps = 8`, in fp32. (a) How much transient memory does the score tensor take *per layer*, and why does this not simply add up across 30 layers? (b) Roughly how many extra FLOPs per optimizer step does the score matmul cost, versus the model's $6ND \approx 3.19\times10^{14}$? (c) Give two ways to cut this by more than 100× and state what each gives up.

??? note "Solution"
    **(a)** The score tensor is $(B, H, T, T) = (32, 8, 2048, 2048)$ in fp32:
    $$
    32 \times 8 \times 2048 \times 2048 \times 4\ \text{bytes} = 4.29\ \text{GB}.
    $$
    It does *not* accumulate across layers because it is built inside `torch.no_grad()` and dropped as soon as `amax` is taken, so at most one (well, transiently two, counting the `masked_fill` copy) lives at a time. The real damage is different: to build it at all you must take the eager branch, which means you have given up FlashAttention's memory savings *for that pass* and pay a large allocator spike that can fragment the 80GB pool and OOM a run that was otherwise comfortable.

    **(b)** The $QK^\top$ product is $2BHT^2d_h$ FLOP per layer:
    $$
    2 \times 32 \times 8 \times 2048^2 \times 64 \approx 1.37\times10^{11}\ \text{FLOP/layer},
    $$
    times 30 layers $\approx 4.1\times10^{12}$ per micro-batch, times 8 micro-batches $\approx 3.3\times10^{13}$ FLOP per optimizer step. Against $3.19\times10^{14}$ that is roughly **10%** — an order of magnitude more than Muon's entire orthogonalization cost, spent on a diagnostic. (And that ignores the masking, the `amax` reduction, and the lost kernel fusion, so the wall-clock hit is worse than 10%.)

    **(c)** Two cuts, composable:

    1. **Gate it in time.** `qk_clip_every = 200` and a single probe micro-batch instead of all 8: the cost falls by $200 \times 8 = 1600\times$, to well under 0.01%. What you give up: your $S_{\max}$ reading is up to 200 steps stale. That is fine — the gains move by $\eta \approx 6\text{e-}3$ times a bounded update per step, so logit drift over 200 steps is small compared to the gap between the init bound (8) and $\tau$ (30).
    2. **Change the estimator.** Use the Tier-1 Cauchy–Schwarz bound $\max_i\lVert q_i\rVert\max_j\lVert k_j\rVert/\sqrt{d_h}$, which is $O(BHTd_h)$ — about $1/T \approx 5\times10^{-4}$ of the score matmul's cost — and keeps the SDPA/FlashAttention fast path entirely. What you give up: it is an *upper* bound, so the clip may fire on a layer whose true max was below $\tau$. That error is conservative and, since Cauchy–Schwarz is loose only when $q$ and $k$ are far from aligned, usually small.

    In combination the diagnostic costs less than a rounding error, which is the only way a safety net earns its place in a 27-GPU-hour run.
