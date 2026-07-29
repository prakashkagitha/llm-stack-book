# 14.6 Optimizer & Schedule: Muon + MuonClip and Warmup-Stable-Decay

Every choice we have made so far — the deep-and-thin `Stack-100M` architecture of [Chapter 14.4](../14-capstone/04-architecture.html), the ~20B-token over-training budget, the FineWeb-Edu/Cosmopedia data mix — only pays off if the optimizer can actually walk the loss down cleanly across roughly **40,000 steps** without spiking, and if the learning-rate schedule leaves a hook for the mid-training annealing phase to come. This chapter fixes both. We adopt the exact optimizer stack the capstone spec pins in §5: **Muon** (Jordan et al., 2024) for the 2D hidden weight matrices, **AdamW** for embeddings, norms, and every 1D parameter, **MuonClip / QK-clip** (Kimi K2, Moonshot 2025) for attention-logit stability, and the **Warmup-Stable-Decay (WSD)** schedule (MiniCPM, Hu et al., 2024; used by DeepSeek). Every number here is the one you will type into `stacklm/config.py`.

This is the applied, capstone-specific companion to two deeper reference chapters you should have open in another tab: [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) derives Adam and Muon from first principles, and [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html) covers warmup, cosine, batch/LR coupling, and muP. We will not re-derive those; we will *use* them, and explain the two or three things that are specific to training a 100M model on a single A100 to 200 tokens per parameter. When something threatens to blow the run up, we lean on [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

## Why Muon, and Why a Hybrid

Start with the memory arithmetic, because at 100M parameters it is not the binding constraint — and that itself is the lesson. AdamW keeps two full-precision state tensors per parameter, the first moment $m$ and second moment $v$. For `Stack-100M`'s ~101M parameters that is `101M × 2 × 4 bytes ≈ 0.8 GB` of optimizer state, trivial on an 80GB A100. So unlike a 70B run, we are **not** choosing Muon to save memory. We are choosing it because, at fixed compute, it walks the loss down measurably faster on the 2D weight matrices that dominate a transformer — the attention projections and the SwiGLU MLP. Muon has driven a series of speedrun records on nanoGPT-scale models and was used at frontier scale by **Kimi K2** (Moonshot, 2025). For a project whose whole thesis is "spend training compute once, serve forever," a faster optimizer is free perplexity.

The core idea, developed carefully in [Chapter 3.9](../03-pretraining/09-optimizers.html), is **orthogonalization of the momentum update**. Adam rescales each coordinate independently by its running gradient magnitude; it never looks at the *matrix structure* of a weight tensor. Muon does. It takes the momentum buffer $B_t$ for a weight matrix $W \in \mathbb{R}^{m\times n}$ and replaces it with the nearest **semi-orthogonal** matrix — the $UV^\top$ from its singular value decomposition $B_t = U\Sigma V^\top$, which is exactly the orthogonal factor in the polar decomposition. Intuitively, the raw momentum update is often dominated by a few large singular directions; the network learns fast along those and starves the rest. Orthogonalizing sets **every singular value to 1**, so the update pushes equally hard along all directions of the matrix. This is a spectral condition-number fix, and it is why Muon behaves like a cheap approximation to a full matrix preconditioner (Shampoo, Gupta et al., 2018) without the cost of forming and inverting the preconditioner.

{{fig:muon-orthogonalization-spectrum}}

Computing an SVD every step of every layer would be far too slow. Muon's key engineering move is to approximate the orthogonal factor with a fixed number (typically 5) of **Newton–Schulz iterations** — a matrix polynomial recurrence that converges to $(B B^\top)^{-1/2} B \approx U V^\top$ using only matrix multiplies, which are exactly what GPUs are fastest at. We implement it below.

### Why not Muon on everything?

Muon's orthogonalization only makes sense for a matrix whose two dimensions are both "feature" axes that mix. Three parameter groups in `Stack-100M` fail that test and stay on **AdamW**:

- **The token embedding** (`32768 × 512`, tied to the output head per Press & Wolf 2017). Its rows are per-token and extremely sparse in the gradient — most tokens in a batch touch a tiny fraction of rows. Orthogonalizing across the 32,768-token axis is meaningless; you want per-row adaptive rates, which is exactly Adam.
- **RMSNorm gains and any 1D vector.** A 1D tensor has no matrix structure to orthogonalize.
- **Biases** — `Stack-100M` has essentially none (SwiGLU and RMSNorm are bias-free), but if present they go to AdamW.

This split — **Muon for 2D hidden matrices, AdamW for embeddings/norms/1D** — is not a compromise; it is the standard hybrid used at scale, including Kimi K2. The capstone commits to it. The one subtlety is *matching the two optimizers' effective step size* so a single conceptual learning rate governs the run; we handle that next.

```text
Stack-100M parameter routing
┌─────────────────────────────────────────────┬───────────┐
│ parameter group                              │ optimizer │
├─────────────────────────────────────────────┼───────────┤
│ tok_emb.weight  (32768×512, tied)            │  AdamW    │
│ blocks.*.attn.wq / wk / wv / wo  (2D)        │  Muon     │
│ blocks.*.mlp.w_gate / w_up / w_down (2D)     │  Muon     │
│ blocks.*.norm*.weight (1D RMSNorm gains)     │  AdamW    │
│ final_norm.weight (1D)                        │  AdamW    │
└─────────────────────────────────────────────┴───────────┘
```

That is **7 two-dimensional matrices per block** (four attention projections + three SwiGLU matrices), all routed to Muon; everything else — the tied embedding and every RMSNorm gain — goes to AdamW. Keep that count in mind; it reappears when we estimate Muon's compute overhead.

## Muon: The Newton–Schulz Orthogonalizer

Let us build the numerical core first, because everything else hangs off it. Given the momentum buffer $B \in \mathbb{R}^{m\times n}$, we want an orthogonal factor $O$ with $O = U V^\top$ where $B = U\Sigma V^\top$. The quintic Newton–Schulz iteration Keller Jordan uses starts from a spectrally-normalized $X_0 = B / \lVert B \rVert_F$ (so all singular values land in $[0,1]$) and applies

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
    matmuls, so it runs fast in bf16 on tensor cores. We deliberately do NOT
    aim for exact orthogonality -- 5 steps is enough to give a well-conditioned
    update direction.
    """
    assert G.ndim == 2, "Newton-Schulz orthogonalization is only for 2D matrices"
    a, b, c = 3.4445, -4.7750, 2.0315          # tuned quintic coefficients
    X = G.bfloat16()                            # compute in bf16; matmul-bound
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

Two scale factors turn the orthogonal direction into a usable update. First, the orthogonalized update has a root-mean-square element magnitude of roughly $1/\sqrt{\max(m,n)}$: a semi-orthogonal $m\times n$ matrix has $\min(m,n)$ singular values equal to 1, hence Frobenius norm $\sqrt{\min(m,n)}$, spread over $mn$ entries, so RMS $=\sqrt{\min(m,n)/(mn)} = 1/\sqrt{\max(m,n)}$. Second, to let the **same learning rate** govern both Muon and AdamW parameters — AdamW's per-element update has RMS $\approx 1$ — the Moonshot "Muon is Scalable" report (Liu et al., 2025) multiplies the Muon update by $0.2\cdot\sqrt{\max(m,n)}$, which brings its RMS to $\approx 0.2$, the same ballpark as a well-behaved AdamW step. That single trick is what lets us tune *one* peak LR for the whole run.

{{fig:muon-rms-matching}}

```python
# stacklm/optim/muon.py  (continued)
from torch.optim.optimizer import Optimizer

class Muon(Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalization, for 2D matrices ONLY.
    Route embeddings/norms/1D params to AdamW instead (see build_optimizers()).

    Args mirror the Jordan et al. reference plus the Moonshot RMS-matching scale
    and decoupled weight decay (Liu et al., 2025)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
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
                # RMS-match to AdamW so one LR governs both (Moonshot 2025)
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

## MuonClip / QK-Clip: Taming Attention Logits

Muon's aggressiveness has a failure mode that AdamW largely hides: because it pushes equally along every singular direction, the **query and key projections can grow until attention logits explode**. The attention score before softmax is $s_{ij} = q_i^\top k_j / \sqrt{d_h}$; if $\lVert W_Q\rVert$ and $\lVert W_K\rVert$ drift upward together, $\max_{ij} s_{ij}$ can climb into the hundreds, the softmax saturates to a one-hot, gradients through it vanish, and — in bf16 — you get a loss spike or an outright NaN ([Chapter 3.11](../03-pretraining/11-training-stability.html)). Kimi K2 hit exactly this at scale and introduced **MuonClip**, whose active ingredient is **QK-clip**: a per-head, post-step rescaling that caps the maximum attention logit.

The mechanism is simple and surgical. After the optimizer step, for each attention head $h$ we track the largest attention logit magnitude $S^{(h)}_{\max}$ observed in the just-completed forward pass. If it exceeds a threshold $\tau$ (Kimi K2 used $\tau=100$), we scale that head's query and key weights so the logit would have landed at $\tau$. Because the logit is bilinear in $W_Q$ and $W_K$ ($s \propto W_Q W_K^\top$), scaling **each** of them by $\sqrt{\tau / S_{\max}}$ multiplies the logit by exactly $\tau / S_{\max}$, pulling the worst case back to the cap. Heads below $\tau$ are untouched, so the clip is a rare, targeted intervention rather than a constant drag.

The one wrinkle is **GQA**: in `Stack-100M`, `n_heads = 8` query heads share `n_kv_heads = 2` key/value heads (a 4:1 ratio, [Chapter 14.4](../14-capstone/04-architecture.html) and [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)), so four query heads read the *same* $W_K$ slice. That means we must scale each $W_Q$ slice **per query head**, but each shared $W_K$ slice only **once** — otherwise a KV slice read by four tripping heads would be multiplied four times over, badly over-shrinking it. The clean fix is two separate loops: scale $W_Q$ per query head by its own $\sqrt{\tau/S_{\max}}$, then scale each $W_K$ group once using the **worst logit among the query heads that share it**.

{{fig:qk-clip-gqa}}

```python
# stacklm/optim/qk_clip.py
import torch

@torch.no_grad()
def qk_clip_(model, max_logits_per_head, tau: float = 100.0):
    """QK-clip (the core of MuonClip, Kimi K2 / Moonshot 2025), GQA-aware.

    For any attention head whose max pre-softmax logit exceeded tau this step,
    rescale W_Q (per query head) and W_K (per shared KV head) by sqrt(tau/S_max),
    so the offending logit is pulled back to ~tau. Bilinearity means scaling BOTH
    W_Q and W_K by the sqrt scales the logit by the full ratio.

    We scale each W_Q slice per query head, but each shared W_K slice only ONCE
    (using the strongest trigger in its group), so a KV head read by several
    tripping query heads is not multiplied repeatedly.

    `max_logits_per_head[layer_idx]` is a tensor of shape (n_heads,) recorded by
    the attention module during the forward pass (the running max of q@k^T/sqrt(d)).
    """
    for layer_idx, block in enumerate(model.blocks):
        s_max = max_logits_per_head[layer_idx]           # (n_heads,)
        attn = block.attn
        hd = attn.head_dim
        group = attn.n_heads // attn.n_kv_heads           # q-heads per kv-head (=4)

        # (1) Per-query-head scale on W_Q.
        for h in range(attn.n_heads):
            if s_max[h] <= tau:
                continue                                  # this head is fine
            eta = (tau / float(s_max[h])) ** 0.5          # sqrt so q AND k share it
            qs = slice(h * hd, (h + 1) * hd)
            attn.wq.weight[qs].mul_(eta)

        # (2) Per-KV-head scale on the SHARED W_K, using the group's worst logit.
        for kv in range(attn.n_kv_heads):
            s_grp = float(s_max[kv * group:(kv + 1) * group].max())
            if s_grp <= tau:
                continue
            eta = (tau / s_grp) ** 0.5
            ks = slice(kv * hd, (kv + 1) * hd)
            attn.wk.weight[ks].mul_(eta)
```

The head that *set* the group maximum lands exactly at $\tau$; the other three query heads in its group get their shared key scaled by the same amount and so are pulled slightly further below $\tau$ — a conservative, safe outcome. Recording `max_logits_per_head` costs one `.amax()` per layer inside the attention forward and is essentially free.

!!! tip "Practitioner tip: QK-norm first, QK-clip as a safety net"
    `Stack-100M` already applies **QK-norm** (RMSNorm on Q and K before attention) per the spec — that alone removes most logit blow-ups by bounding $\lVert q\rVert,\lVert k\rVert$. Treat QK-clip as the belt-and-suspenders backstop for the rare head that still drifts under Muon's aggressive updates. In practice at 100M you may see it fire a handful of times early in training and then never again. Log every trigger; a *rising* trigger rate late in training is a red flag that your LR is too high.

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

where $f$ decays from 1 to (near) 0 over the final phase. MiniCPM found a **$1-\sqrt{\cdot}$** or exponential-style decay works better than linear; we use a simple but effective form below. Two properties make WSD ideal for the capstone:

1. **The stable phase is schedule-agnostic in length.** You hold $\eta_{\max}$ constant for as long as your token budget allows. Because there is no baked-in endpoint, you can *decide to keep going* — extend the stable phase, or branch off multiple decay runs from the same stable checkpoint to compare data mixes. This is the "continuable pretraining" property MiniCPM highlighted.
2. **The decay phase is where the loss drops fastest — and that is precisely where we anneal on premium data.** Empirically WSD shows a characteristic sharp loss drop once decay begins. The capstone exploits this: the WSD **decay phase *is* the mid-training annealing phase** of [Chapter 14.8](../14-capstone/08-mid-training.html). We spend the constant-LR stable phase on the bulk FineWeb-Edu mix, then, as the LR decays, we shift the data toward higher-quality Cosmopedia, math, and code. Low LR + high-quality data = the model "polishes" on the best tokens without the high-LR thrash that would otherwise wash them out. Cosine cannot cleanly do this because its LR is already low through most of the middle of training.

```python
# stacklm/optim/schedule.py
import math

def wsd_lr(step: int, *, peak_lr: float, warmup_steps: int,
           total_steps: int, decay_frac: float = 0.2,
           final_frac: float = 0.0) -> float:
    """Warmup-Stable-Decay learning rate (MiniCPM, Hu et al. 2024).

    - Linear warmup for `warmup_steps`.
    - Constant `peak_lr` through the stable phase.
    - 1 - sqrt() decay over the final `decay_frac` of steps, down to
      `final_frac * peak_lr` (we use 0, i.e. anneal fully to ~0).

    Returns the *actual* LR (not a multiplier), so you can set it directly.
    """
    decay_steps = int(decay_frac * total_steps)
    stable_end = total_steps - decay_steps               # first decay step
    if step < warmup_steps:                              # --- warmup ---
        return peak_lr * (step + 1) / warmup_steps
    if step < stable_end:                                # --- stable ---
        return peak_lr
    # --- decay: 1 - sqrt(progress), progress in [0, 1] ---
    progress = (step - stable_end) / max(1, decay_steps)
    decay_mult = 1.0 - math.sqrt(progress)               # 1 -> 0
    floor = final_frac * peak_lr
    return floor + (peak_lr - floor) * decay_mult
```

We apply the *same* schedule shape to both optimizers, scaling each group's peak LR by its own base. For `Stack-100M` the peaks are **Muon `lr ≈ 0.02`** and **AdamW `lr ≈ 3e-3`** for the embedding/norm group (both illustrative, on the order of what the reference recipes use; the Moonshot RMS-matching keeps them in a comparable regime). The step-by-step wiring:

```python
# in the training loop (stacklm/train.py), once per optimizer step
lr_now_muon  = wsd_lr(step, peak_lr=0.02,  warmup_steps=WARMUP,
                      total_steps=TOTAL, decay_frac=0.2)
lr_now_adamw = wsd_lr(step, peak_lr=3e-3, warmup_steps=WARMUP,
                      total_steps=TOTAL, decay_frac=0.2)
for g in muon_opt.param_groups:  g["lr"] = lr_now_muon
for g in adamw_opt.param_groups: g["lr"] = lr_now_adamw
```

!!! note "Aside: why the decay must reach ~0"
    The loss drop in WSD's decay phase comes from the LR going genuinely small, letting the optimizer settle into a sharper minimum on the (now higher-quality) data. If you floor the LR at, say, $\eta_{\max}/10$ the way a cosine schedule often does, you leave that gain on the table. We anneal to essentially zero. The cost is that a decayed checkpoint is "spent" — you cannot productively resume high-LR training from it — which is exactly why we branch decay runs off the *stable* checkpoint, never off a decayed one.

## Putting It Together: Building the Optimizers

Here is the routing function the training loop calls once at startup. It walks the model's named parameters, sends every 2D hidden matrix to Muon and everything else to AdamW, and returns both. The dimensionality test (`p.ndim == 2` and "not the embedding") is the whole trick.

```python
# stacklm/optim/build.py
import torch
from stacklm.optim.muon import Muon

def build_optimizers(model, muon_lr=0.02, adamw_lr=3e-3,
                     weight_decay=0.1, betas=(0.9, 0.95)):
    """Route Stack-100M params: 2D hidden matrices -> Muon, everything else
    (tied embedding, RMSNorm gains, any 1D/bias) -> AdamW. Returns (muon, adamw).
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
    # gains un-decayed (shrinking a RMSNorm gain toward 0 fights the network).
    decay, no_decay = [], []
    for p in adamw_params:
        (decay if p.ndim >= 2 else no_decay).append(p)
    adamw = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=adamw_lr, betas=betas, eps=1e-8, fused=True)
    return muon, adamw
```

The AdamW **betas are `(0.9, 0.95)`**, the LLM-standard choice ([Chapter 3.9](../03-pretraining/09-optimizers.html)): the lower $\beta_2=0.95$ (vs the vision default 0.999) makes the second-moment estimate more responsive, which matters because language gradients are noisier and more non-stationary than image gradients. We enable `fused=True` for a single-kernel AdamW step. Note the param-group split so that 1D norm gains get **no weight decay** — decaying a RMSNorm gain toward zero would fight the network's need to scale activations.

### The full step: grad accumulation, clipping, bf16, QK-clip

The capstone's effective batch is **~0.5M tokens** (spec §5). At `seq_len = 2048` that is `500000 / 2048 ≈ 244` sequences per optimizer step — far more than fits on one A100. We reach it with **gradient accumulation**: run several micro-batches, summing gradients, and step once. If the A100 holds a micro-batch of 16 sequences, we need `244 / 16 ≈ 16` accumulation micro-steps. Everything runs under **bf16 autocast** ([Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)); bf16's wide exponent means we need **no loss scaler**, unlike fp16.

```python
# stacklm/train.py  (the inner optimizer step, abbreviated)
import torch

GRAD_ACCUM = 16          # micro-steps per optimizer step
MICRO_BSZ  = 16          # sequences per micro-step  -> ~0.5M tokens @ seq_len 2048
GRAD_CLIP  = 1.0

def optimizer_step(model, batches, muon_opt, adamw_opt, step,
                   max_logits_per_head):
    muon_opt.zero_grad(set_to_none=True)
    adamw_opt.zero_grad(set_to_none=True)
    for micro in range(GRAD_ACCUM):
        x, y = next(batches)                        # (MICRO_BSZ, seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, targets=y,
                                 record_attn_logits=max_logits_per_head)
            loss = loss / GRAD_ACCUM                 # average, not sum, over accum
        loss.backward()                              # accumulate grads

    # Clip the GLOBAL grad norm across BOTH optimizers' params to 1.0.
    all_params = [p for g in muon_opt.param_groups  for p in g["params"]] + \
                 [p for g in adamw_opt.param_groups for p in g["params"]]
    torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)

    # WSD learning rates for this step (see schedule.py wiring above).
    set_wsd_lrs(muon_opt, adamw_opt, step)

    muon_opt.step()
    adamw_opt.step()
    # MuonClip: post-step QK-clip using the logits recorded this forward pass.
    qk_clip_(model, max_logits_per_head, tau=100.0)
    max_logits_per_head.zero_()                      # reset the running max
```

Three details that bite people. **(1)** We average the loss by `GRAD_ACCUM` (not sum) so the gradient magnitude — and thus the effective LR — is independent of how many micro-steps we chose; change the accumulation count and the run behaves identically. **(2)** Gradient clipping is applied to the **global** norm across *both* optimizers' parameters, `max_norm = 1.0`, so a spike in any one layer is tamed relative to the whole update. **(3)** QK-clip runs **after** the step, on the logits recorded during this step's forward, then we reset the running max.

### Why 0.5M tokens, and the critical batch size

The 0.5M-token effective batch is not arbitrary. [Chapter 3.10](../03-pretraining/10-lr-schedules-hparams.html) develops the **critical batch size** (McCandlish et al., 2018): below it, doubling the batch nearly halves the step count for the same loss (compute is "wasted" on gradient noise you could average away); above it, extra batch buys diminishing returns and mostly costs memory. For a 100M model the critical batch sits on the order of a few hundred thousand to ~1M tokens, so 0.5M is a sweet spot — large enough that the gradient estimate is clean (which Muon likes, since orthogonalizing a noisy buffer wastes the preconditioner) and small enough that we still take 40,000 update steps to spend the budget. Push the batch to 4M tokens and you would take only 5,000 steps, each barely better than at 0.5M — a worse use of the same tokens.

Micro-batch sizing is then a pure memory game on the A100. Activation memory scales with `MICRO_BSZ × seq_len × d_model × n_layers`; with **activation checkpointing** (recompute in the backward, [Chapter 3.5](../03-pretraining/05-distributed-data-parallel.html) / [Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html)) we can push `MICRO_BSZ` to 16 or 32 sequences of 2048 tokens comfortably, then set `GRAD_ACCUM = round(244 / MICRO_BSZ)` to hit the target. On a 24GB 4090 the micro-batch shrinks and accumulation grows to compensate — same recipe, same effective batch, more wall-clock, exactly as the spec's secondary tier promises.

### One table for the shared knobs

| Knob | Value | Why |
|---|---|---|
| Muon peak LR | ~0.02 | RMS-matched to AdamW via $0.2\sqrt{\max(m,n)}$ scale |
| AdamW peak LR | ~3e-3 | embeddings / norms; standard small-model range |
| AdamW betas | (0.9, 0.95) | $\beta_2$ lowered from 0.999 for noisy LM gradients |
| Weight decay | 0.1 | decoupled; **0.0** on 1D RMSNorm gains |
| Grad clip (global) | 1.0 | across both optimizers' params |
| Momentum (Muon) | 0.95, Nesterov | orthogonalize the look-ahead direction |
| Newton–Schulz steps | 5 | enough to condition the direction; not exact |
| Precision | bf16 autocast | wide exponent ⇒ no loss scaler |
| Effective batch | ~0.5M tokens | near critical batch size for 100M |
| QK-clip $\tau$ | 100 | Kimi K2 default; belt-and-suspenders to QK-norm |

## Worked Example: Step Budget, Warmup, and a Newton–Schulz Trace

!!! example "The Stack-100M schedule, in real numbers"
    **Total steps.** Budget = 20B tokens, effective batch = 0.5M tokens/step:
    $$
    T = \frac{20\times10^{9}}{0.5\times10^{6}} = 40{,}000 \text{ optimizer steps.}
    $$

    **Warmup.** A common rule is ~1–2% of steps, or a few hundred to a couple thousand. Take $T_w = 2{,}000$ steps (~1B tokens). At `peak_lr = 0.02` for Muon, the LR near step 500 is $\approx 0.02\times 500/2000 = 0.005$.

    **Phase split** with `decay_frac = 0.2`: decay = $0.2\times 40{,}000 = 8{,}000$ steps (the last ~4B tokens); stable = $40{,}000 - 2{,}000 - 8{,}000 = 30{,}000$ steps (~15B tokens on the bulk mix). The **8,000-step decay phase is the mid-training window** where we anneal on premium data ([Chapter 14.8](../14-capstone/08-mid-training.html)).

    **Muon LR during decay.** Halfway through decay ($\text{progress}=0.5$): $1-\sqrt{0.5}=0.293$, so $\eta = 0.02\times0.293 \approx 0.0059$. At progress $0.9$: $1-\sqrt{0.9}=0.051$, $\eta\approx 0.001$. The LR falls off fast early in decay — matching the "sharp loss drop" WSD is known for.

    **Newton–Schulz cost.** A SwiGLU `w_gate` is `1408 × 512`; iterating on the short (512) side, each of 5 steps does a handful of matmuls dominated by $512^2\times 1408$, so on the order of `~5–8 GFLOP` per large matrix (attention `512×512` matrices are cheaper, `wk/wv` at `128×512` cheaper still). `Stack-100M` has **7 such 2D matrices per block × 30 blocks = 210 matrices**, summing to roughly **0.5–1 TFLOP per optimizer step** for orthogonalization. Against the model's forward+backward of $6ND \approx 6 \times 101\text{e}6 \times 0.5\text{e}6 \approx 3\times10^{14}$ FLOP/step (≈ 300 TFLOP), Muon's overhead is on the order of **a few tenths of a percent** — negligible, as promised.

The takeaway from the last line: Muon buys faster convergence for essentially free compute at this scale. The Newton–Schulz iterations are dwarfed by the transformer's own matmuls. (A common mistake here is to write the model's per-step FLOPs as $\sim10^{17}$; that is the cost of *many* steps, not one — $6ND$ with $D$ = the 0.5M **tokens per step** lands at $\sim3\times10^{14}$.)

## Contrasting the Choices, and What Would Break

It is worth stating plainly why each alternative was rejected, because an interviewer will ask.

- **Pure AdamW (no Muon).** Perfectly safe, and what nanoGPT-2024 used. It simply converges slower per token on the 2D matrices; at a fixed 20B-token budget you leave perplexity on the table. If you distrust Muon's stability, this is the fallback — route all 2D matrices to AdamW too.
- **Pure Muon (no AdamW).** Breaks on the embedding: orthogonalizing a `32768 × 512` matrix with sparse per-row gradients is both wrong and slow, and 1D norms cannot be orthogonalized at all. The hybrid is not optional.
- **Cosine instead of WSD.** Works, but forces you to commit to a step count and gives no clean data-annealing hook. Since the capstone's entire mid-training story ([Chapter 14.8](../14-capstone/08-mid-training.html)) rides on the decay phase, WSD is the natural fit. A useful mental model: **WSD is cosine with the middle stretched into a flat plateau you can extend at will.**
- **No QK-clip.** Fine *until* it isn't — a single head's logit blow-up produces a NaN that ends the run. With QK-norm already in the architecture, QK-clip fires rarely, but the cost of keeping it is one `.amax()` per layer and the benefit is never losing a 20-GPU-hour run to a preventable spike.

!!! warning "Common pitfall: mismatched Muon/AdamW learning rates"
    The single most common way to get a bad Muon run is to forget the RMS-matching scale ($0.2\sqrt{\max(m,n)}$) and then reuse an AdamW-tuned learning rate directly. Without the scale, Muon's raw orthogonal update has RMS $\sim 1/\sqrt{\max(m,n)}$ — tiny for a `1408×512` matrix — so a "normal" LR of `3e-3` barely moves the weights and the run looks dead. With the scale, `lr ≈ 0.02` on Muon and `≈ 3e-3` on AdamW live in comparable regimes. If Muon training stalls, check this scale *first*.

!!! interview "Interview Corner"
    **Q:** You're training a 100M model and choose Muon for the weight matrices but keep AdamW for embeddings and norms. Why the split, and what stability problem does Muon specifically introduce that you must mitigate?

    **A:** Muon orthogonalizes the momentum update of a 2D matrix — it replaces the update with the $UV^\top$ polar factor via a few Newton–Schulz iterations, so every singular direction gets an equal-magnitude push. That's a spectral preconditioner that converges faster than Adam's per-coordinate rescaling on the dense, well-structured attention and MLP matrices. But it only makes sense for 2D "feature-mixing" matrices: the token embedding has sparse per-row gradients (most rows untouched each batch) and 1D RMSNorm gains have no matrix structure, so both stay on AdamW, which gives them per-coordinate adaptive rates. The stability problem Muon introduces is **attention-logit blow-up**: its aggressive equal-direction updates let $W_Q$ and $W_K$ grow until $\max q^\top k/\sqrt{d}$ saturates the softmax and NaNs the loss in bf16. The fix is **QK-clip** (the core of MuonClip / Kimi K2): after each step, any head whose max logit exceeded a threshold $\tau$ has its query and key weights rescaled by $\sqrt{\tau/S_{\max}}$, pulling the worst-case logit back to $\tau$ (with GQA, the shared key slice is scaled once by the group's worst logit, not once per query head). In our config QK-norm already bounds most of this, so QK-clip is a rarely-firing safety net. I'd also mention the RMS-matching scale ($0.2\sqrt{\max(m,n)}$) that lets one conceptual LR govern both optimizers.

!!! key "Key Takeaways"
    - **Muon orthogonalizes the momentum update** of 2D weight matrices via ~5 Newton–Schulz iterations (a matmul-only approximation to the $UV^\top$ polar factor), setting all singular values to 1 so the update pushes equally in every direction — faster per-token convergence than AdamW on attention/MLP matrices, at a fraction of a percent extra FLOPs.
    - The standard **hybrid is mandatory**: Muon for 2D hidden matrices, **AdamW for the tied embedding, RMSNorm gains, and all 1D params**. Route by `p.ndim == 2` (and "not the embedding").
    - A **RMS-matching scale** of $0.2\sqrt{\max(m,n)}$ on the Muon update lets a single conceptual LR govern both optimizers; forget it and Muon appears "dead." Capstone peaks: Muon `≈0.02`, AdamW `≈3e-3`.
    - **MuonClip / QK-clip** caps attention logits per head: if $S_{\max}>\tau$, scale that head's $W_Q,W_K$ by $\sqrt{\tau/S_{\max}}$; under GQA scale each shared $W_K$ once by its group's worst logit. With QK-norm present it fires rarely but prevents run-ending NaNs.
    - **WSD (Warmup–Stable–Decay)** replaces cosine: linear warmup, a long **constant-LR stable phase you can extend at will**, then a short $1-\sqrt{}$ **decay to ~0**. No baked-in endpoint.
    - The **decay phase is the mid-training annealing phase** ([Chapter 14.8](../14-capstone/08-mid-training.html)): low LR + higher-quality data (more Cosmopedia/math/code) yields a sharp loss drop for little compute — a hook cosine can't cleanly provide.
    - Shared knobs: **weight decay 0.1** (decoupled; none on 1D norms), **global grad-clip 1.0**, AdamW **betas (0.9, 0.95)**, **bf16** autocast with no loss scaler, **~0.5M-token effective batch** via gradient accumulation (~16 micro-steps of 16 seqs @ 2048).
    - For `Stack-100M`'s 20B-token budget: **40,000 steps**, ~2,000 warmup, ~30,000 stable, ~8,000 decay.

!!! sota "State of the Art & Resources (2026)"
    Muon (with the Moonshot RMS-matching fix) and QK-clip have moved from nanoGPT speedrun tricks to production ingredients at frontier scale, and WSD is now a common default for schedule-agnostic pretraining — the capstone's stack mirrors what Kimi K2, Moonlight, and DeepSeek actually shipped.

    **Foundational work**

    - [Gupta, Koren & Singer, *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018)](https://arxiv.org/abs/1802.09568) — the full-matrix preconditioner Muon cheaply approximates via Newton–Schulz.
    - [McCandlish, Kaplan & Amodei, *An Empirical Model of Large-Batch Training* (2018)](https://arxiv.org/abs/1812.06162) — introduces the gradient-noise-scale / critical-batch-size argument used to justify the ~0.5M-token effective batch.
    - [Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2019)](https://arxiv.org/abs/1711.05101) — the AdamW paper; Muon's decoupled weight decay follows the same recipe.

    **Recent advances (2023–2026)**

    - [Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024)](https://arxiv.org/abs/2404.06395) — introduces the Warmup–Stable–Decay schedule and its continuable-pretraining property.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — a frontier-scale run using a WSD-style multi-stage schedule.
    - [Liu et al. (Moonshot AI), *Muon is Scalable for LLM Training* (2025)](https://arxiv.org/abs/2502.16982) — the Moonlight report: decoupled weight decay for Muon and the update-RMS matching that lets Muon and AdamW share one learning rate.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — introduces MuonClip and QK-clip to tame attention-logit blow-up under Muon at trillion-parameter scale.

    **Open-source & tools**

    - [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — the reference PyTorch `Optimizer` implementation this chapter's `muon.py` follows.
    - [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — the speedrun repo where Muon, QK-norm, and related tricks were stress-tested step by step.
    - [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) — open-source distributed Muon (ZeRO-1) plus checkpoints from the "Muon is Scalable" report.

    **Go deeper**

    - [Keller Jordan, *Muon: An optimizer for hidden layers in neural networks* (blog, 2024)](https://kellerjordan.github.io/posts/muon/) — the original writeup deriving Newton–Schulz orthogonalization and the nanoGPT speedrun results.

## Further Reading

- **Keller Jordan et al.**, *Muon: An optimizer for the hidden layers of neural networks* (2024) — the original Muon and Newton–Schulz orthogonalization; the nanoGPT speedrun context.
- **Liu et al. (Moonshot AI)**, *Muon is Scalable for LLM Training* (2025) — decoupled weight decay for Muon and the update-RMS matching that lets Muon and AdamW share a learning rate.
- **Kimi K2 Technical Report** (Moonshot AI, 2025) — MuonClip and the QK-clip mechanism for attention-logit stability at scale.
- **Hu et al.**, *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024) — the Warmup–Stable–Decay schedule and its continuable-pretraining / annealing properties.
- **DeepSeek-V2 / V3 technical reports** (DeepSeek-AI, 2024) — WSD-style multi-step schedules used at frontier scale.
- **McCandlish, Kaplan, Amodei et al.**, *An Empirical Model of Large-Batch Training* (2018) — the critical batch size that justifies the ~0.5M-token effective batch.
- **Kingma & Ba**, *Adam: A Method for Stochastic Optimization* (2015) and **Loshchilov & Hutter**, *Decoupled Weight Decay Regularization (AdamW)* (2019) — the baseline this whole stack extends.
- **Gupta, Roy & Anandkumar**, *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018) — the full-preconditioner method Muon cheaply approximates.

## Exercises

**1.** In `build_optimizers()` the tied token embedding (`32768 x 512`) is routed to AdamW even though it is a 2D tensor and would pass a naive `p.ndim == 2` test. Explain why orthogonalizing this matrix with Muon is both *wrong* and *wasteful*, and name the property of embedding gradients that makes AdamW the right choice.

??? note "Solution"
    Muon's orthogonalization only makes sense when both axes of a matrix are "feature" axes that mix together, because setting every singular value to 1 spreads the update equally across all directions of the matrix. The embedding's row axis is the **token axis** (32,768 entries), not a feature axis. In any given batch only a tiny fraction of the 32,768 rows are touched at all — most tokens do not appear — so the gradient is **extremely sparse per row**. Orthogonalizing across the token axis mixes an untouched row's (zero) gradient direction with the handful of active rows, which is semantically meaningless: there is no shared low-dimensional structure across arbitrary vocabulary rows to precondition.

    It is also wasteful: the Newton-Schulz iteration would run on a `32768 x 512` matrix (iterating on the 512 side, its Gram matrix is still fed a huge outer dimension), spending real FLOPs to produce a bad direction.

    What the embedding actually wants is a **per-row adaptive learning rate** so that a rarely-seen token's row is not swamped by the scale of frequent tokens. That is exactly what AdamW's per-coordinate second-moment rescaling gives. Hence the hybrid: 2D *hidden* feature-mixing matrices to Muon, embedding/norms/1D to AdamW.

**2.** The RMS-matching scale is $0.2\sqrt{\max(m,n)}$. Take the SwiGLU gate weight `w_gate` of shape `1408 x 512`. (a) What is the root-mean-square element magnitude of the raw orthogonalized update $O = UV^\top$ *before* scaling? (b) Compute the scale factor. (c) Confirm the post-scale RMS. (d) In one sentence, why does this matter for choosing the learning rate?

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

    **(d)** A well-behaved AdamW step has per-element RMS $\approx 1$; scaling Muon's update to RMS $\approx 0.2$ puts the two optimizers' effective step magnitudes in the same regime, so a **single conceptual learning rate** (peaks Muon $\approx 0.02$, AdamW $\approx 3\text{e-}3$) governs the whole run. Forget the scale and Muon's raw update has RMS $\sim 1/\sqrt{1408} \approx 0.027$ -- tiny -- so a normal LR barely moves the weights and training looks dead.

**3.** Suppose you re-budget the run to **30B tokens** at the same **0.5M-token** effective batch, use a **2% warmup**, and keep `decay_frac = 0.2`. (a) How many total optimizer steps? (b) Give the warmup / stable / decay step counts. (c) Using the $1-\sqrt{\cdot}$ decay, what Muon LR (peak `0.02`) is in effect at **one quarter** of the way through the decay phase?

??? note "Solution"
    **(a)** Total steps:
    $$
    T = \frac{30\times10^{9}}{0.5\times10^{6}} = 60{,}000 \text{ steps.}
    $$

    **(b)** Warmup $= 0.02 \times 60{,}000 = 1{,}200$ steps. Decay $= 0.2 \times 60{,}000 = 12{,}000$ steps. Stable $= 60{,}000 - 1{,}200 - 12{,}000 = 46{,}800$ steps.

    **(c)** At progress $= 0.25$ through decay:
    $$
    \text{decay\_mult} = 1 - \sqrt{0.25} = 1 - 0.5 = 0.5,
    $$
    so (with `final_frac = 0`) $\eta = 0.02 \times 0.5 = 0.01$. Note the LR is already halved only a quarter of the way in -- the $1-\sqrt{\cdot}$ shape drops fast early, which is what produces WSD's characteristic sharp loss drop right as decay (and premium-data annealing) begins.

**4.** A block has `n_heads = 8` query heads and `n_kv_heads = 2` (GQA group size 4), with $\tau = 100$. The per-head max pre-softmax logits recorded this step are

    s_max = [120, 90, 100, 80, 150, 60, 70, 200]   # heads 0..7

    (a) Which query heads get their $W_Q$ slice scaled, and by what factor? (b) What single factor scales each shared $W_K$ group? (c) Verify that the head that *set* group 1's maximum lands exactly at $\tau$, and show that head 4 (also in group 1) ends up *below* $\tau$.

??? note "Solution"
    Group 0 = heads 0-3, group 1 = heads 4-7. The per-weight scale is $\eta = \sqrt{\tau / S_{\max}}$ (the square root, because the logit is bilinear in $W_Q$ and $W_K$, so scaling *both* multiplies the logit by the full ratio $\tau/S_{\max}$).

    **(a) $W_Q$, per query head** (only heads with $S_{\max} > \tau$; head 2 at exactly 100 is *not* $> 100$, so it is untouched):
    - head 0: $\sqrt{100/120} = 0.9129$
    - head 4: $\sqrt{100/150} = 0.8165$
    - head 7: $\sqrt{100/200} = 0.7071$
    - heads 1, 2, 3, 5, 6: untouched.

    **(b) $W_K$, once per group, using the group's worst logit:**
    - group 0: $\max(120,90,100,80) = 120 \Rightarrow \eta = \sqrt{100/120} = 0.9129$
    - group 1: $\max(150,60,70,200) = 200 \Rightarrow \eta = \sqrt{100/200} = 0.7071$

    **(c)** Head 7 set group 1's max. Its logit is scaled by $\eta_{W_Q} \cdot \eta_{W_K} = 0.7071 \times 0.7071 = 0.5$, giving $200 \times 0.5 = 100 = \tau$ exactly. Head 4 shares group 1's $W_K$ ($0.7071$) but keeps its own $W_Q$ scale ($0.8165$): its logit becomes $150 \times 0.8165 \times 0.7071 = 150 \times 0.5774 \approx 86.6 < \tau$. So the group-setting head lands on the cap and the other members are pulled conservatively below it -- a safe outcome, and the reason $W_K$ is scaled once per group rather than once per query head (which would over-shrink a shared KV slice).

**5.** The chapter stresses WSD's *continuable* property: you branch multiple decay runs off the same **stable** checkpoint, and the decay length is not tied to any pre-committed total. But `wsd_lr()` derives `decay_steps` from `total_steps` via `decay_frac`, so its decay length is coupled to a fixed horizon. Implement `wsd_lr_branch()` that takes `stable_steps` and `decay_steps` **directly**, so you can spawn a decay of any length from a checkpoint saved at the end of the stable phase. Keep the linear warmup, the constant stable phase, and the $1-\sqrt{\cdot}$ decay to `final_frac * peak_lr`.

??? note "Solution"
    The fix is to stop computing the decay length from a global `total_steps` and instead pass the three phase lengths independently. This makes the stable checkpoint a genuine fork point: save at `warmup_steps + stable_steps`, then launch any number of decay runs, each with its own `decay_steps`, all reading the same weights.

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
        progress = min(progress, 1.0)                     # stay flat at floor past end
        decay_mult = 1.0 - math.sqrt(progress)            # 1 -> 0
        floor = final_frac * peak_lr
        return floor + (peak_lr - floor) * decay_mult
    ```

    Two things to notice. First, `total_steps` never appears -- the schedule is fully determined by the three phase lengths, so extending the stable phase or trying a longer/shorter decay is a config change, not a re-derivation. Second, the `min(progress, 1.0)` clamp means that if you ask for a step past the end of the decay, the LR sits at the floor rather than going negative (since $\sqrt{\text{progress}} > 1$ would make `decay_mult` negative). To reproduce the original 40k-step recipe you would call it with `stable_steps = 30000`, `decay_steps = 8000`, `warmup_steps = 2000`.

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

    **Expected output (approximately):**

    ```text
    input  sigma: min 1.000e-01  max 1.000e+01  cond 1.000e+02
    output sigma: min 0.68..     max 1.20..     cond ~1.8
    ```

    The input condition number is $\sim 10^2$; after 5 Newton-Schulz steps every singular value has been pulled toward 1, so the output's spread collapses to roughly $[0.68, 1.20]$ and its condition number drops from $\sim100$ to under 2. That is the whole point of orthogonalization -- turn a spectrum dominated by a few large directions into a nearly isotropic one so the update pushes equally along all directions of the matrix. Notice the largest singular values slightly **overshoot** 1 (to $\approx1.2$): the tuned quintic is designed to converge *fast*, not monotonically, so it rings a little around 1 rather than approaching it from below.

    Why not more steps for exact orthogonality? Because we only need a **well-conditioned direction**, not $\sigma = 1$ to machine precision. The optimizer immediately scales the result by $0.2\sqrt{\max(m,n)}$ and the learning rate and applies it as a step; the residual $\pm20\%$ deviation from perfect orthogonality is utterly washed out by that. Five matmul-only iterations in bf16 are far cheaper than an exact SVD and give a direction that is already as good as the optimization needs -- which is exactly why Muon's per-step overhead stays at a fraction of a percent of the transformer's own FLOPs. (Push the input condition number to $10^5$ instead and 5 steps would *not* fully recover the tiniest singular values -- Newton-Schulz converges only linearly near 0 -- but real momentum buffers are nowhere near that ill-conditioned.)
