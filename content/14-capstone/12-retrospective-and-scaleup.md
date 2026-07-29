# 14.12 Retrospective: Cost Accounting, Reproducibility, and the Path to 1B

Eleven chapters ago we set out to do something specific: take *the entire* LLM stack — every idea in this book — and compress it into one project small enough to run on a single rented GPU for about the price of a nice dinner, yet real enough that nothing was faked. We trained a byte-level BPE tokenizer, fit our *own* scaling law on a ladder of tiny models, pretrained **Stack-100M** on ~20B tokens with a Muon+AdamW hybrid under a WSD schedule, mid-trained it for long context and quality, aligned it with SFT / DPO / GRPO, distilled a narrow tool-using ReAct agent into it, evaluated it honestly, and quantized it to int4 to run on a laptop. The plane is in the air. This chapter lands it.

Landing means four concrete things, and this chapter is those four things: (1) an **audit of the prediction** — did the scaling law we fit in Chapter 14.5 actually forecast the flagship run's loss, and how far is such a law allowed to reach; (2) a **full cost accounting** — every GPU-hour and dollar, stage by stage, derived rather than asserted, and re-priced onto 2026 hardware; (3) a **reproducibility discipline** — the seeds, config hashes, data manifests, environment pins, checkpoint formats, and release artifacts that separate "I trained a model once" from "anyone can rebuild this"; and (4) an honest **scale-up analysis** — what actually *breaks* when you take this exact recipe from 100M to 1B parameters, which libraries you switch to, what it costs, and what you change (data, memory, parallelism, learning-rate and batch scaling, and the Mixture-of-Experts option) to get there.

This chapter builds directly on [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for the FLOP arithmetic, [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) for the utilization analysis, [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) for the parallelism transitions, and [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html) for the reproducibility machinery. It stays strictly consistent with the canonical `Stack-100M` config (`vocab_size=32768`, `d_model=512`, `n_layers=30`, GQA 8:2, SwiGLU `intermediate=1408`, tied embeddings, ≈101M params) fixed in the capstone plan.

---

## 14.12.1 Where We Landed

Before we count the money, let us name the artifact precisely, because the cost only means something relative to what it bought.

At the end of the pipeline the reader owns four checkpoints, all derived from one 101M-parameter base:

```text
stacklm-100m/
├── base/           # 18B-token pretrained base (WSD stable phase)   ~2.9-3.1 nats/tok held-out
├── mid/            # +2B tokens: quality anneal -> long-context (8192) -> capability injection
├── chat/           # SFT + DPO on the ChatML-style template
├── agent/          # distilled ReAct traces -> narrow auto-research tool-user
└── agent-int4/     # round-to-nearest int4 weights, ~55-65 MB on disk, runs on a CPU laptop
```

None of these is a frontier model, and the capstone never pretended otherwise. A 100M model is a *narrow instrument*: within a scaffolded, retrieval-grounded domain it can produce coherent, useful, grounded text; outside that scaffold it hallucinates freely. What is remarkable is not the model's raw quality but that **every layer of the stack that produces a GPT-4-class system is present here in miniature and actually ran** — the tokenizer, the scaling-law fit, the optimizer, the distributed-ready loop, the alignment stack, the agent harness, the quantized deployment. You now have a mental model of the whole machine that is *load-bearing*, not decorative, because you built each part.

The single most important lesson the capstone teaches — the one that makes a 100M model in 2026 vastly better than GPT-2 (117M) in 2019 at the same size — is **deliberate over-training**. Chinchilla-optimal for our ~84.5M non-embedding parameters is ~1.7B tokens; we trained on ~20B, roughly 200 tokens/param and ~12× past compute-optimal. That is economically irrational for a model you train and throw away, and completely rational for a model you will *serve*, because you pay training compute once and save inference cost forever. Keep that asymmetry in mind — it reappears in the cost table (over-training is most of the bill) and in the scale-up section (it only gets more extreme at 1B).

---

## 14.12.2 Did Our Scaling Law Hold?

The capstone's differentiator was that we did not import someone else's scaling constants. In [Chapter 14.5](05-mini-scaling-laws.html) we trained a four-rung ladder ({3.9M, 8.8M, 19M, 44M} non-embedding parameters), fit $L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$ to ~17 measured losses, cross-checked with IsoFLOP profiles, and used the result to *predict* the flagship's loss before spending the flagship's money. A retrospective that never asks whether the prediction landed is not a retrospective. So: did it?

```python
# stacklm/scaling_check.py -- close the loop on the Ch. 14.5 ladder fit.
"""Compare the loss the ladder PREDICTED against the loss the flagship ACHIEVED.

Two rules carried over from Ch. 14.5, both easy to get wrong:
  1. N is the NON-EMBEDDING parameter count (the 30 blocks: 84.5M), because the
     ladder fit N against the work the 6ND rule counts. Feeding 101.4M here
     silently corrupts the comparison.
  2. The law predicts loss on the PRETRAIN mix. Evaluate `base/` on a held-out
     shard of that same 70/15/10/5 mix -- not on the mid-training anneal mix,
     whose entropy floor E is different.
"""
LADDER_FIT = dict(E=2.45, A=124.0, alpha=0.33, B=234.0, beta=0.30)  # Ch. 14.5
N_NONEMBED = 84_500_000      # 30 blocks x ~2.82M; excludes the tied 16.8M embedding
FIT_BAND = 0.10              # +/- nats: the honest resolution of a 4-rung ladder


def predicted_loss(n_nonembed: int, n_tokens: float,
                   E: float, A: float, alpha: float,
                   B: float, beta: float) -> float:
    """Chinchilla parametric form, evaluated at (N, D)."""
    return E + A * n_nonembed ** (-alpha) + B * n_tokens ** (-beta)


for label, D in [("Chinchilla-optimal (~20 tok/param)", 1.69e9),
                 ("flagship, stable phase only",        1.80e10),
                 ("flagship, full campaign",            2.00e10)]:
    L = predicted_loss(N_NONEMBED, D, **LADDER_FIT)
    print(f"{label:36s} D={D:.2e}  L_pred = {L:.3f} +/- {FIT_BAND:.2f} nats")
# Chinchilla-optimal (~20 tok/param)   D=1.69e+09  L_pred = 3.150 +/- 0.10 nats
# flagship, stable phase only          D=1.80e+10  L_pred = 2.949 +/- 0.10 nats
# flagship, full campaign              D=2.00e+10  L_pred = 2.940 +/- 0.10 nats
```

The ladder's forecast for the flagship base checkpoint is therefore **≈ 2.94 nats/token, with an honest ±0.1-nat band** (2.84–3.04), which is exactly the range [Chapter 14.11](11-evaluation-and-serving.html) documents for the held-out perplexity measurement (a loss of ~2.9 nats corresponds to PPL ≈ 18). The prediction and the artifact agree to within the band the fit deserves. That is the outcome you should *expect*, and it is worth being precise about why, because the failure modes are more instructive than the success.

**Why it was allowed to work.** The ladder was trained under the *identical* recipe — same tokenizer, same 32,768 vocab, same 70/15/10/5 mix, same `seq_len=2048`, same Muon+AdamW hybrid, same WSD shape. Every one of those choices is baked into $E$, $A$, and $B$; change any of them and the constants move. And the extrapolation in $N$ is short: from a top rung of 44M to a target of 84.5M is a factor of 1.92, or **0.28 decades**. Extrapolating a factor of two is engineering; extrapolating a factor of a hundred is a prayer.

**Where the strain actually is.** It is not in $N$ — it is in $D$. The ladder's rungs were trained near compute-optimal (tens of tokens per parameter); the flagship is deliberately run at ~200 tokens/parameter. So the $B/D^{\beta}$ term is being evaluated an order of magnitude outside the token range that constrained $\beta$. Three consequences follow, and all three are things to check rather than assume:

1. **The data term is the extrapolated one.** At 20B tokens the data penalty $B/D^{\beta}$ has fallen to ~0.19 nats while the capacity penalty $A/N^{\alpha}$ sits at ~0.30 nats. The model is capacity-limited, not data-limited — which is the correct diagnosis of an over-trained model, and the quantitative reason further over-training has diminishing returns while a bigger model would not.
2. **Repetition is not counted by the law.** $L(N,D)$ assumes $D$ *unique* tokens. If your 20B-token budget contains 2 epochs of a 10B-token corpus, the law over-predicts the benefit. Muennighoff et al. (2023) quantify the discount; below ~4 epochs it is small, which is why the plan's mix is sized for roughly single-epoch coverage.
3. **$E$ is mix-specific.** The mid-training anneal deliberately shifts to a cleaner, denser mix, which *lowers* measured loss partly because the text is more predictable, not because the model got better. Comparing `mid/` against a law fit on the pretrain mix will make the model look better than it is. Always evaluate `base/` on the pretrain mix's held-out shard, and treat mid-training's gain as a separate, separately-measured number.

!!! warning "How far may a fitted law reach?"

    A practical rule, and the one that governs the rest of this chapter: **trust an extrapolation to roughly 2–3× beyond your top rung; distrust anything past ~10×.** Our 44M → 84.5M leap is 1.9× — fine. The same law asked to predict a 1B model is being stretched ~20× (1.3 decades) past the top rung, and the weakly-identified $E,A,B$ offset that cancels harmlessly at 2× does not cancel at 20×. The honest move before a 1B run is not to reuse this fit — it is to **re-run the ladder with a higher top rung** (something like {30M, 70M, 150M, 350M} non-embedding), which costs a few percent of the 1B budget and re-earns the right to extrapolate. Recall also from Ch. 14.5 that the *allocation exponent* $\beta/(\alpha+\beta) \approx 0.47$ is far better identified than the individual constants; it transfers further than the absolute loss does.

Finally, the caveat the loss curve cannot tell you: **the law predicts loss, not capability.** Nothing in $L(N,D)$ says whether the model can do two-digit arithmetic or follow a ReAct format. Downstream abilities are threshold-y and must be measured, which is what the probes in Chapter 14.11 are for. A scaling law is a budgeting instrument, not an evaluation.

---

## 14.12.3 Cost Accounting: Anatomy of the ~USD 100 Model

The headline is "the ~USD 100 model," but a headline is not an accounting. Let us derive the number from first principles — including the FLOPs the usual rule of thumb *omits* — then reconcile it against a realistic itemized bill, then re-price the whole thing onto 2026 hardware, including the parts nobody advertises: the teacher API calls, the object storage, and the *re-run reality tax* of failed launches and OOM debugging.

### The compute floor: 6ND, and the FLOPs 6ND forgets

Training FLOPs for a dense transformer follow the **6ND** rule (2 FLOPs/param/token forward, 4 backward), covered in depth in [Scaling Laws](../03-pretraining/04-scaling-laws.html). For the full pretraining campaign (18B tokens in the stable phase plus ~2B in mid-training):

$$
C_{6ND} = 6 \, N \, D = 6 \times (1.014\times 10^{8}) \times (2.0\times 10^{10}) \approx 1.22 \times 10^{19} \ \text{FLOPs.}
$$

That number is a *floor*, and at Stack-100M's shape it is a surprisingly loose one. The 6ND rule counts only the parameterized matmuls; it ignores the attention score computation, which has no parameters at all. Per token, per layer, causal attention costs $2 s\, d_{\text{model}}$ FLOPs in the forward pass ($QK^\top$ over the causal half plus the $AV$ product, summed across heads), and backward is ~2× forward, so:

$$
C_{\text{attn}} = 6 \, L \, s \, d_{\text{model}} \, D
\quad\Longrightarrow\quad
\frac{C_{\text{attn}}}{C_{6ND}} = \frac{L \, s \, d_{\text{model}}}{N}.
$$

For Stack-100M ($L=30$, $s=2048$, $d=512$, $N=1.014\times10^8$) that ratio is $30 \times 2048 \times 512 / 1.014\times10^{8} = \mathbf{0.31}$. **Attention adds 31% on top of the 6ND count.** Two things follow immediately:

- **Deep-and-thin models pay this tax hardest.** For a standard block the ratio simplifies to roughly $s/(12\,d_{\text{model}})$ — it grows with sequence length and *shrinks* with width. Our narrow $d=512$ is why it is 31% here, while a 1B model at $d=2048$ and the same sequence length pays only ~8%. The deep-thin recipe that buys quality per parameter (MobileLLM, Liu et al., 2024) costs efficiency per FLOP. That is a real trade, not a free win.
- **Long context multiplies it.** At the mid-training length $s=8192$ the ratio quadruples to ~1.24, i.e. attention now costs *more* than all the matmuls combined. This is the quantitative version of Chapter 14.8's warning that pretraining at 8192 from step zero would have been ruinous.

Note that **GQA does not help here**: 2 KV heads shrink the KV *cache* 4×, but Q, K and V are still broadcast to 8 query heads, so the attention FLOPs are unchanged. GQA is a memory optimization, not a compute one.

### MFU, HFU, and which one your loop logs

An NVIDIA A100 (80GB) has a bf16 tensor-core peak of ~312 TFLOP/s dense. You never get peak; you get **Model FLOPs Utilization (MFU)** — sustained useful FLOP/s divided by peak. Three conventions get called "MFU" and they differ by tens of percent, so state yours:

| Convention | Numerator | Notes |
|---|---|---|
| **MFU (6ND)** | $6N \times$ tokens/s | What `estimate_mfu` in [Chapter 14.7](07-pretraining-run.html) logs. Ignores attention: **under-reports** by 31% at our shape. |
| **MFU (model FLOPs)** | $(6N + 6Lsd)\times$ tokens/s | Counts attention. The PaLM-style definition. |
| **HFU (hardware FLOPs)** | model FLOPs **+ recomputation** | With full activation checkpointing you re-run the forward pass: $\text{HFU} \approx \tfrac{8}{6}\,\text{MFU}$. |

The flagship config runs *without* activation checkpointing (Chapter 14.7's table), so for us HFU = model-FLOPs MFU, and the only correction that matters is the 1.31× attention factor. Concretely: a loop logging **MFU(6ND) = 0.45** is really pushing $0.45 \times 1.31 \approx 0.59$ of A100 peak. That is at the *top* of what this shape sustains — large-$M$ GEMMs with a shallow $K=512$ reduction, 30 layers of kernel launches, and elementwise RMSNorm/RoPE/SiLU traffic between them — and it requires `torch.compile`, a FlashAttention-backed SDPA, and a micro-batch large enough to keep the GEMMs in the compute-bound regime.

!!! note "Are these GEMMs even compute-bound? A one-line roofline check"

    Take the SwiGLU up-projection at the flagship micro-batch: $M = 32 \times 2048 = 65{,}536$ tokens, $K = 512$, $N = 1408$. FLOPs $= 2MNK \approx 9.4\times10^{10}$; bf16 bytes moved $= 2(MK + KN + MN) \approx 2.5\times10^{8}$. Arithmetic intensity $\approx 377$ FLOP/byte, versus the A100's ridge point of $312\times10^{12} / 2.0\times10^{12} \approx 156$ FLOP/byte. Comfortably compute-bound — the large token dimension rescues us. Now halve the micro-batch to 4 sequences ($M = 8192$): intensity falls to ~186 FLOP/byte, still above the ridge but with far less headroom, and the shallow $K=512$ reduction leaves the tensor cores under-fed. **This is why "raise the micro-batch and use gradient accumulation" is the single highest-leverage throughput knob in the capstone**, and why the 24GB and 16GB tiers in Chapter 14.7 report materially lower MFU. See [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).

Putting the pieces together for the 18B-token stable phase. Hardware FLOPs $= 6ND \times 1.31 = (6 \times 1.014\times10^{8} \times 1.8\times10^{10}) \times 1.31 \approx 1.43\times10^{19}$. At a sustained $u = 0.59$ of A100 peak:

$$
t_{\text{pretrain}} = \frac{1.43\times 10^{19}}{0.59 \times 3.12\times 10^{14}} \approx 7.8\times10^{4}\ \text{s} \approx 21.6\ \text{GPU-hours},
$$

which corresponds to ~231,500 tokens/s and **MFU(6ND) ≈ 45%** — matching the worked example in Chapter 14.7 almost exactly. Run the same arithmetic at $u = 0.45$ (an un-compiled loop, an unfused attention path, or a smaller micro-batch) and you get **28.4 GPU-hours at MFU(6ND) ≈ 34%**. So the honest planning band is **21–29 GPU-hours for the stable phase**, and the bill below uses the well-tuned end while the re-run tax and the price axis absorb the rest.

The following helper turns any measured throughput into a cost — feed it the tokens/sec your loop actually logs, not the theoretical peak.

```python
# stacklm/cost.py
"""Turn a training run's measured throughput into GPU-hours and dollars.

Everything here is derived from numbers the training loop already logs
(tokens/sec, total tokens) -- no theoretical peaks required for the bill.
"""
from dataclasses import dataclass

# Stack-100M canonical constants (from the capstone plan; keep in sync with stacklm.config)
N_PARAMS = 101_400_000          # ~101.4M total params (tied embedding counted once)
N_LAYERS, D_MODEL = 30, 512
A100_BF16_PEAK_FLOPS = 312e12   # A100-80GB bf16 tensor-core peak (dense)


def training_flops(n_params: int, n_tokens: float) -> float:
    """Dense-transformer training FLOPs via the 6ND rule (matmuls only)."""
    return 6.0 * n_params * n_tokens


def attention_flops(n_tokens: float, seq_len: int,
                    n_layers: int = N_LAYERS, d_model: int = D_MODEL) -> float:
    """The FLOPs 6ND forgets: causal QK^T and AV, fwd+bwd, summed over layers.

    Per token per layer: 2*s*d forward (causal halves the s^2 term), x3 for
    fwd+bwd => 6*L*s*d per token. Parameter-free, so 6ND misses it entirely.
    """
    return 6.0 * n_layers * seq_len * d_model * n_tokens


def hardware_flops(n_tokens: float, seq_len: int, n_params: int = N_PARAMS,
                   recompute: bool = False) -> float:
    """What the GPU actually executes: matmuls + attention (+ recomputation)."""
    total = training_flops(n_params, n_tokens) + attention_flops(n_tokens, seq_len)
    return total * (8.0 / 6.0) if recompute else total   # full ckpt re-runs the fwd


def gpu_hours_from_throughput(n_tokens: float, tokens_per_sec: float) -> float:
    """The honest number: wall-clock GPU-hours from *measured* throughput."""
    return n_tokens / tokens_per_sec / 3600.0


def mfu_6nd(tokens_per_sec: float, n_params: int = N_PARAMS,
            peak_flops: float = A100_BF16_PEAK_FLOPS) -> float:
    """MFU under the 6ND convention -- what stacklm.train logs. Under-reports."""
    return 6.0 * n_params * tokens_per_sec / peak_flops


def peak_fraction(tokens_per_sec: float, seq_len: int,
                  peak_flops: float = A100_BF16_PEAK_FLOPS, **kw) -> float:
    """Fraction of the accelerator's peak actually being used (attention counted)."""
    return hardware_flops(tokens_per_sec, seq_len, **kw) / peak_flops


@dataclass
class Stage:
    name: str
    gpu_hours: float
    usd_per_gpu_hour: float = 1.80   # illustrative A100-80GB spot price; RE-PRICE IT
    extra_usd: float = 0.0           # non-GPU line items: teacher API, storage, egress

    @property
    def usd(self) -> float:
        return self.gpu_hours * self.usd_per_gpu_hour + self.extra_usd


if __name__ == "__main__":
    tps = 231_500                    # sustained tokens/sec logged by the pretrain loop
    hrs = gpu_hours_from_throughput(18e9, tps)
    print(f"stable phase: {hrs:.1f} GPU-hr  "
          f"MFU(6ND)={mfu_6nd(tps):.1%}  peak_frac={peak_fraction(tps, 2048):.1%}")
    # -> stable phase: 21.6 GPU-hr  MFU(6ND)=45.1%  peak_frac=59.2%
```

### 2026 hardware tiers: the same FLOPs, six different bills

The capstone's flagship tier is a single A100-80GB because that is the cheapest thing that comfortably fits the run — but by 2026 it is two generations old, and the *cost* chapter owes you the comparison. Peak figures below are vendor-published **dense** bf16 tensor-core numbers (halve any "with sparsity" marketing figure); prices are illustrative mid-2026 spot bands from the usual GPU-rental market and move fast enough that you must **re-price before you quote a number**.

| Tier | Memory | Dense bf16 peak | Dense FP8 peak | Illustrative spot USD/GPU-hr |
|---|---:|---:|---:|---|
| RTX 4090 (owned) | 24 GB | ~165 TFLOP/s | — | ~0 (electricity) |
| RTX 5090 (owned) | 32 GB | higher than 4090 | yes (Blackwell) | ~0 (electricity) |
| **A100-80GB SXM** | 80 GB | **312 TFLOP/s** | — | ~1.0–1.8 |
| H100-SXM | 80 GB | ~989 TFLOP/s | ~1,979 TFLOP/s | ~1.8–3.0 |
| H200-SXM | 141 GB | ~989 TFLOP/s | ~1,979 TFLOP/s | ~2.3–3.5 |
| B200 (Blackwell) | 192 GB | ~2.2 PFLOP/s | ~4.5 PFLOP/s | ~4–6 |
| AMD MI300X (ROCm) | 192 GB | ~1.3 PFLOP/s | yes | ~2–4 |

Three consequences for this project, and one for the next one:

- **The newest chip is often the cheapest, not just the fastest.** Repricing the 18B-token stable phase onto an H100 at a (deliberately conservative) $u = 0.45$ — small models get *less* efficient on bigger chips because they are more launch-bound and less able to fill the tensor cores — gives $1.43\times10^{19}/(0.45 \times 9.89\times10^{14}) \approx 9.0$ GPU-hours. At USD 2.50/hr that is USD 22.4 against the A100's USD 38.9 at USD 1.80/hr, *and* it finishes in 9 hours instead of 22. Exercise 7 works out the break-even rental price.
- **FP8 is a ≥1B lever, not a 100M one.** Hopper and Blackwell roughly double peak throughput in FP8, and per-tensor/per-block scaling recipes (see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html), and `torchao`'s `float8` integration used by torchtitan) make it usable for pretraining. But the realized end-to-end gain is well under the 2× peak ratio, and at 100M you are not GEMM-limited enough to collect most of it, while you *are* taking on real numerical risk. Turn it on at 1B+, where the GEMMs are fat enough to pay you back.
- **Consumer cards are a legitimate tier, not a consolation prize.** A 4090 at 24 GB cannot hold the flagship's 34 GB of activations (see §14.12.5), so you cut the micro-batch ~4× and lean on gradient accumulation; wall-clock roughly triples but the marginal dollar cost of a card you already own is electricity. The plan's USD 40 low end assumes exactly this.
- **Memory tier, not model size, decides when you shard.** Hold that thought — it is why the "7B does not fit one GPU" folklore is now hardware-generation-dependent (§14.12.5).

### The itemized bill

Every stage of the capstone, priced at an illustrative **USD 1.80/GPU-hour** A100-80GB spot rate. GPU-hours are the *sustained* wall-clock the loop would log at the well-tuned end of the band, not the theoretical floor. Non-GPU line items (teacher-model API for agent distillation, object storage for the ~200 GB of tokenized shards and checkpoints, egress) are called out explicitly.

| Stage (chapter) | GPU-hr | GPU USD | Non-GPU USD | Stage USD |
|---|---:|---:|---:|---:|
| Tokenizer BPE training (14.3) — mostly CPU | 0.3 | 0.54 | — | 0.54 |
| Scaling-law ladder {4M,9M,19M,44M} sweep (14.5) | 2.5 | 4.50 | — | 4.50 |
| **Pretrain, 18B tokens, WSD stable phase (14.7)** | 21.6 | 38.88 | — | 38.88 |
| Mid-training: anneal + 8192 ctx + capability, ~2B tok (14.8) | 3.5 | 6.30 | — | 6.30 |
| SFT on chat template (14.9) | 1.0 | 1.80 | — | 1.80 |
| DPO preference optimization (14.9) | 1.2 | 2.16 | — | 2.16 |
| GRPO / narrow RLVR on arithmetic (14.9) | 3.0 | 5.40 | — | 5.40 |
| Agent distillation: teacher traces + SFT (14.10) | 1.0 | 1.80 | 8.00 | 9.80 |
| Eval + int8/int4 quantization + export (14.11) | 1.2 | 2.16 | — | 2.16 |
| Object storage + egress (~200 GB, one month) | — | — | 5.00 | 5.00 |
| **Subtotal** | **35.3** | **63.54** | **13.00** | **76.54** |
| Re-run reality tax (~25% of GPU USD: OOMs, bad launches, 2 restarts) | — | 15.89 | — | 15.89 |
| **Grand total** | — | — | — | **≈ USD 92** |

Two of those lines deserve to be *derived* rather than accepted, because they are the ones a reader would otherwise have to take on faith:

**The mid-training line.** Chapter 14.8 slices ~2B tokens into ~1.2B of anneal at `seq_len=2048`, ~0.6B of long-context extension at 8192, and ~0.2B of capability injection at 8192. Applying the attention correction per sub-phase: the 2048 portion costs $6ND \times 1.31 = 9.6\times10^{17}$ hardware FLOPs, and the 8192 portion costs $6ND \times 2.24 = 1.09\times10^{18}$ — the shorter sub-phase is the more expensive one, entirely because of the quadratic attention term. Sum ≈ $2.0\times10^{18}$ FLOPs, and at the reduced $u \approx 0.50$ you sustain once the micro-batch drops 4× to fit 8192-token sequences, that is ~3.4 GPU-hours. The table's 3.5 falls out; it was never a guess.

**The ladder line.** Chapter 14.5 budgets the four-rung ladder at roughly 10% of the flagship's compute — ~2.5 GPU-hours, ~USD 4.50. That is the cheapest insurance in the project: it is what told you 20B tokens was the right budget before you spent 22 GPU-hours finding out.

Four things this table teaches that a bare "USD 92" hides:

1. **Pretraining is ~50% of the bill and over-training is ~90% of *that*.** A Chinchilla-optimal run (1.7B tokens) would have cost ~2.0 GPU-hours, ~USD 3.7. We deliberately spent the other ~USD 41 to buy a permanently cheaper-to-serve model. That is the deployment-economics trade made *visible*.
2. **Alignment + agent is cheap; the teacher API is the surprise.** All of SFT+DPO+GRPO+distill is ~USD 19, and USD 8 of that is *not* your GPU at all — it is API calls to a large teacher to generate ReAct trajectories you then filter and distill. At 1B and beyond, teacher/data-generation cost often *exceeds* your own training cost (§14.12.5 puts a number on it).
3. **The reality tax is real, and 25% is the optimistic version.** No first run of a 30-layer model at high LR survives cleanly. You will hit an OOM from a mis-set gradient-accumulation count, a loss spike from an un-clipped attention logit, a corrupted shard. A practitioner who has done this before pays ~25%; a first-timer pays closer to 50%, which pushes the same bill to ~USD 108. **That gap — not the FLOPs — is why the sticker says "~USD 100."**
4. **The bill is a function of three inputs, only one of which is the model.** Tokens, MFU, and USD/GPU-hr. Two of the three are yours to control; the third is a market.

!!! note "Why the bill is a band, not a point"

    The plan quotes **USD 40–100**, and both ends are honest. Re-run the same table with 2026-typical A100 spot at USD 1.20/hr and it lands at ~USD 66; run it on an H100 (2.4× the throughput at USD 2.50/hr) and it lands at ~USD 59 *and finishes in a third of the wall-clock*; run it on an owned 4090 and the GPU column collapses to electricity, leaving the ~USD 13 of API and storage. The high end is the fully-loaded invoice: A100 at USD 1.80/hr, a 34-hour un-tuned pretrain, a first-timer's re-run tax. Same recipe, same tokens — the ~2.5× spread is *entirely* GPU market price, kernel quality, and how many times you fat-finger a launch. Quote the number with its assumptions attached: **a dollar figure without a USD/GPU-hr, an MFU, and the MFU's convention is not reproducible.**

{{fig:capstone-cost-anatomy}}

!!! example "Worked example: does over-training actually pay off?"

    Suppose you will serve Stack-100M for one billion inference requests, each generating 256 tokens. Inference FLOPs are $\approx 2ND$, so total serving compute is

    $$
    C_{\text{serve}} \approx 2 \times (1.014\times10^8) \times (10^{9}\times 256) \approx 5.2\times10^{19}\ \text{FLOPs.}
    $$

    That is ~4.3× the *entire* 20B-token pretraining budget ($1.22\times10^{19}$). Now imagine over-training let you hit your target quality at 100M instead of needing a 200M model (roughly 2× the serving FLOPs). The extra ~USD 41 you spent over-training saves you ~$5.2\times10^{19}$ FLOPs — on the order of four full pretrain budgets — *per billion requests*. The over-training pays for itself many times over the moment you deploy at scale. This is exactly the "inference-aware over-training" of Sardana et al. (*Beyond Chinchilla-Optimal*, 2024), and it only sharpens as you scale. Exercise 5 derives the break-even request count, which — pleasingly — is independent of model size.

{{fig:capstone-pay-once-save-forever}}

---

## 14.12.4 Reproducibility: Making the Run Rebuildable

A result you cannot reproduce is an anecdote, not an experiment. At 100M on one GPU, reproducibility is *achievable to the bit* if you are disciplined; at 1B across a cluster it becomes merely *approximate* (non-deterministic all-reduce ordering, hardware variance, cuBLAS split-k) — which is all the more reason to nail the discipline now while you can verify it.

The checklist has six pillars: seeds, config, data, environment, checkpoints, and release. We give real code for each — it lives in `stacklm/repro.py` and is called from `stacklm.train` — and, just as importantly, we name the open-source tool that does each job properly at scale, because hand-rolling all six is a teaching device, not a recommendation.

### Seeds and determinism

```python
# stacklm/repro.py  (part 1: determinism)
import os, random
import numpy as np
import torch


def seed_everything(seed: int = 1337, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline touches. Returns nothing; mutates global state.

    We seed FOUR independent generators: Python's `random`, NumPy (used by the
    tokenizer and data sampler), the CPU torch RNG, and all CUDA device RNGs.
    Missing any one silently reintroduces nondeterminism (e.g. dropout masks,
    data-order shuffles, or Muon's Newton-Schulz init).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)   # affects dict/set iteration order
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Trade a little throughput for reproducible kernels. cuBLAS needs the
        # workspace env var set BEFORE the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False   # disable autotuner (nondeterministic pick)


def rng_state_dict() -> dict:
    """Snapshot every RNG so a resumed run continues the SAME random stream.

    NOTE the numpy conversion: `np.random.get_state()` returns a tuple containing
    a raw ndarray, which `torch.load(weights_only=True)` will refuse. We store it
    as a torch tensor so the whole checkpoint stays loadable under the SAFE
    unpickler (see `load_trainer_state` below). This is not fussiness -- it is
    the difference between a checkpoint you can hand to a stranger and one you
    cannot.
    """
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (kind, torch.from_numpy(keys.copy()), int(pos),
                  int(has_gauss), float(cached)),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    kind, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
```

The subtle one is RNG *state* (not just the seed). If you resume a run and re-seed from the same integer, you replay the random stream *from the start* — reusing data batches you already consumed. You must save and restore the RNG *state* at the resume step, exactly as in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html).

!!! tip "Practitioner tip: let the dataloader own its own position"

    Restoring RNG state fixes the *shuffle*; it does not fix "which shard and offset were we at?" if your sampler is stateful. `torchdata`'s `StatefulDataLoader` (`from torchdata.stateful_dataloader import StatefulDataLoader`) is a drop-in `DataLoader` replacement whose `state_dict()` / `load_state_dict()` capture worker positions, so a resume continues mid-epoch at exactly the right sample instead of restarting the epoch. Save it alongside the model state. This is the same mechanism torchtitan uses for its 1B+ pretraining resumes.

### Config: one frozen object, one hash

Every knob that affects the result — architecture, optimizer, schedule, data mix — lives in one frozen dataclass. Hash its canonical serialization and stamp the hash onto every checkpoint and log line. Two runs with the same config hash are the same experiment; a changed hash is a new one.

```python
# stacklm/repro.py  (part 2: config + data + env provenance)
import hashlib, json, subprocess
from dataclasses import asdict, is_dataclass


def _canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace jitter. Dataclasses -> dict."""
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config) -> str:
    """Short, stable fingerprint of the full run configuration."""
    blob = _canonical_json(config).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]   # 12 hex chars disambiguates plenty
```

**The libraries that do this properly.** A frozen dataclass is the right *mental model*; at more than a handful of experiments you want a config framework. **Hydra** (with **OmegaConf**) gives you composable YAML groups plus command-line overrides and writes the fully-resolved config into each run directory — hash *that* resolved file, not your defaults. **pydantic-settings** is the lighter-weight option when you want validation and environment-variable binding without Hydra's multirun machinery. Both preserve the invariant that matters: exactly one serialized object determines the run.

**And log the hash somewhere you will look.** Print it as line 1, then push it to an experiment tracker so the loss curve and the config travel together:

```python
# stacklm/tracking.py -- one place where the config hash meets the loss curve.
def start_tracking(config, backend: str = "wandb"):
    """Open a run in an experiment tracker keyed by the config hash.

    Using the config hash as the run id is the trick that makes this useful:
    a resumed job reattaches to the SAME run instead of forking a new curve,
    and two runs with identical configs collide loudly instead of silently
    becoming 'experiment 47' and 'experiment 63'.
    """
    h = config_hash(config)
    if backend == "wandb":
        import wandb
        return wandb.init(project="stacklm-100m", id=h, resume="allow",
                          config=asdict(config), tags=[f"cfg:{h}"])
    if backend == "mlflow":
        import mlflow
        mlflow.set_experiment("stacklm-100m")
        run = mlflow.start_run(run_name=h)
        mlflow.log_params(asdict(config))
        return run
    raise ValueError(backend)   # 'aim' and plain TensorBoard are equally fine
```

**Weights & Biases** and **MLflow** are the two defaults; **Aim** is a good self-hosted alternative and plain **TensorBoard** (`torch.utils.tensorboard`) is entirely sufficient for a single-GPU capstone. What matters is not which one, but that *the config hash, the git commit, the corpus hash, and the loss curve are queryable together six months later.*

### The data manifest

The dirtiest source of "it doesn't reproduce" is data drift: a shard silently re-tokenized, a file truncated by a failed download, the mix reweighted. A **data manifest** pins the *content* of every shard by SHA-256, not its path, so you can prove the bytes are identical.

```python
def data_manifest(shard_paths: list[str]) -> dict:
    """Content-addressed manifest of the tokenized corpus.

    For each .bin memmap shard we record its SHA-256, byte size, and token count
    (uint16 tokens => 2 bytes each, valid because vocab_size=32768 < 65536). The
    top-level `corpus_hash` fingerprints the WHOLE dataset: change any shard, or
    the order, and it changes.
    """
    shards, running = [], hashlib.sha256()
    for path in sorted(shard_paths):                 # sorted => order-independent of FS listing
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
                h.update(chunk)
                size += len(chunk)
        digest = h.hexdigest()
        running.update(digest.encode())
        shards.append({"path": path, "sha256": digest,
                       "bytes": size, "tokens": size // 2})
    return {
        "corpus_hash": running.hexdigest()[:16],
        "n_shards": len(shards),
        "total_tokens": sum(s["tokens"] for s in shards),
        "shards": shards,
    }
```

At capstone scale (~200 GB of shards) this file *is* your data version control. Past that, use a tool: **DVC** tracks large files by content hash in git without putting the bytes in git, **git-lfs** does the crude version of the same, and if your corpus lives on the Hub, `datasets` already computes a deterministic **fingerprint** for every transformed dataset (`ds._fingerprint`) plus a per-file `sha256` in the repo metadata — record those and you get the same guarantee for free. The invariant to preserve is: **the corpus is identified by its content, never by a path or a date.**

### Environment pinning

Same code + same data + *different* PyTorch build can still diverge (a changed default in a fused kernel, a new cuDNN heuristic, a different Triton autotune result). Capture the exact environment as data, alongside the run — and capture the *lockfile*, not a prose reminder to make one.

```python
def _sha256_file(path: str, n: int = 16) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:n]
    except OSError:
        return None


def environment_fingerprint(lockfile: str = "uv.lock") -> dict:
    """Everything outside your code that can change the result."""
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return None
    return {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),   # True => uncommitted: DANGER
        # The dependency set, captured as DATA rather than as a comment:
        "lockfile": lockfile,
        "lockfile_sha256": _sha256_file(lockfile),
        # Set by the launcher, e.g. `docker inspect --format='{{index .RepoDigests 0}}'`
        # or an Apptainer/Singularity image hash. The only true hermetic pin.
        "container_digest": os.environ.get("STACKLM_IMAGE_DIGEST"),
    }
```

A `git_dirty=True` at training time is a red flag: it means the code that produced the checkpoint was never committed, so the run is not reproducible from any commit. Fail loudly on it for the flagship run. In 2026 the ergonomic choice for the lockfile is **uv** (`uv lock` → `uv sync --frozen`), which resolves and installs a fully-pinned environment in seconds; `poetry.lock` and a `pip freeze` requirements file are equivalent in kind, weaker in guarantee. And the only *hermetic* pin is a container digest — a **Docker** or **Apptainer** image referenced by `sha256:` rather than by tag, since tags are mutable and `pytorch:latest` is a moving target that will silently change your kernels.

### Checkpoint hygiene, and the format that does not execute code

Weld provenance *into* the checkpoint so a checkpoint directory is self-describing: not just weights but the config hash, the corpus hash, the git commit, the RNG state, and the step.

There is a security dimension here that the naive `torch.save`/`torch.load` pattern gets wrong. `torch.load` deserializes Python pickles, which can execute arbitrary code; since PyTorch 2.6 the default is `weights_only=True`, and passing `weights_only=False` opts *out* of that protection. A checkpoint you downloaded is untrusted input. The fix is to split the checkpoint: **weights in `safetensors`** (a zero-copy, mmap-able, executes-nothing tensor container), trainer state in a `weights_only`-safe torch blob, and provenance in plain JSON you can read without importing anything.

```python
# stacklm/repro.py  (part 3: self-describing, safe-by-default checkpoints)
import shutil
from pathlib import Path
from safetensors.torch import save_file, load_file


def build_provenance(config, shard_paths: list[str]) -> dict:
    """Assemble the full provenance record stamped into every checkpoint & log."""
    return {
        "config_hash": config_hash(config),
        "config": asdict(config) if is_dataclass(config) else dict(config),
        "data_manifest": data_manifest(shard_paths),
        "env": environment_fingerprint(),
    }


def save_checkpoint(path, model, optimizer, step, provenance, dataloader=None):
    """Write a self-describing checkpoint DIRECTORY:

        ckpt_step12000/
        |- model.safetensors   weights only; no pickle, mmap-able, framework-neutral
        |- trainer.pt          optimizer + RNG + dataloader state (weights_only-safe)
        |- provenance.json     config, config_hash, corpus_hash, env, git commit

    Written to a .tmp sibling and atomically renamed, so a crash mid-write can
    never leave a half-written 'latest' behind (checkpoint hygiene rule #1).
    """
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # 1. Weights. `.contiguous()` because safetensors refuses shared/strided storage,
    #    which tied embeddings would otherwise trip over.
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              tmp / "model.safetensors",
              metadata={"step": str(step), "config_hash": provenance["config_hash"]})

    # 2. Trainer state. BOTH the Muon and the AdamW param-group states live here.
    trainer = {"step": step, "optimizer": optimizer.state_dict(), "rng": rng_state_dict()}
    if dataloader is not None and hasattr(dataloader, "state_dict"):
        trainer["dataloader"] = dataloader.state_dict()      # StatefulDataLoader
    torch.save(trainer, tmp / "trainer.pt")

    # 3. Provenance, as human-readable JSON.
    (tmp / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)                                          # atomic on POSIX


def load_checkpoint(path, model, optimizer, expect_config_hash=None, dataloader=None):
    """Load a checkpoint directory, refusing a config mismatch and never unpickling
    anything the safe loader would not accept."""
    path = Path(path)
    provenance = json.loads((path / "provenance.json").read_text())
    got = provenance["config_hash"]
    if expect_config_hash is not None and got != expect_config_hash:
        raise RuntimeError(
            f"Config hash mismatch: checkpoint={got} expected={expect_config_hash}. "
            "You are resuming a run with a DIFFERENT architecture/optimizer config."
        )
    model.load_state_dict(load_file(path / "model.safetensors"))
    trainer = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(trainer["optimizer"])
    load_rng_state(trainer["rng"])
    if dataloader is not None and "dataloader" in trainer:
        dataloader.load_state_dict(trainer["dataloader"])
    return trainer["step"]
```

!!! warning "Checkpoint hygiene footguns"

    Four failures cost real GPU-hours. **(1) Non-atomic writes:** if the process dies mid-write, the file is truncated and unloadable — always write to a `.tmp` sibling then rename. **(2) Dropping optimizer state:** saving only weights makes a resumed run silently re-warm Muon momentum and Adam moments from zero, spiking loss for hundreds of steps (see [Checkpointing](../03-pretraining/12-checkpointing-fault-tolerance.html)). **(3) `weights_only=False`:** loading a third-party `.pt` with the unsafe unpickler executes whatever the author put in it. Ship `safetensors`. **(4) Unbounded retention:** ~**1.28 GB** of model+optimizer state per checkpoint (see the blended byte-count in §14.12.5 — it is *not* 16 B/param, because Muon carries one momentum buffer, not two moments) written every 500 steps fills a disk overnight. Keep a rolling window of the last *k* plus milestone checkpoints (end of stable phase, end of decay), and delete the rest — Exercise 6 implements the policy.

### Release: the artifact other people can use

The pillar most tutorials skip. A run is reproducible-in-principle when the six items above are pinned; it is reproducible-in-practice when someone else can `pip install`, download, and re-measure. Three concrete steps:

```bash
# 1. Publish weights + tokenizer + config, tagged with the provenance you already have.
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("your-org/stack-100m-base", repo_type="model", exist_ok=True)
api.upload_folder(folder_path="stacklm-100m/base",
                  repo_id="your-org/stack-100m-base",
                  commit_message="base @ cfg 4f2a9c1b8e30, corpus a17c93de5b204f61")
PY

# 2. Re-measure the headline evals against a PINNED harness commit, not `main`.
uv pip install "lm_eval @ git+https://github.com/EleutherAI/lm-evaluation-harness@<commit-sha>"
lm_eval --model hf \
        --model_args pretrained=./stacklm-100m/chat,dtype=bfloat16 \
        --tasks arc_easy,hellaswag,piqa --batch_size 16 \
        --output_path evals/results.json          # commit this file next to the weights
```

3. **Write the model card, and mean it.** `README.md` in the model repo carries the intended use and — the part that matters legally as much as scientifically — **data provenance and licensing**. FineWeb and FineWeb-Edu are released under ODC-By; Cosmopedia is synthetic and carries its generator's terms; code from The Stack / StarCoder carries per-repository licenses and an opt-out mechanism you are expected to respect. Derived weights inherit obligations from all of them. State the mix, the token count, the eval numbers with the harness commit that produced them, and the honest capability ceiling from Chapter 14.11. A model card that omits the training mix is not a model card.

### The one-page checklist

Before you press go on a flagship run, this must all be true. It is the difference between science and a story.

```text
REPRODUCIBILITY PREFLIGHT  (stacklm)
[ ] seed_everything(1337, deterministic=True) called before model init
[ ] RNG *state* + dataloader state saved in checkpoint (not just the integer seed)
[ ] config frozen in one object (dataclass / Hydra-resolved YAML); hash on log line 1
[ ] tracker run opened with id=config_hash (W&B / MLflow / Aim / TensorBoard)
[ ] data_manifest.json written; corpus_hash matches the intended mix
[ ] held-out eval shard provably EXCLUDED from the manifest (contamination, Ch. 14.11)
[ ] git commit clean (git_dirty == False); commit hash logged
[ ] uv.lock committed AND its sha256 recorded; container digest recorded
[ ] checkpoints: safetensors weights + weights_only-safe trainer state + provenance.json
[ ] checkpoints atomic (tmp+rename), retention policy set (keep_last + milestones)
[ ] eval harness pinned to a commit sha; results.json committed beside the weights
[ ] a 50-step CPU toy run reproduces bit-identically across two invocations
```

That last line is the cheap insurance: a hermetic 50-step CPU run (tiny vocab, tiny model, synthetic data — the same toy path the book's CI smoke-tests) that must produce *identical* loss across two invocations. If it does not, your determinism is broken and no amount of cluster time will fix it later.

---

## 14.12.5 What Breaks at 1B — and What to Change

The capstone's proudest claim is that the *entire* 100M pipeline fits on one GPU. The instructive question is: run the exact same code targeting 1B parameters — what breaks first? We take the failures in the order you hit them.

### Data: the wall you hit before the compute wall

A 1B model at Chinchilla-optimal wants ~20B tokens; over-trained for deployment (the capstone's whole philosophy) it wants **200B–1T tokens**. Our ~20B-token corpus is now 10–50× too small. You cannot fix this by looping the same data: repetition helps only up to ~4 epochs and its value collapses toward zero by ~16 epochs (Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023). So the first thing that breaks at 1B is *data volume*, and the fix is more *and* cleaner data:

- **Volume, with a real pipeline.** Move from a FineWeb-Edu *sample* to the full FineWeb / FineWeb-Edu (trillions of tokens). At that scale you stop hand-rolling dedup and use the tool that actually built FineWeb: **`datatrove`** (HuggingFace), whose pipeline blocks — `MinhashDedupSignature` → `MinhashDedupBuckets` → `MinhashDedupCluster` → `MinhashDedupFilter` — run MinHash-LSH near-duplicate removal as a sharded, resumable, Slurm- or Ray-schedulable job rather than a laptop script. **NVIDIA NeMo Curator** (GPU-accelerated fuzzy dedup and classifier filtering) and **AI2 `dolma`** (the toolkit behind the Dolma corpus) are the credible alternatives. All three implement the same conceptual pipeline as [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) — the difference is that they survive a trillion documents.
- **Quality and mix.** The marginal token matters more at 1B; retune the mix (the 70/15/10/5 FineWeb-Edu / Cosmopedia / code / math split) using the domain-weighting methods of [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html), and lean harder on synthetic data (Cosmopedia-style, see [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)) to buy quality where raw web runs thin.

The uncomfortable truth: at 1B, *data engineering* — not model code — is where most of your time and a growing share of your budget goes. We will put a number on "growing share" two subsections from now.

### Memory: the accounting the folklore gets wrong

Does 1B even fit on one 80 GB GPU? Do the arithmetic properly, because both the usual per-parameter rule and the usual conclusion are wrong for this project.

**First error: 16 bytes/param assumes pure AdamW.** Stack-100M uses the Muon+AdamW hybrid fixed in the plan — Muon for the 2D hidden matrices, AdamW for embeddings, norms, and 1D parameters. Muon keeps **one** momentum buffer, not Adam's $m$ *and* $v$, so its groups cost 12 bytes/param, not 16.

**Second error: weights+optimizer is not what fills the GPU.** Activations are. Here is the accounting that actually predicts an OOM:

```python
# stacklm/scaleup.py -- what really fills an 80 GB GPU?

def state_bytes_per_param(optimizer: str) -> int:
    """Steady-state bytes per parameter for mixed-precision training.

    Common to all: bf16 weights(2) + bf16 grads(2) + fp32 master copy(4) = 8 B.
      adamw      + fp32 m and v        -> +8  => 16 B/param
      muon       + ONE fp32 momentum   -> +4  => 12 B/param   <-- our 2D hidden matrices
      adamw8bit  + int8 m,v (bnb)      -> +2  => 10 B/param
      adamw_bf16 + bf16 m,v, no master -> ...  =>  8 B/param   (see Ch. 4.10)
    """
    return {"adamw": 16, "muon": 12, "adamw8bit": 10, "adamw_bf16": 8}[optimizer]


def optimizer_state_gb(groups: dict[str, float]) -> float:
    """groups maps optimizer name -> n_params in that group. Returns GB."""
    return sum(n * state_bytes_per_param(o) for o, n in groups.items()) / 1e9


def activation_gb(n_layers: int, d_model: int, seq_len: int, micro_batch: int,
                  n_heads: int, flash: bool = True, recompute: bool = False) -> float:
    """Per-layer activation memory, Korthikanti et al. (2022), 16-bit storage:

           bytes_per_layer = s*b*h * (34 + 5*a*s/h)

    The 5*a*s/h term is the materialized s x s attention matrix. A fused
    FlashAttention kernel never writes it, so with `flash=True` it vanishes --
    which is the entire reason 8192-token mid-training fits at all.
    Full activation checkpointing stores only each layer's input: 2*s*b*h.
    """
    s, b, h, a = seq_len, micro_batch, d_model, n_heads
    if recompute:
        per_layer = 2 * s * b * h
    else:
        per_layer = s * b * h * (34 + (0.0 if flash else 5 * a * s / h))
    return n_layers * per_layer / 1e9


# --- Stack-100M at the flagship config (Ch. 14.7: micro-batch 32 x 2048, no recompute)
STACK_100M = {"muon": 84.5e6,     # 2D hidden matrices across the 30 blocks
              "adamw": 16.83e6}   # tied embedding + norms + 1D params
print(f"weights+opt : {optimizer_state_gb(STACK_100M):5.2f} GB")
print(f"activations : {activation_gb(30, 512, 2048, 32, 8):5.1f} GB  (FlashAttention)")
print(f"  no flash  : {activation_gb(30, 512, 2048, 32, 8, flash=False):5.1f} GB  (!!)")
print(f"  recompute : {activation_gb(30, 512, 2048, 32, 8, recompute=True):5.1f} GB")
# weights+opt :  1.28 GB
# activations :  34.2 GB  (FlashAttention)
#   no flash  : 195.3 GB  (!!)
#   recompute :   2.0 GB
```

Read those four numbers again. On the flagship A100, **weights and optimizer state are 1.28 GB — under 4% of the memory in play — while activations are 34 GB.** Without FlashAttention the run does not fit on any GPU that exists. With full activation checkpointing it would fit in 4 GB, at the cost of a 33% FLOP surcharge (that is the MFU-vs-HFU gap from §14.12.3), which is precisely why Chapter 14.7 leaves recompute *off* at 80 GB and *on* at 24 GB.

Now scale the same two functions to 1B and beyond:

```python
for name, groups in [
    ("Stack-100M", {"muon": 84.5e6,  "adamw": 16.8e6}),
    ("1B  dense",  {"muon": 0.9e9,   "adamw": 0.1e9}),
    ("7B  dense",  {"muon": 6.3e9,   "adamw": 0.7e9}),
    ("70B dense",  {"muon": 63e9,    "adamw": 7e9}),
]:
    print(f"{name:11s} weights+opt = {optimizer_state_gb(groups):7.1f} GB")
# Stack-100M  weights+opt =     1.3 GB
# 1B  dense   weights+opt =    12.4 GB
# 7B  dense   weights+opt =    86.8 GB
# 70B dense   weights+opt =   868.0 GB

# ...and the activations that decide whether it ACTUALLY fits (1B: L=24, d=2048, a=16)
for mb in (4, 8, 16, 32):
    act = activation_gb(24, 2048, 2048, mb, 16)
    print(f"1B micro-batch {mb:2d}: {act:5.1f} GB act + 12.4 GB state = {act+12.4:5.1f} GB")
# 1B micro-batch  4:  13.7 GB act + 12.4 GB state =  26.1 GB
# 1B micro-batch  8:  27.4 GB act + 12.4 GB state =  39.8 GB
# 1B micro-batch 16:  54.8 GB act + 12.4 GB state =  67.2 GB
# 1B micro-batch 32: 109.6 GB act + 12.4 GB state = 122.0 GB   <- OOM on 80 GB
```

So **1B fits a single 80 GB GPU, but the headroom is activations, not weights** — you get micro-batch 8, comfortably, or 16 if you are careful, and 32 only with recompute. The naive "16 GB of state, 64 GB to spare" reading is off by a factor of five in the thing that matters.

Note also what happened to the folklore threshold: 7B needs ~87 GB of weights+optimizer state, so it does *not* fit an 80 GB A100 or H100 — but it *does* fit an **H200 (141 GB)**, a **B200 (192 GB)**, or an **MI300X (192 GB)** with room for a real batch. **"When do I need FSDP?" is a hardware-generation question, not a model-size question.** The rule that survives is the ratio: shard when weights + optimizer + activations exceed your device memory, and re-derive it every time you change tiers.

Levers, in the order you should reach for them, all covered in [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html): raise gradient accumulation (free), turn on activation checkpointing (33% FLOP surcharge), switch AdamW groups to 8-bit (`bitsandbytes.optim.AdamW8bit`, ~6 GB saved at 7B), keep optimizer states in bf16, then — only then — shard.

### Time, and the parallelism ladder (with the libraries you would actually use)

The recipe does not break on memory at 1B. It breaks on *time*: 1B over 200B tokens is $6 \times 10^{9} \times 2\times10^{11} = 1.2\times10^{21}$ FLOPs, ~99× the capstone pretrain, i.e. **thousands of GPU-hours** on one card. That is when you go multi-GPU, and the progression is exactly the one taught in Part III — but the *libraries* have moved, and naming the 2019 ones is no longer sufficient.

1. **DDP / data parallelism first.** Replicate the model on each of $g$ GPUs, split the batch, all-reduce gradients. Near-linear speedup while the model *fits* per GPU (true at 1B). `torch.nn.parallel.DistributedDataParallel` launched with `torchrun`. This is your first move and it turns thousands of GPU-hours of wall-clock into hundreds.
2. **FSDP2 when it stops fitting — and note it is `fully_shard`, not the old wrapper.** Shard parameters, gradients, and optimizer state across data-parallel ranks so each holds $1/g$ of the per-parameter bytes (ZeRO-3 in DeepSpeed's vocabulary). The current PyTorch API is **`torch.distributed.fsdp.fully_shard`** ("FSDP2"), which shards *per parameter* using DTensor and composes cleanly with `torch.compile` and tensor parallelism; the older `FullyShardedDataParallel` module wrapper is the one every stale tutorial shows and the one you should not start from. At 1B you *can* use FSDP2 purely to buy activation headroom; you do not *need* it.

    ```python
    # 100M -> 1B: the distributed diff really is about this small.
    import torch
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in model.blocks:        # shard each transformer block: overlaps
        fully_shard(block, mp_policy=mp)   # all-gather(i+1) with compute(i)
    fully_shard(model, mp_policy=mp)  # then the root, for embeddings/head
    ```

3. **Save with DCP, not `torch.save`.** A sharded run's `state_dict` is spread across ranks, and a plain save bakes in the world size — so resuming on 16 GPUs from an 8-GPU checkpoint fails. **`torch.distributed.checkpoint`** writes a resharding-aware, parallel checkpoint:

    ```python
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    msd, osd = get_state_dict(model, optimizer)
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=f"ckpt/step{step}")
    # ...later, on a DIFFERENT number of ranks:
    dcp.load({"model": msd, "optim": osd}, checkpoint_id=f"ckpt/step{step}")
    set_state_dict(model, optimizer, model_state_dict=msd, optim_state_dict=osd)
    ```

4. **Tensor + pipeline parallelism only when a layer is too big or too slow (tens of B+).** Split individual matmuls across GPUs (tensor) and stages of layers across GPUs (pipeline), the Megatron-style approach of [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html) and [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html). You do **not** reach for this at 1B — pipeline bubbles and tensor-parallel communication are pure overhead you only accept when nothing else works. Naming it here is the point: knowing you *don't* need it at 1B is as valuable as knowing you *will* at 30B.

!!! tip "Do not hand-roll the 1B run: torchtitan is this recipe, at 1B"

    The honest advice for someone taking Stack-100M to 1B is: read your own `train.py` next to **`pytorch/torchtitan`**, PyTorch's reference pretraining stack, and port rather than rebuild. It is the same loop you wrote — bf16 autocast, gradient accumulation, WSD-family schedules, throughput/MFU logging — with FSDP2, tensor/pipeline/context parallelism, `torch.compile`, Float8 training via `torchao`, and DCP already wired together and config-driven. The alternatives, each with a different center of gravity: **NVIDIA Megatron-Core** (the productized Megatron-LM, maximal 3D-parallel performance and the reference for tensor parallelism), **DeepSpeed** (ZeRO-1/2/3, offload, and DeepSpeed-MoE), and **HuggingFace `nanotron`** (a compact, readable 3D-parallel pretrainer in the nanoGPT spirit). Whichever you pick, the value of having written the single-GPU loop yourself is that you can now *read* theirs.

{{fig:capstone-scaleup-ladder}}

### What does 1B actually cost?

A cost chapter that spends a third of its length on "the path to 1B" owes you the number. Run the same machinery as §14.12.3 on a 1B model (24 layers, $d=2048$, so the attention correction is only $1 + Lsd/N = 1.08$) over 200B tokens:

| Path | Hardware FLOPs | Assumed $u$ | GPU-hr | Wall-clock on 8 GPUs | GPU USD |
|---|---:|---:|---:|---:|---:|
| 8×A100-80GB, bf16, DDP | $1.30\times10^{21}$ | 0.55 | ~2,100 | ~12 days | ~USD 2,730 @ 1.30/hr |
| 8×H100-SXM, bf16, FSDP2 | $1.30\times10^{21}$ | 0.50 | ~730 | ~4.2 days | ~USD 1,830 @ 2.50/hr |
| 8×H100-SXM, **FP8** | $1.30\times10^{21}$ | 0.50 (fp8 peak) | ~450–600 | ~3 days | ~USD 1,100–1,500 |

(The FP8 row assumes a realized 1.2–1.6× end-to-end speedup, not the 2× peak ratio — that gap is scaling-factor overhead and the non-GEMM work FP8 does not touch. Add ~10% to every GPU-hour figure for data-parallel communication and stragglers.)

Now the part that is genuinely different at 1B — the non-GPU column:

| Line item | Stack-100M (20B tok) | Stack-1B (200B tok) | Why it moves |
|---|---:|---:|---|
| Tokenizer + scaling-law ladder | USD 5.04 | ~USD 100 | re-run the ladder with a higher top rung (§14.12.2) |
| Pretrain GPU | USD 38.88 | ~USD 1,830 | 98× the FLOPs, ~3× faster hardware |
| Mid-training GPU | USD 6.30 | ~USD 190 | ~20B tokens of anneal + long-context |
| Post-training + agent distill GPU | USD 11.16 | ~USD 250 | GRPO is generation-bound; scales worse than FLOPs |
| Eval + quantize + serve GPU | USD 2.16 | ~USD 60 | bigger harness, more probes |
| **Data pipeline (CPU + storage)** | USD 5.00 | **~USD 800** | datatrove MinHash over ~1T docs; 2–10 TB of shards |
| **Synthetic / teacher data** | USD 8.00 | **~USD 1,500** | e.g. 10B synthetic tokens at ~USD 0.15 / M tokens |
| Re-run reality tax (25% of GPU) | USD 15.89 | ~USD 610 | |
| **Total** | **≈ USD 92** | **≈ USD 5,300** | |

Two readings of that table, and the second is the thesis of the whole chapter:

- **1B is ~58× the cost, not ~98×.** The FLOPs went up 98×, but three generations of hardware and a 3× cheaper FLOP absorbed part of it. The compute wall moves under you; plan against *current* price-performance, never against the number in a two-year-old blog post.
- **The non-GPU share went from 13% to 43%.** At 100M, "cost" means GPU-hours and the data is basically free. At 1B, the dedup pipeline and the synthetic-data bill together (~USD 2,300) rival the entire pretraining run (~USD 1,830). This is the quantitative form of "at 1B, data engineering is where the budget goes" — and it is why the tooling recommendations in the data subsection are not incidental. Extrapolate the trend one more decade and you can see why frontier labs are data organizations that happen to own GPUs.

### Learning rate and batch size: what to rescale

You cannot copy Stack-100M's hyperparameters to 1B unchanged; width and batch both grew. Two rules, both from [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html):

- **Learning rate vs. width.** Optimal LR shrinks as the model widens. The principled tool is **μP (Maximal Update Parametrization)**: tune LR on a small proxy, then transfer it to the large model with width-dependent scaling so the *same* tuned value stays optimal — note this is the same "fit small, predict large" discipline as §14.12.2, applied to hyperparameters instead of loss, and it is well supported in torchtitan and Megatron-Core. A cheaper heuristic is $\eta \propto 1/\sqrt{d_{\text{model}}}$; going from $d{=}512$ to $d{=}2048$ roughly *halves* the peak LR. The AdamW groups (embeddings, norms) and the Muon groups (2D hidden matrices) rescale differently — Muon's orthogonalized update is naturally more scale-robust, one reason the hybrid was chosen (see [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html)) — but you still retune.
- **Batch size and warmup.** Larger models have a larger **critical batch size** — the point past which more parallelism stops helping per-step progress. At 1B you can raise the global batch from ~0.5M tokens toward ~1–2M tokens (more data-parallel ranks make this nearly free) and extend warmup proportionally. The WSD schedule transfers cleanly: keep the long stable phase, keep the short decay phase as your mid-training quality anneal — the schedule's shape is scale-invariant even though the numbers move.

Keep MuonClip / QK-clip on. Attention-logit blow-ups get *worse* at scale and high LR, and QK-clip is exactly the stability fix (Kimi K2, Moonshot AI, 2025) that made Muon usable at scale — do not drop it when you can least afford instability. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

### The MoE fork in the road

At 1B dense you spend 1B params of *compute* per token. **Mixture-of-Experts (MoE)** breaks that link: route each token to a few experts so *total* params (capacity) far exceed *active* params (compute/token). This is the single highest-leverage architecture change on the path beyond 1B, and the capstone's own plan lists it as the scale-up option.

The modern design point is **fine-grained experts with shared experts**, from **DeepSeekMoE** (Dai et al., 2024): split each expert into many smaller ones for finer specialization, and keep a few *always-on* shared experts to absorb common patterns so the routed experts can specialize. **Qwen3-MoE** (Qwen Team, 2025) is a recent production instance of the same family. The trade you are making:

$$
\underbrace{N_{\text{total}}}_{\text{capacity, sets quality}} \gg \underbrace{N_{\text{active}} = N_{\text{shared}} + k \cdot N_{\text{expert}}}_{\text{FLOPs/token, sets cost}}.
$$

A sketch of the fine-grained + shared-expert MoE FFN that would replace Stack-100M's SwiGLU block — the full treatment (load-balancing loss, capacity factor, expert/all-to-all parallelism) is in [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html):

```python
# stacklm/moe.py  -- DeepSeekMoE-style FFN: many fine-grained + few shared experts
import torch, torch.nn as nn, torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """One small SwiGLU expert (same activation as the dense Stack-100M MLP)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekMoEFFN(nn.Module):
    """Fine-grained routed experts + always-on shared experts (Dai et al., 2024).

    - `n_routed` many small experts; top-`k` are activated per token.
    - `n_shared` experts run for EVERY token (capture common structure).
    - `router_bias` is DeepSeek-V3's auxiliary-loss-FREE load balancer: a
      per-expert bias added to the routing logits for SELECTION only (never for
      the gate value), nudged up for under-loaded experts and down for
      over-loaded ones between steps. It balances load without an auxiliary
      loss term fighting the language-modeling objective.

    Active params/token = shared + k*routed, far below the total capacity of
    (n_shared + n_routed) experts. Here: active 3 of 17 experts resident.
    """
    def __init__(self, d_model=512, d_ff=352, n_routed=16, n_shared=1, k=2):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d_model, n_routed, bias=False)   # token -> expert affinities
        self.register_buffer("router_bias", torch.zeros(n_routed))  # updated by the balancer
        self.routed = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(n_routed)])
        self.shared = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x):                        # x: (B, T, d_model)
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        out = torch.zeros_like(flat)
        for e in self.shared:                    # shared experts: always on
            out = out + e(flat)

        scores = self.router(flat)                       # (B*T, n_routed)
        _, topi = (scores + self.router_bias).topk(self.k, dim=-1)   # SELECT with bias
        gates = F.softmax(scores.gather(-1, topi), dim=-1)           # WEIGHT without it
        for slot in range(self.k):               # accumulate the k routed experts
            idx = topi[:, slot]                  # which expert each token chose
            g = gates[:, slot:slot + 1]
            for e_id, expert in enumerate(self.routed):
                mask = idx == e_id
                if mask.any():                   # teaching sketch; see the note below
                    out = out.index_put((mask.nonzero(as_tuple=True)[0],),
                                        g[mask] * expert(flat[mask]), accumulate=True)
        return out.reshape(B, T, D)
```

With those numbers, active compute per token is `n_shared + k = 1 + 2 = 3` experts (`3 × 352 ≈ 1056` FFN width, *below* the dense `1408`), while total resident capacity is `n_shared + n_routed = 17` experts — roughly 6× the capacity per active expert for less-than-dense compute. That is the whole pitch of MoE in one line, and Exercise 3 does the parameter accounting exactly.

!!! warning "That masked loop is a teaching sketch, not a kernel"

    The `for e_id, expert in enumerate(self.routed)` loop launches `n_routed × k` tiny GEMMs per layer per step and materializes a boolean mask for each — fine for reading, catastrophic for throughput, and it gets worse as you make the experts finer-grained. Production MoE uses a **grouped GEMM**: sort tokens by expert, then run one batched matmul over variable-sized groups. **MegaBlocks** (Gale et al., 2022) formulates this as block-sparse matmuls and is *dropless* — no capacity factor, no tokens silently discarded when an expert is oversubscribed — and is integrated into Megatron-Core and used by several open MoE releases. **DeepSpeed-MoE** and Microsoft **Tutel** are the other mature options, both providing the fused **all-to-all** expert-parallel dispatch/combine that this sketch elides entirely. Reach for one of them before your first real MoE run; write the sketch once, to know what they are doing.

The catch — and why MoE is a *fork*, not a free lunch: MoE trades compute for **memory and communication**. All experts must live in memory even though each token uses few, and multi-GPU MoE needs **all-to-all** communication to route tokens to the GPUs holding their experts (expert parallelism, in [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html); the serving-side consequences are in [Serving Mixture-of-Experts](../07-inference-serving/13-serving-moe.html)). At 1B on one GPU, dense is simpler and probably right. MoE is the lever you pull when you want 7B-worth of *capacity* at ~1–2B-worth of *inference cost* — the DeepSeek/Qwen3 playbook.

!!! interview "Interview Corner"

    **Q:** You trained a great 1B dense model. A product team wants "much smarter" but the inference-latency budget per token is fixed. Do you go to a larger dense model or to MoE, and what breaks in each case?

    **A:** A larger dense model raises *active* params, so FLOPs/token and thus latency rise — it violates the fixed latency budget. MoE raises *total* capacity (quality) while holding *active* params (and therefore per-token FLOPs and latency) roughly fixed, which is exactly what the constraint demands — this is the DeepSeekMoE/Qwen3-MoE trade. What breaks with MoE is *serving*, not FLOPs: every expert must be resident in memory, so device memory and cost-per-GPU jump, and multi-GPU deployment needs all-to-all routing (expert parallelism) that adds communication and complicates batching — vLLM and SGLang both support expert parallelism, but your tokens-per-second-per-dollar curve changes shape. There is also a training-side cost the question invites you to miss: load balancing. Without it a few experts absorb most tokens and you have paid for capacity you are not using; DeepSeek-V3's auxiliary-loss-free bias-based balancer is the current answer, and a dropless grouped-GEMM kernel (MegaBlocks) avoids the token-dropping that capacity factors otherwise force. So: MoE for more quality at fixed latency, but budget for more memory, a harder serving stack, and a router you have to babysit; dense if you are memory- or ops-constrained and can afford the latency. The honest answer names the trade explicitly rather than treating MoE as free capacity.

{{fig:capstone-moe-capacity-vs-compute}}

---

## 14.12.6 Landing the Plane

You have now built the whole stack once — small, real, and end to end. Not a tutorial that stops at "and the rest is left as an exercise," but every stage that a frontier lab runs, executed in miniature on hardware you can rent for an afternoon. The tokenizer is yours; the scaling law is one you fit *and then checked against the outcome*; the optimizer, the schedule, the alignment stack, the agent, the quantized deployment — you touched all of it, and you know where every dollar went and why.

The gap between Stack-100M and a frontier model is real and enormous, and this book has been honest about it at every step: a 100M model is a narrow instrument, over-training is the lever that makes it punch above its size, and the path to 1B is mostly *data engineering and parallelism discipline*, not new ideas. But the *shape* of the machine is the same at every scale. The person who has built it once at 100M and understands why each piece is there is far better equipped to reason about a 100B run than someone who has only read about one. That transfer — from a run you can hold in your head to systems you cannot — is the entire point of the capstone.

Go run it. Then over-train it. Then, when you are ready, scale it up.

!!! key "Key Takeaways"

    - **Close the loop on your own prediction.** The Ch. 14.5 ladder forecast ≈2.94 ±0.1 nats/token for the flagship, and the run landed in that band — but only because the extrapolation was 1.9× in $N$ and the recipe was byte-identical. Trust a fitted law to ~2–3× past your top rung; before a 1B run, re-run the ladder with a higher top rung rather than stretching this fit 20×.
    - **6ND is a floor, not the bill.** Attention adds $Lsd/N$ on top — **31% for Stack-100M at seq 2048, 124% at 8192** — so state which MFU convention you log (6ND, model-FLOPs, or HFU with recomputation). An MFU(6ND) of 0.45 here is ~0.59 of A100 peak; deep-and-thin buys quality per parameter and pays for it per FLOP.
    - **The "~USD 100 model" itemizes to ~35 GPU-hours at USD 1.80/hr plus ~USD 13 non-GPU and a ~25% re-run tax ≈ USD 92** — and **over-training is ~90% of the pretraining line by design**, bought back many times over in saved inference. On 2026 hardware the same recipe is ~USD 59 on an H100 *and 2.4× faster in wall-clock*: the newest chip is often the cheapest.
    - **A dollar figure without a USD/GPU-hr, an MFU, and the MFU's convention is not reproducible.** Compute the bill from *measured* throughput; re-price against today's market, never a two-year-old blog post.
    - **Reproducibility has six pillars** — RNG *state* (plus dataloader state via `StatefulDataLoader`), a hashed frozen config (Hydra/pydantic) logged to a tracker (W&B/MLflow), a content-addressed data manifest (DVC / HF fingerprints), an environment pin (uv.lock hash + container digest, fail on a dirty git tree), safetensors checkpoints that never unpickle, and a **release** (HF Hub + model card + licence provenance + a pinned `lm-evaluation-harness` commit). Validate with a hermetic 50-step CPU run that reproduces bit-for-bit.
    - **At 1B, data breaks first**: you need 200B–1T tokens, which means `datatrove` / NeMo Curator / dolma rather than a laptop dedup script; repetition past ~4 epochs does not substitute for volume.
    - **Activations decide memory, not weights.** Stack-100M is 1.28 GB of weights+optimizer (Muon's single momentum buffer makes it 12 B/param, not 16) against **34 GB of activations** — and 195 GB without FlashAttention. A 1B model fits one 80 GB card at micro-batch 8–16; 7B needs ~87 GB and so fits an H200/B200/MI300X but not an A100. When to shard is a hardware-generation question.
    - **Climb the parallelism ladder only as far as forced**: DDP → FSDP2 (`fully_shard`, not the legacy wrapper) with `torch.distributed.checkpoint` for resharding-safe resume → tensor/pipeline only at tens of billions. `torchtitan` is this exact recipe already wired up at 1B–70B; Megatron-Core, DeepSpeed, and nanotron are the alternatives.
    - **1B costs ~USD 5,300, and 43% of it is not GPU** (vs 13% at 100M) — the dedup pipeline plus the synthetic-data bill rival the pretraining run. That ratio, not the FLOP count, is the real lesson about scale.
    - **MoE (DeepSeekMoE fine-grained + shared experts, Qwen3-MoE)** decouples capacity from compute/token — the highest-leverage change beyond 1B — but trades FLOPs for memory, all-to-all communication, and a router you must load-balance (DeepSeek-V3's bias-based balancer; MegaBlocks for dropless grouped GEMMs). It is a serving decision, not a free lunch.

---

!!! sota "State of the Art & Resources (2026)"
    Cost-aware pretraining — over-training small models for cheap inference, then scaling the same recipe with data-constrained-aware mixes, μP-style hyperparameter transfer, and fine-grained MoE — is now the default playbook at every lab that ships a small production model. The 2024→2026 shift in the *systems* layer is just as important: PyTorch-native sharding (FSDP2 + DCP) and reference stacks like torchtitan have largely replaced hand-rolled distributed loops for the 1B–70B range, and `safetensors` plus pinned eval harnesses have become the minimum bar for a releasable artifact.

    **Foundational work**

    - [Kaplan et al., *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361) — the original power-law relationships between loss, params, data, and compute that the capstone's cost arithmetic builds on.
    - [Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022)](https://arxiv.org/abs/2203.15556) — the Chinchilla result showing most trained models are compute-inefficient, the baseline the capstone deliberately trains past.
    - [Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2019)](https://arxiv.org/abs/1910.02054) — the sharded-optimizer-state idea behind FSDP, the first rung of the 1B→beyond parallelism ladder.
    - [Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022)](https://arxiv.org/abs/2205.05198) — the per-layer activation-memory formula this chapter uses to show that activations, not weights, decide what fits.

    **Recent advances (2023–2026)**

    - [Muennighoff et al., *Scaling Data-Constrained Language Models* (2023)](https://arxiv.org/abs/2305.16264) — quantifies how far repeating data can substitute for volume, the data-wall math cited for the 1B scale-up.
    - [Sardana et al., *Beyond Chinchilla-Optimal* (2024)](https://arxiv.org/abs/2401.00448) — the inference-aware scaling law that formally justifies deliberately over-training a model you intend to serve.
    - [Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization* (2024)](https://arxiv.org/abs/2401.06066) — the fine-grained-plus-shared-expert MoE design the chapter's `DeepSeekMoEFFN` sketch is modeled on.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — the auxiliary-loss-free, bias-based expert load balancer used in the sketch, plus FP8 training at scale.
    - [Qwen Team, *Qwen3 Technical Report* (2025)](https://arxiv.org/abs/2505.09388) — a current production MoE family (dense and MoE variants) applying the same capacity/compute-decoupling trade at scale.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — reports the MuonClip / QK-clip stabilization that keeps Muon usable at trillion-parameter scale, the fix the chapter tells you not to drop.
    - [Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* (2022)](https://arxiv.org/abs/2211.15841) — dropless block-sparse grouped GEMMs, the kernel the chapter's masked MoE loop is a stand-in for.

    **Open-source & tools**

    - [pytorch/torchtitan](https://github.com/pytorch/torchtitan) — PyTorch's reference pretraining stack (FSDP2, TP/PP/CP, `torch.compile`, Float8, DCP). The literal answer to "run this recipe at 1B."
    - [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (and Megatron-Core) — the reference tensor/pipeline-parallel implementation; [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) for ZeRO stages, offload, and DeepSpeed-MoE; [huggingface/nanotron](https://github.com/huggingface/nanotron) for a compact readable 3D-parallel pretrainer.
    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — the sharded MinHash-LSH dedup and filtering pipeline that produced FineWeb / FineWeb-Edu; [NVIDIA/NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) and [allenai/dolma](https://github.com/allenai/dolma) are the alternatives.
    - [huggingface/safetensors](https://github.com/huggingface/safetensors) — the zero-copy, no-code-execution tensor format that should hold every checkpoint you publish.
    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — pin a commit and commit the `results.json`; [huggingface/lighteval](https://github.com/huggingface/lighteval) is the lighter alternative.
    - [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — the reference Muon-optimizer implementation, paired with AdamW for embeddings/norms exactly as in Stack-100M's hybrid optimizer.
    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — a minimal, widely-reproduced training loop; useful as a second reference implementation for the cost-accounting discipline this chapter builds.
    - Experiment/config plumbing: [wandb](https://github.com/wandb/wandb), [mlflow](https://github.com/mlflow/mlflow), [aim](https://github.com/aimhubio/aim); [facebookresearch/hydra](https://github.com/facebookresearch/hydra) + OmegaConf; [astral-sh/uv](https://github.com/astral-sh/uv) for lockfiles; [iterative/dvc](https://github.com/iterative/dvc) for data versioning.

    **Go deeper**

    - [EleutherAI, *Transformer Math 101*](https://blog.eleuther.ai/transformer-math/) — the canonical write-up of the 6ND FLOP rule, MFU, and per-parameter memory accounting used throughout this chapter's cost tables.
    - [HuggingFace, *The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — an end-to-end tour of 5D parallelism, activation memory, and throughput tuning, the natural next read after this chapter's scale-up section.

## 14.12.7 Further Reading: The Works Behind Part XIV

Every technique in Stack-100M traces to a real, load-bearing paper. This is the curated list of the works actually cited across the capstone — read these and you have read the sources of the modern small-model recipe. (Illustrative magnitudes throughout the capstone are "on the order of"; these citations are the verifiable ground truth.)

**Small-model architecture & the deep-thin recipe**

- Liu et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models* (2024) — the deep-and-thin insight at fixed parameter budget.
- Su et al. *RoFormer: Rotary Position Embedding* (RoPE, 2021); Kazemnejad et al. *The Impact of Positional Encoding on Length Generalization* (NoPE, 2023); HuggingFace *SmolLM3* (2025) — RoPE + NoPE-on-a-subset for length generalization.
- Ainslie et al. *GQA: Grouped-Query Attention* (2023); Shazeer *GLU Variants Improve Transformer* (SwiGLU, 2020); Zhang & Sennrich *Root Mean Square Layer Normalization* (RMSNorm, 2019); Press & Wolf *Using the Output Embedding to Improve Language Models* (tied embeddings, 2017).
- DeepSeek-AI *DeepSeek-V2* (MLA, 2024) and *DeepSeek-V3* (MTP, FP8 training, auxiliary-loss-free load balancing, 2024); Gloeckle et al. *Better & Faster Large Language Models via Multi-token Prediction* (2024) — the efficiency options.

**Data, scaling, and over-training**

- Penedo et al. *The FineWeb Datasets* (FineWeb / FineWeb-Edu, HuggingFace, 2024); HuggingFace *Cosmopedia* — the data recipe, and (via `datatrove`) the pipeline that produced it.
- Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla, 2022); Kaplan et al. *Scaling Laws for Neural Language Models* (2020) — the scaling-law foundation.
- Besiroglu et al. *Chinchilla Scaling: A Replication Attempt* (2024) — why the fitted constants are fragile and the extrapolated loss is the thing to trust, the basis for §14.12.2's error bars.
- Sardana et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* (2024) — the over-training economics that justify 200 tokens/param.
- Muennighoff et al. *Scaling Data-Constrained Language Models* (2023) — the data wall and epoch limits that bite at 1B.

**Optimizer, schedule, and stability**

- Jordan et al. *Muon: An Optimizer for Hidden Layers in Neural Networks* (2024) and Moonshot AI *Kimi K2* (MuonClip / QK-clip, 2025) — the Muon+AdamW hybrid and its stability fix.
- Hu et al. *MiniCPM* (WSD schedule, 2024); OLMo team *OLMo 2* (2024) — WSD and the mid-training phase.

**Systems, memory, and cost**

- Korthikanti et al. *Reducing Activation Recomputation in Large Transformer Models* (2022) — the activation-memory formula behind §14.12.5.
- Dao et al. *FlashAttention* (2022) and *FlashAttention-2* (2023) — why the $s^2$ activation term disappears and 8192-token mid-training fits at all.
- Chowdhery et al. *PaLM* (2022) — the MFU definition that counts attention FLOPs, the convention §14.12.3 reconciles against.

**Alignment, agents, and deployment**

- Rafailov et al. *Direct Preference Optimization* (DPO, 2023); Shao et al. *DeepSeekMath* (GRPO, 2024); Yao et al. *ReAct* (2022) — SFT/DPO/GRPO and the agent loop.
- Frantar et al. *GPTQ* (2022); Lin et al. *AWQ* (2023) — the quantization behind the int4 laptop deployment.
- Dai et al. *DeepSeekMoE* (2024), Gale et al. *MegaBlocks* (2022), and Qwen Team *Qwen3 Technical Report* (2025) — the MoE path beyond 1B and the kernels that make it fast.

**Distributed training (the scale-up ladder)**

- Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2019); Shoeybi et al. *Megatron-LM* (2019); Narayanan et al. *Efficient Large-Scale LM Training on GPU Clusters* (2021) — the DP → FSDP → tensor/pipeline progression, and the achieved-vs-peak FLOPs reporting methodology this chapter's MFU section extends.

For the annotated, book-wide version of this list see [Key Papers: An Annotated Reading List](../99-appendix/03-papers-reading-list.html).

---

## Exercises

**1.** (Conceptual) The chapter calls training Stack-100M on ~20B tokens (~200 tokens/param) instead of the Chinchilla-optimal ~1.7B "economically irrational for a model you train and throw away, and completely rational for a model you will *serve*." Explain the asymmetry that makes both halves of that sentence true, and name where in the cost table the over-training decision shows up.

??? note "Solution"

    The asymmetry is between *when you pay* training compute and *when you pay* inference compute. Training compute is paid **once**, up front: the 6ND FLOPs to fit the weights. Inference compute is paid **every request, forever**: ~2ND FLOPs per generated token for as long as you serve the model.

    - If you train the model and then throw it away (a research probe, a one-off experiment), you get zero inference back, so any tokens beyond Chinchilla-optimal are pure waste — you spent ~12× the compute for a marginal loss improvement you will never amortize. Irrational.
    - If you will *serve* the model to many requests, over-training buys a permanently smaller/cheaper model at your target quality. A better 100M can replace a 200M that would have cost ~2× the FLOPs on *every* one of billions of future requests. You pay the extra training compute once and harvest the inference saving forever. Rational.

    In the cost table this decision is the **pretrain + mid-training lines**: 25.1 GPU-hours / ~USD 45.2, which is ~49% of the whole bill. A Chinchilla-optimal run (1.69B tokens, per the Ch. 14.5 fit) would take ~2.0 GPU-hours / ~USD 3.7, so roughly USD 41 of that USD 45 — about **91%** — is the over-training premium, spent deliberately to lower serving cost. This is the "inference-aware over-training" of Sardana et al. (2024). Note the §14.12.2 corroboration: at 20B tokens the data-penalty term $B/D^\beta \approx 0.19$ nats has fallen below the capacity term $A/N^\alpha \approx 0.30$ nats, i.e. the model is now capacity-limited — which is exactly the point at which further over-training stops paying and a *bigger* model starts to.

**2.** (Quantitative) The `cost.py` example assumes the stable phase sustained 231,500 tokens/sec, which the chapter says gives 21.6 GPU-hours at MFU(6ND) 45.1% over 18B tokens. Suppose instead your kernels are less well tuned and the loop sustains only **200,000 tokens/sec** on the same 18B-token run. Using the chapter's formulas and constants ($N = 1.014\times10^8$, A100 bf16 peak $= 3.12\times10^{14}$ FLOP/s, USD 1.80/GPU-hr), compute (a) the pretrain GPU-hours, (b) MFU under the 6ND convention, (c) the true fraction of A100 peak being used at `seq_len=2048`, and (d) the pretrain dollar cost. By what fraction does the slower run raise the pretrain bill?

??? note "Solution"

    (a) GPU-hours from measured throughput, `gpu_hours_from_throughput`:

    $$
    t = \frac{1.8\times10^{10}}{200{,}000 \times 3600} = \frac{90{,}000\ \text{s}}{3600} = 25.0\ \text{GPU-hours.}
    $$

    (b) MFU(6ND) is achieved / peak with achieved $= 6N \times$ tokens/sec:

    $$
    \text{MFU}_{6ND} = \frac{6 \times (1.014\times10^{8}) \times 200{,}000}{3.12\times10^{14}}
    = \frac{1.217\times10^{14}}{3.12\times10^{14}} \approx 0.390 = 39.0\%.
    $$

    (c) Add the attention FLOPs the 6ND rule omits. The correction factor is $1 + Lsd/N = 1 + (30 \times 2048 \times 512)/(1.014\times10^{8}) = 1.31$, so the true peak fraction is $0.390 \times 1.31 \approx 0.511$ — the GPU is at ~51% of its bf16 peak, not 39%. This is why the convention must be stated: the same run is "39% utilized" or "51% utilized" depending on which numerator you publish.

    (d) Dollars $= 25.0 \times \text{USD } 1.80 = \text{USD } 45.00$.

    Relative to the tuned run (21.6 GPU-hr, USD 38.88), the slower kernels raise the pretrain bill by $45.00/38.88 - 1 \approx 0.157$, i.e. **~16% more**. Note the ratio is exactly the throughput ratio $231{,}500/200{,}000 = 1.158$: GPU-hours and dollars scale inversely with tokens/sec, which is why the chapter insists a dollar figure is meaningless without a stated MFU *and* its convention.

**3.** (Quantitative) Take the `DeepSeekMoEFFN` config from the chapter: `d_model=512`, `d_ff=352`, `n_routed=16`, `n_shared=1`, `k=2`. Each `SwiGLUExpert` has three bias-free linear layers (`w_gate`, `w_up`: $d_{\text{model}}\times d_{\text{ff}}$ each; `w_down`: $d_{\text{ff}}\times d_{\text{model}}$). Compute (a) parameters per expert, (b) total resident expert parameters, (c) active expert parameters per token, and (d) compare (c) against the dense Stack-100M SwiGLU MLP (`intermediate=1408`). Does the MoE block use less compute per token than the dense block? Then (e): if you replaced all 30 blocks' MLPs this way, what happens to the *whole model's* parameter count, and what does that do to the memory analysis in §14.12.5?

??? note "Solution"

    (a) Per expert, three matmuls each of size $d_{\text{model}}\times d_{\text{ff}}$:

    $$
    3 \times 512 \times 352 = 540{,}672 \approx 0.54\text{M params.}
    $$

    (b) Total resident = all routed + shared experts must live in memory:

    $$
    (n_{\text{routed}} + n_{\text{shared}}) \times 540{,}672 = 17 \times 540{,}672 = 9{,}191{,}424 \approx 9.19\text{M params.}
    $$

    (c) Active per token = shared (always on) + top-$k$ routed:

    $$
    (n_{\text{shared}} + k) \times 540{,}672 = 3 \times 540{,}672 = 1{,}622{,}016 \approx 1.62\text{M params.}
    $$

    (d) Dense SwiGLU with intermediate 1408:

    $$
    3 \times 512 \times 1408 = 2{,}162{,}688 \approx 2.16\text{M params.}
    $$

    Yes: the active MoE compute (1.62M, effective FFN width $3\times352 = 1056$) is **below** the dense block (2.16M, width 1408) — about $1.62/2.16 \approx 0.75\times$ the compute per token — while resident capacity is $9.19/2.16 \approx 4.3\times$ larger (equivalently $17/3 \approx 5.7\times$ the experts). That is the MoE pitch: more capacity than dense at less-than-dense compute/token, paid for in memory.

    (e) Swapping all 30 MLPs: total params become $16.8\text{M (embed)} + 30 \times (0.655\text{M attn} + 9.19\text{M experts}) \approx 312\text{M}$, while *active* params per token fall to $16.8 + 30\times(0.655 + 1.62) \approx 85\text{M}$ — under the dense 101M. So you would have a **312M-parameter model with ~85M-parameter inference cost.** In §14.12.5's memory table, weights+optimizer state triples (from ~1.3 GB to ~3.8 GB at 12 B/param) while activations barely change, so on the flagship A100 the MoE variant still fits comfortably — the memory pressure at 100M is activations, and MoE does not touch those. The MoE memory problem only becomes the binding constraint at scales where weights already dominate, which is exactly the 7B+ regime where all-to-all expert parallelism becomes mandatory.

**4.** (Quantitative) The chapter claims 1B "fits a single 80 GB GPU, but the headroom is activations, not weights." Verify both halves with the chapter's two functions. (a) Using the Muon/AdamW blend (12 B/param for 2D hidden matrices, 16 B/param for embeddings and 1D params), compute weights+optimizer state for Stack-100M (84.5M Muon / 16.8M AdamW) and for a 1B model (0.9B Muon / 0.1B AdamW), and say how each compares to the naive 16 B/param figure. (b) Using Korthikanti's $s\,b\,h\,(34 + 5as/h)$ per layer with FlashAttention, compute activation memory for a 1B model ($L{=}24$, $h{=}2048$, $a{=}16$, $s{=}2048$) at micro-batch 8 and 16, and state the largest micro-batch that fits 80 GB. (c) Compute total pretraining FLOPs for 1B over 200B tokens, express it as a multiple of the capstone's $1.22\times10^{19}$, and estimate the single-GPU A100 wall-clock at MFU(6ND) 0.45.

??? note "Solution"

    **(a) Weights + optimizer.** Bytes/param: bf16 weights (2) + bf16 grads (2) + fp32 master (4) = 8 common, plus Muon's single fp32 momentum (4) → 12, or AdamW's fp32 $m,v$ (8) → 16.

    - Stack-100M: $84.5\times10^{6}\times12 + 16.8\times10^{6}\times16 = 1.014\times10^{9} + 2.69\times10^{8} = 1.28$ GB. The naive 16 B/param figure gives 1.62 GB — **27% too high**, which matters directly for the checkpoint-retention footgun.
    - 1B: $0.9\times10^{9}\times12 + 0.1\times10^{9}\times16 = 1.08\times10^{10} + 1.6\times10^{9} = 12.4$ GB, versus 16 GB naively.

    **(b) Activations.** With FlashAttention the $5as/h$ term vanishes, leaving $34\,s\,b\,h$ bytes per layer:

    $$
    \text{micro-batch }8:\quad 24 \times 34 \times 2048 \times 8 \times 2048 = 2.74\times10^{10}\ \text{B} = 27.4\ \text{GB}
    $$

    $$
    \text{micro-batch }16:\quad 54.8\ \text{GB}
    $$

    Adding the 12.4 GB of state: micro-batch 8 → 39.8 GB (comfortable), 16 → 67.2 GB (fits, but leaves little room for fragmentation, the `torch.compile` workspace, and eval activations), 32 → 122 GB (**OOM**). So the practical answer is **16, and 8 if you want to sleep**. Note that *without* FlashAttention the $5as/h = 5\times16\times2048/2048 = 80$ term more than triples the per-layer cost and micro-batch 8 alone needs ~92 GB — the fused kernel is not an optimization here, it is a prerequisite.

    **(c) FLOPs and time.** $C = 6 \times 10^{9} \times 2\times10^{11} = 1.2\times10^{21}$ FLOPs $= 1.2\times10^{21}/1.22\times10^{19} \approx 98\times$ the capstone campaign. At MFU(6ND) 0.45 on one A100:

    $$
    t = \frac{1.2\times10^{21}}{0.45 \times 3.12\times10^{14}} \approx 8.5\times10^{6}\ \text{s} \approx 2{,}370\ \text{GPU-hours} \approx 99\ \text{days.}
    $$

    So 1B does not break on memory — it breaks on *time*: ~2,400 GPU-hours on one card is unacceptable wall-clock, which is exactly why you go multi-GPU with DDP (near-linear speedup while the model still fits per GPU, which it does at 1B). Sanity check on the MFU assumption: at $d=2048$ the attention correction is only $1 + 24\times2048\times2048/1.2\times10^{9} = 1.08$, so MFU(6ND) 0.45 means ~49% of peak — plausible, and notably *easier* to reach than the same nominal MFU at Stack-100M's narrower shape.

**5.** (Implementation) Extend `stacklm/cost.py` with an inference-side accounting to make the "worked example" reproducible in code. Implement `serving_flops(n_params, n_requests, tokens_per_request=256)` using the $2ND$ inference rule, verify it reproduces the chapter's $\approx 5.2\times10^{19}$ FLOPs for a billion 256-token requests, and add `breakeven_requests(...)` returning how many 256-token requests it takes for cumulative serving FLOPs to equal the 20B-token pretraining budget. Report that number and explain why it does not depend on model size.

??? note "Solution"

    ```python
    # stacklm/cost.py  (append)
    def serving_flops(n_params: int, n_requests: int,
                      tokens_per_request: int = 256) -> float:
        """Inference compute via the 2ND rule: 2 FLOPs/param/generated token.

        Decode-only approximation: it ignores prefill (which is prompt-length
        dependent) and the attention-over-KV term, both of which push the real
        number UP -- so this is a lower bound on serving cost, which makes the
        over-training argument conservative.
        """
        return 2.0 * n_params * n_requests * tokens_per_request


    def breakeven_requests(n_params: int = N_PARAMS,
                           pretrain_tokens: float = 2.0e10,
                           tokens_per_request: int = 256) -> float:
        """How many requests until cumulative serving FLOPs == pretraining FLOPs."""
        pretrain = training_flops(n_params, pretrain_tokens)        # 6ND
        per_request = 2.0 * n_params * tokens_per_request           # 2ND, one request
        return pretrain / per_request


    if __name__ == "__main__":
        c = serving_flops(N_PARAMS, 1_000_000_000, 256)
        print(f"serve 1e9 reqs x256 tok: {c:.2e} FLOPs")   # -> 5.19e+19
        print(f"breakeven: {breakeven_requests():,.0f} requests")
    ```

    Check on part 1: $2 \times (1.014\times10^{8}) \times 10^{9} \times 256 = 5.19\times10^{19}$ FLOPs — matches the chapter's $\approx 5.2\times10^{19}$ (it is ~4.3× the entire pretraining campaign).

    Break-even: the $N$ cancels, so it is independent of model size:

    $$
    n = \frac{6ND}{2N \cdot 256} = \frac{6 \times 2\times10^{10}}{2 \times 256} = \frac{1.2\times10^{11}}{512} \approx 2.34\times10^{8}.
    $$

    **~234 million requests.** The cancellation is the point: the break-even depends only on the *tokens-per-parameter ratio you trained at* and the tokens per request — $n = 3D/(2\cdot\text{tok per request})$ — not on how big the model is. Train at 200 tokens/param and you break even after a fixed number of requests regardless of scale. After roughly a quarter-billion 256-token requests, cumulative serving compute equals the *entire* 20B-token pretraining budget — which is exactly why serving economics, not training economics, dominate the decision to over-train.

**6.** (Implementation) Footgun #4 in the "Checkpoint hygiene" admonition warns that ~1.28 GB of state written every 500 steps "fills a disk overnight," and prescribes keeping "a rolling window of the last *k* plus milestone checkpoints ... and delete the rest." Implement `prune_checkpoints(ckpt_dir, keep_last=3, milestones=())` consistent with `stacklm/repro.py`'s **directory-per-checkpoint** layout: given a directory containing sub-directories named `ckpt_step{N}`, delete every checkpoint except the `keep_last` highest steps and any step in `milestones`; return the list of deleted names. Be careful about the `.tmp` sibling an in-flight atomic write leaves behind.

??? note "Solution"

    ```python
    # stacklm/repro.py  (part 4: retention policy)
    import re, shutil
    from pathlib import Path

    _CKPT_RE = re.compile(r"^ckpt_step(\d+)$")     # anchored: excludes ckpt_step900.tmp


    def prune_checkpoints(ckpt_dir: str, keep_last: int = 3,
                          milestones=()) -> list[str]:
        """Enforce checkpoint retention: keep the `keep_last` newest checkpoints
        plus any `milestones` steps (e.g. end of stable phase, end of decay);
        delete the rest. Returns the names removed.

        Prevents the 'fills a disk overnight' footgun: ~1.28 GB/checkpoint
        (84.5M x 12 B Muon + 16.8M x 16 B AdamW) every 500 steps is unsustainable.
        """
        root = Path(ckpt_dir)
        milestones = set(milestones)
        found = []                                   # (step, name)
        for entry in root.iterdir():
            m = _CKPT_RE.match(entry.name)
            if m and entry.is_dir():                 # dirs only: skips stray files
                found.append((int(m.group(1)), entry.name))
        found.sort()                                 # ascending by step

        newest = {step for step, _ in found[-keep_last:]} if keep_last > 0 else set()
        keep = newest | milestones

        removed = []
        for step, name in found:
            if step not in keep:
                shutil.rmtree(root / name)           # rmtree, not unlink: it's a dir
                removed.append(name)
        return removed
    ```

    Walking through it: we discover checkpoints by the anchored `ckpt_step{N}` pattern, which by construction *excludes* the `ckpt_step{N}.tmp` sibling an in-flight `save_checkpoint` may be writing — deleting that mid-write would be a spectacular own-goal. We sort by *step number* rather than filesystem order or lexicographic name (so `ckpt_step900` is correctly older than `ckpt_step1000`), and form the keep-set as the union of the `keep_last` newest steps and the explicit `milestones`. Everything else is `rmtree`d and its name returned for logging.

    Example: with checkpoints at steps `{500, 1000, 1500, 2000, 2500}`, `keep_last=2`, `milestones={500}` keeps `{2500, 2000}` (newest two) plus `{500}` (a milestone), and returns `["ckpt_step1000", "ckpt_step1500"]`. Guarding `keep_last > 0` avoids `found[-0:]` accidentally selecting the whole list when a caller passes `keep_last=0` to retain milestones only. In a sharded (DCP) run the same policy applies to `checkpoint_id` directories — with the extra rule that pruning must run on rank 0 only, after a barrier, or ranks will race each other into a half-deleted directory.

**7.** (Quantitative) Your cloud offers A100-80GB at USD 1.30/GPU-hr and H100-SXM at USD 2.50/GPU-hr. The 18B-token stable phase requires $1.43\times10^{19}$ hardware FLOPs (6ND plus the 31% attention correction). Assume you sustain $u = 0.59$ of peak on the A100 (312 TFLOP/s dense bf16) but only $u = 0.45$ on the H100 (989 TFLOP/s dense bf16), because the smaller model leaves the bigger chip more launch-bound. Compute (a) GPU-hours and cost on each, (b) the speedup, and (c) the H100 rental price at which the two become equally expensive. What does this say about how to choose a tier?

??? note "Solution"

    (a) $t = C_{\text{hw}} / (u \cdot F_{\text{peak}})$:

    $$
    t_{\text{A100}} = \frac{1.43\times10^{19}}{0.59 \times 3.12\times10^{14}} \approx 7.77\times10^{4}\,\text{s} \approx 21.6\ \text{GPU-hr}
    \;\Rightarrow\; 21.6 \times 1.30 \approx \text{USD } 28.1
    $$

    $$
    t_{\text{H100}} = \frac{1.43\times10^{19}}{0.45 \times 9.89\times10^{14}} \approx 3.21\times10^{4}\,\text{s} \approx 8.9\ \text{GPU-hr}
    \;\Rightarrow\; 8.9 \times 2.50 \approx \text{USD } 22.3
    $$

    (b) Speedup $= 21.6/8.9 \approx 2.4\times$ — note this is *well below* the 3.17× peak ratio, precisely because the assumed utilization dropped from 0.59 to 0.45. Small models do not scale onto big chips for free.

    (c) Break-even: $p^{\star} = \text{USD }28.1 / 8.9 \approx \text{USD } 3.15$/GPU-hr. Any H100 price below ~USD 3.15/hr makes the H100 both cheaper *and* 2.4× faster.

    The lesson is that **the right tier is decided by (peak × achievable utilization) ÷ price, not by price alone**, and that wall-clock has option value the dollar figure does not capture: a 9-hour run can be launched, inspected, and relaunched the same day, while a 22-hour run costs you a day per iteration. The counter-pressure is that utilization *falls* as the chip grows relative to your model — which is why this arithmetic must be redone (with a measured $u$ from a 200-step probe run, not an assumed one) each time you change tiers.

**8.** (Conceptual + quantitative) In §14.12.2 the Ch. 14.5 ladder fit ($E{=}2.45$, $A{=}124$, $\alpha{=}0.33$, $B{=}234$, $\beta{=}0.30$) predicts $L \approx 2.94$ nats/token for the flagship. Suppose your run instead lands at **3.35** nats/token on the held-out pretrain-mix shard. (a) Is that inside the fit's honest band? (b) Decompose the predicted loss into its three terms and use the decomposition to argue which failure hypotheses the discrepancy is and is not consistent with. (c) Name three checks you would run, in order, before concluding the scaling law was wrong.

??? note "Solution"

    (a) No. The ladder's honest resolution is about ±0.1 nats (Ch. 14.5), so 2.84–3.04 is "as predicted." A measured 3.35 is **0.41 nats high — four times the band.** Ch. 14.5 states the diagnostic threshold explicitly: a broken run misses by 0.3+ nats, and that is what you are looking at. Treat it as a bug report, not as new science.

    (b) Decomposition at $N = 8.45\times10^{7}$, $D = 2\times10^{10}$:

    $$
    E = 2.45,\qquad A/N^{\alpha} = 124 \times (8.45\times10^{7})^{-0.33} \approx 0.30,\qquad
    B/D^{\beta} = 234 \times (2\times10^{10})^{-0.30} \approx 0.19.
    $$

    A 0.41-nat gap is larger than the *entire* capacity term and more than double the entire data term. That rules out the gentle hypotheses immediately: you cannot explain it by "the model is slightly too small" (driving $A/N^\alpha$ to zero — an infinitely large model — buys only 0.30) or by "we should have trained longer" (driving $B/D^{\beta}$ to zero buys only 0.19). Even both together do not reach 0.41. Since $E$ is by construction the irreducible entropy of *the mixture under this tokenizer*, a gap this large is only consistent with **the run or the measurement being different from what the ladder fit**: a different effective data distribution, a different tokenizer, a broken optimizer, or an evaluation against the wrong text.

    (c) In order of cost-to-check:

    1. **Are you measuring the same thing?** Confirm the eval shard is from the *pretrain* mix (not the annealed mid-training mix, and not a different domain), that it is tokenized with the same 32,768-vocab tokenizer, that the loss is per-*token* natural log (not per-byte, not base-2 — a factor of $\ln 2$ turns 2.32 into 3.35 and this mistake is embarrassingly common), and that padding/BOS positions are excluded from the mean. This costs minutes and explains the majority of such gaps.
    2. **Is the run the run you think it is?** Compare the checkpoint's `config_hash` and `corpus_hash` against the ladder's. A silently different mix (a shard that failed to download, so the 70/15/10/5 weights drifted), a different `seq_len`, or a different LR schedule all change $E$, $A$, $B$ — and the provenance record in §14.12.4 exists precisely so this check takes one command.
    3. **Did training actually converge?** Plot the loss curve: an un-decayed WSD stable-phase checkpoint legitimately sits well above its final value (Ch. 14.8's whole premise), and a mid-run loss spike that never fully recovered (Ch. 14.11 / Ch. 3.11) leaves a persistent offset. Also re-check the ladder's own held-out-rung test: fit on S1–S3 and predict S4. If *that* still holds, the law is fine and the flagship run is not.

    Only if all three pass should you suspect the law — and then the correct conclusion is not "scaling laws don't work" but "my ladder did not span the regime I extrapolated into," which is fixed by adding rungs, not by abandoning the method.
