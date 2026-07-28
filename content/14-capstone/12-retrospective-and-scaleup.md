# 14.12 Retrospective: Cost Accounting, Reproducibility, and the Path to 1B

Eleven chapters ago we set out to do something specific: take *the entire* LLM stack — every idea in this book — and compress it into one project small enough to run on a single rented GPU for about the price of a nice dinner, yet real enough that nothing was faked. We trained a byte-level BPE tokenizer, fit our *own* scaling law on a ladder of tiny models, pretrained **Stack-100M** on ~20B tokens with a Muon+AdamW hybrid under a WSD schedule, mid-trained it for long context and quality, aligned it with SFT / DPO / GRPO, distilled a narrow tool-using ReAct agent into it, evaluated it honestly, and quantized it to int4 to run on a laptop. The plane is in the air. This chapter lands it.

Landing means three concrete things, and this chapter is those three things: (1) a **full cost accounting** — every GPU-hour and dollar, stage by stage, adding up to the "~\$100" headline, so you know exactly where the money went; (2) a **reproducibility discipline** — the seeds, config hashes, data manifests, environment pins, and checkpoint hygiene that separate "I trained a model once" from "anyone can rebuild this bit-for-bit"; and (3) an honest **scale-up analysis** — what actually *breaks* when you take this exact recipe from 100M to 1B parameters, and what you change (data, parallelism, learning-rate and batch scaling, and the Mixture-of-Experts option) to get there.

This chapter builds directly on [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for the FLOP arithmetic, [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) for the parallelism transitions, and [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html) for the reproducibility machinery. It stays strictly consistent with the canonical `Stack-100M` config (`vocab_size=32768`, `d_model=512`, `n_layers=30`, GQA 8:2, SwiGLU `intermediate=1408`, tied embeddings, ≈101M params) fixed in the capstone plan.

---

## 14.12.1 Where We Landed

Before we count the money, let us name the artifact precisely, because the cost only means something relative to what it bought.

At the end of the pipeline the reader owns four checkpoints, all derived from one 101M-parameter base:

```text
stacklm-100m/
├── base/           # 20B-token pretrained base (WSD stable phase)   ~2.8–3.2 nats/tok train loss
├── mid/            # long-context (8192) + quality-annealed decay phase
├── chat/           # SFT + DPO on the ChatML-style template
├── agent/          # distilled ReAct traces → narrow auto-research tool-user
└── agent-int4/     # round-to-nearest int4 weights, ~55–65 MB on disk, runs on a CPU laptop
```

None of these is a frontier model, and the capstone never pretended otherwise. A 100M model is a *narrow instrument*: within a scaffolded, retrieval-grounded domain it can produce coherent, useful, grounded text; outside that scaffold it hallucinates freely. What is remarkable is not the model's raw quality but that **every layer of the stack that produces a GPT-4-class system is present here in miniature and actually ran** — the tokenizer, the scaling-law fit, the optimizer, the distributed-ready loop, the alignment stack, the agent harness, the quantized deployment. You now have a mental model of the whole machine that is *load-bearing*, not decorative, because you built each part.

The single most important lesson the capstone teaches — the one that makes a 100M model in 2026 vastly better than GPT-2 (117M) in 2019 at the same size — is **deliberate over-training**. Chinchilla-optimal for 100M is ~2B tokens; we trained on ~20B, roughly 200 tokens/param and ~10× past compute-optimal. That is economically irrational for a model you train and throw away, and completely rational for a model you will *serve*, because you pay training compute once and save inference cost forever. Keep that asymmetry in mind — it reappears in the cost table (over-training is most of the bill) and in the scale-up section (it only gets more extreme at 1B).

---

## 14.12.2 Cost Accounting: Anatomy of the ~\$100 Model

The headline is "the ~\$100 model," but a headline is not an accounting. Let us derive the number from first principles and then reconcile it against a realistic itemized bill, including the parts nobody advertises: the teacher API calls, the object storage, and the *re-run reality tax* of failed launches and OOM debugging.

### The compute floor: 6ND and MFU

Training FLOPs for a dense transformer follow the **6ND** rule (2 FLOPs/param/token forward, 4 backward), covered in depth in [Scaling Laws](../03-pretraining/04-scaling-laws.html). For the flagship pretraining run:

$$
C = 6 \, N \, D = 6 \times (1.01\times 10^{8}) \times (2.0\times 10^{10}) \approx 1.21 \times 10^{19} \ \text{FLOPs.}
$$

An NVIDIA A100 (80GB) has a bf16 tensor-core peak of ~312 TFLOP/s. You never get peak; you get **Model FLOPs Utilization (MFU)** — the fraction of peak your loop actually sustains after memory movement, kernel-launch overhead, and non-matmul work. Deep-and-thin models like Stack-100M (30 narrow layers) sit at the *lower* end of the usual 0.4–0.6 band that wide-and-shallow models enjoy, because each layer's matmuls are smaller relative to the fixed per-layer launch/normalization overhead. A realistic sustained MFU here is on the order of **0.40–0.50**.

$$
t_{\text{pretrain}} = \frac{C}{\text{MFU} \times F_{\text{peak}}} = \frac{1.21\times 10^{19}}{0.45 \times 3.12\times 10^{14}} \approx 8.6\times 10^{4}\ \text{s} \approx 23.9\ \text{GPU-hours.}
$$

At a well-tuned MFU of 0.50 that drops to ~21.7 hours; at a pessimistic 0.40 it climbs to ~27.1. So the pretraining wall-clock lands around **22–27 GPU-hours**. The bill below uses **22** — the optimistic end, a well-optimized run with fused kernels and `torch.compile` — which is consistent with the plan's 15–25 GPU-hour sticker, and we let the re-run tax absorb the slippage between a clean run and a real one. The following helper turns any measured throughput into a cost — feed it the tokens/sec your loop actually logs, not the theoretical peak.

```python
# stacklm/cost.py
"""Turn a training run's measured throughput into GPU-hours and dollars.

Everything here is derived from numbers the training loop already logs
(tokens/sec, total tokens) — no theoretical peaks required for the bill.
"""
from dataclasses import dataclass

# Stack-100M canonical constants (from the capstone plan; keep in sync with stacklm.config)
N_PARAMS = 101_400_000          # ~101.4M total params (tied embedding counted once)
A100_BF16_PEAK_FLOPS = 312e12   # A100-80GB bf16 tensor-core peak


def training_flops(n_params: int, n_tokens: int) -> float:
    """Dense-transformer training FLOPs via the 6ND rule."""
    return 6.0 * n_params * n_tokens


def gpu_hours_from_throughput(n_tokens: int, tokens_per_sec: float) -> float:
    """The honest number: wall-clock GPU-hours from *measured* throughput."""
    return n_tokens / tokens_per_sec / 3600.0


def mfu(tokens_per_sec: float, n_params: int = N_PARAMS,
        peak_flops: float = A100_BF16_PEAK_FLOPS) -> float:
    """Model FLOPs Utilization: sustained / peak. 6N FLOPs per token."""
    achieved = 6.0 * n_params * tokens_per_sec
    return achieved / peak_flops


@dataclass
class Stage:
    name: str
    gpu_hours: float
    usd_per_gpu_hour: float = 1.80   # illustrative A100-80GB spot price (Lambda/RunPod-class)
    extra_usd: float = 0.0           # non-GPU line items: teacher API, storage, egress

    @property
    def usd(self) -> float:
        return self.gpu_hours * self.usd_per_gpu_hour + self.extra_usd


if __name__ == "__main__":
    # Example: our pretrain logged ~252k tokens/sec sustained on 20B tokens.
    tps = 252_000
    hrs = gpu_hours_from_throughput(20_000_000_000, tps)
    print(f"pretrain: {hrs:.1f} GPU-hr, MFU={mfu(tps):.2%}")
    # -> pretrain: 22.0 GPU-hr, MFU=49.14%
```

### The itemized bill

Every stage of the capstone, priced at an illustrative **\$1.80/GPU-hour** A100-80GB spot rate. GPU-hours are the *sustained* wall-clock the loop would log, not the theoretical floor. Non-GPU line items (teacher-model API for agent distillation, object storage for the ~200 GB of tokenized shards, egress) are called out explicitly.

| Stage (chapter) | GPU-hr | GPU \$ | Non-GPU \$ | Stage \$ |
|---|---:|---:|---:|---:|
| Tokenizer BPE training (14.3) — mostly CPU | 0.3 | 0.54 | — | 0.54 |
| Scaling-law ladder {4M,9M,19M,43M} sweep (14.5) | 4.5 | 8.10 | — | 8.10 |
| **Pretrain, 20B tokens, WSD stable (14.7)** | 22.0 | 39.60 | — | 39.60 |
| Mid-training: 8192 ctx + quality anneal, ~3B tok (14.8) | 4.5 | 8.10 | — | 8.10 |
| SFT on chat template (14.9) | 1.0 | 1.80 | — | 1.80 |
| DPO preference optimization (14.9) | 1.2 | 2.16 | — | 2.16 |
| GRPO / narrow RLVR on arithmetic (14.9) | 3.0 | 5.40 | — | 5.40 |
| Agent distillation: teacher traces + SFT (14.10) | 1.0 | 1.80 | 8.00 | 9.80 |
| Eval + int8/int4 quantization + export (14.11) | 1.2 | 2.16 | — | 2.16 |
| Object storage + egress (~200 GB, one month) | — | — | 5.00 | 5.00 |
| **Subtotal** | **38.7** | **69.66** | **13.00** | **82.66** |
| Re-run reality tax (~25% of GPU \$: OOMs, bad launches, 2 restarts) | — | 17.40 | — | 17.40 |
| **Grand total** | — | — | — | **≈ \$100** |

Three things this table teaches that a bare "\$100" hides:

1. **Pretraining is ~40% of the bill and over-training is most of *that*.** Chinchilla-optimal (2B tokens) would cost roughly one-tenth of the pretrain line — ~\$4 instead of ~\$40. We deliberately spent the other ~\$36 to buy a permanently cheaper-to-serve model. That is the deployment-economics trade made *visible*.
2. **Alignment + agent is cheap; the teacher API is the surprise.** All of SFT+DPO+GRPO+distill is ~\$19, and \$8 of that is *not* your GPU at all — it is API calls to a large teacher to generate ReAct trajectories you then filter and distill. At 1B and beyond, teacher/data-generation cost often *exceeds* your own training cost.
3. **The reality tax is real.** No first run of a 30-layer model at high LR survives cleanly. You will hit an OOM from a mis-set gradient-accumulation count, a loss spike from an un-clipped attention logit, a corrupted shard. Budgeting ~25% for re-runs is not pessimism; it is the difference between the sticker and the invoice.

!!! note "Why the bill is a band, not a point"

    The plan quotes **\$40–\$100**, and both ends are honest. The low end assumes an owned or deeply-discounted GPU and a clean first run: ~22 GPU-hours near the ~\$1/GPU-hr floor of spot A100 pricing is ~\$22 of compute plus a few dollars of API and storage. The high end is the fully-loaded invoice above: on-demand-class spot at ~\$1.80/GPU-hr, the teacher API, storage, and a realistic re-run tax. Same recipe, same tokens — the roughly 2.5× spread is *entirely* GPU market price and how many times you fat-finger a launch. Quote the number with its assumptions attached: a dollar figure without a price-per-GPU-hour *and* an MFU is not reproducible, because someone else's \$/GPU-hr and someone else's kernels move it by 2×.

{{fig:capstone-cost-anatomy}}

!!! example "Worked example: does over-training actually pay off?"

    Suppose you will serve Stack-100M for one billion inference requests, each generating 256 tokens. Inference FLOPs are $\approx 2ND$, so total serving compute is

    $$
    C_{\text{serve}} \approx 2 \times (1.01\times10^8) \times (10^{9}\times 256) \approx 5.2\times10^{19}\ \text{FLOPs.}
    $$

    That is ~4× the *entire* 20B-token pretraining budget ($1.21\times10^{19}$). Now imagine over-training let you hit your target quality at 100M instead of needing a 200M model (roughly 2× the serving FLOPs). The extra ~\$36 you spent over-training saves you ~$5.2\times10^{19}$ FLOPs — on the order of one full pretrain-budget's worth of compute — *per billion requests*. The over-training pays for itself many times over the moment you deploy at scale. This is exactly the "inference-aware over-training" of Sardana et al. (*Beyond Chinchilla-Optimal*, 2024), and it only sharpens as you scale.

{{fig:capstone-pay-once-save-forever}}

---

## 14.12.3 Reproducibility: Making the Run Rebuildable

A result you cannot reproduce is an anecdote, not an experiment. At 100M on one GPU, reproducibility is *achievable to the bit* if you are disciplined; at 1B across a cluster it becomes merely *approximate* (non-deterministic all-reduce ordering, hardware variance) — which is all the more reason to nail the discipline now while you can verify it. The checklist has five pillars: seeds, config hashes, a data manifest, environment pinning, and checkpoint hygiene. We give real code for each; it lives in `stacklm/repro.py` and is called from the training loop in `stacklm.train`.

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
    Save this inside the checkpoint; restore it before the next data batch."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
```

The subtle one is `PYTHONHASHSEED` and RNG *state* (not just the seed). If you resume a run and re-seed from the same integer, you replay the random stream *from the start* — reusing data batches you already consumed. You must save and restore the RNG *state* at the resume step, exactly as in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html).

### Config hashing

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
    return hashlib.sha256(blob).hexdigest()[:12]   # 12 hex chars is plenty to disambiguate
```

### The data manifest

The dirtiest source of "it doesn't reproduce" is data drift: a shard silently re-tokenized, a file truncated by a failed download, the mix reweighted. A **data manifest** pins the *content* of every shard by SHA-256, not its path, so you can prove the bytes are identical.

```python
def data_manifest(shard_paths: list[str]) -> dict:
    """Content-addressed manifest of the tokenized corpus.

    For each .bin memmap shard we record its SHA-256, byte size, and token count
    (uint16 tokens => 2 bytes each). The top-level `corpus_hash` fingerprints the
    WHOLE dataset: change any shard, or the order, and it changes.
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

### Environment pinning

Same code + same data + *different* PyTorch build can still diverge (a changed default in a fused kernel, a new cuDNN heuristic). Capture the exact environment as data, alongside the run.

```python
def environment_fingerprint() -> dict:
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
        "git_dirty": bool(_git("status", "--porcelain")),   # True => uncommitted changes: DANGER
        # For a hermetic rebuild also freeze deps: `pip freeze > requirements.lock`
        # or better, commit a uv.lock / poetry.lock and record its hash here.
    }
```

A `git_dirty=True` at training time is a red flag: it means the code that produced the checkpoint was never committed, so the run is not reproducible from any commit. Fail loudly on it for the flagship run.

### Checkpoint hygiene

Finally, weld provenance *into* the checkpoint so a lone `.pt` file is self-describing. A checkpoint should carry not just weights but the config hash, the corpus hash, the git commit, the RNG state, and the step — everything needed to prove where it came from and to resume it identically.

```python
# stacklm/repro.py  (part 3: self-describing checkpoints)
def build_provenance(config, shard_paths: list[str]) -> dict:
    """Assemble the full provenance record stamped into every checkpoint & log."""
    return {
        "config_hash": config_hash(config),
        "config": asdict(config) if is_dataclass(config) else dict(config),
        "data_manifest": data_manifest(shard_paths),
        "env": environment_fingerprint(),
    }


def save_checkpoint(path, model, optimizer, step, provenance):
    """Self-describing checkpoint: weights + opt state + RNG + provenance.
    Uses a temp file + atomic rename so a crash mid-write never corrupts the
    'latest' checkpoint (checkpoint hygiene rule #1)."""
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),   # BOTH Muon and AdamW group states
        "rng": rng_state_dict(),
        "provenance": provenance,
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)                       # atomic on POSIX


def load_checkpoint(path, model, optimizer, expect_config_hash=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    got = ckpt["provenance"]["config_hash"]
    if expect_config_hash is not None and got != expect_config_hash:
        raise RuntimeError(
            f"Config hash mismatch: checkpoint={got} expected={expect_config_hash}. "
            "You are resuming a run with a DIFFERENT architecture/optimizer config."
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    load_rng_state(ckpt["rng"])
    return ckpt["step"]
```

!!! warning "Checkpoint hygiene footguns"

    Three failures cost real GPU-hours. **(1) Non-atomic writes:** if the process dies mid-`torch.save`, the file is truncated and unloadable — always write to `.tmp` then `os.replace`. **(2) Dropping optimizer state:** saving only weights makes a resumed run silently re-warm Muon/Adam momentum from zero, spiking loss for hundreds of steps (see [Checkpointing](../03-pretraining/12-checkpointing-fault-tolerance.html)). **(3) Unbounded retention:** ~1.6 GB of model+optimizer state per checkpoint (16 bytes/param × 101M) written every 500 steps fills a disk overnight. Keep a rolling window of the last *k* plus milestone checkpoints (end of stable phase, end of decay), and delete the rest.

### The one-page checklist

Before you press go on a flagship run, this must all be true. It is the difference between science and a story.

```text
REPRODUCIBILITY PREFLIGHT  (stacklm)
[ ] seed_everything(1337, deterministic=True) called before model init
[ ] RNG *state* saved in checkpoint (not just the integer seed)
[ ] config frozen in one dataclass; config_hash printed to log line 1
[ ] data_manifest.json written; corpus_hash matches the intended mix
[ ] git commit clean (git_dirty == False); commit hash logged
[ ] requirements.lock / uv.lock committed; torch+cuda+cudnn versions logged
[ ] checkpoints atomic (tmp+rename), carry provenance, retention policy set
[ ] a 50-step CPU toy run reproduces bit-identically across two invocations
```

That last line is the cheap insurance: a hermetic 50-step CPU run (tiny vocab, tiny model, synthetic data — the same toy path the book's CI smoke-tests) that must produce *identical* loss across two invocations. If it does not, your determinism is broken and no amount of cluster time will fix it later.

---

## 14.12.4 What Breaks at 1B — and What to Change

The capstone's proudest claim is that the *entire* 100M pipeline fits on one GPU. The instructive question is: run the exact same code targeting 1B parameters — what breaks first? We take the failures in the order you hit them.

### Data: the wall you hit before the compute wall

A 1B model at Chinchilla-optimal wants ~20B tokens; over-trained for deployment (the capstone's whole philosophy) it wants **200B–1T tokens**. Our ~20B-token corpus is now 10–50× too small. You cannot fix this by looping the same data: repetition helps only up to ~4 epochs and its value collapses toward zero by ~16 epochs (Muennighoff et al., *Scaling Data-Constrained Language Models*, 2023). So the first thing that breaks at 1B is *data volume*, and the fix is more *and* cleaner data:

- **Volume:** move from a FineWeb-Edu *sample* to the full FineWeb / FineWeb-Edu (trillions of tokens), which forces the dedup and quality pipeline of [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) to actually scale — MinHash-LSH over trillions of documents, not a laptop-sized set.
- **Quality and mix:** the marginal token matters more at 1B; retune the mix (the 70/15/10/5 FineWeb-Edu/Cosmopedia/code/math split) using the domain-weighting methods of [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html), and lean harder on synthetic data (Cosmopedia-style) to buy quality where raw web runs thin.

The uncomfortable truth: at 1B, *data engineering* — not model code — is where most of your time and a growing share of your budget goes.

### Memory and parallelism: DP → FSDP → pipeline

Does 1B even fit on one 80 GB A100? Do the arithmetic, because it is closer than intuition suggests. With mixed-precision training (see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)) and an AdamW-style optimizer, per-parameter memory is: bf16 weights (2B) + bf16 grads (2B) + fp32 master weights (4B) + fp32 Adam $m,v$ (8B) = **16 bytes/param**.

```python
# stacklm/scaleup.py  -- when does the model stop fitting on one GPU?
def optimizer_state_gb(n_params: int, bytes_per_param: int = 16) -> float:
    """bf16 weights(2) + bf16 grads(2) + fp32 master(4) + fp32 Adam m,v(8) = 16 B/param."""
    return n_params * bytes_per_param / 1e9


for n in [1e8, 1e9, 7e9, 70e9]:
    print(f"{n/1e9:5.1f}B params -> {optimizer_state_gb(int(n)):7.1f} GB weights+opt state")
# 0.1B ->   1.6 GB   (Stack-100M: trivial)
# 1.0B ->  16.0 GB   (fits on 80GB A100 with room for activations)
# 7.0B -> 112.0 GB   (does NOT fit one GPU)
# 70.0B -> 1120.0 GB  (needs a whole cluster)
```

So **1B still fits a single 80 GB A100** for weights+optimizer (~16 GB), leaving room for activations at modest batch/seq. The recipe does not break on memory at 1B — it breaks on *time*: 1B over 200B tokens is $6 \times 10^{9} \times 2\times10^{11} = 1.2\times10^{21}$ FLOPs, ~100× the capstone pretrain, i.e. **thousands of GPU-hours** on one card. That is when you go multi-GPU, and the progression is exactly the one taught in Part III:

1. **DDP / data parallelism first** — the simplest scale-out. Replicate the model on each of $g$ GPUs, split the batch, all-reduce gradients. Near-linear speedup while the model *fits* per GPU (true at 1B). This is your first move: 8×A100 with DDP turns thousands of GPU-hours of wall-clock into hundreds.
2. **FSDP / ZeRO when it stops fitting (~7B+)** — shard parameters, gradients, and optimizer state across data-parallel ranks so each holds $1/g$ of the 16 bytes/param. FSDP is what lets a 7B–70B model train at all; see [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html). At 1B you *can* use FSDP purely to raise batch size / activation headroom, but you do not *need* it.
3. **Tensor + pipeline parallelism when even a layer is too big / too slow (tens of B+)** — split individual matmuls across GPUs (tensor) and stages of layers across GPUs (pipeline), the Megatron-style approach of [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html) and [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html). You do **not** reach for this at 1B — pipeline bubbles and tensor-parallel communication are pure overhead you only accept when nothing else works. Naming it here is the point: knowing you *don't* need it at 1B is as valuable as knowing you *will* at 30B.

The single-controller message: at 1B, **DDP is enough**; FSDP is a convenience; pipeline parallelism is premature. The complexity ladder is something you climb only as far as the model forces you.

{{fig:capstone-scaleup-ladder}}

### Learning rate and batch size: what to rescale

You cannot copy Stack-100M's hyperparameters to 1B unchanged; width and batch both grew. Two rules, both from [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html):

- **Learning rate vs. width.** Optimal LR shrinks as the model widens. The principled tool is **μP (Maximal Update Parametrization)**: tune LR on a small proxy, then transfer it to the large model with width-dependent scaling so the *same* tuned value stays optimal. A cheaper heuristic is $\eta \propto 1/\sqrt{d_{\text{model}}}$; going from $d{=}512$ to $d{=}2048$ (a typical 1B width) roughly *halves* the peak LR. The AdamW groups (embeddings, norms) and the Muon groups (2D hidden matrices) rescale differently — Muon's orthogonalized update is naturally more scale-robust, one reason the hybrid was chosen (see [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html)), but you still retune.
- **Batch size and warmup.** Larger models have a larger **critical batch size** — the point past which more parallelism stops helping per-step progress. At 1B you can raise the global batch from ~0.5M tokens toward ~1–2M tokens (more data-parallel ranks make this nearly free) and extend warmup proportionally. The WSD schedule transfers cleanly: keep the long stable phase, keep the short decay phase as your mid-training quality anneal — the schedule's shape is scale-invariant even though the numbers move.

Keep MuonClip / QK-clip on. Attention-logit blow-ups get *worse* at scale and high LR, and QK-clip is exactly the stability fix (Kimi K2, 2025) that made Muon usable at scale — do not drop it when you can least afford instability. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

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
    Active params/token = shared + k*routed, far below the total capacity of
    (n_shared + n_routed) experts. Here: active 3 of 17 experts resident.
    """
    def __init__(self, d_model=512, d_ff=352, n_routed=16, n_shared=1, k=2):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d_model, n_routed, bias=False)   # token -> expert affinities
        self.routed = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(n_routed)])
        self.shared = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(n_shared)])

    def forward(self, x):                        # x: (B, T, d_model)
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        out = sum(e(flat) for e in self.shared)  # shared experts: always on

        scores = self.router(flat)               # (B*T, n_routed)
        topv, topi = scores.topk(self.k, dim=-1) # pick k experts per token
        gates = F.softmax(topv, dim=-1)          # renormalize over the chosen k
        for slot in range(self.k):               # accumulate the k routed experts
            idx = topi[:, slot]                  # which expert each token chose
            g = gates[:, slot:slot + 1]
            for e_id, expert in enumerate(self.routed):
                mask = idx == e_id
                if mask.any():                   # (real systems use grouped all-to-all here)
                    out[mask] += g[mask] * expert(flat[mask])
        return out.reshape(B, T, D)
```

With those numbers, active compute per token is `n_shared + k = 1 + 2 = 3` experts (`3 × 352 ≈ 1056` FFN width, *below* the dense `1408`), while total resident capacity is `n_shared + n_routed = 17` experts — roughly 6× the capacity per active expert for less-than-dense compute. That is the whole pitch of MoE in one line.

The catch — and why MoE is a *fork*, not a free lunch: MoE trades compute for **memory and communication**. All experts must live in memory even though each token uses few, and multi-GPU MoE needs **all-to-all** communication to route tokens to the GPUs holding their experts (expert parallelism, in [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html)). At 1B on one GPU, dense is simpler and probably right. MoE is the lever you pull when you want 7B-worth of *capacity* at ~1–2B-worth of *inference cost* — the DeepSeek/Qwen3 playbook.

!!! interview "Interview Corner"

    **Q:** You trained a great 1B dense model. A product team wants "much smarter" but the inference-latency budget per token is fixed. Do you go to a larger dense model or to MoE, and what breaks in each case?

    **A:** A larger dense model raises *active* params, so FLOPs/token and thus latency rise — it violates the fixed latency budget. MoE raises *total* capacity (quality) while holding *active* params (and therefore per-token FLOPs and latency) roughly fixed, which is exactly what the constraint demands — this is the DeepSeekMoE/Qwen3-MoE trade. What breaks with MoE is *serving*, not FLOPs: every expert must be resident in memory, so device memory and cost-per-GPU jump, and multi-GPU deployment needs all-to-all routing (expert parallelism) that adds communication and complicates batching. So: MoE for more quality at fixed latency, but budget for more memory and a harder serving stack; dense if you are memory- or ops-constrained and can afford the latency. The honest answer names the trade explicitly rather than treating MoE as free capacity.

{{fig:capstone-moe-capacity-vs-compute}}

---

## 14.12.5 Landing the Plane

You have now built the whole stack once — small, real, and end to end. Not a tutorial that stops at "and the rest is left as an exercise," but every stage that a frontier lab runs, executed in miniature on hardware you can rent for an afternoon. The tokenizer is yours; the scaling law is one you fit; the optimizer, the schedule, the alignment stack, the agent, the quantized deployment — you touched all of it, and you know where every dollar went.

The gap between Stack-100M and a frontier model is real and enormous, and this book has been honest about it at every step: a 100M model is a narrow instrument, over-training is the lever that makes it punch above its size, and the path to 1B is mostly *data engineering and parallelism discipline*, not new ideas. But the *shape* of the machine is the same at every scale. The person who has built it once at 100M and understands why each piece is there is far better equipped to reason about a 100B run than someone who has only read about one. That transfer — from a run you can hold in your head to systems you cannot — is the entire point of the capstone.

Go run it. Then over-train it. Then, when you are ready, scale it up.

!!! key "Key Takeaways"

    - The "~\$100 model" itemizes to ~39 GPU-hours across all stages at ~\$1.80/GPU-hr, plus ~\$13 non-GPU (teacher API + storage) and a ~25% re-run reality tax — and **over-training is ~90% of the pretraining bill by design**, bought back many times over in saved inference.
    - Cost is FLOPs / (MFU × peak): the 6ND rule gives $\approx 1.2\times10^{19}$ pretraining FLOPs, and at a realistic deep-thin MFU of ~0.45 that is ~22–27 GPU-hours — compute the bill from *measured* throughput, not theoretical peak, and always quote \$/GPU-hr and MFU alongside the dollar figure.
    - Reproducibility has five pillars — seed *state* (not just the integer), config hash, content-addressed data manifest, environment pin (fail on a dirty git tree), and atomic self-describing checkpoints — validated by a hermetic 50-step CPU run that must reproduce bit-for-bit.
    - At 1B, **data breaks first**: you need 200B–1T tokens, forcing real dedup/quality/mix engineering; repetition past ~4 epochs does not substitute for volume.
    - 1B still *fits* one 80 GB A100 (~16 bytes/param ⇒ ~16 GB); it breaks on *time*, so scale DP → FSDP → (only much later) tensor/pipeline — knowing you do **not** need pipeline parallelism at 1B is as valuable as knowing when you will.
    - Rescale hyperparameters with model width: peak LR shrinks (μP or ~$1/\sqrt{d}$), critical batch size grows, warmup extends — but the WSD schedule shape and MuonClip stability fix transfer unchanged.
    - **MoE (DeepSeekMoE fine-grained + shared experts, Qwen3-MoE)** decouples capacity from compute/token — the highest-leverage change beyond 1B — but trades FLOPs for memory and all-to-all communication, so it is a serving decision, not a free lunch.
    - You have built the entire stack once, small and real; the machine has the same shape at every scale, and that is what makes the 100M run a genuine on-ramp to the 100B one.

---

## 14.12.6 Further Reading: The Works Behind Part XIV

Every technique in Stack-100M traces to a real, load-bearing paper. This is the curated list of the works actually cited across the capstone — read these and you have read the sources of the modern small-model recipe. (Illustrative magnitudes throughout the capstone are "on the order of"; these citations are the verifiable ground truth.)

**Small-model architecture & the deep-thin recipe**

- Liu et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models* (2024) — the deep-and-thin insight at fixed parameter budget.
- Su et al. *RoFormer: Rotary Position Embedding* (RoPE, 2021); Kazemnejad et al. *The Impact of Positional Encoding on Length Generalization* (NoPE, 2023); HuggingFace *SmolLM3* (2025) — RoPE + NoPE-on-a-subset for length generalization.
- Ainslie et al. *GQA: Grouped-Query Attention* (2023); Shazeer *GLU Variants Improve Transformer* (SwiGLU, 2020); Zhang & Sennrich *Root Mean Square Layer Normalization* (RMSNorm, 2019); Press & Wolf *Using the Output Embedding to Improve Language Models* (tied embeddings, 2017).
- DeepSeek-AI *DeepSeek-V2* (MLA, 2024) and *DeepSeek-V3* (MTP, 2024); Gloeckle et al. *Better & Faster Large Language Models via Multi-token Prediction* (2024) — the efficiency options.

**Data, scaling, and over-training**

- Penedo et al. *The FineWeb Datasets* (FineWeb / FineWeb-Edu, HuggingFace, 2024); HuggingFace *Cosmopedia* — the data recipe.
- Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla, 2022); Kaplan et al. *Scaling Laws for Neural Language Models* (2020) — the scaling-law foundation.
- Sardana et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* (2024) — the over-training economics that justify 200 tokens/param.
- Muennighoff et al. *Scaling Data-Constrained Language Models* (2023) — the data wall and epoch limits that bite at 1B.

**Optimizer, schedule, and stability**

- Jordan et al. *Muon: An Optimizer for Hidden Layers in Neural Networks* (2024) and Moonshot AI *Kimi K2* (MuonClip / QK-clip, 2025) — the Muon+AdamW hybrid and its stability fix.
- Hu et al. *MiniCPM* (WSD schedule, 2024); OLMo team *OLMo 2* (2024) — WSD and the mid-training phase.

**Alignment, agents, and deployment**

- Rafailov et al. *Direct Preference Optimization* (DPO, 2023); Shao et al. *DeepSeekMath* (GRPO, 2024); Yao et al. *ReAct* (2022) — SFT/DPO/GRPO and the agent loop.
- Frantar et al. *GPTQ* (2022); Lin et al. *AWQ* (2023) — the quantization behind the int4 laptop deployment.
- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization* (2024) and Qwen Team *Qwen3 Technical Report* (Qwen3-MoE, 2025) — the MoE path beyond 1B.

**Distributed training (the scale-up ladder)**

- Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2019); Shoeybi et al. *Megatron-LM* (2019) — the DP → FSDP → tensor/pipeline progression referenced above.

For the annotated, book-wide version of this list see [Key Papers: An Annotated Reading List](../99-appendix/03-papers-reading-list.html).

---

## Exercises

**1.** (Conceptual) The chapter calls training Stack-100M on ~20B tokens (~200 tokens/param) instead of the Chinchilla-optimal ~2B "economically irrational for a model you train and throw away, and completely rational for a model you will *serve*." Explain the asymmetry that makes both halves of that sentence true, and name where in the cost table the over-training decision shows up.

??? note "Solution"

    The asymmetry is between *when you pay* training compute and *when you pay* inference compute. Training compute is paid **once**, up front: the 6ND FLOPs to fit the weights. Inference compute is paid **every request, forever**: ~2ND FLOPs per generated token for as long as you serve the model.

    - If you train the model and then throw it away (a research probe, a one-off experiment), you get zero inference back, so any tokens beyond Chinchilla-optimal are pure waste — you spent ~10x the compute for a marginal loss improvement you will never amortize. Irrational.
    - If you will *serve* the model to many requests, over-training buys a permanently smaller/cheaper model at your target quality. A better 100M can replace a 200M that would have cost ~2x the FLOPs on *every* one of billions of future requests. You pay the extra training compute once and harvest the inference saving forever. Rational.

    In the cost table this decision is the **pretrain line**: ~22 GPU-hours / ~\$39.60, which is ~40% of the whole bill. Chinchilla-optimal (2B tokens) would be ~one-tenth of that (~\$4), so roughly \$36 of the \$40 pretrain line — about 90% of it — is the over-training premium, spent deliberately to lower serving cost. This is the "inference-aware over-training" of Sardana et al. (2024).

**2.** (Quantitative) The `cost.py` example assumes the pretrain loop sustained 252,000 tokens/sec, which the chapter says gives 22.0 GPU-hours at MFU 49.1%. Suppose instead your kernels are less well tuned and the loop sustains only **200,000 tokens/sec** on the same 20B-token run. Using the chapter's formulas and constants ($N = 1.014\times10^8$, A100 bf16 peak $= 3.12\times10^{14}$ FLOP/s, \$1.80/GPU-hr), compute (a) the pretrain GPU-hours, (b) the MFU, and (c) the pretrain dollar cost. By what fraction does the slower run raise the pretrain bill?

??? note "Solution"

    (a) GPU-hours from measured throughput, `gpu_hours_from_throughput`:

    $$
    t = \frac{2.0\times10^{10}}{200{,}000 \times 3600} = \frac{100{,}000\ \text{s}}{3600} \approx 27.8\ \text{GPU-hours.}
    $$

    (b) MFU is achieved / peak, with achieved $= 6N \times$ tokens/sec:

    $$
    \text{MFU} = \frac{6 \times (1.014\times10^{8}) \times 200{,}000}{3.12\times10^{14}}
    = \frac{1.217\times10^{14}}{3.12\times10^{14}} \approx 0.390 = 39.0\%.
    $$

    (c) Dollars $= 27.8 \times \$1.80 \approx \$50.0$.

    Relative to the tuned run (22.0 GPU-hr, \$39.6), the slower kernels raise the pretrain bill by $50.0/39.6 - 1 \approx 0.26$, i.e. **~26% more**. Note the ratio is exactly the throughput ratio $252{,}000/200{,}000 = 1.26$: GPU-hours and dollars scale inversely with tokens/sec, which is why the chapter insists a dollar figure is meaningless without a stated MFU.

**3.** (Quantitative) Take the `DeepSeekMoEFFN` config from the chapter: `d_model=512`, `d_ff=352`, `n_routed=16`, `n_shared=1`, `k=2`. Each `SwiGLUExpert` has three bias-free linear layers (`w_gate`, `w_up`: $d_{\text{model}}\times d_{\text{ff}}$ each; `w_down`: $d_{\text{ff}}\times d_{\text{model}}$). Compute (a) parameters per expert, (b) total resident expert parameters, (c) active expert parameters per token, and (d) compare (c) against the dense Stack-100M SwiGLU MLP (`intermediate=1408`). Does the MoE block use less compute per token than the dense block?

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

**4.** (Quantitative) The chapter claims 1B "still fits a single 80 GB A100 ... it breaks on *time*." Verify both halves. Using 16 bytes/param for weights+optimizer state, compute the weights+optimizer footprint of a 1B and a 7B model, and check each against 80 GB. Then compute the total pretraining FLOPs for 1B over 200B tokens, express it as a multiple of the capstone's $1.21\times10^{19}$, and estimate the single-GPU wall-clock at MFU 0.45.

??? note "Solution"

    Footprint via `optimizer_state_gb` ($16$ bytes/param):

    - 1B: $10^{9} \times 16 = 1.6\times10^{10}$ B $= 16$ GB — well under 80 GB, leaving room for activations. **Fits.**
    - 7B: $7\times10^{9} \times 16 = 1.12\times10^{11}$ B $= 112$ GB $> 80$ GB. **Does not fit** one GPU (hence FSDP/ZeRO at ~7B+).

    Pretraining FLOPs for 1B over 200B tokens (6ND):

    $$
    C = 6 \times 10^{9} \times 2\times10^{11} = 1.2\times10^{21}\ \text{FLOPs.}
    $$

    As a multiple of the capstone: $1.2\times10^{21} / 1.21\times10^{19} \approx 99\times$ — about **100x** the capstone pretrain.

    Single-GPU wall-clock at MFU 0.45:

    $$
    t = \frac{1.2\times10^{21}}{0.45 \times 3.12\times10^{14}} = \frac{1.2\times10^{21}}{1.404\times10^{14}} \approx 8.5\times10^{6}\ \text{s} \approx 2{,}370\ \text{GPU-hours} \approx 99\ \text{days.}
    $$

    So 1B does not break on memory — it breaks on *time*: ~2,400 GPU-hours on one card is unacceptable wall-clock, which is exactly why you go multi-GPU with DDP (near-linear speedup while the model still fits per GPU, which it does at 1B).

**5.** (Implementation) Extend `stacklm/cost.py` with an inference-side accounting to make the "worked example" reproducible in code. Implement `serving_flops(n_params, n_requests, tokens_per_request=256)` using the $2ND$ inference rule, verify it reproduces the chapter's $\approx 5.2\times10^{19}$ FLOPs for a billion 256-token requests, and add `breakeven_requests(...)` returning how many 256-token requests it takes for cumulative serving FLOPs to equal the 20B-token pretraining budget. Report that number.

??? note "Solution"

    ```python
    # stacklm/cost.py  (append)
    def serving_flops(n_params: int, n_requests: int,
                      tokens_per_request: int = 256) -> float:
        """Inference compute via the 2ND rule: 2 FLOPs/param/generated token."""
        return 2.0 * n_params * n_requests * tokens_per_request


    def breakeven_requests(n_params: int = N_PARAMS,
                           pretrain_tokens: int = 20_000_000_000,
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

    Check on part 1: $2 \times (1.014\times10^{8}) \times 10^{9} \times 256 = 5.19\times10^{19}$ FLOPs — matches the chapter's $\approx 5.2\times10^{19}$ (it is ~4x the entire pretraining budget).

    Break-even: the $N$ cancels, so it is independent of model size:

    $$
    n = \frac{6ND}{2N \cdot 256} = \frac{6 \times 2\times10^{10}}{2 \times 256} = \frac{1.2\times10^{11}}{512} \approx 2.34\times10^{8}.
    $$

    **~234 million requests.** After roughly a quarter-billion 256-token requests, cumulative serving compute equals the *entire* 20B-token pretraining budget — which is exactly why serving economics, not training economics, dominate the decision to over-train.

**6.** (Implementation) Footgun #3 in the "Checkpoint hygiene" admonition warns that ~1.6 GB of state written every 500 steps "fills a disk overnight," and prescribes keeping "a rolling window of the last *k* plus milestone checkpoints ... and delete the rest." Implement `prune_checkpoints(ckpt_dir, keep_last=3, milestones=())` consistent with `stacklm/repro.py`: given a directory of files named `ckpt_step{N}.pt`, delete every checkpoint except the `keep_last` highest steps and any step in `milestones`; return the list of deleted filenames.

??? note "Solution"

    ```python
    # stacklm/repro.py  (part 4: retention policy)
    import os, re

    _CKPT_RE = re.compile(r"^ckpt_step(\d+)\.pt$")


    def prune_checkpoints(ckpt_dir: str, keep_last: int = 3,
                          milestones=()) -> list[str]:
        """Enforce checkpoint retention: keep the `keep_last` newest checkpoints
        plus any `milestones` steps (e.g. end of stable phase, end of decay);
        delete the rest. Returns the filenames removed.

        Prevents the 'fills a disk overnight' footgun: ~1.6 GB/checkpoint
        (16 B/param x 101M) every 500 steps is unsustainable without pruning.
        """
        milestones = set(milestones)
        found = []                                   # (step, filename)
        for name in os.listdir(ckpt_dir):
            m = _CKPT_RE.match(name)
            if m:
                found.append((int(m.group(1)), name))
        found.sort()                                 # ascending by step

        newest = {step for step, _ in found[-keep_last:]} if keep_last > 0 else set()
        keep = newest | milestones

        removed = []
        for step, name in found:
            if step not in keep:
                os.remove(os.path.join(ckpt_dir, name))
                removed.append(name)
        return removed
    ```

    Walking through it: we discover checkpoints by the `ckpt_step{N}.pt` pattern (ignoring `.tmp` files from an in-flight atomic write and any unrelated file), sort by *step number* rather than filesystem order or lexicographic name (so `ckpt_step900.pt` is correctly older than `ckpt_step1000.pt`), and form the keep-set as the union of the `keep_last` newest steps and the explicit `milestones`. Everything else is deleted and its name returned for logging.

    Example: with checkpoints at steps `{500, 1000, 1500, 2000, 2500}`, `keep_last=2`, `milestones={500}` keeps `{2500, 2000}` (newest two) plus `{500}` (a milestone), and returns `["ckpt_step1000.pt", "ckpt_step1500.pt"]`. Guarding `keep_last > 0` avoids `found[-0:]` accidentally selecting the whole list when a caller passes `keep_last=0` to retain milestones only.
