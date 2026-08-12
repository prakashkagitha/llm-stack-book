# 15.4 Pretraining With a Real Trainer: torchtitan and nanotron

In [Chapter 14.7](../14-capstone/07-pretraining-run.html) we wrote `train.py` by hand: a single
Python file that pulled together the model, the packed dataset, the Muon+AdamW pair, and the
Warmup-Stable-Decay (WSD) schedule, then spent ~20 GPU-hours turning them into `ckpt_stable.pt`. We
did that on purpose — every mechanism was visible, every tensor was budgeted, every failure mode was
made structurally impossible in code you could read top to bottom. That is exactly the right way to
*learn* pretraining. It is not how you *ship* it.

When a practitioner sits down at work to pretrain a model — even a 100M one, and certainly anything
that needs more than one GPU — they do not hand-roll the loop. They reach for a **pretraining
platform**: a maintained codebase that already implements FSDP2, tensor parallelism, activation
checkpointing, distributed checkpointing, fault tolerance, and MFU logging correctly, and lets you
describe your run in a config file instead of a thousand lines of Python. In 2026 the PyTorch-native
default is **torchtitan**; the HuggingFace-ecosystem alternative used to build SmolLM is
**nanotron**; the heavyweight cluster incumbents are **Megatron-LM** and **DeepSpeed**; and for the
single-GPU or few-GPU small path, HuggingFace **`accelerate`** wraps FSDP with almost no code.

This chapter takes the *same* Stack-100M target from `capstone/PLAN.md` and launches a real
distributed run with these tools — mapping every knob back onto the hand-written loop of Ch. 14.7 so
you can see that the config file is not magic; it is the loop you already wrote, exposed as TOML.

## Why a Real Trainer (and What It Buys You Over Ch. 14.7)

Our Ch. 14.7 loop is correct and complete for one GPU. The moment you want a second GPU, or a bigger
model, or a run that survives a spot-instance reclaim across a 512-GPU cluster, you start
re-implementing — badly, under deadline — things these platforms already got right. Here is the
honest crosswalk between what we built by hand and what the library hands you.

| Stage / knob | Ch. 14.7 hand-rolled | torchtitan gives you |
|---|---|---|
| Data-parallel sharding | single GPU; a one-paragraph DDP/FSDP note | FSDP2 (`fully_shard`) as the default, degree set in config |
| Tensor / context parallel | not attempted at 100M | `tensor_parallel_degree`, `context_parallel_degree` knobs |
| Activation checkpointing | hand-wrapped `CheckpointedBlock` | `[activation_checkpoint] mode = "selective"` in config |
| WSD schedule | our `wsd_lr(step, ...)` | `[lr_scheduler]` warmup + decay-ratio |
| Chunked loss head | our `fused_ce_z_loss` | built-in chunked/compiled cross-entropy |
| Checkpoint + resume | our `save_checkpoint`/`_rng_snapshot` | Distributed Checkpoint (DCP), async, sharded |
| MFU logging | our `utilization()` | logged every step from `num_flops_per_token` |
| Fault tolerance | atomic rename + NaN guard | DCP + optional `torchft` semi-sync |
| `torch.compile` | one line, easy to break with graph breaks | integrated, tested against the parallelism plan |

The thesis of Part XV in one sentence: **you hand-roll to understand the mechanism; you reach for
the library to get the mechanism at scale, tested, and maintained.** Nothing below contradicts
Ch. 14.7 — the config keys *are* our variables, and where a default differs we will say so.

!!! note "Aside: torchtitan is a platform, not a framework you import"

    A subtle but important distinction. `torchtitan` is not a `pip install` you call from your own
    script the way you call `transformers`. It is a *reference codebase you run and extend*: you
    clone the repo, point `torchrun` at its `train.py`, and pass a TOML config. To train a model
    that is not one of its built-in flavors (Llama 3/4, DeepSeek-V3, Qwen3, and others), you add a
    small `TrainSpec` in code that registers your model class, its args, and its build functions. It
    is deliberately hackable and deliberately not backward-compatible — the maintainers reserve the
    right to move flags between releases. Pin a commit for reproducibility; read the example TOMLs in
    `torchtitan/models/*/train_configs/` for the exact keys your checkout uses.

## torchtitan: The PyTorch-Native Pretrainer

torchtitan's design goal is to be *the* place PyTorch's newest distributed features (FSDP2, the
`DTensor`-based tensor/context/pipeline parallel APIs, `torch.compile`, Float8, Distributed
Checkpoint) are composed into one working pretraining loop, so you get them without gluing them
together yourself. Under the hood it is exactly the structure of Ch. 14.7 — a step loop with
forward, backward, grad-clip, optimizer step, schedule, logging, checkpoint — but each piece is a
swappable component selected by config.

### Install and launch shape

```bash
# torchtitan is run from its repo, driven by torchrun. Pin a commit for reproducibility;
# the maintainers move flags between releases and say so.
git clone https://github.com/pytorch/torchtitan && cd torchtitan
git checkout <a-pinned-commit-sha>        # do NOT float on main for a real run
pip install -r requirements.txt
pip install -e .                          # installs the `torchtitan` package + `torchtitan` CLI

# Launch: torchrun spawns one process per GPU; the TOML is the whole run description.
# --nproc_per_node = GPUs on this node. CONFIG_FILE points at your TOML.
CONFIG_FILE=./train_configs/stack100m.toml \
  torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
           -m torchtitan.train --job.config_file $CONFIG_FILE
```

That `torchrun --nproc_per_node=8 ... -m torchtitan.train` is doing what our Ch. 14.7 "Scaling Out"
section only sketched: it launches 8 processes, initializes the process group, and hands each rank
its shard of the work. The single-GPU Stack-100M run is just `--nproc_per_node=1` with a
`data_parallel_shard_degree` of 1 — the same code path, no sharding, so torchtitan is a legitimate
choice even for the flagship single-A100 tier, not only for clusters.

### The config file is the loop

Below is a torchtitan-style TOML for Stack-100M. Every value is annotated with the Ch. 14.7 variable
it replaces. Treat the exact key names as *illustrative of the current schema* — verify against the
example TOML shipped with your pinned checkout, because sections get renamed (for instance
`[training]` fields have migrated in and out of `[parallelism]` and `[activation_checkpoint]` across
releases).

```yaml
# train_configs/stack100m.toml   —  Stack-100M, torchtitan schema (annotate to Ch. 14.7)

[job]
dump_folder = "./outputs/stack-100m"        # where logs, metrics, checkpoints land
description  = "Stack-100M pretraining, WSD stable phase"

[model]
name          = "stack100m"                  # the name we REGISTER via TrainSpec (below)
flavor        = "100M"                        # selects our StackConfig-equivalent model args
tokenizer_path = "./tokenizer/stack100m-32768"   # the Ch. 15.3 / 14.3 tokenizer

[training]
seq_len         = 2048                        # cfg.model.max_seq_len
local_batch_size = 32                         # cfg.micro_batch_size (per-rank micro-batch)
# global batch = local_batch_size * dp_degree * grad_accum; see the batch-size note below
steps           = 34332                       # cfg.stop_at_step  (stable-phase end; 18.0B tokens)
max_norm        = 1.0                         # cfg.grad_clip  (global grad-norm clip)
seed            = 1337                         # cfg.seed
compile         = true                        # cfg.compile_model  (torch.compile the model)
mixed_precision_param  = "bfloat16"           # bf16 autocast params (Ch. 14.7: no fp16 scaler)
mixed_precision_reduce = "float32"            # fp32 gradient reduction — the master-copy analogue

[optimizer]
name        = "AdamW"                          # built-in; Muon hybrid needs a custom builder (below)
lr          = 3e-3                             # cfg.adamw_peak_lr  (the AdamW group's peak)
weight_decay = 0.1                             # cfg.weight_decay
beta1 = 0.9                                    # cfg.betas[0]
beta2 = 0.95                                   # cfg.betas[1]

[lr_scheduler]
warmup_steps = 2000                           # cfg.warmup_steps
decay_ratio  = 0.0                             # NO decay leg in this run — see the note below.
# decay_ratio is a FRACTION OF `steps`, not of some longer schedule: leaving it at 0.10 here
# would decay the LR to zero over the LAST ~3,433 of these 34,332 steps. Ch. 14.8 owns the decay.
decay_type   = "sqrt"                          # WSD's short sqrt decay leg (MiniCPM-style)
min_lr_factor = 0.0                            # decays to 0 (cfg.final_frac)

[parallelism]
data_parallel_shard_degree = -1               # -1 = "use all remaining GPUs for FSDP2 sharding"
data_parallel_replicate_degree = 1            # pure FSDP (no HSDP replication) at this size
tensor_parallel_degree = 1                    # 100M does not need TP; > 1 shards each matmul
context_parallel_degree = 1                   # for long context; 1 here (seq_len 2048)
pipeline_parallel_degree = 1                  # no PP at this size

[activation_checkpoint]
mode = "none"                                 # cfg.activation_checkpointing=False on the A100 tier
# mode = "selective"; selective_ac_option = "op"   # the recommended memory/compute trade at scale

[checkpoint]
enable   = true
folder   = "checkpoints"
interval = 1000                               # cfg.ckpt_every  (steps between checkpoints)
keep_latest_k = 5                             # cfg.keep_last_ckpts
async_mode = "async"                          # DCP async save: overlap checkpoint I/O with compute

[metrics]
log_freq = 10                                 # cfg.log_every
enable_tensorboard = true
# enable_wandb = true                         # torchtitan logs loss, tok/s, MFU, memory, grad_norm
```

Read that TOML next to the `TrainConfig` dataclass in Ch. 14.7 and the correspondence is
one-to-one. The library did not invent new concepts; it *named the same knobs* and wired them to
tested implementations. That is the whole value proposition.

### Registering Stack-100M: the `TrainSpec`

torchtitan will not know what `name = "stack100m"` means until you register it. The registration hook
is where Ch. 14.4's model, Ch. 14.6's optimizer, and Ch. 14.2's dataloader plug in — the library
supplies the loop, you supply the components. This is the single most important mechanism in the
chapter, because it is exactly the "swap my hand-rolled parts into the production loop" move.

```python
# torchtitan/models/stack100m/__init__.py   (added to your torchtitan checkout)
# A TrainSpec bundles the model, its args, and the build_* functions the loop calls.
# The exact import paths/field names track your pinned commit — check protocols/train_spec.py.
from dataclasses import dataclass
import torch
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.protocols.model import BaseModelArgs                  # model args must subclass this
from torchtitan.components.optimizer import build_optimizers          # default AdamW builder
from torchtitan.components.lr_scheduler import build_lr_schedulers    # default WSD-capable scheduler
from torchtitan.components.loss import build_cross_entropy_loss       # the chunked/compiled CE default

from stacklm.config import StackConfig
from stacklm.data import build_stack100m_dataloader   # our Ch. 14.2 packed-shard loader
from stacklm.model.transformer import Stack100M       # our Ch. 14.4 model, unchanged


@dataclass
class Stack100MArgs(BaseModelArgs):
    """torchtitan calls this the model's *args*; it IS StackConfig by another name.
    `flavor = "100M"` in the TOML selects the entry in the registry below.
    Subclassing `BaseModelArgs` is not cosmetic: the loop calls `update_from_config()`
    (to push TOML values like seq_len into the args) and a per-token FLOP hook
    (`get_nparams_and_flops`) that the MFU logging below depends on — implement both,
    and check which of the args/model class owns them in your pinned commit."""
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 30
    n_heads: int = 8
    n_kv_heads: int = 2
    intermediate: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    # z_loss, qk_norm, nope_every, tie_embeddings ... all of PLAN.md §1 travels here.


# The flavor registry: "100M" -> the frozen PLAN.md §1 numbers.
stack100m_configs = {
    "100M":  Stack100MArgs(),                                   # defaults ARE the frozen config
    "debug": Stack100MArgs(n_layers=2, d_model=128, vocab_size=512),  # CI toy scale
}


def build_stack100m(model_args: Stack100MArgs) -> torch.nn.Module:
    """torchtitan hands us the parsed args; we return an nn.Module it will
    shard with FSDP2 and (optionally) torch.compile. The parallelize_fn (next)
    is what actually applies fully_shard / tensor-parallel plans to it."""
    # Skip the private bookkeeping fields BaseModelArgs contributes; only our own
    # config keys are meaningful to StackConfig.
    keys = {k: v for k, v in vars(model_args).items() if not k.startswith("_")}
    return Stack100M(StackConfig(**keys))


def parallelize_stack100m(model, world_mesh, parallel_dims, job_config):
    """Apply the parallelism PLAN to the model: FSDP2 `fully_shard` per block,
    optional tensor-parallel row/col-wise sharding of the projections, activation
    checkpointing, and torch.compile. torchtitan ships a reference
    `parallelize_llama` you can copy almost verbatim — the plan is per-module,
    and Stack100M's blocks look like Llama blocks (attention + SwiGLU MLP)."""
    from torchtitan.models.llama3.infra.parallelize import parallelize_llama
    return parallelize_llama(model, world_mesh, parallel_dims, job_config)


register_train_spec(TrainSpec(
    name="stack100m",
    model_cls=build_stack100m,
    model_args=stack100m_configs,
    parallelize_fn=parallelize_stack100m,
    pipelining_fn=None,                       # no pipeline parallel at 100M
    build_optimizers_fn=build_optimizers,     # <- swap for a Muon+AdamW builder (below)
    build_lr_schedulers_fn=build_lr_schedulers,
    build_dataloader_fn=build_stack100m_dataloader,   # our packed-shard loader (Ch. 14.2)
    build_tokenizer_fn=None,                  # the one genuinely optional hook (`| None`)
    build_loss_fn=build_cross_entropy_loss,   # torchtitan's chunked/compiled CE (see below)
))
# Note: `build_dataloader_fn` and `build_loss_fn` are REQUIRED callables — the trainer invokes
# them unconditionally in __init__, so passing None raises `TypeError: 'NoneType' object is not
# callable` rather than selecting a default. Name the default explicitly, as above.
```

Two of those hooks deserve a closer look because they are where our capstone diverges from
torchtitan's stock defaults.

## Mapping Every Knob: From the Hand-Written Loop to the Config

### FSDP2, and how it differs from our single-GPU assumption

Ch. 14.7 assumed the whole model, its gradients, and its optimizer state fit on one GPU — true at
101M params (~1.3 GB of weights+grads+opt-state, from that chapter's memory table). torchtitan's
default is **FSDP2** (`fully_shard`), the successor to the FSDP1 wrapper, built on `DTensor`. The
mechanism is the one from
[Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html):
each rank stores only its shard of every parameter, gradient, and optimizer-state tensor; before a
block's forward it **all-gathers** the full parameters for that block, runs the block, and frees the
gather; in backward it recomputes the gather, computes the gradient, and **reduce-scatters** it back
to shards. Memory per GPU falls roughly like $1/\text{dp\_degree}$; communication rises to pay for it.

`data_parallel_shard_degree = -1` tells torchtitan to use every GPU it was given for sharding. For
Stack-100M this is almost pure throughput scaling — the model already fits, so FSDP2 buys you
*larger effective batch and more tokens/second*, not the ability to fit at all. That is the honest
framing from Ch. 14.7's scale-out note: at 100M, distributed training is optional and about wall
clock, not feasibility.

```python
# What `fully_shard` does, in miniature — this is the FSDP2 API torchtitan calls per block.
from torch.distributed.fsdp import fully_shard
# Shard each transformer block across the data-parallel mesh, then the whole model.
for block in model.blocks:
    fully_shard(block, mesh=dp_mesh)          # per-block gather/free keeps peak memory low
fully_shard(model, mesh=dp_mesh)              # embeddings, final norm, lm_head
# torchtitan wraps this with mixed-precision (bf16 param, fp32 reduce) and reshard policies.
```

### The WSD schedule maps onto `[lr_scheduler]`

Ch. 14.6's WSD schedule — linear warmup, a long constant "stable" plateau, then a short sqrt/linear
decay leg — is what torchtitan's scheduler expresses through `warmup_steps` plus `decay_ratio`. The
`decay_ratio` is the *fraction of total steps spent in the decay leg*; the stable phase is
everything between warmup and the start of decay. So with `steps = 38147`, `warmup_steps = 2000`, and
`decay_ratio = 0.10` you get 2,000 warmup / ~32,332 stable / ~3,815 decay — Ch. 14.6's frozen split.

But recall the deliberate design decision from Ch. 14.7: **this chapter stops at the end of the
stable phase** (`stop_at_step = 34332`) and hands a *pre-decay* checkpoint to
[mid-training](../14-capstone/08-mid-training.html), which spends the decay leg annealing on premium
data. You reproduce that in torchtitan two ways. Either set `steps = 34332` **and**
`decay_ratio = 0.0`, so the whole run is warmup + stable and the LR never leaves its plateau (the
config above does this — it is exactly Ch. 14.7's `mult == 1.0` for every stable step); or keep
`steps = 38147` with `decay_ratio = 0.10` and let the decay leg run as an integrated mid-training
anneal. What you must *not* do is the tempting middle — `steps = 34332` with `decay_ratio = 0.10`
still decays, because the ratio is taken against `steps` itself, giving 2,000 warmup / ~28,899 stable
/ ~3,433 decay and handing mid-training an already-annealed checkpoint, exactly the costly
re-warming case the split exists to avoid. The general theory of why a stable-then-decay shape beats
cosine here, and why re-warming a decayed checkpoint is costly, is in
[Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

!!! warning "Common pitfall: assuming your trainer's 'WSD' is *your* WSD"

    "WSD" names a *shape*, not a formula. Trainers differ in the decay curve (linear vs. sqrt vs.
    cosine-to-zero vs. cosine-to-`min_lr`), in whether `decay_ratio` counts from the end or names an
    absolute step, and in whether warmup is measured in steps or tokens. Two runs both labelled "WSD"
    can put materially different LRs at step 30,000. Before trusting a config, *plot the realized LR*:
    step the scheduler in a loop, record `optimizer.param_groups[0]["lr"]`, and eyeball the curve
    against Ch. 14.6's. This is the single cheapest way to catch a schedule that silently disagrees
    with the one your hyperparameters were tuned for.

### Muon + AdamW: the optimizer hook torchtitan does not fill by default

torchtitan ships Adam and AdamW. Stack-100M's recipe (PLAN.md §5, Ch. 14.6) is the **Muon + AdamW
hybrid**: Muon's Newton-Schulz-orthogonalized momentum update for the 2-D hidden weight matrices,
AdamW for the tied embedding and every 1-D norm/QK-norm gain. You get it by passing your own
`build_optimizers_fn` in the `TrainSpec` — the same parameter-group split we wrote in Ch. 14.6,
returned as an object torchtitan can `.step()`.

```python
# A Muon+AdamW builder that satisfies torchtitan's optimizer-container interface.
# torchtitan expects an object exposing .step(), .zero_grad(), and a state_dict for DCP.
from stacklm.optim import build_optimizers as build_muon_adamw   # our Ch. 14.6 factory


def build_stack100m_optimizers(model_parts, job_config, parallel_dims=None):
    """model_parts is a list (pipeline stages); at PP=1 it is [model].
    We route 2-D matrices -> Muon, embeddings/1-D -> AdamW, exactly as Ch. 14.6,
    reading peak LRs from the TOML so the config stays the single source of truth."""
    (model,) = model_parts
    muon, adamw = build_muon_adamw(
        model,
        muon_lr=6e-3,                                  # cfg.muon_peak_lr
        adamw_lr=job_config.optimizer.lr,              # 3e-3 from the TOML
        weight_decay=job_config.optimizer.weight_decay,
        betas=(job_config.optimizer.beta1, job_config.optimizer.beta2),
    )
    # Wrap [muon, adamw] so step()/zero_grad()/state_dict() fan out to both —
    # the "one clip, two optimizers" pattern of Ch. 14.7, hosted.
    return PairedOptimizers([muon, adamw])


class PairedOptimizers:
    """Duck-types the container interface torchtitan's loop and DCP expect:
    step/zero_grad on every inner optimizer, and a state_dict keyed per optimizer
    so a resume restores both. torchtitan's own `OptimizersContainer` *builds* its
    optimizers from a class + kwargs, which cannot express "two different optimizers
    over two parameter groups", so we supply the container ourselves. Check the
    protocol in `torchtitan/components/optimizer.py` at your pinned commit — if it
    requires more (e.g. an `optimizers` attribute or lr-scheduler hooks), subclass it."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {f"opt{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, sd):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(sd[f"opt{i}"])
```

The important honesty here: **the library gives you the loop, not the research optimizer.** Muon is
recent (Jordan et al., 2024) and torchtitan may or may not have first-class support by the time you
read this; the `build_optimizers_fn` hook is precisely the seam that lets you bring it anyway. This
is the general shape of using a real trainer for a not-yet-standard recipe — you inherit the tested
distributed loop and inject the one component that is your contribution.

### The chunked loss head, for free

Ch. 14.7 spent a whole section proving that at `d_model = 512`, `vocab = 32768`, the *loss head*
(not attention) dominates memory — the unchunked `(B·T, V)` logits peak near 30 GB — and built
`fused_ce_z_loss` to chunk it. torchtitan reaches the same conclusion and ships a chunked/compiled
cross-entropy as its default `build_loss_fn`; recent versions integrate variants of the same fused
linear-cross-entropy kernels we recommended (Liger-Kernel, cut-cross-entropy). You take that default
by naming it — `build_loss_fn=build_cross_entropy_loss` in the `TrainSpec`; the hook is required, so
there is no "leave it None and get the default". If you need the z-loss term
(PLAN.md §1's PaLM-style `logsumexp` penalty), supply a `build_loss_fn` that adds it — the same term
our `_chunk_ce` computed from the `logsumexp` it already needed. See
[Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html) for the
memory arithmetic this default is quietly saving you.

### Activation checkpointing: `mode = "selective"`

Ch. 14.7 hand-wrote `CheckpointedBlock` and then noted, correctly, that all-or-nothing recompute is
not frontier practice — **selective** recompute (Korthikanti et al., 2022) recomputes only the
cheap-to-recompute, expensive-to-store ops. torchtitan exposes exactly this as one config line:

```yaml
[activation_checkpoint]
mode = "selective"            # "none" | "selective" | "full"
selective_ac_option = "op"    # "op" = policy-based per-op; or an int N = every Nth block
```

`mode = "selective"` with `selective_ac_option = "op"` is torchtitan's default recommendation and
implements the per-op `CheckpointPolicy` we described in Ch. 14.7 — it keeps the results of the
expensive matmuls and recomputes the cheap SwiGLU elementwise intermediates. On the A100 flagship
tier we leave it `"none"` (the extra ~33% compute of full recompute costs GPU-hours we would rather
spend on tokens, and 101M fits comfortably); on the 24 GB and 16 GB tiers `"selective"` is close to
free memory relief.

### Distributed Checkpoint (DCP): our `save_checkpoint`, sharded and async

Our Ch. 14.7 checkpoint was a single `torch.save` with an atomic rename and a hand-built RNG
snapshot — perfect for one GPU, useless across 512 ranks where a monolithic save would serialize the
whole cluster through one process. torchtitan uses **PyTorch Distributed Checkpoint (DCP)**: each
rank saves its shard in parallel to a checkpoint *directory*, the format is resharding-tolerant (you
can resume a 64-GPU checkpoint on 8 GPUs), and `async_mode = "async"` copies tensors to CPU and
writes them off the critical path so the step loop barely stalls. This is the production version of
[Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html);
the guarantee we cared about in Ch. 14.7 — resume to the exact step, optimizer state, and data
position — is the same, implemented for the distributed case.

```bash
# Resume is implicit: torchtitan finds the latest step in [checkpoint].folder and continues.
# To convert a sharded DCP checkpoint to a single consolidated file for export/serving:
python -m torch.distributed.checkpoint.format_utils dcp_to_torch \
       ./outputs/stack-100m/checkpoints/step-34332 ./stack-100m-consolidated.pt
```

### MFU logging: our `utilization()`, computed every step

Ch. 14.7 built `utilization()` and stressed the discipline of **stating your FLOP convention** —
6ND-only understates a deep-thin model at seq_len 2048 by ~31% because it omits attention's
score/value matmuls. torchtitan computes a `num_flops_per_token` for the registered model and logs
MFU every `log_freq` steps against the device's known peak (it maintains a table of peak bf16 FLOP/s
per accelerator). You still owe the reader the convention, and here it is a *third* one — not
Ch. 14.7's. torchtitan's reference count is
$6(N - N_{\text{embed}}) + 12\,L\,H\,Q\,s$: attention-inclusive, but with the attention term **not**
halved for causality (the PaLM convention Ch. 14.1 flags, with an explicit source comment saying it
deliberately does not credit causal sparsity) and with the embedding parameters **excluded** from
the $6N$ term. For Stack-100M that is
$6(101.3 - 16.8)\text{e}6 + 12 \times 30 \times 512 \times 2048 = 5.07\text{e}8 + 3.78\text{e}8
\approx 8.85\text{e}8$ FLOP/token, against Ch. 14.1/14.7's causal-halved
$6N + 6Lsd_q \approx 7.97\text{e}8$ — about **11% higher**, so the *same* run reports ~11% more MFU
under torchtitan's meter than under ours. Convert before you compare: multiply torchtitan's MFU by
$7.97/8.85 \approx 0.90$ to put it on Ch. 14.7's footing (and neither number is the 6ND-only one,
which is lower again). A representative torchtitan log line looks like:

```text
step: 12000  loss:  3.11  grad_norm:  0.42  lr: 3.00e-03
  tps: 1.42e5  mfu: 40.3%  memory: 21.7GiB(27.4%)  tflops: 125.7  end_to_end(s): 0.92
```

Those three throughput fields are one number in three costumes, and checking that they agree is a
free sanity test on your registration: `tflops = num_flops_per_token × tps` ($8.85\text{e}8 \times
1.42\text{e}5 = 1.257\text{e}14$) and `mfu = tflops / peak` ($125.7 / 312 = 40.3\%$ on an A100 —
consistent with the `memory` field's 79 GiB device). If `tflops` and `tps` do not reconcile through
the FLOP formula above, your `get_nparams_and_flops` is wrong and every MFU you report is wrong with
it. Every field there is a variable we logged by hand in Ch. 14.7 — `tps` is our `tokens_per_sec`,
`mfu` our `utilization()[0]`, `memory` our `max_memory_allocated`, `grad_norm` our pre-clip
`clip_grad_norm_` return. The trainer did not add observability you did not already understand; it
made it free.

## A Worked Run and Its Numbers

Let us put concrete magnitudes on a torchtitan Stack-100M run so the config above is not abstract.

!!! example "Worked example: batch, steps, and MFU on 8×H100 vs 1×A100"

    **Single A100 (the flagship tier).** `--nproc_per_node=1`,
    `data_parallel_shard_degree = 1` (no sharding), `local_batch_size = 32`, `seq_len = 2048`. With
    no data parallelism and no built-in gradient accumulation, one optimizer step sees
    $32 \times 2048 = 65{,}536$ tokens — an *eighth* of Ch. 14.6's target ≈0.5M-token effective
    batch. To recover the 524,288-token batch you either set
    `gradient_accumulation_steps = 8` if your torchtitan version supports it (recent ones do; older
    ones do not — check), or accept the smaller batch and note the Muon/WSD hyperparameters were
    tuned for ≈0.5M. This is the first real version-drift trap: **the from-scratch loop always had
    grad accumulation; not every trainer release exposes it, and torchtitan historically preferred
    scaling data-parallel degree instead.**

    **8×H100 (a realistic small cluster).** `--nproc_per_node=8`,
    `data_parallel_shard_degree = 8`, `local_batch_size = 8`. Global batch is
    $8 \text{ ranks} \times 8 \times 2048 = 131{,}072$ tokens per step — still shy of 0.5M, so pair
    it with `gradient_accumulation_steps = 4` for $524{,}288$. Now the step math: at
    $C_{\text{step}} \approx (6N + 6 L s d_q)\,B_{\text{tok}}$ per Ch. 14.1's attention-inclusive
    convention, with $N = 101.4\text{M}$, $L = 30$, $s = 2048$, $d_q = 512$, and
    $B_{\text{tok}} = 524{,}288$, the per-step FLOPs are on the order of
    $6 \times 101.4\text{e}6 \times 524{,}288 + 6 \times 30 \times 2048 \times 512 \times 524{,}288$
    $\approx 3.19\text{e}14 + 9.89\text{e}13 \approx 4.2\text{e}14$ FLOP. At an H100 bf16 peak of
    ~989 TFLOP/s each (8 GPUs ⇒ ~7.9 PFLOP/s peak) and, say, an achieved MFU of 0.45, sustained
    throughput is $\approx 0.45 \times 7.9\text{e}15 = 3.6\text{e}15$ FLOP/s, so a step takes
    $\approx 4.2\text{e}14 / 3.6\text{e}15 \approx 0.12$ s and the 18.0B-token stable phase
    ($\approx 34{,}332$ steps) finishes in **on the order of an hour of wall clock**, i.e. 8 GPUs ×
    ~1 hr ≈ **8 GPU-hr** plus communication overhead. Read that against the single A100's ~22–29
    GPU-hours carefully, because two different effects are stacked in it. FSDP2 buys the ~8×
    *wall-clock* compression at (approximately) constant GPU-hours — data parallelism never reduces
    total compute. The remaining ~3× in GPU-hours is the *device*: an H100 at 989 TFLOP/s bf16 dense
    against an A100 at 312. Compare like with like — 8 GPU-hr on H100s is ≈25 GPU-hr of A100
    time, squarely inside the 22–29 band Ch. 14.1 budgeted. Always
    quote MFU **with its convention** and confirm the peak-FLOP number for your exact SKU; treat
    989 TFLOP/s as the *denominator*, never as achievable throughput.

## nanotron: The YAML-Config Alternative

**nanotron** is HuggingFace's minimalist pretraining library — the codebase behind the SmolLM family
whose data recipe Stack-100M borrows. Where torchtitan leans on the newest PyTorch-native `DTensor`
APIs, nanotron implements 3D parallelism (data, tensor, pipeline) more explicitly and configures a
run through a **single YAML file** consumed by `run_train.py`. If your team already lives in the
HuggingFace ecosystem (its tokenizers, its `datasets`, its hub), nanotron is the natural neighbor.

```yaml
# stack100m.yaml  —  nanotron config (schema tracks your pinned nanotron commit)
general:
  project: stack-100m
  run: wsd-stable
  seed: 1337

model:
  model_config:
    hidden_size: 512                 # d_model
    num_hidden_layers: 30            # n_layers
    num_attention_heads: 8           # n_heads
    num_key_value_heads: 2           # GQA
    intermediate_size: 1408          # SwiGLU
    max_position_embeddings: 2048    # seq_len
    vocab_size: 32768
    rope_theta: 10000.0
    tie_word_embeddings: true

tokens:
  sequence_length: 2048
  micro_batch_size: 8                # per-rank micro-batch (cfg.micro_batch_size analogue)
  batch_accumulation_per_replica: 4  # nanotron DOES expose grad accum natively
  # global batch = dp x micro_batch_size x accumulation x seq_len = 8 x 8 x 4 x 2048 = 524,288 tokens
  train_steps: 34332                 # stable-phase end (Ch. 14.7 stop_at_step)

optimizer:
  optimizer_factory:
    name: adamW
    adam_beta1: 0.9
    adam_beta2: 0.95
  weight_decay: 0.1
  clip_grad: 1.0                     # cfg.grad_clip
  learning_rate_scheduler:
    learning_rate: 3.0e-3            # peak LR
    lr_warmup_steps: 2000            # cfg.warmup_steps
    lr_decay_style: "1-sqrt"         # WSD-style stable-then-inverse-sqrt decay
    lr_decay_steps: 3815             # LENGTH of the decay leg (Ch. 14.6's 3,815) — not its position
    lr_decay_starting_step: 34332    # WHERE it starts. Omit this and decay begins right after
                                     # warmup, flattening the LR to 0 by step ~5,800. Setting it to
                                     # train_steps keeps this run entirely on the stable plateau;
                                     # Ch. 14.8's anneal owns the leg itself.
    min_decay_lr: 0.0

parallelism:
  dp: 8                              # data-parallel degree
  tp: 1                              # tensor-parallel (raise for big hidden sizes)
  pp: 1                              # pipeline-parallel
  tp_mode: REDUCE_SCATTER            # sequence-parallel-style TP comm when tp>1

checkpoints:
  checkpoint_interval: 1000          # cfg.ckpt_every
  save_initial_state: false
  checkpoints_path: ./checkpoints/stack-100m
```

```bash
# nanotron launch: torchrun over run_train.py with --config-file. Same torchrun shape as torchtitan.
git clone https://github.com/huggingface/nanotron && cd nanotron
git checkout <pinned-sha> && pip install -e .
CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --nproc_per_node=8 run_train.py --config-file ../stack100m.yaml
```

Two things to notice about nanotron versus torchtitan, both honest trade-offs rather than a verdict.
First, **nanotron exposes gradient accumulation directly** (`batch_accumulation_per_replica`), so the
≈0.5M-token effective batch of Ch. 14.6 is a one-line setting rather than a version-dependent
feature — a real ergonomic advantage for the small, single-node runs this capstone targets. Do the
arithmetic explicitly, though: the global batch is `dp × micro_batch_size × accumulation × seq_len`,
so with `dp: 8` and `micro_batch_size: 8` the accumulation that lands on 524,288 tokens is **4**, not
8. Second, its `lr_decay_style: "1-sqrt"` with an explicit `lr_decay_steps` is a faithful WSD leg (the
SmolLM recipe used exactly this shape), and `lr_decay_steps` being *absolute* mirrors Ch. 14.7's
deliberate choice to pass `decay_steps` rather than a fraction — but it is an absolute *length*, not
an absolute *position*. The position is `lr_decay_starting_step`, which defaults to the end of warmup:
leave it out and your "long stable plateau" becomes a 3,815-step decay to zero starting at step 2,000,
followed by ~28,500 steps at LR 0. This is the exact same trap as torchtitan's `decay_ratio`, wearing
different clothes, and the same defense catches it — plot the realized LR. As with torchtitan, the
field names above are representative of the current schema — nanotron is research code and renames
things; validate against the examples in the repo you actually cloned.

!!! tip "Practitioner tip: pick the trainer your problem already points at"

    torchtitan if you want the newest PyTorch-native distributed features (FSDP2, DTensor
    tensor/context parallel, Float8 on Hopper/Blackwell) and are comfortable extending a reference
    codebase; nanotron if you live in the HuggingFace ecosystem, want explicit 3D parallelism and
    native grad accumulation, and value the SmolLM recipe lineage; Megatron-LM/DeepSpeed if you are
    at true cluster scale with an ops team; `accelerate` if you have one to a few GPUs and want your
    own Ch. 14.7-style loop with sharding bolted on and almost no new code. For Stack-100M at 100M
    params, *any* of them works and the single-GPU `accelerate` path is the least friction.

## Megatron-LM, DeepSpeed, and the `accelerate` Small Path

### The heavyweight incumbents

For runs far larger than Stack-100M, the two names you will hear at every frontier lab are
**Megatron-LM** (NVIDIA) and **DeepSpeed** (Microsoft), covered mechanistically in
[Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).
Megatron-LM is the origin of the tensor-parallel and sequence-parallel implementations that
everything else (torchtitan included) learned from; you configure a run through `pretrain_gpt.py`
command-line arguments rather than a config file:

```bash
# Megatron-LM, illustrative arg shape (versions move flags; read examples/ in your checkout).
# UNITS TRAP: --global-batch-size counts SEQUENCES, not tokens (the rest of this chapter counts
# tokens). 256 sequences x 2048 = 524,288 tokens/step, i.e. 256/(8 micro x 8 dp) = 4 accum steps.
# Passing 524288 here would ask for 1.07e9 tokens per optimizer step — a ~2000x overshoot.
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --num-layers 30 --hidden-size 512 --num-attention-heads 8 \
  --group-query-attention --num-query-groups 2 \
  --seq-length 2048 --max-position-embeddings 2048 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --micro-batch-size 8 --global-batch-size 256 \
  --lr 3.0e-3 --min-lr 0.0 --lr-warmup-iters 2000 \
  --lr-decay-style WSD --lr-wsd-decay-iters 3815 \
  --clip-grad 1.0 --bf16 --use-distributed-optimizer \
  --recompute-activations
  # --recompute-activations is shorthand for --recompute-granularity selective, and Megatron's
  # validate_args then asserts recompute_method is None — so do NOT add --recompute-method uniform
  # here. That flag belongs only with --recompute-granularity full (+ --recompute-num-layers N).
```

Note `--lr-decay-style WSD` — Megatron added WSD as a first-class schedule, `--global-batch-size`
(**in sequences**) handling the gradient-accumulation arithmetic for you, and
`--use-distributed-optimizer` giving you ZeRO-1-style optimizer sharding. **DeepSpeed** is the
other half of the pair: a ZeRO
(Zero Redundancy Optimizer) implementation you attach to an existing model via a `ds_config.json`,
sharding optimizer state (stage 1), gradients (stage 2), and parameters (stage 3), with CPU/NVMe
offload for when even the shards do not fit.

```json
{
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "bf16": { "enabled": true },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "reduce_scatter": true
  }
}
```

Note the units once more: DeepSpeed derives `train_batch_size = train_micro_batch_size_per_gpu ×
gradient_accumulation_steps × world_size`, so on 8 GPUs the values above are
$8 \times 4 \times 8 \times 2048 = 524{,}288$ tokens per step. Re-derive that product every time the
GPU count changes, or the batch your hyperparameters were tuned for changes with it.

Honesty check: Megatron-LM and DeepSpeed are *cluster* tools. Running them for a 101M single-node
model is using a forklift to carry a grocery bag — everything works, but the operational overhead
(specific NCCL/CUDA/driver pins, launcher quirks, config surface) is not worth it below the scale
where their tensor/pipeline parallelism actually earns its keep. We name them because the reader
will meet them at work, and because their concepts (ZeRO stages, tensor/sequence/pipeline
parallelism) are the vocabulary of
[Distributed Training II](../03-pretraining/06-distributed-model-parallel.html) — not because
Stack-100M needs them.

### The small path: `accelerate` + your Ch. 14.7 loop

The lowest-friction way to take the *exact* loop from Ch. 14.7 and give it FSDP/multi-GPU without
rewriting it into a platform is HuggingFace **`accelerate`**. You keep your `train.py`; `accelerate`
wraps the model, optimizer, and dataloader and injects the distributed backend and mixed precision
from a config you generate interactively.

```bash
accelerate config      # answer prompts: FSDP? yes; sharding strategy FULL_SHARD; bf16; etc.
# writes ~/.cache/huggingface/accelerate/default_config.yaml, then:
accelerate launch train.py    # your Ch. 14.7 script, now sharded across all visible GPUs
```

```python
# The minimal diff to Ch. 14.7's train.py — accelerate absorbs device placement,
# gradient accumulation, mixed precision, and grad clipping into the Accelerator.
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",                 # replaces the manual torch.autocast(bf16) block
    gradient_accumulation_steps=8,          # replaces the hand-written accumulate() loop;
                                            # 8 is the ONE-GPU value — divide by the rank count to
                                            # hold the ≈0.5M-token global batch when sharding
)
model, muon, adamw, train_loader = accelerator.prepare(model, muon, adamw, train_loader)

for batch in train_loader:
    with accelerator.accumulate(model):     # handles the /grad_accum scaling + zero_grad timing
        _, loss = model(batch["input_ids"], targets=batch["targets"],
                        position_ids=batch["position_ids"], seq_ids=batch["seq_ids"])
        accelerator.backward(loss)          # replaces loss.backward(); syncs grads on the last accum
        if accelerator.sync_gradients:      # true only on the real optimizer step
            accelerator.clip_grad_norm_(model.parameters(), 1.0)   # cfg.grad_clip
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
```

That is the whole spectrum: `accelerate` keeps *your* loop and adds sharding; torchtitan and
nanotron replace your loop with a config-driven platform; Megatron/DeepSpeed are the cluster-scale
end of the same continuum. All four run the identical mathematics you wrote by hand in Ch. 14.7 — the
choice is how much of the loop you want to own versus inherit.

!!! interview "Interview Corner"

    **Q:** You trained a 100M model fine on one GPU with a hand-written loop. Your lead asks you to
    move it to torchtitan on an 8-GPU node "for speed." Walk me through what actually changes, and
    one subtle correctness risk in the port.

    **A:** Almost nothing about the *mathematics* changes — torchtitan runs the same forward,
    backward, grad-clip, optimizer step, and WSD schedule; I express them as a TOML instead of
    Python, and register my model via a `TrainSpec`. What changes operationally: FSDP2
    (`data_parallel_shard_degree = 8`) shards params/grads/optimizer state across the 8 GPUs, so I
    get more tokens/second at the *same* total GPU-hours — a wall-clock win, not a feasibility one,
    since 100M already fit on one GPU. The subtle correctness risk is the **effective batch size**.
    My single-GPU loop used `grad_accum_steps = 8` to hit ≈0.5M tokens/step, which is what the Muon
    and WSD hyperparameters were tuned for. On 8 ranks with `local_batch_size = 8`, one step is now
    $8 \times 8 \times 2048 = 131{,}072$ tokens — a *different* effective batch — unless I also set
    gradient accumulation, and older torchtitan releases may not expose that flag at all. Get this
    wrong and the loss curve still falls, so it is invisible, but the LR schedule and optimizer are
    now mismatched to the batch. I would compute the global batch explicitly, set grad accum (or DP
    degree) to restore ≈0.5M, and verify by logging tokens-per-step before trusting the run. The
    secondary risks are the same order: confirm the trainer's "WSD" decay curve matches mine by
    plotting the realized LR, and confirm its MFU FLOP convention (attention-inclusive vs 6ND) before
    comparing numbers to my single-GPU baseline.

!!! warning "Common pitfall: floating on `main` and trusting flag names from a blog post"

    Every trainer in this chapter is fast-moving research code that renames configuration keys
    between releases — `[training]` fields migrate into `[parallelism]`, `decay_ratio` becomes
    `lr_decay_fraction`, `data_parallel_degree` splits into `_shard_degree` and `_replicate_degree`.
    A config copied from a six-month-old tutorial will silently ignore keys it no longer recognizes
    (many parsers do not error on unknown fields) and quietly run with defaults you did not intend.
    Two defenses: **pin a commit SHA** for any run whose numbers you need to reproduce, and derive
    your config from the example TOML/YAML *shipped in that exact checkout*, not from memory or a
    blog. Then dump the fully-resolved config the trainer actually parsed (torchtitan and nanotron
    both log it) and diff it against your intent before spending a single GPU-hour.

## Key Takeaways

!!! key "Key Takeaways"

    - **The config file is the Ch. 14.7 loop, exposed as TOML/YAML.** torchtitan and nanotron do not
      invent new concepts; every key — `seq_len`, `max_norm`, `warmup_steps`, `interval` — is a
      variable you already wrote by hand. You hand-roll to understand; you reach for the trainer to
      get it at scale, tested, and maintained.
    - **torchtitan is a PyTorch-native platform you extend, not a package you import.** Clone it,
      drive `torchtitan.train` with `torchrun`, and register your model via a `TrainSpec` whose
      `build_*` hooks are exactly where Ch. 14.4's model, Ch. 14.6's Muon+AdamW, and Ch. 14.2's
      loader plug in.
    - **FSDP2 at 100M is a throughput win, not a feasibility one.** Stack-100M fits on one GPU
      (~1.3 GB weights+grads+opt-state), so `data_parallel_shard_degree` buys wall-clock at the same
      total GPU-hours — matching Ch. 14.7's honest scale-out framing.
    - **Watch the effective batch size on the port.** Global batch = local_batch × dp_degree ×
      grad_accum, and gradient accumulation is version-dependent in torchtitan while native in
      nanotron. Get it wrong and the loss still falls while the LR/optimizer silently mismatch the
      batch they were tuned for.
    - **"WSD" is a shape, not a formula.** Trainers differ on decay curve, whether the decay length is
      absolute or a ratio, warmup units, and — the one that bites hardest — where the decay leg is
      *placed*: torchtitan's `decay_ratio` is a fraction of `steps` (so shortening the run does not
      remove the decay), while nanotron's `lr_decay_steps` is a length whose position comes from
      `lr_decay_starting_step` (which defaults to the end of warmup). Reproduce Ch. 14.6's split by
      plotting the *realized* LR before trusting any config, and stop at the stable-phase checkpoint
      so mid-training owns the decay leg.
    - **The chunked loss head, DCP checkpointing, selective activation checkpointing, and MFU logging
      all come free** — the exact mechanisms we built by hand in Ch. 14.7, now one config line each.
    - **Pick the tool your problem points at:** torchtitan for newest PyTorch-native distributed;
      nanotron for the HuggingFace/SmolLM lineage with native grad accum; Megatron-LM/DeepSpeed only
      at true cluster scale; `accelerate` to keep your own loop and add sharding with almost no code.
    - **Pin a commit and derive configs from the shipped examples.** These are fast-moving research
      codebases; flag names move, unknown keys are often ignored silently, and a stale tutorial config
      runs on defaults you did not choose.

## Further reading

- **pytorch/torchtitan** — the PyTorch-native pretraining platform (FSDP2, DTensor tensor/context
  parallel, `torch.compile`, Distributed Checkpoint, selective activation checkpointing); read the
  `train_configs/` example TOMLs and the `TrainSpec` protocol.
- Liang et al., *torchtitan: One-stop PyTorch native solution for production-ready LLM pre-training*
  (2024) — the paper describing torchtitan's design and the composition of its parallelism features.
- **huggingface/nanotron** — minimalist 3D-parallel pretraining library; the codebase behind the
  SmolLM models, whose data recipe Stack-100M borrows.
- Allal et al., *SmolLM2* / the SmolLM technical reports (HuggingFace, 2024–2025) — the small-model
  recipe (data mix, WSD schedule, over-training) this capstone follows, trained with nanotron.
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
  Parallelism* (2019) and Narayanan et al., *Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM* (2021) — the origin of tensor/pipeline parallelism and of
  achieved-vs-peak-FLOPs reporting.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) —
  the DeepSpeed ZeRO stages that FSDP later generalized.
- Hu et al., *MiniCPM* (2024) — the Warmup-Stable-Decay schedule and the decay-phase-as-mid-training
  insight that Ch. 14.6 and this chapter both build on.
