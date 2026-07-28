# %% [markdown]
# # Capstone: Pretrain Stack-100M on a Single H100 (Real Run)
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the book —
# > run it to get your own numbers.
#
# You will build the real ~101M-parameter `Stack-100M` model from the capstone's
# `stacklm` package and run genuine, bounded-step GPU pretraining on it — bf16
# autocast, gradient accumulation to a ~0.5M-token effective batch, the Muon+AdamW
# hybrid optimizer under a WSD learning-rate schedule, MuonClip/QK-clip, live
# tokens/s + MFU logging measured with `torch.cuda.Event`, checkpoint/resume, a
# plotted loss curve, and a sample generation — then extrapolate your own measured
# throughput to the capstone's ~20B-token full run.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/14-capstone/07-pretraining-run.html)
# for the full explanation, and `capstone/PLAN.md` in the book's repo for the
# canonical Stack-100M spec this notebook stays faithful to.

# %%
# torch is preinstalled on GPU images; numpy + matplotlib almost always are too, but
# pip install them defensively. No flash-attn / bitsandbytes / transformers needed:
# attention below uses torch's built-in `F.scaled_dot_product_attention`, which
# dispatches to a fused flash/cuDNN kernel on H100 in bf16 automatically.
%pip install -q numpy matplotlib

import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

# --- get the capstone package (`stacklm`) -------------------------------------
# This notebook is meant to run standalone on a fresh GPU box, so it clones the
# book's repo if `stacklm` isn't already importable (e.g. you're not running this
# from inside a checkout of the repo already).
REPO_URL = "https://github.com/prakashkagitha/llm-stack-book.git"
REPO_DIR = "llm-stack-book"

if not os.path.isdir(os.path.join(REPO_DIR, "capstone", "stacklm")):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
sys.path.insert(0, os.path.join(REPO_DIR, "capstone"))

from stacklm.config import StackConfig, count_params
from stacklm.model import Stack100M
from stacklm.tokenizer import StackTokenizer, SPECIAL_TOKENS
from stacklm.data import STACK100M_MIX, synthetic_corpus, build_shards, PackedMemmapDataset
from stacklm.optim import build_optimizers, wsd_lr, qk_clip_
from stacklm.train import autocast_ctx, save_checkpoint, load_checkpoint
from stacklm.serve import generate

# --- device / precision / seeding --------------------------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

assert torch.cuda.is_available(), "This notebook targets a CUDA GPU (1x H100 80GB)."
device = torch.device("cuda")
torch.cuda.manual_seed_all(SEED)

props = torch.cuda.get_device_properties(device)
print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB | "
      f"compute capability {props.major}.{props.minor}")
assert props.major >= 8, "bf16 tensor cores need compute capability >= 8.0 (Ampere+); H100 is 9.0."
assert torch.cuda.is_bf16_supported(), "This notebook trains in bf16 -- needs bf16 tensor-core support."

# TF32 for any stray fp32 matmul outside the autocast regions (embeddings, norms);
# the actual training math runs in bf16 under `autocast_ctx` below.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# %% [markdown]
# ## The canonical Stack-100M config
#
# `StackConfig()`'s defaults ARE the frozen capstone spec (`capstone/PLAN.md` sec. 1):
# `vocab_size=32768`, `d_model=512`, `n_layers=30`, GQA `n_heads=8`/`n_kv_heads=2`,
# SwiGLU `intermediate=1408`, tied embeddings, QK-norm, and NoPE (RoPE skipped) on
# every 4th layer. `count_params` reproduces the ~101.4M arithmetic by hand; we
# assert it matches `model.num_params()` exactly, the same invariant the capstone's
# CPU smoke test checks at toy scale.
#
# **Expected result:** total parameters print as **101,353,728 (~101.35M)** — this
# number is exact arithmetic, not a benchmark, so it should match on every machine.

# %%
cfg = StackConfig()  # the full, canonical Stack-100M config -- nothing scaled down here
acct = count_params(cfg)
for name, n in acct.items():
    print(f"{name:26s} {n:>12,}")

model = Stack100M(cfg).to(device)
n_params = model.num_params()
assert n_params == acct["total"], "hand arithmetic must match the model's real parameter count"
print(f"\nmodel.num_params() = {n_params:,}  ({n_params / 1e6:.2f}M)")

fp32_bytes = n_params * 4
bf16_bytes = n_params * 2
print(f"fp32 master weights ~ {fp32_bytes / 1e9:.3f} GB  |  bf16 compute copy ~ {bf16_bytes / 1e9:.3f} GB")
print("(both trivially fit in an 80GB H100 alongside activations and Muon+AdamW optimizer state)")

# %% [markdown]
# ## Data: packed synthetic corpus, document-aware masking
#
# The real Stack-100M pretraining mix is 70% FineWeb-Edu / 15% Cosmopedia v2 / 10%
# StarCoder / 5% FineMath, streamed and packed to 2048-token, document-aware
# sequences (Ch. 14.2). To keep this notebook self-contained and network-free we
# use the capstone's own **synthetic corpus generator** (`stacklm.data.synthetic`)
# instead — the same in-process fallback the CI smoke test uses, just at a bigger,
# GPU-appropriate scale.
#
# One honest shortcut: we train the from-scratch byte-BPE tokenizer (Ch. 14.3) to a
# **much smaller** vocabulary than the model's. We *ask* for up to `TOK_VOCAB` merges,
# but this tiny synthetic corpus has so few distinct byte-pairs that BPE saturates
# early — expect only a **few hundred** learned merges, not thousands (the cell prints
# the actual `tok.vocab_size`). Training the real 32,768-merge tokenizer in pure
# Python on a full corpus is a multi-minute CPU job — that's Ch. 14.3's chapter, not
# this GPU chapter's. The model's `cfg.vocab_size` stays fixed at **32768** (the number
# that fixes the ~101M param count); every id this smaller tokenizer emits is
# `< tok.vocab_size <= 32768`, a valid subset of `[0, 32768)`, so the embedding lookup
# and cross-entropy are exactly correct -- the embedding table is simply
# under-saturated versus a real run, exactly like early in a real 32768-vocab run
# before the tokenizer has seen enough distinct byte-pairs.
#
# **Expected result:** a few hundred packed training sequences of length 2048 built
# in well under a minute of CPU time (this is the one CPU-bound, non-GPU cell in
# the notebook).

# %%
TOK_VOCAB = 4096          # requested CAP; the tiny synthetic vocab saturates far below this
N_DOCS_PER_SOURCE = 800   # synthetic docs per mix source (4 sources -> 3,200 docs total)

print("training a byte-level BPE tokenizer on a synthetic sample (CPU)...")
t0 = time.time()
sample_text = "\n".join(
    doc["text"] for entry in STACK100M_MIX for doc in synthetic_corpus(entry, n_docs=150)
)
tok = StackTokenizer()
tok.train(sample_text, vocab_size=TOK_VOCAB, special_tokens=SPECIAL_TOKENS)
print(f"tokenizer: vocab_size={tok.vocab_size}  merges={len(tok.merges)}  ({time.time() - t0:.1f}s)")
assert tok.vocab_size <= cfg.vocab_size, "tokenizer ids must fit inside the model's embedding table"

print("packing the synthetic mix into uint16 memmap shards (Ch. 14.2 format)...")
t0 = time.time()
docs = []
for entry in STACK100M_MIX:
    docs.extend(list(synthetic_corpus(entry, n_docs=N_DOCS_PER_SOURCE)))
random.shuffle(docs)
n_val = max(32, len(docs) // 20)
train_docs, val_docs = docs[n_val:], docs[:n_val]

data_dir = tempfile.mkdtemp(prefix="stack100m_")
train_dir, val_dir = f"{data_dir}/train", f"{data_dir}/val"
build_shards(train_docs, tok, train_dir, seq_len=cfg.max_seq_len, tokens_per_shard=cfg.max_seq_len * 4096)
build_shards(val_docs, tok, val_dir, seq_len=cfg.max_seq_len, tokens_per_shard=cfg.max_seq_len * 1024)

train_ds = PackedMemmapDataset(train_dir)
val_ds = PackedMemmapDataset(val_dir)
print(f"{len(train_docs)} train docs -> {len(train_ds)} packed sequences of {cfg.max_seq_len} tokens "
      f"({time.time() - t0:.1f}s)")
print(f"{len(val_docs)} val docs -> {len(val_ds)} packed sequences")
assert len(train_ds) >= 256, "need at least one optimizer step's worth of packed sequences (see below)"
assert len(val_ds) >= 1, "need at least one val sequence to evaluate on"

# %% [markdown]
# ## Optimizer & schedule: Muon+AdamW, WSD, ~0.5M-token effective batch
#
# `build_optimizers` (Ch. 14.6) routes every 2-D hidden weight matrix (attention
# Q/K/V/O, SwiGLU up/gate/down) to **Muon** (Newton-Schulz-orthogonalized momentum;
# Jordan et al., 2024 -- used at scale by Kimi K2) and the tied embedding plus every
# 1-D tensor (RMSNorm/QK-norm gains) to **AdamW**. Muon and AdamW keep *different*
# absolute peak learning rates, so we scale each optimizer's own base LR by a
# common **WSD** (Warmup-Stable-Decay; MiniCPM/Hu et al., 2024) shape rather than
# overwriting every group to one literal value.
#
# `MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS` reproduces the chapter's
# `32 x 2048 x 8 = 524,288`-token effective-batch *target* -- the H100's 80GB could
# fit a much bigger *micro*-batch, but the effective batch is what the schedule and
# optimizer hyperparameters are tuned around. (Each packed window yields a
# 2047-token input after the next-token shift, so the measured tokens/step is a hair
# under the round 524,288 target; the throughput numbers below count the real
# `input_ids.numel()`, not the target.)

# %%
MICRO_BATCH_SIZE = 32     # sequences per forward/backward micro-batch
GRAD_ACCUM_STEPS = 8      # 32 * 2048 * 8 = 524,288 tokens/optimizer-step (PLAN.md's ~0.5M target)
SEQ_LEN = cfg.max_seq_len  # 2048

PEAK_LR = 6e-3             # Muon group's peak LR; AdamW group runs at peak_lr/2 (Ch. 14.6 default)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_STEPS = 20
TOTAL_STEPS = 200          # a BOUNDED demo run; the real flagship run is 38,147 steps (see below)
QK_CLIP_EVERY = 20         # MuonClip/QK-clip cadence (Kimi K2, 2025)
QK_TAU = 100.0

muon, adamw = build_optimizers(model, muon_lr=PEAK_LR, adamw_lr=PEAK_LR / 2, weight_decay=WEIGHT_DECAY)
optimizers = [muon, adamw]
for opt in optimizers:
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]  # remember each group's own peak LR before the WSD shape rescales it


def set_lr(step: int) -> float:
    """Scale each optimizer group's OWN base LR by a shared WSD shape (peak_lr=1
    normalizes wsd_lr to a pure 0..1 multiplier), so Muon and AdamW keep their
    different absolute peak LRs instead of both being overwritten to one value."""
    mult = wsd_lr(step, peak_lr=1.0, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS)
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
    return mult


tokens_per_step_target = MICRO_BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
print(f"effective batch (target) = {tokens_per_step_target:,} tokens/optimizer-step")

# H100 SXM dense bf16 tensor-core peak, per NVIDIA's published spec sheet (no
# structured sparsity); a PCIe H100 is somewhat lower, an A100 SXM is 312e12.
H100_BF16_PEAK_FLOPS = 989e12


def estimate_mfu(n_params: int, tokens_per_sec: float, peak_flops: float = H100_BF16_PEAK_FLOPS) -> float:
    """6ND rule (Kaplan et al., 2020): one forward+backward pass costs ~6 FLOPs/param/token."""
    return (6 * n_params * tokens_per_sec) / peak_flops

# %% [markdown]
# ## The training loop: bf16 autocast, global grad clip, CUDA-event timing
#
# Each optimizer step: zero grads once, accumulate `GRAD_ACCUM_STEPS` micro-batches
# of loss/backward under bf16 autocast (loss divided by the accumulation count
# *before* `.backward()`), clip the **global** gradient norm across every
# parameter, then step both optimizers. Every `QK_CLIP_EVERY` steps we run one extra
# no-grad forward to harvest the per-head max attention logit and then apply
# MuonClip/QK-clip -- kept *outside* the timed region so this diagnostic doesn't
# pollute the throughput numbers. Throughput is measured with `torch.cuda.Event`
# (not `time.time()`, which mostly measures how fast the CPU enqueues asynchronous
# kernels) bracketing a `torch.cuda.synchronize()`-guarded region, and we track peak
# memory with `torch.cuda.max_memory_allocated`. A short, untimed warmup runs first
# so kernel autotuning doesn't pollute the first "real" measurement.
#
# We checkpoint (model + both optimizers + step) every `CKPT_EVERY` steps and
# auto-resume from `latest.pt` if this cell is re-run after an interruption --
# exactly the crash-safety the chapter argues a multi-hour rented-GPU job needs.
#
# **Expected result:** loss starts near `ln(32768) ~= 10.4` nats/token (uniform
# guessing over the model's full vocab) and drops into single digits within the
# first few dozen steps, then continues down more slowly -- on this small,
# repetitive synthetic corpus it will likely fall faster and lower than the
# **~2.8-3.2 nats/token** the real 20B-token FineWeb-Edu/Cosmopedia mix lands at
# (`capstone/PLAN.md`); treat this run as a mechanics/throughput proof, not a
# quality benchmark. MFU for this un-fused, uncompiled eager loop is plausibly on
# the order of **~20-45%** of the H100's peak (compare the book's own A100 worked
# example, in the ~40-45% range) -- add `torch.compile` for a real shot at closing
# that gap further. **Peak GPU memory** is on the order of **tens of GB** — the
# `32 x 2048` logits over a 32,768-way softmax dominate — but comfortably under the
# 80GB ceiling.

# %%
CKPT_DIR = os.path.join(data_dir, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "latest.pt")
LOG_EVERY, EVAL_EVERY, CKPT_EVERY = 10, 50, 50
COMPILE_MODEL = False  # flip True on H100 for extra throughput; left off so compile
                        # time doesn't eat into this notebook's "a few minutes" budget.
                        # Note: `stacklm`'s save/load_checkpoint don't unwrap
                        # `torch.compile`'s `_orig_mod.*` state-dict prefix (the
                        # chapter's `_unwrap()` helper handles that) -- if you flip
                        # this on, save/load the uncompiled module's state dict yourself.


def infinite_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    while True:
        yield from loader


train_iter = infinite_loader(train_ds, MICRO_BATCH_SIZE)


@torch.no_grad()
def evaluate(model, dataset, batch_size=8, max_batches=8):
    # small batch + drop_last=False so a modest val set still yields >= 1 batch
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=False)
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx(device):
            _, loss = model(batch["input_ids"], targets=batch["targets"], seq_ids=batch["seq_ids"])
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / max(1, len(losses))
    return mean, math.exp(min(mean, 20.0))


if COMPILE_MODEL:
    model = torch.compile(model)

history = {"step": [], "loss": [], "lr": [], "grad_norm": [], "tokens_per_sec": [], "mfu": [], "val_loss": []}

start_step = 0
if os.path.exists(CKPT_PATH):
    start_step, _extra = load_checkpoint(CKPT_PATH, model, optimizers, map_location=device)
    print(f"resumed from checkpoint at step {start_step}")

# --- untimed warmup: let cuDNN/SDPA kernel selection and the caching allocator settle ---
model.train()
for _ in range(2):
    batch = {k: v.to(device) for k, v in next(train_iter).items()}
    with autocast_ctx(device):
        _, warm_loss = model(batch["input_ids"], targets=batch["targets"], seq_ids=batch["seq_ids"])
    warm_loss.backward()
    model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats(device)

start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

for step in range(start_step, TOTAL_STEPS):
    set_lr(step)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    # ---- timed region: pure forward/backward/optimizer-step for this batch ----
    start_ev.record()
    loss_acc, tokens_this_step = 0.0, 0
    for _ in range(GRAD_ACCUM_STEPS):
        batch = {k: v.to(device, non_blocking=True) for k, v in next(train_iter).items()}
        with autocast_ctx(device):
            _, loss = model(batch["input_ids"], targets=batch["targets"], seq_ids=batch["seq_ids"])
            loss = loss / GRAD_ACCUM_STEPS   # scale BEFORE backward, since grads simply add across micro-batches
        loss.backward()
        loss_acc += loss.item()
        tokens_this_step += batch["input_ids"].numel()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # global norm, once, jointly
    for opt in optimizers:
        opt.step()

    end_ev.record()
    torch.cuda.synchronize()                       # wait for all queued GPU work before reading the timer
    dt_s = start_ev.elapsed_time(end_ev) / 1000.0   # cuda events report elapsed time in milliseconds

    # ---- MuonClip / QK-clip: an untimed diagnostic forward + in-place rescale ----
    if QK_CLIP_EVERY and step % QK_CLIP_EVERY == 0:
        rec = {}
        with torch.no_grad():
            b = {k: v.to(device) for k, v in next(train_iter).items()}
            model(b["input_ids"], seq_ids=b["seq_ids"], record=rec)
        qk_clip_(model, rec, tau=QK_TAU)

    tokens_per_sec = tokens_this_step / dt_s
    mfu = estimate_mfu(n_params, tokens_per_sec)

    history["step"].append(step)
    history["loss"].append(loss_acc)
    history["lr"].append(optimizers[0].param_groups[0]["lr"])
    history["grad_norm"].append(float(grad_norm))
    history["tokens_per_sec"].append(tokens_per_sec)
    history["mfu"].append(mfu)

    val_loss = None
    if step % EVAL_EVERY == 0:
        val_loss, _val_ppl = evaluate(model, val_ds)
    history["val_loss"].append(val_loss)

    if step % LOG_EVERY == 0:
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        vl = f"  val {val_loss:.3f}" if val_loss is not None else ""
        print(f"step {step:>4}/{TOTAL_STEPS}  loss {loss_acc:.3f}{vl}  lr {history['lr'][-1]:.2e}  "
              f"|g| {float(grad_norm):.2f}  tok/s {tokens_per_sec:,.0f}  mfu {mfu * 100:.1f}%  "
              f"peak_mem {peak_mem_gb:.1f}GB")

    if step % CKPT_EVERY == 0 and step > 0:
        save_checkpoint(CKPT_PATH, model, optimizers, step)

save_checkpoint(os.path.join(CKPT_DIR, "final.pt"), model, optimizers, TOTAL_STEPS)
print(f"\ndone: {TOTAL_STEPS} steps, final loss {history['loss'][-1]:.3f}, "
      f"peak GPU memory {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB")

# %% [markdown]
# ## Proving checkpoint/resume actually works
#
# A 15-25 GPU-hour job on a rented H100 *will* occasionally be interrupted -- a
# spot reclaim, a driver hiccup. The chapter's own practitioner tip is to dry-run
# the resume path before trusting it on money you're paying for by the hour: build
# a **fresh** model + optimizers, load the checkpoint just written, and confirm the
# loss on a fresh batch looks like a *trained* model, not a randomly-initialized
# one (`~ln(vocab_size) ~= 10.4`).
#
# **Expected result:** the resumed model's eval loss should sit close to the last
# logged training loss above -- nowhere near the ~10.4 nats/token a fresh random
# init would give.

# %%
resumed_model = Stack100M(cfg).to(device)
resumed_muon, resumed_adamw = build_optimizers(
    resumed_model, muon_lr=PEAK_LR, adamw_lr=PEAK_LR / 2, weight_decay=WEIGHT_DECAY)
resumed_optimizers = [resumed_muon, resumed_adamw]

resumed_step, _extra = load_checkpoint(
    os.path.join(CKPT_DIR, "final.pt"), resumed_model, resumed_optimizers, map_location=device)
print(f"loaded checkpoint written at step {resumed_step}")

resumed_model.eval()
with torch.no_grad():
    batch = {k: v.to(device) for k, v in next(train_iter).items()}
    with autocast_ctx(device):
        _, resumed_loss = resumed_model(batch["input_ids"], targets=batch["targets"], seq_ids=batch["seq_ids"])

random_init_loss = math.log(cfg.vocab_size)
print(f"resumed-model eval loss on a fresh batch: {resumed_loss.item():.3f}  "
      f"(last training loss was {history['loss'][-1]:.3f}; a fresh random init would be ~{random_init_loss:.1f})")
assert resumed_loss.item() < 0.9 * random_init_loss, "a correctly resumed model should not look randomly initialized"

# %% [markdown]
# ## Loss curve, throughput, and MFU
#
# Small multiples, one axis each -- loss, grad norm, tokens/s, and MFU do not share
# a scale, so we plot them as separate panels rather than a dual-axis chart.

# %%
import matplotlib.pyplot as plt

steps = history["step"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].plot(steps, history["loss"], color="#4C72B0", linewidth=1.6)
axes[0, 0].set_title("train loss (nats/token)")
axes[0, 0].set_xlabel("optimizer step")
axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(steps, history["grad_norm"], color="#DD8452", linewidth=1.6)
axes[0, 1].axhline(GRAD_CLIP, color="#888888", linestyle="--", linewidth=1.0, label=f"clip={GRAD_CLIP}")
axes[0, 1].set_title("pre-clip global grad norm")
axes[0, 1].set_xlabel("optimizer step")
axes[0, 1].legend(frameon=False)
axes[0, 1].grid(alpha=0.25)

axes[1, 0].plot(steps, history["tokens_per_sec"], color="#55A868", linewidth=1.6)
axes[1, 0].set_title("throughput (tokens/s)")
axes[1, 0].set_xlabel("optimizer step")
axes[1, 0].grid(alpha=0.25)

mfu_pct = [m * 100 for m in history["mfu"]]
axes[1, 1].plot(steps, mfu_pct, color="#8172B2", linewidth=1.6)
axes[1, 1].set_title("MFU (% of H100 bf16 peak)")
axes[1, 1].set_xlabel("optimizer step")
axes[1, 1].grid(alpha=0.25)

fig.suptitle("Stack-100M bounded pretraining demo (1x H100)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## A sample generation
#
# `stacklm.serve.generate` is the same naive, no-KV-cache autoregressive sampler
# the serving chapter (Ch. 14.11) runs on a quantized model on a laptop -- correct,
# just not the throughput-optimized path.
#
# One demo-specific wrinkle: the model has a full **32,768**-slot output head, but
# our shortcut tokenizer only defined a few hundred ids, and `decode` only knows
# those. A barely-trained model would happily sample an id no byte-pair maps to and
# crash the decoder. So we register a temporary forward hook on `lm_head` that masks
# every logit `>= tok.vocab_size` to `-inf` for the duration of generation --
# restricting sampling to the tokenizer's live id range. (In a real 32,768-vocab run
# the tokenizer covers the whole head and no such mask is needed.)
#
# **Expected result:** at only `TOTAL_STEPS` steps on a small synthetic corpus, do
# not expect coherent language -- expect a plausible-looking but largely
# meaningless stream of the synthetic vocabulary's words/punctuation. This cell is
# a mechanics check (does generation run, does it stop at EOS, is decoding
# lossless) rather than a quality demonstration; the real 20B-token run on
# FineWeb-Edu/Cosmopedia produces the actually-readable model.

# %%
model.eval()
_valid_vocab = tok.vocab_size  # only ids < this are decodable by our shortcut tokenizer


def _restrict_to_tokenizer_vocab(module, inputs, output):
    output[..., _valid_vocab:] = float("-inf")
    return output


_hook = model.lm_head.register_forward_hook(_restrict_to_tokenizer_vocab)
try:
    sample = generate(model, tok, prompt="the process", max_new_tokens=40, temperature=0.8)
finally:
    _hook.remove()
print(f"generated: {sample!r}")

# %% [markdown]
# ## Extrapolating to the real ~20B-token / ~$100 run
#
# `capstone/PLAN.md` frames the flagship run as **~15-25 GPU-hours, ~USD 40-100,
# on a single A100**. We extrapolate from *your own measured* steady-state
# throughput above (never a number we invent) to the real ~20B-token budget, and
# compare against the A100 figure using the two GPUs' published bf16 peak FLOPs.

# %%
steady = slice(-min(20, len(history["tokens_per_sec"])), None)  # skip early/warm-up steps
measured_tps = statistics.median(history["tokens_per_sec"][steady])
measured_mfu = statistics.median(history["mfu"][steady])

TARGET_TOKENS = 20_000_000_000  # ~20B tokens, ~200 tokens/param, deliberately over-trained (PLAN.md sec. 2)
projected_hours = (TARGET_TOKENS / measured_tps) / 3600

H100_SPOT_USD_PER_HR = 2.50  # illustrative on-demand/spot price -- varies by cloud, region, and date
projected_cost = projected_hours * H100_SPOT_USD_PER_HR

A100_BF16_PEAK_FLOPS = 312e12  # A100 80GB SXM bf16 dense peak, vendor spec (same figure the chapter uses)
speedup_vs_a100 = H100_BF16_PEAK_FLOPS / A100_BF16_PEAK_FLOPS

print(f"measured (this bounded demo): {measured_tps:,.0f} tok/s at MFU ~{measured_mfu * 100:.1f}%")
print(f"naive extrapolation to {TARGET_TOKENS / 1e9:.0f}B tokens: "
      f"~{projected_hours:.1f} H100-hours, ~${projected_cost:,.0f} at ~${H100_SPOT_USD_PER_HR:.2f}/GPU-hr")
print(f"H100 vs A100 bf16 peak FLOPs ratio: ~{speedup_vs_a100:.1f}x (vendor spec sheets)")
print("Compare to the capstone's own flagship estimate: ~15-25 A100 GPU-hours, ~$40-100 (capstone/PLAN.md). "
      "An H100's higher peak throughput should land you toward the faster/cheaper end of a similar order of "
      "magnitude -- but real wall-clock also depends on real (not synthetic) data I/O, dedup, periodic eval, "
      "and whether torch.compile is on, none of which this bounded demo fully exercises. Treat every number "
      "on this line as an order-of-magnitude estimate, not a promise.")

# %% [markdown]
# ### The exact command for the real run (not executed here)
#
# Scaling this notebook's own loop up to the real run means: (1) set `TOTAL_STEPS
# = 38_147` (`ceil(20e9 / 524_288)`, from `capstone/PLAN.md`'s ~20B-token budget),
# (2) replace the synthetic-corpus cells above with `stream_source(entry,
# offline=False)` for each `STACK100M_MIX` entry (streams the real FineWeb-Edu /
# Cosmopedia v2 / StarCoder / FineMath mix), (3) train the *full* 32,768-vocab
# tokenizer once (Ch. 14.3) instead of the tiny-vocab shortcut above, and (4) run
# it as a background script, not inside a notebook, so a lost connection cannot
# kill a 15-25 hour job. We only *print* the command below -- it is not invoked,
# since it would try to launch a multi-hour job the moment this cell runs.

# %%
real_run_cmd = (
    "nohup python3 -m stacklm.train "  # a CLI entry point you build from this notebook's loop (Ch. 14.7)
    "--total-steps 38147 --micro-batch-size 32 --grad-accum-steps 8 "
    "--train-shards data/train/*.bin --val-shards data/val/*.bin "
    "--ckpt-dir checkpoints/stack100m --peak-lr 6e-3 "
    "> train.log 2>&1 &"
)
print("Real single-H100 full-run command (run in a terminal / tmux session, NOT this notebook cell):\n")
print(f"  {real_run_cmd}")
print("\nOptional scale-out (unnecessary at 101M params, but available if you have an 8xH100 box; "
      "see the chapter's DDP/FSDP section): "
      "torchrun --nproc_per_node=8 -m stacklm.train --total-steps 38147 ...")

# %% [markdown]
# ## What you should see
#
# - **Param count is exact, not hedged:** `count_params(StackConfig())` and
#   `model.num_params()` should both print **101,353,728 (~101.35M)** on every
#   machine -- this is arithmetic, not a benchmark.
# - **Loss** starts near `ln(32768) ~= 10.4` nats/token and drops into single
#   digits within the first few dozen steps; on this small, repetitive synthetic
#   corpus it will likely fall faster/lower than the real run's **~2.8-3.2
#   nats/token** target (`capstone/PLAN.md`) -- that number describes the real
#   20B-token FineWeb-Edu/Cosmopedia/StarCoder/FineMath mix, not this demo's data.
# - **MFU** for this un-fused, uncompiled eager loop is plausibly on the order of
#   **~20-45%** of the H100's bf16 peak; the book's own worked A100 example lands
#   in the ~40-45% range with a comparably simple loop, so expect roughly that
#   ballpark or somewhat better on H100's newer tensor cores, and expect a further
#   boost from `torch.compile` + fused kernels. **Peak GPU memory** is on the order
#   of **tens of GB** (the `32 x 2048` logits over a 32,768-way softmax dominate) --
#   comfortably under the 80GB ceiling at this model size and micro-batch.
#
# **Key takeaways**
#
# - A single H100 (or A100) comfortably trains the *full* ~101M-parameter
#   Stack-100M config end to end; nothing here needs a distributed launcher.
# - **MFU (6ND FLOPs / peak FLOPs), measured with `torch.cuda.Event` +
#   `torch.cuda.synchronize()`, is the number that tells you if you're getting
#   your money's worth** -- `nvidia-smi`'s GPU-Util can read near 100% while MFU
#   sits far below what the hardware can deliver.
# - A checkpoint must capture **model + optimizer state + step** (and, for a
#   production run, the data-loader cursor and every RNG state) to resume as if an
#   interruption never happened -- we proved that above by loading into a fresh
#   model and checking the loss, not just trusting that `torch.save` succeeded.
# - This bounded, synthetic-data run is a **mechanics and throughput proof**, not
#   a quality benchmark -- the real ~20B-token run against real data is what turns
#   Stack-100M into an actually useful base model.
#
# **Next step:** [Ch. 14.8, Mid-Training](https://prakashkagitha.github.io/llm-stack-book/14-capstone/08-mid-training.html)
# picks up exactly where this notebook's `final.pt` checkpoint leaves off -- the
# WSD decay phase annealed onto a higher-quality data mix, plus RoPE-base
# rescaling for long-context extension.
