# 3.17 Training an LLM From Scratch: The End-to-End Recipe

Every chapter so far has handed you one gear of the machine: a tokenizer, a model, an optimizer, a distributed loop, a stability playbook, an evaluation harness, a decoding algorithm. Each was built and verified in isolation. This chapter's job is different — and, in a sense, simpler: assemble the gears, point at the exact bolts that connect them, and run the machine end to end, from a folder of raw text to a checkpoint you can sample from and score.

We will not re-derive anything. Every internal — why AdamW decouples weight decay, why FSDP all-gathers a layer just-in-time, why perplexity is `exp` of cross-entropy — was built from first principles elsewhere in the book, and we link to it rather than repeat it. What this chapter contributes that no other chapter does is the **glue**: the exact commands, the exact file formats, the exact flags that make chapter 3.5's `train.py`, chapter 2.7's `GPT`, and chapter 2.1's tokenizer cooperate on the same run — at four concrete scales, from a laptop CPU to a multi-node cluster. Treat this chapter as the lab manual you keep open in a second window while chapters 2.1 through 3.12 are open in the first.

## The Whole Pipeline on One Page

Here is the entire path from bytes on disk to a sampled sentence, as eight stages. Read it left to right, then top to bottom (it wraps because eight boxes don't fit one terminal width):

```text
+-------------+     +-------------+     +--------------+     +-------------+
| 1. raw      | --> | 2. tokenizer| --> | 3. uint16    | --> | 4. GPT      |
|    corpus   |     |    (BPE)    |     |    .bin      |     |    model    |
+-------------+     +-------------+     |    shards    |     +-------------+
                                         +--------------+            |
                                                                      v
+-------------+     +-------------+     +--------------+     +-------------+
| 8. inference| <-- | 7. eval     | <-- | 6. monitoring| <-- | 5. AdamW +  |
|    / sample |     |  (ppl+bench)|     | (loss/gnorm/ |     |  cosine-wu  |
+-------------+     +-------------+     |  MFU)        |     +-------------+
                                         +--------------+            ^
                                                                      |
                                              torchrun training loop
                                              (grad clip + checkpoint)
```

We do not re-derive any internal here; every stage links to the chapter that builds it from scratch.

{{fig:pretrain-pipeline-spine}}

What we *do* provide is concrete configs at **four scales** — a laptop/CPU toy (~10M params), a single GPU (~124M, GPT-2-small-sized), an 8-GPU node (~1-2B), and a multi-node cluster (~7B) — with the exact `torchrun` invocation, the exact hyperparameters, and the exact order-of-magnitude numbers you should expect to see at each one. By the end you will have run (or at least be able to run, verbatim) the smallest of these on a laptop in an afternoon, and understand precisely which three flags change to scale the same code to a cluster.

Here is the stage-to-deep-chapter map you'll use throughout:

- **Tokenization** — [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html)
- **Data pipeline** — [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html)
- **Objective** — [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html)
- **The GPT itself** — [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html)
- **Optimizer** — [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html)
- **LR schedule / hparams** — [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html)
- **Distributed loop** — [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)
- **MFU / roofline** — [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)
- **Stability** — [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)
- **Checkpointing** — [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html)
- **Evaluation** — [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html)
- **Sampling** — [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html)

## Stage 0: Environment, Running Example & Repo Layout

Install the handful of packages this whole chapter depends on:

```bash
pip install torch numpy tiktoken datasets lm-eval
# bf16 autocast/FSDP MixedPrecision need a reasonably recent torch (2.1+).
# torch.compile (optional but recommended for the GPU scales) needs 2.0+
# and a matching CUDA toolkit; it is a no-op fallback on CPU/MPS.
```

We lay the project out exactly the way `train.py` (from chapter 3.5) expects to find it:

```text
.
|-- data/              # raw HF dataset cache + tokenized .bin shards
|   |-- train_000.bin
|   |-- train_001.bin
|   `-- val_000.bin
|-- prepare.py          # Stage 2: corpus -> uint16 .bin shards
|-- train.py             # Stage 5: the reusable training loop (from 3.5)
|-- sample.py            # Stage 9: load a checkpoint, generate text
|-- eval_ppl.py          # Stage 8: held-out perplexity
`-- ckpt/                # step_*.pt checkpoints written by train.py
```

**The one running example, carried through every scale.** For the laptop toy we use **TinyStories** (Eldan & Li) — a corpus of short, simple children's stories deliberately small in vocabulary and complexity, so a ~10M-parameter model can learn *something coherent* in an afternoon on a CPU. For every GPU-scale run we switch to the **FineWeb-Edu `sample-10BT`** subset (Penedo et al., via the HuggingFace `datasets` library) — a deduplicated, quality-filtered, education-focused slice of the web, small enough to stream comfortably but large enough to feel like real pretraining. We deliberately do **not** re-derive where such a corpus comes from or how it was cleaned and deduplicated here — that is the full subject of [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) and [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html). This chapter assumes that work is done and starts from a `datasets.load_dataset(..., streaming=True)` call.

Fix two seeds before anything else, and keep them fixed across every run you compare:

```python
import torch
torch.manual_seed(0)          # model init -- identical weights on every rank (3.5)
# ShardedTokenLoader (below) seeds its own generator with 1234, independently,
# so data order is reproducible across resumes and across ddp/fsdp switches.
```

The exact torch/CUDA versions, GPU model names, and wall-clock numbers throughout this chapter are illustrative — treat every number as "on the order of," not a benchmark claim, per this book's numbers policy. Your mileage will vary with driver versions, interconnect, and how warmed-up your page cache is.

## Stage 1: Corpus to Tokenizer

The decision here is almost always **reuse, not retrain**. Training a BPE tokenizer from scratch is expensive, and unless you have a real reason (a non-English or code-heavy domain, a much smaller toy vocabulary, a specialized byte alphabet) you should default to a tokenizer someone has already trained on a huge, diverse corpus. For this recipe we reuse **GPT-2's byte-level BPE** via `tiktoken`:

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")     # vocab_size = 50257
ids = enc.encode_ordinary("The transformer processes tokens in parallel.")
text = enc.decode(ids)
print(ids[:8], "..." , "roundtrip ok:", text == "The transformer processes tokens in parallel.")
print("EOT token id:", enc.eot_token)   # 50256 -- the document/sequence separator
```

**Why 50304 and not 50257.** `tiktoken`'s `gpt2` encoding has exactly 50257 entries (256 byte tokens + 50,000 BPE merges + 1 special `<|endoftext|>` token). But 50257 is an awkward number for a GPU matmul — it is not a multiple of 64 (or 128, the tile size on newer tensor cores), so the final `lm_head` matmul and the softmax over it fall off the fast tensor-core path. The standard fix, and the one chapter 3.5's `train.py` bakes in as its default (`vocab=50304`), is to **pad the embedding table and output head up to the next multiple of 64** — 50304 — and simply never train the extra 47 rows (they start near-zero and stay there, or you can mask them out of the loss). This is a pure efficiency trick with zero effect on model quality; it is exactly why every config in this chapter uses `vocab=50304` rather than `50257`.

If you *do* need your own tokenizer — a smaller vocabulary for the TinyStories toy scale (roughly 8k merges is a reasonable choice for such a narrow, simple corpus), a non-English corpus, or a code-specific vocabulary — the from-scratch BPE trainer, complete with the pre-tokenization regex and merge-selection algorithm, lives in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html). Everything downstream in this chapter is agnostic to which tokenizer produced the integer IDs; only the `vocab_size` argument to `GPT` changes.

Once you have integer token IDs, the next step is turning them into vectors — the embedding table lookup that opens [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html).

## Stage 2: Tokenize to uint16 .bin Shards (`prepare.py`)

This is the single most important piece of glue in the chapter, because it is the contract between "a corpus" and "a training loop." Chapter 3.5's `ShardedTokenLoader` reads its data with `np.memmap(f, dtype=np.uint16)` and slices out `(ctx+1)`-token windows — so whatever `prepare.py` writes must be `uint16` on disk, or the loader silently reinterprets garbage bytes as token IDs.

```python
# prepare.py -- stream a HF dataset, tokenize with GPT-2 BPE, write uint16 .bin shards.
#
# Run:
#   python prepare.py --dataset tinystories  --out ./data           # laptop toy
#   python prepare.py --dataset fineweb-edu  --out ./data           # GPU scales
import argparse
import itertools
import os

import numpy as np
import tiktoken
from datasets import load_dataset

N_VAL_DOCS = 2000          # a small held-out slice, fixed doc count regardless of corpus size


def write_shards(doc_iter, text_field, enc, eot, out_dir, split, shard_tokens):
    """Encode every document, append EOT as a document separator, and flush
    fixed-size uint16 shards to disk as we go (so we never hold the whole
    tokenized corpus in memory)."""
    shard_idx, buf, total = 0, [], 0
    for doc in doc_iter:
        ids = enc.encode_ordinary(doc[text_field])   # no special-token parsing of raw text
        ids.append(eot)                               # EOT marks the document boundary
        buf.extend(ids)
        total += len(ids)
        while len(buf) >= shard_tokens:
            shard = np.array(buf[:shard_tokens], dtype=np.uint16)   # uint16: vocab 50304 < 65536
            path = os.path.join(out_dir, f"{split}_{shard_idx:03d}.bin")
            shard.tofile(path)
            print(f"wrote {path}: {len(shard):,} tokens ({shard.nbytes / 1e9:.3f} GB)")
            buf = buf[shard_tokens:]
            shard_idx += 1
    if buf:                                            # flush the final, smaller shard
        shard = np.array(buf, dtype=np.uint16)
        path = os.path.join(out_dir, f"{split}_{shard_idx:03d}.bin")
        shard.tofile(path)
        print(f"wrote {path}: {len(shard):,} tokens ({shard.nbytes / 1e9:.3f} GB)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fineweb-edu", "tinystories"], required=True)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)   # ~200 MB/shard as uint16
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    if args.dataset == "fineweb-edu":
        # HuggingFaceFW/fineweb-edu, the 10-billion-token curated sample -- see 3.1/3.2
        # for how this corpus was crawled, filtered, and deduplicated upstream.
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           split="train", streaming=True)
    else:
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    ds_iter = iter(ds)
    val_docs = itertools.islice(ds_iter, N_VAL_DOCS)   # first N_VAL_DOCS become the held-out set
    val_tokens = write_shards(val_docs, "text", enc, eot, args.out, "val", args.shard_tokens)
    train_tokens = write_shards(ds_iter, "text", enc, eot, args.out, "train", args.shard_tokens)

    total = train_tokens + val_tokens
    print(f"total: {total:,} tokens, {total * 2 / 1e9:.2f} GB on disk (uint16)")


if __name__ == "__main__":
    main()
```

Expected stdout for the FineWeb-Edu run looks like:

```text
wrote ./data/val_000.bin: 4,213,882 tokens (0.008 GB)
wrote ./data/train_000.bin: 100,000,000 tokens (0.200 GB)
wrote ./data/train_001.bin: 100,000,000 tokens (0.200 GB)
...
total: 9,842,113,207 tokens, 19.68 GB on disk (uint16)
```

**The dtype reconciliation, stated explicitly.** [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) works with `int32` token buffers in its production sharding/streaming discussion, because that chapter is agnostic to any particular vocabulary size and wants headroom for tokenizers with vocabularies above 65,536 (e.g. 128k+ multilingual vocabularies). Here, because we've fixed `vocab=50304 < 65536`, every token ID fits in an *unsigned 16-bit* integer, so we use `np.uint16` — this is the nanoGPT convention, and it halves both the on-disk footprint and the dataloader's memory-mapped I/O relative to `int32`, for zero loss of information. The one rule that must never be violated: **the writer's dtype and the loader's dtype must match exactly.** `prepare.py` writes `uint16`; `ShardedTokenLoader` in 3.5 opens with `np.memmap(f, dtype=np.uint16, mode="r")`. If you ever swap to a >65k vocabulary, you must change *both* sides to `uint32`/`int32` together, or every token ID silently wraps and corrupts training.

The `val_*.bin` shard is a small, held-out slice used only for the perplexity computation in Stage 8 — it never appears in a training batch. Production pipelines hold out entire documents or domains chosen deliberately (not a stream prefix, which can correlate with crawl order); see [Pretraining Data](../03-pretraining/01-pretraining-data.html) for the real-world version of this split.

{{fig:pretrain-shard-contract-window}}

## Stage 3: The Model — One `GPTConfig`, Four Sizes

The model is not reprinted here. The runnable one is the compact, weight-tied `GPT` class embedded inside chapter 3.5's `train.py` (constructor `GPT(vocab, d, h, n_layers, ctx)`) — the same GPT-2-style architecture built module-by-module in [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html): token embedding, learned positional embedding, a stack of pre-norm causal-attention + MLP blocks, a final LayerNorm, and a weight-tied `lm_head`. (2.7 also develops a modern variant — RoPE, RMSNorm, SwiGLU, GQA — which drops into the same training script unchanged, as noted below.) We do not reprint the modules; we only fix the numbers.

| Scale | `n_layer` | `n_head` | `d_model` | `ctx` | Params |
|---|---|---|---|---|---|
| Laptop / CPU toy | 6 | 8 | 256 | 256 | ~10M |
| Single GPU | 12 | 12 | 768 | 1024 | 124M |
| 8-GPU node | 24 | 16 | 2048 | 1024 | ~1.3B |
| Multi-node | 32 | 32 | 4096 | 4096 | ~7B |

**Weight tying and init.** The `GPT` in 2.7 ties `wte.weight` (the token embedding) and `lm_head.weight` (the output projection) — they are the *same* tensor, saving `vocab_size * d_model` parameters and, per Press & Wolf, acting as a mild regularizer. It also applies a "two-headed" initialization: a standard `N(0, 0.02)` init for most weights, but residual-stream-writing projections (the attention output projection and the MLP's second linear) get their standard deviation scaled by `1/sqrt(2 * n_layer)`, because those are the two places every block adds *directly* into the residual stream, and without the extra shrink their variance compounds across depth and destabilizes early training. See the `_init_weights` method in [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html) for the exact code. 3.5's compact `train.py` keeps the weight tying but relies on PyTorch's default initialization for brevity; porting `_init_weights` across is a recommended upgrade for the deeper/wider rows, where the residual-variance growth this scaling controls actually begins to bite.

**Where the parameter count comes from.** A convenient approximation, accurate to within a few percent for these shapes, is

$$
\text{params} \approx 12 \cdot n_{\text{layer}} \cdot d_{\text{model}}^2 + \text{vocab} \cdot d_{\text{model}}
$$

The first term is the four large matmuls per block (Q, K, V-and-output projections at roughly $4 d^2$, plus an MLP up/down pair at roughly $8d^2$, i.e. $12d^2$ per block) and the second is the (tied) embedding/head. Worked once for the 124M row: $12 \times 12 \times 768^2 = 84{,}934{,}656$, plus $50304 \times 768 = 38{,}633{,}472$, totalling $\approx 123.6\text{M}$ — matching the "124M" label (this is, not coincidentally, GPT-2-small's exact shape).

**Modern upgrades are one swap each.** Every knob above describes the *original* GPT-2-style block. Swapping in RoPE (rotary position embeddings, replacing the learned positional table), GQA (grouped-query attention, shrinking the KV heads), SwiGLU (replacing the GELU MLP), and RMSNorm (replacing LayerNorm) — the Llama-family recipe — is exactly the `ModernGPT` assembled in the "A Modern GPT, Assembled" section of [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html); nothing in this chapter's data pipeline, optimizer, or training loop changes when you make that swap, only the `Block` and `GPT` class definitions imported by `train.py`. The multi-node 7B row in this chapter's table is written in that modern style deliberately, since essentially no one trains a 7B-class model with the vanilla GPT-2 block anymore.

**Choosing params vs. tokens.** Nothing about the model config tells you how *long* to train it. That is a scaling-law question — the Chinchilla-optimal rule of thumb is roughly 20 tokens per parameter for a compute-optimal run (so a 124M model wants on the order of 2.5B tokens, a 7B model on the order of 140B) — developed with its derivation and the compute-vs-tokens tradeoff curve in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html). The token budgets in this chapter's scale table follow that rule loosely, adjusted for what is practical to actually run at each hardware tier.

## Stage 4: Optimizer and Schedule, Wired to the Loop

The optimizer is **AdamW**, `betas=(0.9, 0.95)`, `weight_decay=0.1`. 3.5's compact `train.py` constructs it as a plain `AdamW(model.parameters(), weight_decay=0.1)` for brevity; the recommended refinement — `configure_optimizer` in [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html) — splits parameters so that 1-D tensors (biases, LayerNorm/RMSNorm gains) are *excluded* from weight decay and only 2-D weight matrices get decayed, because decaying a norm gain toward zero destabilizes the normalization it controls. Swapping the plain optimizer for that param-group split is a one-function change worth making for any real run, and the *why* behind decoupled decay (as opposed to L2-regularization folded into the gradient) is derived in [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

The schedule is linear warmup into cosine decay — the same `lr_at` function that lives in chapter 3.5's `train.py`:

```python
def lr_at(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup          # linear ramp 0 -> base_lr
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * r))
```

We set `min_lr = 0.1 * base_lr` and clip the global gradient norm to `1.0` at every scale — both are training-loop defaults, not per-scale tunables. Peak learning rate scales *down* as the model grows (roughly `6e-4` for the 124M scale, lower for the 1.3B and 7B rows — see the per-scale table at the end of this chapter), which is the standard observation that wider models tolerate smaller steps before the loss becomes unstable. Warmup length is conventionally quoted as "about 1% of total steps" for a full-scale, long-token-budget run (e.g. 2000 steps out of a 200k-step run); for the shorter single-GPU run below, where the *entire* run is on the order of 9k steps, a proportionally shorter warmup of a few hundred steps (`train.py`'s own default is 100) is enough — scale the warmup fraction, not a fixed step count, when you change the total run length.

**Tokens-per-step, worked with real numbers.** The arithmetic that ties your batch-size flags to your token budget is:

$$
\text{tokens/step} = \text{local\_bsz} \times \text{ctx} \times \text{world\_size} \times \text{grad\_accum}
$$

For the single-GPU 124M row: `local_bsz=8`, `ctx=1024`, `world_size=1`, and a gradient-accumulation factor of `grad_accum=40` (to reach a reasonable effective batch without OOMing a single GPU) gives

$$
8 \times 1024 \times 1 \times 40 = 327{,}680 \approx 0.33\text{M tokens/step}
$$

Targeting a Chinchilla-ish 3B-token run: $3\times10^9 / 0.33\times10^6 \approx 9{,}100$ steps. That number — roughly 9k steps — is what you should see in the `--steps` flag for the single-GPU command in Stage 5 (paired with `--grad-accum 40`), and it is the total against which the cosine schedule and warmup fraction above are computed. The per-step body in Stage 5 implements this accumulation directly: it runs `grad_accum` micro-batches, each scaled by `1/grad_accum`, before a single `opt.step()`, and under DDP wraps all but the last micro-step in `model.no_sync()` to skip the redundant all-reduce — exactly the pattern described in [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html). Set `grad_accum=1` on the laptop toy (its batch already fits); raise it on the GPU tiers to hit the effective batch each row's token budget assumes.

For the deeper theory of warmup's role in taming Adam's early-training variance, the critical-batch-size regime where larger batches stop helping, and $\mu$P (maximal-update parameterization) for transferring a small model's tuned hyperparameters to a larger one without re-sweeping, see [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

## Stage 5: The Training Loop — `torchrun` at Four Scales

The training loop is not reprinted in full here — it is the single `train.py` built end-to-end in "Putting It Together: A Runnable Distributed `train.py`" in [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html). That file bundles the compact `GPT`, the `ShardedTokenLoader` (reading the `uint16` shards from Stage 2), `lr_at`, `save_checkpoint`, and a `--parallel {ddp,fsdp}` switch into one `torchrun`-launchable script. For orientation, here is just the per-step body — the part that actually touches loss, gradients, and the optimizer:

```python
# (contextlib is imported at the top of train.py)
for grp in opt.param_groups:
    grp["lr"] = lr_at(step, args.warmup, args.steps, args.lr, args.lr * 0.1)
opt.zero_grad(set_to_none=True)
for micro in range(args.grad_accum):          # gradient accumulation: G micro-steps per opt step
    x, y = loader.batch(device)
    last = micro == args.grad_accum - 1
    # Under DDP, suppress the all-reduce on every micro-step but the last.
    sync = contextlib.nullcontext() if last or args.parallel != "ddp" else model.no_sync()
    with sync:
        loss = nn.functional.cross_entropy(model(x).view(-1, 50304),
                                           y.view(-1)) / args.grad_accum   # mean over G micro-batches
        loss.backward()                       # collectives fire on the last micro-step
clip()                                         # global-norm clip, DDP or FSDP-sharded-aware
opt.step()
if step > 0 and step % args.ckpt_every == 0:
    save_checkpoint(model, step, args.ckpt_dir, rank)
```

And here are the exact four launches — the same file, four flag changes:

```bash
# --- Laptop / CPU (or Apple Silicon via MPS): learning, not speed. -----------------
# dist.init_process_group needs RANK/WORLD_SIZE/MASTER_ADDR set, so even ONE process
# goes through torchrun; and since there's no NCCL on CPU, pass --backend gloo
# (train.py selects a CPU device automatically on the gloo backend).
torchrun --nproc_per_node=1 train.py --data ./data --parallel ddp --backend gloo \
    --ctx 256 --local-bsz 8 --grad-accum 1 --steps 2000 --warmup 100

# --- Single GPU: ordinary DDP with world_size=1 (no-op all-reduce). ----------------
torchrun --nproc_per_node=1 train.py --data ./data --parallel ddp \
    --ctx 1024 --local-bsz 8 --grad-accum 40 --steps 9000 --warmup 200 --lr 6e-4

# --- 8-GPU node: FSDP FULL_SHARD = ZeRO-3 across the eight ranks. ------------------
torchrun --nproc_per_node=8 train.py --data ./data --parallel fsdp \
    --ctx 1024 --local-bsz 16 --grad-accum 4 --steps 40000 --warmup 1000 --lr 3e-4

# --- 2+ nodes: same file, add --nnodes and a shared rendezvous endpoint. -----------
torchrun --nnodes=2 --node_rank=$RANK --nproc_per_node=8 \
    --rdzv_endpoint=$HEAD:29500 train.py --data ./data --parallel fsdp \
    --ctx 4096 --local-bsz 4 --grad-accum 3 --steps 200000 --warmup 2000 --lr 1.5e-4
```

`--parallel {ddp,fsdp}` is the entire switch between "replicate the model" and "shard it." Under `fsdp`, `ShardingStrategy.FULL_SHARD` gives you ZeRO-3 semantics (parameters, gradients, and optimizer state all sharded across the group), and bf16 arrives via FSDP's `MixedPrecision` config rather than a separate autocast call — see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) for why bf16, not fp16, is the default here (its wider dynamic range avoids the overflow failure mode in Stage 7 below).

**Resuming.** `--ckpt-dir` and `--ckpt-every` control a full (unsharded) state-dict checkpoint, gathered onto rank 0 and written every `ckpt-every` steps — simple, and correct at the 8-GPU scale, but it pays a memory spike on rank 0 and does not save optimizer state. For a real multi-day run you want a *sharded* checkpoint (each rank writes only its slice, no single-host memory spike) that also captures the AdamW moment buffers and the data-loader's RNG position, so a restart is bit-continuous rather than restarting the LR schedule and optimizer momentum from cold — that machinery, built with `torch.distributed.checkpoint` and `FSDP.optim_state_dict`, is the subject of [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html).

{{fig:pretrain-four-scale-ladder}}

The multi-node command above is still pure data parallelism — every GPU still runs the *whole* forward/backward, just on a shard of the model's storage. At the point where a single node's aggregate memory can no longer hold even a `FULL_SHARD`-ed model comfortably, or the inter-node link becomes the bottleneck, you compose FSDP with tensor and pipeline parallelism (`HYBRID_SHARD` plus a device mesh with TP/PP axes) — developed in [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) and made concrete with the Megatron-LM and DeepSpeed configuration files in [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

## Stage 6: Monitoring — Reading a Live Run

Every `N` steps, log six numbers: **loss**, the **pre-clip** gradient norm (clip *after* you've logged it, or you'll only ever see 1.0), the **learning rate**, **ms/step**, **tokens/s**, and **MFU**. The pre-clip grad norm is the one that actually tells you something — a post-clip norm is clamped by construction and hides exactly the spikes you want to catch (see Stage 7).

**MFU**, reproduced from [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html):

```python
def mfu(num_params, tokens_per_sec, num_gpus, peak_tflops_per_gpu):
    flops_per_token = 6 * num_params                      # the 6N forward+backward rule
    achieved = flops_per_token * tokens_per_sec
    peak = num_gpus * peak_tflops_per_gpu * 1e12
    return achieved / peak

# Worked example: 124M model, single A100 (bf16 dense peak ~312 TFLOP/s).
u = mfu(num_params=124e6, tokens_per_sec=170_000, num_gpus=1, peak_tflops_per_gpu=312)
print(f"MFU = {u * 100:.1f}%")     # -> MFU = 40.5%
```

That 40.5% is squarely in the healthy range for a single-GPU bf16 run with `torch.compile` enabled and flash-attention-backed attention kernels.

**The loss curve's expected shape.** For a `vocab=50304` model, the loss of a uniform random guesser — and therefore the loss you should see logged at step 0 before any learning has happened — is $\ln(50304) \approx 10.83$. A healthy run drops fast over the first few hundred steps (the model is learning trivial statistics: token frequency, short local n-grams) and then settles into a long, slow grind as it learns genuinely harder long-range structure. If step 0's loss is not close to $\ln(\text{vocab})$, suspect a labeling or masking bug before suspecting the model — see the parallel warning in [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html).

A **healthy-run** mini-table to keep next to your terminal:

| Signal | Healthy range | Cause for concern |
|---|---|---|
| Pre-clip grad norm | steady, roughly 0.1-1.0 | growing trend, or repeated spikes >5x baseline |
| NaN/Inf count per step | 0 | any nonzero |
| MFU | ~40-55% (GPU scales) | <30% with idle-looking GPUs |

These thresholds, and the diagnostic playbook for when they're violated, are developed in full in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html). MFU is **not** a meaningful number on the CPU toy — there is no "peak FLOP/s" denominator worth computing for a laptop CPU in this context, so skip it there and watch loss and wall-clock only.

A tiny snippet to turn the printed log lines into a plot, assuming lines of the form `step  123 | loss 3.4521 | lr 3.21e-04 | 210 ms/step`:

```python
import re
import matplotlib.pyplot as plt

steps, losses = [], []
with open("train.log") as f:
    for line in f:
        m = re.search(r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))

plt.plot(steps, losses)
plt.xlabel("step"); plt.ylabel("train loss"); plt.yscale("log")
plt.savefig("loss_curve.png")
```

## Stage 7: What to Expect and Common Failures

A first real run rarely goes perfectly. Here are the four failure signatures you will actually see, in the order you're likely to hit them:

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss stuck near $\ln(\text{vocab})$ | targets not shifted, LR far too low, corrupt/repeated-token shard | decode a batch to real text; verify `x = w[:-1], y = w[1:]`; raise LR |
| NaN / sudden loss spike | fp16 overflow, or Adam amplifying a bad batch's gradient | switch fp16 -> bf16; keep grad clip at 1.0; skip-batch on high-loss steps; lower peak LR; QK-norm |
| Out of memory | activations overflow (not the sharded model state) | lower `local_bsz`, raise `grad_accum`; enable activation checkpointing; FSDP `FULL_SHARD`; bf16 |
| Low MFU, GPUs idle | cold page cache, `ctx` too small (memory-bound), or comms-bound multi-node | warm the shards; raise `ctx`/batch; switch `FULL_SHARD` -> `HYBRID_SHARD` |

**(1) Loss not dropping.** If the very first logged loss is already near $\ln(\text{vocab}) \approx 10.8$ and *stays* there for hundreds of steps, the model is not learning anything — three usual suspects. First, an off-by-one in the target shift: if `y` is not exactly `x` shifted one position to the right, the model is being asked to solve an impossible or trivial task and gradients are meaningless. Second, the learning rate might simply be too low to move a randomly-initialized model off its starting entropy within your logging window — try a 10x increase as a sanity probe. Third, a corrupted or degenerate shard (e.g. one that is accidentally all padding or one repeated token) will produce gradients but never reduce the *aggregate* loss meaningfully once mixed with real data at low probability. The fastest diagnostic is always the same: pull one training batch, decode `x` back to text with your tokenizer, and read it — a surprising fraction of "the model won't learn" bugs are visible the instant you look at the actual bytes going in.

**(2) NaN or a loss spike.** Two distinct mechanisms produce this, and they compound. First, fp16's narrow dynamic range (max ~65,504) overflows silently in attention logits or MLP activations well before training is "done," producing Inf that propagates to NaN on the very next backward pass — this is why bf16, with its much larger exponent range at the same 16-bit budget, is the default in every command in Stage 5. Second, even with bf16, an anomalous batch (a near-duplicate or repetitive sequence the model assigns very low probability, producing an unusually large gradient) gets **amplified** by Adam: if the historical second-moment estimate $\hat v_t$ hasn't yet absorbed a gradient this large, the effective step size $\hat m_t / \sqrt{\hat v_t}$ can be an order of magnitude larger than a typical step — the worked example in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html) shows a spike update **100x** larger than normal from exactly this mechanism, purely from the ratio of an anomalous gradient to the recent baseline. The standing defenses are: bf16 (not fp16), global-norm gradient clipping at 1.0, a skip-bad-batch guard that discards steps with implausibly high loss before they reach the optimizer, a somewhat lower peak LR, and QK-normalization inside attention to keep logits bounded before the softmax.

**(3) Out of memory.** With FSDP `FULL_SHARD`, the *model state* (params, gradients, optimizer moments) is sharded across every GPU and, per the accounting in [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html), typically shrinks to a few GB per GPU even for billion-parameter models — so an OOM at that scale is almost always **activations**, which are *not* sharded by data parallelism (each rank holds its own local batch's full activation tensors). The fix ladder: lower `local_bsz` and compensate with more gradient-accumulation steps to hold the effective batch size constant; enable activation checkpointing (recompute instead of store, trading compute for memory); confirm you are actually on `FULL_SHARD` and not accidentally on `NO_SHARD`; and confirm bf16, not fp32, compute. See [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html) for the sharding ladder and [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html) for the activation-checkpointing mechanics and the compute/memory tradeoff it buys.

**(4) Dataloader bottleneck / low MFU with idle GPUs.** If `nvidia-smi` shows GPUs mostly idle between steps, the bottleneck is upstream of compute. The usual causes: a cold page cache on the memory-mapped `.bin` shards (the very first pass over a large shard pays real disk I/O; subsequent epochs are fast once the OS has cached it — "warm the shards" by reading through them once before timing a run); a `ctx` too small to make the attention and MLP kernels compute-bound rather than memory-bound (very short sequences spend disproportionate time on fixed per-kernel-launch overhead); or, at the multi-node scale, a comms-bound `FULL_SHARD` where inter-node all-gathers dominate — switch to `HYBRID_SHARD`, which keeps the bandwidth-heavy parameter all-gathers on fast intra-node NVLink and only synchronizes gradients (the cheaper collective) across the slow inter-node link. The roofline framing for diagnosing exactly which of these applies — arithmetic intensity, memory- vs. compute-bound kernels — is [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).

## Stage 8: Evaluation — Perplexity plus a Quick Benchmark

A checkpoint is not "done" until you can put a defensible number on it. We use two: an **intrinsic** one (perplexity on held-out data, cheap, always available) and an **extrinsic** one (a standard benchmark, expensive, tells you whether the model can actually *do* anything).

```python
# eval_ppl.py -- held-out perplexity for a trained checkpoint.
# Run: python eval_ppl.py --data ./data --ckpt ckpt/step_9000.pt --ctx 1024
import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from train import GPT   # reuse the exact model class from train.py (3.5)


@torch.no_grad()
def eval_ppl(model, data_dir, ctx, device, batch_size=8):
    files = sorted(glob.glob(os.path.join(data_dir, "val_*.bin")))
    assert files, "no val_*.bin shards -- did prepare.py write a val split?"
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for f in files:
        mm = np.memmap(f, dtype=np.uint16, mode="r")     # must match prepare.py's writer dtype
        n_windows = (len(mm) - 1) // ctx                  # NON-overlapping (ctx+1)-token windows
        for i in range(0, n_windows, batch_size):
            ws = range(i, min(i + batch_size, n_windows))
            xs, ys = [], []
            for w in ws:
                off = w * ctx
                chunk = torch.from_numpy(mm[off:off + ctx + 1].astype(np.int64))
                xs.append(chunk[:-1]); ys.append(chunk[1:])
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                    reduction="sum")       # sum, not mean -- we accumulate by hand
            total_nll += loss.item()
            total_tokens += y.numel()
    mean_ce = total_nll / total_tokens                     # nats/token
    return mean_ce, math.exp(mean_ce)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--h", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=12)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(vocab=50304, d=args.d, h=args.h, n_layers=args.n_layers, ctx=args.ctx).to(device)
    model.load_state_dict(ckpt["model"])

    mean_ce, ppl = eval_ppl(model, args.data, args.ctx, device)
    print(f"val cross-entropy: {mean_ce:.4f} nats/token | perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
```

`ppl = exp(mean_ce)` is the standard definition — the effective branching factor of the model's next-token distribution. Because perplexity is tokenizer-dependent (a coarser vocabulary reports a lower perplexity for the same underlying predictive power), the tokenizer-agnostic **bits-per-byte** normalization is the fairer number for cross-tokenizer comparisons; its conversion from nats/token is covered in [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html). Expected perplexity, reading straight off the scale table at the end of this chapter: on the order of **~4** for the TinyStories toy, **~22** for the single-GPU 124M model, and **~14** for the 8-GPU 1.3B model — lower is better, and the trend (bigger model + more tokens = lower perplexity) is the entire content of scaling laws made concrete.

Then the quick benchmark — an *extrinsic* check that the model can do something beyond predicting the next token well. `train.py` saves a raw `{'model': state_dict}` checkpoint, **not** a HuggingFace-`transformers` directory, so `lm-eval`'s `hf` adapter cannot load `./ckpt` directly. Two bridges, both built in [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html): export the state dict into a `GPT2LMHeadModel` (whose config matches this GPT-2-shaped model almost exactly) and point `lm_eval` at that directory, or — simpler here — wrap the compact `GPT` in the thin custom `LM` subclass from 11.3 and score it directly. Once exported, the run is:

```bash
lm_eval --model hf --model_args pretrained=./ckpt_hf --tasks hellaswag --num_fewshot 0
``` Set honest expectations before you run this: HellaSwag is a 4-way multiple-choice task, so random guessing scores ~25%, and **both the TinyStories toy and the 124M single-GPU model will score at or barely above that random floor.** This is not a bug — it is the expected, correct outcome of a model this small trained on this few tokens; benchmark performance on tasks like HellaSwag, MMLU, or GSM8K only starts to separate meaningfully from random around the 1B-parameter-and-tens-of-billions-of-tokens range, which is exactly why the intrinsic perplexity number is the more useful signal at the two smallest scales in this chapter. Also note the harness reports both `acc` (raw argmax over answer choices) and `acc_norm` (length-normalized) as separate metrics — they can disagree by a few points, and comparing your number to a third-party leaderboard entry requires knowing which one they reported. For the landscape of *why* benchmark scores are noisy, contaminated, and hard to compare across papers, see [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html).

## Stage 9: Inference and Sampling

Closing the loop means loading a checkpoint and generating text — reusing the `generate()` method built in [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html) rather than re-deriving decoding.

```python
# sample.py -- load a checkpoint and generate text.
# Run: python sample.py --ckpt ckpt/step_2000.pt --prompt "Once upon a time" \
#          --temperature 0.8 --top_k 200
import argparse

import tiktoken
import torch
import torch.nn.functional as F

from train import GPT   # the same compact model class used everywhere in this chapter


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    # Identical algorithm to generate() in 2.7; adapted only in that this file's
    # compact GPT tracks its context length as `model.ctx`, not a GPTConfig.block_size.
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.ctx else idx[:, -model.ctx:]
        logits = model(idx_cond)[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        if top_p is not None:
            sp, si = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sp, dim=-1)
            sp[cum - sp > top_p] = 0.0
            sp /= sp.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(1, si, sp)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--h", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(vocab=50304, d=args.d, h=args.h, n_layers=args.n_layers, ctx=args.ctx).to(device)
    model.load_state_dict(ckpt["model"])

    ids = torch.tensor([enc.encode_ordinary(args.prompt)], device=device)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
```

```bash
python sample.py --ckpt ckpt/step_2000.pt --prompt "Once upon a time" \
    --temperature 0.8 --top_k 200
```

What to expect, per scale: the **TinyStories toy** (a few thousand steps on ~10-20M tokens) produces a short, coherent children's story — simple sentences, consistent characters within the passage, correct grammar, because that is exactly the narrow distribution it was trained on. The **124M single-GPU model**, trained on a few billion tokens of FineWeb-Edu, produces text that is locally fluent — grammatical clauses, plausible next words, topically consistent for a sentence or two — but wanders globally, losing the thread of an argument or drifting topic over a paragraph; this is the classic signature of a model with enough capacity and data to nail local statistics but not yet enough to hold long-range coherence.

Everything past this point — KV-cache management, continuous batching, multi-request serving — is production **inference serving**, not sampling theory, and is explicitly out of scope here; it begins in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html). The theory behind the sampling knobs themselves — temperature, top-k, top-p/nucleus, min-p, and beam search, with the entropy and calibration arguments for when to use which — is developed in full in [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html).

## The Four-Scale Recipe Table

The one artifact worth keeping open in a pinned tab — every config, launch command, and expectation for all four scales in one place:

| Scale | Hardware | Model (config) | Params | Tokens | Wall-clock (ballpark) | Final val loss / ppl | MFU | Launch |
|---|---|---|---|---|---|---|---|---|
| Laptop / CPU toy | 1 laptop CPU (or MPS / 1 small GPU) | n_layer=6, d_model=256, n_head=8, ctx=256, TinyStories | ~10M | ~10-20M (~1 short epoch) | ~30-90 min CPU; ~5-15 min MPS/GPU | loss ~1.2-1.5 / ppl ~3.5-4.5 | not meaningful on CPU | `python train.py --data ./data --parallel ddp --ctx 256` |
| Single GPU | 1x A100/H100 (or 4090) | GPT-2 small: n_layer=12, d_model=768, n_head=12, ctx=1024 | 124M | 2-3B (FineWeb-Edu) | ~0.5-1 day on 1 A100 | loss ~3.0-3.3 / ppl ~20-28 | ~35-45% (bf16 + torch.compile) | `torchrun --nproc_per_node=1 train.py --parallel ddp` |
| 8-GPU node | 8x H100/A100 (NVLink) | n_layer=24, d_model=2048, n_head=16, ctx=1024-2048 | ~1.3B | ~25-40B | ~1-2 days | loss ~2.5-2.8 / ppl ~12-16 | ~40-50% (FSDP FULL_SHARD) | `torchrun --nproc_per_node=8 train.py --parallel fsdp` |
| Multi-node | 2-8 nodes x 8 GPU (16-64 GPUs, InfiniBand) | Llama-style 7B (RoPE/GQA/SwiGLU/RMSNorm), ctx=4096 | ~7B | 140B (Chinchilla) to 1T+ | days to weeks | loss ~1.9-2.2 / ppl ~7-9 | ~40-50% (HYBRID_SHARD, or FSDP+TP) | `torchrun --nnodes=N --node_rank=$RANK --nproc_per_node=8 --rdzv_endpoint=$HEAD:29500 train.py --parallel fsdp` |

All wall-clock, loss, and MFU figures are order-of-magnitude planning ballparks — they move with data quality, kernel/compile settings, and network. The one invariant: the SAME `train.py` and the SAME uint16 shard format carry across every row; only the config, the launch flags, and the sharding strategy change.

!!! tip "If you only do one thing"

    Run the laptop TinyStories path — Stages 1, 2, 3, 5, and 9 — end to end in an afternoon before you spend a single GPU-day on anything bigger. It exercises every piece of glue in this chapter (tokenizer, shard format, model, training loop, sampling) on a scale where a bug costs you minutes, not a canceled cluster reservation. Everything that goes wrong at the 8-GPU or multi-node scale has a smaller, faster-to-debug cousin that already went wrong on your laptop.

## Interview Corner, Key Takeaways & Further Reading

!!! interview "Interview Corner"

    **Q:** You have one 80GB GPU and a 124M-parameter model. Walk me through every stage from a text corpus to a sampled sentence, and tell me where each stage would break if you tried to scale it 100x (to a ~12B model on a full cluster).

    **A:** Tokenize once with a reused byte-level BPE (`tiktoken`'s `gpt2`, padded to vocab 50304 for tensor-core alignment) and write `uint16` `.bin` shards — that step is essentially scale-invariant; only the total corpus size and shard count grow. Build the `GPT` from the config table and train with AdamW (betas 0.9/0.95, decay on 2-D weights only) under a linear-warmup-cosine schedule, launched via a single `train.py` under `torchrun`. At 124M on one GPU, plain **DDP** is not even doing anything interesting — `world_size=1` skips the all-reduce entirely, and the whole ~16-bytes-per-parameter mixed-precision Adam state (about 2GB for 124M) fits trivially in 80GB with huge headroom for activations. At 100x the parameters (~12B), the same DDP call would try to replicate ~192GB of persistent state per GPU and OOM immediately — so the first break is the DDP-to-FSDP-`FULL_SHARD` transition, sharding params/grads/optimizer state across the group. If the cluster spans multiple nodes, the second break is FSDP's inter-node all-gather becoming the bottleneck on the slower fabric, forcing `FULL_SHARD` -> `HYBRID_SHARD` (shard within a fast-NVLink node, replicate the lighter gradient sync across the slow inter-node link). If a 12B model still doesn't fit within one node's aggregate memory even hybrid-sharded, the third break is data-parallelism alone becoming insufficient, forcing composition with tensor parallelism (shard the compute of every layer, not just its storage) and pipeline parallelism (split layers across nodes). Monitoring, evaluation, and sampling are unaffected by any of this — they read logs and load a gathered checkpoint identically at every scale, which is precisely the point of keeping the model-storage concern (parallelism strategy) separate from everything downstream of a trained checkpoint.

!!! key "Key Takeaways"

    - The format contract is the load-bearing detail: `prepare.py` writes `uint16` token IDs, `ShardedTokenLoader` reads `uint16` — vocab 50304 fits comfortably under 65,536, so this halves I/O versus the `int32` convention used in the production data pipeline (3.1). Writer and loader dtype must always match exactly.
    - One `train.py`, one `--parallel {ddp,fsdp}` flag, scales from a single CPU process to a multi-node cluster — the model, loss, and schedule code never change; only the launch command and the sharding strategy do.
    - "Use the least sharding that fits": DDP if the model and its Adam state fit on one GPU; climb ZeRO-1 -> ZeRO-2 -> ZeRO-3/FSDP `FULL_SHARD` only as far as memory forces you, then `HYBRID_SHARD` when the inter-node network becomes the bottleneck.
    - bf16 (not fp16) plus a global gradient-norm clip of 1.0 is the standing defense against the two dominant instability mechanisms: floating-point overflow and Adam's amplification of anomalous-batch gradients.
    - Target MFU of roughly 40-55% on GPU scales as your north-star efficiency number; below ~30% with idle-looking GPUs points at a dataloader, kernel, or communication bottleneck, not a model problem.
    - Compute held-out **perplexity before** reaching for a standard benchmark at small scale — benchmark scores near the random floor at 124M and below are the expected, correct outcome, not a sign anything is broken.
    - The laptop-scale run (TinyStories, ~10M params) exercises every piece of this chapter's glue in an afternoon; treat it as the cheapest possible integration test before spending a GPU-day on anything larger.

**Further reading**

- **Andrej Karpathy — nanoGPT and minGPT** (GitHub) — the minimal, readable reference implementations this chapter's `train.py`, `GPT`, and `generate()` deliberately follow in spirit.
- **Radford et al. — "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019** — the architecture and byte-level BPE tokenizer this chapter's model and vocabulary are built from.
- **Hoffmann et al. — "Training Compute-Optimal Large Language Models" (Chinchilla), 2022** — the tokens-vs-parameters tradeoff behind this chapter's per-scale token budgets.
- **Penedo et al. — "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale," 2024** — the corpus (and its `sample-10BT` subset) used for every GPU-scale run in this chapter.
- **EleutherAI — lm-evaluation-harness** (GitHub) — the benchmark harness behind Stage 8's `lm_eval` command.

## End-to-End Checklist

- [ ] Environment installed (`torch`, `numpy`, `tiktoken`, `datasets`, `lm-eval`); seeds fixed (`torch.manual_seed(0)`, loader seed `1234`).
- [ ] `prepare.py` run to completion; `train_*.bin` and `val_*.bin` exist under `./data`, and their dtype (`uint16`) matches `ShardedTokenLoader`'s `np.memmap` dtype.
- [ ] `GPTConfig` chosen from the four-scale table; vocab fixed at 50304 (or your custom tokenizer's padded vocab).
- [ ] `torch.manual_seed(0)` set before model construction so every rank initializes identically.
- [ ] `train.py` launched via the correct `torchrun` command for your scale, `--parallel` set to `ddp` (fits on one GPU) or `fsdp` (does not).
- [ ] First logged loss lands near $\ln(\text{vocab\_size})$; if not, stop and debug the data pipeline before training further.
- [ ] Live monitoring shows loss falling, pre-clip grad norm steady, zero NaNs, and (on GPU) MFU in a healthy range.
- [ ] `eval_ppl.py` run against the held-out `val_*.bin` shard; perplexity in the right order of magnitude for your scale.
- [ ] (Optional, larger scales) `lm_eval` run on a standard task with honest expectations about small-model scores.
- [ ] `sample.py` run against the latest checkpoint; output read by a human, not just its loss number.
