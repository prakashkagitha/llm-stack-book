#!/usr/bin/env python3
"""LIGHT REAL RUN (Part XIV evidence): actually train the mini scaling-law ladder + a short
Stack-100M pretrain on real data, fit the law, and report predicted-vs-actual loss + a real
generation. Bounded so it finishes in ~1 hour on a single H100; every number it prints is
MEASURED, not documented.

Good-neighbor: uses whatever GPU(s) CUDA_VISIBLE_DEVICES pins (launcher picks 2 free ones).
Env knobs (all optional): VOCAB, SEQ_LEN, TOK_DOCS, LADDER_TOKENS, TARGET_TOKENS, MB.
Writes capstone/experiments/results/{metrics.json, ladder.json, target_curve.json, sample.txt}.
"""
import os, sys, json, time, math, random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # import stacklm
import dataclasses
from stacklm.config import StackConfig
from stacklm.model import Stack100M
from stacklm.tokenizer.bpe import StackTokenizer
from stacklm.data import build_shards, PackedMemmapDataset
from stacklm.train.loop import pretrain, evaluate, estimate_mfu
from stacklm.serve.generate import generate
from stacklm.scaling.ladder import LADDER, TARGET, LadderConfig

VOCAB      = int(os.environ.get("VOCAB", 8192))
SEQ_LEN    = int(os.environ.get("SEQ_LEN", 512))
TOK_DOCS   = int(os.environ.get("TOK_DOCS", 20000))     # docs to train the BPE on
LADDER_TOK = int(os.environ.get("LADDER_TOKENS", 25_000_000))   # tokens per ladder rung
TARGET_TOK = int(os.environ.get("TARGET_TOKENS", 120_000_000))  # tokens for the short 100M run
MB         = int(os.environ.get("MB", 24))              # micro-batch size
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def real_corpus(n_docs):
    """Real text if HF is reachable (TinyStories: small, real, a tiny model can actually learn
    it), else the package's offline synthetic sample so the run still completes."""
    try:
        if os.environ.get("STACKLM_OFFLINE") == "1":
            raise RuntimeError("STACKLM_OFFLINE=1")
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        docs, it = [], iter(ds)
        for _ in range(n_docs):
            docs.append({"text": next(it)["text"]})
        log(f"corpus: TinyStories (real), {len(docs)} docs")
        return docs, "TinyStories (roneneldan/TinyStories)"
    except Exception as e:
        log(f"corpus: HF unavailable ({str(e)[:60]}); gathering a REAL on-disk corpus")
        return ondisk_corpus(), "on-disk: this book's prose (146 chapters) + installed-library Python source"


def ondisk_corpus():
    """A real, fully-offline corpus: the book's own markdown prose (~1.17M words of real
    technical English) + a sample of the installed libraries' Python source (real code).
    Genuinely structured text, so bigger models measurably beat smaller ones (unlike the
    tiny synthetic sample). Each file/paragraph is one document."""
    import glob
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    docs = []
    # 1) the book's prose
    for f in glob.glob(os.path.join(root, "content", "**", "*.md"), recursive=True):
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        for para in t.split("\n\n"):
            if len(para.strip()) > 40:
                docs.append({"text": para.strip()})
    n_prose = len(docs)
    # 2) real Python source from installed packages (code text with real structure)
    mods = []
    for name in ("numpy", "torch", "transformers", "datasets", "sklearn", "trl", "peft", "einops"):
        try:
            mods.append(__import__(name))
        except Exception:
            pass
    py_roots = {os.path.dirname(os.path.dirname(m.__file__)) for m in mods}
    py_files = []
    for r in py_roots:
        py_files += glob.glob(os.path.join(r, "**", "*.py"), recursive=True)
    py_files = sorted(set(py_files))
    for f in py_files[:25000]:
        try:
            t = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if 200 < len(t) < 40000:
            docs.append({"text": t})
    random.shuffle(docs)
    log(f"  on-disk corpus: {n_prose} prose paragraphs + {len(docs)-n_prose} source files = {len(docs)} docs")
    return docs


def cfg_from_rung(r: LadderConfig, vocab, seq_len):
    return StackConfig(vocab_size=vocab, d_model=r.d_model, n_layers=r.n_layers,
                       n_heads=r.d_model // 64, n_kv_heads=r.n_kv_heads, head_dim=64,
                       intermediate=r.intermediate, max_seq_len=seq_len)


def nonembed_params(model, cfg):
    tot = model.num_params()
    emb = cfg.vocab_size * cfg.d_model  # tied embedding, counted once
    return tot - emb


def train_one(cfg, ds, eval_ds, tokens, tag):
    tok_per_step = MB * SEQ_LEN
    steps = max(50, tokens // tok_per_step)
    model = Stack100M(cfg)
    n = model.num_params(); ne = nonembed_params(model, cfg)
    log(f"  {tag}: {n/1e6:.1f}M params ({ne/1e6:.2f}M non-embed), {steps} steps, ~{steps*tok_per_step/1e6:.0f}M tok")
    t0 = time.time()
    hist = pretrain(model, ds, device=DEV, steps=steps, micro_batch_size=MB, grad_accum=1,
                    warmup_steps=max(10, steps // 20), total_steps=steps, decay_frac=0.15,
                    eval_dataset=eval_ds, log_every=max(1, steps // 10))
    dt = time.time() - t0
    losses = [float(x) for x in hist["loss_history"]]
    final = float(np.mean(losses[-10:])) if losses else float("nan")
    val = float(hist.get("val_loss", float("nan")))         # pretrain evaluated eval_ds for us
    tps = (steps * tok_per_step) / dt if dt > 0 else 0.0     # measured end-to-end throughput
    mfu = float(estimate_mfu(n, tps))                        # real MFU from measured tok/s
    log(f"  {tag}: final train loss {final:.3f} | val {val:.3f} | {dt:.0f}s | {tps:,.0f} tok/s | mfu {mfu*100:.1f}%")
    return dict(tag=tag, params=n, nonembed=ne, tokens=steps * tok_per_step, steps=steps,
                final_train_loss=final, val_loss=val, seconds=dt, tok_per_s=tps, mfu=mfu, losses=losses), model


def fit_law_N(points):
    """Fit L(N) = E + A / N^alpha across ladder rungs (D held ~fixed). Returns (E,A,alpha,fn)."""
    N = np.array([p["nonembed"] for p in points], float)
    L = np.array([p["val_loss"] for p in points], float)
    best = None
    for E in np.linspace(0.5, min(L) - 0.05, 60):
        y = np.log(np.clip(L - E, 1e-6, None)); x = np.log(N)
        a, b = np.polyfit(x, y, 1)            # y = a*x + b  => L-E = e^b * N^a, alpha=-a
        pred = E + np.exp(b) * N ** a
        sse = float(np.sum((pred - L) ** 2))
        if best is None or sse < best[0]:
            best = (sse, E, math.exp(b), -a)
    _, E, A, alpha = best
    return E, A, alpha, (lambda n: E + A / n ** alpha)


def main():
    log(f"device={DEV} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(all)')} "
        f"| gpu={torch.cuda.get_device_name(0) if DEV=='cuda' else 'cpu'}")
    torch.manual_seed(1337); random.seed(1337); np.random.seed(1337)

    docs, source = real_corpus(TOK_DOCS)
    log("training byte-level BPE...")
    tok = StackTokenizer()
    tok.train_from_iterable((d["text"] for d in docs[:TOK_DOCS]), vocab_size=VOCAB)
    log(f"tokenizer: vocab={tok.vocab_size}")

    # tokenize once, pack enough tokens for the biggest budget we need
    need = max(LADDER_TOK, TARGET_TOK) + 5_000_000
    log(f"tokenizing + packing ~{need/1e6:.0f}M tokens (seq_len={SEQ_LEN})...")
    packed, total = [], 0
    for d in docs:
        ids = tok.encode(d["text"]); packed.append({"ids": ids}); total += len(ids)
        if total >= need: break
    if total < need:  # repeat the corpus if the real sample was small
        i = 0
        while total < need:
            packed.append({"ids": packed[i % len(packed)]["ids"]}); total += len(packed[-1]["ids"]); i += 1
    shard_dir = os.path.join(OUT, "shards")
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)   # never mix shards across runs / seq_lens
    build_shards(packed, tok, shard_dir, seq_len=SEQ_LEN, tokens_per_shard=SEQ_LEN * 20000)
    ds = PackedMemmapDataset(shard_dir)
    # small held-out split: last 5% of sequences
    n = len(ds); split = int(n * 0.95)
    train_ds = torch.utils.data.Subset(ds, range(split))
    eval_ds = torch.utils.data.Subset(ds, range(split, n))
    log(f"packed {total/1e6:.0f}M tokens -> {n} seqs (train {split}, eval {n-split})")

    # ---- ladder ----
    log("=== SCALING LADDER ===")
    ladder = []
    for r in LADDER:
        cfg = cfg_from_rung(r, tok.vocab_size, SEQ_LEN)
        rec, _ = train_one(cfg, train_ds, eval_ds, LADDER_TOK, r.name)
        ladder.append(rec)
        json.dump(ladder, open(os.path.join(OUT, "ladder.json"), "w"), indent=1)

    E, A, alpha, law = fit_law_N(ladder)
    tgt_cfg = cfg_from_rung(TARGET, tok.vocab_size, SEQ_LEN)
    tgt_ne = nonembed_params(Stack100M(tgt_cfg), tgt_cfg)
    predicted = law(tgt_ne)
    log(f"fitted L(N) = {E:.3f} + {A:.3g}/N^{alpha:.3f}  ->  predicted val loss @ {tgt_ne/1e6:.1f}M non-embed = {predicted:.3f}")

    # ---- short target run ----
    log("=== SHORT Stack-100M PRETRAIN ===")
    tgt_rec, tgt_model = train_one(tgt_cfg, train_ds, eval_ds, TARGET_TOK, "Stack-100M(short)")
    actual = tgt_rec["val_loss"]
    json.dump(tgt_rec, open(os.path.join(OUT, "target_curve.json"), "w"), indent=1)

    # ---- real generation ----
    try:
        prompt = "Once upon a time" if "TinyStories" in source else "The attention mechanism"
        sample = generate(tgt_model, tok, prompt, max_new_tokens=60, temperature=0.7)
        sample = prompt + sample
    except Exception as e:
        sample = f"(generation failed: {str(e)[:80]})"
    open(os.path.join(OUT, "sample.txt"), "w").write(str(sample))
    log("SAMPLE:", str(sample)[:200])

    metrics = dict(
        source=source, vocab=tok.vocab_size, seq_len=SEQ_LEN, device=DEV,
        gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        ladder=[{k: r[k] for k in ("tag", "params", "nonembed", "tokens", "val_loss", "seconds", "mfu")} for r in ladder],
        law=dict(E=E, A=A, alpha=alpha),
        target=dict(nonembed=tgt_ne, params=tgt_rec["params"], tokens=tgt_rec["tokens"],
                    predicted_val_loss=predicted, actual_val_loss=actual,
                    abs_error=abs(predicted - actual), mfu=tgt_rec["mfu"], seconds=tgt_rec["seconds"]),
        sample=str(sample)[:400],
        total_seconds=sum(r["seconds"] for r in ladder) + tgt_rec["seconds"],
    )
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    log("=== DONE ===")
    log(f"predicted {predicted:.3f} vs actual {actual:.3f} (|err| {abs(predicted-actual):.3f}) | "
        f"total {metrics['total_seconds']/60:.1f} min")
    print("REAL_RUN_OK")


if __name__ == "__main__":
    main()
