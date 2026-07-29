#!/usr/bin/env python3
"""Full-run mid-training (Ch. 14.8): resume ckpt_stable.pt and run the WSD decay
across anneal @2048 -> long-context @8192 -> capability injection @8192.

    python capstone/scripts/midtrain.py \
        --stable artifacts/ckpt_stable.pt --out artifacts/ --device cuda
"""
import argparse
from dataclasses import asdict

from stacklm.config import StackConfig
from stacklm.model import Stack100M
from stacklm.optim import build_optimizers
from stacklm.train.loop import load_checkpoint, save_checkpoint     # Ch. 14.7
from stacklm.mid import SubPhase, run_mid_training
from stacklm.mid.mixture import (build_mixture_loader, ANNEAL_MIX,
                                 LONGCTX_MIX, CAPABILITY_MIX)

# 32 x 2048 x 8 accumulation steps = 524,288 tokens/step, identical to the
# pretraining loop's effective batch (Ch. 14.6/14.7). A POWER OF TWO on purpose:
# it divides exactly by every (micro_bs, seq_len) pair we run, so the realized
# batch equals the nominal one at 2048 and at 8192 alike.
GLOBAL_BATCH_TOKENS = 524_288
MICRO_BATCH_TOKENS = 65_536      # 32 x 2048 at pretrain length; 8 x 8192 when long
MUON_PEAK_LR = 6e-3              # Muon group's stable-phase peak (Ch. 14.6)
ADAMW_PEAK_LR = 3e-3             # AdamW group's peak = muon_lr / 2 (Ch. 14.6)

# The three moves of mid-training, in order. Token budgets are illustrative
# (~2B total = ~10% of the 20B pretrain budget); tune per Ch. 14.5.
PHASES = [
    SubPhase("anneal", tokens=1_200_000_000, seq_len=2048, mix=ANNEAL_MIX),
    SubPhase("longctx", tokens=600_000_000, seq_len=8192, extend_now=True,
             mix=LONGCTX_MIX),
    SubPhase("capability", tokens=200_000_000, seq_len=8192, mix=CAPABILITY_MIX),
]


def main(stable_ckpt: str, out_dir: str, device: str):
    # ---- 1. Resume the *pre-decay* model AND optimizer state ------------------
    model = Stack100M(StackConfig()).to(device)
    optimizers = list(build_optimizers(model, muon_lr=MUON_PEAK_LR,
                                       adamw_lr=ADAMW_PEAK_LR))   # (muon, adamw)
    step, extra = load_checkpoint(stable_ckpt, model, optimizers,
                                  map_location=device)
    # `extra` is the payload Ch. 14.7 stores alongside the tensors: tokens_seen,
    # the PackedDataset cursor, and the training config. Trust it over this file.
    muon_peak = extra.get("muon_lr", MUON_PEAK_LR)
    adamw_peak = extra.get("adamw_lr", ADAMW_PEAK_LR)
    print(f"resumed {stable_ckpt} @ global step {step} "
          f"({extra.get('tokens_seen', 0)/1e9:.1f}B tokens, LR still at peak); "
          f"peaks muon={muon_peak} adamw={adamw_peak}")

    # ---- 2. One loader factory per sub-phase, from that sub-phase's mixture ---
    def loader_fn(sub, micro_bs):
        return build_mixture_loader(sub.mix, sub.seq_len, micro_bs,
                                    root="data/mid", seed=1234 + len(sub.name))

    def checkpoint_fn(name, model, opts, mid_step):
        save_checkpoint(f"{out_dir}/ckpt_mid_{name}.pt", model, opts,
                        step=step + mid_step,
                        extra={**extra, "phase": name,
                               "cfg": asdict(model.cfg),   # records the NEW rope geometry
                               "mid_step": mid_step})
        print(f"[{name}] done -> ckpt_mid_{name}.pt")

    # ---- 3. Walk the three sub-phases under one continuous decay --------------
    result = run_mid_training(
        model, PHASES, loader_fn, device=device,
        global_batch_tokens=GLOBAL_BATCH_TOKENS,
        micro_batch_tokens=MICRO_BATCH_TOKENS,
        muon_lr=muon_peak, adamw_lr=adamw_peak,
        optimizers=optimizers,          # CONTINUE Muon momentum + AdamW moments
        log_every=50, checkpoint_fn=checkpoint_fn)

    save_checkpoint(f"{out_dir}/ckpt_mid_final.pt", model, optimizers,
                    step=step + result["mid_steps"],
                    extra={**extra, "cfg": asdict(model.cfg)})
    print(f"mid-training complete: {result['mid_steps']} steps "
          f"-> ckpt_mid_final.pt (ready for SFT in Ch. 14.9)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stable", default="artifacts/ckpt_stable.pt")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.stable, a.out, a.device)
