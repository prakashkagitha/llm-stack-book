#!/usr/bin/env python3
"""End-to-end toy-scale smoke test for the Stack-100M capstone (`stacklm`).

Exercises the WHOLE pipeline on CPU at tiny scale, in order:
  toy config -> byte-BPE -> pack corpus -> build model (+assert REAL param count)
  -> pretrain -> mid-train -> SFT -> DPO -> GRPO -> ReAct agent (fake teacher)
  -> int8/int4 quantize + CPU generate -> "SMOKE OK".

Hermetic: no network, no API, CPU-only, deterministic. Run:
    python3 capstone/smoke_test.py
    python3 scripts/ci_sim_run.py capstone/smoke_test.py
"""
import dataclasses
import os
import random
import sys
import tempfile

import numpy as np
import torch

# Make `import stacklm` work whether run from repo root or from capstone/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stacklm.config import StackConfig, count_params, toy_config
from stacklm.tokenizer import StackTokenizer, SPECIAL_TOKENS
from stacklm.data import (
    STACK100M_MIX, synthetic_corpus, synthetic_text_sample,
    build_shards, PackedMemmapDataset,
)
from stacklm.model import Stack100M
from stacklm.scaling import fit_scaling_law, predicted_loss, compute_optimal_allocation
from stacklm.train import pretrain, evaluate
from stacklm.mid import mid_train, run_mid_training, SubPhase
from stacklm.post import (
    Turn, render_conversation, PackedSFTDataset, sft_train,
    dpo_train, grpo_train, shaped_reward,
)
from stacklm.agent import (
    Passage, ToolEnv, Task, distill, make_stub_teacher, run_agent,
    build_agent_example, END,
)
from stacklm.eval import (
    compute_perplexity, make_arithmetic_probe, eval_arithmetic, eval_retrieval_qa,
)
from stacklm.serve import quantize_stacklm, generate, generate_fn

DEVICE = "cpu"


def seed_all(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def banner(msg):
    print(f"\n=== {msg} ===")


def main():
    seed_all()
    torch.set_num_threads(2)

    # ------------------------------------------------------------------ #
    # 0. REAL-config param accounting (the number the reader reproduces). #
    # ------------------------------------------------------------------ #
    banner("param accounting (REAL Stack-100M config)")
    real_cfg = StackConfig()
    acct = count_params(real_cfg)
    real_model_params = Stack100M(real_cfg).num_params()  # builds on CPU; ~101M, then freed
    print(f"analytic total = {acct['total']:,}  ({acct['total']/1e6:.3f}M)")
    print(f"model .num_params() = {real_model_params:,}")
    assert acct["total"] == real_model_params, "param accounting must match the model"
    assert 100e6 < acct["total"] < 103e6, "Stack-100M must be ~101M params"

    # ------------------------------------------------------------------ #
    # 1. Train a from-scratch byte-level BPE on synthetic text.          #
    # ------------------------------------------------------------------ #
    banner("tokenizer: train byte-level BPE on synthetic corpus")
    text = synthetic_text_sample(n_docs_per_source=120)
    tok = StackTokenizer()
    tok.train(text, vocab_size=384, special_tokens=SPECIAL_TOKENS)
    print(f"vocab_size={tok.vocab_size}  merges={len(tok.merges)}  specials={len(SPECIAL_TOKENS)}")
    rt = "photosynthesis converts sunlight into energy"
    assert tok.decode(tok.encode(rt)) == rt, "tokenizer must round-trip"
    assert tok.encode("<|user|>", add_special_tokens=True) == [tok.id("<|user|>")]

    # Toy model config, with vocab matched to the trained tokenizer.
    cfg = dataclasses.replace(toy_config(), vocab_size=tok.vocab_size, max_seq_len=96)

    # ------------------------------------------------------------------ #
    # 2. Pack a synthetic corpus into uint16 memmap shards.              #
    # ------------------------------------------------------------------ #
    banner("data: pack synthetic corpus into uint16 shards")
    tmp = tempfile.mkdtemp(prefix="stacklm_smoke_")
    docs = []
    for entry in STACK100M_MIX:
        docs.extend(list(synthetic_corpus(entry, n_docs=20)))
    random.shuffle(docs)
    shard_dir = os.path.join(tmp, "shards")
    n_shards = build_shards(docs, tok, shard_dir, seq_len=cfg.max_seq_len,
                            tokens_per_shard=cfg.max_seq_len * 8)
    ds = PackedMemmapDataset(shard_dir)
    print(f"{len(docs)} docs -> {n_shards} shard(s) -> {len(ds)} packed sequences "
          f"of len {cfg.max_seq_len}")
    assert len(ds) >= 8, "need enough packed sequences to train a few steps"
    sample = ds[0]
    assert sample["input_ids"].shape[0] == cfg.max_seq_len - 1
    assert sample["seq_ids"].max() >= 0

    # ------------------------------------------------------------------ #
    # 3. Build the toy model.                                            #
    # ------------------------------------------------------------------ #
    banner("model: build toy Stack100M")
    model = Stack100M(cfg)
    print(f"toy params = {model.num_params():,}  "
          f"(d_model={cfg.d_model}, n_layers={cfg.n_layers}, GQA {cfg.n_heads}:{cfg.n_kv_heads})")

    # ------------------------------------------------------------------ #
    # 4. Mini scaling-law fit (numpy-only, no scipy).                    #
    # ------------------------------------------------------------------ #
    banner("scaling: fit L(N,D)=E+A/N^a+B/D^b from toy ladder points")
    Ns = np.array([4e6, 9e6, 19e6, 43e6])
    Ds = np.array([8e7, 1.8e8, 3.8e8, 8.6e8])
    Ls = 2.6 + 12.0 * Ns ** (-0.33) + 9.0 * Ds ** (-0.29)   # synthetic ground truth
    fit = fit_scaling_law(Ns, Ds, Ls)
    pred = predicted_loss(fit, 1.0e8, 2.0e10)
    opt = compute_optimal_allocation(1.2e19, fit)
    print(f"fit: E={fit['E']:.3f} alpha={fit['alpha']:.3f} beta={fit['beta']:.3f} "
          f"sse={fit['sse']:.2e}")
    print(f"predicted loss @ (100M, 20B tok) = {pred:.3f};  "
          f"compute-optimal @ 1.2e19 FLOPs: d={opt['d_model']} tpp={opt['tpp']:.1f}")
    assert fit["sse"] < 1e-2 and np.isfinite(pred)

    # Ladder / FLOP / sweep arithmetic must reproduce Ch. 14.5's tables exactly.
    from stacklm.scaling import LADDER, TARGET, flops_per_token, batch_tokens, build_runs
    assert LADDER[0].nonembed_params() == 3_932_160
    assert TARGET.nonembed_params() == 84_541_440
    assert TARGET.total_params() == 101_318_656
    fpt = flops_per_token(TARGET)
    assert abs(fpt["total"] - 7.9666e8) / 7.9666e8 < 1e-3          # not 6N = 5.07e8
    assert abs(fpt["total"] / fpt["blocks"] - 1.571) < 1e-3        # the 6ND error
    assert batch_tokens(2.0e10) == 2 ** 19                         # the plan's batch
    runs = build_runs()
    assert len(runs) == 17 and min(r["steps"] for r in runs) >= 2000
    print(f"ladder: target {TARGET.nonembed_params():,} non-embed / "
          f"{TARGET.total_params():,} total; {fpt['total']:.3e} FLOPs/token "
          f"({fpt['total'] / fpt['blocks']:.2f}x the 6ND rule); "
          f"{len(runs)} sweep runs, min steps {min(r['steps'] for r in runs)}")

    # ------------------------------------------------------------------ #
    # 5. Pretrain a few steps -> loss must be finite and generally drop. #
    # ------------------------------------------------------------------ #
    banner("pretrain: a few Muon+AdamW / WSD steps (CPU, fp32)")
    pre = pretrain(model, ds, device=DEVICE, steps=8, micro_batch_size=4,
                   grad_accum=2, peak_lr=3e-3, warmup_steps=2, total_steps=8,
                   qk_clip_every=4, eval_dataset=ds)
    hist = pre["loss_history"]
    print(f"pretrain loss: {hist[0]:.3f} -> {hist[-1]:.3f};  val_ppl={pre['val_ppl']:.2f}")
    assert all(np.isfinite(hist)), "pretrain losses must be finite"
    assert hist[-1] < hist[0] + 0.5, "pretrain loss should not blow up"

    # ------------------------------------------------------------------ #
    # 6. Mid-train: continued training + long-context RoPE rescale.      #
    # ------------------------------------------------------------------ #
    banner("mid-train: anneal + RoPE theta rescale for long context")
    orig_seq_len = cfg.max_seq_len
    mid = mid_train(model, ds, device=DEVICE, steps=4, micro_batch_size=4,
                    grad_accum=2, peak_lr=1.5e-3, extend_to=orig_seq_len * 2)
    assert all(np.isfinite(mid["loss_history"]))
    assert model.cfg.max_seq_len == orig_seq_len * 2, "context should be extended"

    # The three-sub-phase driver: broad -> long -> narrow under ONE decay leg.
    theta_before = model.cfg.rope_theta
    phases = [
        SubPhase("anneal", tokens=0, seq_len=orig_seq_len * 2, steps=2),
        SubPhase("longctx", tokens=0, seq_len=orig_seq_len * 4, steps=2,
                 extend_now=True),
        SubPhase("capability", tokens=0, seq_len=orig_seq_len * 4, steps=2),
    ]

    def toy_loader(sub, micro_bs):
        loader = torch.utils.data.DataLoader(ds, batch_size=micro_bs,
                                             shuffle=True, drop_last=True)
        while True:
            for b in loader:
                yield b

    seen = []
    run = run_mid_training(model, phases, toy_loader, device=DEVICE,
                           global_batch_tokens=orig_seq_len * 8,
                           micro_batch_tokens=orig_seq_len * 4,
                           muon_lr=2e-3, adamw_lr=1e-3, log_every=1,
                           checkpoint_fn=lambda n, m, o, s: seen.append((n, s)))
    print(f"phases: {seen}; theta {theta_before:.0f} -> {model.cfg.rope_theta:.0f}")
    assert run["mid_steps"] == run["total_decay_steps"] == 6
    assert [n for n, _ in seen] == ["anneal", "longctx", "capability"]
    assert model.cfg.max_seq_len == orig_seq_len * 4, "sub-phase B extends context"
    assert model.cfg.rope_theta > theta_before, "RoPE base must be rescaled up"
    assert all(np.isfinite(run["loss_history"]))

    # ------------------------------------------------------------------ #
    # 7. SFT on the chat template (assistant-masked loss).               #
    # ------------------------------------------------------------------ #
    banner("SFT: chat template, loss masked to assistant tokens")
    conversations = []
    for a, b in [(2, 3), (4, 5), (7, 1), (6, 2), (9, 8), (3, 3)]:
        conversations.append([
            Turn("system", "You are Stack-100M, a concise assistant."),
            Turn("user", f"What is {a} plus {b}?"),
            Turn("assistant", f"{a} plus {b} is {a + b}. ####{a + b}"),
        ])
    # Assistant-masking invariant: supervised labels equal the input ids there.
    ids0, mask0 = render_conversation(conversations[0], tok)
    assert any(mask0) and not all(mask0), "some, not all, tokens are supervised"
    sft_ds = PackedSFTDataset(conversations, tok, block=cfg.max_seq_len)
    sft_loader = torch.utils.data.DataLoader(sft_ds, batch_size=2, shuffle=True)
    sft = sft_train(model, sft_loader, epochs=2, lr=1e-3, grad_accum=1, device=DEVICE)
    assert all(np.isfinite(sft["loss_history"])) and len(sft["loss_history"]) > 0

    # ------------------------------------------------------------------ #
    # 8. DPO: a few preference steps against a frozen reference.         #
    # ------------------------------------------------------------------ #
    banner("DPO: preference optimization on toy pairs")
    dpo_batches = [make_pref_batch(tok, cfg.max_seq_len)]
    policy, dpo_stats = dpo_train(model, dpo_batches, epochs=3, lr=1e-4, beta=0.1,
                                  grad_accum=1, device=DEVICE)
    assert all(np.isfinite(dpo_stats["loss_history"]))

    # ------------------------------------------------------------------ #
    # 9. GRPO / RLVR on a verifiable arithmetic task.                    #
    # ------------------------------------------------------------------ #
    banner("GRPO: narrow RLVR on integer arithmetic (exact-match reward)")
    # A shaped reward (small format bonus) gives the untrained toy model reward
    # variance across the group, so the GRPO surrogate produces a real gradient.
    policy, grpo_stats = grpo_train(model, tok, iterations=3, group_size=4,
                                    prompts_per_iter=3, inner_epochs=1, lr=1e-5,
                                    max_new=16, device=DEVICE, reward_fn=shaped_reward)
    print(f"grpo acc history: {grpo_stats['acc_history']}  "
          f"final loss {grpo_stats['loss_history'][-1]:.4f}")
    assert len(grpo_stats["acc_history"]) == 3
    assert all(np.isfinite(grpo_stats["loss_history"]))

    # ------------------------------------------------------------------ #
    # 10. ReAct agent: distill with the OFFLINE fake teacher, then run.  #
    # ------------------------------------------------------------------ #
    banner("agent: distill ReAct traces with a fake teacher, run the loop")
    corpus = [
        Passage("d1", "Toolville was founded in 1500 by early settlers."),
        Passage("d2", "Riverton is a small town known for its annual fair."),
        Passage("d3", "The library in Toolville holds many old maps."),
    ]
    env = ToolEnv(corpus)
    teacher = make_stub_teacher()
    tasks = [Task("Tell me about Toolville. Then double its founding year.", "3000")]
    traces = distill(tasks, teacher, env, samples_per_task=2)
    print(f"kept {len(traces)} successful distilled trajectory/ies")
    assert len(traces) >= 1, "fake teacher must solve the narrow task"
    # The distilled transcript formats into a loss-masked SFT example.
    xi, yi = build_agent_example(traces[0]["text"], tok, max_len=cfg.max_seq_len)
    assert xi.shape[0] == yi.shape[0] and (yi != -100).any()
    # Run the trained (toy) model through the runtime loop -- it just needs to run.
    ans, trace = run_agent(model, tok, tasks[0].question, env, max_steps=3, max_new=16)
    print(f"agent produced {len(trace)} trace step(s); answer repr: {ans[:40]!r}")
    assert isinstance(ans, str)

    # ------------------------------------------------------------------ #
    # 11. Eval probes.                                                   #
    # ------------------------------------------------------------------ #
    banner("eval: perplexity + arithmetic + retrieval-QA probes")
    ppl = compute_perplexity(model, ds, device=DEVICE, max_batches=4)
    arith = eval_arithmetic(model, tok, generate_fn, make_arithmetic_probe(n=6, seed=0))
    rqa = eval_retrieval_qa(model, tok, generate_fn, env.retriever,
                            [{"question": "When was Toolville founded?",
                              "gold_answer": "1500"}])
    print(f"perplexity={ppl['perplexity']:.2f}  arith_acc={arith['accuracy']:.2f}  "
          f"retrieval_em={rqa['exact_match']:.2f}")
    assert np.isfinite(ppl["perplexity"])

    # ------------------------------------------------------------------ #
    # 12. Quantize int8 & int4, then generate on CPU.                    #
    # ------------------------------------------------------------------ #
    banner("serve: int8/int4 round-to-nearest quantization + CPU generate")
    import copy
    for bits in (8, 4):
        qmodel = quantize_stacklm(copy.deepcopy(model), bits=bits, group_size=16)
        out = generate(qmodel, tok, "photosynthesis", max_new_tokens=8, temperature=0.0)
        print(f"int{bits} generate -> {out[:48]!r}")
        assert isinstance(out, str)

    print("\nSMOKE OK")
    return 0


def make_pref_batch(tok, block):
    """Build one DPO batch: chosen vs rejected, padded to equal length, with a
    float mask over the response tokens."""
    chosen_conv = [Turn("user", "What is 2 plus 2?"), Turn("assistant", "2 plus 2 is 4.")]
    rejected_conv = [Turn("user", "What is 2 plus 2?"), Turn("assistant", "It is 5, probably.")]

    def to_ids_mask(conv):
        ids, sup = render_conversation(conv, tok)
        return ids, [float(m) for m in sup]

    ch_ids, ch_m = to_ids_mask(chosen_conv)
    rj_ids, rj_m = to_ids_mask(rejected_conv)
    L = max(len(ch_ids), len(rj_ids))

    def pad(x, fill):
        return x + [fill] * (L - len(x))

    batch = {
        "chosen_ids": torch.tensor([pad(ch_ids, tok.pad_id)], dtype=torch.long),
        "chosen_mask": torch.tensor([pad(ch_m, 0.0)], dtype=torch.float),
        "rejected_ids": torch.tensor([pad(rj_ids, tok.pad_id)], dtype=torch.long),
        "rejected_mask": torch.tensor([pad(rj_m, 0.0)], dtype=torch.float),
    }
    return batch


if __name__ == "__main__":
    sys.exit(main())
