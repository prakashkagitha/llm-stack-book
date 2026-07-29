# 14.9 Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M

By the end of Chapter 14.8 we have a **Stack-100M** *base model*: a 101M-parameter deep-and-thin decoder that has seen ~20B tokens of FineWeb-Edu, Cosmopedia, code, and math, been annealed on a premium mix during its WSD decay phase, and had its context stretched to 8192. It is a competent *text continuer*. Hand it `"The capital of France is"` and it will say `" Paris"`. Hand it `"What is the capital of France?"` and it may well continue with *another question*, because on the pretraining distribution a question is most often followed by more questions. The base model has knowledge and it has fluency, but it has no idea that it is supposed to be a helpful assistant that answers, stops, and waits.

Post-training is the phase that installs that behavior. It is three stages, each cheaper and more surgical than the last, and each answering a different question:

- **SFT (supervised fine-tuning)** teaches *format and instinct*: "when you see a user turn, produce an assistant turn, then stop." This is where the chat template, the special tokens we reserved back in Chapter 14.3, and assistant-only loss masking come in.
- **DPO (Direct Preference Optimization)** teaches *taste*: given two candidate answers, prefer the better one. It does this from preference *pairs* with a single supervised-style loss — no reward model, no rollouts, no critic — which is the only reason preference optimization is affordable at our budget.
- **Narrow RLVR via GRPO** teaches *a verifiable skill*: on a task where correctness can be *checked by a program* (integer arithmetic, simple word problems), we let the model generate, grade its own samples with an exact-match reward, and reinforce what worked. This is where a 100M model can genuinely *improve at a task*, not just imitate.

The honest thesis of this chapter, stated up front so we can hold ourselves to it: **post-training changes what a 100M model *does*, not fundamentally what it *knows***. SFT and DPO reshape behavior that already exists latently in the base model; they cannot conjure reasoning that the base model has no substrate for. RLVR *can* sharpen a narrow, verifiable capability past the base model's zero-shot ceiling — but only when the task is narrow enough that the base model already succeeds often enough to give the reward signal something to grab. We will build all three, run them, and be brutally clear about where the ceiling is.

This chapter builds directly on the deeper book. If you have not read them, keep these open: [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html), [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html), [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html), and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html). Those chapters derive the mathematics from first principles; here we *apply* them to one concrete model, with runnable code that is consistent with the `stacklm` package we have been building.

## Where post-training fits, and what it costs at 100M

The cost asymmetry between the three stages is the whole reason the recipe looks the way it does. Let us anchor the numbers.

Pretraining Stack-100M costs on the order of 15–25 A100-hours (~USD 40–100). Against that, post-training is nearly free:

| Stage | Data volume | Compute | What it changes |
|---|---|---|---|
| SFT | ~10k–100k conversations, 1–3 epochs | ~0.5–2 A100-hr | Format, turn-taking, instruction-following instinct |
| DPO | ~5k–50k preference pairs, 1–2 epochs | ~0.5–2 A100-hr | Relative quality; reduces obvious failure modes |
| GRPO (narrow) | ~1k–10k prompts × G samples | ~2–6 A100-hr | One *verifiable* skill, sharpened past base zero-shot |

Two structural facts drive this. First, post-training touches a *tiny* number of tokens compared to the 20B of pretraining — a few tens of millions at most — so it is a rounding error on the compute bill. Second, the *effective learning rate* is small: we are nudging a converged model, not shaping it from scratch, so a handful of passes suffices and a large LR will simply destroy the pretrained knowledge (catastrophic forgetting). This is why we do full-parameter fine-tuning here rather than LoRA — at 100M the model is small enough that full fine-tuning fits comfortably on the A100, and there is no serving-multiplexing reason to keep adapters separate. (LoRA and QLoRA, covered in [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html), matter when the base is 7B+ or when you serve many task-specialized variants; neither applies to us.)

The classical alternative to DPO+GRPO is the full **PPO-RLHF pipeline** — train a reward model on preferences, then run PPO with a policy, a frozen reference, *and* a value-network critic (see [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) and [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)). At 100M this is technically possible but strategically wrong: the reward model would be *another* ~100M network to train and tune, the critic *another* one, and the whole online feedback loop is the most fragile machinery in ML. DPO deletes the reward model and the rollouts; GRPO deletes the critic. What remains is exactly what our budget can afford and our task actually needs.

{{fig:posttraining-ladder-100m}}

## SFT: chat template, packing, and assistant-only loss masking

### The Stack-100M chat template

Back in Chapter 14.3 we trained a byte-level BPE tokenizer with `vocab_size=32768` and *reserved* a block of special tokens for exactly this moment: `<|bos|> <|eos|> <|pad|>`, the chat-role tokens `<|system|> <|user|> <|assistant|> <|end|>`, and the tool tokens `<|tool_call|> <|tool_result|>`. Reserving them at tokenizer-training time (rather than "adding" them later) matters: each is a *single, atomic* token with its own embedding row, so the model can learn a crisp, unambiguous representation of "a turn just started/ended" instead of having to compose the boundary out of literal characters like `<`, `|`, `im`, `_`, `start`.

Our template is ChatML-like. A single-turn conversation renders to:

```text
<|bos|><|system|>You are Stack-100M, a concise, honest assistant.<|end|>
<|user|>What is 17 times 4?<|end|>
<|assistant|>17 times 4 is 68.<|end|><|eos|>
```

Each role token is immediately followed by that turn's content and terminated by `<|end|>`. At inference, the server appends `<|assistant|>` after the last user turn and decodes until the model emits `<|end|>` (or `<|eos|>`). The distinction between `<|end|>` (end of *one turn*) and `<|eos|>` (end of the *whole sequence*) lets multi-turn dialogues pack cleanly: turns are separated by `<|end|>`, conversations by `<|eos|>`.

Here is the template rendering, consistent with `stacklm.tokenizer.StackTokenizer` from Chapter 14.3:

```python
# stacklm/posttrain/chat.py
from dataclasses import dataclass
from stacklm.tokenizer import StackTokenizer  # trained in Ch. 14.3, vocab_size=32768

# The reserved special-token *strings*. StackTokenizer maps each to a single id.
SPECIAL = {
    "bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>",
    "system": "<|system|>", "user": "<|user|>",
    "assistant": "<|assistant|>", "end": "<|end|>",
}

DEFAULT_SYSTEM = "You are Stack-100M, a concise, honest assistant."

@dataclass
class Turn:
    role: str      # "system" | "user" | "assistant"
    content: str

def render_conversation(turns, tok: StackTokenizer, add_generation_prompt=False):
    """
    Render a list[Turn] into (token_ids, assistant_mask).

    assistant_mask[i] == 1  iff  token i is a *supervised* target: it belongs to
    an assistant turn's CONTENT or its closing <|end|>. Everything else — the
    system prompt, user turns, and the structural role tokens — is context the
    model conditions on but is NOT trained to produce.

    We build ids and mask token-by-token so the mask boundary is exact. Getting
    this boundary wrong is the single most common SFT bug (see the pitfall box).
    """
    ids, mask = [], []

    def emit(text, supervised):
        piece = tok.encode(text, add_special_tokens=False)
        ids.extend(piece)
        mask.extend([1 if supervised else 0] * len(piece))

    def emit_special(name, supervised):
        ids.append(tok.special_token_id(SPECIAL[name]))
        mask.append(1 if supervised else 0)

    emit_special("bos", supervised=False)
    for t in turns:
        emit_special(t.role, supervised=False)          # role marker: context
        if t.role == "assistant":
            emit(t.content, supervised=True)            # <-- the only tokens we learn
            emit_special("end", supervised=True)        # learn to STOP the turn
        else:
            emit(t.content, supervised=False)           # system/user: context only
            emit_special("end", supervised=False)
    if add_generation_prompt:                           # inference-time only
        emit_special("assistant", supervised=False)
    else:
        emit_special("eos", supervised=False)
    return ids, mask
```

Two design decisions deserve emphasis. First, **we supervise the closing `<|end|>` of assistant turns**. If we mask it, the model never receives gradient teaching it to *stop*, and at inference it will happily run past the end of its answer into hallucinated user turns — the "never shuts up" failure. Second, **the `<|assistant|>` role marker itself is masked** (supervised=False): the *harness* emits it to cue generation; the model should learn what comes *after* it, not to emit it spontaneously in the middle of a turn.

!!! note "Where the SFT data comes from at 100M"

    We are not writing 50k conversations by hand. The modern practice — and the one behind the small models we are emulating — is to assemble a compact, *high-quality* SFT mix from public instruction datasets and light synthetic generation. Concretely: **SmolTalk** (HuggingFace, 2024), the ~1M-conversation mix curated for **SmolLM2**, is a near-drop-in source; it blends **UltraChat**-style multi-turn dialogues (Ding et al., 2023), rewriting/summarization tasks, and a slice of math/code so the assistant is well-rounded. At 100M, *less is more*: a few tens of thousands of clean, on-format conversations beat a noisy million, because a small model spends its scarce capacity memorizing whatever regularities dominate the set. We deliberately seed the mix with a handful of arithmetic exemplars in the exact `####` answer format the RLVR stage will grade (below) — this is how the verifier later finds an answer to check. Filter aggressively for turns that *stop*, that respect the format, and that a 100M model can plausibly imitate; drop anything requiring long chains of reasoning it cannot represent.

### Assistant-only loss masking and why it matters

{{fig:sft-assistant-mask-stackml}}

The SFT objective is the ordinary causal-LM cross-entropy from [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html), with one change: we zero the loss on every non-assistant token. Formally, for a rendered sequence of tokens $t_1,\dots,t_L$ with supervision mask $m_i\in\{0,1\}$,

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_i m_i}\sum_{i=1}^{L-1} m_{i+1}\,\log \pi_\theta\!\left(t_{i+1}\mid t_{\le i}\right).
$$

The mask is on the *target* position: we supervise the prediction of token $t_{i+1}$ only when $t_{i+1}$ is an assistant token. In code we implement this by setting masked label positions to `-100`, the sentinel that `torch.nn.functional.cross_entropy` ignores.

Why not train on the whole sequence, prompt included? Two reasons, both real at our scale. (1) The instruction distribution in an SFT set is narrow and repetitive ("Summarize the following…", "Translate…"); training the model to *generate* those prompts wastes capacity and can degrade the diverse generation ability the base model earned. (2) The gradient signal we care about is "what a good assistant says," and diluting it with prompt tokens — which often outnumber response tokens — literally down-weights the thing we are trying to teach. The effect is modest for well-formatted data but the convention is universal, and at 100M, where capacity is precious, it is worth doing right. See [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) for the full argument.

### Packing masked conversations

Conversations vary wildly in length; padding each to `max_seq_len=2048` would waste most of the batch on `<|pad|>`. As in pretraining we **pack**: concatenate rendered conversations end-to-end and slice into fixed-length windows. The subtlety is that packing must carry the mask *and* prevent cross-conversation attention — a token in conversation B must not attend to conversation A sharing its window. We solve this exactly as in [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) and Chapter 14.2: a per-token `seq_id` that resets the document-aware attention mask and the RoPE position ids at each `<|eos|>`.

```python
# stacklm/posttrain/sft_data.py
import numpy as np, torch
from torch.utils.data import Dataset
from stacklm.posttrain.chat import render_conversation

IGNORE = -100  # F.cross_entropy(ignore_index=-100) skips these target positions

class PackedSFTDataset(Dataset):
    """
    Pack rendered conversations into fixed-length windows of `block`.
    Emits input_ids, labels (with IGNORE on masked positions), and seq_ids
    (document boundaries) so the model's document-aware attention (Ch. 14.4)
    blocks cross-conversation attention and resets position ids per conversation.
    """
    def __init__(self, conversations, tok, block=2048):
        self.block = block
        ids_buf, lbl_buf, seg_buf = [], [], []
        seg = 0
        for turns in conversations:
            ids, mask = render_conversation(turns, tok, add_generation_prompt=False)
            # Labels are next-token targets; here we store *aligned* labels and let
            # the training step do the shift. A masked (context) token -> IGNORE.
            labels = [tid if m == 1 else IGNORE for tid, m in zip(ids, mask)]
            ids_buf.extend(ids); lbl_buf.extend(labels); seg_buf.extend([seg]*len(ids))
            seg += 1
        # Truncate to a whole number of blocks (drop the ragged tail).
        n = (len(ids_buf) // block) * block
        self.ids = np.array(ids_buf[:n], dtype=np.int32).reshape(-1, block)
        self.lbl = np.array(lbl_buf[:n], dtype=np.int64).reshape(-1, block)
        self.seg = np.array(seg_buf[:n], dtype=np.int32).reshape(-1, block)

    def __len__(self):  return self.ids.shape[0]
    def __getitem__(self, i):
        return (torch.from_numpy(self.ids[i].astype(np.int64)),
                torch.from_numpy(self.lbl[i]),
                torch.from_numpy(self.seg[i].astype(np.int64)))
```

### The SFT training step

The training step is deliberately boring — that is the point. It reuses the same `Stack100M` model, bf16 autocast, gradient accumulation, and gradient clipping from the pretraining loop in Chapter 14.7. The only new element is the label shift with the ignore index and a *low* peak learning rate with a short warmup and decay to zero.

```python
# stacklm/posttrain/sft.py
import torch, torch.nn.functional as F
from stacklm.model import Stack100M, StackConfig   # Ch. 14.4
from stacklm.optim import build_optimizer          # Muon+AdamW hybrid, Ch. 14.6

def sft_train(model, loader, *, epochs=3, lr=2e-5, warmup=100,
              grad_accum=8, max_grad_norm=1.0, device="cuda"):
    """
    Full-parameter SFT of Stack-100M.

    lr=2e-5 is ~50-100x smaller than the pretraining peak LR: we are nudging a
    converged model. Too large an LR here erases pretrained knowledge (the
    'catastrophic forgetting' failure) — the model gets fluent-but-dumber.
    """
    model.train()
    # Reuse the pretraining optimizer factory but with the small SFT LR.
    opt = build_optimizer(model, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    total_steps = epochs * len(loader) // grad_accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warmup) * max(0.0, 1 - max(0, s - warmup) /
                                                   max(1, total_steps - warmup)))
    step = 0
    for ep in range(epochs):
        for micro, (ids, labels, seg) in enumerate(loader):
            ids, labels, seg = ids.to(device), labels.to(device), seg.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # model returns logits (B, T, V); seg drives document-aware masking
                logits = model(ids, seq_ids=seg)
                # Standard causal shift: predict token t+1 from tokens <= t.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()   # already IGNORE-masked
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,          # masked (context) tokens skipped
                ) / grad_accum
            loss.backward()
            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"ep{ep} step{step} loss {loss.item()*grad_accum:.3f}")
    return model
```

!!! warning "Common pitfall: the off-by-one mask that silently trains on the prompt"

    The mask must land on the *target* position after the causal shift. A classic bug is to mask `logits`/`labels` before shifting, or to mark the assistant role token `<|assistant|>` as supervised. Both leak prompt tokens into the loss or, worse, teach the model to *emit* the role marker mid-turn. Always assert the invariant: after shifting, every non-`IGNORE` label id equals an assistant-turn token id. A one-line `assert (shift_labels[shift_labels!=-100] == ids[:,1:][shift_labels!=-100]).all()` on a debug batch catches this instantly.

!!! note "Optimizer choice for post-training: why AdamW is the safe default"

    We reuse `build_optimizer` for continuity with the pretraining stack, but a word on the optimizer is warranted. **Muon** (Jordan et al., 2024) — which orthogonalizes the momentum update of 2D weight matrices — earned its place in *pretraining* Stack-100M (Ch. 14.6), where its geometry-aware step buys real speedups over many thousands of updates. Post-training is a different regime: a few hundred low-LR steps on a converged model. Here the standard, safest choice across the open literature (Tülu 3, Zephyr, and essentially every public SFT/DPO recipe) is **plain AdamW on all parameters**, precisely because we want *small, well-behaved* nudges rather than aggressive reshaping. `build_optimizer` accepts a flag to route every tensor to AdamW; at these learning rates the hybrid behaves benignly either way, but if you see instability in DPO or GRPO, switch to AdamW-only first before touching anything else.

Expected outcome: on the order of a 1.5–2.5 nats/token loss on assistant tokens (illustrative — the exact figure depends on your SFT set). More telling than the number is the *behavior*: after SFT, Stack-100M answers direct questions, respects the turn structure, and stops. It also becomes noticeably more brittle to inputs unlike its SFT distribution — the first hint of the scale ceiling we return to at the end.

## DPO: preference optimization without a reward model

SFT teaches the model to imitate *one* good answer. But "good" is relative: for the prompt "Explain gravity to a child," there are many acceptable completions and many bad ones, and imitation learning has no way to express "this answer is better than that one." Preference optimization does. We collect **pairs** $(x, y_w, y_l)$ — a prompt, a *chosen* (winner) response, and a *rejected* (loser) response — and train the model to raise the relative likelihood of $y_w$ over $y_l$. At our budget the pairs come from a public preference set such as **UltraFeedback** (Cui et al., 2023) — prompts with GPT-4-scored completions from many models — optionally augmented with pairs mined from Stack-100M's *own* SFT samples (sample two responses, keep the better-formatted / correct one as chosen).

### The DPO loss, briefly (full derivation in Ch. 5.7)

The derivation is done in full in [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html); here is the one-paragraph version so the code is grounded. The KL-regularized RLHF objective has a closed-form optimal policy $\pi^*(y\mid x)\propto \pi_{\text{ref}}(y\mid x)\exp(\tfrac1\beta r(x,y))$. Invert it to write the reward as an *implicit* function of the policy, $r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$, then plug that into the Bradley–Terry preference likelihood. The intractable partition term $\beta\log Z(x)$ appears in both the winner and loser reward and *cancels in the difference*. What survives is a clean logistic loss:

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\Big(\beta\big(\underbrace{\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{winner log-ratio}} - \underbrace{\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{loser log-ratio}}\big)\Big)\right].
$$

Read it as: push the winner's log-probability *up relative to the reference* and the loser's *down relative to the reference*, with $\beta$ (typically 0.1) controlling how far we let the policy drift from $\pi_{\text{ref}}$. The reference $\pi_{\text{ref}}$ is our frozen SFT model. Crucially there is **no reward model and no generation** — DPO consumes a static dataset of pairs, so at 100M it costs about the same as another epoch of SFT.

{{fig:dpo-relative-reshaping}}

### Implementation: per-sequence log-probs and the loss

The one quantity we need is the summed log-probability that a policy assigns to a response's tokens, given the prompt — the sequence log-likelihood, masked to the response.

```python
# stacklm/posttrain/dpo.py
import torch, torch.nn.functional as F

def sequence_logprob(model, input_ids, loss_mask, seg=None):
    """
    Sum of log p(token_t | token_<t) over the *response* tokens only.

    input_ids : (B, T)   full rendered sequence (prompt + response)
    loss_mask : (B, T)   1.0 on response tokens (incl. closing <|end|>), else 0.0
    Returns   : (B,)     per-sequence response log-likelihood.
    """
    logits = model(input_ids, seq_ids=seg)                 # (B, T, V)
    logits = logits[:, :-1, :]                             # predict t+1 from <=t
    targets = input_ids[:, 1:]                             # (B, T-1)
    mask = loss_mask[:, 1:]                                # align mask to targets
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)  # (B,T-1)
    return (tok_logp * mask).sum(dim=-1)                   # sum over response tokens

def dpo_loss(policy, ref, batch, beta=0.1, device="cuda"):
    """
    One DPO step's loss on a batch of preference pairs.

    batch supplies chosen/rejected token ids + response masks. `ref` is the frozen
    SFT model (no grad). We do FOUR forward passes: policy(chosen), policy(rejected),
    ref(chosen), ref(rejected). The ref passes are under no_grad and can be cached.
    """
    ch_ids  = batch["chosen_ids"].to(device);  ch_m = batch["chosen_mask"].to(device)
    rj_ids  = batch["rejected_ids"].to(device); rj_m = batch["rejected_mask"].to(device)

    # Policy log-probs (differentiable).
    pi_ch = sequence_logprob(policy, ch_ids, ch_m)
    pi_rj = sequence_logprob(policy, rj_ids, rj_m)
    # Reference log-probs (frozen). Compute once and cache across epochs in practice.
    with torch.no_grad():
        ref_ch = sequence_logprob(ref, ch_ids, ch_m)
        ref_rj = sequence_logprob(ref, rj_ids, rj_m)

    # Implicit-reward difference; the beta*logZ(x) terms cancelled analytically.
    chosen_logratio   = pi_ch - ref_ch          # log pi/pi_ref for the winner
    rejected_logratio = pi_rj - ref_rj          # log pi/pi_ref for the loser
    logits = beta * (chosen_logratio - rejected_logratio)   # the DPO "margin"

    loss = -F.logsigmoid(logits).mean()

    # Diagnostics that actually tell you if DPO is working:
    with torch.no_grad():
        acc = (logits > 0).float().mean()                    # implicit reward accuracy
        chosen_reward   = beta * chosen_logratio.mean()      # should trend UP
        rejected_reward = beta * rejected_logratio.mean()    # should trend DOWN
        margin = (chosen_reward - rejected_reward)
    return loss, {"acc": acc.item(), "chosen_r": chosen_reward.item(),
                  "rejected_r": rejected_reward.item(), "margin": margin.item()}
```

The training loop is the SFT loop with `dpo_loss` in place of the cross-entropy, a *frozen copy* of the SFT model as `ref`, and an even smaller LR (≈ 5e-7 to 1e-6) — DPO is sensitive and a large LR drives the loss down while *degrading* the model, the notorious DPO failure mode where both chosen and rejected log-probs *fall* (the loss only cares about the *difference*).

```python
# stacklm/posttrain/dpo.py  (continued)
import copy

def dpo_train(sft_model, loader, *, epochs=1, lr=5e-7, beta=0.1,
              grad_accum=8, max_grad_norm=1.0, device="cuda"):
    policy = sft_model                                   # trainable
    ref = copy.deepcopy(sft_model).eval()                # frozen reference
    for p in ref.parameters(): p.requires_grad_(False)
    from stacklm.optim import build_optimizer
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    step = 0
    for ep in range(epochs):
        for micro, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = dpo_loss(policy, ref, batch, beta=beta, device=device)
                loss = loss / grad_accum
            loss.backward()
            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    print(f"dpo step{step} loss {loss.item()*grad_accum:.3f} "
                          f"acc {stats['acc']:.2f} margin {stats['margin']:.3f}")
    return policy
```

!!! tip "Cache the reference log-probs — DPO's one free lunch"

    The reference model $\pi_{\text{ref}}$ is frozen, and the preference dataset is static, so $\log\pi_{\text{ref}}(y_w\mid x)$ and $\log\pi_{\text{ref}}(y_l\mid x)$ never change across epochs. Compute them *once* in a pre-pass, store two floats per pair, and delete the reference model from GPU memory entirely. This halves the forward passes per step (two policy forwards instead of four) and frees ~100M parameters of VRAM — at our scale it turns DPO into "SFT with a cleverer loss." The code above keeps `ref` resident for clarity, but the production `stacklm` path precomputes and caches. If you later switch to **reference-free** variants like **SimPO** or **CPO** (see [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)), the reference disappears from the loss altogether — attractive at 100M, at the cost of a length-normalization hyperparameter you must tune.

!!! example "Worked example: what one DPO step actually does to the logits"

    Take $\beta = 0.1$ and a single pair. Suppose the frozen reference assigns the winner and loser these response log-likelihoods: $\log\pi_{\text{ref}}(y_w)=-30.0$, $\log\pi_{\text{ref}}(y_l)=-28.0$ (the reference actually finds the *loser* slightly more likely — a case DPO should fix). Early in training the policy still matches the reference, so both log-ratios are ~0, the margin $\beta(0-0)=0$, and $\sigma(0)=0.5$: the loss is $-\log 0.5 = 0.693$ nats and implicit-reward accuracy is a coin flip.

    Now suppose after some steps the policy has moved to $\log\pi_\theta(y_w)=-27.0$ (winner up by 3 nats) and $\log\pi_\theta(y_l)=-29.0$ (loser down by 1 nat). The log-ratios are $+3.0$ and $-1.0$; the margin is $\beta(3.0-(-1.0)) = 0.1\times 4.0 = 0.40$. The loss drops to $-\log\sigma(0.40) = -\log(0.599) = 0.512$ nats and accuracy for this pair is 1 (margin $>0$).

    The lesson in the magnitudes: because $\beta=0.1$, even a *4-nat* separation in log-likelihood produces only a **0.4-logit** margin, i.e. a gentle $\sigma(0.4)\approx 0.60$ preference probability. DPO deliberately moves the policy in small steps. If you crank $\beta$ up to make the margin bigger, you also tighten the implicit KL leash and the policy barely moves at all; the sweet spot near $0.1$ is what keeps chosen-reward rising *without* both rewards collapsing.

The honest read on DPO at 100M: it reliably removes *obvious* failure modes present in the base/SFT model (rambling, ignoring the format, repeating the prompt) when the preference pairs target those failures. It does **not** install new reasoning. If your rejected/chosen pairs differ mainly in a capability the base model lacks — say, correct multi-step arithmetic — DPO will happily raise the log-probability of the "chosen" correct answer *relative to* the reference, but the model still cannot *produce* correct arithmetic on a new problem, because the log-ratio objective only reshapes the distribution over responses it can already generate. For a *capability* gain we need a signal tied to *correctness on new inputs*. That is RLVR.

## Narrow RLVR with GRPO: making RL actually work at 100M

Reinforcement learning with **verifiable rewards** (RLVR) replaces the learned, hackable reward model with a *program* that checks correctness. On a math problem, the checker parses the model's final answer and compares it to the ground truth: reward 1 if exactly right, 0 otherwise. There is nothing to reward-hack (short of the model finding the checker's bugs), the signal is dense in the sense of always-available, and — the key point for us — it rewards *being correct on inputs the model has not memorized*, which is exactly the capability signal DPO could not provide. See [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html) for the general recipe.

The catch, and the reason this section is titled "narrow," is the **cold-start problem**. RL can only reinforce behavior the model *sometimes* produces. If Stack-100M gets a task right 0% of the time, every sample earns reward 0, every advantage is 0, and the gradient is exactly zero — RL has nothing to climb. RLVR works when the base+SFT model already succeeds *occasionally* (say 10–40% of the time) so that the reward signal has variance to exploit. At 100M this restricts us to genuinely narrow, in-distribution tasks. We pick **integer arithmetic and one-step word problems** — a task Stack-100M, after math-heavy mid-training (Ch. 14.8), solves often enough to bootstrap.

### The task and the verifier

```python
# stacklm/posttrain/rlvr_task.py
import random, re

def make_arithmetic_prompt(rng, max_val=99):
    """Generate a simple integer-arithmetic problem with a known answer."""
    a, b = rng.randint(2, max_val), rng.randint(2, max_val)
    op = rng.choice(["+", "-", "*"])
    ans = {"+": a + b, "-": a - b, "*": a * b}[op]
    question = f"Compute {a} {op} {b}. Give the final integer after '####'."
    return question, ans

_FINAL = re.compile(r"####\s*(-?\d+)")

def exact_match_reward(completion_text, gold_answer):
    """
    Verifiable reward: 1.0 iff the integer after the '####' marker equals gold.
    We reward the ANSWER, not the reasoning — the model may show work or not.
    Returns (reward, parsed) so we can log parse-failure rate separately.
    """
    m = _FINAL.search(completion_text)
    if m is None:
        return 0.0, None                      # no answer in the required format
    try:
        pred = int(m.group(1))
    except ValueError:
        return 0.0, None
    return (1.0 if pred == gold_answer else 0.0), pred
```

Format-following (`####` marker) is itself a behavior we bootstrapped in SFT — a handful of arithmetic exemplars in the SFT set teach the model to *emit* the marker, so the verifier can find an answer to grade. Without that, parse-failure rate is ~100% and RL never starts. This SFT→RLVR ordering is not optional; it is the recipe. (The `####` convention is borrowed straight from **GSM8K** (Cobbe et al., 2021), whose answers are delimited exactly this way — a small nod to keeping our task in a well-trodden format.)

### GRPO advantage and loss (full derivation in Ch. 5.8)

**GRPO** (Group Relative Policy Optimization; Shao et al., *DeepSeekMath*, 2024) is the critic-free RL algorithm behind the open reasoning-model wave (it is the workhorse of **DeepSeek-R1**, 2025). Its one idea: to get a baseline for the advantage without training a value network, sample a *group* of $G$ responses to the same prompt and use the group's own statistics. For prompt $q$ with sampled rewards $R_1,\dots,R_G$, the advantage of response $i$ is the standardized reward,

$$
\hat{A}_i = \frac{R_i - \operatorname{mean}(R_1,\dots,R_G)}{\operatorname{std}(R_1,\dots,R_G) + \varepsilon},
$$

and — because RLVR gives every token in a response the same terminal reward — this scalar $\hat A_i$ is broadcast to *every token* of response $i$. GRPO then optimizes the PPO-style clipped surrogate so we can safely take a few gradient steps on the same batch of rollouts, plus an optional KL penalty to a frozen reference:

$$
\mathcal{L}_{\text{GRPO}} = -\,\mathbb{E}\!\left[\frac{1}{\sum_i |o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\!\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_i\big)\right] + \beta_{\text{KL}}\,\mathbb{D}_{\text{KL}}\!\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big),
$$

where $\rho_{i,t} = \dfrac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}$ is the per-token importance ratio between the current policy and the policy that generated the rollouts. When $\hat A_i>0$ (this sample beat its group) the objective pushes $\rho$ up (make these tokens more likely); when $\hat A_i<0$ it pushes them down; the clip prevents any single update from moving too far. Notice a group where *all* rewards are equal has zero std and thus zero advantage — those prompts contribute no gradient, which is both a feature (no noise) and the cold-start trap (too-hard or too-easy prompts are wasted).

{{fig:grpo-group-advantage-arith}}

### A minimal GRPO loop

Here is a complete, runnable GRPO loop for the arithmetic task. It is intentionally minimal — single GPU, synchronous generate-then-train, no distributed rollout engine — which is exactly right for a 100M model where generation is cheap. (The production version of this loop, with vLLM rollouts and weight sync, is the subject of Part VI; see [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).)

```python
# stacklm/posttrain/grpo.py
import torch, torch.nn.functional as F, copy
from stacklm.posttrain.chat import render_conversation, Turn, SPECIAL
from stacklm.posttrain.rlvr_task import make_arithmetic_prompt, exact_match_reward

@torch.no_grad()
def sample_group(model, tok, prompt_ids, G, max_new=64, temperature=1.0, device="cuda"):
    """
    Sample G completions for ONE prompt via temperature sampling.
    Returns (seqs, gen_masks): each seq is prompt+completion; gen_mask marks the
    generated (completion) tokens — the only ones we compute the loss on.
    Batched over the group: replicate the prompt G times and decode together.
    """
    model.eval()
    end_id = tok.special_token_id(SPECIAL["end"])
    eos_id = tok.special_token_id(SPECIAL["eos"])
    x = prompt_ids.to(device).unsqueeze(0).repeat(G, 1)     # (G, Tp)
    Tp = x.size(1)
    done = torch.zeros(G, dtype=torch.bool, device=device)
    for _ in range(max_new):
        logits = model(x)[:, -1, :]                          # (G, V)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)                    # (G, 1)
        nxt[done] = eos_id                                   # pad finished rows
        x = torch.cat([x, nxt], dim=1)
        done |= (nxt.squeeze(1) == end_id) | (nxt.squeeze(1) == eos_id)
        if done.all(): break
    gen_mask = torch.zeros_like(x, dtype=torch.float)
    gen_mask[:, Tp:] = 1.0                                    # completion tokens
    # zero the mask past each row's first stop token so padding earns no loss
    for i in range(G):
        row = x[i, Tp:]
        stop = ((row == end_id) | (row == eos_id)).nonzero()
        if len(stop): gen_mask[i, Tp + stop[0].item() + 1:] = 0.0
    return x, gen_mask

def token_logprobs(model, seqs, seg=None):
    """Per-token log-prob of the realized next token: (B, T-1)."""
    logits = model(seqs, seq_ids=seg)[:, :-1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, 2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)

def grpo_train(sft_model, tok, *, iterations=200, group_size=8, prompts_per_iter=16,
               inner_epochs=2, lr=1e-6, clip_eps=0.2, kl_beta=0.02,
               temperature=1.0, device="cuda", seed=0):
    """
    Minimal single-GPU GRPO on integer arithmetic with exact-match reward.
    Each iteration: (1) sample rollouts with the *current* policy (theta_old),
    (2) grade them with the verifier, (3) compute group-relative advantages,
    (4) take a few clipped-surrogate gradient steps.
    """
    import random
    rng = random.Random(seed)
    policy = sft_model.to(device)
    ref = copy.deepcopy(policy).eval()                       # frozen KL anchor
    for p in ref.parameters(): p.requires_grad_(False)
    from stacklm.optim import build_optimizer
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    for it in range(iterations):
        # ---- 1 & 2: rollout + reward, accumulated across several prompts ----
        batch_seqs, batch_gmask, batch_adv, batch_oldlp = [], [], [], []
        n_correct, n_total = 0, 0
        for _ in range(prompts_per_iter):
            q, gold = make_arithmetic_prompt(rng)
            turns = [Turn("user", q)]
            p_ids, _ = render_conversation(turns, tok, add_generation_prompt=True)
            p_ids = torch.tensor(p_ids, dtype=torch.long)
            seqs, gmask = sample_group(policy, tok, p_ids, group_size,
                                       temperature=temperature, device=device)
            # grade each completion
            rewards = torch.zeros(group_size, device=device)
            for i in range(group_size):
                text = tok.decode(seqs[i, p_ids.size(0):].tolist())
                r, _ = exact_match_reward(text, gold)
                rewards[i] = r
            n_correct += int(rewards.sum().item()); n_total += group_size
            # ---- 3: group-relative standardized advantage ----
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)   # (G,)
            # cache old (theta_old) token log-probs for the importance ratio
            with torch.no_grad():
                old_lp = token_logprobs(policy, seqs)                   # (G, T-1)
            batch_seqs.append(seqs); batch_gmask.append(gmask)
            batch_adv.append(adv);   batch_oldlp.append(old_lp)

        # ---- 4: clipped-surrogate updates (a few inner epochs on the rollouts) ----
        policy.train()
        for _ in range(inner_epochs):
            for seqs, gmask, adv, old_lp in zip(batch_seqs, batch_gmask,
                                                batch_adv, batch_oldlp):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    new_lp = token_logprobs(policy, seqs)              # (G, T-1)
                    m = gmask[:, 1:]                                    # align to targets
                    ratio = torch.exp(new_lp - old_lp)                 # rho_{i,t}
                    a = adv.unsqueeze(1)                               # (G,1) broadcast
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
                    surrogate = torch.min(unclipped, clipped)
                    # per-token KL(pi || ref), unbiased k3 estimator (Schulman)
                    with torch.no_grad():
                        ref_lp = token_logprobs(ref, seqs)
                    logr = ref_lp - new_lp
                    kl = torch.exp(logr) - logr - 1.0
                    per_tok = -(surrogate - kl_beta * kl)
                    # mask to generated tokens, average per token (Dr.GRPO-style)
                    loss = (per_tok * m).sum() / m.sum().clamp(min=1)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        acc = n_correct / max(1, n_total)
        if it % 10 == 0:
            print(f"grpo it{it}  train_acc {acc:.3f}  loss {loss.item():.4f}")
    return policy
```

A few implementation notes that matter for correctness. The importance ratio uses `old_lp`, the log-probs *under the policy that generated the rollouts*, cached before any update — this is what makes the inner-epoch updates valid off-policy corrections rather than a bug. On the very first inner step $\rho\equiv 1$ (new = old), so `torch.min` and the clip are no-ops and the update is a plain group-relative REINFORCE step; the clipping only bites once the policy has moved. The KL term uses the **k3 unbiased estimator** $e^{\log r}-\log r-1$ (always non-negative, low variance) rather than the naive $\log$-ratio; this is the standard choice discussed in [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html). We average the loss per token (dividing by the number of generated tokens) rather than per sequence, one of the "Dr. GRPO" corrections (Liu et al., 2025) that removes a length bias in the original formulation — see the discussion in [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html). Finally, generation here recomputes the full forward each step (no KV cache) — fine for a 100M model and 64-token completions, but the first thing you would replace with a real rollout engine at any larger scale.

!!! example "Worked example: a single GRPO group on '17 * 4'"

    Prompt: "Compute 17 * 4. Give the final integer after '####'." Gold = 68. We sample $G=8$ completions at temperature 1.0. Suppose the graded rewards come back as

    $$R = [1, 0, 1, 0, 0, 1, 0, 0]$$

    — three of eight correct, a 37.5% hit rate, right in the RLVR sweet spot. The group mean is $\bar R = 3/8 = 0.375$ and the *population* std is $\sqrt{0.375(1-0.375)} = \sqrt{0.234} = 0.484$. (The code uses `torch.std`, whose default Bessel correction divides by $G-1$ and gives $0.518$; with $\varepsilon=10^{-6}$ the difference is immaterial to the sign and near-magnitude of the advantage — the point is the *ranking*, not the third decimal.) Using the population std, the standardized advantages are:

    - correct samples: $\hat A = (1 - 0.375)/0.484 = +1.29$
    - incorrect samples: $\hat A = (0 - 0.375)/0.484 = -0.775$

    Every token of a *correct* completion gets advantage $+1.29$ (make these tokens more likely); every token of a *wrong* completion gets $-0.775$ (make them less likely). Because there are more wrong samples, each correct one is pushed up harder than each wrong one is pushed down — the group balances itself. Now the two degenerate cases: if all 8 were wrong, $\bar R=0$, std $=0$, every $\hat A = 0/\varepsilon = 0$ and the prompt contributes **no gradient** (too hard — the cold-start trap). If all 8 were right, same thing (nothing left to learn). Only groups with *mixed* outcomes teach anything — which is precisely why the base model must already be partially competent for RLVR to lift it.

### What we honestly observe

Run this loop on Stack-100M after math-focused mid-training and SFT, and the arithmetic accuracy climbs — on the order of from ~25–35% to ~60–80% on in-distribution problems over a couple hundred iterations, with the exact figure depending on operand range and the SFT warm-start (these are illustrative magnitudes, not a measured benchmark). That is a *real* capability gain from RL, on a 100M model, and it is the payoff the chapter promised. But watch the boundaries:

- **It generalizes narrowly.** Push the operand range past training (three-digit multiplication), and accuracy collapses — the model learned a better *distribution over the arithmetic it practiced*, not a general multiplication algorithm. This is the 100M ceiling, not a bug in GRPO.
- **Format drift and reward hacking creep in.** Even with a strict verifier, the model may learn to emit the `####` marker early and guess, or to pad reasoning that doesn't help. A KL leash to the SFT reference and a small format penalty keep it honest; see [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).
- **It can quietly forget.** Aggressive RL on one narrow task degrades general chat quality. Keep `kl_beta` non-trivial (≈ 0.02–0.05) and stop early; the goal is a *narrow tool*, and we accept the trade.

## What post-training does and does not buy at 100M

It is worth stating the scope plainly, because the temptation with a shiny chat model is to over-claim.

**What post-training buys.** A base model that continues text becomes a model that *takes turns, follows instructions, stays in format, and stops*. That is entirely a post-training gift and it is transformational for *usability*: Stack-100M goes from "autocomplete" to "assistant-shaped." DPO sands off obvious quality failures. And RLVR genuinely lifts a *narrow, verifiable* skill above the base model's zero-shot rate — the one place at this scale where RL adds capability rather than just reshaping behavior.

**What it does not buy.** Post-training cannot install knowledge or reasoning the base model has no substrate for. A 100M model has, very roughly, on the order of a few tens of millions of "facts" worth of capacity; no amount of SFT or DPO conjures broad world knowledge, reliable multi-step reasoning, or robust instruction-following on out-of-distribution prompts. The failure modes are characteristic: confident wrong answers, brittleness to phrasings unlike the SFT set, and collapse the moment a task leaves the narrow band RLVR was trained on. This is *expected*, not a defect of our recipe — it is the reason the capstone's north star (Ch. 14.10) is a *narrow, scaffolded, tool-using* agent rather than a general chatbot. Post-training is what makes the narrow tool *usable*; the scale is what keeps it narrow.

The strategic reading, and the reason this ordering (SFT → DPO → narrow GRPO) is the right one for our budget: each stage is cheaper than the last and each depends on the previous. SFT gives DPO and GRPO a model that already produces the right *format* (so DPO's pairs are comparable and GRPO's verifier can parse an answer). DPO gives a cleaner starting policy. GRPO then spends its narrow budget where verification is possible. Skip SFT and DPO has nothing coherent to prefer; skip both and GRPO's reward is 0 everywhere and RL never starts. The pipeline is a ladder, and every rung is load-bearing. This is exactly the shape of modern open post-training recipes — **Tülu 3** (Lambert et al., 2024) runs SFT → preference optimization → verifiable-reward RL in the same order, and we are simply running the 100M-scale, single-GPU edition of it.

!!! interview "Interview Corner"

    **Q:** You are asked to add a *verifiable* skill (say, arithmetic) to a small already-instruction-tuned model. Walk through why you would reach for GRPO/RLVR over just doing more SFT on correct solutions — and name the precondition that decides whether RLVR will work at all.

    **A:** SFT on correct solutions is *imitation*: it maximizes the likelihood of a fixed set of gold answers. It teaches the model to reproduce those specific solutions but gives no signal about the answers it generates itself, so it plateaus at "sound like the training solutions" and inherits their distribution. RLVR with GRPO optimizes a different objective — *be correct on new inputs*, graded by a program — so it can push the model's own sampling distribution toward whatever completions actually verify, including reasoning paths not in any SFT set. Concretely, GRPO samples a group of $G$ completions per prompt, grades each with the exact-match checker, standardizes the rewards within the group to get advantages ($\hat A_i=(R_i-\bar R)/\text{std}$), and reinforces high-advantage samples with a clipped PPO surrogate and a KL leash to the reference — no reward model, no critic. The decisive precondition is **non-degenerate reward variance within groups**: the model must already succeed *sometimes* (roughly 10–40%) so that groups contain both wins and losses. If it is right 0% (or 100%) of the time, every advantage is zero and the gradient vanishes — the cold-start problem. That is why RLVR is paired with an SFT warm-start and restricted to narrow tasks the base model can partially do, and why at 100M it only works on genuinely narrow, in-distribution skills.

!!! key "Key Takeaways"

    - Post-training on Stack-100M is three cheap stages — SFT, then DPO, then narrow GRPO — each under a couple of A100-hours, a rounding error against the ~USD 40–100 pretraining bill.
    - **SFT** installs the chat template using the reserved special tokens (`<|system|>/<|user|>/<|assistant|>/<|end|>`), packs conversations with document-aware masking, and supervises **only assistant tokens (including the closing `<|end|>`, so the model learns to stop)** at a small LR to avoid catastrophic forgetting; source the data from a compact, clean mix (SmolTalk-style) rather than a noisy million conversations.
    - **DPO** optimizes a single logistic loss on preference *pairs* against a frozen SFT reference — the reward model and rollouts of PPO-RLHF are gone, and the intractable partition term cancels — making preference optimization affordable at 100M; keep $\beta\approx0.1$ and a tiny LR, cache the reference log-probs, and watch that chosen-reward rises rather than both rewards falling.
    - **GRPO/RLVR** replaces the reward model with a *program* (exact-match on `####` answers) and the critic with a *group baseline*: standardize rewards within a group of $G$ samples, broadcast the advantage to every token, and update with a clipped surrogate plus a k3-estimated KL leash.
    - RLVR only climbs when groups have **mixed outcomes** — the base+SFT model must already succeed occasionally, which is why the task must be *narrow* and *in-distribution* at this scale (the cold-start trap: all-right or all-wrong groups give zero gradient).
    - The ladder is load-bearing and ordered: SFT gives format so DPO's pairs are comparable and GRPO's verifier can parse; skip a rung and the next one has nothing to work with — the same SFT → preference → verifiable-RL order as Tülu 3, scaled to one GPU.
    - Honest ceiling: post-training changes what the model *does* (turn-taking, format, taste, one narrow verifiable skill), not fundamentally what it *knows*; expect narrow generalization, confident errors off-distribution, and forgetting under aggressive RL — which is exactly why the capstone's endpoint is a scaffolded, tool-using *narrow* agent.

!!! sota "State of the Art & Resources (2026)"
    The SFT → DPO → RLVR ladder this chapter builds is now the industry-standard open post-training recipe; the open-source tooling that implements it at scale (TRL, open-instruct, verl) is exactly what you would reach for beyond a single GPU.

    **Foundational work**

    - [Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017)](https://arxiv.org/abs/1706.03741) — the pairwise-preference-to-reward idea that RLHF and, downstream, DPO both trace back to.
    - [Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN, 2021)](https://arxiv.org/abs/2109.01652) — established that instruction-formatted fine-tuning, not just scale, drives zero-shot instruction-following.

    **Recent advances (2023–2026)**

    - [Ding et al., *UltraChat: Enhancing Chat Language Models by Scaling High-quality Instructional Conversations* (2023)](https://arxiv.org/abs/2305.14233) — the multi-turn synthetic-dialogue dataset behind UltraChat-200k and much of the modern SFT-mix lineage.
    - [Meng, Xia, Chen, *SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024)](https://arxiv.org/abs/2405.14734) — drops DPO's reference model entirely by using length-normalized sequence log-probability as the implicit reward; the natural next step past the DPO code in this chapter.
    - [Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* (2025)](https://arxiv.org/abs/2502.02737) — documents the SmolTalk SFT mix referenced above and the full small-model pretrain-to-instruct pipeline it was built for.
    - [Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO, 2025)](https://arxiv.org/abs/2503.20783) — identifies and removes GRPO's length-bias, the correction this chapter's loop already applies (per-token rather than per-sequence normalization).

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — production `SFTTrainer`/`DPOTrainer`/`GRPOTrainer` implementations; the natural upgrade path from this chapter's from-scratch loops.
    - [allenai/open-instruct](https://github.com/allenai/open-instruct) — the actual Tülu 3 codebase: SFT, DPO, and RLVR stages chained exactly as this chapter narrows them to 100M.
    - [volcengine/verl](https://github.com/volcengine/verl) — a flexible, high-throughput RL-for-LLMs library (PPO, GRPO, RLOO, and more) for when rollouts need to scale past one GPU.
    - [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) — the ~1.1M-conversation SFT dataset this chapter's data-source note points to.

    **Go deeper**

    - [The RLHF Book](https://rlhfbook.com/) — Nathan Lambert's freely-readable, continuously updated book covering RLHF and post-training end to end, from reward modeling through direct alignment algorithms to RLVR.
    - [Tülu 3: The Next Era in Open Post-Training](https://allenai.org/blog/tulu-3-technical) — AI2's write-up of the five-stage SFT → DPO → RLVR pipeline this chapter's single-GPU recipe is a miniature of.

## Further reading

- Rafailov, Sharma, Mitchell, Manning, Ermon, Finn — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023). The DPO derivation and loss used here.
- Shao et al. — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024). Introduces GRPO.
- DeepSeek-AI — *DeepSeek-R1* (2025). RLVR/GRPO at scale for reasoning; the recipe this chapter narrows to 100M.
- Ouyang et al. — *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, 2022). The full PPO-RLHF pipeline we deliberately avoid at this budget.
- Ahmadian et al. — *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback* (2024). RLOO and the critic-free baseline family.
- Lambert et al. — *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* (2024). A modern, reproducible SFT → DPO → RLVR post-training pipeline.
- Cui et al. — *UltraFeedback: Boosting Language Models with High-quality Feedback* (2023). A standard open source of DPO preference pairs.
- Cobbe et al. — *Training Verifiers to Solve Math Word Problems* (GSM8K, 2021). Source of the `####` final-answer convention our verifier parses.
- Schulman et al. — *Proximal Policy Optimization Algorithms* (2017), and Schulman's note on KL estimators (the k3 estimator used in the GRPO loop).

## Exercises

**1.** In `render_conversation`, the closing `<|end|>` of an *assistant* turn is emitted with `supervised=True`, but the `<|assistant|>` role marker is emitted with `supervised=False`. Explain, in behavioral terms, what would go wrong at inference time if you flipped *each* of these two choices: (a) masking the assistant `<|end|>`, and (b) supervising the `<|assistant|>` marker.

??? note "Solution"
    Both choices are about *which* tokens receive gradient, and each controls a distinct behavior the chapter identifies.

    (a) **Masking the assistant `<|end|>`** removes the only gradient that teaches the model to *stop*. The closing `<|end|>` is the token that terminates the assistant turn; if the model never receives loss on producing it, it is never trained to emit it after finishing an answer. At inference the server decodes until it sees `<|end|>` (or `<|eos|>`), so a model that never learned to emit that token "runs past the end of its answer into hallucinated user turns" — the *never shuts up* failure. This is why the chapter supervises the closing `<|end|>`.

    (b) **Supervising the `<|assistant|>` marker** teaches the model to *emit the role marker itself*. But the harness is responsible for emitting `<|assistant|>` to cue generation; the model's job is to learn what comes *after* it. If we put gradient on the marker, the model learns to spontaneously produce `<|assistant|>` mid-turn, corrupting the turn structure the template exists to enforce. So the marker is context the model conditions on but is not trained to generate.

    In short: supervise the token that ends a turn (so the model stops), mask the token that starts the assistant turn (so the model does not impersonate the harness).

**2.** DPO with $\beta = 0.1$ on a single preference pair. The frozen reference assigns response log-likelihoods $\log\pi_{\text{ref}}(y_w) = -26.0$ and $\log\pi_{\text{ref}}(y_l) = -28.0$. After some training the policy assigns $\log\pi_\theta(y_w) = -25.0$ and $\log\pi_\theta(y_l) = -31.0$.
(a) Compute the winner and loser log-ratios, the DPO margin, and the loss $-\log\sigma(\text{margin})$.
(b) Compute the implicit `chosen_reward` and `rejected_reward` diagnostics ($\beta \times$ log-ratio). Is the implicit-reward accuracy 0 or 1 for this pair?
(c) Now suppose instead the policy had drifted to $\log\pi_\theta(y_w) = -35.0$ and $\log\pi_\theta(y_l) = -40.0$. Recompute the margin and both rewards. What failure mode does this illustrate?

??? note "Solution"
    (a) Log-ratios are $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ for each response:

    - winner: $-25.0 - (-26.0) = +1.0$
    - loser: $-31.0 - (-28.0) = -3.0$

    Margin $= \beta(\text{winner} - \text{loser}) = 0.1 \times (1.0 - (-3.0)) = 0.1 \times 4.0 = 0.40$.

    Loss $= -\log\sigma(0.40)$. With $\sigma(0.40) = 1/(1 + e^{-0.40}) = 1/(1 + 0.6703) = 0.5987$, the loss is $-\log(0.5987) = 0.513$ nats.

    (b) `chosen_reward` $= \beta \times (+1.0) = +0.10$; `rejected_reward` $= \beta \times (-3.0) = -0.30$. The chosen reward is up and the rejected reward is down — exactly the healthy trend. Since the margin $0.40 > 0$, the implicit-reward accuracy for this pair is $1$.

    (c) New log-ratios: winner $-35.0 - (-26.0) = -9.0$; loser $-40.0 - (-28.0) = -12.0$. Margin $= 0.1 \times (-9.0 - (-12.0)) = 0.1 \times 3.0 = 0.30 > 0$, so the *loss still decreases* and accuracy is still $1$. But now `chosen_reward` $= 0.1 \times (-9.0) = -0.90$ and `rejected_reward` $= 0.1 \times (-12.0) = -1.20$: **both rewards have fallen**. This is the notorious DPO failure mode — the logistic loss only cares about the *difference*, so it is perfectly happy to push the winner's absolute log-probability down as long as it pushes the loser's down faster. The model is degrading (it is making the good answer *less* likely) while the loss and accuracy look fine, which is why the chapter says to watch that chosen-reward *rises* rather than merely that the margin is positive, and to use a tiny LR.

**3.** A single GRPO group on one prompt, $G = 5$, exact-match reward. The graded rewards come back as $R = [1, 0, 0, 1, 0]$.
(a) Compute the group mean and the *population* standard deviation, then the standardized advantage assigned to a correct sample and to an incorrect sample. Verify the advantages sum to zero.
(b) The code uses `torch.std` (Bessel-corrected, divides by $G-1$). Recompute the std and the two advantages under that convention.
(c) If the same prompt had instead returned $R = [1,1,1,1,1]$, what advantage does every token receive, and how much does this prompt contribute to the gradient?

??? note "Solution"
    (a) Mean $\bar R = 2/5 = 0.40$. Population variance $= \bar R(1 - \bar R) = 0.40 \times 0.60 = 0.24$ (valid because the rewards are 0/1), so population std $= \sqrt{0.24} = 0.4899$.

    - correct sample: $\hat A = (1 - 0.40)/0.4899 = 0.60/0.4899 = +1.225$
    - incorrect sample: $\hat A = (0 - 0.40)/0.4899 = -0.40/0.4899 = -0.8165$

    Sum: $2(+1.225) + 3(-0.8165) = 2.449 - 2.449 = 0$. Advantages sum to zero, as standardization guarantees.

    (b) Bessel-corrected variance $= \frac{1}{G-1}\sum (R_i - \bar R)^2 = \frac{1}{4}\big[2(0.6)^2 + 3(0.4)^2\big] = \frac{1}{4}(0.72 + 0.48) = \frac{1.20}{4} = 0.30$, so std $= \sqrt{0.30} = 0.5477$.

    - correct: $\hat A = 0.60/0.5477 = +1.095$
    - incorrect: $\hat A = -0.40/0.5477 = -0.730$

    Same signs and ranking; only the magnitude shifts. As the chapter notes, with $\varepsilon = 10^{-6}$ the choice of convention is immaterial to the sign and near-magnitude — the point is the ranking, not the third decimal.

    (c) With $R = [1,1,1,1,1]$: $\bar R = 1$, std $= 0$. Every advantage is $(1 - 1)/(0 + \varepsilon) = 0$. Every token gets advantage $0$, so the surrogate is zero and this prompt contributes **no gradient**. This is the all-correct degenerate case (mirror image of all-wrong): with no reward variance in the group there is nothing to reinforce or suppress. Only *mixed* groups teach anything.

**4.** The chapter's "one free lunch" tip says to cache the reference log-probs once, since $\pi_{\text{ref}}$ is frozen and the preference set is static. Implement a `precompute_ref_logprobs` pass and a cache-consuming `dpo_loss_cached`, consistent with `stacklm/posttrain/dpo.py`. State the one constraint the data loader must satisfy for the cache to be valid, and say how many *policy* forward passes per step this saves.

??? note "Solution"
    ```python
    # stacklm/posttrain/dpo.py  (continued)
    import torch, torch.nn.functional as F

    @torch.no_grad()
    def precompute_ref_logprobs(ref, loader, device="cuda"):
        """
        One pass over the STATIC preference set: cache the frozen reference's
        per-sequence response log-likelihoods for chosen and rejected. After this
        the reference model can be deleted from GPU memory entirely.
        Returns two 1-D tensors, indexed by pair in loader-iteration order.
        """
        ref.eval()
        ref_ch_all, ref_rj_all = [], []
        for batch in loader:
            ch_ids = batch["chosen_ids"].to(device);  ch_m = batch["chosen_mask"].to(device)
            rj_ids = batch["rejected_ids"].to(device); rj_m = batch["rejected_mask"].to(device)
            ref_ch_all.append(sequence_logprob(ref, ch_ids, ch_m).cpu())
            ref_rj_all.append(sequence_logprob(ref, rj_ids, rj_m).cpu())
        return torch.cat(ref_ch_all), torch.cat(ref_rj_all)

    def dpo_loss_cached(policy, batch, ref_ch, ref_rj, beta=0.1, device="cuda"):
        """
        DPO loss using PRECOMPUTED reference log-probs. Only two forward passes
        (policy on chosen + rejected); the reference passes are gone.
        ref_ch, ref_rj : the cached slices for exactly this batch's pairs.
        """
        ch_ids = batch["chosen_ids"].to(device);  ch_m = batch["chosen_mask"].to(device)
        rj_ids = batch["rejected_ids"].to(device); rj_m = batch["rejected_mask"].to(device)

        pi_ch = sequence_logprob(policy, ch_ids, ch_m)          # differentiable
        pi_rj = sequence_logprob(policy, rj_ids, rj_m)
        ref_ch = ref_ch.to(device); ref_rj = ref_rj.to(device)  # cached, no grad

        chosen_logratio   = pi_ch - ref_ch
        rejected_logratio = pi_rj - ref_rj
        logits = beta * (chosen_logratio - rejected_logratio)
        return -F.logsigmoid(logits).mean()
    ```

    **Constraint:** the cache is indexed by pair position, so the loader must present pairs in a *fixed, reproducible order* — i.e. **no shuffling** (or, equivalently, key the cache by a stable pair id and look up by that id each step). If the loader shuffles between the precompute pass and training, `ref_ch[k]` no longer corresponds to the $k$-th pair the training loop sees, and every log-ratio is silently mismatched.

    **Savings:** the original `dpo_loss` does four forwards per step (policy chosen, policy rejected, ref chosen, ref rejected). Caching removes the two reference forwards, leaving **two policy forwards per step** — the reference passes are amortized into a single pre-pass — and frees ~100M parameters of VRAM once `ref` is deleted.

**5.** The cold-start trap, quantitatively. Model a group of $G$ i.i.d. samples where each succeeds independently with probability $p$. A group produces *zero gradient* exactly when it is degenerate — all correct or all wrong.
(a) Write the probability that a group is degenerate as a function of $p$ and $G$.
(b) Evaluate it for $G = 8$ at a "sweet spot" success rate $p = 0.30$ and at a cold-start rate $p = 0.05$.
(c) Interpret: what fraction of prompts is "wasted" in each regime, and why does this make the SFT warm-start non-optional?

??? note "Solution"
    (a) A group is all-wrong with probability $(1-p)^G$ and all-right with probability $p^G$; these are disjoint, so

    $$P_{\text{degenerate}}(p, G) = (1-p)^G + p^G.$$

    (b) For $G = 8$:

    - $p = 0.30$: $(0.70)^8 + (0.30)^8 = 0.05765 + 0.0000656 \approx 0.0577$, i.e. about **5.8%** of groups are degenerate.
    - $p = 0.05$: $(0.95)^8 + (0.05)^8 = 0.6634 + (\sim\!4\times10^{-11}) \approx 0.663$, i.e. about **66%** of groups are degenerate.

    (c) At the sweet-spot rate $p = 0.30$, only ~6% of prompts give zero gradient, so ~94% of the compute spent on rollouts actually produces a learning signal — RLVR has plenty to climb. At the cold-start rate $p = 0.05$, two-thirds of all groups are all-wrong (or, negligibly, all-right), so two-thirds of the rollouts are wasted and the effective learning signal is throttled to a trickle; as $p \to 0$ the wasted fraction $\to 1$ and the gradient vanishes entirely. This is why the SFT warm-start is non-optional: SFT (plus math-heavy mid-training and a few `####` format exemplars) lifts the base model's success rate from near-zero into the 10-40% band where groups are *mixed* often enough that the reward signal has variance to exploit. RL can only reinforce behavior the model already sometimes produces; SFT is what puts $p$ into the range where "sometimes" is frequent enough to bootstrap.

**6.** The chapter warns that "format drift and reward hacking creep in" and suggests a small format penalty on top of the exact-match reward. Implement a `shaped_reward(completion_text, gold_answer)` for the GRPO loop that (i) keeps exact-match correctness as the dominant term and (ii) mildly penalizes emitting the wrong number of `####` markers (zero, or more than one). Explain why the *magnitude* of the shaping term must stay small relative to the correctness reward, and why within-group standardization limits how much a constant format bonus can distort learning.

??? note "Solution"
    ```python
    # stacklm/posttrain/rlvr_task.py  (continued)
    import re
    from stacklm.posttrain.rlvr_task import exact_match_reward

    _MARKER = re.compile(r"####")

    def shaped_reward(completion_text, gold_answer, fmt_weight=0.1):
        """
        Exact-match correctness (0/1) plus a SMALL shaping term that rewards
        emitting exactly one well-formed answer marker and penalizes zero or many.
        Correctness stays dominant: fmt_weight (0.1) << the 1.0 correctness gap.
        """
        r_correct, pred = exact_match_reward(completion_text, gold_answer)
        n_markers = len(_MARKER.findall(completion_text))
        fmt = fmt_weight if n_markers == 1 else -fmt_weight
        return r_correct + fmt, pred
    ```

    Drop `shaped_reward` in for `exact_match_reward` where the loop grades completions:

    ```python
    r, _ = shaped_reward(text, gold)     # was exact_match_reward(text, gold)
    rewards[i] = r
    ```

    **Why the shaping magnitude must stay small.** GRPO reinforces whatever raises reward. If the format bonus is comparable to (or larger than) the correctness gap of $1.0$, then a completion that has the right format but the *wrong answer* can out-score, or tie, a differently-formatted *correct* one — the model can maximize reward by getting the format right and the arithmetic wrong. That is textbook reward hacking: optimizing the proxy (format) instead of the goal (correctness). Keeping $\text{fmt\_weight} = 0.1 \ll 1.0$ ensures correctness always dominates the ranking within a group, so the format term only breaks ties among equally-correct (or equally-wrong) samples.

    **Why standardization limits the damage.** The advantage is the reward *standardized within the group*, $\hat A_i = (R_i - \bar R)/(\text{std} + \varepsilon)$. Any component of the reward that is *constant across the group* — e.g. if all $G$ samples already emit exactly one marker, every sample gets the same $+0.1$ — shifts the mean by that same amount and *cancels* in $R_i - \bar R$, contributing nothing to the advantage. The format term therefore only produces gradient when samples in the group *differ* in their format (some emit one marker, some do not), which is exactly the drift we want to correct. Combined with the KL leash to the SFT reference ($\text{kl\_beta} \approx 0.02$-$0.05$), this keeps the shaping honest: it nudges the model back toward one-marker format without letting it trade correctness for format.
