# 14.9 Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M

By the end of Chapter 14.8 we have a **Stack-100M** *base model*: a 101M-parameter deep-and-thin decoder that has seen ~20B tokens of FineWeb-Edu, Cosmopedia, code, and math, been annealed on a premium mix during its WSD decay phase, and had its context stretched to 8192. It is a competent *text continuer*. Hand it `"The capital of France is"` and it will say `" Paris"`. Hand it `"What is the capital of France?"` and it may well continue with *another question*, because on the pretraining distribution a question is most often followed by more questions. The base model has knowledge and it has fluency, but it has no idea that it is supposed to be a helpful assistant that answers, stops, and waits.

Post-training is the phase that installs that behavior. It is three stages, each cheaper and more surgical than the last, and each answering a different question:

- **SFT (supervised fine-tuning)** teaches *format and instinct*: "when you see a user turn, produce an assistant turn, then stop." This is where the chat template, the special tokens we reserved back in Chapter 14.3, and assistant-only loss masking come in.
- **DPO (Direct Preference Optimization)** teaches *taste*: given two candidate answers, prefer the better one. It does this from preference *pairs* with a single supervised-style loss — no reward model, no rollouts, no critic — which is the only reason preference optimization is affordable at our budget.
- **Narrow RLVR via GRPO** teaches *a verifiable skill*: on a task where correctness can be *checked by a program* (integer arithmetic, simple word problems), we let the model generate, grade its own samples with an exact-match reward, and reinforce what worked. This is where a 100M model can genuinely *improve at a task*, not just imitate.

The honest thesis of this chapter, stated up front so we can hold ourselves to it: **post-training changes what a 100M model *does*, not fundamentally what it *knows***. SFT and DPO reshape behavior that already exists latently in the base model; they cannot conjure reasoning that the base model has no substrate for. RLVR *can* sharpen a narrow, verifiable capability past the base model's zero-shot ceiling — but only when the task is narrow enough that the base model already succeeds often enough to give the reward signal something to grab. We will build all three, run them, and be brutally clear about where the ceiling is.

This chapter builds directly on the deeper book. If you have not read them, keep these open: [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html), [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html), [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html), and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html). Those chapters derive the mathematics from first principles; here we *apply* them to one concrete model, with runnable code that is consistent with the `stacklm` package we have been building.

!!! note "Where this chapter's code lives"

    Every code block below is labelled with its file in the capstone package: `capstone/stacklm/post/chat.py`, `post/sft.py`, `post/dpo.py`, `post/grpo.py`. The shipped modules carry CI-scale defaults (tiny blocks, a handful of steps, CPU-safe) so the book's smoke test can run the whole pipeline hermetically; the blocks here show the *full-run* versions, with the flagship hyperparameters. Where a block adds something the shipped module does not yet have (the preference-data pipeline, DAPO-style dynamic sampling), the header says so — append it to the named file and it will import cleanly. Model calls follow the package contract: `Stack100M.forward` **always returns `(logits, loss)`**, so every call site unpacks two values.

## Where post-training fits, and what it costs at 100M

The cost asymmetry between the three stages is the whole reason the recipe looks the way it does. Let us anchor the numbers.

Pretraining Stack-100M costs on the order of 15–25 A100-hours (~USD 40–100). Against that, post-training is nearly free:

| Stage | Data volume | Compute | What it changes |
|---|---|---|---|
| SFT | ~10k–100k conversations, 1–3 epochs | ~0.5–2 A100-hr | Format, turn-taking, instruction-following instinct |
| DPO | ~5k–50k preference pairs, 1–2 epochs | ~0.5–2 A100-hr | Relative quality; reduces obvious failure modes |
| GRPO (narrow) | ~1k–10k prompts × G samples | ~2–6 A100-hr | One *verifiable* skill, sharpened past base zero-shot |

Two structural facts drive this. First, post-training touches a *tiny* number of tokens compared to the 20B of pretraining — a few tens of millions at most — so it is a rounding error on the compute bill. Second, the *effective learning rate* is small: we are nudging a converged model, not shaping it from scratch, so a handful of passes suffices and a large LR will simply destroy the pretrained knowledge (catastrophic forgetting). This is why we do full-parameter fine-tuning here rather than LoRA — at 100M the model is small enough that full fine-tuning fits comfortably on the A100, and there is no serving-multiplexing reason to keep adapters separate. (LoRA and QLoRA, covered in [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html), matter when the base is 7B+ or when you serve many task-specialized variants; neither applies to us.)

Note that only the GRPO row scales with *generation*, not just with data. SFT and DPO consume a static corpus; GRPO must sample $G$ completions per prompt with the model in the loop, which is why its cost is the largest of the three despite touching the fewest gradient steps. At 100M, generation is cheap enough to do naively inside the training process. At 1B+ this inverts — rollouts dominate, and you need a real inference engine (vLLM/SGLang) colocated with the trainer plus a weight-sync path, which is exactly the machinery that [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html) and [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html) exist to explain.

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
# capstone/stacklm/post/chat.py
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
        # add_special_tokens=False => a literal "<|end|>" inside user CONTENT is
        # encoded as ordinary bytes, never as the atomic control token. This is
        # what makes the template injection-proof (see the note below).
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

Three design decisions deserve emphasis. First, **we supervise the closing `<|end|>` of assistant turns**. If we mask it, the model never receives gradient teaching it to *stop*, and at inference it will happily run past the end of its answer into hallucinated user turns — the "never shuts up" failure. Second, **the `<|assistant|>` role marker itself is masked** (supervised=False): the *harness* emits it to cue generation; the model should learn what comes *after* it, not to emit it spontaneously in the middle of a turn. Third, **user content is encoded with `add_special_tokens=False`**, so a user who types the literal string `<|assistant|>` gets a sequence of ordinary byte-level BPE tokens, not the control token. This is the concrete reason to make role markers *reserved atomic ids* rather than literal text: forging a turn boundary becomes impossible at the tokenizer level, not merely discouraged by a regex.

!!! note "Where the SFT data comes from at 100M"

    We are not writing 50k conversations by hand. The modern practice — and the one behind the small models we are emulating — is to assemble a compact, *high-quality* SFT mix from public instruction datasets and light synthetic generation. Concretely: **SmolTalk** (HuggingFace, 2024), the ~1M-conversation mix curated for **SmolLM2**, is a near-drop-in source; it blends **UltraChat**-style multi-turn dialogues (Ding et al., 2023), rewriting/summarization tasks, and a slice of math/code so the assistant is well-rounded. (A successor mix, *SmolTalk2*, was released alongside SmolLM3 in 2025; check the Hub for the current revision.) At 100M, *less is more* — the **LIMA** finding (Zhou et al., 2023) that a thousand carefully-curated examples beat a noisy hundred thousand is *more* true at small scale, not less, because a small model spends its scarce capacity memorizing whatever regularities dominate the set.

    ```python
    # Turning a Hub dataset into the chapter's Turn lists. `datasets` is the
    # standard HF loader; it streams and caches to Arrow, so a 1M-row mix costs
    # no RAM. pip install datasets
    from datasets import load_dataset
    from stacklm.post.chat import Turn, DEFAULT_SYSTEM

    raw = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")

    def to_turns(row, max_chars=4000):
        msgs = row["messages"]                 # [{"role": ..., "content": ...}, ...]
        if not msgs or msgs[-1]["role"] != "assistant":
            return None                        # must END on an assistant turn
        if sum(len(m["content"]) for m in msgs) > max_chars:
            return None                        # a 100M model cannot imitate essays
        return ([Turn("system", DEFAULT_SYSTEM)]
                + [Turn(m["role"], m["content"]) for m in msgs])

    conversations = [t for t in map(to_turns, raw) if t is not None]
    ```

    We deliberately seed the mix with a few thousand arithmetic exemplars in the exact `####` answer format the RLVR stage will grade (below) — this is how the verifier later finds an answer to check, and it is what lifts the GRPO success rate out of the cold-start zone. Filter aggressively for turns that *stop*, that respect the format, and that a 100M model can plausibly imitate; drop anything requiring long chains of reasoning it cannot represent.

### Assistant-only loss masking and why it matters

{{fig:sft-assistant-mask-stackml}}

The SFT objective is the ordinary causal-LM cross-entropy from [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html), with one change: we zero the loss on every non-assistant token. Formally, for a rendered sequence of tokens $t_1,\dots,t_L$ with supervision mask $m_i\in\{0,1\}$,

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_i m_i}\sum_{i=1}^{L-1} m_{i+1}\,\log \pi_\theta\!\left(t_{i+1}\mid t_{\le i}\right).
$$

The mask is on the *target* position: we supervise the prediction of token $t_{i+1}$ only when $t_{i+1}$ is an assistant token. In code we implement this by setting masked label positions to `-100`, the sentinel that `torch.nn.functional.cross_entropy` ignores.

Why not train on the whole sequence, prompt included? Two reasons, both real at our scale. (1) The instruction distribution in an SFT set is narrow and repetitive ("Summarize the following…", "Translate…"); training the model to *generate* those prompts wastes capacity and can degrade the diverse generation ability the base model earned. (2) The gradient signal we care about is "what a good assistant says," and diluting it with prompt tokens — which often outnumber response tokens — literally down-weights the thing we are trying to teach. The effect is modest for well-formatted data but the convention is universal, and at 100M, where capacity is precious, it is worth doing right. See [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) for the full argument.

The denominator $\sum_i m_i$ in that formula looks innocuous. It is not — it is the source of the most widely-shipped bug in open SFT code, and we handle it explicitly in the training step below.

### Packing masked conversations

Conversations vary wildly in length; padding each to `max_seq_len=2048` would waste most of the batch on `<|pad|>`. As in pretraining we **pack**: concatenate rendered conversations end-to-end and slice into fixed-length windows. The subtlety is that packing must carry the mask *and* prevent cross-conversation attention — a token in conversation B must not attend to conversation A sharing its window. We solve this exactly as in [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) and Chapter 14.2: a per-token `seq_id` that resets the document-aware attention mask and the RoPE position ids at each conversation boundary.

```python
# capstone/stacklm/post/sft.py
import numpy as np, torch
from torch.utils.data import Dataset
from stacklm.post.chat import render_conversation

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
        # Pad the ragged tail up to a whole number of blocks with <|pad|>/IGNORE.
        n = max(1, len(ids_buf) // block) * block
        while len(ids_buf) < n:
            ids_buf.append(tok.pad_id); lbl_buf.append(IGNORE); seg_buf.append(seg)
        self.ids = np.array(ids_buf[:n], dtype=np.int64).reshape(-1, block)
        self.lbl = np.array(lbl_buf[:n], dtype=np.int64).reshape(-1, block)
        self.seg = np.array(seg_buf[:n], dtype=np.int64).reshape(-1, block)

    def __len__(self):  return self.ids.shape[0]
    def __getitem__(self, i):
        return (torch.from_numpy(self.ids[i]),
                torch.from_numpy(self.lbl[i]),
                torch.from_numpy(self.seg[i]))
```

Because a conversation can be *cut in half* by a block boundary, the tail half begins mid-assistant-turn with no prompt in view. That is acceptable (it is a small fraction of windows and the labels are still correct next-token targets), but it is also why a window can end up containing *zero* supervised tokens — a window that happens to hold nothing but a long user prompt. The training step must survive that case; see below.

### The SFT training step, and the normalization bug hiding in it

The training step reuses the same `Stack100M` model, bf16 autocast, gradient accumulation, and gradient clipping from the pretraining loop in Chapter 14.7. It has two new elements: the label shift with the ignore index, and a *low* peak learning rate with a short warmup and decay to zero.

It also has a trap. The obvious implementation computes `F.cross_entropy(..., ignore_index=-100)` — which returns the **mean over the supervised tokens in this microbatch** — and then divides by `grad_accum`. Under assistant-only masking the supervised-token count per 2048-token window varies by an order of magnitude: one window may hold 40 assistant tokens, the next 900. Dividing each microbatch mean by the same constant weights every microbatch *equally*, so the 40-token window contributes as much gradient as the 900-token one. The gradient you take is therefore **not** the gradient of the loss over the accumulation window; changing `grad_accum` silently changes the objective. This is the gradient-accumulation normalization bug that HuggingFace and Unsloth publicized in late 2024 and fixed across `transformers` and TRL, and assistant-only masking makes it much worse than it is in plain pretraining (where every window has exactly `block-1` supervised tokens and the bug vanishes).

The fix is to accumulate **sums**, not means, and divide once by the window's true token count. Since that count is only known at the accumulation boundary, we scale the *gradients* rather than the loss — exact, single-pass, and no extra forward.

```python
# capstone/stacklm/post/sft.py  (continued)
import torch, torch.nn.functional as F
from stacklm.model import Stack100M, StackConfig   # Ch. 14.4
from stacklm.optim import build_optimizer          # plain AdamW; see the note below

def sft_train(model, loader, *, epochs=3, lr=2e-5, warmup=100,
              grad_accum=8, max_grad_norm=1.0, device="cuda", log_every=20):
    """
    Full-parameter SFT of Stack-100M.

    lr=2e-5 is ~50-100x smaller than the pretraining peak LR: we are nudging a
    converged model. Too large an LR here erases pretrained knowledge (the
    'catastrophic forgetting' failure) — the model gets fluent-but-dumber.

    Loss normalization: we sum the per-token CE over the whole accumulation
    window and divide the accumulated GRADIENT by the window's supervised-token
    count. Per-microbatch means would weight a 40-token window the same as a
    900-token one — the classic grad-accum bug.
    """
    device = torch.device(device)
    model.to(device).train()
    opt = build_optimizer(model, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    total_steps = max(1, epochs * len(loader) // grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) *
                       max(0.0, 1 - max(0, s - warmup) / max(1, total_steps - warmup)))
    step, history = 0, []
    win_loss, win_tok = 0.0, 0          # running sums for the CURRENT window
    for ep in range(epochs):
        for micro, (ids, labels, seg) in enumerate(loader):
            ids, labels, seg = ids.to(device), labels.to(device), seg.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(ids, seq_ids=seg)     # forward returns (logits, loss)
                # Standard causal shift: predict token t+1 from tokens <= t.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()    # already IGNORE-masked
                n_tok = int((shift_labels != IGNORE).sum().item())
                if n_tok > 0:
                    loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)).float(),
                        shift_labels.view(-1),
                        ignore_index=IGNORE,
                        reduction="sum",                # SUM, not mean
                    )
            if n_tok > 0:            # a window with zero assistant tokens would
                loss_sum.backward()  # give 0/0 = NaN and poison the whole window
                win_loss += loss_sum.item(); win_tok += n_tok
            if (micro + 1) % grad_accum == 0:
                if win_tok > 0:
                    inv = 1.0 / win_tok                 # exact per-token mean grad
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(inv)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step(); sched.step(); step += 1
                    history.append(win_loss / win_tok)
                    if log_every and step % log_every == 0:
                        print(f"ep{ep} step{step} loss {history[-1]:.3f} ntok {win_tok}")
                opt.zero_grad(set_to_none=True)
                win_loss, win_tok = 0.0, 0
    return {"loss_history": history}
```

Two details in that loop are worth naming. `loss_sum.backward()` on an *unnormalized* sum makes the raw gradients ~$10^3\times$ larger than a mean would; that is fine because parameters and their `.grad` buffers are fp32 even under bf16 autocast, and we rescale before clipping so `max_grad_norm=1.0` still means what it says. And the zero-token guard is placed so it skips only the backward, never the accumulation-boundary bookkeeping — an early `continue` there would desynchronize the microbatch counter and quietly change your effective batch size.

!!! warning "Common pitfall: the off-by-one mask that silently trains on the prompt"

    The mask must land on the *target* position after the causal shift. A classic bug is to mask `logits`/`labels` before shifting, or to mark the assistant role token `<|assistant|>` as supervised. Both leak prompt tokens into the loss or, worse, teach the model to *emit* the role marker mid-turn. Always assert the invariant: after shifting, every non-`IGNORE` label id equals an assistant-turn token id. A one-line `assert (shift_labels[shift_labels!=-100] == ids[:,1:][shift_labels!=-100]).all()` on a debug batch catches this instantly.

    The same invariant is worth checking in *any* framework you use. TRL's `SFTTrainer` implements assistant-only masking with `assistant_only_loss=True`, which requires the tokenizer's chat template to wrap assistant content in a `{% generation %}` block so that `apply_chat_template(..., return_assistant_tokens_mask=True)` can return the mask. If the template lacks that block, the flag silently trains on everything — print one decoded batch and check.

!!! note "Optimizer choice for post-training: which factory, and why"

    `stacklm.optim` exposes **two** factories, and post-training deliberately uses the *singular* one:

    - `build_optimizers(model, muon_lr=0.02, adamw_lr=3e-3, ...)` — **plural** — returns the `(muon, adamw)` **hybrid pair** from Ch. 14.6: **Muon** (Jordan et al., 2024) on the 2D hidden matrices, AdamW on the tied embedding, RMSNorm gains, and 1D params. This is what the *pretraining* loop steps.
    - `build_optimizer(model, lr=2e-5, weight_decay=0.0, betas=(0.9, 0.95))` — **singular** — returns a single **AdamW over all parameters** (with weight decay applied only to $\ge$2D tensors).

    Muon earned its place in pretraining, where orthogonalizing the momentum update buys a real speedup over many thousands of steps. Post-training is a different regime: a few hundred low-LR steps on a converged model, where we want *small, well-behaved* nudges rather than aggressive reshaping, and where a geometry-changing update interacts badly with the tiny trust region DPO's $\beta$ and GRPO's KL leash are trying to enforce. Plain AdamW is the standard, safest choice across the open literature (Tülu 3, Zephyr, and essentially every public SFT/DPO recipe), so all three post-training loops in `stacklm.post` call `build_optimizer`. If you want to experiment with Muon here, swap in `build_optimizers` and step both optimizers — but change one thing at a time, and if you see instability in DPO or GRPO, come back to AdamW-only before touching anything else.

Expected outcome: on the order of a 1.5–2.5 nats/token loss on assistant tokens (illustrative — the exact figure depends on your SFT set). More telling than the number is the *behavior*: after SFT, Stack-100M answers direct questions, respects the turn structure, and stops. It also becomes noticeably more brittle to inputs unlike its SFT distribution — the first hint of the scale ceiling we return to at the end.

## DPO: preference optimization without a reward model

SFT teaches the model to imitate *one* good answer. But "good" is relative: for the prompt "Explain gravity to a child," there are many acceptable completions and many bad ones, and imitation learning has no way to express "this answer is better than that one." Preference optimization does. We collect **pairs** $(x, y_w, y_l)$ — a prompt, a *chosen* (winner) response, and a *rejected* (loser) response — and train the model to raise the relative likelihood of $y_w$ over $y_l$.

### Where the pairs come from — and why on-policy is the default at 100M

The tempting shortcut is to download **UltraFeedback** (Cui et al., 2023) — prompts with GPT-4-scored completions from a pool of models — and train on it directly. At 7B that works. At 100M it is close to counterproductive, and understanding *why* is the most important thing in this section.

UltraFeedback's chosen responses were written by 7B–70B models. Stack-100M assigns them log-probabilities in the neighborhood of $-$hundreds of nats: they are, from its perspective, essentially impossible strings. The DPO gradient (derived below) is proportional to $\sigma(-\text{margin})$ times the difference of the two responses' score functions, and when *both* responses are far off-policy the update that most easily lowers the loss is to push the rejected sequence's probability down hard — dragging the chosen one down with it, because they share vocabulary, style, and length statistics the model has no way to separate. This is exactly the "both chosen and rejected log-probs fall" failure the chapter warns about, and with fully off-policy pairs at 100M it is not a risk, it is the *expected* outcome. The Tülu 3 report's headline post-training finding points the same way: the quality of preference optimization is dominated by whether the preference data is **on-policy** — generated by the very model you are about to train.

So we invert the usual advice. **Default path: mine your own pairs from the SFT model.** Sample $k$ completions per prompt at temperature 1.0, score them, keep the best and the worst. Every pair is then a string the policy actually produces, the log-ratios start near zero, and the update is a genuine re-ranking of the model's own distribution rather than a doomed push toward a foreign one.

```python
# capstone/stacklm/post/dpo.py  (new in this chapter — append to the shipped module)
import torch
from stacklm.post.chat import Turn, render_conversation, SPECIAL
from stacklm.post.grpo import sample_group, exact_match_reward

def _strip_stop(text):
    """Cut a decoded completion at its first stop marker; drop the marker."""
    for stop in (SPECIAL["end"], SPECIAL["eos"]):
        text = text.split(stop)[0]
    return text.strip()

@torch.no_grad()
def mine_onpolicy_pairs(model, tok, prompts, score_fn, *, k=4, max_new=96,
                        temperature=1.0, device="cuda"):
    """
    Generate k completions per prompt from the CURRENT SFT policy, score them,
    and keep (prompt, best, worst) whenever the scores actually differ.

    prompts  : iterable of (prompt_text, meta) — `meta` is whatever score_fn needs
               (a gold answer for arithmetic; None for open-ended prompts).
    score_fn : (completion_text, meta) -> float, higher is better. Cheap options:
               exact-match for verifiable prompts, a format/length heuristic for
               chat prompts, or a larger open model used as a judge (Ch. 5.11).

    A prompt whose k samples all score the same is DROPPED — the same degeneracy
    that makes a GRPO group contribute no gradient makes a preference pair carry
    no information. Expect to discard a large fraction; that is normal and cheap.
    """
    model.eval()
    pairs = []
    for prompt_text, meta in prompts:
        p_ids, _ = render_conversation([Turn("user", prompt_text)], tok,
                                       add_generation_prompt=True)
        p_ids = torch.tensor(p_ids, dtype=torch.long)
        seqs, _gmask, Tp = sample_group(model, tok, p_ids, k, max_new=max_new,
                                        temperature=temperature, device=device)
        texts = [_strip_stop(tok.decode(seqs[i, Tp:].tolist())) for i in range(k)]
        scores = [score_fn(t, meta) for t in texts]
        hi = max(range(k), key=lambda i: scores[i])
        lo = min(range(k), key=lambda i: scores[i])
        if scores[hi] - scores[lo] < 1e-6 or not texts[hi] or not texts[lo]:
            continue                                     # no signal in this prompt
        pairs.append({"prompt": prompt_text,
                      "chosen": texts[hi], "rejected": texts[lo]})
    return pairs

def arithmetic_score(text, gold):
    """A verifiable score_fn: exact match dominates, well-formedness breaks ties."""
    r, pred = exact_match_reward(text, gold)
    return r + (0.1 if pred is not None else 0.0) - 0.001 * len(text)
```

The last term of `arithmetic_score` — a tiny length penalty — is not decoration; it is the antidote to the length bias we analyze at the end of this section. When the judge is a heuristic you control, put the anti-verbosity pressure *into the scoring function*, where it is explicit and tunable.

**UltraFeedback still has a role**, as a *supplementary* source: its prompts are far more diverse than anything you will write, and diversity of $x$ is exactly what mining needs. The high-value pattern is to take UltraFeedback's *prompts*, throw away its completions, and generate your own — the same trick Tülu 3's preference pipeline uses at scale. If you nevertheless want to train on its native pairs (worth doing once, to see the failure mode with your own eyes), here is the loader:

```python
# capstone/stacklm/post/dpo.py  (continued)
from datasets import load_dataset

def load_ultrafeedback_pairs(split="train_prefs", limit=20000):
    """
    HuggingFaceH4/ultrafeedback_binarized: each row has a `prompt` string and
    `chosen`/`rejected` message lists whose LAST message is the response.
    Returns the same dict shape as mine_onpolicy_pairs so the two mix freely.
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    out = []
    for row in ds.select(range(min(limit, len(ds)))):
        out.append({"prompt": row["prompt"],
                    "chosen": row["chosen"][-1]["content"],
                    "rejected": row["rejected"][-1]["content"]})
    return out
```

### The DPO loss, briefly (full derivation in Ch. 5.7)

The derivation is done in full in [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html); here is the one-paragraph version so the code is grounded. The KL-regularized RLHF objective has a closed-form optimal policy $\pi^*(y\mid x)\propto \pi_{\text{ref}}(y\mid x)\exp(\tfrac1\beta r(x,y))$. Invert it to write the reward as an *implicit* function of the policy, $r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$, then plug that into the Bradley–Terry preference likelihood. The intractable partition term $\beta\log Z(x)$ appears in both the winner and loser reward and *cancels in the difference*. What survives is a clean logistic loss:

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\Big(\beta\big(\underbrace{\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}}_{\text{winner log-ratio}} - \underbrace{\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}}_{\text{loser log-ratio}}\big)\Big)\right].
$$

Read it as: push the winner's log-probability *up relative to the reference* and the loser's *down relative to the reference*, with $\beta$ (typically 0.1) controlling how far we let the policy drift from $\pi_{\text{ref}}$. The reference $\pi_{\text{ref}}$ is our frozen SFT model. Crucially there is **no reward model and no generation** during training — DPO consumes a static dataset of pairs, so at 100M it costs about the same as another epoch of SFT. (The *mining* pass above does generate, but once, offline, and it is embarrassingly parallel.)

{{fig:dpo-relative-reshaping}}

### The preference dataset and collator

Pairs are variable-length and come in two per example, so packing (which worked for SFT) is the wrong tool: we need the chosen and rejected sequences of a pair to stay aligned in the batch. We right-pad instead.

```python
# capstone/stacklm/post/dpo.py  (new in this chapter — append to the shipped module)
import torch
from torch.utils.data import Dataset
from stacklm.post.chat import Turn, render_conversation

class PreferenceDataset(Dataset):
    """
    Renders {"prompt", "chosen", "rejected"} dicts into two masked sequences per
    example. The mask is exactly the assistant_mask from render_conversation, so
    the DPO log-probabilities are summed over RESPONSE tokens only — the prompt
    is identical in both branches and would cancel anyway, but including it would
    add its (large, noisy) magnitude to both log-ratios.

    Truncation policy: if a rendered pair exceeds max_len, DROP it. Truncating a
    chosen response mid-answer teaches the model that good answers stop abruptly,
    which is precisely the behavior SFT worked to prevent.
    """
    def __init__(self, pairs, tok, max_len=1024, system=None):
        self.items = []
        for p in pairs:
            rec = {}
            for key in ("chosen", "rejected"):
                turns = ([Turn("system", system)] if system else []) + [
                    Turn("user", p["prompt"]), Turn("assistant", p[key])]
                ids, mask = render_conversation(turns, tok, add_generation_prompt=False)
                if len(ids) > max_len:
                    rec = None; break
                rec[key] = (ids, mask)
            if rec:
                self.items.append(rec)

    def __len__(self):  return len(self.items)
    def __getitem__(self, i):  return self.items[i]

def collate_preferences(items, pad_id):
    """
    Right-pad chosen and rejected (independently) to the batch max.

    Why right-padding needs no attention mask here: attention is CAUSAL, so a
    token at position t can only attend to positions <= t. Pads sit at the END,
    strictly after every real token, so they cannot influence any real token's
    representation. Their own logits are computed and then discarded by the loss
    mask. (Left-padding would be wrong for the same reason it is right for
    batched *generation*: it shifts every real token's RoPE position.)
    """
    out = {}
    for key, prefix in (("chosen", "chosen"), ("rejected", "rejected")):
        seqs = [it[key] for it in items]
        T = max(len(ids) for ids, _ in seqs)
        ids_b = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        msk_b = torch.zeros((len(seqs), T), dtype=torch.float)
        for r, (ids, mask) in enumerate(seqs):
            ids_b[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            msk_b[r, :len(mask)] = torch.tensor(mask, dtype=torch.float)
        out[f"{prefix}_ids"] = ids_b
        out[f"{prefix}_mask"] = msk_b
    return out

# Usage:
#   ds = PreferenceDataset(pairs, tok, max_len=1024)
#   loader = torch.utils.data.DataLoader(
#       ds, batch_size=8, shuffle=False, collate_fn=lambda b: collate_preferences(b, tok.pad_id))
# shuffle=False matters if you cache reference log-probs by position (see the tip).
```

One constraint: `max_len` must stay $\le$ `model.cfg.max_seq_len`, since the RoPE cache is built to that length (Ch. 14.4).

### Implementation: per-sequence log-probs and the loss

The one quantity we need is the summed log-probability that a policy assigns to a response's tokens, given the prompt — the sequence log-likelihood, masked to the response.

```python
# capstone/stacklm/post/dpo.py
import torch, torch.nn.functional as F

def sequence_logprob(model, input_ids, loss_mask, seg=None):
    """
    Sum of log p(token_t | token_<t) over the *response* tokens only.

    input_ids : (B, T)   full rendered sequence (prompt + response), right-padded
    loss_mask : (B, T)   1.0 on response tokens (incl. closing <|end|>), else 0.0
    Returns   : (B,)     per-sequence response log-likelihood.
    """
    logits, _ = model(input_ids, seq_ids=seg)              # forward -> (logits, loss)
    logits = logits[:, :-1, :]                             # predict t+1 from <=t
    targets = input_ids[:, 1:]                             # (B, T-1)
    mask = loss_mask[:, 1:]                                # align mask to targets
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)  # (B,T-1)
    return (tok_logp * mask).sum(dim=-1)                   # sum over response tokens

def dpo_loss(policy, ref, batch, beta=0.1, device="cuda"):
    """
    One DPO step's loss on a batch of preference pairs.

    batch comes from collate_preferences: chosen_ids/chosen_mask and
    rejected_ids/rejected_mask. `ref` is the frozen SFT model (no grad). We do
    FOUR forward passes: policy(chosen), policy(rejected), ref(chosen),
    ref(rejected). The ref passes are under no_grad and can be cached away.
    """
    ch_ids  = batch["chosen_ids"].to(device);   ch_m = batch["chosen_mask"].to(device)
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

The training loop is the SFT loop with `dpo_loss` in place of the cross-entropy, a *frozen copy* of the SFT model as `ref`, and an even smaller LR (≈ 5e-7 to 1e-6) — DPO is sensitive and a large LR drives the loss down while *degrading* the model.

```python
# capstone/stacklm/post/dpo.py  (continued)
import copy
from stacklm.optim import build_optimizer          # plain AdamW over all params

def dpo_train(sft_model, loader, *, epochs=1, lr=5e-7, beta=0.1,
              grad_accum=8, max_grad_norm=1.0, device="cuda", log_every=20):
    device = torch.device(device)
    policy = sft_model.to(device)                        # trainable
    ref = copy.deepcopy(sft_model).to(device).eval()     # frozen reference
    for p in ref.parameters(): p.requires_grad_(False)
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    step, history = 0, []
    for ep in range(epochs):
        for micro, batch in enumerate(loader):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, stats = dpo_loss(policy, ref, batch, beta=beta, device=device)
                loss = loss / grad_accum       # safe here: every pair contributes
                                               # exactly one term to the mean
            loss.backward()
            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                history.append(loss.item() * grad_accum)
                if log_every and step % log_every == 0:
                    print(f"dpo step{step} loss {history[-1]:.3f} acc {stats['acc']:.2f} "
                          f"chosen_r {stats['chosen_r']:+.3f} "
                          f"rejected_r {stats['rejected_r']:+.3f}")
    return policy, {"loss_history": history}
```

Note the contrast with SFT: here `loss / grad_accum` *is* correct, because `dpo_loss` averages over *pairs* and every microbatch carries the same number of pairs (fixed `batch_size`). The rule is general — dividing by `grad_accum` is valid exactly when the per-microbatch denominator is constant. It is constant for pairs; it is emphatically not constant for supervised tokens.

!!! tip "Cache the reference log-probs — DPO's one free lunch"

    The reference model $\pi_{\text{ref}}$ is frozen, and the preference dataset is static, so $\log\pi_{\text{ref}}(y_w\mid x)$ and $\log\pi_{\text{ref}}(y_l\mid x)$ never change across epochs. Compute them *once* in a pre-pass, store two floats per pair, and delete the reference model from GPU memory entirely. This halves the forward passes per step (two policy forwards instead of four) and frees ~100M parameters of VRAM — at our scale it turns DPO into "SFT with a cleverer loss." The code above keeps `ref` resident for clarity; Exercise 4 builds the cached path. TRL exposes the same optimization as `DPOConfig(precompute_ref_log_probs=True)`. If you later switch to **reference-free** variants like **SimPO**, **CPO**, or **ORPO** (below), the reference disappears from the loss altogether.

### Length bias, and the variant zoo

Look again at `sequence_logprob`: it returns a raw **sum** of token log-probabilities. Every token contributes a negative number, so *longer responses have systematically lower sequence log-probability*, and the DPO margin conflates "better" with "shorter." Which way this biases the model depends on your data. With UltraFeedback, whose chosen responses are systematically *longer* than its rejected ones, the objective must fight the length term to satisfy the preference — and the easiest way to raise $\log\pi_\theta(y_w)$ across a long sequence is to raise the probability of generic filler, which is one mechanism behind DPO's well-known **verbosity drift**. With on-policy pairs mined at fixed `max_new`, lengths are much better matched and the bias is milder — another reason on-policy is the default here.

The remedies, in increasing order of departure from vanilla DPO:

- **Length-normalized DPO.** Divide each sequence log-probability by its response length before forming the log-ratio (i.e. use the mean token log-prob). One line — `return (tok_logp * mask).sum(-1) / mask.sum(-1).clamp(min=1)` — and the length term is gone. Tülu 3 reports length-normalized DPO as its production choice.
- **SimPO** (Meng et al., 2024) makes length-normalization the *definition* of the implicit reward, drops the reference model entirely, and adds a target margin $\gamma$. Attractive at 100M — no reference model at all — at the cost of two hyperparameters ($\beta$, $\gamma$) you must tune.
- **ORPO** (Hong et al., 2024) merges SFT and preference optimization into a **single stage**: the loss is the ordinary SFT cross-entropy on the chosen response plus an odds-ratio penalty on the rejected one, with no reference model and no separate SFT checkpoint. At our budget this is genuinely tempting — one pass instead of two — and it is the variant to reach for if the SFT-then-DPO ladder is eating your GPU-hours.
- **KTO** (Ethayarajh et al., 2024) drops the *pairing requirement*: it learns from independently-labelled good/bad responses using a prospect-theoretic utility. This matters when your feedback is a thumbs-up/thumbs-down log rather than head-to-head comparisons, which is what most real deployments actually collect.

All four are implemented in TRL (`DPOTrainer` with `loss_type` variants, plus dedicated `CPOTrainer`, `ORPOTrainer`, `KTOTrainer`), so trying them is a config change, not a rewrite. The full comparison lives in [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

!!! example "Worked example: what one DPO step actually does to the logits"

    Take $\beta = 0.1$ and a single pair. Suppose the frozen reference assigns the winner and loser these response log-likelihoods: $\log\pi_{\text{ref}}(y_w)=-30.0$, $\log\pi_{\text{ref}}(y_l)=-28.0$ (the reference actually finds the *loser* slightly more likely — a case DPO should fix). Early in training the policy still matches the reference, so both log-ratios are ~0, the margin $\beta(0-0)=0$, and $\sigma(0)=0.5$: the loss is $-\log 0.5 = 0.693$ nats and implicit-reward accuracy is a coin flip.

    Now suppose after some steps the policy has moved to $\log\pi_\theta(y_w)=-27.0$ (winner up by 3 nats) and $\log\pi_\theta(y_l)=-29.0$ (loser down by 1 nat). The log-ratios are $+3.0$ and $-1.0$; the margin is $\beta(3.0-(-1.0)) = 0.1\times 4.0 = 0.40$. The loss drops to $-\log\sigma(0.40) = -\log(0.599) = 0.512$ nats and accuracy for this pair is 1 (margin $>0$).

    The lesson in the magnitudes: because $\beta=0.1$, even a *4-nat* separation in log-likelihood produces only a **0.4-logit** margin, i.e. a gentle $\sigma(0.4)\approx 0.60$ preference probability. DPO deliberately moves the policy in small steps. If you crank $\beta$ up to make the margin bigger, you also tighten the implicit KL leash and the policy barely moves at all; the sweet spot near $0.1$ is what keeps chosen-reward rising *without* both rewards collapsing.

    Now contrast the off-policy case. Suppose $y_w$ and $y_l$ came from a 70B model, so $\log\pi_{\text{ref}}(y_w)=-310$ and $\log\pi_{\text{ref}}(y_l)=-290$ (both astronomically unlikely, and the *loser* — shorter — is the more likely of the two under the reference). The pair asks the policy to close a 20-nat gap in the wrong direction across ~150 tokens the model has essentially never produced. The gradient available to do that is dominated by suppressing shared high-frequency tokens, and the measured outcome is both `chosen_r` and `rejected_r` marching downward together. That is why the diagnostics in `dpo_loss` report the two rewards *separately* rather than only the margin.

The honest read on DPO at 100M: it reliably removes *obvious* failure modes present in the base/SFT model (rambling, ignoring the format, repeating the prompt) when the preference pairs target those failures and come from the model itself. It does **not** install new reasoning. If your rejected/chosen pairs differ mainly in a capability the base model lacks — say, correct multi-step arithmetic — DPO will happily raise the log-probability of the "chosen" correct answer *relative to* the reference, but the model still cannot *produce* correct arithmetic on a new problem, because the log-ratio objective only reshapes the distribution over responses it can already generate. For a *capability* gain we need a signal tied to *correctness on new inputs*. That is RLVR.

## Narrow RLVR with GRPO: making RL actually work at 100M

Reinforcement learning with **verifiable rewards** (RLVR) replaces the learned, hackable reward model with a *program* that checks correctness. On a math problem, the checker parses the model's final answer and compares it to the ground truth: reward 1 if exactly right, 0 otherwise. There is nothing to reward-hack (short of the model finding the checker's bugs), the signal is always available, and — the key point for us — it rewards *being correct on inputs the model has not memorized*, which is exactly the capability signal DPO could not provide. See [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html) for the general recipe and [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html) for how real verifiers are built and sandboxed.

The catch, and the reason this section is titled "narrow," is the **cold-start problem**. RL can only reinforce behavior the model *sometimes* produces. If Stack-100M gets a task right 0% of the time, every sample earns reward 0, every advantage is 0, and the gradient is exactly zero — RL has nothing to climb. RLVR works when the base+SFT model already succeeds *occasionally* (say 10–40% of the time) so that the reward signal has variance to exploit. At 100M this restricts us to genuinely narrow, in-distribution tasks. We pick **integer arithmetic and one-step word problems** — a task Stack-100M, after math-heavy mid-training (Ch. 14.8), solves often enough to bootstrap.

### The task and the verifier

```python
# capstone/stacklm/post/grpo.py
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

Format-following (`####` marker) is itself a behavior we bootstrapped in SFT — a few thousand arithmetic exemplars in the SFT set teach the model to *emit* the marker, so the verifier can find an answer to grade. Without that, parse-failure rate is ~100% and RL never starts. This SFT→RLVR ordering is not optional; it is the recipe. (The `####` convention is borrowed straight from **GSM8K** (Cobbe et al., 2021), whose answers are delimited exactly this way — a small nod to keeping our task in a well-trodden format.)

!!! tip "Verifiers are a library problem, not a regex problem"

    Our `_FINAL` regex is honest for integer arithmetic and dishonest for anything else: real math answers arrive as `\frac{3}{4}`, `0.75`, `3/4`, or `\boxed{0.75}`, and a naive exact-match verifier scores three of those wrong. Beyond this chapter's toy task, use a purpose-built checker: **`math-verify`** (HuggingFace) parses and *symbolically* compares LaTeX/expression answers; **`verifiers`** (Brown) packages RLVR environments and rubric-based reward functions behind a common interface that TRL and verl can consume. And log the **parse-failure rate separately from the accuracy** — the two failure modes ("wrong answer" vs "unparseable output") demand opposite fixes, and conflating them is how teams spend a week tuning RL hyperparameters to fix a formatting bug.

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

Compare the *baseline* choice with **RLOO** (Ahmadian et al., 2024), which uses the leave-one-out mean $\frac{1}{G-1}\sum_{j\ne i} R_j$ instead of the group mean and does *not* divide by the std. RLOO's advantage is an unbiased policy-gradient estimator; GRPO's std division is not, which is precisely the issue Dr. GRPO raises below. At $G=8$ the practical difference is small; at $G=4$ or less, RLOO's bias-free baseline is the safer choice.

{{fig:grpo-group-advantage-arith}}

### A minimal GRPO loop

Here is a complete, runnable GRPO loop for the arithmetic task. It is intentionally minimal — single GPU, synchronous generate-then-train, no distributed rollout engine — which is exactly right for a 100M model where generation is cheap. (The production version of this loop, with vLLM rollouts and weight sync, is the subject of Part VI; see [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).)

```python
# capstone/stacklm/post/grpo.py  (continued)
import torch, copy
from stacklm.post.chat import render_conversation, Turn, SPECIAL
from stacklm.optim import build_optimizer          # plain AdamW over all params

@torch.no_grad()
def sample_group(model, tok, prompt_ids, G, max_new=64, temperature=1.0, device="cuda"):
    """
    Sample G completions for ONE prompt via temperature sampling.
    Returns (seqs, gen_mask, Tp): each seq is prompt+completion; gen_mask marks
    the generated tokens — the only ones we compute the loss on; Tp is the prompt
    length, so callers can slice the completion out for decoding.
    Batched over the group: replicate the prompt G times and decode together.
    """
    model.eval()
    end_id = tok.special_token_id(SPECIAL["end"])
    eos_id = tok.special_token_id(SPECIAL["eos"])
    x = prompt_ids.to(device).unsqueeze(0).repeat(G, 1)     # (G, Tp)
    Tp = x.size(1)
    cap = model.cfg.max_seq_len                              # stay inside the RoPE cache
    done = torch.zeros(G, dtype=torch.bool, device=device)
    for _ in range(max_new):
        if x.size(1) >= cap:
            break
        logits, _ = model(x)                                 # forward -> (logits, loss)
        logits = logits[:, -1, :]                            # (G, V)
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
    return x, gen_mask, Tp

def token_logprobs(model, seqs):
    """Per-token log-prob of the realized next token: (B, T-1)."""
    logits, _ = model(seqs)                                   # forward -> (logits, loss)
    logits = logits[:, :-1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, 2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)

def grpo_train(sft_model, tok, *, iterations=200, group_size=8, prompts_per_iter=16,
               inner_epochs=2, lr=1e-6, clip_eps_low=0.2, clip_eps_high=0.28,
               kl_beta=0.02, temperature=1.0, max_new=64, device="cuda", seed=0,
               reward_fn=exact_match_reward, log_every=10):
    """
    Minimal single-GPU GRPO on integer arithmetic with exact-match reward.
    Each iteration: (1) sample rollouts with the *current* policy (theta_old),
    (2) grade them with the verifier, (3) compute group-relative advantages,
    (4) take a few clipped-surrogate gradient steps.
    """
    import random
    rng = random.Random(seed)
    device = torch.device(device)
    policy = sft_model.to(device)
    ref = copy.deepcopy(policy).to(device).eval()             # frozen KL anchor
    for p in ref.parameters(): p.requires_grad_(False)
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    accs, losses = [], []

    for it in range(iterations):
        # ---- 1 & 2: rollout + reward, accumulated across several prompts ----
        batch_seqs, batch_gmask, batch_adv, batch_oldlp = [], [], [], []
        n_correct, n_total, n_degenerate = 0, 0, 0
        for _ in range(prompts_per_iter):
            q, gold = make_arithmetic_prompt(rng)
            p_ids, _ = render_conversation([Turn("user", q)], tok,
                                           add_generation_prompt=True)
            p_ids = p_ids[: max(1, policy.cfg.max_seq_len - max_new - 1)]
            p_ids = torch.tensor(p_ids, dtype=torch.long)
            seqs, gmask, Tp = sample_group(policy, tok, p_ids, group_size,
                                           max_new=max_new, temperature=temperature,
                                           device=device)
            # grade each completion
            rewards = torch.zeros(group_size, device=device)
            for i in range(group_size):
                text = tok.decode(seqs[i, Tp:].tolist())
                r, _ = reward_fn(text, gold)
                rewards[i] = r
            n_correct += int((rewards >= 1.0).sum().item()); n_total += group_size
            if rewards.max() == rewards.min():
                n_degenerate += 1        # zero std -> zero advantage -> no gradient
            # ---- 3: group-relative standardized advantage ----
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)   # (G,)
            # cache old (theta_old) token log-probs for the importance ratio
            with torch.no_grad():
                old_lp = token_logprobs(policy, seqs)                   # (G, T-1)
            batch_seqs.append(seqs); batch_gmask.append(gmask)
            batch_adv.append(adv);   batch_oldlp.append(old_lp)

        # ---- 4: clipped-surrogate updates (a few inner epochs on the rollouts) ----
        policy.train()
        last_loss, clip_frac, nll = 0.0, 0.0, 0.0
        for _ in range(inner_epochs):
            for seqs, gmask, adv, old_lp in zip(batch_seqs, batch_gmask,
                                                batch_adv, batch_oldlp):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    new_lp = token_logprobs(policy, seqs)              # (G, T-1)
                    m = gmask[:, 1:]                                    # align to targets
                    ratio = torch.exp(new_lp - old_lp)                 # rho_{i,t}
                    a = adv.unsqueeze(1)                               # (G,1) broadcast
                    unclipped = ratio * a
                    # clip-higher (DAPO): a wider UPPER bound leaves room to raise
                    # the probability of rare-but-correct tokens; the lower bound
                    # stays tight so nothing collapses to zero.
                    clipped = torch.clamp(ratio, 1 - clip_eps_low,
                                          1 + clip_eps_high) * a
                    surrogate = torch.min(unclipped, clipped)
                    # per-token KL(pi || ref), k3 estimator (Schulman)
                    with torch.no_grad():
                        ref_lp = token_logprobs(ref, seqs)
                    logr = ref_lp - new_lp
                    kl = torch.exp(logr) - logr - 1.0
                    per_tok = -(surrogate - kl_beta * kl)
                    # token-level normalization over the GENERATED tokens (DAPO)
                    loss = (per_tok * m).sum() / m.sum().clamp(min=1)
                    with torch.no_grad():        # health metrics, see the table below
                        clip_frac = (((ratio > 1 + clip_eps_high) |
                                      (ratio < 1 - clip_eps_low)) * m).sum().item() \
                                    / m.sum().clamp(min=1).item()
                        nll = -(new_lp * m).sum().item() / m.sum().clamp(min=1).item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                last_loss = loss.item()
        acc = n_correct / max(1, n_total)
        accs.append(acc); losses.append(last_loss)
        if log_every and it % log_every == 0:
            print(f"grpo it{it} acc {acc:.3f} loss {last_loss:.4f} "
                  f"degen {n_degenerate}/{prompts_per_iter} "
                  f"clip {clip_frac:.3f} logp/tok {-nll:.3f}")
    return policy, {"acc_history": accs, "loss_history": losses}
```

A few implementation notes that matter for correctness.

**The importance ratio.** `old_lp` holds the log-probs *under the policy that generated the rollouts*, cached before any update — this is what makes the inner-epoch updates valid off-policy corrections rather than a bug. On the very first inner step $\rho\equiv 1$ (new = old), so `torch.min` and the clip are no-ops and the update is a plain group-relative REINFORCE step; the clipping only bites once the policy has moved. If you set `inner_epochs=1` the algorithm degenerates (correctly) to on-policy REINFORCE with a group baseline, and you can delete the ratio machinery entirely.

**The KL term.** We use the **k3 unbiased estimator** $e^{\log r}-\log r-1$ with $\log r = \log\pi_{\text{ref}} - \log\pi_\theta$ (always non-negative, low variance) rather than the naive log-ratio; since the samples come from $\pi_\theta$, this estimates $\mathbb{D}_{\text{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}})$, which is the direction the objective asks for. This is the standard choice discussed in [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html). Note that DAPO *removes* the KL term entirely for long-horizon reasoning runs, on the argument that the policy is supposed to move far from the SFT reference. At 100M we keep it: our whole strategy is a narrow gain without collateral damage to general chat, and the KL leash is what buys that.

**Which normalizer is whose.** This is worth getting exactly right, because the two named corrections are routinely conflated. **Dr. GRPO** (Liu et al., 2025) makes *two* changes to the original GRPO objective: (i) it **removes the division by the group reward std** in the advantage, and (ii) it replaces the per-sequence $1/|o_i|$ normalizer with a **constant** normalizer. The code above applies *neither* of those literally. It divides by $\sum_i |o_i|$ — the total number of generated tokens in the group — which is the **token-level policy-gradient loss** introduced by **DAPO** (Yu et al., 2025), and it **keeps** the std normalization.

Why keep the std? Dividing by the group std reweights prompts by difficulty: a prompt where 1 of 8 samples is correct gets $\hat A \approx +2.6$ on the winner, while a prompt where 4 of 8 are correct gets $\hat A = +1.0$. Dr. GRPO is right that this is a bias — it is not the policy gradient of any objective — but in practice it acts as a free difficulty curriculum, amplifying exactly the hard-but-not-impossible prompts that carry the most information. At 100M, where the accuracy band is narrow anyway, we keep it and note the ablation is one line:

```python
adv = rewards - rewards.mean()                 # Dr. GRPO: no std division
```

Try both. TRL exposes the same switch (`GRPOConfig(scale_rewards=...)`, and a `loss_type` selecting the sequence-level, token-level, or constant normalizer — names track the current TRL release).

**Generation cost.** Generation here recomputes the full forward each step with no KV cache — fine for a 100M model and 64-token completions, but the first thing you would replace with a real rollout engine at any larger scale.

### The 2026 patch set: dynamic sampling, and what to do about wasted groups

Exercise 5 below works out the arithmetic that motivates this section: with $G=8$ and a per-sample success rate $p=0.05$, **two-thirds of your groups are all-wrong**, produce zero advantage, and burn their entire rollout budget for nothing. Early in an RLVR run — precisely when the model is weakest and you most need signal — this is the dominant cost.

**DAPO's dynamic sampling** (Yu et al., 2025) is the standard fix, and it is embarrassingly simple: *oversample prompts and throw away every degenerate group*, continuing until you have filled a batch with groups whose accuracy is strictly between 0 and 1. You pay more generation to get a batch in which every example contributes gradient, and — this is the part that surprises people — total wall-clock to a given accuracy typically *improves*, because gradient steps stop being diluted by zeros. DAPO pairs it with three other changes, of which we already adopted one: **clip-higher** (decouple $\epsilon_{\text{low}}$ from a larger $\epsilon_{\text{high}}$, so the update has room to raise the probability of rare-but-correct tokens — the reported setting is roughly 0.2/0.28), **token-level policy-gradient loss** (our normalizer), and **overlong reward shaping** (a soft penalty as completions approach the length cap, instead of a hard zero that injects reward noise).

```python
# capstone/stacklm/post/grpo.py  (new in this chapter — append to the shipped module)
@torch.no_grad()
def collect_nondegenerate_groups(policy, tok, rng, *, target_groups, group_size,
                                 reward_fn=exact_match_reward, max_new=64,
                                 temperature=1.0, oversample=3.0, device="cuda"):
    """
    DAPO-style dynamic sampling: keep drawing prompts and DISCARD every group
    whose samples all score the same (accuracy 0 or 1), until we have
    `target_groups` groups that actually carry gradient.

    Returns (groups, tries) where each group is (seqs, gmask, rewards, Tp).
    `tries / len(groups)` is the single most informative RLVR health metric: it
    is 1.0 in the sweet spot and blows up as the task drifts out of reach.
    Budget-capped by `oversample` so a hopeless task fails fast instead of hanging.
    """
    kept, tries, budget = [], 0, int(target_groups * oversample)
    while len(kept) < target_groups and tries < budget:
        tries += 1
        q, gold = make_arithmetic_prompt(rng)
        p_ids, _ = render_conversation([Turn("user", q)], tok,
                                       add_generation_prompt=True)
        p_ids = p_ids[: max(1, policy.cfg.max_seq_len - max_new - 1)]
        p_ids = torch.tensor(p_ids, dtype=torch.long)
        seqs, gmask, Tp = sample_group(policy, tok, p_ids, group_size,
                                       max_new=max_new, temperature=temperature,
                                       device=device)
        rewards = torch.tensor(
            [reward_fn(tok.decode(seqs[i, Tp:].tolist()), gold)[0]
             for i in range(group_size)], device=device)
        if rewards.max() == rewards.min():
            continue                       # degenerate: zero std, zero gradient
        kept.append((seqs, gmask, rewards, Tp))
    return kept, tries
```

Drop it into `grpo_train` by replacing the `for _ in range(prompts_per_iter)` rollout block with:

```python
groups, tries = collect_nondegenerate_groups(
    policy, tok, rng, target_groups=prompts_per_iter, group_size=group_size,
    reward_fn=reward_fn, max_new=max_new, temperature=temperature, device=device)
for seqs, gmask, rewards, Tp in groups:
    n_correct += int((rewards >= 1.0).sum().item()); n_total += group_size
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    with torch.no_grad():
        old_lp = token_logprobs(policy, seqs)
    batch_seqs.append(seqs); batch_gmask.append(gmask)
    batch_adv.append(adv);   batch_oldlp.append(old_lp)
# NOTE: `n_correct / n_total` is now conditioned on non-degenerate groups and is
# NO LONGER an unbiased estimate of accuracy. Track true accuracy on a fixed
# held-out prompt set instead — this is a real reporting trap.
```

That last comment is not a footnote. Dynamic sampling changes the meaning of your training-accuracy curve: it is now computed over a *filtered* population that excludes both the all-wrong and the all-right prompts, so it is biased toward 50% and will look flat while the model genuinely improves. Evaluate on a fixed held-out set (Ch. 14.11) and treat the training curve purely as a health signal.

Two further 2026 developments are worth knowing even though we do not need them at 100M. **GSPO** (Group Sequence Policy Optimization; Zheng et al., Qwen team, 2025) replaces the per-token importance ratio with a **length-normalized sequence-level** ratio, $\rho_i = \big(\pi_\theta(o_i\mid q)/\pi_{\theta_{\text{old}}}(o_i\mid q)\big)^{1/|o_i|}$, and clips at the sequence level. The motivation is that per-token ratios accumulate variance over long rollouts and interact badly with MoE routing changes between the rollout and training policies; at 64-token completions on a dense 100M model, neither problem bites. And **entropy collapse** — the policy's per-token entropy sliding toward zero over an RLVR run, after which every sample in a group is identical, every group is degenerate, and learning stops dead — is the failure mode most likely to end your run. The `logp/tok` metric printed by the loop above is your early warning; the remedies are the ones already in the loop (keep the KL leash on, keep `temperature` at 1.0, do not over-train) plus, at larger scale, explicit entropy regularization. See [Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html).

| Metric | Healthy | What it means when it is not |
|---|---|---|
| degenerate-group fraction | < 20% | Task too hard (all-wrong) or exhausted (all-right); fix with SFT warm-start or curriculum |
| `tries / groups` under dynamic sampling | ~1.0–1.5 | Rising = the task is drifting out of the model's reach |
| clip fraction | 5–20% | ~0% = LR too small / nothing happening; > 40% = LR too large, policy lurching |
| mean log-prob per generated token | slowly rising | Rising *fast* = entropy collapse; falling = the KL penalty is winning, lower `kl_beta` |
| parse-failure rate | < 5% and falling | The model is losing the SFT format; raise `kl_beta` or add a format shaping term |
| KL to reference | small, bounded | Growing without bound = collateral damage to general chat |

!!! example "Worked example: a single GRPO group on '17 * 4'"

    Prompt: "Compute 17 * 4. Give the final integer after '####'." Gold = 68. We sample $G=8$ completions at temperature 1.0. Suppose the graded rewards come back as

    $$R = [1, 0, 1, 0, 0, 1, 0, 0]$$

    — three of eight correct, a 37.5% hit rate, right in the RLVR sweet spot. The group mean is $\bar R = 3/8 = 0.375$ and the *population* std is $\sqrt{0.375(1-0.375)} = \sqrt{0.234} = 0.484$. (The code uses `torch.std`, whose default Bessel correction divides by $G-1$ and gives $0.518$; with $\varepsilon=10^{-6}$ the difference is immaterial to the sign and near-magnitude of the advantage — the point is the *ranking*, not the third decimal.) Using the population std, the standardized advantages are:

    - correct samples: $\hat A = (1 - 0.375)/0.484 = +1.29$
    - incorrect samples: $\hat A = (0 - 0.375)/0.484 = -0.775$

    Every token of a *correct* completion gets advantage $+1.29$ (make these tokens more likely); every token of a *wrong* completion gets $-0.775$ (make them less likely). Because there are more wrong samples, each correct one is pushed up harder than each wrong one is pushed down — the group balances itself, and the advantages sum to zero.

    Now compare a *harder* prompt in the same batch, where only 1 of 8 samples is correct. Then $\bar R = 0.125$, population std $=\sqrt{0.125\times0.875}=0.331$, and the single winner gets $\hat A = (1-0.125)/0.331 = +2.65$ — more than double the $+1.29$ a winner earns on the easier prompt. That is the difficulty reweighting the std division buys (and that Dr. GRPO removes): hard-but-solvable prompts shout louder. Under Dr. GRPO's unnormalized advantage the two winners would get $1-0.125 = +0.875$ and $1-0.375 = +0.625$ — the hard prompt is still favored, but by a factor of $1.4$ instead of $2.05$. The *ordering* survives either way; the *amplification* is what the std division adds, and whether that amplification is a helpful curriculum or an unprincipled bias is the whole disagreement.

    Finally the two degenerate cases: if all 8 were wrong, $\bar R=0$, std $=0$, every $\hat A = 0/\varepsilon = 0$ and the prompt contributes **no gradient** (too hard — the cold-start trap). If all 8 were right, same thing (nothing left to learn). Only groups with *mixed* outcomes teach anything — which is precisely why the base model must already be partially competent for RLVR to lift it, and why dynamic sampling refuses to spend a gradient step on the others.

### What we honestly observe

Run this loop on Stack-100M after math-focused mid-training and SFT, and the arithmetic accuracy climbs — on the order of from ~25–35% to ~60–80% on in-distribution problems over a couple hundred iterations, with the exact figure depending on operand range and the SFT warm-start (these are illustrative magnitudes, not a measured benchmark). That is a *real* capability gain from RL, on a 100M model, and it is the payoff the chapter promised. But watch the boundaries:

- **It generalizes narrowly.** Push the operand range past training (three-digit multiplication), and accuracy collapses — the model learned a better *distribution over the arithmetic it practiced*, not a general multiplication algorithm. This is the 100M ceiling, not a bug in GRPO. Hold out a harder operand band from the start so you can *measure* the collapse instead of guessing.
- **Format drift and reward hacking creep in.** Even with a strict verifier, the model may learn to emit the `####` marker early and guess, or to pad reasoning that doesn't help. A KL leash to the SFT reference and a small format penalty keep it honest; see [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).
- **It can quietly forget.** Aggressive RL on one narrow task degrades general chat quality. Keep `kl_beta` non-trivial (≈ 0.02–0.05), stop early, and re-run your held-out chat eval *before and after* — the goal is a *narrow tool*, and we accept the trade only once we have measured it.
- **Entropy is the resource you are spending.** RLVR does not create new behaviors; it concentrates probability on the good ones the model already samples. Once entropy is gone, so is the search. Every RLVR run is a race between accuracy going up and diversity going down.

## The same three stages with real libraries

Writing all of this from scratch is the point of the capstone — you now know exactly what each line does. But nobody ships a from-scratch GRPO loop, and knowing which library owns which layer is part of knowing the stack. Here is the same ladder in **TRL** (HuggingFace's post-training library), which is the right tool the moment you leave 100M:

```python
# pip install trl transformers datasets accelerate
from datasets import load_dataset
from trl import (SFTTrainer, SFTConfig,
                 DPOTrainer, DPOConfig,
                 GRPOTrainer, GRPOConfig)

# --- 1. SFT -----------------------------------------------------------------
sft = SFTTrainer(
    model="./stack100m-base",                    # or a loaded nn.Module
    train_dataset=load_dataset("HuggingFaceTB/smoltalk", "all", split="train"),
    args=SFTConfig(
        max_length=2048,
        packing=True,                            # the packing we built by hand
        assistant_only_loss=True,                # the mask we built by hand;
                                                 # needs {% generation %} in the
                                                 # tokenizer's chat template
        learning_rate=2e-5, num_train_epochs=3,
        per_device_train_batch_size=8, gradient_accumulation_steps=8, bf16=True,
    ),
)
sft.train()

# --- 2. DPO -----------------------------------------------------------------
dpo = DPOTrainer(
    model="./stack100m-sft",
    ref_model=None,                              # None => an internal frozen copy
    train_dataset=load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                               split="train_prefs"),
    args=DPOConfig(beta=0.1, learning_rate=5e-7,
                   max_length=1024, max_prompt_length=512,
                   precompute_ref_log_probs=True,   # the free lunch, built in
                   bf16=True),
)
dpo.train()

# --- 3. GRPO / RLVR ---------------------------------------------------------
from stacklm.post.grpo import _FINAL           # the '####(-?\d+)' answer regex

def reward_exact_match(completions, answer, **kwargs):
    """TRL reward functions take the batch of completions and any extra dataset
    columns (here `answer`) and return one float per completion."""
    return [1.0 if _FINAL.search(c) and int(_FINAL.search(c).group(1)) == a else 0.0
            for c, a in zip(completions, answer)]

grpo = GRPOTrainer(
    model="./stack100m-dpo",
    reward_funcs=[reward_exact_match],           # a LIST: rewards are summed
    train_dataset=arithmetic_prompts,            # columns: prompt, answer
    args=GRPOConfig(num_generations=8,           # our group_size G
                    max_completion_length=64,
                    beta=0.02,                   # our kl_beta
                    temperature=1.0,
                    use_vllm=False,              # True => vLLM rollout backend
                    bf16=True),
)
grpo.train()
```

Argument names track TRL's current API and do move between releases — read the `SFTConfig`/`DPOConfig`/`GRPOConfig` dataclasses in the version you install. The mapping to what we built is one-to-one, which is the point: `packing` is our `PackedSFTDataset`, `assistant_only_loss` is our `render_conversation` mask, `precompute_ref_log_probs` is our caching tip, `num_generations` is our $G$, `beta` in `GRPOConfig` is our `kl_beta`, and `reward_funcs` is our `exact_match_reward`. Recent releases also expose the DAPO knobs directly (`epsilon`/`epsilon_high` for clip-higher, `scale_rewards` for the Dr. GRPO std ablation, `num_iterations` for our `inner_epochs`).

When do you graduate past TRL? Roughly, when rollouts stop fitting in the training process:

| Layer | Library | Why you would reach for it |
|---|---|---|
| Single-node SFT/DPO/GRPO, HF-native | **TRL** | Default. Everything above, plus `ORPOTrainer`/`KTOTrainer`/`CPOTrainer`. See [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html) |
| A reproducible full pipeline (SFT → DPO → RLVR) | **allenai/open-instruct** | The actual Tülu 3 codebase; data pipelines and eval included |
| Multi-node RL with disaggregated rollouts | **volcengine/verl** | HybridFlow single-controller; PPO/GRPO/RLOO with vLLM or SGLang workers. See [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html) |
| Ray-native RL, large actor/critic splits | **OpenRLHF**, **NeMo-Aligner** | See [OpenRLHF, NeMo-Aligner & Ray-Based Systems](../06-rl-infra/05-openrlhf-nemo-ray.html) |
| Async / decentralized RL | **PrimeIntellect/prime-rl** | See [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html) |
| Reward/verifier definitions | **math-verify**, **verifiers** | Robust answer checking and packaged RLVR environments |

At 100M, all of this is overkill — one process, one GPU, `model.generate()` inline. That is a *feature* of working small: you can see the whole loop.

## What post-training does and does not buy at 100M

It is worth stating the scope plainly, because the temptation with a shiny chat model is to over-claim.

**What post-training buys.** A base model that continues text becomes a model that *takes turns, follows instructions, stays in format, and stops*. That is entirely a post-training gift and it is transformational for *usability*: Stack-100M goes from "autocomplete" to "assistant-shaped." DPO — on *on-policy* pairs — sands off obvious quality failures. And RLVR genuinely lifts a *narrow, verifiable* skill above the base model's zero-shot rate: the one place at this scale where RL adds capability rather than just reshaping behavior.

**What it does not buy.** Post-training cannot install knowledge or reasoning the base model has no substrate for. A 100M model has, very roughly, on the order of a few tens of millions of "facts" worth of capacity; no amount of SFT or DPO conjures broad world knowledge, reliable multi-step reasoning, or robust instruction-following on out-of-distribution prompts. The failure modes are characteristic: confident wrong answers, brittleness to phrasings unlike the SFT set, and collapse the moment a task leaves the narrow band RLVR was trained on. This is *expected*, not a defect of our recipe — it is the reason the capstone's north star (Ch. 14.10) is a *narrow, scaffolded, tool-using* agent rather than a general chatbot. Post-training is what makes the narrow tool *usable*; the scale is what keeps it narrow.

The strategic reading, and the reason this ordering (SFT → DPO → narrow GRPO) is the right one for our budget: each stage is cheaper than the last and each depends on the previous. SFT gives DPO and GRPO a model that already produces the right *format* — so DPO's mined pairs are comparable and GRPO's verifier can parse an answer — and it is SFT that lifts the per-sample success rate $p$ out of the zone where two-thirds of RL groups are wasted. DPO gives a cleaner starting policy. GRPO then spends its narrow budget where verification is possible. Skip SFT and DPO has nothing coherent to prefer; skip both and GRPO's reward is 0 everywhere and RL never starts. The pipeline is a ladder, and every rung is load-bearing. This is exactly the shape of modern open post-training recipes — **Tülu 3** (Lambert et al., 2024) runs SFT → preference optimization → verifiable-reward RL in the same order, and we are simply running the 100M-scale, single-GPU edition of it.

!!! interview "Interview Corner"

    **Q:** You are asked to add a *verifiable* skill (say, arithmetic) to a small already-instruction-tuned model. Walk through why you would reach for GRPO/RLVR over just doing more SFT on correct solutions — and name the precondition that decides whether RLVR will work at all.

    **A:** SFT on correct solutions is *imitation*: it maximizes the likelihood of a fixed set of gold answers. It teaches the model to reproduce those specific solutions but gives no signal about the answers it generates itself, so it plateaus at "sound like the training solutions" and inherits their distribution. RLVR with GRPO optimizes a different objective — *be correct on new inputs*, graded by a program — so it can push the model's own sampling distribution toward whatever completions actually verify, including reasoning paths not in any SFT set. Concretely, GRPO samples a group of $G$ completions per prompt, grades each with the exact-match checker, standardizes the rewards within the group to get advantages ($\hat A_i=(R_i-\bar R)/\text{std}$), and reinforces high-advantage samples with a clipped PPO surrogate and a KL leash to the reference — no reward model, no critic. The decisive precondition is **non-degenerate reward variance within groups**: the model must already succeed *sometimes* (roughly 10–40%) so that groups contain both wins and losses. If it is right 0% (or 100%) of the time, every advantage is zero and the gradient vanishes — the cold-start problem. That is why RLVR is paired with an SFT warm-start and restricted to narrow tasks the base model can partially do. If the success rate is low but non-zero, the standard mitigation is DAPO's dynamic sampling: oversample prompts and discard every group with accuracy 0 or 1 so that every gradient step is spent on a group that actually carries signal.

    **Q:** Your DPO run shows the loss falling steadily and implicit-reward accuracy above 0.9, but the model's outputs are visibly worse than the SFT checkpoint. What do you check, in order?

    **A:** First, the two rewards *separately*, not just the margin: if `chosen_r` and `rejected_r` are both falling, the loss is being minimized by pushing everything down, which is degradation dressed as progress. Second, how on-policy the pairs are — compute the SFT model's mean per-token log-prob on the chosen responses; if it is far below what the model assigns to its own samples, the pairs are off-policy and the fix is to mine pairs from the model itself rather than to tune hyperparameters. Third, response length: summed sequence log-probs are length-biased, so check whether the chosen responses are systematically longer or shorter than the rejected ones, and switch to length-normalized DPO (or SimPO) if they are. Fourth, the learning rate and $\beta$ — DPO wants ~5e-7 to 1e-6 and $\beta \approx 0.1$; a large LR drives the loss down while shredding the policy. Only after those four would I look at the data quality of individual pairs.

!!! key "Key Takeaways"

    - Post-training on Stack-100M is three cheap stages — SFT, then DPO, then narrow GRPO — each under a couple of A100-hours, a rounding error against the ~USD 40–100 pretraining bill. Only GRPO scales with *generation*, which is why it is the most expensive of the three despite the fewest steps.
    - **SFT** installs the chat template using the reserved special tokens (`<|system|>/<|user|>/<|assistant|>/<|end|>`), packs conversations with document-aware masking, and supervises **only assistant tokens (including the closing `<|end|>`, so the model learns to stop)** at a small LR. Under assistant-only masking you must normalize the loss by the *accumulation window's* supervised-token count, not per microbatch — the grad-accum bug that shipped everywhere in 2024.
    - **DPO** optimizes a single logistic loss on preference *pairs* against a frozen SFT reference — no reward model, no rollouts, and the partition term cancels. At 100M the pairs must be **on-policy** (mined from your own SFT model's samples); UltraFeedback's 70B-authored responses are so far off-policy that the expected outcome is both chosen and rejected log-probs falling together.
    - Summed sequence log-probs are **length-biased**; length-normalized DPO, SimPO, ORPO (one-stage, reference-free), and KTO (unpaired feedback) are the named fixes, all available in TRL.
    - **GRPO/RLVR** replaces the reward model with a *program* (exact-match on `####` answers) and the critic with a *group baseline*: standardize rewards within a group of $G$ samples, broadcast the advantage to every token, and update with a clipped surrogate plus a k3-estimated KL leash. Dr. GRPO removes the std division *and* the per-sequence normalizer; the token-level normalizer we use is DAPO's — do not conflate them.
    - RLVR only climbs when groups have **mixed outcomes**. At $p=0.05$ and $G=8$, two-thirds of groups are degenerate; **DAPO's dynamic sampling** (oversample, discard accuracy-0 and accuracy-1 groups) is the standard fix — but it biases the training-accuracy curve, so evaluate on a held-out set.
    - Watch entropy: RLVR spends diversity to buy accuracy, and entropy collapse ends the run. Track degenerate-group fraction, clip fraction, mean log-prob per token, parse-failure rate, and KL-to-reference.
    - The ladder is load-bearing and ordered: SFT gives format *and* lifts $p$ into the RLVR-viable band; skip a rung and the next one has nothing to work with — the same SFT → preference → verifiable-RL order as Tülu 3, scaled to one GPU. Beyond one GPU the same three stages are TRL, then open-instruct or verl.

!!! sota "State of the Art & Resources (2026)"
    The SFT → DPO → RLVR ladder this chapter builds is now the industry-standard open post-training recipe; the open-source tooling that implements it at scale (TRL, open-instruct, verl) is exactly what you would reach for beyond a single GPU.

    **Foundational work**

    - [Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017)](https://arxiv.org/abs/1706.03741) — the pairwise-preference-to-reward idea that RLHF and, downstream, DPO both trace back to.
    - [Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN, 2021)](https://arxiv.org/abs/2109.01652) — established that instruction-formatted fine-tuning, not just scale, drives zero-shot instruction-following.
    - [Rafailov et al., *Direct Preference Optimization* (2023)](https://arxiv.org/abs/2305.18290) — the closed-form inversion of the KL-regularized objective that makes preference optimization a supervised loss.

    **Recent advances (2023–2026)**

    - [Zhou et al., *LIMA: Less Is More for Alignment* (2023)](https://arxiv.org/abs/2305.11206) — a thousand curated SFT examples beat a hundred thousand noisy ones; the justification for the compact SFT mix at 100M.
    - [Ding et al., *UltraChat* (2023)](https://arxiv.org/abs/2305.14233) — the multi-turn synthetic-dialogue dataset behind UltraChat-200k and much of the modern SFT-mix lineage.
    - [Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization* (2024)](https://arxiv.org/abs/2402.01306) — learns from *unpaired* good/bad labels, which is what production feedback logs actually contain.
    - [Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model* (2024)](https://arxiv.org/abs/2403.07691) — merges SFT and preference optimization into a single reference-free stage; very attractive at a 100M budget.
    - [Meng, Xia, Chen, *SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024)](https://arxiv.org/abs/2405.14734) — drops DPO's reference model by using *length-normalized* sequence log-probability as the implicit reward, which also removes the length bias analyzed above.
    - [Ahmadian et al., *Back to Basics: Revisiting REINFORCE-Style Optimization for RLHF* (2024)](https://arxiv.org/abs/2402.14740) — RLOO's leave-one-out baseline: unbiased where GRPO's std division is not.
    - [Lambert et al., *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* (2024)](https://arxiv.org/abs/2411.15124) — the reproducible SFT → (length-normalized) DPO → RLVR pipeline this chapter miniaturizes, and the source of the on-policy-preference-data finding.
    - [Ben Allal et al., *SmolLM2* (2025)](https://arxiv.org/abs/2502.02737) — documents the SmolTalk SFT mix referenced above and the full small-model pretrain-to-instruct pipeline.
    - [Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO, 2025)](https://arxiv.org/abs/2503.20783) — identifies GRPO's two biases: the std division in the advantage and the per-sequence length normalizer.
    - [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — clip-higher, **dynamic sampling**, token-level policy-gradient loss, and overlong reward shaping; the fix for the wasted-group problem this chapter quantifies.
    - [Zheng et al., *Group Sequence Policy Optimization* (GSPO, 2025)](https://arxiv.org/abs/2507.18071) — sequence-level, length-normalized importance ratios for stability on long rollouts and MoE policies.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — production `SFTTrainer`/`DPOTrainer`/`GRPOTrainer`/`ORPOTrainer`/`KTOTrainer`; the natural upgrade path from this chapter's from-scratch loops.
    - [allenai/open-instruct](https://github.com/allenai/open-instruct) — the actual Tülu 3 codebase: SFT, DPO, and RLVR stages chained exactly as this chapter narrows them to 100M.
    - [volcengine/verl](https://github.com/volcengine/verl) — a flexible, high-throughput RL-for-LLMs library (PPO, GRPO, RLOO, and more) for when rollouts need to scale past one GPU.
    - [huggingface/Math-Verify](https://github.com/huggingface/Math-Verify) — robust symbolic checking of LaTeX/expression answers; what a real math verifier looks like once `re.compile(r"####")` stops being enough.
    - [willccbb/verifiers](https://github.com/willccbb/verifiers) — packaged RLVR environments and reward rubrics that plug into TRL/verl-style trainers.
    - [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) — the ~1.1M-conversation SFT dataset this chapter's data-source note points to.
    - [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — the standard binarized preference set; best used here as a *prompt* source for on-policy mining.

    **Go deeper**

    - [The RLHF Book](https://rlhfbook.com/) — Nathan Lambert's freely-readable, continuously updated book covering RLHF and post-training end to end, from reward modeling through direct alignment algorithms to RLVR.
    - [Tülu 3: The Next Era in Open Post-Training](https://allenai.org/blog/tulu-3-technical) — AI2's write-up of the SFT → DPO → RLVR pipeline this chapter's single-GPU recipe is a miniature of.

## Further reading

- Rafailov, Sharma, Mitchell, Manning, Ermon, Finn — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023). The DPO derivation and loss used here.
- Shao et al. — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024). Introduces GRPO.
- DeepSeek-AI — *DeepSeek-R1* (2025). RLVR/GRPO at scale for reasoning; the recipe this chapter narrows to 100M.
- Yu et al. — *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025). Dynamic sampling, clip-higher, token-level loss, overlong shaping.
- Liu et al. — *Understanding R1-Zero-Like Training: A Critical Perspective* (2025). The Dr. GRPO corrections, stated precisely.
- Zheng et al. — *Group Sequence Policy Optimization* (2025). Sequence-level importance ratios.
- Ouyang et al. — *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, 2022). The full PPO-RLHF pipeline we deliberately avoid at this budget.
- Ahmadian et al. — *Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback* (2024). RLOO and the critic-free baseline family.
- Lambert et al. — *Tülu 3: Pushing Frontiers in Open Language Model Post-Training* (2024). A modern, reproducible SFT → DPO → RLVR post-training pipeline.
- Hong et al. — *ORPO: Monolithic Preference Optimization without Reference Model* (2024); Ethayarajh et al. — *KTO* (2024); Meng et al. — *SimPO* (2024). The direct-alignment variant family.
- Zhou et al. — *LIMA: Less Is More for Alignment* (2023). Why a small, clean SFT set wins.
- Cui et al. — *UltraFeedback: Boosting Language Models with High-quality Feedback* (2023). A standard open source of DPO preference pairs (and, better, of prompts).
- Cobbe et al. — *Training Verifiers to Solve Math Word Problems* (GSM8K, 2021). Source of the `####` final-answer convention our verifier parses.
- Schulman et al. — *Proximal Policy Optimization Algorithms* (2017), and Schulman's note on KL estimators (the k3 estimator used in the GRPO loop).

## Exercises

**1.** In `render_conversation`, the closing `<|end|>` of an *assistant* turn is emitted with `supervised=True`, but the `<|assistant|>` role marker is emitted with `supervised=False`. Explain, in behavioral terms, what would go wrong at inference time if you flipped *each* of these two choices: (a) masking the assistant `<|end|>`, and (b) supervising the `<|assistant|>` marker. (c) Separately: `emit()` calls `tok.encode(text, add_special_tokens=False)`. What attack does that one keyword argument prevent?

??? note "Solution"
    Both of the first two choices are about *which* tokens receive gradient, and each controls a distinct behavior the chapter identifies.

    (a) **Masking the assistant `<|end|>`** removes the only gradient that teaches the model to *stop*. The closing `<|end|>` is the token that terminates the assistant turn; if the model never receives loss on producing it, it is never trained to emit it after finishing an answer. At inference the server decodes until it sees `<|end|>` (or `<|eos|>`), so a model that never learned to emit that token "runs past the end of its answer into hallucinated user turns" — the *never shuts up* failure.

    (b) **Supervising the `<|assistant|>` marker** teaches the model to *emit the role marker itself*. But the harness is responsible for emitting `<|assistant|>` to cue generation; the model's job is to learn what comes *after* it. If we put gradient on the marker, the model learns to spontaneously produce `<|assistant|>` mid-turn, corrupting the turn structure the template exists to enforce.

    In short: supervise the token that ends a turn (so the model stops), mask the token that starts the assistant turn (so the model does not impersonate the harness).

    (c) With `add_special_tokens=False`, the encoder does *not* treat reserved special-token strings as atomic; a user who types the literal characters `<|end|><|assistant|>` gets ordinary byte-level BPE tokens for `<`, `|`, `end`, … and never the control ids. That blocks **role-marker injection**: without it, a user could close their own turn and open a fake assistant turn inside their message, making the model treat attacker-authored text as if it were its own prior output (or as a system instruction). This is the concrete payoff of reserving special tokens as atomic ids at tokenizer-training time rather than pattern-matching them in text — the trust boundary lives in the tokenizer, where it can be enforced, not in a regex.

**2.** DPO with $\beta = 0.1$ on a single preference pair. The frozen reference assigns response log-likelihoods $\log\pi_{\text{ref}}(y_w) = -26.0$ and $\log\pi_{\text{ref}}(y_l) = -28.0$. After some training the policy assigns $\log\pi_\theta(y_w) = -25.0$ and $\log\pi_\theta(y_l) = -31.0$.
(a) Compute the winner and loser log-ratios, the DPO margin, and the loss $-\log\sigma(\text{margin})$.
(b) Compute the implicit `chosen_reward` and `rejected_reward` diagnostics ($\beta \times$ log-ratio). Is the implicit-reward accuracy 0 or 1 for this pair?
(c) Now suppose instead the policy had drifted to $\log\pi_\theta(y_w) = -35.0$ and $\log\pi_\theta(y_l) = -40.0$. Recompute the margin and both rewards. What failure mode does this illustrate, and which of the chapter's data-sourcing recommendations most directly prevents it?

??? note "Solution"
    (a) Log-ratios are $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ for each response:

    - winner: $-25.0 - (-26.0) = +1.0$
    - loser: $-31.0 - (-28.0) = -3.0$

    Margin $= \beta(\text{winner} - \text{loser}) = 0.1 \times (1.0 - (-3.0)) = 0.1 \times 4.0 = 0.40$.

    Loss $= -\log\sigma(0.40)$. With $\sigma(0.40) = 1/(1 + e^{-0.40}) = 1/(1 + 0.6703) = 0.5987$, the loss is $-\log(0.5987) = 0.513$ nats.

    (b) `chosen_reward` $= \beta \times (+1.0) = +0.10$; `rejected_reward` $= \beta \times (-3.0) = -0.30$. The chosen reward is up and the rejected reward is down — exactly the healthy trend. Since the margin $0.40 > 0$, the implicit-reward accuracy for this pair is $1$.

    (c) New log-ratios: winner $-35.0 - (-26.0) = -9.0$; loser $-40.0 - (-28.0) = -12.0$. Margin $= 0.1 \times (-9.0 - (-12.0)) = 0.1 \times 3.0 = 0.30 > 0$, so the *loss still decreases* and accuracy is still $1$. But now `chosen_reward` $= 0.1 \times (-9.0) = -0.90$ and `rejected_reward` $= 0.1 \times (-12.0) = -1.20$: **both rewards have fallen**. The logistic loss only cares about the *difference*, so it is perfectly happy to push the winner's absolute log-probability down as long as it pushes the loser's down faster. The model is degrading while loss and accuracy look fine.

    The most direct prevention is **on-policy pair mining**. This failure is driven by pairs the policy assigns near-zero probability to: with both responses far off the model's distribution, the cheapest way to open a margin is to suppress shared structure, dragging the winner down too. Pairs sampled from the model itself start with log-ratios near zero and stay in a region where raising $\log\pi_\theta(y_w)$ is actually achievable. (A tiny LR and $\beta\approx0.1$ slow the damage; on-policy data removes the pressure.)

**3.** A single GRPO group on one prompt, $G = 5$, exact-match reward. The graded rewards come back as $R = [1, 0, 0, 1, 0]$.
(a) Compute the group mean and the *population* standard deviation, then the standardized advantage assigned to a correct sample and to an incorrect sample. Verify the advantages sum to zero.
(b) The code uses `torch.std` (Bessel-corrected, divides by $G-1$). Recompute the std and the two advantages under that convention.
(c) If the same prompt had instead returned $R = [1,1,1,1,1]$, what advantage does every token receive, and how much does this prompt contribute to the gradient?
(d) Recompute (a) under **Dr. GRPO**'s unnormalized advantage $\hat A_i = R_i - \bar R$, and under **RLOO**'s leave-one-out baseline $\hat A_i = R_i - \frac{1}{G-1}\sum_{j\ne i}R_j$. Which of the three treats the correct and incorrect samples most asymmetrically?

??? note "Solution"
    (a) Mean $\bar R = 2/5 = 0.40$. Population variance $= \bar R(1 - \bar R) = 0.40 \times 0.60 = 0.24$ (valid because the rewards are 0/1), so population std $= \sqrt{0.24} = 0.4899$.

    - correct sample: $\hat A = (1 - 0.40)/0.4899 = 0.60/0.4899 = +1.225$
    - incorrect sample: $\hat A = (0 - 0.40)/0.4899 = -0.40/0.4899 = -0.8165$

    Sum: $2(+1.225) + 3(-0.8165) = 2.449 - 2.449 = 0$. Advantages sum to zero, as standardization guarantees.

    (b) Bessel-corrected variance $= \frac{1}{G-1}\sum (R_i - \bar R)^2 = \frac{1}{4}\big[2(0.6)^2 + 3(0.4)^2\big] = \frac{1}{4}(0.72 + 0.48) = \frac{1.20}{4} = 0.30$, so std $= \sqrt{0.30} = 0.5477$.

    - correct: $\hat A = 0.60/0.5477 = +1.095$
    - incorrect: $\hat A = -0.40/0.5477 = -0.730$

    Same signs and ranking; only the magnitude shifts.

    (c) With $R = [1,1,1,1,1]$: $\bar R = 1$, std $= 0$. Every advantage is $(1 - 1)/(0 + \varepsilon) = 0$. Every token gets advantage $0$, so the surrogate is zero and this prompt contributes **no gradient**. This is the all-correct degenerate case (mirror image of all-wrong): with no reward variance in the group there is nothing to reinforce or suppress. Only *mixed* groups teach anything — and DAPO's dynamic sampling exists precisely so you never pay for a rollout like this one.

    (d) **Dr. GRPO**: $\hat A = 1 - 0.4 = +0.60$ for a correct sample, $0 - 0.4 = -0.40$ for an incorrect one. **RLOO**: for a correct sample the leave-one-out mean over the other four is $1/4 = 0.25$, so $\hat A = 1 - 0.25 = +0.75$; for an incorrect sample the leave-one-out mean is $2/4 = 0.50$, so $\hat A = 0 - 0.50 = -0.50$. All three schemes give correct samples a positive advantage of larger magnitude than the incorrect samples' negative one — the ratio is $1.225/0.8165 = 1.5$ for GRPO, $0.60/0.40 = 1.5$ for Dr. GRPO, $0.75/0.50 = 1.5$ for RLOO. The asymmetry is identical (it is just $(1-\bar R)/\bar R$); what differs is the *scale*, and therefore how this prompt is weighted against other prompts in the same batch. Only GRPO's std division makes that scale depend on the prompt's difficulty — which is the bias Dr. GRPO objects to and the curriculum effect practitioners often keep.

**4.** The chapter's "one free lunch" tip says to cache the reference log-probs once, since $\pi_{\text{ref}}$ is frozen and the preference set is static. Implement a `precompute_ref_logprobs` pass and a cache-consuming `dpo_loss_cached`, consistent with `stacklm/post/dpo.py`. State the one constraint the data loader must satisfy for the cache to be valid, and say how many *policy* forward passes per step this saves.

??? note "Solution"
    ```python
    # capstone/stacklm/post/dpo.py  (continued)
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
            ch_ids = batch["chosen_ids"].to(device);   ch_m = batch["chosen_mask"].to(device)
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
        ch_ids = batch["chosen_ids"].to(device);   ch_m = batch["chosen_mask"].to(device)
        rj_ids = batch["rejected_ids"].to(device); rj_m = batch["rejected_mask"].to(device)

        pi_ch = sequence_logprob(policy, ch_ids, ch_m)          # differentiable
        pi_rj = sequence_logprob(policy, rj_ids, rj_m)
        ref_ch = ref_ch.to(device); ref_rj = ref_rj.to(device)  # cached, no grad

        chosen_logratio   = pi_ch - ref_ch
        rejected_logratio = pi_rj - ref_rj
        logits = beta * (chosen_logratio - rejected_logratio)
        return -F.logsigmoid(logits).mean()
    ```

    **Constraint:** the cache is indexed by pair position, so the loader must present pairs in a *fixed, reproducible order* — i.e. **no shuffling** (this is why the `PreferenceDataset` usage snippet passes `shuffle=False`) — or, equivalently, key the cache by a stable pair id and look up by that id each step. If the loader shuffles between the precompute pass and training, `ref_ch[k]` no longer corresponds to the $k$-th pair the training loop sees, and every log-ratio is silently mismatched. A second, easily-missed constraint: **padding must be deterministic too**. `collate_preferences` pads to the *batch* maximum, so re-batching with a different `batch_size` changes each sequence's padded length; the masked sum is unaffected, but any code that keys the cache by shape will break.

    **Savings:** the original `dpo_loss` does four forwards per step (policy chosen, policy rejected, ref chosen, ref rejected). Caching removes the two reference forwards, leaving **two policy forwards per step** — the reference passes are amortized into a single pre-pass — and frees ~100M parameters of VRAM once `ref` is deleted. TRL's `DPOConfig(precompute_ref_log_probs=True)` implements exactly this.

**5.** The cold-start trap, quantitatively. Model a group of $G$ i.i.d. samples where each succeeds independently with probability $p$. A group produces *zero gradient* exactly when it is degenerate — all correct or all wrong.
(a) Write the probability that a group is degenerate as a function of $p$ and $G$.
(b) Evaluate it for $G = 8$ at a "sweet spot" success rate $p = 0.30$ and at a cold-start rate $p = 0.05$.
(c) Interpret: what fraction of prompts is "wasted" in each regime, and why does this make the SFT warm-start non-optional?
(d) Under DAPO's dynamic sampling we keep drawing prompts until we have $N$ non-degenerate groups. Derive the expected number of prompts sampled, and evaluate it at $p=0.30$ and $p=0.05$ for $N=16$, $G=8$. What does this say about the `oversample=3.0` budget cap in `collect_nondegenerate_groups`?

??? note "Solution"
    (a) A group is all-wrong with probability $(1-p)^G$ and all-right with probability $p^G$; these are disjoint, so

    $$P_{\text{degenerate}}(p, G) = (1-p)^G + p^G.$$

    (b) For $G = 8$:

    - $p = 0.30$: $(0.70)^8 + (0.30)^8 = 0.05765 + 0.0000656 \approx 0.0577$, i.e. about **5.8%** of groups are degenerate.
    - $p = 0.05$: $(0.95)^8 + (0.05)^8 = 0.6634 + (\sim\!4\times10^{-11}) \approx 0.663$, i.e. about **66%** of groups are degenerate.

    (c) At the sweet-spot rate $p = 0.30$, only ~6% of prompts give zero gradient, so ~94% of the compute spent on rollouts actually produces a learning signal. At the cold-start rate $p = 0.05$, two-thirds of all groups are all-wrong, so two-thirds of the rollouts are wasted and the effective learning signal is throttled to a trickle; as $p \to 0$ the wasted fraction $\to 1$ and the gradient vanishes entirely. This is why the SFT warm-start is non-optional: SFT (plus math-heavy mid-training and a few thousand `####` format exemplars) lifts the base model's success rate from near-zero into the 10–40% band where groups are *mixed* often enough that the reward signal has variance to exploit. RL can only reinforce behavior the model already sometimes produces; SFT is what puts $p$ into the range where "sometimes" is frequent enough to bootstrap.

    (d) Each prompt is non-degenerate independently with probability $q = 1 - P_{\text{degenerate}}$, so the number of prompts needed to collect $N$ of them is negative-binomial with expectation

    $$\mathbb{E}[\text{prompts}] = N/q.$$

    - $p = 0.30$: $q = 0.942$, so $\mathbb{E} = 16/0.942 = 17.0$ prompts — a 6% surcharge, essentially free.
    - $p = 0.05$: $q = 0.337$, so $\mathbb{E} = 16/0.337 = 47.5$ prompts — a **3× surcharge** in generation compute.

    Two readings. First, dynamic sampling is *cheap exactly where you do not need it and expensive exactly where you do* — but the expensive case is the one where the alternative (plain GRPO) is taking two-thirds of its gradient steps on zeros, so you are paying generation to buy real gradient rather than paying it for nothing. Second, this is precisely why `collect_nondegenerate_groups` caps the budget at `oversample=3.0`: at $p=0.05$ the expected cost is right at the cap, so the run completes but the returned `tries/len(kept)` ratio screams that the task is out of reach. If you hit the cap, the answer is not a bigger budget; it is more SFT, easier prompts, or a curriculum (see [RL Data, Curriculum & Replay Management](../06-rl-infra/12-rl-data-curriculum-replay.html)).

**6.** The chapter warns that "format drift and reward hacking creep in" and suggests a small format penalty on top of the exact-match reward. Write a custom `reward_fn` for `grpo_train` that (i) keeps exact-match correctness as the dominant term and (ii) mildly penalizes emitting the wrong number of `####` markers (zero, or more than one). Explain why the *magnitude* of the shaping term must stay small relative to the correctness reward, and why within-group standardization limits how much a constant format bonus can distort learning.

??? note "Solution"
    `grpo_train` takes `reward_fn=` (defaulting to `exact_match_reward`), and any callable with the signature `(completion_text, gold) -> (reward, parsed)` drops in. No new module is needed:

    ```python
    # anywhere that imports the package — e.g. capstone/scripts/run_grpo.py
    import re
    from stacklm.post.grpo import exact_match_reward, grpo_train

    _MARKER = re.compile(r"####")

    def marker_shaped_reward(completion_text, gold_answer, fmt_weight=0.1):
        """
        Exact-match correctness (0/1) plus a SMALL shaping term that rewards
        emitting exactly one well-formed answer marker and penalizes zero or many.
        Correctness stays dominant: fmt_weight (0.1) << the 1.0 correctness gap.
        """
        r_correct, pred = exact_match_reward(completion_text, gold_answer)
        n_markers = len(_MARKER.findall(completion_text))
        fmt = fmt_weight if n_markers == 1 else -fmt_weight
        return r_correct + fmt, pred

    policy, stats = grpo_train(sft_model, tok, reward_fn=marker_shaped_reward)
    ```

    (The shipped `stacklm.post.grpo.shaped_reward` is a slightly different variant of the same idea: it adds a small bonus for containing `####` and another for containing any integer, giving a denser early signal when the model has not yet learned the format at all. Either is fine; the design rules below apply to both.)

    **Why the shaping magnitude must stay small.** GRPO reinforces whatever raises reward. If the format bonus is comparable to (or larger than) the correctness gap of $1.0$, then a completion that has the right format but the *wrong answer* can out-score, or tie, a differently-formatted *correct* one — the model can maximize reward by getting the format right and the arithmetic wrong. That is textbook reward hacking: optimizing the proxy (format) instead of the goal (correctness). Keeping $\text{fmt\_weight} = 0.1 \ll 1.0$ ensures correctness always dominates the ranking within a group, so the format term only breaks ties among equally-correct (or equally-wrong) samples.

    **Why standardization limits the damage.** The advantage is the reward *standardized within the group*, $\hat A_i = (R_i - \bar R)/(\text{std} + \varepsilon)$. Any component of the reward that is *constant across the group* — e.g. if all $G$ samples already emit exactly one marker, every sample gets the same $+0.1$ — shifts the mean by that same amount and *cancels* in $R_i - \bar R$, contributing nothing to the advantage. The format term therefore only produces gradient when samples in the group *differ* in their format, which is exactly the drift we want to correct. Combined with the KL leash to the SFT reference ($\text{kl\_beta} \approx 0.02$–$0.05$), this keeps the shaping honest.

    **One caveat the standardization argument hides.** Shaping *does* change which groups are degenerate. A group where all 8 samples are wrong has zero variance under `exact_match_reward` and is discarded by dynamic sampling; under a shaped reward, if some of those 8 got the format right and some did not, the group now has variance and will be kept — spending a gradient step teaching format rather than arithmetic. Early in a run that is exactly what you want. Late in a run it is wasted compute, which is why shaping weights are usually annealed toward zero as the parse-failure rate drops.

**7.** The SFT loop accumulates `reduction="sum"` losses and divides the *gradients* by the window's supervised-token count, rather than the more familiar `loss / grad_accum`. Consider an accumulation window of `grad_accum=2` microbatches, where microbatch A has 900 supervised tokens with mean per-token loss 2.0, and microbatch B has 100 supervised tokens with mean per-token loss 4.0.
(a) What per-token loss does the correct (window-normalized) objective report, and what does the naive `mean/grad_accum` scheme effectively report?
(b) By what factor does the naive scheme over-weight microbatch B's gradient relative to the correct objective?
(c) The DPO loop *does* use `loss / grad_accum`. Why is that correct there but not here?

??? note "Solution"
    (a) The correct objective is the sum over all supervised tokens divided by their count:

    $$\frac{900\times 2.0 + 100\times 4.0}{900 + 100} = \frac{1800 + 400}{1000} = \frac{2200}{1000} = 2.20 \text{ nats/token}.$$

    The naive scheme averages the two microbatch *means* with equal weight: $(2.0 + 4.0)/2 = 3.00$ nats/token. It reports a number that is 36% too high, and — far worse — the *gradient* it produces is the equally-weighted average of the two microbatch mean-gradients rather than the token-weighted one.

    (b) Under the correct objective microbatch B carries weight $100/1000 = 0.10$; under the naive scheme it carries $0.5$. B's gradient is over-weighted by a factor of $0.5/0.10 = \mathbf{5\times}$. Symmetrically, A is under-weighted by $0.5/0.9 = 0.56\times$. Since assistant-token counts per packed window routinely vary by 10× under assistant-only masking, distortions of this size are the norm, not the worst case — and the distortion changes whenever you change `grad_accum` or the batch size, so a "reproducible" config silently trains a different objective on a different GPU count.

    (c) Because the validity condition for `loss / grad_accum` is that **every microbatch's denominator is the same**. `dpo_loss` returns a mean over *preference pairs*, and every microbatch contains exactly `batch_size` pairs, so averaging the microbatch means is exactly the window mean. In SFT the denominator is the *supervised-token count*, which varies per window by an order of magnitude, so the two are not equal. The rule generalizes: whenever a loss is a mean over a variable-size population (tokens, valid positions, non-padding elements), accumulate sums and normalize once at the step boundary.

    Finally, note the practical consequence for logging: `history.append(win_loss / win_tok)` reports true nats/token over the accumulation window, which is directly comparable to the pretraining loss curve from Ch. 14.7. The naive scheme's logged value is not comparable to anything, including itself at a different `grad_accum`.
