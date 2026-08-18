# 15.5 Post-Training With TRL and the alignment-handbook

In [Chapter 14.9](../14-capstone/09-post-training.html) we post-trained **Stack-100M** with about 400 lines of our own PyTorch: a chat-template renderer, an assistant-only loss mask, a DPO loss written out term by term, and a GRPO loop that generated its own rollouts, graded them with an exact-match reward, and reinforced what worked. Every gradient was visible. That was the point — you cannot debug a preference-optimization run whose loss you have never derived, and you cannot reason about a reward-hacking failure ([Ch. 5.13](../05-posttraining-alignment/13-reward-hacking-failures.html)) if `DPOTrainer` is an opaque box.

This chapter rebuilds those same three stages — **SFT, DPO, GRPO** — with the library a working post-training engineer actually reaches for: **TRL** (Transformer Reinforcement Learning), Hugging Face's post-training library, plus the **alignment-handbook** recipes that wrap it into reproducible YAML, and the scale-out systems (**veRL**, **OpenRLHF**) you graduate to when 100M becomes 100B. The thesis of Part XV holds here more sharply than anywhere else: **you hand-roll to learn the mechanism; you reach for TRL to ship**, because the library has already absorbed a hundred bug-fixes you would otherwise rediscover — the off-by-one in the DPO reference logprobs, the loss-mask that leaks the prompt, the vLLM weight-sync race in GRPO.

We build directly on the from-first-principles chapters — keep [SFT & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [Chat Templates & Packing](../05-posttraining-alignment/02-chat-templates-packing.html), [DPO & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html), [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html), and [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html) open — and we do **not** re-derive their math. Here we map each derivation onto a real `Trainer` call, one config field at a time, and we are honest about the thing that bites everyone: TRL's flag names move between releases.

!!! warning "Version drift is the whole game — pin, then verify"

    TRL is a fast-moving library; between roughly 0.9 and 0.20 (2024–2026) several flags were **renamed or relocated**: the SFT trainer's `tokenizer=` became `processing_class=`, `max_seq_length` moved out of the trainer signature into `SFTConfig(max_length=...)`, `dataset_text_field` semantics shifted, and `assistant_only_loss` and the vLLM-server GRPO mode were added midway. **Pin an exact version in your lockfile** (`trl==<x.y.z>`, `transformers`, `accelerate`, `peft`, `datasets`, `vllm` all pinned together) and, before trusting any snippet in this chapter, print the dataclass fields your installed version actually exposes:

    ```bash
    python -c "from trl import SFTConfig; print(sorted(SFTConfig.__dataclass_fields__))"
    python -c "from trl import GRPOConfig; print(sorted(GRPOConfig.__dataclass_fields__))"
    ```

    When a field below is missing in your version, the upstream `trl/examples/scripts/` and the `SFTConfig`/`DPOConfig`/`GRPOConfig` docstrings are the ground truth, not this book.

## The crosswalk: from-scratch loop → TRL Trainer

Every TRL trainer is a subclass of `transformers.Trainer`. That single fact explains most of the ergonomics: you get gradient accumulation, mixed precision, `accelerate`/DeepSpeed/FSDP integration, checkpoint-and-resume, and logging **for free**, and you only supply what is *special* about the post-training objective — the data collator, the loss, and (for GRPO) the rollout engine. Here is the map from what we wrote by hand to what the library owns.

| Stage | Our from-scratch code (Ch. 14.9) | TRL object | What TRL owns that we hand-wrote |
|---|---|---|---|
| SFT | `post/sft.py`: manual loss mask, LR lambda, accum loop | `SFTTrainer` + `SFTConfig` | Chat-template application, assistant-only masking, packing with a boundary-aware collator, the whole training loop |
| DPO | `post/dpo.py`: 4 forwards, log-ratio loss | `DPOTrainer` + `DPOConfig` | Reference-logprob precompute+cache, 12+ loss variants (`sigmoid`, `hinge`, `ipo`, …), length-normalization flags |
| GRPO | `post/grpo.py`: generate → reward → group-normalize → policy-grad | `GRPOTrainer` + `GRPOConfig` | vLLM-backed rollouts, group advantage, KL-to-ref, async generation/weight-sync |
| Recipe | argparse defaults | **alignment-handbook** YAML + `accelerate launch` | Reproducible configs, multi-GPU DeepSpeed/FSDP recipes |
| Gate | our tiny probe suite | **lm-evaluation-harness** | Standardized tasks, few-shot templating, stderr/CI on every metric |

Before the bridge, one thing that saves hours: **each trainer expects a specific dataset schema**, and feeding the wrong shape produces confusing errors or, worse, silent mis-training.

!!! note "The three dataset schemas TRL expects"

    - **SFT** wants either a **conversational** row `{"messages": [{"role","content"}, ...]}` (TRL applies `tok.chat_template`) or a plain-text row `{"text": "..."}` (no template). Prompt-completion rows `{"prompt", "completion"}` are also accepted and trigger `completion_only_loss`.
    - **DPO** wants a **preference triple** `{"prompt", "chosen", "rejected"}`, where each field is either a string or a conversational message list. `chosen`/`rejected` are the *completions*, not full conversations — a frequent mistake is to include the prompt inside `chosen`, which double-counts it.
    - **GRPO** wants only a **prompt** row `{"prompt": ...}` plus whatever extra columns your reward function needs (e.g. `answer`) — there are no target completions, because the model generates them and the reward grades them. TRL passes the extra columns to `reward_funcs` as keyword lists.

    Get the schema right and most of TRL "just works"; get it wrong and you will chase a `KeyError` or, far worse, a model that trained on the wrong tokens without any error at all.

One bridge has to be crossed before any of this runs. TRL trainers expect a Hugging Face `PreTrainedModel` — an `AutoModelForCausalLM` with a `config`, a `generate()`, and a chat-aware tokenizer. Our hand-rolled `Stack100M` (a bare `nn.Module` whose `forward` returns `(logits, loss)`) is **not** that. The clean move is to export Stack-100M into a HF-compatible checkpoint.

```python
# capstone/stacklm/export/to_hf.py
# Stack-100M's components (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings) — and, the
# distinguishing detail, QK-norm as an RMSNorm over head_dim applied BEFORE RoPE —
# are exactly the Qwen3 recipe, which is why Ch. 14.4's canonical exporter targets
# Qwen3ForCausalLM (a Llama config has no q_norm/k_norm at all). The ONE thing that
# does not map is NoPE-on-every-4th-layer: with nope_every=0 this is a pure key
# rename; keep NoPE and you owe a modeling file loaded with trust_remote_code=True.
from transformers import Qwen3Config, Qwen3ForCausalLM
import torch

def stacklm_to_qwen3(stack_model, tokenizer_len=32768):
    cfg = Qwen3Config(
        vocab_size=tokenizer_len,
        hidden_size=512,
        intermediate_size=1408,      # our SwiGLU inner dim
        num_hidden_layers=30,
        num_attention_heads=8,
        num_key_value_heads=2,       # GQA 4:1
        head_dim=64,
        max_position_embeddings=8192,
        rope_theta=41830.0,          # the NTK-rescaled base mid-training left in the
                                     # checkpoint (Ch. 14.8), NOT the pretrain 10000 —
                                     # read rope_theta/max_seq_len off the saved config
                                     # rather than retyping defaults, or the export
                                     # silently rotates positions at the wrong rate.
        rms_norm_eps=1e-5,
        attention_bias=False,
        tie_word_embeddings=True,    # Press & Wolf, saves 16.8M params
    )
    hf = Qwen3ForCausalLM(cfg)
    # to_qwen3's rename map (Ch. 14.4, `stacklm/serve/export_hf.py`) is the canonical
    # one: ~15 lines of renames per block, including self_attn.q_norm/k_norm, which
    # Qwen3 carries and Llama does not — drop them and you have silently changed the
    # architecture, not just the weights. Load non-strictly and re-tie, because
    # lm_head.weight is tied to model.embed_tokens.weight and so is absent from the map.
    hf.load_state_dict(rename_to_qwen3(stack_model.state_dict()), strict=False)
    hf.tie_weights()
    # Then EARN the export: assert logits match on a fixed batch before trusting it.
    return hf
```

!!! note "The custom-architecture escape hatch"

    QK-norm *is* expressible — that is the whole reason Qwen3 is the export target ([Ch. 14.4](../14-capstone/04-architecture.html)). **NoPE-on-every-4th-layer is the one thing that is not**: no stock architecture supports "skip RoPE on this layer." Two honest options: (1) ship a `modeling_stacklm.py` next to the checkpoint and load with `trust_remote_code=True` — TRL trains any `PreTrainedModel`, custom or not; or (2) train the run you intend to post-train with `nope_every=0`, so the Qwen3 export is a pure key rename and the whole ecosystem opens up. We take (1) for the flagship and note where (2) is fine. Either way, do not drop a component and pretend the exported model is identical — verify logits match on a fixed batch first.

## SFT with `SFTTrainer`: chat template, packing, assistant-only loss

SFT teaches *format and instinct* — "when you see a user turn, emit an assistant turn, then stop." In [Ch. 14.9](../14-capstone/09-post-training.html) we did three things by hand that `SFTTrainer` now does for us: (1) render conversations through the chat template, (2) mask the loss to assistant tokens only, and (3) pack short conversations so we do not waste compute padding. Let us do all three the library way.

### The chat template lives on the tokenizer

TRL applies whatever chat template your tokenizer carries. A chat template is a **Jinja string stored in `tokenizer.chat_template`** that turns a list of `{"role", "content"}` dicts into the exact token stream the model was trained on. We reserved `<|system|> <|user|> <|assistant|> <|end|>` at tokenizer-training time in [Ch. 14.3](../14-capstone/03-tokenizer.html) precisely so each turn boundary is one atomic token. The template must also emit the `{% generation %}` markers that let TRL recover *which tokens are assistant tokens* for masking:

```python
# capstone/stacklm/post/chat_template.py
# A ChatML-like template. The {% generation %}...{% endgeneration %} block is the
# critical bit: tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)
# uses it to return a 0/1 mask that SFTConfig(assistant_only_loss=True) consumes.
STACK_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'assistant' %}"
    "<|assistant|>{% generation %}{{ m['content'] }}<|end|>{% endgeneration %}"
    "{% else %}"
    "<|{{ m['role'] }}|>{{ m['content'] }}<|end|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./stack-100m-base")
tok.chat_template = STACK_CHAT_TEMPLATE
# Sanity check the mask BEFORE training — this is the #1 silent SFT bug:
enc = tok.apply_chat_template(
    [{"role": "user", "content": "What is 17 times 4?"},
     {"role": "assistant", "content": "17 times 4 is 68."}],
    return_assistant_tokens_mask=True, return_dict=True, tokenize=True,
)
assert sum(enc["assistant_masks"]) > 0, "template emits no assistant tokens — check {% generation %}"
```

### `assistant_only_loss`, packing, and the config

Now the trainer. Compare this to the ~120-line loop of `post/sft.py`: the mechanism is identical (cross-entropy on next-token prediction, loss zeroed on prompt tokens), but every knob is a config field.

```python
# capstone/stacklm/post/sft_trl.py
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Conversational dataset: each row is {"messages": [ {role, content}, ... ]}.
# SFTTrainer auto-detects the conversational format and applies tok.chat_template.
ds = load_dataset("json", data_files="data/sft/stack_chat.jsonl", split="train")

args = SFTConfig(
    output_dir="ckpts/stack-100m-sft",
    # --- what makes this SFT and not raw LM training ---
    assistant_only_loss=True,     # mask loss to {% generation %} spans only
    packing=True,                 # concatenate short convos to fill max_length...
    packing_strategy="bfd",       # ...with a boundary-aware (best-fit-decreasing) packer
                                  #    so packed examples still respect turn boundaries;
                                  #    with flash-attn the collator also blocks
                                  #    cross-document attention (no attending across the join)
    max_length=2048,              # matches Stack-100M pretrain context
    # --- ordinary Trainer knobs, free from the base class ---
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,   # effective batch ~ 16*4*2048 tokens/GPU
    num_train_epochs=3,
    learning_rate=2e-5,              # small: we are nudging a converged model
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
)

trainer = SFTTrainer(
    model="./stack-100m-base",       # path or a PreTrainedModel instance
    args=args,
    train_dataset=ds,
    processing_class=tok,            # NOTE: was `tokenizer=` in older TRL
)
trainer.train()
trainer.save_model()                 # writes a HF checkpoint you can serve with vLLM (Ch. 15.6)
```

### What the collator actually hands the model

It is worth seeing the tensor `SFTTrainer` builds, because it is exactly the object our `post/sft.py` assembled by hand — just produced by a battle-tested collator instead. For the two-turn example above, the collator emits `input_ids`, a `labels` tensor that is a *copy* of `input_ids` — same length, same alignment, since the causal LM shifts internally when it computes the loss — with **prompt/user positions set to `-100`** (the ignore index PyTorch's cross-entropy skips), and, under packing with flash-attention, the block-diagonal attention metadata:

```python
# Conceptually what SFTTrainer's collator produces for assistant_only_loss=True.
# -100 is torch.nn.CrossEntropyLoss's ignore_index: those positions contribute
# zero loss and zero gradient. This is the ONLY difference between SFT and raw LM.
input_ids = [ <|bos|> <|system|> ...sys... <|end|> <|user|> ...q... <|end|>
              <|assistant|> ...answer... <|end|> ]
labels    = [ -100     -100        -100     -100    -100     -100  -100
              -100         ...answer...     <|end|> ]   # only assistant span + its <|end|>
#             ^ everything up to and including the <|assistant|> tag is masked;
#               the model learns "produce THIS answer and THIS terminator", nothing else.
```

The reason to let the library own this is not that the mask is hard — it is ten lines — but that the *interactions* are: the `<|assistant|>` tag itself must be masked (it is the cue, not the target), the terminating `<|end|>` must be **un**masked (the model has to learn to stop), and under packing every one of those boundaries has to stay aligned as sequences are concatenated. Each is a one-character bug that never shows up in the loss curve. TRL has already paid for those bugs.

Two subtleties the library gets right that are easy to get wrong by hand. First, **packing + assistant-only masking must compose**: when you glue three conversations into one 2048-token row, the assistant mask has to stay aligned across the joins and the model must not attend across document boundaries. TRL's `bfd` packing with the flash-attention path builds the block-diagonal attention mask for you; our hand-rolled version needed explicit `position_ids` resets and a document mask (the same machinery as document-aware packing in [Ch. 5.2](../05-posttraining-alignment/02-chat-templates-packing.html)). Second, **`completion_only_loss` vs `assistant_only_loss`**: for a prompt→completion (non-conversational) dataset, TRL masks the prompt via `completion_only_loss`; for multi-turn conversational data you want `assistant_only_loss`, which masks *every* user/system turn across *all* turns, not just the first prompt.

!!! example "Worked example: how much signal does assistant-only masking actually save?"

    Take a realistic Stack-100M SFT batch. A conversation renders to 512 tokens, of which the system + user turns are 180 tokens and the two assistant turns are 332 tokens (roughly a 35/65 split — typical when answers are longer than questions). Cross-entropy is averaged over *unmasked* tokens.

    - **No masking (train on everything):** the loss is computed over all 512 tokens. About $180/512 = 35\%$ of the gradient signal is spent teaching the model to *predict the user's questions* — behavior we do not want at inference and that actively pulls the model toward continuing with more questions.
    - **Assistant-only masking:** loss over 332 tokens; $100\%$ of the gradient teaches "given the turn, produce this answer, then `<|end|>`."

    Now scale it. At 100k conversations × 512 tokens = $5.1\times10^{7}$ tokens/epoch, masking discards $\sim1.8\times10^{7}$ token-losses per epoch. That is not wasted compute — the forward pass is identical — but it *is* $35\%$ of your gradient budget redirected from "sound like a user" to "answer like an assistant." At 100M params, where the model is easily pushed around by a few epochs, this is the difference between a model that answers-and-stops and one that answers-then-asks-a-follow-up-question. It is the single highest-leverage flag in the whole SFT config.

### Cheap adapters: `peft` and `unsloth`

At 100M we full-fine-tune — the model fits on any GPU and there is no multi-adapter serving reason to keep LoRA weights separate ([Ch. 5.3](../05-posttraining-alignment/03-peft-lora-qlora.html) explains when LoRA is the right call: 7B+ bases, or many task-specialized variants served off one frozen base as in [Ch. 7.14](../07-inference-serving/14-multi-tenant-lora-serving.html)). But you will use adapters constantly at work, and TRL wires to `peft` in one argument: pass a `LoraConfig` and `SFTTrainer` wraps the model for you.

```python
# LoRA-SFT: identical trainer, one extra arg. At Stack-100M's width this trains
# ~4% of parameters (r=16 against d_model=512 is a *fat* adapter); the familiar
# ~0.1-0.5% figure is a 7B+ number, where r/d_model is an order of magnitude smaller.
from peft import LoraConfig
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # attn + SwiGLU
    task_type="CAUSAL_LM",
)
trainer = SFTTrainer(model="./stack-100m-base", args=args,
                     train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
```

For memory-constrained work (a 7B base on a single 24GB card) **unsloth** is the 2025–2026 practitioner default: it provides drop-in `FastLanguageModel.from_pretrained(...)` with fused Triton kernels and 4-bit QLoRA that cut memory roughly in half and speed up training on the order of 2×, and it is TRL-compatible — you hand the unsloth model to `SFTTrainer` unchanged. The mechanism (LoRA math, NF4 quantization) is exactly what we cover in [Ch. 5.3](../05-posttraining-alignment/03-peft-lora-qlora.html) and [Ch. 4.10](../04-kernels-efficiency/10-memory-efficient-training.html); unsloth is the tuned implementation. Treat its headline speedups as version- and hardware-dependent — benchmark on your own shape.

## DPO with `DPOTrainer`: preference optimization without a reward model

DPO teaches *taste* from preference pairs `(prompt, chosen, rejected)` with a single supervised-style loss and no rollouts, no reward model, no critic — which is the only reason preference optimization is affordable at our budget. [Ch. 5.7](../05-posttraining-alignment/07-dpo-and-variants.html) derives it; the loss we implemented by hand in `post/dpo.py` was

$$
\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right]\right)\right],
$$

where $\pi_\theta$ is the policy we are training, $\pi_{\text{ref}}$ is the frozen SFT model, $y_w$/$y_l$ are the chosen/rejected completions, and $\beta$ controls how far the policy may drift from the reference. `DPOTrainer` computes exactly this — the four forward passes (policy-chosen, policy-rejected, ref-chosen, ref-rejected), the log-ratio, the logistic loss — and adds the operational machinery that makes it robust.

```python
# capstone/stacklm/post/dpo_trl.py
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# Preference dataset: rows are {"prompt", "chosen", "rejected"} (conversational or text).
ds = load_dataset("json", data_files="data/dpo/stack_prefs.jsonl", split="train")

args = DPOConfig(
    output_dir="ckpts/stack-100m-dpo",
    beta=0.1,                          # KL strength; 0.1 is the common default
    loss_type="sigmoid",              # the original DPO loss; see menu below
    max_length=1024,
    truncation_mode="keep_start",     # prompt+completion truncation lives in max_length now;
                                      #   older TRL had a separate max_prompt_length field
    # Precompute & CACHE the frozen reference logprobs once, then drop the ref model
    # from the loop entirely — reclaims the reference weights (2 bytes/param) and
    # removes 2 of every 4 forwards.
    precompute_ref_log_probs=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,               # DPO wants a MUCH smaller LR than SFT
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
)

trainer = DPOTrainer(
    model="./stack-100m-sft",          # start from the SFT checkpoint, not the base
    ref_model=None,                    # None ⇒ TRL clones the policy as the frozen ref
                                       #   (or, with LoRA, disables the adapter to get ref)
    args=args,
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()
```

Three things the library owns that our hand-roll left on the table:

- **Reference-logprob caching.** With `precompute_ref_log_probs=True`, TRL runs one pass over the dataset to store $\log\pi_{\text{ref}}(y_w\mid x)$ and $\log\pi_{\text{ref}}(y_l\mid x)$, then trains with the reference model *removed from memory*. Our version kept the ref model resident and paid two extra forwards every step. This matters more as the base grows: at 7B the reference is 14GB you get to reclaim.
- **LoRA-as-reference.** When training a LoRA-DPO run, you do not need a separate reference model at all — TRL gets $\pi_{\text{ref}}$ by *disabling the adapter* (the base weights are the reference by construction). One model, two behaviors. This is a genuinely clever memory win that is annoying to implement by hand.
- **The loss-variant menu.** `loss_type` selects among a family that all share the DPO scaffolding but change the objective's shape: `"sigmoid"` (original), `"ipo"` (Azar et al., replaces the logistic loss with a squared-error target on the log-ratio margin to fight the DPO over-optimization discussed in [Ch. 5.13](../05-posttraining-alignment/13-reward-hacking-failures.html)), `"hinge"`, `"robust"`, `"apo_zero"`, and others. Switching is a one-string change; deriving each from scratch is a chapter. Note that not every DPO-adjacent objective is a `DPOConfig` loss variant: CPO and SimPO are their own objectives with their own trainers, so `loss_type="cpo"` is *not* a legal value and raises rather than switching objectives. Print `DPOConfig.__dataclass_fields__["loss_type"].metadata["help"]` for the exact list your version accepts before assuming a variant is one string away.

There is also a **length-bias trap** DPO practitioners hit constantly: the implicit reward $\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ is an **unnormalized sum of per-token log-ratios**, so the margin a completion can attain grows with its token count — the optimizer can widen `rewards/margins` by nudging the ratio up over *more* tokens, i.e. by preferring *longer* text, rather than better text. That is reward-hacking length rather than quality. TRL exposes a length-normalized variant to counter this (`loss_type="sigmoid_norm"`, which divides each completion's score by its token count before the logistic loss — the SimPO-style average-log-prob idea); the honest move is to log mean chosen/rejected token lengths alongside `rewards/margins` and watch for the policy drifting long. This is the concrete, in-the-trainer face of the over-optimization theory in [Ch. 5.13](../05-posttraining-alignment/13-reward-hacking-failures.html).

The metrics TRL logs are your only window into whether preference optimization is healthy. The ones to watch:

| TRL metric | Healthy behavior | Bad sign |
|---|---|---|
| `rewards/margins` | rises steadily | flat (no learning) or explodes (LR too high) |
| `rewards/chosen` | near 0 or slightly negative | falling fast ⇒ destroying the SFT model |
| `rewards/rejected` | goes negative | rising ⇒ policy prefers rejected |
| `rewards/accuracies` | rises toward 1.0 | stuck at 0.5 ⇒ no signal in the pairs |
| chosen − rejected token length | roughly stable | chosen growing ⇒ length-hacking |

!!! warning "The DPO learning rate is not the SFT learning rate"

    A recurring failure: engineers copy `learning_rate=2e-5` from their SFT config into DPO and watch the model collapse — margins explode, both chosen and rejected logprobs crater, and the model outputs degenerate text. DPO operates on *log-ratios of already-fluent completions*; it needs an LR one-to-two orders of magnitude smaller (on the order of `5e-7` to `5e-6`). Watch `rewards/margins`, `rewards/chosen`, and `rewards/rejected` in the TRL logs: healthy DPO shows the **margin rising while chosen stays near zero and rejected goes negative**. If chosen logprob is falling fast, your LR (or $\beta$) is too high.

## GRPO with `GRPOTrainer` + vLLM rollouts

GRPO (Group Relative Policy Optimization, from DeepSeekMath, Shao et al. 2024) teaches a *verifiable skill*: generate a group of completions per prompt, score each with a reward function, and reinforce the above-average ones — with the group mean/std serving as a **critic-free baseline**, which is what makes it so much cheaper than PPO ([Ch. 5.8](../05-posttraining-alignment/08-grpo-rloo.html), [Ch. 5.6](../05-posttraining-alignment/06-ppo-for-llms.html)). For a prompt $x$ with group $\{y_1,\dots,y_G\}$ and scalar rewards $\{r_1,\dots,r_G\}$, the group-normalized advantage is

$$
A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon},
$$

and each token in completion $y_i$ is pushed by $A_i$ under a clipped policy-gradient objective with a KL penalty to the reference. Our `post/grpo.py` did all of this on a single GPU with cacheless `model.generate` — which, as we computed in [Ch. 14.9](../14-capstone/09-post-training.html), spends **five times more FLOPs on generation than on gradients**. That asymmetry is why real GRPO offloads rollouts to a proper inference engine.

### The reward function is your code

The single most important interface: TRL passes each reward function the list of generated completions and returns a list of floats. This is where RL-with-verifiable-rewards ([Ch. 5.9](../05-posttraining-alignment/09-rlvr-reasoning.html)) lives — the reward is a *program*, not a learned model.

```python
# capstone/stacklm/post/grpo_reward.py
import re
# Signature contract: reward_fn(completions, **kwargs) -> list[float].
# `completions` is a list (len = batch*num_generations); other dataset columns
# (e.g. the ground-truth answer) arrive as keyword lists of the same length.
def exact_match_reward(completions, answer, **kwargs):
    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp[-1]["content"] if isinstance(comp, list) else comp  # conversational vs text
        m = re.search(r"####\s*(-?\d+)", text)         # our SFT taught "#### <int>" answers
        pred = m.group(1) if m else None
        rewards.append(1.0 if pred is not None and pred == str(gold) else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    # Small shaping reward: encourage the <think>...</think> #### <int> format.
    out = []
    for comp in completions:
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        out.append(0.2 if re.search(r"####\s*-?\d+", text) else 0.0)
    return out
```

### The trainer, with vLLM doing the generation

```python
# capstone/stacklm/post/grpo_trl.py
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from grpo_reward import exact_match_reward, format_reward

ds = load_dataset("json", data_files="data/rlvr/arith.jsonl", split="train")
# rows: {"prompt": [{role, content}...], "answer": 68}

args = GRPOConfig(
    output_dir="ckpts/stack-100m-grpo",
    num_generations=8,                 # group size G — the baseline is the group mean
    max_completion_length=256,         # (prompt-side truncation was `max_prompt_length` in
                                       #  older TRL; current releases dropped that field)
    temperature=1.0,                   # exploration: too low ⇒ no reward variance ⇒ no signal
    beta=0.04,                         # KL-to-reference coefficient
    # --- offload rollouts to vLLM (Ch. 7.3) — the reason this scales ---
    use_vllm=True,
    vllm_mode="colocate",              # "colocate": vLLM shares this process's GPUs;
                                       # "server": a separate `trl vllm-serve` process
    # generation/optimization split
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_iterations=1,                  # PPO-style inner epochs per rollout batch (μ)
    learning_rate=1e-6,
    bf16=True,
    logging_steps=1,
)

trainer = GRPOTrainer(
    model="./stack-100m-sft",
    reward_funcs=[exact_match_reward, format_reward],  # summed (optionally weighted)
    args=args,
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()
```

Under the hood `GRPOTrainer` optimizes the clipped surrogate we derived in [Ch. 5.8](../05-posttraining-alignment/08-grpo-rloo.html) — for each token it forms the importance ratio $\rho_{i,t} = \pi_\theta(y_{i,t}\mid\cdot)/\pi_{\theta_\text{old}}(y_{i,t}\mid\cdot)$ against the policy that *generated* the rollouts, and maximizes

$$
\mathcal{J} = \mathbb{E}\Big[\min\big(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)A_i\big)\Big] - \beta\,\mathrm{KL}\!\left[\pi_\theta \,\|\, \pi_\text{ref}\right].
$$

The `num_iterations` field ($\mu$) is how many gradient epochs you take on one batch of rollouts before regenerating — $\mu=1$ makes $\theta_\text{old}=\theta$ so every ratio is exactly 1 and the clip never bites (pure on-policy, the cheap default); $\mu>1$ reuses expensive rollouts at the cost of the policy drifting from the generator, which is precisely when the clip earns its keep. This is the same generation-reuse tradeoff that dominates large-scale RL systems ([Ch. 6.2](../06-rl-infra/02-generation-training-loop.html)).

The `vllm_mode` choice is the crux of RL-for-LLM systems ([Ch. 6.2](../06-rl-infra/02-generation-training-loop.html), [Ch. 6.7](../06-rl-infra/07-colocated-vs-disaggregated.html)):

- **`"colocate"`** runs vLLM in the same process, sharing GPU memory with the trainer. Simplest to launch; you split VRAM between the training model and the KV cache. Good for a single node — including our single-A100 flagship.
- **`"server"`** runs a standalone, persistent inference server you start separately with `trl vllm-serve --model ...`; the trainer streams prompts to it over HTTP and syncs updated weights after each optimizer step. This is the disaggregated pattern that scales to many GPUs, at the cost of a weight-synchronization path (the trainer must push its new weights into the server's model between steps — the exact race-prone machinery [Ch. 6.7](../06-rl-infra/07-colocated-vs-disaggregated.html) dissects).

Either way, TRL replaces our slow cacheless `model.generate` with vLLM's PagedAttention + continuous batching ([Ch. 4.6](../04-kernels-efficiency/06-paged-attention-kv.html), [Ch. 7.3](../07-inference-serving/03-vllm-internals.html)), which is where essentially all of GRPO's wall-clock goes. There is one correctness subtlety the colocate mode forces you to confront: **the sampling engine (vLLM) and the training engine (transformers) must agree on the policy's probabilities**, or the importance ratios are computed against a distribution the model never sampled from. TRL syncs the trainer's updated weights into the vLLM worker after each optimizer step; if that sync is stale, or if vLLM's kernels produce logits that differ from the training forward pass by more than rounding, you get a subtle *off-policy* bias. This is exactly the rollout/train mismatch that [Ch. 6.7](../06-rl-infra/07-colocated-vs-disaggregated.html) treats as a first-class systems problem, and it is the reason TRL ships a *sampler-vs-trainer* diagnostic. Do not look at the policy ratio for this: at $\mu=1$, $\pi_{\theta_\text{old}}$ *is* $\pi_\theta$ (TRL reuses the training forward's own logprobs), so $\rho\equiv1$ by construction and tells you nothing. The quantity that actually measures the mismatch is the ratio between vLLM's returned logprobs and the trainer's forward on those same tokens — turn on `vllm_importance_sampling_correction=True` and watch the logged `sampling/importance_sampling_ratio/{min,mean,max}`: a mean near 1.0 with tight tails means the two engines agree; heavy tails mean your gradients are being computed against a distribution the sampler never used. Launch a colocated run with `accelerate`:

```bash
# Single node. For the server split, first: trl vllm-serve --model ./stack-100m-sft
accelerate launch --config_file recipes/accelerate/single_gpu.yaml \
    capstone/stacklm/post/grpo_trl.py
```

!!! example "Worked example: the group advantage that GRPO actually computes"

    Take one arithmetic prompt, `"What is 17 times 4? Answer with #### <int>."`, group size $G=8$, temperature 1.0. The eight sampled completions get exact-match rewards (1.0 correct, 0.0 wrong) plus a 0.2 format bonus each for producing `#### <int>`:

    | $i$ | correct? | $r_i$ (match + format) |
    |---|---|---|
    | 1 | ✓ | 1.2 |
    | 2 | ✗ | 0.2 |
    | 3 | ✓ | 1.2 |
    | 4 | ✗ | 0.2 |
    | 5 | ✓ | 1.2 |
    | 6 | ✗ | 0.0 (no `####`) |
    | 7 | ✓ | 1.2 |
    | 8 | ✗ | 0.2 |

    Mean $\bar r = (1.2\cdot4 + 0.2\cdot3 + 0.0)/8 = 5.4/8 = 0.675$. The squared deviations sum to $4(0.525)^2 + 3(0.475)^2 + (0.675)^2 = 2.235$, so the *population* std is $\sqrt{2.235/8}\approx0.529$ — but TRL's `nanstd` applies Bessel's correction, so what `GRPOTrainer` divides by is $\sqrt{2.235/7}\approx0.565$. The normalized advantages $A_i = (r_i - 0.675)/0.565$ are $\approx +0.93$ for the four correct completions, $\approx -0.84$ for the three format-but-wrong ones, and $\approx -1.19$ for the completely-unformatted one. Every **token** of a correct completion is pushed *up* by $\approx0.93$; every token of the worst one is pushed *down* by $\approx1.19$ — no value network anywhere, the group is its own baseline. (The $n$ vs $n-1$ convention shifts every advantage by $\sqrt{(G-1)/G}$, a uniform $\approx7\%$ at $G=8$; it rescales the effective step size, not the ranking.)

    The failure mode this makes visible: if all eight completions were correct, $\operatorname{std}=0$, every $A_i=0$, and the batch contributes **zero gradient** (the $\varepsilon$ just prevents a divide-by-zero). GRPO learns only from prompts where the model *sometimes* succeeds and *sometimes* fails — which is exactly why RLVR needs a curriculum pitched at the edge of the model's competence ([Ch. 6.12](../06-rl-infra/12-rl-data-curriculum-replay.html)), and why it works at 100M **only** on narrow tasks the base model already solves part of the time.

!!! warning "Your reward function is the specification — and GRPO will exploit it"

    GRPO optimizes *exactly* what your reward returns, not what you meant. Two failures bite immediately at 100M. First, a **shaping reward becomes the whole objective**: give a 0.2 bonus for emitting `#### <int>` and the model may learn to emit `#### 0` on every prompt — collecting the format bonus while ignoring correctness — because a guaranteed 0.2 beats a risky shot at 1.0. Keep shaping rewards small relative to the correctness reward, and *watch the components separately* (log `exact_match` and `format` rewards as distinct metrics). Second, **a buggy verifier is a gift the model will take**: if your regex accepts `#### 68.0` as matching gold `68` in some cases and not others, or strips whitespace inconsistently, the model finds the crack. The from-scratch lesson from [Ch. 5.13](../05-posttraining-alignment/13-reward-hacking-failures.html) and [Ch. 6.8](../06-rl-infra/08-reward-verifiers-sandboxes.html) is unchanged by the library: the reward function is the most security-critical code in the pipeline, and it deserves unit tests before a single GRPO step runs.

## Recipes and scale: the alignment-handbook, veRL, OpenRLHF

### The alignment-handbook: TRL as reproducible YAML

Writing a Python script per run is fine for experiments; for a *reproducible* pipeline you want configuration, not code. The **alignment-handbook** (the Hugging Face H4 team's recipe repo that produced the Zephyr models) is a thin, opinionated layer over TRL: each stage is a YAML config plus a launch command, with tested `accelerate` + DeepSpeed ZeRO-3 / FSDP configs for multi-GPU. The Stack-100M SFT recipe looks like:

```yaml
# recipes/stack-100m/sft/config_full.yaml
model_name_or_path: ./stack-100m-base
torch_dtype: bfloat16
# data
dataset_mixer:
  stack-chat-sft: 1.0            # weight each dataset; the handbook mixes and shuffles
dataset_splits: [train]
preprocessing_num_workers: 12
# SFT (these map 1:1 onto SFTConfig fields)
max_length: 2048
packing: true
assistant_only_loss: true
per_device_train_batch_size: 16
gradient_accumulation_steps: 4
learning_rate: 2.0e-05
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 3
bf16: true
gradient_checkpointing: true
output_dir: ckpts/stack-100m-sft
```

```bash
# Multi-GPU with DeepSpeed ZeRO-3, straight from the handbook's tested configs:
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    scripts/run_sft.py recipes/stack-100m/sft/config_full.yaml
```

The DPO recipe is the same shape with a `run_dpo.py` entry point and `beta`/`loss_type` fields. The payoff is that a run is now a *file you commit*: config hash, data mixer, and seed are captured, which is exactly the reproducibility discipline the capstone retrospective ([Ch. 14.12](../14-capstone/12-retrospective-and-scaleup.html)) demands. Be honest about drift here too: the handbook tracks a specific TRL/transformers matrix, and field names in the YAML follow whatever TRL version it pins — read its `setup.py` before assuming a field exists.

### When TRL stops scaling: veRL and OpenRLHF

TRL's GRPO is excellent up to roughly a single node. Past that — 70B policies, thousands of concurrent rollouts, multi-node weight sync — you graduate to a purpose-built RL system, and the two 2026 standards are **veRL** and **OpenRLHF** ([Ch. 6.4](../06-rl-infra/04-verl.html), [Ch. 6.5](../06-rl-infra/05-openrlhf-nemo-ray.html)).

- **veRL** (HybridFlow, Volcano Engine) uses a **single-controller** programming model over Ray: you write the RL algorithm as ordinary Python control flow, and veRL maps the heavy pieces — a **vLLM/SGLang rollout worker pool** and an **FSDP/Megatron training pool** — onto the cluster, handling the placement and the weight resharding between them. It is the system frontier labs reach for when rollouts dominate the FLOP bill (as they do at 1B+; see the cost inversion in [Ch. 14.9](../14-capstone/09-post-training.html)).
- **OpenRLHF** is a Ray + DeepSpeed + vLLM stack with a CLI-driven design; a run is launched as a Ray job where you *declare the cluster topology as flags* — how many GPUs host the vLLM rollout engines, how many host the actor (policy), how many the critic — and Ray places them:

```bash
# OpenRLHF: disaggregated placement declared on the command line (illustrative flags —
# check the pinned version's `train_ppo_ray --help`, as names track releases).
ray job submit -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain ./stack-1b-sft \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --advantage_estimator group_norm    # GRPO-style critic-free advantage
```

  It pioneered the disaggregated actor/rollout placement that TRL's `vllm_mode="server"` is a single-node echo of. NeMo-Aligner (NVIDIA) is the third member of this family, Megatron-native and built for the largest scales ([Ch. 6.5](../06-rl-infra/05-openrlhf-nemo-ray.html)).

The mechanism is identical to what TRL does — generate, reward, group-normalize, policy-gradient — but the *systems* problem (colocated vs disaggregated placement, weight synchronization, load-balancing straggler rollouts) is the entire game, and that is what [Part VI](../06-rl-infra/01-anatomy-rl-system.html) is about. The transfer lesson: **learn the objective in TRL, learn the system in veRL/OpenRLHF.** You would not run Stack-100M on veRL — the Ray overhead dwarfs a 100M model — and that scale-appropriateness is itself the point.

## Gating with `lm-evaluation-harness`

A post-training stage is only "done" when a *held-out, standardized* eval says the model got better and no capability regressed. The 2026 standard is EleutherAI's **lm-evaluation-harness** (`lm-eval`), the same harness the Open LLM Leaderboard and most papers use, which matters because it makes your numbers *comparable* and its templating removes a giant source of silent measurement error ([Ch. 11.3](../11-evaluation/03-eval-harnesses.html)). We built a tiny probe suite by hand in [Ch. 14.11](../14-capstone/11-evaluation-and-serving.html); `lm-eval` is the industrial version.

```bash
# Evaluate the SFT checkpoint on a few lightweight tasks. Every metric comes with a
# stderr — report it; a 1-point move inside the stderr is noise (Ch. 11.6).
lm_eval --model hf \
    --model_args pretrained=ckpts/stack-100m-sft,dtype=bfloat16 \
    --tasks gsm8k,arc_easy,hellaswag \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path evals/stack-100m-sft.json

# Faster: run the eval THROUGH vLLM (the same engine you serve with, Ch. 15.6):
lm_eval --model vllm \
    --model_args pretrained=ckpts/stack-100m-sft,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks gsm8k --num_fewshot 5 --batch_size auto
```

A caution specific to small models: `lm-eval`'s default tasks (GSM8K, ARC, HellaSwag) are pitched at billion-parameter models, and a 100M model will score at or near chance on most of them — that is honest, not a bug ([Ch. 14.11](../14-capstone/11-evaluation-and-serving.html) is blunt about the 100M ceiling). The right gate for Stack-100M mixes a couple of standard tasks (so your numbers are *comparable* to the literature and you can watch for regressions) with the *narrow, custom* probes that actually track the capability you post-trained for — the arithmetic exact-match your GRPO run targeted, a small held-out format-adherence set, a retrieval-QA exact-match. `lm-eval` supports custom tasks via a YAML task spec, so you register those alongside the standard ones and run them in one command. And always report the stderr: [Ch. 11.6](../11-evaluation/06-statistical-rigor-eval.html) shows why a 0.8-point "improvement" inside a 1.5-point stderr is a coin flip, not progress.

Wire this into CI as a **gate**: a stage promotes only if its eval clears the previous checkpoint by more than the reported stderr on the target metric *and* does not regress a guardrail metric (e.g. a safety or format-adherence probe). This is the data-flywheel discipline of [Ch. 12.5](../12-production-mlops/05-data-flywheel.html) applied to post-training: SFT → gate → DPO → gate → GRPO → gate, each stage earning its place.

!!! tip "Match the eval's chat template to training"

    A brutal, common bug: you train with the Stack-100M ChatML template but `lm-eval` scores the model with *no* template (raw prompt), and every instruct metric looks broken. Pass `--apply_chat_template` (and `--system_instruction` where relevant) so the harness renders prompts the way the model was trained, or you are measuring a distribution the model never saw. The from-scratch lesson — that the template *is* part of the model's interface — does not go away because a library renders it.

!!! interview "Interview Corner"

    **Q:** Your team fine-tunes a chat model with TRL's `SFTTrainer` on multi-turn conversational data, but at inference it keeps generating the *user's* next turn instead of stopping after its answer. Walk me through the likely causes and fixes.

    **A:** This is almost always a **loss-masking or template bug**, in one of three places. (1) `assistant_only_loss` was left `False` — the model trained on *predicting user turns too*, so it learned to continue the conversation as a user. Verify with `apply_chat_template(..., return_assistant_tokens_mask=True)` and assert the mask is nonzero. (A missing `{% generation %}` block used to fail *silently* here, producing an empty mask; current TRL either substitutes a generation-marked training template or raises a `RuntimeError` when an example has no assistant tokens. The version of this bug that is still quiet is a template whose end-of-turn token falls *outside* the `{% generation %}` span — TRL only warns, and the model never learns to stop.) (2) The **turn-terminator token isn't being learned as a stop**: if `<|end|>`/`<|eos|>` weren't consistently emitted after assistant turns in training, or the generation config's `eos_token_id` doesn't include the turn terminator, decoding runs past the boundary. (3) **Packing crossed document boundaries**: if short conversations were concatenated without a boundary-aware packer and cross-document attention mask, the model saw "assistant turn → next user turn" as an in-context continuation and learned exactly that. The fixes, in order: turn on `assistant_only_loss` and confirm the mask, ensure the eval uses the same chat template with the terminator as a stop token, and use TRL's `bfd` packing (or `packing=False`) so joins are respected. The deeper point: SFT format bugs are usually *masking* bugs, and they're invisible in the training loss curve — you catch them only by inspecting the mask and by evaluating with the production template.

!!! key "Key Takeaways"

    - **Hand-roll to learn, reach for TRL to ship.** Every TRL post-training trainer subclasses `transformers.Trainer`, so you inherit accum/mixed-precision/FSDP/checkpointing for free and supply only the objective — the loss you already derived in [Ch. 14.9](../14-capstone/09-post-training.html) and [Ch. 5.x](../05-posttraining-alignment/01-sft-instruction-tuning.html).
    - **Bridge the model first.** TRL needs a HF `PreTrainedModel`; export Stack-100M into a Qwen3-shaped checkpoint (Qwen3 carries the `q_norm`/`k_norm` our QK-norm needs; only NoPE requires a `trust_remote_code` modeling file), and *verify logits match* before trusting the export.
    - **SFT is a masking problem.** `assistant_only_loss=True` + a `{% generation %}` chat template + boundary-aware `packing` is the whole recipe; the single highest-leverage flag redirects ~35% of gradient from "sound like a user" to "answer like an assistant."
    - **DPO deletes the reward model; cache the reference.** `precompute_ref_log_probs=True` (or LoRA-as-reference) removes two of four forwards; DPO needs an LR one-to-two orders of magnitude below SFT, and you watch `rewards/margins` to know it's healthy.
    - **GRPO's cost is generation, so offload it to vLLM.** `use_vllm=True` with `vllm_mode` colocate (single node) or server (disaggregated, weight-sync); the reward function is *your program* (RLVR), and the group std being zero means zero gradient — RL needs prompts at the edge of competence.
    - **Recipes, not scripts.** The alignment-handbook turns TRL runs into committed YAML with tested DeepSpeed/FSDP configs — reproducibility as a file.
    - **Scale changes the system, not the objective.** veRL (single-controller HybridFlow) and OpenRLHF (Ray+DeepSpeed+vLLM) exist for when rollouts dominate the FLOP bill at 1B+; at 100M their overhead dwarfs the model.
    - **Gate every stage with lm-evaluation-harness**, apply the training chat template at eval time, and promote only past the reported stderr — post-training is a flywheel of train → gate → repeat.
    - **Pin versions and verify fields.** TRL flags move between releases; lock your dependency matrix and print `Config.__dataclass_fields__` rather than trusting any snippet, including this one.

## Further reading

- **TRL** — Hugging Face, *TRL: Transformer Reinforcement Learning* (library and docs; `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`). The `trl/examples/scripts/` directory is the canonical, version-matched source of truth.
- **The alignment-handbook** — Hugging Face H4 team (the recipe repo behind the Zephyr models); `run_sft.py`, `run_dpo.py`, and the tested `accelerate`/DeepSpeed configs.
- Rafailov, Sharma, Mitchell, et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023).
- Shao, Wang, Zhu, et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024) — introduces GRPO.
- Sheng, Zhang, Ye, et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (veRL, 2024).
- Hu, Wu, Zhang, et al., *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework* (2024).
- **peft** (Hugging Face) and **unsloth** (Han et al.) — parameter-efficient and memory-efficient fine-tuning implementations.
- Gao, Tow, Abbasi, et al., *The Language Model Evaluation Harness* (EleutherAI, `lm-evaluation-harness`).
- Kwon, Li, Zhuang, et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023) — the rollout engine behind GRPO at scale.
