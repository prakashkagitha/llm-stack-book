# 14.11 Evaluation & Serving: Honest Benchmarks, int4 Quantization, and Running on a Laptop

By the end of [14.9 Post-Training](09-post-training.html) and [14.10 The Narrow Agent](10-agentic-narrow.html), you have a checkpoint: `Stack-100M`, pretrained on ~20B tokens, mid-trained for long context and capability injection, SFT+DPO-aligned, RLVR-sharpened on tool use, and distilled onto a ReAct agent loop. It is tempting to declare victory and ship it. Resist that. A checkpoint is not a result — a **number you can defend**, next to an **honest description of what the model cannot do**, is a result. This chapter builds that: a small, principled evaluation harness that measures every capability the capstone claimed (including the agent), and then the second half of the payoff — squeezing the ~101M-parameter, ~405MB fp32 checkpoint down to something that runs, from cold start to generated text, on the CPU of the laptop you are reading this on.

Two halves, one spirit: **measure honestly, then serve efficiently** — and then, crucially, *measure again after serving*, because quantization is a model edit and every model edit is a hypothesis about quality that has to be tested. Neither half is about chasing a leaderboard. A 100M model trained on a single GPU for the price of a nice dinner is never going to touch a frontier benchmark, and pretending otherwise — via a leaky eval set or a cherry-picked prompt — teaches the wrong lesson. The right lesson is that a small model, evaluated honestly and served efficiently, is still a genuinely useful, genuinely *yours* piece of software.

## 1. What "Done" Means for Stack-100M

Recall the pipeline from [14.1 The Capstone Overview](01-overview-and-landscape.html): tokenizer → pretraining (~20B tokens, WSD schedule, Muon+AdamW) → mid-training → post-training (SFT, DPO, narrow RLVR) → agentic distillation. Every stage produced a checkpoint. This chapter is the final gate:

1. **Settle the serving surface** — which inference path every measurement runs through, and the one string-level wrapper this chapter adds on top of Ch. 14.4.
2. **Evaluate** against a small, fixed battery: held-out perplexity, an arithmetic probe *in the format RLVR actually trained*, a tiny cloze-scored multiple-choice set, retrieval-QA exact match, and — the capstone's flagship deliverable — the ReAct **agent loop** itself.
3. **Write down what the model cannot do.** Not padding: this is the deliverable that makes the numbers trustworthy.
4. **Quantize** to int8, then int4, understanding exactly what post-training quantization (PTQ) does and why round-to-nearest (RTN) is a baseline rather than the state of the art.
5. **Re-run the battery on the quantized model** and tabulate the deltas. "int8 is free, int4 costs you something on the hard tail" is an empirical claim; check it rather than believe it.
6. **Export and serve** on CPU, with measured latency and memory on your own machine, and a named upgrade path into the real 2026 quantization/serving stack (torchao, llm-compressor / compressed-tensors, GGUF + llama.cpp, vLLM).

We evaluate the *final* post-trained checkpoint, but every function below also runs against the raw pretrained base from [14.7 The Pretraining Run](07-pretraining-run.html) — comparing base vs. instruct/agent perplexity is a useful check that post-training did not quietly regress core language modeling ability (a real failure mode called *alignment tax*).

## 2. The Serving Surface: From `model.generate` to Text In, Text Out

Everything in this chapter — every probe, the agent loop, the quantized laptop demo — runs through one inference path, so it is worth being precise about who owns which piece.

**Ch. 14.4 already owns the hard part.** `Stack100M` ships with a preallocated `KVCache` (one slab per layer, shape `(n_layers, B, n_kv_heads, max_seq, head_dim)` — the *narrow*, post-RoPE, GQA keys and values), a `forward(idx, targets=None, position_ids=None, seq_ids=None, kv_cache=None, start_pos=0, logits_to_keep=0)` that supports incremental decode, a `_build_mask` that constructs the causal mask **from positions** (never `is_causal=True` on a rectangular score matrix — see 14.4's warning on SDPA's top-left alignment), and a token-level `generate()` with a `use_cache=False` oracle branch. Its CI asserts that cached incremental decode reproduces the full recompute and that greedy generation is cache-invariant. Do not reimplement any of it; import it.

**Two things this chapter needs on top.** First, a **tokenizer-level wrapper**: `model.generate` speaks token-id tensors, and every probe below speaks strings, special tokens, and a stop condition. Second, a discipline about `logits_to_keep`.

!!! warning "Common pitfall: `logits_to_keep=1` is a sampling optimization, not a default"

    `Stack100M.forward` computes the LM head over only the last `logits_to_keep` positions when that argument is non-zero. At $V = 32768$ and $d = 512$ the head is $\approx 16.8$M MACs *per position*, so skipping it on a 2048-token prefill is the difference between a fast prefill and a slow one — which is exactly why `generate()` passes `logits_to_keep=1`.

    Every **evaluation** function in this chapter needs the opposite: `logits_to_keep=0` (the default), giving `(B, T, V)`. Perplexity needs a log-probability at every position; cloze scoring needs them at every *continuation* position. Copy a sampling call into an eval function without changing that argument and you get a `(B, 1, V)` tensor that either crashes on the reshape or — worse — silently scores one position and reports a number that looks plausible. Assert the shape.

    The flip side is memory: an all-positions logits tensor is $B \times T \times V \times 4$ bytes in fp32. At $B=4$, $T=2047$, $V=32768$ that is **1.07 GB** — on a CPU eval box the logits, not the weights, are your memory ceiling. Keep the eval batch size small (1–4 on CPU).

```python
"""
stacklm/infer.py — the string-in/string-out layer over Ch. 14.4's generate().
Every probe in Section 4, the agent loop in Ch. 14.10, and the CLI in Section 9
call exactly these two functions.
"""
import torch

from stacklm.tokenizer import SPECIAL_TOKENS               # Ch. 14.3
ALL_SPECIALS = frozenset(SPECIAL_TOKENS)


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 64,
                  temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                  stop_id: int | None = None, allowed_special=ALL_SPECIALS,
                  device: str = "cpu", return_n_tokens: bool = False):
    """Encode -> Ch. 14.4's cached generate() -> decode ONLY the new tokens.

    `allowed_special` is not decoration. Ch. 14.3's `StackTokenizer.encode`
    emits the reserved special ids ONLY for strings you explicitly allow;
    forget it and `<|system|>` is tokenized as literal angle-bracket bytes, the
    model never sees the chat frame it was SFT'd on, and your probe measures a
    distribution shift you introduced yourself. Default it ON here so no caller
    can forget. (Ch. 14.3's default is the opposite, deliberately: untrusted
    user text must never be able to forge a role boundary.)

    `return_n_tokens` returns the EXACT count of generated ids. Re-encoding the
    decoded string is not guaranteed to reproduce it, so tok/s computed that way
    is an estimate; the CLI in Section 9 uses this flag instead.
    """
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt, allowed_special=allowed_special)],
                       dtype=torch.long, device=device)
    assert ids.shape[1] + max_new_tokens <= model.cfg.max_seq_len, (
        "prompt + generation exceeds the context; call model.rebuild_rope() or "
        "truncate — Ch. 14.4's generate() asserts this too, loudly.")
    eos = tokenizer.eos_id if stop_id is None else stop_id
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=top_p, top_k=top_k, eos_id=eos)
    new = out[0, ids.shape[1]:].tolist()                        # new tokens only
    if eos in new:                                              # 14.4's generate emits the
        new = new[:new.index(eos)]                              # stop token; drop it
    text = tokenizer.decode(new)
    return (text, len(new)) if return_n_tokens else text


def generate_fn(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0, **kw):
    """The exact signature the Section 4 probes expect. Greedy by default:
    a probe score you cannot reproduce bit-for-bit is not a measurement."""
    return generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                         temperature=temperature, **kw)
```

!!! tip "Practitioner tip: the KV cache is not a rounding error at 100M"

    `KVCache(cfg, batch_size=1, max_seq=2048, device="cpu", dtype=torch.bfloat16).nbytes()` returns **31,457,280** bytes — 30 MiB, or ≈31.5 MB. Hold that next to the ≈63 MB int4 weight budget we will fight for in §7: at the pretrain context the cache is *half the model*. Small models are relatively *more* KV-bound than large ones, because the cache scales with layers × context while the weights scale with layers × width². Keep the cache in **bf16** (Ch. 14.4's default): keys and values are activations, re-read once per step and never accumulated, so fp32 doubles the cost for no benefit. Exercise 7 works out where the cache overtakes the weights. Everything past that — fp8/int8 KV quantization, paged blocks, prefix reuse — is a serving-systems problem and lives in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) and [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html).

{{tool:kv-cache-budgeter}}

!!! warning "Re-run the cache-equivalence test after quantization"

    Ch. 14.4's CI proves that *the fp32 model's* cached decode matches its full recompute. That proof does not transfer to the quantized model, because §7 replaces every `nn.Linear` with a different module, and a packing or row-gather bug in `QuantizedLinear` shows up as fluent-but-wrong text rather than a crash. Re-run the same assertion against the quantized model at toy scale — it is three lines, and it is the difference between "the export loads" and "the export is correct":

    ```python
    q = quantize_stacklm(copy.deepcopy(tiny_model), bits=4, group_size=64).eval()
    torch.manual_seed(0); a = q.generate(idx, 6, temperature=0.0)
    torch.manual_seed(0); b = q.generate(idx, 6, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)          # cache-invariance survives quantization
    ```

    Necessary, not sufficient — see Exercise 7(c) for the second half of the test pair.

## 3. Held-Out Perplexity: The One Number You Can Trust

Perplexity is the exponentiated average negative log-likelihood the model assigns to held-out text it never trained on:

$$
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i})\right)
$$

It is the most trustworthy number in this chapter precisely because it is *cheap to make honest*: as long as your held-out shard was excluded from the training manifest (see the manifest/hash bookkeeping in [14.2 The Data Pipeline](02-data-pipeline.html)), there is no way to game it by prompt-crafting or answer-formatting tricks. It directly measures the thing pretraining optimizes. Its weakness is the flip side: it tells you how well the model predicts *held-out web text*, not whether it can multiply two-digit numbers or drive a tool loop — for that you need §4.

```python
"""
stacklm/eval/ppl.py — held-out perplexity.

Consumes exactly the batch dict Ch. 14.2's `PackedMemmapDataset` yields:
{"input_ids", "position_ids", "targets"} over uint16 memmap shards packed to
seq_len 2048 with document-aware position resets.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stacklm.model.transformer import Stack100M       # Ch. 14.4
from stacklm.data.dataset import PackedMemmapDataset  # Ch. 14.2


def _autocast(device: str):
    """bf16 autocast on CUDA; a no-op context on CPU (where the rest of this
    chapter lives). Hardcoding device_type='cuda' is the classic way to make an
    eval function that only runs on the machine you wrote it on."""
    dev_type = torch.device(device).type
    if dev_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=dev_type, enabled=False)


@torch.no_grad()
def compute_perplexity(model: Stack100M, val_shard_dir: str, pad_id: int,
                       batch_size: int = 4, max_batches: int | None = None,
                       device: str = "cpu") -> dict:
    """Held-out cross-entropy (nats/token) and perplexity.

    Three details decide whether the number is honest (or fits in RAM):
      1. PAD MASKING. Ch. 14.2 pads the trailing window with `pad_id`, and the
         training loop masks the loss there. If we do not, those positions are
         trivially predictable and drag the mean loss DOWN — a silently
         optimistic perplexity. We remap pad targets to -100 and let
         `ignore_index` do the rest, then count only the surviving tokens.
      2. BATCH SIZE. The all-positions logits tensor is B x T x V x 4 bytes;
         at B=4, T=2047, V=32768 that is 1.07 GB. On CPU the logits, not the
         weights, are the memory ceiling — keep this small.
      3. DOCUMENT MASKING. Ch. 14.2 stores `position_ids` that reset to 0 at
         every document start; `seq_ids = cumsum(position_ids == 0) - 1` turns
         that into the segment id `Stack100M.forward` uses to forbid
         cross-document attention. `position_ids` also drives RoPE, so a
         document starting mid-window is rotated from its OWN position 0,
         exactly as in training. Evaluate without them and you measure the
         model in a geometry it was never trained in.

    Note what is NOT passed: `logits_to_keep`. Its default of 0 is what makes
    `logits` a (B, T, V) tensor instead of the (B, 1, V) a sampling call
    returns — Section 2's pitfall, where it does the most damage.
    """
    model.to(device).eval()
    ds = PackedMemmapDataset(val_shard_dir)          # never seen in ANY stage
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    total_nll, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)             # (B, T)
        targets = batch["targets"].to(device).clone()         # (B, T), shifted by 1
        position_ids = batch["position_ids"].to(device)
        seq_ids = (position_ids == 0).cumsum(dim=-1) - 1      # (B, T) segment ids

        targets = targets.masked_fill(targets == pad_id, -100)     # rule 1
        # Also drop the boundary target: the last token of doc k "predicts" the
        # BOS of doc k+1, which is a packing artifact, not language modeling.
        boundary = torch.zeros_like(targets, dtype=torch.bool)
        boundary[:, :-1] = position_ids[:, 1:] == 0
        targets = targets.masked_fill(boundary, -100)

        with _autocast(device):
            logits, _ = model(input_ids, position_ids=position_ids,
                              seq_ids=seq_ids)              # (B, T, V)
        assert logits.shape[1] == input_ids.shape[1], "logits_to_keep leaked in"
        # sum (not mean) so we can average correctly over the whole eval set
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               targets.reshape(-1), reduction="sum", ignore_index=-100)
        total_nll += loss.item()
        total_tokens += int((targets != -100).sum())

    mean_nll = total_nll / max(1, total_tokens)
    return {"loss_nats_per_token": mean_nll,
            "perplexity": math.exp(mean_nll),
            "n_tokens_evaluated": total_tokens}
```

Two practical rules make this number defensible rather than decorative. First, **the held-out shard must be provably excluded** from every stage of training — pretraining, mid-training annealing, and SFT/DPO data — which is why Ch. 14.2 hashes and manifests every source file up front. Second, **report loss in nats/token, not just perplexity**, since it composes linearly and is what your training curve already plots — you can drop an "eval loss" point onto the same chart as the training loss and read the generalization gap off by eye.

!!! example "Worked example: reading a perplexity number"

    Suppose `compute_perplexity` reports a held-out loss of **2.9 nats/token**, in the "on the order of ~2.8–3.2 nats/token" range documented for the flagship run in [14.7](07-pretraining-run.html). That converts to $\text{PPL} = e^{2.9} \approx 18.2$. In plain terms: on held-out text from the training mix, the model's predictive distribution has an effective branching factor of about 18 tokens at each position — not "18 equally likely tokens" (real text is far more skewed), but a useful single-number summary of average surprise. A random-guessing model over the 32768-token vocabulary would sit at PPL ≈ 32768, i.e. a loss of $\ln 32768 \approx 10.4$ nats — which is exactly where the training curve *started* (Ch. 14.7). The honest framing: this number says the model learned real structure in FineWeb-Edu-like text, and nothing more.

!!! warning "Common pitfall: perplexity is not comparable across tokenizers"

    A perplexity number is defined per *token*, and Stack-100M's tokens are its own 32768-entry byte-level BPE (Ch. 14.3). Comparing PPL 18 here against PPL 12 from a model with a 128k vocabulary is meaningless — the second model's tokens are longer, so it makes fewer, harder predictions per unit of text. The tokenizer-independent version is **bits per byte**: $\text{BPB} = \frac{\text{total nll in nats}}{\ln(2)\cdot\text{total UTF-8 bytes}}$. If you ever want to compare Stack-100M against an external model, compute BPB on the same raw text, not PPL. This is the same trap `lm-evaluation-harness`'s `bits_per_byte` metric exists to avoid.

## 4. The Capability Battery

Perplexity says nothing about whether the model can do anything *useful*. We add four small, cheap, interpretable probes — each deliberately tiny (tens to low hundreds of examples), because at 100M parameters a large eval suite would cost more compute than the pretraining run itself and buy very little additional signal. These are diagnostic probes, not benchmark claims; treat every number as "this model, on this exact small set, on this exact day."

They probe *different* skills, and the asymmetry between them — together with §3's predictive fit — is the finding: reliable narrow reasoning that RLVR targeted (§4.1), broad parametric knowledge the model has no capacity for (§4.2), reading comprehension when the answer is handed over (§4.3), and multi-step tool use, the capstone's flagship deliverable (§4.4).

### 4.0 One chat frame, shared by every probe

Before any probe, fix the wire format. Ch. 14.9's `render_conversation` is the *only* definition of how Stack-100M sees a conversation, and an eval that improvises its own spacing is measuring a distribution shift you introduced. Reconstruct it exactly, once:

```python
# stacklm/eval/probes.py
import re
import torch

from stacklm.post.chat import SPECIAL            # Ch. 14.9's role-token strings


def chat_prompt(user: str, system: str | None = None) -> str:
    """Byte-identical to `render_conversation(..., add_generation_prompt=True)`:
    <|bos|>[<|system|>SYS<|end|>]<|user|>USER<|end|><|assistant|>

    No newlines between turns — Ch. 14.9's template does not emit any, and a
    stray "\\n" is a token the model never saw in that slot. Ch. 14.9's RLVR
    rollouts render the arithmetic task with NO system turn, so §4.1 passes
    `system=None` to match the RL distribution exactly.
    """
    s = SPECIAL["bos"]
    if system is not None:
        s += f"{SPECIAL['system']}{system}{SPECIAL['end']}"
    return s + f"{SPECIAL['user']}{user}{SPECIAL['end']}{SPECIAL['assistant']}"
```

### 4.1 Arithmetic accuracy — grade the task you actually trained

Integer arithmetic is the cleanest probe of *reliable reasoning*: unambiguous, exactly gradable, and — because [14.9](09-post-training.html) ran narrow RLVR on exactly this task — a direct check of whether that RL stage stuck. "Exactly this task" has to be taken literally. Ch. 14.9's prompts read `Compute {a} {op} {b}. Give the final integer after '####'.` with $\text{op} \in \{+,-,\times\}$, and its verifier looks for `#### <int>`. A probe that invents a different surface form (`"37 + 8 = "`) and parses the *first* integer in the completion scores a well-trained RLVR model near zero, because the model dutifully shows work and then emits `#### 45` — and you would blame the model. **Import the task and the verifier; do not re-derive them.**

```python
import random
from stacklm.post.grpo import make_arithmetic_prompt, exact_match_reward  # Ch. 14.9


def make_arithmetic_probe(n: int = 200, seed: int = 0, max_val: int = 99) -> list[dict]:
    """Freeze n problems from Ch. 14.9's RLVR distribution. Seeded, so the probe
    set is reproducible from the seed alone — a hashed, versioned eval set in
    the cheapest possible form."""
    rng = random.Random(seed)
    return [dict(zip(("question", "answer"), make_arithmetic_prompt(rng, max_val=max_val)))
            for _ in range(n)]


@torch.no_grad()
def eval_arithmetic(model, tokenizer, generate_fn, problems: list[dict],
                    max_new_tokens: int = 64, label: str = "in-distribution") -> dict:
    """Greedy-generate, grade with Ch. 14.9's OWN verifier.

    Two disciplines Ch. 14.9 insists on and an eval must not drop:
      * `max_new_tokens=64` matches the RLVR rollout budget. Truncate to 8 and
        you cut off the '####' line and measure your own budget, not the model.
      * PARSE-FAILURE RATE IS REPORTED SEPARATELY from accuracy. "Wrong answer"
        and "unparseable output" demand opposite fixes; conflating them is how
        teams spend a week tuning RL to fix a formatting bug.
    """
    n_correct = n_parsed = 0
    for p in problems:
        completion = generate_fn(model, tokenizer, prompt=chat_prompt(p["question"]),
                                 max_new_tokens=max_new_tokens, temperature=0.0)
        reward, pred = exact_match_reward(completion, p["answer"])
        n_parsed += int(pred is not None)
        n_correct += int(reward == 1.0)
    n = len(problems)
    return {"label": label, "accuracy": n_correct / n,
            "parse_rate": n_parsed / n, "n": n}


# Two rows, always reported together. The second is the generalization probe
# Ch. 14.9 asks for: same format, operands an order of magnitude larger than
# anything RLVR saw. Expect a large gap — RLVR at 100M generalizes NARROWLY,
# and hiding that behind a single averaged number is the dishonest option.
ARITH_PROBES = [
    (make_arithmetic_probe(n=200, seed=0, max_val=99),  "2-digit (in-distribution)"),
    (make_arithmetic_probe(n=100, seed=1, max_val=999), "3-digit (out-of-distribution)"),
]
```

### 4.2 A tiny multiple-choice set, scored the way real harnesses do

We score multiple-choice not by asking the model to output the letter "A"/"B"/"C"/"D" — a 100M model's instruction-following is too weak to reliably format that — but by **cloze scoring**: computing the model's total log-probability of each full answer string appended to the question, and picking the highest. This is the technique EleutherAI's `lm-evaluation-harness` (Gao et al., *A Framework for Few-Shot Language Model Evaluation*) uses for MMLU-style tasks (Hendrycks et al., *Measuring Massive Multitask Language Understanding*, 2021), and it sidesteps formatting fragility that would otherwise dominate the result at this scale.

```python
import torch.nn.functional as F


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, continuation: str,
                     device="cpu") -> tuple[float, int]:
    """Return (sum of log p(token | prefix) over `continuation`, n_cont_tokens).

    The subtle part is the boundary. BPE is not guaranteed to satisfy
        encode(prompt + continuation)[:len(encode(prompt))] == encode(prompt)
    — a merge can straddle the join and re-tokenize the last prompt token. When
    that happens, slicing at len(prompt_ids) silently scores the WRONG tokens
    and the whole probe becomes noise. `lm-evaluation-harness` has the identical
    construction (`whole_enc[len(context_enc):]`) and the identical exposure; we
    make the assumption explicit and loud rather than implicit and silent.

    The practical mitigation is baked into TINY_MC_SET below: put the separating
    space at the START of every continuation (" Paris", not "Paris" after a
    trailing space), so the join lands on a whitespace boundary that byte-level
    BPE pre-tokenization does not merge across.
    """
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + continuation)
    n_ctx = len(prompt_ids)
    if full_ids[:n_ctx] != prompt_ids:
        raise ValueError(
            "BPE boundary violation: encode(prompt) is not a prefix of "
            "encode(prompt + continuation). Move the separator into the "
            "continuation, or score with a byte-aligned re-tokenization.")
    if len(full_ids) == n_ctx:
        raise ValueError("empty continuation after tokenization")

    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits, _ = model(ids)                                   # (1, T, V): logits_to_keep=0
    logp = F.log_softmax(logits[0].float(), dim=-1)          # (T, V)

    # logits[i] predicts token i+1. Continuation tokens live at n_ctx .. T-1,
    # so the predicting rows are n_ctx-1 .. T-2.
    targets = ids[0, n_ctx:]                                 # (n_cont,)
    pred_rows = logp[n_ctx - 1: -1, :]                       # (n_cont, V)
    tok_logp = pred_rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(tok_logp.sum()), int(tok_logp.numel())


# A hand-written, deliberately tiny probe (2 items shown; the set used in
# practice is ~50-100 items spanning basic science, geography, and word sense —
# small enough to eyeball every item for leakage).
TINY_MC_SET = [
    {"question": "The chemical symbol for water is",
     "choices": [" H2O", " CO2", " NaCl", " O2"], "answer_idx": 0},
    {"question": "The capital of France is",
     "choices": [" Paris", " Berlin", " Madrid", " Rome"], "answer_idx": 0},
    # ... remaining items omitted; same {question, choices, answer_idx} shape.
]


def eval_mc_probe(model, tokenizer, mc_set: list[dict] = TINY_MC_SET,
                  device="cpu") -> dict:
    """Reports BOTH metrics `lm-evaluation-harness` reports:
      acc      — raw summed log-prob (the harness's `acc`)
      acc_norm — summed log-prob divided by the continuation's UTF-8 BYTE length
                 (the harness's `acc_norm`), removing the length bias that
                 otherwise favours whichever option tokenizes shortest.
    If they disagree, your choices are length-imbalanced and the raw number is
    measuring string length as much as knowledge."""
    n_raw = n_norm = 0
    for item in mc_set:
        raw, norm = [], []
        for choice in item["choices"]:
            s, _ = sequence_logprob(model, tokenizer, item["question"], choice, device)
            raw.append(s)
            norm.append(s / max(1, len(choice.encode("utf-8"))))
        n_raw += int(max(range(len(raw)), key=lambda i: raw[i]) == item["answer_idx"])
        n_norm += int(max(range(len(norm)), key=lambda i: norm[i]) == item["answer_idx"])
    n = len(mc_set)
    return {"acc": n_raw / n, "acc_norm": n_norm / n, "n": n}
```

Note that the MC probe is scored on the **base LM distribution**, with no chat frame: cloze scoring compares continuations of a plain prefix, and wrapping it in `<|user|>`/`<|assistant|>` would change every candidate's likelihood by a shared but prompt-dependent amount and add a formatting confound for nothing.

### 4.3 Retrieval-QA exact match

This probe reuses the small local corpus and retriever built for the narrow agent in [14.10](10-agentic-narrow.html). It measures a *different* skill than the closed-book MC set: given a **retrieved passage that actually contains the answer**, can the model extract and state it? That isolates reading comprehension from parametric knowledge — a fairer test for a model this small, which has nowhere near the capacity to memorize broad factual knowledge in its weights (Natural Questions, Kwiatkowski et al., 2019, is the large-scale ancestor of this idea).

```python
def normalize_answer(s: str) -> str:
    """Standard SQuAD/NQ-style EM normalization: lowercase, strip punctuation,
    articles, and extra whitespace."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def eval_retrieval_qa(model, tokenizer, generate_fn, retriever,
                      qa_pairs: list[dict]) -> dict:
    """qa_pairs: [{"question": str, "gold_answer": str}, ...]
    retriever: BM25Retriever or HashEmbedRetriever from Ch. 14.10.

    Ch. 14.10's `.search(query, k)` returns list[tuple[Passage, float]] — a
    (passage, score) PAIR, and `Passage` is a dataclass with `.doc_id`/`.text`.
    Interpolating the tuple straight into the prompt (a genuinely easy mistake)
    feeds the model a Python repr and scores ~0 for reasons that have nothing to
    do with the model.

    We also report RETRIEVER RECALL@1 separately. With k=1 this is a joint test
    of two components; without the recall number a bad EM is unattributable —
    you cannot tell whether the model failed to read or the retriever failed to
    fetch."""
    n_correct, n_retrieved = 0, 0
    for pair in qa_pairs:
        hits = retriever.search(pair["question"], k=1)
        passage_text = hits[0][0].text if hits else ""        # (Passage, score) -> .text
        n_retrieved += int(normalize_answer(pair["gold_answer"])
                           in normalize_answer(passage_text))
        prompt = chat_prompt(f"Passage: {passage_text}\nQuestion: {pair['question']}",
                             system="Answer using only the passage below.")
        completion = generate_fn(model, tokenizer, prompt=prompt, max_new_tokens=16,
                                 temperature=0.0)   # specials enabled by default (§2)
        n_correct += int(normalize_answer(completion) == normalize_answer(pair["gold_answer"]))
    n = len(qa_pairs)
    return {"exact_match": n_correct / n, "retriever_recall@1": n_retrieved / n, "n": n}
```

### 4.4 Evaluating the agent loop

PLAN §9 makes the narrow ReAct "auto-research" agent the capstone's flagship deliverable — and a single-shot retrieval-QA probe that calls `retriever.search` itself *bypasses the agent entirely*. If you stop at §4.3, you have not measured the thing you spent Ch. 14.10 building. Agent quality is not one number; it decomposes, and the decomposition is the diagnostic:

| Metric | What it tells you | What a bad value means |
|---|---|---|
| Tool-call **format validity** | fraction of emitted tool-call blocks that parse as valid JSON naming a known tool | distillation didn't lock in the grammar; more/cleaner traces, or constrained decoding |
| Tool-**choice accuracy** | first tool chosen == the gold tool for that task | the model learned the *format* but not the *policy* |
| **Turns** (mean/median) and **cap rate** | efficiency, and how often it never terminates | non-termination is a harness problem as much as a model problem |
| Final-answer **EM** | end-to-end success | the only metric a user feels |
| Failure **taxonomy** | *why* the failures happen | tells you which of the four above to fix first |

```python
"""
stacklm/eval/agent.py — evaluate the Ch. 14.10 ReAct loop end to end.
Runs the REAL loop (model drives, tools execute) and decomposes the outcome.
"""
from dataclasses import dataclass
from collections import Counter
import statistics

from stacklm.agent.react import parse_assistant_step
from stacklm.agent.loop import run_agent          # returns (answer, trace)
from stacklm.agent.distill import ToolEnv, normalize


@dataclass
class AgentTask:
    question: str
    gold: str                    # verifiable final answer (exact match)
    gold_tool: str               # "search" or "calc" — the tool a competent agent picks first
    gold_evidence: str = ""      # substring that MUST appear in an observation if
                                 # retrieval worked; enables the "read it and
                                 # ignored it" diagnosis


def eval_agent(model, tok, env: ToolEnv, tasks: list[AgentTask],
               max_steps: int = 6) -> dict:
    n_calls = n_parseable = n_tool_right = n_solved = n_capped = 0
    turns: list[int] = []
    taxonomy: Counter[str] = Counter()

    for task in tasks:
        answer, trace = run_agent(model, tok, task.question, env, max_steps=max_steps)
        assistant = [t for kind, t in trace if kind == "assistant"]
        observations = [t for kind, t in trace if kind == "observation"]
        acts = [parse_assistant_step(t) for t in assistant]
        calls = [a for a in acts if a.kind == "tool"]

        n_calls += len(calls)
        n_parseable += sum(1 for a in calls if a.tool != "__malformed__")
        first_tool = calls[0].tool if calls else None
        n_tool_right += int(first_tool == task.gold_tool)

        # CAPPING IS A TRACE-SHAPE PROPERTY, NOT A LAST-ACTION PROPERTY.
        # Ch. 14.10's run_agent returns the instant an emission parses as
        # "final", so every terminating path leaves AT MOST max_steps assistant
        # emissions. The only way to get max_steps + 1 is the forced-synthesis
        # fallback appended after the step budget is exhausted — and that
        # fallback ("Thought: I must answer now.\nAnswer: ...") itself parses as
        # kind == "final". Testing `acts[-1].kind != "final"` therefore never
        # fires: it is dead code, and cap_rate would be a structural zero.
        capped = len(assistant) > max_steps
        n_capped += int(capped)
        turns.append(max_steps if capped else len(assistant))   # don't count the fallback

        solved = normalize(answer) == normalize(task.gold)
        n_solved += int(solved)

        # Failure taxonomy, in priority order — each task lands in exactly one bucket.
        evidence_seen = bool(task.gold_evidence) and any(
            task.gold_evidence.lower() in o.lower() for o in observations)
        if solved:
            taxonomy["ok"] += 1
        elif any(a.tool == "__malformed__" for a in calls):
            taxonomy["malformed_call"] += 1
        elif any(o.startswith("RepeatedCall:") for o in observations):
            taxonomy["looped"] += 1
        elif capped:
            taxonomy["non_termination"] += 1
        elif not calls:
            taxonomy["never_called_a_tool"] += 1
        elif first_tool != task.gold_tool:
            taxonomy["wrong_tool"] += 1
        elif evidence_seen:
            taxonomy["retrieved_but_ignored"] += 1
        else:
            taxonomy["other"] += 1

    n = len(tasks)
    return {
        "n_tasks": n,
        "tool_format_validity": (n_parseable / n_calls) if n_calls else float("nan"),
        "tool_choice_acc": n_tool_right / n,
        "turns_mean": statistics.mean(turns),
        "turns_median": statistics.median(turns),
        "cap_rate": n_capped / n,
        "exact_match": n_solved / n,
        "taxonomy": dict(taxonomy),
    }
```

!!! tip "Practitioner tip: make the harness *say* it capped"

    Inferring "we hit the step cap" from `len(assistant) > max_steps` is correct for Ch. 14.10's loop, but it is an inference about someone else's control flow, and it silently breaks the day `run_agent` grows a second early-exit path. The robust version is a one-line change on the producer side: have the forced-synthesis branch append a sentinel to the trace — `trace.append(("forced_synthesis", ""))` — and let the eval check `any(kind == "forced_synthesis" for kind, _ in trace)`. Prefer explicit signals from the harness over reconstructions in the scorer; this is the same instinct that makes `inspect_ai` log per-step tool records rather than a flat transcript.

Thirty hand-written tasks are enough to be informative and small enough that you can read every trace. Read them. The taxonomy is a *hypothesis generator*, not a verdict — `retrieved_but_ignored` dominating means the bottleneck is the model's extraction, not the retriever; `malformed_call` dominating means you need more distillation traces or **constrained decoding** (a JSON grammar enforced at sampling time, per [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)), which is by far the cheapest fix and the standard production answer.

!!! tip "Practitioner tip: the libraries that own this layer"

    Three open-source harnesses do this properly; graduate to them once your probe set outgrows a Python list.

    - **`lm-evaluation-harness`** (EleutherAI) — the de-facto standard for log-likelihood/cloze tasks; backs the Hugging Face Open LLM Leaderboard. You plug in a custom model by subclassing `lm_eval.api.model.LM` and implementing exactly two primitives — `loglikelihood(requests)` and `generate_until(requests)` — which are *precisely* `sequence_logprob` and `generate_text` from this chapter. Not a coincidence: those two primitives span nearly all static LLM evaluation.
    - **`lighteval`** (Hugging Face) — a lighter, more hackable harness with first-class custom-task and custom-model entry points; good when you want your own metric without forking a large codebase.
    - **`inspect_ai`** (UK AI Safety Institute) — the framework built for *agentic* evaluation: solvers, tool sandboxes, scorers, and full trace logging with a viewer. It formalizes exactly the decomposition in `eval_agent` (per-step tool records, turn limits, scorer-per-metric). If your agent eval grows past one file, port it rather than reinventing the trace format. See [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html) and [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html).

!!! example "Worked example: a plausible probe report"

    Running the full battery against the post-trained Stack-100M checkpoint produces a table with this *structure* (magnitudes are illustrative — replace every cell with your own run's numbers):

    | Probe | Metric | Score | n |
    |---|---|---|---|
    | Held-out perplexity | PPL | ~18 | 2M tokens |
    | Arithmetic, 2-digit (in-dist.) | EM / parse rate | on the order of 60-85% / >95% | 200 |
    | Arithmetic, 3-digit (OOD) | EM / parse rate | far lower; parse rate ~unchanged | 100 |
    | Tiny MC set | acc / acc_norm | on the order of 40-60% | ~80 |
    | Retrieval-QA | EM / recall@1 | on the order of 50-75% / >90% | ~50 |
    | Agent loop | tool-format validity | on the order of 85-98% | ~30 |
    | Agent loop | tool-choice acc | on the order of 60-85% | ~30 |
    | Agent loop | end-to-end EM | on the order of 30-55% | ~30 |

    Three patterns such a table typically reveals, each of which should be *reported*, not smoothed over. (1) Arithmetic and retrieval-QA — narrowly scoped, RL/SFT-targeted, unambiguously gradable — score well above chance, while the closed-book MC set, which needs broad parametric world knowledge the model never had capacity to store, sits closer to a coin flip. (2) The in-distribution/OOD arithmetic gap with a *flat parse rate* is the signature of narrow RLVR: the model kept the format everywhere and the competence only where it was trained. (3) The agent's *format* validity is far higher than its *end-to-end* success — distillation reliably teaches a 100M model the wire grammar and much less reliably teaches it the policy. That third gap is the single most characteristic signature of a small distilled agent.

## 5. The Honest Capability Report

The single most important artifact of this chapter is not a number — it is a paragraph. Every probe measures a narrow slice; the reader of your model card needs the slices assembled into an honest picture. Write it down explicitly, next to the numbers:

- **Stack-100M is a narrow tool, not a general oracle.** At ~101M parameters and ~20B training tokens, it sits roughly three orders of magnitude below a frontier model in both parameters and effective training FLOPs. It will confidently state incorrect facts, struggle with multi-step reasoning beyond what narrow RLVR explicitly trained, and its "knowledge" is a lossy compression of a filtered web+synthetic corpus, not a queryable database.
- **What it is reliably good at** is precisely the narrow, scaffolded tasks it was pointed at: short-form chat in the SFT/DPO style, two-digit arithmetic *in the trained format*, and grounded retrieval-QA when the answer is handed to it in context — the ReAct loop from [14.10](10-agentic-narrow.html) exists specifically because retrieval + a small model beats a small model alone on knowledge-heavy questions.
- **What it is not good at**: long-horizon reasoning, closed-book trivia outside the training mix's coverage, arithmetic outside the operand range RLVR saw, code beyond the toy StarCoder-subset flavor, and — like every language model — it will hallucinate fluently with no internal uncertainty signal a downstream system can cheaply detect.
- **The agent is format-reliable and policy-fragile.** Quote the §4.4 decomposition explicitly: a model that emits well-formed tool calls 95% of the time and solves 40% of tasks is *not* "40% as good as a real agent" — it has memorized a grammar and is guessing at a policy. Say that, because an integrator who sees only the EM number will add guardrails in the wrong place.
- **State the eval-set provenance and size.** "n = 30 hand-written tasks, seed 0, committed at commit `abc1234`" is worth more than a number with two decimal places and no provenance.

This is the evaluation ethic developed at length in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html): a benchmark score is only as trustworthy as the disclosure that accompanies it.

!!! warning "Contamination: the probe you build is the probe you must audit"

    Because you constructed the MC set and the arithmetic generator yourself, contamination risk is lower than for a public leaderboard — but it is not zero. Three concrete leaks:

    1. **Data-pipeline leakage.** If FineWeb-Edu or Cosmopedia contains near-duplicates of your hand-written MC questions (surprisingly plausible for well-known trivia like "capital of France"), the model may be pattern-matching memorized web text rather than reasoning. Cross-check MC items against the deduplication index from [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html).
    2. **Retrieval-QA leakage.** If the small local corpus overlaps pretraining data verbatim, the model may already have the passage memorized and the probe silently degrades into a closed-book test wearing an open-book costume. Build the retrieval corpus from sources *excluded* from the pretraining manifest.
    3. **Agent-task leakage into distillation.** The nastiest, because it is self-inflicted: if the ~30 evaluation tasks overlap the tasks you generated teacher trajectories for in Ch. 14.10, you are measuring memorization of specific trajectories, not tool-use ability. Hold the eval tasks out *before* the distillation rollout, by hash, and never let them into the SFT mix.

    More generally: any number you cannot regenerate from a documented, hashed, versioned eval set is not a number — it is an anecdote. At $n = 30$ and an observed 40% success rate, a Wilson 95% interval spans roughly 25–58%, so a rival configuration scoring 50% is *not* meaningfully better. Reporting it as an improvement is the single most common error in small-scale eval; see [Statistical Rigor in Evaluation](../11-evaluation/06-statistical-rigor-eval.html).

{{fig:honest-eval-four-probes-asymmetry}}

## 6. Post-Training Quantization: RTN, GPTQ, and AWQ

With honest numbers in hand, the second half is serving. The fp32 checkpoint is ~405MB (101.3M quantizable parameters × 4 bytes — exact accounting in [14.4](04-architecture.html)); even at bf16 that is ~203MB, comfortably in RAM on any laptop but wasteful given how little precision a well-trained weight actually needs at inference. **Post-training quantization (PTQ)** converts trained fp32/bf16 weights to low-bit integers *after* training, no gradient updates required — the natural complement to the quantization-aware and mixed-precision *training* techniques in [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html). This section is a hands-on companion to the book's dedicated PTQ chapters — [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html) and [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html) — read those for full derivations; here we implement the baseline end to end and *use* it.

### 6.1 Round-to-nearest (RTN): the baseline

RTN is exactly what it sounds like: pick a scale (and, for asymmetric schemes, a zero-point) per group of weights from that group's min/max, then round each weight to the nearest representable integer. For a $b$-bit **symmetric** scheme over a group of weights $w$:

$$
s = \frac{\max(|w|)}{2^{b-1}-1}, \qquad q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right),\, -(2^{b-1}-1),\, 2^{b-1}-1\right), \qquad \hat w = s\,q
$$

For a $b$-bit **asymmetric** scheme (uses the full unsigned range, better for skewed distributions):

$$
s = \frac{\max(w)-\min(w)}{2^b-1}, \qquad z = \operatorname{round}\!\left(\frac{-\min(w)}{s}\right), \qquad q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right)+z,\, 0,\, 2^b-1\right), \qquad \hat w = s\,(q-z)
$$

RTN's virtue is that it is a single pass over the weights with no calibration data and no per-layer optimization — you can quantize a checkpoint in seconds. Its vice is that it treats every weight independently: it has no notion that some weights matter more to the model's output than others, so at 4 bits (only 16 representable values per group) it can measurably hurt quality, especially on outlier-heavy channels.

{{fig:rtn-quantization-number-line}}

### 6.2 GPTQ: reconstruction-aware quantization

**GPTQ** (Frantar, Ashkboos, Hoefler & Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022) fixes RTN's blind spot by quantizing each linear layer's weight matrix **column by column**, and after quantizing each column, *updating all not-yet-quantized columns to compensate for the error just introduced*. For a layer with calibration activations $X$ (real hidden states run through the layer) and weight $W$, GPTQ builds the Hessian of the layer's squared-error reconstruction objective, $H = 2XX^\top + \lambda I$ (the damping term keeps it invertible), and processes input-feature columns $q = 1, 2, \dots$ in order:

$$
\hat w_{:,q} = \operatorname{quant}(w_{:,q}), \qquad
\delta = \frac{w_{:,q} - \hat w_{:,q}}{[H^{-1}]_{qq}}, \qquad
w_{:,j} \mathrel{-}= \delta \cdot [H^{-1}]_{qj} \;\; \forall\, j > q
$$

The update spreads each column's quantization error onto the columns not yet quantized, weighted by how strongly the Hessian says those columns interact — an efficient, layer-local instance of the classic Optimal Brain Surgeon idea (LeCun et al., 1990; Hassibi & Stork, 1993) applied to quantization instead of pruning. The payoff is that GPTQ can push to 4 or even 3 bits with far less quality loss than RTN, at the cost of calibration data and $O(d_{in}^3)$ work per layer for the Hessian inverse (real GPTQ batches this via Cholesky decomposition; the sketch below is the pedagogical, unoptimized version):

```python
import torch


def gptq_quantize_column_by_column(W: torch.Tensor, X_calib: torch.Tensor, bits: int = 4,
                                   damp: float = 1e-2) -> torch.Tensor:
    """Simplified reference implementation of GPTQ's per-layer reconstruction.
    W: (d_out, d_in) weight matrix. X_calib: (n_samples, d_in) calibration
    activations captured from a real forward pass on held-out text.

    Pedagogical: real GPTQ batches columns and uses a running Cholesky
    factorization for speed. For production use GPTQModel or llm-compressor
    (Section 9) — do not ship this."""
    d_out, d_in = W.shape
    W = W.clone().float()

    # Hessian of the layer's local least-squares reconstruction objective.
    H = 2 * (X_calib.T @ X_calib) / X_calib.shape[0]
    H += damp * torch.eye(d_in, device=W.device)          # damping for stability
    H_inv = torch.linalg.inv(H)

    qmax = 2 ** (bits - 1) - 1
    for q in range(d_in):
        col = W[:, q]
        scale = col.abs().max() / qmax if col.abs().max() > 0 else 1.0
        q_col = torch.clamp(torch.round(col / scale), -qmax, qmax)
        w_hat = q_col * scale
        error = (col - w_hat) / H_inv[q, q]                # per-output-row error
        W[:, q] = w_hat
        if q + 1 < d_in:
            # Spread the reconstruction error onto not-yet-quantized columns.
            W[:, q + 1:] -= torch.outer(error, H_inv[q, q + 1:])
    return W
```

### 6.3 AWQ: protect the salient channels instead of correcting for them

**AWQ** (Lin, Tang, Tang, Yang, Dang & Han, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023; MLSys 2024 Best Paper) takes a cheaper angle. Its empirical observation: a small fraction of *input channels* (about 0.1–1% in practice) receive systematically large activation magnitudes and disproportionately affect the layer's output — quantization error on the weights feeding those channels hurts far more than error elsewhere. Rather than correcting errors after the fact like GPTQ, AWQ **protects the salient channels before quantizing** by rescaling: it multiplies weight columns corresponding to high-activation channels by a per-channel scale $s>1$ (making them larger and therefore relatively less perturbed by rounding) and divides the corresponding activations by the same $s$ — leaving the mathematical output of $W^\top x$ unchanged while shifting quantization error away from the channels that matter most:

$$
y = W^\top x = (W \cdot \operatorname{diag}(s))^\top (x / s)
$$

The scale vector $s$ is found by a small grid search over a single exponent $\alpha$ (with $s = a^{\alpha}$ for per-channel activation magnitudes $a$), using calibration data but no backprop and no Hessian inversion — substantially cheaper than GPTQ while empirically competitive at 4-bit. That is a good default when calibration compute is tight, which is exactly our situation on a single rented GPU.

```python
def _fake_quant_grouped(W: torch.Tensor, bits: int = 4, group_size: int = 64):
    """Symmetric per-group RTN, dequantized straight back to fp32 ('fake quant').
    Same grid the reference path in Section 7 ships, so the search optimizes the
    quantizer we actually use rather than a different, coarser one."""
    d_out, d_in = W.shape
    g = W.view(d_out, d_in // group_size, group_size)
    qmax = 2 ** (bits - 1) - 1
    s = (g.abs().amax(dim=2, keepdim=True) / qmax).clamp(min=1e-8)
    return (torch.round(g / s).clamp(-qmax, qmax) * s).view(d_out, d_in)


def awq_search_channel_scales(W: torch.Tensor, X_calib: torch.Tensor, bits: int = 4,
                              group_size: int = 64, n_grid: int = 20) -> torch.Tensor:
    """AWQ's salient-channel search. W: (d_out, d_in). X_calib: (n_samples, d_in).

    THE OBJECTIVE MUST BE OUTPUT ERROR MEASURED AFTER DE-SCALING. A tempting
    shortcut — the squared error of the SCALED matrix against its quantization —
    is monotonically increasing in alpha (larger salient channels => larger
    group max => larger step size), so its argmin is always alpha = 0 and the
    whole search silently degenerates into a no-op. Measure what you care about:
    the layer's output.

    Because alpha = 0 (s = 1, plain RTN) is in the grid, the returned scale can
    never be worse than RTN on the calibration set. That is the property that
    makes a grid search over one scalar a legitimate algorithm rather than a
    hopeful heuristic.
    """
    act = X_calib.abs().mean(dim=0).clamp(min=1e-5)   # per-input-channel saliency
    act = act / act.mean()                            # >1 on salient channels
    ref = X_calib @ W.T                               # fp32 reference output

    best_s, best_err = torch.ones_like(act), float("inf")
    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = act.pow(alpha)
        W_hat = _fake_quant_grouped(W * s.unsqueeze(0), bits, group_size) / s.unsqueeze(0)
        err = (X_calib @ W_hat.T - ref).pow(2).mean().item()
        if err < best_err:
            best_err, best_s = err, s
    return best_s     # fold s into W; fold 1/s into the PRECEDING norm/layer
```

That last comment is where AWQ's real engineering lives: the rescaling is only free if you can absorb $1/s$ into something already applying a per-channel multiply. In a pre-norm LLaMA-style block — which Stack-100M is — that something is the RMSNorm weight $\gamma$ immediately before the projection. This is exactly why AWQ (and SmoothQuant) are described as "norm-folding" methods, and why they need architecture awareness rather than being a generic tensor operation. It also means the two projections that consume the *same* normalized hidden state must share one $s$: in Stack-100M, `wq`/`wk`/`wv` all read the attention norm's output, and `gate`/`up` both read the MLP norm's output, so each group has to be searched jointly. That constraint is why you should reach for a maintained implementation (§9) rather than shipping the sketch above.

Stack-100M ships with RTN as the reference, fully implemented path in §7 — correct, simple, sufficient at int8. Whether it is sufficient at int4 is an empirical question, and §8 answers it with code instead of assertion. GPTQ and AWQ are the production upgrade path once you find the gap.

{{fig:gptq-vs-awq-two-philosophies}}

## 7. Implementing int8 and int4 Weight-Only Quantization

The implementation below quantizes every `nn.Linear` weight in `Stack100M` (attention projections, SwiGLU gate/up/down, and the tied embedding/output projection) and leaves RMSNorm scales at fp32 — 35,072 parameters total, a vanishingly small fraction, and extremely error-sensitive. int8 uses per-output-channel (row-wise) symmetric scales, nearly free in memory and already capturing most of the benefit at 8 bits. int4 uses per-group asymmetric scales with `group_size=64` — `d_model=512`, `intermediate=1408`, and `n_heads*head_dim=512` all divide evenly by 64 — because at 4 bits a per-row scale is not enough precision.

### 7.1 The primitives

```python
"""
stacklm/serve/quantize.py — round-to-nearest int8/int4 weight-only PTQ, with a
packed on-disk format for CPU inference (Ch. 14.11 reference path).
"""
import json
import torch
import torch.nn as nn


def quantize_int8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8. weight: (d_out, d_in).
    Returns (int8 codes, fp32 scales of shape (d_out,))."""
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0          # (d_out,)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127)
    return q.to(torch.int8), scale


def dequantize_int8_per_row(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(1)


def quantize_int4_grouped(weight: torch.Tensor, group_size: int = 64):
    """Asymmetric per-group int4, packed two values per uint8 byte.
    weight: (d_out, d_in), d_in divisible by group_size.
    Returns (packed uint8 (d_out, d_in//2), scale, zero_point),
    scale/zero_point of shape (d_out, d_in // group_size)."""
    d_out, d_in = weight.shape
    assert d_in % group_size == 0, f"d_in={d_in} not divisible by group_size={group_size}"
    n_groups = d_in // group_size

    w = weight.float().view(d_out, n_groups, group_size)
    w_min, w_max = w.amin(dim=2), w.amax(dim=2)                  # (d_out, n_groups)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)             # 4-bit unsigned: 0..15
    zero_point = torch.round(-w_min / scale)                     # integer zero-point

    q = torch.round(w / scale.unsqueeze(2) + zero_point.unsqueeze(2)).clamp(0, 15)
    q = q.view(d_out, d_in).to(torch.uint8)

    # Pack two nibbles per byte: low nibble = even column, high nibble = odd column.
    q_even, q_odd = q[:, 0::2], q[:, 1::2]
    packed = (q_even | (q_odd << 4)).to(torch.uint8)             # (d_out, d_in // 2)
    return packed, scale, zero_point


def dequantize_int4_grouped(packed, scale, zero_point, d_in: int, group_size: int = 64):
    d_out = packed.shape[0]
    n_groups = d_in // group_size
    q_even = (packed & 0x0F).to(torch.float32)
    q_odd = ((packed >> 4) & 0x0F).to(torch.float32)
    q = torch.empty(d_out, d_in, device=packed.device)
    q[:, 0::2], q[:, 1::2] = q_even, q_odd
    q = q.view(d_out, n_groups, group_size)
    w = (q - zero_point.unsqueeze(2)) * scale.unsqueeze(2)
    return w.view(d_out, d_in)
```

### 7.2 `QuantizedLinear`, and the row-gather the tied embedding needs

```python
class QuantizedLinear(nn.Module):
    """Memory-efficient stand-in for nn.Linear. Stores weights packed at int8 or
    int4; on each forward, dequantizes to fp32 and does a standard matmul.

    This is the *pedagogically* correct approach — it is exactly what the weight
    compression buys you — but it does NOT give a matmul speedup, since PyTorch
    still computes in fp32 and the unpack is extra work. Section 9.1 measures
    that honestly. Real speedups need integer-native kernels."""

    def __init__(self, bits: int, group_size: int = 64):
        super().__init__()
        assert bits in (4, 8)
        self.bits, self.group_size = bits, group_size
        self.d_in = self.d_out = None

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int, group_size: int = 64):
        layer = cls(bits, group_size)
        layer.d_in, layer.d_out = linear.in_features, linear.out_features
        # register (not plain-assign) so a bias round-trips through state_dict;
        # Stack-100M is LLaMA-style bias-free, so this is normally None.
        layer.register_buffer(
            "bias", linear.bias.detach().clone() if linear.bias is not None else None)
        if bits == 8:
            q, scale = quantize_int8_per_row(linear.weight.data)
            layer.register_buffer("q_weight", q)
            layer.register_buffer("scale", scale)
        else:
            packed, scale, zp = quantize_int4_grouped(linear.weight.data, group_size)
            layer.register_buffer("q_weight", packed)
            layer.register_buffer("scale", scale)
            layer.register_buffer("zero_point", zp)
        return layer

    @classmethod
    def empty(cls, in_features: int, out_features: int, bits: int,
              group_size: int = 64, device="meta"):
        """Allocate correctly-shaped buffers WITHOUT ever materializing the fp32
        weight. This is what you want at LOAD time: building a real fp32
        Stack100M just to quantize and throw it away costs 405 MB of peak RSS on
        a machine whose whole point is that it only has to hold 63 MB."""
        layer = cls(bits, group_size)
        layer.d_in, layer.d_out = in_features, out_features
        layer.register_buffer("bias", None)
        if bits == 8:
            layer.register_buffer("q_weight", torch.empty(out_features, in_features,
                                                          dtype=torch.int8, device=device))
            layer.register_buffer("scale", torch.empty(out_features, dtype=torch.float32,
                                                       device=device))
        else:
            n_g = in_features // group_size
            layer.register_buffer("q_weight", torch.empty(out_features, in_features // 2,
                                                          dtype=torch.uint8, device=device))
            layer.register_buffer("scale", torch.empty(out_features, n_g,
                                                       dtype=torch.float32, device=device))
            layer.register_buffer("zero_point", torch.empty(out_features, n_g,
                                                            dtype=torch.float32, device=device))
        return layer

    # ---- dequantization --------------------------------------------------
    def dequantize(self) -> torch.Tensor:
        if self.bits == 8:
            return dequantize_int8_per_row(self.q_weight, self.scale)
        return dequantize_int4_grouped(self.q_weight, self.scale, self.zero_point,
                                       self.d_in, self.group_size)

    def dequantize_rows(self, idx: torch.Tensor) -> torch.Tensor:
        """Dequantize ONLY the requested output rows — the embedding lookup path.
        Gathering 2048 rows of a 32768-row table costs 6% of a full dequantize."""
        if self.bits == 8:
            return self.q_weight[idx].float() * self.scale[idx].unsqueeze(1)
        return dequantize_int4_grouped(self.q_weight[idx], self.scale[idx],
                                       self.zero_point[idx], self.d_in, self.group_size)

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility shim for Ch. 14.4, which reaches for `.weight` in two
        places: `fused_ce_z_loss(x, self.lm_head.weight, ...)` when
        `cfg.loss_chunk > 0`, and `estimate_params()`'s `tok_emb.weight.numel()`.
        Without this property both raise AttributeError the moment the model is
        quantized — a sibling chapter's documented API breaking silently.

        It MATERIALIZES the full fp32 tensor, so it is a correctness shim, not a
        fast path. `quantize_stacklm` forces `cfg.loss_chunk = 0` for exactly
        that reason."""
        return self.dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize()
        return torch.nn.functional.linear(x, w.to(x.dtype), self.bias)


class QuantizedEmbedding(nn.Module):
    """Token lookup that reads the SAME packed buffers as the quantized, tied
    `lm_head`, dequantizing only the gathered rows.

    Why this class must exist: Ch. 14.4 ties the weights by aliasing
    (`lm_head.weight = tok_emb.weight`). Replacing `lm_head` with a
    QuantizedLinear BREAKS the alias — `tok_emb` keeps its own fp32 Parameter,
    it stays in state_dict(), and the "quantized" export silently carries a
    16.78M x 4 B = 67 MB fp32 table alongside the packed weights, roughly
    doubling every headline in this chapter."""

    def __init__(self, head: QuantizedLinear):
        super().__init__()
        # Deliberately NOT a submodule: nn.Module.__setattr__ would register it,
        # duplicating every key as `tok_emb._head.*` in state_dict(). We want the
        # packed buffers to appear exactly once, under `lm_head.*`.
        object.__setattr__(self, "_head", head)

    @property
    def weight(self) -> torch.Tensor:
        return self._head.dequantize()          # same shim, same reason

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        rows = self._head.dequantize_rows(idx.reshape(-1))     # (n, d_model)
        return rows.view(*idx.shape, -1)
```

### 7.3 Walking the model, tie included

```python
def _swap_linears(module: nn.Module, bits: int, group_size: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            assert bits == 8 or child.in_features % group_size == 0, (
                f"{name}: in_features={child.in_features} does not tile at "
                f"group_size={group_size}")
            setattr(module, name, QuantizedLinear.from_float(child, bits, group_size))
        else:
            _swap_linears(child, bits, group_size)


def quantize_stacklm(model: nn.Module, bits: int, group_size: int = 64,
                     embedding_bits: int | None = None) -> nn.Module:
    """Replace every nn.Linear with a QuantizedLinear, then re-establish the
    embedding tie against the quantized head.

    RMSNorm scales stay fp32 (35,072 params ~= 0.14 MB — negligible and
    error-sensitive; every production format does the same).

    `embedding_bits` keeps the tied embedding/head at a HIGHER precision than
    the body — e.g. `bits=4, embedding_bits=8`. Not a hedge: the 32768 x 512
    table is 17% of the parameters and its rare-token rows are thinly trained,
    so it is disproportionately fragile at 4 bits. llama.cpp's mixed "K-quant"
    recipes make the same choice. Section 8 lets you measure whether it matters
    for YOUR checkpoint instead of guessing."""
    tied = (getattr(model, "cfg", None) is not None
            and model.cfg.tie_embeddings
            and model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr())

    head_linear = model.lm_head if tied else None
    if tied and embedding_bits is not None and embedding_bits != bits:
        # Quantize the head separately, at its own bit-width, BEFORE the walk.
        model.lm_head = QuantizedLinear.from_float(head_linear, embedding_bits, group_size)

    _swap_linears(model, bits, group_size)

    if tied:
        model.tok_emb = QuantizedEmbedding(model.lm_head)   # drops the fp32 Parameter

    if getattr(model, "cfg", None) is not None and model.cfg.loss_chunk:
        # Ch. 14.4's chunked fused CE path calls lm_head.weight per chunk, which
        # on a quantized model means a 67 MB materialization per chunk. We never
        # need a training loss at serve time; turn it off explicitly rather than
        # letting the shim hide a 100x slowdown.
        model.cfg.loss_chunk = 0
    return model


def build_quantized_shell(cfg, bits: int, group_size: int = 64,
                          embedding_bits: int | None = None):
    """Construct the quantized module tree on the `meta` device — right shapes,
    zero bytes — ready for `load_state_dict(..., assign=True)`."""
    from stacklm.model.transformer import Stack100M
    with torch.device("meta"):
        model = Stack100M(cfg)
    tied = cfg.tie_embeddings
    eb = embedding_bits or bits
    model.lm_head = QuantizedLinear.empty(cfg.d_model, cfg.vocab_size, eb, group_size)
    for blk in model.blocks:
        a, m = blk.attn, blk.mlp
        a.wq = QuantizedLinear.empty(cfg.d_model, cfg.n_heads * cfg.head_dim, bits, group_size)
        a.wk = QuantizedLinear.empty(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bits, group_size)
        a.wv = QuantizedLinear.empty(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bits, group_size)
        a.wo = QuantizedLinear.empty(cfg.n_heads * cfg.head_dim, cfg.d_model, bits, group_size)
        m.gate = QuantizedLinear.empty(cfg.d_model, cfg.intermediate, bits, group_size)
        m.up = QuantizedLinear.empty(cfg.d_model, cfg.intermediate, bits, group_size)
        m.down = QuantizedLinear.empty(cfg.intermediate, cfg.d_model, bits, group_size)
    if tied:
        model.tok_emb = QuantizedEmbedding(model.lm_head)
    return model


def state_dict_bytes(model: nn.Module) -> float:
    """Actual bytes, in MB, deduplicated by storage.

    Deduplication is not pedantry: an UNQUANTIZED tied model has BOTH
    `tok_emb.weight` and `lm_head.weight` in its state_dict pointing at one
    16.78M-element tensor, so the naive sum reports ~472 MB for a 405 MB model
    and every compression ratio you compute against it is wrong."""
    seen, total = set(), 0
    for t in model.state_dict().values():
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total / 1e6
```

Verify the tie was actually handled, because this is the failure that costs you 2× silently:

```python
>>> m = Stack100M(StackConfig())
>>> [k for k in m.state_dict() if "tok_emb" in k or k == "lm_head.weight"]
['tok_emb.weight', 'lm_head.weight']              # aliased, but BOTH keys exist
>>> round(state_dict_bytes(m), 1)
405.4                                             # MB, counting the tie ONCE
>>> _ = quantize_stacklm(m, bits=4, group_size=64)
>>> [k for k in m.state_dict() if "tok_emb" in k or "lm_head" in k]
['lm_head.q_weight', 'lm_head.scale', 'lm_head.zero_point']   # no fp32 table anywhere
>>> round(state_dict_bytes(m), 1)
63.5                                              # 63.3 packed + 0.14 fp32 norms
```

That last line is the acceptance test for this whole section: **sum the actual bytes in the actual `state_dict`.** A quantization implementation that cannot pass a byte-count assertion has not quantized anything.

### 7.4 Export: safetensors, not pickle

```python
from safetensors.torch import save_file, load_file


def export_quantized(model: nn.Module, path_prefix: str, bits: int, group_size: int,
                     config: dict, embedding_bits: int | None = None) -> None:
    """Serialize to `<prefix>.safetensors` + `<prefix>.json`.

    safetensors (huggingface/safetensors) over torch.save: no pickle (so loading
    an untrusted checkpoint cannot execute code), a readable header, and
    zero-copy mmap on load — which is why cold-start load time in Section 9 is
    dominated by page faults rather than deserialization.

    safetensors REFUSES aliased tensors, which is a feature here: it is a
    machine-checked version of the Section 7.3 tie assertion."""
    sd = model.state_dict()
    ptrs = [t.data_ptr() for t in sd.values()]
    assert len(set(ptrs)) == len(ptrs), (
        "aliased tensors in state_dict — the embedding tie was not collapsed")
    save_file({k: v.contiguous() for k, v in sd.items()}, path_prefix + ".safetensors")
    with open(path_prefix + ".json", "w") as f:
        json.dump({"bits": bits, "group_size": group_size,
                   "embedding_bits": embedding_bits or bits,
                   "format": "stacklm-rtn-v1", "architecture": config}, f, indent=2)


def load_quantized(path_prefix: str, device="cpu"):
    """Load without ever allocating the fp32 model."""
    from stacklm.config import StackConfig
    with open(path_prefix + ".json") as f:
        meta = json.load(f)
    cfg = StackConfig(**meta["architecture"])
    model = build_quantized_shell(cfg, meta["bits"], meta["group_size"],
                                  meta.get("embedding_bits"))
    sd = load_file(path_prefix + ".safetensors", device=device)
    # assign=True (PyTorch >= 2.1) REPLACES the meta params/buffers with the
    # loaded tensors rather than copying into them — required here, because you
    # cannot copy into a meta tensor.
    model.load_state_dict(sd, assign=True, strict=True)
    # RoPE tables are persistent=False, so they are NOT in the state_dict and are
    # still on `meta`. Rebuild them or the first forward dies on a meta tensor.
    model.rebuild_rope(cfg.max_seq_len, cfg.rope_theta, device=device)
    model.eval()
    return model, meta
```

And the six lines that actually *produce* the artifact §9's CLI loads — without this module, `--bits 4` has nothing to open:

```python
"""
stacklm/serve/export.py — fp32 post-trained checkpoint -> quantized artifact.
Run: `python -m stacklm.serve.export --bits 4 --embedding_bits 8`
"""
import argparse
from dataclasses import asdict

import torch

from stacklm.config import StackConfig
from stacklm.model.transformer import Stack100M
from stacklm.serve.quantize import (quantize_stacklm, export_quantized,
                                    state_dict_bytes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/stack100m/post_final.pt")
    p.add_argument("--out_dir", default="checkpoints/stack100m")
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--embedding_bits", type=int, default=8,
                   help="keep the tied table at higher precision than the body")
    args = p.parse_args()

    # Ch. 14.7's save_checkpoint stores {"model", "config", "step", ...} with
    # `config` already asdict()-ed, so weights_only=True is enough to load it.
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = StackConfig(**ck["config"])
    model = Stack100M(cfg)                     # the constructor re-establishes the tie
    model.load_state_dict(ck["model"])
    model.eval()

    quantize_stacklm(model, bits=args.bits, group_size=args.group_size,
                     embedding_bits=args.embedding_bits)
    prefix = f"{args.out_dir}/stack100m_int{args.bits}"
    export_quantized(model, prefix, args.bits, args.group_size,
                     asdict(cfg), embedding_bits=args.embedding_bits)
    print(f"wrote {prefix}.safetensors  ({state_dict_bytes(model):.1f} MB)")
    print(f"wrote {prefix}.json")


if __name__ == "__main__":
    main()
```

!!! example "Worked example: the memory budget, computed exactly"

    Stack-100M has 101,353,728 parameters, of which **101,318,656 live in `nn.Linear` weights** (30 blocks × 2,818,048 body params + the 16,777,216-entry tied embedding) and 35,072 in RMSNorm scales that stay fp32. Here is what each format costs, computed rather than guessed:

    | Format | Weight storage | Scale/zero-point overhead | Total | vs. fp32 |
    |---|---|---|---|---|
    | fp32 | 101.32M × 4 B = 405.3 MB | — | **≈405 MB** | 1.0× |
    | bf16 | 101.32M × 2 B = 202.6 MB | — | **≈203 MB** | 2.0× |
    | int8 (row-wise) | 101.32M × 1 B = 101.3 MB | 171,008 scales × 4 B ≈ 0.68 MB | **≈102 MB** | 4.0× |
    | int4 (group=64) | 101.32M × 0.5 B = 50.66 MB | 1,583,104 groups × 8 B ≈ 12.66 MB | **≈63.3 MB** | 6.4× |
    | int4 body + int8 embed | 42.27 + 16.78 = 59.05 MB | 10.57 + 0.13 ≈ 10.70 MB | **≈69.8 MB** | 5.8× |

    Reproduce the two counts yourself. **int8 row scales**: one per output row, so $30 \times (512 + 128 + 128 + 512)$ attention rows $+\ 30 \times (1408 + 1408 + 512)$ MLP rows $+\ 32768$ head rows $= 38{,}400 + 99{,}840 + 32{,}768 = 171{,}008$. **int4 groups**: $101{,}318{,}656 \div 64 = 1{,}583{,}104$ exactly, each storing one fp32 scale **and** one fp32 zero-point (8 B) — $\approx 12.66$ MB, a full **20% of the int4 total**. That overhead is why quoting "int4 = 8× smaller" is a lie you should never repeat; the honest number is 6.4×.

    Production formats attack precisely that 20%. Storing the scale at fp16 and the zero-point at int8 costs ≈5 B/group ≈ 7.9 MB, dropping int4 to ≈58.6 MB (6.9×). GGUF's `Q4_K` goes further: it packs 256 weights into a *super-block* of eight 32-weight sub-blocks, storing 6-bit sub-scales plus a single fp16 super-scale — roughly 4.5 bits/weight all in. The headline: **the whole model, quantized, is smaller than a folder of phone photos**, and sits comfortably in RAM alongside a browser and an IDE.

{{fig:quant-memory-ladder-and-packing}}

## 8. Measuring the Damage: The Quantization Sweep

Everything in §6–§7 is a *claim* about quality: "int8 is free, int4 costs something on the hard tail." A textbook that asserts this and moves on has taught you to trust a heuristic. Wire the two halves of the chapter together instead — the eval battery is already a function, the quantizer is already a function, so composing them is fifteen lines:

```python
"""stacklm/eval/sweep.py — run the full battery across quantization settings."""
import copy

from stacklm.serve.quantize import quantize_stacklm, state_dict_bytes


def evaluate_quantization_sweep(fp32_model, tok, val_shard_dir, probes, *,
                                configs=((None, None, None), (8, None, None),
                                         (4, 128, None), (4, 64, None), (4, 64, 8)),
                                device="cpu", max_batches=20):
    """configs: (bits, group_size, embedding_bits). `probes` bundles the
    Section 4 fixtures:
       {"arith": [(problems, label), ...], "mc": [...], "qa": [...],
        "retriever": ..., "env": ..., "agent_tasks": [...], "pad_id": int}
    Returns one row per config, with deltas against the fp32 baseline."""
    rows = []
    for bits, gs, eb in configs:
        # deepcopy preserves ALIASING, so the tied lm_head/tok_emb stays tied in
        # the copy — which is exactly what `quantize_stacklm`'s data_ptr check
        # relies on. Rebuilding via Stack100M(cfg)+load_state_dict also re-ties,
        # because the constructor performs the tie; a naive tensor-by-tensor
        # "clone" helper would NOT.
        m = copy.deepcopy(fp32_model)
        if bits is not None:
            quantize_stacklm(m, bits=bits, group_size=gs or 64, embedding_bits=eb)
        m.to(device).eval()

        name = "fp32" if bits is None else f"int{bits}/g{gs or 64}" + (f"/e{eb}" if eb else "")
        ppl = compute_perplexity(m, val_shard_dir, probes["pad_id"],
                                 device=device, max_batches=max_batches)
        arith_in = eval_arithmetic(m, tok, generate_fn, probes["arith"][0][0])
        agent = eval_agent(m, tok, probes["env"], probes["agent_tasks"])
        rows.append({
            "config": name,
            "bytes_mb": state_dict_bytes(m),          # dedup'd: fp32 reads 405, not 472
            "ppl": ppl["perplexity"],
            "arith": arith_in["accuracy"],
            "parse": arith_in["parse_rate"],
            "mc": eval_mc_probe(m, tok, probes["mc"], device=device)["acc_norm"],
            "qa": eval_retrieval_qa(m, tok, generate_fn, probes["retriever"],
                                    probes["qa"])["exact_match"],
            "fmt": agent["tool_format_validity"],
            "agent_em": agent["exact_match"],
        })
        del m

    base = rows[0]
    for r in rows:
        r["d_ppl"] = r["ppl"] - base["ppl"]
        for k in ("arith", "mc", "qa", "fmt", "agent_em"):
            r["d_" + k] = r[k] - base[k]
    return rows


def print_sweep(rows) -> None:
    hdr = (f"{'config':<15}{'MB':>7}{'PPL':>8}{'dPPL':>8}{'arith':>8}"
           f"{'MC':>7}{'QA':>7}{'fmt':>7}{'agent':>7}")
    print(hdr, "\n", "-" * len(hdr), sep="")
    for r in rows:
        print(f"{r['config']:<15}{r['bytes_mb']:>7.1f}{r['ppl']:>8.2f}{r['d_ppl']:>+8.2f}"
              f"{r['arith']:>8.1%}{r['mc']:>7.1%}{r['qa']:>7.1%}"
              f"{r['fmt']:>7.1%}{r['agent_em']:>7.1%}")
```

Fill in this table from your own run — we will not invent the numbers for you:

| Config | Size (MB) | PPL | ΔPPL | Arithmetic | MC (acc_norm) | Retr-QA EM | Agent fmt | Agent EM |
|---|---|---|---|---|---|---|---|---|
| fp32 | ≈405 | *baseline* | 0 | | | | | |
| int8 / row | ≈102 | | | | | | | |
| int4 / g=128 | ≈57 | | | | | | | |
| int4 / g=64 | ≈63 | | | | | | | |
| int4 / g=64 + int8 embed | ≈70 | | | | | | | |

What to look for, and what each pattern means:

- **int8 deltas indistinguishable from noise on every probe.** The expected result, and why int8 weight-only PTQ is treated as a commodity: 8 bits with a per-row scale reconstructs a well-trained weight matrix to well within the model's own run-to-run variance. If int8 *does* move your numbers, suspect a bug in the packing, not a discovery about quantization.
- **int4 ΔPPL small (a few hundredths of a nat) while a capability probe drops several points.** The central phenomenon of this chapter, and the thing perplexity is structurally incapable of catching. Watch *arithmetic* and *agent EM* hardest: both need a long chain of exactly-right tokens, so they compound small per-token degradations that perplexity averages away.
- **g=128 worse than g=64 despite being 6 MB smaller.** The empirical version of Exercise 2's arithmetic: wider groups amortize the scale overhead but coarsen the grid.
- **Agent EM falling faster than everything else.** A 6-turn trajectory needs ~6 consecutive well-formed emissions, so a per-emission degradation $\epsilon$ compounds roughly as $(1-\epsilon)^6$. Multi-step evaluation is a *magnifier* of quantization damage. The `fmt` column tells you which fix to reach for: if format validity is degrading, spend bits on the embedding (`embedding_bits=8`, the last row) or add constrained decoding; if it is flat, the damage is in the policy and you want GPTQ/AWQ rather than a decoder change.

Small-$n$ discipline still applies (§5): at $n = 200$ arithmetic items, a 3-point difference is roughly one standard error and means nothing on its own.

## 9. Running Stack-100M on a Laptop CPU

The final step: load the quantized checkpoint and generate text with no GPU involved.

```python
"""
stacklm/serve/cli.py — CPU text generation from a quantized checkpoint, with
measured latency and memory.
Run (after `python -m stacklm.serve.export --bits 4`):
    python -m stacklm.serve.cli --bits 4
"""
import argparse
import os
import resource
import time

import torch

from stacklm.tokenizer import StackTokenizer                     # Ch. 14.3
from stacklm.serve.quantize import load_quantized                # Section 7.4
from stacklm.infer import generate_text                          # Section 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/stack100m")
    p.add_argument("--prompt", type=str, default="The mitochondria is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--threads", type=int, default=0, help="0 = all cores")
    args = p.parse_args()

    # CPU decode at batch 1 is memory-bandwidth bound, so more threads help only
    # until they saturate DRAM. Sweep this; on many laptops the optimum is the
    # PHYSICAL core count, not the SMT thread count.
    torch.set_num_threads(args.threads or os.cpu_count())

    t0 = time.perf_counter()
    model, meta = load_quantized(f"{args.checkpoint_dir}/stack100m_int{args.bits}")
    tok = StackTokenizer.load(f"{args.checkpoint_dir}/tokenizer.json")
    load_time = time.perf_counter() - t0

    # Warm up once: the first forward pays lazy-init and page-fault costs that
    # would otherwise be charged to your tok/s number.
    generate_text(model, tok, args.prompt, max_new_tokens=4, temperature=0.0)

    t1 = time.perf_counter()
    text, n_new = generate_text(model, tok, args.prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=0.7, top_p=0.95, return_n_tokens=True)
    gen_time = time.perf_counter() - t1

    # Peak RSS counts the mmap'd weights and the KV cache, which a Python-level
    # tracer like tracemalloc cannot see at all. (POSIX only; `ru_maxrss` is KiB
    # on Linux and BYTES on macOS — check yours before reporting a number.)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"--- Stack-100M (int{args.bits}, group={meta['group_size']}) on CPU ---")
    print(f"Threads:          {torch.get_num_threads()}")
    print(f"Load time:        {load_time:.2f} s")
    print(f"Generation:       {gen_time:.2f} s  ({n_new / gen_time:.1f} tok/s)")
    print(f"Peak RSS:         {peak_rss_mb:.0f} MB")
    print(f"Output: {args.prompt}{text}")


if __name__ == "__main__":
    main()
```

!!! example "Worked example: the full laptop memory budget"

    Every piece measured or computed above, for a generation call with a 2048-token context at int4:

    - **Quantized weights**: ≈63.3 MB (§7 table — 50.66 MB packed int4 + 12.66 MB fp32 scales/zero-points), plus 0.14 MB of fp32 RMSNorm scales.
    - **KV cache** (Ch. 14.4's `KVCache`), GQA with `n_kv_heads=2`, `head_dim=64`, 30 layers, bf16: per token per layer, K and V together are $2 \times 2 \times 64 = 256$ elements × 2 B = **512 B**; across 30 layers, **15,360 B ≈ 15 KB/token**; at the full 2048-token context, ≈**31.5 MB**. (Leave the cache in fp32 and it is 63 MB — as big as the weights.)
    - **Transient dequantization buffer**: `QuantizedLinear.forward` materializes one fp32 weight at a time. The largest is the tied head, $32768 \times 512 \times 4\ \text{B} = 67$ MB — *bigger than the entire quantized model*. This is the reference path's real cost, and the reason `dequantize_rows` exists for the embedding direction.
    - **Activations** (single-token decode): a few MB at most.

    Total steady-state: **on the order of 100 MB** of resident memory to hold the entire model plus a full 2048-token conversation, with a transient peak set by the largest dequantized layer. That is the concrete payoff of stacking GQA (4× smaller KV cache than plain MHA), a right-sized 32768-token vocabulary, and int4 weight quantization — each individually a modest win, compounding into "runs anywhere." State the context length whenever you quote it: at the 8192-token context mid-training unlocks, the cache alone is 126 MB (Exercise 7).

### 9.1 int4 weights are not int4 speed — and our reference path is *slower*

`QuantizedLinear` calls `self.dequantize()` on every forward. It buys **memory** (RAM and disk), not compute. Be blunt about the consequence, because the chapter's own thesis is "measure, don't assume": on a dequantize-then-matmul path, int4 is **slower than fp32 eager**, and a reader who runs the CLI and sees a small tok/s number deserves to know that is *correct behavior*, not a broken setup.

The arithmetic is simple. Single-stream decode is memory-bandwidth bound: a matrix–vector product does ~1 MAC per weight loaded, so time is dominated by moving weights, and per decoded token each path moves:

| Path | Weight bytes touched per decoded token | Time at ~20 GB/s | Ceiling |
|---|---|---|---|
| fp32 eager `nn.Linear` | read 405 MB | ≈20 ms | ≈50 tok/s |
| **our int4 reference** | read 63 MB packed **+ write 405 MB + read 405 MB** dequantized | ≈44 ms | ≈23 tok/s |
| GGUF `Q4_K_M` in llama.cpp (fused unpack-in-matmul) | read ≈57 MB | ≈3 ms | ≈350 tok/s |

Substitute your own machine's achievable bandwidth (a modern laptop lands roughly in the 10–50 GB/s range for a single process; measure it, do not trust a spec sheet) — the *ratios* are the claim, not the absolute numbers, and every row is an upper bound that real ALU work, threading, and cache behavior will pull down. The ordering, though, is robust: our int4 path pays a full fp32 round-trip that fp32 eager does not, including a 67 MB dequantize of the tied head **at every single decode step**. Expect llama.cpp's Q4_K_M to be several times faster than either PyTorch path.

The bandwidth argument in favor of low-bit weights is real — it just requires a kernel that unpacks *inside* the matmul loop instead of materializing a dequantized weight first (and note that even then the honest streaming speedup is ≈6.4×, not 8×, once the fp32 scales and zero-points are counted). See [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) for why this is a general property of decode. Ours does not fuse. These do:

**torchao** — PyTorch's supported quantization library for LLMs, and the correct answer to "what does PyTorch actually use in 2026." The legacy eager-mode `torch.ao.quantization.quantize_dynamic` (FBGEMM/QNNPACK) still exists and still works for int8 on CPU, but it is not what anyone quantizes an LLM with today.

```python
# pip install torchao
import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig

model = load_fp32_stack100m().eval()

quantize_(model, Int8WeightOnlyConfig())                    # W8A16, works broadly
# quantize_(model, Int4WeightOnlyConfig(group_size=64))     # W4A16, grouped like ours

# torch.compile is not optional here: torchao's quantized tensor subclasses are
# designed to be traced and fused, and the eager-mode numbers are misleading.
model = torch.compile(model, mode="max-autotune")
```

`quantize_` mutates the model in place, swapping each `nn.Linear` weight for a quantized *tensor subclass* that intercepts `F.linear` and dispatches to a packed kernel — conceptually the same swap as `quantize_stacklm`, done at the tensor level rather than the module level, which is why it composes with `torch.compile`, FSDP, and the rest of the stack. Two caveats to check against your installed version rather than assume: older torchao releases spell these configs as the factory functions `int8_weight_only()` / `int4_weight_only(group_size=64)`, and int4 kernel coverage differs by backend and layout (the int4 path was CUDA-first; CPU support depends on the layout your build ships). Microbenchmark before you believe a speedup.

**llm-compressor + compressed-tensors** — the vLLM project's one-shot quantization pipeline and the on-disk format vLLM reads natively. This is the path when you want GPTQ or AWQ rather than RTN, applied by a maintained implementation:

```python
# pip install llmcompressor
from llmcompressor import oneshot                          # entrypoint moved across
from llmcompressor.modifiers.quantization import GPTQModifier  # releases; check yours

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model, dataset=calibration_dataset, recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512)
model.save_pretrained("Stack-100M-W4A16", save_compressed=True)  # compressed-tensors
```

Note `ignore=["lm_head"]` — the default recipes leave the output head at higher precision, the same instinct as our `embedding_bits`. This pipeline expects a Hugging Face `PreTrainedModel`, so using it on Stack-100M means first porting `StackConfig` to a `PretrainedConfig` + `PreTrainedModel` wrapper (Ch. 14.4 sketches the state-dict remapping). That port is worth doing once: it is also what unlocks `lighteval`, `vllm serve`, and the Hub.

**GGUF + llama.cpp** — the fastest production path for int4 on CPU and the format that dominates edge deployment. llama.cpp fuses unpack-and-matmul into a single hand-written kernel per ISA (AVX2 / AVX-512 / NEON), which is precisely the thing our reference path does not do:

```bash
# In a llama.cpp checkout, after exporting Stack-100M in HF layout:
python convert_hf_to_gguf.py ./stack100m-hf --outfile stack100m-f16.gguf --outtype f16
./llama-quantize stack100m-f16.gguf stack100m-q4_k_m.gguf Q4_K_M
./llama-cli -m stack100m-q4_k_m.gguf -p "The mitochondria is" -n 64
./llama-server -m stack100m-q4_k_m.gguf --port 8080   # OpenAI-compatible endpoint
```

!!! warning "Common pitfall: converting a not-quite-Llama architecture to GGUF"

    Stack-100M is Llama-shaped *except* for two things: QK-norm, and NoPE on every 4th layer. `convert_hf_to_gguf.py` maps a declared architecture to a fixed graph, so converting it *as* Llama would silently apply RoPE to the NoPE layers and drop the QK-norm weights. Nothing errors; the model just generates degraded text, and you will spend a day blaming the quantizer. The honest routes are (a) add a small architecture entry describing the per-layer RoPE mask and the QK-norm tensors, or (b) accept the mismatch only after verifying token-level logit agreement against the PyTorch reference on a fixed prompt. **Always diff logits against the reference implementation after any format conversion** — the same discipline as Ch. 14.4's cache-equivalence assertion, applied across runtimes.

**Where the GPU stacks fit.** Nothing about `vLLM`, `SGLang`, or `TensorRT-LLM` is wrong for this model — they solve a problem Stack-100M does not have at batch 1 on a laptop. Their wins are throughput wins: continuous batching, PagedAttention, prefix-cache reuse across requests ([vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html), [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html)). The moment you serve the §4.4 agent to more than one concurrent user — where every request shares the same long system prompt and tool schemas — prefix caching alone is worth more than everything in this section. vLLM loads GPTQ, AWQ, and compressed-tensors checkpoints directly, which is why the llm-compressor route composes with production serving. And `bitsandbytes` is a different problem again: its NF4 path is the *training*-time 4-bit quantization behind QLoRA, not a serving format.

Measure your own tok/s with the CLI before and after each of these. The gap is the lesson.

!!! interview "Interview Corner"

    **Q1:** You quantize a model to int4 with round-to-nearest and perplexity barely moves, but a downstream multi-step reasoning eval drops sharply. What's going on, and how would GPTQ or AWQ help?

    **A:** Perplexity averages error across the entire vocabulary and every position, so it is dominated by the easy majority of predictions and can mask damage to a small number of high-leverage weights — the ones a specific reasoning chain depends on at a specific step. Two effects compound. First, RTN quantizes every weight independently with no notion of which weights the model's *output* is sensitive to, so it can wreck a handful of outlier channels that barely move the average loss but matter enormously for a token the model must get exactly right mid-chain. Second, a multi-step eval is a product of per-step success probabilities: a per-token degradation of 1% is invisible in a mean but costs roughly $1-(0.99)^{k}$ on a $k$-step trajectory. GPTQ addresses the first directly: after quantizing each column it uses the layer's Hessian to redistribute the resulting error onto not-yet-quantized columns, explicitly minimizing the *layer's output reconstruction error* rather than treating weights as independent. AWQ takes a cheaper, complementary route: identify the small set of activation channels with systematically large magnitude and rescale so those channels see less quantization error in the first place, folding the inverse scale into the preceding norm. Both beat RTN exactly where RTN is weakest — which is why production int4 deployments use GPTQ/AWQ, and why this chapter treats RTN as a pedagogical baseline. The process lesson matters more than either method: never ship a quantized model on a perplexity delta alone; run the capability battery.

    **Q2:** Your team ships a small tool-using agent. It scores 40% end-to-end on your task set. What do you measure next, and what would each result tell you to change?

    **A:** 40% end-to-end is not actionable — decompose it. (1) **Tool-call format validity**: what fraction of emissions parse as a well-formed call? If that is low, the model never learned the grammar, and the cheapest fixes are constrained/grammar-guided decoding at sampling time and more distillation traces — not a bigger model. (2) **Tool-choice accuracy** against a gold first tool: high format validity with low choice accuracy means the model learned the *syntax* and not the *policy*, which is a data-coverage problem. (3) **Turns and cap rate**: if it is burning the step cap, the harness guards matter more than the model — and make sure your cap detector actually fires, because a harness that forces a synthesis attempt on exhaustion emits something that *looks* like a normal final answer. (4) **A failure taxonomy over the traces**: if `retrieved_but_ignored` dominates, retrieval is fine and extraction is the bottleneck, so target SFT on grounded-answer formatting; if `wrong_tool` dominates, target the routing decision. (5) Finally, **confidence intervals**: at n=30, 40% has a 95% Wilson interval of roughly 25–58%, so before acting, check whether the differences you are chasing are larger than the noise. In production this decomposition is what `inspect_ai` gives you for free — per-step tool records, turn accounting, and per-metric scorers over a logged trace.

## 10. Closing the Loop

You now have every artifact the capstone promised: a trained, mid-trained, aligned, tool-using ~101M-parameter model; a real inference path with a KV cache that provably matches the training-time function; an honest, reproducible evaluation report — including the agent, not just the language model — with explicit disclosure of what it cannot do; a measured account of what quantization costs rather than an assumed one; and an export that runs on ordinary CPU hardware. [14.12 Retrospective & Scale-Up](12-retrospective-and-scaleup.html) closes the capstone with the full cost accounting and a concrete map of what changes if you push past 100M parameters.

Before that, two commands on your own laptop, on your own checkpoint:

```bash
python -m stacklm.serve.export --bits 4 --embedding_bits 8
python -m stacklm.serve.cli    --bits 4 --prompt "The mitochondria is"
```

Read what it wrote. That is the actual point of the entire capstone — not the perplexity number, not the tok/s number, but the fact that a model you trained end to end, on hardware you rented for less than a dinner out, just generated a sentence on the machine in front of you.

!!! key "Key Takeaways"

    - Evaluation and sampling need *opposite* settings of the same knob: `logits_to_keep=1` is a prefill optimization for generation, `logits_to_keep=0` is a hard requirement for perplexity and cloze scoring. Assert the shape. And pass `position_ids`/`seq_ids` through — evaluating without the document geometry the model trained in measures a different model.
    - **Evaluate the format you trained.** Import Ch. 14.9's `make_arithmetic_prompt` and `exact_match_reward` and Ch. 14.9's chat template rather than re-deriving a lookalike; a probe with the wrong surface form or too small a token budget scores a working model near zero. Report parse-failure rate separately from accuracy, and report an out-of-distribution row so "RLVR generalizes narrowly" is visible instead of averaged away.
    - A model is not "done" at a loss curve. It is done at a defensible held-out perplexity (provably excluded from every training stage), cheap probes (arithmetic EM + parse rate, cloze MC reporting both `acc` and `acc_norm`, retrieval-QA EM *with retriever recall separately*), an **agent-loop evaluation** decomposed into format validity / tool choice / turns / cap rate / EM / failure taxonomy, and an honest statement of what the model cannot do. Contamination is real even for hand-built probes, and the nastiest leak is self-inflicted: hold agent eval tasks out of the Ch. 14.10 distillation rollout by hash.
    - Perplexity measures predictive fit to held-out text and nothing else; it is not comparable across tokenizers (use bits-per-byte), and it can stay flat while a capability collapses — multi-step evals *magnify* damage that averaging hides, roughly as $(1-\epsilon)^k$ over $k$ steps.
    - Derive metrics from signals the harness actually emits. Ch. 14.10's loop returns as soon as an emission parses as `final` *and* force-synthesizes a final-looking answer when the step budget runs out, so "did it cap?" is `len(assistant) > max_steps`, not "was the last action non-final" — the latter is dead code and a structurally-zero `cap_rate`.
    - RTN is a fast, calibration-free baseline; GPTQ (Frantar et al., 2022) redistributes each column's error onto not-yet-quantized columns via the layer's Hessian; AWQ (Lin et al., 2023) protects high-activation channels by rescaling and folds $1/s$ into the preceding norm. AWQ's search must optimize *output* error after de-scaling — a weight-space proxy on the scaled matrix is monotone in the search variable and degenerates into a no-op.
    - **Tied embeddings break naively.** Replacing `lm_head` with a quantized module severs the alias to `tok_emb`, leaving a 67 MB fp32 table in the export. Detect the tie by `data_ptr()`, replace the embedding with a row-gathering view over the quantized head's buffers, give the quantized modules a `.weight` property so Ch. 14.4's `fused_ce_z_loss`/`estimate_params` call sites keep working, and assert the *deduplicated* byte count of the actual `state_dict`.
    - Exact accounting for Stack-100M's 101.32M quantizable parameters: fp32 ≈405MB → bf16 ≈203MB → int8 ≈102MB (4.0×) → int4/g64 ≈63MB (6.4×, counting the 12.66MB of fp32 scales/zero-points honestly — 20% of the total, which is why "int4 = 8×" is a lie). Add the KV cache before quoting a memory number: 15 KB/token, 31.5 MB at 2048 in bf16, 126 MB at 8192.
    - int4 weights shrink memory, not FLOPs. A dequantize-then-matmul path is *slower than fp32 eager* — it pays a full fp32 round-trip per token, including a 67 MB dequantize of the tied head at every decode step. Real throughput needs fused integer kernels: **torchao** (`quantize_(model, Int4WeightOnlyConfig(...))` + `torch.compile`), **llm-compressor**/**compressed-tensors** for GPTQ/AWQ into vLLM, and **GGUF + llama.cpp** for CPU/edge — with the caveat that converting a not-quite-Llama architecture silently corrupts NoPE and QK-norm unless you diff logits against the reference.
    - The payoff of the entire capstone is two commands: `python -m stacklm.serve.export --bits 4` then `python -m stacklm.serve.cli --bits 4`, and text your own model produced on your own machine.

!!! sota "State of the Art & Resources (2026)"
    Weight-only PTQ has matured into a commodity operation — GPTQ and AWQ are standard export paths in every major serving stack, `compressed-tensors` is the format the vLLM ecosystem converges on, GGUF's block-quantized K-quants dominate CPU/edge deployment, and the frontier has pushed toward sub-4-bit and ternary weights. Evaluation has consolidated around a small number of reproducible open harnesses, with agentic evaluation the fastest-moving frontier.

    **Foundational work**

    - [Frantar, Ashkboos, Hoefler & Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022)](https://arxiv.org/abs/2210.17323) — the Hessian-based, column-by-column reconstruction method §6.2 implements a pedagogical version of.
    - [Dettmers, Lewis, Belkada & Zettlemoyer, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022)](https://arxiv.org/abs/2208.07339) — the mixed-precision decomposition that made int8 inference practical for transformers with outlier activation features.

    **Recent advances (2023–2026)**

    - [Lin, Tang, Tang, Yang, Dang & Han, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023)](https://arxiv.org/abs/2306.00978) — the salient-channel-protection method §6.3 implements; MLSys 2024 Best Paper.
    - [Xiao, Lin, Seznec, Wu, Demouth & Han, *SmoothQuant* (2023)](https://arxiv.org/abs/2211.10438) — migrates quantization difficulty from activations to weights via a per-channel smoothing factor, enabling accurate W8A8 (not just weight-only) quantization.
    - [Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (2024)](https://arxiv.org/abs/2402.17764) — ternary {-1, 0, 1} weights, the current frontier past int4.
    - [Dettmers, Pagnoni, Holtzman & Zettlemoyer, *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)](https://arxiv.org/abs/2305.14314) — NF4 and double quantization; the *training*-side 4-bit story, contrasted with this chapter's serving-side one.

    **Open-source & tools**

    - [pytorch/ao (torchao)](https://github.com/pytorch/ao) — PyTorch's quantization/sparsity library for LLMs; `quantize_(model, Int8WeightOnlyConfig())`, tensor-subclass based, designed to compose with `torch.compile`. The modern replacement for eager-mode `torch.ao.quantization`.
    - [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) — one-shot GPTQ/AWQ/SmoothQuant/FP8 pipelines producing [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) checkpoints that vLLM loads natively.
    - [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — the production reference for fused, quantized CPU/edge inference; `convert_hf_to_gguf.py`, `llama-quantize`, `llama-server`.
    - [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) — actively maintained GPTQ/AWQ/GGUF toolkit with Transformers, vLLM, and SGLang integration.
    - [huggingface/safetensors](https://github.com/huggingface/safetensors) — the pickle-free, mmap-able tensor format §7.4 exports to; its refusal to store aliased tensors is a free correctness check on the embedding tie.
    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the cloze-scoring framework §4.2 borrows from; plug in a custom model by implementing `loglikelihood` and `generate_until`.
    - [huggingface/lighteval](https://github.com/huggingface/lighteval) — lighter, hackable harness with first-class custom-task/custom-model entry points.
    - [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) — the UK AI Safety Institute's framework for agentic evaluation: solvers, tool sandboxes, scorers, and a trace viewer; the right home for §4.4 once it outgrows one file.

    **Go deeper**

    - [GGUF (Hugging Face Hub docs)](https://huggingface.co/docs/hub/en/gguf) — the on-disk block-quantization format (Q4_K, Q6_K, IQ-series) referenced throughout §6–§9, including the super-block layout that beats our flat per-group scales.

## Further reading

- Frantar, Ashkboos, Hoefler & Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022.
- Lin, Tang, Tang, Yang, Dang & Han, *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, 2023.
- Dettmers, Lewis, Belkada & Zettlemoyer, *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, 2022.
- Jacob, Kligys, Chen, Zhu, Tang, Howard, Adam & Kalenichenko, *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, 2018.
- Gao et al. (EleutherAI), *A Framework for Few-Shot Language Model Evaluation* (`lm-evaluation-harness`).
- Hendrycks, Burns, Basart, Zou, Mazeika, Song & Steinhardt, *Measuring Massive Multitask Language Understanding* (MMLU), 2021.
- Kwiatkowski et al., *Natural Questions: A Benchmark for Question Answering Research*, 2019.
- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K), 2021 — source of the `####` answer convention §4.1 grades against.
- Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao, *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022 — the loop §4.4 evaluates.
- Gerganov et al., `llama.cpp` and the GGUF format — the production reference for fused, quantized CPU/edge inference.
- Cross-reference: [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html), [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html), [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html), [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html), [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html), [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html), [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html), [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html).

## Exercises

**1.** Your `compute_perplexity` run on the held-out shard reports a mean loss of **3.1 nats/token**. (a) Convert this to perplexity. (b) The function returns *both* `loss_nats_per_token` and `perplexity`. Give the specific reason §3 offers for reporting the nats/token number rather than perplexity alone. (c) A random-guessing model over Stack-100M's vocabulary would sit at what perplexity, and what does the gap between that and your number tell you?

??? note "Solution"
    (a) $\text{PPL} = \exp(3.1)$. Since $e^{3} \approx 20.09$ and $e^{0.1} \approx 1.105$, $\text{PPL} \approx 20.09 \times 1.105 \approx 22.2$. The model's predictive distribution has an effective branching factor of about 22 tokens per position on this held-out text.

    (b) §3 gives two reasons: loss in nats/token **composes linearly** (perplexity does not — it is an exponential), and it is **exactly the quantity the training curve in [14.7] already plots**. So you can drop an "eval loss" point directly onto the same chart as the training loss and read off the generalization gap by eye. Perplexity is the human-facing summary; nats/token is the composable, chart-compatible one.

    (c) A model that assigns uniform probability $1/V$ to every token over the $V = 32768$ vocabulary has $\text{PPL} = V = 32768$ (loss $\ln 32768 \approx 10.4$ nats — the value the training curve starts at). Your $\approx 22.2$ is roughly three orders of magnitude below that, which is the concrete evidence that the model learned real predictive structure in the training-mix text (and, per §3, *nothing more than that*). Note also that this number is only comparable to another model trained with the **same tokenizer** — across tokenizers you must use bits-per-byte.

**2.** Using the exact memory accounting in §7 (101,318,656 quantizable parameters), compute the int4 on-disk size and compression-vs-fp32 ratio for **`group_size = 128`** instead of the default 64. The reference int4 layout stores one fp32 scale plus one fp32 zero-point per group. Compare against the default (group=64) numbers and state the tradeoff you are making by widening the group — then say which line of §8's sweep table settles the question empirically.

??? note "Solution"
    Packed 4-bit weights are independent of group size: $101{,}318{,}656 \times 0.5\ \text{B} = 50{,}659{,}328\ \text{B} \approx 50.66\ \text{MB}$.

    Group count at `group_size = 128`: $101{,}318{,}656 \div 128 = 791{,}552$ groups (half the $1{,}583{,}104$ at group=64). Each group stores one fp32 scale + one fp32 zero-point $= 8\ \text{B}$:

    $$791{,}552 \times 8\ \text{B} = 6{,}332{,}416\ \text{B} \approx 6.33\ \text{MB}.$$

    Total $\approx 50.66 + 6.33 \approx 57.0\ \text{MB}$. Compression vs. fp32 ($\approx 405.3$ MB): $405.3 \div 57.0 \approx 7.1\times$ (versus $\approx 63.3\ \text{MB}$ / $6.4\times$ at group=64).

    Tradeoff: doubling the group size **halves the scale/zero-point overhead** ($\approx 12.66 \to \approx 6.33$ MB), but each scale/zero-point must now cover 128 weights with a single min/max instead of 64, so the quantization grid is **coarser** and reconstruction error grows — exactly the "not enough precision at 4 bits" concern §7 raises. Empirically, the `int4/g128` row of §8's sweep versus the `int4/g64` row settles it: if `d_arith` and `d_agent_em` are materially worse at g=128, the 6 MB is not worth it. Measure, don't assume.

**3.** Work RTN by hand. Take a single group of four weights $w = [\,0.90,\ -0.40,\ 0.10,\ -0.85\,]$ and apply the **$b=4$ symmetric** scheme from §6.1 (so $2^{b-1}-1 = 7$). (a) Compute the scale $s$. (b) Compute the integer codes $q$ and the dequantized weights $\hat w$. (c) Report the maximum absolute reconstruction error, and say which weight carries it — and why that is the point.

??? note "Solution"
    (a) Symmetric scale: $s = \dfrac{\max(|w|)}{2^{b-1}-1} = \dfrac{0.90}{7} \approx 0.12857$.

    (b) $q_i = \operatorname{clip}(\operatorname{round}(w_i/s),\,-7,\,7)$:

    - $0.90/0.12857 = 7.00 \to 7$
    - $-0.40/0.12857 = -3.11 \to -3$
    - $0.10/0.12857 = 0.78 \to 1$
    - $-0.85/0.12857 = -6.61 \to -7$

    So $q = [\,7,\ -3,\ 1,\ -7\,]$. Dequantize $\hat w_i = s\,q_i$:

    - $7 \times 0.12857 = 0.9000$
    - $-3 \times 0.12857 = -0.3857$
    - $1 \times 0.12857 = 0.1286$
    - $-7 \times 0.12857 = -0.9000$

    So $\hat w = [\,0.900,\ -0.386,\ 0.129,\ -0.900\,]$.

    (c) Per-element errors $w_i - \hat w_i$: $0.000,\ -0.0143,\ -0.0286,\ +0.0500$. The **maximum absolute error is $0.050$, on the $-0.85$ weight** — whose scaled value $-6.61$ lands near a rounding boundary — *not* on the largest-magnitude weight $0.90$, which sits exactly on the grid endpoint with zero error. That is RTN's blind spot in miniature (§6.1): it rounds every weight independently by grid proximity alone, with no notion of which one matters more to the layer's output. GPTQ and AWQ are the two ways of injecting that missing notion.

**4.** `sequence_logprob` in §4.2 scores a candidate answer by the **sum** of its per-token log-probabilities. (a) Explain the systematic bias this introduces when options tokenize to different lengths. (b) Why is it not a problem for the `TINY_MC_SET` items actually shown? (c) `eval_mc_probe` reports both `acc` and `acc_norm` — describe exactly what `acc_norm` normalizes by and why *byte* length rather than *token* length is the more defensible denominator.

??? note "Solution"
    (a) Every token contributes a $\log p \le 0$ term, so each additional continuation token can only push the summed score **downward**. Longer candidates accumulate more negative mass purely by being longer, independent of correctness, so `max(scores)` is biased toward the **shorter** option — a 1-token choice can beat a correct-but-longer 5-token choice just by stopping sooner.

    (b) The shown choices are near-uniform in length (`" H2O"`, `" CO2"`, `" NaCl"`, `" O2"`; `" Paris"`, `" Berlin"`, `" Madrid"`, `" Rome"`), so the length bias is roughly constant across options and cancels out of the arg-max. The construction ("small enough to eyeball every item") maintains this invariant by hand.

    (c) `acc_norm` divides the summed log-probability by `len(choice.encode("utf-8"))` — the continuation's **byte** length — matching `lm-evaluation-harness`'s `acc_norm`. Bytes are the more defensible denominator because token counts are a property of *your tokenizer*, not of the answer: the same string can be 2 tokens under one BPE and 5 under another, so token-normalized scores are not comparable across models, while byte-normalized ones are. (Bytes are also what bits-per-byte uses, for the same reason.) Reporting both is the discipline: if `acc` and `acc_norm` disagree, your option set is length-imbalanced and the raw number is partly measuring string length.

**5.** Compute the KV-cache budget the way §9's worked example does, using the same GQA config (`n_kv_heads = 2`, `head_dim = 64`, 30 layers, bf16 = 2 bytes/element). (a) Bytes per token per layer, and per token across all layers. (b) Total cache for a **512-token** context. (c) The chapter says GQA gives a "4× smaller KV cache than plain multi-head attention." What would the same 512-token cache cost under plain MHA, and what does that imply about the number of query heads? (d) `KVCache` preallocates the full `max_seq_len` buffer at construction. What does that cost at 512 tokens of *actual* use with `max_seq_len = 2048`, and why is it still the right design?

??? note "Solution"
    (a) Per token per layer we store both K and V, each of shape `n_kv_heads x head_dim`:

    $$2\ (\text{K and V}) \times 2\ (\text{kv heads}) \times 64\ (\text{head dim}) = 256\ \text{elements} \times 2\ \text{B} = 512\ \text{B}.$$

    Across 30 layers: $512 \times 30 = 15{,}360\ \text{B} \approx 15\ \text{KB per token}$.

    (b) At 512 tokens: $15{,}360 \times 512 = 7{,}864{,}320\ \text{B} \approx 7.86\ \text{MB}$.

    (c) "4× smaller" means plain MHA would cost $4 \times 7.86 \approx 31.5\ \text{MB}$ for the same context, because MHA stores K and V for every query head rather than sharing 2 KV heads. A 4×-larger cache at fixed `head_dim` means MHA uses $4 \times 2 = 8$ KV heads — i.e. the model has **8 query heads**, and GQA collapses them onto 2 shared KV heads (a 4:1 grouping).

    (d) Preallocation reserves the full $15{,}360 \times 2048 \approx 31.5$ MB regardless, so at 512 tokens of use you are holding ~23.6 MB of zeros — a 4× over-allocation. It is still right for a single-stream laptop decoder because the alternative (growing the cache by `torch.cat` each step) reallocates and copies the entire cache every token, turning $O(1)$ decode into $O(T)$ memory traffic per step and thrashing the allocator. The production answer to the waste is not dynamic growth but **paging**: allocate fixed-size blocks on demand and keep a block table, which is exactly PagedAttention. Sizing the cache to `len(prompt) + max_new_tokens` (as `generate` does) already recovers most of the slack.

**6.** Implement a **symmetric** per-group int4 variant of `quantize_int4_grouped` / `dequantize_int4_grouped` (call them `..._symmetric`), following §6.1's symmetric scheme and reusing §7's two-nibbles-per-byte packing. Then answer: (a) how much on-disk memory does dropping the zero-point save for Stack-100M's 101,318,656 quantizable parameters at `group_size = 64`, and (b) why did §7 nonetheless choose *asymmetric* int4 as the reference path?

??? note "Solution"
    Symmetric int4 uses codes in $[-7, 7]$ with $s = \max(|w|)/7$ and no zero-point. The signed range spans 15 levels, which still fits a 4-bit nibble; we shift by $+7$ to $[0, 14]$ only for packing, and undo it on dequantize.

    ```python
    def quantize_int4_grouped_symmetric(weight: torch.Tensor, group_size: int = 64):
        """Symmetric per-group int4, packed two values per uint8 byte.
        Returns (packed uint8 (d_out, d_in//2), scales (d_out, d_in//group_size))."""
        d_out, d_in = weight.shape
        assert d_in % group_size == 0, f"d_in={d_in} not divisible by {group_size}"
        n_groups = d_in // group_size

        w = weight.float().view(d_out, n_groups, group_size)
        scale = (w.abs().amax(dim=2) / 7.0).clamp(min=1e-8)        # (d_out, n_groups)
        q = torch.round(w / scale.unsqueeze(2)).clamp(-7, 7)        # codes in [-7, 7]
        q = (q + 7).to(torch.uint8).view(d_out, d_in)              # shift to [0, 14] to pack

        q_even, q_odd = q[:, 0::2], q[:, 1::2]
        packed = (q_even | (q_odd << 4)).to(torch.uint8)           # (d_out, d_in // 2)
        return packed, scale


    def dequantize_int4_grouped_symmetric(packed: torch.Tensor, scale: torch.Tensor,
                                          d_in: int, group_size: int = 64) -> torch.Tensor:
        d_out = packed.shape[0]
        n_groups = d_in // group_size
        q_even = (packed & 0x0F).to(torch.float32)
        q_odd = ((packed >> 4) & 0x0F).to(torch.float32)
        q = torch.empty(d_out, d_in, device=packed.device)
        q[:, 0::2], q[:, 1::2] = q_even, q_odd
        q = (q - 7.0).view(d_out, n_groups, group_size)            # undo the +7 shift
        w = q * scale.unsqueeze(2)                                 # no zero-point term
        return w.view(d_out, d_in)
    ```

    (This drops into `QuantizedLinear.from_float`/`dequantize`: for `bits == 4` you register `q_weight` and `scale` only, with no `zero_point` buffer — and remember to update `QuantizedLinear.empty` to match, or `load_state_dict` will fail on a missing key. Note this is the same fake-quant grid §6.3's AWQ search optimizes against.)

    (a) The reference asymmetric layout stores an fp32 scale **and** an fp32 zero-point (8 B) per group; symmetric stores only the scale (4 B). At $1{,}583{,}104$ groups that saves $1{,}583{,}104 \times 4\ \text{B} \approx 6.33\ \text{MB}$ — int4 drops from $\approx 63.3$ MB to $\approx 57.0$ MB (about $7.1\times$ vs. fp32).

    (b) §7 chose asymmetric because at only 4 bits the extra precision matters more than 6 MB. A symmetric grid is forced to center at 0 and cover $[-\max|w|, +\max|w|]$; for a **skewed** group this wastes codes on a range the weights never occupy. Asymmetric fits the actual $[\min(w), \max(w)]$ with an integer zero-point, giving finer effective resolution exactly where 4-bit RTN is most fragile. The saving is real but small relative to that quality risk — and §8's sweep is how you would check whether the risk materializes on *your* checkpoint.

**7.** (Quantitative) Stack-100M's KV cache costs 15,360 B/token in bf16 (§2). (a) At what context length does the bf16 KV cache exceed the ≈63.3 MB int4 weight budget, and what is the answer if you leave the cache in fp32? (b) After mid-training extends the context to 8192 (Ch. 14.8), what does the bf16 cache cost, and what does that do to the "runs on a Raspberry Pi" claim? (c) Ch. 14.4's CI asserts that greedy generation is identical with and without the cache. Explain precisely why that assertion must be re-run against the *quantized* model rather than inherited from the fp32 one, and name the specific `QuantizedLinear` bug it would *fail* to catch.

??? note "Solution"
    (a) bf16: $63.3\times10^{6} \div 15{,}360 \approx 4{,}121$ tokens — beyond roughly a 4k context, the conversation costs more memory than the model. fp32 doubles the per-token cost to 30,720 B, halving the crossover to $\approx 2{,}060$ tokens, i.e. **just past the 2048 pretrain context**. That is the concrete argument for Ch. 14.4's bf16 cache default: at fp32 the cache overtakes the weights inside the model's own training context.

    (b) $15{,}360 \times 8192 = 125{,}829{,}120$ B $\approx 126$ MB — **twice** the int4 weights. The "≈100 MB total" figure in §9 is stated for a 2048-token context and does not survive a full 8192-token conversation; the honest number there is ≈190 MB (63 MB weights + 126 MB cache), which still fits a 512MB-class device but no longer leaves a comfortable margin. Report the context length alongside any memory claim, or the claim is not falsifiable.

    (c) The assertion is a property of *the modules the decode path executes*, not of the architecture. §7 replaces every `nn.Linear` with a `QuantizedLinear`, so the fp32 proof covers code that is no longer running; re-running it catches cache, mask, or `position_ids` regressions introduced by the module swap. What it would **not** catch is a **packing** bug: `dequantize_int4_grouped` reconstructs `q[:, 0::2], q[:, 1::2]` from the low/high nibbles, and any nibble-order or `view(d_out, n_groups, group_size)` mistake produces a *deterministic* wrong weight that is applied identically on both the cached and uncached paths, so the two agree perfectly while both being wrong. The test to run is therefore the pair: (i) cache-invariance on the quantized model, and (ii) a logit diff of the quantized model against `dequantize()`-then-`nn.Linear` on the same input. Neither alone is sufficient — the same "diff against a reference after every model edit" discipline §9.1 applies to GGUF conversion.

**8.** (Analysis) You run `eval_agent` on 30 tasks and get: `tool_format_validity = 0.97`, `tool_choice_acc = 0.83`, `turns_median = 3`, `cap_rate = 0.03`, `exact_match = 0.40`, and taxonomy `{ok: 12, retrieved_but_ignored: 11, wrong_tool: 4, looped: 1, non_termination: 1, other: 1}`. (a) Which single intervention has the highest expected value, and why? (b) A colleague proposes fixing this by scaling to 300M parameters. Give the argument against, using the taxonomy. (c) You then quantize to int4 and `exact_match` falls to 0.27 while held-out PPL rises only 0.03 nats. Which §4.4 sub-metric would you check first, and what would each outcome imply? (d) Suppose instead `cap_rate` had come back as exactly `0.00` on every configuration you ever tried. What should you suspect before believing it?

??? note "Solution"
    (a) `retrieved_but_ignored` at 11/30 is the largest failure bucket by a wide margin and is *not* a retrieval problem: by construction that bucket means the gold evidence string appeared in a tool observation and the model still answered wrong. The intervention with the highest expected value is therefore targeted SFT on **grounded extraction** — more traces of the form "observation contains the answer → emit exactly that span in the `Answer:` slot" — plus tightening the answer-format grammar. Format validity (0.97) and tool choice (0.83) are already high enough that improving them can move end-to-end EM by at most a few points; extraction can move it by up to ~37 points.

    (b) The taxonomy says the bottleneck is a *specific, learnable behavior* the distillation data under-covers, not a capacity ceiling. Scaling to 300M is roughly 3× the training and serving cost for an intervention that does not target the dominant failure mode; if the traces never show the model copying an answer span out of an observation, a bigger model will learn the same wrong habit more fluently. Fix the data and the answer grammar first, and use scale only after the taxonomy is dominated by `other` — i.e. after the failures stop being explainable.

    (c) Check **`tool_format_validity`** first, because a 13-point EM drop against a 0.03-nat PPL change is exactly the "perplexity is flat, capability collapsed" signature, and multi-step trajectories magnify per-token degradation as $(1-\epsilon)^k$. If format validity has fallen (say 0.97 → 0.80), quantization damaged the model's grip on the rare special-token grammar — fix it with precision where it matters (`embedding_bits=8` for the tied table, or GPTQ/AWQ instead of RTN) or with constrained decoding, which makes malformed output impossible by construction. If format validity is *unchanged*, the grammar survived and the damage is in the policy/extraction — expect `retrieved_but_ignored` and `wrong_tool` to have grown, which points at reconstruction-aware quantization rather than at the decoder.

    (d) A metric that is exactly zero across every configuration is far more likely to be **structurally unreachable** than genuinely never observed — suspect the detector before you congratulate the model. That is precisely the trap §4.4 flags: Ch. 14.10's `run_agent` force-synthesizes an `Answer:` line when the step budget runs out, so that emission parses as `kind == "final"` and a detector written as `acts[-1].kind != "final"` can never fire, pinning `cap_rate` at 0.0 and starving the `non_termination` bucket. The correct signal is the trace shape (`len(assistant) > max_steps`), and the more robust fix is to make the harness emit an explicit sentinel. General rule: for every metric, write down the input that would make it non-zero and confirm the code path exists.
