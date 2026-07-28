# %% [markdown]
#
# # Speculative Decoding: Measuring the Real Wall-Clock Speedup
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes (plus one-time weight
# > download). Not executed in the book — run it to get your own numbers.
#
# You will load a real target model and a real, much smaller draft model from
# the same tokenizer family, implement the speculative-decoding accept/reject
# rule by hand to measure the token-level acceptance rate directly, and then
# use `transformers`' production `assistant_model=` path to measure the actual
# end-to-end tokens/s speedup against plain autoregressive decoding.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/07-inference-serving/06-speculative-decoding.html)
# for the full explanation.

# %%
# This notebook downloads real model weights from the Hugging Face Hub, so it
# needs network access. GPT-2 checkpoints are small, openly downloadable, and
# permissively licensed (see each model card) — no gated-access approval is
# required, unlike some newer instruction-tuned families (e.g. Llama, Gemma).
%pip install -q "transformers>=4.42" accelerate

# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.is_available(), "This notebook needs a CUDA GPU (target: 1x H100 80GB)."
device = torch.device("cuda")
dtype = torch.bfloat16  # Hopper's native fast dtype; both models run in bf16.

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

print("Device:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())

# %% [markdown]
# ## Choosing a target/draft pair
#
# Speculative decoding needs a **target** model `p` (large, accurate, slow)
# and a **draft** model `q` (small, fast, approximate) that share a tokenizer,
# so the draft's proposed token ids are directly meaningful to the target.
#
# We use **GPT-2 XL (~1.5B params, 48 layers)** as the target and **GPT-2
# small (~124M params, 12 layers)** as the draft — they share the exact same
# BPE tokenizer/vocabulary, both download without any license gate, and the
# size gap (~12x fewer params, 4x fewer layers) is large enough to make the
# draft meaningfully cheaper per token. In production you'd more commonly see
# a bigger disparity (e.g. a 70B target with a ~1B sibling, or a distilled/
# Medusa/EAGLE drafter, per the chapter) — that widens the cost-ratio `c` even
# further and pushes the achievable speedup up, so treat the numbers below as
# a conservative, reproducible lower bound rather than the ceiling.
#
# Expected result of this cell: two models loaded in bf16 on the GPU, GPT-2
# XL's parameter count roughly 12x GPT-2 small's, and a few GB of GPU memory
# used in total (well within an H100's 80GB).

# %%
TARGET_NAME = "gpt2-xl"   # ~1.5B params, the accurate but slow model p
DRAFT_NAME = "gpt2"       # ~124M params, the fast but weaker model q

tokenizer = AutoTokenizer.from_pretrained(TARGET_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

# `torch_dtype=` is the load-time dtype kwarg accepted across transformers>=4.42
# (newer releases also accept `dtype=`; on the very latest it may emit a
# deprecation warning but still works).
target_model = AutoModelForCausalLM.from_pretrained(TARGET_NAME, torch_dtype=dtype).to(device).eval()
draft_model = AutoModelForCausalLM.from_pretrained(DRAFT_NAME, torch_dtype=dtype).to(device).eval()

n_target = sum(p.numel() for p in target_model.parameters())
n_draft = sum(p.numel() for p in draft_model.parameters())
print(f"target ({TARGET_NAME}): {n_target/1e6:.0f}M params")
print(f"draft  ({DRAFT_NAME}):  {n_draft/1e6:.0f}M params  (~{n_target/n_draft:.1f}x smaller)")
print(f"GPU memory after loading both models: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# ## Measuring the draft/target cost ratio `c`
#
# The chapter's speedup formula depends on `c`, the draft's per-token cost as
# a *fraction* of one target forward pass. Autoregressive decoding is
# memory-bandwidth bound (one token at a time -> the bottleneck is streaming
# parameters from HBM, not FLOPs), so `c` should track the parameter-count
# ratio roughly, modulo the fixed per-launch overhead that hurts the tiny
# draft model relatively more. We measure it directly with `torch.cuda.Event`
# rather than assuming it.
#
# Expected result: `c` on the order of 0.1-0.3 for this pair — noticeably
# larger than the ~0.02 a truly tiny (sub-100M) drafter would give against a
# 70B target, which is exactly why this demo's speedup will be more modest
# than the headline 1.5-2.5x band for production-scale pairs.

# %%
def time_forward_ms(model, input_ids, n_iters=30, n_warmup=10):
    """Median-ish per-call latency (ms) of a single forward pass, via CUDA events."""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            _ = model(input_ids)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iters


probe_ids = tokenizer("The", return_tensors="pt").input_ids.to(device)  # single-token decode-step proxy
target_ms = time_forward_ms(target_model, probe_ids)
draft_ms = time_forward_ms(draft_model, probe_ids)
c_measured = draft_ms / target_ms
print(f"target forward: {target_ms:.3f} ms/step   draft forward: {draft_ms:.3f} ms/step")
print(f"measured cost ratio c = draft/target = {c_measured:.3f}")

# %% [markdown]
# ## Baseline: plain autoregressive decoding
#
# The reference point every speedup number is measured against: the target
# model alone, one token at a time, with its own KV cache. Decoding a small
# batch=1 sequence like this is memory-bandwidth bound, so tokens/s should be
# roughly stable regardless of prompt content.
#
# Expected result: on one H100, a ~1.5B model in bf16 at batch=1 should decode
# on the order of tens of tokens/second (the exact number depends on prompt
# length, sampling settings, and kernel/driver versions — treat it as a rough
# baseline to compare the speculative numbers against, not an absolute spec).

# %%
PROMPT_CODE = "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is a prime number.\"\"\"\n"
PROMPT_CHAT = "The most surprising thing about the history of the internet is"
MAX_NEW_TOKENS = 64

def timed_generate(model, inputs, **gen_kwargs):
    """Time a full .generate() call with CUDA events (includes warmup + sync)."""
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id, **gen_kwargs)  # warmup
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id, **gen_kwargs)
        end.record()
        torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return out, n_new / (ms / 1000.0), peak_gb


inputs_code = tokenizer(PROMPT_CODE, return_tensors="pt").to(device)
_, base_tps, base_mem = timed_generate(target_model, inputs_code, do_sample=False)
print(f"baseline (target-only, greedy): {base_tps:.1f} tokens/s   peak mem {base_mem:.2f} GB")

# %% [markdown]
# ## The accept/reject rule, from scratch
#
# Matching the chapter's notation exactly: at each proposal step the draft
# samples a token `x` from `q(x)`, and we accept it with probability
# `min(1, p(x)/q(x))`. On rejection, we resample from the **residual
# distribution** `p_res(x) ∝ max(0, p(x) - q(x))`, which guarantees the
# emitted token is distributed exactly as `p`, regardless of `q`'s quality.
# For greedy decoding this degenerates to an exact rule: accept iff the
# draft's argmax equals the target's argmax at that position.
#
# For clarity and to make this reference implementation trivially auditable
# (no KV-cache bookkeeping to get subtly wrong), the target's verification
# pass below **recomputes the full running sequence from scratch** every
# round instead of reusing a cache across rounds. That keeps the causal-LM
# indexing unambiguous (`logits[:, t, :]` predicts the token at position
# `t+1`, full stop) at the cost of some wall-clock efficiency — so treat this
# cell's own timing as a correctness check on the acceptance rate `alpha`,
# not as the headline speedup number. The headline number comes from the
# cache-optimized production path in the next cell.
#
# Expected result: acceptance rate on the order of 0.5-0.9 on the more
# templated/predictable code prompt, and somewhat lower on the open-ended
# prose prompt — illustrating exactly why `alpha` (and hence the speedup)
# is workload-dependent.

# %%
def sample_from_residual(p, q):
    """Sample from p_res(x) ∝ max(0, p(x) - q(x)); falls back to p if p ⊆ q numerically."""
    # multinomial wants float32 (and bf16 logits/probs are numerically noisy for this anyway)
    p, q = p.float(), q.float()
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total.item() <= 1e-8:
        residual, total = p, p.sum()
    residual = residual / total
    return torch.multinomial(residual, num_samples=1).item()


@torch.no_grad()
def speculative_generate_reference(prompt, gamma, max_new_tokens, do_sample=False, temperature=1.0, max_rounds=200):
    """Reference (non-cached) speculative decoding loop. Returns (tokens, alpha, n_rounds)."""
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = tokens.shape[1]
    accept_count, propose_count = 0, 0

    for _ in range(max_rounds):
        if tokens.shape[1] - prompt_len >= max_new_tokens:
            break
        prefix_len = tokens.shape[1]

        # 1) Draft proposes up to `gamma` tokens autoregressively (HF manages the draft's own KV cache).
        #    Pass an explicit all-ones attention_mask (no padding here) to silence HF's mask warning.
        draft_out = draft_model.generate(
            tokens, attention_mask=torch.ones_like(tokens),
            max_new_tokens=gamma, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.eos_token_id,
        )
        draft_seq = draft_out.sequences[:, prefix_len:]
        n_drafted = draft_seq.shape[1]
        if n_drafted == 0:
            break
        q_logits = draft_out.scores[:n_drafted]  # q's distribution at each drafted position

        # 2) Target verifies all drafted tokens in ONE forward pass over the full sequence (no cache reuse).
        full_seq = torch.cat([tokens, draft_seq], dim=1)
        target_logits_full = target_model(full_seq).logits
        # logits[:, prefix_len - 1 + i, :] is p's distribution used to verify draft_seq[i]
        p_logits = [target_logits_full[:, prefix_len - 1 + i, :] for i in range(n_drafted)]

        # 3) Accept/reject each drafted token in order; stop at the first rejection.
        accepted_all = True
        for i in range(n_drafted):
            x_i = draft_seq[0, i].item()
            propose_count += 1
            if do_sample:
                p_i = F.softmax(p_logits[i].squeeze(0).float(), dim=-1)
                q_i = F.softmax(q_logits[i].squeeze(0).float(), dim=-1)
                accept_prob = min(1.0, (p_i[x_i] / q_i[x_i].clamp_min(1e-12)).item())
                accepted = torch.rand(1).item() <= accept_prob
            else:
                accepted = torch.argmax(p_logits[i], dim=-1).item() == x_i

            if accepted:
                accept_count += 1
                tokens = torch.cat([tokens, draft_seq[:, i:i + 1]], dim=1)
            else:
                correction = (
                    sample_from_residual(p_i, q_i) if do_sample
                    else torch.argmax(p_logits[i], dim=-1).item()
                )
                tokens = torch.cat([tokens, torch.tensor([[correction]], device=device)], dim=1)
                accepted_all = False
                break

        if accepted_all:
            # All drafted tokens accepted -> one bonus token, free, from the target's next distribution.
            bonus_logits = target_logits_full[:, -1, :]
            bonus = (
                torch.multinomial(F.softmax(bonus_logits.float(), dim=-1), num_samples=1)
                if do_sample else torch.argmax(bonus_logits, dim=-1, keepdim=True)
            )
            tokens = torch.cat([tokens, bonus], dim=1)

    alpha = accept_count / max(propose_count, 1)
    return tokens, alpha, propose_count


GAMMA = 5  # matches the chapter's worked numeric example
for name, prompt in [("code completion", PROMPT_CODE), ("open-ended prose", PROMPT_CHAT)]:
    _, alpha, n_proposed = speculative_generate_reference(prompt, GAMMA, MAX_NEW_TOKENS, do_sample=False)
    expected_tokens_per_step = (1 - alpha ** (GAMMA + 1)) / (1 - alpha) if alpha < 1 else GAMMA + 1
    predicted_speedup = expected_tokens_per_step / (1 + GAMMA * c_measured)
    print(f"[{name:16s}] alpha={alpha:.2f}  (n_proposed={n_proposed})  "
          f"formula-predicted speedup ~= {predicted_speedup:.2f}x  (with measured c={c_measured:.2f})")

# %% [markdown]
# ## The production path: `transformers`' built-in assisted generation
#
# Re-implementing the accept/reject math above is what makes the mechanism
# legible, but `transformers`' `model.generate(assistant_model=..., ...)`
# already implements this with proper KV-cache reuse/cropping across rounds
# and an adaptive draft length (`num_assistant_tokens`), so it's the right
# tool for the actual wall-clock comparison. Passing `do_sample=True` runs the
# sampling-based accept/reject rule (the exact math above); `do_sample=False`
# runs the greedy special case. The draft length is read from the assistant
# model's own generation config (`draft_model.generation_config.num_assistant_tokens`).
#
# Expected result: assisted generation should decode noticeably more
# tokens/second than the plain baseline above at greedy, with a smaller (but
# still positive) gain at low temperature (sampling accepts less often than
# greedy's exact-match rule, all else equal).

# %%
draft_model.generation_config.num_assistant_tokens = GAMMA

_, spec_tps_greedy, spec_mem_greedy = timed_generate(
    target_model, inputs_code, assistant_model=draft_model, do_sample=False,
)
torch.manual_seed(0)
_, spec_tps_sample, spec_mem_sample = timed_generate(
    target_model, inputs_code, assistant_model=draft_model, do_sample=True, temperature=0.7, top_p=0.95,
)

print(f"baseline (greedy):            {base_tps:6.1f} tok/s")
print(f"speculative (greedy):         {spec_tps_greedy:6.1f} tok/s   speedup {spec_tps_greedy/base_tps:.2f}x   peak mem {spec_mem_greedy:.2f} GB")
print(f"speculative (T=0.7 sampling): {spec_tps_sample:6.1f} tok/s   speedup {spec_tps_sample/base_tps:.2f}x   peak mem {spec_mem_sample:.2f} GB")

# %% [markdown]
# ## Why `alpha` and `c` set the speedup — and when it stops helping
#
# From the chapter, expected tokens emitted per verification round and the
# resulting speedup over plain decoding are:
#
# ```
# E[tokens/step] = (1 - alpha^(gamma+1)) / (1 - alpha)
# Speedup        = E[tokens/step] / (1 + gamma * c)
# ```
#
# Two forces fight each other as you increase `gamma` (the draft length):
# more speculative tokens means more free tokens *if* they're accepted, but
# also more wasted draft-forward-pass cost when they aren't — so there's an
# optimal `gamma*` for a given `alpha`, and it's only worth pushing `gamma` up
# when `alpha` is high (predictable, templated workloads like code far more
# than open-ended chat). A quick, fixed-`gamma` sweep with the production API
# makes this trade-off directly visible.
#
# Expected result: tokens/s should rise from `gamma=2` to some middle value
# and then flatten or dip again as `gamma` grows further, especially on the
# lower-`alpha` prose prompt.

# %%
for g in (2, 4, 8):
    draft_model.generation_config.num_assistant_tokens = g
    draft_model.generation_config.num_assistant_tokens_schedule = "constant"  # pin gamma; HF's default is adaptive
    inputs_g = tokenizer(PROMPT_CODE, return_tensors="pt").to(device)
    _, tps, _ = timed_generate(target_model, inputs_g, assistant_model=draft_model, do_sample=False)
    print(f"gamma={g}: {tps:6.1f} tok/s   speedup {tps/base_tps:.2f}x")

# %% [markdown]
# ## The gating condition: this only helps when decode is memory-bound
#
# Everything above is single-sequence (batch=1) decoding, which is exactly
# the memory-bandwidth-bound regime speculative decoding targets: the
# bottleneck is streaming the target's weights from HBM once per token, and
# verifying `gamma` draft tokens in one target pass amortizes that streaming
# cost over several tokens. At large batch sizes, decode becomes
# compute-bound instead (the matmuls are big enough to saturate the SMs), and
# the *extra* verification FLOPs for rejected tokens cost real time with no
# corresponding memory-bandwidth win — production engines (vLLM, TensorRT-LLM,
# etc.) gate speculative decoding on batch size / queue depth for exactly this
# reason. This notebook does not sweep batch size; treat that as the natural
# follow-up experiment.

# %% [markdown]
# ## What you should see
#
# - **Acceptance rate `alpha`**: on the order of 0.5-0.9 for the templated
#   code prompt and somewhat lower for open-ended prose — this is the single
#   biggest lever on speedup, and it is a property of how *predictable* the
#   target's distribution is at each position, not of the draft model alone.
# - **Wall-clock speedup**: expect something in the rough 1.2-2x range for
#   this GPT-2 XL / GPT-2 pair — likely below the ~1.5-2.5x band you'd see
#   with a more disparate target/draft pair (e.g. a ~10-50x parameter gap,
#   or a distilled/Medusa/EAGLE drafter) because our measured cost ratio `c`
#   is comparatively large. Your exact numbers will depend on prompt,
#   hardware, driver/kernel versions, and the `transformers` version's
#   adaptive-`gamma` heuristic — treat everything here as "roughly", not a
#   spec.
# - **Greedy vs. low-temperature sampling**: greedy's exact-match accept rule
#   is usually more forgiving than the probabilistic `min(1, p/q)` rule used
#   under sampling, so expect somewhat higher acceptance (and speedup) at
#   greedy than at `T=0.7`.
# - **Key takeaway**: speculative decoding trades *extra FLOPs* (the draft's
#   forward passes, plus wasted target verification on rejected tokens) for
#   *fewer round trips to HBM* — a good trade only in the memory-bandwidth-
#   bound, small-batch regime, and only when the draft is a decent-enough
#   predictor of the target (`alpha` not too low).
#
# Next step: swap in a bigger/more disparate pair (e.g. a real instruct model
# with a same-family small sibling, if you have Hub access to a gated repo)
# and re-run the `gamma` sweep; then read the chapter's Medusa/EAGLE/tree-
# attention sections for drafters that avoid a second model entirely.
