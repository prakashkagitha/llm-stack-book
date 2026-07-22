"""
Runs the CPU-runnable Python blocks from
content/02-transformer/12-diffusion-nonAR-lms.md, concatenated in order so
that later blocks can rely on names defined by earlier ones (as they do in
the chapter itself). Each block is copied verbatim from the (now-fixed)
chapter; only minimal glue needed to make blocks that only *define*
something (a function) actually execute has been added, and is clearly
marked "GLUE".

Blocks covered (2 CPU-runnable blocks per task spec):
  #0 (line ~108) - masked_diffusion_sample: the from-scratch absorbing-state
                    masked-diffusion sampler (confidence-based remasking).
  #3 (line ~325) - infill_example: infilling built on top of the block #0
                    sampler (clamp prefix AND suffix, denoise the hole).

Blocks explicitly SKIPPED (per task spec):
  #1 - SKIP(non-python): ```text ascii-art block-diffusion diagram
                          (line ~235, "Block 1 Block 2 Block 3 ..." schematic).
  #2 - SKIP(fragment): block_causal_mask (line ~252) builds an attention mask
                        for block diffusion but is never wired to an actual
                        attention computation in the chapter (no QK^T call
                        uses it) -- it is a standalone illustrative fragment,
                        not exercised end-to-end. Per the task spec this is
                        listed as "#2=fragment" and is a default SKIP. (We
                        could trivially call it and check the mask shape, but
                        that would not exercise any interesting logic beyond
                        what block_of[None,:] <= block_of[:,None] already
                        states in one line, so we leave it as documented-only.)

REAL BUG FOUND & FIXED IN THE BOOK:
  The chapter's `infill_example` (block #3) built a fully-clamped array `x`
  (prefix tokens on the left, suffix tokens on the right, `is_clamped` mask
  marking both) but then called `masked_diffusion_sample(..., prompt=None)`,
  which silently DISCARDED all of that clamping -- `masked_diffusion_sample`
  only supported clamping a contiguous *prefix* via the `prompt` argument, so
  passing `prompt=None` meant the function generated an entirely fresh,
  unconstrained sequence and the returned tokens had no relationship to
  `prefix_ids` / `suffix_ids` at all. The chapter's own inline comment even
  flagged this half-finished state: "(in practice, pass is_clamped into the
  sampler as is_prompt)". This is a genuine correctness bug: the whole point
  of the example is to demonstrate bidirectional infilling conditioned on
  BOTH sides, and as written it demonstrated nothing of the kind.

  Fix applied to content/02-transformer/12-diffusion-nonAR-lms.md:
    - `masked_diffusion_sample` (block #0) gained two new optional
      parameters, `clamp_mask` and `clamp_values`, that clamp an ARBITRARY
      subset of positions (not just a contiguous prefix). `prompt` is kept
      as the backward-compatible contiguous-prefix special case.
    - `infill_example` (block #3) now passes `clamp_mask=is_clamped,
      clamp_values=x` instead of `prompt=None`, so the prefix and suffix
      tokens really are held fixed while the hole is denoised.
  This test asserts the fixed behavior: the returned sequence's prefix and
  suffix positions exactly equal the original clamped input tokens.
"""

import torch
import torch.nn.functional as F

# ======================================================================
# Block #0 (line ~108): masked_diffusion_sample
# ======================================================================
print("=" * 70)
print("Block #0 (line ~108): masked_diffusion_sample")
print("=" * 70)

# --- verbatim from the chapter (as fixed: clamp_mask/clamp_values added) ---
MASK_ID = 0  # reserve index 0 in the vocab for the [MASK] absorbing token

@torch.no_grad()
def masked_diffusion_sample(
    denoiser,            # callable: (LongTensor[L]) -> FloatTensor[L, V] logits
    L: int,              # sequence length to generate
    num_steps: int,      # number of denoising iterations N (the serial depth)
    vocab_size: int,
    temperature: float = 0.0,   # 0.0 = greedy/argmax per position
    prompt: torch.Tensor = None,   # optional LongTensor of clamped prefix tokens
    clamp_mask: torch.Tensor = None,    # optional (L,) bool mask of arbitrary clamped positions
    clamp_values: torch.Tensor = None,  # (L,) values at those positions; used with clamp_mask
    device: str = "cpu",
):
    """
    Absorbing-state masked diffusion sampler.

    Strategy ("confidence-based remasking", a la LLaDA's low-confidence remasking):
    at each step we predict ALL masked positions, but only *keep* (unmask) the
    most confident predictions, sized so that the fraction of masked tokens
    follows a linear schedule from 1.0 down to 0.0 over num_steps. Less confident
    positions are returned to [MASK] and revisited in later steps.
    """
    # 1. Start fully masked.
    x = torch.full((L,), MASK_ID, dtype=torch.long, device=device)

    # 2. Clamp fixed context (these positions are never masked and never predicted).
    #    `clamp_mask`/`clamp_values` clamp an ARBITRARY subset of positions (e.g. a
    #    prefix AND a suffix for infilling); `prompt` is the common special case of
    #    clamping only a contiguous prefix.
    is_prompt = torch.zeros(L, dtype=torch.bool, device=device)
    if clamp_mask is not None:
        is_prompt = clamp_mask.to(device)
        x[is_prompt] = clamp_values.to(device)[is_prompt]
    elif prompt is not None:
        x[: prompt.numel()] = prompt.to(device)
        is_prompt[: prompt.numel()] = True

    # 3. Denoising schedule: target number of STILL-masked tokens after step k.
    #    Linear from "all generatable positions masked" down to 0.
    n_gen = int((~is_prompt).sum().item())   # positions we actually generate
    # masked_target[k] = how many gen-positions remain masked AFTER step k+1
    masked_target = [
        round(n_gen * (1.0 - (k + 1) / num_steps)) for k in range(num_steps)
    ]

    for step in range(num_steps):
        masked = (x == MASK_ID) & (~is_prompt)   # positions still to fill
        if masked.sum() == 0:
            break

        logits = denoiser(x)                     # (L, V) — full bidirectional pass
        logits[:, MASK_ID] = -float("inf")       # never predict [MASK] itself

        if temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (L,)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)     # (L,)
        else:
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)       # greedy + its confidence

        # Candidate fill: tentatively set every masked position to its prediction.
        x_candidate = x.clone()
        x_candidate[masked] = pred[masked]

        # Decide how many of the currently-masked positions to KEEP unmasked.
        n_keep = int(masked.sum().item()) - masked_target[step]
        n_keep = max(0, n_keep)

        # Rank masked positions by confidence; keep the top-n_keep, remask the rest.
        conf_masked = conf.clone()
        conf_masked[~masked] = -float("inf")     # only compete among masked positions
        if n_keep > 0:
            keep_idx = torch.topk(conf_masked, n_keep).indices
            new_x = x.clone()
            new_x[masked] = MASK_ID              # provisionally remask all
            new_x[keep_idx] = x_candidate[keep_idx]  # commit the confident ones
            x = new_x
        # if n_keep == 0 we commit nothing this step (rare; only at the very start)

    # Final cleanup: fill any leftover masks greedily (last step should handle this).
    leftover = (x == MASK_ID) & (~is_prompt)
    if leftover.any():
        logits = denoiser(x)
        logits[:, MASK_ID] = -float("inf")
        x[leftover] = logits[leftover].argmax(dim=-1)

    return x

# --- GLUE: a tiny toy "denoiser" bidirectional model to exercise the sampler ---
torch.manual_seed(0)

VOCAB_SIZE = 20

class ToyDenoiser(torch.nn.Module):
    """A minimal bidirectional token->logits model, just enough to give
    masked_diffusion_sample something real to call. Standing in for "a
    bidirectional transformer returning logits of shape (L, V)" per the
    chapter's own description of the denoiser callable's contract."""
    def __init__(self, vocab_size: int, hidden: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.mix = torch.nn.Linear(hidden, hidden)
        self.proj = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)                       # (L, hidden)
        ctx = h.mean(dim=0, keepdim=True)        # crude bidirectional mixing
        h = torch.tanh(self.mix(h + ctx))
        return self.proj(h)                      # (L, vocab_size)


_toy_model = ToyDenoiser(VOCAB_SIZE)


def denoiser(x: torch.Tensor) -> torch.Tensor:
    return _toy_model(x)


# --- GLUE: call masked_diffusion_sample (block #0) with a tiny prompt ---
L = 12
prompt = torch.tensor([5, 6, 7], dtype=torch.long)
out = masked_diffusion_sample(
    denoiser, L=L, num_steps=6, vocab_size=VOCAB_SIZE, prompt=prompt, device="cpu",
)
print("sampled sequence:", out.tolist())

assert out.shape == (L,)
assert out.dtype == torch.long
# The prompt prefix must be preserved verbatim.
assert torch.equal(out[: prompt.numel()], prompt), "prompt prefix was overwritten"
# Every position must have been filled in -- no leftover [MASK] tokens.
assert (out != MASK_ID).all(), "sampler left [MASK] tokens in the output"

print("Block #0 OK: masked_diffusion_sample ran, preserved the prompt, filled all masks.\n")


# ======================================================================
# Block #3 (line ~325): infill_example
# ======================================================================
print("=" * 70)
print("Block #3 (line ~325): infill_example")
print("=" * 70)

# --- verbatim from the chapter (as fixed: clamp_mask/clamp_values used) ---
# Infilling with the same sampler: clamp prefix AND suffix, denoise the middle.
def infill_example(denoiser, prefix_ids, suffix_ids, hole_len, vocab_size):
    L = len(prefix_ids) + hole_len + len(suffix_ids)
    x = torch.full((L,), MASK_ID, dtype=torch.long)
    is_clamped = torch.zeros(L, dtype=torch.bool)

    # left context
    x[: len(prefix_ids)] = torch.tensor(prefix_ids)
    is_clamped[: len(prefix_ids)] = True
    # right context (note: this is the future, which AR cannot use!)
    x[len(prefix_ids) + hole_len :] = torch.tensor(suffix_ids)
    is_clamped[len(prefix_ids) + hole_len :] = True

    # Reuse the diffusion loop, but treat ALL clamped positions (prefix AND
    # suffix) like a "prompt" so they are never masked and never overwritten.
    # The hole is denoised conditioned on BOTH sides simultaneously.
    return masked_diffusion_sample(
        denoiser, L=L, num_steps=hole_len, vocab_size=vocab_size,
        clamp_mask=is_clamped, clamp_values=x,
    ), is_clamped

# --- GLUE: call infill_example with a tiny prefix/suffix and a small hole ---
prefix_ids = [1, 2, 3]
suffix_ids = [8, 9]
hole_len = 4

result_x, is_clamped = infill_example(denoiser, prefix_ids, suffix_ids, hole_len, VOCAB_SIZE)
L_expected = len(prefix_ids) + hole_len + len(suffix_ids)
print("infilled sequence:", result_x.tolist())
print("is_clamped mask:  ", is_clamped.tolist())

assert result_x.shape == (L_expected,)
assert is_clamped.shape == (L_expected,)
assert is_clamped.sum().item() == len(prefix_ids) + len(suffix_ids)
# The prefix and suffix must come back EXACTLY as given -- this is the whole
# point of infilling (bidirectional clamped context), and is what the
# book's original code failed to guarantee before the fix above.
assert torch.equal(result_x[: len(prefix_ids)], torch.tensor(prefix_ids, dtype=torch.long))
assert torch.equal(result_x[len(prefix_ids) + hole_len :], torch.tensor(suffix_ids, dtype=torch.long))
# The hole itself must be fully denoised -- no leftover [MASK] tokens.
hole = result_x[len(prefix_ids): len(prefix_ids) + hole_len]
assert (hole != MASK_ID).all(), "infilled hole still contains [MASK] tokens"

print("Block #3 OK: infill_example preserved prefix/suffix exactly and filled the hole.\n")

print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
