"""
Executable test for content/07-inference-serving/09-sampling-decoding.md

Concatenates the chapter's 4 CPU-runnable Python blocks in order and exercises
each one with tiny CPU tensors / a tiny dummy language model so the book's
actual code runs end to end.

Blocks covered:
  #0  (line ~17,  26 lines) sample_token()
  #7  (line ~281, 48 lines) RepetitionPenaltyProcessor, FrequencyPenaltyProcessor
  #8  (line ~349, 66 lines) BeamHypothesis, beam_search()
  #10 (line ~522, 48 lines) CompositeLogitsProcessor, build_logits_processor_list()
       (needs `transformers`, which is NOT in the guaranteed CI import list ->
        guarded with try/except; executed only if transformers happens to be
        importable, otherwise honestly SKIPPED per the hard rules)

Skipped blocks (fragments that are not standalone / need a real HF model, or
would need a GPU / network):
  #1  greedy_decode            -- needs a real `model(generated).logits` callable;
                                   non-standalone fragment reused conceptually by
                                   beam_search below, not separately exercised.
  #2  TemperatureProcessor     -- fragment (single class, no standalone demo)
  #3  TopKProcessor            -- fragment
  #4  TopPProcessor            -- fragment
  #5  MinPProcessor            -- fragment
  #6  TypicalProcessor         -- fragment
  #9  dola_logits               -- needs output_hidden_states from a real HF model
  #11 ProductionSampler + demo -- heuristically flagged needs-net / out of scope
                                   for this run; not one of the 4 assigned blocks
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from types import SimpleNamespace

# ============================================================
# Block #0 (line ~17) -- logits -> token pipeline: sample_token()
# ============================================================
import torch  # noqa: F811 (book repeats this import per-block; kept verbatim)
import torch.nn.functional as F  # noqa: F811
from typing import List, Optional  # noqa: F811

def sample_token(
    logits: torch.Tensor,          # shape: (vocab_size,), raw model outputs
    processors: List["LogitProcessor"],
    do_sample: bool = True,
) -> int:
    """
    Apply a list of logit processors in sequence, then sample one token.
    Returns the integer token id.
    """
    logits = logits.clone().float()  # always upcast to fp32 for numerical safety

    for proc in processors:
        logits = proc(logits)        # each processor modifies logits in-place or returns a new tensor

    if not do_sample:
        # greedy: just the argmax
        return int(logits.argmax())

    # convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


# ============================================================
# Block #7 (line ~281) -- Repetition and frequency/presence penalties
# ============================================================
class RepetitionPenaltyProcessor:
    """
    Multiplicative repetition penalty (CTRL-style).
    theta > 1 reduces repetition; theta = 1 is no-op.
    """
    def __init__(self, penalty: float, input_ids: torch.Tensor):
        assert penalty >= 1.0
        self.penalty = penalty
        # Track unique tokens seen in the context
        self.seen = set(input_ids.flatten().tolist())

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        for token_id in self.seen:
            if logits[token_id] > 0:
                logits[token_id] /= self.penalty
            else:
                logits[token_id] *= self.penalty
        return logits


class FrequencyPenaltyProcessor:
    """
    Additive frequency penalty: subtract alpha * count(t).
    Also supports presence penalty (binary).
    """
    def __init__(
        self,
        frequency_penalty: float,
        presence_penalty: float,
        input_ids: torch.Tensor,
    ):
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        # Count occurrences of each token in context
        token_list = input_ids.flatten().tolist()
        from collections import Counter
        self.counts = Counter(token_list)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        for token_id, count in self.counts.items():
            logits[token_id] -= (
                self.frequency_penalty * count
                + self.presence_penalty
            )
        return logits


# ============================================================
# Block #8 (line ~349) -- Beam search
# ============================================================
import heapq  # noqa: F811
from dataclasses import dataclass, field
from typing import Tuple

@dataclass(order=True)
class BeamHypothesis:
    score: float          # negative log-prob (min-heap)
    tokens: list = field(compare=False)

def beam_search(
    model,
    input_ids: torch.Tensor,     # (1, prefix_len)
    beam_size: int = 4,
    max_new_tokens: int = 100,
    eos_id: int = 2,
    length_alpha: float = 0.6,
) -> List[int]:
    """
    Minimal beam search. Returns the best sequence as a list of token ids.
    NB: production implementations use KV-cache for each beam; this toy
    version re-encodes each step for clarity.
    """
    prefix = input_ids[0].tolist()
    # heap items: (neg_score, token_list)
    active_beams: List[Tuple[float, List[int]]] = [(0.0, prefix[:])]
    finished: List[Tuple[float, List[int]]] = []

    for _ in range(max_new_tokens):
        if not active_beams:
            break
        candidates: List[Tuple[float, List[int]]] = []

        for neg_score, tokens in active_beams:
            ids = torch.tensor([tokens], dtype=torch.long)
            with torch.no_grad():
                logits = model(ids).logits[0, -1, :]   # (vocab,)
            log_probs = F.log_softmax(logits, dim=-1)

            # Expand: consider all tokens in the vocabulary
            topk_logprob, topk_ids = torch.topk(log_probs, beam_size)
            for lp, tid in zip(topk_logprob.tolist(), topk_ids.tolist()):
                new_tokens = tokens + [tid]
                new_neg_score = neg_score - lp   # minimise negative log-prob
                if tid == eos_id:
                    # Normalise by length
                    n = len(new_tokens) - len(prefix)
                    normalised = new_neg_score / (n ** length_alpha)
                    finished.append((normalised, new_tokens))
                else:
                    candidates.append((new_neg_score, new_tokens))

        # Keep the best beam_size active hypotheses
        candidates.sort(key=lambda x: x[0])
        active_beams = candidates[:beam_size]

    # Fall back to active beams if none finished
    if not finished:
        n_prefix = len(prefix)
        for neg_score, tokens in active_beams:
            n = len(tokens) - n_prefix
            normalised = neg_score / max(n, 1) ** length_alpha
            finished.append((normalised, tokens))

    finished.sort(key=lambda x: x[0])
    return finished[0][1]   # best hypothesis


# ============================================================
# Block #10 (line ~522) -- HuggingFace-compatible logits processor pipeline
# ============================================================
# `transformers` is a third-party dependency NOT in the guaranteed CI import
# list (numpy, torch, einops, sklearn, stdlib only). Per the hard rules this
# import must be guarded so the module still loads without it; the block is
# then executed only when the package happens to be importable, else it is
# defined-but-not-called (an honest SKIP).
try:
    from transformers import LogitsProcessor, LogitsProcessorList
    HAS_TRANSFORMERS = True
except Exception:
    LogitsProcessor = object
    LogitsProcessorList = None
    HAS_TRANSFORMERS = False

import torch  # noqa: F811

class CompositeLogitsProcessor(LogitsProcessor):
    """
    A single LogitsProcessor that applies a pipeline of sub-processors,
    each expecting (input_ids, scores) -> scores.
    This follows the HuggingFace LogitsProcessor protocol.
    """

    def __init__(self, processors: list):
        self.processors = processors

    def __call__(
        self,
        input_ids: torch.LongTensor,     # (batch, seq_len)
        scores: torch.FloatTensor,       # (batch, vocab_size)
    ) -> torch.FloatTensor:
        for proc in self.processors:
            scores = proc(input_ids, scores)
        return scores


# Example: build a typical production pipeline
def build_logits_processor_list(
    temperature: float = 0.8,
    repetition_penalty: float = 1.1,
    top_k: int = 50,
    top_p: float = 0.9,
) -> "LogitsProcessorList":
    from transformers import (
        TemperatureLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    procs = LogitsProcessorList()
    if repetition_penalty != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if temperature != 1.0:
        procs.append(TemperatureLogitsWarper(temperature=temperature))
    if top_k > 0:
        procs.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p < 1.0:
        procs.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
    return procs


# ============================================================
# Test harness
# ============================================================

class TinyLM(torch.nn.Module):
    """A tiny deterministic stand-in for a real causal LM: embeds the token
    sequence, mean-pools it, and projects to vocab-sized logits. It exposes
    a `.logits` attribute like a HuggingFace model output, which is exactly
    the interface `beam_search` (and `greedy_decode` elsewhere in the
    chapter) rely on. CPU-only, tiny shapes, fixed seed."""

    def __init__(self, vocab_size: int = 16, dim: int = 8, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.head = torch.nn.Linear(dim, vocab_size)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab_size, dim, generator=g))
            self.head.weight.copy_(torch.randn(vocab_size, dim, generator=g) * 0.5)
            self.head.bias.copy_(torch.randn(vocab_size, generator=g) * 0.1)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)          # (batch, seq, dim)
        h = x.mean(dim=1)                  # simple pooling over context
        logits = self.head(h)              # (batch, vocab)
        seq_len = input_ids.shape[1]
        logits = logits.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, vocab)
        return SimpleNamespace(logits=logits)


def main():
    # --- Block #0: sample_token ---------------------------------------------
    torch.manual_seed(0)
    vocab_size = 32
    logits = torch.randn(vocab_size)
    logits[5] = 20.0  # dominant peak

    class _IdentityProc:
        def __call__(self, lg):
            return lg

    class _ScaleProc:
        def __init__(self, t):
            self.t = t
        def __call__(self, lg):
            return lg / self.t

    # do_sample=False path: greedy argmax through the processor pipeline
    greedy_tok = sample_token(logits, [_IdentityProc()], do_sample=False)
    assert greedy_tok == 5, f"expected argmax token 5, got {greedy_tok}"

    # do_sample=True path: sharpen with a temperature-like scale processor,
    # then sample many times -- dominant peak should win overwhelmingly.
    counts = {}
    for _ in range(200):
        t = sample_token(logits.clone(), [_ScaleProc(0.3)], do_sample=True)
        counts[t] = counts.get(t, 0) + 1
    assert counts.get(5, 0) > 150, f"expected token 5 to dominate, got counts={counts}"
    print(f"[OK] block #0 sample_token: greedy={greedy_tok}, sampled token 5 in {counts.get(5,0)}/200 draws")

    # --- Block #7: RepetitionPenaltyProcessor, FrequencyPenaltyProcessor -----
    torch.manual_seed(0)
    base_logits = torch.randn(vocab_size)
    base_logits[3] = 5.0
    context = torch.tensor([3, 3, 7, 3])  # token 3 appears 3 times

    rep_proc = RepetitionPenaltyProcessor(penalty=1.5, input_ids=context)
    penalized = rep_proc(base_logits)
    # token 3's positive logit should shrink under the multiplicative penalty
    assert penalized[3] < base_logits[3]
    assert torch.isclose(penalized[3], base_logits[3] / 1.5)
    # unseen token's logit is untouched
    unseen_id = 0 if 0 not in {3, 7} else 1
    assert torch.equal(penalized[unseen_id], base_logits[unseen_id])

    freq_proc = FrequencyPenaltyProcessor(
        frequency_penalty=0.5, presence_penalty=0.2, input_ids=context
    )
    freq_penalized = freq_proc(base_logits)
    # token 3 appears 3x -> subtract 0.5*3 + 0.2 = 1.7
    expected_3 = base_logits[3] - (0.5 * 3 + 0.2)
    assert torch.isclose(freq_penalized[3], expected_3), (freq_penalized[3], expected_3)
    # token 7 appears 1x -> subtract 0.5*1 + 0.2 = 0.7
    expected_7 = base_logits[7] - (0.5 * 1 + 0.2)
    assert torch.isclose(freq_penalized[7], expected_7), (freq_penalized[7], expected_7)
    print("[OK] block #7 RepetitionPenaltyProcessor + FrequencyPenaltyProcessor")

    # --- Block #8: BeamHypothesis + beam_search -------------------------------
    torch.manual_seed(0)
    tiny_model = TinyLM(vocab_size=16, dim=8, seed=0)
    input_ids = torch.tensor([[1]], dtype=torch.long)  # (1, prefix_len=1)

    best_seq = beam_search(
        tiny_model,
        input_ids,
        beam_size=3,
        max_new_tokens=4,
        eos_id=2,
        length_alpha=0.6,
    )
    assert isinstance(best_seq, list)
    assert best_seq[0] == 1  # prefix token preserved
    assert len(best_seq) > 1  # at least one token was generated
    # sanity: BeamHypothesis dataclass itself is usable / orderable
    h1 = BeamHypothesis(score=1.0, tokens=[1, 2])
    h2 = BeamHypothesis(score=0.5, tokens=[1, 3])
    assert h2 < h1  # ordered by score (min-heap semantics)
    print(f"[OK] block #8 beam_search returned sequence of length {len(best_seq)}: {best_seq}")

    # --- Block #10: CompositeLogitsProcessor + build_logits_processor_list ---
    if HAS_TRANSFORMERS:
        pipeline = build_logits_processor_list(
            temperature=0.8, repetition_penalty=1.1, top_k=8, top_p=0.9
        )
        composite = CompositeLogitsProcessor(list(pipeline))
        dummy_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        dummy_scores = torch.randn(1, vocab_size)
        out_scores = composite(dummy_input_ids, dummy_scores)
        assert out_scores.shape == dummy_scores.shape
        assert torch.isfinite(out_scores[out_scores != float("-inf")]).all()
        print("[OK] block #10 CompositeLogitsProcessor + build_logits_processor_list "
              "(transformers available)")
    else:
        # Honest SKIP: `transformers` is not in the guaranteed CI dependency
        # set (only numpy, torch, einops, sklearn, stdlib are). The classes
        # above are still defined (with LogitsProcessor falling back to
        # `object`) and importable/parseable, but the HF-dependent pipeline
        # construction is not invoked, per the hard rules for optional
        # third-party imports.
        assert CompositeLogitsProcessor is not None  # class at least defined
        print("[SKIP] block #10 CompositeLogitsProcessor + build_logits_processor_list "
              "-- `transformers` not installed (optional dependency, guarded import)")

    print("\nAll assigned blocks executed (block #10 executed if transformers "
          "is available, otherwise honestly skipped).")


if __name__ == "__main__":
    main()
