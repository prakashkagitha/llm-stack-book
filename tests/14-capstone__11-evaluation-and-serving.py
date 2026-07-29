"""
Runs the CPU-runnable Python code blocks from:
    content/14-capstone/11-evaluation-and-serving.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, exactly as the chapter's own `stacklm` package would assemble them
across files (infer.py -> probes.py -> agent.py -> quantize.py -> sweep.py ->
export.py -> cli.py). The chapter's cross-chapter imports (`from stacklm.model
.transformer import Stack100M`, `from stacklm.post.chat import SPECIAL`, ...)
are mechanically dropped -- every symbol they import is defined earlier in
THIS file, either verbatim (glue for things this chapter itself defines) or as
a small, explicitly-labeled toy stand-in for a class/function that lives in a
DIFFERENT chapter (14.3's tokenizer, 14.4's Stack100M, 14.9's chat/grpo,
14.10's agent loop/retriever). Every stand-in is called out in a comment where
it is defined. The chapter's OWN logic (evaluation probes, quantization
primitives, the QuantizedLinear/Embedding swap, export/load, the sweep, the
CLI) is copied and exercised unmodified.

Tested blocks (chapter's own numbering):
    #0  (Sec 2,  infer.py)          -- generate_text / generate_fn
    #3  (Sec 4.0, probes.py)        -- chat_prompt
    #4  (Sec 4.1, probes.py)        -- make_arithmetic_probe / eval_arithmetic / ARITH_PROBES
    #5  (Sec 4.2, probes.py)        -- sequence_logprob / eval_mc_probe / TINY_MC_SET
    #6  (Sec 4.3, probes.py)        -- normalize_answer / eval_retrieval_qa  [BONUS: given
                                        classification said "fragment"/SKIP, but on inspection
                                        it only needs a `.search(query,k)` retriever, which is a
                                        trivially-buildable toy fixture -- tested for real]
    #7  (Sec 4.4, agent.py)         -- AgentTask / eval_agent
    #8  (Sec 6.2)                   -- gptq_quantize_column_by_column
    #10 (Sec 7.1, quantize.py)      -- quantize_int8_per_row / quantize_int4_grouped + dequant
    #12 (Sec 7.3, quantize.py)      -- _swap_linears / quantize_stacklm / build_quantized_shell
                                        / state_dict_bytes
    #13 (Sec 7.3, REPL doctest)     -- tie-collapse acceptance test (adapted to toy sizes)
    #14 (Sec 7.4, quantize.py)      -- export_quantized / load_quantized  (safetensors, guarded)
    #15 (Sec 7.4, export.py)        -- the export CLI's main()
    #16 (Sec 8,   sweep.py)         -- evaluate_quantization_sweep / print_sweep
    #17 (Sec 9,   cli.py)           -- the serving CLI's main()

Skipped blocks:
    #1  SKIP(fragment): cache-equivalence snippet under the "Re-run the cache-
        equivalence test" warning -- uses undefined `tiny_model`/`idx` at
        module scope, an inline illustration, not a standalone unit.
    #2  SKIP(needs-gpu / non-hermetic): compute_perplexity (ppl.py) consumes
        Ch. 14.2's real `PackedMemmapDataset` over on-disk shards and a
        `torch.autocast(device_type="cuda", ...)` branch; per the assignment's
        own heuristic classification this is left untested here. Block #16
        (tested) still needs a `compute_perplexity` NAME to exist, so a
        clearly-labeled toy stand-in (NOT block #2's code) is defined just
        below the QA section so the sweep's own orchestration logic executes.
    #9  SKIP(fragment): _fake_quant_grouped / awq_search_channel_scales --
        self-contained but not referenced by any tested block, and not on the
        assigned tested list.
    #11 the QuantizedLinear/QuantizedEmbedding classes ("fragment" per the
        given classification) are copied verbatim and INCLUDED, because block
        #12 (quantize_stacklm, tested) cannot run without them -- this is the
        ordinary "later blocks depend on names defined earlier" concatenation
        the assignment describes, not a separate test entry.
    #18 SKIP(optional-dep torchao, not installed + illustrative pseudocode):
        the torchao snippet calls an undefined `load_fp32_stack100m()` in the
        chapter text itself. Guarded import; if torchao were present we would
        substitute our own toy model and run `quantize_(model,
        Int8WeightOnlyConfig())` for real (code path included, gated).
    #19 SKIP(optional-dep llmcompressor, not installed + illustrative
        pseudocode): references undefined `model`/`calibration_dataset` in the
        chapter text itself; a documentation sketch of the GPTQModifier API,
        not standalone-runnable regardless of the dependency.
    #20, #21 SKIP(shell): bash CLI invocations, not Python.
    #22 SKIP(fragment): Exercise 6's solution defines a *_symmetric variant of
        block #10's functions for a hand-worked exercise; not part of the
        chapter's main path and not depended on by any tested block.

No network access. Every third-party import beyond numpy/torch/stdlib
(safetensors, torchao, llmcompressor) is wrapped in try/except so the module
loads even where the package is absent; the corresponding block degrades to a
guarded, explicit SKIP print rather than crashing the whole file.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import re
import resource
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import save_file, load_file
except Exception:
    save_file = load_file = None

try:
    import torchao  # noqa: F401
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    _HAS_TORCHAO = True
except Exception:
    _HAS_TORCHAO = False

try:
    import llmcompressor  # noqa: F401
    _HAS_LLMCOMPRESSOR = True
except Exception:
    _HAS_LLMCOMPRESSOR = False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

print(f"[setup] safetensors available: {save_file is not None}")
print(f"[setup] torchao available: {_HAS_TORCHAO}")
print(f"[setup] llmcompressor available: {_HAS_LLMCOMPRESSOR}")


# =====================================================================
# GLUE: a tiny Stack100M-shaped model and a byte-level tokenizer.
#
# Ch. 14.4's real Stack100M (KV cache, RoPE, GQA, QK-norm, NoPE layers) and
# Ch. 14.3's real BPE StackTokenizer live in OTHER chapters. This chapter
# (14.11) only *consumes* their API surface -- forward(idx, targets=None,
# position_ids=None, seq_ids=None, kv_cache=None, start_pos=0,
# logits_to_keep=0), generate(...), .cfg, .blocks[i].attn.{wq,wk,wv,wo},
# .blocks[i].mlp.{gate,up,down}, tied .tok_emb/.lm_head -- so this stand-in
# reproduces exactly that surface at toy scale (real matmuls, real causal +
# document masking, real GQA head-sharing, real weight tying by aliasing) so
# every block UNDER TEST in this chapter runs against real tensors rather
# than mocks. Group size 8 (not the book's 64) is used throughout so int4
# quantization's group-tiling assertion is satisfiable at this toy width.
# =====================================================================

SPECIAL = {                                    # stand-in for Ch. 14.9's SPECIAL
    "bos": "<|bos|>", "system": "<|system|>", "user": "<|user|>",
    "assistant": "<|assistant|>", "end": "<|end|>",
}
PAD_TOKEN = "<|pad|>"
SPECIAL_TOKENS = list(SPECIAL.values()) + [PAD_TOKEN]     # stand-in for Ch. 14.3's SPECIAL_TOKENS
_SPECIAL_TO_ID = {s: 256 + i for i, s in enumerate(SPECIAL_TOKENS)}
_ID_TO_SPECIAL = {v: k for k, v in _SPECIAL_TO_ID.items()}
VOCAB_SIZE = 256 + len(SPECIAL_TOKENS)


class StackTokenizer:
    """Byte-level stand-in for Ch. 14.3's BPE tokenizer: base ids 0-255 are
    raw UTF-8 bytes (so encode(prompt) is always an exact prefix of
    encode(prompt+continuation) -- no merge can straddle the join), plus one
    reserved id per SPECIAL_TOKENS string."""
    vocab_size = VOCAB_SIZE
    bos_id = _SPECIAL_TO_ID[SPECIAL["bos"]]
    eos_id = _SPECIAL_TO_ID[SPECIAL["end"]]
    pad_id = _SPECIAL_TO_ID[PAD_TOKEN]

    def encode(self, text: str, allowed_special=frozenset()) -> list:
        ids, pos = [], 0
        if allowed_special:
            pat = re.compile("|".join(re.escape(s) for s in allowed_special))
            for m in pat.finditer(text):
                if m.start() > pos:
                    ids.extend(text[pos:m.start()].encode("utf-8"))
                ids.append(_SPECIAL_TO_ID[m.group()])
                pos = m.end()
        if pos < len(text):
            ids.extend(text[pos:].encode("utf-8"))
        return ids

    def decode(self, ids) -> str:
        parts, buf = [], bytearray()
        for i in ids:
            if i in _ID_TO_SPECIAL:
                if buf:
                    parts.append(bytes(buf).decode("utf-8", errors="ignore"))
                    buf.clear()
                parts.append(_ID_TO_SPECIAL[i])
            else:
                buf.append(int(i))
        if buf:
            parts.append(bytes(buf).decode("utf-8", errors="ignore"))
        return "".join(parts)

    @classmethod
    def load(cls, path: str):     # Section 9's cli.py calls StackTokenizer.load(...)
        return cls()


@dataclass
class StackConfig:
    """Toy stand-in for Ch. 14.4's StackConfig: same field names the chapter's
    code reaches for (vocab_size, d_model, n_heads, n_kv_heads, head_dim,
    intermediate, tie_embeddings, loss_chunk, max_seq_len, rope_theta), toy
    magnitudes."""
    vocab_size: int = VOCAB_SIZE
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 16
    intermediate: int = 128
    n_layers: int = 2
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    loss_chunk: int = 0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class ToyAttention(nn.Module):
    """GQA attention with the exact `wq/wk/wv/wo` nn.Linear names Section 7.3
    walks and re-wires."""

    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, mask):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class ToyMLP(nn.Module):
    """SwiGLU MLP with the exact `gate/up/down` nn.Linear names Section 7.3
    walks and re-wires."""

    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ToyBlock(nn.Module):
    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = ToyAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = ToyMLP(cfg)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Stack100M(nn.Module):
    """Toy stand-in for Ch. 14.4's Stack100M: real causal + document-aware
    masking, real GQA head-sharing, real weight tying (lm_head.weight IS
    tok_emb.weight, by aliasing -- exactly the property Section 7.3's tie
    detection and Section 7.2's QuantizedEmbedding exist to handle
    correctly), and the exact forward()/generate() signatures this chapter's
    code calls."""

    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([ToyBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None, position_ids=None, seq_ids=None,
                kv_cache=None, start_pos=0, logits_to_keep=0):
        B, T = idx.shape
        x = self.tok_emb(idx)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        if seq_ids is not None:
            doc_diff = seq_ids.unsqueeze(-1) != seq_ids.unsqueeze(-2)   # (B,T,T)
            mask = causal.unsqueeze(0) | doc_diff
        else:
            mask = causal.unsqueeze(0).expand(B, T, T)
        mask = mask.unsqueeze(1)                                        # (B,1,T,T)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.final_norm(x)
        if logits_to_keep:
            x = x[:, -logits_to_keep:, :]
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_p=1.0, top_k=0,
                eos_id=None, use_cache=True):
        idx = idx.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(idx_cond, logits_to_keep=1)
            logits = logits[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx

    def rebuild_rope(self, max_seq_len, rope_theta, device="cpu"):
        pass    # no-op: this toy model has no persisted RoPE buffers to rebuild

    def estimate_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


GROUP_SIZE = 32   # toy-scale stand-in for the book's group_size=64: large enough
                  # relative to d_model=64 that int4's fp32 scale/zero-point
                  # overhead doesn't swamp the packed savings (Section 7's own
                  # point about group-size overhead, at toy scale)

print("[glue OK] StackTokenizer + toy Stack100M (GQA, tied embeddings) constructed.\n")


# =====================================================================
# Block #0 (Sec 2, chapter: stacklm/infer.py) -- verbatim
# =====================================================================

ALL_SPECIALS = frozenset(SPECIAL_TOKENS)


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 64,
                  temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
                  stop_id: int | None = None, allowed_special=ALL_SPECIALS,
                  device: str = "cpu", return_n_tokens: bool = False):
    """Encode -> Ch. 14.4's cached generate() -> decode ONLY the new tokens."""
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
    """The exact signature the Section 4 probes expect. Greedy by default."""
    return generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                         temperature=temperature, **kw)


# --- exercise block #0 ------------------------------------------------------
_tok = StackTokenizer()
_model = Stack100M(StackConfig())
_model.eval()

_txt, _n = generate_text(_model, _tok, "The mitochondria is", max_new_tokens=6,
                         return_n_tokens=True)
assert isinstance(_txt, str) and 0 <= _n <= 6
_txt2 = generate_fn(_model, _tok, "hello there", max_new_tokens=4)
assert isinstance(_txt2, str)
# logits_to_keep discipline (Section 2's warning): default 0 gives (B,T,V), a
# sampling call's logits_to_keep=1 gives (B,1,V).
_ids = torch.tensor([_tok.encode("abc")])
_logits_all, _ = _model(_ids)
assert _logits_all.shape == (1, 3, _tok.vocab_size)
_logits_last, _ = _model(_ids, logits_to_keep=1)
assert _logits_last.shape == (1, 1, _tok.vocab_size)
# generate() raises rather than silently truncating on a too-long prompt+budget
try:
    generate_text(_model, _tok, "x" * 10, max_new_tokens=10**9)
    raise AssertionError("expected the context-length assertion to fire")
except AssertionError as e:
    assert "exceeds the context" in str(e)

print("[block #0 OK] generate_text / generate_fn: encode -> generate -> decode, "
      "logits_to_keep=0 vs 1 shapes, context-length guard.\n")


# =====================================================================
# Block #3 (Sec 4.0, chapter: stacklm/eval/probes.py) -- verbatim
# =====================================================================


def chat_prompt(user: str, system: str | None = None) -> str:
    """Byte-identical to `render_conversation(..., add_generation_prompt=True)`:
    <|bos|>[<|system|>SYS<|end|>]<|user|>USER<|end|><|assistant|>"""
    s = SPECIAL["bos"]
    if system is not None:
        s += f"{SPECIAL['system']}{system}{SPECIAL['end']}"
    return s + f"{SPECIAL['user']}{user}{SPECIAL['end']}{SPECIAL['assistant']}"


# --- exercise block #3 ------------------------------------------------------
_p_no_sys = chat_prompt("hello")
assert _p_no_sys == f"{SPECIAL['bos']}{SPECIAL['user']}hello{SPECIAL['end']}{SPECIAL['assistant']}"
_p_sys = chat_prompt("hello", system="be terse")
assert _p_sys.startswith(SPECIAL["bos"])
assert SPECIAL["system"] in _p_sys and "be terse" in _p_sys
assert _p_sys.endswith(SPECIAL["assistant"])
assert "\n" not in _p_sys and "\n" not in _p_no_sys, "no newlines between turns"
# round-trips through the tokenizer's special-token path without forging a role
_enc = _tok.encode(_p_sys, allowed_special=ALL_SPECIALS)
assert _SPECIAL_TO_ID[SPECIAL["bos"]] in _enc and _SPECIAL_TO_ID[SPECIAL["assistant"]] in _enc

print("[block #3 OK] chat_prompt: byte-identical frame, no injected newlines.\n")


# =====================================================================
# Block #4 (Sec 4.1, chapter: stacklm/eval/probes.py) -- verbatim logic.
# `make_arithmetic_prompt`/`exact_match_reward` stand in for Ch. 14.9's RLVR
# task + verifier (same "Compute a op b. ... #### <int>" surface form).
# =====================================================================


def make_arithmetic_prompt(rng: random.Random, max_val: int = 99):
    a, b = rng.randint(0, max_val), rng.randint(0, max_val)
    op = rng.choice(["+", "-", "*"])
    ans = {"+": a + b, "-": a - b, "*": a * b}[op]
    q = f"Compute {a} {op} {b}. Give the final integer after '####'."
    return q, str(ans)


def exact_match_reward(completion: str, gold: str):
    m = re.search(r"####\s*(-?\d+)", completion)
    if not m:
        return 0.0, None
    pred = m.group(1)
    return (1.0 if pred == gold else 0.0), pred


def make_arithmetic_probe(n: int = 200, seed: int = 0, max_val: int = 99) -> list:
    rng = random.Random(seed)
    return [dict(zip(("question", "answer"), make_arithmetic_prompt(rng, max_val=max_val)))
            for _ in range(n)]


@torch.no_grad()
def eval_arithmetic(model, tokenizer, generate_fn, problems: list,
                    max_new_tokens: int = 64, label: str = "in-distribution") -> dict:
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


ARITH_PROBES = [
    (make_arithmetic_probe(n=200, seed=0, max_val=99),  "2-digit (in-distribution)"),
    (make_arithmetic_probe(n=100, seed=1, max_val=999), "3-digit (out-of-distribution)"),
]

# --- exercise block #4 -------------------------------------------------------
assert len(ARITH_PROBES[0][0]) == 200 and len(ARITH_PROBES[1][0]) == 100
assert ARITH_PROBES[0][0] == make_arithmetic_probe(n=200, seed=0, max_val=99), \
    "seeded probe generation must be reproducible from the seed alone"
# Grade a hand-checkable completion directly against the book's own verifier.
assert exact_match_reward("work work #### 45", "45") == (1.0, "45")
assert exact_match_reward("no answer line here", "45") == (0.0, None)
assert exact_match_reward("#### 44", "45") == (0.0, "44")

# Actually drive the (untrained) toy model through the harness -- a tiny slice
# of the real ARITH_PROBES set, small enough to stay well under budget.
_arith_result = eval_arithmetic(_model, _tok, generate_fn, ARITH_PROBES[0][0][:3],
                                max_new_tokens=8, label=ARITH_PROBES[0][1])
assert set(_arith_result) == {"label", "accuracy", "parse_rate", "n"}
assert _arith_result["n"] == 3 and 0.0 <= _arith_result["accuracy"] <= 1.0
assert 0.0 <= _arith_result["parse_rate"] <= 1.0
# Parse-failure rate and accuracy are reported SEPARATELY (Section 4.1's rule).
assert _arith_result["accuracy"] <= _arith_result["parse_rate"] + 1e-9

print(f"[block #4 OK] arithmetic probe on the toy model: {_arith_result}\n")


# =====================================================================
# Block #5 (Sec 4.2, chapter: stacklm/eval/probes.py) -- verbatim
# =====================================================================


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, continuation: str,
                     device="cpu") -> tuple:
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

    targets = ids[0, n_ctx:]                                 # (n_cont,)
    pred_rows = logp[n_ctx - 1: -1, :]                       # (n_cont, V)
    tok_logp = pred_rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(tok_logp.sum()), int(tok_logp.numel())


TINY_MC_SET = [
    {"question": "The chemical symbol for water is",
     "choices": [" H2O", " CO2", " NaCl", " O2"], "answer_idx": 0},
    {"question": "The capital of France is",
     "choices": [" Paris", " Berlin", " Madrid", " Rome"], "answer_idx": 0},
]


def eval_mc_probe(model, tokenizer, mc_set: list = TINY_MC_SET, device="cpu") -> dict:
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


# --- exercise block #5 -------------------------------------------------------
_lp, _n_cont = sequence_logprob(_model, _tok, "The capital of France is", " Paris")
assert isinstance(_lp, float) and _n_cont == len(_tok.encode(" Paris"))
assert _lp <= 0.0 + 1e-6, "a sum of log-probabilities must be <= 0"
# BPE-boundary guard actually fires when a caller lies about the tokenizer.
try:
    sequence_logprob(_model, _tok, "The capital of France is", "")
    raise AssertionError("expected empty-continuation ValueError")
except ValueError as e:
    assert "empty continuation" in str(e)

_mc = eval_mc_probe(_model, _tok, TINY_MC_SET)
assert set(_mc) == {"acc", "acc_norm", "n"} and _mc["n"] == 2
assert 0.0 <= _mc["acc"] <= 1.0 and 0.0 <= _mc["acc_norm"] <= 1.0

print(f"[block #5 OK] sequence_logprob + cloze MC scoring on the toy model: {_mc}\n")


# =====================================================================
# Block #6 (Sec 4.3, chapter: stacklm/eval/probes.py) -- verbatim.
# BONUS block: the given classification defaulted this to SKIP(fragment)
# because it needs Ch. 14.10's BM25Retriever/HashEmbedRetriever, but the only
# real dependency is a `.search(query, k) -> list[(Passage, score)]` object,
# which is a trivially-buildable toy fixture -- so it is tested for real,
# and reused by block #16's sweep below.
# =====================================================================


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def eval_retrieval_qa(model, tokenizer, generate_fn, retriever, qa_pairs: list) -> dict:
    n_correct, n_retrieved = 0, 0
    for pair in qa_pairs:
        hits = retriever.search(pair["question"], k=1)
        passage_text = hits[0][0].text if hits else ""        # (Passage, score) -> .text
        n_retrieved += int(normalize_answer(pair["gold_answer"])
                           in normalize_answer(passage_text))
        prompt = chat_prompt(f"Passage: {passage_text}\nQuestion: {pair['question']}",
                             system="Answer using only the passage below.")
        completion = generate_fn(model, tokenizer, prompt=prompt, max_new_tokens=6,
                                 temperature=0.0)
        n_correct += int(normalize_answer(completion) == normalize_answer(pair["gold_answer"]))
    n = len(qa_pairs)
    return {"exact_match": n_correct / n, "retriever_recall@1": n_retrieved / n, "n": n}


class Passage:
    def __init__(self, doc_id: str, text: str):
        self.doc_id, self.text = doc_id, text


class ToyRetriever:
    """Substring-match stand-in for Ch. 14.10's BM25Retriever/HashEmbedRetriever.
    `.search` returns exactly the (Passage, score) pair shape Section 4.3 warns
    a caller must not flatten into the prompt as a Python repr."""

    def __init__(self, passages: list):
        self.passages = passages

    def search(self, query: str, k: int = 1):
        scored = sorted(self.passages,
                        key=lambda p: -sum(w in p.text.lower() for w in query.lower().split()))
        return [(p, 1.0) for p in scored[:k]]


_QA_PASSAGES = [Passage("d0", "Paris is the capital of France."),
               Passage("d1", "The mitochondria is the powerhouse of the cell.")]
_QA_PAIRS = [{"question": "What is the capital of France?", "gold_answer": "Paris"},
            {"question": "What is the powerhouse of the cell?", "gold_answer": "mitochondria"}]
_retriever = ToyRetriever(_QA_PASSAGES)

# --- exercise block #6 -------------------------------------------------------
assert normalize_answer("The Paris.") == "paris"
_hits = _retriever.search("capital of France", k=1)
assert _hits[0][0].text.startswith("Paris") and _hits[0][1] == 1.0
_qa = eval_retrieval_qa(_model, _tok, generate_fn, _retriever, _QA_PAIRS)
assert set(_qa) == {"exact_match", "retriever_recall@1", "n"} and _qa["n"] == 2
assert _qa["retriever_recall@1"] == 1.0, "both gold answers literally appear in the top-1 passage"
assert 0.0 <= _qa["exact_match"] <= 1.0

print(f"[block #6 OK] retrieval-QA EM + retriever recall@1, decomposed: {_qa}\n")


# =====================================================================
# GLUE: a toy stand-in for the SKIPPED block #2 (compute_perplexity, ppl.py).
# NOT block #2's code -- block #2 needs Ch. 14.2's real on-disk
# PackedMemmapDataset and is left SKIPPED, per the assignment's own
# classification ("needs-gpu"). This exists only so block #16 (tested,
# below) has a `compute_perplexity` name to call, and it still runs a REAL
# forward pass + REAL cross-entropy with logits_to_keep=0 (Section 2's rule),
# matching the book function's *shape*, not its data-loading internals.
# =====================================================================


def compute_perplexity(model, val_shard_dir, pad_id, batch_size: int = 4,
                       max_batches: int | None = None, device: str = "cpu") -> dict:
    model.to(device).eval()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits, _ = model(ids.to(device))                 # logits_to_keep=0 default -> (B,T,V)
    assert logits.shape[1] == ids.shape[1], "logits_to_keep leaked in"
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), ids.reshape(-1))
    return {"loss_nats_per_token": float(loss),
            "perplexity": float(torch.exp(loss)),
            "n_tokens_evaluated": int(ids.numel())}


print("[glue OK] toy compute_perplexity stand-in defined for block #2 (SKIPPED itself).\n")


# =====================================================================
# Block #7 (Sec 4.4, chapter: stacklm/eval/agent.py) -- verbatim.
# `parse_assistant_step`/`run_agent`/`ToolEnv`/`normalize` stand in for
# Ch. 14.10's real ReAct loop; `run_agent` is a deterministic, seeded-by-
# task-order scripted loop that actually drives the toy model once per task
# (via generate_text) AND deliberately walks eval_agent's five taxonomy
# branches (solved / wrong_tool / malformed_call / non_termination), so the
# THIS CHAPTER's decomposition logic in eval_agent is genuinely exercised
# rather than mocked away.
# =====================================================================


@dataclass
class ParsedStep:
    kind: str            # "tool" | "final"
    tool: str = ""
    arg: str = ""
    answer: str = ""


_FINAL_RE = re.compile(r"Answer:\s*(.*)", re.S)
_TOOL_RE = re.compile(r"Action:\s*(\w+)\((.*?)\)", re.S)


def parse_assistant_step(text: str) -> ParsedStep:
    m = _FINAL_RE.search(text)
    if m:
        return ParsedStep(kind="final", answer=m.group(1).strip())
    m = _TOOL_RE.search(text)
    if m:
        return ParsedStep(kind="tool", tool=m.group(1), arg=m.group(2))
    return ParsedStep(kind="tool", tool="__malformed__")


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class ToolEnv:
    """Toy stand-in for Ch. 14.10's ToolEnv: a `search`/`calc` tool surface."""

    def __init__(self):
        self.calls = 0
        self._pattern = itertools.cycle(["solved", "wrong_tool", "malformed", "capped"])

    def call(self, tool: str, arg: str) -> str:
        self.calls += 1
        if tool == "calc":
            return "Observation: result is 42"
        return f"Observation: passage about {arg} found"


def run_agent(model, tok, question: str, env: ToolEnv, max_steps: int = 6):
    """Toy stand-in for Ch. 14.10's real ReAct run_agent. Actually drives the
    model once (touching the real forward/generate path this chapter owns),
    then walks a deterministic outcome cycle so eval_agent's full taxonomy is
    exercised."""
    _ = generate_text(model, tok, chat_prompt(question), max_new_tokens=2)
    outcome = next(env._pattern)
    trace = []
    if outcome == "malformed":
        trace.append(("assistant", "gibberish with no Action or Answer"))
        trace.append(("assistant", "Thought: I must answer now.\nAnswer: unknown"))
        return "unknown", trace
    if outcome == "capped":
        for _ in range(max_steps):
            trace.append(("assistant", "Action: search(more)"))
            trace.append(("observation", env.call("search", "more")))
        trace.append(("assistant", "Thought: I must answer now.\nAnswer: unknown"))
        return "unknown", trace
    tool = "calc" if "calc" in question.lower() else "search"
    if outcome == "wrong_tool":
        tool = "search" if tool == "calc" else "calc"
    trace.append(("assistant", f"Action: {tool}({question})"))
    trace.append(("observation", env.call(tool, question)))
    if outcome == "solved":
        trace.append(("assistant", "Answer: 42"))
        return "42", trace
    trace.append(("assistant", "Answer: unsure"))
    return "unsure", trace


@dataclass
class AgentTask:
    question: str
    gold: str
    gold_tool: str
    gold_evidence: str = ""


def eval_agent(model, tok, env: ToolEnv, tasks: list, max_steps: int = 6) -> dict:
    n_calls = n_parseable = n_tool_right = n_solved = n_capped = 0
    turns: list = []
    taxonomy: Counter = Counter()

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

        capped = len(assistant) > max_steps
        n_capped += int(capped)
        turns.append(max_steps if capped else len(assistant))

        solved = normalize(answer) == normalize(task.gold)
        n_solved += int(solved)

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


# --- exercise block #7 -------------------------------------------------------
_agent_env = ToolEnv()
_agent_tasks = [
    AgentTask(question="calc 6 times 7", gold="42", gold_tool="calc"),   # -> "solved"
    AgentTask(question="search population", gold="99", gold_tool="search"),  # -> "wrong_tool"
    AgentTask(question="a hard question", gold="7", gold_tool="search"),  # -> "malformed"
    AgentTask(question="another question", gold="7", gold_tool="search"),  # -> "capped"
]
_agent_result = eval_agent(_model, _tok, _agent_env, _agent_tasks, max_steps=6)
assert _agent_result["n_tasks"] == 4
assert _agent_result["taxonomy"] == {"ok": 1, "wrong_tool": 1, "malformed_call": 1,
                                      "non_termination": 1}, _agent_result["taxonomy"]
assert _agent_result["exact_match"] == 0.25
assert _agent_result["cap_rate"] == 0.25
assert abs(_agent_result["turns_mean"] - 3.0) < 1e-9 and _agent_result["turns_median"] == 2.0
# tool_format_validity: 9 total calls (1+1+1+6), 8 parseable (all but the
# single deliberately malformed one).
assert abs(_agent_result["tool_format_validity"] - 8 / 9) < 1e-9
# cap_rate is read off the TRACE SHAPE, not "was the last action non-final"
# (Section 4.4's dead-code warning) -- confirm the forced-synthesis emission
# on the capped task really does parse as `final`, so the naive detector
# really would have missed it.
_capped_trace_last = parse_assistant_step("Thought: I must answer now.\nAnswer: unknown")
assert _capped_trace_last.kind == "final", "the forced-synthesis fallback must parse as final"

print(f"[block #7 OK] eval_agent decomposition, exact taxonomy: {_agent_result['taxonomy']}\n")


# =====================================================================
# Block #8 (Sec 6.2) -- verbatim
# =====================================================================


def gptq_quantize_column_by_column(W: torch.Tensor, X_calib: torch.Tensor, bits: int = 4,
                                   damp: float = 1e-2) -> torch.Tensor:
    d_out, d_in = W.shape
    W = W.clone().float()

    H = 2 * (X_calib.T @ X_calib) / X_calib.shape[0]
    H += damp * torch.eye(d_in, device=W.device)
    H_inv = torch.linalg.inv(H)

    qmax = 2 ** (bits - 1) - 1
    for q in range(d_in):
        col = W[:, q]
        scale = col.abs().max() / qmax if col.abs().max() > 0 else 1.0
        q_col = torch.clamp(torch.round(col / scale), -qmax, qmax)
        w_hat = q_col * scale
        error = (col - w_hat) / H_inv[q, q]
        W[:, q] = w_hat
        if q + 1 < d_in:
            W[:, q + 1:] -= torch.outer(error, H_inv[q, q + 1:])
    return W


# --- exercise block #8 -------------------------------------------------------
torch.manual_seed(1)
_W = torch.randn(6, 10)
_X_calib = torch.randn(32, 10)
_W_gptq = gptq_quantize_column_by_column(_W, _X_calib, bits=4)
assert _W_gptq.shape == _W.shape
# GPTQ optimizes RECONSTRUCTION error on real calibration activations, not
# raw weight-space error -- the property worth checking is that it beats a
# plain RTN quantization of the SAME matrix on the layer's actual output.
_qmax4 = 2 ** 3 - 1
_scale_rtn = _W.abs().amax(dim=1, keepdim=True) / _qmax4
_W_rtn = torch.round(_W / _scale_rtn).clamp(-_qmax4, _qmax4) * _scale_rtn
_err_gptq = (_X_calib @ _W_gptq.T - _X_calib @ _W.T).pow(2).mean().item()
_err_rtn = (_X_calib @ _W_rtn.T - _X_calib @ _W.T).pow(2).mean().item()
assert _err_gptq <= _err_rtn * 1.05, (
    f"GPTQ's reconstruction-aware quantization should not be meaningfully "
    f"worse than plain RTN on the calibration set: gptq={_err_gptq:.4f} rtn={_err_rtn:.4f}")

print(f"[block #8 OK] gptq_quantize_column_by_column: output-error "
      f"gptq={_err_gptq:.4f} <= rtn={_err_rtn:.4f}.\n")


# =====================================================================
# Block #10 (Sec 7.1, chapter: stacklm/serve/quantize.py) -- verbatim
# =====================================================================


def quantize_int8_per_row(weight: torch.Tensor):
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127)
    return q.to(torch.int8), scale


def dequantize_int8_per_row(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.unsqueeze(1)


def quantize_int4_grouped(weight: torch.Tensor, group_size: int = 64):
    d_out, d_in = weight.shape
    assert d_in % group_size == 0, f"d_in={d_in} not divisible by group_size={group_size}"
    n_groups = d_in // group_size

    w = weight.float().view(d_out, n_groups, group_size)
    w_min, w_max = w.amin(dim=2), w.amax(dim=2)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zero_point = torch.round(-w_min / scale)

    q = torch.round(w / scale.unsqueeze(2) + zero_point.unsqueeze(2)).clamp(0, 15)
    q = q.view(d_out, d_in).to(torch.uint8)

    q_even, q_odd = q[:, 0::2], q[:, 1::2]
    packed = (q_even | (q_odd << 4)).to(torch.uint8)
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


# --- exercise block #10 ------------------------------------------------------
torch.manual_seed(2)
_Wt = torch.randn(6, GROUP_SIZE * 3) * 0.1

_q8, _s8 = quantize_int8_per_row(_Wt)
assert _q8.dtype == torch.int8 and _q8.shape == _Wt.shape and _s8.shape == (6,)
_Wt_hat8 = dequantize_int8_per_row(_q8, _s8)
_err8 = (_Wt - _Wt_hat8).abs().max().item()

_packed4, _scale4, _zp4 = quantize_int4_grouped(_Wt, group_size=GROUP_SIZE)
assert _packed4.dtype == torch.uint8 and _packed4.shape == (6, _Wt.shape[1] // 2)
assert _scale4.shape == _zp4.shape == (6, 3)
_Wt_hat4 = dequantize_int4_grouped(_packed4, _scale4, _zp4, _Wt.shape[1], group_size=GROUP_SIZE)
_err4 = (_Wt - _Wt_hat4).abs().max().item()

assert _err8 < 0.01, f"int8 per-row RTN should reconstruct tightly, got max err {_err8}"
assert _err4 < 0.05, f"int4 grouped RTN should reconstruct reasonably, got max err {_err4}"
assert _err8 < _err4, "8 bits must reconstruct at least as tightly as 4 bits"
# d_in not divisible by group_size must raise, not silently misalign.
try:
    quantize_int4_grouped(torch.randn(2, 10), group_size=GROUP_SIZE)
    raise AssertionError("expected an assertion on non-tiling group_size")
except AssertionError as e:
    assert "not divisible" in str(e)

print(f"[block #10 OK] int8 row-wise (max err {_err8:.4f}) and int4 grouped "
      f"(max err {_err4:.4f}) RTN round-trips.\n")


# =====================================================================
# Block #11 (chapter: stacklm/serve/quantize.py) -- verbatim.
# Given classification: "fragment"/SKIP as a standalone test entry, but
# copied in full here because block #12 (quantize_stacklm, TESTED) cannot
# run without these two classes -- the ordinary "later blocks depend on
# names defined earlier" concatenation the assignment describes.
# =====================================================================


class QuantizedLinear(nn.Module):
    def __init__(self, bits: int, group_size: int = 64):
        super().__init__()
        assert bits in (4, 8)
        self.bits, self.group_size = bits, group_size
        self.d_in = self.d_out = None

    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int, group_size: int = 64):
        layer = cls(bits, group_size)
        layer.d_in, layer.d_out = linear.in_features, linear.out_features
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

    def dequantize(self) -> torch.Tensor:
        if self.bits == 8:
            return dequantize_int8_per_row(self.q_weight, self.scale)
        return dequantize_int4_grouped(self.q_weight, self.scale, self.zero_point,
                                       self.d_in, self.group_size)

    def dequantize_rows(self, idx: torch.Tensor) -> torch.Tensor:
        if self.bits == 8:
            return self.q_weight[idx].float() * self.scale[idx].unsqueeze(1)
        return dequantize_int4_grouped(self.q_weight[idx], self.scale[idx],
                                       self.zero_point[idx], self.d_in, self.group_size)

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize()
        return torch.nn.functional.linear(x, w.to(x.dtype), self.bias)


class QuantizedEmbedding(nn.Module):
    def __init__(self, head: QuantizedLinear):
        super().__init__()
        object.__setattr__(self, "_head", head)

    @property
    def weight(self) -> torch.Tensor:
        return self._head.dequantize()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        rows = self._head.dequantize_rows(idx.reshape(-1))
        return rows.view(*idx.shape, -1)


print("[block #11 included as glue] QuantizedLinear / QuantizedEmbedding defined "
      "(exercised through block #12 below).\n")


# =====================================================================
# Block #12 (Sec 7.3, chapter: stacklm/serve/quantize.py) -- verbatim, with
# the chapter's `from stacklm.model.transformer import Stack100M` dropped
# (Stack100M is already defined above in this file).
# =====================================================================


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
    tied = (getattr(model, "cfg", None) is not None
            and model.cfg.tie_embeddings
            and model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr())

    head_linear = model.lm_head if tied else None
    if tied and embedding_bits is not None and embedding_bits != bits:
        model.lm_head = QuantizedLinear.from_float(head_linear, embedding_bits, group_size)

    _swap_linears(model, bits, group_size)

    if tied:
        model.tok_emb = QuantizedEmbedding(model.lm_head)

    if getattr(model, "cfg", None) is not None and model.cfg.loss_chunk:
        model.cfg.loss_chunk = 0
    return model


def build_quantized_shell(cfg, bits: int, group_size: int = 64,
                          embedding_bits: int | None = None):
    # (dropped: `from stacklm.model.transformer import Stack100M` -- already defined above)
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
    seen, total = set(), 0
    for t in model.state_dict().values():
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total / 1e6


# --- exercise block #12 ------------------------------------------------------
_cfg = StackConfig()
_fp32_model = Stack100M(_cfg)
_fp32_model.eval()
_fp32_bytes = state_dict_bytes(_fp32_model)

_m8 = quantize_stacklm(copy.deepcopy(_fp32_model), bits=8, group_size=GROUP_SIZE)
_bytes8 = state_dict_bytes(_m8)
_m4 = quantize_stacklm(copy.deepcopy(_fp32_model), bits=4, group_size=GROUP_SIZE)
_bytes4 = state_dict_bytes(_m4)
_m4e8 = quantize_stacklm(copy.deepcopy(_fp32_model), bits=4, group_size=GROUP_SIZE, embedding_bits=8)
_bytes4e8 = state_dict_bytes(_m4e8)

assert _bytes8 < _fp32_bytes, "int8 must be smaller than fp32"
assert _bytes4 < _bytes8, "int4 must be smaller than int8"
# The mixed-precision embedding path: keeping the tied table at 8 bits while
# the body goes to 4 costs MORE than pure int4, less than pure int8.
assert _bytes4 < _bytes4e8 < _bytes8, (_bytes4, _bytes4e8, _bytes8)
# Tie handling: after quantization every module is either QuantizedLinear or
# QuantizedEmbedding, and forward() still runs end to end.
assert isinstance(_m4.lm_head, QuantizedLinear) and isinstance(_m4.tok_emb, QuantizedEmbedding)
_ids4 = torch.tensor([[1, 2, 3]])
_logits4, _ = _m4(_ids4)
assert _logits4.shape == (1, 3, _cfg.vocab_size)
# build_quantized_shell produces the SAME parameter tree shape as the real
# quantize_stacklm walk (needed by load_quantized/block #14 below).
_shell = build_quantized_shell(_cfg, bits=4, group_size=GROUP_SIZE, embedding_bits=8)
assert set(_shell.state_dict()) == set(_m4e8.state_dict()), (
    "build_quantized_shell's meta-device tree must match quantize_stacklm's "
    "real tree key-for-key, or load_state_dict(assign=True) cannot round-trip")

print(f"[block #12 OK] fp32={_fp32_bytes:.3f}MB int8={_bytes8:.3f}MB int4={_bytes4:.3f}MB "
      f"int4+int8-embed={_bytes4e8:.3f}MB; shell keys match quantized model keys.\n")


# =====================================================================
# Block #13 (Sec 7.3, chapter: REPL doctest) -- same assertions the chapter's
# `>>>` transcript demonstrates, adapted to this file's toy sizes instead of
# the book's literal 405.4/63.5 MB (which are specific to the real 101.3M-
# parameter Stack-100M and don't apply to a toy config).
# =====================================================================

_m13 = Stack100M(StackConfig())
_keys_before = [k for k in _m13.state_dict() if "tok_emb" in k or k == "lm_head.weight"]
assert _keys_before == ["tok_emb.weight", "lm_head.weight"], _keys_before   # aliased, BOTH keys exist
_bytes_before = state_dict_bytes(_m13)
_expected_fp32 = sum(p.numel() for p in _m13.parameters()) * 4 / 1e6        # every param is fp32
# tok_emb.weight and lm_head.weight are the SAME storage: state_dict_bytes
# must count it ONCE, not twice.
assert abs(_bytes_before - _expected_fp32) < 1e-4, (_bytes_before, _expected_fp32)

_ = quantize_stacklm(_m13, bits=4, group_size=GROUP_SIZE)
_keys_after = sorted(k for k in _m13.state_dict() if "tok_emb" in k or "lm_head" in k)
assert _keys_after == ["lm_head.q_weight", "lm_head.scale", "lm_head.zero_point"], _keys_after
_bytes_after = state_dict_bytes(_m13)
assert _bytes_after < _bytes_before, "the acceptance test: quantization must shrink the ACTUAL state_dict"

print(f"[block #13 OK] tie-collapse acceptance test: {_bytes_before:.4f}MB -> "
      f"{_bytes_after:.4f}MB, no fp32 table survives the tie.\n")


# =====================================================================
# Block #14 (Sec 7.4, chapter: stacklm/serve/quantize.py) -- verbatim, with
# `from safetensors.torch import save_file, load_file` and
# `from stacklm.config import StackConfig` dropped (both already available
# at module scope above: save_file/load_file guarded-imported, StackConfig
# defined above).
# =====================================================================


def export_quantized(model: nn.Module, path_prefix: str, bits: int, group_size: int,
                     config: dict, embedding_bits: int | None = None) -> None:
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
    # (dropped: `from stacklm.config import StackConfig` -- already defined above)
    with open(path_prefix + ".json") as f:
        meta = json.load(f)
    cfg = StackConfig(**meta["architecture"])
    model = build_quantized_shell(cfg, meta["bits"], meta["group_size"],
                                  meta.get("embedding_bits"))
    sd = load_file(path_prefix + ".safetensors", device=device)
    model.load_state_dict(sd, assign=True, strict=True)
    model.rebuild_rope(cfg.max_seq_len, cfg.rope_theta, device=device)
    model.eval()
    return model, meta


# --- exercise block #14 ------------------------------------------------------
if save_file is not None:
    _export_dir = tempfile.mkdtemp(prefix="stack100m_export_")
    _prefix = f"{_export_dir}/stack100m_int4"
    _model_to_export = quantize_stacklm(Stack100M(_cfg), bits=4, group_size=GROUP_SIZE,
                                        embedding_bits=8)
    _model_to_export.eval()
    export_quantized(_model_to_export, _prefix, bits=4, group_size=GROUP_SIZE,
                     config=asdict(_cfg), embedding_bits=8)
    assert os.path.exists(_prefix + ".safetensors") and os.path.exists(_prefix + ".json")

    _loaded_model, _meta = load_quantized(_prefix)
    assert _meta["bits"] == 4 and _meta["embedding_bits"] == 8
    assert isinstance(_loaded_model.lm_head, QuantizedLinear)
    _loaded_logits, _ = _loaded_model(torch.tensor([[1, 2, 3]]))
    assert _loaded_logits.shape == (1, 3, _cfg.vocab_size)
    # The exported artifact's on-disk byte accounting matches the in-memory one.
    _bytes_exported = state_dict_bytes(_model_to_export)
    _bytes_loaded = state_dict_bytes(_loaded_model)
    assert abs(_bytes_exported - _bytes_loaded) < 1e-6, (_bytes_exported, _bytes_loaded)
    print(f"[block #14 OK] export_quantized/load_quantized round-trip: "
          f"{_bytes_exported:.4f}MB, logits shape preserved.\n")
else:
    print("[block #14 SKIP(optional-dep)] safetensors not installed in this "
          "environment; export_quantized/load_quantized are defined but not called.\n")


# =====================================================================
# Block #15 (Sec 7.4, chapter: stacklm/serve/export.py) -- verbatim main(),
# invoked with argv pointing at a toy Ch. 14.7-shaped checkpoint instead of
# a real trained one.
# =====================================================================


def _export_main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/stack100m/post_final.pt")
    p.add_argument("--out_dir", default="checkpoints/stack100m")
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--embedding_bits", type=int, default=8)
    args = p.parse_args()

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


# --- exercise block #15 ------------------------------------------------------
if save_file is not None:
    _ckpt_dir = tempfile.mkdtemp(prefix="stack100m_ckpt_")
    _ckpt_path = f"{_ckpt_dir}/post_final.pt"
    torch.save({"model": _fp32_model.state_dict(), "config": asdict(_cfg), "step": 100},
              _ckpt_path)
    _out_dir = tempfile.mkdtemp(prefix="stack100m_out_")

    _old_argv = sys.argv
    try:
        sys.argv = ["export.py", "--ckpt", _ckpt_path, "--out_dir", _out_dir,
                   "--bits", "4", "--group_size", str(GROUP_SIZE), "--embedding_bits", "8"]
        _export_main()
    finally:
        sys.argv = _old_argv

    assert os.path.exists(f"{_out_dir}/stack100m_int4.safetensors")
    assert os.path.exists(f"{_out_dir}/stack100m_int4.json")
    _cli_loaded, _cli_meta = load_quantized(f"{_out_dir}/stack100m_int4")
    assert _cli_meta["bits"] == 4
    print(f"[block #15 OK] export.py main(): checkpoint -> {_out_dir}/stack100m_int4."
          f"{{safetensors,json}}\n")
else:
    print("[block #15 SKIP(optional-dep)] safetensors not installed; export.py's "
          "main() is defined but not invoked.\n")


# =====================================================================
# Block #16 (Sec 8, chapter: stacklm/eval/sweep.py) -- verbatim, with
# `from stacklm.serve.quantize import quantize_stacklm, state_dict_bytes`
# dropped (both already defined above).
# =====================================================================


def evaluate_quantization_sweep(fp32_model, tok, val_shard_dir, probes, *,
                                configs=((None, None, None), (8, None, None),
                                         (4, 128, None), (4, 64, None), (4, 64, 8)),
                                device="cpu", max_batches=20):
    rows = []
    for bits, gs, eb in configs:
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
            "bytes_mb": state_dict_bytes(m),
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
        print(f"{r['config']:<15}{r['bytes_mb']:>7.3f}{r['ppl']:>8.2f}{r['d_ppl']:>+8.2f}"
              f"{r['arith']:>8.1%}{r['mc']:>7.1%}{r['qa']:>7.1%}"
              f"{r['fmt']:>7.1%}{r['agent_em']:>7.1%}")


# --- exercise block #16 ------------------------------------------------------
_sweep_probes = {
    "arith": [(ARITH_PROBES[0][0][:2], ARITH_PROBES[0][1])],
    "mc": TINY_MC_SET,
    "qa": _QA_PAIRS,
    "retriever": _retriever,
    "env": _agent_env,
    "agent_tasks": _agent_tasks,
    "pad_id": _tok.pad_id,
}
_toy_configs = ((None, None, None), (8, None, None), (4, GROUP_SIZE, None), (4, GROUP_SIZE, 8))
_rows = evaluate_quantization_sweep(_fp32_model, _tok, "unused-toy-val-dir", _sweep_probes,
                                    configs=_toy_configs, max_batches=1)
assert len(_rows) == len(_toy_configs)
assert _rows[0]["config"] == "fp32" and _rows[0]["d_ppl"] == 0.0
for _r in _rows:
    assert {"config", "bytes_mb", "ppl", "arith", "mc", "qa", "fmt", "agent_em",
           "d_ppl", "d_arith", "d_mc", "d_qa", "d_fmt", "d_agent_em"} <= set(_r)
# The chapter's central empirical claim, checked structurally: quantized
# configs are strictly smaller than fp32, and int4 <= int8.
_bytes_by_config = {r["config"]: r["bytes_mb"] for r in _rows}
assert _bytes_by_config[f"int8/g64"] < _bytes_by_config["fp32"]
assert _bytes_by_config[f"int4/g{GROUP_SIZE}"] < _bytes_by_config[f"int8/g64"]
print_sweep(_rows)     # exercise the pretty-printer too

print(f"\n[block #16 OK] evaluate_quantization_sweep + print_sweep over "
      f"{len(_rows)} configs, sizes strictly decreasing with more aggressive PTQ.\n")


# =====================================================================
# Block #17 (Sec 9, chapter: stacklm/serve/cli.py) -- verbatim main(), run
# against the artifact block #15 exported (guarded on safetensors).
# =====================================================================


def _cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/stack100m")
    p.add_argument("--prompt", type=str, default="The mitochondria is")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(args.threads or os.cpu_count())

    t0 = time.perf_counter()
    model, meta = load_quantized(f"{args.checkpoint_dir}/stack100m_int{args.bits}")
    tok = StackTokenizer.load(f"{args.checkpoint_dir}/tokenizer.json")
    load_time = time.perf_counter() - t0

    generate_text(model, tok, args.prompt, max_new_tokens=4, temperature=0.0)

    t1 = time.perf_counter()
    text, n_new = generate_text(model, tok, args.prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=0.7, top_p=0.95, return_n_tokens=True)
    gen_time = time.perf_counter() - t1

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"--- Stack-100M (int{args.bits}, group={meta['group_size']}) on CPU ---")
    print(f"Threads:          {torch.get_num_threads()}")
    print(f"Load time:        {load_time:.2f} s")
    print(f"Generation:       {gen_time:.2f} s  ({n_new / max(gen_time, 1e-9):.1f} tok/s)")
    print(f"Peak RSS:         {peak_rss_mb:.0f} MB")
    print(f"Output: {args.prompt}{text}")
    return {"load_time": load_time, "gen_time": gen_time, "n_new": n_new,
           "peak_rss_mb": peak_rss_mb}


# --- exercise block #17 ------------------------------------------------------
if save_file is not None:
    _tokenizer_json = f"{_out_dir}/tokenizer.json"
    with open(_tokenizer_json, "w") as f:
        json.dump({}, f)     # our toy StackTokenizer.load ignores the file's contents

    _old_argv = sys.argv
    try:
        sys.argv = ["cli.py", "--bits", "4", "--checkpoint_dir", _out_dir,
                   "--prompt", "The mitochondria", "--max_new_tokens", "4", "--threads", "1"]
        _cli_stats = _cli_main()
    finally:
        sys.argv = _old_argv

    assert _cli_stats["n_new"] >= 0 and _cli_stats["load_time"] >= 0
    assert _cli_stats["peak_rss_mb"] > 0
    print(f"\n[block #17 OK] cli.py main(): loaded quantized export, generated "
          f"{_cli_stats['n_new']} tokens.\n")
else:
    print("[block #17 SKIP(optional-dep)] safetensors not installed; cli.py's "
          "main() is defined but not invoked (needs block #15's exported artifact).\n")


# =====================================================================
# Block #18 (Sec 9.1, torchao snippet) -- SKIP(optional-dep + illustrative).
# The chapter's own snippet calls an undefined `load_fp32_stack100m()`; the
# import is guarded, and IF torchao were installed we substitute our own toy
# model and actually run the real quantize_ call below.
# =====================================================================

if _HAS_TORCHAO:
    _torchao_model = copy.deepcopy(_fp32_model).eval()
    quantize_(_torchao_model, Int8WeightOnlyConfig())
    _out, _ = _torchao_model(torch.tensor([[1, 2, 3]]))
    assert _out.shape == (1, 3, _cfg.vocab_size)
    print("[block #18 OK] torchao Int8WeightOnlyConfig applied to the toy model.\n")
else:
    print("[block #18 SKIP(optional-dep)] torchao not installed; the chapter's own "
          "snippet also calls an undefined load_fp32_stack100m(), so this is a "
          "documentation sketch even when the package is present.\n")


# =====================================================================
# Block #19 (Sec 9.1, llm-compressor snippet) -- SKIP(optional-dep +
# illustrative). References undefined `model`/`calibration_dataset` in the
# chapter's own source; a documentation sketch of the GPTQModifier API.
# =====================================================================

if _HAS_LLMCOMPRESSOR:
    print("[block #19 SKIP(illustrative)] llmcompressor is installed, but the "
          "chapter's snippet references undefined `model`/`calibration_dataset` "
          "and expects a HF PreTrainedModel this chapter never builds.\n")
else:
    print("[block #19 SKIP(optional-dep)] llmcompressor not installed.\n")


# =====================================================================
# SKIP notes (not executed -- see module docstring for full rationale)
# =====================================================================
# #1  cache-equivalence snippet (Sec 2 warning)  -- SKIP(fragment)
# #2  compute_perplexity / ppl.py                -- SKIP(needs-gpu / non-hermetic
#                                                     PackedMemmapDataset); toy
#                                                     stand-in defined for block #16
# #9  _fake_quant_grouped / awq_search_channel_scales -- SKIP(fragment, unused
#                                                     by any tested block)
# #20, #21 bash CLI invocations                  -- SKIP(shell)
# #22 Exercise 6 solution (*_symmetric variants)  -- SKIP(fragment, exercise-only)

print("=== All tested blocks (#0,#3,#4,#5,#6[bonus],#7,#8,#10,#12,#13,#14,#15,#16,#17) "
      "executed and verified successfully. ===")
