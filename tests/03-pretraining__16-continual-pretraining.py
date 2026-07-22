"""
Runs the CPU-runnable Python code blocks from:
    content/03-pretraining/16-continual-pretraining.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:  #0, #1, #2, #3, #4, #5, #6
Skipped blocks: none (block #6 self-skips only if scipy is unavailable).
"""

from __future__ import annotations

import copy
import math
import random
from itertools import cycle
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover - guarded per optional-import rule
    curve_fit = None


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #0 (line ~51) -- CPT LR schedule (re-warm -> stable -> re-decay)
# ============================================================================
_section("Block #0: make_cpt_schedule")


def make_cpt_schedule(
    optimizer,
    base_peak_lr: float,        # the ORIGINAL pretraining peak LR
    rewarm_fraction: float,     # eta_cpt = rewarm_fraction * base_peak_lr
    num_rewarm_steps: int,      # T_w: linear re-warmup
    num_total_steps: int,       # full CPT step budget
    num_decay_steps: int,       # T_d: final cosine re-decay
    start_lr_frac: float = 0.0, # LR at step 0 as a fraction of eta_cpt
    final_lr_frac: float = 0.05 # floor as a fraction of eta_cpt
):
    """
    Three-phase CPT LR schedule (re-warm -> stable -> re-decay).
    Returns a multiplier in [0, 1] RELATIVE to base_peak_lr, so set the
    optimizer's base lr == base_peak_lr.

        |       ____________________
        |      /                    \\
        |     /                      \\___
        |    /                           (floor)
        +---/------------------------------------>
          T_w        stable           T_d
    """
    eta_cpt_mult = rewarm_fraction          # peak as fraction of base peak
    stable_end   = num_total_steps - num_decay_steps

    def lr_lambda(step: int) -> float:
        if step < num_rewarm_steps:
            # Phase 1: linear re-warm from start_lr_frac*eta_cpt -> eta_cpt
            frac = step / max(1, num_rewarm_steps)
            mult = start_lr_frac + frac * (1.0 - start_lr_frac)
            return eta_cpt_mult * mult
        elif step < stable_end:
            # Phase 2: hold at the re-warm peak
            return eta_cpt_mult
        else:
            # Phase 3: cosine re-decay from eta_cpt -> final_lr_frac*eta_cpt
            prog = (step - stable_end) / max(1, num_decay_steps)
            prog = min(prog, 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))   # 1 -> 0
            mult = final_lr_frac + (1.0 - final_lr_frac) * cos
            return eta_cpt_mult * mult

    return LambdaLR(optimizer, lr_lambda)

# Example: base model trained at peak 3e-4, finished at 3e-5.
# We re-warm to 20% of peak (6e-5), over 2% of the CPT budget, then
# decay over the final 20%.
# optimizer base lr is set to 3e-4 (base_peak_lr).
# sched = make_cpt_schedule(opt, base_peak_lr=3e-4, rewarm_fraction=0.20,
#             num_rewarm_steps=400, num_total_steps=20_000,
#             num_decay_steps=4_000, final_lr_frac=0.05)

# --- exercise the schedule on a tiny toy model/optimizer ---
_toy_param = nn.Linear(4, 4)
_base_peak_lr = 3e-4
_opt = torch.optim.Adam(_toy_param.parameters(), lr=_base_peak_lr)
_sched = make_cpt_schedule(
    _opt,
    base_peak_lr=_base_peak_lr,
    rewarm_fraction=0.20,
    num_rewarm_steps=4,
    num_total_steps=20,
    num_decay_steps=4,
    final_lr_frac=0.05,
)

_lrs = []
for _step in range(20):
    _lrs.append(_opt.param_groups[0]["lr"])
    _opt.step()
    _sched.step()
print("LR trace (first 8):", [round(x, 8) for x in _lrs[:8]])

# Sanity checks on the three phases:
# (1) re-warm phase should be monotonically non-decreasing
assert all(_lrs[i] <= _lrs[i + 1] + 1e-12 for i in range(3)), "re-warm should ramp up"
# (2) LR at the stable phase should equal rewarm_fraction * base_peak_lr
_eta_cpt = 0.20 * _base_peak_lr
assert abs(_lrs[10] - _eta_cpt) < 1e-9, f"expected stable-phase LR {_eta_cpt}, got {_lrs[10]}"
# (3) LR should never exceed the re-warm peak
assert max(_lrs) <= _eta_cpt + 1e-9
print("make_cpt_schedule: re-warm/stable/re-decay phases verified.")


# ============================================================================
# Block #1 (line ~135) -- replay-aware data samplers
# ============================================================================
_section("Block #1: replay_mixed_stream / token_balanced_stream")


def replay_mixed_stream(new_iter, replay_iter, replay_ratio: float, seed: int = 0):
    """
    Yield documents from a CPT stream that is `replay_ratio` old-distribution
    and the rest new-distribution. Token-level ratios are approximated at the
    document level here; for exact token budgets, weight by document length.

    new_iter    : iterator over NEW-domain documents (the target)
    replay_iter : iterator over BASE/REPLAY documents (anti-forgetting)
    replay_ratio: probability a given document comes from replay (e.g. 0.05)
    """
    rng = random.Random(seed)
    new_pool    = cycle(new_iter)      # in practice these are sharded streams,
    replay_pool = cycle(replay_iter)   # not in-memory; cycle is illustrative
    while True:
        if rng.random() < replay_ratio:
            yield next(replay_pool)
        else:
            yield next(new_pool)

# For *exact* token accounting you instead track running token counts and
# pull from whichever source is behind its target share -- this matters when
# documents vary wildly in length (code files vs. tweets).
def token_balanced_stream(new_iter, replay_iter, replay_ratio, len_fn=len):
    new_pool, replay_pool = cycle(new_iter), cycle(replay_iter)
    seen_new = seen_replay = 0
    while True:
        total = seen_new + seen_replay
        # current replay share; pull replay if we are below target, else new
        cur_replay_share = seen_replay / total if total else 0.0
        if cur_replay_share < replay_ratio:
            doc = next(replay_pool); seen_replay += len_fn(doc)
        else:
            doc = next(new_pool); seen_new += len_fn(doc)
        yield doc

# --- exercise both generators on tiny toy corpora ---
_new_docs = [f"new_doc_{i}" for i in range(5)]
_replay_docs = [f"replay_doc_{i}" for i in range(5)]

_stream = replay_mixed_stream(_new_docs, _replay_docs, replay_ratio=0.3, seed=42)
_sample = [next(_stream) for _ in range(200)]
_replay_frac = sum(1 for d in _sample if d.startswith("replay_")) / len(_sample)
print(f"replay_mixed_stream: observed replay fraction over 200 draws = {_replay_frac:.3f}")
assert 0.15 < _replay_frac < 0.45, "observed replay share should be roughly near 0.3"

_tb_stream = token_balanced_stream(_new_docs, _replay_docs, replay_ratio=0.25, len_fn=len)
_tb_sample = [next(_tb_stream) for _ in range(200)]
_tb_replay_tokens = sum(len(d) for d in _tb_sample if d.startswith("replay_"))
_tb_total_tokens = sum(len(d) for d in _tb_sample)
_tb_frac = _tb_replay_tokens / _tb_total_tokens
print(f"token_balanced_stream: observed token-weighted replay fraction = {_tb_frac:.3f}")
assert abs(_tb_frac - 0.25) < 0.05, "token-balanced share should converge closely to target"


# ============================================================================
# Block #2 (line ~241) -- function-preserving depth growth
# ============================================================================
_section("Block #2: grow_depth_identity")


def grow_depth_identity(model, insert_after: list):
    """
    Insert function-preserving (identity-at-init) transformer blocks.
    `model.layers` is a ModuleList of residual transformer blocks, each of
    the form: x = x + attn(ln1(x)); x = x + mlp(ln2(x)).
    We clone an existing block and ZERO its residual-output projections so the
    new block is the identity map at initialization -> the grown model
    computes the SAME function as the original at step 0 (loss is preserved).
    """
    new_layers = []
    for i, layer in enumerate(model.layers):
        new_layers.append(layer)
        if i in insert_after:
            twin = copy.deepcopy(layer)
            with torch.no_grad():
                # zero the output projection of attention (o_proj) and MLP
                # (down_proj). Names depend on your block; adapt accordingly.
                twin.attn.o_proj.weight.zero_()
                if twin.attn.o_proj.bias is not None:
                    twin.attn.o_proj.bias.zero_()
                twin.mlp.down_proj.weight.zero_()
                if getattr(twin.mlp.down_proj, "bias", None) is not None:
                    twin.mlp.down_proj.bias.zero_()
            new_layers.append(twin)
    model.layers = nn.ModuleList(new_layers)
    model.config.num_layers = len(new_layers)
    return model


# --- minimal residual toy block matching the docstring's shape, and a toy
# model wrapping a ModuleList of them + a config namespace (glue) ---
class _ToyBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Module()
        self.attn.o_proj = nn.Linear(d, d)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(d, d)

    def forward(self, x):
        x = x + self.attn.o_proj(x)
        x = x + self.mlp.down_proj(x)
        return x


class _ToyModel(nn.Module):
    def __init__(self, d: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(_ToyBlock(d) for _ in range(num_layers))
        self.config = SimpleNamespace(num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


torch.manual_seed(0)
_toy_model = _ToyModel(d=8, num_layers=3)
_x = torch.randn(2, 8)
with torch.no_grad():
    _out_before = _toy_model(_x).clone()

_toy_model = grow_depth_identity(_toy_model, insert_after=[0, 1])
assert len(_toy_model.layers) == 5, "should have inserted 2 identity twins"
assert _toy_model.config.num_layers == 5

with torch.no_grad():
    _out_after = _toy_model(_x).clone()

# Function-preserving: the grown model must compute EXACTLY the same output
# as before growth, since inserted twins are the identity map at init.
assert torch.allclose(_out_before, _out_after, atol=1e-6), \
    "grow_depth_identity should preserve the model's function at init"
print("grow_depth_identity: function preserved after inserting 2 identity layers (5 total).")


# ============================================================================
# Block #3 (line ~280) -- function-preserving width growth (net2wider)
# ============================================================================
_section("Block #3: net2wider_linear")


@torch.no_grad()
def net2wider_linear(W_in: torch.Tensor, W_out: torch.Tensor, new_width: int,
                     noise: float = 1e-3):
    """
    Function-preserving width expansion of one hidden layer.
      W_in : (hidden, in)   -> produces the hidden activations
      W_out: (out, hidden)  -> consumes them
    Returns expanded (W_in', W_out') with hidden -> new_width such that the
    composed map is unchanged (up to small symmetry-breaking noise).
    """
    hidden, _in = W_in.shape
    assert new_width >= hidden
    # choose which existing neurons to duplicate (uniform with replacement)
    idx = torch.randint(0, hidden, (new_width - hidden,))
    pick = torch.cat([torch.arange(hidden), idx])         # (new_width,)
    # count copies of each original neuron, for the divide-by-k correction
    counts = torch.bincount(pick, minlength=hidden).float()
    W_in_new  = W_in[pick].clone()                        # replicate rows
    W_in_new += noise * torch.randn_like(W_in_new)        # break symmetry
    # outgoing weights: divide each replica by the # of copies of its source
    W_out_new = (W_out[:, pick] / counts[pick].unsqueeze(0)).clone()
    return W_in_new, W_out_new


# --- exercise net2wider on a tiny linear->relu->linear stack. With noise=0 the
# expansion is EXACTLY function-preserving (the divide-by-k correction cancels
# the duplicated neurons); verify the composed map is unchanged. ---
torch.manual_seed(0)
_W_in = torch.randn(4, 3)                       # (hidden=4, in=3)
_W_out = torch.randn(2, 4)                       # (out=2, hidden=4)
_x = torch.randn(3, 5)                           # (in, batch)


def _stack(W_in, W_out, x):                      # W_out @ relu(W_in @ x)
    return W_out @ torch.relu(W_in @ x)


_y_before = _stack(_W_in, _W_out, _x)
_W_in2, _W_out2 = net2wider_linear(_W_in, _W_out, new_width=9, noise=0.0)
assert _W_in2.shape == (9, 3), "W_in should grow to (new_width, in)"
assert _W_out2.shape == (2, 9), "W_out should grow to (out, new_width)"
_y_after = _stack(_W_in2, _W_out2, _x)
assert torch.allclose(_y_before, _y_after, atol=1e-5), \
    "net2wider_linear (noise=0) must preserve the composed function"

# with the default tiny noise the map is only approximately preserved
_W_in3, _W_out3 = net2wider_linear(_W_in, _W_out, new_width=9, noise=1e-3)
_y_noisy = _stack(_W_in3, _W_out3, _x)
assert not torch.equal(_y_before, _y_noisy), "symmetry-breaking noise should perturb outputs"
assert torch.allclose(_y_before, _y_noisy, atol=5e-2), "perturbation should stay small"
print("net2wider_linear: exact function-preservation at noise=0, small perturbation at noise=1e-3.")


# ============================================================================
# Block #4 (line ~308) -- dense-to-MoE sparse upcycling
# ============================================================================
_section("Block #4: UpcycledMoE")


class UpcycledMoE(nn.Module):
    """
    Replace a dense MLP with an MoE whose experts are clones of that MLP.
    At init all experts are identical, so with a near-uniform router the layer
    approximately reproduces the dense MLP -> function-preserving upcycle.
    See chapter 2.9 for routing, load balancing, and capacity factors.
    """
    def __init__(self, dense_mlp, num_experts=8, top_k=2):
        super().__init__()
        d_model = dense_mlp.up_proj.in_features
        # one router (gate) added fresh; small init so logits start ~uniform
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=1e-3)
        # each expert is a deep copy of the trained dense MLP
        self.experts = nn.ModuleList(copy.deepcopy(dense_mlp)
                                     for _ in range(num_experts))
        self.top_k = top_k

    def forward(self, x):                       # x: (tokens, d_model)
        logits = self.gate(x)                   # (tokens, E)
        w, idx = torch.topk(logits.softmax(-1), self.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)         # renormalize top-k weights
        out = torch.zeros_like(x)
        for slot in range(self.top_k):          # gather-scatter over experts
            for e, expert in enumerate(self.experts):
                mask = idx[:, slot] == e
                if mask.any():
                    out[mask] += w[mask, slot:slot+1] * expert(x[mask])
        return out


# --- toy dense MLP with the `.up_proj` attribute the code reads, plugged
# into an UpcycledMoE and run on a tiny batch of tokens ---
class _DenseMLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_hidden)
        self.down_proj = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


torch.manual_seed(0)
_dense_mlp = _DenseMLP(d_model=8, d_hidden=16)
_moe = UpcycledMoE(_dense_mlp, num_experts=4, top_k=2)
_tokens = torch.randn(6, 8)
_moe_out = _moe(_tokens)
assert _moe_out.shape == _tokens.shape
# every expert starts as an exact deep copy of the dense MLP
for expert in _moe.experts:
    assert torch.equal(expert.up_proj.weight, _dense_mlp.up_proj.weight)
print(f"UpcycledMoE: forward pass over {_tokens.shape[0]} tokens -> {tuple(_moe_out.shape)}, "
      f"{len(_moe.experts)} experts each a clone of the dense MLP.")


# ============================================================================
# Block #5 (line ~351) -- vocabulary/tokenizer transfer (mean-of-subtokens init)
# ============================================================================
_section("Block #5: init_new_embeddings")


@torch.no_grad()
def init_new_embeddings(old_emb, new_vocab, old_tokenizer, mean_init=True):
    """
    Build a new embedding matrix for an extended/changed vocabulary.
      old_emb       : (|V_old|, d) trained embedding matrix
      new_vocab     : dict {new_token_str -> new_id}
      old_tokenizer : can encode a string into OLD ids
    Shared tokens copy their trained vector; new tokens are initialized to the
    mean of the OLD sub-token embeddings of their surface string (FOCUS-style).
    Falls back to the overall mean (a safe centroid) when no sub-tokens exist.
    """
    d = old_emb.shape[1]
    new_emb = torch.empty(len(new_vocab), d)
    overall_mean = old_emb.mean(0)
    old_vocab = old_tokenizer.get_vocab()       # {token_str -> old_id}
    for tok, new_id in new_vocab.items():
        if tok in old_vocab:                     # shared: copy trained vector
            new_emb[new_id] = old_emb[old_vocab[tok]]
        elif mean_init:                          # new: mean of OLD sub-tokens
            sub_ids = old_tokenizer.encode(tok, add_special_tokens=False)
            if sub_ids:
                new_emb[new_id] = old_emb[torch.tensor(sub_ids)].mean(0)
            else:
                new_emb[new_id] = overall_mean
        else:
            new_emb[new_id] = overall_mean
    return new_emb


class _FakeOldTokenizer:
    """Tiny stand-in tokenizer: BPE-like sub-token split by fixed pieces,
    no network/model download involved."""

    def __init__(self, vocab: dict):
        self._vocab = vocab            # {token_str -> old_id}
        # a hand-built "sub-token split" for a few whole words not in vocab
        self._splits = {
            "hello": ["he", "llo"],
            "world": ["wor", "ld"],
            "unknownxyz": [],           # no sub-tokens found -> falls back to mean
        }

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, tok, add_special_tokens=False):
        pieces = self._splits.get(tok, [])
        return [self._vocab[p] for p in pieces if p in self._vocab]


torch.manual_seed(0)
_old_vocab = {"he": 0, "llo": 1, "wor": 2, "ld": 3, "shared_tok": 4}
_old_emb = torch.randn(len(_old_vocab), 6)
_old_tok = _FakeOldTokenizer(_old_vocab)

# new_vocab: one shared token (kept), two whole words needing mean-of-subtokens,
# and one token with no sub-tokens found (falls back to overall mean)
_new_vocab = {"shared_tok": 0, "hello": 1, "world": 2, "unknownxyz": 3}
_new_emb = init_new_embeddings(_old_emb, _new_vocab, _old_tok, mean_init=True)

assert _new_emb.shape == (4, 6)
# shared token: exact copy of its trained vector
assert torch.equal(_new_emb[0], _old_emb[_old_vocab["shared_tok"]])
# "hello" -> mean of "he" and "llo" embeddings
_expected_hello = _old_emb[torch.tensor([_old_vocab["he"], _old_vocab["llo"]])].mean(0)
assert torch.allclose(_new_emb[1], _expected_hello)
# "world" -> mean of "wor" and "ld" embeddings
_expected_world = _old_emb[torch.tensor([_old_vocab["wor"], _old_vocab["ld"]])].mean(0)
assert torch.allclose(_new_emb[2], _expected_world)
# "unknownxyz" -> no sub-tokens found, falls back to the overall mean
assert torch.allclose(_new_emb[3], _old_emb.mean(0))
print("init_new_embeddings: shared-token copy, mean-of-subtokens, and fallback-to-mean all verified.")


# ============================================================================
# Block #6 (line ~409) -- CPT loss-trajectory fitter (shifted power law)
# ============================================================================
_section("Block #6: fit_cpt_trajectory")


def fit_cpt_trajectory(tokens, losses):
    """
    Fit L(D) = L_inf + A / (D0 + D)**alpha to pilot (tokens, loss) points,
    then return a predictor and the token count to hit a target loss.
    tokens : array of CPT token counts (e.g. [1e9, 2e9, 4e9])
    losses : measured new-domain loss at each.
    """
    def law(D, L_inf, A, D0, alpha):
        return L_inf + A / np.power(D0 + D, alpha)
    p0 = [min(losses) * 0.9, 1.0, 1e9, 0.3]            # rough initial guess
    popt, _ = curve_fit(law, np.asarray(tokens), np.asarray(losses),
                        p0=p0, maxfev=20000)
    L_inf, A, D0, alpha = popt

    def predict(D):                                    # loss at D tokens
        return L_inf + A / (D0 + D) ** alpha

    def tokens_for_target(target_loss):                # invert the law
        if target_loss <= L_inf:
            return float("inf")                        # unreachable: below L_inf
        return (A / (target_loss - L_inf)) ** (1.0 / alpha) - D0

    return predict, tokens_for_target, popt

# Example usage:
# predict, tokens_for, params = fit_cpt_trajectory(
#     [1e9, 2e9, 4e9], [2.41, 2.30, 2.22])
# print(predict(40e9))           # extrapolated loss at the full 40B budget
# print(tokens_for(2.10))        # tokens needed to reach loss 2.10

if curve_fit is None:
    print("SKIP(optional-dependency): scipy not available, skipping fit_cpt_trajectory call.")
else:
    # pilot points generated from a KNOWN shifted power law + tiny noise, so we
    # can check the fitter recovers a sane, monotonically-decreasing curve.
    _true_law = lambda D, L_inf=2.0, A=50.0, D0=1e9, alpha=0.3: L_inf + A / (D0 + D) ** alpha
    _pilot_tokens = np.array([1e9, 2e9, 4e9, 8e9])
    _pilot_losses = np.array([_true_law(d) for d in _pilot_tokens])

    _predict, _tokens_for, _popt = fit_cpt_trajectory(_pilot_tokens, _pilot_losses)
    print("fitted params (L_inf, A, D0, alpha):", [round(float(p), 4) for p in _popt])

    # predictions at the pilot points should closely match the observed losses
    for d, observed in zip(_pilot_tokens, _pilot_losses):
        pred = _predict(d)
        assert abs(pred - observed) < 1e-2, f"predict({d}) = {pred}, expected ~{observed}"

    # extrapolated loss at a much larger budget should be lower than at the
    # largest pilot point (diminishing returns, still improving)
    _extrapolated = _predict(40e9)
    assert _extrapolated < _pilot_losses[-1]
    print(f"predict(40B tokens) = {_extrapolated:.4f} (< last pilot loss {_pilot_losses[-1]:.4f})")

    # tokens_for_target should invert predict (round-trip close to target D)
    _target_loss = _predict(6e9)
    _recovered_D = _tokens_for(_target_loss)
    assert abs(_recovered_D - 6e9) / 6e9 < 0.05
    print(f"tokens_for_target round-trip: target D=6e9 -> recovered D={_recovered_D:.3e}")


_section("All blocks executed")
print("03-pretraining/16-continual-pretraining.md: blocks #0, #1, #2, #3, #4, #5, #6 ran successfully.")
