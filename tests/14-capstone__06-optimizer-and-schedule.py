"""
Runs the CPU-runnable Python code blocks from:
    content/14-capstone/06-optimizer-and-schedule.md

Blocks are copied faithfully from the chapter (verbatim logic) and concatenated in
document order, then each is actually exercised with tiny fixtures/toy shapes so
every tested block EXECUTES (functions are called, classes are instantiated and
stepped, not merely defined).

Tested blocks (numbering matches the chapter's own code-block index):
    #2  (lines ~76-109)   -- zeropower_via_newtonschulz5 (Newton-Schulz orthogonalizer).
                              Not itself on the required list, but block #3's
                              `Muon.step()` calls it directly on its non-batched path,
                              so it is included verbatim as a real dependency rather
                              than silently stubbed.
    #3  (lines ~133-189)  -- Muon optimizer class (momentum + orthogonalized,
                              RMS-matched, weight-decayed step)
    #4  (lines ~209-255)  -- zeropower_batched / _orthogonalize_bucketed
                              (the `batched=True` shape-bucketed fast path)
    #6  (lines ~324-358)  -- verify_qk_invariance.py: the QK-norm scale-invariance
                              demo, run against the REAL, in-repo `capstone/stacklm`
                              package (`Stack100M` + `toy_config`). This block's own
                              code IS those two imports, so importing the actual
                              shipped package -- pure `torch.nn.Module`, random init,
                              no pretrained weights, no network -- is the most
                              verbatim way to execute it.
    #9  (lines ~427-478)  -- qk_clip_ / _clip_qk_norm_gains_ (QK-norm clip path,
                              the core of MuonClip as adapted here)
    #10 (lines ~490-518)  -- _clip_projections_ (Kimi K2's original, no-QK-norm,
                              GQA-aware clip path)
    #11 (lines ~561-596)  -- wsd_lr (Warmup-Stable-Decay schedule)
    #14 (Exercise 5 solution, lines ~917-942) -- wsd_lr_branch (phase lengths given
                              directly, no baked-in `total_steps` horizon)
    #15 (Exercise 6 solution, lines ~951-979) -- Newton-Schulz conditioning sanity
                              check (ill-conditioned matrix -> orthogonalized
                              spectrum), plus the looped-vs-batched agreement check

Skipped blocks (reason):
    #0  non-python (```text API-contract signature block).
    #1  non-python (```text Stack-100M parameter-routing table).
    #5  SKIP(needs-gpu): the "measure optimizer.step()" timing snippet
        (`torch.cuda.synchronize(); ...; muon.step(); adamw.step()`) is CUDA-only
        and references `muon`/`adamw` objects that are never defined standalone.
    #7  non-python (```text expected console output of block #6).
    #8  SKIP(fragment): the `Attention.forward` excerpt is explicitly introduced as
        "(inside Attention.forward, after QK-norm + RoPE, ...)" -- it references
        `self`, `q`, `k`, `T`, `x` from an enclosing method never given standalone.
        The same logic is exercised end-to-end for real when block #6 below calls
        `model(x, record=rec)` against the real `Attention.forward`.
    #12 SKIP(not required / no additional coverage): `build_optimizers()` routes
        model params to Muon vs AdamW by `p.ndim == 2` (and "not the embedding").
        Not on the required block list; it is a thin wrapper around the
        already-tested `Muon` class and stdlib `AdamW`, and building a real
        (non-toy) ~101M-param `Stack100M` just to exercise the routing loop would
        add real runtime for no new logic coverage.
    #13 SKIP(needs-gpu + fragment): `optimizer_step()`, the canonical inner
        training-loop step, hardcodes `torch.autocast(device_type="cuda", ...)`
        and calls `next(batches)` on an undefined generator -- not standalone
        CPU-runnable.
    #16 non-python (State-of-the-Art / Further-Reading prose; no fenced Python).
    #17 SKIP(fragment): the Exercise 7 per-head QK-norm snippet
        (`self.q_norm = RMSNorm((self.n_heads, 1, self.head_dim), ...)`) is two
        lines meant to replace two lines inside `Attention.__init__`; it references
        `self`/`cfg` from an enclosing constructor never given standalone.

No network access. Only `torch` (CPU) and the standard library are used, plus one
import of this repo's own `capstone/stacklm` package (for block #6) -- itself pure
`torch`, random-init, no pretrained weights, no HF Hub, verified import-clean of
every package on the CI blocklist.
"""

from __future__ import annotations

import math
import os
import sys

import torch
from torch.optim.optimizer import Optimizer

# ---------------------------------------------------------------------------
# Make the in-repo `capstone/stacklm` package importable (needed by block #6,
# and for the fresh model instances used to exercise blocks #9/#10). No PyPI
# install, no network -- this is the book's own shipped package, sitting right
# next to this tests/ directory in the repo.
# ---------------------------------------------------------------------------
_CAPSTONE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capstone")
sys.path.insert(0, os.path.abspath(_CAPSTONE_DIR))


# =============================================================================
# Block #2 (stacklm/optim/muon.py) -- zeropower_via_newtonschulz5
# Verbatim dependency of block #3 (Muon.step's non-batched path needs it).
# =============================================================================
_NS_COEFFS = (3.4445, -4.7750, 2.0315)          # tuned quintic coefficients

@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal (polar) factor UV^T of a 2D matrix G via a
    quintic Newton-Schulz iteration. Returns a matrix with singular values ~1.
    """
    assert G.ndim == 2, "Newton-Schulz orthogonalization is only for 2D matrices"
    a, b, c = _NS_COEFFS
    X = G.float()
    X = X / (X.norm() + 1e-7)                    # normalize so all sigma <= 1
    transposed = G.size(0) > G.size(1)          # keep the short side as rows
    if transposed:
        X = X.T                                  # iterate on the smaller Gram matrix
    for _ in range(steps):
        A = X @ X.T                              # A = X X^T  (small: min(m,n)^2)
        P = b * A + c * (A @ A)                  # polynomial in the Gram matrix
        X = a * X + P @ X                        # quintic Newton-Schulz update
    if transposed:
        X = X.T
    return X


# =============================================================================
# Block #3 (stacklm/optim/muon.py, continued) -- the Muon optimizer
# =============================================================================
class Muon(Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalization, for 2D matrices ONLY.
    Route embeddings/norms/1D params to AdamW instead (see build_optimizers())."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.1, ns_steps=5, batched=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.batched = batched

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            nesterov, wd, ns = group["nesterov"], group["weight_decay"], group["ns_steps"]

            live, dirs = [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon received a non-2D param; check routing"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)                    # B <- mu*B + g
                live.append(p)
                # Nesterov look-ahead: orthogonalize g + mu*B, not the bare buffer.
                dirs.append(g.add(buf, alpha=mu) if nesterov else buf)

            if not live:
                continue
            if self.batched:
                orths = _orthogonalize_bucketed(dirs, ns)     # one bmm per shape class
            else:
                orths = [zeropower_via_newtonschulz5(d, steps=ns) for d in dirs]

            for p, o in zip(live, orths):
                # RMS-match: cancel the shape dependence, land the update RMS at 0.2.
                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5
                # Decoupled weight decay (AdamW-style): shrink the weight itself.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(o.to(p.dtype), alpha=-lr * scale)   # W <- W - lr*scale*O
        return loss


# =============================================================================
# Block #4 (stacklm/optim/muon.py, the batched fast path)
# =============================================================================
@torch.no_grad()
def zeropower_batched(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Same quintic on a STACK of identically-shaped matrices: G is (K, m, n).

    Collapses K*steps*4 kernel launches into steps*4 batched ones. Algebraically
    identical to looping zeropower_via_newtonschulz5; expect last-ulp differences,
    because bmm and mm use different reduction orders and split-k heuristics.
    """
    assert G.ndim == 3, "zeropower_batched expects a (K, m, n) stack"
    a, b, c = _NS_COEFFS
    X = G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)   # per-matrix Frobenius
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT                                          # batched transpose
    for _ in range(steps):
        A = X @ X.mT                                      # (K, r, r) via bmm
        P = b * A + c * (A @ A)
        X = a * X + P @ X
    if transposed:
        X = X.mT
    return X


@torch.no_grad()
def _orthogonalize_bucketed(dirs, steps: int):
    """Orthogonalize a list of 2D directions, bucketing identical shapes.

    Key on the SORTED shape, so (1408, 512) and (512, 1408) land in the SAME
    bucket after the orientation-normalizing transpose. That is what collapses
    Stack-100M's 210 matrices into 3 classes rather than 4.
    """
    out = [None] * len(dirs)
    buckets: dict[tuple, list[int]] = {}
    for i, d in enumerate(dirs):
        m, n = d.shape
        buckets.setdefault((min(m, n), max(m, n)), []).append(i)
    for idxs in buckets.values():
        flip = [dirs[i].size(0) > dirs[i].size(1) for i in idxs]
        stack = torch.stack([dirs[i].T if f else dirs[i] for i, f in zip(idxs, flip)])
        O = zeropower_batched(stack, steps=steps)
        for j, (i, f) in enumerate(zip(idxs, flip)):
            out[i] = O[j].T if f else O[j]
    return out


print("=== Blocks #2-#4: Muon optimizer (looped + batched Newton-Schulz paths) ===")
torch.manual_seed(0)


def _make_toy_params():
    return [torch.nn.Parameter(torch.randn(6, 4)), torch.nn.Parameter(torch.randn(4, 10))]


for _batched in (False, True):
    _params = _make_toy_params()
    _opt = Muon(_params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.1,
                ns_steps=5, batched=_batched)
    _before = [p.detach().clone() for p in _params]
    # A tiny quadratic loss so every param gets a real, non-degenerate gradient.
    _loss = sum((p ** 2).sum() for p in _params)
    _loss.backward()
    _opt.step()
    for _pb, _pa in zip(_before, _params):
        assert _pa.shape == _pb.shape
        assert not torch.allclose(_pb, _pa), "Muon.step() did not move the weight"
        assert torch.isfinite(_pa).all(), "Muon step produced a non-finite weight"
        _delta = (_pa - _pb).abs().max().item()
        assert 0.0 < _delta < 1.0, f"Muon step size looks implausible: {_delta}"
    # A second step must reuse (not reset) the momentum buffer -- covers the
    # `"momentum_buffer" not in state` branch on step 1 vs the accumulation on step 2.
    for p in _params:
        p.grad = None
    _loss2 = sum((p ** 2).sum() for p in _params)
    _loss2.backward()
    _mid = [p.detach().clone() for p in _params]
    _opt.step()
    for _pm, _pa in zip(_mid, _params):
        assert not torch.allclose(_pm, _pa), "Muon.step() (2nd call) did not move the weight"
print("Muon: both the looped (batched=False) and bucketed (batched=True) paths step correctly.")


# =============================================================================
# Block #6 -- verify_qk_invariance.py (run against the real capstone/stacklm pkg)
# =============================================================================
print("\n=== Block #6: verify_qk_invariance.py (QK-norm scale-invariance) ===")

from stacklm.config import StackConfig, toy_config  # noqa: E402  (needs sys.path setup above)
from stacklm.model.transformer import Stack100M  # noqa: E402

torch.manual_seed(0)
cfg = toy_config()                       # head_dim=16 -> a-priori bound sqrt(16)=4
model = Stack100M(cfg)
x = torch.randint(0, cfg.vocab_size, (2, 32))


def max_logit(m):
    rec = {}
    for b in m.blocks:                   # exact per-head max, not the cheap bound
        b.attn.record_exact = True
    with torch.no_grad():
        m(x, record=rec)
    return max(float(v.max()) for v in rec.values())


a = model.blocks[0].attn
gq, gk = a.q_norm.weight.detach(), a.k_norm.weight.detach()
bound = cfg.head_dim ** 0.5 * gq.abs().max() * gk.abs().max()
print(f"a-priori bound        {float(bound):.4f}")
obs0 = max_logit(model)
print(f"observed max logit    {obs0:.4f}")

with torch.no_grad():                    # (1) rescale the PROJECTIONS
    for b in model.blocks:
        b.attn.wq.weight.mul_(0.3); b.attn.wk.weight.mul_(0.3)
obs1 = max_logit(model)
print(f"after W_Q,W_K x0.3    {obs1:.4f}   <- unchanged (no-op)")

with torch.no_grad():                    # (2) rescale the GAINS
    for b in model.blocks:
        b.attn.q_norm.weight.mul_(0.5); b.attn.k_norm.weight.mul_(0.5)
obs2 = max_logit(model)
print(f"after gains x0.5 each {obs2:.4f}   <- exactly 0.25x")

# --- the invariance the chapter proves algebraically ---
assert abs(float(bound) - 4.0) < 1e-6, "toy head_dim=16, gains=1 at init -> bound should be exactly 4"
assert 0.0 < obs0 <= float(bound) * 1.02, "observed logit must respect the a-priori Cauchy-Schwarz bound"
assert abs(obs1 - obs0) < 1e-2 * obs0, "rescaling W_Q/W_K under QK-norm must be (numerically) a no-op"
assert abs(obs2 / obs1 - 0.25) < 1e-3, "rescaling BOTH gains by 0.5 must scale every logit by exactly 0.25"
print("QK-norm invariance confirmed: projection rescale is a no-op, gain rescale is exact.")


# =============================================================================
# Block #9 -- qk_clip_ / _clip_qk_norm_gains_ (QK-norm clip path)
# =============================================================================
@torch.no_grad()
def qk_clip_(model, max_logits, tau: float = 100.0) -> int:
    """QK-clip (the core of MuonClip, Kimi K2 / Moonshot 2025), architecture-aware.

    `max_logits[layer_idx]` is an (n_heads,) tensor recorded by the attention
    module this forward pass. Returns the number of layers that fired -- log it.

    A dense (n_layers, n_heads) tensor is also accepted, because Ch. 14.7's
    train.py harvests into a preallocated running-max buffer. It is CONVERTED,
    not indexed directly: `layer_idx in tensor` is elementwise VALUE membership
    in PyTorch, not key lookup, and would silently do the wrong thing.
    """
    if torch.is_tensor(max_logits):
        max_logits = {i: max_logits[i] for i in range(max_logits.size(0))}
    fired = 0
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx not in max_logits:
            continue
        s_max = max_logits[layer_idx].float()          # (n_heads,)
        attn = block.attn
        if isinstance(attn.q_norm, torch.nn.Identity):
            fired += _clip_projections_(attn, s_max, tau)    # Kimi K2 config
        else:
            fired += _clip_qk_norm_gains_(attn, s_max, tau)  # Stack-100M
    return fired


@torch.no_grad()
def _clip_qk_norm_gains_(attn, s_max, tau: float) -> int:
    """QK-norm path: the gains ARE the temperature, so clip them."""
    s = float(s_max.max())
    if s <= tau:
        return 0                                  # this layer is fine
    eta = (tau / s) ** 0.5                        # sqrt so q AND k share it
    attn.q_norm.weight.mul_(eta)
    attn.k_norm.weight.mul_(eta)
    return 1


print("\n=== Block #9: qk_clip_ / _clip_qk_norm_gains_ (QK-norm path) ===")
torch.manual_seed(1)
cfg9 = toy_config()                       # qk_norm=True, n_heads=4, n_kv_heads=2, head_dim=16
model9 = Stack100M(cfg9)

q0_before = model9.blocks[0].attn.q_norm.weight.detach().clone()
k0_before = model9.blocks[0].attn.k_norm.weight.detach().clone()
q1_before = model9.blocks[1].attn.q_norm.weight.detach().clone()

# Layer 0's worst head (120) exceeds tau=100; layer 1's worst (40) does not.
max_logits9 = {0: torch.tensor([120., 90., 100., 80.]),
               1: torch.tensor([10., 20., 30., 40.])}
fired9 = qk_clip_(model9, max_logits9, tau=100.0)
assert fired9 == 1, f"expected exactly one layer to fire, got {fired9}"

eta9 = (100.0 / 120.0) ** 0.5
q0_after = model9.blocks[0].attn.q_norm.weight.detach()
k0_after = model9.blocks[0].attn.k_norm.weight.detach()
q1_after = model9.blocks[1].attn.q_norm.weight.detach()
assert torch.allclose(q0_after, q0_before * eta9, atol=1e-6)
assert torch.allclose(k0_after, k0_before * eta9, atol=1e-6)
assert torch.allclose(q1_after, q1_before), "layer whose worst head (40) is below tau must be untouched"
print(f"layer 0 gains scaled by {eta9:.4f} (= sqrt(100/120)); layer 1 untouched, as expected.")

# The docstring's other contract: a dense (n_layers, n_heads) tensor is accepted
# and CONVERTED (not indexed by elementwise value), giving identical behaviour.
torch.manual_seed(1)
model9b = Stack100M(toy_config())
dense9 = torch.stack([max_logits9[0], max_logits9[1]])         # (n_layers=2, n_heads=4)
fired9b = qk_clip_(model9b, dense9, tau=100.0)
assert fired9b == fired9
assert torch.allclose(model9b.blocks[0].attn.q_norm.weight, q0_after, atol=1e-6)
assert torch.allclose(model9b.blocks[1].attn.q_norm.weight, q1_after, atol=1e-6)
print("dense (n_layers, n_heads) tensor input matches the dict-keyed input exactly.")


# =============================================================================
# Block #10 -- _clip_projections_ (Kimi K2, no-QK-norm, GQA-aware clip path)
# =============================================================================
@torch.no_grad()
def _clip_projections_(attn, s_max, tau: float) -> int:
    """No-QK-norm path (Kimi K2): per-query-head W_Q, per-KV-head shared W_K.

    Attribute names (n_heads, n_kv_heads, head_dim, groups) are exactly those of
    stacklm.model.attention.Attention -- keep them stable across chapters.
    """
    hd = attn.head_dim
    group = attn.groups                          # q-heads per kv-head (= 4)
    fired = 0

    # (1) Per-query-head scale on W_Q.
    for h in range(attn.n_heads):
        if float(s_max[h]) <= tau:
            continue                             # this head is fine
        eta = (tau / float(s_max[h])) ** 0.5     # sqrt so q AND k share it
        attn.wq.weight[h * hd:(h + 1) * hd].mul_(eta)
        fired += 1

    # (2) Per-KV-head scale on the SHARED W_K, using the group's worst logit.
    for kv in range(attn.n_kv_heads):
        s_grp = float(s_max[kv * group:(kv + 1) * group].max())
        if s_grp <= tau:
            continue
        eta = (tau / s_grp) ** 0.5
        attn.wk.weight[kv * hd:(kv + 1) * hd].mul_(eta)
    return fired


print("\n=== Block #10: _clip_projections_ (no-QK-norm, GQA-aware clip path) ===")
torch.manual_seed(2)
cfg10 = StackConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2,
                     head_dim=8, intermediate=32, max_seq_len=16, qk_norm=False)
model10 = Stack100M(cfg10)
attn10 = model10.blocks[0].attn
assert isinstance(attn10.q_norm, torch.nn.Identity), "qk_norm=False must route through _clip_projections_"

wq_before = attn10.wq.weight.detach().clone()
wk_before = attn10.wk.weight.detach().clone()

# heads 0..3; groups of 2 -> kv group 0 = {heads 0,1}, kv group 1 = {heads 2,3}.
s_max10 = torch.tensor([120., 90., 100., 80.])
fired10 = qk_clip_(model10, {0: s_max10}, tau=100.0)
assert fired10 == 1, f"only head 0 (120) exceeds tau=100, expected fired=1, got {fired10}"

hd10, eta10 = attn10.head_dim, (100.0 / 120.0) ** 0.5
wq_after = attn10.wq.weight.detach()
wk_after = attn10.wk.weight.detach()
assert torch.allclose(wq_after[0:hd10], wq_before[0:hd10] * eta10, atol=1e-5), "head 0's W_Q slice must be scaled"
for _h in (1, 2, 3):
    assert torch.allclose(wq_after[_h * hd10:(_h + 1) * hd10], wq_before[_h * hd10:(_h + 1) * hd10]), \
        f"head {_h} must be untouched (its own logit was <= tau)"
# kv group 0 (heads {0,1}, worst logit 120) is scaled once; kv group 1 (heads {2,3},
# worst logit 100 <= tau) is untouched.
assert torch.allclose(wk_after[0:hd10], wk_before[0:hd10] * eta10, atol=1e-5), \
    "kv group 0's shared W_K must be scaled ONCE by the group's worst logit"
assert torch.allclose(wk_after[hd10:2 * hd10], wk_before[hd10:2 * hd10]), \
    "kv group 1 (worst=100, not > tau) must be untouched"
print(f"W_Q head 0 and shared W_K group 0 both scaled by {eta10:.4f}; other heads/groups untouched.")


# =============================================================================
# Block #11 -- wsd_lr (Warmup-Stable-Decay schedule)
# =============================================================================
def wsd_lr(step: int, *, peak_lr: float, warmup_steps: int, total_steps: int,
           decay_steps: int | None = None, decay_frac: float = 0.2,
           final_frac: float = 0.0) -> float:
    """Warmup-Stable-Decay learning rate (MiniCPM, Hu et al. 2024).

    - Linear warmup for `warmup_steps`.
    - Constant `peak_lr` through the stable phase.
    - 1 - sqrt() decay over the final phase, down to `final_frac * peak_lr`
      (we use 0.0, i.e. anneal fully to ~0).

    Give EITHER `decay_steps` (absolute) or `decay_frac` (a fraction of
    total_steps -- what Ch. 14.7's TrainConfig passes, 0.10). `decay_steps` wins.

    Call it with peak_lr=1.0 to get a pure MULTIPLIER, which is how the training
    loop drives Muon's 0.02 and AdamW's 3e-3 off one shared curve.
    """
    if decay_steps is None:
        decay_steps = int(decay_frac * total_steps)
    stable_end = total_steps - decay_steps               # first decay step
    if step < warmup_steps:                              # --- warmup ---
        return peak_lr * (step + 1) / warmup_steps
    if step < stable_end:                                # --- stable ---
        return peak_lr
    # --- decay: 1 - sqrt(progress) ---
    progress = (step - stable_end) / max(1, decay_steps)
    # CLAMP: without min(...,1.0) any step past total_steps gives sqrt(p) > 1 and
    # a NEGATIVE learning rate -- live risk, since Ch. 14.8 continues past the
    # pretrain horizon. Past the end we sit at the floor.
    decay_mult = 1.0 - math.sqrt(min(progress, 1.0))     # 1 -> 0
    floor = final_frac * peak_lr
    return floor + (peak_lr - floor) * decay_mult


print("\n=== Block #11: wsd_lr (Warmup-Stable-Decay) ===")
# Small round numbers: warmup=10, total=100, decay_frac=0.2 -> decay_steps=20, stable_end=80.
_kw = dict(peak_lr=1.0, warmup_steps=10, total_steps=100, decay_frac=0.2)
assert wsd_lr(0, **_kw) == 1.0 * (0 + 1) / 10
assert wsd_lr(9, **_kw) == 1.0                       # last warmup step reaches full peak
assert wsd_lr(10, **_kw) == 1.0                      # stable phase, constant
assert wsd_lr(79, **_kw) == 1.0                      # still stable (one step before decay)
_mid_decay = wsd_lr(85, **_kw)                       # progress = (85-80)/20 = 0.25
assert abs(_mid_decay - (1.0 - math.sqrt(0.25))) < 1e-9
assert abs(_mid_decay - 0.5) < 1e-9
_past_end = wsd_lr(500, **_kw)                       # far past total_steps: clamp, not negative
assert _past_end == 0.0, "the clamp must prevent a negative learning rate past the horizon"

# The two peaks the chapter drives off ONE shared multiplier (Muon 0.02, AdamW 3e-3):
# the central invariant the chapter calls out explicitly.
_mult = wsd_lr(85, **_kw)
_muon_lr, _adamw_lr = 0.02 * _mult, 3e-3 * _mult
assert abs(_muon_lr / _adamw_lr - 0.02 / 3e-3) < 1e-9, "both groups must ride the SAME multiplier"

# The chapter's own worked numbers (warmup=2000, total=38147, decay_frac=0.10):
# mid-warmup and deep-stable checks only (both well clear of any decay-boundary
# rounding), so this does not depend on how int(decay_frac*total_steps) rounds.
_book_kw = dict(peak_lr=1.0, warmup_steps=2000, total_steps=38147, decay_frac=0.10)
assert abs(wsd_lr(1000, **_book_kw) - 0.5005) < 1e-3      # "mid-warmup, multiplier ~0.5"
assert wsd_lr(20000, **_book_kw) == 1.0                   # deep in the stable phase
print("wsd_lr: warmup / stable / decay / post-horizon clamp all verified.")


# =============================================================================
# Block #14 (Exercise 5 solution) -- wsd_lr_branch (no baked-in total_steps horizon)
# =============================================================================
def wsd_lr_branch(step: int, *, peak_lr: float, warmup_steps: int,
                  stable_steps: int, decay_steps: int,
                  final_frac: float = 0.0) -> float:
    """WSD LR with phase lengths given DIRECTLY (no baked-in total horizon).

    Decouples the decay from any pre-committed step count, so multiple decay
    runs of different lengths can branch off one stable-phase checkpoint
    (MiniCPM's continuable-pretraining property). Call with peak_lr=1.0 to
    get a multiplier and preserve the Muon/AdamW ratio.
    """
    stable_end = warmup_steps + stable_steps          # first decay step
    if step < warmup_steps:                           # --- warmup ---
        return peak_lr * (step + 1) / warmup_steps
    if step < stable_end:                             # --- stable ---
        return peak_lr
    # --- decay: 1 - sqrt(progress), clamped to [0, 1] ---
    progress = (step - stable_end) / max(1, decay_steps)
    progress = min(progress, 1.0)                     # stay at the floor past the end
    decay_mult = 1.0 - math.sqrt(progress)            # 1 -> 0
    floor = final_frac * peak_lr
    return floor + (peak_lr - floor) * decay_mult


print("\n=== Block #14 (Exercise 5): wsd_lr_branch (phase lengths, no total horizon) ===")
# Reproduce the flagship recipe exactly -- pure arithmetic, no loop, so the exact
# 38,147-step numbers cost nothing to check directly.
_branch_kw = dict(peak_lr=1.0, warmup_steps=2_000, stable_steps=32_332, decay_steps=3_815)
assert wsd_lr_branch(0, **_branch_kw) == 1.0 / 2000
assert wsd_lr_branch(1999, **_branch_kw) == 1.0                # last warmup step at peak
assert wsd_lr_branch(2000, **_branch_kw) == 1.0                # stable phase starts
assert wsd_lr_branch(2000 + 32332 - 1, **_branch_kw) == 1.0    # last stable step
_start_decay = 2000 + 32332
assert wsd_lr_branch(_start_decay, **_branch_kw) == 1.0        # progress=0 -> mult=1
_end_decay = _start_decay + 3815
assert wsd_lr_branch(_end_decay, **_branch_kw) == 0.0          # progress=1 -> floor (0)
assert wsd_lr_branch(_end_decay + 10_000, **_branch_kw) == 0.0, "must clamp, never go negative"

# `wsd_lr_branch` never references total_steps at all -- unlike `wsd_lr`, expressing
# the SAME horizon (2000 warmup, 34332 = start of decay, 3815 decay length) via
# `total_steps=38147, decay_steps=3815` must give identical values throughout.
for _step in (0, 1999, 2000, 34000, 34332, 35000, 38147, 50000):
    _a = wsd_lr(_step, peak_lr=1.0, warmup_steps=2_000, total_steps=38_147, decay_steps=3_815)
    _b = wsd_lr_branch(_step, peak_lr=1.0, warmup_steps=2_000, stable_steps=32_332, decay_steps=3_815)
    assert abs(_a - _b) < 1e-12, f"wsd_lr and wsd_lr_branch disagree at step {_step}: {_a} vs {_b}"
print("wsd_lr_branch: no total_steps needed; matches wsd_lr(..., decay_steps=...) exactly.")


# =============================================================================
# Block #15 (Exercise 6 solution) -- Newton-Schulz conditioning sanity check
# =============================================================================
print("\n=== Block #15 (Exercise 6): ill-conditioned matrix -> conditioned output ===")
torch.manual_seed(0)
_m, _n = 256, 128
# Random orthonormal U, V and a bad spectrum from 1e-1 to 1e1 (cond ~1e2).
_U, _ = torch.linalg.qr(torch.randn(_m, _m))
_V, _ = torch.linalg.qr(torch.randn(_n, _n))
_s = torch.logspace(-1, 1, _n)                      # 1e-1 ... 1e1, span 1e2
_G = (_U[:, :_n] * _s) @ _V.T                        # ill-conditioned (m x n)

_input_cond = float(_s.max() / _s.min())
print("input  sigma: min %.3e  max %.3e  cond %.3e" % (_s.min(), _s.max(), _input_cond))

_O = zeropower_via_newtonschulz5(_G, steps=5).float()
_sv = torch.linalg.svdvals(_O)
_output_cond = float(_sv.max() / _sv.min())
print("output sigma: min %.4f  max %.4f  cond %.4f" % (_sv.min(), _sv.max(), _output_cond))

assert abs(_input_cond - 100.0) < 1.0, "the constructed spectrum should have cond ~100"
assert _output_cond < 3.0, "5 Newton-Schulz steps should collapse cond~100 to well under 3"
assert 0.5 < float(_sv.min()) < 1.5 and 0.5 < float(_sv.max()) < 1.5, \
    "orthogonalized singular values should cluster tightly around 1, not be exactly 1"

# Batched path must agree with the looped one to reduction-order noise, on
# Stack-100M's real shape classes (60x 512x512, 60x 128x512, 90x 512x1408/1408x512).
_dirs = [torch.randn(*_sh) for _sh in [(512, 512), (128, 512), (1408, 512), (512, 1408)]]
_looped = [zeropower_via_newtonschulz5(d) for d in _dirs]
_batched = _orthogonalize_bucketed(_dirs, 5)
_max_diff = max(float((_la - _ba).abs().max()) for _la, _ba in zip(_looped, _batched))
print("max |looped - batched| =", _max_diff)
assert _max_diff < 1e-4, f"looped and batched Newton-Schulz should agree to near machine precision, got {_max_diff}"
# And they should NOT be identical bit-for-bit (that would suggest the batched
# path silently fell back to a loop rather than actually using bmm reductions).
assert _max_diff > 0.0

print("\nAll targeted blocks (2/3/4/6/9/10/11/14/15) executed and asserted successfully.")
