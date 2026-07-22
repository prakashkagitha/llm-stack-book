"""
Runnable-code test for content/06-rl-infra/11-scaling-rl-tricks.md

Tests the 3 heuristically CPU-runnable Python blocks from the chapter:
  - block #0 (line ~42): static vs continuous batching token-forward sketch
  - block #1 (line ~74): LPT (longest-processing-time) greedy dispatch
  - block #5 (line ~247): GRPO vs Dr. GRPO advantage + loss normalization

Skipped (fragments, not standalone / need external engine or dataclass glue
that isn't part of the demonstrated logic; see chapter for context):
  - block #2 (line ~120): PartialTrajectory / run_phase -- depends on an
    external `engine.generate(...)` object and a module-level GLOBAL_MAX;
    it's an architecture sketch, not self-contained runnable logic.
  - block #3 (line ~171): dynamic_sampling_step -- depends on external
    `engine` and `reward_fn` objects (rollout engine + verifier), fragment.
  - block #4 (line ~205): length_shaped_reward -- trivial fragment on its
    own; DAPO's over-long soft punishment. (Actually CPU-safe / no external
    deps -- included below as a bonus check since it's essentially free.)
  - block #6 (line ~288): informativeness / allocate_budget -- trivial
    fragment; also CPU-safe with no external deps -- included below.
"""

import numpy as np


# ============================================================
# Block #0 (line ~42): static vs continuous batching sketch
# ============================================================
# Sketch of why continuous batching helps. Two regimes, same 8 prompts.

lengths = np.array([200, 220, 240, 260, 300, 1800, 320, 280])  # one outlier

# Static batching: every step processes ALL live slots padded to max length.
static_token_forwards = len(lengths) * lengths.max()           # 8 * 1800

# Continuous batching: a slot is freed at its own EOS; total real work only.
# (Assume the engine always has the 8 in flight; freed slots stay idle here
#  because there is nothing else queued -> tail still hurts, but no padding.)
continuous_token_forwards = lengths.sum()

print(static_token_forwards)      # 14400
print(continuous_token_forwards)  # 3620  -> ~4x less compute on the same batch

assert static_token_forwards == 8 * 1800 == 14400
assert continuous_token_forwards == int(lengths.sum()) == 3620
assert continuous_token_forwards < static_token_forwards


# ============================================================
# Block #1 (line ~74): LPT greedy dispatch across engines
# ============================================================
# Length-balanced greedy dispatch (LPT, "longest processing time first") when
# you DO have a length estimate (e.g., from a cheap length-predictor or from a
# previous epoch's trace for the same prompt). Minimizes makespan = max load.
import heapq

def lpt_dispatch(prompt_len_estimates, num_engines):
    """Assign prompts to engines to minimize the max per-engine total length.
    prompt_len_estimates: list of (prompt_id, estimated_gen_len)."""
    # Sort longest-first so big jobs are placed before small ones can imbalance.
    jobs = sorted(prompt_len_estimates, key=lambda x: -x[1])
    # Min-heap of (current_load, engine_id); always feed the least-loaded engine.
    heap = [(0, e) for e in range(num_engines)]
    heapq.heapify(heap)
    assignment = {e: [] for e in range(num_engines)}
    for pid, est in jobs:
        load, eng = heapq.heappop(heap)
        assignment[eng].append(pid)
        heapq.heappush(heap, (load + est, eng))
    return assignment

# LPT is a 4/3-approximation to optimal makespan -- in practice near-perfect
# balance when you have many small jobs and a few large ones.

# --- exercise lpt_dispatch on a tiny toy workload ---
toy_prompts = [
    ("p0", 1800), ("p1", 200), ("p2", 220), ("p3", 240),
    ("p4", 260), ("p5", 300), ("p6", 320), ("p7", 280),
]
assignment = lpt_dispatch(toy_prompts, num_engines=3)
print("LPT assignment:", assignment)

# Every prompt must be assigned exactly once across the 3 engines.
all_assigned = sorted(pid for engine_prompts in assignment.values() for pid in engine_prompts)
assert all_assigned == sorted(pid for pid, _ in toy_prompts)
assert set(assignment.keys()) == {0, 1, 2}

# The engine loads should be reasonably balanced -- LPT is a 4/3-approximation
# to the optimal makespan, so no engine should be wildly more loaded than the
# ideal average.
len_by_id = dict(toy_prompts)
loads = {e: sum(len_by_id[pid] for pid in pids) for e, pids in assignment.items()}
total = sum(len_by_id.values())
ideal_avg = total / 3
makespan = max(loads.values())
assert makespan <= (4 / 3) * ideal_avg + max(l for _, l in toy_prompts)


# ============================================================
# Block #5 (line ~247): Dr. GRPO -- removing normalization biases
# ============================================================
import torch

def grpo_advantage(rewards):                       # rewards: (G,)
    mean = rewards.mean()
    std = rewards.std() + 1e-6
    return (rewards - mean) / std                  # vanilla GRPO

def dr_grpo_advantage(rewards):                    # Dr. GRPO: mean-only baseline
    return rewards - rewards.mean()

def sequence_loss_vanilla(token_logp_ratio, adv, mask):
    # per-sequence mean over tokens (divides by length) -> length bias
    per_tok = -(token_logp_ratio * adv.unsqueeze(-1)) * mask
    return (per_tok.sum(-1) / mask.sum(-1).clamp(min=1)).mean()

def token_loss_drgrpo(token_logp_ratio, adv, mask):
    # flat mean over ALL valid tokens (no per-length division) -> length-neutral
    per_tok = -(token_logp_ratio * adv.unsqueeze(-1)) * mask
    return per_tok.sum() / mask.sum().clamp(min=1)

# --- exercise the advantage functions on a tiny reward group ---
torch.manual_seed(0)

rewards = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)  # G=4
vanilla_adv = grpo_advantage(rewards)
dr_adv = dr_grpo_advantage(rewards)
print("vanilla GRPO advantage:", vanilla_adv)
print("Dr. GRPO advantage:    ", dr_adv)

assert vanilla_adv.shape == rewards.shape
assert dr_adv.shape == rewards.shape
# Dr. GRPO advantage is just the mean-subtracted reward, no std division.
assert torch.allclose(dr_adv, rewards - rewards.mean())
# Vanilla divides by (approximately) std, so magnitudes differ from dr_adv
# whenever std != 1.
assert not torch.allclose(vanilla_adv, dr_adv)

# Degenerate all-same-reward group -> both baselines give ~0 advantage
# (this is exactly the DAPO zero-gradient problem discussed in the chapter).
degenerate_rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
assert torch.allclose(dr_grpo_advantage(degenerate_rewards), torch.zeros(4))

# --- exercise the two loss variants on a tiny batch of 2 sequences ---
# Batch of 2 sequences, max length 3 tokens; sequence 0 has 3 real tokens,
# sequence 1 has only 1 real token (rest is padding).
token_logp_ratio = torch.tensor([
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
])
adv = torch.tensor([1.0, -1.0])  # one good, one bad sequence
mask = torch.tensor([
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
])

vanilla_loss = sequence_loss_vanilla(token_logp_ratio, adv, mask)
token_loss = token_loss_drgrpo(token_logp_ratio, adv, mask)
print("vanilla per-sequence loss:", vanilla_loss.item())
print("Dr. GRPO token-level loss:", token_loss.item())

assert vanilla_loss.dim() == 0
assert token_loss.dim() == 0
# The two normalizations give different values on an imbalanced-length batch
# (this asymmetry is exactly the length bias the chapter describes).
assert not torch.isclose(vanilla_loss, token_loss)


# ============================================================
# Bonus (free, no external deps): block #4 length_shaped_reward
# and block #6 informativeness / allocate_budget fragments.
# ============================================================

def length_shaped_reward(task_reward, gen_len, soft_start, hard_cap, lam=1.0):
    """DAPO-style soft overlong penalty.
    No penalty below soft_start; linearly ramps to -lam at hard_cap.
    Truncated (>= hard_cap, no EOS) sequences are handled by masking elsewhere."""
    if gen_len <= soft_start:
        return task_reward
    if gen_len >= hard_cap:
        # over-long: mask this trajectory's loss instead of training on noise
        return None   # sentinel -> caller drops/masks it
    frac = (gen_len - soft_start) / (hard_cap - soft_start)
    return task_reward - lam * frac

assert length_shaped_reward(1.0, gen_len=100, soft_start=500, hard_cap=1000) == 1.0
assert length_shaped_reward(1.0, gen_len=1000, soft_start=500, hard_cap=1000) is None
mid = length_shaped_reward(1.0, gen_len=750, soft_start=500, hard_cap=1000)
assert 0.0 < mid < 1.0


def informativeness(pass_rate):
    """Expected per-group reward variance for Bernoulli reward; gradient signal
    peaks at pass_rate = 0.5 and vanishes at 0 or 1 (the DAPO-filtered cases)."""
    return pass_rate * (1.0 - pass_rate)

def allocate_budget(prompt_pass_rates, total_budget):
    w = {pid: informativeness(pr) + 1e-3 for pid, pr in prompt_pass_rates.items()}
    z = sum(w.values())
    return {pid: int(total_budget * wi / z) for pid, wi in w.items()}

assert informativeness(0.5) > informativeness(0.9)
assert informativeness(0.0) == 0.0
budget = allocate_budget({"easy": 0.95, "medium": 0.5, "hard": 0.05}, total_budget=1000)
print("allocated budget:", budget)
assert budget["medium"] > budget["easy"]
assert budget["medium"] > budget["hard"]

print("\nAll blocks executed successfully.")
