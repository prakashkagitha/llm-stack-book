"""
Runs the CPU-runnable Python blocks from
content/06-rl-infra/12-rl-data-curriculum-replay.md, concatenated in order so
that later blocks can rely on names defined by earlier ones (as they do in the
chapter itself). Each block is copied verbatim from the chapter; only the
minimal glue needed to make blocks that only *define* something actually
execute has been added, and is clearly marked "GLUE". The book's own
`if __name__ == "__main__":` demo guard (in block #2) is left in place and
fires naturally, since this file itself plays the role of "__main__".

Blocks covered (all 4 heuristically CPU-runnable blocks the task named):

  #0 (line ~42)  - RLTask dataclass + make_checker() (math-domain checker)
  #1 (line ~108) - estimate_offline_difficulty() / difficulty_bucket()
  #2 (line ~257) - TaskState / FakeEngine / select_candidates / rl_step +
                    the chapter's own __main__ demo loop
  #3 (line ~379) - prompt_priority() / sample_from_buffer() (prioritized
                    prompt buffer bolt-on)

No block touches the network, GPU, or any third-party package outside the
allowed list (numpy + stdlib only), so nothing is skipped.

GLUE note: block #0's `make_checker` calls three helper functions the prose
explicitly says are only "sketched" for the math path --
`normalize_math`, `extract_boxed_answer`, `math_equal` -- and are never
defined anywhere in the chapter. Minimal, honest stand-in implementations are
provided below (regex \\boxed{} extraction + string-normalized equality) so
`make_checker`'s own logic (unchanged, verbatim) actually executes end to end.
"""

import random
import re

# ===========================================================================
# GLUE: stand-in implementations for the chapter's "sketched" math-checker
# helpers (normalize_math / extract_boxed_answer / math_equal). The chapter
# says "here we sketch the math path" and never defines these; this is the
# minimal honest glue needed so block #0's make_checker() can run.
# ===========================================================================
def normalize_math(ans):
    if ans is None:
        return ans
    return ans.strip().replace(" ", "")


def extract_boxed_answer(completion):
    m = re.search(r"\\boxed\{([^}]*)\}", completion)
    return m.group(1) if m else None


def math_equal(a, b):
    return normalize_math(a) == normalize_math(b)


# ===========================================================================
# Block #0 (line ~42) -- verbatim from the chapter
# ===========================================================================
from dataclasses import dataclass, field
from typing import Callable, Optional
import hashlib

@dataclass
class RLTask:
    task_id: str                      # stable hash, used for dedup + replay keys
    prompt: str                       # the rendered chat prompt fed to the policy
    answer: Optional[str] = None      # ground-truth answer for math-style checking
    tests: Optional[list] = None      # unit tests for code tasks
    domain: str = "unknown"           # math / code / logic / agentic / ...
    # difficulty state, maintained ONLINE (see later sections):
    pass_rate_ema: float = 0.5        # exponential moving avg of empirical pass rate
    n_attempts: int = 0               # how many groups we've spent on this task
    n_solved_groups: int = 0          # groups where at least one sample passed
    static_difficulty: Optional[float] = None  # offline estimate, optional prior

    def fingerprint(self) -> str:
        # Normalize before hashing so trivial whitespace/casing diffs collapse.
        norm = " ".join(self.prompt.lower().split())
        return hashlib.sha256(norm.encode()).hexdigest()[:16]


def make_checker(task: RLTask) -> Callable[[str], float]:
    """Return a verifier closure. In production the code path runs in a sandbox;
    here we sketch the math path. The checker MUST be robust to extraction noise."""
    if task.domain == "math":
        gold = normalize_math(task.answer)
        def check(completion: str) -> float:
            pred = extract_boxed_answer(completion)   # parse \boxed{...}
            return 1.0 if pred is not None and math_equal(pred, gold) else 0.0
        return check
    raise NotImplementedError(f"no checker for domain {task.domain}")


# ---------------------------------------------------------------------------
# GLUE: actually execute block #0 -- instantiate RLTask, hash it, build a
# checker, and run it on both a correct and an incorrect completion.
# ---------------------------------------------------------------------------
task0 = RLTask(task_id="t0", prompt="What is 2+2?", answer="4", domain="math")
fp = task0.fingerprint()
assert isinstance(fp, str) and len(fp) == 16, "fingerprint() must be a 16-char hex digest"

checker0 = make_checker(task0)
assert checker0("The answer is \\boxed{4}.") == 1.0
assert checker0("The answer is \\boxed{5}.") == 0.0
assert checker0("no boxed answer here") == 0.0

# The unsupported-domain branch is normal (non-crash) control flow in
# make_checker -- exercise it too, narrowly, since it's part of block #0.
try:
    make_checker(RLTask(task_id="t1", prompt="write code", domain="code"))
    raise AssertionError("expected NotImplementedError for domain='code'")
except NotImplementedError:
    pass

print("[block 0] RLTask.fingerprint() + make_checker() OK "
      f"(fingerprint={fp}, correct=1.0, incorrect=0.0)")


# ===========================================================================
# Block #1 (line ~108) -- verbatim from the chapter
# ===========================================================================
import numpy as np

def estimate_offline_difficulty(tasks, policy, checker_fn, k=8, batch_gen=None):
    """One pass over the pool: k rollouts each, record empirical pass rate.
    Returns buckets and prunes the dead tails. batch_gen() should call your
    rollout engine (vLLM/SGLang) -- batch ALL prompts*k together for throughput."""
    for t in tasks:
        completions = batch_gen(t.prompt, n=k)            # k samples
        rewards = [checker_fn(t)(c) for c in completions]
        t.static_difficulty = float(np.mean(rewards))     # p_hat_0 in [0,1]
        t.pass_rate_ema = t.static_difficulty             # seed the online EMA

    keep, pruned = [], []
    for t in tasks:
        # Prune the dead tails: never-solved (maybe broken) and always-solved.
        if 0.0 < t.static_difficulty < 1.0:
            keep.append(t)
        else:
            pruned.append(t)
    return keep, pruned


def difficulty_bucket(p, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Map a pass rate to a coarse bucket index. Buckets, not raw p, because
    raw p from small k is too noisy to act on at fine granularity."""
    for b in range(len(edges) - 1):
        if edges[b] <= p < edges[b + 1]:
            return b
    return len(edges) - 2  # p == 1.0 edge case


# ---------------------------------------------------------------------------
# GLUE: actually execute block #1 -- a tiny pool of math tasks with distinct
# per-task solve probabilities, and a fake batch_gen() standing in for the
# rollout engine (mirrors the book's own FakeEngine.rollout in spirit).
# ---------------------------------------------------------------------------
random.seed(0)

_prompt_to_answer = {
    "What is 3+4?": "7",     # near-mastered -> should get pruned
    "What is 10-2?": "8",    # mid-difficulty -> should get kept
    "What is 6*7?": "42",    # near-impossible (for this fake policy) -> pruned
}
_solve_prob = {
    "What is 3+4?": 0.95,
    "What is 10-2?": 0.5,
    "What is 6*7?": 0.02,
}

tasks_b1 = [
    RLTask(task_id=f"m{i}", prompt=p, answer=a, domain="math")
    for i, (p, a) in enumerate(_prompt_to_answer.items())
]


def fake_batch_gen(prompt, n):
    correct = _prompt_to_answer[prompt]
    p_solve = _solve_prob[prompt]
    return [
        f"\\boxed{{{correct}}}" if random.random() < p_solve else "\\boxed{wrong}"
        for _ in range(n)
    ]


kept_b1, pruned_b1 = estimate_offline_difficulty(
    tasks_b1, policy=None, checker_fn=make_checker, k=32, batch_gen=fake_batch_gen
)

for t in tasks_b1:
    assert t.static_difficulty is not None
    b = difficulty_bucket(t.static_difficulty)
    assert 0 <= b <= 4, f"bucket out of range: {b}"

assert len(kept_b1) + len(pruned_b1) == len(tasks_b1)

print(f"[block 1] estimate_offline_difficulty(): kept={len(kept_b1)} "
      f"pruned={len(pruned_b1)} "
      f"static_difficulties={[round(t.static_difficulty, 3) for t in tasks_b1]}")


# ===========================================================================
# Block #2 (line ~257) -- verbatim from the chapter, including its own
# `if __name__ == "__main__":` demo, which fires naturally since this test
# file itself is run as __main__.
# ===========================================================================
import numpy as np
from dataclasses import dataclass, field

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Task state: a Beta(s+1, f+1) posterior over the *current-policy* pass rate.
# We decay old counts so the posterior tracks the moving policy (not lifetime).
# ---------------------------------------------------------------------------
@dataclass
class TaskState:
    task_id: int
    true_p: float                 # SIMULATION ONLY: the latent pass rate
    s: float = 1.0                # decayed success pseudo-count (+1 prior)
    f: float = 1.0                # decayed failure pseudo-count (+1 prior)
    n_groups: int = 0             # how many groups we've spent here (priority/age)

    def posterior_mean(self):
        return self.s / (self.s + self.f)

    def sample_p(self):
        # Thompson draw: sample a plausible pass rate from the posterior.
        return rng.beta(self.s, self.f)

    def update(self, successes, G, decay=0.9):
        # Decay then add this group's evidence. Decay makes the posterior
        # forget stale (old-policy) observations so it tracks current p.
        self.s = decay * self.s + successes
        self.f = decay * self.f + (G - successes)
        self.n_groups += 1


# ---------------------------------------------------------------------------
# Stubs standing in for the real rollout engine and trainer.
# ---------------------------------------------------------------------------
class FakeEngine:
    """Simulates generating G samples for a task; returns #successes ~ Binomial.
    A real engine returns text completions; the checker turns them into r in {0,1}.
    We also let true_p drift UP a touch each time a task is trained on, to mimic
    the policy mastering material (the non-stationarity the EMA/Beta must track)."""
    def rollout(self, task: TaskState, G: int) -> int:
        succ = int(rng.binomial(G, task.true_p))
        return succ

    def learn_drift(self, task: TaskState, lr=0.02):
        # Mastering nudges pass rate toward 1; harder material drifts slower.
        task.true_p = min(0.999, task.true_p + lr * (1.0 - task.true_p))


# ---------------------------------------------------------------------------
# Difficulty-targeted selection by Thompson sampling toward a target pass rate.
# ---------------------------------------------------------------------------
def select_candidates(tasks, n_select, target_p=0.5):
    scored = []
    for t in tasks:
        p_tilde = t.sample_p()              # explore via posterior uncertainty
        score = -abs(p_tilde - target_p)    # prefer tasks near the target band
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:n_select]]


# ---------------------------------------------------------------------------
# One RL step: select -> generate -> DYNAMIC SAMPLING filter -> update -> train.
# Dynamic sampling: keep only non-zero-variance groups; oversample to refill.
# ---------------------------------------------------------------------------
def rl_step(tasks, engine, B_keep=64, G=8, target_p=0.5,
            oversample_cap=6, buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    kept = []                               # (task, successes) that have signal
    generated_groups = 0
    rounds = 0
    while len(kept) < B_keep and rounds < oversample_cap:
        rounds += 1
        # Oversample a generous candidate set so we can refill after filtering.
        need = B_keep - len(kept)
        cands = select_candidates(tasks, n_select=2 * need, target_p=target_p)
        for t in cands:
            successes = engine.rollout(t, G)
            generated_groups += 1
            t.update(successes, G)          # online difficulty (Beta) update
            # DYNAMIC SAMPLING: drop zero-variance groups (all pass / all fail).
            if 0 < successes < G:
                kept.append((t, successes))
                if len(kept) >= B_keep:
                    break

    # --- "Train" on the kept (informative) groups: here we just record stats
    #     and apply the simulated learning drift so difficulty is non-stationary.
    bucket_counts = np.zeros(len(buckets) - 1, dtype=int)
    for t, successes in kept:
        engine.learn_drift(t)               # policy improves -> p drifts up
        p_hat = successes / G
        b = min(np.searchsorted(buckets, p_hat, side="right") - 1, len(buckets) - 2)
        bucket_counts[max(b, 0)] += 1

    survival = len(kept) / max(generated_groups, 1)
    return {
        "kept": len(kept),
        "generated_groups": generated_groups,
        "survival_rho": survival,           # what fraction survived the filter
        "oversample_factor": generated_groups / max(len(kept), 1),
        "bucket_counts": bucket_counts,     # distribution of kept difficulties
    }


# ---------------------------------------------------------------------------
# Run it. Watch survival rho recover toward 1 (selection feeds in-band prompts)
# and the kept-difficulty histogram concentrate near the target band.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A pool spanning the full difficulty range, incl. the dead tails we must avoid.
    pool = [TaskState(i, true_p=rng.uniform(0.02, 0.98)) for i in range(4000)]
    for step in range(8):
        stats = rl_step(pool, FakeEngine(), B_keep=64, G=8, target_p=0.5)
        print(f"step {step:2d} | rho={stats['survival_rho']:.2f} "
              f"| oversample={stats['oversample_factor']:.2f}x "
              f"| buckets[0.0-1.0]={stats['bucket_counts']}")
        assert stats["kept"] == 64
        assert 0.0 < stats["survival_rho"] <= 1.0


# ===========================================================================
# Block #3 (line ~379) -- verbatim from the chapter
# ===========================================================================
# Bolt-on: a prioritized PROMPT buffer (PER at the task level). Priority = how
# close a task is to the target band, with a small age bonus so we revisit
# under-sampled tasks. This is the on-policy, FREE kind of replay.
def prompt_priority(t: TaskState, target_p=0.5, age_w=0.05):
    closeness = 1.0 / (abs(t.posterior_mean() - target_p) + 0.05)  # near band -> high
    uncertainty = (t.s * t.f) / ((t.s + t.f) ** 2 * (t.s + t.f + 1))  # Beta variance
    return closeness + age_w * uncertainty  # exploit band + explore uncertain tasks

def sample_from_buffer(tasks, n, target_p=0.5, temperature=1.0):
    pr = np.array([prompt_priority(t, target_p) for t in tasks])
    probs = (pr ** (1.0 / temperature))
    probs /= probs.sum()
    idx = rng.choice(len(tasks), size=n, replace=False, p=probs)
    return [tasks[i] for i in idx]


# ---------------------------------------------------------------------------
# GLUE: actually execute block #3 -- draw a prioritized sample of prompts
# from the pool the block #2 demo just trained on.
# ---------------------------------------------------------------------------
sampled = sample_from_buffer(pool, n=16, target_p=0.5)
assert len(sampled) == 16
assert all(isinstance(t, TaskState) for t in sampled)
mean_closeness = float(np.mean([abs(t.posterior_mean() - 0.5) for t in sampled]))
print(f"[block 3] sample_from_buffer(): drew {len(sampled)} tasks, "
      f"mean |p_hat - 0.5| = {mean_closeness:.3f}")

print("\nAll 4 blocks executed successfully.")
