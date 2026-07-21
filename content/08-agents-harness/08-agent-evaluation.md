# 8.8 Agent Evaluation & Benchmarks

Evaluating a language model on a multiple-choice question is easy: you check whether the top-1 token matches the gold label. Evaluating an *agent* is structurally harder: the agent must execute a sequence of actions, possibly across dozens of tool calls, browser interactions, or shell commands, and the outcome depends on both what it does and *how* the harness around it behaves. A 3% improvement on SWE-bench might reflect a better model, a different scaffold, a looser time limit, or a lucky leak of test data into pretraining. Understanding which of these is true is the central challenge of agent evaluation.

This chapter covers the major benchmarks — SWE-bench, SWE-bench Verified, WebArena, GAIA, tau-bench, terminal-bench, and others — and then digs into the measurement science: trajectory versus outcome scoring, pass@k estimators, harness effects, reproducibility, and contamination. We close with a worked example and practitioner guidance for building trustworthy evals in your own projects.

## Why Agent Evaluation Is Different

Before cataloguing benchmarks, it's worth internalizing what makes agent eval fundamentally harder than standard LLM eval.

**Long-horizon dependencies.** A coding agent might write a fix in step 3, run tests in step 7, and interpret failure in step 11. An error at any point can cascade. You cannot meaningfully score a single step in isolation; the entire trajectory matters.

**Non-determinism stacks up.** Even with temperature 0, sampling LLMs are not truly deterministic across batches or hardware. A 50-step trajectory multiplies this: a different branch taken in step 4 yields a completely different state by step 30. This creates high variance in outcomes and demands more samples to get a reliable estimate.

**The harness is part of the system.** The scaffold that decides *when* to execute tool calls, *how* to format context, *how long* to let the agent run, and *whether* to retry on errors is not a neutral observer. Different harnesses running the same model on the same task routinely produce results 10–20 percentage points apart. See [Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html) for how harness decisions compound.

**Outcome vs. trajectory ambiguity.** Two agents might both fail a task — but one explored a sensible plan and hit an environment bug, while the other hallucinated the entire approach. A binary pass/fail metric does not distinguish them.

**Contamination is hard to detect.** SWE-bench tasks are drawn from real GitHub issues. If a model has seen the merged PR in pretraining, the task is effectively leaked. Standard n-gram decontamination is not sufficient because the overlap may be semantic, not lexical.

## The Major Benchmarks

### SWE-bench and SWE-bench Verified

SWE-bench (Jimenez et al., 2023) is the canonical benchmark for software-engineering agents. Each task is a real GitHub issue from a popular Python repository (Django, Flask, NumPy, Sympy, and others). The agent receives the issue text and the full repository, and must produce a *patch* — a `git diff` — that makes the failing test suite pass.

The original SWE-bench has approximately 2,300 tasks (split into dev and test). **SWE-bench Verified** is a human-curated subset of about 500 tasks where annotators confirmed the task is well-specified, the reference solution is correct, and no annotation errors exist. Because the full set has a long tail of noisy tasks, Verified is now the preferred reporting target: scores are higher and more meaningful.

**Evaluation protocol.**

{{fig:ageval-swebench-protocol}}

The score is *resolve rate* = fraction of tasks where all tests pass. Notice that this is a *binary outcome* metric — a patch that fixes 9 of 10 failing tests still scores 0.

**The harness matters enormously.** The original paper used a simple single-attempt scaffold. Community leaderboards (SWE-bench.com) allow any scaffold. Reported differences include:

- **Context window usage:** How much of the codebase the agent sees (full repo, BM25-retrieved files, oracle file localization).
- **Iteration count:** Number of edit-debug-retry loops allowed.
- **Execution feedback:** Whether the agent can run tests mid-trajectory or only at submission.
- **Model calls:** Some entries use multi-agent systems with a planner and executor.

This means comparing two entries requires checking their scaffold, not just their model. See [Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html) for how this connects to multi-turn training.

**Contamination.** The SWE-bench test set was constructed from issues merged before mid-2023. Models trained on data scraped after that date may have seen the solution in GitHub history. SWE-bench Verified includes a temporal split, but no benchmark fully escapes this problem for frontier models.

```python
# Minimal SWE-bench task runner (illustrative, not the official harness)
import subprocess, tempfile, os, json

def run_swebench_task(repo_path: str, patch: str, test_cmd: str) -> bool:
    """
    Apply a patch to a repo clone and run the test suite.
    Returns True if all tests pass.
    
    repo_path: path to a fresh checkout at the pre-patch commit
    patch:     git-diff-style string produced by the agent
    test_cmd:  e.g. "pytest tests/test_models.py -x -q"
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch)
        patch_file = f.name

    try:
        # Step 1: apply the patch
        result = subprocess.run(
            ["git", "apply", "--check", patch_file],
            cwd=repo_path,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # patch doesn't apply cleanly → immediate FAIL
            return False

        subprocess.run(["git", "apply", patch_file], cwd=repo_path, check=True)

        # Step 2: run the test suite
        result = subprocess.run(
            test_cmd.split(),
            cwd=repo_path,
            capture_output=True, text=True,
            timeout=120  # hard wall-clock limit per task
        )
        # pytest returns 0 only when all tests pass
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(patch_file)
        # Reset the repo for the next task
        subprocess.run(["git", "checkout", "--", "."], cwd=repo_path)
```

### WebArena

WebArena (Zhou et al., 2023) measures whether an agent can complete realistic web tasks — booking travel, searching an e-commerce site, navigating a codebase on GitLab, managing a Reddit-like forum, and similar. Tasks are expressed as natural-language instructions, and the agent interacts with live web environments via a browser API.

**Key design decisions:**

- *Sandboxed instances.* WebArena spins up isolated copies of real open-source web apps (GitLab, shopping, Reddit clones) so that agents cannot accidentally interact with the real internet and results are reproducible.
- *Functional evaluation.* Unlike pixel-based web tests, WebArena verifies the *state* of the application after task completion (e.g., "does a new issue exist with title X?") rather than checking button clicks.
- *Task diversity.* About 800 tasks spanning single-site and multi-site scenarios. Many require multi-step navigation with backtracking.

**Scoring.** Each task is binary (success/failure). The success criterion is verified by an automated checker that queries application state. Success rate across all tasks is the primary metric.

WebArena scores for frontier models have grown substantially as agents learned to reason about HTML structure and leverage screenshots. It tests a different capability than SWE-bench: navigation and form-filling under real UI constraints rather than code editing.

### GAIA

GAIA (Mialon et al., 2023) — the General AI Assistants benchmark — poses questions that require a combination of reasoning, web search, file parsing, and multi-step tool use, with exact-match graded answers. Tasks span three difficulty levels:

| Level | Description | Example |
|-------|-------------|---------|
| 1 | Simple tool use, 1–3 steps | "What is the capital of the country where X was born?" |
| 2 | Multi-hop, 4–8 steps | Parsing a PDF, looking up a value, doing arithmetic |
| 3 | Complex chained reasoning, 8+ steps | Cross-referencing multiple documents, code execution |

GAIA is deliberately designed so that the answers are short and verifiable (a number, a name, a date), reducing ambiguity in grading. Level 3 tasks remain very hard even for frontier models.

**What GAIA measures.** Unlike SWE-bench, GAIA is a general-purpose assistant benchmark. Succeeding requires knowing *when* to use which tool, correct tool invocation, and accurate synthesis of returned results. It does not test specialized coding ability.

### tau-bench

tau-bench (Yao et al., 2024) evaluates tool-augmented agents in *realistic customer service* settings. An agent must interact with a user (simulated by a model) and a database of tools to fulfill requests like modifying orders, refunding purchases, or looking up account status.

**Design novelties:**

- *Simulated user.* The "customer" is another LLM instructed to behave realistically, including asking follow-up questions and providing information in pieces. This tests turn-level conversation management.
- *Policy compliance.* Many tasks have explicit policy rules (e.g., "refunds are only allowed within 30 days"). The agent must follow policy while still satisfying the user, creating a tension that tests instruction-following under constraint.
- *Multi-turn scoring.* tau-bench records whether the agent correctly resolves the issue AND whether it violates any policy, producing a two-dimensional score.

tau-bench is particularly relevant for production deployments of service agents. Its simulated-user design avoids the need for human annotators during evaluation while keeping the dynamics realistic.

### terminal-bench

terminal-bench evaluates agents in a raw terminal environment without web or GUI scaffolding. Tasks are given as natural-language instructions inside a bash session; the agent must run commands, interpret output, install packages, write scripts, and navigate the filesystem. Inspired by earlier work on computer-using agents, it focuses on the low-level "can it actually operate a Unix system?" question.

**Key features:**

- Tasks run inside Docker containers for isolation and reproducibility.
- Time and command-count limits prevent pathological behavior.
- Scoring is automated: a checker script verifies the final system state (e.g., "does file X exist with content Y?").

terminal-bench is newer and smaller than SWE-bench or WebArena, but it isolates a distinct capability: raw terminal proficiency that is a prerequisite for coding agents.

### Other Notable Benchmarks

| Benchmark | Domain | Key Feature |
|-----------|--------|-------------|
| AgentBench (Liu et al., 2023) | Multi-domain | 8 environments: OS, DB, code, web |
| OSWorld | Desktop GUI | Screenshot-based computer use |
| InterCode | Bash/SQL | Interactive code execution |
| AppAgent / ScreenAgent | Mobile/Desktop | Vision-based GUI interaction |
| HumanEval-X | Code (pass@k) | Multilingual code generation |
| SciCode | Scientific coding | Domain knowledge + coding |

## Trajectory vs. Outcome Scoring

The choice between trajectory-level and outcome-level metrics is not just a technical detail — it fundamentally shapes what you are optimizing for.

**Outcome scoring** (used by SWE-bench, WebArena, GAIA) measures only the final state. Did the agent succeed? This is maximally objective and easy to compute. The downside is low signal: a model that consistently makes the same wrong first move on 50% of tasks and then never recovers will have the same score as one that sometimes finds a clever workaround — even though they represent very different capability profiles.

**Trajectory scoring** measures the quality of intermediate steps. Approaches include:

- *Step accuracy:* Fraction of actions that match a reference trajectory (human demonstration or oracle solution).
- *Subtask completion:* Credit for correctly completing each step in a decomposed task.
- *Process reward modeling:* A trained model scores each reasoning step for correctness (see [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)).
- *Efficiency metrics:* Number of steps taken, tokens consumed, wall-clock time.

The tension between them is real. Training a model with outcome rewards (e.g., RL on pass/fail signals) can lead to "hacking" the outcome metric via trajectories that are semantically bizarre but happen to produce the right file diff. Trajectory rewards add supervision signal but require annotated demonstrations, which are expensive.

In practice, most published agent benchmarks report outcome scores because they scale better. But for diagnosing failures — e.g., "our agent correctly localizes the bug but then applies the wrong fix" — trajectory-level analysis is indispensable.

```python
import json
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TrajectoryStep:
    action: str          # e.g. "read_file", "edit_file", "run_tests"
    action_input: dict
    observation: str     # what the environment returned
    reward: float = 0.0  # step-level reward if using process reward model

@dataclass
class Trajectory:
    task_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_outcome: bool = False   # did the task succeed?

def compute_outcome_score(trajectories: List[Trajectory]) -> float:
    """Binary pass rate — the standard SWE-bench metric."""
    return sum(t.final_outcome for t in trajectories) / len(trajectories)

def compute_trajectory_efficiency(traj: Trajectory) -> dict:
    """
    Compute trajectory-level metrics beyond binary pass/fail.
    Returns a dict with step count, token estimate, and action diversity.
    """
    step_count = len(traj.steps)
    
    # Rough token estimate: action_input + observation for each step
    total_chars = sum(
        len(json.dumps(s.action_input)) + len(s.observation)
        for s in traj.steps
    )
    token_estimate = total_chars // 4  # ~4 chars/token as approximation
    
    # How many distinct action types were used?
    action_types = {s.action for s in traj.steps}
    
    return {
        "task_id": traj.task_id,
        "success": traj.final_outcome,
        "step_count": step_count,
        "token_estimate": token_estimate,
        "distinct_actions": len(action_types),
        "tokens_per_step": token_estimate / max(step_count, 1),
    }

def compare_trajectories(
    model_a: List[Trajectory],
    model_b: List[Trajectory]
) -> dict:
    """
    Compare two sets of agent trajectories on both outcome and efficiency.
    Assumes matching task order between model_a and model_b.
    """
    outcome_a = compute_outcome_score(model_a)
    outcome_b = compute_outcome_score(model_b)

    # Only compare trajectories where outcomes differ (diagnostic cases)
    a_pass_b_fail = [(a, b) for a, b in zip(model_a, model_b)
                     if a.final_outcome and not b.final_outcome]
    b_pass_a_fail = [(a, b) for a, b in zip(model_a, model_b)
                     if not a.final_outcome and b.final_outcome]

    return {
        "outcome_A": outcome_a,
        "outcome_B": outcome_b,
        "delta": outcome_a - outcome_b,
        "A_wins": len(a_pass_b_fail),
        "B_wins": len(b_pass_a_fail),
        "n_tasks": len(model_a),
    }
```

## The pass@k Estimator

For stochastic agents, running a single sample per task produces an unreliable estimate. The **pass@k** metric, originally introduced for code generation in the Codex paper (Chen et al., 2021), addresses this by measuring whether *at least one* of $k$ independent samples solves the task.

### Definition

Given $n$ samples per task of which $c$ are correct, the probability that at least one of $k$ randomly chosen samples is correct is:

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

This is an unbiased estimator when $n \geq k$. You run $n$ samples, check which are correct, then compute the combinatorial probability. You never need to enumerate all $\binom{n}{k}$ subsets.

### Why Not Just Report the Best-of-k Score?

Reporting "we ran 10 samples and took the best" is problematic for two reasons:

1. **Selection requires a verifier.** You can only pick the best if you know which is correct, which requires running the test suite for all $k$ samples. In open-ended domains without a ground-truth checker, you cannot do this at test time.
2. **Overestimates agent utility.** Real deployments usually cannot afford $k = 10$ full runs. pass@1 is what users experience; pass@10 describes what is possible with a strong verifier.

The clean way to report agentic capability is to show pass@1 (agent performance in single-shot settings) and pass@k for larger $k$ (upper bound with oracle selection, useful for measuring the benefit of sampling diversity).

!!! example "Worked Example: Estimating pass@k"

    Suppose we run $n = 8$ independent agent trajectories on a SWE-bench task and $c = 2$ of them produce a passing patch. We want to estimate pass@1, pass@2, pass@4.

    Using the unbiased estimator:

    $$
    \text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} = 1 - \frac{\binom{6}{k}}{\binom{8}{k}}
    $$

    For $k=1$:

    $$
    \text{pass@1} = 1 - \frac{\binom{6}{1}}{\binom{8}{1}} = 1 - \frac{6}{8} = 0.25
    $$

    For $k=2$:

    $$
    \text{pass@2} = 1 - \frac{\binom{6}{2}}{\binom{8}{2}} = 1 - \frac{15}{28} \approx 0.464
    $$

    For $k=4$:

    $$
    \text{pass@4} = 1 - \frac{\binom{6}{4}}{\binom{8}{4}} = 1 - \frac{15}{70} \approx 0.786
    $$

    Interpretation: if we run 4 independent samples and pick the correct one (assuming we have a verifier), we'd succeed about 79% of the time on this task, compared to 25% for a single sample. This is a strong argument for investing in fast test execution that enables multi-sample selection.

```python
import math
from typing import Sequence

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.
    
    n: total samples drawn per task
    c: number of correct samples among n
    k: number of samples to select (k <= n)
    
    Returns probability that at least one of k samples is correct.
    Uses the combinatorial formula from the Codex paper (Chen et al. 2021).
    """
    if n < k:
        raise ValueError(f"Cannot compute pass@{k} with only {n} samples")
    if c == 0:
        return 0.0
    if c == n:
        return 1.0
    # 1 - P(all k selected are wrong) = 1 - C(n-c, k) / C(n, k)
    # Compute in log space for numerical stability with large n
    log_num = sum(math.log(n - c - i) for i in range(k))
    log_den = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_num - log_den)

def aggregate_pass_at_k(
    results: Sequence[tuple[int, int]],  # list of (n, c) per task
    k: int
) -> float:
    """
    Aggregate pass@k over multiple tasks (unweighted mean).
    results: list of (n_samples, n_correct) per task.
    """
    scores = [pass_at_k(n, c, k) for n, c in results]
    return sum(scores) / len(scores)

# Example: 500 tasks, each run 8 times
import random
random.seed(42)
task_results = [(8, random.randint(0, 3)) for _ in range(500)]

for k in [1, 2, 4, 8]:
    score = aggregate_pass_at_k(task_results, k)
    print(f"pass@{k}: {score:.3f}")
# Example output (will vary by random seed):
# pass@1: 0.188
# pass@2: 0.340
# pass@4: 0.560
# pass@8: 0.737
```

{{fig:ageval-pass-at-k}}

## Harness Effects on Scores

The harness — the scaffolding around the model — is not a neutral measurement instrument. It is a design choice that can swing scores dramatically. This section catalogs the main harness dimensions and their approximate effects.

### File Localization

SWE-bench tasks involve large codebases (often hundreds of files). How the agent identifies which files to edit affects performance:

- **Oracle localization:** Tell the agent exactly which files need editing (upper bound on localization). Typically lifts scores by 10–20 percentage points on SWE-bench.
- **BM25 retrieval:** Rank files by keyword similarity to the issue, pass top-N. Cheaper but noisy.
- **Agent self-localization:** Let the agent explore the repo (via `find`, `grep`, code search) to identify relevant files. Most realistic but most expensive.

### Iteration Budget

Most agents improve when allowed to observe test failures and retry. Giving an agent 3 edit-run-test loops instead of 1 typically improves SWE-bench resolve rate by several percentage points. The diminishing returns curve matters: going from 1 to 3 iterations helps; going from 10 to 30 iterations rarely does.

### Tool Suite

The set of available tools shapes what strategies are even possible. An agent with only `read_file` + `edit_file` cannot run tests mid-trajectory; it must produce a correct patch in one shot. Adding `run_command` (bash execution) enables iterative debugging but also increases trajectory length and cost.

### Context Truncation Policy

Long codebases don't fit in context. The policy for truncating — keep first N lines, summarize, or chunk and retrieve — interacts with the model's positional bias. See [Context Engineering & Management](../08-agents-harness/04-context-engineering.html) for the mechanics.

### System Prompt Engineering

Instruction format affects whether the model produces valid `git diff` output, uses tools correctly, or stalls in thought loops. Prompting choices that add or remove a single sentence about output format have caused observed score shifts of 2–5 percentage points.

!!! warning "Common pitfall: reporting scores without harness disclosure"

    A leaderboard entry that says "Model X achieves 35% on SWE-bench Verified" is nearly uninterpretable without knowing the harness. Always report: (1) oracle vs. retrieved file localization, (2) maximum iterations, (3) tools available, (4) scaffold name and version. Without these, comparison across entries is meaningless.

{{fig:ageval-harness-swing}}

## Reproducibility and Variance

Agent benchmarks have notoriously high variance. The main sources are:

**Sampling stochasticity.** Even at temperature 0, results may differ across runs due to non-deterministic CUDA kernels, API load balancing, or batch-size-dependent numerics. For small task sets, a ±2% difference can be noise.

**Environment instability.** Web environments spin up Docker containers; container startup time, network latency, and website version differences all introduce variance. WebArena's sandboxed design reduces this, but not entirely.

**Test suite fragility.** Some SWE-bench test suites are flaky — tests that pass and fail non-deterministically regardless of the patch. The Verified subset was curated to reduce this, but it persists in the full set.

### Confidence Intervals for Agent Scores

For a binary outcome metric on $N$ tasks, the standard error of the proportion is:

$$
\text{SE} = \sqrt{\frac{p(1-p)}{N}}
$$

where $p$ is the observed resolve rate. A 95% confidence interval is approximately $p \pm 1.96 \cdot \text{SE}$.

For $N = 500$ (SWE-bench Verified size) and $p = 0.30$:

$$
\text{SE} = \sqrt{\frac{0.30 \times 0.70}{500}} = \sqrt{0.00042} \approx 0.020
$$

So the 95% CI is roughly $[0.26, 0.34]$. A reported improvement from 30% to 33% is statistically indistinguishable from noise at this sample size. This is alarming for a field that regularly claims "state-of-the-art" improvements of 1–3 percentage points.

The correct response is to run multiple seeds, report CIs, and use paired tests (McNemar's test for the same task set) rather than comparing raw percentages.

```python
import numpy as np
from scipy import stats

def agent_benchmark_ci(n_tasks: int, n_correct: int, confidence: float = 0.95) -> dict:
    """
    Wilson score interval for a binary agent benchmark.
    More accurate than normal approximation at extreme proportions.
    """
    p_hat = n_correct / n_tasks
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)  # e.g. 1.96 for 95% CI
    
    # Wilson score interval
    denom = 1 + z**2 / n_tasks
    center = (p_hat + z**2 / (2 * n_tasks)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_tasks + z**2 / (4 * n_tasks**2))) / denom
    
    return {
        "estimate": p_hat,
        "ci_lower": max(0.0, center - margin),
        "ci_upper": min(1.0, center + margin),
        "n": n_tasks,
        "n_correct": n_correct,
    }

def mcnemar_test(outcomes_a: list[bool], outcomes_b: list[bool]) -> dict:
    """
    McNemar's test for paired binary outcomes.
    Tests whether model A and model B have significantly different success rates
    on the SAME task set (the right comparison, vs. unpaired z-test).
    
    outcomes_a[i], outcomes_b[i]: True/False for task i under model A/B.
    """
    assert len(outcomes_a) == len(outcomes_b)
    # Count concordant and discordant pairs
    b = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if a and not bb)   # A passes, B fails
    c = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if not a and bb)   # B passes, A fails
    
    # McNemar's statistic (with continuity correction)
    chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0.0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        "A_only": b,    # tasks only A solves
        "B_only": c,    # tasks only B solves
        "chi2": chi2,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
    }

# Example: 500 tasks, model A gets 160 right, model B gets 175 right
# but 30 are tasks A solves that B doesn't, and 45 are tasks B solves that A doesn't
outcomes_a = [True] * 130 + [True] * 30 + [False] * 45 + [False] * 295
outcomes_b = [True] * 130 + [False] * 30 + [True] * 45 + [False] * 295
result = mcnemar_test(outcomes_a, outcomes_b)
print(result)
# {"A_only": 30, "B_only": 45, "chi2": ..., "p_value": ..., "significant_at_05": ...}
```

{{fig:ageval-score-noise-band}}

## Contamination in Agent Benchmarks

Data contamination — the presence of benchmark tasks or solutions in pretraining data — is a documented problem for LLM benchmarks and is *worse* for agent benchmarks for several reasons.

**SWE-bench and GitHub overlap.** SWE-bench tasks are real GitHub issues and PRs. The merged patch is public on GitHub, often in the training crawl of every major model. Even if the issue text isn't in training, the diff is. The question is whether the model is *using* that memory or solving the problem fresh.

**Detection methods and their limits.**

1. *N-gram overlap detection* (Membership Inference, Min-K% Prob): Check whether the test instances appear verbatim in training. Works for exact matches, fails for paraphrased or semantically equivalent content.
2. *Temporal splits*: Only use issues filed and resolved after a model's training cutoff. SWE-bench Verified includes recency filtering, but models may still have seen the PR through commit history.
3. *Differential perturbation*: Create modified versions of the task (rename variables, change error message) and check if the model's solve rate drops. A large drop suggests memorization; robustness suggests generalization.
4. *Canary insertion*: Insert synthetic "planted" tasks into the benchmark and check if any model exhibits disproportionately high solve rates on them.

**Practical guidance.** For any claimed state-of-the-art result on an agent benchmark:
- Check the model's knowledge cutoff against the benchmark's task date range.
- Prefer benchmarks that release task IDs but not task content publicly (held-out test sets).
- Request solve-rate stratified by task creation date: if the model does disproportionately better on older tasks, contamination is likely.

!!! note "SWE-bench Verified's approach to contamination"

    The Verified split was annotated in mid-2024. Some providers filter out SWE-bench task IDs from training. However, since the filter operates on identifiers (not semantic content), and since many tasks appear in model-training web crawls through Stack Overflow answers, blog posts, and pull request discussions, contamination is difficult to fully eliminate.

## Measuring Agentic Progress: What the Numbers Actually Say

Stepping back, what does the trajectory of agent benchmark scores tell us about real progress?

**SWE-bench as a case study.** In mid-2023, the best published resolve rates on SWE-bench were around 3–5% (a single model without retrieval). By early 2025, leaderboard-leading entries using multi-agent scaffolds and oracle file localization were reporting rates on the order of 40–50% on SWE-bench Verified. That is a genuine capability jump — the tasks are real software engineering problems and the evaluation is objective.

But much of the improvement came from scaffolding, not just the base model. The signal is real, but it is a *system* signal: (model + harness + compute budget) rather than model-in-isolation.

**Metrics that capture system capability honestly** should include:

- **Solve rate at fixed compute budget** (e.g., 2 × 10^6 tokens per task): forces apples-to-apples comparison.
- **Solve rate with and without oracle localization**: isolates code-generation ability from retrieval ability.
- **Cost per solved task** (inference API cost): increasingly important for production deployment.

**The floor and ceiling problem.** Benchmarks saturate. When any single benchmark approaches 70–80% solve rates, the remaining tasks are either pathologically hard, noisy, or require capabilities genuinely absent from current models. The field regularly needs new, harder benchmarks. terminal-bench and SciCode are recent examples of raising the difficulty ceiling.

See [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html) for broader context on how agent benchmarks fit into the full evaluation landscape, and [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) for the meta-question of what benchmarks are actually measuring.

!!! interview "Interview Corner"

    **Q:** SWE-bench Verified shows Model A achieves 40% and Model B achieves 37%. A colleague says Model A is strictly better. What would you push back on?

    **A:** Several important caveats: First, with 500 tasks and the observed proportion near 0.40, the 95% confidence interval is roughly ±4 points, so the difference may not be statistically significant. Use McNemar's test on paired task outcomes (same task set) rather than comparing raw percentages. Second, the scores are harness-dependent: ask whether both models used the same scaffold, the same file localization strategy, the same iteration budget, and the same tools. A difference in oracle-vs-retrieved localization alone can explain a 3-point gap. Third, check for contamination: if Model A's training cutoff is closer to the SWE-bench task dates, it may have seen more of the benchmark data. Fourth, consider stratified performance: maybe Model A is much better on Django tasks and worse on everything else. A single aggregate number obscures capability profiles. None of this means A isn't better — it might well be — but the claim requires more evidence than a 3-point gap on a single number.

## Building Trustworthy Agent Evals in Practice

If you are building an agent system and need reliable internal evaluation, the public benchmarks are a starting point but not a complete answer. Here is a practitioner checklist:

**Define the right unit of measurement.** What matters to your users: per-task success rate, multi-turn session completion rate, time to resolution, or user satisfaction? Pick the metric before you look at any results.

**Use a held-out task set.** Once your team has seen the tasks during development, those tasks are contaminated for evaluation purposes. Create a locked-down holdout set that no one on the team inspects until the final evaluation.

**Run multiple seeds.** Even at temperature 0, environmental non-determinism exists. Run at least 3–5 independent seeds per task for your primary metric, and report the mean and standard deviation.

**Ablate your harness.** Run your model with and without each harness feature (retrieval, iteration, tools). This tells you what the model contributes versus the scaffold.

**Track cost alongside quality.** A system that costs \$50 per solved task and one that costs \$5 per solved task are not equivalent even if they have the same solve rate. Cost-normalized solve rate (tasks solved per dollar) is a legitimate production metric.

**Automate the evaluator.** Human evaluation is gold-standard but does not scale. The gold standard for agentic eval is automated state-checking (does the final system state match the goal?), not model-based judging. Reserve LLM-as-a-judge (see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html)) for tasks where no automated checker exists.

```python
# Minimal reproducible eval harness with seeding and logging

import json, hashlib, datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable, Any

@dataclass
class EvalConfig:
    benchmark: str         # e.g. "swe-bench-verified"
    model: str             # e.g. "gpt-4o-2024-08-06"
    scaffold: str          # e.g. "acr-v2.1"
    n_seeds: int = 3
    max_iterations: int = 5
    file_localization: str = "bm25-top10"  # or "oracle", "agent-only"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat()

    def fingerprint(self) -> str:
        """Stable hash of this config for deduplication."""
        s = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:12]


def run_eval(
    config: EvalConfig,
    tasks: list[dict],
    agent_fn: Callable[[dict, int], bool],  # (task, seed) -> success
    output_dir: Path,
) -> dict:
    """
    Run an agent eval with multiple seeds, log results, and compute statistics.
    
    agent_fn: your agent callable. Takes a task dict and a random seed int,
              returns True if the task was resolved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = config.fingerprint()
    results_path = output_dir / f"eval_{fp}.jsonl"

    all_per_task = []  # list of (n_correct, n_total) per task

    with open(results_path, "w") as out_f:
        for task in tasks:
            task_results = []
            for seed in range(config.n_seeds):
                success = agent_fn(task, seed)
                record = {
                    "task_id": task["id"],
                    "seed": seed,
                    "success": success,
                    "config_fingerprint": fp,
                }
                out_f.write(json.dumps(record) + "\n")
                task_results.append(success)

            n_correct = sum(task_results)
            all_per_task.append((config.n_seeds, n_correct))

    # Aggregate
    pass1 = aggregate_pass_at_k(all_per_task, k=1)
    passN = aggregate_pass_at_k(all_per_task, k=config.n_seeds)
    ci = agent_benchmark_ci(len(tasks), int(pass1 * len(tasks)))

    summary = {
        "config": asdict(config),
        "n_tasks": len(tasks),
        "pass_at_1": pass1,
        f"pass_at_{config.n_seeds}": passN,
        "ci_95_lower": ci["ci_lower"],
        "ci_95_upper": ci["ci_upper"],
        "results_file": str(results_path),
    }

    summary_path = output_dir / f"summary_{fp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
```

## Summary: The Benchmark Landscape at a Glance

```text
┌────────────────┬──────────────┬──────────────────┬─────────────────────────┐
│  Benchmark     │  Domain      │  Metric          │  Key Design Feature     │
├────────────────┼──────────────┼──────────────────┼─────────────────────────┤
│ SWE-bench      │ Code/GitHub  │ resolve rate     │ Real issues, unit tests │
│ SWE-bench      │ Code/GitHub  │ resolve rate     │ Human-verified subset   │
│   Verified     │              │                  │ (≈500 tasks)            │
│ WebArena       │ Web browsing │ success rate     │ Sandboxed web apps      │
│ GAIA           │ General AI   │ exact-match acc  │ 3 difficulty levels     │
│ tau-bench      │ Customer svc │ resolve + policy │ Simulated user partner  │
│ terminal-bench │ Bash/Linux   │ success rate     │ Docker-isolated shell   │
│ AgentBench     │ Multi-domain │ mean success     │ 8 environments          │
│ OSWorld        │ Desktop GUI  │ success rate     │ Screenshot-based        │
└────────────────┴──────────────┴──────────────────┴─────────────────────────┘
```

Progress on these benchmarks reflects real capability improvements in the underlying models and scaffolds, but interpreting that progress requires understanding harness effects, statistical uncertainty, and contamination risk. The tools in this chapter — pass@k estimators, confidence intervals, McNemar's test, harness ablations — are not academic exercises; they are the difference between knowing whether your agent actually got better and just hoping it did.

For how these evaluations connect to training, see [Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html). For tool-calling fundamentals that underpin all agentic benchmarks, see [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html). For the full eval framework that governs how these benchmarks fit into broader LLM evaluation, see [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html).

!!! key "Key Takeaways"
    - SWE-bench (especially the Verified subset) is the standard for coding agent evaluation: agents produce git patches graded by whether the task test suite passes.
    - WebArena tests web navigation, GAIA tests general multi-tool reasoning, tau-bench tests conversational service agents, and terminal-bench tests raw shell proficiency — each isolating a different capability.
    - Outcome scoring (binary pass/fail) is the norm; trajectory scoring provides richer diagnostics but requires annotated demonstrations or process reward models.
    - pass@k is the correct estimator when you run multiple samples: $\text{pass@}k = 1 - \binom{n-c}{k}/\binom{n}{k}$, unbiased and numerically stable.
    - Harness choices — file localization, iteration budget, tools available, context truncation — routinely shift SWE-bench scores by 10–20 percentage points, making harness disclosure mandatory for fair comparison.
    - With 500 tasks and ~30% solve rate, a 95% CI spans about ±4 points; a 3-point improvement may be noise. Use McNemar's paired test and report CIs, not just point estimates.
    - Contamination is a structural risk: SWE-bench tasks live on GitHub and may appear in training crawls; prefer recency-filtered splits and stratify results by task creation date.
    - In production, augment pass@k with cost-normalized metrics (tasks solved per dollar) and ablation studies that separate model contribution from scaffold contribution.

!!! sota "State of the Art & Resources (2026)"
    Agent evaluation has matured rapidly: SWE-bench Verified scores rose from under 2% in late 2023 to over 75% by 2026, driven by both stronger base models and scaffold engineering. The field is now converging on richer metrics — cost-normalized solve rates, harness-ablated comparisons, and multi-domain benchmarks — to separate genuine capability gains from scaffolding and contamination effects.

    **Foundational work**

    - [Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* (2023)](https://arxiv.org/abs/2310.06770) — the canonical coding-agent benchmark; each task is a real GitHub issue graded by whether the agent's patch passes the test suite.
    - [Chen et al., *Evaluating Large Language Models Trained on Code* (Codex, 2021)](https://arxiv.org/abs/2107.03374) — introduced the unbiased pass@k estimator used by virtually every subsequent agent benchmark.

    **Recent advances (2023–2026)**

    - [Zhou et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents* (2023)](https://arxiv.org/abs/2307.13854) — sandboxed web benchmark across e-commerce, GitLab, and forum tasks; state-based functional evaluation.
    - [Mialon et al., *GAIA: a benchmark for General AI Assistants* (2023)](https://arxiv.org/abs/2311.12983) — three-level benchmark requiring multi-hop tool use with exact-match grading; humans score 92% vs. ~15% for GPT-4 with plugins.
    - [Yao et al., *τ-bench: Tool-Agent-User Interaction in Real-World Domains* (2024)](https://arxiv.org/abs/2406.12045) — simulated customer-service benchmark with policy-compliance scoring and a multi-turn pass^k metric.
    - [Xie et al., *OSWorld: Benchmarking Multimodal Agents in Real Computer Environments* (2024)](https://arxiv.org/abs/2404.07972) — 369 tasks spanning Ubuntu, Windows, and macOS; best model achieves ~12% vs. 72% human performance.
    - [Yehudai et al., *Survey on Evaluation of LLM-based Agents* (2025)](https://arxiv.org/abs/2503.16416) — comprehensive survey across five evaluation dimensions: core capabilities, application benchmarks, generalist agents, benchmark analysis, and evaluation frameworks.

    **Open-source & tools**

    - [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench) — official Docker-based evaluation harness for SWE-bench and SWE-bench Verified; includes dataset, inference scripts, and the sb-cli cloud runner.
    - [web-arena-x/webarena](https://github.com/web-arena-x/webarena) — self-hostable web environment with 812 tasks across sandboxed web apps; Playwright-based browser automation.
    - [THUDM/AgentBench](https://github.com/THUDM/AgentBench) — multi-environment evaluation suite covering OS, database, knowledge-graph, web shopping, and household tasks (ICLR 2024).

    **Go deeper**

    - [SWE-bench Official Leaderboard](https://www.swebench.com/) — live rankings across Verified, Lite, Multilingual, and Multimodal splits with harness and model filters.

## Further Reading

- Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2023) — original SWE-bench paper.
- Chowdhury et al., "SWE-bench Verified" (2024) — the human-verified subset methodology.
- Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents" (2023).
- Mialon et al., "GAIA: A Benchmark for General AI Assistants" (2023).
- Yao et al., "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" (2024).
- Chen et al., "Evaluating Large Language Models Trained on Code" (Codex, 2021) — introduced the pass@k estimator.
- Liu et al., "AgentBench: Evaluating LLMs as Agents" (2023) — multi-environment benchmark.
- Xie et al., "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments" (2024).
- SWE-bench leaderboard and harness code: github.com/princeton-nlp/SWE-bench
