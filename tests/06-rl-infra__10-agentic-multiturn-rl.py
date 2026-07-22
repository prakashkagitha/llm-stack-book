"""
Executable extraction of the CPU-runnable code blocks from
content/06-rl-infra/10-agentic-multiturn-rl.md

Blocks tested (chapter's own numbering):
  #0 (line ~25,  40 lines) - Role / Segment / Trajectory dataclasses,
      including Trajectory.flatten() (the loss-mask construction).
  #1 (line ~85,  69 lines) - Environment Protocol + the `rollout()` loop
      that interleaves policy generation and environment steps.
  #2 (line ~186, 36 lines) - masked_grpo_loss(): the token-level clipped
      surrogate with a loss mask, including the next-token shift.
  #3 (line ~246, 15 lines) - assign_trajectory_advantage(): GRPO-style
      group-relative advantage at the trajectory level. The harness
      flags this block as a "fragment", but it is in fact a complete,
      trivially CPU-safe function (it only needs Trajectory objects
      from block #0), so we exercise it too as bonus coverage.
  #4 (line ~307, 59 lines) - SearchQAEnv: a complete ReAct-style
      search-and-answer environment (reset/step/_match).

Blocks explicitly SKIPPED:
  # SKIP(needs-gpu): block #5 (line ~416, agentic_grpo_step +
    masked_token_kl) calls `.cuda()` on every batched tensor and is the
    full training step wired to an FSDP-wrapped `policy`/`ref_policy`
    and a real optimizer. None of that is CPU-runnable or a toy-fixture
    fit; it is defined nowhere in this file since it adds no additional
    *logic* beyond block #2's masked loss plus a KL term (masked_token_kl
    mirrors masked_grpo_loss's shift-and-mask pattern, already verified
    by block #2's off-by-one test below).

Since `rollout()` (block #1) needs a policy_engine and tokenizer, and the
chapter itself supplies a complete, real Environment implementation
(SearchQAEnv, block #4), we wire the two together in the block #1 test:
a small scripted FakePolicyEngine + FakeTokenizer stand in only for the
external vLLM/SGLang engine and HF tokenizer (never mocking the book's
own trajectory/environment logic), and `rollout()` drives a real
SearchQAEnv instance end-to-end, producing a real `Trajectory` that is
then flattened (block #0) to check the loss mask.

No real bugs were found in this chapter's code; all four blocks ran
correctly as printed in the book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import re

import torch
import torch.nn.functional as F


# =====================================================================
# Block #0 (line ~25) - Trajectory data structure
# =====================================================================

class Role(Enum):
    SYSTEM = "system"          # task description, tool schemas (no loss)
    ASSISTANT = "assistant"    # model-generated action (LOSS — this is the policy)
    OBSERVATION = "observation"  # tool output / env feedback (no loss, MASKED)


@dataclass
class Segment:
    role: Role
    token_ids: list[int]       # tokens for this segment
    logprobs: list[float] | None = None  # only for ASSISTANT segments (from rollout)


@dataclass
class Trajectory:
    segments: list[Segment] = field(default_factory=list)
    reward: float = 0.0                 # terminal reward
    turn_rewards: list[float] | None = None  # optional per-turn dense rewards
    metadata: dict = field(default_factory=dict)  # task_id, success flag, n_turns...

    def flatten(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate all segments into one token sequence plus a 0/1
        'loss mask' that is 1 exactly on ASSISTANT tokens. This mask is
        the single most important object in agentic RL — get it wrong
        and you train the model to predict tool outputs (catastrophic).
        """
        all_ids: list[int] = []
        loss_mask: list[int] = []
        for seg in self.segments:
            all_ids.extend(seg.token_ids)
            is_action = 1 if seg.role == Role.ASSISTANT else 0
            loss_mask.extend([is_action] * len(seg.token_ids))
        return (
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )


def test_block_0_trajectory():
    traj = Trajectory()
    traj.segments.append(Segment(Role.SYSTEM, [1, 2, 3]))
    traj.segments.append(Segment(Role.ASSISTANT, [4, 5], logprobs=[-0.1, -0.2]))
    traj.segments.append(Segment(Role.OBSERVATION, [6, 7, 8, 9]))
    traj.segments.append(Segment(Role.ASSISTANT, [10], logprobs=[-0.3]))
    traj.reward = 1.0

    ids, mask = traj.flatten()
    assert ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    assert mask.sum().item() == 3  # exactly the ASSISTANT tokens
    assert mask.dtype == torch.float

    print("[block #0] Trajectory/Segment/flatten(): OK")


# =====================================================================
# Block #1 (line ~85) - Environment Protocol + rollout()
# =====================================================================

class Environment(Protocol):
    """RAGEN/Gym-style environment interface for tool-using agents."""
    def reset(self, task) -> str:
        """Return the initial prompt (system + task) as a string."""
        ...

    def step(self, action_text: str) -> tuple[str, float, bool]:
        """
        Execute the model's action against the world.
        Returns (observation_text, turn_reward, done).
        - observation_text: what the model sees next (tool result, error, etc.)
        - turn_reward: dense per-turn reward (often 0.0 until terminal)
        - done: True if the episode has ended (success, failure, or give-up)
        """
        ...


def rollout(policy_engine, tokenizer, env: Environment, task,
            max_turns: int = 8, max_action_tokens: int = 512) -> Trajectory:
    """
    Run one agentic episode and record a fully-tagged Trajectory.
    `policy_engine.generate` returns (text, token_ids, logprobs) and STOPS
    at a tool-call boundary (a stop string like '</tool_call>' or an EOS).
    """
    traj = Trajectory()
    prompt = env.reset(task)

    # The initial prompt is context the model conditions on but never produced:
    # role=SYSTEM => loss_mask 0.
    sys_ids = tokenizer.encode(prompt)
    traj.segments.append(Segment(Role.SYSTEM, sys_ids))

    turn_rewards: list[float] = []
    running_text = prompt

    for turn in range(max_turns):
        # ---- 1. GENERATE an action (on-policy, differentiable) ----
        text, action_ids, logprobs = policy_engine.generate(
            running_text,
            max_tokens=max_action_tokens,
            stop=["</tool_call>", tokenizer.eos_token],
        )
        traj.segments.append(Segment(Role.ASSISTANT, action_ids, logprobs))
        running_text += text

        # ---- 2. STEP the environment with the action ----
        obs_text, turn_reward, done = env.step(text)
        turn_rewards.append(turn_reward)

        if done:
            traj.reward = sum(turn_rewards)          # terminal credit
            traj.turn_rewards = turn_rewards
            traj.metadata["n_turns"] = turn + 1
            return traj

        # ---- 3. INJECT the observation (NOT generated => loss_mask 0) ----
        # Wrap so the model can parse it, e.g. <tool_response>...</tool_response>
        obs_wrapped = f"<tool_response>\n{obs_text}\n</tool_response>"
        obs_ids = tokenizer.encode(obs_wrapped)
        traj.segments.append(Segment(Role.OBSERVATION, obs_ids))
        running_text += obs_wrapped

    # Hit max_turns without finishing: episode truncated.
    traj.reward = sum(turn_rewards)
    traj.turn_rewards = turn_rewards
    traj.metadata["n_turns"] = max_turns
    traj.metadata["truncated"] = True
    return traj


# ---- Test fixtures for block #1 (fake inference engine + tokenizer) ----
# These stand in ONLY for the external vLLM/SGLang engine and HF tokenizer.
# The environment used below (SearchQAEnv, block #4) and the rollout()
# function itself are the book's real, unmodified code.

class FakeTokenizer:
    """Deterministic whitespace tokenizer standing in for a real HF tokenizer."""
    eos_token = "<eos>"

    def encode(self, text: str) -> list[int]:
        return [abs(hash(tok)) % 5000 for tok in text.split()]


class FakePolicyEngine:
    """
    Deterministic stand-in for a vLLM/SGLang inference engine: replays a
    fixed script of actions instead of actually sampling from a model.
    """

    def __init__(self, tokenizer, script: list[str]):
        self.tokenizer = tokenizer
        self.script = list(script)
        self._i = 0

    def generate(self, running_text, max_tokens=512, stop=None):
        text = self.script[self._i]
        self._i += 1
        ids = self.tokenizer.encode(text)
        logprobs = [-0.1] * len(ids)
        return text, ids, logprobs


def test_block_1_rollout():
    tokenizer = FakeTokenizer()
    retriever = lambda q: [f"passage about {q} says the answer is 42", "irrelevant passage"]
    env = SearchQAEnv(retriever=retriever, max_turns=6, step_penalty=0.02)

    policy_engine = FakePolicyEngine(tokenizer, script=[
        "Thought: I should search first. <search>meaning of life</search>",
        "Thought: Now I know it. <answer>42</answer>",
    ])
    task = {"question": "What is the meaning of life?", "answer": "42"}

    traj = rollout(policy_engine, tokenizer, env, task, max_turns=6, max_action_tokens=64)

    assert isinstance(traj, Trajectory)
    assert traj.metadata["n_turns"] == 2
    assert not traj.metadata.get("truncated", False)
    # turn 0: search -> step_cost only (-0.02); turn 1: correct answer -> 1.0 - 0.02
    assert abs(traj.reward - (-0.02 + 0.98)) < 1e-9, traj.reward
    assert traj.turn_rewards is not None and len(traj.turn_rewards) == 2

    # segments: SYSTEM, ASSISTANT(search), OBSERVATION, ASSISTANT(answer) — no
    # trailing observation because the episode terminates on the 2nd action.
    roles = [s.role for s in traj.segments]
    assert roles == [Role.SYSTEM, Role.ASSISTANT, Role.OBSERVATION, Role.ASSISTANT]

    ids, mask = traj.flatten()
    assert ids.shape == mask.shape
    action_len = sum(len(s.token_ids) for s in traj.segments if s.role == Role.ASSISTANT)
    assert mask.sum().item() == action_len

    print("[block #1] rollout(): OK")


# =====================================================================
# Block #2 (line ~186) - masked_grpo_loss
# =====================================================================

def masked_grpo_loss(logits, token_ids, loss_mask, advantages,
                     old_logprobs, epsilon=0.2):
    """
    Token-level clipped surrogate with a loss mask.

    logits:       [B, L, V] current-policy logits (teacher-forced on the trajectory)
    token_ids:    [B, L]    the realized tokens (actions AND observations)
    loss_mask:    [B, L]    1.0 on action tokens, 0.0 on observation/system tokens
    advantages:   [B, L]    per-token advantage (broadcast from trajectory or turn)
    old_logprobs: [B, L]    log-probs recorded at rollout time (behaviour policy)
    """
    # Shift for next-token prediction: logits at position i predict token i+1.
    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)        # [B, L-1, V]
    target = token_ids[:, 1:]                                   # [B, L-1]
    new_logprobs = torch.gather(
        logp_all, dim=-1, index=target.unsqueeze(-1)
    ).squeeze(-1)                                               # [B, L-1]

    # Align the other tensors to the shifted positions.
    mask = loss_mask[:, 1:]
    adv  = advantages[:, 1:]
    old  = old_logprobs[:, 1:]

    ratio = torch.exp(new_logprobs - old)                      # ρ_i
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    per_token_loss = -torch.min(unclipped, clipped)            # PPO surrogate

    # Mask out observation/system tokens, then average over ACTION tokens only.
    denom = mask.sum().clamp(min=1.0)
    loss = (per_token_loss * mask).sum() / denom
    return loss


def test_block_2_masked_grpo_loss():
    torch.manual_seed(0)
    V = 5
    logits = torch.randn(1, 6, V)
    token_ids = torch.randint(0, V, (1, 6))
    # SYS(2), ACTION(2), OBS(2)
    loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    advantages = loss_mask * 2.0

    with torch.no_grad():
        logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)
        target = token_ids[:, 1:]
        old_lp_shifted = torch.gather(
            logp_all, dim=-1, index=target.unsqueeze(-1)
        ).squeeze(-1)
    # old_logprobs must be full length L (function itself shifts it).
    old_logprobs = torch.cat([torch.zeros(1, 1), old_lp_shifted], dim=1)

    loss1 = masked_grpo_loss(logits, token_ids, loss_mask, advantages, old_logprobs)
    assert torch.isfinite(loss1)

    # Position index 0 in `logits` (pre-shift) predicts target index 1, which
    # has loss_mask 0 (system) -> perturbing it must NOT move the loss.
    # This directly checks the off-by-one pitfall the book calls out.
    # (Perturb a single vocab entry, not the whole row: log_softmax is
    # shift-invariant to adding a constant across the whole row, so that
    # would be a no-op regardless of masking and wouldn't test anything.)
    logits_a = logits.clone()
    logits_a[:, 0, 0] += 5.0
    loss_a = masked_grpo_loss(logits_a, token_ids, loss_mask, advantages, old_logprobs)
    assert torch.allclose(loss1, loss_a), "masked-out (post-shift) position changed the loss"

    # Position index 1 in `logits` predicts target index 2, which has
    # loss_mask 1 (action) -> perturbing it MUST move the loss.
    logits_b = logits.clone()
    logits_b[:, 1, 0] += 5.0
    loss_b = masked_grpo_loss(logits_b, token_ids, loss_mask, advantages, old_logprobs)
    assert not torch.allclose(loss1, loss_b), "masked-in (post-shift) position did not change the loss"

    print("[block #2] masked_grpo_loss(): OK")


# =====================================================================
# Block #3 (line ~246) - assign_trajectory_advantage
# Flagged "fragment" by the extraction heuristic, but it is a complete,
# trivially CPU-safe function over Trajectory objects (block #0); we
# exercise it as bonus coverage of the chapter's credit-assignment code.
# =====================================================================

def assign_trajectory_advantage(group: list[Trajectory], delta: float = 1e-6):
    """
    GRPO-style group-relative advantage at the TRAJECTORY level.
    All trajectories in `group` are rollouts for the SAME task/prompt.
    Returns each trajectory annotated with a scalar advantage that will
    be broadcast onto every one of its action tokens.
    """
    rewards = torch.tensor([t.reward for t in group])
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)
    for t, r in zip(group, rewards):
        adv = (r - mu) / (sigma + delta)
        t.metadata["advantage"] = adv.item()   # one number for the whole trajectory
    return group


def test_block_3_assign_trajectory_advantage():
    # The chapter's own worked example: G=4 trajectories, terminal rewards
    # [0.75, 0.50, 0.00, 0.25].
    rewards = [0.75, 0.50, 0.00, 0.25]
    group = [Trajectory(reward=r) for r in rewards]

    out = assign_trajectory_advantage(group)
    assert out is group  # returns the same list, mutated in place

    r = torch.tensor(rewards)
    mu = r.mean()
    sigma = r.std(unbiased=False)
    expected = (r - mu) / (sigma + 1e-6)

    for t, e in zip(group, expected):
        assert "advantage" in t.metadata
        assert abs(t.metadata["advantage"] - e.item()) < 1e-5

    # The best trajectory (reward 0.75) should be ~ +1.36 per the book's
    # worked example.
    assert abs(group[0].metadata["advantage"] - 1.36) < 0.05

    print("[block #3] assign_trajectory_advantage(): OK")


# =====================================================================
# Block #4 (line ~307) - SearchQAEnv
# =====================================================================

class SearchQAEnv:
    """
    A multi-turn ReAct environment: the agent must answer a question by
    issuing <search>query</search> calls and finally <answer>...</answer>.
    Terminal reward = 1.0 for a correct answer, with a small format bonus
    and a per-turn step penalty to discourage dithering.
    """
    SYSTEM_PROMPT = (
        "You are a research agent. To find information, emit "
        "<search>your query</search>. When you know the answer, emit "
        "<answer>your final answer</answer>. Think step by step before acting."
    )

    def __init__(self, retriever, max_turns: int = 6, step_penalty: float = 0.02):
        self.retriever = retriever          # callable: query -> list[str] passages
        self.max_turns = max_turns
        self.step_penalty = step_penalty

    def reset(self, task: dict) -> str:
        # task = {"question": ..., "answer": "<gold>"}
        self.gold = task["answer"]
        self.turns = 0
        return f"{self.SYSTEM_PROMPT}\n\nQuestion: {task['question']}\n"

    def step(self, action_text: str) -> tuple[str, float, bool]:
        self.turns += 1
        step_cost = -self.step_penalty       # discourage long episodes

        # --- Parse a final answer first ---
        ans = re.search(r"<answer>(.*?)</answer>", action_text, re.DOTALL)
        if ans:
            pred = ans.group(1).strip()
            correct = self._match(pred, self.gold)
            reward = (1.0 if correct else 0.0) + step_cost
            return ("", reward, True)        # terminal

        # --- Parse a search action ---
        srch = re.search(r"<search>(.*?)</search>", action_text, re.DOTALL)
        if srch:
            query = srch.group(1).strip()
            passages = self.retriever(query)[:3]
            obs = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
            done = self.turns >= self.max_turns
            # If we ran out of turns mid-search, that's a failure (reward 0).
            return (obs if not done else "Out of turns.", step_cost, done)

        # --- Malformed action: tell the model, penalize lightly ---
        obs = ("Invalid action. Use <search>...</search> or "
               "<answer>...</answer>.")
        done = self.turns >= self.max_turns
        return (obs, step_cost, done)

    @staticmethod
    def _match(pred: str, gold: str) -> bool:
        norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
        return norm(gold) in norm(pred)      # lenient containment match


def test_block_4_search_qa_env():
    env = SearchQAEnv(retriever=lambda q: ["dummy passage"], max_turns=3, step_penalty=0.1)

    prompt = env.reset({"question": "What is 2+2?", "answer": "four"})
    assert "What is 2+2?" in prompt
    assert env.turns == 0

    # Malformed action -> observation, not an exception; small penalty; not done.
    obs, r, done = env.step("no tags here")
    assert not done
    assert "Invalid action" in obs
    assert abs(r - (-0.1)) < 1e-9

    # Search action -> retriever called, observation contains passage, not done.
    obs2, r2, done2 = env.step("<search>2+2</search>")
    assert not done2
    assert "dummy passage" in obs2
    assert abs(r2 - (-0.1)) < 1e-9

    # Correct (lenient) final answer -> terminal, positive reward net of step cost.
    obs3, r3, done3 = env.step("<answer>The answer is FOUR!</answer>")
    assert done3
    assert obs3 == ""
    assert abs(r3 - (1.0 - 0.1)) < 1e-9

    # Wrong answer -> terminal, reward is just the negative step cost.
    env2 = SearchQAEnv(retriever=lambda q: [], max_turns=3, step_penalty=0.1)
    env2.reset({"question": "Q", "answer": "42"})
    _, r4, done4 = env2.step("<answer>seven</answer>")
    assert done4
    assert abs(r4 - (-0.1)) < 1e-9

    print("[block #4] SearchQAEnv: OK")


# =====================================================================
# Block #5 (line ~416) - agentic_grpo_step + masked_token_kl — SKIP(needs-gpu)
# Calls .cuda() on every batched tensor and is wired to an FSDP-wrapped
# policy/ref_policy/optimizer; not CPU-runnable and not toy-fixture-able
# without rewriting its own logic. Not defined in this test file. Its
# masking pattern (shift + mask + normalize over action tokens) is the
# same one already verified end-to-end by block #2's off-by-one test.
# =====================================================================


# =====================================================================
# Run all tests
# =====================================================================

if __name__ == "__main__":
    test_block_0_trajectory()
    test_block_1_rollout()
    test_block_2_masked_grpo_loss()
    test_block_3_assign_trajectory_advantage()
    test_block_4_search_qa_env()
    print("\nAll CPU-runnable blocks in 06-rl-infra/10-agentic-multiturn-rl.md executed successfully.")
