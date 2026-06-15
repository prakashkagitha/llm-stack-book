# 8.2 The Agentic Loop: ReAct, Plan-Execute & Reflection

An LLM that only generates a single response is a brilliant encyclopedia. An LLM that can *act*, observe the results of those actions, and revise its plan is something qualitatively different: an autonomous agent. This chapter unpacks the machinery that makes the difference — the closed-loop architectures that turn a next-token predictor into a system that can write and execute code, browse the web, call APIs, and self-correct when things go wrong.

We will build from the simplest agentic pattern (observe–think–act) through to tree-search-based planning and systematic reflection, culminating in a minimal but complete ReAct agent you can run today. Throughout, we pay equal attention to the failure modes — infinite loops, derailment, and spurious tool calls — because understanding what breaks is half the engineering challenge.

Before diving in, make sure you are comfortable with [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html), which covers how individual tool calls work at the API level. For the memory systems that agents rely on to maintain state across many steps, see [Memory Systems for Agents](../08-agents-harness/05-agent-memory.html). For the multi-agent case where several agents collaborate, see [Multi-Agent Systems & Orchestration](../08-agents-harness/07-multi-agent-systems.html).

## The Core Problem: Why One-Shot Generation Fails

Consider asking a model to write a Python function, run the tests, fix the failures, and commit the result. A single-shot generation cannot do this: it does not know whether the tests passed. The model would have to hallucinate the outcome.

The pattern that breaks this deadlock is deceptively simple:

```text
while not done:
    action = model.think(context)
    observation = environment.execute(action)
    context += (action, observation)
```

This is the *agentic loop*. Its power comes from the fact that observations are real: a Python interpreter actually ran the code and returned a stack trace. The model now has ground-truth feedback that it could not have generated internally.

The agentic loop introduces a new set of engineering problems that do not exist in single-shot inference:

1. **Context growth.** Each iteration adds tokens. After $k$ steps the context is $O(k \cdot s)$ tokens where $s$ is the average step length. Context management strategies are covered in [Context Engineering & Management](../08-agents-harness/04-context-engineering.html).
2. **Stopping.** The model must decide *when* the task is finished, a question the architecture has to answer explicitly.
3. **Error propagation.** A wrong action early can send the agent down an unrecoverable path.
4. **Latency.** Each step is a full LLM forward pass plus a tool round trip. Ten steps at 2 seconds each is 20 seconds of wall-clock time.

The rest of this chapter examines the principal architectures that address these problems.

## ReAct: Interleaving Reasoning and Acting

ReAct (Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2023) is the foundational paper for modern agentic loops. The key insight is that purely acting (calling tools without explanation) and purely reasoning (chain-of-thought without any real execution) are both weaker than interleaving them.

### The ReAct Format

A ReAct agent produces alternating *Thought* and *Action* tokens, with *Observation* tokens injected by the harness after each action completes:

{{fig:agentloop-react-trace}}

The *Thought* traces are not sent to any tool — they are purely for the model's internal reasoning, influencing the next token but not executed. This is precisely the chain-of-thought mechanism from [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html), applied inside a loop.

### The ReAct Prompt Template

Getting the model to reliably produce this format requires a system prompt that establishes the convention and a few-shot examples:

```python
REACT_SYSTEM_PROMPT = """\
You are a helpful AI assistant that solves tasks by alternating between
Thought, Action, and Observation steps.

Format:
    Thought: <your reasoning about what to do next>
    Action: <tool_name>(<arguments>)
    Observation: <filled in by the system>
    ... (repeat as needed)
    Action: finish(<your final answer>)

Available tools:
{tool_descriptions}

Rules:
- Always think before acting.
- Actions must be exactly one tool call per step.
- When you have a final answer, call finish() — do not keep looping.
- If a tool returns an error, reason about the error and try a different approach.
"""
```

### Formal Loop Specification

Let $\mathcal{M}$ denote the language model, $\mathcal{T}$ the set of available tools, and $e_t$ the execution environment. At step $t$ the context is:

$$
c_t = [s_0, (a_1, o_1), (a_2, o_2), \ldots, (a_{t-1}, o_{t-1})]
$$

where $s_0$ is the system prompt + task, $a_i$ is the model's action (thought + tool call), and $o_i$ is the observation returned by the environment. The model generates:

$$
a_t \sim \mathcal{M}(\cdot \mid c_t)
$$

The environment produces:

$$
o_t = e_t(a_t)
$$

The loop terminates when $a_t$ calls `finish(·)` or when $t > T_{\max}$ (the hard step limit).

## A Minimal ReAct Agent: Full Implementation

{{fig:react-loop}}

Here is a complete, runnable ReAct agent that uses the OpenAI-compatible chat API (works with any provider supporting function/tool calling or raw text generation). We implement both a tool-calling variant and a raw-text variant for clarity.

```python
"""
minimal_react.py — A self-contained ReAct agent.

Requires:
    pip install openai>=1.0

Usage:
    OPENAI_API_KEY=sk-... python minimal_react.py
"""

import os
import re
import math
import json
import datetime
from typing import Callable

from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Tool registry
# ---------------------------------------------------------------------------

# Each tool is a plain Python function.  We store its name and a short
# description that goes into the system prompt.
TOOLS: dict[str, tuple[Callable, str]] = {}

def register_tool(description: str):
    """Decorator that registers a function as an agent tool."""
    def decorator(fn):
        TOOLS[fn.__name__] = (fn, description)
        return fn
    return decorator


@register_tool("Search the web for a query. Returns a short summary string.")
def search(query: str) -> str:
    # In production, call a real search API (e.g. Serper, Brave, Tavily).
    # Here we fake it for the purpose of the demo.
    fake_db = {
        "Tokyo population": "Tokyo metro ~37 million (2024).",
        "New York population": "NYC metro ~20 million (2024).",
        "speed of light": "299,792,458 metres per second.",
        "Python creator": "Guido van Rossum created Python.",
    }
    for key, value in fake_db.items():
        if key.lower() in query.lower():
            return value
    return f"No results found for: {query}"


@register_tool("Evaluate a Python arithmetic expression and return the result.")
def calculator(expression: str) -> str:
    # Restrict to safe arithmetic — no exec() on untrusted input.
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: only arithmetic expressions allowed."
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@register_tool("Return today's date.")
def get_date() -> str:
    return datetime.date.today().isoformat()


# ---------------------------------------------------------------------------
# 2. ReAct agent class
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    A minimal ReAct agent that uses raw text generation (no function-calling
    API) so the Thought/Action/Observation format is fully visible.
    """

    MAX_STEPS = 10       # hard safety limit
    STOP_TOKEN = "finish"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self._build_system_prompt()

    def _build_system_prompt(self) -> None:
        tool_descriptions = "\n".join(
            f"  - {name}({self._sig(fn)}): {desc}"
            for name, (fn, desc) in TOOLS.items()
        )
        self.system_prompt = f"""\
You are a helpful AI assistant that solves tasks step-by-step.
Use the following format EXACTLY:

Thought: <your reasoning>
Action: <tool_name>(<comma-separated arguments>)
...
(repeat Thought/Action until you have the answer)
Action: finish(<your final answer as a string>)

Available tools:
{tool_descriptions}

Important:
- Each Action line must contain exactly one tool call.
- Do NOT fabricate Observation lines — they will be injected for you.
- Stop immediately once you call finish().
"""

    @staticmethod
    def _sig(fn: Callable) -> str:
        """Return a compact signature string for a tool function."""
        import inspect
        sig = inspect.signature(fn)
        params = ", ".join(
            f"{p}: {v.annotation.__name__}"
            if v.annotation is not inspect.Parameter.empty
            else p
            for p, v in sig.parameters.items()
        )
        return params

    def _parse_action(self, text: str) -> tuple[str, list[str]] | None:
        """
        Extract the tool name and arguments from a line like:
            Action: search("Tokyo population")
        Returns (tool_name, [arg1, arg2, ...]) or None if no action found.
        """
        # Match lines starting with "Action:" then capture tool_name(args)
        match = re.search(r"Action:\s*(\w+)\(([^)]*)\)", text)
        if not match:
            return None
        tool_name = match.group(1)
        # Split args by comma and strip whitespace / surrounding quotes
        raw_args = match.group(2)
        args = [a.strip().strip('"\'') for a in raw_args.split(",") if a.strip()]
        return tool_name, args

    def _execute_action(self, tool_name: str, args: list[str]) -> str:
        """Call the registered tool and return its string output."""
        if tool_name == self.STOP_TOKEN:
            # finish() is a pseudo-tool handled by the loop.
            return "__FINISH__"
        if tool_name not in TOOLS:
            return f"Error: unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
        fn, _ = TOOLS[tool_name]
        try:
            result = fn(*args)
            return str(result)
        except TypeError as e:
            return f"Error calling {tool_name}: {e}"

    def run(self, task: str, verbose: bool = True) -> str:
        """
        Run the ReAct loop for a given task.
        Returns the final answer string.
        """
        # Conversation history: system + user task, then alternating
        # assistant (thought+action) and user (observation) messages.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": task},
        ]

        for step in range(self.MAX_STEPS):
            # ---- Model generation ----
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # Stop generation right after "Action: finish(...)"
                # We use a stop sequence to prevent the model generating
                # a fake Observation line.
                stop=["Observation:"],
                temperature=0.0,   # deterministic for agentic tasks
                max_tokens=512,
            )
            assistant_text = response.choices[0].message.content.strip()

            if verbose:
                print(f"\n[Step {step+1}]\n{assistant_text}")

            # ---- Parse action ----
            parsed = self._parse_action(assistant_text)
            if parsed is None:
                # Model produced text but no parseable Action — inject an
                # error observation to get it back on track.
                observation = "Error: no Action found. Please follow the format."
            else:
                tool_name, args = parsed
                observation = self._execute_action(tool_name, args)

            # ---- Check for finish ----
            if observation == "__FINISH__":
                # Extract the argument to finish() as the final answer.
                match = re.search(r"finish\(([^)]*)\)", assistant_text)
                final_answer = match.group(1).strip('"\'') if match else assistant_text
                if verbose:
                    print(f"\n[Final Answer] {final_answer}")
                return final_answer

            # ---- Append to context ----
            # Assistant turn includes the thought + action.
            messages.append({"role": "assistant", "content": assistant_text})
            # User turn carries the environment observation.
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

            if verbose:
                print(f"Observation: {observation}")

        # Exceeded MAX_STEPS
        return "Error: agent exceeded maximum steps without finishing."


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = ReActAgent(model="gpt-4o-mini")
    answer = agent.run(
        "How many times larger is Tokyo's metro population than New York's? "
        "Give me the ratio to 2 decimal places."
    )
    print(f"\nAnswer: {answer}")
```

The key implementation details to notice:

- **Stop sequence `"Observation:"`** prevents the model from hallucinating its own observations, a common failure mode. The harness injects the real observation instead.
- **`temperature=0.0`** for determinism — agentic tasks rarely benefit from high temperature, and stochastic actions make debugging much harder.
- **`MAX_STEPS`** is a hard safety valve. Without it, a buggy agent can run indefinitely.

!!! example "Worked example: ratio calculation"
    Given the task "How many times larger is Tokyo's metro population than New York's?", the agent might take the following steps (output abbreviated):

    1. `search("Tokyo population")` → "Tokyo metro ~37 million (2024)."
    2. `search("New York population")` → "NYC metro ~20 million (2024)."
    3. `calculator("37 / 20")` → "1.85"
    4. `finish("Tokyo's metro area is approximately 1.85x larger than New York's.")`

    Total steps: 4. Total tokens: roughly 600 input + 200 output across all calls. At gpt-4o-mini pricing (~\$0.15/M input, ~\$0.60/M output), this trace costs on the order of USD 0.0002 — essentially free per query, but note that a 50-step agent task at similar density would approach USD 0.01.

## Planning: Decomposition and Plan-Execute

ReAct is reactive — it decides each action one step at a time. For longer-horizon tasks, this is fragile: early actions may lack the context to be chosen correctly. *Plan-Execute* architectures (also called "planner-executor" or "hierarchical agents") separate the task into two phases:

1. **Plan phase.** A planning LLM (possibly the same model, possibly a more capable one) produces a structured plan: an ordered list of subtasks.
2. **Execute phase.** An executor agent carries out each subtask, potentially using its own inner ReAct loop.

{{fig:agentloop-plan-execute}}

### When to Plan vs React

| Criterion | ReAct (reactive) | Plan-Execute |
|---|---|---|
| Task length | Short (< 10 steps) | Long (10+ steps) |
| Subtask interdependence | Low | High — later steps depend on earlier |
| Parallelism | Sequential | Subtasks can sometimes run in parallel |
| Error recovery | Local (per step) | May require re-planning |
| Latency | Lower (no plan phase) | Higher (extra LLM call up front) |

### Structured Planning with JSON

Asking the model to output a plan as structured JSON makes it easier for the harness to parse and execute:

```python
PLAN_PROMPT = """\
Break the following task into an ordered list of subtasks.
Each subtask should be atomic enough to be completed in a single
focused ReAct session. Output valid JSON only.

Task: {task}

Output format:
{{
  "goal": "<restate the goal>",
  "steps": [
    {{"id": 1, "description": "<step>", "depends_on": []}},
    {{"id": 2, "description": "<step>", "depends_on": [1]}},
    ...
  ]
}}
"""
```

```python
import json

def make_plan(client, model: str, task: str) -> dict:
    """Ask the LLM to decompose a task into steps."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PLAN_PROMPT.format(task=task)}],
        response_format={"type": "json_object"},  # JSON mode
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


def execute_plan(agent: ReActAgent, plan: dict) -> dict[int, str]:
    """
    Execute each step in dependency order.
    Returns a dict mapping step id → result string.
    """
    results: dict[int, str] = {}

    for step in plan["steps"]:
        step_id = step["id"]
        desc = step["description"]
        deps = step.get("depends_on", [])

        # Build a context-enriched task that incorporates prior results
        prior_context = "\n".join(
            f"Step {d} result: {results[d]}"
            for d in deps
            if d in results
        )
        full_task = f"{desc}\n\nContext from prior steps:\n{prior_context}" if prior_context else desc

        print(f"\n=== Executing step {step_id}: {desc} ===")
        results[step_id] = agent.run(full_task, verbose=True)

    return results
```

The `depends_on` field enables a simple topological sort; steps with no dependencies can in principle run in parallel using `asyncio` or `concurrent.futures`.

## Reflection and Self-Correction

Even a well-planned agent will sometimes make mistakes — a wrong tool call, a misinterpreted observation, an off-track line of reasoning. *Reflection* is the mechanism by which an agent reviews its own outputs and corrects them.

### Inline Reflection (Thought-level)

The simplest form is already baked into ReAct: because the model reads its own prior thoughts and observations before each step, it can notice that an earlier approach failed and pivot. No extra machinery is needed — the context window provides the "memory."

### Post-Hoc Reflection: Reflexion

*Reflexion* (Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, 2023) formalizes this into an outer loop:

{{fig:agentloop-reflexion-outer-loop}}

The reflection step asks: "What went wrong and what should I do differently?" This verbal summary is prepended to the next attempt, acting as a form of episodic memory.

```python
REFLECT_PROMPT = """\
The following agent trajectory failed to complete the task correctly.
Please analyze what went wrong and provide specific, actionable advice
for how the agent should approach this task differently next time.

Task: {task}

Trajectory:
{trajectory}

Error or failure reason: {failure_reason}

Reflection (be specific and concrete):
"""

def reflect(client, model: str, task: str, trajectory: str, failure: str) -> str:
    """Produce a verbal reflection on a failed trajectory."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": REFLECT_PROMPT.format(
                task=task,
                trajectory=trajectory,
                failure_reason=failure,
            )
        }],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()
```

The key difference from simply retrying is that the reflection is *persistent* — it accumulates across trials, providing stronger and stronger guidance over time.

### Critic-Based Reflection

A more systematic approach uses a dedicated *critic* model (or a separate prompt to the same model) that evaluates each step before it is executed:

$$
\text{score}(a_t, c_t) = \text{critic}(c_t, a_t) \in [0, 1]
$$

If the score is below a threshold $\tau$, the action is rejected and the model is asked to reconsider. This adds latency (an extra LLM call per step) but can prevent catastrophic early errors, particularly in tool-heavy pipelines where mistakes are expensive to undo (e.g., sending an email, deleting a file).

```python
def critic_gate(client, model: str, context: str, proposed_action: str,
                threshold: float = 0.6) -> tuple[bool, str]:
    """
    Ask a critic whether the proposed action is correct/safe.
    Returns (should_proceed, explanation).
    """
    prompt = f"""\
Context so far:
{context}

Proposed action:
{proposed_action}

On a scale of 0 to 1, rate whether this action is correct, safe,
and aligned with the task goal. Then explain briefly.
Format: SCORE: <float>\\nEXPLANATION: <text>
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=128,
    )
    text = response.choices[0].message.content
    match = re.search(r"SCORE:\s*([\d.]+)", text)
    score = float(match.group(1)) if match else 0.0
    return (score >= threshold), text
```

## Tree Search Over Actions

For tasks where mistakes have high cost (coding challenges, long-form document editing) or where many valid solution paths exist, the linear ReAct loop may be too narrow. *Tree-of-Thought* (Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, 2023) and related approaches turn the action space into a tree:

{{fig:agentloop-tree-of-thought}}

At each node, the model proposes $k$ candidate next actions. A value function (another LLM call, a trained verifier, or a simple heuristic) scores them. The search proceeds by depth-first, breadth-first, or beam search, keeping only the top-$b$ beams at each depth.

### Beam Search Agent

```python
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class AgentNode:
    """A node in the search tree."""
    messages: list         # conversation history up to this point
    score: float = 1.0     # cumulative log-prob or value estimate
    depth: int = 0
    actions: list = field(default_factory=list)  # actions taken so far

def beam_react_agent(
    client,
    model: str,
    task: str,
    beam_width: int = 3,
    max_depth: int = 8,
    value_fn=None,   # optional callable(node) -> float
) -> str:
    """
    A beam-search version of the ReAct loop.
    Returns the answer from the highest-scoring terminal node.
    """
    system = "You solve tasks step-by-step using Thought/Action format."
    root = AgentNode(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": task},
        ]
    )
    beam = [root]

    for depth in range(max_depth):
        candidates = []
        for node in beam:
            # Sample multiple continuations from this node
            for _ in range(beam_width):
                resp = client.chat.completions.create(
                    model=model,
                    messages=node.messages,
                    stop=["Observation:"],
                    temperature=0.7,  # need diversity for beam search
                    max_tokens=256,
                )
                text = resp.choices[0].message.content.strip()

                # Check for finish
                if "finish(" in text:
                    match = re.search(r"finish\(([^)]*)\)", text)
                    answer = match.group(1).strip('"\'') if match else text
                    # Score this leaf
                    leaf_score = value_fn(node) if value_fn else node.score
                    candidates.append((leaf_score, answer, None))
                    continue

                # Execute action and get observation
                parsed = ReActAgent._parse_action(None, text)  # static-style call
                if parsed:
                    tool_name, args = parsed
                    obs = ReActAgent._execute_action(None, tool_name, args)
                else:
                    obs = "Error: no valid action found."

                # Compute node score (simple: add log-prob proxy from token count)
                new_msgs = deepcopy(node.messages)
                new_msgs.append({"role": "assistant", "content": text})
                new_msgs.append({"role": "user", "content": f"Observation: {obs}"})
                child_score = node.score * (0.95 ** 1)  # decay per step
                new_node = AgentNode(
                    messages=new_msgs,
                    score=child_score,
                    depth=depth + 1,
                    actions=node.actions + [text],
                )
                candidates.append((child_score, None, new_node))

        # Separate finished and unfinished candidates
        finished = [(s, a) for s, a, n in candidates if n is None]
        if finished:
            best_score, best_answer = max(finished, key=lambda x: x[0])
            return best_answer

        # Keep top beam_width nodes
        next_nodes = [(s, n) for s, _, n in candidates if n is not None]
        next_nodes.sort(key=lambda x: x[0], reverse=True)
        beam = [n for _, n in next_nodes[:beam_width]]

    # If we exhaust depth, return best partial result
    return "Exhausted search depth without finding answer."
```

Tree search multiplies the LLM calls by roughly $k \times d$ (beam width × depth), making it expensive. It is typically reserved for offline evaluation or tasks where quality clearly outweighs cost.

## When to Stop: Termination Criteria

One of the most consequential design decisions in an agentic loop is *when to stop*. There are four mechanisms, and robust agents use all of them:

### 1. Model-Initiated Termination
The model calls `finish()` (or analogous). This is the primary mechanism. The format must be part of the training/prompt so the model learns to distinguish "I have the answer" from "I need more information."

### 2. Hard Step Limit ($T_{\max}$)
Always implement a ceiling. A reasonable default is 10–25 steps for most tasks. For complex coding agents, up to 50 may be appropriate. Beyond this, logs almost always show the agent stuck in a loop.

### 3. Token Budget
Because each step adds tokens to the context, you can also terminate when the total context length approaches the model's limit minus a safety margin:

```python
MAX_CONTEXT_FRACTION = 0.85  # stop at 85% of context window

def context_full(messages: list, model_context_limit: int = 128_000) -> bool:
    """Rough check: count characters as a proxy for tokens (4 chars ≈ 1 token)."""
    total_chars = sum(len(m["content"]) for m in messages)
    estimated_tokens = total_chars / 4
    return estimated_tokens > MAX_CONTEXT_FRACTION * model_context_limit
```

### 4. Task-Specific Success Detectors
For verifiable tasks (code execution, math), the harness can check success directly: unit tests pass, the math answer matches, the file exists. This is the gold standard but requires task-specific logic.

!!! interview "Interview Corner"
    **Q:** An interviewer asks: "You're building a production ReAct agent. It sometimes gets stuck in an infinite loop, calling the same tool with the same arguments repeatedly. How do you diagnose and fix this?"

    **A:** First, add a *deduplication check*: if the last $n$ actions are identical (same tool + same args), immediately inject an observation like "You have already tried this approach. It did not work. Consider a different strategy." and optionally decrement the step budget.

    Second, diagnose *why* the loop occurs. Common causes: (a) the model never received a meaningful observation (tool returned empty or the format was not parsed correctly), (b) the model's instruction-following is weak for the specific tool format, (c) the task is genuinely unsolvable with the available tools and the model doesn't know how to give up gracefully. Fix (a) by improving observation parsing; (b) by adding few-shot examples of recovery; (c) by adding a `give_up(reason)` tool alongside `finish(answer)`.

    Third, implement *action hashing*: maintain a set of `(tool_name, frozenset(args))` tuples seen so far. If a new action is in the set, block it and force the model to explain why it is trying something it already tried.

## Failure Modes and Defensive Engineering

### Infinite Loops

The agent calls `search("Python syntax")` five times because each observation is slightly different (or identical) and the model never synthesizes an answer. Fix: action deduplication (above) + an explicit "I'm stuck" detection heuristic.

```python
from collections import Counter

def detect_loop(action_history: list[str], window: int = 4, threshold: int = 3) -> bool:
    """
    Returns True if the same action appears >= threshold times
    in the last 'window' steps.
    """
    recent = action_history[-window:]
    counts = Counter(recent)
    return any(c >= threshold for c in counts.values())
```

### Derailment (Context Drift)

Over many steps, the model "forgets" the original goal and starts pursuing a tangential sub-task. This happens because the user's original instruction is diluted by many turns of tool outputs.

Mitigations:
- **Goal injection.** Re-inject the original task as a user message every $k$ steps: "Reminder: your goal is `{task}`."
- **Context compression.** Summarize older turns to keep the context relevant. See [Context Engineering & Management](../08-agents-harness/04-context-engineering.html).
- **Structured scratchpad.** Maintain a short "current goal" field separate from the message history that is always prepended to the system prompt.

### Spurious Tool Calls

The agent calls tools that are unnecessary for the task (e.g., running a database query to answer a simple arithmetic question). This wastes tokens and time.

Mitigation: add a short *rationale check* to the thought step. If the thought does not logically require the chosen tool, the model should reconsider. Few-shot examples of "I already know this; I don't need to search" are very effective.

### Hallucinated Observations

Without the `stop=["Observation:"]` trick, the model will sometimes write both the Action *and* a fake Observation in a single generation, bypassing real tool execution entirely. This is arguably the most dangerous failure mode because it looks correct in the logs.

Always terminate generation at `"Observation:"` and inject the real result from your harness.

!!! warning "Common pitfall: trusting model-generated observations"
    If your harness does not use a stop sequence at `"Observation:"`, the model will sometimes generate plausible-sounding but completely fabricated observations. A model generating its own search results is no longer grounded — it is just doing chain-of-thought with extra steps. Always enforce that observations come from real tool execution.

## Agentic RL: Training the Loop End-to-End

The architectures above treat the LLM as a frozen policy and engineer around it. A richer approach trains the model to be a better agentic policy using reinforcement learning. This connects deeply to [Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html) and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

The key insight is that the reward signal for an agent can be *sparse and delayed*: `+1` if the task is completed successfully, `0` otherwise. This is harder than token-level reward but more directly aligned with task success. The KL penalty $\beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$ is essential for preventing the policy from collapsing into degenerate tool-call patterns.

For the purposes of this chapter: agentic RL is the mechanism by which a model learns *internally* to do what ReAct does with *external* prompting. A model trained with agentic RL writes better Thought traces, chooses better tools, and stops more reliably — without needing the hand-crafted system prompt.

## Summary Comparison of Agentic Architectures

| Architecture | Loop type | Planning | Reflection | Typical use case |
|---|---|---|---|---|
| ReAct | Reactive | None | Inline (via Thought) | Short Q&A, web browsing |
| Plan-Execute | Two-phase | Explicit decomposition | After each subtask | Multi-step research, coding |
| Reflexion | Multi-trial | None per trial | Explicit verbal reflection | Tasks with verifiable outcomes |
| Tree of Thought | Beam/BFS/DFS | Implicit (branching) | Via scoring | Hard reasoning, math proofs |
| Agentic RL | Trained policy | Learned | Learned | General purpose (post-training) |

!!! key "Key Takeaways"
    - The agentic loop — observe, think, act — is the minimal pattern that grounds an LLM in real-world state. Without it, the model can only hallucinate outcomes.
    - ReAct interleaves Thought (chain-of-thought reasoning) and Action (tool calls) within a single context window, with Observations injected by the harness after each action. Always terminate generation at `"Observation:"` to prevent hallucinated observations.
    - Use `temperature=0.0` for agentic tasks, a hard `MAX_STEPS` safety limit, and action deduplication to detect and break loops before they waste tokens or cause real-world side effects.
    - Plan-Execute architectures front-load planning into a separate LLM call that produces a structured dependency graph of subtasks, enabling parallel execution and better long-horizon coherence.
    - Reflexion wraps the ReAct loop in an outer retry loop, using verbal reflection on failed trajectories to guide subsequent attempts — a form of few-shot self-supervised improvement.
    - Tree-of-Thought extends ReAct to a beam-search tree, sampling $k$ candidate actions at each step and pruning with a value function. It is powerful but multiplies LLM calls by $O(k \times d)$.
    - Key failure modes are: infinite loops (action deduplication + early termination), context drift/derailment (goal re-injection + context compression), and hallucinated observations (stop sequences). All three must be actively defended against in production.
    - Agentic RL trains the model to internalize the agentic loop as a policy, replacing fragile prompt engineering with learned behavior that generalizes across task distributions.

!!! sota "State of the Art & Resources (2026)"
    The agentic loop — observe, think, act — is now the dominant paradigm for deploying LLMs on real-world tasks, with ReAct as its canonical formulation. Since 2023 the field has moved rapidly from prompting-based agents to trained agentic policies and standardized benchmarks, with production frameworks like LangGraph and SWE-agent pushing reliability into software-engineering workflows.

    **Foundational work**

    - [Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2023)](https://arxiv.org/abs/2210.03629) — the paper that established the Thought/Action/Observation loop and showed it beats pure chain-of-thought or pure acting on QA and decision-making benchmarks.
    - [Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (2023)](https://arxiv.org/abs/2303.11366) — introduces the outer reflection loop: verbal self-critique stored as episodic memory guides subsequent trial attempts without weight updates.
    - [Yao et al., *Tree of Thoughts: Deliberate Problem Solving with LLMs* (2023)](https://arxiv.org/abs/2305.10601) — extends the linear loop into a beam-search tree, sampling multiple candidate thoughts and pruning with a value function; lifts GPT-4 success on Game of 24 from 4% to 74%.

    **Recent advances (2023–2026)**

    - [Yang et al., *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering* (2024)](https://arxiv.org/abs/2405.15793) — shows that a carefully designed agent-computer interface (ACI) is as important as the underlying model; NeurIPS 2024.
    - [Liu et al., *AgentBench: Evaluating LLMs as Agents* (2023)](https://arxiv.org/abs/2308.03688) — multi-environment benchmark across 8 tasks revealing large gaps between commercial and open-source models as agents; ICLR 2024.
    - [Wang et al., *A Survey on Large Language Model based Autonomous Agents* (2024)](https://arxiv.org/abs/2308.11432) — comprehensive taxonomy of agent construction, memory, planning, and action, covering 150+ papers.

    **Open-source & tools**

    - [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — graph-based orchestration framework for stateful, long-running agents with cycles, conditionals, and durable execution; the de-facto production choice for Python agentic pipelines.
    - [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent) — open-source ReAct-style agent that resolves real GitHub issues; reference implementation for tool-interface design.
    - [ysymyth/ReAct](https://github.com/ysymyth/ReAct) — official code release for the ICLR 2023 ReAct paper, with prompt notebooks for HotpotQA, FEVER, AlfWorld, and WebShop.

    **Go deeper**

    - [Anthropic, *Building Effective AI Agents* (2024)](https://www.anthropic.com/research/building-effective-agents) — practitioner guide from Anthropic distilling patterns (prompt chaining, routing, orchestrator-worker) from production deployments; argues for simple composable patterns over heavy frameworks.

## Further Reading

- Yao, S., Zhao, J., Yu, D., et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
- Shinn, N., Cassano, F., Labash, B., et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023.
- Yao, S., Yu, D., Zhao, J., et al. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023.
- Wei, J., Wang, X., Schuurmans, D., et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
- Wang, L., Ma, C., Feng, X., et al. *A Survey on Large Language Model based Autonomous Agents.* Frontiers of Computer Science, 2024.
- Significant Gravitas / AutoGPT — one of the earliest open-source agentic loop implementations. [github.com/Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- LangChain Agents documentation and the LangGraph framework for graph-based agentic pipelines.
