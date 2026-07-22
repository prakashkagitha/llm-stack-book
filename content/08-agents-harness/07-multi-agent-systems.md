# 8.7 Multi-Agent Systems & Orchestration

A single LLM call has a bounded context window, bounded compute budget, and bounded reliability — it can hallucinate, lose track of a long task, or simply not have the right "persona" for every sub-problem. Multi-agent systems attack all three limits by decomposing a task across multiple model invocations that communicate through structured messages or shared state.

This chapter maps the design space: topologies, communication patterns, shared memory, frameworks, and the often-ignored question of *when multi-agent is the wrong answer*. We build real code throughout, because the footguns in this space are easiest to understand by running into them.

Related chapters you should know first: [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html), [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html), [Memory Systems for Agents](../08-agents-harness/05-agent-memory.html), and [The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html).

## The Case for Multiple Agents

The intuitive motivation is task decomposition, but the real drivers are more specific.

**Context limits.** A 200k-token context sounds enormous, but a realistic software engineering task can easily involve hundreds of files totalling millions of tokens. Assigning each file to a specialized worker agent that produces a compact summary allows the orchestrator to reason at a higher level of abstraction without blowing its window.

**Error isolation.** When a single monolithic agent fails halfway through a 40-step task, you lose all intermediate work. A workflow with explicit handoff points lets you checkpoint, retry just the failed sub-task, and resume.

**Parallelism.** Many tasks have independent sub-tasks. Searching three databases, writing two code modules, and translating a document into five languages can proceed concurrently when each sub-task runs in a separate agent thread.

**Role specialization.** A "critic" agent can be prompted with different priorities than a "generator" agent. When the same model plays both roles in sequence, with different system prompts and therefore different prior distributions over outputs, the critique is often sharper than self-critique in a single context.

**Debate and verification.** Multiple agents can independently solve a problem and then reconcile answers, reducing variance in the same way that ensembling reduces variance for classifiers.

We can express the expected quality gain from $N$ independent agents who each succeed with probability $p$ (for tasks where any one correct solution suffices) as:

$$
P(\text{at least one correct}) = 1 - (1-p)^N
$$

For $p = 0.7$ and $N = 3$, this is $1 - 0.3^3 = 0.973$. The gain is largest when $p$ is moderate — for tasks that are either trivial ($p \approx 1$) or impossibly hard ($p \approx 0$), multiple agents add cost without benefit.

{{fig:ensembling-gain-vs-p}}

## Topologies: How Agents Are Wired

The wiring diagram is the first design decision. There are five canonical patterns; real systems combine them.

{{fig:marl-topologies-five-patterns}}

Each topology has a natural failure mode:

| Topology | Natural strength | Natural failure mode |
|---|---|---|
| Orchestrator-workers | Clear ownership, easy to retry workers | Orchestrator becomes bottleneck / single point of context |
| Pipeline | Deterministic, debuggable | Early errors propagate unchecked |
| Debate | Reduces overconfidence | Agents converge to same wrong answer; judge is biased |
| Blackboard | Concurrent writers, emergent synthesis | Race conditions, stale reads, key collisions |
| Hierarchical | Scales to large tasks | Latency explodes; cross-tree communication is awkward |

## Workflow vs. Agent: The Critical Distinction

The LLM agent community overuses the word "agent." It is worth being precise.

A **workflow** is a directed acyclic graph (DAG) of LLM calls where the routing logic is deterministic and written in code by the developer. Each node calls a model; the edges between nodes are if/else or fixed sequences. The model does *not* decide what to do next.

An **agent** lets the model itself decide the next action at each step, including whether to call a tool, which tool, and what to pass. The control flow is dynamic and emerges from model outputs.

```text
WORKFLOW (control flow in code)                AGENT (control flow in model)

code: if intent == "math":                     model output: {"action": "calculator",
          call(math_model, query)                            "input": "sqrt(2)"}
      elif intent == "search":
          call(search_model, query)
```

The practical significance: workflows are cheaper to run, easier to test, easier to observe, and less likely to spiral. Agents are more flexible and can handle tasks where the exact steps cannot be enumerated in advance.

A common design pattern is to use a workflow as the outer shell (so the overall task is deterministic and auditable) and embed agents as the *leaves* of the workflow — the workers that handle unpredictable sub-tasks.

We will return to this distinction when discussing frameworks, because LangGraph, AutoGen, CrewAI, and Swarm each stake out a different point on this spectrum.

{{fig:workflow-vs-agent-control-flow-spectrum}}

## Orchestrator-Worker in Code

Let us build the most common pattern from scratch, without a framework, so the mechanics are transparent.

```python
"""
orchestrator_worker.py
A minimal orchestrator-worker multi-agent system using the OpenAI chat API.
Runs end-to-end: the orchestrator decomposes a task, dispatches workers,
and synthesises their results.

Requirements:
    pip install openai
    OPENAI_API_KEY must be set.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

client = OpenAI()

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    """A sub-task produced by the orchestrator."""
    worker_id: str          # human-readable label, e.g. "researcher_1"
    system_prompt: str      # role specialisation
    task: str               # the concrete instruction
    context: str = ""       # any shared context the worker needs


@dataclass
class WorkResult:
    worker_id: str
    output: str
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Low-level model call
# ---------------------------------------------------------------------------

def call_model(
    system: str,
    user: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Thin wrapper around chat completions — easy to swap for any provider."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Orchestrator: produces a plan as structured JSON
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM = """\
You are a task orchestrator. Given a high-level goal, decompose it into
independent sub-tasks that can run in parallel. Each sub-task should be
self-contained. Return a JSON array of objects, each with keys:
  worker_id   – short snake_case label
  system_prompt – the specialist role for this worker
  task        – the full instruction for this worker

Return ONLY the JSON array. No prose.
"""

def plan(goal: str, shared_context: str = "") -> list[WorkItem]:
    """Ask the orchestrator to produce a parallel work plan."""
    user_msg = f"Goal: {goal}\n\nShared context:\n{shared_context}" if shared_context else f"Goal: {goal}"
    raw = call_model(ORCHESTRATOR_SYSTEM, user_msg, temperature=0.0)

    # Strip markdown fences if model wrapped the JSON
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    items_data = json.loads(raw)
    return [
        WorkItem(
            worker_id=d["worker_id"],
            system_prompt=d["system_prompt"],
            task=d["task"],
            context=shared_context,
        )
        for d in items_data
    ]


# ---------------------------------------------------------------------------
# Worker: executes a single sub-task
# ---------------------------------------------------------------------------

def run_worker(item: WorkItem) -> WorkResult:
    """Execute one work item; capture any model or network error."""
    try:
        user_msg = item.task
        if item.context:
            user_msg = f"Context:\n{item.context}\n\nTask:\n{item.task}"
        output = call_model(item.system_prompt, user_msg)
        return WorkResult(worker_id=item.worker_id, output=output)
    except Exception as exc:
        return WorkResult(worker_id=item.worker_id, output="", success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Synthesiser: combines parallel results
# ---------------------------------------------------------------------------

SYNTHESISER_SYSTEM = """\
You are a synthesis agent. You receive the outputs of multiple specialist
workers who each tackled a sub-task. Your job is to integrate their outputs
into a single coherent, well-structured response. Be concise.
"""

def synthesise(goal: str, results: list[WorkResult]) -> str:
    """Combine worker outputs into a final answer."""
    parts = [f"=== {r.worker_id} ===\n{r.output}" for r in results if r.success]
    failed = [r.worker_id for r in results if not r.success]
    user_msg = f"Original goal: {goal}\n\nWorker outputs:\n\n" + "\n\n".join(parts)
    if failed:
        user_msg += f"\n\nNote: workers {failed} failed and produced no output."
    return call_model(SYNTHESISER_SYSTEM, user_msg)


# ---------------------------------------------------------------------------
# Top-level orchestrate function
# ---------------------------------------------------------------------------

def orchestrate(goal: str, max_workers: int = 4) -> str:
    """Full orchestrator-worker pipeline for a given goal."""
    print(f"[Orchestrator] Planning: {goal}")
    work_items = plan(goal)
    print(f"[Orchestrator] {len(work_items)} sub-tasks: {[w.worker_id for w in work_items]}")

    results: list[WorkResult] = []
    # Run workers in parallel using a thread pool (each call is IO-bound)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_worker, item): item for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            status = "OK" if result.success else f"FAILED: {result.error}"
            print(f"  [Worker {result.worker_id}] {status}")
            results.append(result)

    print("[Orchestrator] Synthesising results …")
    answer = synthesise(goal, results)
    return answer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    goal = (
        "Write a brief market analysis report for a new AI-powered note-taking app: "
        "competitor landscape, target user segments, and key risks."
    )
    answer = orchestrate(goal)
    print("\n=== FINAL ANSWER ===")
    print(answer)
```

The key engineering choices above:

1. **ThreadPoolExecutor for parallelism** — LLM calls are network I/O so Python threads work fine; `asyncio` with `httpx` is equally valid and slightly lower overhead.
2. **Structured JSON plan** — The orchestrator emits a machine-readable plan rather than prose, making the handoff deterministic. Always strip markdown fences defensively.
3. **Isolated error handling per worker** — A single worker failure does not crash the pipeline; the synthesiser knows which workers succeeded.

## Shared Memory and the Blackboard Pattern

In the orchestrator-worker model, the orchestrator is the sole integration point — it reads all worker outputs and synthesises them. This works well for 3–8 workers but becomes a bottleneck as the number of workers grows, because every result must fit in the orchestrator's context window at synthesis time.

The **blackboard pattern** replaces the orchestrator's context with an external shared data structure. Workers read from and write to the blackboard; an optional coordinator polls the blackboard and decides when to advance.

```python
"""
blackboard.py
A minimal blackboard for multi-agent coordination.
Thread-safe for concurrent workers using a threading.Lock.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlackboardEntry:
    author: str           # which agent wrote this
    content: Any          # arbitrary Python object (str, dict, etc.)
    timestamp: float = 0.0


class Blackboard:
    """
    Shared key-value store for agent communication.
    Keys are namespaced: "agent_id:key_name".
    """

    def __init__(self):
        self._store: dict[str, list[BlackboardEntry]] = defaultdict(list)
        self._lock = threading.Lock()

    def write(self, key: str, content: Any, author: str) -> None:
        """Append a versioned entry. Never overwrites — full audit trail."""
        import time
        entry = BlackboardEntry(author=author, content=content, timestamp=time.time())
        with self._lock:
            self._store[key].append(entry)

    def read_latest(self, key: str) -> BlackboardEntry | None:
        """Return the most recently written entry for a key."""
        with self._lock:
            entries = self._store.get(key, [])
            return entries[-1] if entries else None

    def read_all(self, key: str) -> list[BlackboardEntry]:
        """Return full history for a key (useful for debate/critique)."""
        with self._lock:
            return list(self._store.get(key, []))

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def snapshot(self) -> dict[str, Any]:
        """Return a plain dict of latest values — useful for debugging."""
        with self._lock:
            return {k: v[-1].content for k, v in self._store.items() if v}
```

Agents interact with the blackboard rather than passing messages directly:

```python
# Example: two agents using the blackboard

def research_agent(board: Blackboard, topic: str) -> None:
    """Writes findings to the blackboard."""
    findings = call_model(
        "You are a research specialist. Be factual and concise.",
        f"Research: {topic}",
    )
    board.write(key="research:findings", content=findings, author="researcher")


def critic_agent(board: Blackboard) -> None:
    """Reads the latest research and writes a critique."""
    entry = board.read_latest("research:findings")
    if entry is None:
        board.write("critic:status", "No findings to critique yet.", author="critic")
        return
    critique = call_model(
        "You are a rigorous critic. Identify gaps and unsupported claims.",
        f"Critique this research:\n\n{entry.content}",
    )
    board.write(key="critic:critique", content=critique, author="critic")
```

The blackboard's append-only design gives you a full audit trail for free. This matters for debugging: you can replay the entire agent interaction post-hoc to find where reasoning went wrong.

## Frameworks: LangGraph, AutoGen, CrewAI, and OpenAI Swarm

Four frameworks dominate the multi-agent landscape as of mid-2025. They differ fundamentally in their mental models.

### LangGraph

LangGraph (from the LangChain team) models agent workflows as **explicit state machines** with typed state. Nodes are Python functions or model-calling steps; edges can be conditional. This is close to the *workflow* end of the spectrum — the developer writes the graph topology.

```python
"""
langgraph_example.py
A minimal LangGraph orchestrator-worker for a research task.
Requires: pip install langgraph langchain-openai
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# 1. Define state — the "blackboard" is strongly typed
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    question: str
    research_notes: str     # filled by researcher node
    critique: str           # filled by critic node
    final_answer: str       # filled by synthesiser node
    iteration: int          # how many research-critique loops so far

# ---------------------------------------------------------------------------
# 2. Define nodes (each is a pure function: state → state patch)
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def researcher(state: ResearchState) -> dict:
    """Node: produce research notes on the question."""
    prompt = f"Research the following and produce concise notes:\n{state['question']}"
    if state.get("critique"):
        prompt += f"\n\nPrevious critique to address:\n{state['critique']}"
    result = llm.invoke(prompt)
    return {"research_notes": result.content}


def critic(state: ResearchState) -> dict:
    """Node: critique the research notes, or approve them."""
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Research notes:\n{state['research_notes']}\n\n"
        "Identify missing information or unsupported claims. "
        "If the notes are sufficient, reply with exactly: APPROVED"
    )
    result = llm.invoke(prompt)
    return {
        "critique": result.content,
        "iteration": state.get("iteration", 0) + 1,
    }


def synthesiser(state: ResearchState) -> dict:
    """Node: produce the final answer from approved notes."""
    prompt = (
        f"Question: {state['question']}\n\n"
        f"Approved research notes:\n{state['research_notes']}\n\n"
        "Write a clear, concise final answer."
    )
    result = llm.invoke(prompt)
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# 3. Routing: decide whether to loop or to synthesise
# ---------------------------------------------------------------------------

def route_after_critic(state: ResearchState) -> str:
    """Conditional edge: approve ends the loop; critique sends us back."""
    if "APPROVED" in state["critique"] or state.get("iteration", 0) >= 3:
        return "synthesise"   # edge label → goes to synthesiser node
    return "revise"           # edge label → goes back to researcher


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(ResearchState)

builder.add_node("researcher",   researcher)
builder.add_node("critic",       critic)
builder.add_node("synthesiser",  synthesiser)

builder.set_entry_point("researcher")
builder.add_edge("researcher", "critic")
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "revise":     "researcher",   # loop back
        "synthesise": "synthesiser",  # exit
    },
)
builder.add_edge("synthesiser", END)

graph = builder.compile()

# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = graph.invoke({
        "question": "What are the main differences between RLHF and DPO?",
        "research_notes": "",
        "critique": "",
        "final_answer": "",
        "iteration": 0,
    })
    print(result["final_answer"])
```

LangGraph's chief virtue is **observability**: the state at every node transition is a typed dict you can log, checkpoint, and resume from. Its chief cost is verbosity — simple pipelines require more boilerplate than raw Python.

### AutoGen

Microsoft's AutoGen models multi-agent systems as **conversational agents** that exchange natural-language messages. The programmer registers agents and defines their reply functions; the conversation drives control flow. AutoGen sits closer to the *agent* end of the spectrum.

```python
"""
autogen_debate.py
A two-agent debate using AutoGen's ConversableAgent API.
Requires: pip install pyautogen
"""

import autogen

# Shared config for both agents
llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": "..."}]}

# Agent A: takes the PRO position
pro_agent = autogen.ConversableAgent(
    name="ProAgent",
    system_message=(
        "You argue FOR the proposition that large language models will "
        "replace most knowledge workers within 10 years. Be specific and "
        "evidence-based. Keep responses under 200 words."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",  # fully automated
)

# Agent B: takes the CON position
con_agent = autogen.ConversableAgent(
    name="ConAgent",
    system_message=(
        "You argue AGAINST the proposition that large language models will "
        "replace most knowledge workers within 10 years. Be specific and "
        "evidence-based. Keep responses under 200 words."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Judge: produces a verdict after the debate
judge = autogen.ConversableAgent(
    name="Judge",
    system_message=(
        "You are a neutral judge. After reading the debate, you produce a "
        "balanced verdict on who made the stronger arguments and why."
    ),
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "VERDICT:" in msg.get("content", ""),
)

if __name__ == "__main__":
    # Initiate a 3-round debate between pro and con agents
    pro_agent.initiate_chat(
        recipient=con_agent,
        message="LLMs will replace most knowledge workers within 10 years.",
        max_turns=3,
    )
    # Then ask the judge to weigh in on the full transcript
    # (In practice you'd collect and pass the conversation history)
```

### CrewAI

CrewAI provides a higher-level abstraction: you define **Agents** with `role`, `goal`, and `backstory`, then assemble them into a **Crew** with a list of **Tasks**. Control flow is either sequential or hierarchical (where a manager LLM decides task order). The framework is opinionated and quick to prototype.

```python
"""
crewai_example.py
A minimal CrewAI pipeline for writing a technical blog post.
Requires: pip install crewai
"""

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# ---------------------------------------------------------------------------
# Define specialist agents
# ---------------------------------------------------------------------------

researcher = Agent(
    role="Technical Researcher",
    goal="Find accurate, up-to-date information on the given topic.",
    backstory="A meticulous research scientist who double-checks every claim.",
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write a clear, engaging technical blog post from research notes.",
    backstory="An experienced writer who makes complex ideas accessible.",
    llm=llm,
    verbose=True,
)

editor = Agent(
    role="Editor",
    goal="Polish the draft for clarity, accuracy, and style.",
    backstory="A demanding editor who improves every sentence.",
    llm=llm,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Define tasks (sequential pipeline)
# ---------------------------------------------------------------------------

research_task = Task(
    description="Research the key concepts behind mixture-of-experts LLM architectures.",
    agent=researcher,
    expected_output="A structured set of notes covering architecture, training, and key papers.",
)

writing_task = Task(
    description="Write a 500-word technical blog post based on the research notes.",
    agent=writer,
    expected_output="A polished 500-word blog post draft.",
    context=[research_task],  # this task receives research_task's output
)

editing_task = Task(
    description="Edit the blog post draft for accuracy, flow, and style.",
    agent=editor,
    expected_output="A final, publication-ready blog post.",
    context=[writing_task],
)

# ---------------------------------------------------------------------------
# Assemble and run the crew
# ---------------------------------------------------------------------------

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # or Process.hierarchical for a manager LLM
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
```

### OpenAI Swarm

OpenAI's Swarm (released as an educational framework) is deliberately minimal: agents are plain Python functions plus a system prompt, and they communicate by **handing off** control to one another. There is no persistent orchestrator — instead, an agent hands control to a different agent by returning a `Result` object with a `agent` field. This is the *agent-as-router* pattern.

```python
"""
swarm_handoff.py
A triage-and-specialist handoff pattern using OpenAI Swarm.
Requires: pip install openai-swarm
"""

from swarm import Swarm, Agent

client = Swarm()

# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

billing_agent = Agent(
    name="BillingAgent",
    instructions=(
        "You handle billing inquiries: invoices, payment methods, refunds. "
        "If the question is not billing-related, transfer back to triage."
    ),
)

technical_agent = Agent(
    name="TechnicalAgent",
    instructions=(
        "You handle technical support: bugs, API errors, configuration issues. "
        "If the question is not technical, transfer back to triage."
    ),
)

# ---------------------------------------------------------------------------
# Triage agent: routes to the right specialist
# ---------------------------------------------------------------------------

def transfer_to_billing() -> Agent:
    """Handoff function — returning an Agent transfers control."""
    return billing_agent

def transfer_to_technical() -> Agent:
    return technical_agent

triage_agent = Agent(
    name="TriageAgent",
    instructions=(
        "You are a customer service triage agent. "
        "For billing questions call transfer_to_billing(). "
        "For technical questions call transfer_to_technical(). "
        "Otherwise answer directly."
    ),
    functions=[transfer_to_billing, transfer_to_technical],
)

# ---------------------------------------------------------------------------
# Run a conversation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    messages = [{"role": "user", "content": "My invoice shows the wrong amount."}]
    response = client.run(agent=triage_agent, messages=messages)
    print(response.messages[-1]["content"])
    print(f"Final agent: {response.agent.name}")
```

### Framework Comparison

| Dimension | LangGraph | AutoGen | CrewAI | Swarm |
|---|---|---|---|---|
| Mental model | State machine / DAG | Conversational agents | Crew with roles & tasks | Agent-as-router, handoffs |
| Control flow owner | Developer (graph edges) | Model (message replies) | Framework + model | Model (function calls) |
| State management | Typed dict per node | Message history | Task context passing | Context variables |
| Debugging | Excellent (typed state) | Moderate | Moderate | Minimal |
| Best for | Complex stateful workflows | Open-ended debates | Structured pipelines | Simple routing |
| Maturity (mid-2025) | High | High | High | Low (educational) |

## Handoff Protocols and Communication Patterns

How agents pass information to each other determines much of the system's reliability.

**String-to-string (opaque)** is the simplest: agent A's output text becomes agent B's input text. It is easy to implement but fragile — B must parse prose that A may format inconsistently.

**Structured handoff** uses typed schemas (JSON, Pydantic models) at every boundary. The orchestrator validates the payload before forwarding. This adds a small overhead but makes errors explicit rather than silently propagating malformed text.

```python
from pydantic import BaseModel

class ResearchHandoff(BaseModel):
    """Validated payload from researcher → writer."""
    topic: str
    key_facts: list[str]          # at most 10 bullet points
    source_urls: list[str]        # cited sources (may be empty)
    confidence: float             # researcher's self-assessed confidence 0–1

class WritingHandoff(BaseModel):
    """Validated payload from writer → editor."""
    draft_text: str
    word_count: int
    flagged_sections: list[str]   # writer's own uncertainty flags
```

**Message bus** — agents subscribe to topics (e.g. "findings", "code_output") rather than passing messages directly. This decouples the topology and makes it easy to add new agents without rewiring the graph. Apache Kafka or even Redis pub/sub serve as the bus in production systems; for local development a simple Python `queue.Queue` suffices.

**Context forwarding** is the mechanism used by most LLM-native systems: each agent's context window includes a summary or verbatim copy of the preceding step's output. The cost is linear in chain length — a 10-step pipeline may inject up to 10 prior summaries into each new model call. Managing this is the subject of [Context Engineering & Management](../08-agents-harness/04-context-engineering.html).

## When Multi-Agent Hurts: Costs, Anti-Patterns, and Failure Modes

Adding agents is not free. Before designing a multi-agent system, run through this checklist.

**Latency.** Even with parallelism, the critical path latency of a multi-agent system is the sum of sequential LLM calls on the critical path. A 4-node sequential chain where each call takes 2 s adds 8 s of minimum latency before any response.

**Token cost.** Each agent call incurs its own prompt tokens (system prompt + context). A 5-agent pipeline with a 2,000-token system prompt each and 1,000 tokens of shared context per call spends 15,000 input tokens just on overhead, before any actual work.

**Error amplification.** In a pipeline, a hallucination in stage 1 is treated as ground truth by stage 2. Without a verification step, errors compound across stages. The same hallucination that a careful single agent might self-correct (because it contradicts itself within one context) goes undetected when each agent starts fresh.

**Coordination overhead.** Agents that communicate extensively spend a growing fraction of their token budget on housekeeping — summarising their own state, re-establishing context, and formatting handoffs — rather than on the actual task.

**Anti-patterns to avoid:**

- **Agent soup** — spawning many agents without clear role boundaries. If two agents' instructions overlap, they produce contradictory outputs and the synthesiser cannot reconcile them.
- **Infinite loops** — without a hard iteration cap, debate and reflection loops can cycle indefinitely (and rack up large bills). Always set `max_turns` or `max_iterations`.
- **Blind trust** — an orchestrator that passes a worker's output to the next stage without validation. A worker that returns malformed JSON or a code snippet containing an injection should fail loudly, not silently corrupt downstream agents.
- **Over-parallelism** — spawning 20 workers in parallel when the rate limit is 60 requests/minute will cause most to fail or be throttled. Always model the rate-limit as a resource constraint on your thread pool.

!!! warning "The multi-agent tax"
    Every inter-agent boundary costs tokens and latency. A task that takes 500 tokens and 1 s as a single call may cost 3,000 tokens and 5 s when decomposed into a 3-agent pipeline with context forwarding. Benchmark the single-agent baseline before reaching for multi-agent decomposition.

!!! example "Cost worked example: 5-agent pipeline"

    Suppose you have a pipeline with 5 sequential agents:

    - Each agent has a system prompt of ~1,000 tokens.
    - The shared "briefing" context forwarded from earlier stages grows: stage 1 receives 0, stage 2 receives ~500, stage 3 receives ~1,000, stage 4 receives ~1,500, stage 5 receives ~2,000 tokens of prior output.
    - Each agent generates ~400 output tokens.
    - Model price: USD 0.15 / 1M input tokens, USD 0.60 / 1M output tokens (gpt-4o-mini tier, illustrative).

    **Input tokens:** $5 \times 1000_\text{system} + (0+500+1000+1500+2000)_\text{context} = 5000 + 5000 = 10000$ input tokens.

    **Output tokens:** $5 \times 400 = 2000$ output tokens.

    **Cost per pipeline run:**

    $$
    C = \frac{10000 \times 0.15}{10^6} + \frac{2000 \times 0.60}{10^6} = \$0.0015 + \$0.0012 = \$0.0027
    $$

    At 10,000 pipeline runs per day, that is USD 27/day or roughly USD 800/month just for this pipeline — before any tool calls or RAG retrieval costs. Contrast with a single well-prompted agent that might cost USD 0.0008/call at the same volume: USD 240/month. The multi-agent premium here is roughly 3.4×.

    The premium is justified only if the quality gain exceeds the cost. Measure both.

{{fig:multi-agent-tax-compounding}}

## Agentic RL and Learning to Orchestrate

Multi-agent systems are not only engineered; they can be *trained*. Reinforcement learning from verifiable rewards (see [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)) can be applied at the orchestrator level: the orchestrator is the policy, its actions are "dispatch worker X with context Y", and the reward is the quality of the final task completion.

This is the regime where [Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html) becomes relevant: the value function must account for multi-step credit assignment across agent boundaries, not just within a single model call.

A simpler approach — and one deployable today — is **self-critique with routing**: the orchestrator scores each worker's output against a rubric and re-dispatches to the same worker (with the critique as additional context) if the score is below a threshold. This is a policy gradient in disguise, but the gradient is computed by a scoring LLM rather than a reward model.

```python
def self_critique_loop(
    orchestrator_prompt: str,
    worker_fn,
    scoring_fn,
    task: str,
    threshold: float = 0.8,
    max_retries: int = 3,
) -> str:
    """
    Run a worker, score its output, and retry with critique if below threshold.
    scoring_fn: (task, output) -> float in [0, 1]
    """
    for attempt in range(max_retries):
        output = worker_fn(task)
        score = scoring_fn(task, output)
        if score >= threshold:
            return output
        # Generate a critique and append it to the task for the next attempt
        critique = call_model(
            "You are a quality assessor. Identify what is wrong or missing.",
            f"Task: {task}\n\nOutput:\n{output}\n\nScore: {score:.2f}",
        )
        task = f"{task}\n\n[Previous attempt scored {score:.2f}. Critique: {critique}]"
    return output  # return best-effort after max_retries
```

!!! interview "Interview Corner"
    **Q:** You're designing a multi-agent system for a code-generation task. The system has an orchestrator, a code-writing worker, a test-running worker, and a documentation worker. The latency SLA is 5 seconds. How do you architect this?

    **A:** The critical insight is to identify which steps must be sequential (the test runner depends on code the writer produces) and which can be parallel (documentation can be generated at the same time tests are running, given access to the same code). A minimal architecture:

    1. **Orchestrator** (step 1, ~0.5 s): decomposes the task, produces a function signature and test specification as structured JSON.
    2. **Writer** (step 2, ~1.5 s): generates code given the spec.
    3. **Test runner and doc writer in parallel** (step 3, ~1.5 s each): test runner executes the code and returns pass/fail + error messages; doc writer generates docstrings and README from the code and spec simultaneously.
    4. **Orchestrator again** (step 4, ~0.5 s): if tests pass, assemble the final output. If tests fail, dispatch the writer again with the error messages; but this retry exceeds the 5-second SLA, so the system should return a partial result with a flag rather than timing out silently.

    Total on the happy path: 0.5 + 1.5 + 1.5 + 0.5 = **4 seconds**, within SLA. The critical path goes through writer → test runner; documentation can run in parallel without contributing to critical-path latency.

    Secondary concerns: rate limits (if all three parallel agents share the same API key), token budget (pass only the function signature to the doc writer, not full context history), and retry policy (exponential backoff on transient API errors, hard cutoff at 4.5 s to leave room for the final assembly step).

## Putting It Together: A Reference Architecture

A production multi-agent system for a complex task (e.g., automated software engineering, research synthesis, data pipeline construction) typically has this layered structure:

{{fig:marl-reference-architecture-layers}}

The harness layer (addressed further in [Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html)) enforces SLAs, cost limits, and per-agent sandboxing. The tool layer is where [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html) and [The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html) operate. Agent evaluation — how you know the system is working — is covered in [Agent Evaluation & Benchmarks](../08-agents-harness/08-agent-evaluation.html).

!!! key "Key Takeaways"
    - Multi-agent systems attack context limits, error isolation, and parallelism — but every agent boundary costs tokens and latency. Benchmark the single-agent baseline first.
    - The five canonical topologies are orchestrator-worker (star), pipeline (chain), debate (peer-to-peer), blackboard (shared state), and hierarchical (tree). Real systems compose them.
    - The workflow vs. agent distinction matters: a workflow has developer-written control flow; an agent has model-driven control flow. Use workflows for predictable, auditable steps; use agents at the leaves where flexibility is needed.
    - LangGraph is best for stateful workflows where you want typed, inspectable state. AutoGen suits open-ended multi-turn conversations. CrewAI speeds up structured pipelines. OpenAI Swarm illustrates the handoff pattern but is educational-grade.
    - Shared-state (blackboard) architectures enable concurrent writers and emergent synthesis, but require thread-safe access and explicit conflict resolution.
    - Structured handoffs (Pydantic schemas at every inter-agent boundary) reduce silent error propagation. Validate payloads before forwarding.
    - The multi-agent "tax" is real: a 5-agent pipeline with context forwarding can cost 3–5× more tokens than a single well-prompted call. The premium is justified only when measurable quality gain exceeds the cost.
    - Self-critique loops (worker → scorer → retry with critique) are simple forms of policy gradient that can be applied without any model fine-tuning.
    - Anti-patterns: agent soup (unclear role boundaries), infinite loops (no iteration cap), blind trust (no output validation), and over-parallelism (ignoring rate limits).

!!! sota "State of the Art & Resources (2026)"
    Multi-agent LLM orchestration has matured rapidly since 2023: production frameworks now support typed state machines, conversational handoffs, and role-based crews, while research on debate, self-critique, and agentic RL continues to push quality ceilings. The central engineering challenge remains balancing the quality gains of decomposition against the latency, token cost, and error-propagation penalties of inter-agent boundaries.

    **Foundational work**

    - [Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* (2023)](https://arxiv.org/abs/2308.08155) — introduced the conversational-agent model where agents exchange natural-language messages; the most-cited multi-agent framework paper.
    - [Hong et al., *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework* (2023)](https://arxiv.org/abs/2308.00352) — encodes software-engineering SOPs into structured roles and shared message pools, achieving SOTA on code-generation benchmarks.
    - [Sumers et al., *Cognitive Architectures for Language Agents* (2024)](https://arxiv.org/abs/2309.02427) — unified theoretical framework (CoALA) connecting memory, action space, and planning to cognitive science; the canonical survey for understanding *why* agent architectures are designed the way they are.

    **Recent advances (2023–2026)**

    - [Liang et al., *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate* (2023)](https://arxiv.org/abs/2305.19118) — empirical study showing structured debate between agents reduces degeneration-of-thought and improves factual accuracy over single-model self-reflection.
    - [Chen et al., *A Survey on LLM-based Multi-Agent System: Recent Advances and New Frontiers* (2024)](https://arxiv.org/abs/2412.17481) — comprehensive survey covering topologies, communication patterns, and open challenges across 200+ recent papers.
    - [Anthropic, *Building Effective Agents* (2024)](https://www.anthropic.com/research/building-effective-agents) — practitioner guide distinguishing workflows from agents, advocating simple composable patterns before complex frameworks; widely referenced in production teams.

    **Open-source & tools**

    - [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — low-level state-machine orchestration framework for stateful, long-running agents; typed state dicts, conditional edges, and built-in checkpointing.
    - [microsoft/autogen](https://github.com/microsoft/autogen) — conversational-agent framework where agents coordinate via natural-language message exchange; 58k+ GitHub stars.
    - [crewAIInc/crewai](https://github.com/crewaiinc/crewai) — role-based multi-agent framework with Crews (autonomous collaboration) and Flows (production event-driven pipelines); 52k+ stars, independent of LangChain.
    - [openai/swarm](https://github.com/openai/swarm) — minimal educational reference implementation of the agent-handoff pattern; superseded in production by the OpenAI Agents SDK but ideal for studying core mechanics.

    **Go deeper**

    - [OpenAI Agents SDK — Agent Orchestration Guide](https://openai.github.io/openai-agents-python/multi_agent/) — official docs covering orchestration-via-code vs. LLM-driven routing, handoffs, and agents-as-tools patterns; the production successor to Swarm.

## Further Reading

- **Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation," Microsoft Research, 2023** — the paper introducing the AutoGen conversational-agent framework.
- **Hong et al., "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework," 2023** — introduces structured role assignment and shared message pools for software engineering tasks.
- **OpenAI Swarm repository (github.com/openai/swarm)** — minimal reference implementation of the handoff pattern, useful for studying the core mechanics.
- **LangGraph documentation and examples (github.com/langchain-ai/langgraph)** — the canonical source for state-machine-based agent orchestration.
- **Liang et al., "Encouraging Divergent Thinking in Large Language Models through Debate," 2023** — empirical study of multi-agent debate improving factual accuracy over single-model baselines.
- **Chase, "LangChain Expression Language (LCEL) and agent patterns," LangChain blog** — practical treatment of chain composition and routing in production systems.
- **Sumers et al., "Cognitive Architectures for Language Agents," TMLR 2024** — a unified theoretical framework covering memory, action, and planning in LLM agents, connecting multi-agent patterns to cognitive science.

## Exercises

**1.** For each of the following designs, decide whether the chapter would call it a **workflow** or an **agent**, and justify your answer in one sentence using the chapter's definition (who owns the control flow):

- (a) A customer-support system where the model reads a ticket and emits `{"action": "transfer_to_billing"}` or `{"action": "transfer_to_technical"}`, and the framework routes accordingly.
- (b) A document pipeline in which Python code runs `translate()`, then `summarize()`, then `format()` on every input in that fixed order.
- (c) The LangGraph research graph from this chapter, whose `route_after_critic` edge sends the state back to the researcher unless the critique contains `APPROVED`.

??? note "Solution"
    The chapter's test is: *does the model decide the next action, or does developer-written code?* A **workflow** is a DAG whose routing is deterministic and written in code; an **agent** lets the model itself choose the next action.

    - (a) **Agent.** The model's own output (`transfer_to_billing` vs `transfer_to_technical`) selects the next step — this is exactly the "agent-as-router" / Swarm handoff pattern, where control flow emerges from model output.
    - (b) **Workflow.** The three stages run in a fixed, code-specified sequence; the model never chooses what runs next. It is a deterministic pipeline (chain topology).
    - (c) **Hybrid, but the routing is a workflow.** The *edges* are developer-written code: `route_after_critic` is a Python function whose branch is decided by a string check (`"APPROVED" in state["critique"]`) plus a hard iteration cap. The model produces content, but the graph — not the model — owns control flow. This is the chapter's recommended pattern: a workflow shell (LangGraph edges) with model-driven work at the nodes.

**2.** The chapter models the probability that at least one of $N$ independent agents (each succeeding with probability $p$) solves a task where any single correct solution suffices, as $P = 1 - (1-p)^N$.

- (a) Compute $P$ for $p = 0.6$ with $N = 1, 2, 3, 4$.
- (b) Your API budget lets you run either **one agent on the hard task** ($p = 0.6$) or **spend the same tokens elsewhere**. Roughly how many agents $N$ do you need so that $P \ge 0.95$?
- (c) The chapter says the gain is largest for moderate $p$ and negligible for trivial or impossible tasks. Verify this by comparing the *absolute* gain from going $N=1 \to N=3$ at $p = 0.6$ versus at $p = 0.95$.

??? note "Solution"
    (a) Using $P = 1 - (1-p)^N$ with $1-p = 0.4$:

    - $N=1:\ 1 - 0.4 = 0.600$
    - $N=2:\ 1 - 0.4^2 = 1 - 0.16 = 0.840$
    - $N=3:\ 1 - 0.4^3 = 1 - 0.064 = 0.936$
    - $N=4:\ 1 - 0.4^4 = 1 - 0.0256 = 0.9744$

    (b) We need $1 - 0.4^N \ge 0.95$, i.e. $0.4^N \le 0.05$. Take logs:

    $$
    N \ge \frac{\ln 0.05}{\ln 0.4} = \frac{-2.9957}{-0.9163} \approx 3.27
    $$

    So $N = 4$ agents (from part (a), $N=3$ gives $0.936 < 0.95$ and $N=4$ gives $0.974 \ge 0.95$). Four agents.

    (c) Absolute gain going $N=1 \to N=3$:

    - At $p = 0.6$: $0.936 - 0.600 = 0.336$ (a 33.6-point jump).
    - At $p = 0.95$: $\big(1 - 0.05^3\big) - 0.95 = (1 - 0.000125) - 0.95 = 0.999875 - 0.95 = 0.0499$ (about 5 points).

    The moderate-$p$ task gains ~0.34 while the near-certain task gains only ~0.05 for the *same* 3x cost, confirming that ensembling multiple agents pays off most when a single agent is unreliable but not hopeless.

**3.** Reuse the chapter's cost model from the "5-agent pipeline" worked example (input USD 0.15 / 1M tokens, output USD 0.60 / 1M tokens). A team proposes cutting the pipeline from **5 stages to 3 stages** while keeping the same per-stage numbers: each stage still has a 1,000-token system prompt and generates 400 output tokens, and the forwarded context still grows by 500 tokens per stage starting at 0 (so stage 1 = 0, stage 2 = 500, stage 3 = 1,000).

- (a) Compute the input tokens, output tokens, and dollar cost per run of the 3-stage pipeline.
- (b) At 10,000 runs/day for 30 days, what is the monthly cost, and what fraction of the 5-stage pipeline's ~USD 800/month is saved?

??? note "Solution"
    (a) **Input tokens:** system prompts $3 \times 1000 = 3000$; forwarded context $0 + 500 + 1000 = 1500$. Total input $= 3000 + 1500 = 4500$ tokens.

    **Output tokens:** $3 \times 400 = 1200$ tokens.

    **Cost per run:**

    $$
    C = \frac{4500 \times 0.15}{10^6} + \frac{1200 \times 0.60}{10^6} = \$0.000675 + \$0.00072 = \$0.001395
    $$

    So about **USD 0.0014 per run**.

    (b) Monthly cost: $0.001395 \times 10{,}000 \times 30 = \$418.5 \approx$ **USD 419/month**.

    The 5-stage pipeline cost about USD 800/month (USD 0.0027/run). Savings $\approx 800 - 419 = \$381$, i.e. roughly **48%** cheaper. The reduction is more than linear in stage count because dropping the last two stages removes both their fixed system-prompt overhead *and* the largest forwarded-context blocks (the 1,500- and 2,000-token injections), which is the compounding "multi-agent tax" the chapter warns about.

**4.** The chapter lists **over-parallelism** as an anti-pattern: "spawning 20 workers in parallel when the rate limit is 60 requests/minute will cause most to fail or be throttled." The `orchestrate` function submits *every* work item to the `ThreadPoolExecutor` at once. Modify the worker layer so that no more than `rate_limit` model calls start per 60-second window, while still running workers concurrently up to that bound. Keep the chapter's `WorkItem` / `WorkResult` types and `call_model` unchanged.

??? note "Solution"
    A simple, correct approach is a token-bucket-style throttle enforced by a shared, thread-safe gate that every worker must pass before it calls the model. We block a worker until a slot in the current 60-second window is free.

    ```python
    import threading
    import time
    from collections import deque

    class RateLimiter:
        """Allow at most `max_calls` acquisitions per `period` seconds.

        Thread-safe: shared by all worker threads. Each worker calls
        acquire() and blocks until a slot in the rolling window is free.
        """
        def __init__(self, max_calls: int, period: float = 60.0):
            self.max_calls = max_calls
            self.period = period
            self._calls: deque[float] = deque()   # timestamps of recent starts
            self._lock = threading.Lock()

        def acquire(self) -> None:
            while True:
                with self._lock:
                    now = time.monotonic()
                    # Drop timestamps older than the window
                    while self._calls and now - self._calls[0] >= self.period:
                        self._calls.popleft()
                    if len(self._calls) < self.max_calls:
                        self._calls.append(now)
                        return
                    # Otherwise, compute how long until the oldest call ages out
                    wait = self.period - (now - self._calls[0])
                time.sleep(max(wait, 0.01))   # sleep OUTSIDE the lock


    def run_worker_limited(item: WorkItem, limiter: RateLimiter) -> WorkResult:
        """Same as run_worker, but gated by the shared rate limiter."""
        try:
            limiter.acquire()                 # blocks if the window is full
            user_msg = item.task
            if item.context:
                user_msg = f"Context:\n{item.context}\n\nTask:\n{item.task}"
            output = call_model(item.system_prompt, user_msg)
            return WorkResult(worker_id=item.worker_id, output=output)
        except Exception as exc:
            return WorkResult(worker_id=item.worker_id, output="",
                              success=False, error=str(exc))
    ```

    Wire it into `orchestrate` by creating one shared limiter and passing it to every worker:

    ```python
    def orchestrate(goal: str, max_workers: int = 4, rate_limit: int = 60) -> str:
        print(f"[Orchestrator] Planning: {goal}")
        work_items = plan(goal)
        print(f"[Orchestrator] {len(work_items)} sub-tasks: "
              f"{[w.worker_id for w in work_items]}")

        limiter = RateLimiter(max_calls=rate_limit, period=60.0)
        results: list[WorkResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_worker_limited, item, limiter): item
                for item in work_items
            }
            for future in as_completed(futures):
                result = future.result()
                status = "OK" if result.success else f"FAILED: {result.error}"
                print(f"  [Worker {result.worker_id}] {status}")
                results.append(result)

        print("[Orchestrator] Synthesising results ...")
        return synthesise(goal, results)
    ```

    Key points: (1) the limiter is *shared* across threads and guarded by a `Lock`, so the rolling window is counted correctly under concurrency; (2) `time.sleep` happens **outside** the lock so a waiting worker does not block others from checking the window; (3) `ThreadPoolExecutor(max_workers=...)` still caps in-flight concurrency, while the limiter caps the *rate* — the two constraints compose. Note the plan itself already fits the pool, but the limiter protects against the case where `len(work_items)` (or a burst of retries) exceeds the provider's requests-per-minute ceiling.

**5.** The chapter's `self_critique_loop` retries a single worker with an appended critique until its score clears a threshold. Two problems with using it in production: (i) it can keep the *worse* of two attempts, since it returns the last attempt after `max_retries` rather than the best-scoring one; and (ii) the critique text accumulates unboundedly in `task`, inflating token cost on every retry. Rewrite the loop to (a) track and return the **highest-scoring** output seen, and (b) keep only the **most recent** critique in the prompt rather than appending forever. Preserve the signature and the early-return-on-success behavior.

??? note "Solution"
    We separate the *base task* (never mutated) from the *latest critique* (replaced, not appended), and remember the best (score, output) pair so a late low-scoring attempt cannot overwrite an earlier good one.

    ```python
    def self_critique_loop(
        orchestrator_prompt: str,
        worker_fn,
        scoring_fn,
        task: str,
        threshold: float = 0.8,
        max_retries: int = 3,
    ) -> str:
        """
        Run a worker, score it, retry with critique if below threshold.
        Returns the highest-scoring output seen across all attempts.
        scoring_fn: (task, output) -> float in [0, 1]
        """
        base_task = task                 # immutable reference task
        latest_critique = ""             # replaced each round, not appended
        best_output = ""
        best_score = -1.0

        for attempt in range(max_retries):
            # Build the prompt from the base task + only the most recent critique
            prompt = base_task
            if latest_critique:
                prompt = (f"{base_task}\n\n[Revise. Previous critique: "
                          f"{latest_critique}]")

            output = worker_fn(prompt)
            score = scoring_fn(base_task, output)

            # (a) track the best attempt regardless of ordering
            if score > best_score:
                best_score = score
                best_output = output

            if score >= threshold:
                return output            # early return on success, unchanged

            # (b) overwrite the critique instead of appending
            latest_critique = call_model(
                "You are a quality assessor. Identify what is wrong or missing.",
                f"Task: {base_task}\n\nOutput:\n{output}\n\nScore: {score:.2f}",
            )

        return best_output               # best-effort = highest-scoring attempt
    ```

    Why this fixes both problems:

    - **(a) Best-of-N return.** `best_output`/`best_score` are updated on every attempt with a strict `>` comparison, so after `max_retries` we return the highest-scoring attempt rather than blindly the last one. If attempt 2 scored 0.75 and attempt 3 regressed to 0.60, we now return attempt 2.
    - **(b) Bounded prompt growth.** `base_task` is fixed and `latest_critique` is *reassigned* each round, so the retry prompt stays roughly constant in size (base task + one critique) instead of growing by one full critique per iteration. This keeps the per-retry input token count flat, directly addressing the "coordination overhead" cost the chapter flags.

    The early success path (`score >= threshold`) is preserved exactly.
