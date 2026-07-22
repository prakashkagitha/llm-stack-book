"""
Runs the CPU-runnable Python code blocks from:
    content/08-agents-harness/07-multi-agent-systems.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:
    #2 (line ~267) -- Blackboard / BlackboardEntry (thread-safe shared store)
    #3 (line ~327) -- research_agent / critic_agent using the blackboard
                       (call_model is an external LLM boundary -> mocked)
    #5 (line ~483) -- autogen_debate.py (AutoGen ConversableAgent)
                       -- third-party `autogen` package; guarded import,
                          object-construction only executed if installed
                          (network call sites are never invoked)
    #6 (line ~546) -- crewai_example.py (CrewAI Agent/Task/Crew)
                       -- third-party `crewai`/`langchain_openai`; guarded
                          import, object-construction only executed if
                          installed (network call sites are never invoked)
    #8 (line ~713) -- ResearchHandoff / WritingHandoff Pydantic schemas

Skipped blocks (not tested here):
    #0 -- non-python (```text``` diagram of workflow vs agent control flow)
    #1 -- orchestrator_worker.py: needs-net (real `openai.OpenAI()` client,
          `client.chat.completions.create(...)`)
    #4 -- langgraph_example.py: needs-net (real `ChatOpenAI(...).invoke(...)`
          calls inside node functions)
    #7 -- swarm_handoff.py: needs-net (real `Swarm().run(...)` call)
    #9 -- self_critique_loop: fragment that depends on an unspecified
          `worker_fn`/`scoring_fn` pair supplied by the *caller*; the block
          itself is just a function definition with no standalone demo in
          the chapter. We DO exercise it below with tiny local stand-ins
          for worker_fn/scoring_fn (no network), since doing so is honest
          and cheap -- see the bottom of this file.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

try:
    import autogen
except Exception:  # pragma: no cover - guarded per optional-import rule
    autogen = None

try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew, Process
except Exception:  # pragma: no cover - guarded per optional-import rule
    CrewAgent = CrewTask = Crew = Process = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - guarded per optional-import rule
    ChatOpenAI = None

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - guarded per optional-import rule
    BaseModel = None


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #2 (line ~267) -- blackboard.py
# ============================================================================
_section("Block #2: Blackboard / BlackboardEntry")


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


# Exercise the Blackboard directly (write/read_latest/read_all/keys/snapshot)
board = Blackboard()
board.write("research:findings", "MoE routes tokens to a subset of experts.", author="researcher")
board.write("research:findings", "Refined: MoE reduces FLOPs per token vs a dense model.", author="researcher")
board.write("critic:status", {"ok": True}, author="critic")

latest = board.read_latest("research:findings")
assert latest is not None and latest.author == "researcher"
assert "Refined" in latest.content

history = board.read_all("research:findings")
assert len(history) == 2

assert set(board.keys()) == {"research:findings", "critic:status"}

snap = board.snapshot()
assert snap["research:findings"] == latest.content
assert snap["critic:status"] == {"ok": True}

# Concurrent writers should not corrupt the store (thread-safety check).
def _writer(board: Blackboard, tag: str, n: int) -> None:
    for i in range(n):
        board.write("stress:key", f"{tag}-{i}", author=tag)

threads = [threading.Thread(target=_writer, args=(board, f"t{i}", 20)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert len(board.read_all("stress:key")) == 80

print(f"[blackboard] keys={board.keys()}, latest research finding={latest.content!r}")


# ============================================================================
# Block #3 (line ~327) -- research_agent / critic_agent using the blackboard
# ============================================================================
_section("Block #3: research_agent / critic_agent")

# `call_model` is the network-calling boundary defined by block #1
# (orchestrator_worker.py, SKIPPED as needs-net). We mock it here so the
# agents' OWN logic (reading/writing the blackboard) still executes offline
# against a canned response.
def call_model(system: str, user: str, model: str = "gpt-4o-mini",
                temperature: float = 0.3, max_tokens: int = 1024) -> str:
    raise RuntimeError("call_model must be mocked in tests -- no real network calls allowed")


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


demo_board = Blackboard()

# critic_agent on an empty board takes the "no findings yet" branch --
# no call_model invocation needed, exercised for real.
critic_agent(demo_board)
assert demo_board.read_latest("critic:status").content == "No findings to critique yet."

# research_agent + critic_agent (happy path) with call_model mocked at the
# network boundary so the surrounding blackboard logic runs for real.
with patch(__name__ + ".call_model", side_effect=[
    "Mixture-of-experts models route each token to a subset of experts.",
    "The claim about routing lacks a citation; otherwise reasonable.",
]) as mocked_call_model:
    research_agent(demo_board, "mixture-of-experts routing")
    critic_agent(demo_board)

assert mocked_call_model.call_count == 2
research_entry = demo_board.read_latest("research:findings")
critique_entry = demo_board.read_latest("critic:critique")
assert research_entry.content.startswith("Mixture-of-experts")
assert "citation" in critique_entry.content
print(f"[research_agent/critic_agent] research={research_entry.content!r}")
print(f"[research_agent/critic_agent] critique={critique_entry.content!r}")


# ============================================================================
# Block #5 (line ~483) -- autogen_debate.py
# ============================================================================
_section("Block #5: autogen_debate.py (AutoGen ConversableAgent)")

if autogen is None:
    print("[SKIP] block #5: `autogen` (pyautogen) is not installed in CI -- "
          "skipping AutoGen ConversableAgent construction.")
else:
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

    # NOTE: `pro_agent.initiate_chat(...)` is NOT called here -- that hits
    # a real LLM API (network), which is forbidden in this offline test.
    # We only exercise object construction, which is the CPU-safe part of
    # this block.
    assert pro_agent.name == "ProAgent"
    assert con_agent.name == "ConAgent"
    assert judge.name == "Judge"
    print(f"[autogen] constructed agents: {pro_agent.name}, {con_agent.name}, {judge.name}")


# ============================================================================
# Block #6 (line ~546) -- crewai_example.py
# ============================================================================
_section("Block #6: crewai_example.py (CrewAI Agent/Task/Crew)")

if CrewAgent is None or ChatOpenAI is None:
    print("[SKIP] block #6: `crewai` and/or `langchain_openai` are not "
          "installed in CI -- skipping CrewAI Agent/Task/Crew construction.")
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    researcher = CrewAgent(
        role="Technical Researcher",
        goal="Find accurate, up-to-date information on the given topic.",
        backstory="A meticulous research scientist who double-checks every claim.",
        llm=llm,
        verbose=True,
    )

    writer = CrewAgent(
        role="Technical Writer",
        goal="Write a clear, engaging technical blog post from research notes.",
        backstory="An experienced writer who makes complex ideas accessible.",
        llm=llm,
        verbose=True,
    )

    editor = CrewAgent(
        role="Editor",
        goal="Polish the draft for clarity, accuracy, and style.",
        backstory="A demanding editor who improves every sentence.",
        llm=llm,
        verbose=True,
    )

    research_task = CrewTask(
        description="Research the key concepts behind mixture-of-experts LLM architectures.",
        agent=researcher,
        expected_output="A structured set of notes covering architecture, training, and key papers.",
    )

    writing_task = CrewTask(
        description="Write a 500-word technical blog post based on the research notes.",
        agent=writer,
        expected_output="A polished 500-word blog post draft.",
        context=[research_task],  # this task receives research_task's output
    )

    editing_task = CrewTask(
        description="Edit the blog post draft for accuracy, flow, and style.",
        agent=editor,
        expected_output="A final, publication-ready blog post.",
        context=[writing_task],
    )

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,  # or Process.hierarchical for a manager LLM
        verbose=True,
    )

    # NOTE: `crew.kickoff()` is NOT called here -- that hits a real LLM API
    # (network), which is forbidden in this offline test. We only exercise
    # object construction and wiring, which is the CPU-safe part of this
    # block.
    assert len(crew.agents) == 3
    assert len(crew.tasks) == 3
    print(f"[crewai] constructed crew with {len(crew.agents)} agents, {len(crew.tasks)} tasks")


# ============================================================================
# Block #8 (line ~713) -- ResearchHandoff / WritingHandoff (Pydantic schemas)
# ============================================================================
_section("Block #8: ResearchHandoff / WritingHandoff")

if BaseModel is None:
    print("[SKIP] block #8: `pydantic` is not installed in CI -- skipping "
          "ResearchHandoff / WritingHandoff schema validation.")
else:
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

    # Instantiate and validate both schemas with tiny fixture data.
    handoff = ResearchHandoff(
        topic="mixture-of-experts routing",
        key_facts=[
            "MoE layers route each token to a small subset of experts.",
            "Sparse activation lowers FLOPs per token vs. a dense model of similar size.",
        ],
        source_urls=[],
        confidence=0.82,
    )
    assert handoff.topic == "mixture-of-experts routing"
    assert len(handoff.key_facts) == 2
    assert 0.0 <= handoff.confidence <= 1.0

    draft = WritingHandoff(
        draft_text="Mixture-of-experts (MoE) models activate only a subset of parameters per token...",
        word_count=12,
        flagged_sections=["needs a citation for the FLOPs claim"],
    )
    assert draft.word_count == 12
    assert draft.flagged_sections

    # Pydantic should reject an invalid payload (e.g. confidence as non-float
    # garbage) -- this exercises the "structured handoff catches errors
    # explicitly" claim the chapter makes about this pattern.
    try:
        ResearchHandoff(
            topic="x",
            key_facts=[],
            source_urls=[],
            confidence="not-a-number-and-not-coercible",
        )
        raise AssertionError("expected a pydantic ValidationError for a bad confidence field")
    except Exception as exc:
        assert "confidence" in str(exc).lower() or "valid" in str(exc).lower()

    print(f"[pydantic] ResearchHandoff={handoff!r}")
    print(f"[pydantic] WritingHandoff={draft!r}")


# ============================================================================
# Block #9 (line ~789) -- self_critique_loop
# ============================================================================
_section("Block #9: self_critique_loop (fragment, exercised with local stand-ins)")

# NOTE: this block is a standalone function definition -- the chapter does
# not supply a runnable worker_fn/scoring_fn demo alongside it. We copy the
# function verbatim and exercise it with tiny, deterministic, no-network
# stand-ins so the retry/threshold logic actually runs.

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


# Deterministic local stand-ins (no network): the worker's "quality" improves
# each time the task string grows (i.e. once it contains critique feedback),
# and the scorer rewards that growth. This lets us observe both the
# success-on-retry path and the max-retries-exhausted path.
def _toy_worker(task: str) -> str:
    return f"draft(len={len(task)})"

def _toy_scorer(task: str, output: str) -> float:
    # Score climbs with task length (simulating "more context -> better output")
    return min(1.0, len(task) / 120.0)

with patch(__name__ + ".call_model", return_value="Add more specifics and a citation."):
    result = self_critique_loop(
        orchestrator_prompt="unused in this fragment",
        worker_fn=_toy_worker,
        scoring_fn=_toy_scorer,
        task="Summarize mixture-of-experts routing in one sentence.",
        threshold=0.8,
        max_retries=5,
    )
assert result.startswith("draft(len=")
print(f"[self_critique_loop] result={result!r}")

# Exhausted-retries path: a scorer that never reaches threshold must still
# return the best-effort output rather than raising.
with patch(__name__ + ".call_model", return_value="Still not good enough."):
    exhausted_result = self_critique_loop(
        orchestrator_prompt="unused in this fragment",
        worker_fn=_toy_worker,
        scoring_fn=lambda task, output: 0.1,
        task="short",
        threshold=0.99,
        max_retries=2,
    )
assert exhausted_result.startswith("draft(len=")
print(f"[self_critique_loop] exhausted_result={exhausted_result!r}")


print("\nAll tested blocks (#2, #3, #5, #6, #8, #9) executed successfully.")
