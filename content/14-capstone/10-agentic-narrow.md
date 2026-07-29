# 14.10 A Narrow Auto-Research Agent: ReAct, Tool-Use & Retrieval by Distillation

We now do the most ambitious thing a 100M-parameter model can honestly do: turn **Stack-100M** into a *narrow auto-research agent*. Given a question, it will interleave **thought → tool-call → observation** — searching a small local corpus, reading what it finds, doing exact arithmetic with a calculator, and finally synthesizing a short, grounded answer. This is the **ReAct** pattern (Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022), and it is the capstone of the capstone: everything we built — the tokenizer's reserved tool tokens (Ch. 14.3), the SFT loop (Ch. 14.9), the narrow-RLVR machinery (Ch. 14.9) — comes together here.

There is one non-negotiable truth to state up front, because it shapes the entire chapter. **A 100M base model will not discover multi-step tool use on its own.** It has neither the in-context-learning strength of a 70B model nor the reasoning depth to plan across turns from a few prompt examples. The only thing that works at this scale is **distillation**: we let a large teacher model produce many ReAct trajectories, keep only the ones that actually solved the task, reformat them with our tool special tokens, and **supervise-fine-tune Stack-100M to imitate the successful traces**. The 100M model does not learn to *reason about* tool use; it learns to *reproduce a narrow, well-worn groove* of tool use. Inside that groove it is genuinely useful. One millimeter outside it, it falls apart. We will be brutally honest about exactly where that wall is.

This chapter builds directly on three deeper chapters you should have open in another tab: [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html) for the loop mechanics, [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html) for the schema/parse/dispatch primitives, and [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html) for the retriever. For the training half we lean on [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html), and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

## Why Distillation Is the Only Path at 100M

### The capability gap, quantified

Multi-step tool use is a *compositional* skill. To answer "In what year was the author of the paper that introduced RoPE born, and what is that year times two?" the agent must (1) decide to search, (2) form a good query, (3) read the observation, (4) extract a fact, (5) decide to search *again* for a different fact, (6) call the calculator, and (7) synthesize. Each step conditions on the outcome of the previous one. A single wrong branch — a malformed tool call, a query that retrieves nothing, a misread number — cascades.

Large models absorb this pattern from a handful of in-context examples because their pretraining already contains millions of implicit "act, observe, revise" structures (code with REPL sessions, forum threads, worked solutions). A 100M model pretrained on ~20B tokens of FineWeb-Edu and Cosmopedia (Ch. 14.2) has seen far less of this and has far less capacity to generalize it. If you few-shot-prompt Stack-100M with three ReAct exemplars, you will observe the classic small-model failure modes: it emits a plausible-looking `Thought:` and then an **answer with no tool call at all** (it "hallucinates the observation"), or it emits a tool call and then **ignores the returned observation**, or it loops forever re-issuing the same search.

{{fig:distilled-groove-vs-skill}}

The fix is to make the *behavior* itself the training signal. This is knowledge distillation in the behavioral sense (see [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html)): rather than matching the teacher's soft logits token-for-token, we match the teacher's *trajectories* — its sequence of thoughts, actions, and the resulting observations — as ordinary SFT targets. This is often called **trajectory distillation** or **rejection-sampling fine-tuning** (the same family as STaR, Zelikman et al., 2022, and the "distill from a stronger model" recipes behind essentially every small open instruction model, including the SmolLM series from HuggingFace, 2024–2025). The distinction matters at 100M: logit distillation needs the teacher and student to share a tokenizer and be queried in lockstep, which is expensive and brittle across model families; trajectory distillation only needs the teacher's *text output*, so any strong model can be the teacher and the student trains with the plain cross-entropy loop it already has.

### The pipeline in one diagram

```text
   ┌─────────────────────────────────────────────────────────────────┐
   │ 1. TASK POOL         narrow auto-research questions (+ gold ans) │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  each task
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 2. TEACHER ROLLOUT   large model runs the REAL ReAct loop with   │
   │                      our two tools -> raw trajectory             │
   │                      (in CI: teacher is STUBBED/offline)         │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  trajectory + did it solve the task?
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 3. FILTER            keep only trajectories whose final answer   │
   │                      exactly matches gold (verifiable reward)    │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  successful traces
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 4. FORMAT            splice in tool special tokens, mask loss so │
   │                      the model is NOT trained on observations    │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  SFT dataset
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 5. SFT Stack-100M    imitate the groove  (Ch. 14.9 loop reused)  │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  optional
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 6. RLVR (GRPO)       reward = final-answer exact match; sharpen  │
   └─────────────────────────────────────────────────────────────────┘
```

{{fig:distill-rejection-funnel}}

Steps 2–3 are exactly **rejection sampling against a verifiable reward** — the same reward we use in RLVR (Ch. 14.9), just applied offline to *filter demonstrations* instead of online to *estimate advantages*. That symmetry is deliberate and we will exploit it in the final section.

## The Two Tools: A Calculator and a Tiny Retriever

An agent is only as good as its tools, and at 100M we keep the toolset tiny and the tool *interface* rigidly regular so the model has the smallest possible surface to memorize. Two tools:

- `calc(expr)` — evaluates an arithmetic expression exactly. Transformers are notoriously bad at multi-digit arithmetic; delegating it is a textbook win ([Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html)).
- `search(query, k)` — retrieves the top-`k` passages from a small local corpus. We implement two backends: **BM25** (lexical) and an **embedding-lite** cosine retriever (a hashed bag-of-words, so it is fully self-contained with no external model). See [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html), [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html), and [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html) for the real versions.

These live in `stacklm.agent.tools`. Everything is pure-Python and dependency-light so CI can run it hermetically on CPU.

```python
# stacklm/agent/tools.py
"""Two narrow tools for the Stack-100M auto-research agent: an exact
calculator and a small local retriever (BM25 + an embedding-lite backend).
Everything here is deterministic, offline, and CI-safe."""

from __future__ import annotations
import ast, math, operator, re, hashlib
from collections import Counter
from dataclasses import dataclass


# ----------------------------------------------------------------------
# TOOL 1: a SAFE calculator.  We never call eval() on model output.
# Instead we parse to an AST and walk it, allowing only arithmetic nodes.
# ----------------------------------------------------------------------
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):              # a literal number
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def calc(expr: str) -> str:
    """Evaluate an arithmetic expression, returning a compact string.
    Errors are returned AS OBSERVATIONS (never raised) so the agent can
    read them and recover -- this is important for teaching self-correction."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        val = _eval_node(tree)
    except Exception as e:                          # noqa: BLE001 (we want to catch all)
        return f"CalcError: {e}"
    # normalise: integers print without a trailing .0
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


# ----------------------------------------------------------------------
# TOOL 2: a tiny retriever over an in-memory corpus of passages.
# ----------------------------------------------------------------------
_TOK = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


@dataclass
class Passage:
    doc_id: str
    text: str


class BM25Retriever:
    """Textbook Okapi BM25 (Robertson & Zaragoza). Lexical, exact, fast,
    and shockingly hard to beat on keyword-heavy narrow corpora."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [_tokenize(p.text) for p in passages]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.docs))
        # document frequency of each term
        df: Counter[str] = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        N = len(self.docs)
        # BM25 idf with the +0.5 smoothing; the +1.0 inside the log keeps it
        # non-negative even for terms in more than half the corpus.
        self.idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1.0)
                    for t, n in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query_terms: list[str], i: int) -> float:
        s, dl = 0.0, self.doc_len[i]
        for t in query_terms:
            if t not in self.idf:
                continue
            f = self.tf[i][t]
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf[t] * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        q = _tokenize(query)
        scored = [(self.passages[i], self.score(q, i)) for i in range(len(self.docs))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, sc) for p, sc in scored[:k] if sc > 0.0]


class HashEmbedRetriever:
    """'Embedding-lite': hash each token into a fixed-width vector via the
    hashing trick (Weinberger et al., 2009), L2-normalise, cosine-rank.
    No trained encoder, no network -- a stand-in for a real dense retriever
    (see Ch. 9.1) that is still meaningfully semantic-ish on paraphrases."""

    def __init__(self, passages: list[Passage], dim: int = 512):
        self.passages, self.dim = passages, dim
        self.vecs = [self._embed(p.text) for p in passages]

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in _tokenize(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0     # signed hashing reduces collision bias
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        qv = self._embed(query)
        scored = [(self.passages[i], sum(a * b for a, b in zip(qv, self.vecs[i])))
                  for i in range(len(self.passages))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, sc) for p, sc in scored[:k] if sc > 0.0]
```

The calculator returns errors *as strings* rather than raising. This is a deliberate agent-design choice: an observation like `CalcError: disallowed expression node` is something the model can *read* and recover from, and the teacher's successful trajectories will occasionally contain exactly such recover-then-retry sequences — which is precisely the robust behavior we want Stack-100M to imitate.

!!! tip "Practitioner tip: hybrid retrieval, even here"

    On narrow, keyword-heavy corpora BM25 usually wins; on paraphrased queries the hashed-embedding backend catches synonyms BM25 misses. The production move is a **hybrid** — union the two candidate sets and rerank — exactly as in [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html). For the 100M agent we keep it to a single backend per run to reduce the observation-format variety the model must memorize, but we expose both so you can ablate which one distills better.

## The ReAct Trace Format & Tool Special Tokens

The single most important design decision for a small agent is the **wire format** — the exact token sequence that represents a step. We keep it brutally regular. Recall the reserved special tokens from the tokenizer (Ch. 14.3): the chat roles `<|system|> <|user|> <|assistant|> <|end|>` and the tool tokens `<|tool_call|> <|tool_result|>`. A trajectory is a single chat turn whose assistant span *contains* the interleaved calls:

```text
<|system|>You are a narrow research assistant. Think, then either call a
tool or give a final answer. Tools: search(query,k), calc(expr).<|end|>
<|user|>How old was RoPE's introduction year doubled? Use the corpus.<|end|>
<|assistant|>Thought: I need the year RoPE was introduced. I'll search.
<|tool_call|>{"tool": "search", "args": {"query": "RoPE rotary introduced year", "k": 2}}<|end|>
<|tool_result|>[1] RoFormer introduced RoPE in 2021. [2] RoPE rotates query/key vectors...<|end|>
<|assistant|>Thought: The year is 2021. Now double it with the calculator.
<|tool_call|>{"tool": "calc", "args": {"expr": "2021*2"}}<|end|>
<|tool_result|>4042<|end|>
<|assistant|>Thought: I have everything.
Answer: 4042<|end|>
```

Three rules make this format learnable at 100M:

1. **Tool calls are JSON, always, with a fixed key order** (`tool` then `args`). A single canonical schema means the model memorizes one grammar, not many. In production you would enforce this with constrained decoding ([Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)); at 100M, format regularity in the training data does most of that work for free.
2. **Observations are wrapped in `<|tool_result|> … <|end|>` and are NOT part of the loss.** The model must *read* observations but must never be trained to *produce* them — they come from the environment. We enforce this with loss masking (next section). This is the same masking discipline as assistant-only SFT loss ([Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html)), extended to also mask tool results.
3. **The final answer is delimited** by the literal `Answer: ` prefix so a verifier can extract it with a trivial regex — which is what makes the reward *verifiable* for both filtering and RLVR.

Here is the parser/serializer. It lives in `stacklm.agent.react` and is deliberately a small state machine — no clever regex soup, because the harness must fail *loudly and recoverably* on malformed model output.

```python
# stacklm/agent/react.py
"""ReAct trace (de)serialisation for Stack-100M. A trajectory is a list of
Steps; we render it to the tool-token wire format and parse the model's
streamed output back into an action."""

from __future__ import annotations
import json, re
from dataclasses import dataclass

# The reserved special tokens (must match stacklm/tokenizer specials, Ch. 14.3).
BOS, EOS, END = "<|bos|>", "<|eos|>", "<|end|>"
SYS, USER, ASST = "<|system|>", "<|user|>", "<|assistant|>"
TOOL_CALL, TOOL_RESULT = "<|tool_call|>", "<|tool_result|>"

ANSWER_RE = re.compile(r"Answer:\s*(.+?)\s*$", re.DOTALL)
# A tool call is the JSON object sitting between <|tool_call|> and <|end|>.


@dataclass
class Action:
    kind: str                    # "tool" or "final"
    thought: str = ""
    tool: str | None = None
    args: dict | None = None
    answer: str | None = None


def parse_assistant_step(text: str) -> Action:
    """Parse ONE assistant emission (thought + optional tool call OR answer).
    `text` is what the model generated up to (and excluding) <|end|>."""
    thought = ""
    m = re.search(r"Thought:\s*(.*?)(?:\n|$)", text)
    if m:
        thought = m.group(1).strip()

    if TOOL_CALL in text:
        payload = text.split(TOOL_CALL, 1)[1].split(END, 1)[0].strip()
        try:
            obj = json.loads(payload)
            return Action("tool", thought, obj["tool"], obj.get("args", {}))
        except Exception:
            # Malformed call: surface it so the harness returns an error obs.
            return Action("tool", thought, tool="__malformed__", args={"raw": payload})

    m = ANSWER_RE.search(text)
    if m:
        return Action("final", thought, answer=m.group(1).strip())

    # No tool call, no answer: the classic small-model derailment. Treat the
    # whole emission as the (probably wrong) final answer so the loop halts.
    return Action("final", thought, answer=text.strip())


def render_tool_result(obs: str) -> str:
    return f"{TOOL_RESULT}{obs}{END}"


def render_call(tool: str, args: dict) -> str:
    # Canonical key order (tool, args) -> one grammar for the model to learn.
    body = json.dumps({"tool": tool, "args": args}, separators=(", ", ": "))
    return f"{TOOL_CALL}{body}{END}"
```

{{fig:react-wire-format-loss-mask}}

## Distilling Trajectories From the Teacher

Now the heart of the chapter. We generate trajectories by running the *real ReAct loop* with a large teacher model, keep the winners, and turn them into SFT data.

### The real teacher call (prose) vs. the CI stub

In the full run you point the rollout at a strong instruction model — anything in the "solidly agentic" tier. The teacher is prompted with the tool schemas and a couple of gold exemplars, and it drives the *same* environment (`calc`, `search`) our agent will use. A single teacher rollout looks like this (real API sketch; do not run this in CI):

```python
# Illustrative ONLY -- the real teacher call. In CI this path is never taken.
import anthropic  # or any strong-model client

def teacher_step(client, messages, tools_desc):
    resp = client.messages.create(
        model="a-strong-agentic-model",     # the large teacher
        system=tools_desc,                    # tool schemas + 2 gold exemplars
        messages=messages,
        max_tokens=512, temperature=0.7,      # some temperature -> diverse traces
    )
    return resp.content[0].text
```

Temperature > 0 matters: we want *diverse* trajectories per task so that after filtering we retain several distinct successful solution shapes, not one. This is rejection sampling — sample many, keep the good — and its yield is a number you should actually measure (see the worked example below).

For CI and for anyone without a teacher budget, we ship a **deterministic stub teacher** that solves the toy tasks with a hand-written policy. It exercises *every* code path — tool dispatch, observation formatting, filtering, SFT export — with zero network. The interface is identical, so swapping in the real teacher is a one-line change.

```python
# stacklm/agent/distill.py
"""Generate ReAct trajectories, filter to successes, export SFT data.
The teacher is an injected callable so CI can pass a hermetic stub while
the full run passes a real large-model client."""

from __future__ import annotations
from dataclasses import dataclass
from stacklm.agent.react import (
    Action, parse_assistant_step, render_tool_result,
    SYS, USER, ASST, END,
)
from stacklm.agent.tools import calc, BM25Retriever, Passage


@dataclass
class Task:
    question: str
    gold: str                     # the verifiable final answer (exact match)


# ---- the environment: dispatch a parsed Action to a real tool -----------
class ToolEnv:
    def __init__(self, corpus: list[Passage]):
        self.retriever = BM25Retriever(corpus)

    def run_tool(self, act: Action) -> str:
        if act.tool == "calc":
            return calc(str(act.args.get("expr", "")))
        if act.tool == "search":
            hits = self.retriever.search(act.args.get("query", ""),
                                         int(act.args.get("k", 2)))
            if not hits:
                return "NoResults"
            return " ".join(f"[{i+1}] {p.text}" for i, (p, _) in enumerate(hits))
        return f"ToolError: unknown or malformed tool '{act.tool}'"


SYSTEM_PROMPT = ("You are a narrow research assistant. Think, then either "
                 "call a tool or give a final answer. Tools: search(query,k), "
                 "calc(expr).")


def rollout(task: Task, teacher, env: ToolEnv, max_steps: int = 6):
    """Run ONE ReAct trajectory with `teacher` (a callable prompt->text).
    Returns (transcript_str, solved: bool). The transcript is the full
    tool-token wire format including observations."""
    transcript = f"{SYS}{SYSTEM_PROMPT}{END}{USER}{task.question}{END}"
    solved = False
    for _ in range(max_steps):
        gen = teacher(transcript + ASST)           # teacher emits one step
        step_text = gen.split(END, 1)[0]           # up to the first <|end|>
        act = parse_assistant_step(step_text)
        # record the assistant emission verbatim (thought + call OR answer)
        transcript += f"{ASST}{step_text}{END}"
        if act.kind == "final":
            solved = (normalize(act.answer) == normalize(task.gold))
            break
        obs = env.run_tool(act)                     # REAL tool call
        transcript += render_tool_result(obs)      # inject observation
    return transcript, solved


def normalize(s: str | None) -> str:
    """Verifier-side normalisation so '4042', ' 4042 ', '4042.0' all match."""
    if s is None:
        return ""
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return " ".join(s.lower().split())


def distill(tasks, teacher, env, samples_per_task: int = 4):
    """Rejection sampling: multiple rollouts per task, keep only solved ones.
    Yields SFT-ready transcripts (deduplicated per task)."""
    kept = []
    for task in tasks:
        seen = set()
        for _ in range(samples_per_task):
            transcript, solved = rollout(task, teacher, env)
            if solved and transcript not in seen:
                seen.add(transcript)
                kept.append({"task": task.question, "text": transcript})
    return kept
```

### A hermetic stub teacher for CI

```python
# stacklm/agent/stub_teacher.py
"""A deterministic 'teacher' that solves the toy auto-research tasks by a
fixed policy. It parses the transcript so far and decides the next step --
exactly mimicking what a real large model would emit, but offline."""

import re
from stacklm.agent.react import TOOL_RESULT, END, render_call

def make_stub_teacher():
    def teacher(prompt: str) -> str:
        # How many observations have we already seen? Drives the policy.
        n_obs = prompt.count(TOOL_RESULT)
        last_obs = prompt.rsplit(TOOL_RESULT, 1)[-1].split(END, 1)[0] if n_obs else ""
        if n_obs == 0:
            # Step 1: always search for the entity in the question.
            q = re.search(r"about (.+?)[\?\.]", prompt)
            query = q.group(1) if q else "topic"
            return (f"Thought: I should look this up.\n"
                    f"{render_call('search', {'query': query, 'k': 2})}")
        if n_obs == 1 and re.search(r"\d{3,4}", last_obs):
            # Step 2: found a year -> double it (the toy task shape).
            year = re.search(r"\d{3,4}", last_obs).group(0)
            return (f"Thought: Found {year}; the task wants it doubled.\n"
                    f"{render_call('calc', {'expr': f'{year}*2'})}")
        # Final step: emit the last numeric observation as the answer.
        num = re.search(r"-?\d+", last_obs)
        ans = num.group(0) if num else last_obs.strip()
        return f"Thought: I have the result.\nAnswer: {ans}"
    return teacher
```

The stub is not cheating — it is a *test double*. Its job is to guarantee that the plumbing (rollout → filter → format → SFT) is correct and stays correct under CI, so that on the day you swap in a real teacher the only variable is trajectory *quality*, never code correctness.

!!! example "Worked example: rejection-sampling yield and token budget"

    Suppose our narrow task pool has **200 questions**, we take **`samples_per_task = 8`** teacher rollouts each, and the teacher solves this narrow task family with a per-rollout success probability of **on the order of 0.6**.

    Expected raw successes: $200 \times 8 \times 0.6 = 960$ solved trajectories. After deduping (successful traces for the same task often collapse to 2–3 distinct shapes), keep on the order of $200 \times 3 = 600$ unique traces.

    Now the **token budget**. Each trace is short: system + question ≈ 60 tokens, two `<|assistant|>` steps ≈ 40 tokens each, two observations ≈ 50 tokens each, final answer ≈ 15 tokens ⇒ ~**255 tokens/trace**, of which the **loss-bearing** (assistant) tokens are only ~95 (observations and the prompt are masked). So the SFT set is $600 \times 255 \approx 153{,}000$ tokens total, ~**57k supervised tokens**.

    That is *tiny* — a few minutes of fine-tuning. The scarce resource is not compute; it is **successful, diverse trajectories**. At a teacher cost of, say, on the order of \$1–3 per thousand rollouts, generating $200 \times 8 = 1600$ rollouts costs a few dollars — genuinely "the cost of a coffee," and the dominant line item is the teacher, not the SFT.

    The lesson: at 100M, *data curation is the whole game*. Spend your budget getting more distinct winning traces, not more gradient steps.

## SFT on the Traces: Teaching Stack-100M the Groove

With a folder of successful transcripts, training is just SFT (Ch. 14.9) with one twist: the loss mask must cover **prompt tokens, tool-call JSON we want to keep, AND tool-result spans we must not learn**. Concretely, the model is supervised on the assistant's thoughts, its tool calls, and its final answer — but *not* on the system prompt, the user question, or anything inside `<|tool_result|> … <|end|>`.

```python
# stacklm/agent/sft_format.py
"""Turn a distilled transcript into (input_ids, labels) with the correct
loss mask: supervise ASSISTANT emissions (thoughts, tool CALLS, final
answer) only; mask the system/user prompt and every tool RESULT span."""

from __future__ import annotations
import numpy as np
from stacklm.agent.react import ASST, END, TOOL_RESULT, USER, SYS

IGNORE = -100   # standard CrossEntropyLoss ignore_index (matches Ch. 14.9)


def build_example(transcript: str, tok, max_len: int = 1024):
    """`tok` is the Stack-100M byte-level BPE tokenizer (Ch. 14.3) exposing
    encode(str)->list[int] and the special-token ids."""
    ids, labels = [], []
    # Walk the transcript as alternating spans; a span is 'supervised' iff it
    # is assistant-generated content and NOT inside a tool_result.
    # We segment on the role markers to decide supervision per span.
    segments = _segment(transcript)   # list of (text, supervised: bool)
    for text, supervised in segments:
        piece = tok.encode(text)
        ids.extend(piece)
        labels.extend(piece if supervised else [IGNORE] * len(piece))

    ids, labels = ids[:max_len], labels[:max_len]
    # Teacher forcing: predict token t+1 from tokens <= t, so shift labels.
    input_ids = np.array(ids[:-1], dtype=np.int64)
    target    = np.array(labels[1:], dtype=np.int64)
    return input_ids, target


def _segment(t: str):
    """Split into supervised/unsupervised spans. Assistant content between
    <|assistant|> and the NEXT role/result marker is supervised; everything
    else (system, user, and <|tool_result|>...<|end|>) is masked. Note that
    <|tool_call|> is NOT a segmentation marker, so tool calls stay INSIDE the
    supervised assistant span -- we very much want the model to emit them."""
    out, i, supervised = [], 0, False
    markers = [ASST, USER, SYS, TOOL_RESULT, END]
    while i < len(t):
        nxt = min([(t.find(m, i), m) for m in markers if t.find(m, i) != -1],
                  default=(len(t), None), key=lambda x: x[0])
        j, m = nxt
        if j > i:
            out.append((t[i:j], supervised))
        if m == ASST:
            supervised = True                 # begin supervising assistant span
            out.append((m, False))            # the marker token itself: masked
            i = j + len(m)
        elif m in (USER, SYS, TOOL_RESULT):
            supervised = False                # stop supervising
            out.append((m, False))
            i = j + len(m)
        elif m == END:
            out.append((m, supervised))       # <|end|> IS supervised if in asst
            supervised = False                # ...then close the span
            i = j + len(m)
        else:
            out.append((t[i:], supervised)); break
    return out
```

Two subtleties that matter enormously in practice:

- **The `<|end|>` that closes an assistant turn is supervised.** The model must learn to *emit* the stop token, or at inference it will run past its answer into garbage. But the `<|end|>` that closes a `<|tool_result|>` is masked — the environment writes that one.
- **We do not distill the teacher's raw chain-of-thought verbatim if it is long or off-format.** For a 100M student, terse thoughts distill better than sprawling ones; a light post-processing pass that trims each `Thought:` to one sentence measurably improves imitation (fewer tokens to get wrong). This mirrors the general distillation finding that *shorter, cleaner rationales transfer better to small students* (see [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html)).

Training reuses the Ch. 14.9 SFT loop verbatim — same `stacklm.train.sft` entry point, same AdamW-on-embeddings + Muon-on-matrices optimizer split (Ch. 14.6), same bf16 autocast. The only new thing is the dataset builder above. That reuse is the point of a coherent capstone: the agent is not a new training system, it is a new *dataset* for the training system you already built.

## The Auto-Research Loop at Inference

At serving time there is no teacher. Stack-100M drives the loop itself: we generate until it emits `<|tool_call|>` or `<|end|>`, run the tool if asked, splice the observation back in, and continue — with strict guards, because a small model *will* try to derail.

```python
# stacklm/agent/loop.py
"""The runtime auto-research loop: Stack-100M interleaves thought, tool
calls, and observations, then synthesizes a final grounded answer.
Guards against the known small-model failure modes."""

from __future__ import annotations
from stacklm.agent.react import (parse_assistant_step, render_tool_result,
                                  SYS, USER, ASST, END)
from stacklm.agent.distill import ToolEnv, SYSTEM_PROMPT


def run_agent(model, tok, question: str, env: ToolEnv,
              max_steps: int = 6, max_new: int = 160):
    transcript = f"{SYS}{SYSTEM_PROMPT}{END}{USER}{question}{END}"
    trace = []                                   # for observability / eval
    seen_calls = set()                           # loop guard
    for step in range(max_steps):
        # Generate ONE assistant emission: stop at <|end|> (the model's own
        # stop token) so we hand control back to the harness after each step.
        gen = generate(model, tok, transcript + ASST,
                       max_new=max_new, stop_id=tok.id(END))
        step_text = gen.split(END, 1)[0]
        act = parse_assistant_step(step_text)
        transcript += f"{ASST}{step_text}{END}"
        trace.append(("assistant", step_text))

        if act.kind == "final":
            return act.answer, trace

        # --- GUARDS ---------------------------------------------------
        call_key = (act.tool, str(act.args))
        if call_key in seen_calls:               # exact repeat -> break the loop
            obs = "RepeatedCall: you already ran this; use the prior result."
        elif act.tool == "__malformed__":        # bad JSON -> teach recovery
            obs = "FormatError: emit a valid JSON tool call."
        else:
            obs = env.run_tool(act)
            seen_calls.add(call_key)
        # --------------------------------------------------------------
        transcript += render_tool_result(obs)
        trace.append(("observation", obs))

    # Ran out of steps without a final answer: force a synthesis attempt.
    gen = generate(model, tok, transcript + ASST + "Thought: I must answer now.\nAnswer:",
                   max_new=40, stop_id=tok.id(END))
    return gen.split(END, 1)[0].strip(), trace


def generate(model, tok, prompt: str, max_new: int, stop_id: int) -> str:
    """Greedy/low-temp decode until stop_id or max_new. (Full sampling
    machinery lives in stacklm.infer -- Ch. 14.11; see also
    Sampling Strategies & Decoding Algorithms.)"""
    import torch
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = []
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(ids)[:, -1, :]         # [1, vocab]
        nxt = int(logits.argmax(-1))              # greedy: agents want determinism
        if nxt == stop_id:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return tok.decode(out)
```

The three guards — **repeated-call detection**, **malformed-call recovery**, and **forced-synthesis on step exhaustion** — are not optional garnish. They are the difference between a demo that runs and a demo that hangs. A 100M model *will* re-issue the same search; the guard converts an infinite loop into an observation the model has been trained (via the recover-then-retry traces) to respond to. This is the same defensive-harness philosophy as [Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html), scaled down to what a tiny model needs.

!!! note "Aside: why greedy decoding at serve time"

    We decode the agent greedily (`argmax`), not with temperature. For a chatbot you often *want* sampling for diversity; for a tool-using agent you want **determinism and format discipline**. Every extra bit of entropy is another chance to emit a malformed `<|tool_call|>`, drift the JSON key order, or hallucinate an observation the environment never returned. The distilled groove is narrow, and greedy decoding keeps the model in it. If you must sample (e.g. to generate the GRPO groups below), keep the temperature low (~0.7) and lean hard on the runtime guards — see [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html).

!!! warning "Common pitfall: training on-format, serving off-format"

    The number-one silent failure is a **mismatch between the special-token strings/ids used in distillation and those used at inference**. If your SFT data wrote `<|tool_call|>` but your runtime tokenizer maps that to a different id (or splits it into sub-tokens because you forgot to register it as a *special* token in Ch. 14.3), the model has learned a groove it can never re-enter. Assert, in a unit test, that `tok.encode("<|tool_call|>")` is a single id and that it is identical in the distill, SFT, and serve paths. This one assert prevents a whole category of "it worked in training, produces gibberish in serving" bugs.

## Optional: RLVR on Tool-Use Success

Distillation gets Stack-100M *onto* the groove; a small dose of **RLVR** (Ch. 14.9) can sharpen it — nudging the policy toward trajectories that actually solve the task rather than merely *look* like solutions. The reward could not be simpler: run the agent loop, extract the final answer, compare to gold.

$$
R(\tau) = \mathbf{1}\!\left[\operatorname{normalize}(\text{answer}(\tau)) = \operatorname{normalize}(\text{gold})\right] \;-\; \lambda \cdot \frac{\text{steps}(\tau)}{\text{max\_steps}}
$$

The first term is the verifiable correctness reward; the small step penalty $\lambda$ (on the order of 0.05) discourages needless tool calls. This is *exactly* the RLVR reward of Ch. 14.9, applied to a whole multi-turn trajectory instead of a single answer — the agent is the "policy," a full ReAct rollout is the "completion," and the environment (our tools) is inside the rollout.

We optimize it with **GRPO** (Shao et al., *DeepSeekMath*, 2024; see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)), which needs no value network — ideal at this budget. For each question we sample a group of $G$ trajectories, compute the group-relative advantage, and take a masked policy-gradient step over *only the model-generated tokens* (never over observations — the same mask as SFT).

```python
# stacklm/agent/rlvr.py  (sketch; reuses stacklm.rl.grpo from Ch. 14.9)
"""Narrow RLVR over tool-use success with GRPO. One 'rollout' = one full
ReAct trajectory; reward = final-answer exact match minus a step penalty."""

import numpy as np
from stacklm.agent.loop import run_agent
from stacklm.agent.distill import normalize

def trajectory_reward(answer, gold, n_steps, max_steps, lam=0.05):
    correct = 1.0 if normalize(answer) == normalize(gold) else 0.0
    return correct - lam * (n_steps / max_steps)

def grpo_advantages(rewards):
    r = np.asarray(rewards, dtype=np.float64)
    # group-relative: subtract the group mean, scale by group std (GRPO)
    return (r - r.mean()) / (r.std() + 1e-6)

def rlvr_step(model, tok, env, task, G=8, max_steps=6):
    traj, rews = [], []
    for _ in range(G):                              # a GRPO group per task
        answer, trace = run_agent(model, tok, task.question, env, max_steps)
        n_steps = sum(1 for role, _ in trace if role == "assistant")
        rews.append(trajectory_reward(answer, task.gold, n_steps, max_steps))
        traj.append(trace)
    adv = grpo_advantages(rews)
    # ... feed (trajectory tokens, advantage, ASSISTANT-only mask) into the
    #     GRPO loss from stacklm.rl.grpo and take one optimizer step ...
    return float(np.mean(rews))                     # group success rate
```

Be honest about what this buys at 100M. RLVR here is a **polish pass, not a capability creator**. On a narrow verifiable task family where the SFT model already succeeds, say, on the order of 55–65% of the time, a short GRPO run can lift that by a modest margin and — often more valuably — *shorten* trajectories and cut malformed calls. It will **not** teach the model a tool-use skill that was absent from the distilled traces. The zero-reward problem is brutal at small scale: if the SFT policy almost never solves a task, every trajectory in the group gets reward 0, the advantages are all ~0, and there is no gradient signal. RLVR needs the SFT groove to *already reach the answer sometimes*. Distill first, RL second — never the reverse.

## The Ceiling: Brutally Honest About What 100M Can Do

It is time to state, without hedging, what you have and have not built.

**What genuinely works.** Inside the narrow task family the traces cover — "search the local corpus for a fact, maybe do one arithmetic step, answer" — Stack-100M is a real, functioning ReAct agent. It emits well-formed tool calls, reads observations, chains two or three steps, and returns grounded answers with a correct-answer rate that is *far* above what few-shot prompting the base model achieves (which is near zero for multi-step). The retrieval grounding is the load-bearing element: the model does not need to *know* facts, only to *find and copy* them, which is exactly the kind of task a small model can do reliably.

**Where the wall is.** Everything is scaffolding-shaped:

- **Distribution-brittleness.** Ask a question whose *shape* differs from the distilled traces — three retrieval hops instead of two, a comparison instead of a lookup, a unit conversion the calculator traces never showed — and the model produces confident, malformed, or hallucinated steps. It learned a groove, not a skill. This is the defining limitation and no amount of RLVR on the *same* narrow tasks fixes it; you must distill the new shapes.
- **No robust error recovery beyond what was demonstrated.** It recovers from the exact error patterns present in training (a `CalcError`, a `NoResults`) because those patterns were in successful traces. Novel failure modes derail it.
- **Query formulation is weak.** The single biggest source of end-to-end failure is a *bad search query* that retrieves nothing relevant; the downstream steps then have nothing to stand on. A larger model writes better queries. This bounds the whole system.
- **Context length.** Each step adds tokens; at `d_model=512`, 30 layers, and a mid-training context of 8192 (Ch. 14.8), you have headroom for ~6 short steps but not a 20-step research session. Long-horizon agency is out of reach.

!!! example "Worked example: where the accuracy goes"

    Decompose end-to-end accuracy on the narrow eval into a product of per-stage success rates (a useful mental model, illustrative magnitudes):

    $$
    \text{Acc}_{\text{e2e}} \approx p_{\text{well-formed call}} \cdot p_{\text{good query}} \cdot p_{\text{reads obs}} \cdot p_{\text{correct calc}} \cdot p_{\text{clean synth}}
    $$

    With on-the-order-of values after distillation — $0.95 \times 0.75 \times 0.90 \times 0.98 \times 0.90 \approx 0.57$ — you land near **57%** end-to-end, dominated by the **0.75 query term**. The arithmetic (0.98) and format (0.95) terms are nearly solved *because we offloaded them to a tool and regularized the format*. The bottleneck is the one genuinely cognitive step — writing a good query — which is precisely the thing a 100M model is worst at. This is why the honest headline is: **tool-use offloads the parts small models fail at; the residual failure is the reasoning we could not offload.** Push the corpus to be small and keyword-rich, and $p_{\text{good query}}$ rises — narrowing the domain is the highest-leverage knob you have.

{{fig:acc-decomposition-query-bottleneck}}

This is the frontier of what 100M can honestly do, and it is a genuinely satisfying place to end: not a chatbot oracle, but a *narrow, grounded, tool-using research assistant* that you trained end-to-end for the cost of a nice dinner. Ch. 14.11 evaluates it honestly (retrieval-QA exact-match, arithmetic accuracy) and Ch. 14.12 lays out exactly what to change — more data shapes, a bigger model for query formulation, longer context — to break through this ceiling on the road to 1B.

!!! interview "Interview Corner"

    **Q:** You want a 100M model to do multi-step tool use. Why not just few-shot prompt it with ReAct exemplars like you would a large model, and why does distillation-then-SFT work when that fails?

    **A:** Few-shot ReAct relies on strong in-context learning and latent planning ability — the model has to *induce* the act-observe-revise procedure from a couple of examples and *generalize* it to a new question, holding the whole loop in working memory across turns. A 100M model has neither the ICL strength nor the reasoning depth for that; it degrades into the canonical failure modes — hallucinating observations, ignoring returned results, or looping. Distillation changes the learning problem from "induce and generalize a procedure at inference time" to "reproduce a demonstrated behavior seen thousands of times in training." We generate trajectories with a strong teacher, **filter to the ones that verifiably solved the task** (rejection sampling against an exact-match reward), reformat with tool special tokens, and SFT with the loss masked to assistant tokens (never the observations). Note this is *trajectory* distillation — we imitate the teacher's text, not its logits — so the teacher can be any strong model and need not share our tokenizer. The model then imitates a narrow, well-worn groove rather than reasoning from scratch. The catch, which a good candidate volunteers: it learns the *distribution of demonstrated trajectory shapes*, not a general skill — so it is brittle to task shapes absent from the distilled data, and RLVR can only sharpen what SFT already reaches, not create new capability.

!!! key "Key Takeaways"

    - At 100M, **distillation is the only path to multi-step tool use**: generate ReAct trajectories with a large teacher, keep the verifiably-successful ones (rejection sampling), reformat with tool special tokens, and SFT the small model to imitate them.
    - Prefer **trajectory distillation over logit distillation** here: matching the teacher's *text* (thoughts, calls, answers) needs no shared tokenizer or lockstep querying, so any strong model can teach and the student trains with its ordinary cross-entropy loop.
    - Keep the **tool interface brutally regular** — one canonical JSON schema, tools wrapped in `<|tool_call|>`/`<|tool_result|>`, final answer delimited by `Answer:` — so the model memorizes one grammar and a verifier can extract the answer trivially.
    - **Mask the loss** to assistant emissions only: supervise thoughts, tool calls, and the final answer (including the closing `<|end|>`), never the system/user prompt or tool-result spans.
    - The scarce resource is **diverse successful trajectories**, not compute — the SFT set is tiny (tens of thousands of supervised tokens); spend your budget on the teacher, not on gradient steps.
    - Build **defensive runtime guards** — repeated-call detection, malformed-call recovery, forced synthesis on step exhaustion — decode greedily for determinism, and assert that special-token ids are identical across distill, SFT, and serve.
    - **RLVR (GRPO) with a final-answer exact-match reward** is a polish pass that sharpens and shortens trajectories; it cannot create capability the distilled traces lacked, and it dies from zero-signal groups if SFT does not already solve the task sometimes.
    - Tool use **offloads exactly what small models fail at** (exact arithmetic, format discipline); the residual bottleneck is query formulation — the one cognitive step you cannot offload — which bounds the whole system.
    - Be honest: the result is a **narrow, grounded, scaffolding-shaped research assistant**, useful inside its distilled groove and brittle one step outside it. That is the genuine frontier of 100M.

!!! sota "State of the Art & Resources (2026)"
    Trajectory distillation into small tool-using agents has gone from a research curiosity (2023) to a standard recipe with dedicated frameworks and papers explicitly targeting sub-3B students — the narrow-agent approach this chapter builds is now mainstream practice, not a toy simplification.

    **Foundational work**

    - [Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)](https://arxiv.org/abs/2210.03629) — the interleaved thought/action/observation loop this chapter implements.
    - [Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023)](https://arxiv.org/abs/2302.04761) — self-supervised tool-call insertion, a complementary route to tool use.
    - [Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning* (2022)](https://arxiv.org/abs/2203.14465) — rejection-sampling fine-tuning on self-generated successful traces, the pattern behind the filter step.

    **Recent advances (2023–2026)**

    - [Chen et al., *FireAct: Toward Language Agent Fine-tuning* (2023)](https://arxiv.org/abs/2310.05915) — fine-tuning on diverse GPT-4-generated ReAct trajectories substantially outperforms few-shot prompting and improves robustness to noisy tool observations, the same finding this chapter's worked examples reproduce at 100M scale.
    - [Kang et al., *Distilling LLM Agent into Small Models with Retrieval and Code Tools* (2025)](https://arxiv.org/abs/2505.17612) — transfers full task-solving trajectories (not just chain-of-thought) into 0.5B–3B students with retrieval and code tools, matching much larger CoT-distilled baselines.
    - [Belcak et al., *Small Language Models are the Future of Agentic AI* (NVIDIA, 2025)](https://arxiv.org/abs/2506.02153) — a position paper arguing narrow, repetitive agent nodes (tool-calling, structured reasoning) are exactly where small specialized models belong, with a strong LLM reserved for planning.

    **Open-source & tools**

    - [huggingface/smolagents](https://github.com/huggingface/smolagents) — a barebones (~1,000 line) agent library where the model writes actions as code rather than JSON; useful reference for the wire-format tradeoffs this chapter discusses.
    - [huggingface/trl](https://github.com/huggingface/trl) — includes a production `GRPOTrainer`, the same critic-free algorithm used in the optional RLVR polish pass.
    - [Nardien/agent-distillation](https://github.com/Nardien/agent-distillation) — official code for the Kang et al. small-agent-distillation paper above: trajectory logging, SFT training, and benchmarking for distilled agents.

    **Go deeper**

    - [Anthropic, *Building Effective Agents* (2024)](https://www.anthropic.com/engineering/building-effective-agents) — practitioner guidance that simple, composable agent patterns beat complex frameworks, echoing this chapter's "keep the interface brutally regular" design choice.

## Further reading

- Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao, *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022 — the loop this chapter implements.
- Schick, Dwivedi-Yu, Dessì et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, 2023 — self-supervised tool-call insertion; a complementary path to tool use.
- Zelikman, Wu, Mu & Goodman, *STaR: Bootstrapping Reasoning With Reasoning*, 2022 — rejection-sampling fine-tuning on self-generated successful traces, the pattern behind our filter step.
- Shao, Wang, Zhu et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (GRPO), 2024 — the critic-free RL algorithm for the optional RLVR polish.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009 — the lexical retriever.
- Weinberger, Dasgupta, Langford, Smola & Attenberg, *Feature Hashing for Large Scale Multitask Learning*, 2009 — the hashing trick behind the embedding-lite retriever.
- Lewis, Perez, Piktus et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020 — the RAG foundation the retriever tool sits on.
- Allal, Lozhkov, Bakouch et al., *SmolLM* (HuggingFace), 2024–2025 — the small-model recipe (data + distillation) whose spirit this capstone follows.
- Karpathy, *nanoGPT* / *llm.c*, 2024 — the "reproduce a real model on a budget" spirit this whole capstone updates.

## Exercises

**1.** In the loss-mask design, the `<|end|>` token that closes an *assistant* turn is supervised, but the `<|end|>` that closes a `<|tool_result|>` span is masked. Explain the reasoning behind this asymmetry, and describe the concrete failure you would observe at inference time if you accidentally masked *both* kinds of `<|end|>`.

??? note "Solution"
    The two `<|end|>` tokens are written by different actors and therefore have different learning requirements.

    - The `<|end|>` closing an assistant turn is *emitted by the model*. At serve time the runtime loop generates one assistant step and stops at `<|end|>` (`stop_id=tok.id(END)` in `run_agent`), then hands control back to the harness to run a tool or return the answer. If the model never learned to *produce* this token, it will not know when to stop; it will run past its answer or past its tool call into garbage, and the harness will never regain control at the right boundary. So this token must be *in the loss*.
    - The `<|end|>` closing a `<|tool_result|>` is *written by the environment*, not the model. The model must learn to *read* observations (to condition on them) but must never be trained to *produce* them, or it will start hallucinating observations the environment never returned. The entire `<|tool_result|> ... <|end|>` span, including its closing `<|end|>`, is therefore masked with `IGNORE = -100`.

    This is exactly what `_segment` encodes: `END` is appended with `supervised` equal to whatever the current span's supervision flag is (`out.append((m, supervised))`), so an `<|end|>` inside an assistant span is supervised and one inside a masked tool-result span is not.

    If you masked *both* kinds of `<|end|>`, the model would never be supervised to emit its own stop token. At inference, greedy decoding would not reliably produce `<|end|>` after the answer or tool call, so `generate` would keep decoding until it hit `max_new`, spilling the answer into trailing garbage and breaking the "stop at `<|end|>` after each step" contract the loop depends on. The symptom is a model that "worked in training" but at serve time never cleanly terminates a step.

**2.** The chapter insists that few-shot prompting a 100M base model with ReAct exemplars fails, while distillation-then-SFT works. Name the three canonical small-model failure modes the chapter lists for few-shot ReAct, and explain in one or two sentences *why distillation changes the learning problem* so those modes stop dominating.

??? note "Solution"
    The three canonical failure modes (from the "capability gap" section and the Interview Corner):

    1. **Hallucinating the observation** — it emits a plausible `Thought:` and then an answer with *no tool call at all*, inventing the tool's output instead of calling it.
    2. **Ignoring the returned observation** — it emits a tool call but then does not condition on what came back.
    3. **Looping forever** — it re-issues the same search over and over without progressing.

    Why distillation fixes the framing: few-shot ReAct asks the model to *induce* the act-observe-revise procedure from a couple of in-context examples and *generalize* it to a new question, holding the whole multi-turn loop in working memory. A 100M model has neither the in-context-learning strength nor the reasoning depth for that. Distillation changes the task from "induce and generalize a procedure at inference time" to "reproduce a demonstrated behavior seen thousands of times during training." The model imitates a narrow, well-worn groove seen as ordinary SFT targets rather than reasoning from scratch, so the failure modes above (which are failures of *induction/generalization*) no longer dominate inside the distilled distribution. The catch: it learns the distribution of demonstrated trajectory shapes, not a general skill, so it stays brittle one step outside that groove.

**3.** (Quantitative.) You run the distillation pipeline with a task pool of **150 questions**, **`samples_per_task = 6`** teacher rollouts each, and a per-rollout success probability of **0.5**. Assume that after dedup, successful traces collapse to about **2 distinct shapes per task**. Using the chapter's per-trace figures (~255 tokens/trace, of which ~95 are loss-bearing assistant tokens), estimate: (a) the expected number of raw solved trajectories, (b) the number of unique traces kept, (c) the total and supervised token counts of the SFT set, and (d) at a teacher cost of \$1-3 per thousand rollouts, the dollar cost of teacher rollouts. State the one-line lesson the numbers reinforce.

??? note "Solution"
    (a) Expected raw successes: $150 \times 6 \times 0.5 = 450$ solved trajectories.

    (b) Unique traces after dedup: $150 \times 2 = 300$ traces.

    (c) Token counts, using the chapter's per-trace figures:
    - Total: $300 \times 255 = 76{,}500$ tokens.
    - Supervised (loss-bearing): $300 \times 95 = 28{,}500$ tokens.

    (d) Rollout count: $150 \times 6 = 900$ rollouts. At \$1-3 per thousand rollouts, that is $0.9 \times (\$1\text{ to }\$3) \approx \$0.90\text{ to }\$2.70$ — a couple of dollars.

    Lesson: the SFT set is tiny (tens of thousands of supervised tokens = a few minutes of fine-tuning), so at 100M the scarce resource is not compute but **diverse successful trajectories**. Spend the budget on the teacher (more distinct winning traces), not on gradient steps.

**4.** (Quantitative.) Using the chapter's per-stage decomposition
$$
\text{Acc}_{\text{e2e}} \approx p_{\text{well-formed call}} \cdot p_{\text{good query}} \cdot p_{\text{reads obs}} \cdot p_{\text{correct calc}} \cdot p_{\text{clean synth}},
$$
suppose after distillation you measure $p_{\text{well-formed call}} = 0.95$, $p_{\text{good query}} = 0.60$, $p_{\text{reads obs}} = 0.90$, $p_{\text{correct calc}} = 0.98$, $p_{\text{clean synth}} = 0.90$. (a) Compute the end-to-end accuracy. (b) Identify the bottleneck stage. (c) You can either narrow the corpus to raise $p_{\text{good query}}$ to $0.85$, or spend the same effort raising $p_{\text{well-formed call}}$ to a perfect $1.00$. Compute the resulting accuracy for each option and say which is the higher-leverage knob and why this matches the chapter's thesis.

??? note "Solution"
    (a) End-to-end accuracy:
    $$
    0.95 \times 0.60 \times 0.90 \times 0.98 \times 0.90.
    $$
    Step by step: $0.95 \times 0.60 = 0.570$; $0.570 \times 0.90 = 0.513$; $0.513 \times 0.98 = 0.50274$; $0.50274 \times 0.90 \approx 0.4525$. So $\text{Acc}_{\text{e2e}} \approx \mathbf{45\%}$.

    (b) The bottleneck is $p_{\text{good query}} = 0.60$ — by far the smallest factor, the one dragging the product down.

    (c) Option A (narrow corpus, query $0.60 \to 0.85$):
    $$
    0.95 \times 0.85 \times 0.90 \times 0.98 \times 0.90.
    $$
    $0.95 \times 0.85 = 0.8075$; $\times 0.90 = 0.72675$; $\times 0.98 = 0.712215$; $\times 0.90 \approx 0.6410$, i.e. **~64%**.

    Option B (perfect format, $0.95 \to 1.00$): this multiplies the original 45% by $1.00/0.95 \approx 1.053$, giving $0.4525 \times 1.0526 \approx 0.4763$, i.e. **~48%**.

    Narrowing the corpus to fix the query stage (+19 points) dwarfs perfecting the format stage (+3 points). This matches the chapter's thesis: tool use and format regularization already push the arithmetic and format terms near 1.0 — they are *offloaded* and nearly solved — so the residual failure lives in the one genuinely cognitive step we cannot offload, query formulation. Narrowing the domain (small, keyword-rich corpus) is the highest-leverage knob.

**5.** (Implementation.) The practitioner tip argues that the production retrieval move is a *hybrid* — union the BM25 and hashed-embedding candidate sets and rerank. Implement a `HybridRetriever` in the style of `stacklm/agent/tools.py` that wraps a `BM25Retriever` and a `HashEmbedRetriever`, exposes the same `search(query, k) -> list[tuple[Passage, float]]` interface, and fuses the two candidate lists. Use **Reciprocal Rank Fusion (RRF)** so you do not have to reconcile BM25's and cosine's incompatible score scales. Then state one reason we still default to a *single* backend per run for the 100M agent.

??? note "Solution"
    RRF assigns each document a score $\sum_{\text{backends}} \frac{1}{c + \text{rank}}$ (rank 0-indexed), which depends only on *rank position*, not on the raw score magnitude — so BM25's unbounded scores and cosine's $[-1, 1]$ scores combine cleanly. A document that appears high in either list, or moderately in both, floats to the top.

    ```python
    # stacklm/agent/tools.py  (append)
    class HybridRetriever:
        """Union BM25 + hashed-embedding candidates and rerank with Reciprocal
        Rank Fusion (RRF). RRF is scale-free -- it uses only rank position, so
        there is no need to reconcile BM25's and cosine's incompatible score
        magnitudes. See Ch. 9.4 (Chunking, Reranking & Hybrid Search)."""

        def __init__(self, passages: list[Passage], dim: int = 512,
                     k1: float = 1.5, b: float = 0.75, c: int = 60):
            self.passages = passages
            self.bm25 = BM25Retriever(passages, k1, b)
            self.embed = HashEmbedRetriever(passages, dim)
            self.c = c                      # RRF damping constant (Cormack 2009)

        def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
            # Pull a slightly wider candidate pool from each backend so fusion
            # can promote a doc that is mid-ranked in both lists.
            pool = max(k, 2 * k)
            fused: dict[str, list] = {}     # doc_id -> [Passage, rrf_score]
            for retr in (self.bm25, self.embed):
                for rank, (p, _) in enumerate(retr.search(query, pool)):
                    slot = fused.setdefault(p.doc_id, [p, 0.0])
                    slot[1] += 1.0 / (self.c + rank)
            ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
            return [(p, sc) for p, sc in ranked[:k]]
    ```

    A quick sanity check: if a passage is ranked 0 by BM25 and rank 1 by the embedder with $c = 60$, its fused score is $\frac{1}{60} + \frac{1}{61} \approx 0.0330$, beating a passage that is only rank 0 in BM25 and absent from the embedder list ($\frac{1}{60} \approx 0.0167$). Documents both backends like win.

    Why still default to a single backend for the 100M agent: fusing two backends produces observations with more *format variety* (different passage sets, orders, and phrasings), and the whole distillation strategy relies on keeping the observation format brutally regular so the tiny model memorizes one grammar rather than many. We expose both backends so you can *ablate* which one distills better, but keep a single one per run at 100M to minimize the surface the model must memorize.

**6.** (Conceptual + quantitative.) The chapter warns that RLVR "dies from zero-signal groups if SFT does not already solve the task sometimes." Consider a GRPO group of $G = 8$ trajectories for a single task where the SFT policy currently solves this task essentially never, so all 8 rollouts return the wrong answer, each using the full `max_steps`. Using the reward $R(\tau) = \mathbf{1}[\text{correct}] - \lambda \cdot \frac{\text{steps}}{\text{max\_steps}}$ with $\lambda = 0.05$ and the `grpo_advantages` function, compute the reward of each trajectory and the resulting advantages, and explain why no learning happens. Then state the chapter's ordering rule and why it is not reversible.

??? note "Solution"
    Each trajectory is wrong, so $\mathbf{1}[\text{correct}] = 0$, and each uses the full `max_steps`, so $\frac{\text{steps}}{\text{max\_steps}} = 1$. Thus every trajectory has the same reward:
    $$
    R(\tau) = 0 - 0.05 \times 1 = -0.05.
    $$

    All 8 rewards are identical: $r = [-0.05, -0.05, \dots, -0.05]$. In `grpo_advantages`, the group mean is $-0.05$ and the standard deviation is $0$, so each advantage is
    $$
    \frac{-0.05 - (-0.05)}{0 + 10^{-6}} = \frac{0}{10^{-6}} = 0.
    $$

    Every advantage is $0$. GRPO's policy-gradient update scales the log-prob gradient of each trajectory by its advantage, so a group of all-zero advantages contributes **no gradient signal** — there is nothing to push the policy toward, because no trajectory in the group did any better or worse than its peers. This is the zero-reward (here zero-*variance*) problem: group-relative advantages require *within-group spread*, which requires the policy to *sometimes* succeed and sometimes fail on the same task.

    The chapter's ordering rule: **distill first, RL second — never the reverse.** It is not reversible because RLVR can only *sharpen* behavior the SFT groove already reaches sometimes; it cannot *create* a capability that is absent. If SFT never solves the task, every GRPO group is degenerate (uniform reward, zero advantage, no gradient), so RL has no foothold. You must first get the model onto the groove with distillation so that some fraction of rollouts land on the answer, producing the reward variance GRPO needs.
