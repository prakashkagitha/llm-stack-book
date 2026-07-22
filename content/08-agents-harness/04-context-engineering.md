# 8.4 Context Engineering & Management

A frontier model is, in one sense, a pure function: tokens in, a probability distribution over the next token out. Everything the model "knows" about your task in this moment — the user's goal, the files it has read, the tool outputs it has seen, the plan it is following — must be encoded as tokens inside a single finite buffer called the **context window**. The model has no other working memory. When the window fills, something must be dropped, and what you drop determines what the agent forgets.

This is why context is the scarce resource of agentic systems, and why **context engineering** — the discipline of deciding what goes into the window, in what form, in what order, and for how long — has become as load-bearing as the prompt itself. An agent that runs for fifty turns will assemble its context fifty times. If each assembly is sloppy, errors compound: stale file contents mislead edits, redundant tool output crowds out the plan, and the model's accuracy decays long before it hits the hard token limit.

In [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html) we built the control flow of an agent; in [Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html) we built the scaffolding around it. This chapter is about the *substance* that flows through that scaffolding: the context. We will treat the window as a memory hierarchy to be managed, quantify how performance degrades as it fills, and build the concrete machinery — token budgeting, retrieval, compaction, structured external memory, and prompt caching — that keeps a long-running agent coherent and cheap.

## The Context Window as a Scarce Resource

### What actually lives in the window

At any decode step, the model sees a flat sequence of tokens. In an agent, that sequence is *assembled* from logically distinct sources. A useful mental model is a stack of segments:

```text
+-----------------------------------------------------------+
|  SYSTEM PROMPT      role, rules, output format  (static)  |  <- cache-friendly
+-----------------------------------------------------------+
|  TOOL DEFINITIONS   JSON schemas for callable tools       |  <- cache-friendly
+-----------------------------------------------------------+
|  RETRIEVED CONTEXT  docs, code, RAG chunks    (per-task)  |
+-----------------------------------------------------------+
|  CONVERSATION       user msgs, assistant msgs, tool calls |  <- grows every turn
|     ...             tool results (often huge)             |
+-----------------------------------------------------------+
|  SCRATCHPAD / PLAN  the agent's own working notes         |
+-----------------------------------------------------------+
|  CURRENT QUERY      the immediate instruction             |  <- highest salience
+-----------------------------------------------------------+
```

Every one of these competes for the same budget. A modern model might advertise a 200K-token window, but that headline number is a *capacity*, not a *recommendation*. Three separate costs push you to use far less than the maximum:

1. **Quality.** Accuracy degrades as the window fills (Sections on context rot and lost-in-the-middle below). The useful window is smaller than the advertised window.
2. **Latency.** Prefill is roughly linear in input length, and attention is quadratic in sequence length; a 150K-token prompt can take seconds to prefill before the first token streams. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).
3. **Money.** You pay per input token, every turn. A 50-turn agent that carries a 100K context pays for that 100K *fifty times* unless caching intervenes.

### The cost arithmetic of a naive agent

Consider an agent that appends everything and never prunes. Let turn $t$ add $a_t$ new tokens (the assistant message plus any tool result). The prompt length at turn $t$ is the running sum of everything before it:

$$
L_t = L_0 + \sum_{i=1}^{t} a_i
$$

The total *input* tokens billed across an $N$-turn run is the sum of every prefix:

$$
T_{\text{in}} = \sum_{t=1}^{N} L_t = \sum_{t=1}^{N}\left(L_0 + \sum_{i=1}^{t} a_i\right)
$$

If every turn adds a constant $a$ tokens and $L_0$ is small, this is $\sum_{t=1}^N a t \approx \tfrac{1}{2} a N^2$. **Naive context accumulation is quadratic in the number of turns.** This single fact is the reason every serious agent harness has a context-management strategy. Because the dominant term $\tfrac{1}{2}aN^2$ is *linear* in $a$, halving $a$ (the per-turn footprint) halves that term — while capping $L_t$ at a ceiling $C$ turns the quadratic back into a linear $O(NC)$, attacking the far larger $N^2$ factor.

{{fig:quadratic-cost-of-accumulation}}

!!! example "Worked example: the cost of not pruning"

    An agent runs for $N = 40$ turns. Each turn the model reads a file or runs a command whose output averages $a = 2{,}500$ tokens; the system prompt and tools are $L_0 = 3{,}000$ tokens. Input is priced at USD 3 per million tokens, output (assume 400 tokens/turn) at USD 15 per million.

    Naive accumulation. Prompt length at turn $t$ is $L_t \approx 3000 + 2500\,t$. Total input billed:

    $$
    T_{\text{in}} = \sum_{t=1}^{40}(3000 + 2500\,t) = 40\cdot3000 + 2500\cdot\frac{40\cdot41}{2} = 120{,}000 + 2{,}050{,}000 = 2.17\text{M tokens}
    $$

    Input cost: $2.17 \times 3 = $ **USD 6.51**. Output adds $40 \times 400 = 16{,}000$ tokens $\to$ USD 0.24. Total ≈ **USD 6.75** for one task.

    Now cap the context at $C = 20{,}000$ tokens via compaction (Section below). Once $L_t$ hits the cap it stays there, so $T_{\text{in}} \approx 12\cdot(3000+2500t)|_{\text{ramp}} + 28\cdot20000 \approx 100{,}000 + 560{,}000 = 0.66\text{M}$. Input cost drops to ≈ **USD 1.98** — a 3× saving, *before* prompt caching, which can take another 5–10× off the cacheable prefix.

    The lesson: the agent that thinks about its context is not just smarter, it is dramatically cheaper.

## Context Rot, Lost-in-the-Middle & Position Effects

A bigger window does not mean a model uses all of it equally. Two empirical phenomena dominate practical context engineering.

{{fig:effective-vs-advertised-window}}

### Lost-in-the-middle

When relevant information is placed in the *middle* of a long context, retrieval accuracy is systematically worse than when the same information sits at the **beginning** or **end**. The accuracy-versus-position curve is U-shaped: strong at the edges, sagging in the middle. This was characterized by Liu et al. in *Lost in the Middle: How Language Models Use Long Contexts* (2023) and has held up across model generations, even as absolute scores improved.

The causes are several and reinforcing. Causal attention plus the softmax give early tokens an outsized influence (they are attended to by every later position, and "attention sinks" frequently form on the first tokens). Recency from positional encodings and the autoregressive objective privileges the most recent tokens. RoPE-based long-context extension (see [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html)) can attenuate long-range signal. The middle gets neither edge advantage.

The engineering consequence is direct and actionable: **place the most important material at the start or the very end of the window, never buried in the middle.** When you stuff $k$ retrieved documents, put the highest-scoring ones at the boundaries.

### Context rot

"Context rot" is the broader, more recent observation that model performance on a fixed task **degrades as irrelevant or merely long context is added**, even when the needle is present and the model can in principle find it. It is not only about position; it is about *distractors* and *length itself*. Adding 80K tokens of plausible-but-irrelevant code to a debugging prompt makes the model worse at the bug, because every distractor token is something attention can be pulled toward, and every extra token dilutes the signal-to-noise ratio.

A clean way to think about it: a long context increases the chance that some span looks locally more relevant to the next token than the span you actually care about. The model is doing soft retrieval over its own context at every step; more haystack means more chances to grab hay.

```python
# A "needle in a haystack" harness to MEASURE rot/position effects for your stack.
# Run this against your own model+prompt; do not trust generic claims.
import random

def build_haystack(needle: str, filler_tokens: list[str],
                   total_tokens: int, needle_frac: float) -> str:
    """Embed `needle` at a fractional depth in a haystack of `total_tokens`.

    needle_frac=0.0 -> needle at start, 0.5 -> middle, 1.0 -> end.
    `filler_tokens` is a large pool of irrelevant words to pad with.
    """
    n_pad = max(0, total_tokens - len(needle.split()))
    insert_at = int(n_pad * needle_frac)
    body = [random.choice(filler_tokens) for _ in range(n_pad)]
    words = body[:insert_at] + needle.split() + body[insert_at:]
    return " ".join(words)

def eval_grid(call_model, needle, question, expected,
              lengths=(2000, 8000, 32000, 128000),
              depths=(0.0, 0.25, 0.5, 0.75, 1.0),
              filler_pool=None, trials=3):
    """Sweep (length x depth). Returns accuracy per cell.
    `call_model(prompt) -> str`. We score by substring match on `expected`."""
    filler_pool = filler_pool or ["the","system","logs","report","value",
                                  "module","request","status","cache","node"]
    results = {}
    for L in lengths:
        for d in depths:
            hits = 0
            for _ in range(trials):
                hay = build_haystack(needle, filler_pool, L, d)
                prompt = (f"Read the following text and answer the question.\n\n"
                          f"{hay}\n\nQuestion: {question}\nAnswer:")
                out = call_model(prompt)
                hits += int(expected.lower() in out.lower())
            results[(L, d)] = hits / trials
    return results

# Reading the result grid you will typically see:
#   - accuracy falling as L grows (context rot),
#   - within a row, a dip around d=0.5 (lost-in-the-middle).
# Use it to choose YOUR effective window, not the advertised one.
```

!!! tip "Practitioner tip"

    Treat the advertised window as a hardware spec and your *effective* window as a measured quantity. Run the grid above on your actual model + system prompt and pick the length at which accuracy is still acceptable (say ≥ 95% of the short-context score). Budget to *that* number. For many production setups in 2024–2026 the effective window is a fraction — often a quarter to a half — of the advertised maximum for retrieval-style tasks.

## Retrieval vs Stuffing: Choosing What Goes In

The first decision in context engineering is binary per piece of information: **include it now, or fetch it on demand?** Stuffing means putting everything potentially relevant into the prompt up front. Retrieval means keeping a large corpus *outside* the window and pulling in only what the current step needs.

### The trade-off

Stuffing is simple, has zero retrieval latency, and never "misses" — if the answer is in the corpus and the corpus fits, the answer is in the window. But it pays the full quality/latency/cost penalty of a long context on *every* turn, and it falls off a cliff when the corpus exceeds the window.

Retrieval keeps the window small and the per-turn cost low, and it scales to corpora of millions of tokens. But it adds a retrieval component that can miss (the relevant chunk is not in the top-$k$), it adds latency for the lookup, and it requires infrastructure: embeddings, an index, chunking, reranking. The full machinery is the subject of Part IX — see [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html) and [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html). Here we care about the *decision*.

A simple rule of thumb in terms of corpus size $S$ (tokens) and effective window $W_{\text{eff}}$:

- $S \ll W_{\text{eff}}$ and information is reused every turn → **stuff** (e.g., the file currently being edited, the active plan).
- $S \gg W_{\text{eff}}$, or any single query touches only a small slice → **retrieve** (e.g., a 2M-line codebase, a documentation set, prior conversations).
- In between, or when recall matters more than cost → **hybrid**: stuff a small curated core, retrieve the long tail.

### Agentic retrieval: let the model fetch

The most powerful pattern for agents is to expose retrieval *as a tool* rather than pre-stuffing. The model decides what it needs and when. This is "just-in-time" context: instead of front-loading 50 files, give the agent `grep`, `read_file`, and `search_docs` tools (see [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html)) and let it pull exactly the lines it needs. This keeps the window lean and lets the agent's own reasoning act as the relevance function — often better than a static embedding similarity.

```python
# Just-in-time context via tools. The agent pulls minimal slices on demand
# instead of us pre-stuffing whole files. Note that we return *windows*
# around hits, not whole files, to protect the budget.
import subprocess, re

def grep_tool(pattern: str, path: str = ".", max_hits: int = 20) -> str:
    """Return file:line: matches. Cheap to put in context; points the agent
    at *where* to look without dumping whole files."""
    try:
        out = subprocess.run(["grep", "-rni", pattern, path],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return f"grep error: {e}"
    lines = out.splitlines()[:max_hits]
    return "\n".join(lines) if lines else "(no matches)"

def read_window(path: str, center_line: int, radius: int = 40) -> str:
    """Read only lines [center-radius, center+radius]. Returning a 80-line
    window instead of a 4000-line file is the difference between a 600-token
    and a 30000-token tool result."""
    with open(path) as f:
        lines = f.readlines()
    lo = max(0, center_line - radius)
    hi = min(len(lines), center_line + radius)
    body = "".join(f"{i+1:>5} | {lines[i]}" for i in range(lo, hi))
    return f"# {path} lines {lo+1}-{hi}\n{body}"

# The agent's loop: grep to localize, read_window to inspect, then act.
# At no point does a whole 4000-line file enter the context.
```

!!! warning "Common pitfall: dumping whole tool outputs"

    The single most common way agents blow their context budget is by piping raw, unbounded tool output straight into the window: an entire file, a 5,000-line `pytest` log, a directory listing of 10,000 entries, a full HTTP response body. *Always* bound tool results — head/tail, truncate with an explicit `[... 4,213 lines elided ...]` marker, or summarize — before they re-enter the context. A tool wrapper that caps every result at, say, 2,000 tokens and tells the model how to fetch more is worth more than a cleverer prompt.

## Token Budgeting: Allocating the Window

Once you accept that the effective window is finite and precious, you manage it like a memory allocator. Give each segment a **budget**, enforce it, and have an eviction policy for overflow.

### A budget model

Partition the effective window $W_{\text{eff}}$ into segments with target fractions that sum to (less than) one, leaving headroom for the model's own output:

$$
W_{\text{eff}} = b_{\text{sys}} + b_{\text{tools}} + b_{\text{retrieved}} + b_{\text{history}} + b_{\text{scratch}} + b_{\text{output}}
$$

The system prompt and tool schemas are fixed and small; retrieval, history, and scratchpad are the elastic segments you actively manage. A concrete starting split for a 32K *effective* budget on a coding agent might be: 2K system+tools, 8K retrieved code, 16K conversation history, 2K scratchpad, leaving 4K for the response.

{{fig:token-budget-allocator}}

### A concrete budget manager

You need an accurate token count, not a character heuristic, because over-counting wastes budget and under-counting causes hard API failures. Use the model's real tokenizer (see [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html)).

```python
# A budget manager that assembles a prompt under a hard token ceiling.
# It counts with the REAL tokenizer and enforces per-segment caps,
# evicting from the middle of history (oldest-but-not-first) on overflow.
from dataclasses import dataclass, field

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")  # stand-in tokenizer
    def ntok(s: str) -> int: return len(_enc.encode(s))
except Exception:
    # Fallback heuristic ~4 chars/token; ONLY for environments without tiktoken.
    def ntok(s: str) -> int: return max(1, len(s) // 4)

@dataclass
class Segment:
    name: str
    budget: int                 # max tokens this segment may occupy
    items: list[str] = field(default_factory=list)
    pinned: list[bool] = field(default_factory=list)  # never-evict flags

    def add(self, text: str, pin: bool = False):
        self.items.append(text); self.pinned.append(pin)

    def tokens(self) -> int:
        return sum(ntok(x) for x in self.items)

    def fit(self):
        """Evict UNPINNED items, oldest first, until under budget.
        Keeps pinned items (e.g., the system prompt, the current task)."""
        while self.tokens() > self.budget:
            # find oldest unpinned item to drop
            idx = next((i for i, p in enumerate(self.pinned) if not p), None)
            if idx is None:
                break  # everything pinned; nothing legal to drop
            dropped = self.items.pop(idx); self.pinned.pop(idx)
            # leave a breadcrumb so the model knows something was removed
            self.items.insert(idx, f"[elided {ntok(dropped)} tokens]")
            self.pinned.insert(idx, True)

class ContextBuilder:
    def __init__(self, w_eff: int, output_reserve: int = 4000):
        self.w_eff = w_eff
        self.output_reserve = output_reserve
        self.segments: list[Segment] = []

    def segment(self, name: str, budget: int) -> Segment:
        s = Segment(name, budget); self.segments.append(s); return s

    def build(self) -> str:
        # 1) enforce per-segment budgets
        for s in self.segments:
            s.fit()
        # 2) enforce the GLOBAL ceiling: if total still too big,
        #    squeeze the largest elastic segment further.
        ceiling = self.w_eff - self.output_reserve
        while sum(s.tokens() for s in self.segments) > ceiling:
            big = max(self.segments, key=lambda s: s.tokens())
            big.budget = int(big.budget * 0.85)   # tighten and re-fit
            big.fit()
            if all(s.tokens() <= 1 for s in self.segments):
                break
        return "\n\n".join("\n".join(s.items) for s in self.segments)

# --- usage ---------------------------------------------------------------
cb = ContextBuilder(w_eff=32_000, output_reserve=4000)
sys_seg   = cb.segment("system",    budget=2_000)
ret_seg   = cb.segment("retrieved", budget=8_000)
hist_seg  = cb.segment("history",   budget=16_000)
task_seg  = cb.segment("task",      budget=2_000)

sys_seg.add("You are a coding agent. Follow the plan. Be terse.", pin=True)
ret_seg.add(read_window("server.py", 120))           # only the relevant slice
for turn in []:  # ... append conversation turns here ...
    hist_seg.add(turn)
task_seg.add("Fix the off-by-one in pagination.", pin=True)

prompt = cb.build()
print(f"assembled {ntok(prompt)} tokens (ceiling {32_000-4000})")
```

The key design choices: per-segment caps prevent any one source (usually tool output or history) from cannibalizing the rest; *pinning* protects the load-bearing pieces (system prompt, current task, the active file); eviction leaves an explicit breadcrumb so the model knows context was removed rather than silently hallucinating continuity; and a global ceiling backstop guarantees you never exceed the API's hard limit.

## Compaction & Summarization

When eviction is too lossy — you do not want to simply *drop* the last 30 turns of a debugging session — you **compact**: replace a long span of low-density context with a short, high-density summary that preserves the decisions and facts the agent still needs.

### When to compact

Trigger compaction when the conversation segment crosses a high-water mark, e.g. 75% of its budget. Do not wait for the hard limit; compaction itself costs a model call and you want headroom to absorb the next few turns.

### What to preserve

A good agent summary is not a prose recap; it is a **structured state snapshot**. The art is choosing what is *durable* (must survive) versus *ephemeral* (safe to drop):

- **Durable:** the original goal; decisions made and *why*; files/functions modified and their current state; constraints discovered ("the API rate-limits at 10 rps"); open TODOs; the current hypothesis.
- **Ephemeral:** raw tool output already acted upon; superseded plans; verbose reasoning that led to a conclusion already recorded; failed attempts whose lesson is captured in a single line.

```python
# Compaction: collapse old turns into a structured snapshot via a model call.
# The summary is itself a context artifact, so we constrain its length hard.
COMPACTION_PROMPT = """\
You are compacting an agent's working memory. Produce a STRUCTURED state \
snapshot under {max_tokens} tokens. Preserve only what is needed to CONTINUE \
the task. Use exactly these sections; omit a section if empty.

## Goal
<the user's original objective, one or two sentences>
## Decisions
<bulleted: decision -> rationale>
## Files changed
<path: what changed and current state>
## Constraints discovered
<bulleted facts that constrain future actions>
## Open TODOs
<ordered, actionable>
## Current hypothesis / next step
<one short paragraph>

Do NOT include raw logs, superseded plans, or verbose reasoning.
--- CONVERSATION TO COMPACT ---
{history}
"""

def compact(call_model, turns: list[str], max_tokens: int = 1200) -> str:
    history = "\n\n".join(turns)
    prompt = COMPACTION_PROMPT.format(max_tokens=max_tokens, history=history)
    summary = call_model(prompt)
    # Hard-trim as a safety net in case the model over-runs.
    try:
        toks = _enc.encode(summary)[:max_tokens]
    except NameError:
        toks = None  # tiktoken unavailable; `_enc` was never defined
    return _enc.decode(toks) if toks is not None else summary

def maybe_compact(hist_seg: "Segment", call_model, keep_recent: int = 6):
    """If history exceeds 75% of budget, compact everything except the most
    recent `keep_recent` turns (recency is high-value; never summarize it)."""
    if hist_seg.tokens() < 0.75 * hist_seg.budget:
        return
    old, recent = hist_seg.items[:-keep_recent], hist_seg.items[-keep_recent:]
    snapshot = compact(call_model, old)
    # Replace the old span with one pinned snapshot; keep recent verbatim.
    hist_seg.items = [f"## COMPACTED STATE\n{snapshot}"] + recent
    hist_seg.pinned = [True] + [False] * len(recent)
```

Two subtleties make or break compaction in practice. First, **never compact the most recent turns** — recency is exactly where the model's attention is strongest and where the immediate task state lives; summarizing it throws away your best signal. Keep the last few turns verbatim and compact only the older tail. Second, **compaction is lossy and compounding**: summaries of summaries drift. Mitigate by anchoring durable facts in *external* storage (next section) that is regenerated from ground truth rather than re-summarized, so the snapshot can be partly rebuilt from files on disk rather than from prior summaries.

!!! note "Aside: compaction vs. truncation vs. eviction"

    Three different operations, often conflated. **Truncation** cuts tokens by position (drop the oldest N) — cheap, zero model calls, maximally lossy. **Eviction** drops whole semantic items (a tool result, a turn) under a policy — cheap, semantically aware, still lossy. **Compaction** replaces a span with a model-generated summary — costs a model call, preserves the most information per token, but introduces drift. Mature harnesses use all three: evict obviously-dead items continuously, truncate over-long single results immediately, and compact the conversation periodically.

## Structured External Context: Files, Scratchpads & State

The deepest idea in modern context engineering is that **the context window should not be the agent's only memory.** Treat the window as fast, small, volatile RAM, and treat the file system (or a database, or a structured store) as slow, large, durable disk. The agent reads relevant state into the window when it needs it and writes durable state back out — exactly the memory hierarchy from [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html), one level up.

{{fig:context-window-memory-hierarchy}}

### The scratchpad / plan file

Give the agent a persistent **scratchpad** — often literally a file like `PLAN.md` or `TODO.md` — that it owns and updates. This externalizes working memory so it survives compaction, and it gives the model a stable, re-readable anchor for the task structure. The pattern: the agent writes its plan to a file, and at the top of each turn the *current* plan file is read back into a small pinned segment. Compaction can wipe the conversation history, but the plan persists on disk.

```python
# An externalized scratchpad. The plan lives on disk, NOT only in the window.
# Each turn we re-read the (small) current plan into a pinned segment, so it
# survives compaction and the model always sees an up-to-date task structure.
import json, os

class Scratchpad:
    def __init__(self, path="PLAN.md"):
        self.path = path
        if not os.path.exists(path):
            self._write({"goal": "", "todos": [], "notes": [], "done": []})

    def _write(self, state: dict):
        with open(self.path, "w") as f:
            f.write("# PLAN\n")
            f.write(f"## Goal\n{state['goal']}\n\n")
            f.write("## TODO\n" + "".join(f"- [ ] {t}\n" for t in state["todos"]))
            f.write("\n## Done\n" + "".join(f"- [x] {d}\n" for d in state["done"]))
            f.write("\n## Notes\n" + "".join(f"- {n}\n" for n in state["notes"]))
            # Keep a machine-readable mirror for reliable parsing.
            with open(self.path + ".json", "w") as g:
                json.dump(state, g)

    def state(self) -> dict:
        with open(self.path + ".json") as f:
            return json.load(f)

    def render(self) -> str:
        """Small, dense view to pin into context every turn."""
        with open(self.path) as f:
            return f.read()

    # Tool surface the agent calls:
    def set_goal(self, goal: str):
        s = self.state(); s["goal"] = goal; self._write(s)
    def add_todo(self, item: str):
        s = self.state(); s["todos"].append(item); self._write(s)
    def complete(self, item: str):
        s = self.state()
        if item in s["todos"]:
            s["todos"].remove(item); s["done"].append(item); self._write(s)
    def note(self, text: str):
        s = self.state(); s["notes"].append(text); self._write(s)

# Each turn:
#   plan_seg.items = [pad.render()]; plan_seg.pinned = [True]
# The plan is ~300 tokens, always fresh, and immune to history compaction.
```

This "model as an operating system" framing — context window as RAM, external store as disk, with the agent paging state in and out — is the conceptual core of agent memory systems generally, which we develop fully in [Memory Systems for Agents](../08-agents-harness/05-agent-memory.html). The distinction worth holding here: the *scratchpad* is short-lived per-task working memory; *memory systems* (vector stores of past episodes, user preferences, learned facts) are long-lived cross-task memory. Both are forms of externalized context.

### Sub-agents as context isolation

A complementary technique: spin up a **sub-agent** with its own fresh context window to handle a bounded subtask (e.g., "find every call site of `deprecated_fn`"), and return only its *conclusion* to the parent. The verbose exploration — dozens of `grep` and `read` results — happens in the sub-agent's window and is discarded; the parent's window receives a clean two-line answer. This is context engineering via decomposition, and it is why multi-agent decomposition (see [Multi-Agent Systems & Orchestration](../08-agents-harness/07-multi-agent-systems.html)) is as much a context-management strategy as a capability strategy.

## Prompt Caching for Agents

Even with a tight context, a long-running agent re-sends a large, *identical* prefix on every turn: the system prompt, the tool definitions, and the unchanged head of the conversation. **Prompt caching** lets the inference server reuse the previously computed KV cache for that prefix instead of re-prefilling it, turning repeated input tokens from full price into a steep discount (often on the order of 10× cheaper for cache hits) and cutting time-to-first-token dramatically. The serving mechanism is covered in [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html) and SGLang's [RadixAttention](../07-inference-serving/04-sglang-radixattention.html); here we focus on *structuring context to maximize hits*.

### The one rule: keep the prefix byte-stable

Prefix caching matches the **longest common prefix** of the new request against cached entries. The cache hit ends at the *first byte that differs*. Therefore: order your context from **most stable to least stable**, and never mutate an earlier segment if you can avoid it.

```text
GOOD ordering (stable prefix -> cache hit on the whole gray region):

[ system prompt ][ tool schemas ][ retrieved (stable) ][ history ... ][ new turn ]
 ^---------------- cached prefix (reused) --------------^^-- new tokens, prefilled --

BAD: a timestamp or turn-counter near the TOP invalidates EVERYTHING after it:

[ "It is 14:32:07. " ][ system prompt ][ tools ] ...   <- cache miss every turn
  ^ changes each call -> longest common prefix is ~0 -> you pay full price always
```

{{fig:prompt-cache-prefix-stability}}

Common cache-busting mistakes, all fixable: putting the current timestamp, a random request ID, or a per-turn counter at the *top* of the system prompt; reordering tool definitions between calls (serialize them in a fixed, sorted order); rewriting earlier conversation turns during compaction in a way that changes their bytes (instead, compact into a *new* trailing segment and only occasionally rebuild). Move all volatile content to the *end*, just before the model's turn.

```python
# Assemble an agent request that is prefix-cache friendly (Anthropic-style
# cache_control shown; the principle is identical for vLLM/SGLang auto-prefix).
def build_cached_request(system: str, tools: list[dict],
                         stable_context: str, history: list[dict],
                         current_turn: str) -> dict:
    # Tools serialized in a FIXED order so bytes never shift between calls.
    tools_sorted = sorted(tools, key=lambda t: t["name"])
    return {
        "model": "claude-style-model",
        # System + tools: the most stable block. Mark a cache breakpoint at
        # its end so this large prefix is reused on every subsequent turn.
        "system": [
            {"type": "text", "text": system},
            {"type": "text", "text": stable_context,
             "cache_control": {"type": "ephemeral"}},  # <- cache up to here
        ],
        "tools": tools_sorted,
        "messages": history + [{"role": "user", "content": current_turn}],
        # Volatile bits (timestamp, nonce) go INSIDE current_turn, at the end,
        # so they never invalidate the cached prefix above.
    }
```

### The interaction with compaction

There is a real tension: **compaction rewrites the prefix, which busts the cache.** Right after a compaction, the next request is a full cache miss because the conversation head changed. This is fine — you compact precisely because the old context was too expensive to keep re-sending — but it means you should not compact *too eagerly*. Each compaction trades a one-time cache-miss prefill (and a summarization call) for cheaper subsequent turns. Compact when the projected savings over the next several turns exceed that one-time cost, not on every turn.

!!! example "Worked example: caching math for an agent turn"

    An agent carries a stable prefix of $P = 12{,}000$ tokens (system + tools + pinned plan) and adds $\Delta = 1{,}500$ new tokens per turn. Input is USD 3/M uncached; cached reads are USD 0.30/M (10× off); cache *writes* cost USD 3.75/M (a 25% surcharge, paid once when the prefix is first cached).

    Without caching, every turn prefills $P + \Delta = 13{,}500$ tokens at full price: $13{,}500 \times 3/\text{M} = $ USD 0.0405 per turn.

    With caching: turn 1 writes the prefix ($12{,}000 \times 3.75/\text{M} = $ USD 0.045) plus $1{,}500$ new at full price (USD 0.0045) = USD 0.0495. Turns 2+: read the cached $12{,}000$ at the discount ($12{,}000 \times 0.30/\text{M} = $ USD 0.0036) plus $1{,}500$ new at full price (USD 0.0045) = **USD 0.0081 per turn** — a 5× reduction on every steady-state turn. Over a 40-turn run: naive ≈ USD 1.62, cached ≈ USD 0.049 (write) + 39 × 0.0081 ≈ **USD 0.37**. Caching is the highest-leverage single change you can make to agent economics, which is why prefix stability is worth engineering for.

!!! interview "Interview Corner"

    **Q:** Your coding agent works great on short tasks but its answers get noticeably worse and its bill explodes on long, multi-file refactors. Walk me through how you would diagnose and fix this.

    **A:** Two coupled problems — degrading quality and exploding cost — both rooted in unmanaged context. For *cost*, naive accumulation is $O(N^2)$ in turns: the agent re-sends a growing prompt every step. I would first instrument per-turn input token counts to confirm the quadratic ramp, then (a) cap each tool result (truncate logs/files, return only relevant windows), (b) add a context cap with periodic compaction to flatten the growth to $O(NC)$, and (c) order context most-stable-first and enable prompt caching so the unchanged prefix bills at ~10× discount. For *quality*, the long context is causing context rot and lost-in-the-middle: distractor tokens dilute attention and the relevant code may be buried mid-window. I would measure the effective window with a needle-in-haystack sweep, then move to just-in-time retrieval (grep/read tools instead of pre-stuffing files), place the current task and active file at the very end of the prompt, and externalize the plan to a `PLAN.md` that is re-read each turn so it survives compaction. The unifying principle: treat the window as scarce RAM — put in the minimum high-signal context, page the rest in on demand, and keep the prefix stable for caching.

!!! key "Key Takeaways"

    - The context window is the agent's *only* working memory and the scarce resource of the whole system; engineering what enters it, in what form and order, matters as much as the prompt.
    - Naive context accumulation is $O(N^2)$ in turns and quietly dominates cost; capping context and compacting flattens it to $O(NC)$.
    - The *effective* window is smaller than the *advertised* window: context rot (long/distracting context degrades quality) and lost-in-the-middle (U-shaped position accuracy) are real and measurable — sweep them on your own stack.
    - Place the most important content at the start or the very end of the window; never bury it in the middle.
    - Prefer just-in-time retrieval (tools that grep/read minimal slices) over stuffing whole files; always bound tool outputs before they re-enter context.
    - Budget the window per segment, pin load-bearing pieces (system prompt, current task, active plan), and evict/truncate/compact in that order of increasing cost.
    - Externalize durable state to files/scratchpads (RAM-vs-disk): the plan survives compaction and can be rebuilt from ground truth rather than from lossy summaries.
    - Structure context most-stable-first and keep the prefix byte-stable so prompt caching gives a ~10× discount on the repeated prefix; compact deliberately, since it busts the cache.

!!! sota "State of the Art & Resources (2026)"
    Context engineering — the discipline of deciding what enters the LLM's finite window, in what form, and when — crystallised as a named practice in mid-2025 and is now the central skill for building reliable, cost-efficient long-running agents. Techniques such as structured compaction, just-in-time retrieval, budget management, and prefix-stable caching have all seen first-party API support and rigorous empirical study.

    **Foundational work**

    - [Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023)](https://arxiv.org/abs/2307.03172) — the canonical measurement of U-shaped position accuracy in long contexts; motivates placing critical content at window boundaries.
    - [Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2023)](https://arxiv.org/abs/2310.08560) — introduced the RAM-vs-disk framing: the context window as volatile fast memory, external storage as slow durable memory, with OS-style paging between them.

    **Recent advances (2023–2026)**

    - [Lumer et al., *Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks* (2026)](https://arxiv.org/abs/2601.06007) — rigorous measurement showing prefix caching cuts API cost 41–80% and TTFT 13–31% across providers on multi-turn agentic benchmarks; quantifies the compaction–cache-miss trade-off.

    **Open-source & tools**

    - [gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — the standard harness for sweeping context length × needle depth to measure your model's effective window; widely adopted by Anthropic, Google, and OpenAI.
    - [nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle) — code and data from the Lost-in-the-Middle paper; multi-doc QA and key-value retrieval benchmarks for position-effect evaluation.
    - [mem0ai/mem0](https://github.com/mem0ai/mem0) — 57 k-star universal memory layer for agents; provides managed persistent memory that offloads long-term context from the window to an external store.

    **Go deeper**

    - [Anthropic Engineering, *Effective context engineering for AI agents* (Sep 2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Anthropic's practitioner guide: compaction triggers, tool-result clearing, scratchpad patterns, and multi-agent isolation as context strategy.
    - [Anthropic Cookbook, *Context engineering: memory, compaction, and tool clearing* (Mar 2026)](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools) — runnable code comparing the three first-party context primitives (compaction, tool-result clearing, memory tool) on a realistic long-running research agent.
    - [Anthropic, *Prompt caching* (claude.com)](https://claude.com/blog/prompt-caching) — the announcement and economics of prefix caching in Claude: up to 90% cost reduction and 85% latency reduction for stable-prefix agent turns.
    - [Simon Willison, *Context engineering* (Jun 2025)](https://simonwillison.net/2025/jun/27/context-engineering/) — a lucid 5-minute read on why "context engineering" displaced "prompt engineering" as the right framing for this discipline, with pointers to Karpathy and Lütke's formulations.

## Further reading

- Liu, Lin, Hewitt, et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023) — the canonical study of position effects in long contexts.
- *Lin et al. / "Needle in a Haystack"* evaluation methodology (Greg Kamradt's NIAH harness) — the practical recipe for measuring effective context length.
- Anthropic engineering, *Prompt caching* and *Effective context engineering for AI agents* — vendor guidance on cache-friendly structuring and just-in-time context.
- Packer et al., *MemGPT: Towards LLMs as Operating Systems* (2023) — the RAM-vs-disk framing of context as a managed memory hierarchy.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (RadixAttention) — automatic prefix sharing as a serving-side complement to context engineering.
- Lewis et al., *Retrieval-Augmented Generation* (2020) — the retrieval-vs-stuffing foundation; pairs with Part IX of this book.

## Exercises

**1.** A retrieval step returns 8 documents ranked by relevance score, $d_1$ (best) through $d_8$ (worst), which you will stuff into the window. A colleague proposes concatenating them in rank order, $d_1 d_2 \dots d_8$, reasoning that the most relevant document should come first. Using the position effects described in this chapter, explain why this ordering is suboptimal and give a better one.

??? note "Solution"

    Rank order puts $d_1$ at the start (good) but leaves the *second*-best document, $d_2$, and the rest of the strong documents progressively deeper toward the middle, exactly where **lost-in-the-middle** predicts the worst retrieval accuracy. The accuracy-versus-position curve is U-shaped: strong at the beginning and end, sagging in the middle. Concatenating $d_1 \dots d_8$ places your two most valuable documents ($d_1, d_2$) adjacent at the front, so $d_2$ still enjoys an edge, but $d_3, d_4$ land in the low-accuracy trough while the weakest documents occupy the high-salience *end* of the window.

    The chapter's actionable rule is: "place the most important material at the start or the very end of the window, never buried in the middle... When you stuff $k$ retrieved documents, put the highest-scoring ones at the boundaries." So a better layout brackets the boundaries with the best documents and hides the weak ones in the middle, e.g.:

    $$
    d_1 \; d_3 \; d_5 \; \underbrace{d_7 \; d_8}_{\text{middle: weakest}} \; d_6 \; d_4 \; d_2
    $$

    Here $d_1$ sits at the very start and $d_2$ at the very end (the two highest-salience slots), $d_3$/$d_4$ just inside them, and the two least relevant documents ($d_7, d_8$) absorb the middle trough where a loss of signal costs the least.

**2.** An agent runs for $N = 30$ turns and appends everything, never pruning. The system prompt plus tools are $L_0 = 3{,}000$ tokens, and each turn adds $a = 2{,}000$ tokens (assistant message plus tool result). Input is billed at USD 3 per million tokens.

    (a) Compute the total input tokens billed across the run and the input cost. (b) If you instead halve the per-turn footprint to $a = 1{,}000$ tokens, what is the new input cost, and by what factor did the bill change? Explain the factor using the chapter's cost model.

??? note "Solution"

    **(a)** The prompt length at turn $t$ is $L_t = L_0 + a\,t = 3000 + 2000\,t$. The total input billed is the sum of every prefix:

    $$
    T_{\text{in}} = \sum_{t=1}^{30}(3000 + 2000\,t) = 30\cdot 3000 + 2000\cdot\frac{30\cdot 31}{2} = 90{,}000 + 2000\cdot 465 = 90{,}000 + 930{,}000 = 1{,}020{,}000
    $$

    So $T_{\text{in}} = 1.02$M tokens. Input cost $= 1.02 \times 3 = $ **USD 3.06**.

    **(b)** With $a = 1{,}000$: $T_{\text{in}} = 30\cdot 3000 + 1000\cdot 465 = 90{,}000 + 465{,}000 = 555{,}000$ tokens $= 0.555$M. Cost $= 0.555 \times 3 = $ **USD 1.665**.

    The bill fell by a factor of $3.06 / 1.665 \approx 1.84$, i.e. a bit less than 2x. This follows directly from the chapter's cost model $T_{\text{in}} = L_0 N + a\,\tfrac{N(N+1)}{2}$, which is *linear* in $a$: the $a$-dependent term halves exactly ($930{,}000 \to 465{,}000$) when you halve $a$, so if the fixed $L_0 N = 90{,}000$ term were negligible the total would halve too — a clean 2x. But $L_0 N$ does not depend on $a$, so it is left unchanged and dilutes the reduction to 1.84x. (Note it is *halving*, not quartering: $a$ scales the dominant $\tfrac{1}{2}aN^2$ term linearly; it is halving $N$ that would quarter the bill.) The pure-quadratic $\tfrac{1}{2}aN^2$ approximation dominates only once $a N \gg L_0$; at $N = 30$ the constant $L_0 N$ still contributes enough to hold the factor below 2. Halving the footprint of a *longer* run (larger $N$, so the $a$-term dwarfs $L_0 N$) would approach the clean 2x more closely.

**3.** An agent carries a stable prefix of $P = 10{,}000$ tokens (system + tools + pinned plan) and adds $\Delta = 1{,}000$ new tokens per turn. Pricing: uncached input USD 3/M; cached *reads* USD 0.30/M; a one-time cache *write* USD 3.75/M for the prefix. Compute (a) the per-turn cost with no caching, (b) the steady-state per-turn cost once the prefix is cached, and (c) the total cost of a 20-turn run with and without caching. What is the overall ratio?

??? note "Solution"

    **(a) No caching.** Every turn re-prefills the whole prompt $P + \Delta = 11{,}000$ tokens at full price:

    $$
    11{,}000 \times \frac{3}{10^6} = \text{USD } 0.033 \text{ per turn.}
    $$

    **(b) Cached steady state (turns 2+).** The unchanged $P = 10{,}000$ prefix is read from cache at the discount, and only the $\Delta = 1{,}000$ new tokens prefill at full price:

    $$
    \underbrace{10{,}000 \times \tfrac{0.30}{10^6}}_{\text{cached read }=\,0.003} + \underbrace{1{,}000 \times \tfrac{3}{10^6}}_{\text{new }=\,0.003} = \text{USD } 0.006 \text{ per turn.}
    $$

    That is a $0.033/0.006 = 5.5\times$ reduction on every steady-state turn.

    **(c) 20-turn totals.** Without caching: $20 \times 0.033 = $ **USD 0.66**.

    With caching, turn 1 pays the write surcharge on the prefix plus the new tokens at full price:

    $$
    \text{turn 1} = 10{,}000 \times \tfrac{3.75}{10^6} + 1{,}000 \times \tfrac{3}{10^6} = 0.0375 + 0.003 = \text{USD } 0.0405,
    $$

    then turns 2-20 (19 turns) cost USD 0.006 each: $19 \times 0.006 = 0.114$. Total $= 0.0405 + 0.114 = $ **USD 0.1545**.

    Overall ratio $= 0.66 / 0.1545 \approx 4.3\times$ cheaper. The one-time write surcharge (USD 0.0375 vs 0.03 uncached, a 25% premium paid once) is repaid within the first cached turn, and every subsequent turn compounds the saving — which is why the chapter calls prefix stability "the highest-leverage single change you can make to agent economics."

**4.** The chapter gives one rule for cacheable context ("keep the prefix byte-stable") and separately warns that compaction "busts the cache." (a) Explain mechanically why placing a live timestamp at the *top* of the system prompt causes a cache miss on every turn, whereas placing it at the *end* of the current turn does not. (b) Explain why compaction busts the cache, and state the rule for *when* it is nonetheless worth doing.

??? note "Solution"

    **(a)** Prefix caching matches the **longest common prefix** of the new request against a cached entry, and "the cache hit ends at the *first byte that differs*." A timestamp like `"It is 14:32:07. "` changes every call. If it sits at the *top*, the very first bytes of the request differ from the cached entry, so the longest common prefix is essentially zero — everything after it (system prompt, tools, history) must be re-prefilled at full price, every turn. If instead the volatile timestamp goes at the *end*, inside the current user turn, then the entire stable block before it (system + tools + retrieved + history) is a byte-identical prefix that matches the cache; only the short trailing turn — which was going to be new anyway — is prefilled. The chapter's directive: "Move all volatile content to the *end*, just before the model's turn."

    **(b)** Compaction *rewrites the conversation head*: it replaces a long span of old turns with a freshly generated summary, changing the bytes near the front of the messages. Since the cache match ends at the first differing byte, that rewrite invalidates the cached prefix downstream of it, so the turn immediately after a compaction is a full cache miss (plus the cost of the summarization model call itself).

    The rule: **compact deliberately, not eagerly.** Each compaction trades a one-time cost (a cache-miss prefill of the new, shorter prefix + one summarization call) for cheaper subsequent turns (a smaller prefix re-sent at the cache discount). It is worth doing only when "the projected savings over the next several turns exceed that one-time cost" — i.e. when you expect enough remaining turns to amortize the busted cache, not on every turn.

**5.** The "Common pitfall" admonition warns that piping raw, unbounded tool output into the window is the most common way agents blow their budget, and prescribes truncating "with an explicit `[... 4,213 lines elided ...]` marker." Implement a function `bound_tool_output(text, max_tokens=2000, head_frac=0.5)` that returns `text` unchanged if it is within budget, and otherwise keeps a head and a tail slice (splitting the budget by `head_frac`) joined by an explicit elided-token breadcrumb. Use the chapter's real-tokenizer helper `_enc` / `ntok`. Then show, with a short example, why keeping both a head *and* a tail (rather than just a head) matters for tool output.

??? note "Solution"

    We slice on real tokens (not characters) so the cap is exact, keep the first `head_frac` of the budget from the top and the remainder from the bottom, and stitch them with a breadcrumb that both records how much was dropped and tells the model how to recover it — mirroring the chapter's "a tool wrapper that caps every result... and tells the model how to fetch more is worth more than a cleverer prompt."

    ```python
    # Bound a single tool result to a hard token budget, keeping head + tail
    # with an explicit breadcrumb. Uses the chapter's _enc / ntok helpers.
    def bound_tool_output(text: str, max_tokens: int = 2000,
                          head_frac: float = 0.5) -> str:
        toks = _enc.encode(text)
        if len(toks) <= max_tokens:
            return text                      # already within budget: untouched
        n_head = int(max_tokens * head_frac)
        n_tail = max_tokens - n_head
        n_elided = len(toks) - n_head - n_tail
        head = _enc.decode(toks[:n_head])
        tail = _enc.decode(toks[-n_tail:]) if n_tail else ""
        marker = (f"\n[... {n_elided} tokens elided; "
                  f"narrow the query or call read_window to fetch more ...]\n")
        return head + marker + tail

    # (The breadcrumb itself costs a handful of tokens, so the result is a hair
    #  over max_tokens; reserve a small margin upstream if the cap is hard.)
    ```

    **Why head *and* tail.** Tool output is frequently most informative at *both* ends. A `pytest` run, for example, prints the collected-tests banner and progress dots at the top but the actual `FAILED test_x - AssertionError...` summary and the exit status at the *bottom*. A head-only truncation ($head\_frac = 1.0$) would keep the useless progress noise and elide precisely the failure summary the agent needs to act. Concretely, for a 5,000-line log at `max_tokens=2000`, head+tail preserves both the invocation/first errors and the final failure roster and exit code, while the middle — the repetitive bulk — is exactly the low-signal span it is safest to drop, echoing the lost-in-the-middle intuition that the boundaries carry the load. The breadcrumb keeps the elision honest so the model does not hallucinate continuity across the gap, and it points at the just-in-time tools (`read_window`, `grep`) for pulling any elided detail back on demand.
