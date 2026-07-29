"""
Runs the CPU-runnable Python code blocks from:
    content/14-capstone/10-agentic-narrow.md

Blocks are copied faithfully from the chapter (verbatim logic) and assembled into
one module. The chapter presents the modules in *narrative* order, which is not
always dependency order (e.g. the book's own `test_agent_pipeline.py` block sits
*before* `sft_format.py` in the document even though it imports from it) -- this
file reorders the code to dependency order while keeping every block's logic
byte-for-byte as printed, and drops the chapter's `from stacklm.agent.X import Y`
lines because every symbol is already defined earlier in this single file.

Tested blocks:
    (glue) corpus.py, doc line ~76  -- the heuristic classifier tagged this
        "needs-gpu"; it is pure stdlib (random + dataclasses, no torch/cuda
        anywhere) and is included as REQUIRED glue because every one of the 11
        target blocks below transitively depends on Task/build_corpus/
        make_task_pool. Exercised via build_corpus()/make_task_pool()/make_tasks().
    #2  (tools.py, line ~300)        -- calc, BM25Retriever, HashEmbedRetriever
    #5  (react.py, line ~521)        -- wire format, parse/render, enc()
    #7  (line ~621)                  -- TOOL_CALL_SCHEMA (illustrative dict)
    #8  (grammar.py, line ~638)      -- character acceptor + token-mask lifting
    #12 (distill.py, line ~822)      -- ToolEnv, rollout, distill (rejection sampling)
    #13 (stub_teacher.py, line ~910) -- hermetic CI test-double teacher
    #14 (test_agent_pipeline.py, line ~999) -- the chapter's own 3 CI tests,
        called directly (with `tok` passed as a plain argument instead of a
        pytest fixture)
    #15 (sft_format.py, line ~1105)  -- _segment, build_example, build_dataset
    #16 (loop.py, line ~1201)        -- Trace, run_agent, generate (grammar-
        constrained decoding); driven by a tiny scripted stand-in "model" (see
        below -- no real weights, no network)
    #17 (roles.py, line ~1368)       -- split_roles

Skipped blocks:
    #3  (retriever_dense.py, line ~462): SKIP(network). The chapter's own
        comment on this block says "NOT imported by CI: needs network + deps";
        it does `from sentence_transformers import SentenceTransformer` and
        `import faiss` and would instantiate a real embedding model. Per the
        hard rule against instantiating real models, this block is not run.
    #0, #4, #6, #9 -- non-python (ASCII diagrams / prose / JSON schema shown
        for MCP, not executed).
    #1  (mcp_adapter.py, `tools_line`/`schemas`) -- non-standalone fragment
        that only matters with a real MCP server; not exercised.
    #10 (illustrative teacher_step()) -- real `anthropic` API call, network.
    #18 (roles.py's Ch. discussion / rlvr.py, line ~1431) -- needs-gpu: real
        SFT training loop invocation, not CPU-runnable in the CI time budget.
    #19, #20 (indented Exercise-appendix fragments, line ~1709/1791) -- fragments.

No network access and no optional third-party imports are exercised: only
numpy, torch, and the standard library, all in the guaranteed CI list.

Bug found and fixed while writing this test: NONE. Every block ran verbatim
against the chapter's own text and its own assertions (block #14's three CI
tests) as written.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch

# =====================================================================
# GLUE (not a chapter block): a tiny tokenizer standing in for
# `stacklm.tokenizer.StackTokenizer`. It satisfies the contract used
# throughout the chapter: encode(text, allowed_special=...) -> list[int],
# decode(ids) -> str, special_token_id(str) -> int, .vocab_size. Every
# special string below is byte-identical to the ones `react.py` defines.
#
# It is byte-level with a crude "merge" on top -- every maximal run of
# alphanumeric characters becomes ONE token (a stand-in for real BPE's
# multi-character merges), anything else (space/punctuation/JSON syntax) is
# one byte per token. This is what keeps ~1.5KB transcripts under
# `sft_format.py`'s real max_len=1024 default, matching the chapter's own
# measured ~4-chars/token compression closely enough to exercise the actual
# code path instead of an artifact of an unrealistically granular test double.
# It is exactly invertible (decode(encode(t)) == t for all ASCII t), which the
# grammar acceptor, the JSON parser, and the loss-mask segmenter all require.
# =====================================================================
_SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|end|>", "<|system|>",
             "<|user|>", "<|assistant|>", "<|tool_call|>", "<|tool_result|>"]


class ByteTokenizer:
    def __init__(self):
        self._sp2id = {s: i for i, s in enumerate(_SPECIALS)}
        self._id2sp = {v: k for k, v in self._sp2id.items()}
        self._byte_base = len(_SPECIALS)                 # bytes start right after specials
        self._chunk2id: dict[str, int] = {}
        self._id2chunk: dict[int, str] = {}
        self._next_chunk_id = self._byte_base + 256       # chunks start after the byte range
        self.vocab_size = self._next_chunk_id

    def special_token_id(self, s: str) -> int:
        return self._sp2id[s]

    def _byte_id(self, ch: str) -> int:
        return self._byte_base + (ord(ch) % 256)

    def _chunk_id(self, chunk: str) -> int:
        cid = self._chunk2id.get(chunk)
        if cid is None:
            cid = self._next_chunk_id
            self._chunk2id[chunk] = cid
            self._id2chunk[cid] = chunk
            self._next_chunk_id += 1
            self.vocab_size = self._next_chunk_id
        return cid

    def encode(self, text: str, allowed_special=frozenset()) -> list[int]:
        specials = sorted(allowed_special, key=len, reverse=True)
        ids, i, n = [], 0, len(text)
        while i < n:
            hit = next((s for s in specials if text.startswith(s, i)), None)
            if hit is not None:
                ids.append(self._sp2id[hit])
                i += len(hit)
                continue
            if text[i].isalnum():
                j = i
                while j < n and text[j].isalnum():
                    j += 1
                ids.append(self._chunk_id(text[i:j]))
                i = j
            else:
                ids.append(self._byte_id(text[i]))
                i += 1
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in self._id2sp:
                out.append(self._id2sp[i])
            elif i in self._id2chunk:
                out.append(self._id2chunk[i])
            else:
                out.append(chr(i - self._byte_base))
        return "".join(out)


tok = ByteTokenizer()

# =====================================================================
# GLUE block (chapter: capstone/stacklm/agent/corpus.py, doc line ~76).
# Heuristically mistagged "needs-gpu" -- it is pure stdlib. Included
# verbatim because blocks #12/#13/#14 all need Task/build_corpus/make_task_pool.
# (`from stacklm.agent.tools import Passage` dropped: Passage is defined below,
# in block #2, before build_corpus() is ever CALLED, which is all Python needs.)
# =====================================================================
SPEC_FACTS = [
    ("Stack-100M", "hidden size", "512"),
    ("Stack-100M", "layer count", "30"),
    ("Stack-100M", "vocabulary size", "32768"),
    ("Stack-100M", "query head count", "8"),
    ("Stack-100M", "key-value head count", "2"),
    ("Stack-100M", "head dimension", "64"),
    ("Stack-100M", "MLP inner dimension", "1408"),
    ("Stack-100M", "pretraining sequence length", "2048"),
    ("Stack-100M", "mid-training sequence length", "8192"),
    ("Stack-100M", "RoPE base", "10000"),
    ("Stack-100M", "parameter count in millions", "101"),
    ("Stack-100M", "pretraining token budget in billions", "20"),
]

PAPER_FACTS = [
    ("the Transformer paper", "publication year", "2017"),
    ("Adam", "publication year", "2014"),
    ("RMSNorm", "publication year", "2019"),
    ("SwiGLU", "publication year", "2020"),
    ("RoFormer", "publication year", "2021"),
    ("LoRA", "publication year", "2021"),
    ("Chinchilla", "publication year", "2022"),
    ("FlashAttention", "publication year", "2022"),
    ("ReAct", "publication year", "2022"),
    ("STaR", "publication year", "2022"),
    ("Toolformer", "publication year", "2023"),
    ("DPO", "publication year", "2023"),
    ("grouped-query attention", "publication year", "2023"),
    ("DeepSeekMath", "publication year", "2024"),
    ("Muon", "publication year", "2024"),
    ("MobileLLM", "publication year", "2024"),
]

_DISTRACTOR_ATTRS = ["hidden size", "layer count", "MLP inner dimension",
                     "vocabulary size", "head dimension"]

_TEMPLATES = [
    "Reference note {n}. In the Stack-100M project notes, the {attr} of {ent} "
    "is {val}. This value is fixed by the capstone specification and is not "
    "tuned per run.",
    "Configuration record {n}: the {attr} of {ent} is {val}. Downstream stages "
    "(mid-training, supervised fine-tuning, serving) all assume this setting "
    "without re-deriving it.",
    "Design log entry {n}. We record here that the {attr} of {ent} is {val}, a "
    "choice made during the architecture pass and left unchanged since.",
]


def _render(n: int, ent: str, attr: str, val: str) -> str:
    return _TEMPLATES[n % len(_TEMPLATES)].format(n=n, ent=ent, attr=attr, val=val)


def chunk_words(text: str, size: int = 80, overlap: int = 16) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    out, step = [], max(1, size - overlap)
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if piece:
            out.append(" ".join(piece))
        if i + size >= len(words):
            break
    return out


def build_corpus(seed: int = 0, n_distractors: int = 120) -> list["Passage"]:
    rng = random.Random(seed)
    facts = list(SPEC_FACTS) + list(PAPER_FACTS)
    for i in range(n_distractors):
        ent = f"ablation run R-{i:03d}"
        for attr in rng.sample(_DISTRACTOR_ATTRS, 2):
            facts.append((ent, attr, str(rng.choice([128, 192, 256, 384, 640,
                                                     768, 896, 1024, 1536]))))
    passages: list["Passage"] = []
    for n, (ent, attr, val) in enumerate(facts):
        for j, piece in enumerate(chunk_words(_render(n, ent, attr, val))):
            passages.append(Passage(doc_id=f"doc-{n:04d}-{j}", text=piece))
    return passages


@dataclass
class Task:
    question: str
    gold: str
    kind: str
    hints: dict = field(default_factory=dict)


def _is_num(v: str) -> bool:
    return v.lstrip("-").isdigit()


def make_tasks(seed: int = 0) -> list[Task]:
    rng = random.Random(seed + 1)
    facts = list(SPEC_FACTS) + list(PAPER_FACTS)
    numeric = [f for f in facts if _is_num(f[2])]
    tasks: list[Task] = []

    for ent, attr, val in facts:
        tasks.append(Task(
            question=f"What is the {attr} of {ent}? Use the corpus.",
            gold=val, kind="lookup",
            hints={"queries": [f"{ent} {attr}"], "facts": [(ent, attr)], "op": None}))

    for ent, attr, val in numeric:
        tasks.append(Task(
            question=f"What is the {attr} of {ent}, multiplied by 2? Use the corpus.",
            gold=str(int(val) * 2), kind="double",
            hints={"queries": [f"{ent} {attr}"], "facts": [(ent, attr)], "op": "double"}))

    for _ in range(120):
        (e1, a1, v1), (e2, a2, v2) = rng.sample(numeric, 2)
        tasks.append(Task(
            question=(f"What do you get if you add the {a1} of {e1} to the "
                      f"{a2} of {e2}? Use the corpus."),
            gold=str(int(v1) + int(v2)), kind="sum",
            hints={"queries": [f"{e1} {a1}", f"{e2} {a2}"],
                   "facts": [(e1, a1), (e2, a2)], "op": "sum"}))

    for _ in range(120):
        (e1, a1, v1), (e2, a2, v2) = rng.sample(numeric, 2)
        if e1 == e2 or int(v1) == int(v2):
            continue
        tasks.append(Task(
            question=(f"Which is larger, the {a1} of {e1} or the {a2} of {e2}? "
                      f"Answer with the entity name. Use the corpus."),
            gold=e1 if int(v1) > int(v2) else e2, kind="compare",
            hints={"queries": [f"{e1} {a1}", f"{e2} {a2}"],
                   "facts": [(e1, a1), (e2, a2)], "op": "compare",
                   "names": [e1, e2]}))
    return tasks


def make_task_pool(seed: int = 0, n_train: int = 200, n_heldout: int = 50):
    tasks = make_tasks(seed)
    random.Random(seed + 2).shuffle(tasks)
    assert len(tasks) >= n_train + n_heldout, "pool too small; raise the counts"
    return tasks[:n_train], tasks[n_train:n_train + n_heldout]


def qa_pairs(tasks: list[Task]) -> list[dict]:
    return [{"question": t.question, "gold_answer": t.gold} for t in tasks]


# =====================================================================
# Block #2 (chapter: capstone/stacklm/agent/tools.py, doc line ~300)
# =====================================================================
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def calc(expr: str) -> str:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        val = _eval_node(tree)
    except Exception as e:                          # noqa: BLE001 (catch all)
        return f"CalcError: {e}"
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


_TOK = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


@dataclass
class Passage:
    doc_id: str
    text: str


class BM25Retriever:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [_tokenize(p.text) for p in passages]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.docs))
        df: Counter = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        N = len(self.docs)
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
    def __init__(self, passages: list[Passage], dim: int = 512):
        self.passages, self.dim = passages, dim
        self.vecs = [self._embed(p.text) for p in passages]

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in _tokenize(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            idx, rest = h % self.dim, h // self.dim
            v[idx] += 1.0 if rest & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        qv = self._embed(query)
        scored = [(self.passages[i], sum(a * b for a, b in zip(qv, self.vecs[i])))
                  for i in range(len(self.passages))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, sc) for p, sc in scored[:k] if sc > 0.0]


# Block #3 (retriever_dense.py, doc line ~462): SKIP(network).
# The chapter's own comment on this block reads:
#   "capstone/stacklm/agent/retriever_dense.py  (NOT imported by CI: needs
#    network + deps)"
# It does `from sentence_transformers import SentenceTransformer` and
# `import faiss`, and would instantiate a real embedding model
# (google/embeddinggemma-300m). Never executed here.


# =====================================================================
# Block #5 (chapter: capstone/stacklm/agent/react.py, doc line ~521)
# =====================================================================
BOS, EOS, PAD, END = "<|bos|>", "<|eos|>", "<|pad|>", "<|end|>"
SYS, USER, ASST = "<|system|>", "<|user|>", "<|assistant|>"
TOOL_CALL, TOOL_RESULT = "<|tool_call|>", "<|tool_result|>"

ALL_SPECIAL = frozenset({BOS, EOS, PAD, END, SYS, USER, ASST,
                         TOOL_CALL, TOOL_RESULT})

ANSWER_RE = re.compile(r"Answer:\s*(.+?)\s*$", re.DOTALL)


def enc(tok, text: str) -> list[int]:
    return tok.encode(text, allowed_special=ALL_SPECIAL)


@dataclass
class Action:
    kind: str
    thought: str = ""
    tool: str | None = None
    args: dict | None = None
    answer: str | None = None


def parse_assistant_step(text: str) -> Action:
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
            return Action("tool", thought, tool="__malformed__", args={"raw": payload})

    m = ANSWER_RE.search(text)
    if m:
        return Action("final", thought, answer=m.group(1).strip())

    return Action("final", thought, answer=text.strip())


def render_tool_result(obs: str) -> str:
    return f"{TOOL_RESULT}{obs}{END}"


def render_call(tool: str, args: dict) -> str:
    body = json.dumps({"tool": tool, "args": args}, separators=(", ", ": "))
    return f"{TOOL_CALL}{body}{END}"


# =====================================================================
# Block #7 (doc line ~621): illustrative JSON Schema handed to a served model.
# =====================================================================
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["search", "calc"]},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}


# =====================================================================
# Block #8 (chapter: capstone/stacklm/agent/grammar.py, doc line ~638)
# =====================================================================
_DIGITS = frozenset("123456789")
_QUERY  = frozenset("abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._,()-")
_EXPR   = frozenset("0123456789+-*/.() ")


@dataclass(frozen=True)
class Hole:
    chars: frozenset
    min_len: int
    max_len: int


SEARCH_T = ('{"tool": "search", "args": {"query": "', Hole(_QUERY, 1, 64),
            '", "k": ', Hole(_DIGITS, 1, 1), '}}')
CALC_T   = ('{"tool": "calc", "args": {"expr": "', Hole(_EXPR, 1, 40), '"}}')
TEMPLATES = (SEARCH_T, CALC_T)

State = tuple


def start_states() -> frozenset:
    return frozenset((ti, 0, 0) for ti in range(len(TEMPLATES)))


def _succ(state: State, ch: str):
    ti, ii, n = state
    tmpl = TEMPLATES[ti]
    if ii >= len(tmpl):
        return
    item = tmpl[ii]
    if isinstance(item, str):
        if item[n] == ch:
            yield (ti, ii, n + 1) if n + 1 < len(item) else (ti, ii + 1, 0)
        return
    if ch in item.chars and n < item.max_len:
        yield (ti, ii, n + 1)
    if n >= item.min_len:
        yield from _succ((ti, ii + 1, 0), ch)


def step(states: frozenset, ch: str) -> frozenset:
    out: set = set()
    for s in states:
        out.update(_succ(s, ch))
    return frozenset(out)


def accepts(states: frozenset) -> bool:
    return any(ii >= len(TEMPLATES[ti]) for ti, ii, _ in states)


_MASK_CACHE: dict = {}


def build_vocab_strings(tok) -> list[str]:
    return [tok.decode([i]) for i in range(tok.vocab_size)]


def token_transitions(states: frozenset, vocab_strings: list[str]):
    hit = _MASK_CACHE.get(states)
    if hit is not None:
        return hit
    allowed, nxt = [], {}
    for tid, s in enumerate(vocab_strings):
        if not s:
            continue
        cur = states
        for ch in s:
            cur = step(cur, ch)
            if not cur:
                break
        if cur:
            allowed.append(tid)
            nxt[tid] = cur
    _MASK_CACHE[states] = (allowed, nxt)
    return allowed, nxt


# =====================================================================
# Block #12 (chapter: capstone/stacklm/agent/distill.py, doc line ~822)
# =====================================================================
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


def normalize(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return " ".join(s.lower().split())


def rollout(task: Task, teacher, env: ToolEnv, max_steps: int = 6):
    transcript = f"{SYS}{SYSTEM_PROMPT}{END}{USER}{task.question}{END}"
    solved = False
    for _ in range(max_steps):
        gen = teacher(transcript + ASST)
        step_text = gen.split(END, 1)[0]
        act = parse_assistant_step(step_text)
        transcript += f"{ASST}{step_text}{END}"
        if act.kind == "final":
            solved = (normalize(act.answer) == normalize(task.gold))
            break
        obs = env.run_tool(act)
        transcript += render_tool_result(obs)
    return transcript, solved


def distill(tasks, teacher, env, samples_per_task: int = 4):
    kept = []
    for task in tasks:
        seen = set()
        for _ in range(samples_per_task):
            transcript, solved = rollout(task, teacher, env)
            if solved and transcript not in seen:
                seen.add(transcript)
                kept.append({"task": task.question, "text": transcript,
                             "kind": task.kind})
    return kept


# =====================================================================
# Block #13 (chapter: capstone/stacklm/agent/stub_teacher.py, doc line ~910)
# =====================================================================
def _observations(prompt: str) -> list[str]:
    return [chunk.split(END, 1)[0]
            for chunk in prompt.split(TOOL_RESULT)[1:]]


def _extract(observations: list[str], ent: str, attr: str) -> str | None:
    pat = re.compile(rf"the {re.escape(attr)} of {re.escape(ent)} is ([^.,;]+)[.,;]")
    for obs in observations:
        m = pat.search(obs)
        if m:
            return m.group(1).strip()
    return None


def make_stub_teacher(tasks):
    by_q = {t.question: t for t in tasks}

    def teacher(prompt: str) -> str:
        question = prompt.split(USER, 1)[1].split(END, 1)[0]
        task = by_q[question]
        obs = _observations(prompt)
        queries, facts = task.hints["queries"], task.hints["facts"]
        op = task.hints["op"]

        if len(obs) < len(queries):
            i = len(obs)
            return (f"Thought: I need the {facts[i][1]} of {facts[i][0]}. "
                    f"I'll search the corpus.\n"
                    f"{render_call('search', {'query': queries[i], 'k': 2})}")

        vals = [_extract(obs, e, a) for e, a in facts]
        if any(v is None for v in vals):
            return "Thought: The corpus did not give me the fact.\nAnswer: unknown"

        n_calc = len(obs) - len(queries)
        if n_calc == 0 and op is not None:
            if op == "double":
                expr = f"{vals[0]}*2"
            elif op == "sum":
                expr = f"{vals[0]}+{vals[1]}"
            else:
                expr = f"{vals[0]}-{vals[1]}"
            return (f"Thought: Now I compute {expr} with the calculator.\n"
                    f"{render_call('calc', {'expr': expr})}")

        if op is None:
            answer = vals[0]
        elif op == "compare":
            diff = obs[-1].strip()
            names = task.hints["names"]
            answer = names[0] if not diff.startswith("-") else names[1]
        else:
            answer = obs[-1].strip()
        return f"Thought: I have everything I need.\nAnswer: {answer}"

    return teacher


# =====================================================================
# Block #15 (chapter: capstone/stacklm/agent/sft_format.py, doc line ~1105)
# Moved ahead of block #14's test suite (below) because that suite imports
# `build_example`/`_segment`/`IGNORE` from it -- a dependency-order fix, not a
# logic change; the document itself presents the test file BEFORE this module.
# =====================================================================
IGNORE = -100


def _segment(t: str):
    out, i, supervised = [], 0, False
    markers = [ASST, USER, SYS, TOOL_RESULT, END]
    while i < len(t):
        j, m = min([(t.find(x, i), x) for x in markers if t.find(x, i) != -1],
                   default=(len(t), None), key=lambda z: z[0])
        if j > i:
            out.append((t[i:j], supervised))
        if m is None:
            break
        if m == ASST:
            supervised = True
            out.append((m, False))
        elif m in (USER, SYS, TOOL_RESULT):
            supervised = False
            out.append((m, False))
        else:                                 # END
            out.append((m, supervised))
            supervised = False
        i = j + len(m)
    return out


def build_example(transcript: str, tok, max_len: int = 1024):
    ids, labels = [], []
    for text, supervised in _segment(transcript):
        piece = enc(tok, text)
        ids.extend(piece)
        labels.extend(piece if supervised else [IGNORE] * len(piece))
    if len(ids) > max_len:
        return None
    input_ids = np.array(ids[:-1], dtype=np.int64)
    target    = np.array(labels[1:], dtype=np.int64)
    return input_ids, target


def build_dataset(kept: list[dict], tok, max_len: int = 1024):
    out = [ex for ex in (build_example(r["text"], tok, max_len) for r in kept)
           if ex is not None]
    dropped = len(kept) - len(out)
    if dropped:
        print(f"[sft_format] dropped {dropped}/{len(kept)} over-length traces "
              f"(max_len={max_len}); raise max_len if this is more than a few %")
    return out


# =====================================================================
# Block #14 (chapter: capstone/tests/test_agent_pipeline.py, doc line ~999)
# `tok` is passed in directly rather than injected as a pytest fixture --
# same functions, same asserts, called explicitly below.
# =====================================================================
def test_distill_yields_correct_transcripts():
    corpus = build_corpus()
    train, held = make_task_pool()
    assert len(corpus) == 268 and len(train) == 200 and len(held) == 50
    env = ToolEnv(corpus)
    teacher = make_stub_teacher(train)
    kept = distill(train[:40], teacher, env, samples_per_task=2)

    assert len(kept) >= 35, f"teacher yield collapsed: {len(kept)}/40"
    gold = {t.question: t.gold for t in train}
    for row in kept:
        final = row["text"].rsplit("Answer:", 1)[1].split(END, 1)[0]
        assert normalize(final) == normalize(gold[row["task"]])
    assert {r["kind"] for r in kept} == {"lookup", "double", "sum", "compare"}


def test_special_tokens_are_single_ids_and_masking_is_nonvacuous(tok):
    for s in (TOOL_CALL, TOOL_RESULT, END):
        assert len(enc(tok, s)) == 1, f"{s} is not a single special id"
    corpus = build_corpus()
    train, _ = make_task_pool()
    env, teacher = ToolEnv(corpus), make_stub_teacher(train)
    kept = distill(train[:4], teacher, env, samples_per_task=1)
    segs = _segment(kept[0]["text"])
    assert "".join(s for s, _ in segs) == kept[0]["text"]
    ids, labels = build_example(kept[0]["text"], tok)
    n_sup = int((labels != IGNORE).sum())
    assert 0 < n_sup < len(labels), "loss mask is empty or masks nothing"


def test_grammar_accepts_what_we_render_and_rejects_what_we_fear():
    train, _ = make_task_pool()
    bodies = [render_call("calc", {"expr": "2021*2"}),
              render_call("calc", {"expr": "2019+10000"})]
    bodies += [render_call("search", {"query": q, "k": 2})
               for t in train for q in t.hints["queries"]]
    for body in bodies:
        json_part = body.split(TOOL_CALL, 1)[1].split(END, 1)[0]
        st = start_states()
        for ch in json_part:
            st = step(st, ch)
            assert st, f"grammar rejected our own output at {ch!r}"
        assert accepts(st)
    st = start_states()
    for ch in '{"tool": search':
        st = step(st, ch)
    assert not st


# =====================================================================
# Block #16 (chapter: capstone/stacklm/agent/loop.py, doc line ~1201)
# =====================================================================
@dataclass
class Trace:
    steps: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    gen: list = field(default_factory=list)

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def add(self, ids: list[int], generated: int) -> None:
        self.ids.extend(ids)
        self.gen.extend([generated] * len(ids))


def run_agent(model, tok, question: str, env: ToolEnv,
              max_steps: int = 6, max_new: int = 160,
              temperature: float = 0.0, constrain: bool = True):
    asst_id, end_id = tok.special_token_id(ASST), tok.special_token_id(END)
    tr = Trace()
    tr.add(enc(tok, f"{SYS}{SYSTEM_PROMPT}{END}{USER}{question}{END}"), 0)
    seen_calls = set()

    for _ in range(max_steps):
        tr.add([asst_id], 0)
        new_ids, closed = generate(model, tok, tr.ids, max_new=max_new,
                                   temperature=temperature, constrain=constrain)
        tr.add(new_ids, 1)
        tr.add([end_id], 1 if closed else 0)

        step_text = tok.decode(new_ids)
        tr.steps.append(("assistant", step_text))
        act = parse_assistant_step(step_text)
        if act.kind == "final":
            return act.answer, tr

        call_key = (act.tool, str(act.args))
        if call_key in seen_calls:
            obs = "RepeatedCall: you already ran this; use the prior result."
        elif act.tool == "__malformed__":
            obs = "FormatError: emit a valid JSON tool call."
        else:
            obs = env.run_tool(act)
            seen_calls.add(call_key)
        tr.add(enc(tok, render_tool_result(obs)), 0)
        tr.steps.append(("observation", obs))

    tr.add(enc(tok, f"{ASST}Thought: I must answer now.\nAnswer:"), 0)
    new_ids, closed = generate(model, tok, tr.ids, max_new=40,
                               temperature=temperature, constrain=False)
    tr.add(new_ids, 1)
    tr.add([end_id], 1 if closed else 0)
    answer = tok.decode(new_ids).strip()
    tr.steps.append(("assistant", f"Thought: I must answer now.\nAnswer: {answer}"))
    return answer, tr


@torch.no_grad()
def generate(model, tok, prompt_ids: list[int], max_new: int,
             temperature: float = 0.0, constrain: bool = True):
    stop_id = tok.special_token_id(END)
    call_id = tok.special_token_id(TOOL_CALL)
    vocab_strings = build_vocab_strings(tok) if constrain else None
    cap = model.cfg.max_seq_len
    dev = next(model.parameters()).device

    ids = torch.tensor([prompt_ids[-cap:]], dtype=torch.long, device=dev)
    out, states, nxt = [], None, None
    for _ in range(max_new):
        logits, _ = model(ids[:, -cap:])
        logits = logits[:, -1, :].float()

        if states is not None:
            allowed, nxt = token_transitions(states, vocab_strings)
            mask = torch.full_like(logits, float("-inf"))
            mask[0, allowed] = 0.0
            if accepts(states):
                mask[0, stop_id] = 0.0
            logits = logits + mask

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            tid = int(torch.multinomial(probs, 1))
        else:
            tid = int(logits.argmax(-1))

        if tid == stop_id:
            return out, True
        out.append(tid)
        ids = torch.cat([ids, torch.tensor([[tid]], device=dev)], dim=1)

        if states is not None:
            states = nxt.get(tid)
        elif constrain and tid == call_id:
            states = start_states()
    return out, False


# =====================================================================
# Block #17 (chapter: capstone/stacklm/agent/roles.py, doc line ~1368)
# =====================================================================
def split_roles(transcript: str):
    qw, syn, prefix = [], [], ""
    for chunk in transcript.split(ASST):
        if not prefix:
            prefix = chunk
            continue
        target, rest = chunk.split(END, 1)
        role = qw if '"tool": "search"' in target else syn
        role.append((prefix + ASST, target + END))
        prefix += ASST + target + END + rest
    return qw, syn


# =====================================================================
# GLUE (not a chapter block): a scripted stand-in for the real Stack-100M
# `nn.Module`. It has zero learned weights -- it drives its next-token logits
# by consulting a `Callable[[str], str]` "plan" (we reuse the SAME stub
# teacher from block #13 as the plan, so it is honestly just "the hermetic
# teacher, replayed one character at a time through the real generate()/
# run_agent() harness code, including grammar-constrained decoding"). This
# satisfies the hard rule against instantiating a real model while still
# exercising every line of block #16's control flow (Trace bookkeeping, the
# repeated-call/malformed-call guards, generate()'s masking).
# =====================================================================
class ScriptedAgentModel:
    def __init__(self, tok, plan):
        self.tok = tok
        self.plan = plan
        self.cfg = SimpleNamespace(max_seq_len=4096)

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, ids: torch.Tensor):
        text = self.tok.decode(ids[0].tolist())
        prompt_part, sep, gen_so_far = text.rpartition(ASST)
        assert sep, "scripted model expects an open assistant turn"
        script = self.plan(prompt_part + ASST)          # what the teacher would say
        V = self.tok.vocab_size
        logits = torch.full((1, ids.shape[1], V), -50.0)
        idx = len(gen_so_far)
        if idx < len(script):
            hit = next((s for s in _SPECIALS if script.startswith(s, idx)), None)
            tid = self.tok.special_token_id(hit) if hit else self.tok._byte_id(script[idx])
        else:
            tid = self.tok.special_token_id(END)          # done -> stop
        logits[0, -1, tid] = 50.0
        return logits, torch.tensor(0.0)


def main():
    # ---- block #14: the chapter's own CI test suite, run for real ----
    test_distill_yields_correct_transcripts()
    test_special_tokens_are_single_ids_and_masking_is_nonvacuous(tok)
    test_grammar_accepts_what_we_render_and_rejects_what_we_fear()

    # ---- block #2 extra coverage: HashEmbedRetriever + calc() directly ----
    corpus = build_corpus()
    assert calc("2021*2") == "4042"
    assert calc("1/0").startswith("CalcError")
    hasher = HashEmbedRetriever(corpus[:30], dim=64)
    hits = hasher.search("hidden size Stack-100M", k=3)
    assert 0 < len(hits) <= 3
    assert all(isinstance(p, Passage) and isinstance(sc, float) for p, sc in hits)

    # ---- block #7: the schema is data; assert its shape ----
    assert set(TOOL_CALL_SCHEMA["properties"]) == {"tool", "args"}
    assert TOOL_CALL_SCHEMA["properties"]["tool"]["enum"] == ["search", "calc"]

    # ---- block #8 extra coverage: token-level mask lifting ----
    vocab_strings = build_vocab_strings(tok)
    states = start_states()
    allowed, nxt_states = token_transitions(states, vocab_strings)
    assert len(allowed) > 0
    # the byte id for '{' must be among the tokens allowed at the very start
    assert tok._byte_id("{") in allowed

    # ---- block #12/#13/#17: distill a few traces, then split_roles() ----
    train, _held = make_task_pool()
    env = ToolEnv(corpus)
    teacher = make_stub_teacher(train)
    kept = distill(train[:20], teacher, env, samples_per_task=1)
    assert len(kept) >= 15
    double_trace = next(r for r in kept if r["kind"] == "double")
    qw, syn = split_roles(double_trace["text"])
    assert len(qw) >= 1 and len(syn) >= 1
    assert all('"tool": "search"' in t for _, t in qw)
    assert all('"tool": "search"' not in t for _, t in syn)
    # roles' prefixes accumulate the FULL history verbatim
    assert qw[0][0].endswith(ASST)

    # ---- block #15 extra coverage: build_dataset() over the kept traces ----
    ds = build_dataset(kept, tok)
    assert len(ds) >= 1
    for input_ids, target in ds:
        assert input_ids.shape == target.shape
        assert input_ids.dtype == np.int64

    # ---- block #16: run_agent() driven by the scripted model, end to end ----
    lookup_task = next(t for t in make_tasks() if t.kind == "lookup"
                       and t.hints["facts"][0] == ("Stack-100M", "hidden size"))
    assert lookup_task.gold == "512"
    plan = make_stub_teacher([lookup_task])
    model = ScriptedAgentModel(tok, plan)
    answer, trace = run_agent(model, tok, lookup_task.question, env,
                              max_steps=4, max_new=200, temperature=0.0,
                              constrain=True)
    assert normalize(answer) == normalize(lookup_task.gold), (answer, trace.steps)
    # two assistant turns (search, then answer) and one observation
    assert [role for role, _ in trace.steps].count("assistant") == 2
    assert [role for role, _ in trace.steps].count("observation") == 1
    # Trace carries ids/gen of equal length, and BOTH policy (1) and
    # environment (0) tokens are present -- this is the exact invariant the
    # chapter's warning box calls "the single most common correctness bug".
    assert len(trace.ids) == len(trace.gen)
    assert set(trace.gen) == {0, 1}
    # the grammar-constrained decoder actually produced a well-formed call:
    # re-run the emitted JSON through the acceptor from block #8.
    asst_text = trace.steps[0][1]
    assert TOOL_CALL in asst_text
    json_part = asst_text.split(TOOL_CALL, 1)[1]
    st = start_states()
    for ch in json_part:
        st = step(st, ch)
        assert st, f"the model's own constrained output was rejected at {ch!r}"
    assert accepts(st)

    print("ALL 11 TARGET BLOCKS (#2,#3-skip,#5,#7,#8,#12,#13,#14,#15,#16,#17) "
          "+ corpus.py glue ran OK.")


if __name__ == "__main__":
    main()
