# 14.10 A Narrow Auto-Research Agent: ReAct, Tool-Use & Retrieval by Distillation

We now do the most ambitious thing a 100M-parameter model can honestly do: turn **Stack-100M** into a *narrow auto-research agent*. Given a question, it will interleave **thought → tool-call → observation** — searching a small local corpus, reading what it finds, doing exact arithmetic with a calculator, and finally synthesizing a short, grounded answer. This is the **ReAct** pattern (Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022), and it is the capstone of the capstone: everything we built — the tokenizer's reserved tool tokens (Ch. 14.3), the SFT loop and the narrow-RLVR machinery (Ch. 14.9) — comes together here, on top of one new artifact we build from scratch: the **corpus and the task pool** the whole pipeline consumes.

One non-negotiable truth up front, because it shapes the entire chapter. **A 100M base model will not discover multi-step tool use on its own.** It has neither the in-context-learning strength of a 70B model nor the reasoning depth to plan across turns from a few prompt examples. The only thing that works at this scale is **distillation**: let a large teacher produce many ReAct trajectories, keep only the ones that verifiably solved the task, reformat them with our tool special tokens, and **supervise-fine-tune Stack-100M to imitate the winners**. The 100M model does not learn to *reason about* tool use; it learns to *reproduce a narrow, well-worn groove* of tool use. Inside that groove it is genuinely useful. One millimeter outside it, it falls apart, and we will measure exactly where that wall is.

This chapter builds on three deeper chapters you should have open in another tab: [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html) for the loop mechanics, [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html) for the schema/parse/dispatch primitives, and [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html) for the retriever. For the training half we lean on [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html), and [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

!!! note "Aside: where this code lives, and the package contract"

    Every block below is labelled with its file under `capstone/stacklm/agent/`: `corpus.py`, `tools.py`, `react.py`, `grammar.py`, `distill.py`, `stub_teacher.py`, `sft_format.py`, `loop.py`, `rlvr.py`, plus the optional `roles.py` and `mcp_adapter.py`, and the CI test at `capstone/tests/test_agent_pipeline.py`.

    The package contract from Ch. 14.9 holds unchanged: `Stack100M.forward` **always returns `(logits, loss)`**, so every call site unpacks two values; the tokenizer is `stacklm.tokenizer.StackTokenizer`, exposing `encode(text, allowed_special=...)`, `decode(ids)` and `special_token_id(str)`; and the post-training utilities live in `stacklm.post` (`post/sft.py`, `post/grpo.py`). Every character count, yield figure, and ratio quoted in this chapter's worked examples was **measured by running these modules**, not estimated.

## Why Distillation Is the Only Path at 100M

Multi-step tool use is a *compositional* skill. To answer "What is the sum of the publication year of RoFormer and the publication year of ReAct?" the agent must (1) decide to search, (2) form a good query, (3) read the observation, (4) extract a fact, (5) decide to search *again* for a different fact, (6) call the calculator, and (7) synthesize. Each step conditions on the outcome of the previous one, so a single wrong branch — a malformed call, a query that retrieves nothing, a misread number — cascades.

Large models absorb this pattern from a handful of in-context examples because their pretraining already contains millions of implicit "act, observe, revise" structures (code with REPL sessions, forum threads, worked solutions). A 100M model pretrained on ~20B tokens of FineWeb-Edu and Cosmopedia (Ch. 14.2) has seen far less of this and has far less capacity to generalize it. Few-shot-prompt Stack-100M with three ReAct exemplars and you will observe the three classic small-model failure modes: it emits a plausible `Thought:` and then an **answer with no tool call at all** (it hallucinates the observation); or it emits a tool call and then **ignores the returned observation**; or it **loops forever** re-issuing the same search.

{{fig:distilled-groove-vs-skill}}

The fix is to make the *behavior* itself the training signal. This is knowledge distillation in the behavioral sense (see [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html)): rather than matching the teacher's soft logits token-for-token, we match its *trajectories* — thoughts, actions, and resulting observations — as ordinary SFT targets. This is **trajectory distillation**, a.k.a. rejection-sampling fine-tuning (the STaR family, Zelikman et al., 2022, and the recipe behind essentially every small open instruction model including HuggingFace's SmolLM series). The distinction matters at 100M: logit distillation needs teacher and student to share a tokenizer and be queried in lockstep, which is expensive and brittle across model families; trajectory distillation needs only the teacher's *text*, so any strong model can teach and the student trains with the plain cross-entropy loop it already has.

### The pipeline in one diagram

```text
   ┌─────────────────────────────────────────────────────────────────┐
   │ 0. WORLD BUILD       stacklm.agent.corpus:                      │
   │                      build_corpus()   -> 268 short passages     │
   │                      make_task_pool() -> 200 train + 50 held-out│
   │                                          Tasks with exact gold  │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  tasks + a ToolEnv over the corpus
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 1. TEACHER ROLLOUT   large model runs the REAL ReAct loop with  │
   │                      our two tools -> raw trajectory            │
   │                      (in CI: teacher is STUBBED/offline)        │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  trajectory + did it solve the task?
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 2. FILTER            keep only trajectories whose final answer  │
   │                      exactly matches gold (verifiable reward)   │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  successful traces
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 3. FORMAT            splice in tool special tokens, mask loss so│
   │                      the model is NOT trained on observations   │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  SFT dataset
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 4. SFT Stack-100M    imitate the groove  (Ch. 14.9 loop reused) │
   └───────────────┬─────────────────────────────────────────────────┘
                   │  optional
                   ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ 5. RLVR (GRPO)       reward = final-answer exact match; sharpen │
   │                      multi-turn: loss over GENERATED tokens only│
   └─────────────────────────────────────────────────────────────────┘
```

{{fig:distill-rejection-funnel}}

Steps 1–2 are exactly **rejection sampling against a verifiable reward** — the same reward we use in RLVR (Ch. 14.9), applied offline to *filter demonstrations* instead of online to *estimate advantages*. That symmetry is deliberate, and the final section exploits it: the SFT loss mask and the RL generated-token mask turn out to be the same mask.

## Step 0: Building the Corpus and the Task Pool

Every later stage consumes two artifacts, so we build them first and build them *programmatically*. The design constraint that makes this tractable: **generate the corpus from an explicit fact table, then generate the questions from the same table.** Gold answers are then exact by construction — no annotation step, no ambiguity for the verifier, and no risk that the "gold" answer is not actually derivable from the corpus.

The narrow world we choose is the capstone's own reference notes: the Stack-100M configuration from `capstone/PLAN.md` §1, plus the publication years of landmark components the book has already cited. To that we add a much larger set of **synthetic distractor records** — ablation-run entries with the same attribute vocabulary and different values. The distractors are the whole point of a retrieval corpus: without them, a one-word query hits the right passage every time and `search` degenerates into a lookup table.

```python
# capstone/stacklm/agent/corpus.py
"""The narrow world the auto-research agent lives in.

Two artifacts, both deterministic and offline:
  build_corpus()   -> list[Passage]   268 short passages, chunked, doc_id'd
  make_task_pool() -> (train, heldout) list[Task] with EXACT gold answers

Everything is rendered from an explicit (entity, attribute, value) fact table,
so a task's gold answer is correct BY CONSTRUCTION and is always recoverable
from the corpus. Ch. 14.11's retrieval-QA probe imports these same functions.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from stacklm.agent.tools import Passage

# ----------------------------------------------------------------------
# 1. The fact table.  (entity, attribute, value) -- values are strings so a
#    single code path handles numeric and non-numeric facts; `_is_num`
#    decides which task templates a fact is eligible for.
#    These 12 values are byte-identical to PLAN.md 1 -- do not drift.
# ----------------------------------------------------------------------
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

# Publication years of landmark works this book cites elsewhere. These are
# real, verifiable facts -- we never invent a fact to pad the corpus.
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

# Three surface forms so BM25 and the embedding backend see real lexical
# variation -- but ALL of them contain the canonical clause
# "the {attr} of {entity} is {value}", which is what makes extraction exact.
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
    """Fixed-size word chunking with overlap -- the simplest chunker that still
    respects the two rules from Ch. 9.4: (a) a chunk must be small enough that
    the answer span is not diluted by irrelevant text, (b) chunks must OVERLAP
    so a fact straddling a boundary survives in one of them. Our rendered
    passages average 28 words, so this is a no-op for them; it exists so you can
    drop real documents into the same pipeline."""
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


def build_corpus(seed: int = 0, n_distractors: int = 120) -> list[Passage]:
    """268 short passages: 28 real facts + 120 synthetic ablation records
    (2 facts each). Deterministic given `seed`."""
    rng = random.Random(seed)
    facts = list(SPEC_FACTS) + list(PAPER_FACTS)

    # Synthetic distractors: same attribute vocabulary, different entities and
    # values, so a lazy query like "hidden size" retrieves noise, not gold.
    for i in range(n_distractors):
        ent = f"ablation run R-{i:03d}"
        for attr in rng.sample(_DISTRACTOR_ATTRS, 2):
            facts.append((ent, attr, str(rng.choice([128, 192, 256, 384, 640,
                                                     768, 896, 1024, 1536]))))

    passages: list[Passage] = []
    for n, (ent, attr, val) in enumerate(facts):
        for j, piece in enumerate(chunk_words(_render(n, ent, attr, val))):
            passages.append(Passage(doc_id=f"doc-{n:04d}-{j}", text=piece))
    return passages


# ----------------------------------------------------------------------
# 2. The task pool.  Four SHAPES, deliberately: the held-out split lets
#    Ch. 14.11 measure in-shape accuracy, and holding out a whole shape
#    (Exercise 9) measures the distribution-brittleness this chapter warns
#    about.  make_tasks() yields 262 tasks: 28 + 28 + 120 + 86.
# ----------------------------------------------------------------------
@dataclass
class Task:
    question: str
    gold: str                       # the verifiable final answer (exact match)
    kind: str                       # "lookup" | "double" | "sum" | "compare"
    # `hints` is the ANSWER KEY. The real teacher never sees it; the CI stub
    # teacher does (a test double is allowed to cheat -- see below).
    hints: dict = field(default_factory=dict)


def _is_num(v: str) -> bool:
    return v.lstrip("-").isdigit()


def make_tasks(seed: int = 0) -> list[Task]:
    rng = random.Random(seed + 1)
    facts = list(SPEC_FACTS) + list(PAPER_FACTS)
    numeric = [f for f in facts if _is_num(f[2])]      # all 28, as it happens
    tasks: list[Task] = []

    for ent, attr, val in facts:                                # SHAPE 1: lookup
        tasks.append(Task(
            question=f"What is the {attr} of {ent}? Use the corpus.",
            gold=val, kind="lookup",
            hints={"queries": [f"{ent} {attr}"], "facts": [(ent, attr)], "op": None}))

    for ent, attr, val in numeric:                              # SHAPE 2: x2
        tasks.append(Task(
            question=f"What is the {attr} of {ent}, multiplied by 2? Use the corpus.",
            gold=str(int(val) * 2), kind="double",
            hints={"queries": [f"{ent} {attr}"], "facts": [(ent, attr)], "op": "double"}))

    for _ in range(120):                                        # SHAPE 3: sum
        (e1, a1, v1), (e2, a2, v2) = rng.sample(numeric, 2)
        tasks.append(Task(
            question=(f"What do you get if you add the {a1} of {e1} to the "
                      f"{a2} of {e2}? Use the corpus."),
            gold=str(int(v1) + int(v2)), kind="sum",
            hints={"queries": [f"{e1} {a1}", f"{e2} {a2}"],
                   "facts": [(e1, a1), (e2, a2)], "op": "sum"}))

    for _ in range(120):                                        # SHAPE 4: compare
        (e1, a1, v1), (e2, a2, v2) = rng.sample(numeric, 2)
        # Two rejections, both load-bearing. Equal values leave no unique gold.
        # SAME ENTITY leaves a question that is unanswerable as posed --
        # "which is larger, the hidden size of Stack-100M or the layer count of
        # Stack-100M? Answer with the entity name" has gold "Stack-100M" down
        # EITHER branch, so it is free marks for teacher and student alike and
        # silently inflates compare-shape accuracy. Stack-100M owns 12 of the
        # 28 numeric facts, so this fires on ~17% of draws -- not a corner case.
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
    """Shuffle and split. The held-out slice is NEVER shown to the teacher and
    is what Ch. 14.11 evaluates on."""
    tasks = make_tasks(seed)
    random.Random(seed + 2).shuffle(tasks)
    assert len(tasks) >= n_train + n_heldout, "pool too small; raise the counts"
    return tasks[:n_train], tasks[n_train:n_train + n_heldout]


def qa_pairs(tasks: list[Task]) -> list[dict]:
    """Exactly the shape Ch. 14.11's `eval_retrieval_qa(..., qa_pairs)` wants."""
    return [{"question": t.question, "gold_answer": t.gold} for t in tasks]
```

Three things to notice, because each is a decision you would otherwise make blind.

**Gold answers are derived, not annotated.** `int(v1) + int(v2)` cannot be wrong. This is the property that makes both the rejection-sampling filter and the RLVR reward *verifiable* (see [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html)); a corpus of human-written QA pairs would need an annotation budget and would still be ambiguous.

**The corpus is mostly distractors.** 28 real facts, 240 distractor passages. Skip the distractors and $p_{\text{good query}}$ — the term that will dominate the accuracy decomposition at the end of this chapter — is artificially ~1.0, and every number you report is a lie.

**The corpus must be excluded from the pretraining manifest.** Ch. 14.11 warns about exactly this: if these passages are also in the ~20B-token pretraining mix, the retrieval probe silently becomes a closed-book memorization test wearing an open-book costume. Because we *generate* the corpus rather than scraping it, exclusion is free — but write the assertion anyway, against the dedup index from Ch. 14.2.

!!! tip "Practitioner tip: scale the pool's *shapes*, not its size"

    You can generate 10,000 tasks in a second. Resist. The bottleneck is not task count but task **shape diversity**, and every shape you add must be matched by teacher trajectories that solve it. The honest scaling move is: add a fifth shape (a unit conversion, a three-hop lookup), regenerate the pool, re-run the teacher, and measure whether the fifth shape's held-out accuracy comes up. That loop — not more of shape 1 — is what buys generality. See [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html) for the production version.

## The Two Tools: A Calculator and a Tiny Retriever

An agent is only as good as its tools, and at 100M we keep the toolset tiny and the tool *interface* rigidly regular so the model has the smallest possible surface to memorize. Two tools:

- `calc(expr)` — evaluates an arithmetic expression exactly. Transformers are notoriously bad at multi-digit arithmetic; delegating it is a textbook win ([Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html)).
- `search(query, k)` — retrieves the top-`k` passages from the local corpus. Two backends: **BM25** (lexical) and an **embedding-lite** cosine retriever (hashed bag-of-words, self-contained, no external model). See [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html), [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html), and [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html) for the real versions.

```python
# capstone/stacklm/agent/tools.py
"""Two narrow tools for the Stack-100M auto-research agent: an exact
calculator and a small local retriever (BM25 + an embedding-lite backend).
Everything here is deterministic, offline, pure-stdlib, and CI-safe."""

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
    Errors are returned AS OBSERVATIONS (never raised) so the agent can read
    them and recover -- important for teaching self-correction."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        val = _eval_node(tree)
    except Exception as e:                          # noqa: BLE001 (catch all)
        return f"CalcError: {e}"
    if isinstance(val, float) and val.is_integer():  # 4042.0 -> "4042"
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
    """Textbook Okapi BM25 (Robertson & Zaragoza). Lexical, exact, fast, and
    shockingly hard to beat on keyword-heavy narrow corpora."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [_tokenize(p.text) for p in passages]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.docs))
        df: Counter[str] = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        N = len(self.docs)
        # BM25 idf with +0.5 smoothing; the +1.0 inside the log keeps it
        # non-negative even for terms appearing in more than half the corpus.
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
        """Top-k by BM25. NOTE: hits with score <= 0 are dropped, so this can
        return FEWER than k passages -- including zero, which the environment
        renders as the observation 'NoResults'. That is deliberate: a small
        model must be trained on what a failed search actually looks like."""
        q = _tokenize(query)
        scored = [(self.passages[i], self.score(q, i)) for i in range(len(self.docs))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, sc) for p, sc in scored[:k] if sc > 0.0]


class HashEmbedRetriever:
    """'Embedding-lite': hash each token into a fixed-width vector via the
    hashing trick (Weinberger et al., 2009), L2-normalise, cosine-rank.
    No trained encoder, no network -- a stand-in for a real dense retriever
    (Ch. 9.1) that is still meaningfully robust to paraphrase."""

    def __init__(self, passages: list[Passage], dim: int = 512):
        self.passages, self.dim = passages, dim
        self.vecs = [self._embed(p.text) for p in passages]

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for t in _tokenize(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            # Signed hashing needs the sign bit to be INDEPENDENT of the bucket,
            # or collisions stop cancelling. Taking the sign from a low bit of
            # `h` is a real bug when `dim` is a power of two: `h % dim` is
            # exactly the low bits, so every token in a bucket gets the same
            # sign and signed hashing silently degrades to plain counting.
            # Splitting h into quotient and remainder keeps the two independent.
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
```

The calculator returns errors *as strings* rather than raising. This is a deliberate agent-design choice: an observation like `CalcError: disallowed expression node` is something the model can *read* and recover from, and the teacher's successful trajectories will occasionally contain exactly such recover-then-retry sequences — precisely the robust behavior we want Stack-100M to imitate.

### The real retrieval stack (and why you should use it)

`HashEmbedRetriever` exists so CI stays hermetic. It is not what you should ship, and it would be strange to economize here: the accuracy decomposition at the end of this chapter shows the *retrieval* term dominates end-to-end accuracy, and a real encoder is the cheapest fix available. A 100–600M-parameter embedder runs comfortably on CPU next to a 100M generator — yes, the retriever may be *larger than the model it serves*, and that is the correct allocation when grounding is the load-bearing element.

As of 2026 the concrete stack is:

| Layer | What to use | Notes |
|---|---|---|
| Dense encoder | **sentence-transformers** with **EmbeddingGemma-300M** (Google, 2025), **Qwen3-Embedding-0.6B** (Alibaba, 2025), or **BGE-M3** (BAAI, 2024) | pick via the **MTEB** leaderboard *on your own retrieval slice*, not the global average |
| Vector index | **FAISS** `IndexFlatIP` for <1M vectors (exact, no tuning); **hnswlib** / **Qdrant** / **LanceDB** / **Chroma** for ANN at scale | at 268 passages exact search is instant — never reach for ANN before you need it ([Vector Databases & ANN](../09-rag-retrieval/02-vector-databases-ann.html)) |
| Lexical | **bm25s** (fast, sparse-matrix BM25) or **rank_bm25**; **Pyserini** for the research-grade Lucene path | our from-scratch BM25 is for teaching; these are for shipping |
| Reranking | a cross-encoder such as **bge-reranker-v2-m3** or **Qwen3-Reranker** | rerank the top ~20 fused candidates down to `k=2` ([Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html)) |

The drop-in replacement keeps our `search(query, k)` interface exactly, so nothing downstream changes:

```python
# capstone/stacklm/agent/retriever_dense.py  (NOT imported by CI: needs network + deps)
"""A real dense retriever with the same interface as HashEmbedRetriever.
pip install sentence-transformers faiss-cpu"""

from __future__ import annotations
from stacklm.agent.tools import Passage


class DenseRetriever:
    def __init__(self, passages: list[Passage],
                 model_name: str = "google/embeddinggemma-300m"):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.passages = passages
        self.encoder = SentenceTransformer(model_name)
        # normalize_embeddings=True + inner product == cosine similarity.
        emb = self.encoder.encode([p.text for p in passages],
                                  normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])   # exact search; 268 docs
        self.index.add(emb)

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        q = self.encoder.encode([query], normalize_embeddings=True,
                                convert_to_numpy=True).astype("float32")
        scores, idx = self.index.search(q, k)
        return [(self.passages[i], float(s))
                for i, s in zip(idx[0], scores[0]) if i >= 0]
```

On narrow, keyword-heavy corpora BM25 usually wins; on paraphrased queries the dense backend catches synonyms BM25 misses. The production move is a **hybrid** — union the two candidate sets and rerank with Reciprocal Rank Fusion, as in [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html); Exercise 5 has you implement it, and explains why we still run a *single* backend per distillation run.

## The ReAct Trace Format & Tool Special Tokens

The single most important design decision for a small agent is the **wire format** — the exact token sequence that represents a step. We keep it brutally regular. Recall the reserved special tokens from the tokenizer (Ch. 14.3): the chat roles `<|system|> <|user|> <|assistant|> <|end|>` and the tool tokens `<|tool_call|> <|tool_result|>`. A trajectory is a single chat turn whose assistant span *contains* the interleaved calls (this is a real trace, produced by the stub teacher over the corpus above):

```text
<|system|>You are a narrow research assistant. Think, then either call a
tool or give a final answer. Tools: search(query,k), calc(expr).<|end|>
<|user|>What is the publication year of RoFormer, multiplied by 2? Use the corpus.<|end|>
<|assistant|>Thought: I need the publication year of RoFormer. I'll search the corpus.
<|tool_call|>{"tool": "search", "args": {"query": "RoFormer publication year", "k": 2}}<|end|>
<|tool_result|>[1] Reference note 16. In the Stack-100M project notes, the publication year of RoFormer is 2021. ... [2] ...<|end|>
<|assistant|>Thought: Now I compute 2021*2 with the calculator.
<|tool_call|>{"tool": "calc", "args": {"expr": "2021*2"}}<|end|>
<|tool_result|>4042<|end|>
<|assistant|>Thought: I have everything I need.
Answer: 4042<|end|>
```

Three rules make this format learnable at 100M:

1. **Tool calls are JSON, always, with a fixed key order** (`tool` then `args`). One canonical schema means the model memorizes one grammar — and, as the next subsection shows, one grammar can be *enforced* rather than hoped for.
2. **Observations are wrapped in `<|tool_result|> … <|end|>` and are NOT part of the loss.** The model must *read* observations but must never be trained to *produce* them — they come from the environment. This is the same masking discipline as assistant-only SFT loss ([Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html)), extended to tool results.
3. **The final answer is delimited** by the literal `Answer: ` prefix so a verifier can extract it with a trivial regex — which is what makes the reward verifiable for both filtering and RLVR.

Here is the parser/serializer. It is deliberately a small state machine — no clever regex soup, because the harness must fail *loudly and recoverably* on malformed model output.

```python
# capstone/stacklm/agent/react.py
"""ReAct wire-format constants, (de)serialisation, and the single tokenizer
entry point. The special-token strings and the non-standard JSON spacing in
`render_call` MUST stay byte-identical across distillation, SFT, and serving
or the model's memorized groove breaks."""

from __future__ import annotations
import json, re
from dataclasses import dataclass

# The reserved special tokens (must match stacklm/tokenizer specials, Ch. 14.3).
BOS, EOS, PAD, END = "<|bos|>", "<|eos|>", "<|pad|>", "<|end|>"
SYS, USER, ASST = "<|system|>", "<|user|>", "<|assistant|>"
TOOL_CALL, TOOL_RESULT = "<|tool_call|>", "<|tool_result|>"

# Passed to tok.encode(..., allowed_special=ALL_SPECIAL) EVERYWHERE we encode a
# rendered transcript. Forgetting this is the bug the pitfall box warns about:
# without it the BPE shreds "<|tool_call|>" into a dozen ordinary tokens.
ALL_SPECIAL = frozenset({BOS, EOS, PAD, END, SYS, USER, ASST,
                         TOOL_CALL, TOOL_RESULT})

ANSWER_RE = re.compile(r"Answer:\s*(.+?)\s*$", re.DOTALL)


def enc(tok, text: str) -> list[int]:
    """The ONLY way this package tokenizes wire-format text."""
    return tok.encode(text, allowed_special=ALL_SPECIAL)


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
    # Canonical key order (tool, args) and canonical spacing -> ONE grammar for
    # the model to learn, and one grammar for the acceptor below to enforce.
    body = json.dumps({"tool": tool, "args": args}, separators=(", ", ": "))
    return f"{TOOL_CALL}{body}{END}"
```

{{fig:react-wire-format-loss-mask}}

### Constrain the grammar, don't hope for it

Format regularity in the training data gets $p_{\text{well-formed call}}$ to maybe 0.95 at 100M. **Constrained decoding gets it to 1.00 by construction, for free, at serve time.** At frontier scale this is a production nicety; at 100M it is close to mandatory, because a single stray quote costs a whole trajectory. The idea (see [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)) is to intersect the model's next-token distribution with the set of tokens that keep the output on a legal path through a grammar, setting the rest to $-\infty$ before the argmax.

Our grammar is tiny. In EBNF:

```text
call        ::= '{"tool": "' toolbody '}}'
toolbody    ::= 'search", "args": {"query": "' query '", "k": ' digit
              | 'calc", "args": {"expr": "' expr '"'
query       ::= [A-Za-z0-9 ._,()-]{1,64}
expr        ::= [0-9+\-*/.() ]{1,40}
digit       ::= [1-9]
```

In production you do not hand-roll this. You hand a JSON Schema or an EBNF/GBNF grammar to a compiled engine:

- **XGrammar** (Dong et al., 2024) — the structured-output backend used by default in both **vLLM** and **SGLang**; it precompiles the grammar into per-state token masks over a compressed vocabulary trie, so masking costs microseconds per step rather than a full vocabulary scan.
- **Outlines** (Willard & Louf, 2023) — popularized regex/JSON-Schema-guided generation by precomputing a finite-state-machine index over the vocabulary.
- **llguidance** — the Rust engine behind `guidance`, also selectable inside vLLM.
- **llama.cpp GBNF** grammars, if you are serving the int4 export from Ch. 14.11 on a laptop.

```python
# Illustrative: the tool-call schema you would hand a served model.
# vLLM has renamed this parameter across releases (guided_json ->
# guided_decoding -> structured_outputs); the semantics have been stable.
# Check the version you install.
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["search", "calc"]},
        "args": {"type": "object"},
    },
    "required": ["tool", "args"],
}
```

For our own stack we implement the mechanism, because "call a library" is not an explanation. Our grammar is a *template with holes*, which makes the acceptor short enough to read in one sitting:

```python
# capstone/stacklm/agent/grammar.py
"""A character-level acceptor for the ONE JSON shape our agent may emit, and a
token-level logit mask built from it. This is a miniature, readable version of
what XGrammar / Outlines / llguidance do inside vLLM and SGLang."""

from __future__ import annotations
from dataclasses import dataclass

_DIGITS = frozenset("123456789")
_QUERY  = frozenset("abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._,()-")
_EXPR   = frozenset("0123456789+-*/.() ")


@dataclass(frozen=True)
class Hole:
    chars: frozenset
    min_len: int
    max_len: int


# A template is a tuple of literal strings and Holes, and matches `render_call`
# byte for byte (that is what the CI test below checks). Both templates share
# the prefix '{"tool": "' and diverge at 's' vs 'c'; the acceptor tracks a SET
# of live states, so the branch costs nothing and needs no backtracking.
SEARCH_T = ('{"tool": "search", "args": {"query": "', Hole(_QUERY, 1, 64),
            '", "k": ', Hole(_DIGITS, 1, 1), '}}')
CALC_T   = ('{"tool": "calc", "args": {"expr": "', Hole(_EXPR, 1, 40), '"}}')
TEMPLATES = (SEARCH_T, CALC_T)

# A state is (template_index, item_index, chars_consumed_within_that_item).
State = tuple


def start_states() -> frozenset:
    return frozenset((ti, 0, 0) for ti in range(len(TEMPLATES)))


def _succ(state: State, ch: str):
    """Yield every state reachable from `state` by consuming character `ch`."""
    ti, ii, n = state
    tmpl = TEMPLATES[ti]
    if ii >= len(tmpl):                       # already complete: nothing follows
        return
    item = tmpl[ii]
    if isinstance(item, str):                 # literal: exactly one legal char
        if item[n] == ch:
            yield (ti, ii, n + 1) if n + 1 < len(item) else (ti, ii + 1, 0)
        return
    # a Hole: either extend it...
    if ch in item.chars and n < item.max_len:
        yield (ti, ii, n + 1)
    # ...or (if long enough) close it and hand `ch` to the next item.
    if n >= item.min_len:
        yield from _succ((ti, ii + 1, 0), ch)


def step(states: frozenset, ch: str) -> frozenset:
    out: set = set()
    for s in states:
        out.update(_succ(s, ch))
    return frozenset(out)


def accepts(states: frozenset) -> bool:
    """True iff some live state sits at the end of its template -- i.e. the
    JSON object is complete and <|end|> is now legal."""
    return any(ii >= len(TEMPLATES[ti]) for ti, ii, _ in states)


# ---- lifting the CHARACTER acceptor to a TOKEN mask ---------------------
_MASK_CACHE: dict[frozenset, tuple[list[int], dict[int, frozenset]]] = {}


def build_vocab_strings(tok) -> list[str]:
    """Decode every vocabulary id once. Special tokens decode to their literal
    '<|...|>' form, whose '<' is illegal in every grammar position, so they are
    excluded automatically -- except <|end|>, which the decoder re-admits by
    hand exactly when accepts(states) is True."""
    return [tok.decode([i]) for i in range(tok.vocab_size)]


def token_transitions(states: frozenset, vocab_strings: list[str]):
    """Return (allowed_ids, next_states). A token is allowed iff feeding its
    decoded characters one at a time leaves at least one state alive.

    Cost: O(V * avg_token_len) per DISTINCT state set -- ~10^5 char steps at
    V=32768. We memoize on the state set, which collapses the cost to near zero
    in practice because the same states recur constantly. THIS is the
    inefficiency XGrammar removes properly, by precompiling the vocabulary into
    a trie and caching per-state adaptive masks."""
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
```

Two properties worth internalizing. First, **`<|end|>` is legal only when `accepts(states)` is true** — so the grammar also supplies the halting condition, and the model physically cannot stop mid-JSON. Second, **constraint applies only after the model commits to `<|tool_call|>`**. The choice between "call a tool" and "give a final answer" is a *decision*, not a format, and must stay free; constraining it would let the grammar make the agent's decisions for it. That line — guide syntax, never override semantics — is also why constrained decoding cannot fix the query-formulation bottleneck: it guarantees the JSON is well-formed and says nothing about whether the query inside it is any good.

### The wire format is a compression of a real standard

Our compact schema is a deliberate stripped-down instance of the 2026 tool-interface standard. In **MCP** (the Model Context Protocol; see [The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html)) a server advertises tools via `tools/list`, each entry carrying a JSON Schema:

```json
{
  "name": "search",
  "description": "Search the local corpus and return the top-k passages.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Keyword query."},
      "k": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2}
    },
    "required": ["query"]
  }
}
```

The OpenAI-style `tools` array used by most chat APIs is the same content in a different envelope (`{"type": "function", "function": {"name", "description", "parameters"}}`). Both carry a JSON Schema, which is exactly what you hand XGrammar. So the adapter is short, and worth writing so your agent can be pointed at a real MCP server later:

```python
# capstone/stacklm/agent/mcp_adapter.py
"""Render an MCP tool list into (a) the terse system-prompt tool line the 100M
model was distilled on, and (b) the JSON Schemas a constrained decoder wants."""

def tools_line(mcp_tools: list[dict]) -> str:
    """[{'name': 'search', 'inputSchema': {...}}, ...] -> 'search(query,k), calc(expr)'"""
    parts = []
    for t in mcp_tools:
        props = t.get("inputSchema", {}).get("properties", {})
        parts.append(f"{t['name']}({','.join(props)})")
    return ", ".join(parts)


def schemas(mcp_tools: list[dict]) -> dict[str, dict]:
    return {t["name"]: t.get("inputSchema", {}) for t in mcp_tools}
```

Why keep the terse form for the model? **Context economy.** A full MCP/OpenAI schema for two tools is ~200 tokens of system prompt re-encoded on every step of every trajectory; the terse line is ~15. With the schema baked into the weights by distillation anyway, paying 200 tokens per step to restate what the model already memorized is pure waste. The adapter lets the *harness* speak the standard while the *model* speaks the compressed dialect it was trained on — standards at the boundary, compression on the wire.

!!! note "Aside: JSON actions vs. code actions"

    Our loop is a stripped-down `ToolCallingAgent` in **smolagents** terms — the model emits a JSON action. The alternative, smolagents' `CodeAgent`, has the model emit *Python* that calls the tools, which composes beautifully (loops, conditionals, variable reuse in one action) and measurably reduces step counts for capable models. It is the wrong choice at 100M: code actions have an unbounded grammar, so $p_{\text{well-formed}}$ collapses, and constraining "valid Python that only calls these two functions" is far harder than constraining one JSON template. **LangGraph** (explicit state machines over agent nodes) and **LlamaIndex** workflows are the other frameworks this loop instantiates; both are the right answer once the model is large enough that the harness, not the format, is your problem. We hand-roll because at 100M **the wire format *is* the training target** — the harness and the dataset are the same artifact, and no framework will generate your SFT data for you.

## Distilling Trajectories From the Teacher

Now the heart of the chapter: generate trajectories by running the *real ReAct loop* with a large teacher, keep the winners, turn them into SFT data.

In the full run you point the rollout at a strong instruction model — anything in the "solidly agentic" tier — prompted with the tool schemas and a couple of gold exemplars, driving the *same* environment (`calc`, `search`) our agent will use. A single teacher rollout looks like this (real API sketch; never run in CI):

```python
# Illustrative ONLY -- the real teacher call. In CI this path is never taken.
import anthropic  # or any strong-model client

def teacher_step(client, messages, tools_desc):
    resp = client.messages.create(
        model="a-strong-agentic-model",       # the large teacher
        system=tools_desc,                    # tool schemas + 2 gold exemplars
        messages=messages,
        max_tokens=512, temperature=0.7,      # some temperature -> diverse traces
    )
    return resp.content[0].text
```

Temperature > 0 matters: we want *diverse* trajectories per task so that after filtering we retain several distinct successful solution shapes, not one. That is rejection sampling — sample many, keep the good — and its yield is a number you should actually measure.

!!! tip "Practitioner tip: batch the rollouts, don't for-loop them"

    `distill()` below runs $200 \times 8 = 1600$ multi-step rollouts in a plain Python loop, which is fine for a hosted teacher behind an HTTP API (use the provider's batch endpoint and a thread pool) and *terrible* if you self-host. Self-hosted, serve the teacher with **vLLM** or **SGLang** and run the *whole task pool in lockstep*: at each ReAct step submit all live transcripts as one batch, execute the returned tool calls, resubmit. Two properties make this dramatically faster than it sounds: continuous batching keeps the GPU saturated across trajectories of different lengths ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)), and every step re-sends a transcript sharing a long prefix with the previous step — precisely what **prefix caching** / SGLang's **RadixAttention** eliminate ([Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html), [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html)). The same machinery is what you reuse for the GRPO group rollouts later; see [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).

```python
# capstone/stacklm/agent/distill.py
"""Generate ReAct trajectories, filter to successes, export SFT data.
The teacher is an INJECTED callable (prompt -> one assistant step) so CI can
pass a hermetic stub while the full run passes a real large-model client. No
network import lives anywhere on this path."""

from __future__ import annotations
from stacklm.agent.react import (
    Action, parse_assistant_step, render_tool_result,
    SYS, USER, ASST, END,
)
from stacklm.agent.tools import calc, BM25Retriever, Passage
from stacklm.agent.corpus import Task          # Task lives with the pool


class ToolEnv:
    """The environment: dispatch a parsed Action to a real tool."""

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
    """Verifier-side normalisation so '4042', ' 4042 ', '4042.0' all match."""
    if s is None:
        return ""
    s = s.strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return " ".join(s.lower().split())


def rollout(task: Task, teacher, env: ToolEnv, max_steps: int = 6):
    """Run ONE ReAct trajectory with `teacher`. Returns (transcript, solved).
    The transcript is the full wire format including observations."""
    transcript = f"{SYS}{SYSTEM_PROMPT}{END}{USER}{task.question}{END}"
    solved = False
    for _ in range(max_steps):
        gen = teacher(transcript + ASST)           # teacher emits one step
        step_text = gen.split(END, 1)[0]           # up to the first <|end|>
        act = parse_assistant_step(step_text)
        transcript += f"{ASST}{step_text}{END}"    # record it verbatim
        if act.kind == "final":
            solved = (normalize(act.answer) == normalize(task.gold))
            break
        obs = env.run_tool(act)                    # REAL tool call
        transcript += render_tool_result(obs)      # inject observation
    return transcript, solved


def distill(tasks, teacher, env, samples_per_task: int = 4):
    """Rejection sampling: multiple rollouts per task, keep only solved ones,
    deduplicated per task."""
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
```

### A hermetic stub teacher for CI

The stub is a **test double**, and a test double is allowed to read the answer key. It receives the task list, looks each question up by exact string, and follows a fixed policy driven by `task.hints`. A real teacher sees only the question. Being explicit about this is the point: the stub's job is to prove the *plumbing* is correct, never to prove that anything is *learnable*.

```python
# capstone/stacklm/agent/stub_teacher.py
"""A deterministic 'teacher' that solves the generated auto-research tasks by a
fixed policy driven by Task.hints (the answer key). It emits exactly the wire
format a real large model would emit, but offline and in microseconds.

It is a TEST DOUBLE: it cheats by construction. Its only contract is that every
downstream code path -- dispatch, observation rendering, the success filter,
the SFT exporter -- is exercised end to end on every CI run."""

from __future__ import annotations
import re
from stacklm.agent.react import TOOL_RESULT, END, USER, render_call


def _observations(prompt: str) -> list[str]:
    """Every tool result seen so far, in order."""
    return [chunk.split(END, 1)[0]
            for chunk in prompt.split(TOOL_RESULT)[1:]]


def _extract(observations: list[str], ent: str, attr: str) -> str | None:
    """Pull a value out of an observation using the corpus's canonical clause
    'the {attr} of {ent} is {value}'. Returns None if the search missed --
    which is exactly what makes the rejection filter non-trivial.

    Note the character class EXCLUDES the comma: one of the three passage
    templates continues the sentence with ', a choice made during...', and a
    greedy '[^.]+' would swallow that whole clause into the 'value'. Verifiers
    fail silently in exactly this way; write the test that catches it."""
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
        task = by_q[question]                      # the cheat: the answer key
        obs = _observations(prompt)
        queries, facts = task.hints["queries"], task.hints["facts"]
        op = task.hints["op"]

        # PHASE 1 -- one search per required fact, in order.
        if len(obs) < len(queries):
            i = len(obs)
            return (f"Thought: I need the {facts[i][1]} of {facts[i][0]}. "
                    f"I'll search the corpus.\n"
                    f"{render_call('search', {'query': queries[i], 'k': 2})}")

        vals = [_extract(obs, e, a) for e, a in facts]
        if any(v is None for v in vals):           # retrieval genuinely failed
            return "Thought: The corpus did not give me the fact.\nAnswer: unknown"

        # PHASE 2 -- one calculator call, if the task shape needs arithmetic.
        n_calc = len(obs) - len(queries)           # observations past the searches
        if n_calc == 0 and op is not None:
            if op == "double":
                expr = f"{vals[0]}*2"
            elif op == "sum":
                expr = f"{vals[0]}+{vals[1]}"
            else:                                  # "compare": sign of difference
                expr = f"{vals[0]}-{vals[1]}"
            return (f"Thought: Now I compute {expr} with the calculator.\n"
                    f"{render_call('calc', {'expr': expr})}")

        # PHASE 3 -- synthesize.
        if op is None:
            answer = vals[0]
        elif op == "compare":
            diff = obs[-1].strip()
            names = task.hints["names"]
            answer = names[0] if not diff.startswith("-") else names[1]
        else:
            answer = obs[-1].strip()               # the calculator's output
        return f"Thought: I have everything I need.\nAnswer: {answer}"

    return teacher
```

Trace the `double` shape once: step 1 sees zero observations and emits `search("RoFormer publication year", k=2)`; the environment returns the reference note; step 2 sees one observation, extracts `2021`, emits `calc("2021*2")`; the environment returns `4042`; step 3 sees two observations, `op` is not `None` and `n_calc == 1`, so it answers `4042`. `normalize("4042") == normalize(task.gold)` and the trace is kept. Run over all 200 training tasks the stub keeps **200/200**, spread across all four shapes.

And now the assertion the section implicitly promised — the CI test that makes "hermetic, exercises every code path" a *checked* claim rather than a hope:

```python
# capstone/tests/test_agent_pipeline.py
"""Hermetic, CPU-only, no network. Runs in a couple of seconds.
`tok` is the session-scoped toy StackTokenizer fixture from conftest.py."""

from stacklm.agent.corpus import build_corpus, make_task_pool
from stacklm.agent.distill import ToolEnv, distill, normalize
from stacklm.agent.stub_teacher import make_stub_teacher
from stacklm.agent.sft_format import build_example, _segment, IGNORE
from stacklm.agent.react import TOOL_CALL, TOOL_RESULT, END, render_call, enc
from stacklm.agent import grammar as G


def test_distill_yields_correct_transcripts():
    corpus = build_corpus()
    train, held = make_task_pool()
    assert len(corpus) == 268 and len(train) == 200 and len(held) == 50
    env = ToolEnv(corpus)
    teacher = make_stub_teacher(train)
    kept = distill(train[:40], teacher, env, samples_per_task=2)

    # (1) The pipeline produces data at all -- the bug this test exists for.
    #     The stub reads the answer key, so its yield is ~100%; a real teacher
    #     lands nearer 0.6. Assert on the YIELD, not merely on "no exception":
    #     a stub that silently solves nothing returns [] and every downstream
    #     stage becomes a no-op while CI stays green.
    assert len(kept) >= 35, f"teacher yield collapsed: {len(kept)}/40"
    # (2) Every kept transcript really ends in the gold answer.
    gold = {t.question: t.gold for t in train}
    for row in kept:
        final = row["text"].rsplit("Answer:", 1)[1].split(END, 1)[0]
        assert normalize(final) == normalize(gold[row["task"]])
    # (3) All four task shapes survive the filter (no shape is silently lost).
    assert {r["kind"] for r in kept} == {"lookup", "double", "sum", "compare"}


def test_special_tokens_are_single_ids_and_masking_is_nonvacuous(tok):
    # The pitfall box's assertion, as an actual test.
    for s in (TOOL_CALL, TOOL_RESULT, END):
        assert len(enc(tok, s)) == 1, f"{s} is not a single special id"
    corpus = build_corpus()
    train, _ = make_task_pool()
    env, teacher = ToolEnv(corpus), make_stub_teacher(train)
    kept = distill(train[:4], teacher, env, samples_per_task=1)
    # _segment must PARTITION the transcript -- no dropped and no duplicated
    # spans. The bug this catches: a tail span appended twice when a transcript
    # does not end in a marker, which silently doubles those tokens in the SFT
    # example AND in the RL generated-token mask.
    segs = _segment(kept[0]["text"])
    assert "".join(s for s, _ in segs) == kept[0]["text"]
    ids, labels = build_example(kept[0]["text"], tok)
    n_sup = int((labels != IGNORE).sum())
    assert 0 < n_sup < len(labels), "loss mask is empty or masks nothing"


def test_grammar_accepts_what_we_render_and_rejects_what_we_fear():
    # Every search query the task pool can produce, plus both calc shapes.
    train, _ = make_task_pool()
    bodies = [render_call("calc", {"expr": "2021*2"}),
              render_call("calc", {"expr": "2019+10000"})]
    bodies += [render_call("search", {"query": q, "k": 2})
               for t in train for q in t.hints["queries"]]
    for body in bodies:
        json_part = body.split(TOOL_CALL, 1)[1].split(END, 1)[0]
        st = G.start_states()
        for ch in json_part:
            st = G.step(st, ch)
            assert st, f"grammar rejected our own output at {ch!r}"
        assert G.accepts(st)
    # A single dropped quote must be rejected, and rejected EARLY.
    st = G.start_states()
    for ch in '{"tool": search':
        st = G.step(st, ch)
    assert not st
```

Test (1) is the one that matters most: a stub whose policy silently fails produces `kept == []`, every downstream stage becomes a no-op, and the pipeline "passes" CI while doing nothing. Assert on the *yield*, not on the absence of exceptions.

!!! example "Worked example: rejection-sampling yield and token budget"

    Take **200 training questions**, **`samples_per_task = 8`** teacher rollouts each, and a strong teacher that solves this narrow family with per-rollout success probability on the order of **0.6** (the failures are mostly retrieval misses on the distractor-heavy corpus, not reasoning errors).

    Expected raw successes: $200 \times 8 \times 0.6 = 960$. After deduping — successful traces for one task typically collapse to 2–3 distinct shapes — keep on the order of $200 \times 3 = 600$ unique traces.

    Now the **token budget**, measured by running `distill()` over the real pool. Across all 200 kept traces the mean transcript is **1474 characters**, of which **468** (**31.8%**) sit inside supervised assistant spans. Decomposing the 3-step `double` shape specifically:

    | Component | Count | Mean chars each | Subtotal |
    |---|---|---|---|
    | system + question preamble | 1 | 237 | 237 |
    | assistant emissions (thought + call/answer) | 3 | 107 | 321 |
    | `k=2` observations | 2 | 179 | 358 |
    | special-token markers (3 assistant opens/closes, 2 tool-result opens/closes) | 10 | ~10 | 104 |
    | **total** | | | **1020** |

    At the byte-level BPE's roughly **4 characters/token** on this text, the pool mean of 1474 characters is about **370 tokens/trace**, of which about **120** are loss-bearing — a supervised fraction near **0.32**, because two-thirds of every trace is text the environment wrote. So 600 traces is $600 \times 370 \approx 222{,}000$ tokens total and about **72k supervised tokens**.

    (Characters are the tokenizer-independent quantity, which is why we report them; the *token* fraction shifts with your vocabulary, because tool-call JSON and bare integers tokenize less efficiently than prose. Measure it once on your own SFT set — the diagnostic below depends on your measured baseline, not on ours.)

    72k supervised tokens is *tiny* — a few minutes of fine-tuning. The scarce resource is not compute; it is **successful, diverse trajectories**. At a teacher cost on the order of \$1–3 per thousand rollouts, generating 1600 rollouts costs a few dollars, and the dominant line item is the teacher, not the SFT.

    One caveat the CI stub makes vivid: the stub is *deterministic*, so all 8 rollouts per task are byte-identical and dedup collapses them to **one** trace per task. Diversity comes from the teacher's temperature, not from the sampling loop. Run a real teacher at `temperature=0` and you get the same collapse, having paid 8× for 1/8 the data.

## SFT on the Traces: Teaching Stack-100M the Groove

With a folder of successful transcripts, training is just SFT (Ch. 14.9) with one twist: the loss mask must cover **prompt tokens, tool-call JSON we want to keep, AND tool-result spans we must not learn**. Concretely, the model is supervised on the assistant's thoughts, its tool calls, and its final answer — but not on the system prompt, the user question, or anything inside `<|tool_result|> … <|end|>`.

```python
# capstone/stacklm/agent/sft_format.py
"""Turn a distilled transcript into (input_ids, labels) with the correct loss
mask: supervise ASSISTANT emissions (thoughts, tool CALLS, final answer) only;
mask the system/user prompt and every tool RESULT span."""

from __future__ import annotations
import numpy as np
from stacklm.agent.react import ASST, END, TOOL_RESULT, USER, SYS, enc

IGNORE = -100   # standard CrossEntropyLoss ignore_index (matches Ch. 14.9)


def _segment(t: str):
    """Split a transcript into (text, supervised) spans that PARTITION it
    exactly -- concatenating the pieces must reproduce `t` byte for byte.
    Assistant content between <|assistant|> and the NEXT role/result marker is
    supervised; everything else (system, user, and <|tool_result|>...<|end|>)
    is masked. <|tool_call|> is deliberately NOT a segmentation marker, so tool
    calls stay INSIDE the supervised span -- we very much want them emitted.

    This function is load-bearing twice over: it defines the SFT loss mask
    here, and it defines the RL generated-token mask in `rlvr.py`."""
    out, i, supervised = [], 0, False
    markers = [ASST, USER, SYS, TOOL_RESULT, END]
    while i < len(t):
        j, m = min([(t.find(x, i), x) for x in markers if t.find(x, i) != -1],
                   default=(len(t), None), key=lambda z: z[0])
        if j > i:
            out.append((t[i:j], supervised))
        if m is None:
            break            # trailing text, already appended -- do NOT re-add
        if m == ASST:
            supervised = True                 # begin supervising assistant span
            out.append((m, False))            # the marker token itself: masked
        elif m in (USER, SYS, TOOL_RESULT):
            supervised = False                # stop supervising
            out.append((m, False))
        else:                                 # END
            out.append((m, supervised))       # <|end|> IS supervised if in asst
            supervised = False                # ...then close the span
        i = j + len(m)
    return out


def build_example(transcript: str, tok, max_len: int = 1024):
    """-> (input_ids, target) or None if the transcript does not fit.

    `tok` is the Stack-100M byte-level BPE tokenizer (Ch. 14.3). We encode
    through `enc`, which passes allowed_special so every '<|...|>' marker maps
    to its single reserved id instead of being shredded into byte pieces.

    We DROP over-length transcripts rather than truncating them. `ids[:max_len]`
    looks harmless and is not: it cuts the tail off the longest, most
    compositional traces -- precisely the ones whose `Answer:` span lives at the
    end -- so the model is trained on trajectories that never terminate, and it
    learns not to terminate. Log the drop rate; if it is non-trivial, raise
    `max_len` (mid-training gave us 8192 tokens of context) rather than quietly
    mutilating your best data."""
    ids, labels = [], []
    for text, supervised in _segment(transcript):
        piece = enc(tok, text)
        ids.extend(piece)
        labels.extend(piece if supervised else [IGNORE] * len(piece))
    if len(ids) > max_len:
        return None
    # Teacher forcing: predict token t+1 from tokens <= t, so shift the labels.
    input_ids = np.array(ids[:-1], dtype=np.int64)
    target    = np.array(labels[1:], dtype=np.int64)
    return input_ids, target


def build_dataset(kept: list[dict], tok, max_len: int = 1024):
    """Export the whole distilled set, reporting what was dropped."""
    out = [ex for ex in (build_example(r["text"], tok, max_len) for r in kept)
           if ex is not None]
    dropped = len(kept) - len(out)
    if dropped:
        print(f"[sft_format] dropped {dropped}/{len(kept)} over-length traces "
              f"(max_len={max_len}); raise max_len if this is more than a few %")
    return out
```

Two subtleties that matter enormously in practice:

- **The `<|end|>` that closes an assistant turn is supervised.** The model must learn to *emit* the stop token, or at inference it will run past its answer into garbage. But the `<|end|>` that closes a `<|tool_result|>` is masked — the environment writes that one.
- **Do not distill the teacher's raw chain-of-thought verbatim if it is long or off-format.** For a 100M student, terse thoughts distill better than sprawling ones; a light pass that trims each `Thought:` to one sentence measurably improves imitation. This mirrors the general finding that *shorter, cleaner rationales transfer better to small students* (see [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html)).

Training reuses the Ch. 14.9 SFT loop verbatim — same `stacklm.post.sft` entry point, same AdamW optimizer, same bf16 autocast. The only new thing is the dataset builder above. That reuse is the point of a coherent capstone: the agent is not a new training system, it is a new *dataset* for the training system you already built.

## The Auto-Research Loop at Inference

At serving time there is no teacher. Stack-100M drives the loop itself: generate until it emits `<|tool_call|>`-terminated JSON or `<|end|>`, run the tool if asked, splice the observation back in, and continue — with strict guards, because a small model *will* try to derail.

One structural decision deserves stating before the code. `run_agent` returns, alongside the human-readable step list, **the exact token ids of the whole trajectory and a per-token flag saying whether the policy or the environment produced each one.** It never hands back only text. The reason is in the warning box below, and it is the single most common correctness bug in agentic RL implementations.

```python
# capstone/stacklm/agent/loop.py
"""The runtime auto-research loop: Stack-100M interleaves thought, tool calls,
and observations, then synthesizes a final grounded answer. Guards against the
known small-model failure modes, and carries token ids so the RL stage never
has to re-tokenize a rollout."""

from __future__ import annotations
from dataclasses import dataclass, field
import torch
from stacklm.agent.react import (parse_assistant_step, render_tool_result,
                                 SYS, USER, ASST, END, TOOL_CALL, enc)
from stacklm.agent.distill import ToolEnv, SYSTEM_PROMPT
from stacklm.agent import grammar as G


@dataclass
class Trace:
    """A finished rollout, in two views.

    `steps` is for humans, for logging, and for Ch. 14.11's failure taxonomy;
    it iterates as (role, text) pairs so `for role, text in trace` works.
    `ids` / `gen` are for RL: the exact token sequence the forward pass must
    see, and a 0/1 flag per token that is 1 iff the POLICY sampled it."""
    steps: list[tuple[str, str]] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)
    gen: list[int] = field(default_factory=list)

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
    """-> (answer, Trace).

    temperature=0.0 -> greedy (serve time). temperature>0 -> sampling, which is
    what the GRPO rollouts below REQUIRE: a greedy policy in a deterministic
    environment produces G identical trajectories, a zero-variance group, and
    therefore a zero gradient."""
    asst_id, end_id = tok.special_token_id(ASST), tok.special_token_id(END)
    tr = Trace()
    tr.add(enc(tok, f"{SYS}{SYSTEM_PROMPT}{END}{USER}{question}{END}"), 0)
    seen_calls = set()                                # loop guard

    for _ in range(max_steps):
        tr.add([asst_id], 0)                          # the HARNESS opens the turn
        new_ids, closed = generate(model, tok, tr.ids, max_new=max_new,
                                   temperature=temperature, constrain=constrain)
        tr.add(new_ids, 1)                            # <- the policy's own tokens
        # The harness always closes the turn so the wire format stays legal, but
        # the token counts as OURS only if the policy chose to emit it. If we
        # hit max_new, flagging it 1 would reinforce a token nobody sampled.
        tr.add([end_id], 1 if closed else 0)

        step_text = tok.decode(new_ids)
        tr.steps.append(("assistant", step_text))
        act = parse_assistant_step(step_text)
        if act.kind == "final":
            return act.answer, tr

        # --- GUARDS ---------------------------------------------------
        call_key = (act.tool, str(act.args))
        if call_key in seen_calls:                    # exact repeat -> break loop
            obs = "RepeatedCall: you already ran this; use the prior result."
        elif act.tool == "__malformed__":             # bad JSON -> teach recovery
            obs = "FormatError: emit a valid JSON tool call."
        else:
            obs = env.run_tool(act)
            seen_calls.add(call_key)
        # --------------------------------------------------------------
        tr.add(enc(tok, render_tool_result(obs)), 0)  # the ENVIRONMENT wrote this
        tr.steps.append(("observation", obs))

    # Out of steps: force one synthesis attempt. The forcing text is the
    # HARNESS's, not the policy's, so it is flagged 0 -- otherwise RL would
    # reinforce a string the model never chose to say.
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
    """Decode ONE assistant emission. -> (sampled_ids, closed_with_end).

    Takes and returns IDS, never text: the caller keeps them verbatim.

    `constrain=True` turns on grammar-constrained decoding for the tool-call
    JSON: the constraint activates the moment the model commits to <|tool_call|>
    and releases when the JSON is complete. The choice between calling a tool
    and answering stays FREE -- we constrain syntax, never the decision."""
    stop_id = tok.special_token_id(END)
    call_id = tok.special_token_id(TOOL_CALL)
    vocab_strings = G.build_vocab_strings(tok) if constrain else None
    cap = model.cfg.max_seq_len                       # the RoPE cache length
    dev = next(model.parameters()).device

    ids = torch.tensor([prompt_ids[-cap:]], dtype=torch.long, device=dev)
    out, states, nxt = [], None, None
    for _ in range(max_new):
        logits, _ = model(ids[:, -cap:])              # Stack100M -> (logits, loss)
        logits = logits[:, -1, :].float()             # [1, vocab]

        if states is not None:                        # inside a tool call
            allowed, nxt = G.token_transitions(states, vocab_strings)
            mask = torch.full_like(logits, float("-inf"))
            mask[0, allowed] = 0.0
            if G.accepts(states):                     # JSON complete -> may stop
                mask[0, stop_id] = 0.0
            logits = logits + mask

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            tid = int(torch.multinomial(probs, 1))
        else:
            tid = int(logits.argmax(-1))              # greedy

        if tid == stop_id:                            # the model closed the step
            return out, True
        out.append(tid)
        ids = torch.cat([ids, torch.tensor([[tid]], device=dev)], dim=1)

        if states is not None:
            states = nxt.get(tid)                     # advance the grammar
        elif constrain and tid == call_id:
            states = G.start_states()                 # commit -> constrain
    return out, False                                 # ran out of budget
```

The three guards — **repeated-call detection**, **malformed-call recovery**, and **forced synthesis on step exhaustion** — are not garnish. They are the difference between a demo that runs and a demo that hangs. A 100M model *will* re-issue the same search; the guard converts an infinite loop into an observation the model has been trained (via the recover-then-retry traces) to respond to. Same defensive-harness philosophy as [Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html), scaled to what a tiny model needs.

!!! warning "Common pitfall: two ways the ids stop matching the training groove"

    **(1) Special-token drift.** The number-one silent failure is a mismatch between the special-token strings/ids used in distillation and those used at inference. If your SFT data wrote `<|tool_call|>` but your runtime tokenizer maps it to a different id — or splits it into sub-tokens because you forgot `allowed_special=ALL_SPECIAL`, or never registered it as a *special* token in Ch. 14.3 — the model has learned a groove it can never re-enter. This is why every encode in this package goes through the single `enc()` helper, and why `test_special_tokens_are_single_ids_...` asserts `len(enc(tok, "<|tool_call|>")) == 1`.

    **(2) Round-tripping a rollout through text.** It is tempting to have `run_agent` return only strings, reassemble a transcript, and re-encode it for the RL stage. Do not. **BPE detokenize→retokenize is not the identity** — whitespace at span joins, digit grouping, and the boundary between a `Thought: …\n` and the following `<|tool_call|>` can all re-merge differently. The importance ratio $\rho = \pi_\theta / \pi_{\theta_{\text{old}}}$ would then be evaluated on a token sequence the policy never emitted, so the gradient points at the wrong sequence and the whole update is quietly meaningless. **Never round-trip a rollout through text before computing log-probs — carry the ids.** That is exactly what `Trace.ids` / `Trace.gen` exist for. (You still need `_segment` for SFT, where teacher *text* is genuinely all you have; the two mask definitions agree, but their provenance differs and only one of them is safe for RL.)

!!! note "Aside: greedy at serve, sampled during RL"

    We decode the agent greedily (`temperature=0.0`) at serve time. For a chatbot you often *want* sampling for diversity; for a tool-using agent you want **determinism and format discipline**. Every extra bit of entropy is another chance to drift the JSON key order or hallucinate an observation.

    During **RL rollouts you must do the opposite**. GRPO's advantage is group-relative — it subtracts the group mean. A greedy policy in a deterministic environment emits $G$ byte-identical trajectories with $G$ identical rewards, so the group standard deviation is zero, every advantage is zero, and no gradient flows: the degenerate case of Exercise 6, manufactured by your own decoder rather than by a weak policy. Sample the rollouts (`temperature≈0.7`, matching `sample_group` in `stacklm/post/grpo.py` from Ch. 14.9), keep the guards on, and switch back to greedy for evaluation and serving. See [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html).

### Two roles, one model: the narrow multi-agent variant

The accuracy decomposition below shows that **query formulation** is the binding constraint. That suggests an architectural response rather than a training one: split the agent into two *roles* and specialize each.

- A **query writer** sees the question and the observations so far, and emits only a `search` call. Its whole job is turning "the publication year of RoFormer" into a query that beats 240 distractor passages.
- A **reader/synthesizer** sees the question and the retrieved passages, and emits either a `calc` call or the final `Answer:`.

This is the smallest honest multi-agent system ([Multi-Agent Systems & Orchestration](../08-agents-harness/07-multi-agent-systems.html)), and it costs almost nothing because the distillation data already splits cleanly: every teacher trace decomposes into query-writer examples (prefix → `search` call) and synthesizer examples (prefix + observations → `calc`/`Answer:`). Over our 200 kept traces that is 356 query-writer and 375 synthesizer examples. Train two LoRA adapters (Ch. 5.3) over the same frozen 100M base and serve both from one process — the multi-tenant adapter pattern in [Multi-Tenant LoRA & Adapter Serving at Scale](../07-inference-serving/14-multi-tenant-lora-serving.html) — so the "multi-agent system" costs one model's weights plus two ~2MB adapters.

```python
# capstone/stacklm/agent/roles.py
"""Split one distilled trace into role-specific SFT sets, so each role's
adapter sees ONLY the decisions it will be asked to make at serve time."""

from stacklm.agent.react import ASST, END


def split_roles(transcript: str):
    """-> (query_writer_examples, synthesizer_examples), each a list of
    (prefix, target) pairs in the same wire format. The prefix accumulates the
    FULL history including observations, so each role still conditions on
    everything that happened -- only the target differs."""
    qw, syn, prefix = [], [], ""
    for chunk in transcript.split(ASST):
        if not prefix:                       # the system+user preamble
            prefix = chunk
            continue
        target, rest = chunk.split(END, 1)   # rest = observation, if any
        role = qw if '"tool": "search"' in target else syn
        role.append((prefix + ASST, target + END))
        prefix += ASST + target + END + rest
    return qw, syn
```

Be honest about the tradeoff: two roles means two chances to derail, two adapters to keep in sync with the wire format, and a harder debugging story. The win is that each adapter's training distribution is *narrower*, which is the one thing that reliably helps at 100M. Measure it against the single-agent baseline on the held-out split before you believe it.

## Optional: RLVR on Tool-Use Success

Distillation gets Stack-100M *onto* the groove; a small dose of **RLVR** (Ch. 14.9) can sharpen it — nudging the policy toward trajectories that actually solve the task rather than merely *look* like solutions. The reward could not be simpler: run the agent loop, extract the final answer, compare to gold.

$$
R(\tau) = \mathbf{1}\!\left[\operatorname{normalize}(\text{answer}(\tau)) = \operatorname{normalize}(\text{gold})\right] \;-\; \lambda \cdot \frac{\text{steps}(\tau)}{\text{max\_steps}}
$$

The first term is the verifiable correctness reward; the small step penalty $\lambda$ (on the order of 0.05) discourages needless tool calls. This is *exactly* the RLVR reward of Ch. 14.9 applied to a whole multi-turn trajectory instead of a single answer — the agent is the policy, a full ReAct rollout is the completion, and the environment lives *inside* the rollout.

### What changes when the rollout is multi-turn

This is the genuinely new machinery, and where multi-turn agentic RL differs from the single-turn GRPO of Ch. 14.9 ([Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html)). A trajectory $\tau_i$ is a single token sequence that *interleaves two authors*: tokens the policy generated, and observation tokens the **environment** wrote. Define the **generated-token mask**

$$
m_{i,t} = \mathbf{1}\!\left[\text{token } t \text{ of trajectory } i \text{ was emitted by } \pi_\theta\right],
$$

and note three consequences:

1. **Observation tokens must be in the forward pass** — they condition every subsequent policy decision, so removing them would change the distribution we are differentiating.
2. **Observation tokens must be out of the loss.** They have no $\pi_\theta$ to take a ratio of; including them would train the model to predict the environment, which is both wrong and the fastest route to hallucinated observations.
3. **The normalization changes the answer.** Averaging per token over the *whole batch* (the "Dr. GRPO" correction, Liu et al., 2025) weights each generated token equally; averaging per sequence then over sequences weights each *trajectory* equally, over-weighting short ones. Because our trajectories differ in both length and observation share, this is not cosmetic — Exercise 8 works the numbers.

The objective is the standard clipped surrogate, restricted to the mask:

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_{i,t} m_{i,t}} \sum_{i=1}^{G} \sum_{t} m_{i,t}\,
\min\!\Big(\rho_{i,t}\hat{A}_i,\ \operatorname{clip}\!\big(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon\big)\hat{A}_i\Big)
\;+\; \beta\,\frac{1}{\sum_{i,t} m_{i,t}} \sum_{i,t} m_{i,t}\, \mathbb{KL}_{i,t},
$$

with $\rho_{i,t} = \pi_\theta(o_{i,t}\mid o_{i,<t}) / \pi_{\theta_{\text{old}}}(o_{i,t}\mid o_{i,<t})$ and the group-relative advantage $\hat A_i = (R_i - \bar R)/(\operatorname{std}(R) + \epsilon)$ shared by every token of trajectory $i$.

And here is the unification worth carrying away: **$m$ is the SFT loss mask.** The tokens we supervise in distillation are exactly the tokens the policy generates. Same *definition*, different *provenance*: in SFT all we have is teacher-written text, so `_segment` recovers the mask from the markers; in RL we are the ones sampling, so `run_agent` records the mask as it goes and we never re-tokenize. One concept, two stages, no chance of them drifting apart — and no round-trip.

```python
# capstone/stacklm/agent/rlvr.py
"""Narrow multi-turn RLVR with GRPO. One 'rollout' = one full ReAct trajectory;
reward = final-answer exact match minus a step penalty. The policy-gradient
loss covers GENERATED tokens only -- observations condition the forward pass
but never enter the loss."""

from __future__ import annotations
import numpy as np, torch
from stacklm.agent.loop import run_agent
from stacklm.agent.distill import normalize
from stacklm.agent.react import PAD
from stacklm.post.grpo import token_logprobs          # Ch. 14.9, unchanged


def trajectory_reward(answer, gold, n_steps, max_steps, lam=0.05):
    correct = 1.0 if normalize(answer) == normalize(gold) else 0.0
    return correct - lam * (n_steps / max_steps)


def grpo_advantages(rewards):
    r = np.asarray(rewards, dtype=np.float64)
    # group-relative: subtract the group mean, scale by the group std (GRPO)
    return (r - r.mean()) / (r.std() + 1e-6)


def grpo_agent_step(policy, ref, opt, tok, env, task, *, group_size=8,
                    max_steps=6, temperature=0.7, clip_eps=0.2, kl_beta=0.02,
                    lam=0.05, device="cuda"):
    """ONE GRPO iteration on ONE task. Returns a stats dict."""
    # ---- 1. rollouts. SAMPLED, not greedy (see the aside above). ----------
    traces, rewards = [], []
    for _ in range(group_size):
        answer, tr = run_agent(policy, tok, task.question, env,
                               max_steps=max_steps, temperature=temperature)
        n_steps = sum(1 for role, _ in tr if role == "assistant")
        rewards.append(trajectory_reward(answer, task.gold, n_steps, max_steps, lam))
        traces.append(tr)

    adv_np = grpo_advantages(rewards)
    if float(np.abs(adv_np).max()) < 1e-6:
        # Degenerate group: every rollout scored the same. No gradient exists;
        # skipping is strictly better than taking a zero step (and it is the
        # signal to fix your task difficulty, not your learning rate).
        return {"reward": float(np.mean(rewards)), "skipped": True}

    # ---- 2. pad the rollouts. NO re-tokenization: the ids came from the
    #         sampler, and `tr.gen` came with them. ------------------------
    G, T = group_size, max(len(tr.ids) for tr in traces)
    pad_id = tok.special_token_id(PAD)
    seqs  = torch.full((G, T), pad_id, dtype=torch.long, device=device)
    gmask = torch.zeros((G, T), dtype=torch.float32, device=device)
    pmask = torch.zeros((G, T), dtype=torch.float32, device=device)  # real, non-pad
    for i, tr in enumerate(traces):
        n = len(tr.ids)
        seqs[i, :n]  = torch.tensor(tr.ids, dtype=torch.long, device=device)
        gmask[i, :n] = torch.tensor(tr.gen, dtype=torch.float32, device=device)
        pmask[i, :n] = 1.0
    # Right-padding is exact under CAUSAL attention: no real token can attend
    # to a pad, and every pad position is masked out of the loss below.
    adv = torch.tensor(adv_np, dtype=torch.float32, device=device)

    # ---- 3. old log-probs: the policy that GENERATED these rollouts -------
    with torch.no_grad():
        old_lp = token_logprobs(policy, seqs)          # (G, T-1)
        ref_lp = token_logprobs(ref, seqs)             # frozen KL anchor
    m    = gmask[:, 1:]                                # align to prediction targets
    real = pmask[:, 1:]                                # non-pad positions

    # ---- 4. one clipped-surrogate step over GENERATED tokens only --------
    new_lp = token_logprobs(policy, seqs)
    ratio  = torch.exp(new_lp - old_lp)                # == 1 on the first step
    a      = adv.unsqueeze(1)                          # (G,1) broadcasts over t
    surr   = torch.min(ratio * a,
                       torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    logr   = ref_lp - new_lp                           # k3 KL estimator (Schulman)
    kl     = torch.exp(logr) - logr - 1.0
    # Token-mean over the WHOLE batch: every generated token counts once,
    # regardless of which trajectory it came from (Dr. GRPO length correction).
    denom  = m.sum().clamp(min=1.0)
    loss   = -((surr - kl_beta * kl) * m).sum() / denom

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    return {"reward": float(np.mean(rewards)), "loss": float(loss),
            # `real_tokens`, NOT m.numel(): the latter counts right-padding, so
            # the ratio below would move with trajectory-length variance rather
            # than with the thing we want to monitor.
            "gen_tokens": int(m.sum()), "real_tokens": int(real.sum()),
            "skipped": False}
```

Three implementation notes that are easy to get wrong.

On the **first** gradient step $\rho \equiv 1$ (new = old), so the clip is inert and the update is plain group-relative REINFORCE; the clip only bites if you take multiple inner epochs on the same rollouts, as `grpo_train` does in Ch. 14.9.

The **`gen_tokens / real_tokens` ratio is a metric you should log**. Measure it once on your own SFT set and alert on drift from *that* baseline — on our traces the supervised share is 0.32 of characters and lands in the same neighbourhood in tokens, and the qualitative claim is what matters: it must sit well below 1.0 and stay stable. A drift toward 1.0 means the mask has broken and you are training on observations. Note the denominator: dividing by `m.numel()` would count right-padding, so the ratio would drop whenever trajectory lengths spread out — a diagnostic that fires for reasons unrelated to masking is worse than no diagnostic.

And **the `skipped` counter is a curriculum signal**: a run where most groups are degenerate is telling you the task pool is uniformly too easy or too hard for the current policy, which you fix by re-mixing task shapes ([RL Data, Curriculum & Replay Management](../06-rl-infra/12-rl-data-curriculum-replay.html)), not by tuning the optimizer.

!!! tip "Practitioner tip: at any real scale, don't write this loop"

    Ours is synchronous, single-GPU, and regenerates the KV cache from scratch every step — exactly right for a 100M model where a rollout costs milliseconds, and exactly wrong above ~1B. The 2026 production stack for multi-turn agentic RL is **veRL** (multi-turn rollouts with tool calling over an SGLang/vLLM rollout engine; [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html)), **TRL**'s `GRPOTrainer` with vLLM-backed generation and custom reward functions ([TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html)), or **OpenRLHF** / Ray-based systems ([OpenRLHF, NeMo-Aligner & Ray-Based Systems](../06-rl-infra/05-openrlhf-nemo-ray.html)). What they buy is the thing our loop most conspicuously lacks: an asynchronous, weight-synchronized generation engine so the GPU is not idle while Python runs BM25. Write ours once to understand the masking; then use theirs.

Be honest about what this buys at 100M. RLVR here is a **polish pass, not a capability creator**. On a narrow verifiable family where the SFT model already succeeds on the order of 55–65% of the time, a short GRPO run can lift that modestly and — often more valuably — *shorten* trajectories and cut malformed calls. It will **not** teach a tool-use skill absent from the distilled traces. The zero-signal problem is brutal at small scale: if the SFT policy almost never solves a task, every trajectory in the group gets the same reward, all advantages are ~0, and there is no gradient. Distill first, RL second — never the reverse.

## The Ceiling: Brutally Honest About What 100M Can Do

**What genuinely works.** Inside the narrow task family the traces cover — "search the local corpus for one or two facts, maybe do one arithmetic step, answer" — Stack-100M is a real, functioning ReAct agent. It emits well-formed tool calls (with constrained decoding, *always* well-formed), reads observations, chains two or three steps, and returns grounded answers at a rate far above what few-shot prompting the base model achieves (near zero for multi-step). Retrieval grounding is the load-bearing element: the model does not need to *know* facts, only to *find and copy* them, which is exactly what a small model can do reliably.

**Where the wall is.**

- **Distribution-brittleness.** Ask a question whose *shape* differs from the distilled traces — three retrieval hops instead of two, a unit conversion the calculator traces never showed — and the model produces confident, malformed, or hallucinated steps. It learned a groove, not a skill. This is the defining limitation, and no amount of RLVR on the *same* narrow tasks fixes it; you must distill the new shapes. Our four-shape pool makes this measurable rather than rhetorical: hold out a whole shape (Exercise 9) and watch accuracy fall off a cliff.
- **No robust error recovery beyond what was demonstrated.** It recovers from the exact error patterns present in training (a `CalcError`, a `NoResults`) because those patterns appeared in successful traces. Novel failure modes derail it.
- **Query formulation is weak.** The single biggest source of end-to-end failure is a bad search query that retrieves nothing relevant; every downstream step then has nothing to stand on. A larger model writes better queries. This bounds the whole system, and it is the one term constrained decoding cannot touch.
- **Horizon — and it is *not* context that caps it.** Worth stating precisely, because the intuitive answer is wrong. At ~120 tokens per step, the 8192-token mid-training context (Ch. 14.8) would hold something like 60 steps. The real caps are: (a) `max_steps=6`, a harness policy you can raise with one argument; (b) `build_example(max_len=1024)`, which means the model has *never seen* a trajectory longer than roughly eight steps, so beyond that it is off-distribution and degrades regardless of context; and (c) the distilled trace-length distribution itself, which is 2–4 steps. Context becomes binding only if you raise `k` or feed whole documents instead of 40-word passages. Long-horizon agency is out of reach for **distributional**, not architectural, reasons — which is also the good news, because distributions are something you can change with data.

!!! example "Worked example: where the accuracy goes"

    Decompose end-to-end accuracy on the narrow eval into a product of per-stage success rates (a mental model; illustrative magnitudes):

    $$
    \text{Acc}_{\text{e2e}} \approx p_{\text{well-formed call}} \cdot p_{\text{good query}} \cdot p_{\text{reads obs}} \cdot p_{\text{correct calc}} \cdot p_{\text{clean synth}}
    $$

    With on-the-order-of values after distillation and **unconstrained** decoding — $0.95 \times 0.75 \times 0.90 \times 0.98 \times 0.90 \approx 0.57$ — you land near **57%**. Turn on grammar-constrained decoding and the first term becomes $1.00$ *by construction*: $0.75 \times 0.90 \times 0.98 \times 0.90 \approx 0.60$, i.e. ~**60%** for a two-line change and no extra training.

    But look where it is still stuck: the **0.75 query term** now dominates outright. Arithmetic (0.98) and format (1.00) are solved *because we offloaded them* — arithmetic to a tool, format to a grammar. The bottleneck is the one genuinely cognitive step, writing a good query, which is precisely what a 100M model is worst at. So the honest headline is: **tool use and constrained decoding offload exactly what small models fail at; the residual failure is the reasoning we could not offload.** Make the corpus small and keyword-rich, use a real dense retriever so a mediocre query still lands, or split off a specialized query-writer role — those raise $p_{\text{good query}}$, and they are the highest-leverage knobs you have.

{{fig:acc-decomposition-query-bottleneck}}

**How you would benchmark this for real.** Our held-out 50 tasks are a micro-eval, and you should say so. The public agentic benchmarks measuring the same capabilities are **BFCL** (the Berkeley Function-Calling Leaderboard: single/multiple/parallel calls, irrelevance detection, multi-turn), **τ-bench** (Yao et al., 2024: multi-turn tool use against a simulated user and a database, with a pass^k reliability metric), and **ToolBench**/ToolLLM for API breadth. Stack-100M will score near the floor on all of them, and running it against one is still worth an afternoon: it converts "narrow" from an adjective into a number. See [Agent Evaluation & Benchmarks](../08-agents-harness/08-agent-evaluation.html) and [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html).

This is the frontier of what 100M can honestly do, and it is a satisfying place to end: not a chatbot oracle, but a *narrow, grounded, tool-using research assistant* you trained end-to-end for the cost of a nice dinner. Ch. 14.11 evaluates it honestly — its retrieval-QA probe imports `build_corpus()`, wraps it in the same `BM25Retriever`, and grades against `qa_pairs(heldout)` from this chapter — and Ch. 14.12 lays out what to change to break through this ceiling on the road to 1B.

!!! interview "Interview Corner"

    **Q:** You want a 100M model to do multi-step tool use. Why not just few-shot prompt it with ReAct exemplars like you would a large model, and why does distillation-then-SFT work when that fails? And once it works, what actually changes when you move from single-turn RLVR to RL on the multi-turn agent?

    **A:** Few-shot ReAct relies on strong in-context learning and latent planning: the model has to *induce* the act-observe-revise procedure from a couple of examples and *generalize* it to a new question, holding the whole loop in working memory across turns. A 100M model has neither the ICL strength nor the reasoning depth, so it degrades into the canonical failure modes — hallucinating observations, ignoring returned results, or looping. Distillation changes the learning problem from "induce and generalize a procedure at inference time" to "reproduce a demonstrated behavior seen thousands of times in training." We generate trajectories with a strong teacher, **filter to the ones that verifiably solved the task** (rejection sampling against an exact-match reward), reformat with tool special tokens, and SFT with the loss masked to assistant tokens only. Note this is *trajectory* distillation — we imitate the teacher's text, not its logits — so the teacher can be any strong model and need not share our tokenizer.

    Moving to multi-turn RL changes three concrete things. First, the trajectory has **two authors**: the environment writes the observation spans, so the importance ratio and the clipped surrogate must be restricted to a *generated-token mask* — which is the same mask SFT uses, just recorded during sampling instead of recovered from text. Second, observations still have to be in the **forward pass**, because they condition every later decision; you exclude them from the loss, not from the context. Third, **normalization matters**: token-mean over the whole batch versus per-sequence-then-mean weights long and short trajectories differently, and agent trajectories vary a lot in length.

    Two traps worth volunteering. Your rollouts must be *sampled*, not greedy — a deterministic policy in a deterministic environment gives you G identical trajectories, zero group variance, and zero gradient. And you must carry the sampled token ids through to the loss: reassembling the transcript as text and re-encoding it is not the identity under BPE, so you would compute $\rho$ on a sequence the policy never emitted. The overall catch: the model learns the *distribution of demonstrated trajectory shapes*, not a general skill — so it is brittle to shapes absent from the distilled data, and RLVR can only sharpen what SFT already reaches.

!!! key "Key Takeaways"

    - **Build the world first.** Generate the corpus *and* the task pool from one explicit fact table (`stacklm/agent/corpus.py`), so gold answers are exact by construction, the reward is verifiable, and the corpus is trivially excludable from the pretraining manifest. Most of it should be distractors, or every number you report is inflated — and reject ill-posed task draws (same entity on both sides of a `compare`) before they inflate your accuracy for free.
    - At 100M, **distillation is the only path to multi-step tool use**: generate ReAct trajectories with a large teacher, keep the verifiably-successful ones (rejection sampling), reformat with tool special tokens, and SFT the small model to imitate them. Prefer **trajectory over logit distillation** — matching the teacher's *text* needs no shared tokenizer or lockstep querying.
    - Keep the **tool interface brutally regular** — one canonical JSON schema, `<|tool_call|>`/`<|tool_result|>`, `Answer:` — then **enforce it with constrained decoding** (XGrammar/Outlines/llguidance in vLLM or SGLang; the from-scratch acceptor in `grammar.py`). That drives the format term to 1.0 for free — but constrain *syntax*, never the tool-vs-answer decision.
    - **Mask the loss** to assistant emissions only: thoughts, tool calls, and the final answer including its closing `<|end|>`; never the prompt or a tool result. **The same mask is the RL generated-token mask** — one definition, two stages. **Drop** over-length examples rather than truncating them, or you train the model never to terminate.
    - **Carry the token ids out of the rollout.** Detokenize→retokenize is not the identity, so an RL loss computed on a re-encoded transcript is computed on a sequence the policy never emitted.
    - The scarce resource is **diverse successful trajectories**, not compute — our whole SFT set is ~72k supervised tokens. Spend the budget on the teacher, batch its rollouts through vLLM/SGLang with prefix caching, and never run it at temperature 0.
    - Build **defensive runtime guards** — repeated-call detection, malformed-call recovery, forced synthesis on step exhaustion — decode greedily at serve time but **sample during RL rollouts**, and assert that special-token ids are identical across distill, SFT, and serve.
    - **Multi-turn GRPO** differs from single-turn in three ways: loss over generated tokens only, observations in the forward pass but out of the loss, and a normalization choice (token-mean vs. sequence-mean) that reweights trajectories of different length.
    - Tool use and grammar constraints **offload exactly what small models fail at**; the residual bottleneck is query formulation — the one cognitive step you cannot offload — which is why a real embedder or a specialized query-writer role is the highest-leverage upgrade. What caps the horizon is the *distribution* of distilled trace lengths, not the context window. Be honest about the result: a **narrow, grounded, scaffolding-shaped research assistant**, useful inside its distilled groove and brittle one step outside it. That is the genuine frontier of 100M.

!!! sota "State of the Art & Resources (2026)"
    Trajectory distillation into small tool-using agents has gone from a research curiosity (2023) to a standard recipe with dedicated frameworks and papers explicitly targeting sub-3B students — the narrow-agent approach this chapter builds is mainstream practice, not a toy simplification. Two things changed the engineering around it: **grammar-constrained decoding** became a default feature of the serving stack rather than an add-on, and **multi-turn agentic RL** got first-class support in the open RL frameworks.

    **Foundational work**

    - [Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)](https://arxiv.org/abs/2210.03629) — the interleaved thought/action/observation loop this chapter implements.
    - [Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023)](https://arxiv.org/abs/2302.04761) — self-supervised tool-call insertion, a complementary route to tool use.
    - [Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning* (2022)](https://arxiv.org/abs/2203.14465) — rejection-sampling fine-tuning on self-generated successful traces, the pattern behind the filter step.

    **Recent advances (2023–2026)**

    - [Chen et al., *FireAct: Toward Language Agent Fine-tuning* (2023)](https://arxiv.org/abs/2310.05915) — fine-tuning on diverse GPT-4-generated ReAct trajectories substantially outperforms few-shot prompting and improves robustness to noisy tool observations.
    - [Kang et al., *Distilling LLM Agent into Small Models with Retrieval and Code Tools* (2025)](https://arxiv.org/abs/2505.17612) — transfers full task-solving trajectories (not just chain-of-thought) into 0.5B–3B students with retrieval and code tools, matching much larger CoT-distilled baselines.
    - [Belcak et al., *Small Language Models are the Future of Agentic AI* (NVIDIA, 2025)](https://arxiv.org/abs/2506.02153) — a position paper arguing narrow, repetitive agent nodes are exactly where small specialized models belong, with a strong LLM reserved for planning.
    - [Willard & Louf, *Efficient Guided Generation for Large Language Models* (Outlines, 2023)](https://arxiv.org/abs/2307.09702) and [Dong et al., *XGrammar* (2024)](https://arxiv.org/abs/2411.15100) — the two lineages behind the constrained-decoding section; XGrammar's compressed vocabulary trie is what makes per-token masking essentially free.
    - [Yao et al., *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains* (2024)](https://arxiv.org/abs/2406.12045) — multi-turn tool use against a simulated user, with a pass^k reliability metric; the public benchmark closest in spirit to our held-out split.

    **Open-source & tools**

    - [huggingface/smolagents](https://github.com/huggingface/smolagents) — a barebones agent library with both `ToolCallingAgent` (JSON actions, what we build) and `CodeAgent` (Python actions); the clearest reference for the wire-format tradeoff.
    - [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar), [dottxt-ai/outlines](https://github.com/dottxt-ai/outlines), [guidance-ai/llguidance](https://github.com/guidance-ai/llguidance) — the constrained-decoding engines; XGrammar is the default structured-output backend in both vLLM and SGLang.
    - [vllm-project/vllm](https://github.com/vllm-project/vllm) and [sgl-project/sglang](https://github.com/sgl-project/sglang) — batch the teacher rollouts and the GRPO groups here; SGLang's RadixAttention is purpose-built for the growing-prefix pattern of agent loops.
    - [modelcontextprotocol](https://github.com/modelcontextprotocol) — MCP servers/SDKs; the standard our compact wire format is a deliberate compression of.
    - [huggingface/trl](https://github.com/huggingface/trl) and [volcengine/verl](https://github.com/volcengine/verl) — production `GRPOTrainer` and multi-turn/tool-calling agentic RL over a vLLM/SGLang rollout engine.
    - [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers), [facebookresearch/faiss](https://github.com/facebookresearch/faiss), [xhluca/bm25s](https://github.com/xhluca/bm25s) — the real retrieval stack that replaces our teaching implementations one class at a time.
    - [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) (Berkeley Function-Calling Leaderboard) and [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) — the public agentic benchmarks to place your narrow agent against.
    - [Nardien/agent-distillation](https://github.com/Nardien/agent-distillation) — official code for the Kang et al. small-agent-distillation paper above: trajectory logging, SFT training, and benchmarking for distilled agents.

    **Go deeper**

    - [Anthropic, *Building Effective Agents* (2024)](https://www.anthropic.com/engineering/building-effective-agents) — practitioner guidance that simple, composable agent patterns beat complex frameworks, echoing this chapter's "keep the interface brutally regular" design choice.

## Further reading

- Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao, *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022 — the loop this chapter implements.
- Schick, Dwivedi-Yu, Dessì et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, 2023 — self-supervised tool-call insertion; a complementary path to tool use.
- Zelikman, Wu, Mu & Goodman, *STaR: Bootstrapping Reasoning With Reasoning*, 2022 — rejection-sampling fine-tuning on self-generated successful traces, the pattern behind our filter step.
- Shao, Wang, Zhu et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning* (GRPO), 2024 — the critic-free RL algorithm for the optional RLVR polish.
- Liu, Zhu, Wang et al., *Understanding R1-Zero-Like Training* ("Dr. GRPO"), 2025 — the length/normalization corrections behind our token-mean loss aggregation.
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (Outlines), 2023, and Dong et al., *XGrammar*, 2024 — grammar-constrained decoding, the mechanism `stacklm/agent/grammar.py` reimplements in miniature.
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009 — the lexical retriever.
- Weinberger, Dasgupta, Langford, Smola & Attenberg, *Feature Hashing for Large Scale Multitask Learning*, 2009 — the hashing trick behind the embedding-lite retriever.
- Reimers & Gurevych, *Sentence-BERT*, 2019, and Muennighoff, Tazi, Magne & Reimers, *MTEB: Massive Text Embedding Benchmark*, 2022 — how to pick and use the real encoder that replaces it.
- Lewis, Perez, Piktus et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020 — the RAG foundation the retriever tool sits on.
- Kwon, Li, Zhuang et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), 2023 — the rollout engine you batch the teacher through.
- Allal, Lozhkov, Bakouch et al., *SmolLM* (HuggingFace), 2024–2025 — the small-model recipe (data + distillation) whose spirit this capstone follows.
- Karpathy, *nanoGPT* / *llm.c*, 2024 — the "reproduce a real model on a budget" spirit this whole capstone updates.

## Exercises

**1.** In the loss-mask design, the `<|end|>` token that closes an *assistant* turn is supervised, but the `<|end|>` that closes a `<|tool_result|>` span is masked. Explain the reasoning behind this asymmetry, and describe the concrete failure you would observe at inference time if you accidentally masked *both* kinds of `<|end|>`. Then explain the analogous distinction `run_agent` makes when it appends `<|end|>` after a step that hit `max_new`.

??? note "Solution"
    The two `<|end|>` tokens are written by different actors and therefore have different learning requirements.

    - The `<|end|>` closing an assistant turn is *emitted by the model*. At serve time `generate` stops at `<|end|>` and hands control back to the harness to run a tool or return the answer. If the model never learned to *produce* this token, it will not know when to stop; it will run past its answer or past its tool call into garbage and the harness will never regain control at the right boundary. So this token must be *in the loss*.
    - The `<|end|>` closing a `<|tool_result|>` is *written by the environment*. The model must learn to *read* observations (to condition on them) but must never be trained to *produce* them, or it will start hallucinating observations the environment never returned. The entire `<|tool_result|> ... <|end|>` span, including its closing `<|end|>`, is masked with `IGNORE = -100`.

    This is exactly what `_segment` encodes: `END` is appended with `supervised` equal to the current span's flag (`out.append((m, supervised))`), so an `<|end|>` inside an assistant span is supervised and one inside a masked tool-result span is not.

    If you masked *both*, the model would never be supervised to emit its own stop token. At inference, greedy decoding would not reliably produce `<|end|>` after the answer or the tool call, so `generate` would decode until `max_new`, spilling the answer into trailing garbage and breaking the "stop at `<|end|>` after each step" contract the loop depends on. The symptom is a model that "worked in training" but at serve time never cleanly terminates a step. (Grammar-constrained decoding partially rescues the *tool-call* case — the acceptor re-admits `<|end|>` exactly when `accepts(states)` is true — but does nothing for the final `Answer:` turn, which is unconstrained by design.)

    The RL-side analogue: when a step hits `max_new` without the model emitting `<|end|>`, `run_agent` still appends the token so the wire format stays parseable, but flags it `gen = 0`. The flag answers "did $\pi_\theta$ choose this token?", and here it did not — the harness did. Flagging it 1 would put a term in the policy-gradient loss for an action the policy never took, reinforcing (or, under a negative advantage, punishing) a decision that was never made.

**2.** The chapter insists that few-shot prompting a 100M base model with ReAct exemplars fails, while distillation-then-SFT works. Name the three canonical small-model failure modes the chapter lists for few-shot ReAct, and explain in one or two sentences *why distillation changes the learning problem* so those modes stop dominating.

??? note "Solution"
    The three canonical failure modes:

    1. **Hallucinating the observation** — it emits a plausible `Thought:` and then an answer with *no tool call at all*, inventing the tool's output instead of calling it.
    2. **Ignoring the returned observation** — it emits a tool call but then does not condition on what came back.
    3. **Looping forever** — it re-issues the same search over and over without progressing.

    Why distillation reframes it: few-shot ReAct asks the model to *induce* the act-observe-revise procedure from a couple of in-context examples and *generalize* it to a new question, holding the whole multi-turn loop in working memory. A 100M model has neither the in-context-learning strength nor the reasoning depth for that. Distillation changes the task to "reproduce a demonstrated behavior seen thousands of times during training." The model imitates a narrow, well-worn groove presented as ordinary SFT targets rather than reasoning from scratch, so the failure modes above — which are failures of *induction and generalization* — no longer dominate inside the distilled distribution. The catch: it learns the distribution of demonstrated trajectory shapes, not a general skill, so it stays brittle one step outside that groove.

    Note the runtime guards attack the *residual* of each mode rather than its cause: `RepeatedCall` converts mode 3 from a hang into an observation, and forced synthesis bounds the damage from mode 1.

**3.** (Quantitative.) You run the distillation pipeline with a task pool of **150 questions**, **`samples_per_task = 6`** teacher rollouts each, and a per-rollout success probability of **0.5**. Assume that after dedup, successful traces collapse to about **2 distinct shapes per task**. Using the chapter's measured per-trace figures (~370 tokens/trace, of which ~120 are loss-bearing assistant tokens), estimate: (a) the expected number of raw solved trajectories, (b) the number of unique traces kept, (c) the total and supervised token counts of the SFT set, and (d) at a teacher cost of \$1–3 per thousand rollouts, the dollar cost of teacher rollouts. State the one-line lesson the numbers reinforce, and say what changes if the teacher is run at temperature 0.

??? note "Solution"
    (a) Expected raw successes: $150 \times 6 \times 0.5 = 450$ solved trajectories.

    (b) Unique traces after dedup: $150 \times 2 = 300$ traces.

    (c) Token counts, using the chapter's measured per-trace figures:
    - Total: $300 \times 370 = 111{,}000$ tokens.
    - Supervised (loss-bearing): $300 \times 120 = 36{,}000$ tokens.

    The ratio checks out against the measured supervised fraction: $36{,}000 / 111{,}000 \approx 0.32$.

    (d) Rollout count: $150 \times 6 = 900$. At \$1–3 per thousand rollouts that is $0.9 \times (\$1\text{ to }\$3) \approx \$0.90$ to \$2.70 — a couple of dollars.

    Lesson: the SFT set is tiny (tens of thousands of supervised tokens = a few minutes of fine-tuning), so at 100M the scarce resource is not compute but **diverse successful trajectories**. Spend the budget on the teacher, not on gradient steps.

    At temperature 0 the teacher is deterministic, so all 6 rollouts per task are byte-identical: dedup collapses to **1 trace per task** (150 traces, ~18k supervised tokens). You paid for 900 rollouts to obtain 150 — 6× the money for half the data. This is exactly what the hermetic stub teacher demonstrates in CI, and it is why `temperature=0.7` is not a stylistic choice.

**4.** (Quantitative.) Using the chapter's per-stage decomposition
$$
\text{Acc}_{\text{e2e}} \approx p_{\text{well-formed call}} \cdot p_{\text{good query}} \cdot p_{\text{reads obs}} \cdot p_{\text{correct calc}} \cdot p_{\text{clean synth}},
$$
suppose after distillation you measure $p_{\text{well-formed call}} = 0.95$, $p_{\text{good query}} = 0.60$, $p_{\text{reads obs}} = 0.90$, $p_{\text{correct calc}} = 0.98$, $p_{\text{clean synth}} = 0.90$. (a) Compute the end-to-end accuracy. (b) Identify the bottleneck stage. (c) You can either narrow the corpus to raise $p_{\text{good query}}$ to $0.85$, or raise $p_{\text{well-formed call}}$ to a perfect $1.00$. Compute the resulting accuracy for each option. (d) Given what the chapter says about constrained decoding, why is framing these as an either/or the wrong framing — and what is the correct engineering conclusion?

??? note "Solution"
    (a) $0.95 \times 0.60 = 0.570$; $\times 0.90 = 0.513$; $\times 0.98 = 0.50274$; $\times 0.90 \approx 0.4525$. So $\text{Acc}_{\text{e2e}} \approx \mathbf{45\%}$.

    (b) The bottleneck is $p_{\text{good query}} = 0.60$ — by far the smallest factor, the one dragging the product down.

    (c) Option A (narrow corpus, query $0.60 \to 0.85$): $0.95 \times 0.85 = 0.8075$; $\times 0.90 = 0.72675$; $\times 0.98 = 0.712215$; $\times 0.90 \approx 0.6410$, i.e. **~64%**.

    Option B (perfect format, $0.95 \to 1.00$): multiplies the original 45% by $1.00/0.95 \approx 1.053$, giving $0.4525 \times 1.0526 \approx 0.4763$, i.e. **~48%**.

    (d) Option B is **not an expenditure of effort at all** — grammar-constrained decoding sets $p_{\text{well-formed call}} = 1.00$ by construction, and enabling it is a decode-time flag (`constrain=True`), not a training project. So the correct conclusion is: *take option B for free, then spend all your actual effort on option A.* Doing both gives $1.00 \times 0.85 \times 0.90 \times 0.98 \times 0.90 \approx 0.6747$, i.e. **~67%**.

    This sharpens the chapter's thesis rather than contradicting it. Once arithmetic is offloaded to a tool and format to a grammar, both terms are pinned near 1.0 and *cannot* be the bottleneck; what remains is the one genuinely cognitive step, which is precisely what a 100M model is worst at and what no external machinery can do on its behalf.

**5.** (Implementation.) The production retrieval move is a *hybrid* — union the BM25 and hashed-embedding candidate sets and rerank. Implement a `HybridRetriever` in the style of `stacklm/agent/tools.py` that wraps a `BM25Retriever` and a `HashEmbedRetriever`, exposes the same `search(query, k) -> list[tuple[Passage, float]]` interface, and fuses the two candidate lists using **Reciprocal Rank Fusion (RRF)** so you do not have to reconcile BM25's and cosine's incompatible score scales. Then state one reason we still default to a *single* backend per distillation run.

??? note "Solution"
    RRF assigns each document a score $\sum_{\text{backends}} \frac{1}{c + \text{rank}}$ (rank 0-indexed), which depends only on *rank position*, not raw score magnitude — so BM25's unbounded scores and cosine's $[-1, 1]$ scores combine cleanly. A document ranked high in either list, or moderately in both, floats to the top.

    ```python
    # capstone/stacklm/agent/tools.py  (append)
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
            # Pull a wider candidate pool from each backend so fusion can
            # promote a doc that is mid-ranked in both lists.
            pool = 2 * k
            fused: dict[str, list] = {}     # doc_id -> [Passage, rrf_score]
            for retr in (self.bm25, self.embed):
                for rank, (p, _) in enumerate(retr.search(query, pool)):
                    slot = fused.setdefault(p.doc_id, [p, 0.0])
                    slot[1] += 1.0 / (self.c + rank)
            ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
            return [(p, sc) for p, sc in ranked[:k]]
    ```

    Sanity check: a passage ranked 0 by BM25 and 1 by the embedder with $c = 60$ scores $\frac{1}{60} + \frac{1}{61} \approx 0.0330$, beating a passage that is only rank 0 in BM25 and absent from the embedder list ($\frac{1}{60} \approx 0.0167$). Documents both backends like win.

    Swapping `HashEmbedRetriever` for `DenseRetriever` (sentence-transformers + FAISS) changes nothing above — that is the point of pinning the `search(query, k)` interface.

    Why still one backend per distillation run: fusing two backends produces observations with more *format variety* (different passage sets, orders, phrasings), and the whole distillation strategy depends on keeping the observation format brutally regular so the tiny model memorizes one grammar rather than many. Expose both so you can *ablate* which distills better; ship one per run.

**6.** (Conceptual + quantitative.) The chapter warns that RLVR dies from zero-signal groups if SFT does not already solve the task sometimes. Consider a GRPO group of $G = 8$ trajectories for a single task where the SFT policy currently solves this task essentially never, so all 8 rollouts return the wrong answer, each using the full `max_steps`. Using $R(\tau) = \mathbf{1}[\text{correct}] - \lambda \cdot \frac{\text{steps}}{\text{max\_steps}}$ with $\lambda = 0.05$ and the `grpo_advantages` function, compute the reward of each trajectory and the resulting advantages, and explain why no learning happens. State the chapter's ordering rule and why it is not reversible. Finally, name a *second*, purely mechanical way to produce this same degenerate group even when the policy is perfectly capable.

??? note "Solution"
    Each trajectory is wrong, so $\mathbf{1}[\text{correct}] = 0$, and each uses the full `max_steps`, so $\frac{\text{steps}}{\text{max\_steps}} = 1$. Every trajectory therefore has reward
    $$
    R(\tau) = 0 - 0.05 \times 1 = -0.05.
    $$

    All 8 rewards are identical. In `grpo_advantages` the group mean is $-0.05$ and the standard deviation is $0$, so each advantage is
    $$
    \frac{-0.05 - (-0.05)}{0 + 10^{-6}} = 0.
    $$

    GRPO scales each trajectory's log-prob gradient by its advantage, so a group of all-zero advantages contributes **no gradient signal** — nothing to push the policy toward, because no trajectory did better or worse than its peers. This is the zero-*variance* problem: group-relative advantages require within-group spread, which requires the policy to *sometimes* succeed. `grpo_agent_step` detects exactly this (`np.abs(adv).max() < 1e-6`) and returns `skipped: True` rather than taking a zero step.

    The ordering rule: **distill first, RL second — never the reverse.** It is not reversible because RLVR can only *sharpen* behavior the SFT groove already reaches sometimes; it cannot *create* an absent capability. If SFT never solves the task, every group is degenerate and RL has no foothold.

    The second, mechanical cause: **greedy rollouts**. With `temperature=0.0` the policy is deterministic and our environment is deterministic, so all $G$ trajectories are byte-identical, all rewards are equal, and the group is degenerate *regardless of how good the policy is*. The symptom is indistinguishable from the capability failure above, which is what makes it nasty; the fix is one argument. The diagnostic that separates them: with sampling on, a capability failure gives $G$ *distinct* trajectories that all score $-0.05$, whereas a decoder bug gives $G$ *identical* strings. Log the number of unique trajectories per group and you will never confuse them again.

**7.** (Conceptual.) Walk the acceptor in `grammar.py` by hand. (a) Starting from `start_states()`, feed the characters `{"tool": "c` one at a time — which template indices remain alive after each character, and at which character does the branch resolve? (b) The decoder re-admits `<|end|>` only when `accepts(states)` is true. Explain what would go wrong if you instead allowed `<|end|>` at every step, and what would go wrong if you never allowed it. (c) Why does the chapter insist on constraining only *after* the model emits `<|tool_call|>`, rather than constraining the whole assistant emission?

??? note "Solution"
    (a) `start_states()` is $\{(0,0,0), (1,0,0)\}$ — item 0 of both templates, zero characters consumed. Template 0's item 0 is the literal `{"tool": "search", "args": {"query": "` and template 1's is `{"tool": "calc", "args": {"expr": "`. The two literals agree character-for-character through the shared prefix `{"tool": "` (10 characters), so after each of those both states stay alive, advancing their intra-item counter `n` in lockstep. The 11th character is `s` for template 0 and `c` for template 1. Feeding `c` kills template 0 and the live set collapses to $\{(1, 0, 11)\}$. **The branch resolves at the first character after the shared prefix** — which is precisely why the acceptor tracks a *set* of states: the ambiguity is real but short-lived, and no backtracking is needed.

    (b) Allowing `<|end|>` at every step lets the model terminate mid-JSON — e.g. after `{"tool": "sea` — producing a payload `json.loads` rejects, which `parse_assistant_step` turns into `__malformed__` and the harness turns into a `FormatError` observation. You would have paid the full cost of constrained decoding and still failed to guarantee well-formedness, defeating the point. Never allowing `<|end|>` is worse: the JSON completes, the live set empties on the next character (nothing legally follows `}}`), the mask is all $-\infty$, and the decoder either emits garbage or runs to `max_new` and never returns control. The `accepts()` gate makes the grammar supply both the *legality* condition and the *halting* condition.

    (c) Because the choice between "call a tool" and "give a final answer" is a **decision**, not a format. Constraining the whole emission would force every step into a tool call, so the agent could never answer — the grammar would make the agent's decisions for it. The general line: constrained decoding should restrict the *syntax of a committed action*, never the *space of actions*. It is also why constrained decoding cannot raise $p_{\text{good query}}$: the grammar guarantees the query string is well-formed and says nothing about whether it retrieves the right passage.

**8.** (Quantitative.) A GRPO group contains two trajectories for the same task. Trajectory A takes 2 steps: 40 policy-generated tokens and 120 observation tokens. Trajectory B takes 5 steps: 100 policy-generated tokens and 400 observation tokens. Both have advantage $\hat A = +1$. (a) Under `grpo_agent_step`'s token-mean normalization, what fraction of the gradient comes from each trajectory? (b) Under a sequence-mean scheme (average within each trajectory, then across trajectories), what fraction comes from each? (c) Which trajectory does each scheme favor, and why does this matter *specifically* for agent training with a step penalty? (d) The stats dict reports `gen_tokens / real_tokens`. Compute both for this batch, and explain what would go wrong if you reported `m.numel()` as the denominator instead.

??? note "Solution"
    (a) **Token-mean.** Total generated tokens: $40 + 100 = 140$. Each contributes $1/140$, so trajectory A supplies $40/140 = 2/7 \approx \mathbf{28.6\%}$ of the gradient and B supplies $100/140 = 5/7 \approx \mathbf{71.4\%}$. Weight is proportional to generated length.

    (b) **Sequence-mean.** A's per-token contributions are averaged ($\frac{1}{40}\sum$), likewise B's ($\frac{1}{100}\sum$), and the two averaged: each trajectory gets exactly $\mathbf{50\%}$. Per *token*, A's are weighted $\frac{1}{2\cdot 40} = 1/80$ each and B's $\frac{1}{2 \cdot 100} = 1/200$ each — A's tokens carry 2.5× the weight.

    (c) Token-mean favors the **long** trajectory; sequence-mean favors the **short** one. For agent training this interacts directly with the step penalty $-\lambda \cdot \text{steps}/\text{max\_steps}$, whose entire purpose is to make short trajectories more attractive. Under sequence-mean you apply a *second*, implicit short-trajectory bonus through the normalization, double-counting an effect you already priced explicitly and making $\lambda$ uninterpretable. Token-mean keeps the length preference in exactly one place — the reward — which is the "Dr. GRPO" argument (Liu et al., 2025) and why `grpo_agent_step` divides by `m.sum()`.

    (d) `gen_tokens` $= 140$. `real_tokens` counts every non-pad position: $(40 + 120) + (100 + 400) = 660$, so the ratio is $140/660 \approx 0.21$ — well below 1.0, as it should be.

    If you instead used `m.numel()`, you would count the padded rectangle. With $G=2$ and $T = \max(160, 500) = 500$, that is $2 \times 500 = 1000$ (ignoring the one-token shift), giving $140/1000 = 0.14$. The number is now dominated by how *ragged* the group happened to be: the same policy, the same mask, and a batch whose trajectories differ more in length would report a smaller "supervised fraction." A diagnostic whose value moves for reasons unrelated to what it diagnoses is worse than none — you would either chase phantom masking bugs or miss a real one. (Separately, note that using either denominator in the *loss* rather than as a metric would silently rescale your learning rate by a data-dependent factor, and one that depends on how verbose the *environment* was: switch the retriever from `k=2` to `k=3` and your effective LR changes.)

**9.** (Design + implementation.) `make_task_pool` shuffles all tasks and splits randomly, so the held-out 50 contain the same four shapes as the training 200. (a) Explain why this measures something *weaker* than the distribution-brittleness the Ceiling section calls the defining limitation. (b) Write a `make_shape_holdout(held_kind)` variant that trains on three shapes and evaluates on the fourth, and a `make_entity_holdout()` variant that holds out all tasks mentioning a set of entities. (c) Predict, with reasoning, how the agent's accuracy differs across the three splits — random, entity-held-out, shape-held-out.

??? note "Solution"
    (a) A random split holds out *instances*, not *shapes*. Every held-out task is a `lookup`, `double`, `sum` or `compare` question whose exact template — down to the wording "What do you get if you add the … to the …?" — appeared dozens or hundreds of times in training with different entities filled in. Success on it demonstrates that the model can **slot new entities into a memorized groove**, which is real but modest. The chapter's claim is about *shape* generalization: composing a step pattern never shown. A random split cannot test that, so reporting only random-split accuracy overstates what the agent learned — a version of the train/test-leakage problem in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html).

    (b) Both variants are small changes to the same generator. Note the deliberate absence of a hard-coded `train[:200]`: the pool holds 262 tasks split 28 `lookup` / 28 `double` / 120 `sum` / 86 `compare`, so holding out `sum` leaves only 142 training tasks (and holding out `lookup` or `double` yields only 28 eval tasks, not 50). Silently slicing to fixed sizes would either fail or, worse, quietly return short lists and make the three splits non-comparable — report the actual `len(train)` and `len(held)` alongside every accuracy.

    ```python
    # capstone/stacklm/agent/corpus.py  (append)
    def make_shape_holdout(held_kind: str, seed: int = 0, n_heldout: int = 50):
        """Train on three shapes, evaluate on the fourth. The honest test of
        whether the agent learned a SKILL or a GROOVE. Returns ALL remaining
        tasks as train -- the pool is only 262 tasks and holding out `sum`
        removes 120 of them, so a fixed train size is not available."""
        tasks = make_tasks(seed)
        train = [t for t in tasks if t.kind != held_kind]
        held  = [t for t in tasks if t.kind == held_kind]
        random.Random(seed + 3).shuffle(train)
        random.Random(seed + 4).shuffle(held)
        return train, held[:n_heldout]

    def make_entity_holdout(seed: int = 0, n_entities: int = 4,
                            n_heldout: int = 50):
        """Hold out every task mentioning a chosen set of entities. Tests
        whether retrieval + copying generalizes to unseen facts, holding the
        step pattern fixed. We draw only from PAPER_FACTS: 'Stack-100M' owns
        12 of the 28 facts, so holding it out would strip most of the pool
        rather than a clean slice of it."""
        rng = random.Random(seed + 5)
        held_ents = set(rng.sample(sorted({e for e, _, _ in PAPER_FACTS}),
                                   n_entities))
        tasks = make_tasks(seed)
        def touches(t):                       # entity names appear verbatim
            return any(e in t.question for e in held_ents)
        train = [t for t in tasks if not touches(t)]
        held  = [t for t in tasks if touches(t)]
        rng.shuffle(train); rng.shuffle(held)
        return train, held[:n_heldout]
    ```

    (`make_shape_holdout` must also drop the held shape from the *teacher* pool, not just the eval set — the stub teacher will happily solve a shape the student never saw, and leaving those traces in the SFT data leaks the shape straight back in.)

    (c) Predicted ordering, highest to lowest accuracy:

    1. **Random split** — highest. Same shapes, same wording, new entity fillers. Failures come from retrieval misses on the distractor-heavy corpus, not from planning.
    2. **Entity holdout** — slightly lower but close. The step pattern is unchanged and the model's job is still "retrieve and copy"; only the surface form of the fact is new. The gap between (1) and (2) is a clean measure of how much the model *memorized specific facts* versus *learned to look things up* — if it is large, your corpus leaked into pretraining and you should re-read the contamination warning.
    3. **Shape holdout** — dramatically lower, plausibly near zero for `sum` and `compare` (which need two searches and a specific arithmetic composition never seen). This is the number that honestly reports the ceiling, and it is the number the Ceiling section is about. The fix is not more RLVR on the three trained shapes — it is distilling traces of the fourth.
