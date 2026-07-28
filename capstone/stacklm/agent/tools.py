"""Agent tools (Ch. 14.10): a safe AST calculator and a from-scratch BM25 lexical
retriever (plus an 'embedding-lite' hashing retriever). Pure stdlib -- no
`rank_bm25`, no embedding model, no network.
"""
import ast
import hashlib
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
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
    except Exception as e:  # noqa: BLE001
        return f"CalcError: {e}"
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


_TOK = re.compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return _TOK.findall(text.lower())


@dataclass
class Passage:
    doc_id: str
    text: str


class BM25Retriever:
    """Textbook Okapi BM25 (Robertson & Zaragoza)."""

    def __init__(self, passages, k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [_tokenize(p.text) for p in passages]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.docs))
        df = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        N = len(self.docs)
        self.idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query_terms, i: int) -> float:
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

    def search(self, query: str, k: int = 3):
        q = _tokenize(query)
        scored = [(self.passages[i], self.score(q, i)) for i in range(len(self.docs))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, sc) for p, sc in scored[:k] if sc > 0.0]


class HashEmbedRetriever:
    """'Embedding-lite': the hashing trick (Weinberger et al., 2009), cosine-rank."""

    def __init__(self, passages, dim: int = 256):
        self.passages, self.dim = passages, dim
        self.vecs = [self._embed(p.text) for p in passages]

    def _embed(self, text: str):
        v = [0.0] * self.dim
        for t in _tokenize(text):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 1) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def search(self, query: str, k: int = 3):
        qv = self._embed(query)
        scored = [(self.passages[i], sum(a * b for a, b in zip(qv, self.vecs[i])))
                  for i in range(len(self.passages))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p, sc) for p, sc in scored[:k] if sc > 0.0]
