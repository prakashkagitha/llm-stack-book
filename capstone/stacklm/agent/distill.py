"""ReAct rollout + distillation (Ch. 14.10). A teacher (a plain
`Callable[[str], str]`) emits one step at a time; the environment runs REAL tool
calls; successful trajectories are kept and formatted for SFT. The teacher is
injected, never imported -- so no network/API in any CI path (see stub_teacher).
"""
from dataclasses import dataclass

from .tools import Passage, BM25Retriever, calc
from .react import (
    parse_assistant_step, render_tool_result, Action,
    SYS, USER, ASST, END,
)

SYSTEM_PROMPT = ("You are a narrow research assistant. Think, then either "
                 "call a tool or give a final answer. Tools: search(query,k), "
                 "calc(expr).")


@dataclass
class Task:
    question: str
    gold: str


class ToolEnv:
    def __init__(self, corpus):
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


def normalize(s) -> str:
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
        gen = teacher(transcript + ASST)           # teacher emits one step
        step_text = gen.split(END, 1)[0]           # up to the first <|end|>
        act = parse_assistant_step(step_text)
        transcript += f"{ASST}{step_text}{END}"
        if act.kind == "final":
            solved = (normalize(act.answer) == normalize(task.gold))
            break
        obs = env.run_tool(act)                     # REAL tool call
        transcript += render_tool_result(obs)      # inject observation
    return transcript, solved


def distill(tasks, teacher, env, samples_per_task: int = 2):
    kept = []
    for task in tasks:
        seen = set()
        for _ in range(samples_per_task):
            transcript, solved = rollout(task, teacher, env)
            if solved and transcript not in seen:
                seen.add(transcript)
                kept.append({"task": task.question, "text": transcript})
    return kept
