"""ReAct wire format + parser (Ch. 14.10). The special-token strings and the
non-standard JSON spacing in `render_call` MUST stay byte-identical across
distillation, SFT, and serving or the model's memorized groove breaks.
"""
import json
import re
from dataclasses import dataclass

BOS, EOS, END = "<|bos|>", "<|eos|>", "<|end|>"
SYS, USER, ASST = "<|system|>", "<|user|>", "<|assistant|>"
TOOL_CALL, TOOL_RESULT = "<|tool_call|>", "<|tool_result|>"

ANSWER_RE = re.compile(r"Answer:\s*(.+?)\s*$", re.DOTALL)


@dataclass
class Action:
    kind: str                    # "tool" or "final"
    thought: str = ""
    tool: str = None
    args: dict = None
    answer: str = None


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
