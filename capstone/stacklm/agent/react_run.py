"""Runtime ReAct loop for the trained model (Ch. 14.10). Greedy decode one step,
parse it, run the tool, inject the observation, repeat. Also formats successful
teacher transcripts into loss-masked SFT examples (`build_agent_example`).
"""
import numpy as np
import torch

from .react import (
    parse_assistant_step, render_tool_result,
    ASST, END, TOOL_RESULT, USER, SYS,
)
from .distill import ToolEnv, SYSTEM_PROMPT

IGNORE = -100


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int, stop_id: int) -> str:
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=True)], dtype=torch.long)
    out = []
    for _ in range(max_new):
        logits, _ = model(ids[:, -model.cfg.max_seq_len:])
        nxt = int(logits[:, -1, :].argmax(-1))
        if nxt == stop_id:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return tok.decode(out)


def run_agent(model, tok, question: str, env: ToolEnv,
              max_steps: int = 4, max_new: int = 48):
    transcript = f"{SYS}{SYSTEM_PROMPT}{END}{USER}{question}{END}"
    trace = []
    seen_calls = set()
    for _step in range(max_steps):
        gen = generate(model, tok, transcript + ASST, max_new=max_new,
                       stop_id=tok.id(END))
        step_text = gen.split(END, 1)[0]
        act = parse_assistant_step(step_text)
        transcript += f"{ASST}{step_text}{END}"
        trace.append(("assistant", step_text))

        if act.kind == "final":
            return act.answer, trace

        call_key = (act.tool, str(act.args))
        if call_key in seen_calls:
            obs = "RepeatedCall: you already ran this; use the prior result."
        elif act.tool == "__malformed__":
            obs = "FormatError: emit a valid JSON tool call."
        else:
            obs = env.run_tool(act)
            seen_calls.add(call_key)
        transcript += render_tool_result(obs)
        trace.append(("observation", obs))
    return "", trace


def _segment(t: str):
    """Split a transcript into (text, supervised) spans. Assistant content is
    supervised; system/user/tool-result content is not."""
    out, i, supervised = [], 0, False
    markers = [ASST, USER, SYS, TOOL_RESULT, END]
    while i < len(t):
        cands = [(t.find(m, i), m) for m in markers if t.find(m, i) != -1]
        if not cands:
            out.append((t[i:], supervised))
            break
        j, m = min(cands, key=lambda x: x[0])
        if j > i:
            out.append((t[i:j], supervised))
        if m == ASST:
            supervised = True
            out.append((m, False))
            i = j + len(m)
        elif m in (USER, SYS, TOOL_RESULT):
            supervised = False
            out.append((m, False))
            i = j + len(m)
        else:  # END
            out.append((m, supervised))
            supervised = False
            i = j + len(m)
    return out


def build_agent_example(transcript: str, tok, max_len: int = 512):
    """Format one distilled transcript into (input_ids, labels) with the loss
    masked to assistant spans."""
    ids, labels = [], []
    for text, supervised in _segment(transcript):
        if text in (ASST, USER, SYS, TOOL_RESULT, END):
            piece = [tok.id(text)]
        else:
            piece = tok.encode(text, add_special_tokens=True)
        ids.extend(piece)
        labels.extend(piece if supervised else [IGNORE] * len(piece))
    ids, labels = ids[:max_len], labels[:max_len]
    input_ids = np.array(ids[:-1], dtype=np.int64)
    target = np.array(labels[1:], dtype=np.int64)
    return input_ids, target
