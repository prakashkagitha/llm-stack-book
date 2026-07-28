"""Deterministic OFFLINE teacher (Ch. 14.10). In the book the teacher is a large
model called over an API; that call is illustrative only and must never run in
CI. This stub implements a fixed 'find a year, then double it' policy so
distillation is fully hermetic. Swap in a real `Callable[[str], str]` for a real
teacher.
"""
import re

from .react import TOOL_RESULT, END, render_call


def make_stub_teacher():
    def teacher(prompt: str) -> str:
        n_obs = prompt.count(TOOL_RESULT)
        last_obs = prompt.rsplit(TOOL_RESULT, 1)[-1].split(END, 1)[0] if n_obs else ""
        if n_obs == 0:
            q = re.search(r"about (.+?)[\?\.]", prompt)
            query = q.group(1) if q else "topic"
            return (f"Thought: I should look this up.\n"
                    f"{render_call('search', {'query': query, 'k': 2})}")
        if n_obs == 1 and re.search(r"\d{3,4}", last_obs):
            year = re.search(r"\d{3,4}", last_obs).group(0)
            return (f"Thought: Found {year}; the task wants it doubled.\n"
                    f"{render_call('calc', {'expr': f'{year}*2'})}")
        num = re.search(r"-?\d+", last_obs)
        ans = num.group(0) if num else last_obs.strip()
        return f"Thought: I have the result.\nAnswer: {ans}"
    return teacher
