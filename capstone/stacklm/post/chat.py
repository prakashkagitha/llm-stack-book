"""ChatML-like template over the reserved special tokens (Ch. 14.9). The load-
bearing rule: supervise ONLY assistant-turn content and its closing `<|end|>`
(so the model learns to stop). Everything else -- role markers, system/user text,
the generation prompt -- is masked out of the loss.
"""
from dataclasses import dataclass

SPECIAL = {
    "bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>",
    "system": "<|system|>", "user": "<|user|>",
    "assistant": "<|assistant|>", "end": "<|end|>",
}
DEFAULT_SYSTEM = "You are Stack-100M, a concise, honest assistant."


@dataclass
class Turn:
    role: str      # "system" | "user" | "assistant"
    content: str


def render_conversation(turns, tok, add_generation_prompt=False):
    """Returns (ids, supervised_mask). `tok` is a StackTokenizer."""
    ids, mask = [], []

    def emit(text, supervised):
        piece = tok.encode(text, add_special_tokens=False)
        ids.extend(piece)
        mask.extend([1 if supervised else 0] * len(piece))

    def emit_special(name, supervised):
        ids.append(tok.special_token_id(SPECIAL[name]))
        mask.append(1 if supervised else 0)

    emit_special("bos", supervised=False)
    for t in turns:
        emit_special(t.role, supervised=False)          # role marker: context
        if t.role == "assistant":
            emit(t.content, supervised=True)            # <-- the only tokens we learn
            emit_special("end", supervised=True)        # learn to STOP the turn
        else:
            emit(t.content, supervised=False)
            emit_special("end", supervised=False)
    if add_generation_prompt:                           # inference-time only
        emit_special("assistant", supervised=False)
    else:
        emit_special("eos", supervised=False)
    return ids, mask
