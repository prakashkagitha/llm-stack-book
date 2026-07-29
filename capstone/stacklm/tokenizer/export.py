"""Export a trained StackTokenizer into the artifacts the ecosystem reads
(Ch. 14.3): a `tiktoken.Encoding` (fast, OpenAI-style) and a HuggingFace
`tokenizer.json` (what transformers / TRL / vLLM / llama.cpp actually load).

Both exporters are meant to be EXACT: `capstone/tests/test_tokenizer_export.py`
asserts byte-identical id sequences against the from-scratch encoder.

NOTE: `tiktoken`, `tokenizers`, and `transformers` are NOT part of the hermetic
CI dependency set -- every import here is function-local so `stacklm` still
imports on a stdlib-only box. The tests skip if the libraries are absent.
"""
from __future__ import annotations

from typing import Dict

from .bpe import StackTokenizer, SPLIT_PATTERN


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's byte <-> printable-unicode table (Radford et al., 2019). Maps all
    256 byte values to codepoints that survive a JSON round trip: the printable
    ASCII/Latin-1 ranges map to themselves, and the 68 remaining control/space
    bytes are shifted into U+0100.. -- so byte 32 (space) is 'G-with-dot'
    (U+0120) and byte 10 (newline) is U+010A."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_B2U = bytes_to_unicode()


def _as_hf_string(token: bytes) -> str:
    return "".join(_B2U[b] for b in token)


# --- 1. tiktoken -----------------------------------------------------------
def to_tiktoken(tok: StackTokenizer, name: str = "stack100m-32768"):
    """`mergeable_ranks` maps TOKEN BYTES -> rank. Rank order IS merge order,
    which our id layout already guarantees, so no explicit merge list is needed:
    tiktoken re-derives merges by rank lookup."""
    import tiktoken
    ranks = {b: i for i, b in enumerate(tok.token_bytes())}   # 0..255 then merges
    return tiktoken.Encoding(
        name=name,
        pat_str=tok.pattern,                                  # the artifact's regex
        mergeable_ranks=ranks,
        special_tokens=dict(tok.special_to_id),               # 32759..32767
    )


# --- 2. HuggingFace tokenizers --------------------------------------------
def to_hf_tokenizer(tok: StackTokenizer):
    """Build the equivalent `tokenizers.Tokenizer`. The pre-tokenizer is a
    Sequence, exactly as in Llama 3's tokenizer.json: Split on OUR regex first,
    then ByteLevel with `use_regex=False` (it must NOT re-apply GPT-2's own
    pattern) and `add_prefix_space=False` (we never inject a leading space)."""
    from tokenizers import (AddedToken, Regex, Tokenizer, decoders,
                            models, pre_tokenizers)
    table = tok.token_bytes()
    vocab = {_as_hf_string(b): i for i, b in enumerate(table)}
    merges = [(_as_hf_string(table[a]), _as_hf_string(table[b]))
              for a, b in tok.merges]

    hf = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
    hf.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(tok.pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    hf.decoder = decoders.ByteLevel()
    # Added tokens get ids AFTER the model vocab, i.e. exactly 32759.. -- which
    # is why the shortfall padding matters: len(vocab) must be 32759.
    hf.add_special_tokens([AddedToken(s, special=True, normalized=False)
                           for s in tok.special_to_id])

    assert hf.get_vocab_size() == tok.vocab_size
    for s, i in tok.special_to_id.items():
        assert hf.token_to_id(s) == i, f"{s} landed at {hf.token_to_id(s)}, want {i}"
    return hf


# --- 3. transformers: the file everything else loads ----------------------
# ChatML over the reserved role tokens. This MUST render byte-for-byte what
# stacklm.post.chat.render_conversation emits -- including the leading <|bos|>
# and the trailing <|eos|> -- or the served model sees a prompt format it was
# never trained on. test_chat_template_matches_render_conversation pins it.
CHAT_TEMPLATE = (
    "{{ '<|bos|>' }}"
    "{% for m in messages %}"
    "{{ '<|' + m['role'] + '|>' + m['content'] + '<|end|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}"
    "{% else %}{{ '<|eos|>' }}{% endif %}"
)


def save_pretrained(tok: StackTokenizer,
                    out_dir: str = "tokenizer/stack100m-32768-hf"):
    from transformers import PreTrainedTokenizerFast
    fast = PreTrainedTokenizerFast(
        tokenizer_object=to_hf_tokenizer(tok),
        bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>",
                                   "<|end|>", "<|tool_call|>", "<|tool_result|>"],
        chat_template=CHAT_TEMPLATE,
        # 2048 is the PRETRAIN context (PLAN Sec. 1); mid-training extends to
        # 8192 (PLAN Sec. 7) and Ch. 14.11 serves the mid-trained model, so the
        # exported tokenizer must not silently truncate at 2048.
        model_max_length=8192,
    )
    fast.save_pretrained(out_dir)       # -> tokenizer.json, tokenizer_config.json
    return fast
