# 5.2 Chat Templates, Data Formatting & Sequence Packing

Raw text pretraining teaches a model to continue arbitrary strings. Turning that capability into a *conversational assistant* requires convincing the model that conversations have structure: a system instruction, alternating user and assistant turns, and a definitive end-of-turn marker. This chapter is the engineering manual for that structure — how templates are designed, how loss masks are constructed to train only on completions, how multi-turn dialogues are packed tightly into fixed-length batches, and how tool calls extend the vocabulary of a chat turn.

Related chapters: [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) covers the overall SFT recipe; this chapter zooms into the data side. [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) explains the causal language modelling loss that we are selectively masking here. [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) explains how special tokens are added to a vocabulary.

## Why Chat Templates Exist

A pretrained language model has seen billions of documents, but almost none of them were labelled "this is the human's turn; this is the assistant's turn." Without some delimiter, there is no way for the model—or the training code—to know where one turn ends and the next begins, and there is no way to enforce "generate here but not there" during both fine-tuning and inference.

The solution is a *chat template*: a deterministic function that maps a list of role–content pairs to a single token sequence that the model will both train on and generate from. Critically, a template also defines a *prompt boundary*: everything before this token is prompt; everything after is what the model is expected to generate. During training we mask (zero out the loss on) the prompt side and supervise only the generation side.

Templates must be:

1. **Unambiguous** — every special token must be one that the base model tokenizer will never split or confuse with natural text.
2. **Consistent** — inference must use the exact same string as training. A single missing space or wrong special token is enough to tank benchmark performance.
3. **Efficient** — the template overhead (role tokens, BOS/EOS) should be small relative to the content.

## The Major Template Families

### ChatML

ChatML (popularised by OpenAI and widely adopted thereafter) encodes each turn as:

```text
<|im_start|>role\ncontent<|im_end|>\n
```

A complete two-turn exchange looks like this:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

The special tokens `<|im_start|>` and `<|im_end|>` (short for *image marker* in the original GPT-4-V vocabulary, though the name is now purely historical) are added to the base vocabulary during fine-tuning. The role string (`system`, `user`, `assistant`) is ordinary text immediately followed by a newline; it is not itself a special token.

During generation, the server appends `<|im_start|>assistant\n` after the last user turn and lets the model decode until it emits `<|im_end|>`.

### Llama 2 & Llama 3 Templates

Meta's Llama 2 used a different convention:

```text
[INST] <<SYS>>
{system}
<</SYS>>

{user_turn} [/INST] {assistant_turn} </s><s>[INST] {next_user} [/INST]
```

This is notoriously tricky: the system prompt is embedded inside the *first* `[INST]` block, there is no per-turn `<s>` except between turns, and the closing `</s>` is part of the assistant turn, not a standalone separator. Getting one of these details wrong silently produces bad outputs.

Llama 3 simplified things significantly, adopting a header-token system closer in spirit to ChatML:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Paris.<|eot_id|>
```

Here `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` (end-of-turn) are proper special tokens. The double newline after the header acts as a role separator. `<|eot_id|>` replaces both `[/INST]` and `</s>` in a single unambiguous token.

### Qwen Templates

Qwen models (Alibaba's open-weight series) use a format almost identical to ChatML, with `<|im_start|>` / `<|im_end|>` inherited from OpenAI conventions:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud.<|im_end|>
<|im_start|>user
Translate "bonjour" to English.<|im_end|>
<|im_start|>assistant
Hello.<|im_end|>
```

The Qwen tokenizer's special-token list includes `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, and a set of tool-calling tokens discussed later.

### Comparison

| Model family | Turn-start token | Turn-end token | System handling |
|---|---|---|---|
| ChatML / GPT-4 | `<\|im_start\|>role` | `<\|im_end\|>` | First `system` turn |
| Llama 2 | `[INST]` | `[/INST]` | Embedded in first `[INST]` |
| Llama 3 | `<\|start_header_id\|>role<\|end_header_id\|>` | `<\|eot_id\|>` | First `system` turn |
| Qwen 2+ | `<\|im_start\|>role` | `<\|im_end\|>` | First `system` turn |
| Mistral v1 | `[INST]` | `[/INST]` | No system turn (injected into user) |
| Gemma | `<start_of_turn>role` | `<end_of_turn>` | Injected into user |

The HuggingFace `tokenizers` library stores a Jinja2 template string in the tokenizer's `chat_template` field so that `tokenizer.apply_chat_template(messages)` always returns the correct string for that model. This is the canonical way to apply templates in Python — do not hard-code delimiter strings yourself.

## Building a Chat Template from Scratch

Let us implement a minimal ChatML formatter and a corresponding loss-mask builder. This is the kind of code you would write to process a dataset before passing it to a training loop.

```python
"""
chat_template.py
────────────────
A from-scratch ChatML formatter with a loss mask builder.
Compatible with any tokenizer that has <|im_start|> and <|im_end|>
in its vocabulary (e.g. Qwen, phi-3, many fine-tuned models).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
import torch
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class ChatMessage:
    role: Role
    content: str

# ---------------------------------------------------------------------------
# Template formatter
# ---------------------------------------------------------------------------

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"

def format_chatml(
    messages: Sequence[ChatMessage],
    add_generation_prompt: bool = True,
) -> str:
    """
    Render a list of ChatMessage objects to a ChatML string.

    If add_generation_prompt=True (used during inference and at the
    end of training examples), appends '<|im_start|>assistant\n' so
    the model knows it should generate next.
    """
    pieces: list[str] = []
    for msg in messages:
        # Each turn: <|im_start|>ROLE\nCONTENT<|im_end|>\n
        pieces.append(f"{IM_START}{msg.role}\n{msg.content}{IM_END}\n")

    if add_generation_prompt:
        pieces.append(f"{IM_START}assistant\n")

    return "".join(pieces)


# ---------------------------------------------------------------------------
# Loss mask builder
# ---------------------------------------------------------------------------

def build_chatml_loss_mask(
    tokenizer: PreTrainedTokenizerFast,
    messages: Sequence[ChatMessage],
    max_length: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a chat conversation and return:
        input_ids  : LongTensor of shape (seq_len,)
        loss_mask  : BoolTensor  of shape (seq_len,)
                     True  → compute loss on this position
                     False → mask out (prompt / special tokens)

    Strategy: we tokenize the full sequence, then for each assistant
    turn we find the span [start_of_content, end_of_turn] and set
    loss_mask = True only for those positions.
    """
    # Render full string (no generation prompt needed for training)
    full_text = format_chatml(messages, add_generation_prompt=False)

    # Tokenize full sequence; do not add special tokens automatically
    # (the template already handles BOS/EOS via <|im_start|> etc.)
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids: torch.Tensor = enc["input_ids"][0]  # (seq_len,)
    seq_len = input_ids.shape[0]

    loss_mask = torch.zeros(seq_len, dtype=torch.bool)

    # -----------------------------------------------------------------------
    # Find assistant spans by scanning for the token IDs of
    # <|im_start|>assistant\n  …content…  <|im_end|>
    # -----------------------------------------------------------------------
    im_start_id: int = tokenizer.convert_tokens_to_ids(IM_START)
    im_end_id:   int = tokenizer.convert_tokens_to_ids(IM_END)

    # Tokenize the role string that follows <|im_start|> for an assistant turn.
    # We need to detect "assistant\n" as token(s).
    role_ids: list[int] = tokenizer.encode(
        "assistant\n", add_special_tokens=False
    )

    ids = input_ids.tolist()
    i = 0
    while i < len(ids):
        # Look for <|im_start|>
        if ids[i] != im_start_id:
            i += 1
            continue

        # Check if the next tokens are the assistant role
        role_end = i + 1 + len(role_ids)
        if role_end <= len(ids) and ids[i + 1 : role_end] == role_ids:
            # We are inside an assistant turn.
            # Content starts at role_end; supervise until <|im_end|>.
            content_start = role_end
            # Find the matching <|im_end|>
            j = content_start
            while j < len(ids) and ids[j] != im_end_id:
                j += 1
            # Mark content tokens for loss (include <|im_end|> itself so the
            # model learns to emit the stop token)
            if j < len(ids):
                loss_mask[content_start : j + 1] = True
            i = j + 1
        else:
            # Not an assistant turn; skip to the next <|im_end|>
            i += 1

    return input_ids, loss_mask
```

!!! example "Worked example: token counts and mask positions"
    Consider this three-message exchange tokenised by a Qwen-2 tokenizer (vocabulary size 151,936):

    ```
    messages = [
        ChatMessage("system",    "You are a helpful assistant."),
        ChatMessage("user",      "What is 2+2?"),
        ChatMessage("assistant", "4."),
    ]
    ```

    The formatted string is (spaces shown explicitly for clarity):

    ```text
    <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    <|im_start|>user\nWhat is 2+2?<|im_end|>\n
    <|im_start|>assistant\n4.<|im_end|>\n
    ```

    Approximate token counts (Qwen-2 tokenizer):
    - System turn: 2 (im_start + "system\n") + ~6 (content) + 1 (im_end) + 1 (newline) = ~10 tokens
    - User turn: ~10 tokens
    - Assistant turn: ~6 tokens

    Total ≈ 26 tokens. Of those, **only the assistant content** (tokens for "4." plus `<|im_end|>`) — roughly 3 tokens — have `loss_mask = True`. The rest (system turn, user turn, role headers) are masked to zero. This is the "train on completions only" principle.

    For a dataset of on the order of 100,000 conversations averaging ~200 tokens each, roughly 30–40% of tokens are typically assistant tokens and thus supervised. Packing (discussed below) ensures we do not pay for the other 60–70% with wasted sequence length.

## Loss Masking in Depth

### Why Mask the Prompt?

During causal language modelling pretraining, every token in the sequence is both an input and a target. When we fine-tune on conversations, we do not want the model to "memorise" system prompts or user questions as outputs it should produce; we want it to learn to *respond* to them. Including the prompt in the loss does two things:

1. **Gradient pollution**: the gradient signal for role tokens like `<|im_start|>user` is dominated by examples where those tokens appear, and none of that signal helps the model become a better assistant.
2. **Objective mismatch**: at inference time, the model never generates the user's message; training it to do so creates a distribution mismatch.

Empirically, SFT runs where the prompt is masked converge to lower perplexity on held-out completions (not full sequences) and exhibit fewer repetitions of the system prompt in responses.

### The Loss Computation with a Mask

The standard causal LM loss for a sequence $x_1, \ldots, x_T$ is:

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

With a binary loss mask $m_t \in \{0, 1\}$, we change this to:

$$
\mathcal{L}_{\text{masked}} = -\frac{1}{\sum_t m_t} \sum_{t=1}^{T} m_t \log p_\theta(x_t \mid x_{<t})
$$

The denominator normalises by the number of *supervised* tokens rather than the total sequence length, so that longer prompts do not dilute the gradient.

In PyTorch the standard way is to pass `labels` to a causal LM where masked positions are set to `-100` — `CrossEntropyLoss` ignores index `-100` by default.

```python
def labels_from_mask(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Construct labels for a causal LM: supervised positions keep their
    token id; masked positions are set to -100 (ignored by CrossEntropyLoss).

    Note the standard causal LM shift: the model predicts x_{t+1} given x_{<=t},
    so labels are input_ids shifted left by one, i.e. labels[t] = input_ids[t+1].
    HuggingFace CausalLM models do this shift internally when you pass `labels`,
    so we just pass the full input_ids with -100 at masked positions.
    """
    labels = input_ids.clone()
    labels[~loss_mask] = -100
    return labels
```

### Multi-Turn Loss Masking

In a multi-turn conversation there are multiple assistant turns, each of which should be supervised independently. The loss-mask builder above already handles this: it scans the entire token sequence and marks all assistant-content spans, not just the final one.

A common mistake is to supervise only the *last* assistant turn. This wastes signal — a four-turn conversation has four assistant responses, all of which are valuable training signal.

{{fig:chattmpl-multiturn-loss-mask}}

The causal mask of the transformer still lets each assistant token attend to everything before it (including previous user turns), so learning is coherent — the model sees the full context, it just does not receive gradient for repeating the prompt tokens.

!!! warning "Common pitfall: off-by-one in the loss shift"
    HuggingFace's `CausalLMOutputWithCrossAttentions` shifts labels internally: the model is given `input_ids[:-1]` and predicts `labels[1:]`. If you pre-shift labels yourself and also let the model shift, every label is off by two positions — completely wrong. Use the convention above: pass the *full* `input_ids` as labels with `-100` masking, and let HuggingFace do the single shift internally.

## System Prompts and Role Tokens

### The System Turn

The system prompt is the privileged message that gives the model its persona, its capabilities, and its constraints. It appears as a `system` role turn at the very beginning of the context. During training on conversations that include system prompts, the system turn is part of the prompt and is masked from the loss.

Not all models have a dedicated system role. Llama 2's template embeds the system prompt inside the first `[INST]` block. Mistral v1 has no system slot at all and expects operators to prepend the system content to the first user message. This matters for fine-tuning: if you are training on Mistral but your dataset has a `role: system` field, you must decide whether to drop it, inject it into the user turn, or add `<s>[INST]` tokens — and then be consistent at inference time.

### Role Tokens as Special Tokens

When role markers like `<|im_start|>` are added to the tokenizer vocabulary, they receive their own embedding vectors in the model's embedding table. These are randomly initialised at fine-tuning time (unless the base model already included them during pretraining, as many modern bases do). Because only a handful of examples per token appear in each gradient step, the embeddings for `<|im_end|>` and role names typically need a higher learning rate or more warm-up steps than ordinary parameters, or they remain near their random init.

One practical consequence: if you fine-tune a model on ChatML format but then serve it with a Llama 3-format prompt, the special tokens your model was trained with may not exist in the serving tokenizer — and vice versa. Always version and lock your tokenizer alongside your model weights.

## Sequence Packing

### The Problem with Padding

Naive batching left-pads (or right-pads) every sequence in a batch to the same length, filling the gap with a `[PAD]` token. The model then wastes computation on padding positions. For SFT datasets where individual examples range from 50 to 2,000 tokens, a batch max-length of 2,048 means that short examples may waste 90%+ of their sequence slot on padding.

For a 7B-parameter model with context length 2,048, each forward + backward pass over a batch of size 4 costs on the order of:

$$
\text{FLOPs} \approx 6 \cdot N_{\text{params}} \cdot T_{\text{batch}} = 6 \times 7 \times 10^9 \times (4 \times 2048) \approx 3.4 \times 10^{14}
$$

If the average sequence is only 512 tokens long, padding wastes 75% of those FLOPs. Sequence packing eliminates this waste.

### Pack-Then-Truncate

Sequence packing concatenates multiple independent examples into a single sequence of exactly `max_length` tokens. The simplest algorithm is first-fit-decreasing bin packing: sort examples by length, greedily fill each bin.

```python
"""
packing.py
──────────
First-fit-decreasing bin packer for SFT sequences.
Each bin becomes one training row of length `max_length`.
"""

from __future__ import annotations
from typing import NamedTuple
import torch

class PackedSample(NamedTuple):
    input_ids:  torch.Tensor  # (max_length,)
    labels:     torch.Tensor  # (max_length,) with -100 for masked positions
    # position_ids tracks the per-document position so RoPE is applied
    # correctly within each packed document.
    position_ids: torch.Tensor  # (max_length,)

PAD_ID = 0   # tokenizer.pad_token_id — used only to fill the last bin

def pack_sequences(
    examples: list[tuple[torch.Tensor, torch.Tensor]],  # (input_ids, labels)
    max_length: int = 2048,
    pad_id: int = PAD_ID,
) -> list[PackedSample]:
    """
    Pack variable-length (input_ids, labels) pairs into fixed-length bins.

    For each packed bin we also build position_ids that reset to 0 at the
    start of each new document — critical for correct RoPE / ALiBi behaviour.
    """
    # Sort longest-first for better bin utilisation (first-fit-decreasing)
    examples = sorted(examples, key=lambda x: x[0].shape[0], reverse=True)

    bins: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    bin_lengths: list[int] = []

    for ids, labs in examples:
        L = ids.shape[0]
        if L > max_length:
            # Truncate oversized examples
            ids  = ids[:max_length]
            labs = labs[:max_length]
            L = max_length

        placed = False
        for b_idx, b_len in enumerate(bin_lengths):
            if b_len + L <= max_length:
                bins[b_idx].append((ids, labs))
                bin_lengths[b_idx] += L
                placed = True
                break
        if not placed:
            bins.append([(ids, labs)])
            bin_lengths.append(L)

    packed: list[PackedSample] = []
    for bin_contents, bin_len in zip(bins, bin_lengths):
        # Concatenate all examples in this bin
        all_ids  = torch.cat([x[0] for x in bin_contents])
        all_labs = torch.cat([x[1] for x in bin_contents])

        # Build position_ids: reset to 0 at the start of each document
        pos = []
        for ids, _ in bin_contents:
            pos.append(torch.arange(ids.shape[0]))
        position_ids = torch.cat(pos)

        # Pad the bin to max_length if needed
        pad_len = max_length - bin_len
        if pad_len > 0:
            all_ids  = torch.cat([all_ids,  torch.full((pad_len,), pad_id)])
            all_labs = torch.cat([all_labs, torch.full((pad_len,), -100)])
            position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])

        packed.append(PackedSample(all_ids, all_labs, position_ids))

    return packed
```

### Attention Boundaries: Preventing Cross-Document Leakage

The most important correctness issue with packing is **cross-document attention leakage**. In a naively packed sequence, token 1,024 (the first token of document 2) can attend to token 1,023 (the last token of document 1), creating a spurious dependency that never exists at inference time.

For most SFT training this leakage is a minor nuisance — the model quickly learns that `<|im_start|>` marks a boundary — but it can degrade performance on short conversations and is measurably harmful for preference learning where sequence independence is critical.

The fix is a **block-diagonal attention mask**: within each packed bin, the causal mask is restricted to within-document boundaries. Each document can only attend to its own prior tokens.

```python
def make_block_diagonal_mask(
    document_lengths: list[int],
    dtype: torch.dtype = torch.float32,
    neg_inf: float = float("-inf"),
) -> torch.Tensor:
    """
    Build a causal block-diagonal attention mask for a packed sequence.

    Returns a (total_len, total_len) additive mask M where
        M[i, j] = 0        if j <= i AND same document as i
        M[i, j] = -inf     otherwise

    Add this to raw attention logits before softmax.
    """
    total_len = sum(document_lengths)
    # Start with full -inf (no attention allowed anywhere)
    mask = torch.full((total_len, total_len), neg_inf, dtype=dtype)

    offset = 0
    for doc_len in document_lengths:
        # Within this document, causal attention is allowed
        for i in range(doc_len):
            for j in range(i + 1):  # j <= i → causal
                mask[offset + i, offset + j] = 0.0
        offset += doc_len

    return mask  # (total_len, total_len)
```

In practice, building this mask explicitly is memory-intensive for long sequences. Production implementations (e.g. in HuggingFace TRL's `DataCollatorForCompletionOnlyLM`, or in Megatron-LM) either use Flash Attention's `cu_seqlens` (cumulative sequence lengths) argument which natively handles variable-length batches without materialising the mask, or they pass the `document_ids` array and compute the mask on-the-fly inside the kernel.

```python
# Using Flash Attention's varlen API (pseudo-code showing the key argument)
# flash_attn.flash_attn_varlen_func accepts:
#   qkv : packed Q/K/V for all documents concatenated
#   cu_seqlens_q, cu_seqlens_k : cumulative sequence lengths (int32)
#   max_seqlen_q, max_seqlen_k : max document length in the batch
# This naturally restricts attention to within-document boundaries.
import flash_attn

output = flash_attn.flash_attn_varlen_func(
    q=q_packed,            # (total_tokens, n_heads, head_dim)
    k=k_packed,
    v=v_packed,
    cu_seqlens_q=cu_seqlens,  # e.g. tensor([0, 512, 1024, 2048])
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_doc_len,
    max_seqlen_k=max_doc_len,
    causal=True,
)
```

!!! example "Packing efficiency worked example"
    Suppose your SFT dataset has the following five examples (lengths in tokens):

    | Example | Tokens |
    |---------|--------|
    | A | 900 |
    | B | 600 |
    | C | 400 |
    | D | 350 |
    | E | 200 |

    With `max_length = 2048` and first-fit-decreasing packing:

    - **Bin 1**: A (900) + B (600) + C (400) = 1,900 tokens → pad 148 → **93% utilisation**
    - **Bin 2**: D (350) + E (200) = 550 tokens → pad 1,498 → **27% utilisation**

    Without packing (one sequence per row, right-padded to 2,048):
    - 5 rows × 2,048 = **10,240 total compute tokens**
    - Actual content = 900 + 600 + 400 + 350 + 200 = **2,450 tokens**
    - Utilisation = 2,450 / 10,240 = **24%**

    With packing (2 bins):
    - 2 rows × 2,048 = **4,096 total compute tokens**
    - Utilisation = 2,450 / 4,096 = **60%**

    In practice, real SFT datasets with thousands of examples achieve 85–95% utilisation because length distributions are smoother and the bin-packing problem has many feasible solutions.

## Tool-Call Formatting

Modern assistant models must handle not just text exchanges but also *tool calls* — structured requests to invoke external functions — and their results. The chat template is the natural place to encode these, and several distinct conventions have emerged.

### The Tool-Use Turn

A tool-call turn is an assistant message that contains a structured payload instead of (or in addition to) plain text. The template must encode both the call and the return value. In ChatML-based systems, this is often done with a dedicated `tool` role for the tool's response:

```json
[
  {"role": "user",      "content": "What's the weather in Paris?"},
  {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_1", "type": "function",
                   "function": {"name": "get_weather",
                                "arguments": "{\"city\": \"Paris\"}"}}]},
  {"role": "tool",      "content": "{\"temp\": 18, \"condition\": \"cloudy\"}",
   "tool_call_id": "call_1"},
  {"role": "assistant", "content": "It's 18 °C and cloudy in Paris."}
]
```

When this is rendered to tokens, the template serialises the `tool_calls` JSON (or a subset of it) as the assistant turn's content. Some models (Qwen, Llama 3.1+) define dedicated special tokens for the open/close of a tool call block:

```text
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call><|im_end|>
<|im_start|>tool
{"temp": 18, "condition": "cloudy"}<|im_end|>
<|im_start|>assistant
It's 18 °C and cloudy in Paris.<|im_end|>
```

Here `<tool_call>` and `</tool_call>` may be single special tokens or ordinary text (model-dependent). The key design decision is whether they are special tokens — in which case they are immune to BPE splitting — or ordinary strings that must be stable under tokenization.

### Loss Masking for Tool Calls

For training, we must decide which parts of a tool-call exchange to supervise:

- **The assistant's tool-call invocation** (the JSON payload): supervised — the model must learn to emit valid JSON with the right function name and arguments.
- **The `tool` role turn** (the function return): masked — the model does not generate this; the harness does.
- **The final assistant text** after the tool return: supervised.

The loss-mask scanner from our `build_chatml_loss_mask` function handles this correctly if `tool` turns are treated as non-assistant turns (they will not match the `role_ids == "assistant\n"` check).

```python
"""
Tool-call example: extend build_chatml_loss_mask to recognise
the <tool_call>...</tool_call> block within an assistant turn.

For simplicity we supervise the ENTIRE assistant turn content
(including the tool_call JSON), since the model should learn to
emit both plain text and tool call syntax.
"""

# The loss masking logic in build_chatml_loss_mask already handles this:
# it supervises everything between <|im_start|>assistant\n and <|im_end|>,
# which includes any <tool_call> blocks the model emits.
# The only special case is if you want to EXCLUDE the <tool_call> JSON
# from the loss (e.g. treat it as planning overhead).  For most SFT
# practitioners, supervising it is the right choice.
```

## Putting It All Together: A Training-Ready Collator

Here is a complete data collator that formats, packs, and masks a batch of conversations, ready to be passed to a training loop.

```python
"""
collator.py
───────────
A DataCollator that applies ChatML formatting, loss masking, and
optional sequence packing.  Drop this into any HuggingFace Trainer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
from transformers import PreTrainedTokenizerFast

# Re-use helpers defined above
# from chat_template import build_chatml_loss_mask, labels_from_mask, ChatMessage
# from packing import pack_sequences

@dataclass
class ChatMLCollator:
    tokenizer: PreTrainedTokenizerFast
    max_length: int = 2048
    pack: bool = True

    def __call__(self, raw_batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        raw_batch: list of dicts with key "messages": list[dict] with
                   "role" and "content" keys (HuggingFace-style).
        """
        examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for item in raw_batch:
            # Convert raw dicts to ChatMessage objects
            messages = [
                ChatMessage(role=m["role"], content=m["content"])
                for m in item["messages"]
            ]
            # Build tokens + mask
            input_ids, loss_mask = build_chatml_loss_mask(
                self.tokenizer, messages, max_length=self.max_length
            )
            labels = labels_from_mask(input_ids, loss_mask)
            examples.append((input_ids, labels))

        if self.pack:
            packed = pack_sequences(
                examples,
                max_length=self.max_length,
                pad_id=self.tokenizer.pad_token_id or 0,
            )
            batch_ids   = torch.stack([p.input_ids   for p in packed])
            batch_labs  = torch.stack([p.labels      for p in packed])
            batch_pos   = torch.stack([p.position_ids for p in packed])
        else:
            # Simple right-padding without packing
            max_len = max(ids.shape[0] for ids, _ in examples)
            pad_id  = self.tokenizer.pad_token_id or 0

            def pad(t: torch.Tensor, fill: int) -> torch.Tensor:
                p = max_len - t.shape[0]
                return torch.cat([t, torch.full((p,), fill)])

            batch_ids  = torch.stack([pad(ids,  pad_id) for ids, _    in examples])
            batch_labs = torch.stack([pad(labs, -100)   for _,   labs in examples])
            # Standard sequential positions (no packing, so no reset needed)
            batch_pos  = torch.arange(max_len).unsqueeze(0).expand(len(examples), -1)

        attention_mask = (batch_ids != (self.tokenizer.pad_token_id or 0)).long()

        return {
            "input_ids":      batch_ids,
            "labels":         batch_labs,
            "attention_mask": attention_mask,
            "position_ids":   batch_pos,
        }
```

!!! interview "Interview Corner"
    **Q:** You are fine-tuning a 13B-parameter model on a multi-turn chat dataset with average conversation length of 400 tokens and a context length of 4,096. A colleague suggests just padding everything to 4,096. What is wrong with that approach, and what would you do instead?

    **A:** Padding to 4,096 when conversations are ~400 tokens means roughly 90% of each sequence is padding. The model wastes FLOPs computing attention over PAD tokens, and gradient normalisation by total sequence length (including padding) dilutes the signal. You should apply **sequence packing**: concatenate multiple conversations into a single 4,096-token row using first-fit-decreasing bin packing. With ~400-token conversations you can fit about 10 per row, achieving ~98% utilisation. To prevent cross-conversation attention leakage you either pass `cu_seqlens` to Flash Attention's variable-length API, or build a block-diagonal causal mask. Additionally, apply a **loss mask** so only assistant turns contribute to the cross-entropy loss — prompt tokens should have label `-100`.

## Practical Checklist for Template Consistency

Template bugs are among the most common causes of unexplained degradation in SFT runs. The following checklist covers the most frequent footguns.

1. **Lock tokenizer version.** A tokenizer update can reorder or renumber special tokens. Pin your tokenizer to the same commit as your model weights.

2. **Use `apply_chat_template`, not string concatenation.** `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)` is always right; manual string formatting is often subtly wrong.

3. **Verify the loss mask before training.** Print out a sample batch and confirm that `labels != -100` only for assistant content. A one-liner: `(labels != -100).float().mean()` should match your expected supervised-fraction (typically 0.25–0.50 for chat data).

4. **Match the inference prompt exactly.** The string passed to the model server at inference time must be the output of `apply_chat_template(messages, add_generation_prompt=True)`. Many bugs arise from adding an extra space, a missing BOS token, or the wrong role name.

5. **Check EOS handling.** Most templates include an EOS-like token (`<|im_end|>`, `<|eot_id|>`) at the end of each assistant turn. Make sure your inference server stops generation at this token — add it to `stop_sequences` if it is not the model's default EOS.

6. **Truncate from the left for long prompts.** If a conversation exceeds `max_length`, truncating from the right destroys the final assistant turn. Truncate early user turns from the left, preserving the system prompt and the most recent turns.

7. **Tool-turn masking.** If your dataset includes tool-return turns (`role: "tool"`), confirm that these are masked. The model should not learn to generate tool return values.

Cross-reference: [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) for the broader training loop; [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) for parameter-efficient fine-tuning considerations; [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html) for the inference-side view of tool calls.

---

!!! key "Key Takeaways"
    - A **chat template** is a deterministic function from role–content pairs to a token sequence; it must be identical between training and inference.
    - The three dominant families are **ChatML** (`<|im_start|>`/`<|im_end|>`), **Llama 3** (`<|start_header_id|>`/`<|eot_id|>`), and the legacy Llama 2 `[INST]`/`[/INST]` format; HuggingFace's `apply_chat_template` abstracts over all of them.
    - **Loss masking** sets `labels = -100` for all prompt and role tokens; only assistant-turn content contributes to the cross-entropy gradient. The denominator of the loss is the number of supervised tokens, not the total sequence length.
    - **Multi-turn masking** should supervise every assistant turn in the conversation, not just the last — this multiplies usable signal per example by the number of turns.
    - **Sequence packing** eliminates padding waste by concatenating multiple examples into a single max-length row; a block-diagonal causal mask (or Flash Attention's `cu_seqlens` API) prevents cross-document attention leakage.
    - **Tool-call turns** are handled by the template: assistant invocations are supervised; `tool` role returns are masked, because the harness — not the model — generates them.
    - Template inconsistency between training and inference is one of the most common silent causes of fine-tuned model degradation — always lock and test the template as part of your model release process.

!!! sota "State of the Art & Resources (2026)"
    Chat templates and sequence packing are mature but actively evolving: the HuggingFace ecosystem has converged on Jinja2 tokenizer templates and the `apply_chat_template` API as the interoperability standard, while Flash Attention's variable-length (`varlen`) API has made padding-free packed training mainstream at scale.

    **Foundational work**

    - [Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models* (2023)](https://arxiv.org/abs/2307.09288) — canonical description of the `[INST]`/`[/INST]` template and the role of formatting in RLHF data collection.
    - [Krell et al., *Efficient Sequence Packing without Cross-contamination* (2021)](https://arxiv.org/abs/2107.02027) — foundational study showing up to 50% of tokens can be padding, formalises packing as bin-packing, and proves block-diagonal masks prevent accuracy loss.

    **Recent advances (2023–2026)**

    - [Meta, *The Llama 3 Herd of Models* (2024)](https://arxiv.org/abs/2407.21783) — introduces the `<|start_header_id|>` / `<|eot_id|>` header-token template and multi-turn RLHF data formatting at scale.
    - [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691) — the `flash_attn_varlen_func` / `cu_seqlens` API that makes packed training without explicit mask materialisation practical.
    - [Kundu et al., *Enhancing Training Efficiency Using Packing with Flash Attention* (2024)](https://arxiv.org/abs/2407.09105) — empirical analysis showing up to 2× throughput and ~20% memory reduction when combining packing with Flash Attention 2 across 14 model families.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — TRL's `SFTTrainer` with `packing=True` and `assistant_only_loss=True` is the reference implementation of packed, completion-masked SFT.
    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the `flash_attn_varlen_func` used for block-diagonal packed-sequence attention without materialising the mask.

    **Go deeper**

    - [HuggingFace Transformers — Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating) — canonical docs for `apply_chat_template`, Jinja2 template authoring, and the `add_generation_prompt` / `continue_final_message` parameters.
    - [HuggingFace Blog — Improving Training Efficiency Through Packing with Flash Attention 2](https://huggingface.co/blog/packing-with-FA2) — step-by-step guide to enabling `DataCollatorWithFlattening` and `padding_free=True` in TRL, with throughput benchmarks across real SFT datasets.

## Further Reading

- **HuggingFace `chat_templates` documentation and Jinja2 template examples** — the canonical reference for `apply_chat_template` and per-model template strings. Available in the `transformers` repository under `docs/source/en/chat_templating.md`.
- **Meta, "Llama 2: Open Foundation and Fine-Tuned Chat Models"** (Touvron et al., 2023) — describes the `[INST]`/`[/INST]` format and its role in RLHF data collection.
- **Meta, "The Llama 3 Herd of Models"** (Dubey et al., 2024) — introduces the Llama 3 header-token template and discusses multi-turn RLHF data formatting.
- **OpenAI, "ChatML format"** — the original description of `<|im_start|>`/`<|im_end|>` tokens, described in early GPT-4 system-card documentation.
- **Krell et al., "Efficient Sequence Packing without Cross-contamination"** (2021) — empirical study on the impact of cross-document attention in packed pretraining sequences, and the efficacy of document-boundary masks.
- **Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** (2023) — the `flash_attn_varlen_func` API used for packing without explicit mask materialisation.
- **Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs"** (2023) — covers structured generation and how tool-call schemas interact with constrained decoding, complementing the template story.
