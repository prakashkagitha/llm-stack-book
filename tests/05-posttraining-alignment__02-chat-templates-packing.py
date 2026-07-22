"""
Executable test for content/05-posttraining-alignment/02-chat-templates-packing.md

Runs the chapter's CPU-runnable Python blocks (verbatim) and exercises them with
small, honest fixtures:

  - block #5  (~line 106, 133 lines): chat_template.py
                ChatMessage / format_chatml / build_chatml_loss_mask
  - block #9  (~line 364,  87 lines): packing.py
                PackedSample / pack_sequences
  - block #15 (~line 611, 111 lines): collator.py
                ChatMLCollator

block #5 and block #15 need a real tokenizer object (they call
tokenizer.convert_tokens_to_ids / tokenizer.encode / tokenizer(...)). We build a
tiny, fully-offline PreTrainedTokenizerFast (WordLevel vocab, no network, no
downloads) that has <|im_start|> / <|im_end|> as special tokens, exactly the
kind of tokenizer the chapter assumes ("Compatible with any tokenizer that has
<|im_start|> and <|im_end|> in its vocabulary").

`transformers`/`tokenizers` are optional third-party deps: guarded at module
scope so the file still loads (and block #9, which is pure torch, still runs)
even if they are missing in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence, Any, NamedTuple

import torch

try:
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    HAS_TRANSFORMERS = True
except Exception:
    PreTrainedTokenizerFast = None
    HAS_TRANSFORMERS = False


# =============================================================================
# Block #5 (~line 106): chat_template.py — verbatim from the chapter
# =============================================================================

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
IM_END = "<|im_end|>"


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
    tokenizer: "PreTrainedTokenizerFast",
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
    im_end_id: int = tokenizer.convert_tokens_to_ids(IM_END)

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


# ---------------------------------------------------------------------------
# labels_from_mask (from "The Loss Computation with a Mask" section, ~line 299)
# ---------------------------------------------------------------------------

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


# =============================================================================
# Block #9 (~line 364): packing.py — verbatim from the chapter
# =============================================================================

class PackedSample(NamedTuple):
    input_ids: torch.Tensor  # (max_length,)
    labels: torch.Tensor  # (max_length,) with -100 for masked positions
    # position_ids tracks the per-document position so RoPE is applied
    # correctly within each packed document.
    position_ids: torch.Tensor  # (max_length,)
    # seqlens holds the length of each real (non-pad) document in this bin,
    # so the collator can build cu_seqlens for Flash Attention's varlen path.
    seqlens: torch.Tensor  # (num_docs,) int32


PAD_ID = 0  # tokenizer.pad_token_id — used only to fill the last bin


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
            ids = ids[:max_length]
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
        all_ids = torch.cat([x[0] for x in bin_contents])
        all_labs = torch.cat([x[1] for x in bin_contents])

        # Build position_ids: reset to 0 at the start of each document
        pos = []
        for ids, _ in bin_contents:
            pos.append(torch.arange(ids.shape[0]))
        position_ids = torch.cat(pos)

        # Pad the bin to max_length if needed
        pad_len = max_length - bin_len
        if pad_len > 0:
            all_ids = torch.cat([all_ids, torch.full((pad_len,), pad_id)])
            all_labs = torch.cat([all_labs, torch.full((pad_len,), -100)])
            position_ids = torch.cat([position_ids, torch.zeros(pad_len, dtype=torch.long)])

        # Per-document lengths within this bin (real docs only; the trailing
        # pad region is handled by the collator via max_length - content_len).
        doc_lens = torch.tensor(
            [ids.shape[0] for ids, _ in bin_contents], dtype=torch.int32
        )
        packed.append(PackedSample(all_ids, all_labs, position_ids, doc_lens))

    return packed


# ---------------------------------------------------------------------------
# make_block_diagonal_mask (from "Attention Boundaries" section, ~line 464)
# small, pure-torch, CPU-safe helper — exercised below with tiny doc lengths.
# ---------------------------------------------------------------------------

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


# =============================================================================
# Block #15 (~line 611): collator.py — verbatim from the chapter
# =============================================================================

@dataclass
class ChatMLCollator:
    tokenizer: "PreTrainedTokenizerFast"
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
            batch_ids = torch.stack([p.input_ids for p in packed])
            batch_labs = torch.stack([p.labels for p in packed])
            batch_pos = torch.stack([p.position_ids for p in packed])

            # Build a length-based attention_mask and cu_seqlens over the
            # flattened (B * max_length,) batch. The trailing pad region of
            # each row becomes its own segment so offsets sum to max_length.
            attention_mask = torch.zeros_like(batch_ids)
            row_seqlens: list[torch.Tensor] = []
            for r, p in enumerate(packed):
                content_len = int(p.seqlens.sum())  # real (non-pad) tokens
                attention_mask[r, :content_len] = 1
                lens = p.seqlens
                pad_len = self.max_length - content_len
                if pad_len > 0:
                    lens = torch.cat(
                        [lens, torch.tensor([pad_len], dtype=torch.int32)]
                    )
                row_seqlens.append(lens)

            flat_seqlens = torch.cat(row_seqlens)  # (num_segments_total,)
            cu_seqlens = torch.zeros(flat_seqlens.numel() + 1, dtype=torch.int32)
            torch.cumsum(flat_seqlens, dim=0, out=cu_seqlens[1:])
            max_seqlen = int(flat_seqlens.max())
        else:
            # Simple right-padding without packing
            max_len = max(ids.shape[0] for ids, _ in examples)
            pad_id = self.tokenizer.pad_token_id or 0

            def pad(t: torch.Tensor, fill: int) -> torch.Tensor:
                p = max_len - t.shape[0]
                return torch.cat([t, torch.full((p,), fill)])

            batch_ids = torch.stack([pad(ids, pad_id) for ids, _ in examples])
            batch_labs = torch.stack([pad(labs, -100) for _, labs in examples])
            # Standard sequential positions (no packing, so no reset needed)
            batch_pos = torch.arange(max_len).unsqueeze(0).expand(len(examples), -1)

            # Length-based attention_mask. Do NOT use (batch_ids != pad_token_id):
            # when pad_token == eos_token (the Llama-style config from 5.1) that
            # would also zero out every real EOS / <|im_end|> token.
            attention_mask = torch.zeros_like(batch_ids)
            for r, (ids, _) in enumerate(examples):
                attention_mask[r, : ids.shape[0]] = 1
            cu_seqlens = None
            max_seqlen = None

        batch = {
            "input_ids": batch_ids,
            "labels": batch_labs,
            "attention_mask": attention_mask,
            "position_ids": batch_pos,
        }
        # cu_seqlens / max_seqlen drive the flash-attn varlen path (see below)
        # that keeps attention inside document boundaries. A stock HF model does
        # not accept these kwargs, so consume them in a patched attention forward
        # (or a custom Trainer) and pop them before calling model().
        if self.pack:
            batch["cu_seqlens"] = cu_seqlens
            batch["max_seqlen"] = max_seqlen
        return batch


# =============================================================================
# Test fixtures / glue
# =============================================================================

def _build_tiny_chatml_tokenizer() -> "PreTrainedTokenizerFast":
    """
    Minimal, fully-offline (no network/download) tokenizer with <|im_start|>
    and <|im_end|> as special tokens, satisfying the chapter's stated
    compatibility requirement ("any tokenizer that has <|im_start|> and
    <|im_end|> in its vocabulary"). Word-level vocab covers exactly the
    words used in the test fixtures below.
    """
    words = [
        "You", "are", "a", "helpful", "assistant", ".", "What", "is",
        "2", "+", "?", "4", "system", "user", "tool", "Paris", "France",
        "capital", "of", "the", "weather", "in", "Bonjour", "Hello",
        "Translate", "to", "English", ",", "world", "!", "How", "can",
        "I", "help", "you", "today",
    ]
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for idx, w in enumerate(sorted(set(words)), start=2):
        vocab[w] = idx

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens([IM_START, IM_END])

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=[IM_START, IM_END],
    )
    return fast


def _run_block5_chat_template() -> None:
    print("--- block #5: chat_template.py (format_chatml / build_chatml_loss_mask) ---")

    messages = [
        ChatMessage("system", "You are a helpful assistant."),
        ChatMessage("user", "What is 2+2?"),
        ChatMessage("assistant", "4."),
    ]

    # format_chatml, both with and without the generation prompt
    rendered = format_chatml(messages, add_generation_prompt=False)
    assert rendered.count(IM_START) == 3
    assert rendered.count(IM_END) == 3
    assert rendered.startswith(f"{IM_START}system\n")
    assert rendered.endswith(f"{IM_END}\n")

    rendered_gen = format_chatml(messages, add_generation_prompt=True)
    assert rendered_gen.endswith(f"{IM_START}assistant\n")
    print("  format_chatml: OK (rendered", len(rendered), "chars,",
          len(rendered_gen), "with generation prompt)")

    if not HAS_TRANSFORMERS:
        print("  SKIP(missing-dep): transformers/tokenizers not installed — "
              "build_chatml_loss_mask needs a real tokenizer, skipping.")
        return

    tokenizer = _build_tiny_chatml_tokenizer()

    input_ids, loss_mask = build_chatml_loss_mask(tokenizer, messages, max_length=128)
    assert input_ids.shape == loss_mask.shape
    assert loss_mask.dtype == torch.bool
    n_supervised = int(loss_mask.sum())
    assert 0 < n_supervised < input_ids.shape[0], (
        f"expected a strict subset of tokens supervised, got {n_supervised}/{input_ids.shape[0]}"
    )

    # The system and user turns must be entirely masked (loss_mask == False);
    # only the assistant turn's content + <|im_end|> should be supervised.
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    supervised_ids = input_ids[loss_mask].tolist()
    assert supervised_ids[-1] == im_end_id, "assistant span must include the <|im_end|> token"
    assert supervised_ids.count(im_end_id) == 1, "only the assistant turn's <|im_end|> should be supervised"

    labels = labels_from_mask(input_ids, loss_mask)
    assert (labels[~loss_mask] == -100).all()
    assert (labels[loss_mask] == input_ids[loss_mask]).all()
    print(f"  build_chatml_loss_mask + labels_from_mask: OK "
          f"({n_supervised}/{input_ids.shape[0]} tokens supervised)")


def _run_block9_packing() -> None:
    print("--- block #9: packing.py (pack_sequences) ---")

    torch.manual_seed(0)

    def toy_example(length: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.arange(1, length + 1, dtype=torch.long)
        labs = ids.clone()
        return ids, labs

    # Mirrors the chapter's worked packing-efficiency example (scaled down):
    # lengths 9, 6, 4, 3, 2 into bins of max_length=20.
    examples = [toy_example(L) for L in [9, 6, 4, 3, 2]]
    max_length = 20

    packed = pack_sequences(examples, max_length=max_length, pad_id=0)

    # First-fit-decreasing: 9+6+4=19 (bin 1), 3+2=5 (bin 2)
    assert len(packed) == 2, f"expected 2 bins, got {len(packed)}"

    for sample in packed:
        assert sample.input_ids.shape[0] == max_length
        assert sample.labels.shape[0] == max_length
        assert sample.position_ids.shape[0] == max_length

    bin1 = packed[0]
    assert int(bin1.seqlens.sum()) == 19
    assert int((bin1.input_ids[19:] == 0).sum()) == 1  # 1 pad token
    assert int((bin1.labels[19:] == -100).sum()) == 1
    # position_ids reset to 0 at each document boundary: doc1 (len9) starts at
    # 0, doc2 (len6) starts again at 0, doc3 (len4) starts again at 0.
    assert bin1.position_ids[0].item() == 0
    assert bin1.position_ids[9].item() == 0
    assert bin1.position_ids[9 + 6].item() == 0
    assert list(bin1.seqlens.tolist()) == [9, 6, 4]

    bin2 = packed[1]
    assert int(bin2.seqlens.sum()) == 5
    assert int((bin2.input_ids[5:] == 0).sum()) == max_length - 5

    total_content = sum(L for L in [9, 6, 4, 3, 2])
    total_compute = len(packed) * max_length
    utilisation = total_content / total_compute
    print(f"  pack_sequences: OK (2 bins, utilisation={utilisation:.0%}, "
          f"bin1 seqlens={bin1.seqlens.tolist()}, bin2 seqlens={bin2.seqlens.tolist()})")

    # make_block_diagonal_mask: tiny 2-document case, verify block-diagonal
    # causal structure exactly.
    doc_lengths = [3, 2]
    mask = make_block_diagonal_mask(doc_lengths)
    assert mask.shape == (5, 5)
    # doc 1 occupies rows/cols [0,3): causal within, -inf to doc 2
    assert mask[0, 0].item() == 0.0
    assert mask[2, 0].item() == 0.0  # causal: token 2 attends to token 0 (same doc)
    assert mask[2, 1].item() == 0.0
    assert mask[0, 2].item() == float("-inf")  # non-causal within doc: blocked
    assert mask[3, 0].item() == float("-inf")  # doc 2 cannot see doc 1
    assert mask[3, 3].item() == 0.0  # doc 2 causal self-attention
    assert mask[4, 3].item() == 0.0
    assert mask[4, 4].item() == 0.0
    print("  make_block_diagonal_mask: OK (5x5 mask, cross-document leakage correctly blocked)")


def _run_block15_collator() -> None:
    print("--- block #15: collator.py (ChatMLCollator) ---")

    if not HAS_TRANSFORMERS:
        print("  SKIP(missing-dep): transformers/tokenizers not installed — "
              "ChatMLCollator needs a real tokenizer, skipping.")
        return

    tokenizer = _build_tiny_chatml_tokenizer()

    raw_batch = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris."},
            ]
        },
    ]

    max_length = 64

    # --- pack=True path ------------------------------------------------
    collator_packed = ChatMLCollator(tokenizer=tokenizer, max_length=max_length, pack=True)
    batch = collator_packed(raw_batch)

    for key in ("input_ids", "labels", "attention_mask", "position_ids", "cu_seqlens", "max_seqlen"):
        assert key in batch

    # "Verify it." checks from the chapter (~line 751):
    assert int(batch["cu_seqlens"][-1]) == batch["input_ids"].numel(), (
        "segments must cover every packed token"
    )
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all(), (
        "every masked-out (padding) position must be unsupervised"
    )
    print(f"  ChatMLCollator(pack=True): OK (batch input_ids shape={tuple(batch['input_ids'].shape)}, "
          f"cu_seqlens={batch['cu_seqlens'].tolist()})")

    # --- pack=False path -------------------------------------------------
    collator_unpacked = ChatMLCollator(tokenizer=tokenizer, max_length=max_length, pack=False)
    batch2 = collator_unpacked(raw_batch)
    assert "cu_seqlens" not in batch2 and "max_seqlen" not in batch2
    real_token_counts = batch2["attention_mask"].sum(dim=1)
    assert (real_token_counts > 0).all()
    print(f"  ChatMLCollator(pack=False): OK (attention_mask real-token counts={real_token_counts.tolist()})")

    # --- pad_token == eos_token trap (chapter warning ~line 700, 729) ----
    # Build a second tokenizer where the pad token is aliased to <|im_end|>
    # (mirrors Llama-style `tokenizer.pad_token = tokenizer.eos_token` from
    # ch 5.1) and confirm attention_mask.sum() still counts every real token,
    # i.e. is NOT reduced by the naive `(input_ids != pad_token_id)` approach
    # the chapter warns against.
    eos_id = tokenizer.convert_tokens_to_ids(IM_END)
    tokenizer.pad_token_id = eos_id  # pad_token now aliases the real EOS token

    collator_eos_pad = ChatMLCollator(tokenizer=tokenizer, max_length=max_length, pack=False)
    batch3 = collator_eos_pad(raw_batch)

    naive_mask_sum = int((batch3["input_ids"] != eos_id).sum())
    length_based_sum = int(batch3["attention_mask"].sum())
    assert length_based_sum > naive_mask_sum, (
        "the length-based attention_mask must NOT drop real <|im_end|> tokens "
        "the way a naive (input_ids != pad_token_id) mask would"
    )
    print(f"  pad_token==eos_token trap: OK (length-based sum={length_based_sum} "
          f"> naive (ids!=pad) sum={naive_mask_sum}, correctly keeping real EOS tokens)")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    _run_block5_chat_template()
    _run_block9_packing()
    _run_block15_collator()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
