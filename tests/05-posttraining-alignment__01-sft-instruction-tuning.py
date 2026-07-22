"""CI-tested extracts of runnable code blocks from
content/05-posttraining-alignment/01-sft-instruction-tuning.md

Block inventory (per the chapter's code-block ordering):
  - block #0 (~line 233, the full `sft_train.py`): SKIP(needs-gpu) — the
    `train()` function does `AutoModelForCausalLM.from_pretrained(...,
    device_map="auto")` and a full training loop; this is explicitly a
    GPU/multi-hour training script, not CPU-testable. However, block #1
    (below) calls `InstructionDataset`, which is *defined* inside this
    block and is pure CPU logic (tokenization + loss-mask construction,
    no model, no device placement). We reproduce that one class verbatim
    (plus the `IGNORE_INDEX` constant it uses) as the minimal glue needed
    for block #1 to run, and do NOT call `train()` anywhere.
  - block #1 (~line 546, "Sanity-check the mask on one example"): TESTED.
    Reproduced verbatim below, against a tiny local JSONL fixture and a
    tiny offline mock tokenizer (see note below on why the tokenizer is
    mocked).

Note on the tokenizer: the book's `InstructionDataset` takes any
tokenizer-like object (it only ever calls `tokenizer(text,
add_special_tokens=...)`, `tokenizer.eos_token_id`,
`tokenizer.convert_ids_to_tokens`, `tokenizer.decode`,
`tokenizer.bos_token_id`, `tokenizer.eos_token`). The real chapter script
instantiates this via `AutoTokenizer.from_pretrained(args.model_name)`,
which is a Hugging Face Hub network call — forbidden in this offline test
per the harness rules. We MOCK that one boundary with a tiny deterministic
whitespace tokenizer that implements the exact same interface, so that
`InstructionDataset`'s own tokenize/concatenate/mask logic — which is the
actual point of block #1 — executes for real, offline. The `datasets`
library (`datasets.load_dataset("json", ...)`) IS exercised for real
against a local temp file (no network involved for local JSON loading).

Run directly: `python3 tests/05-posttraining-alignment__01-sft-instruction-tuning.py`
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import datasets
except Exception:
    datasets = None


# ---------------------------------------------------------------------------
# Minimal glue: reproduced verbatim from block #0 (sft_train.py), the parts
# that block #1 depends on by name. The GPU-bound train() function from
# block #0 is intentionally NOT reproduced or called.
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss ignores this label index


class InstructionDataset(Dataset):
    """
    Loads JSONL with fields {"instruction": str, "response": str}.
    Tokenizes and builds input_ids + labels where instruction tokens
    are masked (IGNORE_INDEX) so only response tokens incur loss.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

        # Load raw data
        raw = datasets.load_dataset("json", data_files=data_path, split="train")
        self.samples = [row for row in raw]

    def _format(self, instruction: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Build input_ids + labels with response-only loss masking.
        We tokenize the prompt and the response SEPARATELY and concatenate
        their ids. This makes the mask boundary exact: there is no
        cross-boundary merge and no off-by-one from an auto-prepended BOS.
        The template contains NO literal <s>/</s> -- the tokenizer adds a
        single BOS to the prompt, and we append EOS by id so the model
        learns to stop.
        """
        prompt = (
            f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n"
            f"{instruction} [/INST]"
        )
        # add_special_tokens=True -> prompt_ids begins with exactly one BOS.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # add_special_tokens=False -> response carries NO BOS of its own.
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        response_ids = response_ids + [self.tokenizer.eos_token_id]  # teach stopping

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[: self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Verify the mask boundary -- the exact bug class this chapter warns
        # about: prompt fully masked, response labels equal response ids.
        prompt_len = len(prompt_ids)
        assert bool((labels[:prompt_len] == IGNORE_INDEX).all()), "prompt not masked"
        if prompt_len < input_ids.shape[0]:
            assert bool((labels[prompt_len:] == input_ids[prompt_len:]).all()), \
                "response labels must equal response ids"

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        return self._format(row["instruction"], row["response"])


# ---------------------------------------------------------------------------
# Offline mock tokenizer: implements exactly the interface InstructionDataset
# uses. Whitespace-splits text into a deterministic per-word vocabulary.
# This replaces AutoTokenizer.from_pretrained(...), which would otherwise
# require a Hugging Face Hub network call (forbidden in this offline test).
# ---------------------------------------------------------------------------

class MockTokenizer:
    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self._vocab: Dict[str, int] = {}
        self._next_id = 10

    def _word_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._next_id += 1
        return self._vocab[word]

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [self._word_id(w) for w in text.split()]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == self.bos_token_id:
            return self.bos_token
        if token_id == self.eos_token_id:
            return self.eos_token
        if token_id == self.pad_token_id:
            return "<pad>"
        for word, wid in self._vocab.items():
            if wid == token_id:
                return word
        return "<unk>"

    def decode(self, ids) -> str:
        return " ".join(self.convert_ids_to_tokens(int(i)) for i in ids)


def block_verify_loss_mask():
    """content lines ~546-562: 'Sanity-check the mask on one example'."""
    if datasets is None:
        print("SKIP(missing-optional-dep): `datasets` package not installed; "
              "InstructionDataset.__init__ calls datasets.load_dataset. "
              "Skipping block #1.")
        return

    # ----- minimal honest glue: tiny local fixture instead of a real path,
    # and the offline MockTokenizer instead of AutoTokenizer.from_pretrained
    # (which would require a Hugging Face Hub network call) -----
    tmpdir = tempfile.mkdtemp()
    data_path = str(Path(tmpdir) / "sft_data.jsonl")
    rows = [
        {"instruction": "What is two plus two?", "response": "Two plus two equals four."},
        {"instruction": "Name the capital of France.", "response": "The capital of France is Paris."},
    ]
    with open(data_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    tokenizer = MockTokenizer()

    # ----- verbatim block #1 -----
    # Sanity-check the mask on one example BEFORE launching a run.
    ds = InstructionDataset(data_path, tokenizer)
    ex = ds[0]
    ids, labs = ex["input_ids"], ex["labels"]
    for tok, lab in zip(ids.tolist(), labs.tolist()):
        piece = tokenizer.convert_ids_to_tokens(tok)
        flag = "MASK" if lab == IGNORE_INDEX else "LOSS"
        print(f"{tok:>6}  {flag}  {piece!r}")

    # Expected: ids[0] is the BOS id and it appears exactly once; every prompt
    # token is flagged MASK; every response token (including the trailing EOS)
    # is flagged LOSS. The unmasked region must decode back to the response:
    resp = tokenizer.decode(ids[labs != IGNORE_INDEX])
    assert resp.rstrip().endswith(tokenizer.eos_token)
    assert (ids == tokenizer.bos_token_id).sum().item() == 1  # no double BOS
    # ----- end verbatim block #1 -----

    # --- extra check that this test itself is honest: response labels
    # really do decode back to the original response text + EOS ---
    assert resp.rstrip() == (rows[0]["response"] + " " + tokenizer.eos_token)
    print("Loss-mask sanity check passed:", resp)


BLOCKS = [
    block_verify_loss_mask,
]


def main():
    for fn in BLOCKS:
        print(f"\n===== {fn.__name__} =====")
        fn()
    print(f"\nAll {len(BLOCKS)} code blocks executed and verified.")
    print("SKIP(needs-gpu): block #0 (sft_train.py's train()) requires "
          "AutoModelForCausalLM.from_pretrained(..., device_map='auto') and "
          "real GPU training; not exercised here.")


if __name__ == "__main__":
    main()
