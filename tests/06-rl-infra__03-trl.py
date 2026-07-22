"""
Runs the CPU-runnable Python blocks from content/06-rl-infra/03-trl.md end to end.

Blocks tested (source line numbers refer to the chapter as of writing):
  - Block #10 (~line 635): `composite_reward` / `extract_boxed_answer` -- the
    format + correctness + length-penalty reward function used by GRPOTrainer.
    Pure Python (re + typing.Any only); executed unconditionally and asserted
    against the reward semantics documented in its own docstring.
  - Block #8  (~line 545): `GRPOConfig(use_vllm=True, ...)` -- the vLLM-backend
    config snippet. Only object construction (no network call happens until
    generation actually runs against a live vLLM server), so it is safe to
    execute verbatim when `trl` is importable.
  - Block #11 (~line 684): `trainer = GRPOTrainer(model=model, args=grpo_config,
    train_dataset=dataset, reward_funcs=[composite_reward], processing_class=
    tokenizer, peft_config=lora_config)` -- wired up against a tiny, fully
    offline GPT-2-shaped model/tokenizer/dataset built as minimal glue (the
    chapter's own model/dataset/tokenizer come from block #4, which needs
    network access to download Qwen2.5-7B-Instruct and is skipped). We go one
    step further than mere construction and run a single `trainer.train()`
    step, so the composite_reward from block #10 is genuinely exercised by
    TRL's real GRPO loss/advantage machinery on CPU.

SKIPPED blocks (see chapter for full code):
  #0  SFTTrainer example                    - needs-gpu/needs-net (downloads Qwen2.5-7B-Instruct + ultrachat_200k)
  #1  RewardTrainer example                 - needs-net (downloads Qwen2.5-7B + Anthropic/hh-rlhf)
  #2  DPOTrainer example                    - needs-net (downloads Qwen2.5-7B-Instruct + ultrafeedback_binarized)
  #3  PPOTrainer example                    - needs-net (downloads gpt2 + lvwerra/distilbert-imdb, runs a generate loop)
  #4  GRPOTrainer main example              - needs-net (downloads Qwen2.5-7B-Instruct; requires flash-attn)
  #5  `accelerate launch ...`               - shell
  #6  `my_accelerate.yaml`                  - non-python (YAML)
  #7  QLoRA `BitsAndBytesConfig` example    - needs-gpu (bitsandbytes 4-bit requires CUDA)
  #9  TRL metric name listing               - non-python (plain text log keys)
  #12 pip install + accelerate launch CLI   - shell

Third-party imports beyond {numpy, torch, einops, sklearn, stdlib} are not
guaranteed in CI, so `transformers`, `tokenizers`, `datasets`, `trl`, and
`peft` are all guarded at module scope. If any is unavailable, the blocks
that need it are skipped explicitly (never silently) and the rest of the
file still runs and exits 0.
"""

import re
import sys
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast
except Exception:
    AutoModelForCausalLM = GPT2Config = PreTrainedTokenizerFast = None

try:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
except Exception:
    Tokenizer = WordLevel = Whitespace = None

try:
    from datasets import Dataset
except Exception:
    Dataset = None

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = GRPOTrainer = None

try:
    from peft import LoraConfig
except Exception:
    LoraConfig = None


# =====================================================================
# Block #10 (chapter line ~635-680): composite reward function.
# Copied verbatim from "Building a Custom Reward Function Pipeline".
# =====================================================================
def composite_reward(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """
    Composite reward for math reasoning:
    - +0.5 if completion contains a <think>...</think> block (format reward)
    - +1.0 if the boxed answer matches the ground truth (correctness reward)
    - -0.3 penalty if completion exceeds 800 tokens (length penalty)

    This mirrors the structure used in DeepSeek-R1-Zero and Qwen-QwQ.
    """
    ground_truths = kwargs.get("answer", [""] * len(completions))
    rewards = []

    for completion, gt in zip(completions, ground_truths):
        r = 0.0

        # --- Format reward ---
        has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        if has_think:
            r += 0.5

        # --- Correctness reward ---
        pred = extract_boxed_answer(completion)
        if pred is not None and pred.strip() == str(gt).strip():
            r += 1.0

        # --- Length penalty (rough token count via whitespace split) ---
        if len(completion.split()) > 800:
            r -= 0.3

        rewards.append(r)

    return rewards


def extract_boxed_answer(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def test_composite_reward():
    """Execute block #10's reward function with tiny inputs covering each branch."""
    prompts = ["What is 2 + 2?"] * 4

    completions = [
        # format + correct  -> 0.5 + 1.0 = 1.5
        r"<think>2+2=4</think> The answer is \boxed{4}.",
        # correct only, no <think> block -> 1.0
        r"The answer is \boxed{4}.",
        # format only, wrong answer -> 0.5
        r"<think>2+2=5</think> The answer is \boxed{5}.",
        # neither format nor correct, and long enough to trip the length penalty -> -0.3
        "word " * 801,
    ]
    ground_truths = ["4", "4", "4", "4"]

    rewards = composite_reward(prompts, completions, answer=ground_truths)

    assert rewards == [1.5, 1.0, 0.5, -0.3], rewards
    print(f"[block #10] composite_reward -> {rewards} (OK)")

    # extract_boxed_answer edge case: no \boxed{...} present at all.
    assert extract_boxed_answer("no boxed answer here") is None
    assert extract_boxed_answer(r"\boxed{42}") == "42"
    print("[block #10] extract_boxed_answer edge cases OK")


# =====================================================================
# Block #8 (chapter line ~545-555): GRPOConfig with the vLLM generation
# backend enabled. Object construction only -- no network call happens
# until a real vLLM server is contacted during generation, which we never
# trigger here.
# =====================================================================
def test_grpo_config_vllm_backend():
    if GRPOConfig is None:
        print("[block #8] SKIP(optional-dependency): trl is not installed")
        return

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_server_host="localhost",    # host running the vLLM inference server
        vllm_server_port=8000,
        vllm_server_timeout=120,         # seconds to wait for server readiness
        # vLLM generation kwargs
        temperature=1.0,
        top_p=0.95,
    )

    assert grpo_config.use_vllm is True
    assert grpo_config.vllm_server_host == "localhost"
    assert grpo_config.vllm_server_port == 8000
    assert grpo_config.vllm_server_timeout == 120
    assert grpo_config.temperature == 1.0
    assert grpo_config.top_p == 0.95
    print("[block #8] GRPOConfig(use_vllm=True, ...) constructed OK")


# =====================================================================
# Block #11 (chapter line ~684-694): trainer = GRPOTrainer(...) wired up
# with block #10's composite_reward as the (only) reward function.
#
# The chapter's own `model`, `dataset`, `tokenizer`, `lora_config`, and
# `grpo_config` come from block #4 (main GRPO example), which downloads
# Qwen/Qwen2.5-7B-Instruct from the Hub and needs `attn_implementation=
# "flash_attention_2"` -- both require network/GPU and that block is
# skipped. Here we substitute tiny, fully offline stand-ins (a 2-layer
# GPT-2-shaped model built from a local config, a hand-built WordLevel
# tokenizer, a 4-row toy Dataset) as minimal honest glue, then run
# block #11's own line verbatim against them.
# =====================================================================
def test_grpo_trainer_construction_and_step():
    missing = [
        name
        for name, obj in [
            ("trl", GRPOTrainer),
            ("transformers", AutoModelForCausalLM),
            ("tokenizers", Tokenizer),
            ("datasets", Dataset),
            ("peft", LoraConfig),
        ]
        if obj is None
    ]
    if missing:
        print(f"[block #11] SKIP(optional-dependency): missing {missing}")
        return

    torch.manual_seed(0)

    # --- Minimal offline tokenizer (stand-in for AutoTokenizer.from_pretrained) ---
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for w in "what is 2 plus 3 the answer boxed think 4 5".split():
        if w not in vocab:
            vocab[w] = len(vocab)
    tok_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )

    # --- Minimal offline model (stand-in for AutoModelForCausalLM.from_pretrained) ---
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=2,
        eos_token_id=2,
    )
    model = AutoModelForCausalLM.from_config(config)

    # --- Toy dataset (stand-in for the chapter's GSM8K-style prompt/answer set) ---
    dataset = Dataset.from_list([{"prompt": "what is 2 plus 3", "answer": "5"}] * 8)

    # --- grpo_config (stand-in for block #4's grpo_config, scaled to CPU/tiny) ---
    grpo_config = GRPOConfig(
        output_dir="/tmp/trl-chapter-test-grpo-out",
        num_generations=2,
        max_completion_length=8,
        temperature=0.9,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=1,
        learning_rate=1e-6,
        bf16=False,
        logging_steps=1,
        save_steps=50,
        use_vllm=False,
        report_to=[],
    )

    # --- lora_config (stand-in for block #4's lora_config; target the GPT-2 attn proj) ---
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- Block #11, verbatim ----
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        # List of reward functions — TRL sums their outputs
        reward_funcs=[composite_reward],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    # ---- end block #11 ----

    assert isinstance(trainer, GRPOTrainer)
    print("[block #11] GRPOTrainer instantiated OK")

    # Go one step further than mere construction: actually run one GRPO
    # training step, so composite_reward (block #10) is exercised for real
    # inside TRL's rollout/advantage/loss machinery, on CPU.
    train_result = trainer.train()
    assert trainer.state.global_step >= 1
    print(f"[block #11] trainer.train() ran {trainer.state.global_step} step(s) OK ({train_result})")


def main():
    test_composite_reward()
    test_grpo_config_vllm_backend()
    test_grpo_trainer_construction_and_step()
    print("\nAll checks passed (or honestly skipped).")


if __name__ == "__main__":
    main()
