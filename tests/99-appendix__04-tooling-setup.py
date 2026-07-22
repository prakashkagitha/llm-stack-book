"""
Executable test for content/99-appendix/04-tooling-setup.md

Concatenates the chapter's 4 CPU-runnable Python blocks in order and exercises
each one on CPU / with tiny fixtures so the book's actual code runs end to end.

Blocks covered:
  #5  (line ~152) dtype cheatsheet -- per-parameter memory cost table for a
                  7B model across torch dtypes. Runs verbatim (pure torch,
                  no GPU needed).
  #21 (line ~602) SGLang structured-generation example (`sgl.function` /
                  `sgl.gen` / `run_batch`). `sglang` is not in the guaranteed
                  CI import set and the block's whole point is orchestrating
                  a *live* inference backend (constrained decoding over a
                  running model) -- there is no boundary we can mock without
                  reimplementing SGLang's runtime, which would bypass the
                  very logic being demonstrated. SKIPPED, guarded import only.
  #23 (line ~688) GRPO with TRL -- `reward_fn` is pure Python (no external
                  dependency) and is the actual interesting logic of this
                  block, so it is copied verbatim and CALLED with tiny sample
                  completions. `GRPOConfig`/`GRPOTrainer` come from the
                  optional `trl` package (guarded import); config
                  construction is exercised when `trl` is installed, and the
                  actual `trainer.train()` call is always skipped since it
                  needs a real model, dataset and GPU.
  #24 (line ~724) wandb experiment-tracking loop -- `wandb` is an optional
                  third-party package that talks to a network API by
                  default. We patch `sys.modules['wandb']` with a small fake
                  module (init/log/finish/Table) so the block's own logic
                  (the metric-logging loop, the eval Table construction)
                  executes verbatim and offline against canned fixtures for
                  `dataloader`, `train_step`, `scheduler`, `grad_norm`,
                  `tokens_per_sec`, and `eval_samples` (not defined within
                  this chapter's shown blocks -- minimal honest glue).

Blocks intentionally SKIPPED (per task spec):
  #0, #1, #2, #3, #4   -- shell / conda / env-var snippets
  #6, #7, #8, #10      -- needs GPU (torch.compile CUDA warmup, CUDA
                          profiler, `device_map="cuda"` model load, real
                          multi-GPU `accelerate` training loop)
  #9                   -- needs network (streams a real HF dataset)
  #11, #12             -- shell (launch commands / `accelerate config`)
  #13, #14             -- non-python (DeepSpeed ZeRO JSON configs)
  #15                  -- shell (flash-attn build/verify heredoc)
  #16, #17             -- needs GPU (`attn_implementation="flash_attention_2"`
                          load, bitsandbytes 4-bit model load)
  #18                  -- shell (vLLM server + curl)
  #19                  -- needs GPU (real vLLM `LLM(...)` load + generate)
  #20                  -- shell (SGLang server launch)
  #22                  -- needs GPU (full TRL `SFTTrainer` example: loads a
                          real 8B model and calls `trainer.train()`)
  #25                  -- shell (`wandb login` / offline env vars)
  #26                  -- needs network (`wandb.Api()` hits the real service)
  #27                  -- shell (nvidia-smi / NCCL diagnostic heredoc)
  #28                  -- needs GPU (`torch.cuda.memory_allocated()` etc.)
  #29                  -- non-python (error/fix reference text block)
  #30                  -- shell (one-liner reference commands)
"""

import sys
import unittest.mock as mock

import torch

try:
    from trl import GRPOConfig  # noqa: F401  (GRPOTrainer itself never invoked)
except Exception:
    GRPOConfig = None

try:
    import sglang as sgl  # noqa: F401
except Exception:
    sgl = None
# SKIP(import + needs live backend): sglang is not in the guaranteed CI
# import set, and `sgl.function` / `run_batch` require a running model
# server to produce anything -- there is no offline boundary to mock without
# reimplementing SGLang's execution engine. Block #21 is left
# defined-not-called.


# ============================================================
# Block #5 (line ~152) -- dtype cheatsheet for LLM memory footprints
# ============================================================
# import torch  (already imported above; book repeats this import per-block)

# Memory cost per parameter for common dtypes
dtype_bytes = {
    torch.float32: 4,   # full precision (baseline)
    torch.bfloat16: 2,  # preferred for LLM training on Ampere+
    torch.float16: 2,   # legacy; careful with gradient underflow
    torch.int8: 1,      # post-training quantization
    torch.float8_e4m3fn: 1,  # FP8, Hopper/Ada only (PyTorch >= 2.1)
}

# 7B model parameter memory at different dtypes
n_params = 7_000_000_000
computed_gb = {}
for dtype, bytes_per_param in dtype_bytes.items():
    gb = n_params * bytes_per_param / 1e9
    print(f"{dtype}: {gb:.1f} GB")
    computed_gb[dtype] = gb

# Output:
# torch.float32:   28.0 GB
# torch.bfloat16:  14.0 GB
# torch.float16:   14.0 GB
# torch.int8:       7.0 GB
# torch.float8_e4m3fn: 7.0 GB

assert computed_gb[torch.float32] == 28.0
assert computed_gb[torch.bfloat16] == 14.0
assert computed_gb[torch.float16] == 14.0
assert computed_gb[torch.int8] == 7.0
assert computed_gb[torch.float8_e4m3fn] == 7.0
print("[block #5] dtype cheatsheet OK")


# ============================================================
# Block #23 (line ~688) -- GRPO reward function (from TRL example)
# ============================================================
# from trl import GRPOTrainer, GRPOConfig  (guarded above; GRPOTrainer unused
# here since trainer.train() always needs a real model/dataset/GPU)

def reward_fn(completions, prompts, **kwargs):
    """Custom reward: give +1 if response contains a number, else 0."""
    return [1.0 if any(c.isdigit() for c in comp) else 0.0
            for comp in completions]

# Exercise the book's actual reward logic with tiny sample completions.
sample_prompts = ["What is 2+2?", "Say hello."]
sample_completions = ["The answer is 4.", "Hello there, no digits here."]
rewards = reward_fn(sample_completions, sample_prompts)
assert rewards == [1.0, 0.0], rewards
print(f"[block #23] reward_fn({sample_completions!r}) -> {rewards}")

if GRPOConfig is not None:
    # trl is installed: cheap, GPU-free part of the block (building the
    # config dataclass) can run for real too.
    grpo_config = GRPOConfig(
        output_dir="/tmp/llama3-grpo-test",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-6,
        bf16=False,
        num_generations=8,
        max_new_tokens=256,
        report_to=[],
    )
    print(f"[block #23] GRPOConfig constructed: output_dir={grpo_config.output_dir}")
else:
    print("[block #23] SKIP(import): trl not installed -- GRPOConfig/"
          "GRPOTrainer construction and trainer.train() skipped "
          "(needs a real model + dataset + GPU regardless)")


# ============================================================
# Block #24 (line ~724) -- wandb experiment-tracking loop
# ============================================================
# The book's block does `import wandb`, calls wandb.init/log/finish, and
# references `dataloader`, `train_step`, `scheduler`, `grad_norm`,
# `tokens_per_sec`, `eval_samples` from earlier (unshown, GPU-only) chapter
# context. We supply minimal honest CPU fixtures for those names and patch
# `sys.modules["wandb"]` with a tiny fake module so no network call is ever
# made (whether or not the real `wandb` package happens to be installed),
# then exec the block's code VERBATIM so its own logging logic runs for real.

class _FakeWandbTable:
    def __init__(self, columns=None, data=None):
        self.columns = columns
        self.data = data


class _FakeWandbRun:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((step, metrics))

    def finish(self):
        pass


_fake_run = _FakeWandbRun()
_fake_wandb = mock.MagicMock()
_fake_wandb.init = mock.MagicMock(return_value=_fake_run)
_fake_wandb.log = _fake_run.log
_fake_wandb.finish = _fake_run.finish
_fake_wandb.Table = _FakeWandbTable

# Minimal honest fixtures standing in for earlier (unshown / GPU-only)
# training-loop state that block #24 assumes is already in scope.
_dataloader = [{"x": 1.0}, {"x": 2.0}, {"x": 0.5}]


def _train_step(batch):
    return torch.tensor(batch["x"] * 0.1)


class _FixedScheduler:
    def get_last_lr(self):
        return [2e-4]


_scheduler = _FixedScheduler()
_grad_norm = 0.83
_tokens_per_sec = 12345.0
_eval_samples = [
    ("What is 2+2?", "4", 1.0),
    ("Capital of France?", "Paris", 1.0),
    ("Name a color.", "green", 0.5),
]

_block_24_src = '''
import wandb
import os

# --- Initialize a run ---
wandb.init(
    project="llm-stack-experiments",
    name="llama3-sft-lora-r16",
    config={
        "model": "meta-llama/Meta-Llama-3-8B",
        "lora_r": 16,
        "lr": 2e-4,
        "batch_size": 16,
        "epochs": 3,
        "dataset": "ultrachat_200k",
    },
    tags=["sft", "lora", "llama3"],
)

# --- During training: log metrics ---
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    if step % 10 == 0:
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm,
            "train/tokens_per_sec": tokens_per_sec,
        }, step=step)

# --- Log model outputs as a Table for qualitative eval ---
columns = ["prompt", "generation", "reward"]
rows = []
for prompt, gen, reward in eval_samples:
    rows.append([prompt, gen, reward])

wandb.log({"eval/samples": wandb.Table(columns=columns, data=rows)})

# --- Finish the run ---
wandb.finish()
'''

with mock.patch.dict(sys.modules, {"wandb": _fake_wandb}):
    exec(_block_24_src, {
        "dataloader": _dataloader,
        "train_step": _train_step,
        "scheduler": _scheduler,
        "grad_norm": _grad_norm,
        "tokens_per_sec": _tokens_per_sec,
        "eval_samples": _eval_samples,
    })

# The loop above logs at step % 10 == 0 -> only step 0 out of 3 tiny batches.
metric_logs = [m for (s, m) in _fake_run.logged if "train/loss" in m]
assert len(metric_logs) == 1
assert abs(metric_logs[0]["train/loss"] - 0.1) < 1e-6  # train_step(batch0)=0.1*1.0
table_logs = [m for (s, m) in _fake_run.logged if "eval/samples" in m]
assert len(table_logs) == 1
assert table_logs[0]["eval/samples"].columns == ["prompt", "generation", "reward"]
assert len(table_logs[0]["eval/samples"].data) == 3
print(f"[block #24] wandb loop OK, logged {len(_fake_run.logged)} events "
      f"(metrics={metric_logs[0]}, table_rows={len(table_logs[0]['eval/samples'].data)})")


print("\nAll runnable blocks in 99-appendix/04-tooling-setup.md executed successfully.")
