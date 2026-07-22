"""
Executable test for content/99-appendix/04-tooling-setup.md

Concatenates the chapter's 4 CPU-runnable Python blocks in order and exercises
each one on CPU / with tiny fixtures so the book's actual code runs end to end.

Blocks covered:
  #5  (line ~152) dtype cheatsheet -- per-parameter memory cost table for a
                  7B model across torch dtypes. Runs verbatim (pure torch,
                  no GPU needed).
  #21 (line ~602) SGLang structured-generation example (`sgl.function` /
                  `sgl.gen` / `run_batch`). `sglang` is not installed in this
                  environment (nor in the guaranteed CI import set), and even
                  if it were, running it for real means talking to a live
                  model-serving backend (network). We substitute a tiny,
                  shape-correct offline fake for just the `sgl` surface this
                  block touches (system/user/assistant/gen, the
                  `@sgl.function` decorator, `.run_batch`) -- the same
                  "stub the model/runtime, keep the book's own logic" pattern
                  used for SentenceTransformer/AutoModel stubs elsewhere. The
                  book's `classify_sentiment` function and its batched,
                  forked call run verbatim against that fake backend.
  #23 (line ~688) GRPO with TRL -- `reward_fn` is pure Python (no external
                  dependency) and is the actual interesting logic of this
                  block, so it is copied verbatim and CALLED with tiny sample
                  completions. `GRPOConfig`/`GRPOTrainer` construction and
                  `trainer.train()` are intentionally NOT instantiated: they
                  need a real model, dataset and GPU, and GRPOConfig's
                  accepted kwargs are version-fragile across trl releases
                  (confirmed by hand against the trl installed here: the
                  book's `max_new_tokens` kwarg is not a valid GRPOConfig
                  field in trl 1.9.0, which renamed/removed it). Only the
                  pure-Python reward_fn -- the actually CPU-testable logic in
                  this block -- is exercised.
  #24 (line ~724) wandb experiment-tracking loop -- `wandb` is an optional
                  third-party package that talks to a network API by
                  default, and is blocked outright under ci_sim. We ALWAYS
                  substitute a tiny in-memory fake module (init/log/finish/
                  Table) regardless of whether the real package happens to be
                  importable, so no network call can ever be attempted. The
                  block's own logic (the metric-logging loop, the eval Table
                  construction) executes verbatim and offline against canned
                  fixtures for `dataloader`, `train_step`, `scheduler`,
                  `grad_norm`, `tokens_per_sec`, and `eval_samples` (not
                  defined within this chapter's shown blocks -- minimal
                  honest glue).

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

import types

import torch

try:
    import trl  # noqa: F401  (only used to sanity-check attrs exist; never called)
except Exception:
    trl = None

try:
    import sglang  # noqa: F401  (never actually used -- see block #21 below)
except Exception:
    sglang = None

try:
    import wandb  # noqa: F401  (never actually used -- see block #24 below)
except Exception:
    wandb = None


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
# Block #21 (line ~602) -- SGLang structured generation
# ============================================================
# `import sglang as sgl` -- sglang isn't installed here (and isn't part of
# CI's guaranteed set either), and even if it were, `sgl.function`/
# `run_batch` need a live model-serving backend to produce anything. We
# ALWAYS use a tiny offline fake for just the module surface this block
# touches, so no network/model call is ever attempted, and so the book's own
# `classify_sentiment` function + batched call run for real against it.

class _FakeGenSpec:
    def __init__(self, name, choices=None, max_tokens=None):
        self.name = name
        self.choices = choices
        self.max_tokens = max_tokens


class _FakeSglState(dict):
    def __iadd__(self, other):
        if isinstance(other, _FakeGenSpec):
            # Stand-in for constrained decoding: deterministically pick the
            # first allowed choice (a real engine would sample under the
            # choice-list constraint against actual model logits).
            self[other.name] = other.choices[0] if other.choices else None
        # system()/user()/assistant(str) contributions are plain strings and
        # aren't needed for this test's assertions.
        return self


class _FakeSglFunction:
    def __init__(self, fn):
        self._fn = fn

    def run_batch(self, kwargs_list, progress_bar=False):
        states = []
        for kwargs in kwargs_list:
            state = _FakeSglState()
            self._fn(state, **kwargs)
            states.append(state)
        return states


sgl = types.SimpleNamespace(
    function=lambda fn: _FakeSglFunction(fn),
    system=lambda text: text,
    user=lambda text: text,
    assistant=lambda x: x,
    gen=lambda name, choices=None, max_tokens=None: _FakeGenSpec(name, choices, max_tokens),
)

# SGLang's structured generation: constrain output to a JSON schema
@sgl.function
def classify_sentiment(s, review):
    s += sgl.system("You are a sentiment classifier.")
    s += sgl.user(review)
    s += sgl.assistant(
        sgl.gen("sentiment",
                choices=["positive", "negative", "neutral"],  # constrained
                max_tokens=1)
    )

# Fork for parallel evaluation of multiple reviews
reviews = ["This movie was great!", "Terrible service.", "It was okay."]
states = classify_sentiment.run_batch(
    [{"review": r} for r in reviews],
    progress_bar=True,
)
for state, review in zip(states, reviews):
    print(f"{review!r} -> {state['sentiment']}")

assert len(states) == len(reviews)
assert all(state["sentiment"] in ["positive", "negative", "neutral"] for state in states)
print("[block #21] sglang @function/run_batch pattern OK (offline fake backend)")


# ============================================================
# Block #23 (line ~688) -- GRPO reward function (from TRL example)
# ============================================================
# from trl import GRPOTrainer, GRPOConfig  (guarded above; GRPOTrainer/
# GRPOConfig deliberately never instantiated -- see module docstring)

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

# SKIP(needs-gpu, version-fragile): GRPOConfig(...) / GRPOTrainer(...) /
# .train() are intentionally NOT instantiated here, even when `trl` happens
# to be installed. GRPOConfig's accepted kwargs have changed across trl
# releases (e.g. `max_new_tokens` is not a valid GRPOConfig field in trl
# 1.9.0 -- confirmed by hand), and GRPOTrainer.train() needs a real model,
# dataset, and GPU regardless of config-shape issues.
print("[block #23] SKIP(needs-gpu, version-fragile): GRPOConfig/GRPOTrainer "
      "construction and trainer.train() skipped")


# ============================================================
# Block #24 (line ~724) -- wandb experiment-tracking loop
# ============================================================
# The book's block does `import wandb`, calls wandb.init/log/finish, and
# references `dataloader`, `train_step`, `scheduler`, `grad_norm`,
# `tokens_per_sec`, `eval_samples` from earlier (unshown, GPU-only) chapter
# context. We ALWAYS use a tiny in-memory fake wandb object -- regardless of
# whether the real `wandb` package happens to be importable -- so no network
# call is ever attempted, and supply minimal honest CPU fixtures for the
# training-loop names the block assumes are already in scope.

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


class _FakeWandbModule:
    Table = _FakeWandbTable

    def __init__(self):
        self.run = None

    def init(self, **kwargs):
        self.run = _FakeWandbRun(**kwargs)
        return self.run

    def log(self, metrics, step=None):
        assert self.run is not None, "wandb.log called before wandb.init"
        self.run.log(metrics, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()


wandb = _FakeWandbModule()  # offline fake -- see comment above; never the real package

import os

# Minimal honest fixtures standing in for earlier (unshown / GPU-only)
# training-loop state that block #24 assumes is already in scope.
dataloader = [{"x": 1.0}, {"x": 2.0}, {"x": 0.5}]


def train_step(batch):
    return torch.tensor(batch["x"] * 0.1)


class _FixedScheduler:
    def get_last_lr(self):
        return [2e-4]


scheduler = _FixedScheduler()
grad_norm = 0.83
tokens_per_sec = 12345.0
eval_samples = [
    ("What is 2+2?", "4", 1.0),
    ("Capital of France?", "Paris", 1.0),
    ("Name a color.", "green", 0.5),
]

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

_logged = wandb.run.logged
metric_logs = [m for (s, m) in _logged if "train/loss" in m]
assert len(metric_logs) == 1  # only step 0 of 3 tiny batches hits step % 10 == 0
assert abs(metric_logs[0]["train/loss"] - 0.1) < 1e-6  # train_step(batch0)=0.1*1.0
table_logs = [m for (s, m) in _logged if "eval/samples" in m]
assert len(table_logs) == 1
assert table_logs[0]["eval/samples"].columns == ["prompt", "generation", "reward"]
assert len(table_logs[0]["eval/samples"].data) == 3
print(f"[block #24] wandb loop OK, logged {len(_logged)} events "
      f"(metrics={metric_logs[0]}, table_rows={len(table_logs[0]['eval/samples'].data)})")


print("\nAll runnable blocks in 99-appendix/04-tooling-setup.md executed successfully.")
