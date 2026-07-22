"""
Executable test for content/03-pretraining/07-megatron-deepspeed.md

Concatenates the chapter's 4 CPU-runnable Python blocks in order and exercises
each one on CPU / with tiny fixtures so the book's actual code runs end to end.

Blocks covered:
  #1 (line ~129) DeepSpeed ZeRO-3 config JSON construction + dump
  #2 (line ~169) TinyTransformerBlock model + deepspeed.initialize (deepspeed
                 is an optional third-party dep -- guarded import; the pure
                 PyTorch parts of the block (model build + forward + backward)
                 always run, deepspeed.initialize() only runs if the package
                 is actually installed)
  #3 (line ~346) compute_mfu() -- MFU calc + the worked 70B/512-H100 example
  #5 (line ~437) MODEL_CONFIG dict (Llama-style 70B architecture config)
  #8 (line ~565) parse_megatron_logs.py -- regex + toks_per_sec_to_mfu()

Blocks intentionally SKIPPED (per task spec):
  #0  -- needs GPU (NCCL / real multi-GPU process groups)
  #4  -- non-python (cluster topology text block)
  #6  -- shell (SLURM launch script)
  #7  -- non-python (DeepSpeed config JSON, not python code -- shown as a
         plain JSON listing, no `python` fence)
  #9  -- shell (nsys profile command)
  #10 -- needs GPU (Megatron-Core + real torch.distributed init_process_group)
"""

import json
import os
import sys
import tempfile

import torch

try:
    import deepspeed
except Exception:
    deepspeed = None


# ============================================================
# Block #1 (line ~129) -- DeepSpeed config JSON for ZeRO-3 with CPU offload
# ============================================================
# import json  (already imported above; book repeats this import per-block)

zero3_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",       # optimizer states live on CPU RAM
            "pin_memory": True     # page-locked for fast DMA transfer
        },
        "offload_param": {
            "device": "cpu",       # fp16 params also offloaded
            "pin_memory": True
        },
        "overlap_comm": True,      # overlap reduce-scatter with backward pass
        "contiguous_gradients": True,
        "sub_group_size": 1e9,     # process params in 1B-element chunks
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,           # dynamic loss scaling
        "loss_scale_window": 1000
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
}

# NOTE: the book writes to a fixed relative path "ds_config_zero3.json"; we
# redirect to a temp dir so the test doesn't litter the repo, but the dumped
# object and json.dump call are exactly the book's code.
_tmpdir = tempfile.mkdtemp(prefix="megatron_ds_test_")
_ds_config_path = os.path.join(_tmpdir, "ds_config_zero3.json")
with open(_ds_config_path, "w") as f:
    json.dump(zero3_config, f, indent=2)


# ============================================================
# Block #2 (line ~169) -- Initializing a DeepSpeed Engine
# ============================================================
import torch  # noqa: F811 (book repeats this import per-block, kept verbatim)
import torch.nn as nn

class TinyTransformerBlock(nn.Module):
    """A minimal transformer block for demonstration."""
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x):
        # Pre-norm residual style (GPT-style)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Block #3 (line ~346) -- compute_mfu(): Model FLOP Utilization
# ============================================================
def compute_mfu(
    model_params: int,        # number of parameters
    tokens_per_second: float, # observed training throughput
    peak_flops_per_sec: float, # e.g., 5.06e17 for dense bf16 on 512 H100 SXM5 GPUs
    n_layers: int = None,
    d_model: int = None,
    seq_len: int = None,
) -> float:
    """
    Compute Model FLOP Utilization.

    Uses the simplified 6P rule for forward+backward FLOPs per token.
    If n_layers, d_model, seq_len are provided, also adds attention cost.
    """
    flops_per_token = 6 * model_params

    # Attention cost (forward + backward): 12 * n_layers * d_model * seq_len per token.
    # Forward is 4 * d * S per layer (QK^T and AV, each 2 * d * S); x3 for the
    # backward pass, matching the fwd+bwd convention of the 6P base term.
    if all(v is not None for v in [n_layers, d_model, seq_len]):
        attn_flops = 12 * n_layers * d_model * seq_len
        flops_per_token += attn_flops

    achieved_flops = flops_per_token * tokens_per_second
    mfu = achieved_flops / peak_flops_per_sec
    return mfu


# ============================================================
# Block #5 (line ~437) -- model_config.py: Llama-style 70B architecture
# ============================================================
MODEL_CONFIG = {
    "num_layers": 80,
    "hidden_size": 8192,
    "ffn_hidden_size": 28672,   # ~3.5x hidden_size for SwiGLU
    "num_attention_heads": 64,
    "num_key_value_heads": 8,   # GQA with 8 KV heads
    "max_position_embeddings": 8192,
    "vocab_size": 128256,       # Llama-3 vocabulary
    "activation_function": "swiglu",
    "normalization": "rmsnorm",
    "tie_embeddings": False,
}


# ============================================================
# Block #8 (line ~565) -- parse_megatron_logs.py: extract MFU from stdout
# ============================================================
import re
# import sys  (already imported above; book repeats this import per-block)

LOG_LINE_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([\d.]+).*?"
    r"tokens-per-second-per-gpu: ([\d.]+)",
    re.DOTALL,
)

H100_BF16_PEAK_TFLOPS = 989.0  # dense bf16 Tensor Core, per GPU (1979 is the 2:4-sparse rate)
MODEL_PARAMS = 70e9

def toks_per_sec_to_mfu(tps_per_gpu: float) -> float:
    flops_per_tok = 6 * MODEL_PARAMS
    achieved_tflops = flops_per_tok * tps_per_gpu / 1e12
    return achieved_tflops / H100_BF16_PEAK_TFLOPS

def parse_log_stream(stream):
    """Book's `for line in sys.stdin: ...` loop, factored into a function so
    the test can feed it a canned in-memory log instead of real stdin."""
    results = []
    for line in stream:
        m = LOG_LINE_RE.search(line)
        if m:
            iteration = int(m.group(1))
            ms_per_iter = float(m.group(2))
            tps = float(m.group(3))
            mfu = toks_per_sec_to_mfu(tps)
            msg = (f"iter {iteration:6d} | {ms_per_iter:6.0f} ms/it | "
                   f"{tps:5.0f} tok/s/gpu | MFU {mfu:.1%}")
            print(msg)
            results.append((iteration, ms_per_iter, tps, mfu))
    return results


# ============================================================
# Driver -- exercise every block above with tiny CPU-safe inputs
# ============================================================

def main():
    torch.manual_seed(0)

    # --- Block #1: DeepSpeed ZeRO-3 config JSON -----------------------------
    assert os.path.exists(_ds_config_path)
    with open(_ds_config_path) as f:
        reloaded = json.load(f)
    assert reloaded["zero_optimization"]["stage"] == 3
    assert reloaded["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert reloaded["train_micro_batch_size_per_gpu"] == 2
    assert reloaded["gradient_accumulation_steps"] == 8
    print(f"[OK] block #1 ZeRO-3 config dumped to {_ds_config_path} and reloaded")

    # --- Block #2: TinyTransformerBlock + (optional) deepspeed.initialize ---
    # Tiny shapes so this stays CPU-fast: d_model=32, n_heads=4, 2 layers
    # (book uses d_model=1024, n_heads=16, 24 layers -- infeasible on CPU).
    d_model, n_heads, n_layers = 32, 4, 2
    model = nn.Sequential(*[TinyTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    x = torch.randn(2, 8, d_model)  # (batch, seq, d_model)
    out = model(x)
    assert out.shape == x.shape
    loss = out.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    print(f"[OK] block #2 TinyTransformerBlock stack ran forward+backward, loss={loss.item():.4f}")

    if deepspeed is not None:
        # deepspeed.initialize wraps the model, optimizer, and dataloader
        # into a DeepSpeedEngine that handles ZeRO sharding transparently.
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=_ds_config_path,
        )
        out = model_engine(x)
        loss = out.pow(2).mean()
        model_engine.backward(loss)
        model_engine.step()
        print("[OK] block #2 deepspeed.initialize + engine step ran (deepspeed installed)")
    else:
        print("[SKIP] block #2 deepspeed.initialize -- 'deepspeed' package not installed "
              "in this CI environment; exercised the pure-PyTorch model/train-step "
              "portion of the block above instead.")

    # --- Block #3: compute_mfu() + the book's worked 70B/512-H100 example ----
    # Reproduces the exact example from the chapter, whose comment claims the
    # print shows ~54.9% (~55%) MFU including the attention term.
    H100_BF16_TFLOPS = 9.89e14
    n_gpus = 512
    peak_cluster_flops = H100_BF16_TFLOPS * n_gpus  # ~5.06e17 FLOP/s
    tokens_per_sec = 1200 * n_gpus
    mfu = compute_mfu(
        model_params=70e9,
        tokens_per_second=tokens_per_sec,
        peak_flops_per_sec=peak_cluster_flops,
        n_layers=80,
        d_model=8192,
        seq_len=4096,
    )
    assert 0.54 < mfu < 0.56, f"book claims ~54.9% MFU; got {mfu:.4%}"
    # 6P-only variant (no attention term) must be slightly lower but same order.
    mfu_6p = compute_mfu(70e9, tokens_per_sec, peak_cluster_flops)
    assert mfu_6p < mfu, "attention term should raise MFU above the 6P-only estimate"
    assert 0.50 < mfu_6p < 0.52
    print(f"[OK] block #3 compute_mfu MFU={mfu:.2%} (6P-only={mfu_6p:.2%})")

    # --- Block #5: MODEL_CONFIG dict -----------------------------------------
    assert MODEL_CONFIG["num_layers"] == 80
    assert MODEL_CONFIG["hidden_size"] == 8192
    assert MODEL_CONFIG["num_attention_heads"] % MODEL_CONFIG["num_key_value_heads"] == 0
    assert MODEL_CONFIG["vocab_size"] == 128256
    print(f"[OK] block #5 MODEL_CONFIG = {MODEL_CONFIG}")

    # --- Block #8: parse_megatron_logs.py ------------------------------------
    # Canned stdin replacement (no real stdin/network); matches the log format
    # the LOG_LINE_RE regex expects from real Megatron-LM stdout.
    fake_log = [
        "iteration      100/  500000 | consumed samples: 51200 | elapsed time per "
        "iteration (ms): 812.3 | learning rate: 3.000E-04 | tokens-per-second-per-gpu: 1210.5\n",
        "some unrelated log line that should not match\n",
        "iteration      200/  500000 | consumed samples: 102400 | elapsed time per "
        "iteration (ms): 799.1 | learning rate: 3.000E-04 | tokens-per-second-per-gpu: 1235.0\n",
    ]
    results = parse_log_stream(fake_log)
    assert len(results) == 2, f"expected 2 matched log lines, got {len(results)}"
    iters = [r[0] for r in results]
    assert iters == [100, 200]
    for _, _, tps, mfu in results:
        expected_mfu = toks_per_sec_to_mfu(tps)
        assert abs(mfu - expected_mfu) < 1e-12
        assert 0.0 < mfu < 1.0, f"MFU {mfu} outside plausible (0,1) range for tps={tps}"
    print(f"[OK] block #8 parse_megatron_logs matched {len(results)} lines, "
          f"MFUs={[f'{r[3]:.1%}' for r in results]}")

    print("\nAll CPU-runnable blocks in 03-pretraining/07-megatron-deepspeed.md executed OK.")


if __name__ == "__main__":
    main()
