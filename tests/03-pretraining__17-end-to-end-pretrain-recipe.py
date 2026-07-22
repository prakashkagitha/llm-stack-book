"""
Runs the CPU-runnable Python blocks from
content/03-pretraining/17-end-to-end-pretrain-recipe.md, concatenated in
order. Each tested block is copied verbatim from the chapter; only the
minimal glue needed to make it actually execute (a tiny fixture file, a temp
working directory) is added and clearly marked "GLUE".

Blocks named by the task as heuristically CPU-runnable:

  #4  (line ~92)  -- tiktoken.get_encoding("gpt2") round-trip demo
  #11 (line ~351) -- parse a train.log with regex, plot loss with matplotlib

SKIP(network): block #4 (`import tiktoken; enc = tiktoken.get_encoding("gpt2")`).
Despite being pip-installed, tiktoken's "gpt2" encoding does NOT ship its
rank tables in the package -- `tiktoken_ext.openai_public` downloads
`encoder.json`/`vocab.bpe` from `openaipublic.blob.core.windows.net` (via
`tiktoken.load.read_file_cached` / `blobfile`) the first time `get_encoding`
is called on a machine, caching the result under a temp "data-gym-cache"
directory afterward. This is a real network call on a cold cache -- exactly
what CI (no network) cannot do -- and is already established as such
elsewhere in this repo's test suite (see
tests/02-transformer__01-tokenization.py, block #5, which skips the
identical call for the identical reason). The block's entire point is
checking the *real* GPT-2 BPE table's round-trip and its EOT id, so faking a
canned encoding to "run offline" would not honestly test what the block
demonstrates -- it is skipped outright rather than mocked, per the task's
own guidance to prefer SKIP over a mock when the mock would hide the very
thing being tested.

matplotlib is a third-party package not on the guaranteed-available list
(numpy, torch, einops, scikit-learn, stdlib), so its import is guarded;
block #11 only runs if it is importable, else it is SKIP(dependency).
"""

import os
import re
import tempfile

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    import matplotlib
    matplotlib.use("Agg")  # GLUE: headless backend, no display available in CI
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False


print("=" * 70)
print("Block #4 (line ~92): tiktoken.get_encoding('gpt2') round-trip demo")
print("=" * 70)
print("SKIP(network): tiktoken.get_encoding('gpt2') downloads "
      "encoder.json/vocab.bpe from openaipublic.blob.core.windows.net on a "
      "cold cache -- a real network call, forbidden in this harness (CI has "
      "no network). See module docstring for the full justification and the "
      "precedent already set in tests/02-transformer__01-tokenization.py.")


print()
print("=" * 70)
print("Block #10 (line ~324): mfu() 6N flops rule + worked example")
print("=" * 70)

# --- verbatim from the chapter ---
def mfu(num_params, tokens_per_sec, num_gpus, peak_tflops_per_gpu):
    flops_per_token = 6 * num_params                      # the 6N forward+backward rule
    achieved = flops_per_token * tokens_per_sec
    peak = num_gpus * peak_tflops_per_gpu * 1e12
    return achieved / peak

# Worked example: 124M model, single A100 (bf16 dense peak ~312 TFLOP/s).
u = mfu(num_params=124e6, tokens_per_sec=170_000, num_gpus=1, peak_tflops_per_gpu=312)
print(f"MFU = {u * 100:.1f}%")     # -> MFU = 40.5%
# --- end verbatim ---

# The chapter's comment claims this prints "MFU = 40.5%"; verify the book's
# arithmetic is right: 6*124e6*170000 / (312e12) = 0.4053...
assert f"{u * 100:.1f}" == "40.5", f"expected 40.5%, got {u * 100:.4f}%"


if _HAS_MPL:
    print()
    print("=" * 70)
    print("Block #11 (line ~351): parse train.log + plot loss curve")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # GLUE: a tiny fixture standing in for a real training run's stdout
        # log, in exactly the line format the chapter's regex expects:
        # "step  123 | loss 3.4521 | lr 3.21e-04 | 210 ms/step"
        log_path = os.path.join(tmpdir, "train.log")
        with open(log_path, "w") as f:
            f.write("step     0 | loss 10.8321 | lr 6.00e-06 | 480 ms/step\n")
            f.write("step   100 | loss 6.4213  | lr 3.00e-04 | 205 ms/step\n")
            f.write("some unrelated log line that should not match\n")
            f.write("step   500 | loss 4.1150  | lr 6.00e-04 | 198 ms/step\n")
            f.write("step  1000 | loss 3.5602  | lr 5.10e-04 | 201 ms/step\n")

        cwd = os.getcwd()
        os.chdir(tmpdir)  # GLUE: so the verbatim relative paths below resolve
        try:
            # --- verbatim from the chapter ---
            steps, losses = [], []
            with open("train.log") as f:
                for line in f:
                    m = re.search(r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)", line)
                    if m:
                        steps.append(int(m.group(1)))
                        losses.append(float(m.group(2)))

            plt.plot(steps, losses)
            plt.xlabel("step"); plt.ylabel("train loss"); plt.yscale("log")
            plt.savefig("loss_curve.png")
            # --- end verbatim ---

            print("parsed steps:", steps)
            print("parsed losses:", losses)

            assert steps == [0, 100, 500, 1000]
            assert losses == [10.8321, 6.4213, 4.1150, 3.5602]
            assert os.path.exists("loss_curve.png")
            assert os.path.getsize("loss_curve.png") > 0
            print("loss_curve.png written OK, size:",
                  os.path.getsize("loss_curve.png"), "bytes")
        finally:
            plt.close("all")
            os.chdir(cwd)
else:
    print()
    print("SKIP(dependency): matplotlib not importable; skipping block #11 "
          "(train.log parsing + loss-curve plot).")


print()
print("=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)
