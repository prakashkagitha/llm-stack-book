#!/usr/bin/env python3
"""B (GPU-verify): execute the SINGLE-GPU jupytext-percent notebook sources on a real GPU to
confirm the book's GPU code actually runs. Strips %pip/%magic/! lines; skips %%writefile cells
(those are the multi-GPU torchrun notebooks, handled separately). Logs pass/fail + tail per
notebook to notebooks-gpu/exec_results.json. Pin with CUDA_VISIBLE_DEVICES.

Usage: CUDA_VISIBLE_DEVICES=0 python3 scripts/exec_gpu_notebooks.py [slug ...]
"""
import os, sys, re, json, subprocess, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "notebooks-gpu", "src")
VENV_PY = "/local-ssd/pk669/gpuverify-venv/bin/python"   # has bitsandbytes etc.
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable

# single-GPU, no-network notebooks (skip multi-GPU torchrun + weight-download ones by default)
SINGLE_GPU = [
    "04-kernels-efficiency__flash-attention-benchmark",
    "04-kernels-efficiency__triton-fused-kernel",
    "04-kernels-efficiency__torch-compile-speedup",
    "04-kernels-efficiency__int4-int8-quantization",
    "04-kernels-efficiency__memory-efficient-training",
    "03-pretraining__bf16-vs-fp8-throughput",
    "03-pretraining__optimizers-wallclock",
]


def to_script(src_text):
    """percent source -> runnable python: drop %pip/%magic/!shell lines and %%writefile cells."""
    lines = src_text.splitlines()
    out, skip_cell = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("# %%"):
            skip_cell = False          # new cell resets skip
            continue                   # drop the cell marker itself
        if s.startswith("%%writefile") or s.startswith("%%bash"):
            skip_cell = True           # skip the whole writefile/bash cell body
            continue
        if skip_cell:
            continue
        if re.match(r"\s*[%!]", ln):    # line magics / shell escapes
            continue
        out.append(ln)
    return "\n".join(out)


def run_one(slug, timeout=420):
    src = os.path.join(SRC, slug + ".py")
    if not os.path.exists(src):
        return {"slug": slug, "status": "MISSING"}
    script = to_script(open(src).read())
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(script)
        path = f.name
    t0 = time.time()
    try:
        p = subprocess.run([PY, path], capture_output=True, text=True, timeout=timeout,
                           cwd="/tmp", env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        ok = p.returncode == 0
        tail = (p.stdout + "\n" + p.stderr).strip().splitlines()[-6:]
        res = {"slug": slug, "status": "PASS" if ok else "FAIL", "rc": p.returncode,
               "seconds": round(time.time() - t0, 1), "tail": tail}
    except subprocess.TimeoutExpired:
        res = {"slug": slug, "status": "TIMEOUT", "seconds": timeout}
    finally:
        os.unlink(path)
    return res


def main():
    slugs = sys.argv[1:] or SINGLE_GPU
    print(f"[exec] python={PY} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')} | {len(slugs)} notebooks", flush=True)
    results = []
    for slug in slugs:
        print(f"[exec] running {slug} ...", flush=True)
        r = run_one(slug)
        print(f"[exec]   -> {r['status']} ({r.get('seconds','?')}s)", flush=True)
        if r["status"] in ("FAIL", "TIMEOUT"):
            for t in r.get("tail", []):
                print("        " + t, flush=True)
        results.append(r)
    out = os.path.join(ROOT, "notebooks-gpu", "exec_results.json")
    json.dump(results, open(out, "w"), indent=2)
    npass = sum(1 for r in results if r["status"] == "PASS")
    print(f"[exec] DONE: {npass}/{len(results)} PASS -> {out}", flush=True)
    print("EXEC_GPU_DONE", flush=True)


if __name__ == "__main__":
    main()
