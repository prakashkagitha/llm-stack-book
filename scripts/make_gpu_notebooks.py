#!/usr/bin/env python3
"""Convert jupytext 'percent'-format sources under notebooks-gpu/src/ into .ipynb.

Source format (one file per notebook), cells delimited by:
  # %% [markdown]      -> a markdown cell; following '# '-prefixed lines are its source
  # %%                 -> a code cell; following lines are literal Python
The filename encodes the target path: notebooks-gpu/src/<part>__<slug>.py
  -> notebooks-gpu/<part>/<slug>.ipynb

Also (re)writes notebooks-gpu/README.md as an index. No execution — outputs are empty.
Usage: python3 scripts/make_gpu_notebooks.py
"""
import glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "notebooks-gpu", "src")
OUT = os.path.join(ROOT, "notebooks-gpu")


def parse_cells(text):
    """Yield (kind, source_str) cells from percent-format text."""
    lines = text.splitlines()
    cells, cur_kind, cur = [], None, []

    def flush():
        if cur_kind is not None:
            src = "\n".join(cur).strip("\n")
            if src.strip():
                cells.append((cur_kind, src))

    for ln in lines:
        s = ln.strip()
        if s.startswith("# %% [markdown]") or s == "# %%[markdown]":
            flush(); cur_kind, cur = "markdown", []
        elif s.startswith("# %%"):
            flush(); cur_kind, cur = "code", []
        else:
            if cur_kind == "markdown":
                # strip a leading "# " / "#" from comment-embedded markdown
                cur.append(re.sub(r"^# ?", "", ln))
            elif cur_kind == "code":
                cur.append(ln)
            # text before the first cell marker is ignored
    flush()
    return cells


def to_ipynb(cells):
    nb_cells = []
    for kind, src in cells:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": source})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    srcs = sorted(glob.glob(os.path.join(SRC, "*.py")))
    if not srcs:
        print("No sources under notebooks-gpu/src/.")
        return
    index = []
    for sp in srcs:
        base = os.path.basename(sp)[:-3]           # <part>__<slug>
        if "__" not in base:
            print(f"  SKIP (no '__' part separator): {base}")
            continue
        part, slug = base.split("__", 1)
        cells = parse_cells(open(sp).read())
        if not cells:
            print(f"  SKIP (no cells): {base}")
            continue
        outdir = os.path.join(OUT, part)
        os.makedirs(outdir, exist_ok=True)
        outp = os.path.join(outdir, slug + ".ipynb")
        json.dump(to_ipynb(cells), open(outp, "w"), indent=1)
        n_code = sum(1 for k, _ in cells if k == "code")
        # title = first markdown line's heading, if any
        title = slug
        for k, s in cells:
            if k == "markdown":
                m = re.search(r"^#+\s*(.+)", s.strip())
                if m:
                    title = m.group(1).strip()
                break
        index.append((f"{part}/{slug}.ipynb", title, n_code))
        print(f"  wrote notebooks-gpu/{part}/{slug}.ipynb ({len(cells)} cells, {n_code} code)")

    # README index
    lines = ["# GPU-tier notebooks\n",
             "Real, runnable notebooks for the compute-heavy chapters, written for a "
             "**single NVIDIA H100 (80GB)** or **2x A100** — the tier where the ideas in these "
             "chapters actually show up (real FlashAttention/Triton kernels, `torch.compile`, "
             "quantization, fp8, a real training loop + MFU, and multi-GPU DDP/FSDP/tensor-parallel).\n",
             "> These complement the CPU toy-scale notebooks under `notebooks/`. Each has a pip cell "
             "and a hardware banner; expected throughput/MFU/speedup numbers are given as rough "
             "orders of magnitude, not fabricated benchmarks — run them to get your own.\n",
             "\n## Index\n"]
    for path, title, n_code in index:
        lines.append(f"- [`{path}`]({path}) — {title} ({n_code} code cells)")
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n")
    print(f"Wrote notebooks-gpu/README.md ({len(index)} notebooks).")


if __name__ == "__main__":
    main()
