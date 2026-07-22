#!/usr/bin/env python3
"""Milestone 2: companion runnable-labs. Turn each verified tests/<slug>.py into a
Colab-openable notebook under notebooks/<part>/<chapter>.ipynb — "read the chapter,
run the (CI-verified) code in your browser". Deterministic, no LLM.

Each notebook: a header cell (title, Open-in-Colab badge, link to the online chapter),
a pip-install cell, then the test file split into one code cell per code block.
Also writes notebooks/README.md indexing every notebook with an Open-in-Colab link.
"""
import json, os, re, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "prakashkagitha/llm-stack-book"
BRANCH = "main"
SITE = "https://prakashkagitha.github.io/llm-stack-book"
NBDIR = os.path.join(ROOT, "notebooks")
PIP = "!pip install -q numpy torch einops scikit-learn"

COLAB = "https://colab.research.google.com/github/{repo}/blob/{branch}/notebooks/{rel}"


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.rstrip("\n").splitlines(keepends=True)}


def chapter_title(cid):
    md = os.path.join(ROOT, "content", cid + ".md")
    if os.path.exists(md):
        for line in open(md):
            if line.startswith("# "):
                return line[2:].strip()
    return cid


def build_notebook(cid, src):
    part, chap = cid.split("/", 1)
    rel = f"{part}/{chap}.ipynb"
    title = chapter_title(cid)
    header = (f"# {title}\n\n"
              f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
              f"({COLAB.format(repo=REPO, branch=BRANCH, rel=rel)})\n\n"
              f"Runnable, **CI-verified** code from *The LLM Stack* — "
              f"[read the chapter]({SITE}/{part}/{chap}.html).\n\n"
              f"> Every code cell is executed on CPU in the book's CI, so this notebook runs end-to-end. "
              f"A few heavy/networked models are replaced by tiny offline stand-ins for reproducibility; "
              f"swap them for the real package (and a GPU runtime) to scale up.")
    cells = [md_cell(header), code_cell(PIP)]
    # split the test source into one cell per code block (blocks are separated by a
    # `# ==========` banner line in the generated tests)
    chunks = re.split(r"\n(?=# ={6,})", src)
    for ch in chunks:
        ch = ch.strip("\n")
        if ch.strip():
            cells.append(code_cell(ch))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = os.path.join(NBDIR, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(nb, open(out, "w"), indent=1)
    return rel, title, part


def main():
    os.makedirs(NBDIR, exist_ok=True)
    rows = []
    for tf in sorted(glob.glob(os.path.join(ROOT, "tests", "*.py"))):
        cid = os.path.basename(tf)[:-3].replace("__", "/")
        rel, title, part = build_notebook(cid, open(tf).read())
        rows.append((part, rel, title))

    # index
    by_part = {}
    for part, rel, title in rows:
        by_part.setdefault(part, []).append((rel, title))
    idx = ["# Runnable notebooks — The LLM Stack\n",
           "One Colab-openable notebook per chapter, built from the book's **CI-verified** code "
           "([why they run](../.github/workflows/test.yml)). Click a badge to run it in your browser.\n"]
    for part in sorted(by_part):
        idx.append(f"\n## {part}\n")
        for rel, title in sorted(by_part[part]):
            badge = (f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                     f"({COLAB.format(repo=REPO, branch=BRANCH, rel=rel)})")
            idx.append(f"- {badge} [{title}]({rel})")
    open(os.path.join(NBDIR, "README.md"), "w").write("\n".join(idx) + "\n")

    print(f"wrote {len(rows)} notebooks -> notebooks/ (+ README index)")


if __name__ == "__main__":
    main()
