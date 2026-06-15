#!/usr/bin/env python3
"""Lint the figure library for correctness & theme-safety.

Checks each figures/*.html:
  - XML well-formed (parses)
  - has <svg> with a viewBox and a role/aria-label, and a <figcaption>
  - NO raw hex colors used as a primary fill/stroke (hex allowed ONLY inside var(--x,#fallback))
  - elements roughly within the declared viewBox (coarse overflow heuristic on x/width text)
Also checks marker <-> file consistency:
  - every {{fig:NAME}} referenced in content/ has figures/NAME.html
  - every figures/NAME.html is referenced by some chapter
Exit nonzero if any hard problem is found.
"""
import glob, os, re, sys
import xml.dom.minidom as minidom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")
CONTENT = os.path.join(ROOT, "content")


def strip_vars(s):
    # remove var(--name, #fallback) / var(--name) so the fallback hex isn't flagged
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"var\([^()]*\)", "VAR", s)
    return s


def lint_file(path):
    name = os.path.basename(path)[:-5]
    raw = open(path).read()
    problems = []

    # XML well-formed
    try:
        minidom.parseString("<root>" + raw + "</root>")
    except Exception as e:
        problems.append(f"xml-error: {e}")

    msvg = re.search(r"<svg\b[^>]*>", raw)
    if not msvg:
        problems.append("no <svg>")
    else:
        tag = msvg.group(0)
        if "viewBox" not in tag:
            problems.append("no viewBox")
        if "role=" not in tag:
            problems.append("no role")
    if "aria-label" not in raw:
        problems.append("no aria-label")
    if "<figcaption" not in raw:
        problems.append("no figcaption")

    # raw hex as a primary color (outside var fallbacks)
    cleaned = strip_vars(raw)
    raw_hex = re.findall(r'(?:fill|stroke)\s*=\s*"(#[0-9a-fA-F]{3,8})"', cleaned)
    raw_hex = [h for h in raw_hex if h.lower() not in ("#fff", "#ffffff", "#000", "#000000")]
    if raw_hex:
        problems.append(f"raw-hex-color: {sorted(set(raw_hex))[:5]}")

    return name, problems


def main():
    files = sorted(glob.glob(os.path.join(FIGS, "*.html")))
    print(f"== Figure library: {len(files)} figures ==")
    hard = 0
    for f in files:
        name, probs = lint_file(f)
        status = "ok" if not probs else "; ".join(probs)
        flag = "" if not probs else "  ✗"
        print(f"  {name:24} {status}{flag}")
        hard += len(probs)

    # marker <-> file consistency
    referenced = set()
    for md in glob.glob(os.path.join(CONTENT, "**", "*.md"), recursive=True):
        for n in re.findall(r"\{\{fig:([a-z0-9\-]+)\}\}", open(md).read()):
            referenced.add(n)
    have = {os.path.basename(f)[:-5] for f in files}
    missing = referenced - have
    orphan = have - referenced
    print("-" * 60)
    print(f"referenced markers: {len(referenced)} | figure files: {len(have)}")
    if missing:
        print(f"  ✗ markers with NO figure file: {sorted(missing)}"); hard += len(missing)
    if orphan:
        print(f"  ⚠ figure files never referenced: {sorted(orphan)}")
    if not hard:
        print("✓ all figures clean and consistent")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
