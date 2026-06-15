#!/usr/bin/env python3
"""Deterministic figure hygiene: strip forbidden inline <script> (animation/replay is handled by
app.js, not per-figure JS), and escape stray '&'/'<' that break XML well-formedness. Does NOT
touch colors/layout (those need judgment). Run, then re-check with qa_figs.py.

Usage: python3 scripts/clean_figures.py [--dry-run]
"""
import argparse, glob, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")

ENT = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")


# Pastel fills the Sonnet draw stage hard-coded; map to theme-aware equivalents.
HEX_MAP = [
    (r'fill="#e0e7ff"', 'fill="var(--accent2,#3b82f6)" fill-opacity="0.15"'),
    (r'fill="#dbeafe"', 'fill="var(--accent2,#3b82f6)" fill-opacity="0.15"'),
    (r'fill="#fef3c7"', 'fill="var(--warn,#e0a106)" fill-opacity="0.16"'),
    (r'fill="#fee2e2"', 'fill="var(--accent,#c15b39)" fill-opacity="0.13"'),
    (r'fill="#dcfce7"', 'fill="var(--good,#2f9e6e)" fill-opacity="0.15"'),
]


def clean(text):
    changed = []
    # 1) remove inline <script>...</script> (and any stray <script .../>)
    new = re.sub(r"(?is)<script\b.*?</script>", "", text)
    if new != text:
        changed.append("stripped <script>")
    text = new
    # 1b) remove XML comments (can't contain '--' and the draw stage often put '-->' arrows in them)
    new = re.sub(r"(?s)<!--.*?-->", "", text)
    if new != text:
        changed.append("stripped comments")
    text = new
    # 1c) fix a bare valueless attribute the draw stage emitted (e.g. `rx="3" cell-new`)
    new = re.sub(r'(\srx="\d+")\s+cell-new\b', r"\1", text)
    if new != text:
        changed.append("fixed bare attr")
    text = new
    # 1d) map hard-coded pastel fills to theme vars
    for pat, repl in HEX_MAP:
        if pat in text:
            text = text.replace(pat, repl)
            changed.append("themed " + pat[6:13])
    # 2) escape bare & that isn't a valid entity (common cause of 'undefined entity')
    new = ENT.sub("&amp;", text)
    if new != text:
        changed.append("escaped &")
    text = new
    # 3) escape a bare ' < ' used as less-than in text/labels (heuristic: '< ' or ' <' surrounded by spaces/digits)
    #    only inside >...< text runs is hard generically; skip to avoid breaking real tags.
    return text, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    n = 0
    for f in sorted(glob.glob(os.path.join(FIGS, "*.html"))):
        raw = open(f).read()
        new, changed = clean(raw)
        if changed:
            n += 1
            print(f"  {os.path.basename(f)[:-5]:42} {', '.join(changed)}")
            if not args.dry_run:
                open(f, "w").write(new)
    print(f"\n{'(dry-run) ' if args.dry_run else ''}cleaned {n} figures")


if __name__ == "__main__":
    main()
