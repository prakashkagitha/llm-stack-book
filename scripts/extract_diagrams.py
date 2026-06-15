#!/usr/bin/env python3
"""List chapters containing genuine ASCII flow/architecture/process diagrams (```text blocks
rich in box-drawing/arrow glyphs), so the diagram->SVG pipeline can process them per-chapter.

Usage:
  python3 scripts/extract_diagrams.py                 # table of chapters + diagram counts
  python3 scripts/extract_diagrams.py --ids           # space-separated chapter ids (for --ids)
  python3 scripts/extract_diagrams.py --json out.json # full dump with per-block snippets
"""
import argparse, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOW = set("┌┐└┘├┤┬┴┼─│╔╗╚╝║═▶◀▲▼►◄→←↑↓⟶⟵↦⇒⇨╮╭╰╯")


def scan():
    chapters = []
    for f in sorted(glob.glob(os.path.join(ROOT, "content", "**", "*.md"), recursive=True)):
        if "/13-interview-prep/" in f:
            continue
        cid = f.split("/content/")[1][:-3]
        text = open(f).read()
        # skip if a figure marker already present near a diagram? keep simple: count flow diagrams.
        blocks = []
        for m in re.finditer(r"```text\n(.*?)```", text, re.S):
            b = m.group(1)
            nflow = sum(b.count(c) for c in FLOW)
            if nflow >= 6 and b.count("\n") >= 4:
                # capture the preceding heading for context
                pre = text[:m.start()]
                hmatch = list(re.finditer(r"(?m)^#{2,3}\s+(.+)$", pre))
                heading = hmatch[-1].group(1) if hmatch else ""
                blocks.append({"flow": nflow, "lines": b.count("\n"),
                               "first": b.strip().splitlines()[0][:60] if b.strip() else "",
                               "heading": heading})
        if blocks:
            chapters.append({"id": cid, "path": f, "n": len(blocks), "blocks": blocks})
    return chapters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--parts", nargs="*", default=[])
    args = ap.parse_args()
    chs = scan()
    if args.parts:
        chs = [c for c in chs if c["id"].split("/")[0] in args.parts]
    if args.ids:
        print(" ".join(c["id"] for c in chs))
        return
    if args.json:
        json.dump(chs, open(args.json, "w"), indent=2)
        print(f"Wrote {args.json}: {len(chs)} chapters, {sum(c['n'] for c in chs)} diagrams.")
        return
    total = sum(c["n"] for c in chs)
    print(f"{len(chs)} chapters with flow-diagrams | {total} diagrams total\n")
    for c in sorted(chs, key=lambda x: -x["n"]):
        print(f"  {c['n']:2d}  {c['id']}")


if __name__ == "__main__":
    main()
