#!/usr/bin/env python3
"""Regenerate LINKMAP.md from book.json so every chapter (incl. newly-added ones) is linkable.
Run after editing scripts/book_outline.py + regenerating book.json."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    book = json.load(open(os.path.join(ROOT, "book.json")))
    out = [
        "# LINKMAP — Cross-Reference Map for *The LLM Stack*",
        "",
        "When you reference another chapter, link to it like: `[Title](../<part-dir>/<file>.html)`.",
        "Every chapter HTML file lives one directory deep, so the prefix is always `../`.",
        "Below is the full table of contents with the exact link target for every chapter.",
        "",
    ]
    pn = 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        out.append(f"## {p['title']}")
        for i, c in enumerate(p["chapters"], 1):
            num = "" if front else f"{pn}.{i} "
            t = c["title"]
            tgt = f"../{p['dir']}/{c['file']}.html"
            out.append(f"- {num}**{t}** — `[{t}]({tgt})`")
        out.append("")
    # Note the interview companion is a separate site (linked as ../interview/<file>.html if needed).
    out.append("> The interview companion lives at `../interview/<file>.html` (separate site).")
    with open(os.path.join(ROOT, "LINKMAP.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    nch = sum(len(p["chapters"]) for p in book["parts"])
    print(f"Wrote LINKMAP.md: {len(book['parts'])} parts, {nch} chapters.")


if __name__ == "__main__":
    main()
