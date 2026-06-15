#!/usr/bin/env python3
"""Insert any missing {{fig:NAME}} markers for figures that exist on disk but were never
referenced by their chapter (e.g. an authoring agent wrote the figure then stopped before
the Edit). Deterministic placement: pick the chapter H2 whose heading best matches the
figure's section hint, and insert the marker right after that heading. No model usage.

Usage: python3 scripts/insert_fig_markers.py [--dry-run]
"""
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gen_fig_workflow import FIGS  # (chapter_id, name, concept, section)

STOP = set("the a an and or of to for in on with as is are its into not no this that "
           "section its their where via per most via about how why what which "
           "structure overview".split())


def keywords(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def best_heading(text, hint):
    hk = keywords(hint)
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^##\s+(.+)$", text)]
    best, score = None, 0
    for pos, h in heads:
        s = len(hk & keywords(h))
        if s > score:
            best, score = (pos, h), s
    return best, score


def insert_after_heading(text, pos, heading_line, marker):
    # find end of the heading line
    line_end = text.index("\n", pos)
    # insert marker as its own paragraph after the heading
    return text[:line_end] + f"\n\n{marker}" + text[line_end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inserted, skipped = 0, 0
    for cid, name, concept, section in FIGS:
        figpath = os.path.join(ROOT, "figures", name + ".html")
        chpath = os.path.join(ROOT, "content", cid + ".md")
        if not os.path.exists(figpath) or not os.path.exists(chpath):
            continue
        text = open(chpath).read()
        marker = "{{fig:" + name + "}}"
        if marker in text:
            skipped += 1
            continue
        bh, score = best_heading(text, section)
        if bh and score >= 1:
            pos, hl = bh
            where = f'after "## {hl}" (match {score})'
        else:
            # fallback: after the first H2, else after the first paragraph following H1
            m = re.search(r"(?m)^##\s+.+$", text)
            if m:
                pos, hl = m.start(), m.group(0)[3:]
                where = f'after first H2 "## {hl}" (fallback)'
            else:
                print(f"  !! {name}: no H2 found in {cid}; SKIPPED")
                continue
        new = insert_after_heading(text, pos, hl, marker)
        print(f"  + {name:22} -> {cid}  [{where}]")
        if not args.dry_run:
            open(chpath, "w").write(new)
        inserted += 1
    print(f"\n{'(dry-run) would insert' if args.dry_run else 'inserted'}: {inserted} | already-present: {skipped}")


if __name__ == "__main__":
    main()
