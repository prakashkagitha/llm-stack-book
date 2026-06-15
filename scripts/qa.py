#!/usr/bin/env python3
"""Quality lint for written chapters.

Checks each existing content/<part>/<file>.md for:
  - word count vs target
  - single H1 present
  - >=1 fenced code block
  - >=1 `!!! interview` callout and >=1 `!!! key` callout
  - some math ($...$ or $$)
  - cross-links resolve to real chapter ids
Prints a report and a list of chapters that look thin/broken (for re-running).

Usage: python3 scripts/qa.py            # report on everything written
       python3 scripts/qa.py --thin     # just print thin/broken ids (for --ids re-run)
"""
import argparse, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    with open(os.path.join(ROOT, "book.json")) as f:
        return json.load(f)


def flat(book):
    out = []
    pn = 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        for i, c in enumerate(p["chapters"], 1):
            out.append({
                "id": f"{p['dir']}/{c['file']}",
                "path": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md"),
                "title": c["title"], "words": c.get("words", 4500),
            })
    return out


def wc(text):
    # strip code fences for a prose-ish count but still count code lightly
    return len(re.findall(r"\S+", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thin", action="store_true")
    ap.add_argument("--min-frac", type=float, default=0.55, help="flag if words < frac*target")
    args = ap.parse_args()

    book = load()
    chapters = flat(book)
    valid_ids = {c["id"] for c in chapters}
    # The interview companion is a separate flat site under site/interview/; book pages link to
    # it as ../interview/<file>.html. Accept those targets so they aren't flagged as broken.
    iv_path = os.path.join(ROOT, "interview.json")
    if os.path.exists(iv_path):
        with open(iv_path) as f:
            iv = json.load(f)
        for p in iv["parts"]:
            for c in p["chapters"]:
                valid_ids.add(f"interview/{c['file']}")
        valid_ids.add("interview/index")

    rows, thin, total_words, n_written = [], [], 0, 0
    for c in chapters:
        if not os.path.exists(c["path"]):
            continue
        n_written += 1
        text = open(c["path"]).read()
        w = wc(text)
        total_words += w
        code_blocks = len(re.findall(r"(?m)^```", text)) // 2
        # strip fenced code so '# comment' lines don't masquerade as headings/callouts
        prose = re.sub(r"(?ms)^```.*?^```", "", text)
        h1 = len(re.findall(r"(?m)^#\s+\S", prose))
        interview = len(re.findall(r"!!!\s+interview", prose)) + len(re.findall(r"\?\?\?\s+interview", prose))
        key = len(re.findall(r"!!!\s+key", prose))
        math = ("$$" in prose) or bool(re.search(r"\$[^$\n]+\$", prose))
        # cross-link validation
        links = re.findall(r"\]\(\.\./([0-9a-zA-Z\-]+/[0-9a-zA-Z\-]+)\.html\)", text)
        bad_links = sorted({l for l in links if l not in valid_ids})

        problems = []
        if w < args.min_frac * c["words"]:
            problems.append(f"thin({w}<{int(args.min_frac*c['words'])})")
        if h1 != 1:
            problems.append(f"h1={h1}")
        if code_blocks < 1:
            problems.append("nocode")
        if interview < 1:
            problems.append("no-interview")
        if key < 1:
            problems.append("no-key")
        if not math:
            problems.append("no-math")
        if bad_links:
            problems.append("badlinks:" + ",".join(bad_links[:3]))

        rows.append((c["id"], w, c["words"], code_blocks, interview, key, math, problems))
        if problems:
            thin.append(c["id"])

    if args.thin:
        print(" ".join(thin))
        return

    print(f"{'chapter':52} {'words':>6} {'tgt':>5} {'code':>4} {'iv':>2} {'key':>3} {'math':>4}  problems")
    print("-" * 110)
    for cid, w, tgt, code, iv, key, math, probs in rows:
        flag = "  " + ";".join(probs) if probs else ""
        print(f"{cid:52} {w:6} {tgt:5} {code:4} {iv:2} {key:3} {str(math):>4}{flag}")
    print("-" * 110)
    pages = total_words // 300
    print(f"Written: {n_written}/{len(chapters)} chapters | total words: {total_words:,} "
          f"(~{pages:,} pages @300 w/page)")
    print(f"Flagged (thin/broken): {len(thin)}")
    if thin:
        print("Re-run ids:\n  " + " \\\n  ".join(thin))


if __name__ == "__main__":
    main()
