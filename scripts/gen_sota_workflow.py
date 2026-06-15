#!/usr/bin/env python3
"""Emit a Workflow script that adds web-verified "State of the Art & Resources" boxes.

One web-enabled agent per chapter: reads the chapter, researches the current state of the
art + best learning resources, VERIFIES every link with WebFetch, and inserts one
`!!! sota` admonition before the chapter's Further-reading section. Idempotent: agents skip
any chapter that already contains a `!!! sota` box.

Usage:
  python3 scripts/gen_sota_workflow.py --out scripts/wf_sota_all.js --name sota-all \
      --manifest book.json --parts 01-foundations 02-transformer ... [--limit N]
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOWCASE = {
    "02-transformer/03-attention-from-scratch",
    "04-kernels-efficiency/02-flash-attention-1",
    "07-inference-serving/01-anatomy-inference",
}


def flat(book):
    out = []
    for p in book["parts"]:
        for c in p["chapters"]:
            out.append({
                "id": f"{p['dir']}/{c['file']}",
                "abspath": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md"),
                "title": c["title"], "part": p["title"], "dir": p["dir"],
            })
    return out


PROMPT = r"""You are enhancing ONE chapter of a public, award-quality web textbook ("The LLM Stack: From Silicon to Agents") by adding a single, polished "State of the Art & Resources" box containing REAL, CURRENT, VERIFIED external links.

Chapter file (READ IT FIRST, in full): {ABSPATH}
Chapter title: {TITLE}
Part: {PART}

STEP 0 — IDEMPOTENCY: If the file already contains a line beginning with `!!! sota`, do NOTHING, make no edits, and reply exactly: "SKIP (already has SOTA box): {ABSPATH}". Otherwise continue.

STEP 1 — Infer the chapter's specific topic from its content. Then research using WebSearch and WebFetch:
  - For cutting-edge engineering topics (kernels, training, RL, serving, agents, quantization, architectures): the landmark paper(s), the most important RECENT (2023–2026) advances, and the canonical open-source repos/frameworks/tools, plus 1–2 excellent blog posts or official docs/product write-ups.
  - For foundational/math topics (linear algebra, probability, optimization, numerics, ML basics): emphasize the BEST durable learning resources — canonical textbooks, well-known courses, and great visual explainers — alongside any seminal papers. Still all real, verified links.

STEP 2 — VERIFY every link before using it. For each candidate URL, use WebFetch to confirm it resolves (HTTP 200) and is the correct content. Prefer canonical, durable URLs: arXiv ABSTRACT pages (https://arxiv.org/abs/XXXX.XXXXX), official GitHub repos (https://github.com/org/repo), official docs/blogs. DO NOT invent arXiv IDs, repo paths, dates, or benchmark numbers. If you cannot verify a link, DROP it. Better 6 verified links than 12 shaky ones. Aim for 8–12 verified links.

STEP 3 — Insert ONE admonition into the chapter using the Edit tool, placed IMMEDIATELY BEFORE the final "## Further reading" / "## Further Reading" heading. If there is no such heading, insert it at the very end of the file (after a blank line). EXACT format — the marker line, then EVERY content line indented exactly 4 spaces, with a blank line before and after the whole block:

!!! sota "State of the Art & Resources (2026)"
    A 1–2 sentence orientation on where this topic stands today.

    **Foundational work**

    - [Author et al., *Title* (year)](https://arxiv.org/abs/XXXX.XXXXX) — one-line why-it-matters.

    **Recent advances (2023–2026)**

    - [Author et al., *Title* (year)](URL) — one line.

    **Open-source & tools**

    - [org/repo](https://github.com/org/repo) — what it is.

    **Go deeper**

    - [Title](URL) — one line.

Rules:
  - 8–12 VERIFIED links total, grouped under those bold sub-labels (rename/drop a group to fit the topic — e.g. foundational chapters may use "Textbooks & courses" and "Visual explainers").
  - Each bullet: a real linked title + a short, accurate, non-hype description. No fabricated metrics.
  - Change NOTHING else in the file — not the body, not the existing Further-reading list, not any figure markers.
  - The 4-space indent on every line of the admonition is render-critical (Markdown admonition syntax).

STEP 4 — Reply with ONLY one short line: the chapter path and how many links you verified and inserted (or the SKIP line)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest", default="book.json")
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()

    book = json.load(open(os.path.join(ROOT, args.manifest)))
    sel = []
    for c in flat(book):
        if c["dir"].startswith(("00", "99")):
            continue
        if args.parts and c["dir"] not in args.parts:
            continue
        if args.ids and c["id"] not in args.ids:
            continue
        if c["id"] in SHOWCASE:
            continue
        txt = open(c["abspath"]).read()
        if "!!! sota" in txt:
            continue
        sel.append(c)
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        print("No chapters selected.")
        return

    jobs = [{"label": c["id"], "model": args.model,
             "prompt": PROMPT.replace("{ABSPATH}", c["abspath"])
                             .replace("{TITLE}", c["title"]).replace("{PART}", c["part"])}
            for c in sel]
    jobs_json = json.dumps(jobs, ensure_ascii=True)

    js = f"""export const meta = {{
  name: 'sota-{args.name}',
  description: 'Add web-verified State-of-the-Art & Resources boxes to {len(sel)} chapters',
  phases: [{{ title: 'Research & insert' }}],
}}
const JOBS = {jobs_json};
phase('Research & insert')
log('Adding SOTA boxes to ' + JOBS.length + ' chapters (web-verified links)…');
const results = await parallel(JOBS.map(function (j) {{
  return function () {{
    return agent(j.prompt, {{ label: 'sota:' + j.label, phase: 'Research & insert', model: j.model }})
      .then(function (r) {{ return {{ id: j.label, ok: true, note: r }}; }})
      .catch(function (e) {{ return {{ id: j.label, ok: false, note: String(e) }}; }});
  }};
}}));
const ok = results.filter(function (r) {{ return r.ok; }}).length;
log('SOTA batch {args.name}: ' + ok + '/' + JOBS.length + ' agents reported success (verify on disk).');
return {{ batch: '{args.name}', ok: ok, total: JOBS.length, results: results }};
"""
    with open(args.out, "w") as f:
        f.write(js)
    print(f"Wrote {args.out}: {len(sel)} chapters (model={args.model}).")
    for c in sel:
        print(f"  {c['id']}")


if __name__ == "__main__":
    main()
