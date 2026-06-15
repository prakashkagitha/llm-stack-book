#!/usr/bin/env python3
"""Emit a self-contained Workflow script that writes a batch of chapters.

Usage:
  python3 scripts/gen_workflow.py --out scripts/wf_pilot.js --name pilot \
      --ids 02-transformer/03-attention-from-scratch 04-kernels-efficiency/02-flash-attention-1 ...
  python3 scripts/gen_workflow.py --out scripts/wf_b1.js --name batch1 --parts 01-foundations 02-transformer
  python3 scripts/gen_workflow.py --out scripts/wf_fill.js --name fill --missing      # only chapters lacking .md

Each agent reads STYLE.md + LINKMAP.md, writes content/<part>/<file>.md, returns a JSON summary.
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(manifest="book.json"):
    with open(os.path.join(ROOT, manifest)) as f:
        return json.load(f)


def flat(book):
    out = []
    pn = 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        for i, c in enumerate(p["chapters"], 1):
            num = "" if front else f"{pn}.{i}"
            out.append({
                "id": f"{p['dir']}/{c['file']}",
                "file": c["file"],
                "abspath": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md"),
                "num": num,
                "title": c["title"],
                "part": p["title"],
                "scope": c.get("scope", ""),
                "words": c.get("words", 4500),
                "model": c.get("model", "sonnet"),
            })
    return out


PROMPT = (
    "You are co-authoring a definitive, award-quality web textbook on the entire Large Language Model (LLM) stack, "
    "aimed at a working ML/LLM engineer and at someone preparing for a Google ML interview.\n\n"
    "STEP 1 — Read these two files IN FULL before writing (they define the house style and the cross-reference link map):\n"
    "  - {STYLE}\n  - {LINKMAP}\n\n"
    "STEP 2 — Write ONE complete chapter:\n"
    "  Chapter number: {NUM}\n  Title: {TITLE}\n  Part: {PART}\n"
    "  Scope (what this chapter must cover): {SCOPE}\n"
    "  Target length: about {WORDS} words. This is a deep reference chapter — be comprehensive and concrete; "
    "add more mechanism, code, and worked examples rather than padding with filler.\n\n"
    "HARD REQUIREMENTS (follow STYLE.md exactly):\n"
    "  - Begin the file with a single H1: \"# {NUM} {TITLE}\" (omit the number only for front-matter/appendix).\n"
    "  - Use ## for sections and ### for subsections; 4-8 major sections.\n"
    "  - Math in KaTeX: inline $...$ and display $$...$$ on their own lines.\n"
    "  - Every code fence declares a language (python/bash/text/cpp/json/yaml). Include SUBSTANTIAL, correct, "
    "heavily-commented, runnable or from-scratch code — this is a show-the-code book.\n"
    "  - Include at least one worked numerical example (!!! example) with real magnitudes.\n"
    "  - Include at least one Interview Corner (!!! interview) with a sharp Q and a model A.\n"
    "  - End with a Key Takeaways box (!!! key, 5-9 bullets) and a short 'Further reading' list of REAL landmark "
    "papers/repos named by author/title (do NOT fabricate citations, URLs, exact benchmark numbers, or quotes).\n"
    "  - Cross-link several related chapters using the exact targets in LINKMAP.md: [Title](../dir/file.html).\n"
    "  - Admonition bodies MUST be indented exactly 4 spaces; blank line before and after each block.\n\n"
    "STEP 3 — OUTPUT:\n"
    "  - Use the Write tool to write the COMPLETE chapter markdown to EXACTLY this path:\n      {ABSPATH}\n"
    "  - The file must contain ONLY the chapter markdown (no preamble, no 'here is the chapter', no trailing notes).\n"
    "  - Then return the structured summary object. Your returned text is data for the orchestrator, not for a human.\n"
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "file": {"type": "string", "description": "the path you wrote"},
        "words_written": {"type": "integer"},
        "sections": {"type": "integer", "description": "number of ## sections"},
        "code_blocks": {"type": "integer"},
        "interview_corners": {"type": "integer"},
        "one_line_summary": {"type": "string"},
    },
    "required": ["file", "words_written", "sections", "code_blocks", "one_line_summary"],
}


def make_prompt(c, style, linkmap):
    return (PROMPT
            .replace("{STYLE}", style)
            .replace("{LINKMAP}", linkmap)
            .replace("{NUM}", c["num"])
            .replace("{TITLE}", c["title"])
            .replace("{PART}", c["part"])
            .replace("{SCOPE}", c["scope"])
            .replace("{WORDS}", str(c["words"]))
            .replace("{ABSPATH}", c["abspath"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--missing", action="store_true", help="only chapters whose .md does not yet exist")
    ap.add_argument("--no-schema", action="store_true",
                    help="agents return a short text summary instead of a strict StructuredOutput object "
                         "(disk is the source of truth; avoids spurious 'completed without StructuredOutput' failures)")
    ap.add_argument("--force-model", default=None, choices=["sonnet", "opus", "haiku"])
    ap.add_argument("--manifest", default="book.json", help="which manifest to read (book.json or interview.json)")
    args = ap.parse_args()

    book = load(args.manifest)
    chapters = flat(book)
    sel = []
    for c in chapters:
        if args.ids and c["id"] not in args.ids:
            continue
        if args.parts and c["id"].split("/")[0] not in args.parts:
            continue
        if args.missing and os.path.exists(c["abspath"]):
            continue
        if args.force_model:
            c = dict(c, model=args.force_model)
        sel.append(c)

    if not sel:
        print("No chapters selected.")
        return

    style = os.path.join(ROOT, "STYLE.md")
    linkmap = os.path.join(ROOT, "LINKMAP.md")
    jobs = [{
        "label": c["id"],
        "abspath": c["abspath"],
        "model": c["model"],
        "prompt": make_prompt(c, style, linkmap),
        "words": c["words"],
    } for c in sel]

    meta_phases = json.dumps([{"title": "Write"}])
    jobs_json = json.dumps(jobs, ensure_ascii=True)
    schema_json = json.dumps(SCHEMA, ensure_ascii=True)

    if args.no_schema:
        agent_call = ("agent(j.prompt + '\\n\\nWhen done, reply with ONLY one short line: "
                      "the path you wrote and the approximate word count.', "
                      "{ label: 'write:' + j.label, phase: 'Write', model: j.model })")
    else:
        agent_call = ("agent(j.prompt, "
                      "{ label: 'write:' + j.label, phase: 'Write', model: j.model, schema: SCHEMA })")

    js = f"""export const meta = {{
  name: 'write-textbook-{args.name}',
  description: 'Write {len(sel)} chapters of The LLM Stack textbook (batch {args.name})',
  phases: {meta_phases},
}}
// Auto-generated by scripts/gen_workflow.py — batch '{args.name}', {len(sel)} chapters.
const JOBS = {jobs_json};
const SCHEMA = {schema_json};
phase('Write')
log('Writing ' + JOBS.length + ' chapters for batch {args.name}…')
const results = await parallel(JOBS.map(function (j) {{
  return function () {{
    return {agent_call}
      .then(function (r) {{ return r ? {{ id: j.label, ok: true, note: r }} : {{ id: j.label, failed: true }}; }})
      .catch(function (e) {{ return {{ id: j.label, failed: true, note: String(e) }}; }});
  }};
}}));
const ok = results.filter(Boolean).filter(function (r) {{ return !r.failed; }});
log('Batch {args.name}: ' + ok.length + '/' + JOBS.length + ' agents reported success (verify on disk via qa.py).');
return {{ batch: '{args.name}', reported_ok: ok.length, total: JOBS.length, results: results }};
"""
    with open(args.out, "w") as f:
        f.write(js)
    print(f"Wrote {args.out}: {len(sel)} chapters, models: "
          + ", ".join(sorted(set(c['model'] for c in sel)))
          + f"  (~{sum(c['words'] for c in sel):,} target words)")
    for c in sel:
        print(f"  [{c['model']:6}] {c['num']:5} {c['title']}")


if __name__ == "__main__":
    main()
