#!/usr/bin/env python3
"""Emit a Workflow that EXTRACTS a concept record from each chapter (for the glossary +
prerequisite graph + reading tracks). One Sonnet agent per chapter -> concept/<id>.json.

Resumable: skips chapters whose concept/<part>__<file>.json already exists.
Usage: python3 scripts/gen_concept_workflow.py [--parts 03-pretraining ...] [--limit N]
"""
import argparse, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONCEPT_DIR = os.path.join(ROOT, "concept")
EXCLUDE_PARTS = {"00-frontmatter", "13-interview-prep"}


def flat(book):
    out, pn = [], 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        for i, c in enumerate(p["chapters"], 1):
            out.append({"id": f"{p['dir']}/{c['file']}", "dir": p["dir"], "file": c["file"],
                        "abspath": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md"),
                        "num": "" if front else f"{pn}.{i}", "title": c["title"], "part": p["title"]})
    return out


PROMPT = r"""You are indexing ONE chapter of a large LLM-systems textbook to build a cross-book GLOSSARY, a PREREQUISITE GRAPH, and guided READING TRACKS. Extract a precise, structured record — do not rewrite the chapter.

Chapter: {NUM} {TITLE}  (id: {ID})
Part: {PART}
File (read it): {ABSPATH}

Produce:
  - one_liner: what a reader can DO after this chapter (<= 16 words, action-oriented).
  - key_terms: the 5-12 most important technical terms this chapter uses, each with a crisp <= 25-word definition grounded in how the chapter uses it. Prefer terms a newcomer would need defined. Use the canonical term name (expand acronyms: "KV cache (key-value cache)").
  - introduces: of those, the terms/concepts this chapter is the natural PLACE-OF-FIRST-DEFINITION for (a subset of key_terms' names) — where a glossary should point for the definition.
  - prerequisites: 2-6 concepts or topics a reader should already understand before this chapter (short noun phrases, e.g. "matrix multiplication", "softmax", "the attention mechanism", "KV cache"). Name concepts, not chapter numbers.
  - difficulty: one of "intro" | "core" | "advanced".
  - tags: 2-5 short topical tags from this controlled-ish vocabulary where they fit (else invent sparingly): math, systems, gpu, architecture, training, optimization, data, alignment, rl, inference, serving, agents, rag, multimodal, evaluation, safety, production, interpretability, quantization, scaling.

Rules: be accurate and specific to THIS chapter; definitions must be correct and self-contained; no fabrication.

OUTPUT: use the Write tool to save the JSON object (with keys chapter_id, one_liner, key_terms, introduces, prerequisites, difficulty, tags) to EXACTLY:
  {OUT}
Then also return the same structured object."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["chapter_id", "one_liner", "key_terms", "prerequisites", "difficulty", "tags"],
          "properties": {
              "chapter_id": {"type": "string"},
              "one_liner": {"type": "string"},
              "key_terms": {"type": "array", "maxItems": 14, "items": {
                  "type": "object", "additionalProperties": False, "required": ["term", "definition"],
                  "properties": {"term": {"type": "string"}, "definition": {"type": "string"}}}},
              "introduces": {"type": "array", "items": {"type": "string"}},
              "prerequisites": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
              "difficulty": {"type": "string", "enum": ["intro", "core", "advanced"]},
              "tags": {"type": "array", "maxItems": 6, "items": {"type": "string"}}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "wf_concept.js"))
    ap.add_argument("--name", default="all")
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(CONCEPT_DIR, exist_ok=True)
    book = json.load(open(os.path.join(ROOT, "book.json")))
    sel = []
    for c in flat(book):
        if c["dir"] in EXCLUDE_PARTS:
            continue
        if args.parts and c["dir"] not in args.parts:
            continue
        outp = os.path.join(CONCEPT_DIR, c["id"].replace("/", "__") + ".json")
        if os.path.exists(outp):
            continue
        sel.append(c)
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        print("No chapters selected (all have concept/*.json?).")
        return

    jobs = [{"id": c["id"], "out": os.path.join(CONCEPT_DIR, c["id"].replace("/", "__") + ".json"),
             "prompt": (PROMPT.replace("{NUM}", c["num"]).replace("{TITLE}", c["title"])
                        .replace("{ID}", c["id"]).replace("{PART}", c["part"]).replace("{ABSPATH}", c["abspath"])
                        .replace("{OUT}", os.path.join(CONCEPT_DIR, c["id"].replace("/", "__") + ".json")))}
            for c in sel]

    js = f"""export const meta = {{
  name: 'concept-index-{args.name}',
  description: 'Extract glossary/prereq/tag records from {len(sel)} chapters',
  phases: [{{ title: 'Index' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const SCHEMA = {json.dumps(SCHEMA)};
const fs = {{}};
log('Indexing ' + JOBS.length + ' chapters for the concept graph…');
const results = await parallel(JOBS.map(function (j) {{
  return function () {{
    return agent(j.prompt, {{ label: 'index:' + j.id, phase: 'Index', model: 'sonnet', schema: SCHEMA }})
      .then(function (r) {{ return {{ id: j.id, out: j.out, rec: r }}; }})
      .catch(function (e) {{ return {{ id: j.id, failed: String(e) }}; }});
  }};
}}));
// The orchestrator writes each record to disk after the run (agents return data only).
return {{ total: JOBS.length, results: results }};
"""
    open(args.out, "w").write(js)
    print(f"Wrote {args.out}: {len(sel)} chapters to index (model=sonnet).")


if __name__ == "__main__":
    main()
