#!/usr/bin/env python3
"""Emit a resumable Workflow script that audits the book against benchmarks/cs336.json.

Phase 1 of the "established textbook" effort. READ-ONLY with respect to content/:
each agent grades one benchmark unit and writes audit/<unit-id>.json. Units that
already have an audit file are skipped, so re-running resumes cleanly.

Usage:
  python3 scripts/gen_audit_workflow.py --out scripts/wf_audit1.js --name a1 --limit 10
  python3 scripts/gen_audit_workflow.py --out scripts/wf_audit2.js --name a2 --ids A1-basics,A2-systems
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUBRIC_TEXT = """0 ABSENT     - topic not present in the book
1 MENTIONED  - named or gestured at; no working understanding conveyed
2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone
3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE
               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)
4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,
               resource accounting, and pointers into real library implementations"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["unit_id", "grade", "chapters_examined", "deficits",
                 "buildable_verdict", "summary"],
    "properties": {
        "unit_id": {"type": "string"},
        "grade": {"type": "integer", "minimum": 0, "maximum": 4},
        "chapters_examined": {
            "type": "array", "maxItems": 30,
            "items": {"type": "string"},
            "description": "chapter ids actually read, e.g. 02-transformer/01-tokenization",
        },
        "evidence": {
            "type": "array", "maxItems": 20, "items": {"type": "string"},
            "description": "short quotes/observations justifying the grade",
        },
        "buildable_verdict": {
            "type": "boolean",
            "description": "true only if a competent reader could produce a CORRECT working implementation of this unit's deliverable from the book alone",
        },
        "buildable_reasoning": {"type": "string"},
        "deficits": {
            "type": "array", "maxItems": 25,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "target", "missing", "fix"],
                "properties": {
                    "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                    "target": {
                        "type": "string",
                        "description": "existing chapter id to revise, or 'NEW:<proposed-part>/<proposed-slug>' for missing content",
                    },
                    "missing": {"type": "string", "description": "precisely what is absent or wrong"},
                    "fix": {"type": "string", "description": "concrete, actionable change to make"},
                    "kind": {
                        "type": "string",
                        "enum": ["explanation", "code", "math", "verification", "resource-accounting",
                                 "library-mapping", "figure", "exercise", "correctness-bug"],
                    },
                },
            },
        },
        "summary": {"type": "string"},
    },
}


def unit_prompt(u):
    kind = u.get("kind")
    obj = u.get("objectives") or []
    deliv = u.get("deliverables") or ([u["buildable_deliverable"]] if u.get("buildable_deliverable") else [])
    obj_txt = "\n".join(f"  - {o}" for o in obj) or "  (see deliverables)"
    deliv_txt = "\n".join(f"  - {d}" for d in deliv) or "  (none)"
    crit = "\nTHIS IS A CRITICAL UNIT: students must actually BUILD this. Grade harshly.\n" if u.get("critical") else ""
    return f"""You are auditing a 795,000-word technical textbook, "The LLM Stack: From Silicon to Agents",
against Stanford CS336 (Language Modeling from Scratch) and against a second standard:
CAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?

Book root: {ROOT}
Chapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json

## The unit you are auditing
id:    {u['id']}
kind:  {kind}
title: {u['title']}

Learning objectives this unit demands:
{obj_txt}

Implementable deliverable(s) a reader must be able to produce:
{deliv_txt}
{crit}
## Your job
1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those
   chapters in full (use Read/Grep). Do not grade from titles — grade from the prose and code.
2. Grade the book on this unit using the rubric:
{RUBRIC_TEXT}
3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct
   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,
   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,
   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.
4. List specific, actionable deficits. Each deficit must name a real target chapter id (or
   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.
   Do NOT invent deficits to seem thorough — if the coverage is genuinely strong, say so and
   return few or zero deficits. An honest grade of 4 is a valuable result.
5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with
   kind='correctness-bug' and severity='critical'.

## Output
Write your finding as JSON to: {ROOT}/audit/{u['id']}.json
(the file content must be exactly the same object you return)
Then return the structured object.

Be precise and evidence-based. Cite chapter ids you actually read."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", help="comma-separated unit ids to force")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="ignore existing audit files")
    a = ap.parse_args()

    spec = json.load(open(os.path.join(ROOT, "benchmarks", "cs336.json")))
    units = spec["units"]

    if a.ids:
        want = {s.strip() for s in a.ids.split(",") if s.strip()}
        units = [u for u in units if u["id"] in want]

    auditdir = os.path.join(ROOT, "audit")
    os.makedirs(auditdir, exist_ok=True)
    if not a.redo:
        units = [u for u in units if not os.path.exists(os.path.join(auditdir, u["id"] + ".json"))]

    # critical (assignment/capability) units first — they drive the most work
    units.sort(key=lambda u: (0 if u.get("critical") or u["kind"] == "capability" else 1, u["id"]))
    if a.limit:
        units = units[: a.limit]

    if not units:
        print("Nothing to audit — all units already have audit/<id>.json", file=sys.stderr)
        return 1

    items = [{"id": u["id"], "prompt": unit_prompt(u)} for u in units]

    js = f"""export const meta = {{
  name: 'audit-{a.name}',
  description: 'Audit The LLM Stack against CS336 + train-from-scratch buildability',
  phases: [{{ title: 'Audit', detail: '{len(items)} benchmark units graded independently' }}],
}}

const UNITS = {json.dumps(items, indent=2)}
const SCHEMA = {json.dumps(SCHEMA, indent=2)}

// Reasoning-heavy work runs on Opus 4.8 (per user 2026-07-20: Fable was
// consuming too much quota). Fable is reserved for sparing one-off strategy
// tasks only, never fan-out.
async function reason(prompt, opts = {{}}) {{
  return await agent(prompt, {{ ...opts, model: 'opus' }})
}}

phase('Audit')
const results = await parallel(UNITS.map(u => () =>
  reason(u.prompt, {{ label: `audit:${{u.id}}`, phase: 'Audit', schema: SCHEMA, effort: 'high' }})
))

const ok = results.filter(Boolean)
const failed = UNITS.filter((u, i) => !results[i]).map(u => u.id)
const byGrade = {{}}
for (const r of ok) byGrade[r.grade] = (byGrade[r.grade] || 0) + 1
const notBuildable = ok.filter(r => !r.buildable_verdict).map(r => r.unit_id)
const deficits = ok.flatMap(r => (r.deficits || []).map(d => ({{ ...d, unit: r.unit_id }})))

log(`audited ${{ok.length}}/${{UNITS.length}} | grades ${{JSON.stringify(byGrade)}} | ${{deficits.length}} deficits`)
if (failed.length) log(`FAILED (re-run to resume): ${{failed.join(', ')}}`)

return {{
  audited: ok.length,
  failed,
  gradeHistogram: byGrade,
  notBuildable,
  deficitCount: deficits.length,
  criticalDeficits: deficits.filter(d => d.severity === 'critical').length,
}}
"""
    outp = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    open(outp, "w").write(js)
    print(f"wrote {outp}: {len(items)} units")
    for u in items:
        print("  ", u["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
