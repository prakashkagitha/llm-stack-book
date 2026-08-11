#!/usr/bin/env python3
"""Emit an ADVERSARIAL red-team workflow: per chapter, FIND candidate code/math bugs (Opus-5,
hostile) -> VERIFY each independently and FIX only CONFIRMED real bugs in place (Opus-5, skeptical).

One chapter per pipeline item (no concurrent same-file edits). Batched + resumable via a done-file.
Targets content/*.md chapters (code blocks live inside the markdown). Excludes the live capstone
package (separately smoke-tested; don't edit code a running experiment imports).

Usage: python3 scripts/gen_redteam_workflow.py --out scripts/wf_rt1.js --name rt1 \
         --parts 02-transformer 04-kernels-efficiency --exclude-file plan/redteam_done.txt
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_PARTS = {"00-frontmatter", "13-interview-prep", "99-appendix"}

FIND = r"""You are an ADVERSARIAL technical reviewer red-teaming ONE chapter of a definitive LLM-systems textbook. Your ONLY job is to find REAL DEFECTS in its code and math. Be hostile and specific. Do NOT edit anything; do NOT report style/nitpicks.

Chapter: content/{ID}.md  (read it in full)

Hunt for genuine bugs in these categories:
  - MATH: a wrong formula, a broken derivation step, a wrong constant/exponent/factor, an algebra error, a units mismatch, a probability/linear-algebra mistake, a worked example whose arithmetic does not check out.
  - CODE: a shape/dimension error, an off-by-one, an undefined name / bad import, a wrong or hallucinated library API (method/kwarg that does not exist or is misused), a logic bug, an assertion that is wrong or vacuous, a fence that would crash if run.
  - NUMBERS: a fabricated or clearly-wrong benchmark/size/FLOP/memory number that a reader could check and find false (distinguish from an honestly-hedged "on the order of").
  - CLAIMS: a confidently-stated but incorrect claim about how an algorithm/library/mechanism works.

For EACH candidate defect: quote the exact offending text/line, say precisely why it is wrong (show the correct derivation/shape/value), give the minimal fix, and a confidence in {high, medium, low}. Prefer a few HIGH-confidence real bugs over a long list of maybes. If the chapter is clean, say so honestly (empty list) — do NOT invent problems. Re-derive math yourself; mentally execute code.

Return the structured findings object."""

FIXV = r"""You are a SKEPTICAL verifier for an adversarial red-team pass on ONE textbook chapter. A red-teamer proposed candidate code/math defects. Independently verify EACH, then FIX ONLY the ones you CONFIRM are real bugs. Introducing a wrong "fix" is worse than leaving a non-bug, so when in doubt, REJECT.

Chapter: content/{ID}.md
Red-teamer's candidate defects (JSON):
{FINDINGS}

For EACH candidate:
  - Independently re-derive the math / re-check the code (shapes, APIs, constants, arithmetic). Confirm ONLY if you can demonstrate it is genuinely wrong. A plausible-sounding but unverifiable claim is REJECTED.
  - If CONFIRMED, apply the minimal correct fix in place with the Edit tool (fix the formula/code/number/claim). Keep the fix consistent with the rest of the chapter and the book; do not introduce new errors; never fabricate a number/API/citation.
  - Do NOT touch {{fig:...}} / {{tool:...}} markers, the `## Exercises` structure, figures, or correct content. Only change what is genuinely wrong.

Also do your OWN quick independent scan for any obvious high-confidence bug the red-teamer missed, and fix it too.

Return the verdict object: confirmed_and_fixed (count), rejected (count), and a short note per confirmed fix (what was wrong -> what you changed)."""

FSCHEMA = {"type": "object", "additionalProperties": False, "required": ["chapter", "findings"],
           "properties": {"chapter": {"type": "string"},
                          "findings": {"type": "array", "maxItems": 25, "items": {
                              "type": "object", "additionalProperties": False,
                              "required": ["category", "quote", "why_wrong", "fix", "confidence"],
                              "properties": {"category": {"type": "string", "enum": ["math", "code", "numbers", "claims"]},
                                             "quote": {"type": "string"}, "why_wrong": {"type": "string"},
                                             "fix": {"type": "string"},
                                             "confidence": {"type": "string", "enum": ["high", "medium", "low"]}}}}}}
VSCHEMA = {"type": "object", "additionalProperties": False,
           "required": ["chapter", "confirmed_and_fixed", "rejected", "notes"],
           "properties": {"chapter": {"type": "string"}, "confirmed_and_fixed": {"type": "integer"},
                          "rejected": {"type": "integer"}, "fixes": {"type": "array", "items": {"type": "string"}},
                          "notes": {"type": "string"}}}


def flat(book):
    out, pn = [], 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        for i, c in enumerate(p["chapters"], 1):
            out.append({"id": f"{p['dir']}/{c['file']}", "dir": p["dir"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--exclude-file")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    excluded = set()
    if a.exclude_file and os.path.exists(a.exclude_file):
        excluded = {l.strip() for l in open(a.exclude_file) if l.strip() and not l.startswith("#")}

    book = json.load(open(os.path.join(ROOT, "book.json")))
    sel = []
    for c in flat(book):
        if c["dir"] in EXCLUDE_PARTS or c["id"] in excluded:
            continue
        if a.parts and c["dir"] not in a.parts:
            continue
        sel.append(c["id"])
    if a.limit:
        sel = sel[:a.limit]
    if not sel:
        print("No chapters selected."); return

    jobs = [{"id": cid, "find": FIND.replace("{ID}", cid),
             "fixv": FIXV.replace("{ID}", cid)} for cid in sel]

    js = f"""export const meta = {{
  name: 'redteam-{a.name}',
  description: 'Adversarial code+math red-team of {len(sel)} chapters (find -> verify+fix, Opus-5)',
  phases: [{{ title: 'Find' }}, {{ title: 'Verify+Fix' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const FSCHEMA = {json.dumps(FSCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
log('Red-teaming ' + JOBS.length + ' chapters (adversarial find -> skeptical verify+fix)…');
const results = await pipeline(
  JOBS,
  function (j) {{
    return agent(j.find, {{ label: 'find:' + j.id, phase: 'Find', model: 'claude-opus-5', schema: FSCHEMA }})
      .then(function (f) {{ return {{ j: j, findings: f }}; }})
      .catch(function (e) {{ return {{ j: j, findings: null }}; }});
  }},
  async function (prev) {{
    if (!prev || !prev.j) return null;
    const f = prev.findings;
    if (!f || !f.findings || f.findings.length === 0) return {{ id: prev.j.id, clean: true }};
    const v = await agent(prev.j.fixv.replace('{{FINDINGS}}', JSON.stringify(f.findings).slice(0, 10000)),
      {{ label: 'fix:' + prev.j.id, phase: 'Verify+Fix', model: 'claude-opus-5', schema: VSCHEMA }});
    return {{ id: prev.j.id, verdict: v }};
  }}
);
const done = results.filter(Boolean);
const fixed = done.reduce(function (s, r) {{ return s + (r.verdict ? (r.verdict.confirmed_and_fixed||0) : 0); }}, 0);
log('Red-team {a.name}: ' + done.length + '/' + JOBS.length + ' chapters, ' + fixed + ' confirmed bugs fixed.');
return {{ batch: '{a.name}', total: JOBS.length, done: done.length, bugs_fixed: fixed, results: results }};
"""
    open(a.out, "w").write(js)
    print(f"Wrote {a.out}: {len(sel)} chapters (Opus-5 find -> verify+fix).")


if __name__ == "__main__":
    main()
