#!/usr/bin/env python3
"""Milestone 4: exercises + verified solutions. Per chapter, Opus authors 3-6 exercises
with collapsible worked solutions grounded in the chapter, then a second Opus pass
adversarially verifies every solution and fixes errors. Appends an '## Exercises'
section to the chapter .md.

Resumable: chapters that already contain '## Exercises' are skipped.
Model: Opus (author + verify) — reasoning-heavy.
"""
import argparse, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_PARTS = {"13-interview-prep"}          # already Q&A
EXCLUDE_IDS = {"99-appendix/01-glossary", "99-appendix/02-math-reference",
               "99-appendix/03-papers-reading-list", "99-appendix/04-tooling-setup",
               "99-appendix/05-from-scratch-index", "00-frontmatter/00-preface"}

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["chapter", "passed", "n_exercises", "corrections", "notes"],
    "properties": {
        "chapter": {"type": "string"},
        "passed": {"type": "boolean", "description": "every solution correct + grounded + well-formed"},
        "n_exercises": {"type": "integer"},
        "corrections": {"type": "array", "items": {"type": "string"},
                        "description": "solution errors you found and fixed"},
        "notes": {"type": "string"},
    },
}


def author_prompt(cid):
    return f"""You are writing END-OF-CHAPTER EXERCISES for the textbook "The LLM Stack".
Book root: {ROOT}. Chapter: content/{cid}.md  (READ IT FULLY FIRST.)

Append a new section to the END of content/{cid}.md (use the Edit tool; append after the last
line, do not disturb existing content). The section:

## Exercises

Then 3-6 exercises that genuinely test THIS chapter's core ideas — a mix of:
  - conceptual ("why does ... / what happens if ..."),
  - quantitative (a small calculation with concrete numbers the reader can do by hand),
  - and at least one implementation task ("implement / modify ..." tied to the chapter's code).
Order them roughly easy -> hard. Number them **1.**, **2.**, ...

After EACH exercise, put its worked solution in a COLLAPSIBLE block using this exact syntax
(pymdownx.details — starts collapsed):

    ??? note "Solution"
        <the full worked solution, indented 4 spaces>

Solutions must be COMPLETE and CORRECT: show the reasoning/derivation or the code; for quantitative
problems, work the arithmetic to a final number; for implementation problems, give runnable code
consistent with the chapter's style. Ground everything in the chapter — no facts the chapter doesn't
support. Keep math in the book's `$...$`/`$$...$$` style and code in fenced blocks. ASCII-safe.

MUST PRESERVE: do not alter the chapter's existing prose, code, {{{{fig:}}}}/{{{{tool:}}}} markers, or
admonitions. Only APPEND the Exercises section.

Return one short line: how many exercises you wrote."""


def verify_prompt(cid):
    return f"""You are VERIFYING the '## Exercises' section just appended to content/{cid}.md.
Read the chapter AND the exercises. For EACH exercise's solution:
  - Re-derive / re-compute it independently. Is it CORRECT? (Fix any wrong number, formula, or claim.)
  - Is it grounded in the chapter (no unsupported facts)? Is the collapsible `??? note "Solution"`
    syntax well-formed (4-space indented body)? Is any code consistent with the chapter and runnable?
  - Is the exercise itself well-posed and unambiguous?
Fix problems directly by editing the chapter. Also confirm you did NOT break existing content
(markers/admonitions/code fences intact). Set passed only when every solution is correct and
well-formed. Report n_exercises and the corrections you made."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()

    chapters = []
    for md in sorted(glob.glob(os.path.join(ROOT, "content", "*", "*.md"))):
        cid = os.path.relpath(md, os.path.join(ROOT, "content"))[:-3]
        if cid.split("/")[0] in EXCLUDE_PARTS or cid in EXCLUDE_IDS:
            continue
        if re.search(r"^##\s+Exercises\s*$", open(md).read(), re.M):
            continue                            # already has exercises
        chapters.append(cid)

    if a.ids:
        want = {s.strip() for s in a.ids.split(",") if s.strip()}
        chapters = [c for c in chapters if c in want]
    if a.limit and not a.ids:
        chapters = chapters[: a.limit]
    if not chapters:
        print("No chapters left needing exercises.", file=sys.stderr)
        return 1

    items = [{"chapter": c, "author": author_prompt(c), "verify": verify_prompt(c)} for c in chapters]
    import json
    js = f"""export const meta = {{
  name: 'exercises-{a.name}',
  description: 'Author + verify exercises for {len(items)} chapters',
  phases: [{{ title: 'Author' }}, {{ title: 'Verify' }}],
}}
const CH = {json.dumps(items, indent=2)}
const VERIFY_SCHEMA = {json.dumps(VERIFY_SCHEMA)}
const results = await pipeline(
  CH,
  c => agent(c.author, {{ label: `ex:${{c.chapter}}`, phase: 'Author', model: 'opus', effort: 'high' }}).then(() => c),
  c => agent(c.verify, {{ label: `verify:${{c.chapter}}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: 'opus', effort: 'high' }})
        .then(v => ({{ chapter: c.chapter, verdict: v }})),
)
const done = results.filter(Boolean)
const passed = done.filter(r => r.verdict && r.verdict.passed).map(r => r.chapter)
const failed = done.filter(r => !r.verdict || !r.verdict.passed).map(r => r.chapter)
const corr = done.flatMap(r => (r.verdict?.corrections || []).map(c => `${{r.chapter}}: ${{c}}`))
const died = CH.filter((c,i) => !results[i]).map(c => c.chapter)
log(`exercises: ${{done.length}}/${{CH.length}} | passed ${{passed.length}} | corrections ${{corr.length}}`)
log(`PASSED: ${{passed.join(' ')}}`)
if (failed.length) log(`FAILED: ${{failed.join(' ')}}`)
if (died.length) log(`DIED: ${{died.join(' ')}}`)
return {{ passed, failed, died, corrections: corr.length }}
"""
    outp = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    open(outp, "w").write(js)
    print(f"wrote {outp}: {len(items)} chapters")
    for c in items:
        print("  ", c["chapter"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
