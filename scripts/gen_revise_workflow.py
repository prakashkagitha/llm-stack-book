#!/usr/bin/env python3
"""Emit a resumable revise/deepen Workflow from plan/worklist.json (Phase 3).

Per chapter (one pipeline item -> one file, no concurrent same-file edits):
  Stage 1 SPEC   (Opus, high effort): read chapter + its deficits; work out every
                 fix precisely (correct arithmetic, correct code, exact shapes),
                 emit a structured edit spec.
  Stage 2 APPLY  (Sonnet 5): apply the spec to the .md with surgical Edits; write
                 real runnable code / corrected numbers / verification snippets.
  Stage 3 VERIFY (Opus, high effort): re-read edited chapter; confirm each deficit
                 resolved, no new bug, build-safe. Per-deficit pass/fail.

Resumable: pass --exclude-file plan/revise_done.txt; append PASSED chapters after
each batch. --ids forces specific targets; --limit takes the top-N by priority.

Model routing (user 2026-07-20): reasoning=Opus 4.8; drafting=Sonnet 5.
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUILDABILITY = """BUILDABILITY STANDARD (the bar every fix must hit):
- FULL SPECTRUM hardware: where relevant, give configs/expectations for laptop/CPU toy,
  single GPU, 8-GPU node, and multi-node. State expected loss curves / timings / failure signatures.
- INLINE, COPY-PASTE-COMPLETE code: runnable, self-contained blocks with exact tensor shapes,
  dtypes, edge cases (causal masking, padding, numerical stability, the backward pass),
  and concrete hyperparameters. Pseudo-code is NOT acceptable where real code is needed.
- VERIFICATION: give a way to check correctness (round-trip test, reference comparison,
  finite-difference grad check, or an expected numeric value the reader can reproduce).
- LIBRARY MAPPING: point to how the real library (PyTorch/Triton/HF/Megatron/DeepSpeed/
  FSDP/TRL/veRL/vLLM/SGLang/FlashAttention/lm-eval-harness) implements this, and where to read it.
- CS336 reference quality: a Stanford CS336 student should be able to complete the matching
  assignment/lecture using only this chapter."""

MUST_PRESERVE = """MUST PRESERVE (do not break these — the site build depends on them):
- {{fig:NAME}} figure markers — never delete or rename; the figure files stay valid.
- Admonitions (!!! interview, !!! key, !!! note, SOTA/resources boxes) — keep their exact syntax.
- The single H1, chapter front-matter, cross-links [text](../part/chap.html), and code fences.
- Only math-alphanumeric Unicode and emoji are true tofu; ASCII/Greek/arrows are fine in prose.
  In NEW code use ASCII (no fancy Unicode operators). Keep prose within existing style (STYLE.md)."""

SPEC_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["target", "edits", "overall_plan"],
    "properties": {
        "target": {"type": "string"},
        "overall_plan": {"type": "string"},
        "edits": {
            "type": "array", "maxItems": 40,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["addresses", "locator", "action", "instruction"],
                "properties": {
                    "addresses": {"type": "string", "description": "which deficit(s) this edit resolves"},
                    "locator": {"type": "string", "description": "unique anchor text in the file to find the edit site"},
                    "action": {"type": "string", "enum": ["replace", "insert-after", "insert-before", "rewrite-section"]},
                    "instruction": {"type": "string", "description": "exactly what the drafting agent must write, including corrected values/code verbatim where known"},
                    "corrected_value": {"type": "string", "description": "for correctness-bugs: the right number/formula/code, fully worked out"},
                    "is_bug_fix": {"type": "boolean"},
                },
            },
        },
    },
}

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["target", "passed", "deficits_resolved", "deficits_unresolved", "new_problems", "notes"],
    "properties": {
        "target": {"type": "string"},
        "passed": {"type": "boolean", "description": "true only if all deficits resolved AND no new problems AND build-safe"},
        "deficits_resolved": {"type": "integer"},
        "deficits_unresolved": {"type": "array", "items": {"type": "string"}},
        "new_problems": {"type": "array", "items": {"type": "string"}},
        "build_safe": {"type": "boolean", "description": "figure markers/admonitions/links intact, code fences balanced"},
        "notes": {"type": "string"},
    },
}


def spec_prompt(w):
    defs = "\n".join(
        f"  [{d['severity']}/{d['kind']}] (from {d['unit']})\n     MISSING: {d['missing']}\n     SUGGESTED FIX: {d['fix']}"
        for d in w["deficits"])
    return f"""You are the SPEC author (Opus) for a per-chapter revision of the textbook "The LLM Stack".
Book root: {ROOT}. Chapter file: content/{w['target']}.md

Read the ENTIRE chapter file first. Then read any chapter it cross-references that matters.
This chapter has {w['n_total']} audited deficits ({w['n_critical']} critical). Your job is to turn
each into a precise, correct edit instruction that a drafting agent can apply mechanically.

DEFICITS TO RESOLVE:
{defs}

{BUILDABILITY}

{MUST_PRESERVE}

CRITICAL RULE FOR BUGS: for every correctness-bug, actually WORK OUT the correct result yourself
(do the arithmetic, fix the code, derive the formula) and put the fully-correct value/code in
`corrected_value`. Do not defer this to the drafting agent — it will copy your `corrected_value` verbatim.
Verify your own math before emitting it.

For each edit give a `locator` = a short, UNIQUE substring currently in the file so the drafting
agent can find the exact site. Prefer surgical `replace`/`insert-after` over `rewrite-section`.
Keep the chapter's voice and existing correct content; change only what the deficits require
(plus any adjacent text that must stay consistent with a bug fix).

Write nothing to disk. Return the structured spec object."""


def apply_prompt(w):
    return f"""You are the DRAFTING agent (Sonnet 5) applying an edit spec to a textbook chapter.
File to edit: {ROOT}/content/{w['target']}.md

You will receive an edit spec (JSON) computed by a reasoning model. Apply EVERY edit to the file
using the Edit tool (surgical string replacements — do NOT rewrite the whole file). For each edit:
- Find the site using its `locator`.
- Apply `action` with `instruction`; if `corrected_value` is present, use it VERBATIM (it is the
  verified-correct number/formula/code — do not recompute or alter it).
- Write real, runnable, copy-paste-complete code with exact shapes and edge cases where asked.

{MUST_PRESERVE}

After applying all edits, re-read the changed regions to confirm they landed and the file still
has balanced code fences and intact {{fig:}} markers. Return a short summary listing, per edit,
the deficit it addressed and confirmation it was applied. Do not report success for edits you
could not locate — list those explicitly as SKIPPED so the verifier catches them."""


def verify_prompt(w):
    defs = "\n".join(f"  - [{d['severity']}] {d['missing'][:180]}" for d in w["deficits"])
    return f"""You are the VERIFIER (Opus) for a revised textbook chapter.
Re-read the CURRENT state of: {ROOT}/content/{w['target']}.md

The chapter was just edited to resolve these deficits:
{defs}

For EACH deficit, check the current file and decide if it is genuinely resolved to the buildability
standard below. Also independently re-check any code/arithmetic that was changed — recompute it; if a
"fix" introduced a new error, that is a new_problem. Confirm build-safety: every {{fig:}} marker and
admonition intact, cross-links unbroken, code fences balanced, no tofu Unicode in new code.

{BUILDABILITY}

Set `passed=true` ONLY if all deficits are resolved AND there are no new problems AND build_safe.
List any unresolved deficits and any new problems specifically. Return the structured object."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids", help="comma-separated targets to force")
    ap.add_argument("--limit", type=int, default=7)
    ap.add_argument("--exclude-file", help="file of already-done targets, one per line")
    ap.add_argument("--include-new", action="store_true", help="include NEW: work orders (default: revise only)")
    a = ap.parse_args()

    plan = json.load(open(os.path.join(ROOT, "plan", "worklist.json")))
    work = plan["work_orders"]
    if not a.include_new:
        work = [w for w in work if w["kind"] == "REVISE"]

    if a.ids:
        want = {s.strip() for s in a.ids.split(",") if s.strip()}
        work = [w for w in work if w["target"] in want]
    elif a.exclude_file and os.path.exists(a.exclude_file):
        done = {l.strip() for l in open(a.exclude_file) if l.strip() and not l.startswith("#")}
        work = [w for w in work if w["target"] not in done]

    if a.limit and not a.ids:
        work = work[: a.limit]

    if not work:
        print("Nothing to revise — worklist empty after filters.", file=sys.stderr)
        return 1

    items = [{
        "target": w["target"], "kind": w["kind"],
        "n_critical": w["n_critical"], "n_total": w["n_total"],
        "spec": spec_prompt(w), "apply": apply_prompt(w), "verify": verify_prompt(w),
    } for w in work]

    js = f"""export const meta = {{
  name: 'revise-{a.name}',
  description: 'Deepen/fix {len(items)} chapters to CS336 buildability standard',
  phases: [{{ title: 'Spec' }}, {{ title: 'Apply' }}, {{ title: 'Verify' }}],
}}

const WORK = {json.dumps(items, indent=2)}
const SPEC_SCHEMA = {json.dumps(SPEC_SCHEMA)}
const VERIFY_SCHEMA = {json.dumps(VERIFY_SCHEMA)}

// Reasoning on Opus 4.8; drafting on Sonnet 5 (user routing 2026-07-20).
const results = await pipeline(
  WORK,
  // Stage 1: SPEC (Opus)
  w => agent(w.spec, {{ label: `spec:${{w.target}}`, phase: 'Spec', schema: SPEC_SCHEMA,
                       model: 'opus', effort: 'high' }})
        .then(spec => ({{ w, spec }})),
  // Stage 2: APPLY (Sonnet 5) — gets the spec inline
  ({{ w, spec }}) => agent(
     w.apply + "\\n\\n## EDIT SPEC (apply every edit)\\n" + JSON.stringify(spec, null, 2),
     {{ label: `apply:${{w.target}}`, phase: 'Apply', model: 'claude-sonnet-5' }})
     .then(applied => ({{ w, applied }})),
  // Stage 3: VERIFY (Opus)
  ({{ w }}) => agent(w.verify, {{ label: `verify:${{w.target}}`, phase: 'Verify',
                                  schema: VERIFY_SCHEMA, model: 'opus', effort: 'high' }})
                .then(v => ({{ target: w.target, n_critical: w.n_critical, verdict: v }})),
)

const done = results.filter(Boolean)
const passed = done.filter(r => r.verdict && r.verdict.passed).map(r => r.target)
const failed = done.filter(r => !r.verdict || !r.verdict.passed)
                   .map(r => ({{ target: r.target, unresolved: r.verdict?.deficits_unresolved,
                                new_problems: r.verdict?.new_problems }}))
const died = WORK.filter((w, i) => !results[i]).map(w => w.target)

log(`revised ${{done.length}}/${{WORK.length}} | passed ${{passed.length}} | failed ${{failed.length}} | died ${{died.length}}`)
log(`PASSED (append to revise_done.txt): ${{passed.join(' ')}}`)
if (failed.length) log(`FAILED (needs another pass): ${{failed.map(f => f.target).join(' ')}}`)
if (died.length) log(`DIED (re-run to resume): ${{died.join(' ')}}`)

return {{ passed, failed, died }}
"""
    outp = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    open(outp, "w").write(js)
    print(f"wrote {outp}: {len(items)} chapters")
    for w in items:
        print(f"   [{w['n_critical']}c/{w['n_total']}t] {w['target']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
