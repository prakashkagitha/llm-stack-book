#!/usr/bin/env python3
"""Big bet: CI-tested runnable code. Per chapter, assemble the CPU-runnable Python
into a test module, RUN it, fix real bugs in the book, and verify faithfully.

Reads code/inventory.json (from extract_code.py). Per chapter (one pipeline item):
  Stage 1 ASSEMBLE+RUN (Sonnet 5): extract the chapter's CPU-runnable Python blocks,
     write tests/<slug>.py that executes them faithfully with minimal honest glue;
     RUN it; if a block errors from a REAL bug in the book, fix content/<chapter>.md
     and note it; skip GPU/network/fragment blocks with an explicit SKIP comment.
  Stage 2 VERIFY (Opus): confirm the test faithfully runs the book's code (no blanket
     try/except swallowing errors, no trivial stubs that bypass the logic), skips are
     legitimate, book fixes are correct; RE-RUN it to confirm it passes.

Resumable: --exclude-file code/code_done.txt; append PASSED chapters.
Model routing: assemble/run=Sonnet 5, verify=Opus.
"""
import argparse, json, os, collections, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RULES = """HARD RULES (these make the test HONEST — violating them defeats the whole point):
- The test must run the BOOK'S ACTUAL CODE, verbatim where possible. Copy each tested block's
  code faithfully; do not rewrite its logic to make it pass.
- NO cheating: no blanket `try/except: pass` around a block to swallow its error; no trivial stub
  that bypasses the very logic being demonstrated; no deleting asserts the book states.
- If a block ERRORS because the book's code is genuinely WRONG (bug), FIX it in
  content/<chapter>.md (and mirror the fix in the test) and record it as a real bug found.
- If a block is a DELIBERATELY-BROKEN demo (the chapter says "this crashes / this is the bug"),
  wrap ONLY that block in a narrow assert-it-raises check — do not "fix" it.
- NETWORK/API CALLS ARE FORBIDDEN in the test (CI has no network/keys — a real call hangs or 401/403).
  For a block that calls an external API or network (OpenAI/Anthropic/Cohere/HF hub/requests/httpx/
  boto3/socket/model or dataset download): EITHER (a) MOCK the boundary with `unittest.mock` so the
  block's OWN logic still executes offline against a canned response (preferred when the block's point
  IS the surrounding logic, e.g. retry/parse/prompt-assembly), OR (b) `# SKIP(network): ...` it.
  NEVER leave a real network/API call in the test.
- SKIP (don't test) blocks that truly need a GPU, or that are non-standalone fragments —
  leave an explicit `# SKIP(<reason>): ...` comment naming which block.
- Minimal honest glue is allowed: small fixtures/toy tensors, tiny shapes, fixed seeds, and
  CPU dtype (float32) substitutions where a block only differs by device. Keep runtime under
  ~60s and memory small.
- Deps available: numpy, torch (CPU), math, standard library. Do NOT add heavy deps."""

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["chapter", "passed", "blocks_tested", "blocks_skipped", "real_bugs_fixed", "notes"],
    "properties": {
        "chapter": {"type": "string"},
        "passed": {"type": "boolean", "description": "test runs green AND is faithful (no cheating) AND skips are legit"},
        "blocks_tested": {"type": "integer"},
        "blocks_skipped": {"type": "integer"},
        "real_bugs_fixed": {"type": "array", "items": {"type": "string"},
                            "description": "real bugs in the book's code found and fixed"},
        "cheating_found": {"type": "array", "items": {"type": "string"},
                           "description": "any blanket try/except, fake stub, or removed assert you had to fix"},
        "notes": {"type": "string"},
    },
}


def assemble_prompt(cid, slug, blocks):
    ci = [b for b in blocks if b["tier"] == "ci-testable"]
    other = [b for b in blocks if b["tier"] != "ci-testable"]
    ci_desc = "\n".join(f"    - block #{b['idx']} (line ~{b['line']}, {b['n_lines']} lines)" for b in ci)
    skip_desc = ", ".join(f"#{b['idx']}={b['tier']}" for b in other) or "(none)"
    return f"""You are proving that the runnable code in ONE textbook chapter actually RUNS.
Book root: {ROOT}. Chapter: content/{cid}.md
Write the test to: {ROOT}/tests/{slug}.py

This chapter has {len(ci)} heuristically CPU-runnable Python blocks to test:
{ci_desc}
Other blocks (default SKIP unless trivially CPU-safe): {skip_desc}

Do this:
1. Read the chapter. Extract each CPU-runnable Python block's code faithfully.
2. Write {ROOT}/tests/{slug}.py: assemble the blocks in order into a runnable module (later blocks
   may depend on names defined by earlier blocks in the same chapter — that's expected; concatenate
   them so the chapter's code runs as a whole). Add minimal honest glue and tiny fixtures so it
   executes on CPU. Every tested block should actually EXECUTE (a function defined by a block must be
   CALLED with a tiny input; a class must be instantiated and used).
3. RUN it: `cd {ROOT} && python3 tests/{slug}.py` (or `python3 -m pytest tests/{slug}.py -q` if you
   write pytest-style). Iterate until it runs clean OR every remaining failure is an honestly-SKIPPED
   block. If a failure is a REAL bug in the book's code, fix content/{cid}.md and mirror it in the test.

{RULES}

Return a short summary: blocks tested, blocks skipped (with reasons), and any real bugs you fixed."""


def verify_prompt(cid, slug):
    return f"""You are VERIFYING a code test for honesty and correctness. Chapter: content/{cid}.md
Test file: {ROOT}/tests/{slug}.py

Read the test AND the chapter. Then:
1. RE-RUN it: `cd {ROOT} && python3 tests/{slug}.py` (or pytest). It must pass.
2. Check FAITHFULNESS (this is the point): the test runs the book's ACTUAL code; there is NO blanket
   try/except swallowing errors, NO trivial stub bypassing the demonstrated logic, NO removed asserts.
   Each "tested" block is actually executed (functions called, classes used), not just imported.
3. Check SKIPS are legitimate (genuinely GPU/network/fragment, each with a `# SKIP(...)` reason).
3.5. Check HERMETICITY: no real network/API call remains — any OpenAI/Anthropic/HF/requests/httpx
   call must be MOCKED (unittest.mock) or SKIPped. A test that imports `openai` and calls it for real
   is a FAIL; fix it by mocking the client boundary or skipping the block.
4. Check any book fixes (edits to content/{cid}.md) are correct and improve the book.
If anything is wrong, FIX it (edit the test or the chapter) and re-run. Set passed only when the test
runs green AND is faithful AND skips are legit. Report blocks_tested, blocks_skipped, real_bugs_fixed,
and any cheating you had to correct."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int, default=5)   # smaller batches -> lower peak concurrency -> fewer 429/529
    ap.add_argument("--exclude-file")
    a = ap.parse_args()

    inv = json.load(open(os.path.join(ROOT, "code", "inventory.json")))
    by_ch = collections.defaultdict(list)
    for b in inv:
        by_ch[b["chapter"]].append(b)
    # only chapters that have at least one ci-testable block
    chapters = [cid for cid, bs in by_ch.items() if any(b["tier"] == "ci-testable" for b in bs)]

    if a.ids:
        want = {s.strip() for s in a.ids.split(",") if s.strip()}
        chapters = [c for c in chapters if c in want]
    elif a.exclude_file and os.path.exists(a.exclude_file):
        done = {l.strip() for l in open(a.exclude_file) if l.strip() and not l.startswith("#")}
        chapters = [c for c in chapters if c not in done]

    # most ci-testable blocks first (biggest coverage + most likely to surface bugs)
    chapters.sort(key=lambda c: sum(1 for b in by_ch[c] if b["tier"] == "ci-testable"), reverse=True)
    if a.limit and not a.ids:
        chapters = chapters[: a.limit]

    if not chapters:
        print("No chapters left to test (after filters).", file=sys.stderr)
        return 1

    items = []
    for cid in chapters:
        slug = cid.replace("/", "__")
        nci = sum(1 for b in by_ch[cid] if b["tier"] == "ci-testable")
        items.append({"chapter": cid, "slug": slug, "n_ci": nci,
                      "assemble": assemble_prompt(cid, slug, by_ch[cid]),
                      "verify": verify_prompt(cid, slug)})

    js = f"""export const meta = {{
  name: 'codetest-{a.name}',
  description: 'Assemble + run + verify CI-tested code for {len(items)} chapters',
  phases: [{{ title: 'Assemble+Run' }}, {{ title: 'Verify' }}],
}}

const CH = {json.dumps(items, indent=2)}
const VERIFY_SCHEMA = {json.dumps(VERIFY_SCHEMA)}

const results = await pipeline(
  CH,
  c => agent(c.assemble, {{ label: `run:${{c.chapter}}`, phase: 'Assemble+Run', model: 'claude-sonnet-5' }})
        .then(() => c),
  c => agent(c.verify, {{ label: `verify:${{c.chapter}}`, phase: 'Verify', schema: VERIFY_SCHEMA,
                         model: 'opus', effort: 'high' }})
        .then(v => ({{ chapter: c.chapter, verdict: v }})),
)

const done = results.filter(Boolean)
const passed = done.filter(r => r.verdict && r.verdict.passed).map(r => r.chapter)
const failed = done.filter(r => !r.verdict || !r.verdict.passed).map(r => r.chapter)
const bugs = done.flatMap(r => (r.verdict?.real_bugs_fixed || []).map(b => `${{r.chapter}}: ${{b}}`))
const cheats = done.flatMap(r => (r.verdict?.cheating_found || []).map(b => `${{r.chapter}}: ${{b}}`))
const died = CH.filter((c, i) => !results[i]).map(c => c.chapter)

log(`codetest: ${{done.length}}/${{CH.length}} | passed ${{passed.length}} | failed ${{failed.length}} | real-bugs ${{bugs.length}}`)
log(`PASSED (append to code_done.txt): ${{passed.join(' ')}}`)
if (bugs.length) log(`REAL BUGS FIXED: ${{JSON.stringify(bugs, null, 1)}}`)
if (cheats.length) log(`CHEATING CORRECTED: ${{JSON.stringify(cheats)}}`)
if (failed.length) log(`FAILED: ${{failed.join(' ')}}`)
if (died.length) log(`DIED (re-run): ${{died.join(' ')}}`)
return {{ passed, failed, died, bugs, cheats }}
"""
    outp = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    open(outp, "w").write(js)
    print(f"wrote {outp}: {len(items)} chapters")
    for it in items:
        print(f"   [{it['n_ci']} blocks] {it['chapter']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
