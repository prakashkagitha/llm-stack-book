#!/usr/bin/env python3
"""Phases B+C of the figure-intuition effort: DRAW (Sonnet 5) then REVISE (Opus).

Reads figplan/<chapter>.json (from gen_figstrategy_workflow.py). Per chapter
(one pipeline item -> one .md file, no concurrent marker edits):
  Stage B DRAW (Sonnet 5): author each NEW figure and edit each refine/replace
     figure per its brief and the authoring SPEC; self-verify by rendering
     light+dark PNGs with preview_fig.py and VIEWING them; insert {{fig:}} markers.
  Stage C REVISE (Opus): render + view every touched figure in both themes; check
     it truly builds intuition, is correct, theme-safe, unclipped, tofu-free; FIX
     in place if not, re-render; return a per-figure verdict.

Resumable: pass --exclude-file plan/figplan_done.txt; append PASSED chapters.
Model routing: draw=Sonnet 5, revise=Opus (per user 2026-07-20).
"""
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEC = r"""FIGURE AUTHORING SPEC (follow exactly — correct, beautiful, theme-safe):
OUTPUT FILE: a self-contained figures/<NAME>.html: a `<figure class="viz" id="fig-<NAME>">` with
  optionally `<button class="viz-replay" type="button">/replay</button>`, exactly one inline
  `<svg viewBox="0 0 W H" role="img" aria-label="...">` (W ~ 660-820; H to fit), and a
  `<figcaption>` with a <b>bold lead sentence</b> + 1-2 sentences on the takeaway.
  No <script>, no external images. A scoped `<style>` (selectors prefixed #fig-<NAME>) is fine.
LAYOUT: EVERYTHING inside the viewBox with ~10px padding — text must NOT run past edges (it clips).
  No illegible overlaps (>= ~16px gaps between groups). Clear left-to-right or top-to-bottom order.
COLOR (must work in BOTH themes — NEVER hardcode a primary color): prefer classes v-fill, v-accent,
  v-accent-s, v-stroke, v-grid, v-muted, v-label, v-mono; or inline `var(--accent)` /
  `stroke="var(--good,#2f9e6e)"`. Vars: --accent,--accent-2,--ink-soft,--muted,--good(#2f9e6e),
  --warn(#e0a106),--accent2(#3b82f6),--surface,--surface-3,--border-2. Raw hex is FORBIDDEN as a
  primary color (breaks one theme); hex allowed ONLY as the fallback inside var(...).
FONTS: `font:600 13px var(--sans,sans-serif)`; code/labels `var(--mono,monospace)`.
ANIMATION (enhancement only — the figure MUST be fully clear as a STATIC image; never rely on motion):
  stage reveals via `data-anim="1".."5"` on top-level groups; `class="viz-draw" style="--len:NNN"`
  draws a line; `class="viz-sweep"` sweeps; `class="viz-pulse"` pulses. For a `long-animation`, use a
  longer ordered sequence of data-anim stages that walks through the process step by step, but every
  stage's end-state must still read correctly as a static frame.
TEXT SAFETY: ASCII only inside the SVG (no fancy Unicode arrows/operators/emoji/math-alphanumerics —
  they tofu). Use ->, <=, x, pi, sqrt written out, etc. Small illustrative integers only; invent NO
  benchmark numbers. role="img" + a one-sentence aria-label."""

DRAW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["chapter", "figures_done", "figures_skipped"],
    "properties": {
        "chapter": {"type": "string"},
        "figures_done": {"type": "array", "items": {"type": "string"}},
        "figures_skipped": {"type": "array", "items": {"type": "string"},
                            "description": "names you could NOT complete, with why appended"},
        "notes": {"type": "string"},
    },
}
REVISE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["chapter", "passed", "figures", "notes"],
    "properties": {
        "chapter": {"type": "string"},
        "passed": {"type": "boolean", "description": "true iff every planned figure is present, correct, theme-safe, unclipped and builds intuition"},
        "figures": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "verdict"],
                "properties": {
                    "name": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["good", "fixed", "still-broken", "missing"]},
                    "issue": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}


def draw_prompt(plan):
    cid = plan["chapter"]
    new = plan.get("new_figures", [])
    edits = [e for e in plan.get("existing", []) if e.get("verdict") in ("refine", "replace")]
    tasks = []
    for f in sorted(new, key=lambda x: x.get("priority", 2)):
        tasks.append(f"  NEW  [{f['kind']}]  name={f['name']}\n"
                     f"       concept: {f['concept']}\n"
                     f"       placement: after '{f['placement']}'\n"
                     f"       brief: {f['brief']}")
    for e in edits:
        tasks.append(f"  {e['verdict'].upper()}  existing figures/{e['name']}.html\n"
                     f"       issue: {e.get('issue','')}\n"
                     f"       change: {e.get('improvement','')}")
    tasklist = "\n\n".join(tasks) if tasks else "  (nothing to draw)"
    return f"""You are the FIGURE DRAWER (Sonnet 5) for the web textbook "The LLM Stack".
Chapter (read the relevant sections for exact framing/terminology): {ROOT}/content/{cid}.md

Author these figures. Do them ONE AT A TIME; finish and verify each before the next. Do NOT stop
early — complete every task or list what you couldn't in figures_skipped.

{tasklist}

{SPEC}

For EACH figure:
1. Write (NEW) or Edit (refine/replace) {ROOT}/figures/<NAME>.html per the brief and SPEC.
2. VISUALLY VERIFY: `python3 scripts/preview_fig.py <NAME> --theme both` from {ROOT}, then READ
   /tmp/figpreview/<NAME>.light.png and .dark.png. Fix anything clipped, overlapping, low-contrast in
   either theme, or that misrepresents the idea. Iterate up to 3 passes until both look clean.
3. For a NEW figure, insert the marker `{{{{fig:<NAME>}}}}` on its own line (blank line before/after)
   into content/{cid}.md at the planned placement, using Edit. Change NOTHING else in the chapter.
   (refine/replace keep their existing marker — do not add a duplicate.)

Return the structured result: figures_done (names finished + verified) and figures_skipped (with why)."""


def revise_prompt(plan):
    cid = plan["chapter"]
    names = [f["name"] for f in plan.get("new_figures", [])] + \
            [e["name"] for e in plan.get("existing", []) if e.get("verdict") in ("refine", "replace")]
    intu = "; ".join(plan.get("core_intuitions", [])[:4])
    return f"""You are the FIGURE REVISER (Opus) for "The LLM Stack". Chapter: content/{cid}.md
The chapter's core intuitions: {intu}

Review each of these figures (the ones this chapter just drew/edited):
  {', '.join(names) or '(none)'}

For EACH: render and SEE it — `python3 scripts/preview_fig.py <NAME> --theme both` from {ROOT},
then view /tmp/figpreview/<NAME>.light.png and .dark.png. Judge strictly:
  - Does it actually BUILD INTUITION for its concept (not just decorate)? Is it correct vs the chapter?
  - Theme-safe (no hardcoded primary color; readable in BOTH light and dark)?
  - Nothing clipped by the viewBox; no illegible overlaps; no tofu glyphs (ASCII only in the SVG)?
  - Marker present in the chapter exactly once; figcaption has a bold lead + takeaway?
If a figure has ANY problem, FIX it: Edit {ROOT}/figures/<NAME>.html (and re-render to confirm), or
fix a missing/duplicate marker in the chapter. Re-verify after fixing.

Set verdict per figure: good / fixed / still-broken / missing. `passed` = every figure good-or-fixed
AND present. Return the structured object."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--exclude-file")
    ap.add_argument("--max-prio", type=int, default=1,
                    help="draw only NEW figures with priority <= this (default 1 = must-have foundational)")
    ap.add_argument("--long-allow",
                    help="file listing figure names allowed to stay 'long-animation'; all other "
                         "long-animations are downgraded to a lighter 'animated' staged reveal")
    a = ap.parse_args()

    # Curated flagship long-animation allowlist (selective, per user).
    long_allow = None
    if a.long_allow and os.path.exists(a.long_allow):
        long_allow = {l.strip() for l in open(a.long_allow) if l.strip() and not l.startswith("#")}

    def discipline(plan):
        """Draw priority<=max_prio new figures PLUS any flagship-allowlisted figure (even if
        lower priority); downgrade every non-flagship long-animation to a lighter staged reveal."""
        kept = []
        for f in plan.get("new_figures", []):
            is_flagship = long_allow is not None and f["name"] in long_allow
            if f.get("priority", 2) > a.max_prio and not is_flagship:
                continue
            if f.get("kind") == "long-animation" and long_allow is not None and not is_flagship:
                f = {**f, "kind": "animated"}
            kept.append(f)
        plan = {**plan, "new_figures": kept}
        return plan

    plans = []
    for pf in sorted(glob.glob(os.path.join(ROOT, "figplan", "*.json"))):
        try:
            plan = json.load(open(pf))
        except Exception:
            continue
        # Derive the chapter id AUTHORITATIVELY from the filename — agents wrote the
        # plan's own "chapter" field inconsistently (some as content/<id>.md).
        cid = os.path.basename(pf)[:-5].replace("__", "/")
        plan["chapter"] = cid
        plan = discipline(plan)
        has_work = plan.get("new_figures") or [e for e in plan.get("existing", []) if e.get("verdict") in ("refine", "replace")]
        if has_work:
            plans.append(plan)

    if a.ids:
        want = {s.strip() for s in a.ids.split(",") if s.strip()}
        plans = [p for p in plans if p["chapter"] in want]
    elif a.exclude_file and os.path.exists(a.exclude_file):
        done = {l.strip() for l in open(a.exclude_file) if l.strip() and not l.startswith("#")}
        plans = [p for p in plans if p["chapter"] not in done]

    # most new figures first (biggest visible gain), then most edits
    def weight(p):
        return (len(p.get("new_figures", [])), len([e for e in p.get("existing", []) if e.get("verdict") in ("refine", "replace")]))
    plans.sort(key=lambda p: weight(p), reverse=True)
    if a.limit and not a.ids:
        plans = plans[: a.limit]

    if not plans:
        print("No chapters with figure work left (after filters).", file=sys.stderr)
        return 1

    items = [{
        "chapter": p["chapter"],
        "n_new": len(p.get("new_figures", [])),
        "n_edit": len([e for e in p.get("existing", []) if e.get("verdict") in ("refine", "replace")]),
        "draw": draw_prompt(p), "revise": revise_prompt(p),
    } for p in plans]

    js = f"""export const meta = {{
  name: 'figdraw-{a.name}',
  description: 'Draw + revise foundational figures for {len(items)} chapters',
  phases: [{{ title: 'Draw' }}, {{ title: 'Revise' }}],
}}

const CH = {json.dumps(items, indent=2)}
const DRAW_SCHEMA = {json.dumps(DRAW_SCHEMA)}
const REVISE_SCHEMA = {json.dumps(REVISE_SCHEMA)}

// Draw on Sonnet 5, revise on Opus (user routing 2026-07-20).
const results = await pipeline(
  CH,
  c => agent(c.draw, {{ label: `draw:${{c.chapter}}`, phase: 'Draw', schema: DRAW_SCHEMA,
                       model: 'claude-sonnet-5' }}).then(d => ({{ c, d }})),
  ({{ c }}) => agent(c.revise, {{ label: `revise:${{c.chapter}}`, phase: 'Revise', schema: REVISE_SCHEMA,
                                 model: 'opus', effort: 'high' }})
                .then(v => ({{ chapter: c.chapter, verdict: v }})),
)

const done = results.filter(Boolean)
const passed = done.filter(r => r.verdict && r.verdict.passed).map(r => r.chapter)
const failed = done.filter(r => !r.verdict || !r.verdict.passed)
                   .map(r => ({{ chapter: r.chapter, figs: r.verdict?.figures?.filter(f => f.verdict === 'still-broken' || f.verdict === 'missing') }}))
const died = CH.filter((c, i) => !results[i]).map(c => c.chapter)

log(`figures: drawn+revised ${{done.length}}/${{CH.length}} | passed ${{passed.length}} | failed ${{failed.length}} | died ${{died.length}}`)
log(`PASSED (append to figplan_done.txt): ${{passed.join(' ')}}`)
if (failed.length) log(`FAILED: ${{JSON.stringify(failed)}}`)
if (died.length) log(`DIED (re-run to resume): ${{died.join(' ')}}`)
return {{ passed, failed, died }}
"""
    outp = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    open(outp, "w").write(js)
    print(f"wrote {outp}: {len(items)} chapters")
    for it in items:
        print(f"   [{it['n_new']} new, {it['n_edit']} edit] {it['chapter']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
