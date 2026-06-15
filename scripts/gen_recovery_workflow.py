#!/usr/bin/env python3
"""Emit a recovery Workflow: for each chapter, an Opus agent verifies+fixes the figures that were
drafted for it but not yet wired in, then replaces the matching ASCII block with the {{fig}} marker.
This supplies the Opus verification + wiring that an interrupted diagram run skipped.

Reads the orphan map from /tmp/orphan_map.json ({chapter_id: [figure_names]}).
Usage: python3 scripts/gen_recovery_workflow.py --out scripts/wf_rec1.js --name rec1 --slice 0 7
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RULES = (
    "FIGURE RULES (for any fixes): keep ALL content inside the viewBox with margin (text must not clip); "
    "colors ONLY via theme classes (v-fill/v-accent/v-accent-s/v-stroke/v-grid/v-muted/v-label) or var(--accent|--accent2|"
    "--good|--warn|--ink-soft|--muted) with a hex fallback — NEVER a raw hex as a primary color; NEVER use math-alphanumeric "
    "Unicode (no 𝒩 𝒟 𝓜 ℝ — they tofu; use ASCII or π); NO inline <script> and NO XML comments; animate (data-anim/viz-draw/"
    "viz-sweep/viz-pulse) ONLY if it depicts a process, else keep it static; the figure must read clearly as a static image."
)

PROMPT = (
    "You are FINISHING and VERIFYING SVG figures that were drafted for ONE chapter but not yet reviewed or wired in.\n\n"
    "Chapter file: {ABSPATH}\n"
    "Figures drafted for this chapter (files at figures/<name>.html): {NAMES}\n\n"
    + RULES + "\n\n"
    "For EACH figure name:\n"
    "1. RENDER: run `python3 scripts/preview_fig.py <name> --theme both` from the repo root "
    "(/local-ssd/pk669/programming/llm-stack-textbook), then READ both /tmp/figpreview/<name>.light.png and "
    "/tmp/figpreview/<name>.dark.png.\n"
    "2. VERIFY vs the chapter: this figure was drawn to replace one of the chapter's ```text ASCII diagrams. Read figures/"
    "<name>.html and the chapter. Confirm the figure is FAITHFUL to that diagram's meaning and the chapter's content — every "
    "important box/node/edge/label present and correct, nothing misleading — and that it reads cleanly in BOTH light and dark "
    "(no clipping past the viewBox, no illegible overlaps) and is theme-safe. If ANYTHING is wrong, FIX figures/<name>.html "
    "(Edit) and re-render until clean.\n"
    "3. WIRE IT IN: find the ```text ... ``` fenced block in the chapter that this figure depicts, and replace that ENTIRE "
    "block with a single line `{{fig:<name>}}` (blank line before and after) using the Edit tool. If you truly cannot find a "
    "matching block (already replaced), skip only the replacement.\n\n"
    "Change nothing else in the chapter. Reply with one short line per figure: name + (PASS | fixed:<what> | wired | could-not-wire)."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--map", default="/tmp/orphan_map.json")
    ap.add_argument("--slice", nargs=2, type=int, default=None, help="start end (chapter index range)")
    args = ap.parse_args()

    m = json.load(open(args.map))
    chapters = sorted(m.keys())
    if args.slice:
        chapters = chapters[args.slice[0]:args.slice[1]]
    items = []
    for cid in chapters:
        items.append({"id": cid, "abspath": os.path.join(ROOT, "content", cid + ".md"),
                      "names": m[cid]})
    items_json = json.dumps(items, ensure_ascii=True)
    prompt_json = json.dumps(PROMPT, ensure_ascii=True)

    js = f"""export const meta = {{
  name: 'figure-recovery-{args.name}',
  description: 'Verify+fix and wire-in drafted figures for {len(items)} chapters',
  phases: [{{ title: 'Verify & wire' }}],
}}
const ITEMS = {items_json};
const PROMPT = {prompt_json};
phase('Verify & wire')
log('Recovering figures for ' + ITEMS.length + ' chapters…');
const results = await parallel(ITEMS.map(function (it) {{
  return function () {{
    const p = PROMPT.replace('{{ABSPATH}}', it.abspath).replace('{{NAMES}}', JSON.stringify(it.names));
    return agent(p, {{ label: 'recover:' + it.id, phase: 'Verify & wire', model: 'opus' }})
      .then(function (r) {{ return {{ id: it.id, ok: true, note: r }}; }})
      .catch(function (e) {{ return {{ id: it.id, ok: false, note: String(e) }}; }});
  }};
}}));
log('Recovery {args.name}: ' + results.filter(function(r){{return r.ok;}}).length + '/' + ITEMS.length + ' chapters processed.');
return {{ batch: '{args.name}', results: results }};
"""
    open(args.out, "w").write(js)
    print(f"Wrote {args.out}: {len(items)} chapters")
    for it in items:
        print(f"  {it['id']}  ({len(it['names'])} figs)")


if __name__ == "__main__":
    main()
