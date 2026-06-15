#!/usr/bin/env python3
"""Emit a Workflow that converts ASCII flow-diagrams into theme-safe SVG figures, per chapter,
with a 3-stage pipeline: Opus spec/triage -> Sonnet draw+replace -> Opus verify.

Processing one CHAPTER per pipeline item (stages run sequentially per item) guarantees the same
markdown file is never edited by two agents at once. Different chapters run in parallel.

Usage:
  python3 scripts/gen_diagram_workflow.py --out scripts/wf_diag1.js --name diag1 \
      --parts 02-transformer 06-rl-infra ...        # by part(s)
  python3 scripts/gen_diagram_workflow.py --out scripts/wf_diag1.js --name diag1 --ids a/b c/d
  ... --limit 20                                     # cap chapters per batch
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import extract_diagrams
from gen_fig_workflow import SPEC

GOLD = open(os.path.join(ROOT, "figures", "attention-flow.html")).read()
SPEC_FILLED = SPEC.replace("{GOLD}", GOLD)

S1_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "chapter": {"type": "string"},
        "figures": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "description": "unique kebab-case figure name, PREFIXED with a token from this chapter (e.g. 'moe-serving-alltoall') so it cannot collide with other chapters' figures"},
                    "title": {"type": "string"},
                    "animated": {"type": "boolean", "description": "true ONLY if it depicts a process/flow/sequence that motion clarifies; false for static structure/architecture"},
                    "locator": {"type": "string", "description": "the EXACT first non-empty line of the ```text ASCII block this figure replaces (verbatim, for find-and-replace)"},
                    "spec": {"type": "string", "description": "a comprehensive, precise figure spec: every box/node, every arrow/edge and its label, the exact semantics, left-to-right or top-down layout plan with rough coordinates, the color plan using ONLY theme hooks, animated-vs-static rationale, and a viewBox estimate"},
                },
                "required": ["name", "title", "animated", "locator", "spec"],
            },
        },
        "skipped": {"type": "integer", "description": "how many ASCII diagrams you judged fine to leave as text"},
    },
    "required": ["chapter", "figures", "skipped"],
}

S1_PROMPT = (
    "You are the FIGURE ARCHITECT for a definitive web textbook on the LLM stack. Your job: look at ONE chapter's "
    "ASCII/text diagrams and decide which deserve to become precise SVG illustrations, then write an exact spec for each.\n\n"
    "Chapter file (READ IT IN FULL): {ABSPATH}\n\n"
    "Find every fenced ```text block that is a FLOW / ARCHITECTURE / PROCESS / DATA-FLOW diagram (boxes, arrows, pipelines, "
    "layered systems, timelines, trees). For EACH such diagram decide:\n"
    "  - CONVERT it if a real illustration would be clearer/more precise than the ASCII (most architecture & process diagrams).\n"
    "  - SKIP it (leave as text) if it is trivial, is really a table/log/byte-layout/code-trace where ASCII is already clear, "
    "OR if its concept is already shown by an existing `{{fig:...}}` marker already present in the chapter. Count those in 'skipped'.\n\n"
    "For each diagram you convert, produce: a UNIQUE name (kebab-case, PREFIXED with a token from this chapter so it can't collide "
    "with other chapters), a title, whether it should be ANIMATED (only if motion clarifies a process/sequence — otherwise static), "
    "the exact `locator` (the verbatim first non-empty line of that ASCII block, so the next stage can find & replace it), and a "
    "COMPREHENSIVE, PRECISE `spec`. The spec must name every box/node, every arrow and its label, the exact meaning, a concrete "
    "layout plan (positions, left-to-right or top-down), the color plan using ONLY the theme hooks (v-fill/v-accent/v-stroke/etc. or "
    "var(--accent) with hex fallback), the animated-vs-static decision and why, and a viewBox estimate. Be accurate to the chapter's "
    "actual content — a later Opus reviewer will check the figure against the chapter.\n\n"
    "Return the structured object (figures = only the ones to convert)."
)

S2_PROMPT_HEAD = (
    "You are the FIGURE ILLUSTRATOR. Render the specs below into theme-safe, award-quality SVG figures for this chapter, then "
    "replace each original ASCII block with the figure marker.\n\n"
    "Chapter file: {ABSPATH}\n"
    "Specs (one per figure) as JSON:\n{FIGS}\n\n"
    "{SPEC}\n\n"
    "FOR EACH figure in the JSON:\n"
    "1. Author figures/<name>.html following the spec and the rules above (static or animated per the spec's 'animated' flag). Write with the Write tool.\n"
    "2. VISUALLY VERIFY: run `python3 scripts/preview_fig.py <name> --theme both` from the repo root, then READ both "
    "/tmp/figpreview/<name>.light.png and /tmp/figpreview/<name>.dark.png. Fix any clipping/overlap/low-contrast/tofu-glyph issues "
    "(NEVER use 𝒩/𝒟/ℝ-style math-alphanumeric Unicode — they tofu; use ASCII/π). Iterate until clean in BOTH themes.\n"
    "3. REPLACE the original diagram: in the chapter file, find the ```text ... ``` block whose first non-empty line equals the "
    "figure's `locator`, and replace that ENTIRE fenced block with a single line `{{fig:<name>}}` (blank line before and after). "
    "Use the Edit tool. Do not change anything else.\n\n"
    "Reply with one short line per figure: its name and final viewBox."
)

S3_PROMPT = (
    "You are the FIGURE REVIEWER (correctness gate). Verify the SVG figures just created for this chapter are CORRECT and clean.\n\n"
    "Chapter file: {ABSPATH}\n"
    "Specs as JSON:\n{FIGS}\n\n"
    "FOR EACH figure:\n"
    "1. Run `python3 scripts/preview_fig.py <name> --theme both` and READ both PNGs (/tmp/figpreview/<name>.{{light,dark}}.png).\n"
    "2. Read figures/<name>.html AND the chapter. Check: (a) the figure is FAITHFUL to the original diagram's meaning and the "
    "chapter's content — every important node/edge/label present and correct, nothing misleading; (b) no layout clipping past the "
    "viewBox, no illegible overlaps, readable in BOTH light and dark; (c) theme-safe — no raw hex primary colors, no tofu "
    "math-alphanumeric glyphs; (d) the `{{fig:<name>}}` marker is present in the chapter and the original ASCII block was removed.\n"
    "3. If ANYTHING is wrong, FIX it (Edit figures/<name>.html and/or the chapter) and re-render to confirm.\n\n"
    "Reply with one short line per figure: name + PASS, or what you fixed."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-file", default=None, help="file with already-done chapter ids (one per line)")
    args = ap.parse_args()

    chs = extract_diagrams.scan()
    if args.parts:
        chs = [c for c in chs if c["id"].split("/")[0] in args.parts]
    if args.ids:
        chs = [c for c in chs if c["id"] in args.ids]
    if args.exclude_file and os.path.exists(args.exclude_file):
        done = set(open(args.exclude_file).read().split())
        chs = [c for c in chs if c["id"] not in done]
    # most-diagram-heavy first (highest value), then cap
    chs.sort(key=lambda c: -c["n"])
    if args.limit:
        chs = chs[:args.limit]
    if not chs:
        print("No chapters selected.")
        return

    items = [{"id": c["id"], "abspath": c["path"], "n": c["n"]} for c in chs]
    items_json = json.dumps(items, ensure_ascii=True)
    spec_json = json.dumps(SPEC_FILLED, ensure_ascii=True)
    s1 = json.dumps(S1_PROMPT, ensure_ascii=True)
    s2 = json.dumps(S2_PROMPT_HEAD, ensure_ascii=True)
    s3 = json.dumps(S3_PROMPT, ensure_ascii=True)
    schema = json.dumps(S1_SCHEMA, ensure_ascii=True)

    js = f"""export const meta = {{
  name: 'diagrams-{args.name}',
  description: 'Convert ASCII flow-diagrams to SVG figures in {len(items)} chapters (Opus spec -> Sonnet draw -> Opus verify)',
  phases: [{{ title: 'Spec' }}, {{ title: 'Draw' }}, {{ title: 'Verify' }}],
}}
const ITEMS = {items_json};
const SPEC = {spec_json};
const S1 = {s1};
const S2 = {s2};
const S3 = {s3};
const SCHEMA = {schema};
log('Diagram conversion over ' + ITEMS.length + ' chapters (' + ITEMS.reduce(function(a,c){{return a+c.n;}},0) + ' candidate diagrams)…');

const results = await pipeline(ITEMS,
  // Stage 1 — Opus: spec & triage
  function (it) {{
    return agent(S1.replace('{{ABSPATH}}', it.abspath),
      {{ label: 'spec:' + it.id, phase: 'Spec', model: 'opus', schema: SCHEMA }})
      .then(function (s) {{ return {{ id: it.id, abspath: it.abspath, figs: (s && s.figures) || [] }}; }})
      .catch(function (e) {{ return {{ id: it.id, abspath: it.abspath, figs: [], error: String(e) }}; }});
  }},
  // Stage 2 — Sonnet: draw + replace
  function (s1, it) {{
    if (!s1 || !s1.figs.length) return {{ id: it.id, abspath: it.abspath, figs: [], drew: 0 }};
    const prompt = S2.replace('{{ABSPATH}}', it.abspath)
      .replace('{{FIGS}}', JSON.stringify(s1.figs))
      .replace('{{SPEC}}', SPEC);
    return agent(prompt, {{ label: 'draw:' + it.id, phase: 'Draw', model: 'sonnet' }})
      .then(function (note) {{ return {{ id: it.id, abspath: it.abspath, figs: s1.figs, note: note }}; }})
      .catch(function (e) {{ return {{ id: it.id, abspath: it.abspath, figs: s1.figs, error: String(e) }}; }});
  }},
  // Stage 3 — Opus: verify
  function (s2, it) {{
    if (!s2 || !s2.figs.length) return {{ id: it.id, figs: 0, status: 'no-figures' }};
    const prompt = S3.replace('{{ABSPATH}}', it.abspath).replace('{{FIGS}}', JSON.stringify(s2.figs));
    return agent(prompt, {{ label: 'verify:' + it.id, phase: 'Verify', model: 'opus' }})
      .then(function (v) {{ return {{ id: it.id, figs: s2.figs.length, verdict: v }}; }})
      .catch(function (e) {{ return {{ id: it.id, figs: s2.figs.length, error: String(e) }}; }});
  }}
);
const made = results.filter(Boolean).reduce(function (a, r) {{ return a + (r.figs || 0); }}, 0);
log('Diagram batch {args.name}: ~' + made + ' figures created across ' + results.length + ' chapters.');
return {{ batch: '{args.name}', chapters: results.length, figures: made, results: results }};
"""
    with open(args.out, "w") as f:
        f.write(js)
    print(f"Wrote {args.out}: {len(items)} chapters, {sum(c['n'] for c in chs)} candidate diagrams.")
    for c in chs:
        print(f"  {c['n']:2d}  {c['id']}")


if __name__ == "__main__":
    main()
