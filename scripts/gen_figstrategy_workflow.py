#!/usr/bin/env python3
"""Phase A of the figure-intuition effort: Opus STRATEGY per chapter.

Each agent reads one chapter and its existing figures (rendering them to SEE them),
then decides the FOUNDATIONAL infographics that chapter needs for intuitive
understanding: audit existing figures (keep/refine/replace) and specify new ones
(static / animated / long-animation), each with a rich brief the drawer will follow.

Resumable: writes figplan/<chapter-id-slug>.json; re-run skips chapters that have one.
Model: Opus (reasoning/strategy), per user routing.

Usage:
  python3 scripts/gen_figstrategy_workflow.py --out scripts/wf_figplan1.js --name p1 --parts 04 --limit 12
  python3 scripts/gen_figstrategy_workflow.py --out scripts/wf_figplan1.js --name p1 --ids 04-kernels-efficiency/03-flash-attention-2-3
"""
import argparse, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Reference/text collections that do not need infographics.
EXCLUDE_PARTS = {"13-interview-prep"}
EXCLUDE_IDS = {
    "99-appendix/01-glossary", "99-appendix/02-math-reference",
    "99-appendix/03-papers-reading-list", "99-appendix/04-tooling-setup",
    "00-frontmatter/00-preface",
}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["chapter", "core_intuitions", "existing", "new_figures", "summary"],
    "properties": {
        "chapter": {"type": "string"},
        "core_intuitions": {
            "type": "array", "maxItems": 6, "items": {"type": "string"},
            "description": "the handful of ideas a reader MUST grok intuitively to understand this chapter",
        },
        "existing": {
            "type": "array", "maxItems": 15,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "verdict"],
                "properties": {
                    "name": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["keep", "refine", "replace"]},
                    "issue": {"type": "string", "description": "what's weak (for refine/replace)"},
                    "improvement": {"type": "string", "description": "concrete change to make"},
                },
            },
        },
        "new_figures": {
            "type": "array", "maxItems": 5,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "concept", "rationale", "kind", "placement", "brief"],
                "properties": {
                    "name": {"type": "string", "description": "kebab-case file stem, unique, descriptive"},
                    "concept": {"type": "string", "description": "one line: what it visualizes"},
                    "rationale": {"type": "string", "description": "why THIS is foundational for intuition here"},
                    "kind": {"type": "string", "enum": ["static", "animated", "long-animation"]},
                    "placement": {"type": "string", "description": "section heading / anchor text to place the marker after"},
                    "brief": {"type": "string", "description": "rich, specific drawing brief: what to show, the visual metaphor, labels, stages, what must be legible; enough for a drawer to execute without re-reading the chapter"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 3, "description": "1=must-have foundational, 3=nice-to-have"},
                },
            },
        },
        "summary": {"type": "string"},
    },
}


def chapter_prompt(cid, existing_names):
    figs = "\n".join(f"    - figures/{n}.html  (marker {{{{fig:{n}}}}})" for n in existing_names) or "    (none — this chapter has NO figures yet)"
    return f"""You are the FIGURE STRATEGIST (Opus) for a public web textbook, "The LLM Stack".
Your job for ONE chapter: decide the infographics that are FOUNDATIONAL to understanding it —
the visuals that build INTUITION, not decoration. Quality over quantity: a chapter needs the
1-3 figures that make its core mechanism click, not a figure per section.

Book root: {ROOT}
Chapter (READ IT IN FULL FIRST): content/{cid}.md

Existing figures in this chapter:
{figs}

## Do this
1. Read the chapter fully. Identify its `core_intuitions`: the handful of ideas a reader MUST
   grasp intuitively (e.g. "why decode is memory-bound", "how the online-softmax rescale works",
   "what tensor-parallel all-reduce actually moves"). These drive everything else.
2. AUDIT each existing figure: RENDER it to actually SEE it —
   `python3 scripts/preview_fig.py <name> --theme both` from {ROOT}, then view
   /tmp/figpreview/<name>.light.png and .dark.png. Judge: does it correctly and clearly convey
   its idea in BOTH themes? Verdict keep / refine (state the concrete improvement) / replace.
   Do NOT rubber-stamp — look for clipping, clutter, weak metaphors, or a missed chance at intuition.
3. Specify NEW figures for the core_intuitions not already well-served. For each: a unique kebab-case
   `name`, the `concept`, the `rationale` (why it's foundational for intuition HERE), `kind`
   (static / animated / long-animation), `placement` (section to anchor it), and a rich `brief` a
   drawer can execute from. Use `long-animation` ONLY where motion is genuinely essential to the
   intuition (a multi-step process that a static frame cannot convey — e.g. an online-softmax pass,
   a pipeline schedule filling and draining, a diffusion denoising trajectory, autoregressive decode
   with a growing KV cache). Prefer a crisp STATIC infographic when a single frame suffices.
4. Be disciplined: if the chapter is already well-served, return few or zero new figures and mostly
   `keep` verdicts. If it is figure-poor but concept-rich, propose the foundational 1-3.

## Constraints on what you propose (the drawer must obey these; keep briefs within them)
- Theme-safe: no hardcoded primary colors; small illustrative integers only, invent NO benchmark numbers.
- Everything must fit inside the viewBox; no illegible overlaps; clear reading order.
- The figure must be fully understandable as a STATIC frame; animation is enhancement only.

Write your plan as JSON to {ROOT}/figplan/{cid.replace('/', '__')}.json and also return the object."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--parts", help="comma-separated part-dir prefixes, e.g. 04,05")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--only-fig-light", type=int, default=-1,
                    help="if set to N, only chapters with <= N existing figures")
    a = ap.parse_args()

    chapters = []
    for md in sorted(glob.glob(os.path.join(ROOT, "content", "*", "*.md"))):
        cid = os.path.relpath(md, os.path.join(ROOT, "content"))[:-3]
        part = cid.split("/")[0]
        if part in EXCLUDE_PARTS or cid in EXCLUDE_IDS:
            continue
        chapters.append((cid, md))

    if a.ids:
        want = {s.strip() for s in a.ids.split(",") if s.strip()}
        chapters = [c for c in chapters if c[0] in want]
    if a.parts:
        pref = tuple(p.strip() for p in a.parts.split(","))
        chapters = [c for c in chapters if c[0].split("/")[0].startswith(pref)]

    figdir = os.path.join(ROOT, "figplan")
    os.makedirs(figdir, exist_ok=True)

    rows = []
    for cid, md in chapters:
        text = open(md).read()
        names = re.findall(r"\{\{fig:([a-zA-Z0-9_-]+)\}\}", text)
        if a.only_fig_light >= 0 and len(names) > a.only_fig_light:
            continue
        planfile = os.path.join(figdir, cid.replace("/", "__") + ".json")
        if not a.redo and os.path.exists(planfile):
            continue
        rows.append((cid, names, len(names)))

    # figure-poorest + shortest first (highest marginal value)
    rows.sort(key=lambda r: (r[2], len(r[0])))
    if a.limit:
        rows = rows[: a.limit]

    if not rows:
        print("Nothing to strategize (filters left no chapters).", file=sys.stderr)
        return 1

    items = [{"id": cid, "n_existing": n, "prompt": chapter_prompt(cid, names)} for cid, names, n in rows]

    js = f"""export const meta = {{
  name: 'figplan-{a.name}',
  description: 'Strategize foundational infographics for {len(items)} chapters',
  phases: [{{ title: 'Strategize', detail: '{len(items)} chapters, Opus' }}],
}}

const CH = {json.dumps(items, indent=2)}
const SCHEMA = {json.dumps(SCHEMA)}

phase('Strategize')
const results = await parallel(CH.map(c => () =>
  agent(c.prompt, {{ label: `figplan:${{c.id}}`, phase: 'Strategize', schema: SCHEMA,
                    model: 'opus', effort: 'high' }})))

const ok = results.filter(Boolean)
const died = CH.filter((c, i) => !results[i]).map(c => c.id)
let nNew = 0, nRefine = 0, nReplace = 0, nLong = 0
for (const r of ok) {{
  nNew += (r.new_figures || []).length
  nLong += (r.new_figures || []).filter(f => f.kind === 'long-animation').length
  for (const e of (r.existing || [])) {{ if (e.verdict === 'refine') nRefine++; if (e.verdict === 'replace') nReplace++ }}
}}
log(`strategized ${{ok.length}}/${{CH.length}} | new=${{nNew}} (long-anim=${{nLong}}) | refine=${{nRefine}} | replace=${{nReplace}}`)
if (died.length) log(`DIED (re-run to resume): ${{died.join(', ')}}`)
return {{ strategized: ok.length, died, nNew, nLong, nRefine, nReplace }}
"""
    outp = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    open(outp, "w").write(js)
    print(f"wrote {outp}: {len(items)} chapters")
    for it in items[:60]:
        print(f"   [{it['n_existing']} figs] {it['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
