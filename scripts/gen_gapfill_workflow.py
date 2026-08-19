#!/usr/bin/env python3
"""Emit a COMPLETENESS / GAP-FILL workflow for the LLM-stack textbook.

Phase AUDIT (parallel, read-only): one Opus-5 agent per technical Part audits its chapters
  against the book's THREE driving principles and reports genuine, high-value GAPS only:
    (1) whole LLM stack end-to-end, naming the real OSS library at each layer;
    (2) a complete reference for Stanford CS336 "Language Models from Scratch";
    (3) someone with ONLY this book can build a ~100M model end-to-end, no black boxes.
  A gap = a missing subtopic / thin depth / missing derivation / missing runnable code / a
  concept referenced-but-never-explained that a CS336 student or from-scratch builder needs.
  NOT nitpicks, NOT things already covered. Each gap is prioritized 1-5 (5 = most essential).

Curate (JS barrier): keep priority>=4 gaps, group by chapter, cap per chapter and overall so
  the fill stays focused and never bloats.

Phase FILL (parallel by CHAPTER, collision-free): one Opus-5 agent per chapter adds the missing
  content as a focused new section/subsection in the book's house style (prose + optional
  runnable code fence + optional exercise). HARD rules: never add {{fig:}}/{{tool:}} markers
  (keeps figure parity), never alter existing correct content, keep new code correct/CI-runnable,
  no bloat.

Usage: python3 scripts/gen_gapfill_workflow.py --out scripts/wf_gapfill.js [--parts P1 P2 ...]
"""
import argparse, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE = {"00-frontmatter", "13-interview-prep", "99-appendix"}

PRINCIPLES = """The book's THREE driving principles (audit against ALL THREE):
1. WHOLE STACK: teach the entire LLM ecosystem end-to-end, and at every layer name and use the REAL open-source library practitioners use (datatrove, HF tokenizers, TorchTitan, FlashAttention, vLLM, TRL, verl, PEFT, etc.).
2. CS336 REFERENCE: be a complete reference for Stanford CS336 "Language Models from Scratch" — its lectures and its five assignments (basics/tokenizer+transformer+training, systems/FlashAttention+Triton+DDP, scaling laws, data, alignment/RLHF+DPO). A CS336 student should find every concept here, derived, not just named.
3. BUILDABLE, NO BLACK BOXES: someone with ONLY this book can build a ~100M model end-to-end — data -> tokenizer -> pretrain -> mid-train -> post-train -> serve -> agent — with every step derived and runnable, nothing hand-waved."""

AUDIT = r"""You are auditing ONE Part of a large, already-mature LLM-systems textbook for COMPLETENESS GAPS — genuinely missing or too-thin content, NOT bugs and NOT style. The book has already had deep correctness review; do not re-report errors.

""" + PRINCIPLES + r"""

PART TO AUDIT: {PART}   (chapters listed below; read them with Bash `cat`/`grep` over content/{PART}/*.md)

A GAP is one of: a missing subtopic a CS336 student or from-scratch builder would need; a section too thin to actually implement from; a stated result with no derivation where the book's style is to derive; a concept referenced but never explained; a real OSS library/tool for this layer that is never shown. Judge against what is ALREADY in the Part — read enough to be sure it is genuinely absent, not elsewhere in the same Part or an adjacent one. Hold a HIGH bar: this book is thorough, so most Parts will have only a few real gaps, and some none. Do NOT invent gaps to fill a quota; an empty list is a valid, expected answer.

For each real gap: the target chapter (repo-relative path), a short title, gap_type (missing-subtopic|thin-depth|missing-derivation|missing-code|referenced-not-explained|missing-library), a 1-3 sentence description of exactly what is missing, which principle(s) it serves, and priority 1-5 (5 = a CS336 student or 100M-builder is genuinely blocked without it; 1 = nice-to-have). Only report gaps you would stake the book's completeness claim on.

Return the structured findings object."""

FILL = r"""You are FILLING completeness gaps in ONE chapter of a mature LLM-systems textbook. Auditors identified specific missing content; add it as focused new prose/sections in the book's exact house style. This is ADDITIVE — do not rewrite or "improve" existing correct content.

Chapter: {CHAPTER}

Gaps to fill in this chapter (JSON: title, gap_type, description, priority):
{GAPS}

Method: read the chapter first (Bash `cat`) to match its voice, depth, notation, and section conventions, and to be certain the content is genuinely absent. For each gap worth filling, insert a focused new section or subsection at the right place: clear motivating prose, the derivation or mechanism (not just a name), and — where the gap is code/buildable — a SHORT runnable Python fence consistent with the chapter's existing code (correct, minimal imports, actually executes). Add a matching exercise only if the chapter ends in an Exercises section and it fits naturally.

HARD RULES:
- NEVER add a {{fig:...}} or {{tool:...}} marker (they require asset files; adding one breaks the build). Prose, math ($...$/$$...$$), code fences, tables, and admonitions only.
- NEVER alter, delete, or reword existing correct content, exercise structure, or existing markers.
- Any code you add MUST be correct and self-contained enough to run; verify it by reasoning or by executing it with Bash before you write it.
- Match length to need — fill the gap, do not pad. If a listed gap turns out to be already covered or not worth adding, SKIP it.

Return the verdict object (chapter, sections_added, skipped, summary of what you added)."""

FSCHEMA = {"type": "object", "additionalProperties": False, "required": ["part", "gaps"],
           "properties": {"part": {"type": "string"},
                          "gaps": {"type": "array", "maxItems": 30, "items": {
                              "type": "object", "additionalProperties": False,
                              "required": ["chapter", "title", "gap_type", "description", "principles", "priority"],
                              "properties": {"chapter": {"type": "string"}, "title": {"type": "string"},
                                             "gap_type": {"type": "string"}, "description": {"type": "string"},
                                             "principles": {"type": "string"},
                                             "priority": {"type": "integer", "minimum": 1, "maximum": 5}}}}}}
VSCHEMA = {"type": "object", "additionalProperties": False, "required": ["chapter", "sections_added", "skipped", "summary"],
           "properties": {"chapter": {"type": "string"}, "sections_added": {"type": "integer"},
                          "skipped": {"type": "integer"}, "summary": {"type": "string"}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "wf_gapfill.js"))
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--min-priority", type=int, default=4)
    ap.add_argument("--cap-per-chapter", type=int, default=2)
    ap.add_argument("--cap-chapters", type=int, default=16)
    a = ap.parse_args()

    parts = a.parts or sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(os.path.join(ROOT, "content", "*/"))
                              if os.path.basename(p.rstrip("/")) not in EXCLUDE)
    audits = [{"part": p, "prompt": AUDIT.replace("{PART}", p)} for p in parts]

    js = f"""export const meta = {{
  name: 'gapfill',
  description: 'Audit {len(audits)} Parts for completeness gaps -> fill top gaps per chapter (Opus-5)',
  phases: [{{ title: 'Audit' }}, {{ title: 'Fill' }}],
}}
const AUDITS = {json.dumps(audits, ensure_ascii=True)};
const FSCHEMA = {json.dumps(FSCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
const FILL_TMPL = {json.dumps(FILL)};
const MIN_PRIORITY = {a.min_priority};
const CAP_PER_CHAPTER = {a.cap_per_chapter};
const CAP_CHAPTERS = {a.cap_chapters};

log('Gap-fill: auditing ' + AUDITS.length + ' Parts against the 3 principles (read-only)…');
const found = await parallel(AUDITS.map(function (x) {{
  return function () {{
    return agent(x.prompt, {{ label: 'audit:' + x.part, phase: 'Audit', model: 'claude-opus-5', schema: FSCHEMA }})
      .then(function (r) {{ return r; }}).catch(function (e) {{ return null; }});
  }};
}}));

// BARRIER: collect gaps, keep high-priority, group by chapter, cap per chapter + overall.
const byChapter = {{}};
found.filter(Boolean).forEach(function (r) {{
  (r.gaps || []).forEach(function (g) {{
    if (!g.chapter || (g.priority || 0) < MIN_PRIORITY) return;
    (byChapter[g.chapter] = byChapter[g.chapter] || []).push(g);
  }});
}});
// sort chapters by their best gap priority; cap the number of chapters and gaps-per-chapter.
let chapters = Object.keys(byChapter).map(function (c) {{
  const gaps = byChapter[c].sort(function (a, b) {{ return (b.priority||0) - (a.priority||0); }}).slice(0, CAP_PER_CHAPTER);
  const top = Math.max.apply(null, gaps.map(function (g) {{ return g.priority || 0; }}));
  return {{ chapter: c, gaps: gaps, top: top }};
}});
chapters.sort(function (a, b) {{ return b.top - a.top; }});
chapters = chapters.slice(0, CAP_CHAPTERS);
const totalGaps = chapters.reduce(function (s, c) {{ return s + c.gaps.length; }}, 0);
log('Audit surfaced ' + totalGaps + ' priority>=' + MIN_PRIORITY + ' gaps across ' + chapters.length + ' chapters; filling…');

const filled = await parallel(chapters.map(function (c) {{
  return function () {{
    const prompt = FILL_TMPL.replace('{{CHAPTER}}', c.chapter)
      .replace('{{GAPS}}', JSON.stringify(c.gaps.map(function (g) {{ return {{ title: g.title, gap_type: g.gap_type, description: g.description, priority: g.priority }}; }})).slice(0, 6000));
    return agent(prompt, {{ label: 'fill:' + c.chapter, phase: 'Fill', model: 'claude-opus-5', schema: VSCHEMA }})
      .then(function (v) {{ return v; }}).catch(function (e) {{ return {{ chapter: c.chapter, sections_added: 0, skipped: 0, summary: 'ERR ' + e }}; }});
  }};
}}));
const totalAdded = filled.filter(Boolean).reduce(function (s, v) {{ return s + (v.sections_added || 0); }}, 0);
log('Gap-fill: added ' + totalAdded + ' sections across ' + chapters.length + ' chapters.');
return {{ parts: AUDITS.length, gapChapters: chapters.length, gaps: totalGaps, sectionsAdded: totalAdded, byChapter: chapters, results: filled }};
"""
    open(a.out, "w").write(js)
    print(f"Wrote {a.out}: audit {len(audits)} parts -> fill top gaps (min-priority {a.min_priority}, cap {a.cap_per_chapter}/ch, {a.cap_chapters} ch).")


if __name__ == "__main__":
    main()
