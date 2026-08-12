#!/usr/bin/env python3
"""Emit a CROSS-CHAPTER CONSISTENCY workflow (the class the per-chapter red-team can't catch):
  Phase FIND (parallel, read-only): one Opus-5 agent per THEME greps the whole book for that
    theme's recurring facts/numbers, decides the canonical value (naming the authoritative
    chapter), and reports every DISAGREEING mention (file + quote + canonical) — NO edits.
  Barrier: group all reported disagreements by file.
  Phase FIX (parallel by FILE, so no two agents edit the same file): one Opus-5 agent per file
    applies the canonical value to each flagged spot, verified.

Usage: python3 scripts/gen_consistency_workflow.py --out scripts/wf_consistency.js
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THEMES = [
 {"key": "hardware-specs",
  "desc": "GPU hardware specs stated across the book: NVIDIA A100 (peak bf16 dense TFLOP/s and HBM bandwidth), H100 (peak bf16 dense TFLOP/s ~989 and HBM3 bandwidth ~3.35 TB/s), H200, B200/Blackwell, and any per-GPU FLOP/bandwidth figures. Every chapter that quotes these must agree (e.g. H100 dense bf16 must be the SAME number everywhere). The authoritative chapter is 01-foundations/08-gpu-architecture."},
 {"key": "stack100m-config",
  "desc": "The Stack-100M capstone config wherever it appears book-wide: d_model=512, n_layers=30, n_heads=8, n_kv_heads=2, head_dim=64, SwiGLU intermediate=1408, vocab=32768, ~101M / 101,353,728 total params, ~84.6M non-embedding. Authoritative source is capstone/PLAN.md. Flag any chapter that states a different Stack-100M number."},
 {"key": "capstone-cost-mfu",
  "desc": "The capstone cost/throughput/MFU numbers: pretraining ~22-29 GPU-hr (~25 planning), whole project ~USD 90-100, the measured throughput (14.7 is authoritative at 227,951 tok/s; RECONCILE 14.12's 231,500 tok/s and any other value), and the MFU conventions (0.45-0.58 attention-inclusive == 0.34-0.45 under 6ND; 44.5% MFU(6ND) == 58.2% attention-inclusive). Everything must agree across 14.1/14.5/14.7/14.8/14.12 and capstone/PLAN.md/README."},
 {"key": "core-formulas-constants",
  "desc": "Recurring formulas/constants that must be stated identically everywhere: the 6ND training-FLOP rule, KV-cache size = 2*L*n_kv*head_dim*seq*bytes, Chinchilla ~20 tokens/param (compute-optimal), Adam optimizer state 16 bytes/param (mixed precision), roofline ridge point = peak_FLOPs/bandwidth. Flag any chapter that writes one of these with a different coefficient/exponent/factor."},
 {"key": "benchmark-citations",
  "desc": "Recurring benchmark/speedup figures and paper attributions cited in more than one chapter: FlashAttention speedups, Sarathi-Serve (2.6-5.6x), PagedAttention, vLLM star count, Chinchilla/Kaplan, Muon/Kimi, DeepSeek MLA/MTP, etc. Flag any two chapters that cite the SAME result with DIFFERENT numbers or a different author/year."},
 {"key": "model-landscape-facts",
  "desc": "Recurring facts about named real models stated in multiple chapters: parameter counts, context-window lengths, vocab sizes, and release framing for GPT-4o, Llama 3/4, Qwen3, DeepSeek-V3/R1, SmolLM, Gemma, Mistral, etc. Flag any chapter that gives a different number for the same model than another chapter does."},
]

FIND = r"""You are auditing a large LLM-systems textbook for CROSS-CHAPTER INCONSISTENCY on ONE theme — cases where two or more chapters state the SAME fact/number with DIFFERENT values. This is the one defect class a per-chapter review cannot catch. Do NOT edit anything; report only.

THEME: {DESC}

Method: use Bash `grep`/`rg` over content/**/*.md (and capstone/PLAN.md, capstone/README.md where relevant) to find every place this theme's facts/numbers are stated. Determine the CANONICAL value (prefer the authoritative chapter named above; otherwise the value the clear majority use and that is arithmetically/physically correct). Then list every mention that DISAGREES with the canonical value. Read enough context to be sure it is a genuine contradiction, not two different quantities that merely look similar. Ignore an honestly-hedged range vs a point value if they are compatible.

For each disagreement report: the file (repo-relative), a short exact quote of the wrong text, the canonical value it should be, and why (which chapter/source is authoritative). Only HIGH-confidence genuine contradictions. If the theme is already consistent book-wide, return an empty list.

Return the structured findings object."""

FIX = r"""You are fixing CROSS-CHAPTER CONSISTENCY defects in ONE file of a textbook. Auditors found places in this file that state a shared fact with the WRONG value; correct each to the canonical value. Verify before editing; do not fabricate.

File: {FILE}

Flagged inconsistencies in this file (JSON: quote, canonical, why):
{FINDINGS}

For each: locate the quoted text, confirm it genuinely disagrees with the canonical value (re-derive/sanity-check the number), and apply a minimal Edit changing ONLY the wrong value/claim to the canonical one, keeping the surrounding sentence and any dependent arithmetic self-consistent. If a flagged item is actually correct (a different quantity, or the flagged one is itself the canonical/authoritative statement), SKIP it. Do NOT touch code fences unless the wrong value is literally inside one, and never touch {{fig:}}/{{tool:}} markers, exercise structure, or correct content.

Return the verdict object (file, fixed count, skipped count, notes)."""

FSCHEMA = {"type": "object", "additionalProperties": False, "required": ["theme", "canonical_summary", "disagreements"],
           "properties": {"theme": {"type": "string"}, "canonical_summary": {"type": "string"},
                          "disagreements": {"type": "array", "maxItems": 40, "items": {
                              "type": "object", "additionalProperties": False,
                              "required": ["file", "quote", "canonical", "why"],
                              "properties": {"file": {"type": "string"}, "quote": {"type": "string"},
                                             "canonical": {"type": "string"}, "why": {"type": "string"}}}}}}
VSCHEMA = {"type": "object", "additionalProperties": False, "required": ["file", "fixed", "skipped", "notes"],
           "properties": {"file": {"type": "string"}, "fixed": {"type": "integer"},
                          "skipped": {"type": "integer"}, "notes": {"type": "string"}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "wf_consistency.js"))
    a = ap.parse_args()
    themes = [{"key": t["key"], "find": FIND.replace("{DESC}", t["desc"])} for t in THEMES]

    js = f"""export const meta = {{
  name: 'cross-chapter-consistency',
  description: 'Find cross-chapter inconsistencies ({len(themes)} themes) -> fix per file (Opus-5)',
  phases: [{{ title: 'Find' }}, {{ title: 'Fix' }}],
}}
const THEMES = {json.dumps(themes, ensure_ascii=True)};
const FSCHEMA = {json.dumps(FSCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
const FIX_TMPL = {json.dumps(FIX)};

log('Cross-chapter consistency: scanning ' + THEMES.length + ' themes (read-only)…');
// PHASE 1: find per theme (parallel, no edits)
const found = await parallel(THEMES.map(function (t) {{
  return function () {{
    return agent(t.find, {{ label: 'find:' + t.key, phase: 'Find', model: 'claude-opus-5', schema: FSCHEMA }})
      .then(function (r) {{ return r; }}).catch(function (e) {{ return null; }});
  }};
}}));

// BARRIER: regroup all disagreements by FILE (so no two fix-agents edit the same file)
const byFile = {{}};
found.filter(Boolean).forEach(function (r) {{
  (r.disagreements || []).forEach(function (d) {{
    if (!d.file) return;
    (byFile[d.file] = byFile[d.file] || []).push({{ quote: d.quote, canonical: d.canonical, why: d.why }});
  }});
}});
const files = Object.keys(byFile);
log('Found disagreements in ' + files.length + ' files; fixing per file…');

// PHASE 2: fix per file (parallel over distinct files -> no collision)
const fixed = await parallel(files.map(function (f) {{
  return function () {{
    const prompt = FIX_TMPL.replace('{{FILE}}', f).replace('{{FINDINGS}}', JSON.stringify(byFile[f]).slice(0, 8000));
    return agent(prompt, {{ label: 'fix:' + f, phase: 'Fix', model: 'claude-opus-5', schema: VSCHEMA }})
      .then(function (v) {{ return v; }}).catch(function (e) {{ return {{ file: f, fixed: 0, skipped: 0, notes: 'ERR ' + e }}; }});
  }};
}}));
const total = fixed.filter(Boolean).reduce(function (s, v) {{ return s + (v.fixed || 0); }}, 0);
log('Consistency: ' + total + ' cross-chapter fixes across ' + files.length + ' files.');
return {{ themes: THEMES.length, files: files.length, fixes: total, findings: byFile, results: fixed }};
"""
    open(a.out, "w").write(js)
    print(f"Wrote {a.out}: {len(themes)} themes (find -> fix-per-file).")


if __name__ == "__main__":
    main()
