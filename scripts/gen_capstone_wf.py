#!/usr/bin/env python3
"""Emit the Part-XIV Capstone workflow: per chapter DRAFT (model per book.json) -> OPUS verify+revise.

Pipeline (not barrier): each chapter verifies as soon as its draft lands.
Every agent reads STYLE.md + LINKMAP.md + capstone/PLAN.md (the canonical spec).
Usage: python3 scripts/gen_capstone_wf.py            # -> scripts/wf_capstone.js
       python3 scripts/gen_capstone_wf.py --ids 04-architecture 05-mini-scaling-laws  # subset
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE = os.path.join(ROOT, "STYLE.md")
LINKMAP = os.path.join(ROOT, "LINKMAP.md")
PLAN = os.path.join(ROOT, "capstone", "PLAN.md")


def flat(book):
    out, pn = [], 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        for i, c in enumerate(p["chapters"], 1):
            out.append({
                "dir": p["dir"], "file": c["file"],
                "id": c["file"],
                "abspath": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md"),
                "num": "" if front else f"{pn}.{i}",
                "title": c["title"], "part": p["title"],
                "scope": c.get("scope", ""), "words": c.get("words", 5500),
                "model": c.get("model", "sonnet"),
            })
    return out


DRAFT = (
    "You are co-authoring THE definitive, award-quality web textbook on the whole LLM stack. "
    "You are writing one chapter of the flagship CAPSTONE (Part XIV): a single coherent project that "
    "builds a ~100M-parameter model end to end. Coherence across the 12 capstone chapters is critical.\n\n"
    "STEP 1 — Read these THREE files IN FULL before writing:\n"
    "  - {STYLE}      (house style — follow exactly)\n"
    "  - {LINKMAP}    (exact cross-reference targets)\n"
    "  - {PLAN}       (THE canonical capstone spec: Stack-100M config, data mix, ~20B-token budget, "
    "compute tiers, and every SOTA component with its citation — you MUST stay consistent with it)\n\n"
    "STEP 2 — Write ONE complete chapter:\n"
    "  Chapter number: {NUM}\n  Title: {TITLE}\n  Part: {PART}\n  Scope: {SCOPE}\n"
    "  Target length: about {WORDS} words — a deep, concrete reference chapter. Add mechanism, real "
    "runnable heavily-commented code, and worked numerical examples rather than filler.\n\n"
    "HARD REQUIREMENTS (STYLE.md, render-critical):\n"
    "  - Begin with a single H1: \"# {NUM} {TITLE}\".\n"
    "  - ## sections and ### subsections; 4-8 major sections; never skip levels.\n"
    "  - KaTeX math: inline $...$, display $$...$$ on their own lines.\n"
    "  - Every code fence declares a language. Include SUBSTANTIAL, correct, copy-paste-runnable, "
    "heavily-commented PyTorch/Python. Code MUST be consistent with the Stack-100M config and names in PLAN.md "
    "and with the code in sibling capstone chapters (same package `stacklm`, same module/function names when "
    "you reference them). Prefer real, working implementations over pseudocode.\n"
    "  - >=1 worked numerical example (!!! example) with real magnitudes; >=1 Interview Corner (!!! interview); "
    "end with a Key Takeaways box (!!! key, 5-9 bullets) and a 'Further reading' list of REAL landmark works "
    "named by author/title. NEVER fabricate citations, URLs, exact benchmark numbers, dates, or quotes; "
    "illustrative numbers are 'on the order of'.\n"
    "  - Cross-link the deeper book chapters this builds on using LINKMAP.md targets: [Title](../dir/file.html).\n"
    "  - Admonition bodies indented exactly 4 spaces; blank line before and after each block.\n\n"
    "STEP 3 — OUTPUT: use Write to save the COMPLETE chapter markdown to EXACTLY:\n    {ABSPATH}\n"
    "The file must contain ONLY the chapter markdown (no preamble). Then return the summary object."
)

VERIFY = (
    "You are the Opus verifier/reviser for one chapter of the flagship CAPSTONE (Part XIV) of a definitive "
    "LLM-stack textbook. A draft already exists on disk. Your job: make it correct, rigorous, and consistent "
    "with the canonical spec — then REWRITE it in place.\n\n"
    "STEP 1 — Read IN FULL:\n"
    "  - {ABSPATH}   (the current draft)\n"
    "  - {PLAN}      (canonical Stack-100M spec — the draft MUST match its config, numbers, budget, components, citations)\n"
    "  - {STYLE}     (house style)\n"
    "  - {LINKMAP}   (valid cross-link targets)\n\n"
    "STEP 2 — Verify hard, fixing every issue you find:\n"
    "  - CORRECTNESS: re-derive the math and re-check the CODE mentally (shapes, config values, param counts, "
    "APIs). Fix bugs. Code must be runnable and match the Stack-100M config in PLAN.md exactly.\n"
    "  - CONSISTENCY: model name (Stack-100M), package (stacklm), the exact hyperparameters, the ~20B-token "
    "over-training budget, the ~$100/1xA100 framing, and component choices must match PLAN.md and be plausible "
    "against sibling chapters. No drift.\n"
    "  - CITATIONS: every named work must be a real landmark (RoPE/Su, RMSNorm/Zhang-Sennrich, GQA/Ainslie, "
    "SwiGLU/Shazeer, Muon/Jordan, WSD/MiniCPM, DPO/Rafailov, GRPO/DeepSeekMath-Shao, ReAct/Yao, MLA & MTP/DeepSeek, "
    "SmolLM/HuggingFace, MobileLLM, Chinchilla/Hoffmann, etc.). Remove or fix anything fabricated or dubious; "
    "no invented benchmark numbers/dates/quotes.\n"
    "  - STYLE: H1 with number, section depth, admonitions (4-space indent), one Interview Corner, Key Takeaways, "
    "Further reading, generous cross-links via LINKMAP.md. Ensure ~{WORDS} words of real depth.\n\n"
    "STEP 3 — OUTPUT: use Write to save the FULLY-REVISED chapter markdown back to EXACTLY {ABSPATH} "
    "(only the chapter markdown, no preamble). Then return the summary object noting what you fixed."
)

DSCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "file": {"type": "string"}, "words_written": {"type": "integer"},
    "sections": {"type": "integer"}, "code_blocks": {"type": "integer"},
    "one_line_summary": {"type": "string"}},
    "required": ["file", "words_written", "one_line_summary"]}
VSCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "file": {"type": "string"}, "issues_fixed": {"type": "integer"},
    "consistency_ok": {"type": "boolean"}, "citations_ok": {"type": "boolean"},
    "code_ok": {"type": "boolean"}, "notes": {"type": "string"}},
    "required": ["file", "consistency_ok", "notes"]}


def fill(t, c):
    return (t.replace("{STYLE}", STYLE).replace("{LINKMAP}", LINKMAP).replace("{PLAN}", PLAN)
            .replace("{NUM}", c["num"]).replace("{TITLE}", c["title"]).replace("{PART}", c["part"])
            .replace("{SCOPE}", c["scope"]).replace("{WORDS}", str(c["words"])).replace("{ABSPATH}", c["abspath"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "wf_capstone.js"))
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--draft-only", action="store_true")
    args = ap.parse_args()

    book = json.load(open(os.path.join(ROOT, "book.json")))
    caps = [c for c in flat(book) if c["dir"] == "14-capstone"]
    if args.ids:
        caps = [c for c in caps if c["id"] in args.ids]
    jobs = [{"id": c["id"], "abspath": c["abspath"], "model": c["model"], "words": c["words"],
             "draft": fill(DRAFT, c), "verify": fill(VERIFY, c)} for c in caps]

    stage2 = "" if args.draft_only else """,
  async function (draftRes, j) {
    const v = await agent(j.verify, { label: 'verify:' + j.id, phase: 'Verify', model: 'opus', schema: VSCHEMA });
    return { id: j.id, draft: draftRes, verify: v };
  }"""

    js = f"""export const meta = {{
  name: 'capstone-partXIV',
  description: 'Draft + Opus-verify the 12 Capstone chapters (Build Stack-100M end-to-end)',
  phases: [{{ title: 'Draft' }}, {{ title: 'Verify' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const DSCHEMA = {json.dumps(DSCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
log('Capstone Part XIV: drafting ' + JOBS.length + ' chapters (draft -> Opus verify+revise)…');
const results = await pipeline(
  JOBS,
  function (j) {{
    return agent(j.draft, {{ label: 'draft:' + j.id, phase: 'Draft', model: j.model, schema: DSCHEMA }})
      .then(function (r) {{ return j; }});
  }}{stage2}
);
const done = results.filter(Boolean);
log('Capstone: ' + done.length + '/' + JOBS.length + ' chapters through the pipeline (verify on disk via qa.py).');
return {{ total: JOBS.length, done: done.length, results: results }};
"""
    with open(args.out, "w") as f:
        f.write(js)
    print(f"Wrote {args.out}: {len(jobs)} chapters "
          f"(~{sum(c['words'] for c in caps):,} target words). Models: "
          + ", ".join(f"{c['id']}={c['model']}" for c in caps))


if __name__ == "__main__":
    main()
