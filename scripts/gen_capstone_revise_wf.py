#!/usr/bin/env python3
"""Emit a Workflow that REVISES the capstone chapters against the 3 objectives.

Per chapter (pipeline item): parallel[ Opus-5 review , Fable-5 review ] -> Opus-5 revise-in-place.
Run once per round (--round 1, then --round 2). Reviewers = claude-opus-5 + claude-fable-5,
reviser = claude-opus-5.

Usage: python3 scripts/gen_capstone_revise_wf.py --round 1 --out scripts/wf_caprev1.js
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(ROOT, "capstone", "PLAN.md")
STYLE = os.path.join(ROOT, "STYLE.md")
LINKMAP = os.path.join(ROOT, "LINKMAP.md")

OBJECTIVES = r"""THE THREE OBJECTIVES this book must meet at 2026 STATE-OF-THE-ART quality. Ground EVERY judgement and edit in these:
  (1) WHOLE-STACK + OPEN-SOURCE LIBRARIES: the book teaches the ENTIRE LLM ecosystem stack end to end, and at each level names and correctly uses the REAL open-source libraries/tools that operate there (e.g. HF tokenizers, PyTorch, Triton, FlashAttention, Megatron-LM / DeepSpeed / PyTorch FSDP, TRL / veRL / OpenRLHF, vLLM / SGLang / TensorRT-LLM, HF datasets, LangChain / LlamaIndex, etc.). A reader should learn what each layer is, how it works, AND the libraries that implement it.
  (2) CS336 REFERENCE: the book can serve as a complete reference textbook for everything taught in Stanford CS336 "Language Modeling from Scratch" (tokenization, architectures, systems/kernels, parallelism, scaling laws, data, alignment, inference, evaluation) — matching or exceeding its coverage and rigor.
  (3) BUILDABLE FROM SCRATCH: someone with ONLY this book's knowledge can build a ~100M-parameter model end to end — data -> pretraining -> mid-training -> post-training (SFT/DPO/RLVR) -> a narrow multi-agent / tool-using system — with NOTHING left as an unexplained black box. Every step must be reproducible from what is written."""

REVIEW = r"""You are a demanding technical reviewer for the flagship CAPSTONE (Part XIV) of a definitive LLM-systems textbook. Review ONE chapter HARD against three objectives and return a precise, actionable critique. Do NOT edit anything.

{OBJECTIVES}

Read IN FULL:
  - {ABSPATH}          (the chapter to review)
  - {PLAN}             (the canonical Stack-100M spec the capstone must stay consistent with)

Judge the chapter on: (a) does it advance the 3 objectives above; (b) 2026 SoTA correctness & currency (methods, numbers, library APIs, model/hardware names); (c) buildability — could a reader actually DO this step from what is written, or is something a black box; (d) the specific open-source libraries a practitioner would use here — are they named and used correctly; (e) CS336-level rigor and completeness; (f) code correctness and consistency with PLAN.md; (g) clarity and pedagogy.

Be concrete and specific (quote the passage, name the missing library/method, say exactly what to add or fix). Prioritize the highest-leverage gaps. It is fine to say a chapter is already strong — do not invent problems, but hold a high bar.

Return the structured critique object."""

REVISE = r"""You are the Opus-5 reviser for ONE chapter of the flagship CAPSTONE (Part XIV) of a definitive LLM-systems textbook. Two expert reviewers (Opus-5 and Fable-5) critiqued it against three objectives. Apply the well-founded feedback and REWRITE the chapter in place to raise it to 2026 state-of-the-art, high-quality reference standard.

{OBJECTIVES}

Read IN FULL:
  - {ABSPATH}       (the current chapter)
  - {PLAN}          (canonical Stack-100M spec — stay consistent)
  - {STYLE}         (house style — follow exactly)
  - {LINKMAP}       (valid cross-link targets)

The two reviews (JSON):
{REVIEWS}

Your job:
  - Apply every well-founded suggestion that advances objectives (1)-(3); use judgement, ignore any point that is wrong or would hurt the chapter, and note briefly what you skipped.
  - Strengthen: name and correctly use the real open-source libraries at this layer; close any buildability black box; bring methods/numbers/APIs to verified 2026 SoTA (never fabricate — hedge if unsure); raise rigor to CS336 reference level.
  - PRESERVE what already works. Keep the H1/section structure and house style. Keep every ```code fence runnable and consistent with PLAN.md; keep {{fig:...}} / {{tool:...}} markers intact; keep the `## Exercises` section and its `??? note "Solution"` blocks (you may improve them, but do not drop them). Keep worked examples, Interview Corner, Key Takeaways, and Further reading (extend as needed). Do NOT fabricate citations/numbers/dates.
  - The chapter may grow, but stay focused — depth over padding.

OUTPUT: use the Write tool to save the fully-revised chapter markdown to EXACTLY {ABSPATH} (only the chapter markdown, no preamble). Then return the summary object."""

RSCHEMA = {"type": "object", "additionalProperties": False,
           "required": ["chapter", "overall", "gaps"],
           "properties": {
               "chapter": {"type": "string"},
               "overall": {"type": "string", "description": "1-3 sentence verdict"},
               "strengths": {"type": "array", "items": {"type": "string"}},
               "gaps": {"type": "array", "maxItems": 20, "items": {
                   "type": "object", "additionalProperties": False,
                   "required": ["issue", "objective", "fix", "severity"],
                   "properties": {
                       "issue": {"type": "string"},
                       "objective": {"type": "string", "enum": ["1-stack", "2-cs336", "3-buildable", "quality", "correctness"]},
                       "fix": {"type": "string"},
                       "severity": {"type": "string", "enum": ["high", "medium", "low"]}}}}}}
VSCHEMA = {"type": "object", "additionalProperties": False,
           "required": ["file", "applied", "notes"],
           "properties": {"file": {"type": "string"}, "applied": {"type": "integer"},
                          "skipped": {"type": "integer"}, "notes": {"type": "string"}}}


def flat(book):
    out, pn = [], 0
    for p in book["parts"]:
        front = p["dir"][:2] in ("00", "99")
        if not front:
            pn += 1
        for i, c in enumerate(p["chapters"], 1):
            out.append({"dir": p["dir"], "file": c["file"], "id": c["file"],
                        "abspath": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*", default=[])
    args = ap.parse_args()

    book = json.load(open(os.path.join(ROOT, "book.json")))
    caps = [c for c in flat(book) if c["dir"] == "14-capstone"]
    if args.ids:
        caps = [c for c in caps if c["id"] in args.ids]

    def fill(t, ab):
        return (t.replace("{OBJECTIVES}", OBJECTIVES).replace("{ABSPATH}", ab)
                .replace("{PLAN}", PLAN).replace("{STYLE}", STYLE).replace("{LINKMAP}", LINKMAP))

    jobs = [{"id": c["id"], "abspath": c["abspath"],
             "review": fill(REVIEW, c["abspath"]),
             "revise_tmpl": fill(REVISE, c["abspath"])} for c in caps]

    r = args.round
    js = f"""export const meta = {{
  name: 'capstone-revise-r{r}',
  description: 'Capstone revision round {r}: Opus-5 + Fable-5 review -> Opus-5 revise ({len(jobs)} chapters)',
  phases: [{{ title: 'Review' }}, {{ title: 'Revise' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const RSCHEMA = {json.dumps(RSCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
log('Capstone revision round {r}: ' + JOBS.length + ' chapters (Opus-5 + Fable-5 review -> Opus-5 revise)…');
const results = await pipeline(
  JOBS,
  async function (j) {{
    const reviews = await parallel([
      function () {{ return agent(j.review, {{ label: 'opus5:' + j.id, phase: 'Review', model: 'claude-opus-5', schema: RSCHEMA }}); }},
      function () {{ return agent(j.review, {{ label: 'fable5:' + j.id, phase: 'Review', model: 'claude-fable-5', schema: RSCHEMA }}); }},
    ]);
    return {{ j: j, reviews: reviews.filter(Boolean) }};
  }},
  async function (prev) {{
    if (!prev || !prev.j) return null;
    const rj = JSON.stringify(prev.reviews).slice(0, 12000);
    const v = await agent(prev.j.revise_tmpl.replace('{{REVIEWS}}', rj),
      {{ label: 'revise:' + prev.j.id, phase: 'Revise', model: 'claude-opus-5', schema: VSCHEMA }});
    return {{ id: prev.j.id, reviews: prev.reviews, revise: v }};
  }}
);
const done = results.filter(Boolean);
log('Capstone revision r{r}: ' + done.length + '/' + JOBS.length + ' revised.');
return {{ round: {r}, total: JOBS.length, done: done.length, results: results }};
"""
    open(args.out, "w").write(js)
    print(f"Wrote {args.out}: round {r}, {len(jobs)} chapters "
          f"(reviewers claude-opus-5 + claude-fable-5, reviser claude-opus-5).")


if __name__ == "__main__":
    main()
