#!/usr/bin/env python3
"""Emit a Workflow that revises NON-capstone chapters toward the 3 objectives (Opus-5).

One Opus-5 agent per chapter makes TARGETED Edit-based improvements (not a rewrite):
name the real OSS library at this layer, close buildability black boxes, ensure CS336
coverage/rigor, fix correctness/currency, and cross-link the capstone where relevant.
Preserves code / figures / exercises / structure. Batched + resumable via a done-file.

Usage: python3 scripts/gen_bookrevise_workflow.py --out scripts/wf_bookrev1.js --name b1 \
         --parts 02-transformer 03-pretraining --exclude-file plan/bookrevise_done.txt
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE = os.path.join(ROOT, "STYLE.md")
LINKMAP = os.path.join(ROOT, "LINKMAP.md")
EXCLUDE_PARTS = {"00-frontmatter", "13-interview-prep", "14-capstone", "99-appendix"}

OBJECTIVES = r"""THE THREE OBJECTIVES the whole book must meet at 2026 STATE-OF-THE-ART quality. Ground every judgement and edit in these:
  (1) WHOLE-STACK + OPEN-SOURCE LIBRARIES: teach the ENTIRE LLM ecosystem stack end to end, and at each level name and correctly use the REAL open-source libraries/tools that operate there (e.g. HF tokenizers, PyTorch, Triton, FlashAttention, Megatron-LM / DeepSpeed / PyTorch FSDP, TRL / veRL / OpenRLHF, vLLM / SGLang / TensorRT-LLM, HF datasets / datatrove, lm-evaluation-harness, LangChain / LlamaIndex, llama.cpp/GGUF). A reader should learn what each layer is, how it works, AND the library that implements it.
  (2) CS336 REFERENCE: serve as a complete reference for everything in Stanford CS336 "Language Modeling from Scratch" — matching or exceeding its coverage and rigor.
  (3) BUILDABLE FROM SCRATCH: only this book's knowledge should suffice to build a ~100M model end to end — data -> pretraining -> mid-training -> post-training (SFT/DPO/RLVR) -> a narrow multi-agent/tool-using system — with nothing left as an unexplained black box."""

PROMPT = r"""You are improving ONE chapter of a definitive LLM-systems textbook toward three objectives. Make TARGETED improvements IN PLACE with the Edit tool — do NOT rewrite the chapter wholesale and do NOT pad. Most chapters are already strong from prior passes; your job is surgical, high-leverage edits.

{OBJECTIVES}

Read IN FULL first:
  - {ABSPATH}     (the chapter)
  - {STYLE}       (house style)
  - {LINKMAP}     (valid cross-link targets; the capstone is Part XIV under ../14-capstone/)

Then improve ONLY where the chapter genuinely falls short of an objective:
  - OBJ 1: if the real open-source library/tool for this layer is missing or under-explained, name it and show (briefly) how it is used in practice, alongside the from-scratch mechanism.
  - OBJ 3: close any buildability black box relevant to building a ~100M model end to end; make sure a reader could actually DO the thing.
  - OBJ 2: ensure CS336-level coverage and rigor of this topic; fill a genuine conceptual gap if one exists.
  - CORRECTNESS/CURRENCY: fix any wrong or stale claim; bring to verified 2026 SoTA; NEVER fabricate a number/date/citation/API (hedge if unsure).
  - Where this chapter's technique is used to build Stack-100M, add a short cross-link to the relevant Part-XIV capstone chapter (../14-capstone/<file>.html).

HARD CONSTRAINTS:
  - Use targeted Edit calls. PRESERVE the H1/section structure and house style; every ```code fence must stay runnable; keep all {{fig:...}} / {{tool:...}} markers, the `## Exercises` section and its `??? note "Solution"` blocks, the Interview Corner, Key Takeaways, and any `!!! sota` box.
  - Do NOT fabricate. Do NOT bloat: prefer improving or extending an existing sentence over adding paragraphs. If the chapter is already excellent on all three objectives, make minimal or no changes.

Return the summary object (what you changed and which objective each change served)."""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["file", "changed", "notes"],
          "properties": {"file": {"type": "string"}, "changed": {"type": "boolean"},
                         "n_edits": {"type": "integer"},
                         "objectives": {"type": "array", "items": {"type": "string"}},
                         "notes": {"type": "string"}}}


def flat(book):
    out = []
    for p in book["parts"]:
        for c in p["chapters"]:
            out.append({"id": f"{p['dir']}/{c['file']}", "dir": p["dir"],
                        "abspath": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--exclude-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    excluded = set()
    if args.exclude_file and os.path.exists(args.exclude_file):
        excluded = {l.strip() for l in open(args.exclude_file) if l.strip() and not l.startswith("#")}

    book = json.load(open(os.path.join(ROOT, "book.json")))
    sel = []
    for c in flat(book):
        if c["dir"] in EXCLUDE_PARTS:
            continue
        if args.parts and c["dir"] not in args.parts:
            continue
        if c["id"] in excluded:
            continue
        sel.append(c)
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        print("No chapters selected.")
        return

    jobs = [{"id": c["id"],
             "prompt": (PROMPT.replace("{OBJECTIVES}", OBJECTIVES).replace("{ABSPATH}", c["abspath"])
                        .replace("{STYLE}", STYLE).replace("{LINKMAP}", LINKMAP))}
            for c in sel]

    js = f"""export const meta = {{
  name: 'bookrevise-{args.name}',
  description: 'Revise {len(sel)} chapters toward the 3 objectives (Opus-5, targeted edits)',
  phases: [{{ title: 'Revise' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const SCHEMA = {json.dumps(SCHEMA)};
log('Book revision {args.name}: ' + JOBS.length + ' chapters (Opus-5 targeted edits toward 3 objectives)…');
const results = await parallel(JOBS.map(function (j) {{
  return function () {{
    return agent(j.prompt, {{ label: 'revise:' + j.id, phase: 'Revise', model: 'claude-opus-5', schema: SCHEMA }})
      .then(function (r) {{ return {{ id: j.id, rec: r }}; }})
      .catch(function (e) {{ return {{ id: j.id, failed: String(e) }}; }});
  }};
}}));
const done = results.filter(function (r) {{ return r && !r.failed; }});
log('Book revision {args.name}: ' + done.length + '/' + JOBS.length + ' processed.');
return {{ batch: '{args.name}', total: JOBS.length, done: done.length, results: results }};
"""
    open(args.out, "w").write(js)
    print(f"Wrote {args.out}: {len(sel)} chapters (model=claude-opus-5).")


if __name__ == "__main__":
    main()
