#!/usr/bin/env python3
"""Emit a Workflow that REFRESHES chapters to verified 2026 state of the art.

Deeper than gen_sota_workflow (which only adds a resource box). Per chapter, a pipeline:
  Stage REFRESH (Sonnet 5, web): find stale/at-risk claims (model names & versions, benchmark
    numbers, "current best", framework/hardware versions, dated framing, superseded methods),
    research the 2026 reality via WebSearch/WebFetch, and make TARGETED Edits. Also refresh the
    existing `!!! sota` box. NEVER touch code fences, {{fig:}}/{{tool:}} markers, ## Exercises,
    or correct evergreen math. NEVER fabricate a number/date/citation — verify or hedge.
  Stage VERIFY (Opus, web): re-check every changed fact; REVERT or correct anything unverifiable
    or wrong; confirm code/figures/exercises untouched and style intact. Fix in place.

Resumable: pass --exclude-file plan/currency_done.txt (append PASSED ids there after a run).
Idempotent-ish: re-running re-refreshes, so use the exclude file to avoid rework.

Usage:
  python3 scripts/gen_currency_workflow.py --out scripts/wf_cur1.js --name cur1 \
      --parts 07-inference-serving 08-agents-harness [--limit N] [--exclude-file plan/currency_done.txt]
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fast-moving parts get Opus on BOTH stages; evergreen parts keep Sonnet refresh + Opus verify.
HIGH_CHURN = {"02-transformer", "03-pretraining", "05-posttraining-alignment", "06-rl-infra",
              "07-inference-serving", "08-agents-harness", "09-rag-retrieval",
              "10-multimodal-and-arch", "11-evaluation", "13-interp-safety-gov"}


def flat(book):
    out = []
    for p in book["parts"]:
        for c in p["chapters"]:
            out.append({"id": f"{p['dir']}/{c['file']}", "dir": p["dir"],
                        "abspath": os.path.join(ROOT, "content", p["dir"], c["file"] + ".md"),
                        "title": c["title"], "part": p["title"]})
    return out


REFRESH = r"""You are the 2026 CURRENCY EDITOR for ONE chapter of a public, award-quality web textbook, "The LLM Stack: From Silicon to Agents." Your job: bring this chapter up to date with the VERIFIED 2026 state of the art, with surgical edits — not a rewrite. Today is 2026; treat anything framed as 2023/2024/2025-current as potentially stale.

Chapter file (READ IT IN FULL FIRST): {ABSPATH}
Title: {TITLE}
Part: {PART}

STEP 1 — Find currency risks in the BODY and the existing `!!! sota` box:
  - Model names/versions presented as current/frontier (GPT-4/4o, Claude 3.x, Llama 2/3, Gemini 1.x, DeepSeek V2/V3, Qwen2/2.5, Mistral, etc.) — is there a newer standard-bearer in 2026?
  - Benchmark/SOTA numbers, "the best X is…", leaderboards, context-length norms, price-per-token.
  - Framework/library/hardware versions & norms (PyTorch, vLLM, FlashAttention v2/v3, TensorRT-LLM, SGLang; H100/H200 vs B200/GB200; CUDA).
  - Dated framing ("as of 2024", "recently", "current", "state of the art is") that a 2026 reader would find stale.
  - Methods that have been superseded or become standard since the chapter was written.

STEP 2 — RESEARCH the 2026 reality with WebSearch + WebFetch before changing anything. VERIFY every new fact/number/model-name/date against a real, current source (official docs/blog, arXiv abstract, reputable report). Prefer canonical, durable URLs. If you cannot verify a specific number, do NOT invent one — either keep the old value with softened framing ("on the order of", "as of writing") or state the direction of change qualitatively. NEVER fabricate a model name, version, date, arXiv id, repo path, or benchmark score.

STEP 3 — Make TARGETED Edits with the Edit tool (small, precise replacements):
  - Update stale facts to the verified 2026 reality; update tense/framing so it reads correctly in 2026.
  - Where a major newer development materially changes the picture, add at most 1-3 sentences (do not bloat; keep the chapter's length and pedagogy roughly intact). Prefer updating an existing sentence over adding new ones.
  - Refresh the existing `!!! sota "State of the Art & Resources (2026)"` box: verify its links resolve (WebFetch), replace dead/superseded entries with current ones, keep 8-12 verified links, keep the exact 4-space-indented admonition format.
  - Preserve the teaching. First-principles explanations, derivations, and worked mechanics are usually EVERGREEN — leave them alone.

HARD CONSTRAINTS (do NOT violate):
  - Do NOT modify any fenced code block, any `{{fig:...}}` or `{{tool:...}}` marker line, the `## Exercises` section or its `??? note "Solution"` blocks, or correct mathematics. This chapter's code is CI-tested and its figures/exercises are separately verified — leave them byte-for-byte.
  - Do NOT fabricate citations, URLs, dates, versions, or benchmark numbers. Verified-or-hedged only.
  - Keep the H1, section structure, and house style intact. If nothing is stale, make NO changes and say so — do not churn evergreen content.

STEP 4 — Return the structured report object: list every change you made with a short before/after, why, and the verifying source URL; and note anything you deliberately left because you could not verify a newer value."""

VERIFY = r"""You are the Opus fact-checker for a 2026 CURRENCY REFRESH of ONE textbook chapter. A refresh editor just made targeted edits. Verify them hard and fix any problem in place.

Chapter file (READ IN FULL): {ABSPATH}
Title: {TITLE}

The refresh editor's change report (what they claim to have changed and their sources):
{REPORT}

STEP 1 — For EACH claimed change, independently verify with WebSearch/WebFetch that the new fact/number/model-name/version/date is accurate as of 2026 and that the cited source actually supports it. Be skeptical: a plausible-sounding "update" that is not verifiable is WORSE than the original.

STEP 2 — Fix in place with the Edit tool:
  - REVERT or correct any change that is unverifiable, wrong, or fabricated (wrong model name, invented number, dead/incorrect URL, hallucinated version).
  - Ensure NO fabricated numbers/dates/citations remain anywhere you touched.
  - Confirm the HARD CONSTRAINTS held: fenced code, {{fig:}}/{{tool:}} markers, the `## Exercises` section and its solutions, and correct math are UNCHANGED. If the editor altered any of them, restore them.
  - Confirm the `!!! sota` box format is valid (4-space indent) and its links resolve; drop any that 404.
  - Confirm style/structure intact and the chapter still reads coherently.

STEP 3 — Return the verdict object: how many changes you confirmed, how many you reverted/corrected, whether constraints held, and any residual risk."""

RSCHEMA = {"type": "object", "additionalProperties": False, "required": ["chapter", "n_changes", "summary"],
           "properties": {"chapter": {"type": "string"}, "n_changes": {"type": "integer"},
                          "changes": {"type": "array", "maxItems": 40, "items": {"type": "object",
                              "additionalProperties": False, "required": ["what", "verified"],
                              "properties": {"what": {"type": "string"}, "before": {"type": "string"},
                                  "after": {"type": "string"}, "source": {"type": "string"},
                                  "verified": {"type": "boolean"}}}},
                          "left_unverified": {"type": "array", "items": {"type": "string"}},
                          "summary": {"type": "string"}}}
VSCHEMA = {"type": "object", "additionalProperties": False,
           "required": ["chapter", "confirmed", "reverted", "constraints_held", "notes"],
           "properties": {"chapter": {"type": "string"}, "confirmed": {"type": "integer"},
                          "reverted": {"type": "integer"}, "constraints_held": {"type": "boolean"},
                          "notes": {"type": "string"}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest", default="book.json")
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-file", default=None)
    args = ap.parse_args()

    excluded = set()
    if args.exclude_file and os.path.exists(args.exclude_file):
        excluded = {l.strip() for l in open(args.exclude_file) if l.strip() and not l.startswith("#")}

    book = json.load(open(os.path.join(ROOT, args.manifest)))
    sel = []
    for c in flat(book):
        if c["dir"].startswith(("00", "99")) or c["dir"] == "14-capstone":
            continue
        if args.parts and c["dir"] not in args.parts:
            continue
        if args.ids and c["id"] not in args.ids:
            continue
        if c["id"] in excluded:
            continue
        sel.append(c)
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        print("No chapters selected.")
        return

    jobs = [{"id": c["id"], "abspath": c["abspath"], "title": c["title"],
             "refresh_model": "opus" if c["dir"] in HIGH_CHURN else "sonnet",
             "refresh": REFRESH.replace("{ABSPATH}", c["abspath"]).replace("{TITLE}", c["title"]).replace("{PART}", c["part"]),
             "verify_tmpl": VERIFY.replace("{ABSPATH}", c["abspath"]).replace("{TITLE}", c["title"])}
            for c in sel]

    js = f"""export const meta = {{
  name: 'currency-{args.name}',
  description: 'Refresh {len(sel)} chapters to verified 2026 SOTA (web research+edit -> Opus fact-check)',
  phases: [{{ title: 'Refresh' }}, {{ title: 'Verify' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const RSCHEMA = {json.dumps(RSCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
log('Currency refresh {args.name}: ' + JOBS.length + ' chapters (web refresh -> Opus verify)…');
const results = await pipeline(
  JOBS,
  function (j) {{
    return agent(j.refresh, {{ label: 'refresh:' + j.id, phase: 'Refresh', model: j.refresh_model, schema: RSCHEMA }})
      .then(function (r) {{ return {{ j: j, report: r }}; }})
      .catch(function (e) {{ return {{ j: j, report: null, err: String(e) }}; }});
  }},
  async function (prev) {{
    if (!prev || !prev.j) return null;
    const rep = prev.report ? JSON.stringify(prev.report).slice(0, 6000) : '(refresh stage returned no structured report; verify the file directly)';
    const v = await agent(prev.j.verify_tmpl.replace('{{REPORT}}', rep),
      {{ label: 'verify:' + prev.j.id, phase: 'Verify', model: 'opus', schema: VSCHEMA }});
    return {{ id: prev.j.id, report: prev.report, verify: v }};
  }}
);
const done = results.filter(Boolean);
const changed = done.filter(function (r) {{ return r.report && r.report.n_changes > 0; }}).length;
log('Currency {args.name}: ' + done.length + '/' + JOBS.length + ' processed, ' + changed + ' had updates.');
return {{ batch: '{args.name}', total: JOBS.length, done: done.length, results: results }};
"""
    with open(args.out, "w") as f:
        f.write(js)
    n_opus = sum(1 for c in sel if c["dir"] in HIGH_CHURN)
    print(f"Wrote {args.out}: {len(sel)} chapters ({n_opus} high-churn Opus-refresh, {len(sel)-n_opus} Sonnet-refresh; all Opus-verify).")
    for c in sel:
        print(f"  [{'opus  ' if c['dir'] in HIGH_CHURN else 'sonnet'}] {c['id']}")


if __name__ == "__main__":
    main()
