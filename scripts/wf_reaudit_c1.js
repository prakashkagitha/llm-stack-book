export const meta = {
  name: 'audit-c1',
  description: 'Audit The LLM Stack against CS336 + train-from-scratch buildability',
  phases: [{ title: 'Audit', detail: '1 benchmark units graded independently' }],
}

const UNITS = [
  {
    "id": "C1-end-to-end-pretrain",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    C1-end-to-end-pretrain\nkind:  capability\ntitle: END-TO-END: train an LLM from scratch\n\nLearning objectives this unit demands:\n  - A single coherent path a reader can follow from raw text to a trained, sampled-from model\n  - Every stage connected: data -> tokenizer -> model -> optimizer -> training loop -> checkpoint -> eval -> inference\n  - Concrete configs and compute budgets at multiple scales (laptop / 1 GPU / 8 GPU / multi-node)\n  - Explicit statement of what to expect: loss curves, timings, failure signatures\n\nImplementable deliverable(s) a reader must be able to produce:\n  - A reader with GPUs can go from raw corpus to a working trained LM using only this book.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/C1-end-to-end-pretrain.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  }
]
const SCHEMA = {
  "type": "object",
  "additionalProperties": false,
  "required": [
    "unit_id",
    "grade",
    "chapters_examined",
    "deficits",
    "buildable_verdict",
    "summary"
  ],
  "properties": {
    "unit_id": {
      "type": "string"
    },
    "grade": {
      "type": "integer",
      "minimum": 0,
      "maximum": 4
    },
    "chapters_examined": {
      "type": "array",
      "maxItems": 30,
      "items": {
        "type": "string"
      },
      "description": "chapter ids actually read, e.g. 02-transformer/01-tokenization"
    },
    "evidence": {
      "type": "array",
      "maxItems": 20,
      "items": {
        "type": "string"
      },
      "description": "short quotes/observations justifying the grade"
    },
    "buildable_verdict": {
      "type": "boolean",
      "description": "true only if a competent reader could produce a CORRECT working implementation of this unit's deliverable from the book alone"
    },
    "buildable_reasoning": {
      "type": "string"
    },
    "deficits": {
      "type": "array",
      "maxItems": 25,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "severity",
          "target",
          "missing",
          "fix"
        ],
        "properties": {
          "severity": {
            "type": "string",
            "enum": [
              "critical",
              "major",
              "minor"
            ]
          },
          "target": {
            "type": "string",
            "description": "existing chapter id to revise, or 'NEW:<proposed-part>/<proposed-slug>' for missing content"
          },
          "missing": {
            "type": "string",
            "description": "precisely what is absent or wrong"
          },
          "fix": {
            "type": "string",
            "description": "concrete, actionable change to make"
          },
          "kind": {
            "type": "string",
            "enum": [
              "explanation",
              "code",
              "math",
              "verification",
              "resource-accounting",
              "library-mapping",
              "figure",
              "exercise",
              "correctness-bug"
            ]
          }
        }
      }
    },
    "summary": {
      "type": "string"
    }
  }
}

// Reasoning-heavy work runs on Opus 4.8 (per user 2026-07-20: Fable was
// consuming too much quota). Fable is reserved for sparing one-off strategy
// tasks only, never fan-out.
async function reason(prompt, opts = {}) {
  return await agent(prompt, { ...opts, model: 'opus' })
}

phase('Audit')
const results = await parallel(UNITS.map(u => () =>
  reason(u.prompt, { label: `audit:${u.id}`, phase: 'Audit', schema: SCHEMA, effort: 'high' })
))

const ok = results.filter(Boolean)
const failed = UNITS.filter((u, i) => !results[i]).map(u => u.id)
const byGrade = {}
for (const r of ok) byGrade[r.grade] = (byGrade[r.grade] || 0) + 1
const notBuildable = ok.filter(r => !r.buildable_verdict).map(r => r.unit_id)
const deficits = ok.flatMap(r => (r.deficits || []).map(d => ({ ...d, unit: r.unit_id })))

log(`audited ${ok.length}/${UNITS.length} | grades ${JSON.stringify(byGrade)} | ${deficits.length} deficits`)
if (failed.length) log(`FAILED (re-run to resume): ${failed.join(', ')}`)

return {
  audited: ok.length,
  failed,
  gradeHistogram: byGrade,
  notBuildable,
  deficitCount: deficits.length,
  criticalDeficits: deficits.filter(d => d.severity === 'critical').length,
}
