export const meta = {
  name: 'audit-reaudit',
  description: 'Audit The LLM Stack against CS336 + train-from-scratch buildability',
  phases: [{ title: 'Audit', detail: '8 benchmark units graded independently' }],
}

const UNITS = [
  {
    "id": "A1-basics",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    A1-basics\nkind:  assignment\ntitle: Assignment 1: Basics\n\nLearning objectives this unit demands:\n  (see deliverables)\n\nImplementable deliverable(s) a reader must be able to produce:\n  - Byte-level BPE tokenizer: training and encode/decode\n  - Transformer LM from scratch: embeddings, RoPE, RMSNorm, SwiGLU FFN, causal multi-head attention, weight tying\n  - Cross-entropy loss and AdamW optimizer implemented from scratch\n  - Cosine LR schedule with warmup, gradient clipping\n  - Training loop with checkpointing, and a decoding/sampling function\n  - Train a minimal LM to a target loss\n\nTHIS IS A CRITICAL UNIT: students must actually BUILD this. Grade harshly.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/A1-basics.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  },
  {
    "id": "A2-systems",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    A2-systems\nkind:  assignment\ntitle: Assignment 2: Systems\n\nLearning objectives this unit demands:\n  (see deliverables)\n\nImplementable deliverable(s) a reader must be able to produce:\n  - Benchmark and profile model layers (timing, nsys/torch profiler, memory)\n  - Triton FlashAttention-2 forward AND backward kernel\n  - Correctness testing of kernels against a PyTorch reference\n  - Distributed data-parallel training implementation with communication overlap\n  - Optimizer state sharding (ZeRO-style) implementation\n\nTHIS IS A CRITICAL UNIT: students must actually BUILD this. Grade harshly.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/A2-systems.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  },
  {
    "id": "A3-scaling",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    A3-scaling\nkind:  assignment\ntitle: Assignment 3: Scaling\n\nLearning objectives this unit demands:\n  (see deliverables)\n\nImplementable deliverable(s) a reader must be able to produce:\n  - Reason about each Transformer component's contribution to loss and cost\n  - Design a sweep under a constrained compute budget\n  - Fit IsoFLOP / parametric scaling laws to observed runs\n  - Project optimal model size and token count for a target budget\n\nTHIS IS A CRITICAL UNIT: students must actually BUILD this. Grade harshly.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/A3-scaling.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  },
  {
    "id": "A4-data",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    A4-data\nkind:  assignment\ntitle: Assignment 4: Data\n\nLearning objectives this unit demands:\n  (see deliverables)\n\nImplementable deliverable(s) a reader must be able to produce:\n  - Extract text from raw Common Crawl WARC/WET dumps\n  - Language ID, quality filtering, and harmful-content filtering\n  - Exact and fuzzy (MinHash-LSH) deduplication\n  - Measure the downstream effect of filtering choices on model quality\n\nTHIS IS A CRITICAL UNIT: students must actually BUILD this. Grade harshly.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/A4-data.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  },
  {
    "id": "A5-alignment-rl",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    A5-alignment-rl\nkind:  assignment\ntitle: Assignment 5: Alignment and Reasoning RL\n\nLearning objectives this unit demands:\n  (see deliverables)\n\nImplementable deliverable(s) a reader must be able to produce:\n  - Supervised finetuning for math reasoning with correct loss masking\n  - Verifiable reward function and answer parsing for math\n  - GRPO/policy-gradient training loop implementation\n  - Optional: DPO and safety alignment methods\n\nTHIS IS A CRITICAL UNIT: students must actually BUILD this. Grade harshly.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/A5-alignment-rl.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  },
  {
    "id": "C1-end-to-end-pretrain",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    C1-end-to-end-pretrain\nkind:  capability\ntitle: END-TO-END: train an LLM from scratch\n\nLearning objectives this unit demands:\n  - A single coherent path a reader can follow from raw text to a trained, sampled-from model\n  - Every stage connected: data -> tokenizer -> model -> optimizer -> training loop -> checkpoint -> eval -> inference\n  - Concrete configs and compute budgets at multiple scales (laptop / 1 GPU / 8 GPU / multi-node)\n  - Explicit statement of what to expect: loss curves, timings, failure signatures\n\nImplementable deliverable(s) a reader must be able to produce:\n  - A reader with GPUs can go from raw corpus to a working trained LM using only this book.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/C1-end-to-end-pretrain.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  },
  {
    "id": "C2-debug-and-verify",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    C2-debug-and-verify\nkind:  capability\ntitle: Verification, testing and debugging discipline\n\nLearning objectives this unit demands:\n  - How to unit-test each component (tokenizer round-trip, attention vs reference, grad-check)\n  - Expected numeric values and sanity checks at each stage\n  - Diagnosing loss spikes, NaNs, divergence, dead experts, silent shape bugs\n  - Profiling workflow to find the actual bottleneck\n\nImplementable deliverable(s) a reader must be able to produce:\n  - A reader can independently verify each component they build is correct.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/C2-debug-and-verify.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
  },
  {
    "id": "C3-open-source-stack",
    "prompt": "You are auditing a 795,000-word technical textbook, \"The LLM Stack: From Silicon to Agents\",\nagainst Stanford CS336 (Language Modeling from Scratch) and against a second standard:\nCAN A READER TRAIN AN LLM FROM SCRATCH USING ONLY THIS BOOK?\n\nBook root: /local-ssd/pk669/programming/llm-stack-textbook\nChapter sources: content/<part>/<chapter>.md   Table of contents: TOC.md   Manifest: book.json\n\n## The unit you are auditing\nid:    C3-open-source-stack\nkind:  capability\ntitle: Open-source library coverage across the stack\n\nLearning objectives this unit demands:\n  - For each layer, the real libraries: PyTorch, Triton, HF transformers/datasets/tokenizers, Megatron-LM, DeepSpeed, FSDP, TRL, veRL, vLLM, SGLang, llama.cpp, FlashAttention, xformers, lm-eval-harness\n  - How each library implements the concept just taught, and where to read its source\n  - When to use the library vs write it yourself\n\nImplementable deliverable(s) a reader must be able to produce:\n  - A reader knows which library to reach for and how it works internally.\n\n## Your job\n1. Read TOC.md to locate every chapter that plausibly covers this unit. Then ACTUALLY READ those\n   chapters in full (use Read/Grep). Do not grade from titles \u2014 grade from the prose and code.\n2. Grade the book on this unit using the rubric:\n0 ABSENT     - topic not present in the book\n1 MENTIONED  - named or gestured at; no working understanding conveyed\n2 EXPLAINED  - correctly explained, but a reader could NOT implement it from the text alone\n3 BUILDABLE  - a competent reader could implement a correct version from the book ALONE\n               (complete algorithm, exact tensor shapes, edge cases, hyperparameters)\n4 REFERENCE  - buildable AND includes verification (tests/expected values), failure modes,\n               resource accounting, and pointers into real library implementations\n3. Decide `buildable_verdict`: could a competent reader, with ONLY this book, produce a correct\n   working implementation of the deliverable? Be strict and adversarial. Missing tensor shapes,\n   hand-waved algorithms, pseudo-code where real code is needed, absent edge cases (causal masking,\n   numerical stability, padding, the backward pass), or absent hyperparameters all mean FALSE.\n4. List specific, actionable deficits. Each deficit must name a real target chapter id (or\n   'NEW:<part>/<slug>' if genuinely new content is required) and state a concrete fix.\n   Do NOT invent deficits to seem thorough \u2014 if the coverage is genuinely strong, say so and\n   return few or zero deficits. An honest grade of 4 is a valuable result.\n5. Also flag any CORRECTNESS BUGS you find (wrong formula, wrong shape, wrong claim) with\n   kind='correctness-bug' and severity='critical'.\n\n## Output\nWrite your finding as JSON to: /local-ssd/pk669/programming/llm-stack-textbook/audit/C3-open-source-stack.json\n(the file content must be exactly the same object you return)\nThen return the structured object.\n\nBe precise and evidence-based. Cite chapter ids you actually read."
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
