export const meta = {
  name: 'gap-analysis',
  description: 'Audit the LLM-stack book for missing topics across all domains',
  phases: [{ title: 'Audit domains' }],
}

const TOC = '/local-ssd/pk669/programming/llm-stack-textbook/TOC.md'

const DOMAINS = [
  { key: 'foundations-hardware', parts: 'Part I (Mathematical & Systems Foundations)',
    focus: 'math/ML foundations, GPU architecture, parallel/collectives, AND the broader hardware landscape (TPUs, AMD/ROCm, Trainium/Inferentia, interconnects, networking/RDMA, on-device/edge silicon).' },
  { key: 'architecture', parts: 'Part II (The Transformer Architecture)',
    focus: 'tokenization, attention, positional encodings, the block, MoE, SSMs/linear-attention, and any modern architectural directions (e.g. multi-token prediction, mixture-of-depths, hybrid attention, byte/tokenizer-free models, very-long-context architectures, diffusion-LMs / non-autoregressive text).' },
  { key: 'pretraining-data', parts: 'Part III (Pretraining at Scale)',
    focus: 'data sourcing/cleaning/dedup, objectives, scaling laws, distributed training (DP/TP/PP/EP/SP/context-parallel), precision, optimizers, schedules, stability, checkpointing, long-context. Consider gaps like synthetic data generation, data mixing/curriculum, continual/domain-adaptive pretraining, tokenizer training, data attribution/influence, compute-optimal vs inference-optimal.' },
  { key: 'kernels-efficiency', parts: 'Part IV (Kernels, Efficiency & Quantization)',
    focus: 'roofline, FlashAttention, Triton/CUDA, paged attention, quantization (PTQ/QAT/formats), compilers/fusion, memory-efficient training. Consider gaps like sparsity/pruning, KV-cache quantization/compression, profiling & performance debugging tooling (Nsight, torch profiler), CUTLASS/cuDNN, distributed inference kernels, ThunderKittens-style kernel DSLs.' },
  { key: 'posttraining-alignment', parts: 'Part V (Post-Training & Alignment)',
    focus: 'SFT, PEFT, RLHF/PPO/DPO/GRPO/RLVR, reasoning/test-time compute, constitutional/RLAIF, distillation, reward hacking. Consider gaps like preference data collection/annotation, process reward models & verifiers, self-play/self-improvement, tool-use & agentic fine-tuning, long-horizon/credit assignment, safety alignment & refusal training, model spec/values.' },
  { key: 'rl-infra', parts: 'Part VI (RL Infrastructure)',
    focus: 'RL system anatomy, rollout engines, TRL/veRL/OpenRLHF/prime-rl, colocated vs disaggregated, reward/verifiers/sandboxes, advantage/KL tricks, agentic/multi-turn RL, scaling. Consider gaps like environment/gym design for LLM RL, data/replay management, experiment tracking for RL, partial-rollout/streaming, RL eval & reproducibility.' },
  { key: 'inference-serving', parts: 'Part VII (Inference & Serving)',
    focus: 'prefill/decode/KV cache, continuous batching, vLLM/SGLang/TRT-LLM/TGI, speculative decoding, prefix caching, disaggregation, sampling, structured generation, multi-GPU, economics. Consider gaps like MoE serving/expert-parallel inference, LoRA/multi-adapter serving, long-context inference & context-extension at inference (YaRN/NTK), KV offloading/hierarchical cache, request prioritization/SLO scheduling, on-device/edge inference (llama.cpp/MLX), embedding/reranker serving.' },
  { key: 'agents-harness', parts: 'Part VIII (Agents & Harness Engineering)',
    focus: 'tool use, agentic loop, harness/coding agents, context engineering, memory, MCP, multi-agent, agent eval, prompt engineering. Consider gaps like computer-use/GUI & browser agents, agent safety/sandboxing/permissions, planning & world-models, cost/latency control in agent loops, agent observability/tracing, deep-research/long-horizon agents.' },
  { key: 'rag-multimodal', parts: 'Part IX (Retrieval & RAG) and Part X (Multimodal & Generative Frontiers)',
    focus: 'embeddings, vector DBs/ANN, RAG architectures, chunking/reranking/hybrid, advanced RAG; vision transformers, VLMs, audio/speech, diffusion, any-to-any. Consider gaps like multimodal RAG, document AI/OCR/PDF parsing, video understanding & generation, 3D/embodied & vision-language-action (robotics) models, image/video generation systems (production), speech (TTS/ASR) pipelines, multimodal tokenization.' },
  { key: 'eval-production-safety', parts: 'Part XI (Evaluation) and Part XII (Production, Systems & MLOps)',
    focus: 'eval landscape, LLM-as-judge, harnesses, reasoning/coding/agentic evals, red-teaming; serving system design, observability, caching/routing/cost, guardrails, data flywheel, security. Consider gaps like benchmark contamination/leakage, statistical rigor in eval (confidence intervals, item-response), human eval ops, A/B testing & online eval, cost/FinOps, model/prompt versioning & registries, incident response, compliance/regulation (EU AI Act), reliability/SRE for LLMs.' },
  { key: 'cross-cutting-missing', parts: 'ENTIRE BOOK (look for whole AREAS that are absent)',
    focus: 'Identify major areas of the modern LLM stack that have NO real home in the current TOC at all. Strong candidates to evaluate: (1) Mechanistic interpretability & model internals (circuits, SAEs, probing, activation steering); (2) Knowledge/model editing & machine unlearning (ROME/MEMIT, unlearning, RLHF-of-facts); (3) Privacy, memorization, membership inference, differential privacy, PII; (4) Watermarking, provenance & AI-content detection; (5) AI safety/alignment theory (scalable oversight, deception, evals for dangerous capabilities); (6) Data-centric AI / data engineering as a discipline; (7) Economics of training & the compute/market landscape; (8) Open-weights ecosystem, licensing, model lifecycle/governance. Propose chapters (and possibly a NEW PART) where warranted.' },
]

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    domain: { type: 'string' },
    proposals: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          title: { type: 'string', description: 'proposed chapter title' },
          target_part: { type: 'string', description: 'which existing part it belongs in, or "NEW PART: <name>"' },
          scope: { type: 'string', description: '2-3 sentence scope of what the chapter would cover' },
          why_important: { type: 'string', description: 'why a top-notch 2026 book must include it' },
          currently: { type: 'string', enum: ['absent', 'partial'], description: 'absent = no coverage anywhere; partial = touched but deserves its own chapter' },
          priority: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['title', 'target_part', 'scope', 'why_important', 'currently', 'priority'],
      },
    },
  },
  required: ['domain', 'proposals'],
}

phase('Audit domains')
log('Auditing ' + DOMAINS.length + ' domains for missing topics…')
const results = await parallel(DOMAINS.map(function (d) {
  return function () {
    const prompt =
      'You are auditing a definitive, award-quality 2026 textbook on the ENTIRE Large Language Model stack for COMPLETENESS. ' +
      'Its goal is to cover "everything an ML/LLM engineer should know in their career," at the level of the best engineering writing, current to 2026.\n\n' +
      'STEP 1: Read the full current table of contents: ' + TOC + ' (read it in full so you do NOT propose things already covered).\n\n' +
      'STEP 2: Your audit domain: ' + d.parts + '.\nFocus & candidate gaps to weigh: ' + d.focus + '\n\n' +
      'You MAY use WebSearch to confirm what is genuinely important and current in 2026 for this domain.\n\n' +
      'STEP 3: Propose ONLY genuinely missing, high-value chapters — topics a working ML/LLM engineer or a strong interview candidate would expect, that the current TOC lacks or covers only shallowly. ' +
      'Cross-check the ENTIRE TOC before proposing (many topics are covered in a different part than you might expect — do not duplicate). ' +
      'Be selective and precise: propose 0–6 chapters, each clearly distinct from every existing chapter. For each, give a real title, the target part (or a NEW PART), a concrete 2-3 sentence scope, why it matters, whether it is currently absent or only partial, and a priority. ' +
      'It is fine to return an empty list if the domain is already well covered. Return the structured object.'
    return agent(prompt, { label: 'gap:' + d.key, phase: 'Audit domains', model: 'opus', schema: SCHEMA })
      .then(function (r) { return r; }).catch(function (e) { return { domain: d.key, proposals: [], error: String(e) }; })
  }
}))

const all = []
results.filter(Boolean).forEach(function (r) {
  (r.proposals || []).forEach(function (p) { all.push(Object.assign({ domain: r.domain }, p)) })
})
log('Collected ' + all.length + ' raw proposals across ' + results.length + ' domains.')
return { count: all.length, proposals: all, byDomain: results }
