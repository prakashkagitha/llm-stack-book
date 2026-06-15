export const meta = {
  name: 'showcase-sota-boxes',
  description: 'Add web-verified "State of the Art & Resources" boxes to 3 showcase chapters',
  phases: [{ title: 'Research & insert' }],
}

const CH = [
  {
    file: '/local-ssd/pk669/programming/llm-stack-textbook/content/02-transformer/03-attention-from-scratch.md',
    topic: 'the attention mechanism in transformers',
    anchors: 'Vaswani et al. "Attention Is All You Need" (2017, arXiv:1706.03762); FlashAttention (Dao et al.); ' +
             'efficient/linear attention lineage; the Dao-AILab/flash-attention GitHub repo; ' +
             'recent (2024-2026) attention work and good explainer blogs (e.g. Lilian Weng, Jay Alammar).',
  },
  {
    file: '/local-ssd/pk669/programming/llm-stack-textbook/content/04-kernels-efficiency/02-flash-attention-1.md',
    topic: 'FlashAttention and IO-aware exact attention kernels',
    anchors: 'FlashAttention (Dao et al. 2022), FlashAttention-2 (Dao 2023), FlashAttention-3 (Shah et al. 2024, Hopper/FP8); ' +
             'Dao-AILab/flash-attention repo; Triton and CUTLASS/cuDNN fused attention; ' +
             'recent kernel work (2024-2026). Find the real arXiv IDs and the official repo.',
  },
  {
    file: '/local-ssd/pk669/programming/llm-stack-textbook/content/07-inference-serving/01-anatomy-inference.md',
    topic: 'LLM inference anatomy, the KV cache, prefill/decode',
    anchors: 'vLLM + PagedAttention (Kwon et al. 2023, arXiv:2309.06180, vllm-project/vllm); ' +
             'SGLang + RadixAttention (sgl-project/sglang); disaggregated prefill/decode (e.g. DistServe, Splitwise); ' +
             'KV-cache quantization and prefix caching; recent 2024-2026 serving work and official docs/blogs.',
  },
]

const PROMPT = (c) => `You are enhancing one chapter of a public, award-quality web textbook on the LLM stack by adding a single, polished "State of the Art & Resources" box with REAL, CURRENT, VERIFIED external links.

Chapter file (read it first): ${c.file}
Chapter topic: ${c.topic}

STEP 1 — Research the current (2024–2026) state of the art for this topic using WebSearch and WebFetch. Starting anchors (verify, don't trust blindly): ${c.anchors}

STEP 2 — VERIFY every link before using it. For each candidate URL, use WebFetch to confirm it resolves to the right content (HTTP 200, correct paper/repo/post). Prefer canonical, durable URLs: arXiv abstract pages (https://arxiv.org/abs/XXXX.XXXXX), official GitHub repos (https://github.com/org/repo), and official docs/blog posts. DO NOT invent arXiv IDs, repo paths, dates, or benchmark numbers. If you cannot verify a link, DROP it. It is far better to include 6 verified links than 12 shaky ones.

STEP 3 — Insert ONE admonition into the chapter, IMMEDIATELY BEFORE the final "## Further reading" (or "## Further Reading") section heading. Use the Edit tool. The block MUST follow this exact format (admonition marker, then EVERY content line indented exactly 4 spaces, blank line before and after):

!!! sota "State of the Art & Resources (2026)"
    A 1–2 sentence orientation on where this topic stands today.

    **Foundational papers**

    - [Author et al., *Title* (year)](https://arxiv.org/abs/XXXX.XXXXX) — one-line why-it-matters.

    **Pushing the frontier (2024–2026)**

    - [Author et al., *Title* (year)](URL) — one line.

    **Open-source & tools**

    - [org/repo](https://github.com/org/repo) — what it is.

    **Go deeper**

    - [Post/doc title](URL) — one line.

Rules:
- Keep it to roughly 8–12 verified links total, grouped under those bold sub-labels (drop a group if you have nothing solid for it).
- Each bullet: a real linked title + a short, accurate, non-hype description. No fabricated metrics.
- Change NOTHING else in the file. Do not touch the existing "Further reading" list, the body, or the figure marker.
- Markdown only; the 4-space indent on every line of the admonition is render-critical.

STEP 4 — Reply with ONLY one short line: the chapter file and how many links you verified and inserted.`

phase('Research & insert')
const results = await parallel(CH.map(function (c) {
  return function () {
    return agent(PROMPT(c), { label: 'sota:' + c.file.split('/').slice(-1)[0], phase: 'Research & insert', model: 'sonnet' })
      .then(function (r) { return { file: c.file, ok: true, note: r }; })
      .catch(function (e) { return { file: c.file, ok: false, note: String(e) }; });
  };
}));
log('SOTA boxes: ' + results.filter(function (r) { return r.ok; }).length + '/' + CH.length + ' chapters updated.');
return { results: results };
