#!/usr/bin/env python3
"""Emit a Workflow script that authors animated, theme-aware SVG figures for a curated
set of high-value chapters, each VISUALLY self-verified via headless rasterization.

Each agent: reads the chapter, authors figures/<name>.html to spec (mimicking the gold
example), inserts a {{fig:<name>}} marker at the right spot, then renders the figure to
PNG (light+dark) with scripts/preview_fig.py, VIEWS the images, and iterates until clean.

Idempotent: skips a figure whose figures/<name>.html already exists.

Usage: python3 scripts/gen_fig_workflow.py --out scripts/wf_figs.js [--limit N] [--only name ...]
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (chapter_id, figure-name, concept to draw, suggested section to place it in)
FIGS = [
    # Part I
    ("01-foundations/04-numerics-precision", "float-formats",
     "A bit-layout comparison of FP32, TF32, FP16, BF16 and FP8(E4M3): show each format as a horizontal "
     "row of labeled fields — 1 sign bit, then exponent bits, then mantissa bits — with the bit-widths "
     "marked, so the reader sees BF16 trades mantissa for FP32's exponent range. Align rows to compare widths.",
     "the section contrasting the floating-point formats"),
    ("01-foundations/08-gpu-architecture", "memory-hierarchy",
     "The GPU memory hierarchy as a stack of horizontal bars from registers -> SMEM/L1 -> L2 -> HBM, each "
     "annotated with rough capacity and bandwidth, visually showing the bandwidth 'cliff' from on-chip SRAM "
     "(tens of TB/s, tiny) to off-chip HBM (a few TB/s, tens of GB). Use illustrative orders of magnitude only.",
     "the memory hierarchy section"),
    ("01-foundations/09-parallel-collectives", "ring-allreduce",
     "Ring all-reduce across 4 GPUs arranged in a ring: show chunks passed around the ring in the "
     "reduce-scatter then all-gather phases, illustrating why cost is ~2(N-1)/N. Label the GPUs G0..G3 and "
     "arrows for the passes.",
     "the all-reduce / ring algorithm section"),
    # Part II
    ("02-transformer/01-tokenization", "bpe-merges",
     "Byte-Pair Encoding merges: start from characters of a word like 'l o w e s t', then show 2-3 successive "
     "merge steps (e.g. 'es'->'est') building larger subword tokens, as a small bottom-up sequence/tree.",
     "the BPE algorithm section"),
    ("02-transformer/04-mha-gqa-mla", "gqa-kv-sharing",
     "Multi-Head vs Grouped-Query vs Multi-Query attention: three small panels showing query heads (say 8) "
     "mapping to KV heads — MHA 8->8, GQA 8->2 groups, MQA 8->1 — to convey the KV-cache shrink. Arrows from "
     "Q heads to shared KV heads.",
     "the GQA / MQA section"),
    ("02-transformer/05-positional-encoding", "rope-rotation",
     "RoPE as rotation: a 2D plane showing a feature pair (x1,x2) as a vector, rotated by an angle that grows "
     "with token position m (theta*m), with 2-3 positions shown at increasing angles. Convey that relative "
     "position = angle difference.",
     "the RoPE section"),
    ("02-transformer/06-transformer-block", "transformer-block",
     "A pre-norm Transformer block: the residual stream flowing vertically with two sub-blocks branching off "
     "and adding back — LayerNorm -> Multi-Head Attention -> (+residual), then LayerNorm -> MLP -> (+residual). "
     "Boxes and residual add circles.",
     "the block structure / residual section"),
    ("02-transformer/09-mixture-of-experts", "moe-routing",
     "MoE routing: a token going into a gating/router that selects top-2 of N=6 expert FFNs; show only the "
     "selected experts active and their outputs weighted-summed. Convey sparse activation.",
     "the routing / top-k gating section"),
    ("02-transformer/11-ssm-and-alternatives", "ssm-scan",
     "A state-space-model recurrence vs attention: show the sequential scan h_t = A h_{t-1} + B x_t producing "
     "y_t, as a left-to-right chain of state boxes, contrasting O(N) recurrence with attention's all-pairs.",
     "the SSM recurrence / Mamba section"),
    # Part III
    ("03-pretraining/04-scaling-laws", "scaling-law",
     "A log-log plot of loss vs compute showing a straight power-law line trending down to an irreducible "
     "error floor (dashed asymptote). Axes labeled 'compute (FLOPs, log)' and 'loss (log)'. Illustrative, no "
     "specific numbers.",
     "the power-law section"),
    ("03-pretraining/05-distributed-data-parallel", "data-parallel",
     "Data parallelism: N GPU replicas each holding the full model, processing different data shards, then an "
     "all-reduce synchronizing gradients before the optimizer step. Show replicas + the all-reduce.",
     "the DDP mechanism section"),
    ("03-pretraining/06-distributed-model-parallel", "pipeline-bubble",
     "Pipeline parallelism schedule: stages (S0..S3) on the y-axis, time on the x-axis, microbatches flowing "
     "through as a diagonal of colored cells (forward then backward), with the idle 'bubble' regions shaded. "
     "Convey why microbatching shrinks the bubble.",
     "the pipeline parallelism / bubble section"),
    ("03-pretraining/07-megatron-deepspeed", "zero-stages",
     "DeepSpeed ZeRO partitioning: across 4 GPUs, show stage 1 (optimizer states sharded), stage 2 (+gradients "
     "sharded), stage 3 (+parameters sharded) as three rows where colored segments split across GPUs, "
     "shrinking per-GPU memory.",
     "the ZeRO stages section"),
    ("03-pretraining/08-mixed-precision-fp8", "loss-scaling",
     "Loss scaling for FP16: a number line of representable magnitudes; small gradients fall into the FP16 "
     "underflow/denormal zone; multiplying the loss by a scale S shifts the gradient distribution up into the "
     "representable range (then unscaled before the step).",
     "the loss-scaling section"),
    ("03-pretraining/10-lr-schedules-hparams", "lr-schedule",
     "A learning-rate schedule curve over training steps: linear warmup up to a peak, then cosine decay down "
     "to a small floor. Axes: step vs learning rate. Mark the warmup and decay regions.",
     "the warmup/cosine schedule section"),
    # Part IV
    ("04-kernels-efficiency/01-roofline-performance", "roofline",
     "A roofline plot: x-axis arithmetic intensity (FLOP/byte, log), y-axis attainable FLOP/s (log); a "
     "diagonal memory-bandwidth roof rising to a flat compute-peak ceiling; mark a memory-bound point (e.g. "
     "attention/decode) on the slope and a compute-bound point (big GEMM) on the ceiling.",
     "the roofline model section"),
    ("04-kernels-efficiency/06-paged-attention-kv", "paged-kv",
     "PagedAttention like OS paging: a request's logical, contiguous KV sequence mapped via a block table to "
     "non-contiguous physical KV blocks in GPU memory, eliminating fragmentation. Show logical blocks -> block "
     "table -> scattered physical blocks.",
     "the paged KV / block table section"),
    ("04-kernels-efficiency/07-quantization-ptq", "quant-mapping",
     "Quantization mapping: a continuous FP weight range mapped onto a small set of evenly spaced INT8 levels "
     "via a scale and zero-point, with the rounding shown. Convey scale = range / 255 and dequantization.",
     "the affine quantization / scale-zero-point section"),
    # Part V
    ("05-posttraining-alignment/03-peft-lora-qlora", "lora",
     "LoRA: a frozen weight matrix W (large, locked) with a parallel trainable low-rank path B*A (rank r << d) "
     "added to its output: h = Wx + B(Ax). Show dimensions d x d for W and d x r, r x d for B, A to convey the "
     "tiny parameter count.",
     "the low-rank decomposition section"),
    ("05-posttraining-alignment/05-rlhf-reward-modeling", "rlhf-pipeline",
     "The three-stage RLHF pipeline as a left-to-right flow: (1) SFT on demonstrations, (2) train a Reward "
     "Model on human preference comparisons, (3) optimize the policy with PPO against the RM (with a KL "
     "penalty to the SFT model). Boxes + arrows.",
     "the three-stage overview section"),
    ("05-posttraining-alignment/06-ppo-for-llms", "ppo-clip",
     "The PPO clipped surrogate objective: plot the objective vs the probability ratio r, showing the clip "
     "region [1-eps, 1+eps] flattening the curve for positive vs negative advantage, so updates can't move too "
     "far. Two curves (A>0, A<0).",
     "the clipped objective section"),
    ("05-posttraining-alignment/07-dpo-and-variants", "dpo-vs-ppo",
     "DPO vs PPO: two contrasting pipelines — PPO needs a reward model + online rollouts + value model in a "
     "loop; DPO directly optimizes the policy on preference pairs (chosen vs rejected) with a simple "
     "classification-style loss, no RM, no sampling loop.",
     "the comparison / 'DPO removes the RL loop' section"),
    ("05-posttraining-alignment/08-grpo-rloo", "grpo-advantage",
     "GRPO advantage: for one prompt, sample a GROUP of G responses, score each with the reward, then compute "
     "each response's advantage as its reward minus the group mean (normalized) — no value/critic network. "
     "Show the group, rewards, and the baseline (mean).",
     "the group-relative advantage section"),
    # Part VI
    ("06-rl-infra/01-anatomy-rl-system", "rl-loop",
     "The LLM-RL system loop: a cycle of Generation (an inference engine produces rollouts) -> Reward/Verifier "
     "scores them -> Training (policy gradient update) -> Weight Sync back to the inference engine. Emphasize "
     "the inference+training co-location that makes RL infra hard.",
     "the system overview / loop section"),
    ("06-rl-infra/07-colocated-vs-disaggregated", "colocated-disagg",
     "Two placement layouts on a GPU cluster: COLOCATED (actor/generation and learner/training share the same "
     "GPUs, time-sliced) vs DISAGGREGATED (separate GPU pools for generation and training, connected by a "
     "weight-sync + data path). Side-by-side panels.",
     "the colocated vs disaggregated section"),
    # Part VII
    ("07-inference-serving/02-continuous-batching", "continuous-batching",
     "Static vs continuous batching: two timelines (iteration on x-axis, batch slots on y-axis). Static: slots "
     "idle until the whole batch finishes. Continuous: finished sequences are evicted and new requests join "
     "mid-flight, keeping slots full. Show the utilization difference.",
     "the continuous batching section"),
    ("07-inference-serving/04-sglang-radixattention", "radix-tree",
     "RadixAttention prefix sharing: a radix tree of cached KV where multiple requests sharing a common prompt "
     "prefix share the same tree path/KV, branching only where they diverge. Show 2-3 requests sharing a "
     "prefix.",
     "the radix tree / prefix cache section"),
    ("07-inference-serving/06-speculative-decoding", "spec-decoding",
     "Speculative decoding: a small draft model proposes k tokens cheaply; the target model verifies all k in "
     "ONE parallel forward pass; a prefix of accepted tokens is kept and the first rejection is corrected. "
     "Show draft tokens, parallel verify, accept/reject.",
     "the draft-and-verify section"),
    # Part VIII
    ("08-agents-harness/02-agentic-loop", "react-loop",
     "The ReAct agent loop as a cycle: Thought -> Action (tool call) -> Observation (tool result) -> back to "
     "Thought, repeating until a Final Answer. Show the LLM in the center with the tool/environment on the "
     "side.",
     "the ReAct / agent loop section"),
    ("08-agents-harness/06-mcp", "mcp-arch",
     "Model Context Protocol architecture: a Host app with an MCP Client connecting over a standard protocol to "
     "multiple MCP Servers, each exposing tools/resources/prompts (e.g. files, GitHub, a database). Convey the "
     "'USB-C for tools' standardization.",
     "the MCP architecture section"),
    # Part IX
    ("09-rag-retrieval/02-vector-databases-ann", "hnsw",
     "HNSW graph search: a multi-layer navigable small-world graph — sparse top layer for big jumps, denser "
     "lower layers for refinement — with a greedy search path descending from an entry point to the nearest "
     "neighbor of a query. Show 2-3 layers.",
     "the HNSW section"),
    ("09-rag-retrieval/03-rag-architectures", "rag-pipeline",
     "The RAG pipeline as a flow: user query -> embed -> retrieve top-k chunks from a vector store -> augment "
     "the prompt with retrieved context -> LLM generates a grounded answer (with citations). Boxes + arrows.",
     "the RAG pipeline overview section"),
    # Part X
    ("10-multimodal-and-arch/01-vision-transformers", "vit-patchify",
     "Vision Transformer input: an image split into a grid of fixed patches, each flattened and linearly "
     "projected into a patch embedding, plus a [CLS] token and position embeddings, fed as a sequence to a "
     "Transformer. Show image->patches->tokens.",
     "the patch embedding section"),
    ("10-multimodal-and-arch/04-diffusion-generative", "diffusion",
     "Diffusion: a forward process that gradually adds Gaussian noise to an image over T steps (clean -> pure "
     "noise), and a learned reverse process that denoises step by step back to a sample. Show the two "
     "directions as a chain of increasingly/decreasingly noisy thumbnails (abstract).",
     "the forward/reverse process section"),
]


SPEC = r"""FIGURE AUTHORING SPEC (follow exactly — these rules make it correct, beautiful, and theme-safe):

OUTPUT FILE: write a single self-contained snippet to figures/{NAME}.html — a `<figure class="viz" id="fig-{NAME}">` containing:
  - optionally `<button class="viz-replay" type="button">↻ replay</button>`
  - exactly one inline `<svg viewBox="0 0 W H" role="img" aria-label="...">` (choose W ~ 660–820; H to fit)
  - a `<figcaption>` with a <b>bold lead sentence</b> + 1–2 sentences on the takeaway.
  No <script>, no external images. A scoped `<style>` (selectors prefixed with #fig-{NAME}) for font sizes is fine.

LAYOUT (most important):
  - EVERYTHING must stay INSIDE the viewBox with ~10px padding. Text must NOT run past the right/bottom edges (the figure clips). For start-anchored text near the right, keep x small enough; prefer text-anchor="middle" with safe centers.
  - No illegible overlaps between boxes/labels. Leave clear gaps between adjacent groups (>= ~16px).
  - Use a clear left-to-right or top-to-bottom reading order.

COLOR (must work in BOTH light & dark themes — NEVER hardcode a primary color):
  - Prefer these CSS classes: v-fill (panel fill), v-accent (accent fill), v-accent-s (accent stroke),
    v-stroke (neutral stroke), v-grid (faint gridline stroke), v-muted (muted text fill), v-label (text fill), v-mono (monospace).
  - Or inline with a CSS var AND hex fallback: fill="var(--accent)", stroke="var(--good,#2f9e6e)".
    Available vars: --accent, --accent-2, --ink-soft, --muted, --good(#2f9e6e), --warn(#e0a106), --accent2(#3b82f6), --surface, --surface-3, --border-2.
  - A raw hex (e.g. fill="#333") is FORBIDDEN as a primary color — it breaks one theme. Hex is allowed ONLY as the fallback inside var(...).
  - Default text color is inherited (readable in both themes); add class="v-label" or a var fill for emphasis.

FONTS: use e.g. font="..." via style: `font:600 13px var(--sans,sans-serif)`; for code/labels use `var(--mono,monospace)`.

ANIMATION (enhancement only — the figure MUST be fully clear as a STATIC image):
  - Stage reveals: add `data-anim="1"`..`"5"` to top-level groups (they fade in in order).
  - `class="viz-draw" style="--len:NNN"` draws a path/line; `class="viz-sweep"` sweeps a highlight; `class="viz-pulse"` pulses opacity.
  - Never rely on motion to convey meaning.

ACCESSIBILITY: role="img" + a one-sentence aria-label describing the figure.

CORRECTNESS: the figure must faithfully match THIS chapter's explanation. Use small illustrative integers/labels; invent NO benchmark numbers.

GOLD EXAMPLE (study its structure, conventions, and color usage; then design a DIFFERENT figure for your concept):
--------------------------------------------------------------------------------
{GOLD}
--------------------------------------------------------------------------------
"""


PROMPT = r"""You are authoring ONE original, award-quality animated SVG figure for a public web textbook ("The LLM Stack").

Target chapter (READ IT FIRST to get the concept and terminology right): {ABSPATH}
Figure name (file stem): {NAME}
Concept to visualize: {CONCEPT}
Suggested placement: in {SECTION} — put the marker right after that section's heading (or its first paragraph).

{SPEC}

DO THIS:
1. Read the chapter to ground the figure in its exact framing.
2. Author figures/{NAME}.html per the spec. Write it with the Write tool.
3. VISUALLY VERIFY: run `python3 scripts/preview_fig.py {NAME} --theme both` from the repo root
   (/local-ssd/pk669/programming/llm-stack-textbook), then READ both images:
   /tmp/figpreview/{NAME}.light.png and /tmp/figpreview/{NAME}.dark.png.
   Inspect for: anything clipped by the viewBox edges, illegible overlaps, low contrast in EITHER theme,
   misaligned labels, or anything that misrepresents the concept. If you find ANY issue, Edit
   figures/{NAME}.html and re-render. Iterate until BOTH images look clean and correct (up to 3 passes).
4. Insert the marker `{{fig:{NAME}}}` on its own line (blank line before & after) into the chapter at the
   suggested spot, using the Edit tool. Change NOTHING else in the chapter.

Reply with ONLY one short line: the figure name, its final viewBox, and how many verify passes you did."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--model", default="opus")
    args = ap.parse_args()

    gold = open(os.path.join(ROOT, "figures", "attention-flow.html")).read()
    spec = SPEC.replace("{GOLD}", gold)

    sel = []
    for cid, name, concept, section in FIGS:
        if args.only and name not in args.only:
            continue
        if os.path.exists(os.path.join(ROOT, "figures", name + ".html")):
            continue  # idempotent
        abspath = os.path.join(ROOT, "content", cid + ".md")
        if not os.path.exists(abspath):
            print(f"!! chapter missing, skipping: {cid}")
            continue
        sel.append((cid, name, concept, section, abspath))
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        print("No figures selected (all exist?).")
        return

    jobs = []
    for cid, name, concept, section, abspath in sel:
        p = (PROMPT.replace("{SPEC}", spec).replace("{ABSPATH}", abspath)
             .replace("{NAME}", name).replace("{CONCEPT}", concept).replace("{SECTION}", section))
        jobs.append({"label": name, "model": args.model, "prompt": p})
    jobs_json = json.dumps(jobs, ensure_ascii=True)

    js = f"""export const meta = {{
  name: 'author-figures',
  description: 'Author {len(sel)} animated, visually self-verified SVG figures',
  phases: [{{ title: 'Author & verify' }}],
}}
const JOBS = {jobs_json};
phase('Author & verify')
log('Authoring ' + JOBS.length + ' figures (each rasterized & visually self-checked)…');
const results = await parallel(JOBS.map(function (j) {{
  return function () {{
    return agent(j.prompt, {{ label: 'fig:' + j.label, phase: 'Author & verify', model: j.model }})
      .then(function (r) {{ return {{ name: j.label, ok: true, note: r }}; }})
      .catch(function (e) {{ return {{ name: j.label, ok: false, note: String(e) }}; }});
  }};
}}));
const ok = results.filter(function (r) {{ return r.ok; }}).length;
log('Figures: ' + ok + '/' + JOBS.length + ' authored.');
return {{ ok: ok, total: JOBS.length, results: results }};
"""
    with open(args.out, "w") as f:
        f.write(js)
    print(f"Wrote {args.out}: {len(sel)} figures (model={args.model}).")
    for cid, name, *_ in sel:
        print(f"  {name:22} -> {cid}")


if __name__ == "__main__":
    main()
