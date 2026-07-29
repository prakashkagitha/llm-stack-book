#!/usr/bin/env python3
"""Emit a Workflow that BUILDS interactive visualizer tools (Sonnet build -> Opus verify).

Each job writes a self-contained tools/<slug>.html widget and inserts a {{tool:<slug>}} marker
into its mapped chapter (all distinct chapters -> no concurrent .md edits). build.py's
expand_tools() inlines it and write_tools_hub() adds it to /tools automatically.

Usage: python3 scripts/gen_tools_workflow.py [--ids slug1 slug2 ...]
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# slug, title, chapter (for the {{tool:}} marker + "used in"), rebuild (overwrite, no new marker), brief.
SPECS = [
 {"slug":"tokenizer-playground","title":"Tokenizer playground: train a real byte-level BPE","chapter":"02-transformer/01-tokenization","rebuild":True,
  "brief":"REBUILD the existing tool into a REAL byte-level BPE, not a simplified whitespace splitter. In-browser: a textarea seeded with a sample corpus + a 'vocab size / num-merges' slider + a 'Train' button. On train: run actual byte-level BPE — start from the 256 byte tokens, iteratively find the most frequent adjacent pair and merge, recording the ordered merge list; show a live-updating merges table (rank, pair -> new token) and the growing vocab. Then an encode box: type text, byte-encode, apply the learned merges greedily by rank, and render the resulting tokens as colored chips with the token count and the compression ratio (bytes/token). A decode readout reconstructs the exact bytes (prove round-trip). Include a small step-through 'animate merges' that highlights each merge as it is applied to a sample word. Correctness is the point: real GPT-style byte-level BPE (bytes 0-255 base, merge by frequency, deterministic tie-break, greedy rank-ordered encode, exact decode). Keep it fast (cap corpus/merges so training is instant)."},
 {"slug":"self-attention-explorer","title":"Self-attention explorer: Q, K, V and the attention matrix","chapter":"02-transformer/03-attention-from-scratch",
  "brief":"Let the reader FEEL scaled dot-product attention. Controls: a short editable token sequence (default e.g. 'the cat sat on the mat', tokenized by word, up to ~10 tokens); a 'causal mask' toggle; a temperature/1-over-sqrt(dk) scale toggle; a head selector (2-3 preset heads with different fixed random Q/K projections so the pattern changes). Render the S x S attention-weight heatmap (rows=query, cols=key), each cell shaded by softmax weight with the numeric value on hover; grey out masked cells when causal. Below, for a selected query token, show its softmax distribution as a bar row and the resulting weighted-sum 'context' as a small bar vector. Use small fixed pseudo-random d=8 embeddings + fixed random W_q/W_k/W_v seeded deterministically so it is reproducible; compute real QK^T/sqrt(dk), mask, softmax, and V-weighted sum. This is the single most important transformer intuition — make the heatmap the star."},
 {"slug":"softmax-temperature","title":"Softmax & temperature: from logits to a distribution","chapter":"01-foundations/02-probability-information",
  "brief":"Show how a logit vector becomes a probability distribution and how temperature reshapes it. Controls: ~6 editable logit values (number inputs or draggable bars) and a temperature slider (0.1-3.0, log-ish). Render the input logits as bars and the softmax(logits/T) probabilities as a second set of bars that update live; display the Shannon entropy H = -sum p log p (in nats and bits) and the argmax/perplexity. Show the T->0 (one-hot) and T->inf (uniform) limits as annotations. Compute real softmax with the max-subtraction trick. Tie the readout to LLM sampling (low T = sharp/greedy, high T = diverse)."},
 {"slug":"gradient-descent-playground","title":"Gradient descent playground: optimizers on a loss surface","chapter":"01-foundations/03-calculus-optimization",
  "brief":"A 2D loss-surface sandbox. Draw contour lines of a convex quadratic bowl whose condition number kappa is set by a slider (an ill-conditioned valley when kappa is high). Controls: optimizer (SGD, SGD+momentum, Adam), learning-rate slider, momentum/beta slider, a 'start point' the user can drag, and Step / Run / Reset. Animate the optimizer trajectory as a path of dots on the contour plot, computing the REAL update rules each step from the analytic gradient of the quadratic. Show the loss-vs-step curve alongside. Demonstrate the key intuitions: too-high LR diverges (path explodes), momentum overshoots then settles, Adam is scale-invariant across the ill-conditioned axes, and high kappa slows plain SGD. Keep the math exact (gradient of 0.5*(a*x^2 + b*y^2))."},
 {"slug":"backprop-graph","title":"Backprop on a computation graph","chapter":"01-foundations/07-autodiff-pytorch",
  "brief":"An interactive computation graph for a tiny expression (e.g. L = (w*x + b - y)^2, or a 2-node MLP neuron with a sigmoid). Draw the DAG of nodes (inputs, ops, output). Let the user set the leaf values (w, x, b, y) with number inputs. Forward pass: show each node's value. Backward pass (a 'backprop' button that animates from the loss back to the leaves): show the local derivative on each edge and the accumulated gradient dL/dnode on each node, so the chain rule is visible edge by edge. End with dL/dw etc. matching a from-scratch analytic check. This makes autodiff concrete: values flow forward, gradients flow backward, chain rule = multiply along the path."},
 {"slug":"normalization-explorer","title":"LayerNorm vs RMSNorm, live","chapter":"02-transformer/06-transformer-block",
  "brief":"Show what normalization does to an activation vector. The user edits/drags a small vector (~8 dims) of activations shown as bars (some large, some negative). Toggle between LayerNorm (subtract mean, divide by std, then gain*x+bias) and RMSNorm (divide by RMS, then gain*x — no mean subtraction, no bias) with gain/bias sliders. Render the output bars live and print mean, variance/RMS before and after. Demonstrate: LayerNorm centers AND scales (output mean 0, var 1 before affine); RMSNorm only scales (cheaper, keeps direction, no re-centering). Compute the real statistics. Note the pre-norm residual context."},
 {"slug":"embedding-similarity","title":"Embeddings & cosine similarity in 2D","chapter":"02-transformer/02-embeddings-input",
  "brief":"A 2D plane with ~8 draggable word-vector points (labeled tokens). As the user drags, live-update a similarity readout: pick two tokens (or hover) and show their dot product, cosine similarity, and Euclidean distance. Draw the angle between two selected vectors. Optionally a 'king - man + woman ~ queen' analogy demo: draw the vector arithmetic as arrows and highlight the nearest point to the result. Compute real dot/cosine/norm. Teaches: embeddings are directions in space; cosine similarity = angle; analogies = vector arithmetic; magnitude vs direction."},
 {"slug":"kv-cache-growth","title":"KV cache growth during decoding","chapter":"07-inference-serving/01-anatomy-inference",
  "brief":"Animate the KV cache filling up token by token during autoregressive decode, and show why it dominates long-context memory. Controls: n_layers, n_kv_heads (with an MHA-vs-GQA toggle showing the KV-head reduction), head_dim, dtype (fp16/bf16=2B, fp8=1B), and a 'generate' button that advances the sequence length. Render a grid/bar that grows as tokens are added (2 * L * n_kv_heads * head_dim * seq_len * bytes) with a live GB readout and a comparison to model-weights size. Show GQA shrinking the bar by the head-group ratio. Contrast prefill (parallel, one big fill) vs decode (one column per step). Use the exact KV-cache formula; this complements the kv-cache-budgeter calculator with a visual/animated view."},
 {"slug":"moe-router","title":"Mixture-of-Experts routing & load balance","chapter":"02-transformer/09-mixture-of-experts",
  "brief":"Visualize top-k MoE routing. Show N tokens (dots) and E experts (columns). A router assigns each token a softmax over experts (from fixed seeded pseudo-random gate logits per token, re-rollable); route each token to its top-k experts (k slider, 1-2) and draw lines token->expert. Show per-expert load bars, the capacity factor / dropped-tokens when an expert overflows a capacity limit (slider), and the load-balancing (auxiliary) loss / coefficient-of-variation of the loads. A 'shuffle' button re-rolls the gates. Teaches: sparse activation (only k of E experts run per token), the load-imbalance problem, capacity + drop, and why an aux load-balancing loss is needed. Compute real softmax gating, top-k, loads, and the CV."},
 {"slug":"quantization-explorer","title":"Quantization explorer: fp32 -> int8 / int4 / NF4","chapter":"04-kernels-efficiency/07-quantization-ptq",
  "brief":"Show post-training quantization on a weight tensor. Generate a bell-shaped weight distribution (Gaussian, with an outlier slider). Choose a format: int8 (per-tensor absmax affine/symmetric), int4 (16 levels), and NF4 (the non-uniform normal-float levels). Draw the original values as a histogram with the quantization grid overlaid (the representable levels), and a second histogram of the quantization ERROR (dequantized - original). Show bytes/param (4->1->0.5), the RMSE, and the max abs error. Demonstrate: fewer bits = coarser grid = more error; outliers stretch per-tensor absmax scale and hurt everyone (motivating per-channel/group scaling and NF4's normal-optimal levels). Real quantize/dequantize math (scale = absmax/qmax, round, clamp, dequant)."},
 {"slug":"lr-schedule-explorer","title":"Learning-rate schedules: warmup, cosine, WSD","chapter":"03-pretraining/10-lr-schedules-hparams",
  "brief":"Plot the LR-vs-step curve for the schedules the book uses. Controls: total steps, warmup steps, peak LR, min-LR ratio, and schedule = {linear-warmup+cosine, WSD (warmup-stable-decay, with a decay-fraction slider), inverse-sqrt, constant}. Draw the curve on an axis-labeled plot that updates live; mark the warmup end and (for WSD) the decay start. Overlay two schedules to compare. Compute the exact formulas (cosine: min + 0.5*(peak-min)*(1+cos(pi*t)); WSD: linear/constant/sqrt-decay legs). Annotate why WSD's long stable phase + short decay pairs with mid-training annealing (cross-link the capstone)."},
 {"slug":"flash-attention-tiling","title":"FlashAttention tiling & online softmax","chapter":"04-kernels-efficiency/02-flash-attention-1",
  "brief":"Animate why FlashAttention never materializes the full S x S score matrix. Show the Q, K, V as row/column strips split into blocks (block-size slider). Step through the outer loop over K/V blocks: for the current block, highlight the tile of scores computed, and update the running per-row max m and running sum l and the running output accumulator O via the online-softmax rescale (show m_new = max(m, block_max), the correction factor exp(m_old - m_new), and O rescaled). A memory meter contrasts O(S^2) materialized scores (naive) vs O(block) SRAM tiles (flash). Keep the online-softmax recurrence exact on small toy numbers so the running stats visibly converge to the true softmax attention."},
 {"slug":"decoding-tree","title":"Decoding strategies: greedy, beam, sampling","chapter":"07-inference-serving/09-sampling-decoding",
  "brief":"An interactive next-token decoding tree over a tiny toy vocabulary with fixed seeded conditional probabilities. Choose a strategy: greedy (argmax path), beam search (beam-width slider, keep top-B cumulative-logprob sequences, show the surviving beams and pruned branches), temperature sampling (temp slider), and top-k / top-p / nucleus (k, p sliders showing which tokens are kept/cut). Draw the expanding tree for a few steps, labeling edges with probabilities and highlighting the chosen/kept paths and the pruned ones. Show the resulting sequence(s) and their total logprob. Teaches: greedy vs beam vs sampling, how beam width and top-k/top-p reshape the candidate set, the diversity/quality tradeoff. Real probability math (cumulative logprobs, top-k/top-p truncation + renormalize)."},
 {"slug":"speculative-decoding-viz","title":"Speculative decoding: draft, verify, accept","chapter":"07-inference-serving/06-speculative-decoding",
  "brief":"Animate the draft-then-verify loop. A small fast 'draft' model proposes gamma tokens (gamma slider); the big 'target' model verifies them in one parallel pass; accept the longest matching prefix and reject the rest, then sample one bonus token — animate the accept/reject per step. An 'acceptance rate' slider (or draft/target agreement) drives how many are accepted. Show the expected tokens-per-target-call = (1 - alpha^(gamma+1))/(1 - alpha) and the resulting wall-clock speedup vs plain decode given a draft/target cost ratio slider. Teaches: why speculative decoding is lossless (verification distribution is exact), and how speedup depends on acceptance rate and gamma. Real geometric-acceptance math."},
 {"slug":"precision-formats","title":"Floating-point formats: fp32, bf16, fp16, fp8, int8","chapter":"01-foundations/04-numerics-precision",
  "brief":"Show the bit layout and tradeoffs of the numeric formats used in training/inference. A format selector (fp32, tf32, bf16, fp16, fp8-e4m3, fp8-e5m2, int8) draws the sign/exponent/mantissa bit fields with their widths. Enter a real number and show: its nearest representable value in the chosen format, the rounding error, and where it sits relative to the format's dynamic range (max finite, smallest normal/subnormal) and machine epsilon. A comparison table of the selected vs fp32 (exponent bits, mantissa bits, dynamic range, relative precision). Demonstrate the key point: bf16 keeps fp32's exponent range (few overflow issues) but only ~3 decimal digits of precision; fp16 has more mantissa but narrow range (overflow); fp8 needs per-tensor scaling. Compute the actual decomposition and rounding for the entered value."},
 {"slug":"gqa-mla-explorer","title":"MHA vs MQA vs GQA vs MLA: KV heads & cache","chapter":"02-transformer/04-mha-gqa-mla",
  "brief":"Visualize how query heads share key/value heads across the attention variants. Controls: n_heads (query heads), and a variant selector MHA (n_kv=n_heads), MQA (n_kv=1), GQA (n_kv=g, group-size slider), MLA (low-rank latent, latent-dim slider). Draw the query heads as boxes and the KV heads as boxes with lines showing which queries share which KV head (grouping). Show the KV-cache size per token for each variant (2 * n_kv * head_dim * bytes, and for MLA the latent-dim compression) and the ratio vs MHA. Teaches: GQA interpolates MHA<->MQA trading quality for KV-cache; MLA compresses KV into a latent for an even smaller cache (DeepSeek). Compute exact KV-cache bytes and the compression ratios."},
]


BUILD = r"""You are building ONE interactive, self-contained visualizer widget for a definitive LLM-systems web textbook. It must teach a concept through direct manipulation and be CORRECT, beautiful, and dependency-free.

Tool: {TITLE}  (slug: {SLUG})
It teaches / does:
{BRIEF}

WRITE the widget to EXACTLY: {TOOL_PATH}
Structure (match the book's other tools):
  - A single root `<div class="viz-tool" id="tool-{SLUG}">` containing: a `<div class="vt-title">{TITLE}</div>`, the controls (labels + inputs/sliders/selects, grouped in a `<div class="vt-grid">`), the visualization area (inline SVG and/or DOM/canvas), a live readout, and a short `<div class="vt-note">` explaining the takeaway.
  - A scoped `<style>` (EVERY selector prefixed with `#tool-{SLUG}`) and an inline `<script>` (an IIFE; all element ids prefixed `{SLUG}-` to avoid collisions; no globals leaking).
HARD CONSTRAINTS:
  - COMPLETELY SELF-CONTAINED: no external scripts, stylesheets, fonts, images, CDNs, network/fetch/XHR. Pure vanilla JS + inline SVG/canvas. It must work offline pasted into a page.
  - CORRECTNESS IS THE POINT: implement the REAL math (softmax with max-subtraction, exact BPE merges, exact optimizer updates, exact KV-cache bytes, exact quantization scale/round/dequant, exact online-softmax recurrence, etc.). Deterministic where randomness is used (seed a small PRNG). No fabricated formulas.
  - THEME-SAFE (light + dark): NEVER hardcode a primary color. Use CSS vars with hex fallbacks: var(--accent,#4f46e5), var(--ink,#1a1a2e), var(--ink-soft,#556), var(--muted,#889), var(--surface,#fff), var(--surface-2,#f6f7f9), var(--surface-3,#eceef3), var(--border,#e5e7eb), var(--border-2,#e5e7eb), var(--good,#2f9e6e), var(--warn,#e0a106), var(--mono,monospace), var(--sans,sans-serif). Text must be readable in both themes.
  - RESPONSIVE: fluid width, SVG uses viewBox, wide content scrolls inside its own container; never overflow the page horizontally. Inputs are keyboard-accessible with `<label>`s; add aria-labels.
  - ASCII-only in code/markup (use ->, <=, x, theta written out, HTML entities like &times; &rarr; &sigma; are fine). No console errors; guard against empty/invalid input.
INTERACTIVITY: it must update live on input (attach event listeners; render once on load). Keep it snappy.
Then INSERT the marker `{{tool:{SLUG}}}` on its own line into content/{CHAPTER}.md at the most pedagogically relevant spot (right after the section that explains this concept; a blank line before and after). Do NOT remove or alter any existing content, code fence, figure marker, or exercise.{REBUILD_NOTE}
Return a short JSON summary (file, inserted_marker, controls, what_it_teaches)."""

VERIFY = r"""You are the Opus verifier for ONE interactive textbook visualizer. Make it correct, dependency-free, theme-safe, and genuinely instructive, then fix it in place.

Tool file: {TOOL_PATH}   (slug {SLUG})
Chapter with the marker: content/{CHAPTER}.md

Read the tool file (and the chapter section around the {{tool:{SLUG}}} marker). Verify HARD and FIX in place (rewrite {TOOL_PATH}):
  1. CORRECTNESS: re-derive the math the widget computes and check the code implements it exactly (softmax/BPE/optimizer/KV-cache/quantization/online-softmax/etc.). Fix any wrong formula, off-by-one, or misleading visualization. Verify determinism (seeded PRNG).
  2. SELF-CONTAINED: NO external scripts/styles/fonts/images/CDN/network. If any exist, inline or remove them. Pure vanilla JS + inline SVG/canvas only.
  3. JS VALIDITY: the inline <script> must be syntactically valid and error-free on load and on every control change; guard invalid/empty inputs. (Mentally execute the event handlers.)
  4. THEME-SAFETY: no hardcoded primary colors; CSS vars with hex fallbacks; readable in light AND dark. All selectors and element ids scoped/prefixed to #tool-{SLUG} / {SLUG}- (no collisions with other tools on the /tools hub page, which concatenates all of them).
  5. UX: labeled, keyboard-accessible controls; responsive (viewBox SVG, no horizontal page overflow); a clear vt-title and a vt-note stating the takeaway; live updates.
  6. MARKER: confirm exactly one `{{tool:{SLUG}}}` marker exists in content/{CHAPTER}.md at a sensible spot and that no other chapter content/code/figure/exercise was damaged. Fix if needed.
Return the verdict object (file, correct, self_contained, theme_safe, issues_fixed, notes)."""

BSCHEMA = {"type":"object","additionalProperties":False,"required":["file","what_it_teaches"],
           "properties":{"file":{"type":"string"},"inserted_marker":{"type":"boolean"},
                         "controls":{"type":"array","items":{"type":"string"}},"what_it_teaches":{"type":"string"}}}
VSCHEMA = {"type":"object","additionalProperties":False,"required":["file","correct","self_contained","theme_safe","notes"],
           "properties":{"file":{"type":"string"},"correct":{"type":"boolean"},"self_contained":{"type":"boolean"},
                         "theme_safe":{"type":"boolean"},"issues_fixed":{"type":"integer"},"notes":{"type":"string"}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "wf_tools.js"))
    ap.add_argument("--ids", nargs="*", default=[])
    a = ap.parse_args()
    specs = [s for s in SPECS if not a.ids or s["slug"] in a.ids]

    jobs = []
    for s in specs:
        tool_path = os.path.join(ROOT, "tools", s["slug"] + ".html")
        rebuild_note = ("\n\nNOTE: this tool ALREADY EXISTS and is already referenced by {{tool:%s}} markers in the "
                        "book — OVERWRITE %s with the improved version and do NOT insert any new marker (skip the marker step)."
                        % (s["slug"], tool_path)) if s.get("rebuild") else ""
        def fill(t):
            return (t.replace("{TITLE}", s["title"]).replace("{SLUG}", s["slug"])
                    .replace("{BRIEF}", s["brief"]).replace("{TOOL_PATH}", tool_path)
                    .replace("{CHAPTER}", s["chapter"]).replace("{REBUILD_NOTE}", rebuild_note))
        jobs.append({"slug": s["slug"], "build": fill(BUILD), "verify": fill(VERIFY)})

    js = f"""export const meta = {{
  name: 'interactive-tools',
  description: 'Build {len(jobs)} interactive visualizer tools (Sonnet build -> Opus verify)',
  phases: [{{ title: 'Build' }}, {{ title: 'Verify' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const BSCHEMA = {json.dumps(BSCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
log('Building ' + JOBS.length + ' interactive tools (build -> Opus verify)…');
const results = await pipeline(
  JOBS,
  function (j) {{
    return agent(j.build, {{ label: 'build:' + j.slug, phase: 'Build', model: 'sonnet', schema: BSCHEMA }})
      .then(function (r) {{ return j; }});
  }},
  async function (j) {{
    if (!j) return null;
    const v = await agent(j.verify, {{ label: 'verify:' + j.slug, phase: 'Verify', model: 'claude-opus-5', schema: VSCHEMA }});
    return {{ slug: j.slug, verify: v }};
  }}
);
const done = results.filter(Boolean);
log('Interactive tools: ' + done.length + '/' + JOBS.length + ' built+verified.');
return {{ total: JOBS.length, done: done.length, results: results }};
"""
    open(a.out, "w").write(js)
    print(f"Wrote {a.out}: {len(jobs)} tools")
    for s in specs:
        print(f"  {'[rebuild] ' if s.get('rebuild') else '[new]     '}{s['slug']:28} -> {s['chapter']}")


if __name__ == "__main__":
    main()
