#!/usr/bin/env python3
"""Generate self-check auto-graded exercise widgets (tools/ag-<slug>.html) from hand-verified
answers, and insert their {{tool:ag-<slug>}} markers into the mapped chapters.

Deterministic (no LLM): the answers are authored + verified here, so the grader is trustworthy.
Each widget: a question with concrete numbers, a numeric input, a Check button (relative+absolute
tolerance), instant right/wrong, and a collapsible worked solution. Self-contained + theme-safe.
Run: python3 scripts/make_autograded.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# slug (without the ag- prefix), title, chapter, question html, answer, unit, rel tol, abs tol, working html.
EX = [
 {"slug":"chinchilla-tokens","title":"Self-check: Chinchilla-optimal tokens","chapter":"03-pretraining/04-scaling-laws",
  "q":"The Chinchilla compute-optimal recipe uses roughly <b>20 tokens per parameter</b>. About how many training tokens should a <b>1.4-billion-parameter</b> model see? <i>(answer in billions of tokens)</i>",
  "ans":28.0,"unit":"B tokens","rel":0.05,"abs":0.5,
  "work":"20 tokens/param &times; 1.4&times;10<sup>9</sup> params = 2.8&times;10<sup>10</sup> = <b>28 billion tokens</b>. (Modern small models deliberately over-train far past this &mdash; see the capstone.)"},
 {"slug":"flops-6nd","title":"Self-check: the 6ND training-FLOP rule","chapter":"03-pretraining/04-scaling-laws",
  "q":"Using <b>C = 6ND</b>, estimate the training FLOPs to train <b>N = 7&times;10<sup>9</sup></b> parameters on <b>D = 2&times;10<sup>12</sup></b> tokens. <i>(answer as the coefficient a, where C = a &times; 10<sup>22</sup>)</i>",
  "ans":8.4,"unit":"&times;10^22 FLOPs","rel":0.05,"abs":0.1,
  "work":"C = 6 &times; (7&times;10<sup>9</sup>) &times; (2&times;10<sup>12</sup>) = 6 &times; 14&times;10<sup>21</sup> = <b>8.4&times;10<sup>22</sup> FLOPs</b>."},
 {"slug":"kv-cache-gib","title":"Self-check: KV-cache size","chapter":"07-inference-serving/01-anatomy-inference",
  "q":"One sequence's KV cache = <b>2 &times; L &times; n_kv &times; head_dim &times; seq &times; bytes</b> (the leading 2 is K and V). For <b>L=32, n_kv=8, head_dim=128, seq=8192, bf16 (2 bytes)</b>, how big is it? <i>(answer in GiB, 1 GiB = 2<sup>30</sup> bytes)</i>",
  "ans":1.0,"unit":"GiB","rel":0.03,"abs":0.02,
  "work":"2 &times; 32 &times; 8 &times; 128 &times; 8192 &times; 2 bytes = 1,073,741,824 bytes = 2<sup>30</sup> = <b>exactly 1 GiB</b> &mdash; and that is for a <i>single</i> sequence, which is why long-context serving is KV-bound."},
 {"slug":"mlp-swiglu-params","title":"Self-check: SwiGLU MLP parameters","chapter":"02-transformer/06-transformer-block",
  "q":"A SwiGLU feed-forward block has <b>three</b> weight matrices (gate, up, down), each of size d &times; intermediate. How many parameters for <b>d = 4096, intermediate = 11008</b>? <i>(answer in millions)</i>",
  "ans":135.27,"unit":"M params","rel":0.02,"abs":1.0,
  "work":"3 &times; d &times; intermediate = 3 &times; 4096 &times; 11008 = 135,266,304 &asymp; <b>135M parameters</b> (the MLP is ~2/3 of a transformer block's params)."},
 {"slug":"embedding-params","title":"Self-check: embedding-table size","chapter":"02-transformer/02-embeddings-input",
  "q":"The token-embedding matrix is <b>vocab &times; d_model</b>. How many parameters for <b>vocab = 128000, d_model = 4096</b>? <i>(answer in millions)</i>",
  "ans":524.29,"unit":"M params","rel":0.01,"abs":1.0,
  "work":"128000 &times; 4096 = 524,288,000 &asymp; <b>524M parameters</b> &mdash; untied, the model pays for this table twice (input + output), which is why small models tie the embeddings."},
 {"slug":"lora-params","title":"Self-check: LoRA parameter count","chapter":"05-posttraining-alignment/03-peft-lora-qlora",
  "q":"LoRA adds A (r &times; d) and B (d &times; r) to a d &times; d layer, so it trains <b>2 r d</b> parameters. How many for <b>d = 4096, r = 16</b>? <i>(answer in thousands)</i>",
  "ans":131.07,"unit":"K params","rel":0.02,"abs":1.0,
  "work":"2 &times; 16 &times; 4096 = 131,072 &asymp; <b>131K parameters</b> &mdash; versus 16.8M for the full d&times;d matrix, a ~128&times; reduction in trainable weights."},
 {"slug":"perplexity-from-ce","title":"Self-check: perplexity from cross-entropy","chapter":"01-foundations/02-probability-information",
  "q":"A model reaches a cross-entropy loss of <b>2.0 nats/token</b> on held-out text. Its <b>perplexity = exp(cross-entropy)</b>. What is it? <i>(2 decimals)</i>",
  "ans":7.39,"unit":"perplexity","rel":0.0,"abs":0.05,
  "work":"PPL = e<sup>2.0</sup> = <b>7.39</b> &mdash; the model is about as uncertain as an even choice among ~7.4 tokens at each step. (If the loss were in bits, you would use 2<sup>loss</sup> instead.)"},
 {"slug":"mfu","title":"Self-check: Model FLOPs Utilization","chapter":"04-kernels-efficiency/01-roofline-performance",
  "q":"MFU = <b>6 &times; N &times; (tokens/s) / peak_FLOPs</b>. A <b>7&times;10<sup>9</sup></b>-param model trains at <b>3000 tokens/s</b> on an A100 (peak <b>312&times;10<sup>12</sup></b> bf16 FLOP/s). What MFU? <i>(answer as a percentage)</i>",
  "ans":40.4,"unit":"%","rel":0.0,"abs":1.5,
  "work":"6 &times; 7&times;10<sup>9</sup> &times; 3000 = 1.26&times;10<sup>14</sup> FLOP/s of useful work; &divide; 3.12&times;10<sup>14</sup> = <b>&asymp;40%</b> &mdash; a healthy single-node MFU."},
 {"slug":"ridge-point","title":"Self-check: the roofline ridge point","chapter":"04-kernels-efficiency/01-roofline-performance",
  "q":"A kernel becomes compute-bound above the <b>ridge point = peak_FLOPs / memory_bandwidth</b>. For an A100 (<b>312&times;10<sup>12</sup></b> bf16 FLOP/s, <b>2.0&times;10<sup>12</sup></b> B/s), what is it? <i>(answer in FLOP/byte)</i>",
  "ans":156.0,"unit":"FLOP/byte","rel":0.0,"abs":4.0,
  "work":"312&times;10<sup>12</sup> / 2.0&times;10<sup>12</sup> = <b>156 FLOP/byte</b>. A GEMM must reuse each loaded byte ~156&times; to saturate the tensor cores; decode attention reuses each KV byte ~once, so it is permanently memory-bound."},
 {"slug":"adam-optimizer-memory","title":"Self-check: Adam optimizer-state memory","chapter":"03-pretraining/09-optimizers",
  "q":"Mixed-precision Adam keeps an fp32 master copy + two fp32 moments (m, v) = <b>12 bytes/parameter</b> of optimizer state. For a <b>1.0-billion-parameter</b> model, how much? <i>(answer in GB, 1 GB = 10<sup>9</sup> bytes)</i>",
  "ans":12.0,"unit":"GB","rel":0.0,"abs":0.5,
  "work":"12 bytes &times; 10<sup>9</sup> params = 1.2&times;10<sup>10</sup> bytes = <b>12 GB</b> &mdash; on top of the weights and gradients, which is why optimizer state dominates training memory and motivates 8-bit optimizers / sharding."},
]

TMPL = """<div class="viz-tool ag-tool" id="tool-ag-@SLUG@">
  <div class="vt-title">@TITLE@</div>
  <div class="ag-q">@Q@</div>
  <div class="ag-row">
    <input id="ag-@SLUG@-in" class="ag-in" type="number" step="any" inputmode="decimal" placeholder="your answer" aria-label="your answer">
    <span class="ag-unit">@UNIT@</span>
    <button id="ag-@SLUG@-check" class="ag-btn" type="button">Check</button>
  </div>
  <div class="ag-res" id="ag-@SLUG@-res" role="status" aria-live="polite"></div>
  <details class="ag-work">
    <summary>Show working</summary>
    <div class="ag-work-body">@WORK@</div>
  </details>
  <div class="vt-note">Self-check: type a number and press <b>Check</b> (or Enter). Answers use a small tolerance, so round sensibly.</div>
  <style>
  #tool-ag-@SLUG@{border:1px solid var(--border-2,#e7e2dc);border-radius:10px;padding:1rem 1.1rem;margin:1.2rem 0;background:var(--surface-2,#faf8f5)}
  #tool-ag-@SLUG@ .vt-title{font-weight:700;margin-bottom:.4rem}
  #tool-ag-@SLUG@ .ag-q{color:var(--ink,#2b2b2b);line-height:1.55;margin-bottom:.7rem}
  #tool-ag-@SLUG@ .ag-row{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem}
  #tool-ag-@SLUG@ .ag-in{width:9rem;max-width:60vw;padding:.45rem .6rem;font:600 1rem var(--mono,monospace);
    border:1px solid var(--border,#d8d0c6);border-radius:7px;background:var(--surface,#fff);color:inherit}
  #tool-ag-@SLUG@ .ag-in:focus-visible{outline:2px solid var(--accent,#c0562f);outline-offset:1px}
  #tool-ag-@SLUG@ .ag-unit{color:var(--ink-soft,#6b6459);font:.9rem var(--mono,monospace)}
  #tool-ag-@SLUG@ .ag-btn{margin-left:auto;padding:.45rem .95rem;font:600 .95rem var(--sans,sans-serif);cursor:pointer;
    border:1px solid var(--accent,#c0562f);border-radius:7px;background:var(--accent,#c0562f);color:var(--surface,#fff)}
  #tool-ag-@SLUG@ .ag-btn:hover{filter:brightness(1.06)}
  #tool-ag-@SLUG@ .ag-res{min-height:1.3em;margin:.6rem 0 .2rem;font-weight:600}
  #tool-ag-@SLUG@ .ag-res.ag-ok{color:var(--good,#2f9e6e)}
  #tool-ag-@SLUG@ .ag-res.ag-no{color:var(--warn,#c0562f)}
  #tool-ag-@SLUG@ .ag-work{margin-top:.4rem;font-size:.94rem}
  #tool-ag-@SLUG@ .ag-work summary{cursor:pointer;color:var(--accent,#c0562f);font-weight:600}
  #tool-ag-@SLUG@ .ag-work-body{margin-top:.5rem;color:var(--ink-soft,#4a453d);line-height:1.6;
    border-left:3px solid var(--border-2,#e7e2dc);padding-left:.8rem}
  #tool-ag-@SLUG@ .vt-note{margin-top:.7rem;font-size:.85rem;color:var(--muted,#8a8378)}
  </style>
  <script>(function(){
    var ans=@ANS@, relTol=@REL@, absTol=@ABS@;
    var inp=document.getElementById('ag-@SLUG@-in'),
        btn=document.getElementById('ag-@SLUG@-check'),
        res=document.getElementById('ag-@SLUG@-res');
    if(!inp||!btn||!res)return;
    function check(){
      var v=parseFloat(inp.value);
      if(!isFinite(v)){res.textContent='Enter a number first.';res.className='ag-res';return;}
      var tol=Math.max(absTol, relTol*Math.abs(ans));
      if(Math.abs(v-ans)<=tol){res.textContent='\\u2713 Correct \\u2014 \\u2248 '+ans+' @UNIT@';res.className='ag-res ag-ok';}
      else{res.textContent='\\u2717 Not quite. Check your units, then open \\u201cShow working\\u201d.';res.className='ag-res ag-no';}
    }
    btn.addEventListener('click',check);
    inp.addEventListener('keydown',function(e){if(e.key==='Enter'){e.preventDefault();check();}});
  })();</script>
</div>"""


def render(e):
    return (TMPL.replace("@SLUG@", e["slug"]).replace("@TITLE@", e["title"])
            .replace("@Q@", e["q"]).replace("@UNIT@", e["unit"]).replace("@WORK@", e["work"])
            .replace("@ANS@", repr(float(e["ans"]))).replace("@REL@", repr(float(e["rel"])))
            .replace("@ABS@", repr(float(e["abs"]))))


def insert_marker(chapter, marker):
    p = os.path.join(ROOT, "content", chapter + ".md")
    txt = open(p).read()
    if marker in txt:
        return "already"
    block = "\n" + marker + "\n"
    for anchor in ("## Exercises", "## Further reading", "## Further Reading", "## Key Takeaways"):
        i = txt.find("\n" + anchor)
        if i != -1:
            txt = txt[:i] + "\n" + block + txt[i:]
            open(p, "w").write(txt)
            return "before " + anchor
    txt = txt.rstrip() + "\n" + block
    open(p, "w").write(txt)
    return "appended"


def main():
    made = 0
    for e in EX:
        ch = os.path.join(ROOT, "content", e["chapter"] + ".md")
        if not os.path.exists(ch):
            print(f"  SKIP (no chapter): {e['chapter']}"); continue
        slug = "ag-" + e["slug"]
        open(os.path.join(ROOT, "tools", slug + ".html"), "w").write(render(e))
        where = insert_marker(e["chapter"], "{{tool:%s}}" % slug)
        print(f"  {slug:28} -> {e['chapter']}  (marker: {where})")
        made += 1
    print(f"Generated {made} auto-graded exercise widgets.")


if __name__ == "__main__":
    main()
