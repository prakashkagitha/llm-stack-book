#!/usr/bin/env python3
"""Emit a Workflow that adds a GUIDED STEP-THROUGH "lesson mode" to flagship visualizer tools.

Per tool (Opus-5 augment in place -> Opus-5 verify): add a narrated Prev/Next walkthrough that
drives the tool's OWN existing controls and highlights the relevant part at each step, without
breaking any current behavior. The tool stays a free-explore widget; lesson mode is additive.

Usage: python3 scripts/gen_lessons_workflow.py [--ids self-attention-explorer ...]
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# slug, chapter, and a lesson outline (the pedagogical steps; the agent writes the driver).
SPECS = [
 {"slug":"self-attention-explorer","chapter":"02-transformer/03-attention-from-scratch",
  "steps":"1) Here is a sequence of tokens; each becomes a query, a key, and a value vector (point at Q/K/V). 2) A query scores every key by dot product, scaled by 1/sqrt(dk) — watch one query's row of the heatmap light up. 3) Softmax turns those scores into weights that sum to 1 (show the selected query's bar row). 4) The output is the weighted sum of value vectors (show the context bar). 5) Turn on the causal mask: a token can only attend to itself and earlier tokens (the upper triangle greys out). 6) Switch heads: different projections attend to different relationships. 7) Free explore: edit the tokens and watch the pattern change."},
 {"slug":"transformer-forward-pass","chapter":"02-transformer/07-build-gpt-from-scratch",
  "steps":"Walk one forward pass end to end, pausing on each stage and naming the tensor shape: token ids [B,T] -> embeddings [B,T,d] -> (inside a block) RMSNorm -> QKV -> attention (show the heatmap) -> residual add -> RMSNorm -> SwiGLU MLP -> residual add -> repeat over layers -> final norm -> lm_head [B,T,V] -> softmax -> next-token distribution. Emphasize the residual stream as the through-line and that shapes only change at embed and lm_head. End on the predicted next token; then free explore (change layers/seed)."},
 {"slug":"rlhf-ppo-pipeline","chapter":"05-posttraining-alignment/06-ppo-for-llms",
  "steps":"1) Four models in play: policy, reference, reward, value — point them out. 2) The policy generates a response to a prompt. 3) The reward model scores the full response (a scalar). 4) GAE turns that into per-token advantages using the value baseline. 5) The PPO clipped objective updates the policy only within the trust region — show the ratio and the clip band, and which tokens are clipped. 6) A KL penalty to the reference keeps the policy from drifting — move the beta slider and watch reward vs KL trade off. 7) Set beta near zero and step: reward-model score climbs while true reward falls — reward hacking. Then free explore."},
]

AUGMENT = r"""You are adding a GUIDED STEP-THROUGH "lesson mode" to an existing, already-verified interactive textbook visualizer, WITHOUT breaking any of its current behavior. The widget must remain a free-exploration tool; lesson mode is an ADDITIVE layer a reader can start and stop.

Tool file (READ IT FULLY FIRST — understand its controls, element ids, state, and render function): {TOOL_PATH}
Slug: {SLUG}   Chapter: content/{CHAPTER}.md

Add a lesson walkthrough with these pedagogical steps (you write the exact copy + the driver that realizes each step by setting the tool's OWN controls and, where helpful, highlighting the relevant sub-element):
{STEPS}

REQUIREMENTS:
  - Add a compact lesson bar INSIDE the tool's root div (near the top): a "Start guided tour" button; once started, a narration panel showing the current step's text, a "Step k / N" indicator, and Prev / Next / Exit buttons. Keyboard: Left/Right arrows move Prev/Next while the tour is active (do not hijack arrows when a text/number input is focused). Exit returns the widget to normal free-explore.
  - Each step's driver must set the tool's real controls (dispatch input/change events or call its existing update path) so the visualization actually reflects the narration — reuse the existing render/handlers, do NOT fork a parallel renderer. Where a step references a specific element (a heatmap row, a stage box, a bar), add a temporary highlight (an outline/glow class) that clears on step change.
  - PRESERVE everything: the existing controls still work; starting/exiting the tour leaves the widget fully functional; determinism (seeded) is kept. Do not remove or rename existing element ids/classes that the current script relies on — ADD new ones prefixed {SLUG}-lesson-.
  - Self-contained (no new external deps), theme-safe (CSS vars with hex fallbacks), scoped to #tool-{SLUG}, ASCII-only, no console errors, guarded against missing elements. Respect prefers-reduced-motion for the highlight.
  - Keep the tool's vt-note; optionally mention the tour in one short line.
Write the fully-updated widget back to EXACTLY {TOOL_PATH} (only the widget HTML). Do NOT touch the chapter markdown (the {{tool:{SLUG}}} marker already exists). Return a JSON summary (file, steps_added, controls_driven, notes)."""

VERIFY = r"""You are verifying that a GUIDED "lesson mode" was correctly added to an interactive visualizer without regressing it. Fix in place if needed.

Tool file: {TOOL_PATH}   (slug {SLUG})

Read the file and verify (fix and rewrite in place if any fails):
  1. NO REGRESSION: every pre-existing control still works and the free-explore widget is fully functional before starting and after exiting the tour. The lesson driver reuses the existing render/handlers (no forked parallel renderer that could drift). Determinism preserved.
  2. LESSON CORRECTNESS: Start shows step 1; Next/Prev/Exit and Left/Right arrows work (arrows must NOT fire while a text/number input is focused); the step counter is right; each step actually drives the tool's controls so the visual matches the narration; highlights clear on step change; the narration text is accurate to what the widget shows.
  3. SELF-CONTAINED (no external deps/network), THEME-SAFE (vars + hex fallbacks, readable light+dark), all new ids/classes scoped/prefixed to #tool-{SLUG} / {SLUG}-lesson- (no collisions on the concatenated /tools hub), ASCII-only, no console errors on load or on any tour action, guarded for missing elements, prefers-reduced-motion respected.
  4. JS VALIDITY: the inline <script> parses and runs error-free through a full tour (start -> Next to end -> Prev to start -> Exit) and normal control use.
Return the verdict (file, no_regression, lesson_works, self_contained, theme_safe, issues_fixed, notes)."""

ASCHEMA = {"type":"object","additionalProperties":False,"required":["file","steps_added","notes"],
           "properties":{"file":{"type":"string"},"steps_added":{"type":"integer"},
                         "controls_driven":{"type":"array","items":{"type":"string"}},"notes":{"type":"string"}}}
VSCHEMA = {"type":"object","additionalProperties":False,
           "required":["file","no_regression","lesson_works","self_contained","theme_safe","notes"],
           "properties":{"file":{"type":"string"},"no_regression":{"type":"boolean"},"lesson_works":{"type":"boolean"},
                         "self_contained":{"type":"boolean"},"theme_safe":{"type":"boolean"},
                         "issues_fixed":{"type":"integer"},"notes":{"type":"string"}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "wf_lessons.js"))
    ap.add_argument("--ids", nargs="*", default=[])
    a = ap.parse_args()
    specs = [s for s in SPECS if not a.ids or s["slug"] in a.ids]

    jobs = []
    for s in specs:
        tp = os.path.join(ROOT, "tools", s["slug"] + ".html")
        def fill(t):
            return (t.replace("{TOOL_PATH}", tp).replace("{SLUG}", s["slug"])
                    .replace("{CHAPTER}", s["chapter"]).replace("{STEPS}", s["steps"]))
        jobs.append({"slug": s["slug"], "augment": fill(AUGMENT), "verify": fill(VERIFY)})

    js = f"""export const meta = {{
  name: 'guided-lessons',
  description: 'Add guided step-through lesson mode to {len(jobs)} flagship tools (Opus-5 augment -> verify)',
  phases: [{{ title: 'Augment' }}, {{ title: 'Verify' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const ASCHEMA = {json.dumps(ASCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
log('Adding lesson mode to ' + JOBS.length + ' tools (Opus-5 augment -> verify)…');
const results = await pipeline(
  JOBS,
  function (j) {{
    return agent(j.augment, {{ label: 'augment:' + j.slug, phase: 'Augment', model: 'claude-opus-5', schema: ASCHEMA }})
      .then(function (r) {{ return j; }});
  }},
  async function (j) {{
    if (!j) return null;
    const v = await agent(j.verify, {{ label: 'verify:' + j.slug, phase: 'Verify', model: 'claude-opus-5', schema: VSCHEMA }});
    return {{ slug: j.slug, verify: v }};
  }}
);
const done = results.filter(Boolean);
log('Lessons: ' + done.length + '/' + JOBS.length + ' done.');
return {{ total: JOBS.length, done: done.length, results: results }};
"""
    open(a.out, "w").write(js)
    print(f"Wrote {a.out}: {len(jobs)} lesson augmentations")
    for s in specs:
        print(f"  {s['slug']:28} -> {s['chapter']}")


if __name__ == "__main__":
    main()
