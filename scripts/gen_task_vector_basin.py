#!/usr/bin/env python3
"""Generates figures/task-vector-arithmetic-basin.html"""
import math

W, H = 800, 660

def pt(x, y):
    return (round(x, 1), round(y, 1))

svg = []
svg.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="A weight-space schematic: task vectors tau_A and tau_B are arrows from a shared base checkpoint theta_base to fine-tuned points theta_A and theta_B inside a low-loss basin. Their vector sum lands on a merged point theta_merge that stays inside the same basin. A dashed arrow shows theta_base minus tau_A negating a behavior. A separate, unrelated basin from a different base checkpoint shows that merging task vectors computed against different base checkpoints does not compose the same way.">')

svg.append('''<defs>
  <marker id="tvb-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor"/>
  </marker>
  <marker id="tvb-arrow-muted" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
    <path d="M0,0 L7,3.5 L0,7 Z" fill="currentColor" style="color:var(--muted,#64748b)"/>
  </marker>
</defs>''')

svg.append('''<style>
  #fig-task-vector-arithmetic-basin text { fill: currentColor; }
  #fig-task-vector-arithmetic-basin .ax-lbl    { font: 500 10.5px var(--sans,sans-serif); fill: var(--muted,#64748b); }
  #fig-task-vector-arithmetic-basin .pt-lbl    { font: 700 12px var(--mono,monospace); fill: var(--ink-soft,#334155); }
  #fig-task-vector-arithmetic-basin .pt-sub    { font: 500 10.5px var(--sans,sans-serif); fill: var(--muted,#64748b); }
  #fig-task-vector-arithmetic-basin .vec-lbl-a { font: 700 12px var(--mono,monospace); fill: var(--accent,#c15b39); }
  #fig-task-vector-arithmetic-basin .vec-lbl-b { font: 700 12px var(--mono,monospace); fill: var(--accent2,#3b82f6); }
  #fig-task-vector-arithmetic-basin .vec-lbl-m { font: 700 12px var(--mono,monospace); fill: var(--good,#2f9e6e); }
  #fig-task-vector-arithmetic-basin .basin-lbl { font: 600 12px var(--sans,sans-serif); fill: var(--ink-soft,#334155); }
  #fig-task-vector-arithmetic-basin .basin-lbl-faded { font: 600 11px var(--sans,sans-serif); fill: var(--muted,#64748b); }
  #fig-task-vector-arithmetic-basin .cross-lbl { font: 600 11px var(--sans,sans-serif); fill: var(--muted,#64748b); }
  #fig-task-vector-arithmetic-basin .leg-title { font: 700 11.5px var(--sans,sans-serif); }
  #fig-task-vector-arithmetic-basin .leg-body  { font: 500 10.5px var(--sans,sans-serif); fill: var(--muted,#64748b); }
  #fig-task-vector-arithmetic-basin .basin-fill   { fill: var(--accent,#c15b39); fill-opacity: 0.08; }
  #fig-task-vector-arithmetic-basin .basin-fill-2 { fill: var(--accent,#c15b39); fill-opacity: 0.06; }
  #fig-task-vector-arithmetic-basin .basin-stroke { stroke: var(--border-2,#475569); }
  #fig-task-vector-arithmetic-basin .basin-contour{ stroke: var(--border-2,#475569); stroke-dasharray: 3 3; fill: none; opacity: 0.7; }
  #fig-task-vector-arithmetic-basin .basin-other-fill { fill: var(--muted,#64748b); fill-opacity: 0.10; }
  #fig-task-vector-arithmetic-basin .arrow-a   { stroke: var(--accent,#c15b39); color: var(--accent,#c15b39); stroke-width: 2.2; fill: none; marker-end: url(#tvb-arrow); }
  #fig-task-vector-arithmetic-basin .arrow-b   { stroke: var(--accent2,#3b82f6); color: var(--accent2,#3b82f6); stroke-width: 2.2; fill: none; marker-end: url(#tvb-arrow); }
  #fig-task-vector-arithmetic-basin .arrow-m   { stroke: var(--good,#2f9e6e); color: var(--good,#2f9e6e); stroke-width: 3; fill: none; marker-end: url(#tvb-arrow); }
  #fig-task-vector-arithmetic-basin .arrow-neg { stroke: var(--accent,#c15b39); color: var(--accent,#c15b39); stroke-width: 1.8; stroke-dasharray: 5 4; fill: none; marker-end: url(#tvb-arrow); opacity: 0.85; }
  #fig-task-vector-arithmetic-basin .guide     { stroke: var(--muted,#64748b); stroke-width: 1.1; stroke-dasharray: 3 3; fill: none; opacity: 0.6; }
  #fig-task-vector-arithmetic-basin .cross-line{ stroke: var(--muted,#64748b); stroke-width: 1.6; stroke-dasharray: 5 4; fill: none; }
  #fig-task-vector-arithmetic-basin .pt-dot    { fill: var(--surface,#fff); stroke: var(--ink-soft,#334155); stroke-width: 1.6; }
  #fig-task-vector-arithmetic-basin .pt-dot-m  { fill: var(--good,#2f9e6e); stroke: var(--good,#2f9e6e); stroke-width: 1.6; }
  #fig-task-vector-arithmetic-basin .axis-line { stroke: var(--muted,#64748b); stroke-width: 1.3; }
  #fig-task-vector-arithmetic-basin .divider   { stroke: var(--border-2,#475569); stroke-width: 1; stroke-dasharray: 2 4; opacity: 0.6; }
  #fig-task-vector-arithmetic-basin .swatch-line-a { stroke: var(--accent,#c15b39); color: var(--accent,#c15b39); stroke-width: 2.2; }
  #fig-task-vector-arithmetic-basin .swatch-line-m { stroke: var(--good,#2f9e6e); color: var(--good,#2f9e6e); stroke-width: 3; }
  #fig-task-vector-arithmetic-basin .swatch-line-neg { stroke: var(--accent,#c15b39); color: var(--accent,#c15b39); stroke-width: 1.8; stroke-dasharray: 5 4; }
  #fig-task-vector-arithmetic-basin .swatch-line-x { stroke: var(--muted,#64748b); stroke-width: 1.6; stroke-dasharray: 5 4; }
</style>''')

# ---------------- main basin ----------------
base = (300, 300)
A    = (200, 240)
B    = (355, 205)
merge= (255, 145)
neg  = (400, 360)
brx, bry = 210, 190

svg.append('<g data-anim="1">')
svg.append(f'<ellipse cx="{base[0]}" cy="{base[1]}" rx="{brx}" ry="{bry}" class="basin-fill basin-stroke" stroke-width="1.4"/>')
svg.append(f'<ellipse cx="{base[0]}" cy="{base[1]}" rx="{brx*0.71:.0f}" ry="{bry*0.71:.0f}" class="basin-contour"/>')
svg.append(f'<ellipse cx="{base[0]}" cy="{base[1]}" rx="{brx*0.42:.0f}" ry="{bry*0.42:.0f}" class="basin-contour"/>')
svg.append(f'<text x="{base[0]}" y="{base[1]-bry-20}" text-anchor="middle" class="basin-lbl">low-loss basin (shared base checkpoint)</text>')
svg.append('</g>')

# parallelogram guide lines (construction: A->merge parallel to tau_B, B->merge parallel to tau_A)
svg.append('<g data-anim="2">')
svg.append(f'<line x1="{A[0]}" y1="{A[1]}" x2="{merge[0]}" y2="{merge[1]}" class="guide"/>')
svg.append(f'<line x1="{B[0]}" y1="{B[1]}" x2="{merge[0]}" y2="{merge[1]}" class="guide"/>')
svg.append('</g>')

# tau_A, tau_B arrows
svg.append('<g data-anim="2">')
svg.append(f'<line x1="{base[0]}" y1="{base[1]}" x2="{A[0]+8}" y2="{A[1]+6}" class="arrow-a"/>')
svg.append(f'<line x1="{base[0]}" y1="{base[1]}" x2="{B[0]-8}" y2="{B[1]+6}" class="arrow-b"/>')
svg.append(f'<text x="{(base[0]+A[0])/2-34}" y="{(base[1]+A[1])/2-6}" class="vec-lbl-a">tau_A</text>')
svg.append(f'<text x="{(base[0]+B[0])/2+10}" y="{(base[1]+B[1])/2-10}" class="vec-lbl-b">tau_B</text>')
svg.append('</g>')

# merge arrow (distinct: thick green)
svg.append('<g data-anim="3">')
svg.append(f'<line x1="{base[0]}" y1="{base[1]}" x2="{merge[0]+6}" y2="{merge[1]+10}" class="arrow-m"/>')
svg.append('</g>')

# negation arrow
svg.append('<g data-anim="4">')
svg.append(f'<line x1="{base[0]}" y1="{base[1]}" x2="{neg[0]-8}" y2="{neg[1]-6}" class="arrow-neg"/>')
svg.append(f'<text x="{neg[0]+10}" y="{neg[1]+2}" class="vec-lbl-a">- tau_A</text>')
svg.append(f'<text x="{neg[0]+10}" y="{neg[1]+18}" class="pt-sub">negate a behavior</text>')
svg.append('</g>')

# points on top
svg.append('<g data-anim="1">')
for (x, y), lbl, sub, cls, dx, dy, anchor in [
    (base, "theta_base", "pre-trained init", "pt-dot", 0, -14, "middle"),
    (A, "theta_A", "task A fine-tune", "pt-dot", -14, -12, "end"),
    (B, "theta_B", "task B fine-tune", "pt-dot", 14, -12, "start"),
    (neg, "theta_base - tau_A", None, "pt-dot", 0, 0, "start"),
]:
    svg.append(f'<circle cx="{x}" cy="{y}" r="5" class="{cls}"/>')
svg.append('</g>')

svg.append('<g data-anim="1">')
svg.append(f'<text x="{base[0]}" y="{base[1]+22}" text-anchor="middle" class="pt-lbl">theta_base</text>')
svg.append(f'<text x="{base[0]}" y="{base[1]+36}" text-anchor="middle" class="pt-sub">shared pre-trained init</text>')
svg.append(f'<text x="{A[0]-14}" y="{A[1]-10}" text-anchor="end" class="pt-lbl">theta_A</text>')
svg.append(f'<text x="{A[0]-14}" y="{A[1]+6}" text-anchor="end" class="pt-sub">task A fine-tune</text>')
svg.append(f'<text x="{B[0]+14}" y="{B[1]-10}" text-anchor="start" class="pt-lbl">theta_B</text>')
svg.append(f'<text x="{B[0]+14}" y="{B[1]+6}" text-anchor="start" class="pt-sub">task B fine-tune</text>')
svg.append('</g>')

svg.append('<g data-anim="3">')
svg.append(f'<circle cx="{merge[0]}" cy="{merge[1]}" r="5.5" class="pt-dot-m"/>')
svg.append(f'<text x="{merge[0]-14}" y="{merge[1]-16}" text-anchor="end" class="pt-lbl" style="fill:var(--good,#2f9e6e)">theta_merge</text>')
svg.append(f'<text x="{merge[0]-14}" y="{merge[1]-2}" text-anchor="end" class="pt-sub">= theta_base + tau_A + tau_B</text>')
svg.append(f'<text x="{merge[0]-14}" y="{merge[1]+12}" text-anchor="end" class="pt-sub" style="fill:var(--good,#2f9e6e)">still inside the basin</text>')
svg.append('</g>')

# axes indicator (bottom-left of main panel)
ax_x, ax_y = 46, 500
svg.append('<g data-anim="1">')
svg.append(f'<line x1="{ax_x}" y1="{ax_y}" x2="{ax_x+62}" y2="{ax_y}" class="axis-line" marker-end="url(#tvb-arrow-muted)"/>')
svg.append(f'<line x1="{ax_x}" y1="{ax_y}" x2="{ax_x}" y2="{ax_y-62}" class="axis-line" marker-end="url(#tvb-arrow-muted)"/>')
svg.append(f'<text x="{ax_x+66}" y="{ax_y+4}" class="ax-lbl">weight dim i</text>')
svg.append(f'<text x="{ax_x-6}" y="{ax_y-66}" class="ax-lbl" text-anchor="start">weight dim j</text>')
svg.append(f'<text x="{ax_x}" y="{ax_y+22}" class="ax-lbl">(weight-space; shown are 2 of millions of dims)</text>')
svg.append('</g>')

# ---------------- divider ----------------
svg.append(f'<line x1="560" y1="50" x2="560" y2="470" class="divider"/>')

# ---------------- second, unrelated basin ----------------
other = (680, 230)
svg.append('<g data-anim="5">')
svg.append(f'<ellipse cx="{other[0]}" cy="{other[1]}" rx="72" ry="62" class="basin-other-fill basin-stroke" stroke-width="1.2" stroke-dasharray="2 2"/>')
svg.append(f'<circle cx="{other[0]}" cy="{other[1]}" r="5" class="pt-dot"/>')
svg.append(f'<text x="{other[0]}" y="{other[1]-78}" text-anchor="middle" class="basin-lbl-faded">different base checkpoint</text>')
svg.append(f'<text x="{other[0]}" y="{other[1]-64}" text-anchor="middle" class="basin-lbl-faded">unrelated basin</text>')
svg.append(f'<text x="{other[0]}" y="{other[1]+22}" text-anchor="middle" class="pt-lbl">theta_other</text>')

# crossed dashed arrow between the two basins
cx0, cy0 = base[0]+brx-4, base[1]+40
cx1, cy1 = other[0]-72+4, other[1]+40
svg.append(f'<line x1="{cx0}" y1="{cy0}" x2="{cx1}" y2="{cy1}" class="cross-line"/>')
midx, midy = (cx0+cx1)/2, (cy0+cy1)/2
r = 8
svg.append(f'<line x1="{midx-r}" y1="{midy-r}" x2="{midx+r}" y2="{midy+r}" class="cross-line" stroke-dasharray="none" stroke-width="2"/>')
svg.append(f'<line x1="{midx-r}" y1="{midy+r}" x2="{midx+r}" y2="{midy-r}" class="cross-line" stroke-dasharray="none" stroke-width="2"/>')
svg.append(f'<text x="{midx}" y="{midy+24}" text-anchor="middle" class="cross-lbl">merging across bases = noise</text>')
svg.append('</g>')

# ---------------- legend ----------------
leg_y = 560
svg.append('<g data-anim="6">')
svg.append(f'<line x1="30" y1="{leg_y-16}" x2="{W-30}" y2="{leg_y-16}" class="basin-stroke" stroke-width="1" opacity="0.5"/>')

svg.append(f'<line x1="30" y1="{leg_y+2}" x2="58" y2="{leg_y+2}" class="swatch-line-a" marker-end="url(#tvb-arrow)"/>')
svg.append(f'<text x="66" y="{leg_y+6}" class="leg-title" style="fill:var(--accent,#c15b39)">tau = theta_ft - theta_base</text>')
svg.append(f'<text x="66" y="{leg_y+22}" class="leg-body">solid arrow: one task vector</text>')

svg.append(f'<line x1="230" y1="{leg_y+2}" x2="258" y2="{leg_y+2}" class="swatch-line-m" marker-end="url(#tvb-arrow)"/>')
svg.append(f'<text x="266" y="{leg_y+6}" class="leg-title" style="fill:var(--good,#2f9e6e)">tau_A + tau_B</text>')
svg.append(f'<text x="266" y="{leg_y+22}" class="leg-body">sum lands back inside the basin</text>')

svg.append(f'<line x1="470" y1="{leg_y+2}" x2="498" y2="{leg_y+2}" class="swatch-line-neg" marker-end="url(#tvb-arrow)"/>')
svg.append(f'<text x="506" y="{leg_y+6}" class="leg-title" style="fill:var(--accent,#c15b39)">- tau</text>')
svg.append(f'<text x="506" y="{leg_y+22}" class="leg-body">subtract to negate a behavior</text>')

svg.append(f'<line x1="628" y1="{leg_y+2}" x2="656" y2="{leg_y+2}" class="swatch-line-x"/>')
svg.append(f'<text x="664" y="{leg_y+6}" class="leg-title">different bases</text>')
svg.append(f'<text x="664" y="{leg_y+22}" class="leg-body">not composable</text>')
svg.append('</g>')

svg.append('</svg>')

figure = f'''<figure class="viz" id="fig-task-vector-arithmetic-basin">
<button class="viz-replay" type="button">&#8635; replay</button>
{chr(10).join(svg)}
<figcaption><b>Task vectors are literal arrows in weight space, and they add like vectors.</b> Fine-tuning on task A or task B moves the model from a shared checkpoint <span class="v-mono">theta_base</span> to <span class="v-mono">theta_A</span> or <span class="v-mono">theta_B</span>; the task vector is that displacement, <span class="v-mono">tau = theta_ft - theta_base</span>. Adding <span class="v-mono">tau_A + tau_B</span> lands on a merged model that is still inside the same low-loss basin, subtracting a task vector negates its behavior, but composing task vectors computed against two <i>different</i> base checkpoints has no such guarantee -- there is no shared basin for the arithmetic to land in.</figcaption>
</figure>
'''

with open("/local-ssd/pk669/programming/llm-stack-textbook/figures/task-vector-arithmetic-basin.html", "w") as f:
    f.write(figure)

print("generated")
