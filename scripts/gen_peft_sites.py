#!/usr/bin/env python3
"""Generates figures/peft-intervention-sites.html"""

W, H = 820, 1060

def lock_icon(cx, cy, s=1.0):
    # simple padlock: body rect + shackle arc, size scaled by s
    bw, bh = 16*s, 12*s
    x = cx - bw/2
    y = cy - bh/2 + 4*s
    r = 5*s
    return f'''<g class="lock" transform="translate(0,0)">
    <path d="M {x+3*s} {y} v -{5*s} a {r} {r} 0 0 1 {bw-6*s} 0 v {5*s}" fill="none" class="v-stroke" stroke-width="{1.6*s}"/>
    <rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{2*s}" class="v-fill v-stroke" stroke-width="{1.4*s}"/>
  </g>'''

LAYER_H = 206

def layer_block(top_y, title, dim=False):
    x0, w = 90, 320
    h = LAYER_H
    cx = x0 + w/2
    out = []
    fillcls = "v-fill" if not dim else "v-fill dimmed"
    out.append(f'<g class="layer-block">')
    out.append(f'<rect x="{x0}" y="{top_y}" width="{w}" height="{h}" rx="12" class="{fillcls} v-stroke" stroke-width="1.5"/>')
    out.append(f'<text x="{cx}" y="{top_y+22}" text-anchor="middle" class="blk-title">{title}</text>')
    out.append(lock_icon(x0+24, top_y+22, 1.0))
    # FFN subbox
    fx0, fy0, fw, fh = 150, top_y+34, 200, 40
    out.append(f'<rect x="{fx0}" y="{fy0}" width="{fw}" height="{fh}" rx="6" class="sub-fill v-stroke" stroke-width="1.2"/>')
    out.append(f'<text x="{fx0+10}" y="{fy0+25}" class="sub-lbl">FFN</text>')
    # IA3 marker on FFN intermediate activation
    ia3x, ia3y = fx0+fw+14, fy0+fh/2
    out.append(odot(ia3x, ia3y))
    out.append(f'<text x="{ia3x}" y="{ia3y+22}" text-anchor="middle" class="tiny-lbl mono">l_ff</text>')
    out.append(f'<line x1="{fx0+fw}" y1="{fy0+fh/2}" x2="{ia3x-7}" y2="{ia3y}" class="ia3-line"/>')
    # self-attn subbox
    ax0, ay0, aw, ah = 150, top_y+90, 200, 100
    out.append(f'<rect x="{ax0}" y="{ay0}" width="{aw}" height="{ah}" rx="6" class="sub-fill v-stroke" stroke-width="1.2"/>')
    out.append(f'<text x="{ax0+10}" y="{ay0+18}" class="sub-lbl">self-attention</text>')
    # Q K V ports along bottom of attn subbox
    pw, ph, gap = 54, 16, 9
    px0 = ax0 + (aw - (3*pw + 2*gap))/2
    py = ay0 + ah - ph - 10
    labels = ["Q", "K", "V"]
    port_x = {}
    for i, lbl in enumerate(labels):
        px = px0 + i*(pw+gap)
        port_x[lbl] = px + pw/2
        out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="3" class="port-fill v-stroke" stroke-width="1"/>')
        out.append(f'<text x="{px+pw/2}" y="{py+ph-4}" text-anchor="middle" class="port-lbl mono">{lbl}</text>')
    # prefix-tuning chip above K and V ports
    for lbl in ["K", "V"]:
        pcx = port_x[lbl]
        chip_y = py - 26
        out.append(prefix_chip(pcx, chip_y))
        out.append(f'<line x1="{pcx}" y1="{chip_y+12}" x2="{pcx}" y2="{py-2}" class="prefix-line"/>')
    # IA3 markers on K and V paths (between port and Q/attn computation, drawn just above ports opposite side from prefix chip, to the outer side)
    for lbl, side in [("K", -1), ("V", 1)]:
        pcx = port_x[lbl]
        iax = pcx + side*30
        iay = py - 26
        out.append(odot(iax, iay, small=True))
        out.append(f'<line x1="{pcx}" y1="{py+2}" x2="{iax}" y2="{iay+7}" class="ia3-line"/>')
        out.append(f'<text x="{iax}" y="{iay-10}" text-anchor="middle" class="tiny-lbl mono">l_{lbl.lower()}</text>')
    out.append('</g>')
    return "\n".join(out)

def odot(cx, cy, small=False):
    r = 8 if not small else 7
    return (f'<g class="ia3-mark"><circle cx="{cx}" cy="{cy}" r="{r}" class="ia3-fill v-stroke" stroke-width="1.3"/>'
            f'<text x="{cx}" y="{cy+4}" text-anchor="middle" class="odot-x">*</text></g>')

def prefix_chip(cx, cy):
    w, h = 26, 14
    x = cx - w/2
    y = cy - h/2
    cells = 2
    cw = w/cells
    out = [f'<g class="prefix-chip">']
    for i in range(cells):
        out.append(f'<rect x="{x+i*cw}" y="{y}" width="{cw}" height="{h}" class="prefix-fill v-stroke" stroke-width="1"/>')
    out.append(f'</g>')
    return "\n".join(out)

svg = []
svg.append(f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="A transformer stack from input embeddings at the bottom to output logits at the top, with three PEFT intervention sites overlaid: prompt tuning adds k soft-token embeddings only at the input layer, prefix tuning prepends learned key and value vectors at every attention layer, and IA3 multiplies learned scale vectors into the key, value, and FFN activations at every layer, all while the backbone weights stay frozen.">')

svg.append(f'''<style>
  #fig-peft-intervention-sites text {{ fill: currentColor; }}
  #fig-peft-intervention-sites .blk-title  {{ font: 700 13px var(--sans,sans-serif); fill: var(--ink-soft,#334155); }}
  #fig-peft-intervention-sites .sub-lbl    {{ font: 600 11px var(--sans,sans-serif); fill: var(--muted,#64748b); }}
  #fig-peft-intervention-sites .port-lbl   {{ font: 700 10px var(--mono,monospace); fill: var(--ink-soft,#334155); }}
  #fig-peft-intervention-sites .tiny-lbl   {{ font: 600 10px var(--mono,monospace); fill: var(--muted,#64748b); }}
  #fig-peft-intervention-sites .cap-lbl    {{ font: 500 11px var(--sans,sans-serif); fill: var(--muted,#64748b); }}
  #fig-peft-intervention-sites .flow-lbl   {{ font: 600 12px var(--sans,sans-serif); fill: var(--ink-soft,#334155); }}
  #fig-peft-intervention-sites .ann-title  {{ font: 700 12.5px var(--sans,sans-serif); }}
  #fig-peft-intervention-sites .ann-body   {{ font: 500 11px var(--sans,sans-serif); fill: var(--muted,#64748b); }}
  #fig-peft-intervention-sites .leg-title  {{ font: 700 12px var(--sans,sans-serif); }}
  #fig-peft-intervention-sites .leg-body   {{ font: 500 10.5px var(--sans,sans-serif); fill: var(--muted,#64748b); }}
  #fig-peft-intervention-sites .mono       {{ font-family: var(--mono,monospace); }}
  #fig-peft-intervention-sites .v-fill     {{ fill: var(--surface,#fff); }}
  #fig-peft-intervention-sites .v-stroke   {{ stroke: var(--border-2,#475569); }}
  #fig-peft-intervention-sites .sub-fill   {{ fill: var(--surface-3,#eef1f5); }}
  #fig-peft-intervention-sites .port-fill  {{ fill: var(--surface,#fff); }}
  #fig-peft-intervention-sites .dimmed     {{ opacity: 0.85; }}
  #fig-peft-intervention-sites .lock path, #fig-peft-intervention-sites .lock rect {{ stroke: var(--muted,#64748b); fill: var(--surface-3,#eef1f5); }}
  #fig-peft-intervention-sites .flow-arrow {{ stroke: var(--ink-soft,#334155); stroke-width: 1.6; fill: none; marker-end: url(#pis-arrow); }}
  #fig-peft-intervention-sites .flow-arrow-d {{ stroke: var(--muted,#64748b); stroke-width: 1.4; stroke-dasharray: 4 3; fill: none; marker-end: url(#pis-arrow-m); }}
  #fig-peft-intervention-sites .prefix-fill {{ fill: var(--accent2,#3b82f6); fill-opacity: 0.22; }}
  #fig-peft-intervention-sites .prefix-line {{ stroke: var(--accent2,#3b82f6); stroke-width: 1.4; }}
  #fig-peft-intervention-sites .ia3-fill   {{ fill: var(--warn,#e0a106); fill-opacity: 0.25; }}
  #fig-peft-intervention-sites .ia3-line   {{ stroke: var(--warn,#e0a106); stroke-width: 1.2; }}
  #fig-peft-intervention-sites .odot-x     {{ font: 700 9px var(--mono,monospace); fill: var(--warn,#b8790a); }}
  #fig-peft-intervention-sites .soft-fill  {{ fill: var(--good,#2f9e6e); fill-opacity: 0.22; }}
  #fig-peft-intervention-sites .soft-stroke {{ stroke: var(--good,#2f9e6e); stroke-dasharray: 3 2; }}
  #fig-peft-intervention-sites .tok-fill   {{ fill: var(--surface-3,#eef1f5); }}
  #fig-peft-intervention-sites .brace      {{ stroke: var(--muted,#64748b); stroke-width: 1.3; fill: none; }}
  #fig-peft-intervention-sites .card-prefix {{ fill: var(--accent2,#3b82f6); }}
  #fig-peft-intervention-sites .card-ia3    {{ fill: var(--warn,#b8790a); }}
  #fig-peft-intervention-sites .card-soft   {{ fill: var(--good,#2f9e6e); }}
  #fig-peft-intervention-sites .ann-box    {{ fill: var(--surface,#fff); stroke: var(--border-2,#475569); stroke-width: 1; }}
</style>''')

svg.append('''<defs>
  <marker id="pis-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" style="color:var(--ink-soft,#334155)"/>
  </marker>
  <marker id="pis-arrow-m" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" style="color:var(--muted,#64748b)"/>
  </marker>
</defs>''')

# ---------- spine ----------
cx = 250

# output logits (topmost)
out_y = 20
svg.append(f'<g data-anim="1">')
svg.append(f'<rect x="150" y="{out_y}" width="200" height="38" rx="8" class="v-fill v-stroke" stroke-width="1.5"/>')
svg.append(f'<text x="{cx}" y="{out_y+24}" text-anchor="middle" class="flow-lbl">output logits</text>')
svg.append(f'<line x1="{cx}" y1="{out_y+38}" x2="{cx}" y2="{out_y+64}" class="flow-arrow"/>')
svg.append('</g>')

layerL_top = out_y + 64
svg.append(f'<g data-anim="1">{layer_block(layerL_top, "layer L")}</g>')
svg.append(f'<text x="410" y="{layerL_top-6}" text-anchor="end" class="cap-lbl">topmost layer</text>')

arrow1_y0 = layerL_top + LAYER_H
svg.append(f'<line x1="{cx}" y1="{arrow1_y0}" x2="{cx}" y2="{arrow1_y0+26}" class="flow-arrow-d"/>')

mid_top = arrow1_y0 + 26
svg.append(f'<g data-anim="2">{layer_block(mid_top, "x L layers", dim=True)}</g>')
svg.append(f'<text x="410" y="{mid_top-6}" text-anchor="end" class="cap-lbl">stands in for every layer 2 .. L-1</text>')
# ellipsis dots above/below middle block to show repetition
for dy in (-14,):
    pass

arrow2_y0 = mid_top + LAYER_H
svg.append(f'<line x1="{cx}" y1="{arrow2_y0}" x2="{cx}" y2="{arrow2_y0+26}" class="flow-arrow-d"/>')

layer1_top = arrow2_y0 + 26
svg.append(f'<g data-anim="3">{layer_block(layer1_top, "layer 1")}</g>')
svg.append(f'<text x="410" y="{layer1_top-6}" text-anchor="end" class="cap-lbl">bottommost layer</text>')

arrow3_y0 = layer1_top + LAYER_H
arrow3_y1 = arrow3_y0 + 20
svg.append(f'<line x1="{cx}" y1="{arrow3_y0}" x2="{cx}" y2="{arrow3_y1}" class="flow-arrow"/>')

# ---------- input embeddings row + prompt-tuning soft tokens ----------
label_y = arrow3_y1 + 18
svg.append(f'<g data-anim="1">')
svg.append(f'<text x="{cx}" y="{label_y}" text-anchor="middle" class="flow-lbl">input embeddings</text>')

emb_top = label_y + 14
cellw, cellh = 26, 30
n_real = 6
real_x0 = 190
for i in range(n_real):
    x = real_x0 + i*cellw
    svg.append(f'<rect x="{x}" y="{emb_top}" width="{cellw-3}" height="{cellh}" rx="3" class="tok-fill v-stroke" stroke-width="1"/>')
svg.append(f'<text x="{real_x0 + n_real*cellw/2}" y="{emb_top+cellh+16}" text-anchor="middle" class="cap-lbl">n real tokens X</text>')

n_soft = 3
soft_x0 = real_x0 - 12 - n_soft*cellw
for i in range(n_soft):
    x = soft_x0 + i*cellw
    svg.append(f'<rect x="{x}" y="{emb_top}" width="{cellw-3}" height="{cellh}" rx="3" class="soft-fill soft-stroke" stroke-width="1.4"/>')
svg.append(f'<text x="{soft_x0 + n_soft*cellw/2}" y="{emb_top+cellh+16}" text-anchor="middle" class="cap-lbl">k soft tokens P</text>')
svg.append(f'<text x="{real_x0-6}" y="{emb_top+cellh/2+5}" text-anchor="end" class="flow-lbl">+</text>')
svg.append('</g>')

# bracket + annotation for prompt tuning
brk_y = emb_top + cellh + 30
svg.append(f'<path d="M {soft_x0} {brk_y} q -6 0 -6 6 v 4 q 0 6 -6 6 q 6 0 6 6 v 4 q 0 6 6 6" class="brace" fill="none" transform="translate(0,0)"/>')
# simpler: draw a bottom brace under soft token cells
svg.append(f'<path d="M {soft_x0} {brk_y} L {soft_x0+n_soft*cellw-3} {brk_y}" class="brace"/>')
ann1_x = soft_x0 - 6
ann1_y = brk_y + 24
svg.append(f'<g data-anim="2">')
svg.append(f'<rect x="{40}" y="{ann1_y}" width="300" height="56" rx="8" class="ann-box"/>')
svg.append(f'<rect x="{40}" y="{ann1_y}" width="6" height="56" class="card-soft"/>')
svg.append(f'<text x="{58}" y="{ann1_y+20}" class="ann-title card-soft-text" style="fill:var(--good,#2f9e6e)">PROMPT TUNING</text>')
svg.append(f'<text x="{58}" y="{ann1_y+38}" class="ann-body">k learned embeddings,</text>')
svg.append(f'<text x="{58}" y="{ann1_y+52}" class="ann-body">input layer only</text>')
svg.append('</g>')

# ---------- right-side annotation cards for prefix tuning + IA3 ----------
card_x = 480
card_w = 300

svg.append(f'<g data-anim="3">')
prefix_ann_y = layerL_top + 18
svg.append(f'<rect x="{card_x}" y="{prefix_ann_y}" width="{card_w}" height="72" rx="8" class="ann-box"/>')
svg.append(f'<rect x="{card_x}" y="{prefix_ann_y}" width="6" height="72" class="card-prefix"/>')
svg.append(f'<g transform="translate({card_x+24},{prefix_ann_y+20})">{prefix_chip(0,0)}</g>')
svg.append(f'<text x="{card_x+44}" y="{prefix_ann_y+24}" class="ann-title" style="fill:var(--accent2,#3b82f6)">PREFIX TUNING</text>')
svg.append(f'<text x="{card_x+22}" y="{prefix_ann_y+44}" class="ann-body">learned K/V prepended at every layer</text>')
svg.append(f'<text x="{card_x+22}" y="{prefix_ann_y+60}" class="ann-body">[P_K;K], [P_V;V] -- repeated tag on L, mid, 1</text>')
# single leader line from card to the nearest layer block's prefix chip (layer L)
kyL = layerL_top + 90 + 100 - 16 - 10 - 26 + 7
svg.append(f'<path d="M {card_x} {prefix_ann_y+36} C {card_x-40} {prefix_ann_y+36}, {410} {kyL}, {392} {kyL}" class="brace" stroke-dasharray="2 3" opacity="0.7" marker-end="url(#pis-arrow-m)"/>')
svg.append('</g>')

svg.append(f'<g data-anim="4">')
ia3_ann_y = mid_top + 32
svg.append(f'<rect x="{card_x}" y="{ia3_ann_y}" width="{card_w}" height="80" rx="8" class="ann-box"/>')
svg.append(f'<rect x="{card_x}" y="{ia3_ann_y}" width="6" height="80" class="card-ia3"/>')
svg.append(f'<g transform="translate({card_x+30},{ia3_ann_y+22})">{odot(0,0)}</g>')
svg.append(f'<text x="{card_x+50}" y="{ia3_ann_y+26}" class="ann-title" style="fill:var(--warn,#b8790a)">IA3</text>')
svg.append(f'<text x="{card_x+22}" y="{ia3_ann_y+48}" class="ann-body">element-wise scale on K, V, FFN,</text>')
svg.append(f'<text x="{card_x+22}" y="{ia3_ann_y+64}" class="ann-body">every layer -- l_k*K, l_v*V, l_ff*FFN(h)</text>')
# single leader line from card to the middle block's FFN IA3 marker
fyM = mid_top + 34 + 20
svg.append(f'<path d="M {card_x} {ia3_ann_y+40} C {card_x-40} {ia3_ann_y+40}, {410} {fyM}, {392} {fyM}" class="brace" stroke-dasharray="2 3" opacity="0.7" marker-end="url(#pis-arrow-m)"/>')
svg.append('</g>')

# ---------- frozen backbone note ----------
svg.append(f'<g data-anim="1">')
svg.append(f'<text x="{60}" y="{layerL_top-6}" class="cap-lbl">lock = backbone weights frozen</text>')
svg.append('</g>')

# ---------- bottom legend ----------
leg_y = ann1_y + 56 + 46
svg.append(f'<g data-anim="5">')
svg.append(f'<line x1="30" y1="{leg_y-14}" x2="{W-30}" y2="{leg_y-14}" class="v-stroke" stroke-width="1" opacity="0.5"/>')
svg.append(f'<text x="30" y="{leg_y+4}" class="leg-title" style="fill:var(--good,#2f9e6e)">Prompt tuning</text>')
svg.append(f'<rect x="30" y="{leg_y+14}" width="18" height="14" class="soft-fill soft-stroke" stroke-width="1.4"/>')
svg.append(f'<text x="56" y="{leg_y+25}" class="leg-body">dashed cell = k soft-token embeds, input only</text>')

svg.append(f'<text x="300" y="{leg_y+4}" class="leg-title" style="fill:var(--accent2,#3b82f6)">Prefix tuning</text>')
svg.append(f'<g transform="translate(300,{leg_y+14})">{prefix_chip(9,7)}</g>')
svg.append(f'<text x="326" y="{leg_y+25}" class="leg-body">striped cells = learned K/V prepended, every layer</text>')

svg.append(f'<text x="580" y="{leg_y+4}" class="leg-title" style="fill:var(--warn,#b8790a)">IA3</text>')
svg.append(f'<g transform="translate(589,{leg_y+21})">{odot(0,0,small=True)}</g>')
svg.append(f'<text x="606" y="{leg_y+25}" class="leg-body">circled x = elementwise scale, every layer</text>')

svg.append(f'<text x="30" y="{leg_y+46}" class="leg-body">All three methods keep backbone Q/K/V and FFN weights (lock icon) fully frozen -- only the small overlay object shown per method is trained.</text>')
svg.append('</g>')

svg.append('</svg>')

figure = f'''<figure class="viz" id="fig-peft-intervention-sites">
<button class="viz-replay" type="button">&#8635; replay</button>
{chr(10).join(svg)}
<figcaption><b>Three ways to intervene on a frozen transformer without touching its weights.</b> Prompt tuning adds <span class="mono">k</span> learned embeddings only at the input layer; prefix tuning prepends learned key/value vectors at every attention layer; IA3 multiplies learned scale vectors into K, V, and the FFN activation at every layer. All three keep the backbone weights frozen, differing only in <i>where</i> and <i>what kind</i> of small trainable object they attach.</figcaption>
</figure>
'''

with open("/local-ssd/pk669/programming/llm-stack-textbook/figures/peft-intervention-sites.html", "w") as f:
    f.write(figure)

print("done, total height used up to", leg_y+60)
