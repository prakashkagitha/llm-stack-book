#!/usr/bin/env python3
"""Emit figures/capstone-real-scaling-run.html from the MEASURED matched-budget run
(capstone/experiments/results_v2_matched/metrics.json). A log-x scaling plot: the 4 real
ladder points, the fitted L(N) curve, and the target's predicted (on-curve) vs actual
(measured) points nearly coinciding. Theme-safe, ASCII-only, self-contained SVG.
"""
import json, os, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(os.path.join(ROOT, "capstone/experiments/results_v2_matched/metrics.json")))
E, A, al = M["law"]["E"], M["law"]["A"], M["law"]["alpha"]
lad = [(r["nonembed"], r["val_loss"], r["tag"]) for r in M["ladder"]]
tN, tp, ta = M["target"]["nonembed"], M["target"]["predicted_val_loss"], M["target"]["actual_val_loss"]

# plot geometry
W, H = 760, 440
L, R, T, B = 92, 40, 30, 66            # margins
pw, ph = W - L - R, H - T - B
xlo, xhi = 6.45, 8.02                  # log10(N) range
ylo, yhi = 1.90, 2.26                  # val-loss range (inverted on screen)


def X(n):
    return L + (math.log10(n) - xlo) / (xhi - xlo) * pw


def Y(v):
    return T + (yhi - v) / (yhi - ylo) * ph


def law(n):
    return E + A / n ** al


el = []
# axes
el.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" class="v-stroke"/>')
el.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" class="v-stroke"/>')
# y gridlines + labels
for v in [1.9, 2.0, 2.1, 2.2]:
    y = Y(v)
    el.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+pw}" y2="{y:.1f}" class="v-grid"/>')
    el.append(f'<text x="{L-10}" y="{y+4:.1f}" text-anchor="end" class="v-mono v-label">{v:.1f}</text>')
# x ticks (param sizes)
for n, lbl in [(4e6, "4M"), (1e7, "10M"), (2e7, "20M"), (4e7, "40M"), (8.46e7, "85M")]:
    x = X(n)
    el.append(f'<line x1="{x:.1f}" y1="{T+ph}" x2="{x:.1f}" y2="{T+ph+5}" class="v-stroke"/>')
    el.append(f'<text x="{x:.1f}" y="{T+ph+20}" text-anchor="middle" class="v-mono v-label">{lbl}</text>')
# axis titles
el.append(f'<text x="{L+pw/2:.0f}" y="{H-8}" text-anchor="middle" class="v-label">non-embedding parameters N (log scale)</text>')
el.append(f'<text x="20" y="{T+ph/2:.0f}" text-anchor="middle" transform="rotate(-90 20 {T+ph/2:.0f})" class="v-label">held-out val loss (nats/token)</text>')
# fitted curve
pts = []
for i in range(61):
    n = 10 ** (xlo + (xhi - xlo) * i / 60)
    pts.append(f'{X(n):.1f},{Y(law(n)):.1f}')
el.append(f'<polyline points="{" ".join(pts)}" class="v-accent-s" fill="none" stroke-width="2" stroke-dasharray="1 0"/>')
# ladder points
for n, v, tag in lad:
    el.append(f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="5" class="v-fill"/>')
    el.append(f'<text x="{X(n):.1f}" y="{Y(v)-11:.1f}" text-anchor="middle" class="v-mono v-muted">{tag}</text>')
# target: predicted (on curve, open ring) and actual (filled star)
xt = X(tN)
el.append(f'<line x1="{xt:.1f}" y1="{T}" x2="{xt:.1f}" y2="{T+ph}" class="v-grid" stroke-dasharray="4 3"/>')
el.append(f'<circle cx="{xt:.1f}" cy="{Y(tp):.1f}" r="8" fill="none" class="v-accent" stroke-width="2.5"/>')
# star for actual
sx, sy = xt, Y(ta)
star = " ".join(f'{sx+7*math.cos(math.radians(a))*(1 if k%2==0 else 0.42):.1f},{sy-7*math.sin(math.radians(a))*(1 if k%2==0 else 0.42):.1f}'
                for k, a in enumerate(range(90, 90+360, 36)))
el.append(f'<polygon points="{star}" class="v-accent" />')
el.append(f'<text x="{xt-12:.1f}" y="{Y(tp)-16:.1f}" text-anchor="end" class="v-mono v-label">predicted {tp:.3f}</text>')
el.append(f'<text x="{xt+12:.1f}" y="{Y(ta)+22:.1f}" text-anchor="start" class="v-mono v-label">measured {ta:.3f}</text>')
el.append(f'<text x="{L+pw-6}" y="{T+16}" text-anchor="end" class="v-mono v-muted">L(N) = {E:.2f} + {A:.2f}/N^{al:.3f}</text>')

svg = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Measured scaling-law ladder: a fit on four small models predicts the 100M model held-out loss to within 0.002 nats.">' + "".join(el) + "</svg>"

html = f'''<figure class="viz" id="fig-capstone-real-scaling-run">
{svg}
<figcaption><b>A scaling law fit on four small models predicts the 100M model's loss to 0.002 nats.</b> Measured on one H100 (this book's own text as the training corpus): the ladder (S1&ndash;S4, 3.9M&ndash;44M non-embedding params) was trained under one recipe at a matched ~20M-token budget; fitting $L(N)=E+A/N^{{\\alpha}}$ to those four points and extrapolating to Stack-100M's 84.6M non-embedding params predicts {tp:.3f}, and the model actually trained to {ta:.3f}. The exponent itself is loosely constrained by only four points, but the held-out extrapolation is robust &mdash; which is exactly why you run a ladder before spending the full budget.</figcaption>
</figure>
'''
open(os.path.join(ROOT, "figures", "capstone-real-scaling-run.html"), "w").write(html)
print("wrote figures/capstone-real-scaling-run.html")
print(f"  ladder: {[(round(n/1e6,1), round(v,3)) for n,v,_ in lad]}")
print(f"  target: predicted {tp:.4f} vs measured {ta:.4f} (delta {abs(tp-ta):.4f})")
