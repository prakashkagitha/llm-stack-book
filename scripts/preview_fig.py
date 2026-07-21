#!/usr/bin/env python3
"""Rasterize figure SVG(s) to PNG with cairosvg, resolving the theme CSS so colors
and the class-based styling (v-accent, v-grid, …) show concretely. Lets us VISUALLY
verify each figure's layout (overlaps, bounds, labels) as a reader would see it.

Usage:
  python3 scripts/preview_fig.py attention-flow flash-tiling
  python3 scripts/preview_fig.py --all --theme dark
Outputs PNGs into /tmp/figpreview/.
"""
import argparse, glob, os, re
import cairosvg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figures")
OUT = "/tmp/figpreview"

PALETTE = {
    "light": {"bg": "#faf8f3", "surface": "#ffffff", "surface-3": "#ece7da",
              "ink": "#211f1b", "ink-soft": "#45413a", "muted": "#6d675d",
              "border-2": "#d8d1c0", "accent": "#c15b39", "accent-2": "#b04e2d",
              "accent2": "#3b82f6", "good": "#2f9e6e", "warn": "#e0a106",
              "v-stroke": "#45413a", "v-grid": "#d8d1c0", "v-accent": "#c15b39",
              "v-fill": "#ece7da", "v-muted": "#6d675d", "sans": "sans-serif",
              "serif": "Georgia,serif", "mono": "monospace"},
    "dark": {"bg": "#181614", "surface": "#201d1a", "surface-3": "#2f2b27",
             "ink": "#ece7de", "ink-soft": "#cbc4b8", "muted": "#978f83",
             "border-2": "#413c35", "accent": "#e1845f", "accent-2": "#ec9069",
             "accent2": "#3b82f6", "good": "#2f9e6e", "warn": "#e0a106",
             "v-stroke": "#cbc4b8", "v-grid": "#413c35", "v-accent": "#e1845f",
             "v-fill": "#2f2b27", "v-muted": "#978f83", "sans": "sans-serif",
             "serif": "Georgia,serif", "mono": "monospace"},
}

CLASS_STYLE = """
text{{fill:{ink_soft};}}
.v-stroke{{stroke:{v_stroke};}} .v-fill{{fill:{v_fill};}}
.v-accent{{fill:{v_accent};}} .v-accent-s{{stroke:{v_accent};}}
.v-label{{fill:{ink_soft};}} .v-muted{{fill:{v_muted};}} .v-grid{{stroke:{v_grid};}}
.v-mono{{font-family:monospace;}}
"""


def resolve_vars(s, pal):
    # var(--name, fallback) -> fallback ; var(--name) -> palette[name] or currentColor
    def repl(m):
        name = m.group(1).strip()
        fb = m.group(2)
        if fb is not None:
            return fb.strip()
        return pal.get(name, "#888")
    s = re.sub(r"var\(\s*--([a-z0-9\-]+)\s*(?:,\s*([^)]+))?\)", repl, s)
    return s


# class token -> (attribute, palette-key); cairosvg's class-selector support is unreliable,
# so we bake these onto each element as presentation attributes for faithful color preview.
CLASS_ATTR = {
    "v-fill": ("fill", "v-fill"), "v-accent": ("fill", "v-accent"),
    "v-accent-s": ("stroke", "v-accent"), "v-stroke": ("stroke", "v-stroke"),
    "v-grid": ("stroke", "v-grid"), "v-muted": ("fill", "v-muted"),
    "v-label": ("fill", "ink-soft"),
}


def bake_classes(svg, pal):
    tag_re = re.compile(r"<(rect|circle|path|line|text|polygon|ellipse|g|tspan)\b([^>]*?)(/?)>")
    def fix(m):
        tag, attrs, close = m.group(1), m.group(2), m.group(3)
        cm = re.search(r'class="([^"]*)"', attrs)
        if cm:
            for tok in cm.group(1).split():
                if tok in CLASS_ATTR:
                    attr, key = CLASS_ATTR[tok]
                    if not re.search(rf'(?<![\w-]){attr}=', attrs):
                        attrs += f' {attr}="{pal[key]}"'
        return f"<{tag}{attrs}{close}>"
    return tag_re.sub(fix, svg)


def render(name, theme):
    path = os.path.join(FIGS, name + ".html")
    if not os.path.exists(path):
        print(f"✗ no such figure: {name}"); return None
    raw = open(path).read()
    m = re.search(r"<svg\b.*?</svg>", raw, re.S)
    if not m:
        print(f"✗ no <svg> in {name}"); return None
    svg = m.group(0)
    if "xmlns=" not in svg[:200]:
        # Without an explicit xmlns, cairosvg silently drops the whole <style>
        # block (fill AND font-size/family) -- fine in real browsers (HTML5
        # infers the SVG namespace) but it makes this preview misleading, so
        # inject it only for rasterization.
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    if "<svg" in svg[:20] and ' id="' not in svg[:200]:
        # The figures' scoped <style> selectors are prefixed "#fig-<name> ..."
        # but that id lives on the outer <figure>, not on <svg> itself -- once
        # we isolate just the <svg>...</svg> fragment those selectors match
        # nothing at all (real browsers are fine since <figure id="fig-..">
        # wraps the <svg> there). Recover the id from the <figure> tag so the
        # rules actually apply here too.
        fig_id = re.search(r'<figure\b[^>]*\bid="([^"]+)"', raw)
        if fig_id:
            svg = svg.replace("<svg", f'<svg id="{fig_id.group(1)}"', 1)
    pal = PALETTE[theme]
    svg = bake_classes(svg, pal)
    svg = resolve_vars(svg, pal)
    # make [data-anim] visible (final state)
    svg = svg.replace('data-anim=', 'data-shown=')
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, f"{name}.{theme}.png")
    try:
        cairosvg.svg2png(bytestring=svg.encode(), write_to=png, output_width=820,
                         background_color=pal["bg"])
        print(f"✓ {png}")
        return png
    except Exception as e:
        print(f"✗ cairosvg failed for {name} ({theme}): {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--theme", choices=["light", "dark", "both"], default="light")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = ([os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(FIGS, "*.html")))]
             if args.all else args.names)
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    for n in names:
        for t in themes:
            render(n, t)


if __name__ == "__main__":
    main()
