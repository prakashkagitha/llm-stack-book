#!/usr/bin/env python3
"""Static-site generator for *The LLM Stack* textbook.

Builds two sibling sites from the same repo:
  - the main book      (book.json)      -> site/
  - the interview prep  (interview.json) -> site/interview/   (flat layout)

For each chapter it renders content/<part>/<chapter>.md into HTML with sidebar
nav, per-page TOC, search index, syntax highlighting (pygments, light+dark),
KaTeX math, callouts, copy buttons, and inline animated SVG figures included via
`{{fig:NAME}}` markers (resolved from figures/NAME.html).

Usage:  python3 build.py
"""
import json, os, re, html, shutil, subprocess, datetime, glob
from html.parser import HTMLParser

import markdown
from pygments.formatters import HtmlFormatter

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(ROOT, "content")
SITE = os.path.join(ROOT, "site")
ASSETS_SRC = os.path.join(ROOT, "assets")
FIGURES = os.path.join(ROOT, "figures")

# Public deployment base (GitHub Pages project site). Used for canonical URLs,
# Open Graph/Twitter cards, and sitemap.xml — all of which need absolute URLs.
SITE_BASE = "https://prakashkagitha.github.io/llm-stack-book/"
OG_IMAGE = SITE_BASE + "assets/og-image.png"
REPO_URL = "https://github.com/prakashkagitha/llm-stack-book"
BUILD_DATE = datetime.date.today().isoformat()
YEAR = datetime.date.today().year

# chapter-id -> {path, hw, title} for the GPU-tier notebooks (1xH100 / 2xA100), if present.
try:
    with open(os.path.join(ROOT, "notebooks-gpu", "manifest.json")) as _f:
        GPU_NOTEBOOKS = json.load(_f)
except (OSError, ValueError):
    GPU_NOTEBOOKS = {}

_GIT_DATE_CACHE = {}


def git_last_date(path):
    """Last git commit date (YYYY-MM-DD) for a file, for the per-chapter 'last updated'
    stamp. Falls back to the build date (e.g. a shallow CI clone with no per-file history)."""
    if path in _GIT_DATE_CACHE:
        return _GIT_DATE_CACHE[path]
    d = BUILD_DATE
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", path],
                             cwd=ROOT, capture_output=True, text=True, timeout=10)
        s = out.stdout.strip()
        if s:
            d = s
    except Exception:
        pass
    _GIT_DATE_CACHE[path] = d
    return d

MD_EXTENSIONS = [
    "extra", "sane_lists", "smarty", "admonition", "meta", "toc",
    "pymdownx.superfences", "pymdownx.highlight", "pymdownx.inlinehilite",
    "pymdownx.arithmatex", "pymdownx.details", "pymdownx.tabbed",
    "pymdownx.tasklist", "pymdownx.tilde", "pymdownx.caret", "pymdownx.keys",
    "pymdownx.betterem", "pymdownx.smartsymbols",
]
MD_CONFIG = {
    "pymdownx.highlight": {"use_pygments": True, "guess_lang": False,
                            "css_class": "highlight", "pygments_style": "default"},
    "pymdownx.arithmatex": {"generic": True},
    "pymdownx.superfences": {},
    "pymdownx.tasklist": {"custom_checkbox": True},
    "toc": {"permalink": "¶", "toc_depth": "2-3", "permalink_class": "headerlink"},
}


# ----------------------------------------------------------------------------- helpers

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "pre", "code", "svg"):
            self.skip += 1
    def handle_endtag(self, tag):
        if tag in ("script", "style", "pre", "code", "svg") and self.skip:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)
    def text(self):
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def strip_html(s):
    p = TextExtractor()
    try:
        p.feed(s)
    except Exception:
        return re.sub(r"<[^>]+>", " ", s)
    return p.text()


def rel(from_url, to_url):
    """Relative path from one page to another (both relative to the collection root)."""
    from_dir = os.path.dirname(from_url)
    return os.path.relpath(to_url, from_dir or ".").replace(os.sep, "/")


def _flatten_toc(tokens):
    out = []
    for t in tokens or []:
        out.append(t)
        out.extend(_flatten_toc(t.get("children")))
    return out


# Figure-include system: `{{fig:name}}` -> contents of figures/name.html (a self-contained
# <figure class="viz"> ... inline <svg> ... </figure>). Keeps chapter markdown clean and lets
# the figure library grow independently. Missing figures degrade to a visible note (no crash).
_FIG_RE = re.compile(r"\{\{fig:([a-z0-9\-]+)\}\}")
_FIG_CACHE = {}
_FIG_MISSING = set()


def _load_figure(name):
    if name in _FIG_CACHE:
        return _FIG_CACHE[name]
    path = os.path.join(FIGURES, name + ".html")
    if os.path.exists(path):
        with open(path) as f:
            snippet = f.read().strip()
    else:
        _FIG_MISSING.add(name)
        snippet = (f'<figure class="viz viz-missing"><div class="viz-missing-note">'
                   f'Figure <code>{html.escape(name)}</code> not found.</div></figure>')
    _FIG_CACHE[name] = snippet
    return snippet


def expand_figures(raw):
    return _FIG_RE.sub(lambda m: "\n\n" + _load_figure(m.group(1)) + "\n\n", raw)


# Interactive-tool include system: `{{tool:name}}` -> contents of tools/name.html (a
# self-contained interactive widget; unlike figures, tools MAY contain an inline <script>).
_TOOL_RE = re.compile(r"\{\{tool:([a-z0-9\-]+)\}\}")
_TOOLS_DIR = os.path.join(ROOT, "tools")
_TOOL_MISSING = set()


def _load_tool(name):
    path = os.path.join(_TOOLS_DIR, name + ".html")
    if os.path.exists(path):
        return open(path).read().strip()
    _TOOL_MISSING.add(name)
    return (f'<div class="viz viz-missing"><div class="viz-missing-note">'
            f'Tool <code>{html.escape(name)}</code> not found.</div></div>')


def expand_tools(raw):
    return _TOOL_RE.sub(lambda m: "\n\n" + _load_tool(m.group(1)) + "\n\n", raw)


# ----------------------------------------------------------------------------- collection model

def load_manifest(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


class Collection:
    """One buildable site: a manifest + output rules.

    flat=True  -> chapters render to <out_subdir>/<file>.html (interview companion)
    flat=False -> chapters render to <out_subdir>/<part_dir>/<file>.html (main book)
    """
    def __init__(self, manifest, out_subdir="", flat=False, link_rewrites=None,
                 search_name="search-index.json", parent=None):
        self.m = manifest
        self.out_subdir = out_subdir            # "" for book, "interview" for companion
        self.flat = flat
        self.link_rewrites = link_rewrites or []  # list of (regex, replacement)
        self.search_name = search_name
        self.parent = parent or manifest.get("parent")
        self.flat_chapters = self._flatten()

    def _url(self, part_dir, chap_file):
        if self.flat:
            return f"{chap_file}.html"
        return f"{part_dir}/{chap_file}.html"

    def _flatten(self):
        flat = []
        part_no = 0
        for part in self.m["parts"]:
            is_front = part["dir"].startswith(("00", "99"))
            if not is_front:
                part_no += 1
            for i, ch in enumerate(part["chapters"], 1):
                flat.append({
                    "part_dir": part["dir"], "part_title": part["title"], "part_no": part_no,
                    "is_front": is_front, "chap_no": i, "title": ch["title"],
                    "file": ch["file"], "scope": ch.get("scope", ""),
                    "url": self._url(part["dir"], ch["file"]),
                    "md_path": os.path.join(CONTENT, part["dir"], ch["file"] + ".md"),
                })
        return flat

    @property
    def n_parts(self):
        return len([p for p in self.m["parts"] if not p["dir"].startswith(("00", "99"))])

    def out_root(self):
        return os.path.join(SITE, self.out_subdir) if self.out_subdir else SITE

    def asset_base(self, url):
        """Relative prefix from a page back to where assets/ live (always at site/)."""
        depth = url.count("/") + (1 if self.out_subdir else 0)
        return "../" * depth

    def rewrite_links(self, body):
        for pat, repl in self.link_rewrites:
            body = pat.sub(repl, body)
        return body


# ----------------------------------------------------------------------------- rendering

def render_sidebar(coll, current_url):
    out = ['<nav class="sidebar" id="sidebar">']
    if coll.parent:
        out.append(f'<a class="sidebar-parent" href="{html.escape(coll.parent["href"])}">'
                   f'← {html.escape(coll.parent["title"])}</a>')
    part_no = 0
    for part in coll.m["parts"]:
        is_front = part["dir"].startswith(("00", "99"))
        if not is_front:
            part_no += 1
        out.append(f'<div class="nav-part" data-part="{part["dir"]}">')
        out.append(f'<div class="nav-part-title"><span>{html.escape(part["title"])}</span>'
                   f'<span class="chev">▾</span></div>')
        out.append('<ul class="nav-chapters">')
        for i, ch in enumerate(part["chapters"], 1):
            url = coll._url(part["dir"], ch["file"])
            active = "active" if url == current_url else ""
            href = rel(current_url, url)
            num = "" if is_front else f'<span class="nav-num">{part_no}.{i}</span>'
            out.append(f'<li><a class="{active}" href="{href}">{num}{html.escape(ch["title"])}</a></li>')
        out.append("</ul></div>")
    out.append("</nav>")
    return "\n".join(out)


def render_toc_side(toc_tokens):
    if not toc_tokens:
        return ""
    items = []
    def walk(tokens):
        for t in tokens:
            if t["level"] > 3:
                continue
            cls = "toc-l3" if t["level"] == 3 else "toc-l2"
            items.append(f'<li class="{cls}"><a href="#{t["id"]}">{html.escape(t["name"])}</a></li>')
            if t.get("children"):
                walk(t["children"])
    walk(toc_tokens)
    if not items:
        return ""
    return ('<aside class="toc-side"><div class="toc-title">On this page</div>'
            '<ul>' + "\n".join(items) + "</ul></aside>")


PAGE_TMPL = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — {brand}</title>
<meta name="description" content="{desc}">
<link rel="canonical" href="{canonical}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="{brand}">
<meta property="og:title" content="{title} — {brand}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="{og_image}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title} — {brand}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{og_image}">
<script>(function(){{try{{var t=localStorage.getItem('llmbook-theme');if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';document.documentElement.setAttribute('data-theme',t);}}catch(e){{}}}})();</script>
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{base}assets/style.css">
<link rel="stylesheet" href="{base}assets/pygments.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" media="print" onload="this.media='all'">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" media="print" onload="this.media='all'">
<noscript><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"></noscript>
</head>
<body>
<div class="progress-bar"></div>
<header class="topbar">
  <button class="icon-btn menu-toggle" id="menu-toggle" aria-label="Menu">☰</button>
  <a class="brand" href="{home}"><span class="logo">Λ</span><span class="brand-text">{brand}<small>{brand_sub}</small></span></a>
  <div class="spacer"></div>
  <div class="search-box">
    <span class="si">⌕</span>
    <input id="search-input" type="text" placeholder="Search… ( / )" autocomplete="off" spellcheck="false" data-base="{base}" data-index="{search_name}">
    <div class="search-results" id="search-results"></div>
  </div>
  <button class="icon-btn" id="theme-toggle" aria-label="Toggle theme">☾</button>
</header>
<div class="scrim"></div>
<div class="layout">
{sidebar}
<main class="main">
<article class="content">
{breadcrumb}
{chapter_meta}
{body}
{page_nav}
</article>
<footer class="site-footer">
  <span>{brand} · {brand_sub}</span>
  <span class="footer-links"><a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · <a href="{repo_url}">Source &amp; errata</a></span>
</footer>
</main>
{toc_side}
</div>
<script>window.MathJax=null;</script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}},{{left:'$$',right:'$$',display:true}}],throwOnError:false}});"></script>
<script src="{base}assets/app.js"></script>
</body>
</html>
"""


def build_collection(coll):
    flat = coll.flat_chapters
    out_root = coll.out_root()
    os.makedirs(out_root, exist_ok=True)

    brand = coll.m["title"]
    brand_sub = coll.m.get("subtitle", "")
    search_index = []
    total_words = written = placeholders = 0

    for idx, c in enumerate(flat):
        md = markdown.Markdown(extensions=MD_EXTENSIONS, extension_configs=MD_CONFIG)
        if os.path.exists(c["md_path"]):
            with open(c["md_path"]) as f:
                raw = f.read()
            written += 1
        else:
            num = "" if c["is_front"] else f'{c["part_no"]}.{c["chap_no"]} '
            raw = (f"# {num}{c['title']}\n\n"
                   f"!!! note \"Draft in progress\"\n    This chapter is being written.\n\n"
                   f"**Scope.** {c['scope']}\n")
            placeholders += 1

        raw = expand_figures(raw)
        raw = expand_tools(raw)
        body = md.convert(raw)
        body = coll.rewrite_links(body)
        toc_tokens = getattr(md, "toc_tokens", [])
        text = strip_html(body)
        words = len(text.split())
        total_words += words

        base = coll.asset_base(c["url"])
        home = base + (f"{coll.out_subdir}/index.html" if coll.out_subdir else "index.html")
        num = "" if c["is_front"] else f'{c["part_no"]}.{c["chap_no"]} · '
        crumb = (f'<div class="chapter-eyebrow">{html.escape(c["part_title"])}</div>')
        reading = max(1, round(words / 220))     # ~220 wpm technical reading
        updated = git_last_date(c["md_path"])
        nb_rel = f'{c["part_dir"]}/{c["file"]}.ipynb'
        colab = ""
        if os.path.exists(os.path.join(ROOT, "notebooks", nb_rel)):
            colab = ('<span class="cm-dot">·</span>'
                     f'<a class="cm-colab" target="_blank" rel="noopener" '
                     f'href="https://colab.research.google.com/github/prakashkagitha/llm-stack-book/blob/main/notebooks/{nb_rel}">'
                     '&#9654; Run the code (Colab)</a>')
        # Optional GPU-tier notebook (1xH100 / 2xA100) for compute-heavy chapters.
        gpu = ""
        gnb = GPU_NOTEBOOKS.get(f'{c["part_dir"]}/{c["file"]}')
        if gnb:
            gpu = ('<span class="cm-dot">·</span>'
                   f'<a class="cm-colab" target="_blank" rel="noopener" '
                   f'href="https://colab.research.google.com/github/prakashkagitha/llm-stack-book/blob/main/notebooks-gpu/{gnb["path"]}">'
                   f'&#9889; Run on GPU ({html.escape(gnb["hw"])})</a>')
        chapter_meta = (f'<div class="chapter-meta"><span>{reading} min read</span>'
                        f'<span class="cm-dot">·</span>'
                        f'<span>Updated <time datetime="{updated}">{updated}</time></span>'
                        f'{colab}{gpu}</div>')
        canonical = SITE_BASE + (f'{coll.out_subdir}/' if coll.out_subdir else "") + c["url"]

        prev_c = flat[idx - 1] if idx > 0 else None
        next_c = flat[idx + 1] if idx < len(flat) - 1 else None
        pn = ['<nav class="page-nav">']
        if prev_c:
            pn.append(f'<a class="prev" href="{rel(c["url"], prev_c["url"])}">'
                      f'<span class="pn-label">← Previous</span>'
                      f'<span class="pn-title">{html.escape(prev_c["title"])}</span></a>')
        else:
            pn.append("<span></span>")
        if next_c:
            pn.append(f'<a class="next" href="{rel(c["url"], next_c["url"])}">'
                      f'<span class="pn-label">Next →</span>'
                      f'<span class="pn-title">{html.escape(next_c["title"])}</span></a>')
        pn.append("</nav>")

        page = PAGE_TMPL.format(
            title=html.escape(c["title"]), brand=html.escape(brand),
            brand_sub=html.escape(brand_sub), desc=html.escape(c["scope"][:160]),
            base=base, home=home, search_name=coll.search_name,
            canonical=html.escape(canonical), og_image=OG_IMAGE, repo_url=REPO_URL,
            sidebar=render_sidebar(coll, c["url"]),
            breadcrumb=crumb, chapter_meta=chapter_meta, body=body, page_nav="\n".join(pn),
            toc_side=render_toc_side(toc_tokens),
        )
        dest = os.path.join(out_root, c["url"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(page)

        headings = " ".join(t["name"] for t in _flatten_toc(toc_tokens))
        search_index.append({
            "url": c["url"], "title": c["title"], "part": c["part_title"],
            "headings": headings, "text": text[:1200],
        })

    with open(os.path.join(SITE, "assets", coll.search_name), "w") as f:
        json.dump(search_index, f, ensure_ascii=False)

    write_index(coll, total_words, len(flat))
    return total_words, written, placeholders


INDEX_TMPL = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}: {subtitle}</title>
<meta name="description" content="{tagline}">
<link rel="canonical" href="{canonical}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="{title}">
<meta property="og:title" content="{title}: {subtitle}">
<meta property="og:description" content="{tagline}">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="{og_image}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}: {subtitle}">
<meta name="twitter:description" content="{tagline}">
<meta name="twitter:image" content="{og_image}">
<script>(function(){{try{{var t=localStorage.getItem('llmbook-theme');if(!t)t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';document.documentElement.setAttribute('data-theme',t);}}catch(e){{}}}})();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{base}assets/style.css">
<link rel="stylesheet" href="{base}assets/pygments.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" media="print" onload="this.media='all'">
<noscript><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"></noscript>
</head>
<body>
<div class="progress-bar"></div>
<header class="topbar">
  <button class="icon-btn menu-toggle" id="menu-toggle" aria-label="Menu">☰</button>
  <a class="brand" href="{home}"><span class="logo">Λ</span><span class="brand-text">{title}<small>{subtitle}</small></span></a>
  <div class="spacer"></div>
  <div class="search-box">
    <span class="si">⌕</span>
    <input id="search-input" type="text" placeholder="Search… ( / )" autocomplete="off" spellcheck="false" data-base="{base}" data-index="{search_name}">
    <div class="search-results" id="search-results"></div>
  </div>
  <button class="icon-btn" id="theme-toggle" aria-label="Toggle theme">☾</button>
</header>
<div class="scrim"></div>
<div class="layout">
{sidebar}
<main class="main">
<article class="content">
<section class="hero">
  <div class="hero-eyebrow">{eyebrow}</div>
  <h1 class="hero-title">{title}<span class="hero-sub">{subtitle}</span></h1>
  <p class="hero-tagline">{tagline}</p>
  <div class="stats">
    <div class="stat"><div class="num">{nparts}</div><div class="lbl">Parts</div></div>
    <div class="stat"><div class="num">{nchapters}</div><div class="lbl">Chapters</div></div>
    <div class="stat"><div class="num">~{pages}</div><div class="lbl">Pages</div></div>
    <div class="stat"><div class="num">{kwords}k</div><div class="lbl">Words</div></div>
  </div>
  {cta}
</section>
<div class="toc-grid">
{cards}
</div>
{cite}
</article>
<footer class="site-footer">
  <span>{title} · {subtitle}</span>
  <span class="footer-links"><a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · <a href="{repo_url}">Source &amp; errata</a></span>
</footer>
</main>
</div>
<script src="{base}assets/app.js"></script>
</body>
</html>
"""


def write_index(coll, total_words, nch):
    cards = []
    part_no = 0
    for part in coll.m["parts"]:
        is_front = part["dir"].startswith(("00", "99"))
        if not is_front:
            part_no += 1
        tag = part["title"]
        ptag = "" if is_front else f'<span class="part-no">{_roman(part_no)}</span>'
        lis = []
        for i, ch in enumerate(part["chapters"], 1):
            url = coll._url(part["dir"], ch["file"])
            lis.append(f'<li><a href="{url}">{html.escape(ch["title"])}</a></li>')
        cards.append(
            f'<div class="toc-card"><div class="part-tag">{ptag}{html.escape(tag)}</div>'
            f'<ol>{"".join(lis)}</ol></div>'
        )

    base = "../" if coll.out_subdir else ""
    # brand/home points at this collection's own index (same dir as this page)
    home = "index.html"
    cta = ""
    if coll.parent:
        cta = (f'<p class="hero-cta"><a class="btn-ghost" href="{html.escape(coll.parent["href"])}">'
               f'← Back to {html.escape(coll.parent["title"])}</a></p>')
    elif coll.out_subdir == "":
        # main book: link to interview companion + first chapter
        first = coll.flat_chapters[0]["url"] if coll.flat_chapters else "#"
        cta = (f'<p class="hero-cta"><a class="btn-primary" href="{first}">Start reading →</a>'
               f'<a class="btn-ghost" href="map/index.html">Learning map</a>'
               f'<a class="btn-ghost" href="glossary/index.html">Glossary</a>'
               f'<a class="btn-ghost" href="tools/index.html">Tools &amp; calculators</a>'
               f'<a class="btn-ghost" href="interview/index.html">Interview companion</a>'
               f'<a class="btn-ghost" href="print.html">Print / PDF edition</a></p>')

    canonical = SITE_BASE + (f'{coll.out_subdir}/' if coll.out_subdir else "")
    cite = ""
    if coll.out_subdir == "":                 # BibTeX cite block on the main book homepage
        bib = ("@book{kagitha_llm_stack_" + str(YEAR) + ",\n"
               "  title  = {The LLM Stack: From Silicon to Agents},\n"
               "  author = {Kagitha, Prakash},\n"
               "  year   = {" + str(YEAR) + "},\n"
               "  url    = {" + SITE_BASE + "},\n"
               "  note   = {Open web textbook, CC BY 4.0}\n}")
        cite = ('<section class="cite-book"><h2>Cite this book</h2>'
                '<pre class="cite-bibtex"><code>' + html.escape(bib) + '</code></pre></section>')

    html_out = INDEX_TMPL.format(
        title=html.escape(coll.m["title"]),
        subtitle=html.escape(coll.m.get("subtitle", "")),
        tagline=html.escape(coll.m.get("tagline", "")),
        eyebrow="Interview Companion" if coll.parent else "An open, ground-up field guide",
        nparts=coll.n_parts, nchapters=nch,
        pages=f"{total_words // 300:,}", kwords=f"{total_words // 1000}",
        base=base, home=home, search_name=coll.search_name,
        canonical=html.escape(canonical), og_image=OG_IMAGE, repo_url=REPO_URL, cite=cite,
        sidebar=render_sidebar(coll, "index.html"),
        cards="\n".join(cards), cta=cta,
    )
    out_root = coll.out_root()
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "index.html"), "w") as f:
        f.write(html_out)


def write_tools_hub(book):
    """Collect every tools/*.html into one /tools hub page (the tools also live inline in
    their chapters). Title comes from the tool's .vt-title; the 'used in' link is the chapter
    whose markdown references {{tool:NAME}}."""
    tool_files = sorted(glob.glob(os.path.join(_TOOLS_DIR, "*.html")))
    if not tool_files:
        return 0
    # map tool name -> (chapter url, chapter title)
    marker_re = re.compile(r"\{\{tool:([a-z0-9\-]+)\}\}")
    used_in = {}
    for c in book.flat_chapters:
        if os.path.exists(c["md_path"]):
            for name in marker_re.findall(open(c["md_path"]).read()):
                used_in[name] = (c["url"], c["title"])
    blocks = []
    for tf in tool_files:
        name = os.path.basename(tf)[:-5]
        src = open(tf).read()
        m = re.search(r'class="vt-title">([^<]+)<', src)
        title = m.group(1) if m else name
        link = ""
        if name in used_in:
            url, ctitle = used_in[name]
            link = f'<a class="tool-chapter-link" href="../{url}">↳ in “{html.escape(ctitle)}”</a>'
        blocks.append(f'<section class="tool-hub-item"><div class="tool-hub-head">'
                      f'<h2 id="{name}">{html.escape(title)}</h2>{link}</div>\n{src}\n</section>')
    body = ('<div class="chapter-eyebrow">Interactive tools</div>'
            '<h1>Tools &amp; calculators</h1>'
            '<p class="tool-hub-intro">Live calculators and visualizers from across the book — '
            'FLOP/memory/cost budgets, the Chinchilla-optimal split, KV-cache sizing, and more. '
            'Each also appears inline in the chapter that teaches it.</p>\n' + "\n".join(blocks))
    page = PAGE_TMPL.format(
        title="Tools &amp; calculators", brand=html.escape(book.m["title"]),
        brand_sub=html.escape(book.m.get("subtitle", "")), desc="Interactive LLM calculators and visualizers.",
        base="../", home="../index.html", search_name="search-index.json",
        canonical=SITE_BASE + "tools/", og_image=OG_IMAGE, repo_url=REPO_URL,
        sidebar=render_sidebar(book, "index.html"),
        breadcrumb="", chapter_meta="", body=body, page_nav="",
        toc_side="",
    )
    os.makedirs(os.path.join(SITE, "tools"), exist_ok=True)
    open(os.path.join(SITE, "tools", "index.html"), "w").write(page)
    return len(tool_files)


_CONCEPT_DIR = os.path.join(ROOT, "concept")


def _load_concept(name):
    p = os.path.join(_CONCEPT_DIR, name)
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def _diff_badge(d):
    cls = {"intro": "good", "core": "accent", "advanced": "warn"}.get(d, "accent")
    return f'<span class="lm-diff lm-{cls}">{html.escape(d)}</span>'


def write_learning_map(book):
    """Render /map: guided reading tracks, a browse-by-topic index, and the per-chapter
    prerequisite map — all built from concept/{graph,tags,tracks}.json."""
    graph = _load_concept("graph.json")
    if not graph:
        return 0
    tags = _load_concept("tags.json") or {}
    tracks = _load_concept("tracks.json") or []
    by_id = {g["id"]: g for g in graph}

    def chip(url, label):
        return f'<a class="lm-chip" href="../{url}">{html.escape(label)}</a>'

    css = """
<style>
#lm{max-width:920px}
#lm h2{margin:2.2rem 0 .6rem;font-size:1.3rem}
#lm .lm-intro{color:var(--ink-soft,#556)}
.lm-track{border:1px solid var(--border-2,#e5e7eb);border-radius:12px;padding:1rem 1.15rem;margin:.9rem 0;background:var(--surface,#fff)}
.lm-track h3{margin:.1rem 0 .3rem;font-size:1.1rem}
.lm-aud{display:inline-block;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--accent,#4f46e5);font-weight:700;margin-bottom:.3rem}
.lm-track ol{margin:.5rem 0 0;padding-left:1.2rem}
.lm-track ol li{margin:.16rem 0}
.lm-chip{display:inline-block;font-size:.8rem;padding:.12rem .5rem;margin:.12rem .2rem .12rem 0;border:1px solid var(--border-2,#e5e7eb);border-radius:999px;text-decoration:none;color:var(--ink-soft,#556)}
.lm-chip:hover{border-color:var(--accent,#4f46e5);color:var(--accent,#4f46e5)}
.lm-diff{display:inline-block;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.03em;padding:.05rem .4rem;border-radius:4px;margin-left:.4rem;vertical-align:middle}
.lm-good{background:color-mix(in srgb,var(--good,#2f9e6e) 16%,transparent);color:var(--good,#2f9e6e)}
.lm-accent{background:color-mix(in srgb,var(--accent,#4f46e5) 14%,transparent);color:var(--accent,#4f46e5)}
.lm-warn{background:color-mix(in srgb,var(--warn,#e0a106) 20%,transparent);color:var(--warn,#b8860b)}
.lm-row{padding:.55rem 0;border-bottom:1px solid var(--border-2,#eee)}
.lm-row .lm-h{font-weight:600}
.lm-row .lm-ol{color:var(--ink-soft,#556);font-size:.92rem;margin:.15rem 0}
.lm-row .lm-pre{font-size:.82rem;color:var(--muted,#889)}
.lm-part{margin-top:1.6rem;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted,#889);font-weight:700}
.lm-tagbar a{margin:.15rem .3rem .15rem 0}
</style>
"""
    parts = ['<div class="chapter-eyebrow">Navigate the book</div><h1>Learning map</h1>',
             '<p class="lm-intro">Guided reading tracks for specific goals, a topic index, and the '
             'prerequisite map for every chapter — so you can chart a path instead of reading front to back.</p>',
             css, '<div id="lm">']

    # --- reading tracks ---
    if tracks:
        parts.append('<h2>Guided reading tracks</h2>')
        for t in tracks:
            items = []
            for cid in t.get("chapters", []):
                g = by_id.get(cid)
                if g:
                    items.append(f'<li><a href="../{g["url"]}">{html.escape((g["num"]+" ") if g["num"] else "")}'
                                 f'{html.escape(g["title"])}</a></li>')
            parts.append(
                f'<div class="lm-track"><span class="lm-aud">{html.escape(t.get("audience",""))}</span>'
                f'<h3>{html.escape(t.get("title",""))}</h3>'
                f'<p class="lm-ol">{html.escape(t.get("blurb",""))}</p>'
                f'<ol>{"".join(items)}</ol></div>')

    # --- browse by topic ---
    if tags:
        parts.append('<h2>Browse by topic</h2><div class="lm-tagbar">')
        for tag in sorted(tags):
            chs = tags[tag]
            links = " ".join(chip(c["url"], (c["num"] + " " if c["num"] else "") + c["title"]) for c in chs)
            parts.append(f'<div class="lm-row"><div class="lm-h">{html.escape(tag)} '
                         f'<span class="lm-pre">({len(chs)})</span></div>{links}</div>')
        parts.append('</div>')

    # --- prerequisite map, grouped by part ---
    parts.append('<h2>Prerequisite map</h2>'
                 '<p class="lm-ol">Each chapter with what it lets you do and the concepts to know first '
                 '(linked to where they are taught).</p>')
    cur_part = None
    for g in graph:
        if g["part"] != cur_part:
            cur_part = g["part"]
            parts.append(f'<div class="lm-part">{html.escape(cur_part)}</div>')
        pre = " ".join(
            (f'<a class="lm-chip" href="../{p["url"]}">{html.escape(p["text"])}</a>' if p.get("url")
             else f'<span class="lm-chip">{html.escape(p["text"])}</span>')
            for p in g.get("prereqs", []))
        parts.append(
            f'<div class="lm-row"><div class="lm-h"><a href="../{g["url"]}">'
            f'{html.escape((g["num"]+" ") if g["num"] else "")}{html.escape(g["title"])}</a>'
            f'{_diff_badge(g["difficulty"])}</div>'
            f'<div class="lm-ol">{html.escape(g["one_liner"])}</div>'
            + (f'<div class="lm-pre">Prereqs: {pre}</div>' if pre else "") + '</div>')
    parts.append('</div>')

    body = "\n".join(parts)
    page = PAGE_TMPL.format(
        title="Learning map", brand=html.escape(book.m["title"]),
        brand_sub=html.escape(book.m.get("subtitle", "")),
        desc="Guided reading tracks, topic index, and the prerequisite map for the whole book.",
        base="../", home="../index.html", search_name="search-index.json",
        canonical=SITE_BASE + "map/", og_image=OG_IMAGE, repo_url=REPO_URL,
        sidebar=render_sidebar(book, "index.html"),
        breadcrumb="", chapter_meta="", body=body, page_nav="", toc_side="")
    os.makedirs(os.path.join(SITE, "map"), exist_ok=True)
    open(os.path.join(SITE, "map", "index.html"), "w").write(page)
    return len(graph)


def write_glossary(book):
    """Render /glossary: a deduped, filterable A-Z glossary linked to the chapter that defines
    each term. Built from concept/glossary.json."""
    gloss = _load_concept("glossary.json")
    if not gloss:
        return 0
    # group by first letter
    groups = {}
    for g in gloss:
        c = g["term"][:1].upper()
        c = c if c.isalpha() else "#"
        groups.setdefault(c, []).append(g)
    letters = sorted(groups)

    css = """
<style>
#gl{max-width:820px}
#gl .gl-intro{color:var(--ink-soft,#556)}
#gl-filter{width:100%;padding:.6rem .8rem;font-size:1rem;border:1px solid var(--border-2,#e5e7eb);
  border-radius:10px;margin:.8rem 0 .2rem;background:var(--surface,#fff);color:inherit}
.gl-az{position:sticky;top:0;background:var(--surface,#fff);padding:.4rem 0;border-bottom:1px solid var(--border-2,#eee);
  font-size:.85rem;z-index:2}
.gl-az a{margin-right:.45rem;text-decoration:none;color:var(--accent,#4f46e5);font-weight:600}
.gl-letter{font-size:1.4rem;margin:1.4rem 0 .3rem;color:var(--muted,#889)}
.gl-item{padding:.5rem 0;border-bottom:1px solid var(--border-2,#f0f0f2)}
.gl-term{font-weight:700}
.gl-def{color:var(--ink-soft,#556);margin:.12rem 0}
.gl-src{font-size:.82rem;color:var(--muted,#889)}
.gl-src a{color:var(--accent,#4f46e5);text-decoration:none}
.gl-none{color:var(--muted,#889);padding:1rem 0;display:none}
</style>
"""
    parts = ['<div class="chapter-eyebrow">Reference</div><h1>Glossary</h1>',
             f'<p class="gl-intro">{len(gloss):,} terms from across the book, each linked to the chapter that '
             'defines it. Type to filter.</p>', css,
             '<input id="gl-filter" type="search" placeholder="Filter terms… (e.g. KV cache, LoRA, MFU)" '
             'aria-label="Filter glossary terms" autocomplete="off">',
             '<div id="gl">']
    az = " ".join(f'<a href="#gl-{l}">{l}</a>' for l in letters)
    parts.append(f'<div class="gl-az">{az}</div>')
    for l in letters:
        parts.append(f'<h2 class="gl-letter" id="gl-{l}">{l}</h2>')
        for g in sorted(groups[l], key=lambda x: x["key"]):
            src = (f'<a href="../{g["home_url"]}">{html.escape((g["home_num"]+" ") if g["home_num"] else "")}'
                   f'{html.escape(g["home_title"])}</a>')
            extra = f' · used in {g["used_count"]} chapters' if g["used_count"] > 1 else ""
            parts.append(
                f'<div class="gl-item"><div class="gl-term">{html.escape(g["term"])}</div>'
                f'<div class="gl-def">{html.escape(g["definition"])}</div>'
                f'<div class="gl-src">Defined in {src}{extra}</div></div>')
    parts.append('<div class="gl-none">No terms match your filter.</div></div>')
    parts.append("""
<script>
(function(){
  var f=document.getElementById('gl-filter'), items=document.querySelectorAll('#gl .gl-item'),
      letters=document.querySelectorAll('#gl .gl-letter'), az=document.querySelector('#gl .gl-az'),
      none=document.querySelector('#gl .gl-none');
  function apply(){
    var q=f.value.trim().toLowerCase(), shown=0;
    items.forEach(function(it){
      var m=it.textContent.toLowerCase().indexOf(q)>=0; it.style.display=m?'':'none'; if(m)shown++;
    });
    letters.forEach(function(h){
      var n=h.nextElementSibling, any=false;
      while(n && !n.classList.contains('gl-letter')){ if(n.classList.contains('gl-item')&&n.style.display!=='none')any=true; n=n.nextElementSibling; }
      h.style.display=any?'':'none';
    });
    if(az)az.style.display=q?'none':'';
    none.style.display=shown?'none':'block';
  }
  f.addEventListener('input',apply);
})();
</script>
""")
    body = "\n".join(parts)
    page = PAGE_TMPL.format(
        title="Glossary", brand=html.escape(book.m["title"]),
        brand_sub=html.escape(book.m.get("subtitle", "")),
        desc="A filterable glossary of LLM-stack terms, each linked to the chapter that defines it.",
        base="../", home="../index.html", search_name="search-index.json",
        canonical=SITE_BASE + "glossary/", og_image=OG_IMAGE, repo_url=REPO_URL,
        sidebar=render_sidebar(book, "index.html"),
        breadcrumb="", chapter_meta="", body=body, page_nav="", toc_side="")
    os.makedirs(os.path.join(SITE, "glossary"), exist_ok=True)
    open(os.path.join(SITE, "glossary", "index.html"), "w").write(page)
    return len(gloss)


def write_print_edition(book):
    """One self-contained /print.html with every chapter in order (cover + auto TOC),
    cross-links rewritten to in-page anchors, print-optimized CSS, KaTeX for math.
    The reader opens it and uses Print -> Save as PDF (renders figures/math faithfully);
    scripts/make_pdf.sh automates that with headless Chrome where the sandbox allows."""
    try:
        style_css = open(os.path.join(SITE, "assets", "style.css")).read()
        pyg_css = open(os.path.join(SITE, "assets", "pygments.css")).read()
    except OSError:
        return 0

    def anchor(url):
        return url[:-5].replace("/", "__") if url.endswith(".html") else url.replace("/", "__")

    link_re = re.compile(r'href="(?:\.\./)?([0-9a-z][0-9a-z-]+)/([0-9a-z][0-9a-z-]+)\.html((?:#[^"]*)?)"')
    sections, toc, cur_part = [], [], None
    for c in book.flat_chapters:
        md = markdown.Markdown(extensions=MD_EXTENSIONS, extension_configs=MD_CONFIG)
        if os.path.exists(c["md_path"]):
            raw = open(c["md_path"]).read()
        else:
            continue
        raw = expand_figures(raw)
        raw = expand_tools(raw)
        body = md.convert(raw)
        # cross-chapter links -> in-page anchors
        body = link_re.sub(lambda m: f'href="#{m.group(1)}__{m.group(2)}"', body)
        a = anchor(c["url"])
        num = "" if c.get("is_front") else f'{c["part_no"]}.{c["chap_no"]} '
        if c.get("part_title") != cur_part:
            cur_part = c.get("part_title")
            toc.append(f'<li class="pe-toc-part">{html.escape(cur_part or "")}</li>')
        sections.append(f'<section class="pe-chapter" id="{a}">\n{body}\n</section>')
        toc.append(f'<li class="pe-toc-ch"><a href="#{a}">{html.escape(num + c["title"])}</a></li>')

    m = book.m
    cover = (f'<div class="pe-cover">'
             f'<div class="pe-cover-eyebrow">The complete book &middot; print / PDF edition</div>'
             f'<h1 class="pe-cover-title">{html.escape(m["title"])}</h1>'
             f'<div class="pe-cover-sub">{html.escape(m.get("subtitle",""))}</div>'
             f'<p class="pe-cover-tag">{html.escape(m.get("tagline",""))}</p>'
             f'<div class="pe-cover-meta">{len(sections)} chapters &middot; generated {BUILD_DATE} &middot; '
             f'<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a><br>'
             f'{html.escape(SITE_BASE)}</div>'
             f'<p class="pe-cover-hint">Tip: use your browser&rsquo;s <b>Print &rarr; Save as PDF</b> '
             f'(A4/Letter, background graphics on). Interactive widgets print their initial state.</p>'
             f'</div>')
    toc_html = f'<nav class="pe-toc"><h2>Contents</h2><ol>{"".join(toc)}</ol></nav>'
    print_css = """
:root{color-scheme:light}
html[data-theme]{}
body.pe-body{max-width:52rem;margin:0 auto;padding:2rem 1.4rem;background:#fff;color:#222}
.pe-cover{min-height:80vh;display:flex;flex-direction:column;justify-content:center;text-align:center;break-after:page}
.pe-cover-title{font-size:2.6rem;line-height:1.1;margin:.4rem 0}
.pe-cover-sub{font-size:1.3rem;color:#555}
.pe-cover-tag{color:#666;max-width:34rem;margin:1rem auto}
.pe-cover-eyebrow{letter-spacing:.08em;text-transform:uppercase;font-size:.75rem;color:#a03d1f;font-weight:700}
.pe-cover-meta{margin-top:1.4rem;color:#777;font-size:.9rem}
.pe-cover-hint{margin-top:2rem;color:#888;font-size:.85rem}
.pe-toc{break-after:page}
.pe-toc ol{list-style:none;padding-left:0;columns:1}
.pe-toc-part{font-weight:700;margin:1rem 0 .3rem;color:#a03d1f}
.pe-toc-ch a{text-decoration:none;color:#334;font-size:.95rem}
.pe-chapter{break-before:page}
.pe-chapter h1{font-size:1.9rem;border-bottom:2px solid #eee;padding-bottom:.3rem;margin-top:0}
.viz-replay,.viz-tool button,.vt-in button{display:none!important}
.viz svg,.viz-tool svg{max-width:100%;height:auto}
pre{white-space:pre-wrap;word-break:break-word;background:#f6f7f9;border:1px solid #eceef3;border-radius:6px;padding:.7rem;font-size:.8rem}
code{word-break:break-word}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{border:1px solid #ddd;padding:.3rem .5rem}
.admonition{border:1px solid #e5e5e5;border-left:4px solid #a03d1f;border-radius:5px;padding:.5rem .8rem;margin:1rem 0;font-size:.92rem}
.admonition-title{font-weight:700}
img{max-width:100%}
a{color:#a03d1f}
@media print{
  body.pe-body{max-width:none;padding:0}
  .pe-cover-hint,.pe-toc a{color:#000}
  a{color:#000;text-decoration:none}
  @page{margin:1.6cm 1.4cm}
}
"""
    doc = f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(m["title"])} — Print / PDF edition</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<style>{style_css}</style>
<style>{pyg_css}</style>
<style>{print_css}</style>
</head>
<body class="pe-body markdown-body">
{cover}
{toc_html}
{"".join(sections)}
<script>window.MathJax=null;</script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}},{{left:'$$',right:'$$',display:true}}],throwOnError:false}});"></script>
</body>
</html>
"""
    open(os.path.join(SITE, "print.html"), "w").write(doc)
    return len(sections)


def write_sitemap(entries):
    """entries: list of (path_relative_to_SITE_BASE, lastmod_date). Emits sitemap.xml + robots.txt."""
    items = []
    for path, date in entries:
        loc = html.escape(SITE_BASE + path)
        items.append(f"  <url><loc>{loc}</loc><lastmod>{date}</lastmod></url>")
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           + "\n".join(items) + "\n</urlset>\n")
    with open(os.path.join(SITE, "sitemap.xml"), "w") as f:
        f.write(xml)
    with open(os.path.join(SITE, "robots.txt"), "w") as f:
        f.write("User-agent: *\nAllow: /\n\nSitemap: " + SITE_BASE + "sitemap.xml\n")


def _roman(n):
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def main():
    if os.path.isdir(SITE):
        shutil.rmtree(SITE)
    os.makedirs(SITE)
    shutil.copytree(ASSETS_SRC, os.path.join(SITE, "assets"))
    # pygments css (light + dark scoped)
    fmt_light = HtmlFormatter(style="default")
    fmt_dark = HtmlFormatter(style="github-dark")
    with open(os.path.join(SITE, "assets", "pygments.css"), "w") as f:
        f.write('html[data-theme="light"] ' + fmt_light.get_style_defs(".highlight"))
        f.write("\n")
        f.write('html[data-theme="dark"] ' + fmt_dark.get_style_defs(".highlight"))

    book = Collection(load_manifest("book.json"), out_subdir="", flat=False,
                      search_name="search-index.json")

    # Interview companion: flat under site/interview/. Sibling links written in the markdown as
    # ../13-interview-prep/<f>.html must collapse to <f>.html; links to book parts (../<part>/..)
    # already resolve correctly from depth-1 interview pages, so leave them.
    iv_rewrites = [(re.compile(r'(?:\.\./)?13-interview-prep/'), "")]
    interview = Collection(load_manifest("interview.json"), out_subdir="interview", flat=True,
                           link_rewrites=iv_rewrites, search_name="interview-index.json")

    totals = {}
    for coll in (book, interview):
        tw, wr, ph = build_collection(coll)
        totals[coll.m["title"]] = (tw, wr, ph, len(coll.flat_chapters))
        print(f"Built '{coll.m['title']}' -> {coll.out_root()}")
        print(f"  chapters: {wr} written, {ph} placeholders | ~{tw//300:,} pages ({tw:,} words)")

    nt = write_tools_hub(book)
    if nt:
        print(f"  tools hub: {nt} interactive tools -> site/tools/")

    n_map = write_learning_map(book)
    if n_map:
        print(f"  learning map: {n_map} chapters (tracks + prereqs) -> site/map/")
    n_gloss = write_glossary(book)
    if n_gloss:
        print(f"  glossary: {n_gloss} terms -> site/glossary/")
    n_print = write_print_edition(book)
    if n_print:
        print(f"  print edition: {n_print} chapters -> site/print.html")

    # sitemap.xml + robots.txt (git_last_date is cached from the render pass, so this is cheap)
    entries = [("", BUILD_DATE)]
    if nt:
        entries.append(("tools/", BUILD_DATE))
    if n_map:
        entries.append(("map/", BUILD_DATE))
    if n_gloss:
        entries.append(("glossary/", BUILD_DATE))
    if n_print:
        entries.append(("print.html", BUILD_DATE))
    entries += [(c["url"], git_last_date(c["md_path"])) for c in book.flat_chapters]
    entries.append(("interview/", BUILD_DATE))
    entries += [("interview/" + c["url"], git_last_date(c["md_path"])) for c in interview.flat_chapters]
    write_sitemap(entries)
    print(f"  sitemap.xml: {len(entries)} URLs | robots.txt written")

    if _FIG_MISSING:
        print("  ⚠ missing figures: " + ", ".join(sorted(_FIG_MISSING)))


if __name__ == "__main__":
    main()
