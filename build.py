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
import json, os, re, html, shutil, subprocess, datetime
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
        chapter_meta = (f'<div class="chapter-meta"><span>{reading} min read</span>'
                        f'<span class="cm-dot">·</span>'
                        f'<span>Updated <time datetime="{updated}">{updated}</time></span></div>')
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
               f'<a class="btn-ghost" href="interview/index.html">Interview companion</a></p>')

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

    # sitemap.xml + robots.txt (git_last_date is cached from the render pass, so this is cheap)
    entries = [("", BUILD_DATE)]
    entries += [(c["url"], git_last_date(c["md_path"])) for c in book.flat_chapters]
    entries.append(("interview/", BUILD_DATE))
    entries += [("interview/" + c["url"], git_last_date(c["md_path"])) for c in interview.flat_chapters]
    write_sitemap(entries)
    print(f"  sitemap.xml: {len(entries)} URLs | robots.txt written")

    if _FIG_MISSING:
        print("  ⚠ missing figures: " + ", ".join(sorted(_FIG_MISSING)))


if __name__ == "__main__":
    main()
