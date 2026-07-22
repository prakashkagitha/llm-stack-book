# Contributing to The LLM Stack

Thanks for helping make this a better reference for everyone building LLMs. The
book is a living resource — corrections and improvements are genuinely welcome.

## Reporting an error (errata)

Found a wrong formula, a bug in a code block, a broken link, or a claim that's out
of date? Please [open an issue](https://github.com/prakashkagitha/llm-stack-book/issues/new/choose)
using the **Errata** template. Include:

- the chapter (URL or `content/<part>/<chapter>.md` path),
- what's wrong, and
- the correction (with a source, if it's a factual claim).

Small, specific reports are the most useful — one issue per error.

## Suggesting content

Missing a topic, a figure that would build intuition, or an exercise? Open an issue
with the **Suggestion** template describing what and why.

## Making a change (pull requests)

1. Chapters live in `content/<part>/<chapter>.md` (Markdown).
2. Figures are standalone files in `figures/<name>.html`, embedded via a
   `{{fig:NAME}}` marker. Figure names must be lowercase kebab-case.
3. Build locally to preview:
   ```bash
   pip install -r requirements.txt
   python3 build.py
   cd site && python3 -m http.server 8000   # open http://localhost:8000
   ```
4. Keep code blocks runnable and correct; keep prose in the existing voice
   (see `STYLE.md`). Don't hand-edit anything under `site/` — it's generated.

## Ground rules

- **Accuracy first.** Cite primary sources for factual/quantitative claims.
- **Buildability.** Prefer complete, runnable code over pseudo-code.
- By contributing, you agree your contributions are licensed under **CC BY 4.0**,
  the same license as the book.
