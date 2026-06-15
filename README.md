# The LLM Stack — From Silicon to Agents

A book-length, ground-up treatment of the entire large language model stack: from GPU silicon and numerics, through the transformer and its training, to inference serving, agents, retrieval, evaluation, and safety.

**📖 Read it online: https://prakashkagitha.github.io/llm-stack-book/**

- **134 chapters** across 14 parts (foundations → transformer → pretraining → kernels → post-training → RL infra → inference → agents → RAG → multimodal → evaluation → production → interpretability/safety → appendix)
- **326 hand-built SVG figures and animations**, theme-aware and reduced-motion safe
- A companion **interview prep** track at [`/interview/`](https://prakashkagitha.github.io/llm-stack-book/interview/)

## Repository layout

| Path | What it is |
|------|------------|
| `content/` | The book source — Markdown, one folder per part |
| `figures/` | Standalone HTML/SVG figures, expanded into pages via `{{fig:NAME}}` markers |
| `assets/` | Stylesheets and static assets |
| `build.py` | Static-site builder (Markdown → HTML) |
| `book.json` / `interview.json` | Table of contents and metadata for the two collections |
| `scripts/` | Authoring, QA, and figure-generation tooling |
| `site/` | Build output (not tracked on `main`; deployed to the `gh-pages` branch) |

## Building locally

```bash
pip install -r requirements.txt
python3 build.py          # renders content/ + figures/ into site/
cd site && python3 -m http.server 8000   # then open http://localhost:8000
```

## Deployment

Pushing to `main` triggers [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml),
which installs the pinned dependencies, runs `build.py`, and publishes `site/` to
GitHub Pages. No manual deploy step is needed — just edit, commit, and push.

## License

© 2026 Prakash Kagitha. All rights reserved unless a license file is added.
