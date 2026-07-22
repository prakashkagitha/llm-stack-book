# The LLM Stack — From Silicon to Agents

[![Deploy](https://github.com/prakashkagitha/llm-stack-book/actions/workflows/deploy.yml/badge.svg)](https://github.com/prakashkagitha/llm-stack-book/actions/workflows/deploy.yml)
[![Code tests](https://github.com/prakashkagitha/llm-stack-book/actions/workflows/test.yml/badge.svg)](https://github.com/prakashkagitha/llm-stack-book/actions/workflows/test.yml)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A book-length, ground-up treatment of the entire large language model stack: from GPU silicon and numerics, through the transformer and its training, to inference serving, agents, retrieval, evaluation, and safety.

**📖 Read it online: https://prakashkagitha.github.io/llm-stack-book/**

- **134 chapters** across 14 parts (foundations → transformer → pretraining → kernels → post-training → RL infra → inference → agents → RAG → multimodal → evaluation → production → interpretability/safety → appendix)
- **690 hand-built SVG figures and animations**, theme-aware and reduced-motion safe
- **CI-tested code** — every runnable code block is assembled into `tests/` and executed on CPU in CI ([`test.yml`](.github/workflows/test.yml)); ~45 real bugs were caught and fixed this way
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

## Contributing

Corrections and improvements are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Report errors via the **Errata** issue template.

## License & citation

Licensed under [**CC BY 4.0**](LICENSE) — share and adapt freely, even commercially, with attribution.

```bibtex
@book{kagitha_llm_stack_2026,
  title  = {The LLM Stack: From Silicon to Agents},
  author = {Kagitha, Prakash},
  year   = {2026},
  url    = {https://prakashkagitha.github.io/llm-stack-book/},
  note   = {Open web textbook, CC BY 4.0}
}
```
