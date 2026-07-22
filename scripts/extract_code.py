#!/usr/bin/env python3
"""Extract every fenced code block from content/ and heuristically classify it, to
scope the CI-tested-code effort. Deterministic (no LLM). Writes code/inventory.json.

Tiers (heuristic — an LLM pass refines the borderline ones later):
  shell        - bash/sh/console block
  non-python   - other languages / prose in fences
  fragment     - python that is NOT standalone (leading indent, `...` ellipsis, bare snippet)
  needs-gpu    - python referencing CUDA/distributed/Triton/flash-attn
  needs-net    - python that downloads models/data (from_pretrained, load_dataset, requests, hf)
  ci-testable  - python that looks CPU-only, self-contained, and safe to execute in CI
"""
import json, os, re, glob, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FENCE = re.compile(r"^([ \t]*)```+([^\n`]*)\n(.*?)^\1```+\s*$", re.S | re.M)

GPU = re.compile(r"\bcuda\b|\.to\(\s*['\"]?cuda|device_map|nccl|torch\.distributed|"
                 r"\bdist\.|triton|flash_attn|FlashAttention|\.cuda\(\)|bitsandbytes|"
                 r"tensor_parallel|init_process_group|FSDP|DDP\(", re.I)
NET = re.compile(r"from_pretrained|load_dataset|AutoModel|AutoTokenizer|requests\.|urllib|"
                 r"wget|hf_hub|huggingface_hub|snapshot_download|openai\.|anthropic\.", re.I)
# python that clearly isn't a standalone program
FRAG_SIG = re.compile(r"^\s*(\.\.\.|#\s*\.\.\.|@|def\s|class\s|else:|elif|except|return\b|yield\b)")


def classify(lang, code):
    l = (lang or "").strip().lower().split()[0] if lang.strip() else ""
    if l in ("bash", "sh", "shell", "console", "zsh"):
        return "shell", l
    if l not in ("python", "py", "python3", ""):
        return "non-python", l
    # heuristic python detection when unlabeled
    looks_py = bool(re.search(r"\b(import|def|class|print|torch|np\.|numpy)\b", code))
    if l == "" and not looks_py:
        return "non-python", l
    if GPU.search(code):
        return "needs-gpu", "python"
    if NET.search(code):
        return "needs-net", "python"
    first = next((ln for ln in code.splitlines() if ln.strip()), "")
    # standalone-ish: has an import or a top-level def/assignment and no stray leading indent
    if first.startswith((" ", "\t")) or "..." in code or FRAG_SIG.match(first):
        # a leading-indent / ellipsis / bare-continuation block is usually an excerpt
        if not re.search(r"^\s*(import|from)\s", code, re.M):
            return "fragment", "python"
    return "ci-testable", "python"


def main():
    blocks = []
    for md in sorted(glob.glob(os.path.join(ROOT, "content", "*", "*.md"))):
        cid = os.path.relpath(md, os.path.join(ROOT, "content"))[:-3]
        text = open(md).read()
        for i, m in enumerate(FENCE.finditer(text)):
            lang, code = m.group(2), m.group(3)
            tier, l = classify(lang, code)
            blocks.append({
                "chapter": cid, "idx": i, "lang": l or (lang.strip() or "?"),
                "tier": tier, "n_lines": code.count("\n") + 1,
                "line": text[:m.start()].count("\n") + 1,
            })
    os.makedirs(os.path.join(ROOT, "code"), exist_ok=True)
    json.dump(blocks, open(os.path.join(ROOT, "code", "inventory.json"), "w"), indent=1)

    by_tier = collections.Counter(b["tier"] for b in blocks)
    ci = [b for b in blocks if b["tier"] == "ci-testable"]
    print(f"total code blocks: {len(blocks)}")
    for t, n in by_tier.most_common():
        print(f"  {t:12} {n}")
    print(f"\nci-testable blocks span {len(set(b['chapter'] for b in ci))} chapters")
    print(f"ci-testable total lines: {sum(b['n_lines'] for b in ci)}")


if __name__ == "__main__":
    main()
