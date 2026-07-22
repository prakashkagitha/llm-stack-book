#!/usr/bin/env python3
"""Run a test file under a simulated CI environment: only the packages CI actually
installs (stdlib + numpy/torch/einops/sklearn and sklearn's own deps) are importable;
every other top-level import raises ImportError. This catches non-hermetic tests
(secretly needing sentence_transformers, trl, transformers, matplotlib, ...) locally,
without uninstalling anything.

Usage:  python3 scripts/ci_sim_run.py tests/<file>.py
"""
import sys, builtins

# Packages CI does NOT install (test.yml installs only numpy/torch/einops/scikit-learn/pytest,
# whose transitive deps like scipy/joblib ARE present). A test that imports any of these
# un-guarded would ImportError in CI — so block them here to catch it. Blocklist, not allowlist,
# so we don't fight torch/sklearn's own optional internal imports (dill, narwhals, ...).
BLOCK = {
    "sentence_transformers", "mteb", "transformers", "datasets", "tokenizers", "accelerate",
    "peft", "trl", "deepspeed", "bitsandbytes", "vllm", "tensorrt_llm", "flash_attn", "xformers",
    "triton", "dspy", "openai", "anthropic", "cohere", "tiktoken", "litellm", "google",
    "qdrant_client", "faiss", "chromadb", "pinecone", "weaviate", "langchain", "llama_index",
    "wandb", "mlflow", "ray", "jax", "tensorflow", "keras",
    "matplotlib", "pandas", "seaborn", "plotly", "polars",
    "requests", "httpx", "aiohttp", "boto3", "redis",
    "PIL", "cv2", "librosa", "soundfile", "gradio", "streamlit", "fastapi", "flask", "uvicorn",
}

_orig_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Only gate ABSOLUTE top-level imports (level==0); relative imports pass through.
    if level == 0 and name:
        top = name.split(".")[0]
        if top in BLOCK:
            raise ImportError(f"CI-sim: module '{top}' is not available in the CI test environment")
    return _orig_import(name, globals, locals, fromlist, level)


def main():
    if len(sys.argv) != 2:
        print("usage: ci_sim_run.py <test.py>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    builtins.__import__ = _guarded_import
    src = open(path).read()
    g = {"__name__": "__main__", "__file__": path}
    exec(compile(src, path, "exec"), g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
