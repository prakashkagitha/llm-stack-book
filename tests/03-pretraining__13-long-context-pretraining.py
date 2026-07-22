"""
Executes the CPU-runnable Python blocks from
content/03-pretraining/13-long-context-pretraining.md

Blocks tested (in chapter order):
    - block #0 (line ~95):  build_rope_freqs / build_yarn_freqs (YaRN frequency scaling)
    - block #1 (line ~218): build_packed_attention_mask (document-packing attention mask)
    - block #3 (line ~302): build_haystack_with_needle / evaluate_niah (NIAH harness)
    - block #7 (line ~574): plot_niah_heatmap (matplotlib visualisation of NIAH results)

Skipped blocks (named per the chapter's own code-fence order):
    - block #2 (flash_attn_varlen_func example): requires a CUDA device and the
      `flash_attn` package (GPU-only kernel). SKIP(needs-gpu).
    - block #4 (```text``` ring-of-4-GPUs diagram): not Python, prose only. SKIP(non-python).
    - block #5 (ring_attention_forward): requires torch.distributed process group /
      multiple ranks (dist.send/recv) and is meant to run under `torchrun
      --nproc_per_node=4`; not meaningfully CPU-runnable single-process. SKIP(needs-gpu/needs-dist).
    - block #6 (torchrun training recipe): shell script, not Python. SKIP(shell).
"""

import os
import random
import shutil
import tempfile

import torch

# matplotlib is NOT in the guaranteed-CI import list (numpy, torch, einops,
# sklearn, stdlib) so it must be import-guarded at module scope.
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend, no display needed
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    np = None
    HAS_MATPLOTLIB = False


# ============================================================================
# Block #0 (line ~95): YaRN / RoPE frequency scaling
# ============================================================================
import math


def build_rope_freqs(dim: int, base: float, seq_len: int) -> torch.Tensor:
    """Build standard RoPE inverse frequencies for a given dimension and base."""
    # Shape: (dim // 2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # Outer product with positions → (seq_len, dim // 2)
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)      # shape: (seq_len, dim // 2)
    return freqs  # each row is the angles for one position


def build_yarn_freqs(
    dim: int,
    base: float,
    orig_max_seq_len: int,
    target_max_seq_len: int,
    beta_fast: float = 32.0,   # high-freq boundary (wavelengths)
    beta_slow: float = 1.0,    # low-freq boundary
) -> torch.Tensor:
    """
    YaRN frequency computation.
    Returns per-dimension scaling factor to apply to the standard freqs.
    """
    scale = target_max_seq_len / orig_max_seq_len

    # Standard inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # Wavelength for each dimension: λ = 2π / θ_i
    wavelengths = 2 * math.pi / inv_freq

    # Ramp function: 0 for high-freq dims (leave unchanged), 1 for low-freq (full scale)
    ramp = (wavelengths - orig_max_seq_len / beta_fast) / (
        orig_max_seq_len / beta_slow - orig_max_seq_len / beta_fast
    )
    ramp = ramp.clamp(0.0, 1.0)

    # Blend: high-freq → no interpolation, low-freq → PI scaling
    # NTK base for fully-scaled dims: b' = b * scale^(dim/(dim-2))
    ntk_base = base * (scale ** (dim / (dim - 2)))
    ntk_inv_freq = 1.0 / (ntk_base ** (torch.arange(0, dim, 2).float() / dim))

    # Interpolate between unscaled (ramp=0) and NTK-scaled (ramp=1) per dimension
    blended_inv_freq = (1 - ramp) * inv_freq + ramp * ntk_inv_freq

    t = torch.arange(target_max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, blended_inv_freq)
    return freqs  # (target_max_seq_len, dim // 2)


# --- demo ---
# GPT-style model: dim=128, base=10000, trained at 4096, extending to 32768
orig_freqs  = build_rope_freqs(128, 10000, 4096)
yarn_freqs  = build_yarn_freqs(128, 10000, 4096, 32768)

print(f"Original freq range at pos 4096: [{orig_freqs[-1].min():.4f}, {orig_freqs[-1].max():.4f}]")
print(f"YaRN freq range at pos 32768:    [{yarn_freqs[-1].min():.4f}, {yarn_freqs[-1].max():.4f}]")
# YaRN keeps high-freq dims in known territory; low-freq dims gently scaled.

assert orig_freqs.shape == (4096, 64)
assert yarn_freqs.shape == (32768, 64)
assert torch.isfinite(yarn_freqs).all()


# ============================================================================
# Block #1 (line ~218): Document-packing attention mask
# ============================================================================
def build_packed_attention_mask(
    doc_lengths: list,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a boolean causal attention mask for a packed sequence of multiple docs.

    doc_lengths: list of token counts per document, must sum to seq_len.
    Returns: (seq_len, seq_len) bool tensor where True = can attend.
    """
    seq_len = sum(doc_lengths)
    # Start with no attention allowed
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    offset = 0
    for length in doc_lengths:
        # Within this document's block, allow lower-triangular (causal) attention
        block_end = offset + length
        for row in range(offset, block_end):
            # Token at 'row' can attend to [offset .. row] (same doc, causal)
            mask[row, offset:row + 1] = True
        offset = block_end

    return mask  # (seq_len, seq_len)


# Example: two docs packed into a single context of length 8 (4 + 4)
mask = build_packed_attention_mask([4, 4])
print(mask.int())
# tensor([[1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 1, 0, 0],
#         [0, 0, 0, 0, 1, 1, 1, 0],
#         [0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)

expected = torch.tensor([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
assert torch.equal(mask.int(), expected)


# ============================================================================
# Block #3 (line ~302): Needle-in-a-haystack harness
# ============================================================================
def build_haystack_with_needle(
    haystack_text: str,
    needle: str,
    context_length: int,           # target total tokens (approx by words)
    depth_fraction: float,         # 0.0 = beginning, 1.0 = end
    tokenizer_approx_ratio: float = 1.3,  # tokens per word (rough)
):
    """
    Insert a needle sentence at a given depth in the haystack.
    Returns (full_text, needle_start_word_index).
    """
    target_words = int(context_length / tokenizer_approx_ratio)

    # Tile haystack words to fill the context
    words = haystack_text.split()
    if len(words) < target_words:
        multiplier = (target_words // len(words)) + 1
        words = words * multiplier
    words = words[:target_words]

    # Determine insertion point
    insert_idx = int(len(words) * depth_fraction)

    # Insert needle words at that point
    needle_words = needle.split()
    words = words[:insert_idx] + needle_words + words[insert_idx:]

    return " ".join(words), insert_idx


def evaluate_niah(
    model_fn,           # callable(text: str) -> str
    needle: str,
    answer_keyword: str,
    context_lengths: list,
    depths: list,
    haystack: str,
) -> dict:
    """
    Run a full NIAH evaluation grid.
    Returns dict[(context_len, depth)] -> bool
    """
    results = {}
    for ctx_len in context_lengths:
        for depth in depths:
            text, _ = build_haystack_with_needle(haystack, needle, ctx_len, depth)
            prompt = text + "\n\nWhat is the magic phrase?"
            response = model_fn(prompt)
            results[(ctx_len, depth)] = answer_keyword.lower() in response.lower()
    return results


# --- demo / glue: exercise the NIAH harness fully offline (no network / no real model) ---
_toy_haystack = "the quick brown fox jumps over the lazy dog near the river bank " * 3
_needle = "The magic phrase is pineapple-strawberry-7."
_answer_keyword = "pineapple-strawberry-7"


def _mock_model_fn(prompt: str) -> str:
    """Stand-in for a real LLM call: a trivial local retriever over the prompt text.
    No network / API calls — this only exercises evaluate_niah's own plumbing."""
    if _answer_keyword in prompt:
        return f"The magic phrase is {_answer_keyword}."
    return "I could not find the magic phrase."


_context_lengths = [50, 100]
_depths = [0.0, 0.5, 1.0]

niah_results = evaluate_niah(
    model_fn=_mock_model_fn,
    needle=_needle,
    answer_keyword=_answer_keyword,
    context_lengths=_context_lengths,
    depths=_depths,
    haystack=_toy_haystack,
)

print("NIAH results:", niah_results)
assert len(niah_results) == len(_context_lengths) * len(_depths)
# The needle is always inserted, and the mock model always finds it verbatim in the prompt,
# so every cell of the grid should retrieve successfully.
assert all(niah_results.values())


# ============================================================================
# Block #7 (line ~574): NIAH heatmap visualisation (needs matplotlib)
# ============================================================================
if HAS_MATPLOTLIB:
    def plot_niah_heatmap(results: dict, context_lengths: list, depths: list):
        """
        Visualise needle-in-haystack results as a 2D heatmap.
        results: dict[(ctx_len, depth)] -> bool
        """
        grid = np.array([
            [float(results.get((c, d), False)) for c in context_lengths]
            for d in depths
        ])
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(context_lengths)))
        ax.set_xticklabels([f"{c//1000}K" for c in context_lengths], rotation=45)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{int(d*100)}%" for d in depths])
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Needle Depth")
        ax.set_title("Needle-in-Haystack Retrieval Accuracy")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig("niah_heatmap.png", dpi=150)
        print("Saved niah_heatmap.png")
        return grid.mean()  # return overall accuracy

    # Run in a scratch temp dir so we don't litter the repo with niah_heatmap.png
    # (the book's function hardcodes the output filename — code kept verbatim above).
    _tmpdir = tempfile.mkdtemp(prefix="niah_heatmap_")
    _cwd = os.getcwd()
    try:
        os.chdir(_tmpdir)
        accuracy = plot_niah_heatmap(niah_results, _context_lengths, _depths)
        print(f"Overall NIAH accuracy: {accuracy:.2f}")
        assert accuracy == 1.0
        assert os.path.exists("niah_heatmap.png")
    finally:
        os.chdir(_cwd)
        shutil.rmtree(_tmpdir, ignore_errors=True)
else:
    print("SKIP(matplotlib not installed): block #7 (plot_niah_heatmap) not exercised")


print("\nAll CPU-runnable blocks from 03-pretraining/13-long-context-pretraining.md executed successfully.")
