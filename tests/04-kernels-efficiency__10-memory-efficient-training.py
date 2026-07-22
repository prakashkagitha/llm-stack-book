"""
Runnable-code test for content/04-kernels-efficiency/10-memory-efficient-training.md

Blocks tested (CPU-runnable, per the harness's heuristic scan):
  - block #1 (line ~205): SelectiveCheckpointBlock (selective checkpointing)
  - block #2 (line ~254): deepspeed_offload_config.json comment stub (trivial)
  - block #5 (line ~374): LoRALinear + inject_lora (LoRA from scratch)
  - block #12 (line ~729): torch.compile after enabling checkpointing

Blocks explicitly SKIPPED (per the harness's classification), with reasons:
  - #0  (line ~110): TransformerBlock/CheckpointedModel + measure_peak_memory
                      -> calls .cuda() / torch.cuda.reset_peak_memory_stats(); needs GPU.
                      We DO reuse the pure class definitions (TransformerBlock,
                      CheckpointedModel) as glue for block #12, which needs the
                      CheckpointedModel name but does not touch CUDA.
  - #3  (line ~259): deepspeed_offload_config.json contents -> JSON, not Python.
  - #4  (line ~296): gradient accumulation loop -> calls x.cuda(), y.cuda(),
                      torch.autocast(device_type="cuda", ...); needs GPU + a real
                      dataloader/model/optimizer that the chapter never defines
                      in-page (non-standalone fragment). SKIP(needs-gpu).
  - #6  (line ~519): QLoRA with bitsandbytes/transformers/peft -> needs GPU,
                      needs network to download "meta-llama/Llama-2-7b-hf", and
                      needs optional third-party packages. SKIP(network,needs-gpu).
  - #7  (line ~565): memory_snapshot() -> torch.cuda.memory_allocated() etc.,
                      and uses undefined `model`/`x` from prior GPU context. SKIP(needs-gpu).
  - #8  (line ~601): torch.cuda.memory_stats() -> needs GPU. SKIP(needs-gpu).
  - #9  (line ~652): bitsandbytes Adam8bit -> needs GPU + optional bitsandbytes
                      package, and references an undefined `model`. SKIP(needs-gpu).
  - #10 (line ~669): GradScaler + autocast(device_type="cuda") -> needs GPU,
                      references undefined `model`, `batches`, `optimizer`,
                      `N_ACCUM`. SKIP(needs-gpu).
  - #11 (line ~709): pinned-memory GPU<->CPU copy -> torch.cuda.synchronize(),
                      references undefined `gpu_tensor`. SKIP(needs-gpu).
  - #13 (line ~744): "saved_input = input.detach()" snippet -> a non-standalone
                      fragment illustrating detach() semantics, not a runnable unit
                      (uses undefined `input`, has a "rerun from saved_input" comment
                      with no actual rerun code). SKIP(fragment).

No network calls and no optional third-party imports are required by the
blocks under test, so no import-guarding is needed here -- but we still
follow the "guard optional imports" rule defensively for completeness.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =======================================================================
# Glue: pure class definitions from block #0 (needs-gpu block), reused
# ONLY because block #12 references the name `CheckpointedModel`. None
# of the CUDA-touching code from block #0 (measure_peak_memory, the
# __main__ loop) is executed here.
# =======================================================================
class TransformerBlock(nn.Module):
    """A minimal causal transformer block (MHA + FFN + layer norms)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class CheckpointedModel(nn.Module):
    """Wraps a stack of transformer blocks and applies gradient checkpointing."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int,
                 use_checkpointing: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.use_checkpointing = use_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpointing and x.requires_grad:
                # checkpoint() replaces the saved activations with a
                # recomputation graph.  use_reentrant=False is recommended
                # in PyTorch >= 2.0 for compatibility with compiled graphs.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


# =======================================================================
# Block #1 (line ~205, verbatim): Selective checkpointing -- only
# checkpoint the attention sub-block, not the cheaper FFN.
# =======================================================================

# Selective checkpointing: only checkpoint the attention sub-block,
# not the cheaper FFN.  Saves ~60% of attention-related activation memory
# at a small recompute cost.

from torch.utils.checkpoint import checkpoint

class SelectiveCheckpointBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def _attn_only(self, x: torch.Tensor) -> torch.Tensor:
        """The sub-computation we want to recompute in backward."""
        normed = self.ln1(x)
        out, _ = self.attn(normed, normed, normed, need_weights=False)
        return x + out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recompute attention activations; store FFN activations normally.
        if x.requires_grad:
            x = checkpoint(self._attn_only, x, use_reentrant=False)
        else:
            x = self._attn_only(x)
        return x + self.ffn(self.ln2(x))


# =======================================================================
# Block #2 (line ~254, verbatim): trivial comment-only stub describing a
# DeepSpeed JSON config file. Nothing to execute beyond the comments
# themselves; included verbatim for faithfulness.
# =======================================================================

# deepspeed_offload_config.json — enable ZeRO-3 with CPU offload
# Drop this into your DeepSpeed config to offload optimizer states + params.


# =======================================================================
# Block #5 (line ~374, verbatim): LoRA from scratch.
# =======================================================================

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a LoRA side path.

    During forward: output = x @ W^T + (x @ A^T) @ B^T * scale
    where W is frozen and only A, B are updated.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,   # LoRA scaling hyper-param; scale = alpha/rank
        dropout: float = 0.05,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.scale        = alpha / rank  # Hu et al. use this to keep LR independent of r

        # Frozen base weight (will be loaded from pretrained model)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        ) if bias else None

        # Trainable LoRA matrices
        # A is initialized from N(0, 1/sqrt(r)) to give unit-variance init.
        # B is initialized to zero so ΔW = 0 at the start of training.
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        self.lora_dropout = nn.Dropout(dropout)

        # Kaiming init for A (matches standard linear init scale)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 16,
                    alpha: float = 32.0) -> "LoRALinear":
        """Convert an existing nn.Linear to LoRALinear, preserving its weights."""
        bias = linear.bias is not None
        lora = cls(linear.in_features, linear.out_features,
                   rank=rank, alpha=alpha, bias=bias)
        with torch.no_grad():
            lora.weight.copy_(linear.weight)
            if bias:
                lora.bias_param.copy_(linear.bias)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard linear (no grad flows here since weight.requires_grad=False)
        base_out = F.linear(x, self.weight, self.bias_param)

        # LoRA side path: (x @ A^T) @ B^T, scaled
        # shapes: x[..., in_features] -> A[rank, in] -> B[out, rank]
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),  # [..., rank]
            self.lora_B                                     # [..., out]
        )
        return base_out + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge the LoRA update into W for efficient inference.
        Returns a standard nn.Linear with merged weights.
        """
        merged_weight = self.weight + self.scale * (self.lora_B @ self.lora_A)
        linear = nn.Linear(self.in_features, self.out_features,
                           bias=self.bias_param is not None)
        with torch.no_grad():
            linear.weight.copy_(merged_weight)
            if self.bias_param is not None:
                linear.bias.copy_(self.bias_param)
        return linear

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


# -----------------------------------------------------------------------
# Utility: inject LoRA into all attention Q, V projections of a GPT-style
# model.  This is the most common recipe (K and O are often frozen).
# -----------------------------------------------------------------------

def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0,
                target_modules: tuple = ("q_proj", "v_proj")) -> nn.Module:
    """
    Walk the module tree and replace named Linear sub-modules
    whose name ends with any string in target_modules with LoRALinear.
    Freezes all non-LoRA parameters.
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Replace target projections with LoRA versions
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Navigate to parent and set child
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                lora_module = LoRALinear.from_linear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_module)
                break  # Found target for this module; move on

    # Report trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA injection complete: {trainable:,} / {total:,} params trainable "
          f"({100 * trainable / total:.3f}%)")
    return model


# =======================================================================
# Test driver
# =======================================================================

def test_block1_selective_checkpoint_block():
    torch.manual_seed(0)
    d_model, n_heads, batch, seq_len = 32, 4, 2, 6

    block = SelectiveCheckpointBlock(d_model, n_heads)

    # Path with requires_grad=True: exercises the checkpoint() recompute branch.
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)
    out = block(x)
    assert out.shape == (batch, seq_len, d_model)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.attn.in_proj_weight.grad is not None

    # Path with requires_grad=False: exercises the eager (non-checkpoint) branch.
    block.zero_grad(set_to_none=True)
    x2 = torch.randn(batch, seq_len, d_model)
    out2 = block(x2)
    assert out2.shape == (batch, seq_len, d_model)
    print("[block #1] SelectiveCheckpointBlock forward+backward OK, "
          f"out.shape={tuple(out.shape)}")


def test_block2_deepspeed_config_comment():
    # Block #2 is a two-line comment stub -- nothing to execute. Its mere
    # presence above (parsed without error) is the whole test.
    print("[block #2] deepspeed_offload_config.json comment stub parsed OK "
          "(no executable content)")


def test_block5_lora_from_scratch():
    torch.manual_seed(0)

    # --- Exercise LoRALinear directly -----------------------------------
    in_features, out_features, rank = 8, 8, 2
    base_linear = nn.Linear(in_features, out_features)
    lora = LoRALinear.from_linear(base_linear, rank=rank, alpha=4.0)

    x = torch.randn(3, in_features)
    out = lora(x)
    assert out.shape == (3, out_features)

    # Since lora_B is zero-initialized, ΔW = 0 at init -> output must match
    # the frozen base linear exactly.
    base_out = base_linear(x)
    assert torch.allclose(out, base_out, atol=1e-6), \
        "LoRA output at init (B=0) should equal the frozen base linear output"

    # Base weight must not require grad; LoRA A/B must.
    assert lora.weight.requires_grad is False
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad

    # Train one step so B becomes non-zero, then check divergence from base.
    opt = torch.optim.SGD([lora.lora_A, lora.lora_B], lr=1.0)
    loss = lora(x).sum()
    loss.backward()
    assert lora.lora_A.grad is not None and lora.lora_B.grad is not None
    opt.step()

    # merge_weights() is an inference-time operation, so disable dropout
    # (eval mode) before comparing it against the adapted forward pass --
    # otherwise lora_dropout's stochastic masking makes any two forward
    # calls disagree by construction, independent of the merge logic.
    lora.eval()
    out_after = lora(x)
    assert not torch.allclose(out_after, base_out, atol=1e-6), \
        "after a gradient step, LoRA output should diverge from the frozen base"

    # merge_weights() must fold the (now nonzero) adapter into a plain Linear
    # that reproduces the adapted forward pass.
    merged = lora.merge_weights()
    assert isinstance(merged, nn.Linear)
    merged_out = merged(x)
    assert torch.allclose(merged_out, out_after, atol=1e-5), \
        "merged linear should reproduce the LoRA-adapted forward pass"

    # --- Exercise inject_lora() on a tiny GPT-style toy model -----------
    class ToyAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d)
            self.v_proj = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)

    class ToyGPT(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.layer0 = ToyAttn(d)
            self.layer1 = ToyAttn(d)

    d = 8
    toy = ToyGPT(d)
    toy = inject_lora(toy, rank=2, alpha=4.0, target_modules=("q_proj", "v_proj"))

    # q_proj / v_proj must now be LoRALinear; k_proj / o_proj remain plain.
    assert isinstance(toy.layer0.q_proj, LoRALinear)
    assert isinstance(toy.layer0.v_proj, LoRALinear)
    assert isinstance(toy.layer0.k_proj, nn.Linear) and not isinstance(toy.layer0.k_proj, LoRALinear)
    assert isinstance(toy.layer1.q_proj, LoRALinear)

    # Non-LoRA params frozen, LoRA params trainable.
    assert toy.layer0.k_proj.weight.requires_grad is False
    assert toy.layer0.q_proj.lora_A.requires_grad is True
    assert toy.layer0.q_proj.weight.requires_grad is False

    trainable = sum(p.numel() for p in toy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in toy.parameters())
    assert 0 < trainable < total

    # Run data through the modified model end-to-end.
    xin = torch.randn(2, d)
    yq = toy.layer0.q_proj(xin)
    assert yq.shape == (2, d)

    print(f"[block #5] LoRALinear + inject_lora OK, trainable={trainable}/{total} params")


def test_block12_compile_after_checkpointing():
    # Block #12, verbatim intent: "compile the model AFTER enabling
    # checkpointing". We shrink n_layers/d_model/n_heads from the book's
    # illustrative (32, 4096, 32) down to tiny CPU-safe values -- a pure
    # size/device substitution, not a change to the logic being demonstrated.
    torch.manual_seed(0)

    # Compile the model AFTER enabling checkpointing, not before.
    model = CheckpointedModel(n_layers=2, d_model=16, n_heads=2,
                              use_checkpointing=True)
    # torch.compile will trace through checkpoint boundaries correctly
    # with use_reentrant=False
    model = torch.compile(model, mode="reduce-overhead")

    x = torch.randn(2, 5, 16, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 5, 16)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"[block #12] torch.compile(CheckpointedModel, mode='reduce-overhead') "
          f"forward+backward OK, out.shape={tuple(out.shape)}")


if __name__ == "__main__":
    test_block1_selective_checkpoint_block()
    test_block2_deepspeed_config_comment()
    test_block5_lora_from_scratch()
    test_block12_compile_after_checkpointing()
    print("\nAll CPU-runnable blocks executed successfully.")
