"""stacklm -- the Stack-100M capstone package (Part XIV of The LLM Stack).

A coherent, from-scratch pipeline: tokenizer -> data -> model -> scaling laws ->
optimizer/schedule -> pretrain -> mid-train -> SFT/DPO/GRPO -> agent -> eval ->
quantize/serve. Runs full-scale on one A100; the smoke test exercises the whole
thing on CPU at toy scale. See capstone/PLAN.md for the canonical spec.
"""
from .config import StackConfig, count_params, toy_config

__all__ = ["StackConfig", "count_params", "toy_config"]
__version__ = "0.1.0"
