"""Pre-allocated K/V cache for incremental decoding (Ch. 14.4).

One slab per layer, shaped (n_layers, B, n_kv_heads, max_seq, head_dim). We cache the
NARROW (n_kv-head), already-RoPE-rotated K/V: caching before rotation would force a
re-rotation on every read, and caching after GQA head expansion would throw away the
entire 4x memory win. Production servers page this memory instead -- see vLLM's
PagedAttention (Ch. 4.6 / 7.3).
"""
import torch

from ..config import StackConfig


class KVCache:
    def __init__(self, cfg: StackConfig, batch_size: int, max_seq: int,
                 device=None, dtype=torch.bfloat16):
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, max_seq, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq = max_seq

    def update(self, layer_idx: int, k, v, start_pos: int):
        """Write this step's K/V at [start_pos : start_pos+T]; return the whole prefix."""
        T = k.shape[2]
        assert start_pos + T <= self.max_seq, "KV cache overflow -- grow max_seq"
        self.k[layer_idx, :, :, start_pos:start_pos + T] = k.to(self.k.dtype)
        self.v[layer_idx, :, :, start_pos:start_pos + T] = v.to(self.v.dtype)
        return (self.k[layer_idx, :, :, :start_pos + T],
                self.v[layer_idx, :, :, :start_pos + T])

    def nbytes(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()
