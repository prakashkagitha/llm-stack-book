"""Ladder configs + parameter accounting for the mini scaling-law study (Ch. 14.5).

Four tiny rungs {S1..S4} scaled deep-and-thin like `Stack-100M`, plus the target
itself. `nonembed_params()` mirrors `stacklm.config.count_params` minus the norm
gains (which are <0.04% of the total and are not what the A/N^alpha term models).

Reference values (asserted by the smoke test):
    S1     N_nonembed = 3,932,160    total = 10,223,616
    target N_nonembed = 84,541,440   total = 101,318,656
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LadderConfig:
    """A single rung. Every field except (d_model, n_layers) is *derived* so the
    recipe stays frozen: head_dim pinned at 64, SwiGLU width ~= 2.75*d_model
    rounded to a multiple of 64, KV heads = 1 on the small rungs (the MQA limit
    of the target's 8:2 GQA)."""
    name: str
    d_model: int
    n_layers: int
    head_dim: int = 64
    n_kv_heads: int = 1
    vocab_size: int = 32768
    seq_len: int = 2048            # pretraining context; enters the FLOP count

    @property
    def n_heads(self) -> int:
        assert self.d_model % self.head_dim == 0
        return self.d_model // self.head_dim

    @property
    def intermediate(self) -> int:                 # SwiGLU hidden width
        return int(round(2.75 * self.d_model / 64) * 64)

    def nonembed_params(self) -> int:
        """Parameters in the transformer blocks (the capacity variable we fit).
        Per block: Q,O are d*d; K,V are d*(n_kv*head_dim) under GQA; SwiGLU is
        3*d*intermediate (gate, up, down). Norm gains are omitted."""
        d, kv, hd, inter = self.d_model, self.n_kv_heads, self.head_dim, self.intermediate
        attn = 2 * d * d + 2 * d * (kv * hd)
        mlp = 3 * d * inter
        return self.n_layers * (attn + mlp)

    def embed_params(self) -> int:
        return self.vocab_size * self.d_model      # tied: counted once

    def total_params(self) -> int:
        return self.nonembed_params() + self.embed_params()


LADDER = [
    LadderConfig("S1", d_model=192, n_layers=10),
    LadderConfig("S2", d_model=256, n_layers=13),
    LadderConfig("S3", d_model=320, n_layers=17),
    LadderConfig("S4", d_model=448, n_layers=21),
]
TARGET = LadderConfig("Stack-100M", d_model=512, n_layers=30, n_kv_heads=2)
BY_NAME = {c.name: c for c in LADDER}


def family(d_model: int, n_kv_heads: int = 2) -> LadderConfig:
    """Smooth continuation of the ladder's deep-and-thin aspect ratio,
    n_layers ~= d_model / 19.2. This is the SEARCH SPACE for allocation
    questions; the target itself is pinned by PLAN.md at d=512 / L=30."""
    return LadderConfig(f"d{d_model}", d_model=d_model,
                        n_layers=max(4, round(d_model / 19.2)),
                        n_kv_heads=n_kv_heads)
