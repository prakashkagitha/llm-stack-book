"""Post-training quantization (Ch. 14.11): round-to-nearest int8 (per-row) and
int4 (grouped) baselines, and an in-place swap of every `nn.Linear` for a
`QuantizedLinear` that dequantizes on the fly. This is the RTN baseline that
GPTQ/AWQ improve on.
"""
import json

import torch
import torch.nn as nn


def quantize_int8_per_row(weight: torch.Tensor):
    w = weight.float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0          # (d_out,)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-127, 127)
    return q.to(torch.int8), scale


def dequantize_int8_per_row(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale.unsqueeze(1)


def quantize_int4_grouped(weight: torch.Tensor, group_size: int = 64):
    d_out, d_in = weight.shape
    assert d_in % group_size == 0, f"d_in={d_in} not divisible by group_size={group_size}"
    n_groups = d_in // group_size
    w = weight.float().view(d_out, n_groups, group_size)
    w_min = w.amin(dim=2)                                          # (d_out, n_groups)
    w_max = w.amax(dim=2)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)              # 4-bit unsigned: 0..15
    zero_point = torch.round(-w_min / scale)
    q = torch.round(w / scale.unsqueeze(2) + zero_point.unsqueeze(2)).clamp(0, 15)
    q = q.view(d_out, d_in).to(torch.uint8)
    q_even, q_odd = q[:, 0::2], q[:, 1::2]
    packed = (q_even | (q_odd << 4)).to(torch.uint8)             # (d_out, d_in // 2)
    return packed, scale, zero_point


def dequantize_int4_grouped(packed, scale, zero_point, d_in, group_size: int = 64):
    d_out = packed.shape[0]
    n_groups = d_in // group_size
    q_even = (packed & 0x0F).to(torch.float32)
    q_odd = ((packed >> 4) & 0x0F).to(torch.float32)
    q = torch.empty(d_out, d_in)
    q[:, 0::2], q[:, 1::2] = q_even, q_odd
    q = q.view(d_out, n_groups, group_size)
    w = (q - zero_point.unsqueeze(2)) * scale.unsqueeze(2)
    return w.view(d_out, d_in)


class QuantizedLinear(nn.Module):
    def __init__(self, bits: int, group_size: int = 64):
        super().__init__()
        assert bits in (4, 8)
        self.bits = bits
        self.group_size = group_size
        self.d_in = None

    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int, group_size: int = 64):
        layer = cls(bits, group_size)
        layer.d_in = linear.in_features
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        layer.register_buffer("bias", bias)
        if bits == 8:
            q, scale = quantize_int8_per_row(linear.weight.data)
            layer.register_buffer("q_weight", q)
            layer.register_buffer("scale", scale)
        else:
            packed, scale, zp = quantize_int4_grouped(linear.weight.data, group_size)
            layer.register_buffer("q_weight", packed)
            layer.register_buffer("scale", scale)
            layer.register_buffer("zero_point", zp)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 8:
            w = dequantize_int8_per_row(self.q_weight, self.scale)
        else:
            w = dequantize_int4_grouped(self.q_weight, self.scale, self.zero_point,
                                        self.d_in, self.group_size)
        return torch.nn.functional.linear(x, w.to(x.dtype), self.bias)


def quantize_stacklm(model: nn.Module, bits: int, group_size: int = 64) -> nn.Module:
    """In-place: swap every nn.Linear whose in_features is divisible by group_size.
    The tied lm_head shares weights with the embedding; we only quantize the
    hidden Linears (embedding stays fp32) to keep tying intact."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            if bits == 4 and module.in_features % group_size != 0:
                continue  # skip layers that don't tile evenly (rare at real config)
            setattr(model, name, QuantizedLinear.from_float(module, bits, group_size))
        else:
            quantize_stacklm(module, bits, group_size)
    return model


def export_quantized(model: nn.Module, path: str, bits: int, config: dict) -> None:
    torch.save(model.state_dict(), path + ".pt")
    with open(path + ".json", "w") as f:
        json.dump({"bits": bits, "group_size": 64, "architecture": config}, f, indent=2)
