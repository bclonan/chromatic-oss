"""RGBD with an explicit exponent channel rather than an arbitrary fourth byte."""
from __future__ import annotations

import torch

from .base import ScalarCodec, finite_values, pack_rgb, validate_codes


class RGBDCodec(ScalarCodec):
    """Lossless finite FP32 encoding.

    D contains the IEEE-754 biased exponent. RGB contains the sign bit and
    23 stored fraction bits. Normal values have 24 significand bits because
    their leading one is implicit. D=0 represents subnormal numbers or zero.
    D=255 is forbidden. This encoding performs floating-point arithmetic
    during interoperation; it does not provide integer-only arithmetic.
    """

    name = "rgbd"
    channels = 4
    bits_per_value = 32

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        original = finite_values(x)
        fp32 = original.to(torch.float32)
        if not torch.isfinite(fp32).all():
            raise ValueError("RGBD finite FP32 range overflow")
        if ((original != 0) & (fp32 == 0)).any():
            raise ValueError("RGBD value underflows the finite FP32 range")
        bits = fp32.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        sign_fraction = ((bits >> 8) & 0x800000) | (bits & 0x7FFFFF)
        exponent = ((bits >> 23) & 255).to(torch.uint8)
        return torch.cat((pack_rgb(sign_fraction), exponent.unsqueeze(-1)), -1)

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        c = validate_codes(codes, 4)
        if (c[..., 3] == 255).any():
            raise ValueError("D=255 is reserved for nonfinite IEEE values")
        return c

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        c = self.validate(codes)
        packed = (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]
        bits = ((packed & 0x800000) << 8) | (c[..., 3] << 23) | (packed & 0x7FFFFF)
        return bits.to(torch.int32).contiguous().view(torch.float32)

    def to_config(self) -> dict:
        return {"name": self.name, "format": "finite-ieee754-fp32-sign-fraction-exponent"}

