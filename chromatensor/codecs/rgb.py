"""Linear RGB24 and bit-controlled fixed-point baselines."""
from __future__ import annotations

import torch

from .base import ScalarCodec, finite_values, id_dtype, pack_rgb, unpack_rgb, validate_codes, validate_domain


class RGBCodec(ScalarCodec):
    """Quantize a bounded scalar to an RGB byte triplet, high byte first.

    Finite values outside the domain clip to its endpoints. A logarithmic
    domain requires positive endpoints and rejects nonpositive input values.
    """

    name = "rgb"
    channels = 3
    bits_per_value = 24
    levels = 2 ** 24

    def __init__(self, minimum: float = -1, maximum: float = 1, logarithmic: bool = False):
        self.minimum, self.maximum = validate_domain(minimum, maximum)
        self.logarithmic = bool(logarithmic)
        if self.logarithmic and self.minimum <= 0:
            raise ValueError("Logarithmic domains require positive endpoints")
        if self.logarithmic:
            bounds = torch.tensor([self.minimum, self.maximum], dtype=torch.float64).log()
            if not bounds[0] < bounds[1]:
                raise ValueError("Logarithmic endpoints must have distinct FP64 logarithms")

    def _normalized(self, x: torch.Tensor) -> torch.Tensor:
        x = finite_values(x)
        if self.logarithmic:
            if (x <= 0).any():
                raise ValueError("Logarithmic encoding requires positive values")
            bounds = torch.tensor([self.minimum, self.maximum], dtype=x.dtype, device=x.device).log()
            if not bounds[0] < bounds[1]:
                raise ValueError("Logarithmic endpoints must have distinct FP64 logarithms on the encoding device")
            return ((x.clamp(self.minimum, self.maximum).log() - bounds[0]) / (bounds[1] - bounds[0])).clamp(0, 1)
        return ((x.clamp(self.minimum, self.maximum) - self.minimum) / (self.maximum - self.minimum)).clamp(0, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return pack_rgb(torch.round(self._normalized(x) * (self.levels - 1)).to(torch.int64))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        t = unpack_rgb(codes).to(torch.float64) / (self.levels - 1)
        if self.logarithmic:
            bounds = torch.tensor([self.minimum, self.maximum], dtype=t.dtype, device=t.device).log()
            return torch.exp(bounds[0] + t * (bounds[1] - bounds[0]))
        return self.minimum + t * (self.maximum - self.minimum)

    def to_config(self) -> dict:
        return {"name": self.name, "minimum": self.minimum, "maximum": self.maximum, "logarithmic": self.logarithmic}


class UniformCodec(RGBCodec):
    """Uniform integer quantization, with actual byte/word storage reported separately."""

    name = "uniform"
    channels = 1

    def __init__(self, bits: int = 8, minimum: float = -1, maximum: float = 1, logarithmic: bool = False):
        if isinstance(bits, bool) or int(bits) != bits or not 1 <= bits <= 32:
            raise ValueError("bits must be an integer from 1 through 32")
        super().__init__(minimum, maximum, logarithmic)
        self.bits = int(bits)
        self.bits_per_value = self.bits
        self.levels = 2 ** self.bits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ids = torch.round(self._normalized(x) * (self.levels - 1)).to(id_dtype(self.levels))
        return ids.unsqueeze(-1)

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        return validate_codes(codes, 1, self.levels - 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        ids = self.validate(codes)[..., 0]
        t = ids.to(torch.float64) / (self.levels - 1)
        if self.logarithmic:
            bounds = torch.tensor([self.minimum, self.maximum], dtype=t.dtype, device=t.device).log()
            return torch.exp(bounds[0] + t * (bounds[1] - bounds[0]))
        return self.minimum + t * (self.maximum - self.minimum)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        ids = self.validate(codes)[..., 0]
        return pack_rgb(torch.round(ids.double() * (2 ** 24 - 1) / (self.levels - 1)).long())

    def to_config(self) -> dict:
        return {**super().to_config(), "bits": self.bits}

