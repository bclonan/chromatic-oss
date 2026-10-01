"""Bounded HSV and explicitly periodic hue representations."""
from __future__ import annotations

import math

import torch

from .base import ScalarCodec, finite_values, validate_codes, validate_domain


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Vectorized HSV in [0,1] to byte RGB. Hue wraps at one."""
    h, s, v = hsv.unbind(-1)
    h6 = h.remainder(1) * 6
    sector = h6.floor().long() % 6
    fraction = h6 - h6.floor()
    p, q, t = v * (1 - s), v * (1 - s * fraction), v * (1 - s * (1 - fraction))
    options = torch.stack((torch.stack((v, t, p), -1), torch.stack((q, v, p), -1),
                           torch.stack((p, v, t), -1), torch.stack((p, q, v), -1),
                           torch.stack((t, p, v), -1), torch.stack((v, p, q), -1)), -2)
    return torch.round(options.gather(-2, sector[..., None, None].expand(*sector.shape, 1, 3)).squeeze(-2) * 255).clamp(0, 255).to(torch.uint8)


class HSVCodec(ScalarCodec):
    """H stores an 8-bit linear position; S and V remain fully saturated.

    Its numerical domain is bounded and linear. Endpoint colors meet on the
    hue circle, but their byte symbols differ. HueCodec is the periodic codec.
    """

    name = "hsv"
    channels = 3
    bits_per_value = 24
    effective_bits = 8

    def __init__(self, minimum: float = -1, maximum: float = 1):
        self.minimum, self.maximum = validate_domain(minimum, maximum)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = finite_values(x).clamp(self.minimum, self.maximum)
        h = torch.round((x - self.minimum) / (self.maximum - self.minimum) * 255).to(torch.uint8)
        return torch.stack((h, torch.full_like(h, 255), torch.full_like(h, 255)), -1)

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        c = validate_codes(codes, 3)
        if (c[..., 1:] != 255).any():
            raise ValueError("The scalar HSV codec requires S=V=255")
        return c

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        c = self.validate(codes)
        return self.minimum + c[..., 0].double() / 255 * (self.maximum - self.minimum)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        self.validate(codes)
        return hsv_to_rgb(codes.double() / 255)

    def to_config(self) -> dict:
        return {"name": self.name, "minimum": self.minimum, "maximum": self.maximum}


class HueCodec(ScalarCodec):
    """One-byte circular phase code. Decode returns the canonical phase [0, period)."""

    name = "hue"
    channels = 1
    bits_per_value = 8

    def __init__(self, period: float = 2 * math.pi, bins: int = 256, origin: float = 0):
        if not math.isfinite(period) or period <= 0 or not math.isfinite(origin):
            raise ValueError("period must be finite and positive; origin must be finite")
        if isinstance(bins, bool) or int(bins) != bins or not 2 <= bins <= 256:
            raise ValueError("Hue bins must be an integer in [2,256]")
        self.period, self.bins, self.origin = float(period), int(bins), float(origin)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        phase = (finite_values(x) - self.origin).remainder(self.period)
        ids = torch.round(phase / self.period * self.bins).long().remainder(self.bins)
        return ids.to(torch.uint8).unsqueeze(-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.origin + self.validate(codes)[..., 0].double() / self.bins * self.period

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        return validate_codes(codes, 1, self.bins - 1)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        h = self.validate(codes)[..., 0].double() / self.bins
        return hsv_to_rgb(torch.stack((h, torch.ones_like(h), torch.ones_like(h)), -1))

    def circular_error(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        return ((x - reconstructed + self.period / 2).remainder(self.period) - self.period / 2).abs()

    def to_config(self) -> dict:
        return {"name": self.name, "period": self.period, "bins": self.bins, "origin": self.origin}

