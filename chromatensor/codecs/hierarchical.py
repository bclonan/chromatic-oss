"""Residual color symbols with explicit per-value precision depth."""
from __future__ import annotations

import math

import torch

from .base import ScalarCodec, finite_values, palette_for_size, validate_codes, validate_domain


class HierarchicalCodec(ScalarCodec):
    """Coarse symbol followed by bounded residual refinements.

    Each code row contains `levels` token slots and one depth byte. Unused
    slots are zero. This dense tensor representation does not compress unused
    slots; a TokenSequence can omit them with explicit length metadata.
    """

    name = "hierarchical"

    def __init__(self, levels: int = 3, vocabulary_size: int = 16,
                 minimum: float = -1, maximum: float = 1, tolerance: float | None = None):
        if isinstance(levels, bool) or int(levels) != levels or not 1 <= levels <= 32:
            raise ValueError("levels must be an integer in [1,32]")
        if isinstance(vocabulary_size, bool) or int(vocabulary_size) != vocabulary_size or not 2 <= vocabulary_size <= 256:
            raise ValueError("vocabulary_size must be an integer in [2,256]")
        if levels > 1 and vocabulary_size == 2:
            raise ValueError("Multistage residual precision requires at least three vocabulary symbols")
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("tolerance must be finite and nonnegative")
        self.levels, self.vocabulary_size = int(levels), int(vocabulary_size)
        self.minimum, self.maximum = validate_domain(minimum, maximum)
        self.tolerance = None if tolerance is None else float(tolerance)
        self.channels = self.levels + 1
        self.bits_per_value = self.channels * 8
        self.palette = palette_for_size(self.vocabulary_size)

    def _bounds(self, level: int) -> tuple[float, float]:
        if level == 0:
            return self.minimum, self.maximum
        radius = (self.maximum - self.minimum) / (2 * (self.vocabulary_size - 1) ** level)
        return -radius, radius

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = finite_values(x).clamp(self.minimum, self.maximum)
        result = torch.zeros((*x.shape, self.channels), dtype=torch.uint8, device=x.device)
        reconstructed = torch.zeros_like(x)
        active = torch.ones_like(x, dtype=torch.bool)
        for level in range(self.levels):
            low, high = self._bounds(level)
            target = x if level == 0 else x - reconstructed
            ids = torch.round((target.clamp(low, high) - low) / (high - low) * (self.vocabulary_size - 1)).long()
            result[..., level] = torch.where(active, ids, 0).to(torch.uint8)
            update = low + ids.double() / (self.vocabulary_size - 1) * (high - low)
            reconstructed = reconstructed + torch.where(active, update, 0)
            result[..., -1] += active.to(torch.uint8)
            if self.tolerance is not None:
                active = active & ((x - reconstructed).abs() > self.tolerance)
        return result

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        c = validate_codes(codes, self.channels)
        lengths = c[..., -1]
        if ((lengths < 1) | (lengths > self.levels)).any():
            raise ValueError("Hierarchical depth must lie in [1,levels]")
        if (c[..., :-1] >= self.vocabulary_size).any():
            raise ValueError("Hierarchical token is absent from its vocabulary")
        for level in range(self.levels):
            active = lengths > level
            if ((~active) & (c[..., level] != 0)).any():
                raise ValueError("Unused hierarchical token slots must be zero")
        return c

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        c = self.validate(codes)
        lengths = c[..., -1]
        reconstructed = torch.zeros(c.shape[:-1], dtype=torch.float64, device=c.device)
        for level in range(self.levels):
            active = lengths > level
            low, high = self._bounds(level)
            update = low + c[..., level].double() / (self.vocabulary_size - 1) * (high - low)
            reconstructed += torch.where(active, update, 0)
        return reconstructed.clamp(self.minimum, self.maximum)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        self.validate(codes)
        return self.palette.to(codes.device)[codes[..., 0].long()]

    def token_lengths(self, codes: torch.Tensor) -> torch.Tensor:
        self.validate(codes)
        return codes[..., -1].long()

    def to_config(self) -> dict:
        return {"name": self.name, "levels": self.levels, "vocabulary_size": self.vocabulary_size,
                "minimum": self.minimum, "maximum": self.maximum, "tolerance": self.tolerance}

