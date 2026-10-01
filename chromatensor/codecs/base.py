"""Shared validation and byte packing for discrete scalar codecs."""
from __future__ import annotations

import math
from typing import Any

import torch


INTEGER_DTYPES = (torch.uint8, torch.uint16, torch.uint32, torch.uint64,
                  torch.int8, torch.int16, torch.int32, torch.int64)


def finite_values(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.is_complex() or x.dtype == torch.bool:
        raise TypeError("Scalar codecs require real numerical values")
    if not torch.isfinite(x).all():
        raise ValueError("NaN and infinity have no finite chromatic encoding")
    return x.detach().to(torch.float64)


def validate_domain(minimum: float, maximum: float) -> tuple[float, float]:
    minimum, maximum = float(minimum), float(maximum)
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError("minimum and maximum must be finite and minimum < maximum")
    if not math.isfinite(maximum - minimum):
        raise ValueError("The domain width must be finite")
    return minimum, maximum


def validate_codes(codes: torch.Tensor, channels: int, maximum: int = 255) -> torch.Tensor:
    codes = torch.as_tensor(codes)
    if codes.dtype not in INTEGER_DTYPES:
        raise TypeError("Chromatic codes must use an integer dtype")
    if codes.ndim < 1 or codes.shape[-1] != channels:
        raise ValueError(f"Expected a trailing code dimension of {channels}")
    values = codes.to(torch.int64)
    if ((values < 0) | (values > maximum)).any():
        raise ValueError(f"Code values must lie in [0, {maximum}]")
    return values


def pack_rgb(ids: torch.Tensor) -> torch.Tensor:
    ids = ids.to(torch.int64)
    return torch.stack(((ids >> 16) & 255, (ids >> 8) & 255, ids & 255), -1).to(torch.uint8)


def unpack_rgb(codes: torch.Tensor) -> torch.Tensor:
    c = validate_codes(codes, 3)
    return (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]


def id_dtype(size: int) -> torch.dtype:
    if size <= 256:
        return torch.uint8
    if size <= 65536:
        return torch.uint16
    if size <= 2 ** 31:
        return torch.int32
    return torch.int64


def palette_for_size(size: int) -> torch.Tensor:
    """Unique RGB symbols with nearby IDs occupying nearby Morton-grid cells."""
    ids = torch.arange(size, dtype=torch.int64)
    bits = max(1, (size - 1).bit_length())
    channels = []
    for channel in range(3):
        positions = list(range(channel, bits, 3))
        value = torch.zeros_like(ids)
        for target, source in enumerate(positions):
            value |= ((ids >> source) & 1) << target
        denominator = max(1, 2 ** len(positions) - 1)
        channels.append(torch.round(value.double() * 255 / denominator).to(torch.uint8))
    return torch.stack(channels, -1)


class ScalarCodec:
    """Codec protocol with finite inputs and explicit trailing code channels."""

    channels: int
    bits_per_value: int
    name: str

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        """Validate symbol storage with integer operations only."""
        return validate_codes(codes, self.channels)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        values = self.validate(codes)
        return values[..., :3].to(torch.uint8)

    def equivalent(self, other: Any) -> bool:
        return isinstance(other, ScalarCodec) and self.to_config() == other.to_config()

