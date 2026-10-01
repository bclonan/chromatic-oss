"""Discrete forward values with explicitly approximate straight-through gradients.

This is fake quantization. Parameters, gradients, and kernels remain floating point.
The chromatic modes materialize RGB or RGBD bytes and decode those bytes in forward.
The INT modes use the same uniform levels, so color alone brings no accuracy gain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..codecs.rgbd import RGBDCodec


_RGB_MODULUS = 1 << 24
_PALETTE_MULTIPLIER = 0x9E3779
_PALETTE_OFFSET = 0x406080
_PALETTE_INVERSE = pow(_PALETTE_MULTIPLIER, -1, _RGB_MODULUS)


def _configuration(representation: str, bits: int) -> tuple[str, int]:
    representation = representation.lower()
    aliases = {"rgb": "rgb24", "chromatic": "chroma", "uniform": "int"}
    representation = aliases.get(representation, representation)
    if representation in {"rgb24", "fixed24"}:
        bits = 24
    elif representation == "chroma8":
        bits = 8
    elif representation in {"int8", "int4"}:
        bits = int(representation[3:])
    elif representation == "rgbd":
        return representation, 32
    if representation not in {"chroma", "chroma8", "rgb24", "int", "int8", "int4", "fixed24"}:
        raise ValueError(f"Unknown quantization representation {representation!r}")
    if not isinstance(bits, int) or not 1 <= bits <= 24:
        raise ValueError("bits must be an integer between 1 and 24")
    return representation, bits


def codes_to_rgb(codes: Tensor, *, palette: bool = True) -> Tensor:
    """Return a reversible, deterministic 24-bit chromatic symbol for each ID.

    The palette permutes packed RGB integers with an invertible affine map.
    The color geometry has no claimed numerical meaning. RGB24 uses direct packing.
    """
    codes = codes.to(torch.int64)
    packed = (codes * _PALETTE_MULTIPLIER + _PALETTE_OFFSET) & 0xFFFFFF if palette else codes
    return torch.stack(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255), -1).to(torch.uint8)


def rgb_to_codes(rgb: Tensor, *, palette: bool = True) -> Tensor:
    if rgb.dtype != torch.uint8 or rgb.shape[-1] != 3:
        raise ValueError("RGB symbols must be uint8 with a final channel dimension of 3")
    rgb = rgb.to(torch.int64)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return ((packed - _PALETTE_OFFSET) * _PALETTE_INVERSE) & 0xFFFFFF if palette else packed


@dataclass
class QuantizedState:
    """Actual forward codes, color bytes where applicable, and scalar range.

    RGBD uses four bytes per value and has no numerical range metadata. Its
    codes and rgbd fields refer to the same byte tensor.
    """

    codes: Tensor
    rgb: Tensor | None
    offset: Tensor
    scale: Tensor
    bits: int
    representation: str
    original_dtype: torch.dtype
    range_axes: tuple[int, ...] = ()
    range_method: str = "global min/max"
    constant_ranges: int = 0
    rgbd: Tensor | None = None

    def decode(self) -> Tensor:
        if self.rgbd is not None:
            return RGBDCodec().decode(self.rgbd).to(self.original_dtype)
        if self.rgb is not None:
            codes = rgb_to_codes(self.rgb, palette=self.representation != "rgb24")
        else:
            codes = self.codes.to(torch.int64)
        # FP64 avoids losing the last RGB24 levels before the final dtype cast.
        return (codes.to(torch.float64) * self.scale.to(torch.float64) + self.offset.to(torch.float64)).to(self.original_dtype)

    @property
    def payload_bytes(self) -> int:
        payload = self.rgbd if self.rgbd is not None else self.rgb if self.rgb is not None else self.codes
        return payload.numel() * payload.element_size()

    @property
    def retained_state_bytes(self) -> int:
        """Bytes in distinct inspection buffers, including redundant code IDs."""
        buffers = [self.codes, self.offset, self.scale]
        if self.rgb is not None:
            buffers.append(self.rgb)
        if self.rgbd is not None:
            buffers.append(self.rgbd)
        unique = {id(buffer): buffer for buffer in buffers}
        return sum(buffer.numel() * buffer.element_size() for buffer in unique.values())

    @property
    def range_metadata_bytes(self) -> int:
        return self.offset.numel() * self.offset.element_size() + self.scale.numel() * self.scale.element_size()


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, reconstructed: Tensor) -> Tensor:
        return reconstructed

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        # Identity derivative, including clipped values. This is an estimator,
        # not the true derivative of rounding, byte packing, or clipping.
        return grad_output, None


class ChromaQuantizer(nn.Module):
    """Color encode/decode with a straight-through backward pass.

    Uniform modes use detached minimum and maximum values with no fixed range.
    The default range covers the tensor. range_axis=-1 gives each feature vector
    its own range, so other batch members and future tokens cannot change it.
    Constant or singleton vectors reconstruct exactly through their stored offset.
    Range estimation is not differentiated. The palette relabels scalar levels.
    RGBD packs finite FP32 sign, fraction and exponent bytes and decodes those
    bytes each forward. It needs no range reduction. FP16 and BF16 round-trip
    exactly; FP64 rounds to FP32. Kernels and gradients remain floating point.
    """

    def __init__(self, bits: int = 8, representation: str = "chroma", *,
                 min_value: float | None = None, max_value: float | None = None,
                 enabled: bool = True, range_axis: int | None = None):
        super().__init__()
        self.representation, self.bits = _configuration(representation, bits)
        if self.representation == "rgbd" and (min_value is not None or max_value is not None):
            raise ValueError("RGBD bit packing does not accept fixed min_value or max_value")
        if (min_value is None) != (max_value is None):
            raise ValueError("Specify both min_value and max_value, or neither")
        if min_value is not None and (not math.isfinite(min_value) or not math.isfinite(max_value) or max_value <= min_value):
            raise ValueError("The fixed range must be finite with max_value > min_value")
        self.min_value, self.max_value = min_value, max_value
        if range_axis is not None and (type(range_axis) is not int):
            raise TypeError("range_axis must be an integer or None")
        self.range_axis = range_axis
        self.enabled = enabled
        self.forward_calls = 0
        self.encoded_values = 0
        self.last_state: QuantizedState | None = None

    def encode(self, x: Tensor) -> QuantizedState:
        if not x.is_floating_point():
            raise TypeError("Fake quantization expects a floating-point tensor")
        if self.representation == "rgbd":
            rgbd = RGBDCodec().encode(x)
            empty = torch.empty(0, dtype=torch.float64, device=x.device)
            return QuantizedState(rgbd, None, empty, empty, 32, self.representation,
                                  x.dtype, (), "RGBD bit packing", 0, rgbd=rgbd)
        if self.range_axis is None:
            axes = tuple(range(x.ndim))
        elif x.ndim:
            if not -x.ndim <= self.range_axis < x.ndim:
                raise ValueError("range_axis is outside the input tensor dimensions")
            axes = (self.range_axis % x.ndim,)
        else:
            if self.range_axis not in (-1, 0):
                raise ValueError("range_axis is outside the input tensor dimensions")
            axes = ()
        keepdim = self.range_axis is not None
        range_shape = tuple(1 if axis in axes else size for axis, size in enumerate(x.shape)) if keepdim else ()
        if x.numel() == 0:
            lower = torch.zeros(range_shape, dtype=torch.float64, device=x.device)
            upper = torch.ones(range_shape, dtype=torch.float64, device=x.device)
        else:
            if not torch.isfinite(x).all():
                raise ValueError("Fake quantization only supports finite values")
            lower = x.detach().amin(dim=axes, keepdim=keepdim).to(torch.float64) if axes else x.detach().to(torch.float64)
            upper = x.detach().amax(dim=axes, keepdim=keepdim).to(torch.float64) if axes else x.detach().to(torch.float64)
        method = "global min/max" if self.range_axis is None else "axis min/max"
        if self.min_value is not None:
            lower = torch.tensor(self.min_value, dtype=torch.float64, device=x.device)
            upper = torch.tensor(self.max_value, dtype=torch.float64, device=x.device)
            axes, method = (), "fixed range"
        levels = (1 << self.bits) - 1
        scale = (upper - lower) / levels
        constant_ranges = int((scale == 0).sum())
        # Constant tensors get a safe step and code zero, preserving the value.
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = ((x.detach().to(torch.float64) - lower) / scale).round().clamp(0, levels).to(torch.int64)
        storage_dtype = torch.uint8 if self.bits <= 8 else torch.int32
        codes = codes.to(storage_dtype)
        chromatic = self.representation in {"chroma", "chroma8", "rgb24"}
        rgb = codes_to_rgb(codes, palette=self.representation != "rgb24") if chromatic else None
        return QuantizedState(codes, rgb, lower, scale, self.bits, self.representation, x.dtype,
                              axes, method, constant_ranges)

    def forward(self, x: Tensor) -> Tensor:
        self.forward_calls += 1
        if not self.enabled:
            return x
        state = self.encode(x)
        self.last_state = state
        self.encoded_values += x.numel()
        return _StraightThrough.apply(x, state.decode())

    def extra_repr(self) -> str:
        return f"bits={self.bits}, representation={self.representation!r}, enabled={self.enabled}, range_axis={self.range_axis}"


def fake_chroma_quantize(x: Tensor, bits: int = 8, representation: str = "chroma", *,
                         min_value: float | None = None, max_value: float | None = None,
                         range_axis: int | None = None) -> Tensor:
    return ChromaQuantizer(bits, representation, min_value=min_value, max_value=max_value,
                           range_axis=range_axis)(x)


fake_quantize = fake_chroma_quantize
