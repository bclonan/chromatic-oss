"""Globally mixed N-dimensional Fourier cubes with integer polar storage.

"Holographic" describes distributed Fourier coefficients, not optical hardware
or recovery of an arbitrary tensor from an arbitrary fragment. The forward and
inverse transforms use float64 and complex128. Stored coefficients use 24 bits.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np
import torch

from .codecs.hsv import hsv_to_rgb
from .serialization.format import _budget, _decompress, _finite_json, _read_file

PHASE_BINS = 256
MAGNITUDE_LEVELS = 65535
MAGIC = b"HCUBE\x00\x01\x00"
MAX_HEADER = 1024 * 1024
MAX_PAYLOAD = 1024 * 1024 * 1024
REAL_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _real_input(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.is_complex() or x.layout != torch.strided:
        raise TypeError("HolographicCube requires a dense real tensor")
    if x.numel() == 0 or x.ndim > 64:
        raise ValueError("HolographicCube requires nonempty input with at most 64 dimensions")
    x = x.detach().to(torch.float64)
    if not torch.isfinite(x).all():
        raise ValueError("HolographicCube input must contain only finite values")
    return x


def _forward(x: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fftn(x, norm="ortho") if x.ndim else x.to(torch.complex128)
    if not torch.isfinite(spectrum).all():
        raise ValueError("Input exceeds the finite float64 Fourier transform range")
    return spectrum


def _inverse(spectrum: torch.Tensor) -> torch.Tensor:
    output = torch.fft.ifftn(spectrum, norm="ortho") if spectrum.ndim else spectrum
    if not torch.isfinite(output).all():
        raise ValueError("Coefficients exceed the finite float64 inverse transform range")
    return output.real


def _validate_codes(codes: torch.Tensor, amplitude_scale: float) -> None:
    if not isinstance(codes, torch.Tensor) or codes.dtype != torch.uint8:
        raise TypeError("Fourier coefficient codes must use torch.uint8 storage")
    if codes.layout != torch.strided or codes.ndim < 1 or codes.shape[-1] != 3:
        raise ValueError("Fourier codes require a trailing dimension of three bytes")
    if codes.numel() == 0 or codes.ndim > 65:
        raise ValueError("Fourier code shape must be nonempty with at most 64 value dimensions")
    if isinstance(amplitude_scale, bool) or not isinstance(amplitude_scale, (int, float)):
        raise TypeError("amplitude_scale must be a finite nonnegative number")
    if not math.isfinite(amplitude_scale) or amplitude_scale < 0:
        raise ValueError("amplitude_scale must be finite and nonnegative")
    if amplitude_scale == 0 and torch.any(codes[..., 1:] != 0):
        raise ValueError("A zero amplitude scale requires zero magnitude codes")


def _keep_mask(mask: torch.Tensor, shape: torch.Size, device: torch.device) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("Coefficient mask must be a boolean torch.Tensor")
    if mask.layout != torch.strided or mask.shape != shape:
        raise ValueError("Coefficient mask must match the exact cube shape")
    return mask.to(device=device)


def encode_numeric_polar(spectrum: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Equal-bit numerical control: one phase byte and two magnitude bytes.

    This control uses exactly the same quantization as the colored cube. Labels
    do not change its coefficients or numerical accuracy.
    """
    spectrum = torch.as_tensor(spectrum)
    if spectrum.layout != torch.strided or spectrum.numel() == 0:
        raise ValueError("Polar encoding requires a nonempty dense coefficient tensor")
    spectrum = spectrum.detach().to(torch.complex128)
    if not torch.isfinite(spectrum).all():
        raise ValueError("Polar coefficients must contain only finite values")
    amplitude = spectrum.abs()
    scale = float(amplitude.max().item())
    if not math.isfinite(scale):
        raise ValueError("Coefficient magnitude exceeds the finite float64 range")
    if scale == 0:
        return torch.zeros((*spectrum.shape, 3), dtype=torch.uint8, device=spectrum.device), 0.0
    phase = torch.round(torch.angle(spectrum).remainder(2 * math.pi) / (2 * math.pi) * PHASE_BINS)
    phase = phase.to(torch.int64).remainder(PHASE_BINS)
    magnitude = torch.round(amplitude / scale * MAGNITUDE_LEVELS).clamp(0, MAGNITUDE_LEVELS).to(torch.int64)
    # Phase has no meaning when quantized magnitude is zero.
    phase = torch.where(magnitude == 0, torch.zeros_like(phase), phase)
    return torch.stack((phase, magnitude >> 8, magnitude & 255), dim=-1).to(torch.uint8), scale


def decode_numeric_polar(codes: torch.Tensor, amplitude_scale: float) -> torch.Tensor:
    """Decode the equal-bit numerical control into complex128 coefficients."""
    _validate_codes(codes, amplitude_scale)
    phase = codes[..., 0].to(torch.float64) * (2 * math.pi / PHASE_BINS)
    magnitude_ids = codes[..., 1].to(torch.int64) * 256 + codes[..., 2].to(torch.int64)
    magnitude = magnitude_ids.to(torch.float64) / MAGNITUDE_LEVELS * amplitude_scale
    return torch.polar(magnitude, phase)


def numeric_polar_reference(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Reconstruct a real tensor using the cube's 24-bit numerical control."""
    values = _real_input(x)
    codes, scale = encode_numeric_polar(_forward(values))
    spectrum = decode_numeric_polar(codes, scale)
    if mask is not None:
        spectrum = spectrum.masked_fill(~_keep_mask(mask, values.shape, values.device), 0)
    return _inverse(spectrum)


class HolographicCube:
    """N-dimensional global Fourier mixing with phase and magnitude byte codes.

    Each input scalar has one full complex FFT coefficient stored in three
    bytes. Real input produces redundant Hermitian pairs, which this format
    retains. amplitude_scale is a separate float64-equivalent scalar metadata
    value. No original floating tensor or floating coefficient field is kept.
    """

    phase_bins = PHASE_BINS
    magnitude_levels = MAGNITUDE_LEVELS
    bits_per_value = 24

    def __init__(self, codes: torch.Tensor, amplitude_scale: float,
                 axis_names: Sequence[str] | None = None):
        _validate_codes(codes, amplitude_scale)
        names = tuple(f"axis_{index}" for index in range(codes.ndim - 1)) if axis_names is None else tuple(axis_names)
        if isinstance(axis_names, str) or len(names) != codes.ndim - 1:
            raise ValueError("axis_names must supply one name per cube dimension")
        if any(not isinstance(name, str) or not name.strip() or len(name) > 256 for name in names):
            raise ValueError("Axis names must be nonempty strings of at most 256 characters")
        if len(set(names)) != len(names):
            raise ValueError("Axis names must be unique")
        self.codes = codes.detach().clone()
        self.amplitude_scale = float(amplitude_scale)
        self.axis_names = names

    @classmethod
    def from_tensor(cls, x: torch.Tensor, axis_names: Sequence[str] | None = None) -> "HolographicCube":
        codes, scale = encode_numeric_polar(_forward(_real_input(x)))
        return cls(codes, scale, axis_names)

    @property
    def shape(self) -> torch.Size:
        return self.codes.shape[:-1]

    @property
    def ndim(self) -> int:
        return self.codes.ndim - 1

    @property
    def device(self) -> torch.device:
        return self.codes.device

    @property
    def dtype(self) -> torch.dtype:
        return self.codes.dtype

    @property
    def storage_nbytes(self) -> int:
        return self.codes.numel() * self.codes.element_size()

    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def phase(self) -> torch.Tensor:
        """Decoded phase in radians in [0, 2*pi)."""
        return self.codes[..., 0].to(torch.float64) * (2 * math.pi / PHASE_BINS)

    @property
    def magnitudes(self) -> torch.Tensor:
        """Decoded Fourier amplitudes, with 65536 inclusive magnitude codes."""
        ids = self.codes[..., 1].to(torch.int64) * 256 + self.codes[..., 2].to(torch.int64)
        return ids.to(torch.float64) / MAGNITUDE_LEVELS * self.amplitude_scale

    @property
    def rgb(self) -> torch.Tensor:
        """HSV preview bytes: hue is phase, saturation is one, value is amplitude."""
        h = self.codes[..., 0].to(torch.float64) / PHASE_BINS
        ids = self.codes[..., 1].to(torch.int64) * 256 + self.codes[..., 2].to(torch.int64)
        v = ids.to(torch.float64) / MAGNITUDE_LEVELS
        return hsv_to_rgb(torch.stack((h, torch.ones_like(h), v), dim=-1))

    def coefficients(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        spectrum = decode_numeric_polar(self.codes, self.amplitude_scale)
        if mask is not None:
            spectrum = spectrum.masked_fill(~_keep_mask(mask, self.shape, self.device), 0)
        return spectrum

    def decode(self, mask: torch.Tensor | None = None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Inverse FFT and real projection; false mask entries erase coefficients."""
        if dtype not in REAL_DTYPES:
            raise TypeError("Decoded dtype must be a real floating point torch dtype")
        return _inverse(self.coefficients(mask)).to(dtype=dtype)

    def mask(self, keep_mask: torch.Tensor) -> "HolographicCube":
        """Return a new cube with false mask entries replaced by zero byte codes."""
        keep = _keep_mask(keep_mask, self.shape, self.device)
        return type(self)(self.codes.masked_fill(~keep.unsqueeze(-1), 0), self.amplitude_scale, self.axis_names)

    def erasure_mask(self, fraction: float, seed: int = 0) -> torch.Tensor:
        """Keep mask with the nearest integer count erased, reproducible across devices."""
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Erasure fraction must be a finite number in [0, 1]")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Erasure seed must be an integer in [0, 2**63)")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        count = math.floor(self.numel() * fraction + 0.5)
        keep = torch.ones(self.numel(), dtype=torch.bool)
        keep[torch.randperm(self.numel(), generator=generator)[:count]] = False
        return keep.reshape(self.shape).to(self.device)

    def erase(self, fraction: float, seed: int = 0) -> "HolographicCube":
        return self.mask(self.erasure_mask(fraction, seed))

    def to(self, device) -> "HolographicCube":
        if isinstance(device, torch.dtype):
            raise TypeError("to() preserves integer storage; use decode(dtype=...) to change numerical dtype")
        return type(self)(self.codes.to(device=device), self.amplitude_scale, self.axis_names)

    def cpu(self) -> "HolographicCube":
        return self.to("cpu")

    def cuda(self, device=None) -> "HolographicCube":
        return self.to(torch.device("cuda", device) if device is not None else "cuda")

    def clone(self) -> "HolographicCube":
        return type(self)(self.codes, self.amplitude_scale, self.axis_names)

    def save(self, path: str | Path, compress: bool = True) -> int:
        return save_hcube(self, path, compress)

    @classmethod
    def load(cls, path: str | Path, device="cpu", *, max_file_bytes=None, max_header_bytes=None,
             max_payload_bytes=None) -> "HolographicCube":
        return load_hcube(path, device, max_file_bytes=max_file_bytes, max_header_bytes=max_header_bytes,
                          max_payload_bytes=max_payload_bytes)

    def __repr__(self) -> str:
        return (f"HolographicCube(shape={tuple(self.shape)}, axis_names={self.axis_names!r}, "
                f"device={self.device}, payload_bytes={self.storage_nbytes})")


def save_hcube(cube: HolographicCube, path: str | Path, compress: bool = True) -> int:
    """Atomically save byte codes and bounded JSON metadata with a SHA-256 checksum."""
    if not isinstance(cube, HolographicCube):
        raise TypeError("save_hcube requires a HolographicCube")
    _validate_codes(cube.codes, cube.amplitude_scale)
    if type(compress) is not bool:
        raise TypeError("compress must be boolean")
    raw = cube.codes.detach().cpu().contiguous().numpy().tobytes()
    if len(raw) > MAX_PAYLOAD:
        raise ValueError("hcube payload exceeds the one GiB format limit")
    payload = zlib.compress(raw, 6) if compress else raw
    metadata = {"version": 1, "shape": list(cube.shape), "axis_names": list(cube.axis_names),
                "amplitude_scale": cube.amplitude_scale, "code_dtype": "uint8",
                "phase_bins": PHASE_BINS, "magnitude_levels": MAGNITUDE_LEVELS,
                "transform": "fftn", "norm": "ortho", "execution_dtype": "complex128",
                "compression": "zlib" if compress else "none", "raw_bytes": len(raw),
                "payload_bytes": len(payload)}
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(header) > MAX_HEADER:
        raise ValueError("hcube metadata exceeds the one MiB format limit")
    body = MAGIC + struct.pack("<I", len(header)) + header + payload
    data = body + hashlib.sha256(body).digest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(data)
        os.replace(temporary, target)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return len(data)


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate hcube metadata key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite hcube JSON metadata is forbidden")


def load_hcube(path: str | Path, device="cpu", *, max_file_bytes=None, max_header_bytes=None,
               max_payload_bytes=None) -> HolographicCube:
    """Load safe JSON and byte codes. Reject unsupported metadata and corruption."""
    header_limit = _budget(max_header_bytes, MAX_HEADER, "max_header_bytes")
    payload_limit = _budget(max_payload_bytes, MAX_PAYLOAD, "max_payload_bytes")
    file_limit = _budget(max_file_bytes, payload_limit + header_limit + 1024 * 1024, "max_file_bytes")
    data = _read_file(path, file_limit)
    if len(data) < 44 or data[:8] != MAGIC:
        raise ValueError("Invalid hcube magic or version")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("hcube checksum mismatch")
    size = struct.unpack("<I", data[8:12])[0]
    if size > header_limit or size > len(data) - 44:
        raise ValueError("Invalid hcube header size")
    metadata = _finite_json(data[12:12 + size])
    required = {"version", "shape", "axis_names", "amplitude_scale", "code_dtype", "phase_bins",
                "magnitude_levels", "transform", "norm", "execution_dtype", "compression", "raw_bytes", "payload_bytes"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("Invalid hcube metadata fields")
    for field, expected in {"version": 1, "code_dtype": "uint8", "phase_bins": PHASE_BINS,
                            "magnitude_levels": MAGNITUDE_LEVELS, "transform": "fftn", "norm": "ortho",
                            "execution_dtype": "complex128"}.items():
        if type(metadata[field]) is not type(expected) or metadata[field] != expected:
            raise ValueError("Unsupported hcube " + field)
    shape = metadata["shape"]
    if not isinstance(shape, list) or len(shape) > 64 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Invalid hcube shape")
    names = metadata["axis_names"]
    if (not isinstance(names, list) or len(names) != len(shape)
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Invalid hcube axis names")
    scale = metadata["amplitude_scale"]
    try:
        finite_scale = type(scale) in (int, float) and math.isfinite(scale) and scale >= 0
    except OverflowError:
        finite_scale = False
    if not finite_scale:
        raise ValueError("Invalid hcube amplitude scale")
    expected = math.prod(shape) * 3
    if expected > payload_limit or type(metadata["raw_bytes"]) is not int or metadata["raw_bytes"] != expected:
        raise ValueError("Invalid hcube payload allocation")
    payload = data[12 + size:-32]
    if type(metadata["payload_bytes"]) is not int or metadata["payload_bytes"] != len(payload):
        raise ValueError("hcube payload size mismatch")
    if metadata["compression"] == "zlib":
        raw = _decompress(payload, expected, "hcube")
    elif metadata["compression"] == "none":
        raw = payload
    else:
        raise ValueError("Unsupported hcube compression")
    if len(raw) != expected:
        raise ValueError("hcube decoded payload size mismatch")
    codes = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape((*shape, 3)).copy()).to(device)
    return HolographicCube(codes, metadata["amplitude_scale"], names)
