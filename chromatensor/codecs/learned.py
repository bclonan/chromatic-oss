"""Scalar Lloyd codebooks and an RGBD codebook with a learned residual modifier."""
from __future__ import annotations

import math

import torch
from torch import nn

from .base import ScalarCodec, finite_values, id_dtype, palette_for_size, unpack_rgb, validate_codes, validate_domain


_FLOAT_DTYPES = {str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}


class LearnedChromaCodebook(nn.Module):
    """Learned scalar centroids labeled by unique RGB symbols.

    Fitting uses one-dimensional Lloyd updates and quantile initialization.
    Nearest-centroid assignment uses binary search rather than allocating an
    observations-by-vocabulary distance matrix. It supports 65,536 entries.
    Colors label centroids; Lloyd fitting optimizes numerical error rather
    than perceptual RGB distance.
    """

    def __init__(self, size: int = 256, minimum: float = -1, maximum: float = 1,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if isinstance(size, bool) or int(size) != size or not 2 <= size <= 65536:
            raise ValueError("size must be an integer in [2,65536]")
        self.size = int(size)
        self.minimum, self.maximum = validate_domain(minimum, maximum)
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in _FLOAT_DTYPES.values():
            raise ValueError("Centroid dtype must be float16, bfloat16, float32 or float64")
        bounds = torch.tensor([self.minimum, self.maximum], dtype=dtype)
        if not torch.isfinite(bounds).all() or bounds[0] == bounds[1]:
            raise ValueError("Centroid domain endpoints must be distinct finite values in the centroid dtype")
        values = torch.linspace(self.minimum, self.maximum, self.size, dtype=dtype)
        if not torch.isfinite(values).all():
            raise ValueError("Centroid domain cannot produce finite values in the centroid dtype")
        self.values = nn.Parameter(values)
        self.register_buffer("palette", palette_for_size(self.size))
        self.fit_history: list[float] = []

    @property
    def centers(self) -> torch.Tensor:
        return self.values

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = finite_values(x).clamp(self.minimum, self.maximum)
        sorted_values, order = self.values.detach().to(device=x.device, dtype=x.dtype).sort()
        boundaries = sorted_values[:-1] + (sorted_values[1:] - sorted_values[:-1]) / 2
        positions = torch.bucketize(x.contiguous(), boundaries)
        return order[positions]

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(ids)
        validated = validate_codes(ids.unsqueeze(-1), 1, self.size - 1)[..., 0]
        return self.values.to(ids.device)[validated]

    def colors(self, ids: torch.Tensor) -> torch.Tensor:
        validated = validate_codes(torch.as_tensor(ids).unsqueeze(-1), 1, self.size - 1)[..., 0]
        return self.palette.to(validated.device)[validated]

    def ids_from_colors(self, colors: torch.Tensor) -> torch.Tensor:
        keys = unpack_rgb(colors)
        palette_keys = unpack_rgb(self.palette.to(keys.device))
        sorted_keys, order = palette_keys.sort()
        positions = torch.searchsorted(sorted_keys, keys.contiguous())
        valid_positions = positions.clamp(max=self.size - 1)
        if ((positions >= self.size) | (sorted_keys[valid_positions] != keys)).any():
            raise ValueError("RGB symbol is absent from the learned codebook")
        return order[valid_positions]

    @torch.no_grad()
    def fit(self, samples: torch.Tensor, iterations: int = 20, seed: int = 0,
            maximum_samples: int = 131072) -> "LearnedChromaCodebook":
        if iterations < 1 or maximum_samples < 2:
            raise ValueError("iterations must be positive and maximum_samples at least two")
        values = finite_values(samples).flatten().to("cpu").clamp(self.minimum, self.maximum)
        if values.numel() == 0:
            raise ValueError("A learned codebook needs nonempty fit samples")
        if values.numel() > maximum_samples:
            generator = torch.Generator().manual_seed(int(seed))
            values = values[torch.randperm(values.numel(), generator=generator)[:maximum_samples]]
        ordered = values.sort().values
        # Quantile indices avoid torch.quantile's O(K*N) behavior for large K.
        indices = torch.linspace(0, ordered.numel() - 1, self.size).round().long()
        centers = ordered[indices].clone()
        self.fit_history = []
        for _ in range(iterations):
            boundaries = centers[:-1] + (centers[1:] - centers[:-1]) / 2
            assignment = torch.bucketize(values, boundaries)
            counts = torch.bincount(assignment, minlength=self.size)
            sums = torch.zeros(self.size, dtype=torch.float64).scatter_add_(0, assignment, values)
            updated = torch.where(counts > 0, sums / counts.clamp(min=1), centers).sort().values
            error = (values - updated[torch.bucketize(values, updated[:-1] + (updated[1:] - updated[:-1]) / 2)]).square().mean()
            self.fit_history.append(float(error))
            changed = not torch.equal(updated, centers)
            centers = updated
            if not changed:
                break
        self.values.copy_(centers.to(device=self.values.device, dtype=self.values.dtype))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x)).to(dtype=x.dtype)

    def to_config(self) -> dict:
        return {"size": self.size, "minimum": self.minimum, "maximum": self.maximum,
                "dtype": str(self.values.dtype),
                "values": self.values.detach().cpu().tolist(), "palette": self.palette.cpu().tolist(),
                "fit_history": list(self.fit_history)}

    @classmethod
    def from_config(cls, config: dict) -> "LearnedChromaCodebook":
        dtype = _FLOAT_DTYPES.get(config.get("dtype", "torch.float32"))
        if dtype is None:
            raise ValueError("Unsupported learned centroid dtype")
        result = cls(config["size"], config["minimum"], config["maximum"], dtype=dtype)
        values = torch.tensor(config["values"], dtype=result.values.dtype)
        palette = torch.tensor(config["palette"], dtype=torch.int64)
        if values.shape != result.values.shape or not torch.isfinite(values).all():
            raise ValueError("Invalid learned centroid table")
        if palette.shape != (result.size, 3) or ((palette < 0) | (palette > 255)).any():
            raise ValueError("Invalid learned palette table")
        if unpack_rgb(palette.to(torch.uint8)).unique().numel() != result.size:
            raise ValueError("Learned RGB palette symbols must be unique")
        with torch.no_grad():
            result.values.copy_(values)
            result.palette.copy_(palette.to(torch.uint8))
        result.fit_history = [float(value) for value in config.get("fit_history", [])]
        return result


class LearnedChromaEncoder(nn.Module):
    """Nearest-centroid scalar encoder with explicit integer token output."""

    def __init__(self, codebook: LearnedChromaCodebook):
        super().__init__()
        self.codebook = codebook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.codebook.encode(x)


class LearnedChromaDecoder(nn.Module):
    """Trainable centroid lookup decoder. Gradients update selected values."""

    def __init__(self, codebook: LearnedChromaCodebook):
        super().__init__()
        self.codebook = codebook

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.codebook.decode(ids)


class LearnedCodec(ScalarCodec):
    name = "learned"
    channels = 1

    def __init__(self, size: int = 256, minimum: float = -1, maximum: float = 1,
                 codebook: LearnedChromaCodebook | None = None):
        self.codebook = codebook if codebook is not None else LearnedChromaCodebook(size, minimum, maximum)
        self.size = self.codebook.size
        self.minimum, self.maximum = self.codebook.minimum, self.codebook.maximum
        self.bits_per_value = math.ceil(math.log2(self.size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.codebook.encode(x).to(id_dtype(self.size)).unsqueeze(-1)

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        return validate_codes(codes, 1, self.size - 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        ids = self.validate(codes)[..., 0]
        return self.codebook.decode(ids)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        ids = self.validate(codes)[..., 0]
        return self.codebook.colors(ids)

    def fit(self, samples: torch.Tensor, iterations: int = 20, seed: int = 0,
            maximum_samples: int = 131072) -> "LearnedCodec":
        self.codebook.fit(samples, iterations, seed, maximum_samples)
        return self

    def to_config(self) -> dict:
        return {"name": self.name, "codebook": self.codebook.to_config()}


class LearnedRGBDCodec(ScalarCodec):
    """RGB labels a learned coarse centroid; D selects a learned residual.

    The modifier has one byte and one global residual table. There is no
    confidence claim: D literally means an additive reconstruction correction.
    Both tables fit numerical reconstruction error.
    """

    name = "learned_rgbd"
    channels = 4
    bits_per_value = 32

    def __init__(self, size: int = 64, modifier_size: int = 16,
                 minimum: float = -1, maximum: float = 1,
                 codebook: LearnedChromaCodebook | None = None,
                 modifier: LearnedChromaCodebook | None = None):
        if not 2 <= modifier_size <= 256:
            raise ValueError("modifier_size must lie in [2,256]")
        self.codebook = codebook if codebook is not None else LearnedChromaCodebook(size, minimum, maximum)
        width = self.codebook.maximum - self.codebook.minimum
        self.modifier = modifier if modifier is not None else LearnedChromaCodebook(modifier_size, -width / (size - 1), width / (size - 1))
        self.size, self.modifier_size = self.codebook.size, self.modifier.size
        self.minimum, self.maximum = self.codebook.minimum, self.codebook.maximum

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        values = finite_values(x).clamp(self.minimum, self.maximum)
        ids = self.codebook.encode(values)
        residual = values - self.codebook.decode(ids).double()
        modifiers = self.modifier.encode(residual).to(torch.uint8)
        return torch.cat((self.codebook.colors(ids), modifiers.unsqueeze(-1)), -1)

    def validate(self, codes: torch.Tensor) -> torch.Tensor:
        c = validate_codes(codes, 4)
        if (c[..., 3] >= self.modifier_size).any():
            raise ValueError("D selects an absent learned residual")
        self.codebook.ids_from_colors(codes[..., :3])
        return c

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        c = self.validate(codes)
        ids = self.codebook.ids_from_colors(codes[..., :3])
        return (self.codebook.decode(ids) + self.modifier.decode(c[..., 3])).clamp(self.minimum, self.maximum)

    def colors(self, codes: torch.Tensor) -> torch.Tensor:
        self.validate(codes)
        return codes[..., :3].to(torch.uint8)

    @torch.no_grad()
    def fit(self, samples: torch.Tensor, iterations: int = 20, seed: int = 0,
            maximum_samples: int = 131072) -> "LearnedRGBDCodec":
        self.codebook.fit(samples, iterations, seed, maximum_samples)
        samples = finite_values(samples).clamp(self.minimum, self.maximum)
        ids = self.codebook.encode(samples)
        residual = samples - self.codebook.decode(ids).double()
        bound = max(1e-8, float(residual.abs().max()) * 1.001)
        self.modifier = LearnedChromaCodebook(self.modifier_size, -bound, bound)
        self.modifier.fit(residual, iterations, seed, maximum_samples)
        return self

    def to_config(self) -> dict:
        return {"name": self.name, "codebook": self.codebook.to_config(), "modifier": self.modifier.to_config()}

