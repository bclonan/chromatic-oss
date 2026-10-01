"""An integer-backed tensor wrapper with explicit numerical interoperation."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from .codecs import RGBDCodec, ScalarCodec
from .codecs.base import INTEGER_DTYPES


class ChromaTensor:
    """A value tensor stored only as integer codes and codec metadata.

    `dtype` means storage dtype, while `decoded_dtype` records the numerical
    dtype used for interoperation. Shape operations manipulate codes directly.
    Encoding is discrete and does not preserve an autograd graph; use the
    autograd package's fake quantization when training floating-point models.
    """

    def __init__(self, codes: torch.Tensor, codec: ScalarCodec,
                 shape: Sequence[int] | None = None, dtype: torch.dtype = torch.float32):
        if not isinstance(codes, torch.Tensor) or codes.dtype not in INTEGER_DTYPES:
            raise TypeError("ChromaTensor requires integer torch.Tensor storage")
        if codes.ndim < 1 or codes.shape[-1] != codec.channels:
            raise ValueError("Codes must have the codec's trailing channel dimension")
        logical_shape = tuple(codes.shape[:-1])
        if shape is not None and tuple(shape) != logical_shape:
            raise ValueError("Declared tensor shape differs from code storage")
        if not isinstance(dtype, torch.dtype) or dtype.is_complex:
            raise TypeError("Decoded dtype must be a real torch dtype")
        codec.validate(codes)
        self.codes = codes.detach()
        self.codec = codec
        self.decoded_dtype = dtype

    @classmethod
    def from_tensor(cls, x: torch.Tensor, codec: ScalarCodec | None = None) -> "ChromaTensor":
        x = torch.as_tensor(x)
        codec = RGBDCodec() if codec is None else codec
        numerical_dtype = x.dtype if x.is_floating_point() else torch.float32
        return cls(codec.encode(x), codec, dtype=numerical_dtype)

    @property
    def shape(self) -> torch.Size:
        return self.codes.shape[:-1]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.codes.dtype

    @property
    def device(self) -> torch.device:
        return self.codes.device

    @property
    def rgb(self) -> torch.Tensor:
        return self.codec.colors(self.codes)

    @property
    def rgbd(self) -> torch.Tensor:
        """The actual RGBD token bytes, including the required exponent channel."""
        if not isinstance(self.codec, RGBDCodec):
            raise ValueError("This tensor does not use the finite FP32 RGBD codec")
        return self.codes

    @property
    def storage_nbytes(self) -> int:
        return self.codes.numel() * self.codes.element_size()

    def numel(self) -> int:
        return math.prod(self.shape)

    def decode(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.codec.decode(self.codes).to(dtype=dtype or self.decoded_dtype)

    def to_torch(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.decode(dtype)

    def _wrap(self, codes: torch.Tensor) -> "ChromaTensor":
        # Trusted transformations keep the already validated symbols intact.
        # In particular, reshape and indexing need no floating-point decode.
        result = object.__new__(ChromaTensor)
        result.codes, result.codec, result.decoded_dtype = codes, self.codec, self.decoded_dtype
        return result

    def reshape(self, *shape: int | Sequence[int]) -> "ChromaTensor":
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], Sequence) else tuple(shape)
        if shape.count(-1) > 1 or any(size < -1 for size in shape):
            raise ValueError("Invalid reshape dimensions")
        return self._wrap(self.codes.reshape(*shape, self.codec.channels))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "ChromaTensor":
        if self.ndim == 0:
            return self.reshape(1)
        if not -self.ndim <= start_dim < self.ndim or not -self.ndim <= end_dim < self.ndim:
            raise IndexError("Flatten dimension is outside the value tensor")
        start, end = start_dim % self.ndim, end_dim % self.ndim
        if start > end:
            raise ValueError("start_dim must precede end_dim")
        return self.reshape(*self.shape[:start], math.prod(self.shape[start:end + 1]), *self.shape[end + 1:])

    def transpose(self, dim0: int, dim1: int) -> "ChromaTensor":
        if self.ndim == 0 or not -self.ndim <= dim0 < self.ndim or not -self.ndim <= dim1 < self.ndim:
            raise IndexError("Transpose dimension is outside the value tensor")
        return self._wrap(self.codes.transpose(dim0 % self.ndim, dim1 % self.ndim))

    def permute(self, *dims: int) -> "ChromaTensor":
        if any(not -self.ndim <= dim < self.ndim for dim in dims):
            raise IndexError("Permutation dimension is outside the value tensor")
        normalized = tuple(dim % self.ndim for dim in dims) if self.ndim else tuple(dims)
        if sorted(normalized) != list(range(self.ndim)):
            raise ValueError("permute needs each value dimension exactly once")
        return self._wrap(self.codes.permute(*normalized, self.ndim))

    @property
    def T(self) -> "ChromaTensor":
        return self.permute(*reversed(range(self.ndim)))

    def __getitem__(self, index) -> "ChromaTensor":
        index = index if isinstance(index, tuple) else (index,)
        def boolean_sequence(item):
            if isinstance(item, (bool, np.bool_)):
                return True
            return isinstance(item, (list, tuple)) and bool(item) and all(boolean_sequence(value) for value in item)

        def normalize_index(item):
            if isinstance(item, np.bool_):
                return bool(item)
            if ((isinstance(item, np.ndarray) and item.dtype == np.bool_)
                    or isinstance(item, (list, tuple)) and boolean_sequence(item)):
                return torch.as_tensor(item, dtype=torch.bool, device=self.device)
            return item

        index = tuple(normalize_index(item) for item in index)
        # A final ellipsis leaves the code channels intact, including when an
        # ellipsis or newaxis appears in the user's value-tensor index.
        def consumed_dimensions(item):
            if item is None or item is Ellipsis or isinstance(item, bool):
                return 0
            if isinstance(item, torch.Tensor) and item.dtype in (torch.bool, torch.uint8):
                return item.ndim
            return 1

        consuming = sum(consumed_dimensions(item) for item in index)
        if consuming > self.ndim:
            raise IndexError("Index would address chromatic channel storage")
        if sum(item is Ellipsis for item in index) > 1:
            raise IndexError("Only one ellipsis is allowed")
        expanded = []
        for item in index:
            if item is Ellipsis:
                expanded.extend([slice(None)] * (self.ndim - consuming))
            else:
                expanded.append(item)
        if not any(item is Ellipsis for item in index):
            expanded.extend([slice(None)] * (self.ndim - consuming))
        expanded.append(slice(None))
        # CPU advanced indexing of uint16 is absent in some PyTorch builds.
        # Valid scalar codecs use at most 32 bits. int64 also keeps uint64
        # storage values above 2**31 intact through the indexing fallback.
        storage = self.codes.to(torch.int64) if self.codes.dtype in (torch.uint16, torch.uint32, torch.uint64) else self.codes
        return self._wrap(storage[tuple(expanded)].to(self.codes.dtype))

    def to(self, device=None, *, decoded_dtype: torch.dtype | None = None) -> "ChromaTensor":
        if isinstance(device, torch.dtype):
            raise TypeError("to() preserves integer storage. Use decode(dtype=...) for numerical conversion")
        result = self._wrap(self.codes.to(device=device))
        if decoded_dtype is not None:
            if not isinstance(decoded_dtype, torch.dtype) or decoded_dtype.is_complex:
                raise TypeError("decoded_dtype must be a real torch dtype")
            result.decoded_dtype = decoded_dtype
        return result

    def cpu(self) -> "ChromaTensor":
        return self.to("cpu")

    def cuda(self, device=None) -> "ChromaTensor":
        return self.to(torch.device("cuda", device) if device is not None else "cuda")

    def clone(self) -> "ChromaTensor":
        return self._wrap(self.codes.clone())

    def __len__(self) -> int:
        if not self.ndim:
            raise TypeError("A scalar ChromaTensor has no length")
        return self.shape[0]

    @classmethod
    def cat(cls, tensors: Sequence["ChromaTensor"], dim: int = 0) -> "ChromaTensor":
        first = cls._compatible(tensors)
        if not -first.ndim <= dim < first.ndim:
            raise IndexError("Concatenation dimension is outside the value tensor")
        return first._wrap(torch.cat([tensor.codes for tensor in tensors], dim=dim % first.ndim))

    @classmethod
    def stack(cls, tensors: Sequence["ChromaTensor"], dim: int = 0) -> "ChromaTensor":
        first = cls._compatible(tensors)
        if not -first.ndim - 1 <= dim <= first.ndim:
            raise IndexError("Stack dimension is outside the value tensor")
        return first._wrap(torch.stack([tensor.codes for tensor in tensors], dim=dim % (first.ndim + 1)))

    @staticmethod
    def _compatible(tensors: Sequence["ChromaTensor"]) -> "ChromaTensor":
        if not tensors:
            raise ValueError("At least one ChromaTensor is required")
        first = tensors[0]
        if any(not first.codec.equivalent(tensor.codec) or first.decoded_dtype != tensor.decoded_dtype for tensor in tensors[1:]):
            raise ValueError("All ChromaTensors must use equivalent codecs and numerical dtypes")
        return first

    def __repr__(self) -> str:
        return (f"ChromaTensor(shape={tuple(self.shape)}, codec={self.codec.name!r}, "
                f"storage_dtype={self.dtype}, decoded_dtype={self.decoded_dtype}, "
                f"device={self.device}, payload_bytes={self.storage_nbytes})")


def cat(tensors: Sequence[ChromaTensor], dim: int = 0) -> ChromaTensor:
    return ChromaTensor.cat(tensors, dim)


def stack(tensors: Sequence[ChromaTensor], dim: int = 0) -> ChromaTensor:
    return ChromaTensor.stack(tensors, dim)

