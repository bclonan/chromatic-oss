"""Reversible cube geometry over integer chromatic codes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import math
import operator
from pathlib import Path
import zipfile

import numpy as np
import torch

from .codecs import ScalarCodec
from .codecs.base import INTEGER_DTYPES
from .serialization.format import (
    MAX_HEADER, MAX_PAYLOAD, _atomic_write, _budget, _decode_file, _encode_file, _finite_json,
    _open_archive, _read_member,
)
from .tensor import ChromaTensor


class ChromaCube:
    """A tensor with stable cell identities and reversible plane rotations.

    A cell is one logical scalar, including all of its codec channels. Its ID
    records its original row-major position. Turning a layer moves codes and
    IDs together. Axes may be integer indices or the cube's unique axis names.
    No geometry operation decodes the stored values.
    """

    def __init__(self, tensor: ChromaTensor, cell_ids: torch.Tensor | None = None,
                 axis_names: Sequence[str] | None = None):
        if not isinstance(tensor, ChromaTensor):
            raise TypeError("ChromaCube requires a ChromaTensor")
        if tensor.ndim < 3 or any(size == 0 for size in tensor.shape):
            raise ValueError("A cube requires at least three nonempty dimensions")
        if isinstance(axis_names, str):
            raise ValueError("Axis names must be a sequence of names")
        names = tuple(f"axis_{axis}" for axis in range(tensor.ndim)) if axis_names is None else tuple(axis_names)
        if (len(names) != tensor.ndim or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Axis names must be unique nonempty strings, one per dimension")
        if cell_ids is None:
            cell_ids = torch.arange(tensor.numel(), dtype=torch.int64, device=tensor.device).reshape(tensor.shape)
        else:
            if not isinstance(cell_ids, torch.Tensor) or cell_ids.dtype not in INTEGER_DTYPES:
                raise TypeError("Cell IDs must be an integer torch.Tensor")
            if cell_ids.shape != tensor.shape or cell_ids.device != tensor.device:
                raise ValueError("Cell IDs must have the tensor's shape and device")
            cell_ids = cell_ids.detach().to(torch.int64)
            expected = torch.arange(tensor.numel(), device=tensor.device)
            if not torch.equal(torch.sort(cell_ids.flatten()).values, expected):
                raise ValueError("Cell IDs must be a permutation of the original row-major positions")
        self.tensor = tensor
        self.cell_ids = cell_ids
        self.axis_names = names

    @classmethod
    def from_tensor(cls, x: torch.Tensor, codec: ScalarCodec | None = None,
                    axis_names: Sequence[str] | None = None) -> "ChromaCube":
        return cls(ChromaTensor.from_tensor(x, codec), axis_names=axis_names)

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def _wrap(self, tensor: ChromaTensor, cell_ids: torch.Tensor) -> "ChromaCube":
        # Trusted permutations preserve the constructor's identity checks.
        result = object.__new__(ChromaCube)
        result.tensor, result.cell_ids, result.axis_names = tensor, cell_ids, self.axis_names
        return result

    def _axis(self, axis: int | str) -> int:
        if isinstance(axis, str):
            try:
                return self.axis_names.index(axis)
            except ValueError as error:
                raise ValueError(f"Unknown cube axis {axis!r}") from error
        if isinstance(axis, bool):
            raise TypeError("Cube axes must be integer indices or names")
        try:
            value = operator.index(axis)
        except TypeError as error:
            raise TypeError("Cube axes must be integer indices or names") from error
        if not -self.ndim <= value < self.ndim:
            raise IndexError("Cube axis is outside the tensor")
        return value % self.ndim

    def _axes(self, axes: Sequence[int | str], count: int) -> tuple[int, ...]:
        if isinstance(axes, str) or not isinstance(axes, Sequence) or len(axes) != count:
            raise ValueError(f"Select exactly {count} cube axes")
        normalized = tuple(self._axis(axis) for axis in axes)
        if len(set(normalized)) != count:
            raise ValueError("Selected cube axes must be distinct")
        return normalized

    def _index(self, axis: int, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("Layer indices must be integers")
        try:
            value = operator.index(value)
        except TypeError as error:
            raise TypeError("Layer indices must be integers") from error
        if not -self.shape[axis] <= value < self.shape[axis]:
            raise IndexError("Layer index is outside the cube axis")
        return value % self.shape[axis]

    @staticmethod
    def _turns(turns: int) -> int:
        if isinstance(turns, bool):
            raise TypeError("Turns must be an integer number of quarter turns")
        try:
            return operator.index(turns)
        except TypeError as error:
            raise TypeError("Turns must be an integer number of quarter turns") from error

    def twist(self, axes: Sequence[int | str] = (0, 1), layer_axis: int | str | None = 2,
              layer_index: int = 0, turns: int = 1) -> "ChromaCube":
        """Return a new cube after quarter turns in the ordered axis plane.

        Positive turns use torch.rot90's counterclockwise convention. A layer
        turn fixes one other coordinate and rotates all remaining combinations
        of hidden coordinates. With layer_axis=None, every cell participates.
        Odd quarter turns require equal plane dimensions so shape stays fixed.
        Half turns also work for rectangular planes.
        """
        plane = self._axes(axes, 2)
        turns = self._turns(turns)
        if turns % 2 and self.shape[plane[0]] != self.shape[plane[1]]:
            raise ValueError("Odd quarter turns require equal plane dimensions")
        index = [slice(None)] * self.ndim
        if layer_axis is not None:
            layer = self._axis(layer_axis)
            if layer in plane:
                raise ValueError("The layer axis must be distinct from the rotation plane")
            index[layer] = self._index(layer, layer_index)
            rotation_axes = tuple(axis - (axis > layer) for axis in plane)
        else:
            rotation_axes = plane
        # Rotate byte blocks so unsigned word storage also works on PyTorch 2.3.
        # The value and channel dimensions stay in place; no numerical decode occurs.
        source_bytes = self.tensor.codes.contiguous().view(torch.uint8)
        code_bytes = source_bytes.clone()
        identities = self.cell_ids.clone()
        code_index = (*index, slice(None))
        id_index = tuple(index)
        code_bytes[code_index] = torch.rot90(source_bytes[code_index], turns % 4, rotation_axes)
        identities[id_index] = torch.rot90(self.cell_ids[id_index], turns % 4, rotation_axes)
        codes = code_bytes.view(self.tensor.codes.dtype)
        return self._wrap(self.tensor._wrap(codes), identities)

    def inverse_twist(self, axes: Sequence[int | str] = (0, 1), layer_axis: int | str | None = 2,
                      layer_index: int = 0, turns: int = 1) -> "ChromaCube":
        """Undo a twist with the same plane, layer, index, and turn count."""
        return self.twist(axes, layer_axis, layer_index, -self._turns(turns))

    def slice3d(self, axes: Sequence[int | str] = (0, 1, 2),
                fixed: Mapping[int | str, int] | None = None) -> dict:
        """Return codes and IDs for a 3D slice in the requested axis order.

        fixed supplies coordinates for omitted axes. Unspecified coordinates
        default to zero. The returned tensor and cell_ids share the same
        ordering. They are views of the cube where PyTorch indexing permits it.
        """
        selected = self._axes(axes, 3)
        if fixed is not None and not isinstance(fixed, Mapping):
            raise TypeError("Fixed slice coordinates must be a mapping")
        coordinates = {axis: 0 for axis in range(self.ndim) if axis not in selected}
        supplied = set()
        for axis, value in (fixed or {}).items():
            normalized = self._axis(axis)
            if normalized in selected:
                raise ValueError("Fixed coordinates cannot select a displayed axis")
            if normalized in supplied:
                raise ValueError("A fixed axis was supplied more than once")
            coordinates[normalized] = self._index(normalized, value)
            supplied.add(normalized)
        index = tuple(coordinates.get(axis, slice(None)) for axis in range(self.ndim))
        ascending = sorted(selected)
        order = tuple(ascending.index(axis) for axis in selected)
        tensor = self.tensor[index].permute(*order)
        identities = self.cell_ids[index].permute(*order)
        return {"tensor": tensor, "cell_ids": identities, "axes": selected,
                "axis_names": tuple(self.axis_names[axis] for axis in selected), "fixed": coordinates}

    def decode(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.tensor.decode(dtype)

    def to(self, device=None, *, decoded_dtype: torch.dtype | None = None) -> "ChromaCube":
        tensor = self.tensor.to(device, decoded_dtype=decoded_dtype)
        return self._wrap(tensor, self.cell_ids.to(device=tensor.device))

    def save(self, path, compress: bool = True) -> int:
        """Atomically save a checked ZIP container without Python objects."""
        tensor_data = _encode_file(self.tensor, compress=False)
        ids_data = self.cell_ids.detach().cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()
        manifest = {"format": "chromacube", "version": 1, "shape": list(self.shape),
                    "axis_names": list(self.axis_names), "ids_dtype": "int64",
                    "tensor_sha256": hashlib.sha256(tensor_data).hexdigest(),
                    "ids_sha256": hashlib.sha256(ids_data).hexdigest()}
        metadata = json.dumps(manifest, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
        if len(tensor_data) + len(ids_data) > MAX_PAYLOAD or len(metadata) > MAX_HEADER:
            raise ValueError("Cube exceeds the supported archive size")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as archive:
            archive.writestr("tensor.ctensor", tensor_data)
            archive.writestr("cell_ids.bin", ids_data)
            archive.writestr("manifest.json", metadata)
            archive.writestr("manifest.sha256", hashlib.sha256(metadata).hexdigest())
        data = output.getvalue()
        _atomic_write(path, data)
        return len(data)

    @classmethod
    def load(cls, path, device="cpu", *, max_file_bytes=None, max_header_bytes=None,
             max_payload_bytes=None, max_expanded_bytes=None) -> "ChromaCube":
        """Load codes, axis names, and cell identities after shape and checksum checks."""
        header_limit = _budget(max_header_bytes, MAX_HEADER, "max_header_bytes")
        payload_limit = _budget(max_payload_bytes, MAX_PAYLOAD, "max_payload_bytes")
        expanded_limit = _budget(max_expanded_bytes, payload_limit + header_limit, "max_expanded_bytes")
        expected_files = {"tensor.ctensor", "cell_ids.bin", "manifest.json", "manifest.sha256"}
        try:
            with _open_archive(path, max_file_bytes=max_file_bytes, max_header_bytes=header_limit,
                               max_expanded_bytes=expanded_limit, max_entries=4) as archive:
                items = archive.infolist()
                if len(items) != 4 or {item.filename for item in items} != expected_files:
                    raise ValueError("Cube archive contains missing, duplicate, or unknown entries")
                sizes = {item.filename: item.file_size for item in items}
                if (sizes["manifest.json"] > header_limit or sizes["manifest.sha256"] != 64
                        or sizes["cell_ids.bin"] > payload_limit):
                    raise ValueError("Cube expanded size is too large")
                metadata = _read_member(archive, "manifest.json", header_limit)
                if hashlib.sha256(metadata).hexdigest().encode() != _read_member(archive, "manifest.sha256", 64):
                    raise ValueError("Cube manifest checksum mismatch")
                manifest = _finite_json(metadata)
                required = {"format", "version", "shape", "axis_names", "ids_dtype", "tensor_sha256", "ids_sha256"}
                if not isinstance(manifest, dict) or set(manifest) != required:
                    raise ValueError("Invalid cube manifest")
                if (manifest["format"] != "chromacube" or type(manifest["version"]) is not int
                        or manifest["version"] != 1 or manifest["ids_dtype"] != "int64"):
                    raise ValueError("Unsupported cube format or version")
                shape = manifest["shape"]
                if (not isinstance(shape, list) or not 3 <= len(shape) <= 64
                        or any(type(size) is not int or size < 1 for size in shape)
                        or math.prod(shape) * 8 != sizes["cell_ids.bin"]):
                    raise ValueError("Invalid cube shape or identity byte count")
                names = manifest["axis_names"]
                if (not isinstance(names, list) or len(names) != len(shape)
                        or any(not isinstance(name, str) or not name.strip() for name in names)
                        or len(set(names)) != len(names)):
                    raise ValueError("Invalid cube axis names")
                tensor_data = _read_member(archive, "tensor.ctensor", expanded_limit)
                ids_data = _read_member(archive, "cell_ids.bin", payload_limit)
                if (hashlib.sha256(tensor_data).hexdigest() != manifest["tensor_sha256"]
                        or hashlib.sha256(ids_data).hexdigest() != manifest["ids_sha256"]):
                    raise ValueError("Cube payload checksum mismatch")
        except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid cube archive") from error
        tensor = _decode_file(tensor_data, device=device, max_header_bytes=header_limit,
                              max_payload_bytes=payload_limit - len(ids_data))
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError("Cube tensor shape differs from manifest")
        identities = torch.from_numpy(np.frombuffer(ids_data, dtype="<i8").copy().reshape(shape)).to(device)
        return cls(tensor, identities, manifest["axis_names"])

    def __repr__(self) -> str:
        return f"ChromaCube(shape={tuple(self.shape)}, axis_names={self.axis_names!r}, device={self.device})"
