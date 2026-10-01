"""Bounded, stateless cube operations backed by the numerical library."""
from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path
import json

import torch

MAX_CELLS = 4096


def _integer(value, name, minimum=0, maximum=4096):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def _cube_state(cube, source, seed, twist_count=0):
    return {"shape": list(cube.shape), "axis_names": list(cube.axis_names), "codes": cube.tensor.codes.reshape(-1, cube.tensor.codec.channels).cpu().tolist(), "cell_ids": cube.cell_ids.flatten().cpu().tolist(), "codec": cube.tensor.codec.to_config(), "source": source, "initial_seed": seed, "twist_count": twist_count}


def _restore(state):
    from chromatensor import ChromaTensor
    from chromatensor.codecs import codec_from_config
    from chromatensor.cube import ChromaCube
    if not isinstance(state, dict):
        raise ValueError("state must contain an existing cube")
    shape = state.get("shape")
    if not isinstance(shape, list) or not 3 <= len(shape) <= 6:
        raise ValueError("cube shape must have three to six dimensions")
    for extent in shape:
        _integer(extent, "axis size", 2, 8)
    count = math.prod(shape)
    if count > MAX_CELLS:
        raise ValueError("cube exceeds 4096 cells")
    config = state.get("codec", {})
    if not isinstance(config, dict) or config.get("name") != "rgb":
        raise ValueError("the local cube inspector accepts bounded RGB codes")
    if set(config) != {"name", "minimum", "maximum", "logarithmic"}:
        raise ValueError("cube codec metadata must match the RGB format")
    for field in ("minimum", "maximum"):
        value = config[field]
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e30:
            raise ValueError("cube codec domain must stay within the inspector's finite float32 range")
    if type(config["logarithmic"]) is not bool:
        raise ValueError("cube codec logarithmic flag must be true or false")
    codec = codec_from_config(config)
    codes = state.get("codes")
    ids = state.get("cell_ids")
    if not isinstance(codes, list) or len(codes) != count or any(not isinstance(row, list) or len(row) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in row) for row in codes):
        raise ValueError("cube codes must contain three byte channels per cell")
    if not isinstance(ids, list) or len(ids) != count or any(type(v) is not int for v in ids) or sorted(ids) != list(range(count)):
        raise ValueError("cell identities must be a permutation of the original cell IDs")
    names = state.get("axis_names")
    if not isinstance(names, list) or len(names) != len(shape) or any(not isinstance(n, str) or not 1 <= len(n) <= 40 for n in names):
        raise ValueError("one short name is required per axis")
    seed = _integer(state.get("initial_seed", 7), "seed", 0, 2**31 - 1)
    twist_count = _integer(state.get("twist_count", 0), "twist count", 0, 10000)
    source = state.get("source", "Imported color cube")
    if not isinstance(source, str) or len(source) > 240:
        raise ValueError("source label is too long")
    tensor = ChromaTensor(torch.tensor(codes, dtype=torch.uint8).reshape(*shape, 3), codec)
    cube = ChromaCube(tensor, axis_names=names, cell_ids=torch.tensor(ids).reshape(shape))
    return cube, source, seed, twist_count


def _create(payload, results_path=None):
    from chromatensor import RGBCodec
    from chromatensor.cube import ChromaCube
    dimensions = _integer(payload.get("dimensions", 4), "dimensions", 3, 6)
    size = _integer(payload.get("size", 3), "size", 2, 8)
    seed = _integer(payload.get("seed", 7), "seed", 0, 2**31 - 1)
    shape = (size,) * dimensions
    if math.prod(shape) > MAX_CELLS:
        raise ValueError("cube exceeds 4096 cells; use a smaller side length")
    signal = payload.get("signal", "smooth")
    source_key = payload.get("source_key")
    if source_key:
        if not isinstance(source_key, str) or len(source_key) > 160 or results_path is None:
            raise ValueError("invalid recorded tensor source")
        result = json.loads(Path(results_path).read_text(encoding="utf-8"))
        sample = next((s for s in result.get("tensor_samples", []) if s["name"] == source_key), None)
        if sample is None or len(sample["values"]) < math.prod(shape):
            raise ValueError("recorded sample has too few values for this cube")
        values = torch.tensor(sample["values"][:math.prod(shape)]).reshape(shape)
        source = "Recorded tensor sample reshaped into cube: " + source_key
    elif signal == "noise":
        values = torch.rand(shape, generator=torch.Generator().manual_seed(seed)) * 2 - 1
        source = "Seeded random demonstration tensor; not trained model data"
    elif signal == "smooth":
        coordinates = torch.meshgrid(*(torch.arange(size).double() / size for _ in range(dimensions)), indexing="ij")
        values = sum(torch.sin(2 * math.pi * coordinate + axis * 0.37) for axis, coordinate in enumerate(coordinates)) / dimensions
        values += 0.2 * torch.cos(2 * math.pi * (coordinates[0] + coordinates[1]))
        values = (values / values.abs().max().clamp_min(1)).float()
        source = "Analytic periodic demonstration tensor; not trained model data"
    else:
        raise ValueError("unknown cube signal")
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    cube = ChromaCube.from_tensor(values, RGBCodec(lo, hi), axis_names=[f"axis_{i}" for i in range(dimensions)])
    return cube, source, seed, 0


def _display_axes(payload, dimensions):
    axes = payload.get("view_axes", [0, 1, 2])
    if not isinstance(axes, list) or len(axes) != 3 or len(set(axes)) != 3:
        raise ValueError("select three distinct display axes")
    for axis in axes:
        _integer(axis, "display axis", 0, dimensions - 1)
    return axes


def _fixed(payload, shape, axes):
    supplied = payload.get("fixed", [0] * len(shape))
    if isinstance(supplied, list):
        if len(supplied) != len(shape):
            raise ValueError("fixed coordinates must cover every axis")
        supplied = {i: supplied[i] for i in range(len(shape)) if i not in axes}
    elif isinstance(supplied, dict):
        normalized = {}
        for key, value in supplied.items():
            if type(key) is int:
                axis = key
            elif isinstance(key, str) and key.isascii() and key.isdecimal():
                axis = int(key)
            else:
                raise ValueError("invalid fixed coordinate axis")
            if axis in normalized:
                raise ValueError("duplicate fixed coordinate axis")
            normalized[axis] = value
        supplied = normalized
    else:
        raise ValueError("fixed coordinates must be a list or object")
    fixed = {}
    for axis, value in supplied.items():
        _integer(axis, "fixed axis", 0, len(shape) - 1)
        if axis in axes:
            continue
        fixed[axis] = _integer(value, "fixed coordinate", 0, shape[axis] - 1)
    for axis in range(len(shape)):
        if axis not in axes:
            fixed.setdefault(axis, 0)
    return fixed


def inspect_cube(payload, results_path=None):
    if not isinstance(payload, dict):
        raise ValueError("Cube request must be an object")
    operation = payload.get("operation", "create")
    if operation in ("create", "reset"):
        cube, source, seed, twists = _create(payload, results_path)
    elif operation in ("view", "twist", "decode"):
        cube, source, seed, twists = _restore(payload.get("state"))
    else:
        raise ValueError("unknown cube operation")
    axes = _display_axes(payload, cube.ndim)
    fixed = _fixed(payload, cube.shape, axes)
    if operation == "twist":
        if twists == 10000:
            raise ValueError("cube twist count exceeds the inspector limit; reset the cube")
        plane = payload.get("axes", [0, 1])
        if not isinstance(plane, list) or len(plane) != 2 or plane[0] == plane[1]:
            raise ValueError("twist needs two distinct plane axes")
        for axis in plane:
            _integer(axis, "twist axis", 0, cube.ndim - 1)
        layer_axis = payload.get("layer_axis", 2)
        if layer_axis is not None:
            _integer(layer_axis, "layer axis", 0, cube.ndim - 1)
        layer_extent = cube.shape[layer_axis] if layer_axis is not None else max(cube.shape)
        layer_index = _integer(payload.get("layer_index", 0), "layer index", 0, layer_extent - 1)
        turns = _integer(payload.get("turns", 1), "quarter turns", -4, 4)
        cube = cube.twist(axes=tuple(plane), layer_axis=layer_axis, layer_index=layer_index, turns=turns)
        twists += 1
    values = cube.decode().double()
    rgb = cube.tensor.rgb
    all_cells = []
    for coordinate in itertools.product(*(range(size) for size in cube.shape)):
        all_cells.append({"coordinate": [coordinate[axis] for axis in axes], "full_coordinate": list(coordinate), "id": int(cube.cell_ids[coordinate]), "value": float(values[coordinate]), "rgb": rgb[coordinate].tolist(), "codes": cube.tensor.codes[coordinate].tolist()})
    view_cells = [cell for cell in all_cells if all(cell["full_coordinate"][axis] == value for axis, value in fixed.items())]
    state = _cube_state(cube, source, seed, twists)
    digest = hashlib.sha256(cube.tensor.codes.cpu().numpy().tobytes() + cube.cell_ids.cpu().numpy().tobytes()).hexdigest()
    moved = int((cube.cell_ids.flatten() != torch.arange(cube.cell_ids.numel())).sum())
    response = {"state": state, "all_cells": all_cells, "view": {"cells": view_cells, "shape": [cube.shape[axis] for axis in axes], "axes": axes, "fixed": {str(key): value for key, value in fixed.items()}, "axis_names": [cube.axis_names[axis] for axis in axes]}, "metrics": {"dimensions": cube.ndim, "total_cells": cube.tensor.numel(), "visible_cells": len(view_cells), "payload_bytes": cube.tensor.storage_nbytes, "identity_bytes": cube.cell_ids.numel() * cube.cell_ids.element_size(), "moved_cells": moved, "twist_count": twists, "state_sha256": digest, "twist_execution": "integer tensor rotation, no scalar decoding during the move", "display_execution": "floating point numerical decoding and 3D projection"}, "source": source}
    holographic_enabled = payload.get("holographic", True)
    if type(holographic_enabled) is not bool:
        raise ValueError("holographic must be true or false")
    if holographic_enabled:
        coverage = payload.get("coverage", 1)
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError("coverage must lie between zero and one")
        from chromatensor.holographic import HolographicCube
        from chromatensor.codecs.hsv import hsv_to_rgb
        hologram = HolographicCube.from_tensor(values, axis_names=cube.axis_names)
        mask = hologram.erasure_mask(1 - coverage, seed=seed)
        reconstructed = hologram.decode(mask=mask)
        colors = hologram.rgb
        phase, magnitude = hologram.phase, hologram.magnitudes
        lo, hi = float(values.min()), float(values.max())
        normalized = ((reconstructed - lo) / max(hi - lo, 1e-12)).clamp(0, 1)
        recovered_rgb = hsv_to_rgb(torch.stack((normalized * 0.83, torch.ones_like(normalized) * 0.8, torch.ones_like(normalized)), -1))
        spectral, recovered = [], []
        for coordinate in itertools.product(*(range(size) for size in cube.shape)):
            projected = [coordinate[axis] for axis in axes]
            spectral.append({"coordinate": projected, "full_coordinate": list(coordinate), "rgb": colors[coordinate].tolist(), "phase": float(phase[coordinate]), "magnitude": float(magnitude[coordinate]), "kept": bool(mask[coordinate])})
            recovered.append({"coordinate": projected, "full_coordinate": list(coordinate), "id": int(cube.cell_ids[coordinate]), "value": float(reconstructed[coordinate]), "rgb": recovered_rgb[coordinate].tolist(), "error": float(abs(reconstructed[coordinate] - values[coordinate]))})
        delta = reconstructed - values
        response["holographic"] = {"cells": spectral, "decoded_cells": recovered, "coverage": float(mask.double().mean()), "metrics": {"mae": float(delta.abs().mean()), "mse": float(delta.square().mean()), "max_error": float(delta.abs().max()), "payload_bytes": hologram.codes.numel() * hologram.codes.element_size(), "phase_bits": 8, "magnitude_bits": 16, "retained_coefficients": int(mask.sum()), "total_coefficients": mask.numel()}, "execution": "complex128 N-dimensional FFT, 8-bit phase and 16-bit magnitude packed in three uint8 channels, inverse FFT; floating point transform", "meaning": "each spectral coefficient combines information across the cube; erasure loses information and reconstruction is approximate"}
    return response
