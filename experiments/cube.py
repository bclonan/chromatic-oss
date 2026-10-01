"""Measure multidimensional turns and distributed chromatic reconstruction."""
from __future__ import annotations

import math
from pathlib import Path
import time

import torch

from chromatensor import ChromaCube, RGBCodec, save_chroma_checkpoint, load_chroma_checkpoint
from chromatensor.nn import convert_to_chroma, module_quantization_report
from .common import seed_all, error_metrics, dataset_hash, timed, PeakMemory
from .models import StandardMLP


def smooth_cubes(count, dimensions=4, size=3, seed=7):
    if type(count) is not int or count < 1 or type(dimensions) is not int or not 3 <= dimensions <= 6 or type(size) is not int or size < 2:
        raise ValueError("Use a positive count, three to six dimensions, and side length at least two")
    generator = torch.Generator().manual_seed(seed)
    grid = torch.meshgrid(*(torch.arange(size).double() / size for _ in range(dimensions)), indexing="ij")
    phase = torch.rand(count, dimensions + 1, generator=generator, dtype=torch.float64) * (2 * math.pi)
    amplitudes = torch.rand(count, dimensions + 1, generator=generator, dtype=torch.float64) * 0.5 + 0.2
    expanded = (count,) + (1,) * dimensions
    field = torch.zeros((count,) + (size,) * dimensions, dtype=torch.float64)
    for axis, coordinate in enumerate(grid):
        field += amplitudes[:, axis].reshape(expanded) * torch.sin(2 * math.pi * coordinate + phase[:, axis].reshape(expanded))
    field += amplitudes[:, -1].reshape(expanded) * torch.cos(2 * math.pi * (grid[0] + grid[1]) + phase[:, -1].reshape(expanded))
    field /= dimensions + 1
    return field.float()


def batched_polar(values):
    """Batch version of the cube phase8/magnitude16 codec, with per-cube scale."""
    dimensions = tuple(range(1, values.ndim))
    coefficients = torch.fft.fftn(values.double(), dim=dimensions, norm="ortho")
    magnitude = coefficients.abs()
    maximum = magnitude.amax(dim=dimensions, keepdim=True)
    scale = maximum
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    magnitude_ids = (magnitude / safe_scale * 65535).round().clamp(0, 65535).long()
    phase_ids = ((torch.angle(coefficients).remainder(2 * math.pi) / (2 * math.pi)) * 256).round().long().remainder(256)
    phase_ids = torch.where(magnitude_ids == 0, torch.zeros_like(phase_ids), phase_ids)
    reconstructed = torch.polar(magnitude_ids.double() / 65535 * scale, phase_ids.double() * (2 * math.pi / 256))
    codes = torch.stack((phase_ids, (magnitude_ids >> 8) & 255, magnitude_ids & 255), -1).to(torch.uint8)
    return codes, scale, reconstructed


def encoded_inputs(values, representation, coverage=0.5, seed=7):
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("coverage must be a finite fraction in [0, 1]")
    generator = torch.Generator().manual_seed(seed)
    flat_size = math.prod(values.shape[1:])
    keep = math.floor(flat_size * coverage)
    mask = torch.zeros(len(values), flat_size, dtype=torch.bool)
    for index in range(len(values)):
        mask[index, torch.randperm(flat_size, generator=generator)[:keep]] = True
    mask = mask.reshape(values.shape).to(values.device)
    if representation == "direct_rgb24":
        codec = RGBCodec(-1, 1)
        decoded = codec.decode(codec.encode(values)).float()
        incoming = decoded * mask
    else:
        codes, scale, numeric_coefficients = batched_polar(values)
        if representation == "holographic24":
            # Decode the actual packed chromatic payload. The numerical control
            # retains the same quantization levels without RGB labels.
            phase = codes[..., 0].double() * (2 * math.pi / 256)
            magnitude_ids = (codes[..., 1].long() << 8) | codes[..., 2].long()
            coefficients = torch.polar(magnitude_ids.double() / 65535 * scale, phase)
        elif representation == "numeric_polar24":
            coefficients = numeric_coefficients
        else:
            raise ValueError("unknown cube representation")
        incoming = torch.fft.ifftn(coefficients * mask, dim=tuple(range(1, values.ndim)), norm="ortho").real.float()
    # Include the retained-coordinate mask for both paths. It identifies spatial
    # cells for direct RGB and spectral coefficients for the distributed codec.
    return torch.cat((incoming.flatten(1), mask.flatten(1).float()), -1), mask


def train_cube_reconstructor(representation, seed, artifacts, epochs=24):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    from .provenance import unique_run_id
    artifact_dir = Path(artifacts) / unique_run_id(f"model-{representation}-s{seed}")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    torch.set_num_threads(4)
    train = smooth_cubes(1024, seed=seed)
    validation = smooth_cubes(256, seed=seed + 1)
    tx, train_mask = encoded_inputs(train, representation, seed=seed + 2)
    vx, val_mask = encoded_inputs(validation, representation, seed=seed + 3)
    ty, vy = train.flatten(1), validation.flatten(1)
    # All paired representations share initialization and the same quantized
    # neural architecture. Encoding remains a fixed preprocessing operation.
    seed_all(seed)
    model = convert_to_chroma(StandardMLP(tx.shape[1], 128, ty.shape[1]), bits=8, representation="chroma8")
    initial_hash = dataset_hash(*[parameter.detach() for parameter in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    generator = torch.Generator().manual_seed(seed + 4)
    curves = []
    with torch.no_grad():
        initial_loss = float((model(vx) - vy).square().mean())
    curves.append({"epoch": 0, "train_loss": None, "val_loss": initial_loss})
    with PeakMemory("cpu") as memory:
        training_seconds = 0.0
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            model.train()
            ordering = torch.randperm(len(tx), generator=generator)
            losses = []
            for offset in range(0, len(tx), 64):
                indices = ordering[offset:offset + 64]
                optimizer.zero_grad(set_to_none=True)
                loss = (model(tx[indices]) - ty[indices]).square().mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            training_seconds += time.perf_counter() - start
            model.eval()
            with torch.no_grad():
                validation_loss = float((model(vx) - vy).square().mean())
            curves.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": validation_loss})
    with torch.no_grad():
        prediction, inference_seconds = timed(lambda: model(vx), repeats=5)
        noise = torch.rand(validation.shape, generator=torch.Generator().manual_seed(seed + 5)) * 2 - 1
        noise_input, _ = encoded_inputs(noise, representation, seed=seed + 3)
        noise_prediction = model(noise_input)
        noise_mse = float((noise_prediction - noise.flatten(1)).square().mean())
        noise_zero_predictor_mse = float(noise.square().mean())
    path = artifact_dir / f"cube_{representation}_s{seed}.ctcheckpoint"
    checkpoint_bytes = save_chroma_checkpoint(model, path)
    reloaded = convert_to_chroma(StandardMLP(tx.shape[1], 128, ty.shape[1]), bits=8, representation="chroma8")
    load_chroma_checkpoint(path, reloaded)
    reloaded.eval()
    with torch.no_grad():
        reloaded_prediction = reloaded(vx)
    if not torch.equal(reloaded_prediction, prediction):
        raise AssertionError("Cube checkpoint changed reconstructed predictions")
    field_values = math.prod(validation.shape[1:])
    errors = error_metrics(vy, prediction)
    return {
        "id": f"cube-{representation}-s{seed}", "task": "cube_reconstruction",
        "representation": representation, "seed": seed, "device": "cpu",
        "metrics": {
            "initial_validation_loss": initial_loss, "validation_loss": errors["mse"],
            "training_loss": curves[-1]["train_loss"], "mae": errors["mae"], "max_error": errors["max_error"],
            "out_of_distribution_noise_mse": noise_mse, "noise_zero_predictor_mse": noise_zero_predictor_mse,
            "training_seconds": training_seconds, "training_throughput": epochs * len(tx) / training_seconds,
            "inference_throughput": len(vx) / inference_seconds, "inference_batch_ms": inference_seconds * 1000,
            "bits_per_field_value": 24, "field_payload_bytes": field_values * 3,
            "amplitude_scale_metadata_bytes": 0 if representation == "direct_rgb24" else 8,
            "mask_bits_per_field_value": 1, "mask_runtime_bytes_per_value": 1,
            "neural_input_bytes_per_sample": vx[0].numel() * vx.element_size(),
            "validation_input_buffer_bytes": vx.numel() * vx.element_size(),
            "retained_fraction": float(val_mask.float().mean()),
            "live_parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
            "checkpoint_bytes": checkpoint_bytes, **memory.report(),
            "checkpoint_predictions_exact": True,
        },
        "curves": curves,
        "dataset": {
            "shape": list(validation.shape[1:]), "train_samples": len(train), "validation_samples": len(validation),
            "sha256": dataset_hash(train, validation), "mask_sha256": dataset_hash(train_mask, val_mask),
            "initial_parameter_sha256": initial_hash, "source": "independent random phase/amplitude low-rank periodic tensors",
            "training_scope": f"{epochs}-epoch fixed chromatic input transform with chroma8 weight/activation MLP; all matrix products/optimizer FP32",
            "timing_scope": "optimization batches only; input encoding, validation, checkpoint serialization, and initial/final evaluation excluded",
            "memory_scope": "process peak includes optimization and per-epoch validation",
            "generalization_scope": "held-out samples from the same structured generator; cannot reconstruct arbitrary erased data",
            "negative_control": "256 independently seeded uniform random tensors, with the same retained-coordinate masks as structured validation",
        },
        "execution": module_quantization_report(model), "artifact_path": path.as_posix(),
        "model_config": {"inputs": tx.shape[1], "hidden": 128, "outputs": ty.shape[1],
                         "bits": 8, "representation": "chroma8"},
        "validation_prediction_sha256": dataset_hash(prediction),
    }


def cube_benchmarks(seeds=(7, 23)):
    from chromatensor.holographic import HolographicCube, numeric_polar_reference
    rows, turns = [], []
    for dimensions in (3, 4, 5, 6):
        for seed in seeds:
            smooth = smooth_cubes(1, dimensions=dimensions, seed=seed)[0]
            noise = torch.rand(smooth.shape, generator=torch.Generator().manual_seed(seed)) * 2 - 1
            for signal, values in (("smooth", smooth), ("noise", noise)):
                cube = ChromaCube.from_tensor(values, RGBCodec(-1, 1))
                hologram = HolographicCube.from_tensor(values)
                original = cube.tensor.codes.clone()
                restored = cube
                for _ in range(4):
                    restored = restored.twist(axes=(0, 1), layer_axis=dimensions - 1, layer_index=0)
                if not torch.equal(restored.tensor.codes, original) or not torch.equal(restored.cell_ids, cube.cell_ids):
                    raise AssertionError("four quarter turns did not restore the entire cube")
                inverse = cube.twist(axes=(0, 1), layer_axis=dimensions - 1, layer_index=0).inverse_twist(axes=(0, 1), layer_axis=dimensions - 1, layer_index=0)
                if not torch.equal(inverse.tensor.codes, original) or not torch.equal(inverse.cell_ids, cube.cell_ids):
                    raise AssertionError("inverse did not restore the entire cube")
                turns.append({"dimensions": dimensions, "signal": signal, "seed": seed, "cells": values.numel(), "four_turns_exact": True, "inverse_exact": True, "identity_preserved": True, "integer_payload_bytes": cube.tensor.storage_nbytes, "identity_metadata_bytes": cube.cell_ids.numel() * cube.cell_ids.element_size()})
                for coverage in (1.0, 0.75, 0.5, 0.25, 0.0):
                    mask = hologram.erasure_mask(1 - coverage, seed=seed)
                    decoded, seconds = timed(lambda: hologram.decode(mask), repeats=3)
                    numeric = numeric_polar_reference(values, mask=mask)
                    if not isinstance(numeric, torch.Tensor):
                        numeric = numeric[0]
                    numeric_errors = error_metrics(decoded, numeric)
                    direct = cube.decode().double() * mask
                    rows.append({"dimensions": dimensions, "shape": list(values.shape), "signal": signal, "seed": seed, "coverage": float(mask.float().mean()), "field_values": values.numel(), "holographic": error_metrics(values, decoded), "direct_rgb24_erasure": error_metrics(values, direct), "equal_bit_numeric_polar_delta": numeric_errors, "payload_bytes": hologram.codes.numel() * hologram.codes.element_size(), "amplitude_scale_metadata_bytes": 8, "decode_ms": seconds * 1000, "execution": "complex128 FFT/iFFT; uint8 field storage; same 24-bit phase/amplitude control", "erasure_scope": "identical retained-index masks in two different bases; no exact arbitrary-fragment recovery"})
    return rows, turns
