"""Replay saved models on small batches of their real held-out inputs."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from chromatensor import load_chroma_checkpoint
from chromatensor.serialization.format import _budget, _open_archive, _read_member, MAX_HEADER
from .common import autocast_context


ROOT = Path(__file__).resolve().parents[1]
MAX_PROBE_BYTES = 16 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    descriptor = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(descriptor + b"\n" + array.tobytes()).hexdigest()


def _path(location, root):
    path = Path(location)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def _portable_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def load_numeric_npz(path, *, max_bytes=MAX_PROBE_BYTES, expected_keys=None):
    """Check numeric NPY headers and byte limits before NumPy allocates arrays."""
    path = Path(path)
    budget = _budget(max_bytes, MAX_CHECKPOINT_BYTES, "max_bytes")
    arrays = {}
    try:
        with _open_archive(path, max_file_bytes=budget, max_header_bytes=min(budget, MAX_HEADER),
                           max_expanded_bytes=budget, max_entries=256) as archive:
            entries = archive.infolist()
            if not entries:
                raise ValueError("NPZ file exceeds entry or expanded size limits")
            for entry in entries:
                if not entry.filename.endswith(".npy") or "/" in entry.filename or "\\" in entry.filename:
                    raise ValueError("NPZ entries must be named NPY arrays")
                raw = _read_member(archive, entry.filename, budget)
                stream = io.BytesIO(raw)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY array version")
                if len(shape) > 8 or any(type(size) is not int or size < 0 for size in shape):
                    raise ValueError("Invalid NPZ array dimensions")
                expected = math.prod(shape) * dtype.itemsize
                if dtype.kind not in "fiub" or expected > budget or expected != len(raw) - stream.tell():
                    raise ValueError("Invalid NPZ numeric dtype or allocation")
                array = np.load(io.BytesIO(raw), allow_pickle=False)
                if not np.isfinite(array).all():
                    raise ValueError("NPZ arrays must contain finite values")
                arrays[entry.filename[:-4]] = array
    except zipfile.BadZipFile as error:
        raise ValueError("Invalid NPZ ZIP archive") from error
    if expected_keys is not None and set(arrays) != set(expected_keys):
        raise ValueError("NPZ keys do not match the required probe arrays")
    return arrays


def build_checkpoint_model(record, root=ROOT):
    """Build the recorded CPU architecture and load its saved numeric state."""
    config = record.get("model_config")
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Missing or unsupported checkpoint model configuration")
    if config.get("task") != record.get("task") or config.get("representation") != record.get("representation"):
        raise ValueError("Checkpoint model configuration does not match its record")
    if type(config.get("seed")) is not int or config["seed"] != record.get("seed"):
        raise ValueError("Checkpoint model seed does not match its record")
    checkpoint = _path(record["artifact_path"], root)
    if checkpoint.suffix == ".ctcheckpoint":
        state = load_chroma_checkpoint(checkpoint, device="cpu")
    elif checkpoint.suffix == ".npz":
        state = {name: torch.from_numpy(value) for name, value in
                 load_numeric_npz(checkpoint, max_bytes=MAX_CHECKPOINT_BYTES).items()}
    else:
        raise ValueError("Unsupported replay checkpoint format")
    if record["task"] == "attention":
        from .attention import PhaseAttentionPredictor
        model = PhaseAttentionPredictor(record["representation"], hidden=config["hidden"], heads=config["heads"],
                                        sequence_length=config["sequence_length"], num_codes=config["num_codes"])
    elif record["task"] == "decoder":
        from chromatensor.codecs import LearnedRGBCodec
        model = LearnedRGBCodec(minimum=config["minimum"], maximum=config["maximum"],
                                hidden=config["hidden"], seed=config["seed"])
    else:
        from .training import model_for
        dimensions = config.get("dimensions")
        if (not isinstance(dimensions, list) or len(dimensions) != 2
                or any(type(size) is not int or size < 1 for size in dimensions)):
            raise ValueError("Invalid replay model dimensions")
        model = model_for(record["task"], record["representation"], dimensions, record["seed"], "cpu")
    model.load_state_dict(state, strict=True)
    return model.cpu().eval()


def _predict(model, inputs, representation, task):
    with torch.no_grad(), autocast_context(representation, "cpu"):
        output = model.decode(model.encode(inputs)) if task == "decoder" else model(inputs)
    if not torch.isfinite(output).all():
        raise ValueError("CPU checkpoint probe produced nonfinite predictions")
    precision = {
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "autocast_dtype": "torch.float16" if representation == "fp16" else "torch.bfloat16" if representation == "bf16" else None,
        "output_dtype": str(output.dtype), "stored_prediction_dtype": "torch.float32",
    }
    # BF16 has no NumPy dtype. Widening finite FP16/BF16 values to FP32 is exact.
    return output.detach().float().cpu().contiguous().numpy(), precision


@contextmanager
def _cpu_threads(count):
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("Invalid CPU probe thread count")
    previous = torch.get_num_threads()
    if previous != count:
        torch.set_num_threads(count)
    try:
        yield
    finally:
        if previous != count:
            torch.set_num_threads(previous)


def write_checkpoint_probe(record, heldout_inputs, root=ROOT):
    """Save and immediately verify CPU replay, without replacing task metrics."""
    inputs = heldout_inputs[:32].detach().cpu().contiguous()
    if len(inputs) < 1:
        raise ValueError("Checkpoint probes require real held-out inputs")
    model = build_checkpoint_model(record, root)
    predictions, precision = _predict(model, inputs, record["representation"], record["task"])
    checkpoint = _path(record["artifact_path"], root)
    probe_path = checkpoint.with_name(checkpoint.stem + "-probe.npz")
    input_array = inputs.numpy()
    np.savez_compressed(probe_path, inputs=input_array, predictions=predictions)
    record["checkpoint_probe"] = {
        "schema_version": 1, "path": _portable_path(probe_path), "sha256": sha256_file(probe_path),
        "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "probe_bytes": probe_path.stat().st_size, "input_sha256": array_sha256(input_array),
        "prediction_sha256": array_sha256(predictions), "device": "cpu", "examples": len(inputs),
        "input_shape": list(inputs.shape), "input_dtype": str(inputs.dtype),
        "prediction_shape": list(predictions.shape), "precision": precision,
        "dataset_sha256": record["dataset"]["sha256"], "threads": torch.get_num_threads(),
        "source": "first real held-out inputs from the recorded dataset",
        "predictions_exact": True,
        "scope": "CPU checkpoint replay only; original task accuracy, precision and timings are unchanged",
    }
    verify_checkpoint_probe(record, root)
    return record["checkpoint_probe"]


def verify_checkpoint_probe(record, root=ROOT):
    """Recompute hashes, reload the model and require exact saved predictions."""
    probe = record.get("checkpoint_probe")
    if not isinstance(probe, dict) or probe.get("schema_version") != 1 or probe.get("device") != "cpu":
        raise ValueError("Missing or invalid CPU checkpoint probe")
    checkpoint = _path(record["artifact_path"], root)
    path = _path(probe["path"], root)
    if sha256_file(checkpoint) != probe.get("checkpoint_sha256"):
        raise ValueError("Checkpoint probe checkpoint hash mismatch")
    if checkpoint.stat().st_size != probe.get("checkpoint_bytes"):
        raise ValueError("Checkpoint probe checkpoint byte count mismatch")
    recorded_bytes = record.get("metrics", {}).get("checkpoint_bytes")
    if recorded_bytes is not None and recorded_bytes != checkpoint.stat().st_size:
        raise ValueError("Checkpoint size differs from the model metrics")
    if path.stat().st_size != probe.get("probe_bytes") or sha256_file(path) != probe.get("sha256"):
        raise ValueError("Checkpoint probe file hash or byte count mismatch")
    arrays = load_numeric_npz(path, expected_keys={"inputs", "predictions"})
    inputs, expected = arrays["inputs"], arrays["predictions"]
    if inputs.ndim < 1 or not 1 <= len(inputs) <= 32 or expected.ndim < 1 or len(expected) != len(inputs):
        raise ValueError("Checkpoint probe sample counts do not match")
    if len(inputs) != probe.get("examples") or list(inputs.shape) != probe.get("input_shape"):
        raise ValueError("Checkpoint probe input shape mismatch")
    input_tensor = torch.from_numpy(inputs)
    if str(input_tensor.dtype) != probe.get("input_dtype") or list(expected.shape) != probe.get("prediction_shape"):
        raise ValueError("Checkpoint probe dtype or prediction shape mismatch")
    if expected.dtype != np.float32:
        raise ValueError("Checkpoint probe predictions must use FP32 storage")
    if array_sha256(inputs) != probe.get("input_sha256") or array_sha256(expected) != probe.get("prediction_sha256"):
        raise ValueError("Checkpoint probe array hash mismatch")
    if probe.get("dataset_sha256") != record.get("dataset", {}).get("sha256"):
        raise ValueError("Checkpoint probe dataset hash mismatch")
    with _cpu_threads(probe.get("threads")):
        model = build_checkpoint_model(record, root)
        predictions, precision = _predict(model, input_tensor, record["representation"], record["task"])
    if precision != probe.get("precision"):
        raise ValueError("Checkpoint probe execution precision mismatch")
    if not np.array_equal(predictions, expected) or array_sha256(predictions) != probe["prediction_sha256"]:
        raise ValueError("Checkpoint predictions do not replay exactly")
    return {"verified": True, "record_id": record["id"], "checkpoint_path": _portable_path(checkpoint),
            "probe_path": _portable_path(path), "checkpoint_bytes": checkpoint.stat().st_size,
            "probe_bytes": path.stat().st_size, "examples": len(inputs), "precision": precision,
            "predictions_exact": True}
