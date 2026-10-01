"""Add explicit storage accounting and scoped operation estimates to saved runs."""

from __future__ import annotations

import io
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from chromatensor import load_chroma_checkpoint
from chromatensor.autograd import ChromaQuantizer
from chromatensor.nn import module_quantization_report
from .attention import PhaseAttentionPredictor
from .training import model_for


_FLOAT_KERNELS = {"aten::mm", "aten::addmm", "aten::bmm"}
_SELECTED_OPERATIONS = _FLOAT_KERNELS | {
    "aten::linear", "aten::matmul", "aten::round", "aten::clamp", "aten::to",
    "aten::_to_copy", "aten::amin", "aten::amax", "aten::bitwise_and",
    "aten::bitwise_or", "aten::__lshift__", "aten::__rshift__", "aten::stack",
    "aten::softmax", "aten::embedding", "aten::native_layer_norm",
}
_MAX_NUMPY_BYTES = 256 * 1024 * 1024


def _artifact_path(record, root):
    location = record.get("artifact_path")
    if not location:
        return None
    path = Path(location)
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def _source_record(record, experiments):
    if record.get("representation") != "int8_integer_post_training":
        return record
    return next((source for source in experiments if source.get("task") == record.get("task")
                 and source.get("seed") == record.get("seed") and source.get("representation") == "fp32"), record)


def _state_for(record, root, cache):
    path = _artifact_path(record, root)
    if path is None or not path.is_file():
        return None
    # Checkpoints use the project's checksum/shape-validated JSON and integer
    # codec format. This never calls torch.load or unpickles an external object.
    if path not in cache:
        if path.suffix == ".npz":
            cache[path] = _load_numpy_state(path)
        elif path.suffix == ".ctcheckpoint":
            cache[path] = load_chroma_checkpoint(path, device="cpu")
        else:
            raise ValueError(f"Unsupported model checkpoint format {path.suffix!r}")
    return cache[path]


def _load_numpy_state(path):
    """Validate ZIP and NPY sizes before NumPy allocates numeric state arrays."""
    if path.stat().st_size > _MAX_NUMPY_BYTES:
        raise ValueError("NPZ checkpoint exceeds the encoded size limit")
    state = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > 256 or sum(entry.file_size for entry in entries) > _MAX_NUMPY_BYTES:
            raise ValueError("NPZ checkpoint exceeds entry or expanded size limits")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Duplicate NPZ checkpoint entries")
        for entry in entries:
            if not entry.filename.endswith(".npy") or "/" in entry.filename or "\\" in entry.filename:
                raise ValueError("NPZ checkpoint entries must be named NPY arrays")
            raw = archive.read(entry)
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
            if dtype.kind not in "fiub" or expected > _MAX_NUMPY_BYTES or expected != len(raw) - stream.tell():
                raise ValueError("Invalid NPZ numeric array dtype or allocation")
            # The header bounds above rule out object pickles and oversized shapes.
            array = np.load(io.BytesIO(raw), allow_pickle=False)
            state[entry.filename[:-4]] = torch.from_numpy(array)
    return state


def _fallback_linear_shapes(record, config):
    task, representation = record.get("task"), record.get("representation")
    if task in {"scalar", "mnist", "cifar", "periodic"}:
        inputs = 2 if task == "scalar" else 784 if task == "mnist" else 3072 if task == "cifar" else 1 if representation == "scalar8" else 2
        hidden = 128 if task in {"mnist", "cifar"} else 64
        outputs = 3 if task == "scalar" else 10 if task in {"mnist", "cifar"} else 2
        return [("hidden_0", [hidden, inputs]), ("hidden_1", [hidden, hidden]), ("output", [outputs, hidden])]
    if task == "language":
        hidden = 48
        vocabulary = int(record.get("dataset", {}).get("vocabulary_size", 65))
        return [(name, shape) for name, shape in (
            ("q", [hidden, hidden]), ("k", [hidden, hidden]), ("v", [hidden, hidden]),
            ("projection", [hidden, hidden]), ("ff.0", [hidden * 2, hidden]),
            ("ff.2", [hidden, hidden * 2]), ("head", [vocabulary, hidden]))]
    if task == "attention":
        hidden = int(config.get("attention_hidden", 24))
        return [(name, shape) for name, shape in (
            ("input_projection", [hidden, 2]), ("q_proj", [hidden, hidden]),
            ("k_proj", [hidden, hidden]), ("v_proj", [hidden, hidden]),
            ("out_proj", [hidden, hidden]), ("output.0", [hidden * 2, hidden]),
            ("output.2", [2, hidden * 2]))]
    if task == "decoder":
        hidden = 32
        return [(name, shape) for name, shape in (
            ("encoder.0", [hidden, 1]), ("encoder.2", [hidden, hidden]), ("encoder.4", [3, hidden]),
            ("decoder.0", [hidden, 3]), ("decoder.2", [hidden, hidden]), ("decoder.4", [1, hidden]))]
    return []


def _linear_shapes(record, config, state):
    if state is None:
        return _fallback_linear_shapes(record, config), "declared reference architecture"
    shapes = []
    for name, value in state.items():
        if not name.endswith(".weight") or value.ndim != 2:
            continue
        # Embeddings are table gathers, not dense matrix multiplication.
        if record.get("task") == "language" and ("embedding" in name or "position" in name):
            continue
        shapes.append((name, list(value.shape)))
    return shapes, "size-validated numeric NPZ checkpoint weight shapes" if record.get("task") == "decoder" else "checksum-validated checkpoint weight shapes"


def forward_operation_estimate(record, config, state=None):
    """Multiply-plus-add convention, two operations per dense multiply term."""
    shapes, shape_source = _linear_shapes(record, config, state)
    task = record.get("task")
    dataset = record.get("dataset", {})
    validation_count = int(dataset.get("validation_examples", dataset.get("validation_samples", 256)))
    batch = validation_count if task == "decoder" else min(validation_count, 64 if task in {"language", "attention"} else 256)
    length = int(dataset.get("sequence_length", config.get("attention_sequence_length", 12) if task == "attention" else 32)) if task in {"language", "attention"} else 1
    dense_flops = sum(2 * batch * length * shape[0] * shape[1] for _, shape in shapes)
    terms = [{"operation": name, "weight_shape": shape,
              "multiply_add_estimate": 2 * batch * length * shape[0] * shape[1]} for name, shape in shapes]
    attention_flops = assignment_flops = table_flops = 0
    if task == "language":
        hidden = next((shape[1] for name, shape in shapes if name.endswith("q.weight") or name == "q"), 48)
        attention_flops = 4 * batch * length * length * hidden
        terms.append({"operation": "QK plus probability-value matmuls", "multiply_add_estimate": attention_flops})
    elif task == "attention":
        hidden = next((shape[0] for name, shape in shapes if "q_proj" in name), int(config.get("attention_hidden", 24)))
        heads = int(config.get("attention_heads", 3))
        head_dim = hidden // heads
        codebook = "codebook" in record.get("representation", "")
        attention_flops = (2 if codebook else 4) * batch * length * length * hidden
        terms.append({"operation": "probability-value matmul" if codebook else "QK plus probability-value matmuls",
                      "multiply_add_estimate": attention_flops})
        if codebook:
            code_shapes = [(name, value.shape) for name, value in (state or {}).items()
                           if name.endswith("codebook.codebook")]
            code_count = int(code_shapes[0][1][0]) if code_shapes else int(config.get("attention_num_codes", 32))
            assignment_flops = 4 * batch * heads * length * code_count * head_dim
            table_flops = 2 * code_count * code_count * head_dim
            terms.extend([
                {"operation": "Q/K nearest-code matrix products", "multiply_add_estimate": assignment_flops},
                {"operation": "rebuilt code-vector similarity table", "multiply_add_estimate": table_flops},
            ])
    elif record.get("representation") == "latent":
        # One VQ bottleneck after the first scalar hidden layer.
        code_shapes = [(name, value.shape) for name, value in (state or {}).items() if name.endswith(".codebook")]
        codes, features = code_shapes[0][1] if code_shapes else (32, 64)
        assignment_flops = 2 * batch * int(codes) * int(features)
        terms.append({"operation": "nearest-code matrix product", "multiply_add_estimate": assignment_flops})
    integer = record.get("representation") == "int8_integer_post_training"
    total = dense_flops + attention_flops + assignment_flops + table_flops
    return {
        "batch_size": batch, "sequence_length": length, "shape_source": shape_source,
        "forward_flops_estimate": 0 if integer else total,
        "integer_multiply_add_estimate": total if integer else 0,
        "float_matmul_flops_per_example": 0 if integer else total / max(batch, 1),
        "matrix_terms": terms,
        "kernel_dtype": "INT32 matrix products from INT8 weight payload" if integer else
            "mixed FP16/FP32" if record.get("representation") == "fp16" else
            "mixed BF16/FP32" if record.get("representation") == "bf16" else "FP32 dense products",
        "convention": "two multiply/add operations per matrix-product term, including products at causally masked positions",
        "scope": "one inference batch; excludes rounding, RGB packing, scale estimation, clipping, gathers, nearest-code norms/argmin, bias, activations, normalization, softmax, memory traffic and backward",
        "training_scope": "codebook-attention training additionally computes a QK gradient surrogate; inference estimates exclude it",
        "is_hardware_instruction_count": False,
    }


def _runtime_storage(record):
    quantizers = record.get("execution", {}).get("quantizers", [])
    chromatic = [q for q in quantizers if q.get("enabled") and q.get("payload_format") == "RGB uint8 triples"]
    codebooks = record.get("execution", {}).get("codebooks", [])
    return {
        "chromatic_quantizer_count": len(chromatic),
        "runtime_rgb_payload_bytes": sum(q.get("last_payload_bytes", 0) for q in chromatic),
        "runtime_integer_payload_bytes": sum(q.get("last_payload_bytes", 0) for q in quantizers if q.get("enabled") and q.get("payload_format") != "RGB uint8 triples"),
        "retained_quantizer_inspection_bytes": sum(q.get("last_retained_inspection_bytes", 0) for q in quantizers),
        "ideal_packed_uniform_code_bytes": sum(q.get("last_ideal_packed_code_bytes", 0) for q in quantizers if q.get("enabled")),
        "actual_rgb_bits_per_encoded_scalar": 24 if chromatic else None,
        "nominal_quantization_bits": sorted({q.get("bits") for q in quantizers if q.get("enabled")}),
        "learned_codebook_parameter_bytes": sum(book.get("codebook_bytes", 0) for book in codebooks),
        "learned_palette_bytes": sum(book.get("color_bytes", 0) for book in codebooks),
        "last_learned_latent_rgb_bytes": sum(book["last_rgb_bytes"] for book in codebooks)
            if all("last_rgb_bytes" in book for book in codebooks) else None,
        "last_learned_latent_index_bytes": sum(book["last_index_bytes"] for book in codebooks)
            if all("last_index_bytes" in book for book in codebooks) else None,
        "scope": "last captured quantizer states; RGB byte triples retain duplicate integer IDs for inspection; overlapping layer inputs and outputs are counted separately",
    }


def _attention_weight_reconstruction(record, state):
    weights = [value for name, value in state.items()
               if name.endswith(".weight") and any("attention." + projection in name
                   for projection in ("q_proj", "k_proj", "v_proj", "out_proj"))]
    if not weights:
        return None
    chromatic = record.get("representation") != "fp32_attention_dot"
    quantizer = ChromaQuantizer(representation="chroma8")
    absolute, amount = 0., 0
    for weight in weights:
        reconstructed = quantizer.encode(weight).decode() if chromatic else weight.clone()
        absolute += float((reconstructed - weight).abs().sum())
        amount += weight.numel()
    return {"mae": absolute / amount, "values": amount,
            "scope": "Q/K/V and output attention projection weights reconstructed with the actual per-tensor forward quantizer; excludes unquantized prediction-head weights, activations and vector-code assignment"}


def _profile_scalar(record, root, config, state):
    inputs = next(value.shape[1] for name, value in state.items() if name.endswith("layers.0.weight"))
    outputs = next(value.shape[0] for name, value in state.items() if name.endswith("layers.4.weight"))
    model = model_for("scalar", record["representation"], (int(inputs), int(outputs)), record["seed"], "cpu")
    model.load_state_dict(state)
    model.eval()
    batch = min(int(record.get("dataset", {}).get("validation_examples", 256)), 256)
    sample = torch.linspace(-1, 1, batch * inputs).reshape(batch, inputs)
    with torch.no_grad():
        model(sample)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True, with_flops=True) as profiler:
            model(sample)
    events = profiler.key_averages(group_by_input_shape=True)
    selected = [{"operation": event.key, "calls": event.count, "input_shapes": event.input_shapes,
                 "profiler_annotated_flops": event.flops,
                 "self_cpu_time_us": event.self_cpu_time_total}
                for event in events if event.key in _SELECTED_OPERATIONS]
    return {
        "record_id": record["id"], "device": "cpu", "input_shape": list(sample.shape),
        "source": "one torch.profiler forward of the trained checksum-validated checkpoint",
        "sample_source": "deterministic synthetic linspace inputs; this is operation evidence, not another accuracy evaluation",
        "parameter_dtype": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "observed_fp32_dense_kernel_calls": sum(event.count for event in events if event.key in _FLOAT_KERNELS),
        "annotated_float_matmul_flops": sum(event.flops for event in events if event.key in _FLOAT_KERNELS),
        "all_profiler_annotated_flops": sum(event.flops for event in events),
        "selected_operations": selected,
        "runtime_storage_on_this_batch": _runtime_storage({"execution": module_quantization_report(model)}),
        "limits": ["Profiler FLOP annotations cover selected operators, not every instruction or tensor operation.",
                   "No backward pass, optimizer, GPU kernel, or end-to-end training run was profiled.",
                   "Profiler CPU times include profiling overhead and are not used as benchmark latency."],
    }


def enrich_result(result, root):
    """Mutate and return an existing result, without writing its JSON file."""
    root = Path(root).resolve()
    config = result.get("config", {})
    experiments = result.get("experiments", [])
    cache = {}
    failures, profiles = [], []
    profiled_representations = set()
    for record in experiments:
        source = _source_record(record, experiments)
        try:
            state = _state_for(source, root, cache)
        except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
            state = None
            failures.append({"record_id": record.get("id"), "stage": "checkpoint validation", "error": str(error)})
        estimate = forward_operation_estimate(record, config, state)
        record["operation_estimate"] = estimate
        metrics = record.setdefault("metrics", {})
        metrics["forward_flops_estimate"] = estimate["forward_flops_estimate"]
        metrics["integer_multiply_add_estimate"] = estimate["integer_multiply_add_estimate"]
        runtime = _runtime_storage(record)
        if record.get("task") == "decoder":
            # Decoder timings encode and decode the entire held-out scalar batch.
            runtime.update({"runtime_rgb_payload_bytes": estimate["batch_size"] * 3,
                "actual_rgb_bits_per_encoded_scalar": 24, "nominal_quantization_bits": [24],
                "scope": "analytic uint8 RGB payload for the recorded full validation encode/decode batch; encoder/decoder parameters and dense kernels remain FP32"})
        metrics["runtime_rgb_payload_bytes"] = runtime["runtime_rgb_payload_bytes"]
        storage = {
            **runtime, "live_master_parameter_bytes": metrics.get("live_parameter_bytes", record.get("execution", {}).get("master_parameter_bytes", 0)),
            "diagnostic_scalar_payload_bytes": metrics.get("diagnostic_code_payload_bytes"),
            "diagnostic_codec_metadata_bytes": metrics.get("codec_metadata_bytes"),
            "saved_checkpoint_bytes": metrics.get("checkpoint_bytes"),
            "torch_checkpoint_bytes": metrics.get("torch_checkpoint_bytes"),
            "diagnostic_bits_per_value": metrics.get("bits_per_value"),
            "diagnostic_physical_bits_per_parameter":
                8 * metrics["diagnostic_code_payload_bytes"] / metrics["parameter_count"]
                if metrics.get("diagnostic_code_payload_bytes") is not None and metrics.get("parameter_count", 0) > 0 else None,
            "checkpoint_format": "lossless RGBD storage of FP32 master values plus integer buffers" if record.get("artifact_path") else None,
            "stored_payload_is_executable_quantized_model": False,
            "byte_ledger_note": "chroma8 has 8-bit scalar levels but fake forward stores one three-byte RGB symbol per encoded scalar; live FP32 parameters and inspection buffers remain allocated",
        }
        if record.get("task") == "decoder":
            path = _artifact_path(record, root)
            storage["live_master_parameter_bytes"] = metrics.get("live_parameter_bytes", metrics.get("bytes_model"))
            storage["checkpoint_format"] = "compressed numeric NPZ model state and JSON codec sidecar, loaded without pickle"
            storage["saved_checkpoint_bytes"] = path.stat().st_size if path is not None and path.is_file() else None
            sidecar = path.with_suffix(".json") if path is not None else None
            storage["checkpoint_sidecar_bytes"] = sidecar.stat().st_size if sidecar is not None and sidecar.is_file() else None
            storage["byte_ledger_note"] = "learned uint8 RGB channels occupy 24 bits per scalar; FP32 encoder/decoder weights and JSON metadata are separate costs"
            metrics["live_parameter_bytes"] = storage["live_master_parameter_bytes"]
            metrics["checkpoint_bytes"] = storage["saved_checkpoint_bytes"]
        record["storage_accounting"] = storage
        if record.get("task") == "attention" and state is not None:
            weight_reconstruction = _attention_weight_reconstruction(record, state)
            if weight_reconstruction is not None:
                metrics["encoding_error"] = weight_reconstruction["mae"]
                metrics["reconstruction_error"] = weight_reconstruction["mae"]
                record["reconstruction_measurement"] = weight_reconstruction
            if "codebook" in record.get("representation", "") and metrics.get("auxiliary_loss") is not None:
                metrics["train_latent_vector_mse_from_auxiliary"] = metrics["auxiliary_loss"] / 2.5
                record["latent_reconstruction_measurement"] = {
                    "source": "recorded final-epoch auxiliary loss divided by 2*(1+0.25)",
                    "scope": "mean Q/K vector assignment MSE over training batches; default commitment_weight=0.25",
                    "is_validation_measurement": False,
                }
        if record.get("representation") == "int8_integer_post_training":
            storage["live_master_parameter_bytes"] = 0
            storage["integer_inference_buffer_bytes"] = metrics.get("bytes_model")
            storage["stored_payload_is_executable_quantized_model"] = True
            storage["byte_ledger_note"] = "inference-only CPU module keeps INT8 weights and FP32 scales/bias; its FP32 trained control is a separate record"
            metrics.setdefault("training_loss", None)
            metrics.setdefault("training_seconds", None)
            metrics.setdefault("training_throughput", None)
            metrics.setdefault("reconstruction_error", None)
            metrics.setdefault("encoding_error", None)
            metrics.setdefault("peak_memory_bytes", None)
            metrics.setdefault("peak_cuda_allocated_bytes", None)
            metrics.setdefault("rss_increase_bytes", None)
            record["not_applicable_metrics"] = {"training_loss": "post-training inference-only record",
                "training_seconds": "training belongs to paired FP32 control", "training_throughput": "no training occurred",
                "reconstruction_error": "module output error is measured by task loss; scalar codec reconstruction does not apply",
                "encoding_error": "input rounding occurs within timed inference; separate scalar encoding error was not measured",
                "peak_memory_bytes": "memory was not independently sampled for this inference-only conversion",
                "peak_cuda_allocated_bytes": "CPU reference inference",
                "rss_increase_bytes": "memory was not independently sampled for this inference-only conversion"}
        if record.get("task") == "scalar" and record.get("representation") in {"fp32", "chroma8"} and state is not None and record["representation"] not in profiled_representations:
            try:
                profiles.append(_profile_scalar(record, root, config, state))
                profiled_representations.add(record["representation"])
            except (RuntimeError, ValueError, KeyError, OSError) as error:
                failures.append({"record_id": record.get("id"), "stage": "CPU profiler", "error": str(error)})
    result["instrumentation"] = {
        "schema_version": 1, "analytic_estimates": "one recorded inference batch, matrix products only",
        "profiler_runs": profiles, "failures": failures,
        "limits": ["Nominal quantization bits are separate from physical RGB storage and live training memory.",
                   "Matrix FLOP estimates and profiler annotations are partial arithmetic accounts, not exact hardware counts.",
                   "Checkpoint validation recovers trained master values; profiling uses synthetic CPU inputs and does not replace the recorded validation metrics."],
    }
    return result
