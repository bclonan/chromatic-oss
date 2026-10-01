from pathlib import Path

import numpy as np
import pytest
import torch

from chromatensor import RGBDCodec, save_chroma_checkpoint
from chromatensor.nn import module_quantization_report
from experiments.instrumentation import enrich_result, forward_operation_estimate, _load_numpy_state
from experiments.training import model_for


def test_analytic_scalar_integer_and_transformer_operation_counts():
    scalar = {"task": "scalar", "representation": "chroma8", "dataset": {"validation_examples": 256}}
    expected = 2 * 256 * (2 * 64 + 64 * 64 + 64 * 3)
    assert forward_operation_estimate(scalar, {})["forward_flops_estimate"] == expected
    scalar["representation"] = "int8_integer_post_training"
    integer = forward_operation_estimate(scalar, {})
    assert integer["forward_flops_estimate"] == 0
    assert integer["integer_multiply_add_estimate"] == expected
    language = {"task": "language", "representation": "fp32", "dataset": {"validation_examples": 64, "sequence_length": 32, "vocabulary_size": 65}}
    estimate = forward_operation_estimate(language, {})
    dense = 2 * 64 * 32 * (4 * 48 * 48 + 2 * 48 * 96 + 48 * 65)
    attention = 4 * 64 * 32 * 32 * 48
    assert estimate["forward_flops_estimate"] == dense + attention
    assert not estimate["is_hardware_instruction_count"]


def test_codebook_attention_inference_counts_assignment_and_table():
    record = {"task": "attention", "representation": "chroma8_attention_codebook", "dataset": {"validation_examples": 64, "sequence_length": 12}}
    estimate = forward_operation_estimate(record, {})
    by_name = {term["operation"]: term["multiply_add_estimate"] for term in estimate["matrix_terms"]}
    assert by_name["Q/K nearest-code matrix products"] == 4 * 64 * 3 * 12 * 32 * 8
    assert by_name["rebuilt code-vector similarity table"] == 2 * 32 * 32 * 8
    assert by_name["probability-value matmul"] == 2 * 64 * 12 * 12 * 24


def test_enrichment_profiles_safe_trained_checkpoint_and_actual_rgb_bytes(tmp_path):
    torch.set_num_threads(1)
    records = []
    for representation in ("fp32", "chroma8"):
        model = model_for("scalar", representation, (2, 3), 4, "cpu")
        model(torch.randn(16, 2))
        path = tmp_path / (representation + ".ctcheckpoint")
        saved_bytes = save_chroma_checkpoint(model, path, RGBDCodec())
        records.append({"id": "scalar-" + representation + "-s4", "task": "scalar", "representation": representation,
            "seed": 4, "artifact_path": str(path), "dataset": {"validation_examples": 16},
            "execution": module_quantization_report(model),
            "metrics": {"live_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                        "checkpoint_bytes": saved_bytes, "bits_per_value": 8 if representation == "chroma8" else 32}})
    result = enrich_result({"config": {}, "experiments": records}, tmp_path)
    assert not result["instrumentation"]["failures"]
    profiles = result["instrumentation"]["profiler_runs"]
    assert len(profiles) == 2
    assert all(profile["observed_fp32_dense_kernel_calls"] == 3 for profile in profiles)
    for profile in profiles:
        record = next(r for r in records if r["id"] == profile["record_id"])
        assert profile["annotated_float_matmul_flops"] == record["metrics"]["forward_flops_estimate"]
    chroma = records[1]["storage_accounting"]
    assert chroma["actual_rgb_bits_per_encoded_scalar"] == 24
    assert chroma["nominal_quantization_bits"] == [8]
    assert chroma["runtime_rgb_payload_bytes"] == chroma["ideal_packed_uniform_code_bytes"] * 3
    assert chroma["retained_quantizer_inspection_bytes"] > chroma["runtime_rgb_payload_bytes"]
    assert chroma["live_master_parameter_bytes"] > 0
    assert not chroma["stored_payload_is_executable_quantized_model"]


def test_missing_checkpoint_falls_back_without_inventing_profiler_evidence(tmp_path):
    record = {"id": "missing", "task": "scalar", "representation": "fp32", "seed": 1,
              "artifact_path": str(tmp_path / "missing.ctcheckpoint"), "metrics": {}}
    result = enrich_result({"experiments": [record]}, tmp_path)
    assert not result["instrumentation"]["profiler_runs"]
    assert record["operation_estimate"]["shape_source"] == "declared reference architecture"
    assert record["metrics"]["forward_flops_estimate"] > 0


def test_integer_inference_training_metrics_are_explicitly_not_applicable(tmp_path):
    record = {"id": "integer", "task": "scalar", "representation": "int8_integer_post_training", "seed": 1,
              "metrics": {"bytes_model": 8192}}
    result = enrich_result({"experiments": [record]}, tmp_path)
    assert result["experiments"][0]["metrics"]["training_seconds"] is None
    assert record["storage_accounting"]["live_master_parameter_bytes"] == 0
    assert record["storage_accounting"]["integer_inference_buffer_bytes"] == 8192
    assert record["metrics"]["integer_multiply_add_estimate"] > 0


def test_learned_rgb_decoder_npz_has_real_shapes_payload_and_master_bytes(tmp_path):
    from chromatensor.codecs import LearnedRGBCodec

    codec = LearnedRGBCodec(hidden=32, seed=4)
    path = tmp_path / "neural_rgb.npz"
    np.savez_compressed(path, **{name: value.detach().numpy() for name, value in codec.state_dict().items()})
    path.with_suffix(".json").write_text('{"hidden":32}', encoding="utf-8")
    record = {"id": "decoder-learned_rgb-s4", "task": "decoder", "representation": "learned_rgb", "seed": 4,
        "artifact_path": str(path), "dataset": {"validation_samples": 4096},
        "metrics": {"bytes_model": 9744, "bits_per_value": 24}, "execution": {"fully_chromatic": False}}
    result = enrich_result({"experiments": [record]}, tmp_path)
    assert not result["instrumentation"]["failures"]
    assert record["metrics"]["runtime_rgb_payload_bytes"] == 4096 * 3
    assert record["metrics"]["live_parameter_bytes"] == 9744
    assert record["metrics"]["forward_flops_estimate"] == 2 * 4096 * (32 + 1024 + 96 + 96 + 1024 + 32)
    assert record["storage_accounting"]["actual_rgb_bits_per_encoded_scalar"] == 24
    assert record["storage_accounting"]["checkpoint_format"].startswith("compressed numeric NPZ")
    assert record["storage_accounting"]["checkpoint_sidecar_bytes"] > 0
    assert record["storage_accounting"]["saved_checkpoint_bytes"] == path.stat().st_size


def test_npz_object_arrays_are_rejected_without_pickle_loading(tmp_path):
    path = tmp_path / "objects.npz"
    np.savez(path, unsafe=np.array([{"value": 1}], dtype=object))
    with pytest.raises(ValueError, match="dtype or allocation"):
        _load_numpy_state(path)
