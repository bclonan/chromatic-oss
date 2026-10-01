"""Checkpoint replay must use saved state and fail on changed evidence."""

import copy
import io
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest
import torch

from chromatensor import RGBDCodec, save_chroma_checkpoint
from experiments.checkpoint_probe import (
    array_sha256, build_checkpoint_model, load_numeric_npz, sha256_file,
    verify_checkpoint_probe, write_checkpoint_probe,
)
from experiments.datasets import scalar_data
from experiments.training import model_for, train_one


@pytest.fixture(autouse=True)
def small_cpu_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def scalar_record(tmp_path, representation="chroma8"):
    model = model_for("scalar", representation, (2, 3), 7, "cpu")
    path = tmp_path / (representation + ".ctcheckpoint")
    checkpoint_bytes = save_chroma_checkpoint(model, path, RGBDCodec())
    data = scalar_data(7, "cpu", count=16)
    record = {"id": "scalar-" + representation + "-s7", "task": "scalar", "representation": representation,
              "seed": 7, "artifact_path": str(path), "device": "cuda",
              "metrics": {"checkpoint_bytes": checkpoint_bytes, "validation_loss": .123, "inference_batch_ms": 7.89},
              "dataset": data[4], "model_config": {"schema_version": 1, "task": "scalar",
                  "representation": representation, "seed": 7, "dimensions": [2, 3]}}
    return record, data[2]


@pytest.mark.parametrize("representation", ["fp32", "fp16", "bf16", "chroma8", "latent"])
def test_probe_reloads_exact_predictions_and_keeps_original_task_metrics(tmp_path, representation):
    record, heldout = scalar_record(tmp_path, representation)
    old_metrics = copy.deepcopy(record["metrics"])
    probe = write_checkpoint_probe(record, heldout)
    arrays = load_numeric_npz(probe["path"], expected_keys={"inputs", "predictions"})
    np.testing.assert_array_equal(arrays["inputs"], heldout[:32].numpy())
    assert arrays["predictions"].dtype == np.float32
    assert probe["examples"] == 32
    assert probe["device"] == "cpu"
    assert probe["precision"]["autocast_dtype"] == {"fp16": "torch.float16", "bf16": "torch.bfloat16"}.get(representation)
    assert record["metrics"] == old_metrics
    assert record["device"] == "cuda"
    verified = verify_checkpoint_probe(record)
    assert verified["verified"] and verified["predictions_exact"]
    assert verified["checkpoint_bytes"] == Path(record["artifact_path"]).stat().st_size
    assert not build_checkpoint_model(record).training


def test_training_harness_probe_uses_actual_heldout_batch_and_integer_source(tmp_path):
    data = scalar_data(7, "cpu", count=32)
    record, _, integer = train_one("scalar", "fp32", 7, data,
                                  {"device": "cpu", "scalar_epochs": 1, "epochs": 1}, tmp_path)
    arrays = load_numeric_npz(record["checkpoint_probe"]["path"])
    np.testing.assert_array_equal(arrays["inputs"], data[2][:32].numpy())
    assert integer["source_record_id"] == record["id"]
    assert verify_checkpoint_probe(record)["verified"]


def test_decoder_checkpoint_probe_rebuilds_the_actual_encoder_and_decoder(tmp_path):
    from chromatensor.codecs import LearnedRGBCodec

    model = LearnedRGBCodec(minimum=0, maximum=1, hidden=8, seed=9)
    path = tmp_path / "decoder.npz"
    np.savez_compressed(path, **{name: value.detach().numpy() for name, value in model.state_dict().items()})
    inputs = torch.linspace(0, 1, 17)
    record = {"id": "decoder-test", "task": "decoder", "representation": "learned_rgb", "seed": 9,
              "artifact_path": str(path), "dataset": {"sha256": "decoder-test-data"},
              "model_config": {"schema_version": 1, "task": "decoder", "representation": "learned_rgb",
                               "seed": 9, "minimum": 0, "maximum": 1, "hidden": 8}}
    probe = write_checkpoint_probe(record, inputs)
    arrays = load_numeric_npz(probe["path"])
    np.testing.assert_array_equal(arrays["predictions"], model.decode(model.encode(inputs)).numpy())
    assert verify_checkpoint_probe(record)["examples"] == 17


def test_attention_checkpoint_probe_rebuilds_recorded_nondefault_dimensions(tmp_path):
    from experiments.attention import PhaseAttentionPredictor

    mode = "chroma8_attention_codebook"
    model = PhaseAttentionPredictor(mode, hidden=12, heads=3, sequence_length=6, num_codes=8)
    path = tmp_path / "attention.ctcheckpoint"
    save_chroma_checkpoint(model, path)
    record = {"id": "attention-test", "task": "attention", "representation": mode, "seed": 7,
              "artifact_path": str(path), "dataset": {"sha256": "attention-test-data"},
              "model_config": {"schema_version": 1, "task": "attention", "representation": mode,
                               "seed": 7, "hidden": 12, "heads": 3, "sequence_length": 6, "num_codes": 8}}
    write_checkpoint_probe(record, torch.randn(5, 6, 2))
    assert verify_checkpoint_probe(record)["examples"] == 5


@pytest.mark.parametrize("field,value,error", [
    ("examples", 31, "input shape"),
    ("dataset_sha256", "another-data-set", "dataset hash"),
    ("checkpoint_bytes", 1, "checkpoint byte count"),
    ("threads", 0, "thread count"),
])
def test_probe_metadata_mismatches_are_rejected(tmp_path, field, value, error):
    record, inputs = scalar_record(tmp_path)
    write_checkpoint_probe(record, inputs)
    record["checkpoint_probe"][field] = value
    with pytest.raises(ValueError, match=error):
        verify_checkpoint_probe(record)


def test_precision_and_checkpoint_config_mismatches_are_rejected(tmp_path):
    record, inputs = scalar_record(tmp_path)
    write_checkpoint_probe(record, inputs)
    changed = copy.deepcopy(record)
    changed["checkpoint_probe"]["precision"]["output_dtype"] = "torch.float64"
    with pytest.raises(ValueError, match="precision mismatch"):
        verify_checkpoint_probe(changed)
    changed = copy.deepcopy(record)
    changed["model_config"]["representation"] = "fp32"
    with pytest.raises(ValueError, match="configuration does not match"):
        verify_checkpoint_probe(changed)


def test_modified_probe_file_and_checkpoint_are_rejected(tmp_path):
    record, inputs = scalar_record(tmp_path)
    write_checkpoint_probe(record, inputs)
    path = Path(record["checkpoint_probe"]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file hash or byte count"):
        verify_checkpoint_probe(record)
    write_checkpoint_probe(record, inputs)
    checkpoint = Path(record["artifact_path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        verify_checkpoint_probe(record)


def test_wrong_predictions_fail_replay_even_when_file_and_array_hashes_are_updated(tmp_path):
    record, inputs = scalar_record(tmp_path)
    write_checkpoint_probe(record, inputs)
    probe = record["checkpoint_probe"]
    arrays = load_numeric_npz(probe["path"])
    arrays["predictions"] += 1
    np.savez_compressed(probe["path"], **arrays)
    probe["sha256"] = sha256_file(probe["path"])
    probe["probe_bytes"] = Path(probe["path"]).stat().st_size
    probe["prediction_sha256"] = array_sha256(arrays["predictions"])
    with pytest.raises(ValueError, match="do not replay exactly"):
        verify_checkpoint_probe(record)


def test_npz_loader_rejects_pickle_and_declared_allocation_before_loading(tmp_path):
    path = tmp_path / "object.npz"
    np.savez(path, inputs=np.array([{"unsafe": True}], dtype=object), predictions=np.ones(1))
    with pytest.raises(ValueError, match="numeric dtype or allocation"):
        load_numeric_npz(path)
    raw = io.BytesIO()
    np.lib.format.write_array_header_1_0(raw, {"descr": "<f4", "fortran_order": False, "shape": (10**9,)})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("inputs.npy", raw.getvalue())
    with pytest.raises(ValueError, match="numeric dtype or allocation"):
        load_numeric_npz(path)


@pytest.mark.parametrize("attack", ["encoded_size", "directory_size", "entry_count"])
def test_npz_loader_checks_directory_and_file_budgets_before_zip_parsing(tmp_path, monkeypatch, attack):
    path = tmp_path / "bounded.npz"
    np.savez(path, values=np.arange(4, dtype=np.float32))
    data = bytearray(path.read_bytes())
    end = data.rfind(b"PK\x05\x06")
    if attack == "directory_size":
        struct.pack_into("<I", data, end + 12, 2**30)
    elif attack == "entry_count":
        struct.pack_into("<HH", data, end + 8, 65535, 65535)
    path.write_bytes(data)
    def forbidden(*args, **kwargs):
        raise AssertionError("NPZ directory parsed before checking its budget")
    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="max_file_bytes|directory exceeds"):
        load_numeric_npz(path, max_bytes=1 if attack == "encoded_size" else 1024)


@pytest.mark.parametrize("budget", [True, -1, 1.5, float("inf"), float("nan")])
def test_npz_loader_rejects_invalid_budget_types_before_file_access(tmp_path, budget):
    with pytest.raises(ValueError, match="nonnegative integer"):
        load_numeric_npz(tmp_path / "absent.npz", max_bytes=budget)


def test_npz_loader_checks_expansion_budget_before_member_reads(tmp_path, monkeypatch):
    path = tmp_path / "expanded.npz"
    np.savez_compressed(path, values=np.zeros(4096, dtype=np.float32))
    def forbidden(*args, **kwargs):
        raise AssertionError("NPZ payload opened before checking its expanded size")
    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    with pytest.raises(ValueError, match="max_expanded_bytes"):
        load_numeric_npz(path, max_bytes=1024)
