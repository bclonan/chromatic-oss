"""Check disjoint splits, deferred final testing and physical storage controls."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.common import dataset_hash
from scripts import run_release_research as release


def test_default_release_budget_and_complete_disjoint_training_partition():
    config = release.release_config("cpu")
    assert config["seeds"] == [7, 23, 41, 59]
    assert config["epochs"] == 20
    assert config["large_tensor_values"] == 10000000
    labels = torch.arange(10).repeat_interleave(torch.tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]))
    train_ids, development_ids = release.partition_indices(labels, 48000, 12000, 7)
    assert len(train_ids) == 48000 and len(development_ids) == 12000
    joined = torch.cat((train_ids, development_ids))
    torch.testing.assert_close(joined.sort().values, torch.arange(60000))
    again = release.partition_indices(labels, 48000, 12000, 7)
    torch.testing.assert_close(train_ids, again[0])
    torch.testing.assert_close(development_ids, again[1])


def test_large_tensor_uses_matching_physical_payload_and_metadata(tmp_path):
    result = release.large_tensor_experiment(4096, tmp_path)
    first, second = result["representations"]
    assert result["rgb24_fixed24_codes_exact"]
    assert result["rgb24_fixed24_decoded_exact"]
    assert result["rgbd_fp32_bits_exact"]
    assert first["payload_bytes"] == second["payload_bytes"] == 4096 * 3
    assert first["total_bytes"] == second["total_bytes"]
    assert Path(first["payload_path"]).read_bytes() == Path(second["payload_path"]).read_bytes()
    assert Path(first["metadata_path"]).read_bytes() == Path(second["metadata_path"]).read_bytes()
    assert result["rgbd_payload_bytes"] == 4096 * 4


@pytest.fixture
def completed_fixture_run(tmp_path, monkeypatch, request):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    if getattr(request, "param", None) == "fixed_wall_clock":
        monkeypatch.setattr(release, "now", lambda: "2026-10-01T01:00:00+00:00")
    # Unit fixtures have their own provenance and cannot pass the release gate.
    monkeypatch.setattr(release, "capture_provenance", lambda config: {"source": "unit fixture"})
    monkeypatch.setattr(release, "verify_provenance", lambda *args, **kwargs: True)
    monkeypatch.setattr(release, "artifact_manifest", lambda paths: [str(path) for path in paths])
    generator = torch.Generator().manual_seed(12)
    images = torch.randint(256, (80, 28, 28), generator=generator, dtype=torch.uint8)
    labels = torch.arange(80) % 10
    source = (images, labels, {"source": "unit fixture", "namespace": "fixture_train", "source_examples": 80,
                              "source_sha256": dataset_hash(images, labels)})
    config = release.release_config("cpu", True)
    config.update(epochs=1, train_images=32, development_images=16, final_test_images=12, large_tensor_values=128)
    artifacts = tmp_path / "run-artifacts"
    events = []

    def load_final_test():
        checkpoints = list(artifacts.glob("*.ctcheckpoint"))
        assert len(checkpoints) == 3
        assert all(path.with_name(path.stem + "-probe.npz").is_file() for path in checkpoints)
        events.append("final test loaded after all three trained checkpoints")
        return images[:20], labels[:20], {"source": "unit fixture", "namespace": "fixture_test", "source_examples": 20,
                                          "source_sha256": dataset_hash(images[:20], labels[:20])}

    try:
        result = release.run_release_research(config, tmp_path / "result.json", artifacts,
                                             training_source=source, final_test_loader=load_final_test)
        yield result, events
    finally:
        torch.set_num_threads(previous)


def test_smoke_pipeline_defers_final_test_until_all_checkpoints_and_preserves_controls(completed_fixture_run):
    result, events = completed_fixture_run
    assert events == ["final test loaded after all three trained checkpoints"]
    assert release.validate_release_research(result, require_release=False)
    assert result["status"] == "complete"
    assert result["provenance"]["source"] == "unit fixture"
    with pytest.raises(ValueError, match="Reduced runs"):
        release.validate_release_research(result)
    for record in result["experiments"]:
        assert record["id"].startswith(result["run_id"])
        assert record["final_test_metrics"]["examples"] == 12
        assert record["dataset"]["namespace"] == "fixture_train"


def test_final_test_timing_and_metric_tampering_are_rejected(completed_fixture_run):
    result, _ = completed_fixture_run
    changed = copy.deepcopy(result)
    changed["official_test_opened_at"] = changed["created_at"]
    with pytest.raises(ValueError, match="before training completed"):
        release.validate_release_research(changed, require_release=False)
    changed = copy.deepcopy(result)
    changed["experiments"][0]["final_test_metrics"]["accuracy"] = 1.
    with pytest.raises(ValueError, match="reported accuracy"):
        release.validate_release_research(changed, require_release=False)


@pytest.mark.parametrize("completed_fixture_run", ["fixed_wall_clock"], indirect=True)
def test_fixed_wall_clock_still_records_actual_stage_order(completed_fixture_run):
    result, events = completed_fixture_run
    assert events == ["final test loaded after all three trained checkpoints"]
    assert result["created_at"] == result["training_completed_at"] == result["official_test_opened_at"] == result["completed_at"]
    assert 0 < result["training_completed_monotonic_ns"] < result["official_test_opened_monotonic_ns"]
    assert result["stage_clock"]["api"] == "time.perf_counter_ns"
    assert result["stage_clock"]["monotonic"] is True
    assert result["stage_clock"]["adjustable"] is False
    assert release.validate_release_research(result, require_release=False)
    changed = copy.deepcopy(result)
    changed["official_test_opened_monotonic_ns"] = changed["training_completed_monotonic_ns"]
    with pytest.raises(ValueError, match="before training completed according to monotonic timestamps"):
        release.validate_release_research(changed, require_release=False)


def test_missing_or_invalid_monotonic_stage_timestamps_are_rejected(completed_fixture_run):
    result, _ = completed_fixture_run
    for key in ("training_completed_monotonic_ns", "official_test_opened_monotonic_ns"):
        changed = copy.deepcopy(result)
        del changed[key]
        with pytest.raises(ValueError, match="monotonic timestamps"):
            release.validate_release_research(changed, require_release=False)
        for invalid in (None, True, 0, -1, 1.5, "123"):
            changed = copy.deepcopy(result)
            changed[key] = invalid
            with pytest.raises(ValueError, match="monotonic timestamps"):
                release.validate_release_research(changed, require_release=False)
    for invalid_clock in (None, {}, {**result["stage_clock"], "monotonic": False},
                          {**result["stage_clock"], "resolution_seconds": 0}):
        changed = copy.deepcopy(result)
        changed["stage_clock"] = invalid_clock
        with pytest.raises(ValueError, match="monotonic clock metadata"):
            release.validate_release_research(changed, require_release=False)


def test_equal_or_backwards_monotonic_stage_order_is_rejected(completed_fixture_run):
    result, _ = completed_fixture_run
    for offset in (0, -1):
        changed = copy.deepcopy(result)
        changed["official_test_opened_monotonic_ns"] = changed["training_completed_monotonic_ns"] + offset
        with pytest.raises(ValueError, match="before training completed according to monotonic timestamps"):
            release.validate_release_research(changed, require_release=False)


def test_overlapping_partition_indices_are_rejected_even_with_updated_hash(completed_fixture_run):
    result, _ = completed_fixture_run
    info = result["experiments"][0]["dataset"]
    arrays = release.load_numeric_npz(info["index_artifact"])
    arrays["development_ids"][0] = arrays["train_ids"][0]
    np.savez_compressed(info["index_artifact"], **arrays)
    for record in result["experiments"]:
        record["dataset"]["index_sha256"] = release.sha256_file(info["index_artifact"])
    with pytest.raises(ValueError, match="disjoint valid partitions"):
        release.validate_release_research(result, require_release=False)


def test_rgbd_payload_tampering_is_rejected_even_with_a_new_checksum(completed_fixture_run):
    result, _ = completed_fixture_run
    item = result["large_tensor"]["rgbd"]
    path = Path(item["payload_path"])
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    item["sha256"] = release.sha256_file(path)
    with pytest.raises(ValueError, match="does not reconstruct its source"):
        release.validate_release_research(result, require_release=False)


def test_validator_resolves_relative_artifacts_against_its_root(completed_fixture_run, monkeypatch):
    result, _ = completed_fixture_run
    root = release.location(result["experiments"][0]["artifact_path"]).parent.parent
    changed = copy.deepcopy(result)
    path_fields = {"artifact_path", "index_artifact", "prediction_path", "payload_path", "metadata_path", "path"}
    visited = set()

    def make_relative(value):
        if isinstance(value, list):
            for item in value:
                make_relative(item)
        elif isinstance(value, dict):
            if id(value) in visited:
                return
            visited.add(id(value))
            for key, item in tuple(value.items()):
                if key in path_fields and isinstance(item, str):
                    value[key] = release.location(item).relative_to(root).as_posix()
                else:
                    make_relative(item)

    make_relative(changed)
    monkeypatch.chdir(root.parent)
    assert release.validate_release_research(changed, root=root, require_release=False)
