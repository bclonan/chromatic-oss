"""Verify experiment pairing against the public spectral cube codec."""
import math

import pytest
import torch

from chromatensor.codecs import RGBCodec
from chromatensor.holographic import HolographicCube
from experiments.common import dataset_hash
from experiments.cube import batched_polar, encoded_inputs, smooth_cubes


@pytest.mark.parametrize("signal", ["smooth", "noise", "zero", "constant"])
def test_batched_codec_matches_public_cube_codes_scale_and_coefficients(signal):
    if signal == "smooth":
        values = smooth_cubes(4, dimensions=4, seed=7)
    elif signal == "noise":
        values = torch.rand((4, 3, 3, 3, 3), generator=torch.Generator().manual_seed(11)) * 2 - 1
    elif signal == "zero":
        values = torch.zeros(4, 3, 3, 3, 3)
    else:
        values = torch.full((4, 3, 3, 3, 3), 0.37)
    codes, scales, coefficients = batched_polar(values)
    assert codes.dtype == torch.uint8
    assert codes.numel() * codes.element_size() == values.numel() * 3
    for index, field in enumerate(values):
        cube = HolographicCube.from_tensor(field)
        assert torch.equal(codes[index], cube.codes)
        assert scales[index].item() == cube.amplitude_scale
        assert torch.equal(coefficients[index], cube.coefficients())
    magnitude_ids = codes[..., 1].long() * 256 + codes[..., 2].long()
    assert torch.all(codes[..., 0][magnitude_ids == 0] == 0)


@pytest.mark.parametrize("coverage", [0, 0.25, 0.5, 0.75, 1])
def test_equal_bit_spectral_controls_have_identical_network_inputs_and_masks(coverage):
    values = smooth_cubes(12, seed=23)
    colored, chromatic_mask = encoded_inputs(values, "holographic24", coverage=coverage, seed=31)
    numeric, numeric_mask = encoded_inputs(values, "numeric_polar24", coverage=coverage, seed=31)
    direct, direct_mask = encoded_inputs(values, "direct_rgb24", coverage=coverage, seed=31)
    assert torch.equal(colored, numeric)
    assert torch.equal(chromatic_mask, numeric_mask)
    assert torch.equal(chromatic_mask, direct_mask)
    field_size = math.prod(values.shape[1:])
    assert colored.shape == (len(values), field_size * 2)
    assert colored.dtype == numeric.dtype == direct.dtype == torch.float32
    assert torch.equal(colored[:, field_size:], chromatic_mask.flatten(1).float())
    assert torch.equal(direct[:, field_size:], direct_mask.flatten(1).float())
    assert torch.all(chromatic_mask.flatten(1).sum(-1) == math.floor(field_size * coverage))


def test_spectral_network_input_matches_actual_public_decode_with_returned_masks():
    values = smooth_cubes(5, seed=23)
    incoming, masks = encoded_inputs(values, "holographic24", coverage=0.5, seed=31)
    field_size = math.prod(values.shape[1:])
    for index, field in enumerate(values):
        cube = HolographicCube.from_tensor(field)
        expected = cube.decode(masks[index]).float().flatten()
        assert torch.equal(incoming[index, :field_size], expected)
    direct, direct_masks = encoded_inputs(values, "direct_rgb24", coverage=0.5, seed=31)
    codec = RGBCodec(-1, 1)
    expected = (codec.decode(codec.encode(values)).float() * direct_masks).flatten(1)
    assert torch.equal(direct[:, :field_size], expected)


def test_dataset_and_masks_reproduce_exactly_and_holdout_seed_changes_samples():
    global_state = torch.random.get_rng_state().clone()
    training = smooth_cubes(16, seed=7)
    same_training = smooth_cubes(16, seed=7)
    validation = smooth_cubes(16, seed=8)
    assert torch.equal(training, same_training)
    assert dataset_hash(training) == dataset_hash(same_training)
    assert dataset_hash(training) != dataset_hash(validation)
    train_samples = {dataset_hash(field) for field in training}
    validation_samples = {dataset_hash(field) for field in validation}
    assert train_samples.isdisjoint(validation_samples)
    _, mask = encoded_inputs(training, "holographic24", seed=9)
    _, same_mask = encoded_inputs(training, "numeric_polar24", seed=9)
    _, changed_mask = encoded_inputs(training, "holographic24", seed=10)
    assert dataset_hash(mask) == dataset_hash(same_mask)
    assert dataset_hash(mask) != dataset_hash(changed_mask)
    assert torch.equal(global_state, torch.random.get_rng_state())


@pytest.mark.parametrize("representation", ["holographic24", "numeric_polar24", "direct_rgb24"])
def test_erasing_every_input_value_keeps_zero_field_and_explicit_empty_mask(representation):
    values = smooth_cubes(3, seed=7)
    incoming, mask = encoded_inputs(values, representation, coverage=0, seed=31)
    assert not mask.any()
    assert not incoming.any()


def test_unknown_representation_fails_instead_of_becoming_a_default_control():
    with pytest.raises(ValueError, match="representation"):
        encoded_inputs(smooth_cubes(2), "unknown")


@pytest.mark.parametrize("coverage", [-0.1, 1.1, True, "0.5", float("nan"), float("inf")])
def test_invalid_retained_fraction_fails_before_generating_a_mask(coverage):
    with pytest.raises(ValueError, match="coverage"):
        encoded_inputs(smooth_cubes(2), "holographic24", coverage=coverage)


def test_interrupted_cube_rerun_preserves_prior_artifacts_and_record(tmp_path, monkeypatch):
    import json
    from scripts import run_cube_experiments as runner
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "source.py").write_text("x = 1\n")
    old = tmp_path / "artifacts/checkpoints/cube/prior/model.ctcheckpoint"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"prior verified checkpoint")
    output = tmp_path / "results/cube.json"
    output.parent.mkdir()
    output.write_text(json.dumps({"run_id": "prior", "artifact_path": str(old)}))
    before = output.read_bytes()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "cube_benchmarks", lambda seeds: ([], []))
    calls = []
    def train(representation, seed, folder, epochs):
        calls.append(folder)
        (folder / (representation + ".ctcheckpoint")).write_bytes(b"new interrupted run")
        raise RuntimeError("simulated optimization failure")
    monkeypatch.setattr(runner, "train_cube_reconstructor", train)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="optimization failure"):
            runner.main(["--epochs", "1", "--output", str(output)])
    assert output.read_bytes() == before
    assert old.read_bytes() == b"prior verified checkpoint"
    assert len(calls) == 2 and calls[0] != calls[1]
    assert all(path.parent == old.parent.parent for path in calls)


def test_trained_cube_records_actual_reload_prediction_hash(tmp_path):
    from experiments.common import dataset_hash
    from experiments.cube import train_cube_reconstructor
    from chromatensor import load_chroma_checkpoint
    from chromatensor.nn import convert_to_chroma
    from experiments.models import StandardMLP
    record = train_cube_reconstructor("holographic24", 7, tmp_path, epochs=1)
    model = convert_to_chroma(StandardMLP(162, 128, 81), bits=8, representation="chroma8")
    load_chroma_checkpoint(record["artifact_path"], model)
    model.eval()
    inputs, _ = encoded_inputs(smooth_cubes(256, seed=8), "holographic24", seed=10)
    with torch.no_grad():
        prediction = model(inputs)
    assert dataset_hash(prediction) == record["validation_prediction_sha256"]
    assert record["model_config"]["inputs"] == 162


def test_direct_cube_training_calls_cannot_overwrite_existing_checkpoints(tmp_path):
    from experiments.cube import train_cube_reconstructor
    from experiments.provenance import file_hash
    first = train_cube_reconstructor("direct_rgb24", 7, tmp_path, epochs=1)
    digest = file_hash(first["artifact_path"])
    second = train_cube_reconstructor("direct_rgb24", 7, tmp_path, epochs=2)
    assert first["artifact_path"] != second["artifact_path"]
    assert file_hash(first["artifact_path"]) == digest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("coverage", [0, 0.5, 1])
def test_actual_cuda_paired_inputs_and_masks_match_public_decode(coverage):
    cpu_values = smooth_cubes(4, dimensions=4, seed=23)
    values = cpu_values.to("cuda")
    colored, chromatic_mask = encoded_inputs(values, "holographic24", coverage=coverage, seed=31)
    numeric, numeric_mask = encoded_inputs(values, "numeric_polar24", coverage=coverage, seed=31)
    direct, direct_mask = encoded_inputs(values, "direct_rgb24", coverage=coverage, seed=31)
    _, cpu_mask = encoded_inputs(cpu_values, "holographic24", coverage=coverage, seed=31)
    assert colored.device.type == numeric.device.type == direct.device.type == "cuda"
    assert chromatic_mask.device.type == numeric_mask.device.type == direct_mask.device.type == "cuda"
    assert torch.equal(colored, numeric)
    assert torch.equal(chromatic_mask, numeric_mask)
    assert torch.equal(chromatic_mask, direct_mask)
    assert torch.equal(chromatic_mask.cpu(), cpu_mask)
    field_size = math.prod(values.shape[1:])
    assert torch.equal(colored[:, field_size:], chromatic_mask.flatten(1).float())
    assert torch.all(chromatic_mask.flatten(1).sum(-1) == math.floor(field_size * coverage))
    for index, field in enumerate(values):
        expected = HolographicCube.from_tensor(field).decode(chromatic_mask[index]).float().flatten()
        assert torch.equal(colored[index, :field_size], expected)
