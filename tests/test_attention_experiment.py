import pytest
import torch

from experiments.attention import MODES, PhaseAttentionPredictor, phase_sequence_data, run_attention_ablation, _shared_initialization_hash
from experiments.common import seed_all


def test_phase_sequence_data_pairs_all_modes():
    first = phase_sequence_data(1, count=16, validation_count=8, sequence_length=6)
    second = phase_sequence_data(1, count=16, validation_count=8, sequence_length=6)
    assert first[0].shape == (16, 6, 2)
    assert first[4]["sha256"] == second[4]["sha256"]
    for actual, expected in zip(first[:4], second[:4]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(first[0][:, 1:], first[1][:, :-1], rtol=0, atol=0)


def test_all_attention_shared_parameters_have_exact_paired_initialization():
    hashes = []
    for mode in MODES:
        seed_all(4)
        model = PhaseAttentionPredictor(mode)
        hashes.append(_shared_initialization_hash(model))
        prediction = model(torch.randn(2, 12, 2))
        assert prediction.shape == (2, 12, 2)
    assert len(set(hashes)) == 1


def test_small_attention_experiment_trains_and_returns_measured_records(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        records, samples = run_attention_ablation({"device": "cpu", "seeds": [3],
            "attention_epochs": 4, "attention_train_count": 64, "attention_validation_count": 32,
            "attention_sequence_length": 6, "attention_hidden": 12, "attention_heads": 3,
            "attention_num_codes": 8}, artifact_dir=tmp_path)
    finally:
        torch.set_num_threads(previous_threads)
    assert len(records) == 3
    assert len(samples) == 4
    assert len({record["shared_initialization_sha256"] for record in records}) == 1
    assert len({record["dataset"]["sha256"] for record in records}) == 1
    for record in records:
        assert record["metrics"]["validation_loss"] < record["metrics"]["initial_validation_loss"]
        assert record["metrics"]["training_seconds"] > 0
        assert len(record["curves"]) == 5
        assert record["execution"]["fully_chromatic"] is False
    codebook = records[-1]
    assert codebook["execution"]["attention"][0]["last_score_execution"] == "code-table scores without per-token QK dot product"


def test_invalid_attention_experiment_config_fails_before_training(tmp_path):
    with pytest.raises(ValueError):
        run_attention_ablation({"attention_epochs": 0}, artifact_dir=tmp_path)


@pytest.mark.parametrize("mode", MODES)
def test_phase_predictor_causal_claim_holds_when_future_coordinates_change(mode):
    seed_all(7)
    model = PhaseAttentionPredictor(mode).eval()
    values = torch.randn(2, 12, 2)
    altered = values.clone()
    altered[:, 5:] *= 100
    with torch.no_grad():
        full = model(values)
        changed = model(altered)
        prefix = model(values[:, :5])
    torch.testing.assert_close(full[:, :5], changed[:, :5], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full[:, :5], prefix, rtol=1e-6, atol=1e-6)
