"""Integration checks for paired datasets, real training, and integer controls."""

import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from experiments.datasets import scalar_data
from experiments.models import StandardMLP, IntegerLinear, convert_integer_cpu
from experiments.training import auxiliary_loss, model_for, periodic_data, train_one
from chromatensor.nn import ChromaLinear, CodebookBottleneck


@pytest.fixture(scope="module", autouse=True)
def single_thread_small_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_paired_models_keep_exact_initial_parameters_and_dataset():
    ordinary = model_for("scalar", "fp32", (2, 3), 12, "cpu")
    chroma = model_for("scalar", "chroma8", (2, 3), 12, "cpu")
    integer = model_for("scalar", "int8", (2, 3), 12, "cpu")
    for control, chromatic, numeric in zip(ordinary.parameters(), chroma.parameters(), integer.parameters()):
        torch.testing.assert_close(control, chromatic, rtol=0, atol=0)
        torch.testing.assert_close(control, numeric, rtol=0, atol=0)
    first = scalar_data(12, "cpu", count=64)
    second = scalar_data(12, "cpu", count=64)
    for x, y in zip(first[:4], second[:4]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert first[4]["sha256"] == second[4]["sha256"]
    x = first[0]
    torch.testing.assert_close(chroma(x), integer(x), rtol=0, atol=0)


@pytest.mark.parametrize("representation", ["fp32", "int8", "chroma8", "rgb24", "fixed24"])
def test_actual_language_transformer_is_causal_for_tokens_prefixes_and_gradients(representation):
    model = model_for("language", representation, (32, 65), 7, "cpu").eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    altered = tokens.clone()
    altered[:, 3:] = torch.tensor([[10, 20, 30], [30, 20, 10]])
    full = model(tokens)
    torch.testing.assert_close(full[:, :3], model(altered)[:, :3], rtol=1e-6, atol=1e-6)
    for end in range(1, tokens.shape[1] + 1):
        prefix = model(tokens[:, :end])
        torch.testing.assert_close(full[:, :end], prefix, rtol=1e-6, atol=1e-6)
    # Gradients of earlier predictions cannot reach token-embedding rows used
    # only by the later timesteps. Weight range estimation stays detached.
    actual = model.model if hasattr(model, "model") else model
    model.zero_grad()
    model(torch.tensor([[1, 2, 3, 40, 41, 42]]))[:, :3].square().sum().backward()
    assert torch.count_nonzero(actual.embedding.weight.grad[40:43]) == 0
    assert actual.embedding.weight.grad[1:4].norm() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_weight_and_activation_ablations_apply_correct_flags():
    weights = model_for("scalar", "chroma_weights", (2, 3), 3, "cpu")
    activations = model_for("scalar", "chroma_activations", (2, 3), 3, "cpu")
    for module in weights.modules():
        if isinstance(module, ChromaLinear):
            assert module.weight_quantizer.enabled
            assert not module.activation_quantizer.enabled
    for module in activations.modules():
        if isinstance(module, ChromaLinear):
            assert not module.weight_quantizer.enabled
            assert module.activation_quantizer.enabled


def test_latent_auxiliary_loss_does_not_double_count():
    model = model_for("scalar", "latent", (2, 3), 3, "cpu")
    model(torch.randn(12, 2))
    modules = [module for module in model.modules() if isinstance(module, CodebookBottleneck)]
    assert len(modules) == 1
    torch.testing.assert_close(auxiliary_loss(model), modules[0].auxiliary_loss())
    auxiliary_loss(model).backward()
    assert modules[0].codebook.grad.norm() > 0


def test_periodic_hue_and_equal_level_numeric_data_are_paired():
    hue = periodic_data(5, "hue8", "cpu", count=64)
    numeric = periodic_data(5, "numeric_circular8", "cpu", count=64)
    assert hue[4]["sha256"] == numeric[4]["sha256"]
    for actual, expected in zip(hue[:4], numeric[:4]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tiny_real_training_writes_checkpoint_and_integer_cpu_evidence(tmp_path):
    data = scalar_data(7, "cpu", count=128)
    config = {"device": "cpu", "scalar_epochs": 12, "epochs": 1}
    record, samples, integer = train_one("scalar", "fp32", 7, data, config, tmp_path)
    assert record["metrics"]["validation_loss"] < record["metrics"]["initial_validation_loss"] * .8
    assert len(record["curves"]) == 13
    assert Path(record["artifact_path"]).is_file()
    assert record["metrics"]["training_seconds"] > 0
    assert record["metrics"]["training_throughput"] > 0
    assert samples and len(samples[0]["rgb"]) > 0
    assert integer["device"] == "cpu"
    assert integer["execution"]["linear_products"].startswith("INT32")
    assert integer["metrics"]["validation_loss"] < record["metrics"]["initial_validation_loss"]


def test_integer_cpu_linear_uses_discrete_products_and_close_outputs():
    torch.manual_seed(4)
    ordinary = StandardMLP(3, 16, 2).eval()
    quantized = convert_integer_cpu(ordinary)
    x = torch.randn(32, 3)
    assert all(module.weight_codes.dtype == torch.int8 for module in quantized.modules() if isinstance(module, IntegerLinear))
    with torch.no_grad():
        expected = ordinary(x)
        actual = quantized(x)
    assert (actual - expected).abs().mean() < .005
    assert torch.isfinite(quantized(torch.zeros(2, 3))).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("representation", ["fp16", "bf16", "chroma8"])
def test_cuda_mixed_precision_or_chromatic_step_is_finite(representation):
    from experiments.common import autocast_context
    from experiments.training import task_loss

    model = model_for("scalar", representation, (2, 3), 11, "cuda")
    x, target = torch.randn(32, 2, device="cuda"), torch.randn(32, 3, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scaler = torch.amp.GradScaler("cuda", enabled=representation == "fp16")
    with autocast_context(representation, "cuda"):
        loss = task_loss("scalar", model(x), target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
