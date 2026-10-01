"""Neural tests verify gradients, real optimization, controls, and accounting."""

import math

import pytest
import torch
from torch import nn

from chromatensor.autograd import ChromaQuantizer, codes_to_rgb, fake_chroma_quantize, rgb_to_codes
from chromatensor.nn import (
    ChromaActivation, ChromaAttention, ChromaEmbedding, ChromaLayerNorm, ChromaLinear,
    ChromaMLP, CodebookBottleneck, convert_to_chroma, module_quantization_report,
)


@pytest.mark.parametrize("mode,bits", [("chroma8", 8), ("rgb24", 24), ("fixed24", 24), ("int8", 8), ("int4", 4)])
def test_fake_quantization_discrete_forward_identity_backward(mode, bits):
    x = torch.linspace(-1, 1, 103, dtype=torch.float64, requires_grad=True)
    quantizer = ChromaQuantizer(representation=mode)
    reconstructed = quantizer(x)
    assert torch.max(abs(reconstructed - x)) <= 1 / ((1 << bits) - 1) + 1e-12
    weights = torch.linspace(.1, 1., x.numel(), dtype=x.dtype)
    (reconstructed * weights).sum().backward()
    torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)
    state = quantizer.last_state
    assert state.bits == bits
    if mode in {"chroma8", "rgb24"}:
        assert state.rgb.dtype == torch.uint8
        assert state.rgb.shape == (*x.shape, 3)
    else:
        assert state.rgb is None


def test_palette_is_bijective_and_rgb24_matches_fixed24_control():
    indices = torch.arange(65536)
    torch.testing.assert_close(rgb_to_codes(codes_to_rgb(indices)), indices)
    x = torch.linspace(-3.21, 4.73, 2049, dtype=torch.float64)
    rgb = fake_chroma_quantize(x, representation="rgb24")
    fixed = fake_chroma_quantize(x, representation="fixed24")
    torch.testing.assert_close(rgb, fixed, rtol=0, atol=0)
    chroma = fake_chroma_quantize(x, representation="chroma8")
    integer = fake_chroma_quantize(x, representation="int8")
    torch.testing.assert_close(chroma, integer, rtol=0, atol=0)


def test_quantizer_constant_empty_clipping_and_validation():
    constant = torch.full((4, 5), .127)
    torch.testing.assert_close(fake_chroma_quantize(constant), constant, rtol=0, atol=0)
    assert fake_chroma_quantize(torch.empty(0)).shape == (0,)
    x = torch.tensor([-100., 0., 100.], requires_grad=True)
    quantizer = ChromaQuantizer(4, min_value=-1., max_value=1.)
    y = quantizer(x)
    assert y[0] == -1 and y[-1] == 1
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    with pytest.raises(ValueError):
        ChromaQuantizer(0)
    with pytest.raises(ValueError):
        ChromaQuantizer(min_value=0.)
    with pytest.raises(ValueError):
        quantizer(torch.tensor([float("nan")]))
    with pytest.raises(TypeError):
        quantizer(torch.tensor([1], dtype=torch.int64))


@pytest.mark.parametrize("representation", ["int8", "chroma8", "rgb24", "fixed24"])
def test_feature_local_quantization_preserves_prefixes_batches_and_range_bytes(representation):
    values = torch.tensor([[[-1., .07, .4], [2., 2., 2.], [-1000., 0., 1000.]]])
    quantizer = ChromaQuantizer(representation=representation, range_axis=-1)
    whole = quantizer(values)
    state = quantizer.last_state
    assert state.range_axes == (2,)
    assert state.offset.shape == (1, 3, 1)
    assert state.range_metadata_bytes == 3 * 16
    assert state.constant_ranges == 1
    torch.testing.assert_close(whole[:, :2], quantizer(values[:, :2]), rtol=0, atol=0)
    isolated = quantizer(values[:, :1])
    batch = torch.cat((values[:, :1], values[:, :1] * 1000), dim=0)
    torch.testing.assert_close(isolated, quantizer(batch)[:1], rtol=0, atol=0)
    chunks = torch.cat([quantizer(values[:, start:start + 1]) for start in range(3)], dim=1)
    torch.testing.assert_close(whole, chunks, rtol=0, atol=0)


def test_feature_local_singletons_empty_inputs_and_fixed_ranges():
    quantizer = ChromaQuantizer(range_axis=-1)
    values = torch.tensor([[[-1.234], [8.7]]], requires_grad=True)
    torch.testing.assert_close(quantizer(values), values, rtol=0, atol=0)
    assert quantizer.last_state.constant_ranges == 2
    assert quantizer.last_state.range_metadata_bytes == 32
    quantizer(values).sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    assert quantizer(torch.empty(2, 0, 3)).shape == (2, 0, 3)
    assert quantizer(torch.empty(2, 3, 0)).shape == (2, 3, 0)
    assert quantizer(torch.tensor(.7)).item() == torch.tensor(.7).item()
    fixed = ChromaQuantizer(range_axis=-1, min_value=-1, max_value=1)
    fixed(values.detach())
    assert fixed.last_state.range_method == "fixed range"
    assert fixed.last_state.range_metadata_bytes == 16
    with pytest.raises(ValueError, match="range_axis"):
        ChromaQuantizer(range_axis=3)(values)
    with pytest.raises(TypeError, match="range_axis"):
        ChromaQuantizer(range_axis=True)


def test_neural_execution_reports_activation_range_axes_and_metadata():
    layer = ChromaLinear(3, 4)
    layer(torch.tensor([[1., 2., 3.], [4., 4., 4.]]))
    report = module_quantization_report(layer)
    activation = next(row for row in report["quantizers"] if row["module"] == "activation_quantizer")
    weight = next(row for row in report["quantizers"] if row["module"] == "weight_quantizer")
    assert activation["configured_range_axis"] == -1
    assert activation["last_range_axes"] == [1]
    assert activation["last_range_shape"] == [2, 1]
    assert activation["last_range_metadata_bytes"] == 32
    assert activation["last_constant_ranges"] == 1
    assert weight["configured_range_axis"] is None
    assert weight["last_range_metadata_bytes"] == 16
    assert activation["last_retained_inspection_bytes"] == 6 + 18 + 32


def test_linear_ordinary_control_and_gradients():
    torch.manual_seed(2)
    ordinary = nn.Linear(4, 3).double()
    chroma = convert_to_chroma(ordinary, quantize_weights=False, quantize_activations=False)
    x = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(chroma(x), ordinary(x), rtol=0, atol=0)
    chroma(x).square().mean().backward()
    assert chroma.weight.grad.norm() > 0
    assert x.grad.norm() > 0
    assert chroma.weight.data_ptr() != ordinary.weight.data_ptr()


def test_recursive_conversion_roundtrip_keeps_shared_modules_and_parameters():
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(9, 4)
            self.norm = nn.LayerNorm(4)
            self.projection = nn.Linear(4, 9, bias=False)
            self.projection.weight = self.embedding.weight
            self.alias = self.projection

        def forward(self, ids):
            return self.projection(self.norm(self.embedding(ids)))

    torch.manual_seed(1)
    original = TiedModel().eval()
    original.embedding.weight.requires_grad_(False)
    converted = convert_to_chroma(original, quantize_weights=False, quantize_activations=False)
    assert isinstance(converted.model.embedding, ChromaEmbedding)
    assert isinstance(converted.model.norm, ChromaLayerNorm)
    assert converted.model.projection is converted.model.alias
    assert converted.model.embedding.weight is converted.model.projection.weight
    assert not converted.model.embedding.weight.requires_grad
    inputs = torch.tensor([[1, 2, 3], [3, 4, 5]])
    torch.testing.assert_close(converted(inputs), original(inputs), rtol=0, atol=0)
    restored = converted.to_torch()
    assert type(restored.embedding) is nn.Embedding
    assert restored.projection is restored.alias
    assert restored.embedding.weight is restored.projection.weight
    assert not restored.training
    torch.testing.assert_close(restored(inputs), original(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("parent_registered_first", [False, True])
def test_conversion_preserves_parent_child_parameter_aliases_and_optimizer_updates(parent_registered_first):
    class TiedDirect(nn.Module):
        def __init__(self):
            super().__init__()
            parameter = nn.Parameter(torch.tensor([[.2, .3], [.4, .5]], dtype=torch.float64))
            layer = nn.Linear(2, 2, bias=False, dtype=torch.float64)
            layer.weight = parameter
            if parent_registered_first:
                self.direct = parameter
                self.linear = layer
            else:
                self.linear = layer
                self.direct = parameter

        def forward(self, x):
            return self.linear(x) + x @ self.direct.T

    original = TiedDirect().eval()
    converted = convert_to_chroma(original, quantize_weights=False, quantize_activations=False)
    assert converted.model.direct is converted.model.linear.weight
    assert len(list(converted.parameters())) == len(list(original.parameters())) == 1
    assert converted.model.direct is not original.direct
    x = torch.ones(2, 2, dtype=torch.float64)
    torch.testing.assert_close(converted(x), original(x), rtol=0, atol=0)
    optimizer = torch.optim.SGD(converted.parameters(), lr=.01)
    converted(x).square().mean().backward()
    optimizer.step()
    assert converted.model.direct is converted.model.linear.weight
    assert not torch.equal(converted.model.direct, original.direct)
    restored = converted.to_torch()
    assert restored.direct is restored.linear.weight
    assert restored.direct is not converted.model.direct
    assert len(list(restored.parameters())) == 1
    torch.testing.assert_close(restored(x), converted(x), rtol=0, atol=0)


def test_generic_conversion_rejects_attention_and_custom_layer_behavior():
    attention = nn.MultiheadAttention(8, 2, batch_first=True)
    for original in (attention, nn.Sequential(attention), nn.TransformerEncoderLayer(8, 2, batch_first=True)):
        with pytest.raises(TypeError, match="ChromaAttention.from_torch"):
            convert_to_chroma(original)
    assert type(attention.out_proj) is nn.modules.linear.NonDynamicallyQuantizableLinear

    class DoubleLinear(nn.Linear):
        def forward(self, x):
            return 2 * super().forward(x)

    with pytest.raises(TypeError, match="explicit adapter"):
        convert_to_chroma(DoubleLinear(2, 2), quantize_weights=False, quantize_activations=False)


def test_restoration_preserves_parent_ties_to_a_trainable_wrapped_activation():
    from chromatensor.nn.layers import to_torch

    original = nn.Module()
    original.activation = ChromaActivation(nn.PReLU(2))
    original.direct = original.activation.activation.weight
    original.alias = original.activation.activation
    restored = to_torch(original)
    assert restored.direct is restored.activation.weight
    assert restored.activation is restored.alias
    assert restored.direct is not original.direct
    assert len(list(restored.parameters())) == 1


def test_embedding_layernorm_and_activation_train():
    torch.manual_seed(3)
    embedding = ChromaEmbedding(7, 4, padding_idx=0, max_norm=1.)
    norm = ChromaLayerNorm(4)
    activation = ChromaActivation("gelu")
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    value = activation(norm(embedding(ids)))
    value.square().mean().backward()
    assert torch.isfinite(value).all()
    assert embedding.weight.grad[1:].norm() > 0
    assert torch.count_nonzero(embedding.weight.grad[0]) == 0
    assert norm.weight.grad.norm() > 0
    assert torch.max(embedding.weight.norm(dim=1)) <= 1. + 1e-6


def test_embedding_max_norm_cannot_change_previous_token_quantization():
    embedding = ChromaEmbedding(3, 3, max_norm=1.)
    with torch.no_grad():
        embedding.weight.copy_(torch.tensor([[.1, .2, .3], [.3, .2, .1], [100., -100., 50.]]))
    prefix = embedding(torch.tensor([[0, 1]])).detach()
    extended = embedding(torch.tensor([[0, 1, 2]])).detach()
    torch.testing.assert_close(prefix, extended[:, :2], rtol=0, atol=0)
    assert embedding.weight_quantizer.range_axis == -1


def test_chroma_mlp_reduces_real_sine_training_loss():
    # Fixed full-batch regression checks actual optimization, not just a forward.
    torch.manual_seed(12)
    x = torch.linspace(-math.pi, math.pi, 129).unsqueeze(1)
    target = x.sin()
    model = ChromaMLP(1, 24, 1, representation="chroma8")
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    with torch.no_grad():
        initial = (model(x) - target).square().mean().item()
    for _ in range(100):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = (model(x) - target).square().mean().item()
    assert final < initial * .15
    assert final < .03
    report = module_quantization_report(model)
    assert report["fully_chromatic"] is False
    assert report["floating_point_execution"] is True
    assert report["master_parameter_bytes"] == sum(p.numel() * 4 for p in model.parameters())
    assert all(q["encoded_values"] > 0 for q in report["quantizers"])


def test_codebook_has_discrete_rgb_states_and_updates_both_sides():
    torch.manual_seed(14)
    source = torch.randn(16, 3, requires_grad=True)
    bottleneck = CodebookBottleneck(3, 8)
    output = bottleneck(source)
    torch.testing.assert_close(output.detach(), bottleneck.codebook.detach()[bottleneck.last_indices])
    assert bottleneck.last_rgb.dtype == torch.uint8
    assert bottleneck.last_rgb.shape == (16, 3)
    (output.square().mean() + bottleneck.auxiliary_loss()).backward()
    assert source.grad.norm() > 0
    assert bottleneck.codebook.grad.norm() > 0
    assert 1 <= bottleneck.usage()["used_codes"] <= 8


def test_latent_codebook_reconstruction_loss_falls():
    torch.manual_seed(20)
    centers = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    samples = centers.repeat_interleave(16, 0) + .02 * torch.randn(64, 2)
    bottleneck = CodebookBottleneck(2, 16)
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=.05)
    initial = bottleneck(samples).sub(samples).square().mean().item()
    for _ in range(60):
        optimizer.zero_grad()
        bottleneck(samples)
        loss = bottleneck.auxiliary_loss()
        loss.backward()
        optimizer.step()
    final = bottleneck(samples).sub(samples).square().mean().item()
    assert final < initial * .05
    assert final < .004


def test_latent_mlp_optimization_and_conversion_after_forward():
    torch.manual_seed(9)
    model = ChromaMLP(1, [8], 1, latent_codes=16)
    x = torch.linspace(-1, 1, 32).reshape(-1, 1)
    loss = model(x).square().mean() + model.auxiliary_loss()
    loss.backward()
    assert next(m for m in model.modules() if isinstance(m, CodebookBottleneck)).codebook.grad.norm() > 0
    restored = model.to_torch()
    assert isinstance(restored[0], nn.Linear)
    assert any(isinstance(m, CodebookBottleneck) for m in restored.modules())


@pytest.mark.parametrize("batch_first", [True, False])
def test_attention_matches_torch_conventional_control(batch_first):
    torch.manual_seed(13)
    control = nn.MultiheadAttention(12, 3, dropout=0., batch_first=batch_first).double().eval()
    candidate = ChromaAttention.from_torch(control, quantize_weights=False, quantize_activations=False)
    x = torch.randn(2, 5, 12, dtype=torch.float64)
    if not batch_first:
        x = x.transpose(0, 1)
    mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
    expected, expected_weights = control(x, x, x, attn_mask=mask)
    actual, weights = candidate(x, attn_mask=mask, need_weights=True)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-10, atol=1e-10)
    actual.square().mean().backward()
    assert candidate.q_proj.weight.grad.norm() > 0
    restored = candidate.to_torch()
    torch.testing.assert_close(restored(x, x, x, attn_mask=mask)[0], expected, rtol=1e-10, atol=1e-10)


def test_attention_codebook_similarity_has_real_table_scores_and_gradients():
    torch.manual_seed(15)
    attention = ChromaAttention(8, 2, similarity_mode="codebook_similarity", num_codes=16)
    x = torch.randn(2, 5, 8, requires_grad=True)
    output = attention(x)
    (output.square().mean() + attention.auxiliary_loss()).backward()
    assert attention.q_proj.weight.grad.norm() > 0
    assert attention.codebook.codebook.grad.norm() > 0
    assert attention.last_q_codes.shape == (2, 2, 5)
    assert torch.isfinite(output).all()
    with torch.no_grad():
        attention(x)
        table = attention.codebook.codebook @ attention.codebook.codebook.t() / math.sqrt(attention.head_dim)
        scores = table[attention.last_q_codes.unsqueeze(-1), attention.last_k_codes.unsqueeze(-2)]
        torch.testing.assert_close(attention.last_attention, scores.softmax(-1))
    report = module_quantization_report(attention)
    assert report["attention"][0]["last_score_execution"] == "code-table scores without per-token QK dot product"
    with pytest.raises(ValueError):
        attention.to_torch()


def test_codebook_attention_task_loss_reaches_codebook_without_auxiliary_loss():
    torch.manual_seed(18)
    attention = ChromaAttention(8, 2, similarity_mode="codebook_similarity", num_codes=16)
    attention(torch.randn(2, 5, 8)).square().mean().backward()
    assert attention.codebook.codebook.grad is not None
    assert attention.codebook.codebook.grad.norm() > 0


def test_attention_fully_masked_rows_have_finite_gradients():
    attention = ChromaAttention(4, 1)
    x = torch.randn(1, 3, 4, requires_grad=True)
    output = attention(x, key_padding_mask=torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(attention.last_attention) == 0
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_attention_conversion_preserves_double_dtype_and_frozen_parameters():
    control = nn.MultiheadAttention(8, 2, batch_first=True).double().eval()
    for parameter in control.parameters():
        parameter.requires_grad_(False)
    candidate = ChromaAttention.from_torch(control, quantize_weights=False, quantize_activations=False)
    restored = candidate.to_torch()
    assert all(parameter.dtype == torch.float64 for parameter in restored.parameters())
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    torch.testing.assert_close(restored(x, x, x)[0], control(x, x, x)[0])


@pytest.mark.parametrize("mode", ["dot_product", "codebook_similarity"])
@pytest.mark.parametrize("training", [False, True])
def test_chromatic_attention_is_causal_for_prefixes_chunks_and_task_gradients(mode, training):
    torch.manual_seed(19)
    model = ChromaAttention(8, 2, representation="chroma8", similarity_mode=mode)
    model.train(training)
    values = torch.randn(2, 6, 8, requires_grad=True)
    altered = values.detach().clone()
    altered[:, 3:] *= 100
    full = model(values, is_causal=True)
    changed = model(altered, is_causal=True)
    torch.testing.assert_close(full[:, :3], changed[:, :3], rtol=1e-6, atol=1e-6)
    prefix = model(values[:, :3], is_causal=True)
    torch.testing.assert_close(full[:, :3], prefix, rtol=1e-6, atol=1e-6)
    # A query chunk still attends to its preceding keys. Its explicit mask uses
    # absolute token positions rather than resetting the causal diagonal.
    mask = torch.arange(6).unsqueeze(0) > torch.arange(2, 5).unsqueeze(1)
    chunk = model(values[:, 2:5], values, values, attn_mask=mask)
    torch.testing.assert_close(full[:, 2:5], chunk, rtol=1e-6, atol=1e-6)
    full[:, :3].square().sum().backward()
    assert torch.count_nonzero(values.grad[:, 3:]) == 0
    assert values.grad[:, :3].norm() > 0
    assert torch.isfinite(values.grad).all()
    assert model.q_proj.weight.grad.norm() > 0
