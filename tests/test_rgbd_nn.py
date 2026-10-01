"""RGBD neural forwards decode actual color bytes with floating-point kernels."""

import struct

import pytest
import torch
from torch import nn

from chromatensor.autograd import ChromaQuantizer
from chromatensor.codecs import RGBDCodec
from chromatensor.nn import ChromaLinear, convert_to_chroma, module_quantization_report


def _bits(values):
    return values.contiguous().view(torch.int32)


def _independent_rgbd(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    fraction = ((bits >> 8) & 0x800000) | (bits & 0x7FFFFF)
    return [(fraction >> 16) & 255, (fraction >> 8) & 255,
            fraction & 255, (bits >> 23) & 255]


def test_rgbd_forward_stores_exact_sign_fraction_exponent_bytes():
    bitpatterns = torch.tensor([0, -2147483648, 1, -2147483647, 0x007FFFFF,
                                0x00800000, 0x3F800000, -1082130432,
                                0x3FC00000, 0x40000000, 0x7F7FFFFF], dtype=torch.int32)
    values = bitpatterns.view(torch.float32)
    quantizer = ChromaQuantizer(representation="rgbd")
    decoded = quantizer(values)
    state = quantizer.last_state
    assert state.bits == quantizer.bits == 32
    assert state.rgbd.dtype == torch.uint8
    assert state.rgbd.shape == (*values.shape, 4)
    assert state.rgbd.tolist() == [_independent_rgbd(value) for value in values.tolist()]
    assert torch.equal(_bits(decoded), bitpatterns)
    assert state.codes is state.rgbd
    assert state.rgb is None
    assert quantizer.forward_calls == 1
    assert quantizer.encoded_values == values.numel()
    assert state.payload_bytes == state.retained_state_bytes == values.numel() * 4
    assert state.range_metadata_bytes == 0
    assert state.offset.numel() == state.scale.numel() == 0
    assert state.range_axes == ()
    assert state.range_method == "RGBD bit packing"


@pytest.mark.parametrize("channel,byte,expected", [
    (0, 128, -1.0),
    (0, 64, 1.5),
    (1, 1, 1.000030517578125),
    (2, 1, 1.0000001192092896),
    (3, 128, 2.0),
])
def test_rgbd_byte_mutation_changes_neural_state_decode(channel, byte, expected):
    quantizer = ChromaQuantizer(representation="rgbd")
    original = torch.tensor([1.0])
    quantizer(original)
    state = quantizer.last_state
    assert state.rgbd.tolist() == [[0, 0, 0, 127]]
    state.rgbd[0, channel] = byte
    assert state.decode().item() == expected
    assert original.item() == 1.0


def test_rgbd_neural_state_rejects_nonfinite_exponent_mutation():
    state = ChromaQuantizer(representation="rgbd").encode(torch.tensor([1.0]))
    state.rgbd[0, 3] = 255
    with pytest.raises(ValueError, match="D=255"):
        state.decode()


def test_rgbd_enabled_forward_decodes_new_bytes_each_call(monkeypatch):
    encode, decode = RGBDCodec.encode, RGBDCodec.decode
    calls = {"encode": 0, "decode": 0}

    def changed_encode(codec, values):
        calls["encode"] += 1
        codes = encode(codec, values)
        codes[..., 3] = 127 + calls["encode"]
        return codes

    def observed_decode(codec, codes):
        calls["decode"] += 1
        return decode(codec, codes)

    monkeypatch.setattr(RGBDCodec, "encode", changed_encode)
    monkeypatch.setattr(RGBDCodec, "decode", observed_decode)
    quantizer = ChromaQuantizer(representation="rgbd")
    values = torch.tensor([1.0], requires_grad=True)
    assert quantizer(values).item() == 2.0
    second = quantizer(values)
    assert second.item() == 4.0
    assert calls == {"encode": 2, "decode": 2}
    second.sum().backward()
    assert values.grad.item() == 1.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rgbd_forward_preserves_declared_dtype_and_fp32_interoperation(dtype):
    values = torch.tensor([-2.25, -.1, 0., .1, 1.125, 3.75], dtype=dtype, requires_grad=True)
    quantizer = ChromaQuantizer(representation="rgbd", range_axis=-1)
    decoded = quantizer(values)
    assert decoded.dtype == dtype
    expected = values.detach().to(torch.float32).to(dtype)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    if dtype == torch.float64:
        assert decoded[1] != values[1]
    weights = torch.linspace(.25, 1.5, values.numel(), dtype=dtype)
    (decoded * weights).sum().backward()
    torch.testing.assert_close(values.grad, weights, rtol=0, atol=0)
    assert quantizer.last_state.range_axes == ()
    assert quantizer.last_state.range_metadata_bytes == 0


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
def test_rgbd_scalar_empty_and_multidimensional_shapes(shape):
    values = torch.full(shape, -.125)
    quantizer = ChromaQuantizer(representation="rgbd", range_axis=-1)
    decoded = quantizer(values)
    assert decoded.shape == values.shape
    assert quantizer.last_state.rgbd.shape == (*values.shape, 4)
    assert quantizer.last_state.payload_bytes == values.numel() * 4
    assert quantizer.last_state.retained_state_bytes == values.numel() * 4
    assert torch.equal(_bits(decoded), _bits(values))


@pytest.mark.parametrize("values,message", [
    ([float("nan")], "NaN and infinity"),
    ([float("inf")], "NaN and infinity"),
    ([-float("inf")], "NaN and infinity"),
    ([1e39], "overflow"),
    ([1e-100], "underflows"),
])
def test_rgbd_forward_rejects_values_outside_finite_fp32(values, message):
    quantizer = ChromaQuantizer(representation="rgbd")
    with pytest.raises(ValueError, match=message):
        quantizer(torch.tensor(values, dtype=torch.float64))
    assert quantizer.encoded_values == 0
    assert quantizer.last_state is None


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64, torch.bool])
def test_rgbd_neural_encoding_requires_real_floating_inputs(dtype):
    with pytest.raises(TypeError, match="floating-point"):
        ChromaQuantizer(representation="rgbd")(torch.tensor([1], dtype=dtype))


@pytest.mark.parametrize("kwargs", [
    {"min_value": -1.}, {"max_value": 1.}, {"min_value": -1., "max_value": 1.},
])
def test_rgbd_neural_encoding_rejects_fixed_range_quantization(kwargs):
    with pytest.raises(ValueError, match="RGBD bit packing"):
        ChromaQuantizer(representation="rgbd", **kwargs)


def test_rgbd_range_axis_does_not_trigger_minimum_or_maximum_reduction(monkeypatch):
    def reject_reduction(*args, **kwargs):
        raise AssertionError("RGBD encoding must not estimate a numerical range")

    values = torch.tensor([[1., 2., 3.], [-1000., 0., 1000.]])
    monkeypatch.setattr(torch.Tensor, "amin", reject_reduction)
    monkeypatch.setattr(torch.Tensor, "amax", reject_reduction)
    quantizer = ChromaQuantizer(representation="rgbd", range_axis=-1)
    assert torch.equal(_bits(quantizer(values)), _bits(values))
    assert quantizer.last_state.range_axes == ()


def test_disabled_rgbd_quantizer_preserves_input_without_encoding():
    values = torch.tensor([1.], requires_grad=True)
    quantizer = ChromaQuantizer(representation="rgbd", enabled=False)
    assert quantizer(values) is values
    assert quantizer.last_state is None
    assert quantizer.encoded_values == 0


def test_rgbd_model_conversion_matches_fp32_forward_gradients_and_optimizer_step():
    torch.manual_seed(331)
    original = nn.Sequential(nn.Embedding(9, 4), nn.LayerNorm(4), nn.Tanh(), nn.Linear(4, 3))
    converted = convert_to_chroma(original, representation="rgbd")
    token_ids = torch.tensor([[0, 2, 5], [8, 1, 3]])
    expected = original(token_ids)
    actual = converted(token_ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for ordinary, chromatic in zip(original.parameters(), converted.parameters()):
        assert ordinary.grad is not None and chromatic.grad is not None
        torch.testing.assert_close(chromatic.grad, ordinary.grad, rtol=0, atol=0)
    ordinary_optimizer = torch.optim.SGD(original.parameters(), lr=.01)
    chromatic_optimizer = torch.optim.SGD(converted.parameters(), lr=.01)
    ordinary_optimizer.step()
    chromatic_optimizer.step()
    torch.testing.assert_close(converted(token_ids), original(token_ids), rtol=0, atol=0)
    report = module_quantization_report(converted)
    assert report["floating_point_execution"]
    assert not report["fully_chromatic"]
    assert len(report["quantizers"]) == 6
    for quantizer in report["quantizers"]:
        assert quantizer["representation"] == "rgbd"
        assert quantizer["forward_calls"] == 2
        assert quantizer["payload_format"] == "RGBD uint8 quadruples"
        assert quantizer["last_range_axes"] == []
        assert quantizer["last_range_metadata_bytes"] == 0
        assert quantizer["last_payload_bytes"] == quantizer["last_retained_inspection_bytes"]
        assert quantizer["last_payload_bytes"] == quantizer["last_ideal_packed_code_bytes"]
        assert quantizer["last_payload_bytes"] == quantizer["encoded_values"] * 4 // 2


def test_rgbd_unexecuted_and_executed_reports_describe_bytes_without_range_metadata():
    layer = ChromaLinear(3, 4, representation="rgbd")
    before = module_quantization_report(layer)
    for quantizer in before["quantizers"]:
        assert quantizer["range_method"] == "RGBD bit packing"
        assert quantizer["last_payload_bytes"] == 0
        assert quantizer["last_range_metadata_bytes"] == 0
        assert quantizer["payload_format"] == "RGBD uint8 quadruples"
    layer(torch.tensor([[1., 2., 3.], [4., 4., 4.]]))
    after = module_quantization_report(layer)
    activation, weight = (next(row for row in after["quantizers"] if row["module"] == name)
                          for name in ("activation_quantizer", "weight_quantizer"))
    assert activation["configured_range_axis"] == -1
    assert activation["last_payload_bytes"] == activation["last_ideal_packed_code_bytes"] == 24
    assert weight["last_payload_bytes"] == weight["last_ideal_packed_code_bytes"] == 48
    assert activation["last_range_shape"] == weight["last_range_shape"] == [0]
    assert activation["last_range_metadata_bytes"] == weight["last_range_metadata_bytes"] == 0


def test_existing_chroma8_and_int8_controls_keep_identical_forward_values():
    values = torch.linspace(-.71, 2.39, 1025)
    chromatic = ChromaQuantizer(representation="chroma8")
    integer = ChromaQuantizer(representation="int8")
    torch.testing.assert_close(chromatic(values), integer(values), rtol=0, atol=0)
    assert chromatic.last_state.rgbd is None
    assert integer.last_state.rgbd is None
    assert chromatic.bits == integer.bits == 8
    assert chromatic.last_state.range_metadata_bytes == integer.last_state.range_metadata_bytes == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_rgbd_cpu_cuda_forward_bytes_and_gradients_match():
    values = torch.tensor([0., -0., 1.0, -1.0, 1.5, torch.finfo(torch.float32).tiny,
                           torch.finfo(torch.float32).max])
    cpu = ChromaQuantizer(representation="rgbd")
    cuda = ChromaQuantizer(representation="rgbd")
    cpu_result = cpu(values)
    cuda_result = cuda(values.cuda())
    assert torch.equal(cpu.last_state.rgbd, cuda.last_state.rgbd.cpu())
    assert torch.equal(_bits(cpu_result), _bits(cuda_result.cpu()))
    gradient_values = torch.tensor([-.125, .75, 1.5], device="cuda", requires_grad=True)
    cuda(gradient_values).sum().backward()
    torch.testing.assert_close(gradient_values.grad, torch.ones_like(gradient_values), rtol=0, atol=0)
