import json

import pytest
import torch

from chromatensor.codecs import LearnedRGBCodec, codec_from_config, LearnedCodec
from chromatensor.decoders import LearnedNeuralDecoder
from chromatensor.tensor import ChromaTensor


def test_joint_neural_rgb_fit_improves_heldout_and_learns_colors():
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(19)
    train = torch.rand(2048, generator=generator) * 2 - 1
    heldout = torch.rand(1024, generator=generator) * 2 - 1
    codec = LearnedRGBCodec(seed=7)
    before = (codec.decode(codec.encode(heldout)) - heldout).square().mean()
    initial_colors = codec.encode(torch.tensor([-.8, 0., .8])).clone()
    codec.fit(train, iterations=180, seed=19, validation_samples=heldout)
    after = (codec.decode(codec.encode(heldout)) - heldout).square().mean()
    assert after < before / 30
    assert after < .005
    assert not torch.equal(initial_colors, codec.encode(torch.tensor([-.8, 0., .8])))
    assert codec.fit_metadata["training_samples"] == train.numel()
    assert codec.fit_metadata["validation_samples"] == heldout.numel()
    assert codec.fit_history[-1]["validation_loss"] < codec.fit_history[0]["validation_loss"]
    encoded = codec.encode(heldout.reshape(32, 32))
    assert encoded.dtype == torch.uint8
    assert encoded.shape == (32, 32, 3)
    assert torch.equal(codec.colors(encoded), encoded)
    reconstructed = codec_from_config(json.loads(json.dumps(codec.to_config())))
    assert torch.equal(reconstructed.encode(heldout), codec.encode(heldout))
    assert torch.equal(reconstructed.decode(encoded), codec.decode(encoded))
    assert torch.equal(ChromaTensor.from_tensor(heldout, codec).decode(), codec.decode(codec.encode(heldout)))
    for value in [float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            codec.encode(torch.tensor([value]))


def test_neural_palette_decoder_training_and_gradients_are_real():
    torch.set_num_threads(4)
    book = LearnedCodec(size=64)
    ids = torch.arange(64).reshape(-1, 1)
    rgb, values = book.colors(ids), book.decode(ids).detach()
    decoder = LearnedNeuralDecoder(seed=3)
    before = (decoder(rgb) - values).square().mean()
    decoder.fit(rgb, values, steps=120, seed=3)
    after = (decoder(rgb) - values).square().mean()
    assert after < before / 4
    assert decoder.fit_history[-1]["training_loss"] < decoder.fit_history[0]["training_loss"]
    decoder.zero_grad()
    decoder(rgb).sum().backward()
    assert decoder.network[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        decoder.fit(rgb, values[:-1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_neural_rgb_config_preserves_parameter_dtype_when_default_changes(dtype):
    codec = LearnedRGBCodec(seed=7).to(dtype=dtype)
    values = torch.tensor([-.8, 0., .8], dtype=dtype)
    codes = codec.encode(values)
    config = json.loads(json.dumps(codec.to_config()))
    saved_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64 if dtype == torch.float32 else torch.float32)
        restored = codec_from_config(config)
        for name, parameter in codec.state_dict().items():
            assert restored.state_dict()[name].dtype == dtype
            assert torch.equal(restored.state_dict()[name], parameter)
        assert torch.equal(restored.encode(values), codes)
        assert torch.equal(restored.decode(codes), codec.decode(codes))
    finally:
        torch.set_default_dtype(saved_default)


def test_legacy_neural_rgb_config_loads_fp32_and_rejects_bad_dtype_metadata():
    codec = LearnedRGBCodec(seed=3).float()
    legacy = codec.to_config()
    del legacy["state_dtypes"]
    saved_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        restored = codec_from_config(legacy)
        assert all(parameter.dtype == torch.float32 for parameter in restored.parameters())
        assert all(torch.equal(value, restored.state_dict()[name]) for name, value in codec.state_dict().items())
    finally:
        torch.set_default_dtype(saved_default)
    invalid = codec.to_config()
    invalid["state_dtypes"][next(iter(invalid["state_dtypes"]))] = "torch.int64"
    with pytest.raises(ValueError, match="Unsupported neural RGB parameter dtype"):
        codec_from_config(invalid)
    invalid = codec.to_config()
    invalid["state_dtypes"] = {}
    with pytest.raises(ValueError, match="dtype metadata"):
        codec_from_config(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_neural_codec_shape_device_and_cuda_fit():
    codec = LearnedRGBCodec(seed=11).cuda()
    x = torch.linspace(-1, 1, 256, device="cuda")
    codec.fit(x, iterations=30, seed=11)
    codes = codec.encode(x)
    assert codes.device.type == "cuda"
    assert codec.decode(codes).device.type == "cuda"
    assert codec.encode(x.cpu()).device.type == "cpu"
    assert torch.equal(codec.encode(x.cpu()), codes.cpu())

