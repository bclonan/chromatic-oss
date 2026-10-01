import math
import json

import pytest
import torch

from chromatensor.codecs import (RGBCodec, RGBDCodec, UniformCodec, HSVCodec, HueCodec,
                                LearnedCodec, LearnedRGBDCodec, HierarchicalCodec,
                                LearnedChromaCodebook, LearnedChromaEncoder,
                                LearnedChromaDecoder, codec_from_config)
from chromatensor.tokenizer import ChromaVocabulary, ChromaTokenizer, ChromaDecoder, TokenSequence


@pytest.mark.parametrize("codec", [RGBCodec(), RGBDCodec(), UniformCodec(), HSVCodec(),
                                    HueCodec(), LearnedCodec(16), LearnedRGBDCodec(), HierarchicalCodec()])
def test_codec_contract_shape_config_and_invalid_floats(codec):
    x = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
    codes = codec.encode(x)
    assert codes.shape == (*x.shape, codec.channels)
    assert not codes.is_floating_point()
    assert codec.decode(codes).shape == x.shape
    assert codec.colors(codes).shape == (*x.shape, 3)
    assert codec.colors(codes).dtype == torch.uint8
    restored = codec_from_config(codec.to_config())
    assert torch.equal(restored.encode(x), codes)
    assert torch.equal(restored.decode(codes), codec.decode(codes))
    for value in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="NaN|infinity"):
            codec.encode(torch.tensor([value]))
    with pytest.raises(TypeError):
        codec.decode(codes.float())


def test_rgb24_roundtrip_arbitrary_range_and_equal_bit_baseline():
    generator = torch.Generator().manual_seed(37)
    x = torch.rand(10001, generator=generator, dtype=torch.float64) * 84.2 - 41
    codec = RGBCodec(-41, 43.2)
    reconstructed = codec.decode(codec.encode(x))
    assert (x - reconstructed).abs().max() <= 84.2 / (2 * (2 ** 24 - 1)) + 1e-14
    uniform = UniformCodec(24, -41, 43.2)
    assert torch.equal(reconstructed, uniform.decode(uniform.encode(x)))
    assert torch.equal(codec.decode(codec.encode(torch.tensor([-100.0, 100.0]))), torch.tensor([-41, 43.2], dtype=torch.float64))


def test_log_domain_relative_error_and_validation():
    codec = RGBCodec(1e-12, 1e12, logarithmic=True)
    x = torch.logspace(-12, 12, 1000, dtype=torch.float64)
    relative_error = ((codec.decode(codec.encode(x)) - x) / x).abs()
    assert relative_error.max() < 2e-6
    with pytest.raises(ValueError):
        RGBCodec(-1, 1, logarithmic=True)
    with pytest.raises(ValueError):
        codec.encode(torch.tensor([0.0]))
    for minimum, maximum in [(1, 1), (2, 1), (float("nan"), 1), (-1e308, 1e308)]:
        with pytest.raises(ValueError):
            RGBCodec(minimum, maximum)


def test_rgbd_exact_fp32_sign_subnormal_and_dynamic_range():
    bitpatterns = torch.tensor([0, -2147483648, 1, -2147483647, 0x007FFFFF,
                                0x00800000, 0x3F800000, -1082130432, 0x7F7FFFFF], dtype=torch.int32)
    x = bitpatterns.view(torch.float32)
    codec = RGBDCodec()
    codes = codec.encode(x)
    assert torch.equal(codec.decode(codes).view(torch.int32), bitpatterns)
    assert codes[6, 3] == 127
    assert codes[0, 3] == 0
    with pytest.raises(ValueError, match="overflow"):
        codec.encode(torch.tensor([1e39], dtype=torch.float64))
    with pytest.raises(ValueError, match="underflows"):
        codec.encode(torch.tensor([1e-100], dtype=torch.float64))
    malformed = codes.clone()
    malformed[0, 3] = 255
    with pytest.raises(ValueError, match="reserved"):
        codec.decode(malformed)


@pytest.mark.parametrize("bits,dtype,actual_bytes", [(4, torch.uint8, 1), (8, torch.uint8, 1),
                                                    (16, torch.uint16, 2), (24, torch.int32, 4)])
def test_uniform_storage_and_quantization_bound(bits, dtype, actual_bytes):
    codec = UniformCodec(bits, -10, 10)
    x = torch.linspace(-10, 10, 10003, dtype=torch.float64)
    codes = codec.encode(x)
    assert codes.dtype == dtype
    assert codes.element_size() == actual_bytes
    assert (codec.decode(codes) - x).abs().max() <= 20 / (2 * (2 ** bits - 1)) + 1e-14
    assert codec.bits_per_value == bits


def test_periodic_hue_sine_phase_rotation_and_sequences():
    codec = HueCodec()
    phases = torch.linspace(-math.pi * 8, math.pi * 8, 3001, dtype=torch.float64)
    reconstructed = codec.decode(codec.encode(phases))
    assert codec.circular_error(phases, reconstructed).max() <= math.pi / 256 + 1e-14
    assert (phases.sin() - reconstructed.sin()).abs().max() <= math.pi / 256 + 1e-14
    assert torch.equal(codec.encode(phases), codec.encode(phases + 2 * math.pi))
    quarter_turn = 2 * math.pi / 4
    shifted = codec.encode(phases + quarter_turn)[..., 0].long()
    assert torch.equal(shifted, (codec.encode(phases)[..., 0].long() + 64) % 256)
    assert torch.equal(codec.colors(codec.encode(torch.tensor([0.0]))), torch.tensor([[255, 0, 0]], dtype=torch.uint8))
    assert codec.circular_error(torch.tensor([0.001]), torch.tensor([2 * math.pi - 0.001])) < .003


def test_learned_fit_improves_skewed_values_and_seed_reproducibility():
    generator = torch.Generator().manual_seed(43)
    x = (torch.randn(10000, generator=generator) * .08 + .3).clamp(-1, 1)
    codec = LearnedCodec(size=16)
    before = (x - codec.decode(codec.encode(x))).square().mean()
    codec.fit(x, seed=7)
    after = (x - codec.decode(codec.encode(x))).square().mean()
    assert after < before / 4
    other = LearnedCodec(size=16).fit(x, seed=7)
    assert torch.equal(codec.codebook.values, other.codebook.values)
    assert codec.codebook.fit_history[-1] <= codec.codebook.fit_history[0]


@pytest.mark.parametrize("minimum,maximum", [(1e-50, 2e-50), (1., 1. + 1e-8), (1e40, 2e40)])
def test_learned_codebook_rejects_unrepresentable_default_domains(minimum, maximum):
    with pytest.raises(ValueError, match="Centroid domain"):
        LearnedChromaCodebook(16, minimum, maximum, dtype=torch.float32)


def test_fp64_codebook_domain_and_metadata_preserve_centroids_and_tokens():
    book = LearnedChromaCodebook(16, minimum=1e-50, maximum=2e-50, dtype=torch.float64)
    values = torch.linspace(1e-50, 2e-50, 16, dtype=torch.float64)
    assert torch.equal(book.decode(book.encode(values)), values)
    vocabulary = ChromaVocabulary(codebook=book)
    sequence = ChromaTokenizer(vocabulary).encode(values)
    saved_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        restored = LearnedChromaCodebook.from_config(json.loads(json.dumps(book.to_config())))
        assert restored.values.dtype == torch.float64
        assert torch.equal(restored.values, book.values)
        assert torch.equal(ChromaDecoder().decode(sequence), values)
    finally:
        torch.set_default_dtype(saved_default)


def test_learned_config_with_legacy_dtype_is_independent_of_default_dtype():
    book = LearnedChromaCodebook(16, dtype=torch.float32)
    config = book.to_config()
    del config["dtype"]
    saved_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        restored = LearnedChromaCodebook.from_config(config)
        assert restored.values.dtype == torch.float32
        assert torch.equal(restored.values, book.values)
    finally:
        torch.set_default_dtype(saved_default)


def test_learned_encoder_decoder_centroid_gradients():
    codebook = LearnedChromaCodebook(8)
    encoder, decoder = LearnedChromaEncoder(codebook), LearnedChromaDecoder(codebook)
    ids = encoder(torch.tensor([-.8, .1, .9]))
    reconstructed = decoder(ids)
    reconstructed.sum().backward()
    assert codebook.values.grad is not None
    assert codebook.values.grad.sum() == 3
    assert ids.dtype == torch.int64


def test_learned_rgbd_residual_improves_coarse_codebook():
    x = torch.linspace(-1, 1, 3001)
    codec = LearnedRGBDCodec(size=16, modifier_size=16).fit(x, seed=3)
    coarse = codec.codebook.decode(codec.codebook.encode(x))
    decoded = codec.decode(codec.encode(x))
    assert (decoded - x).square().mean() < (coarse - x).square().mean() / 20
    codes = codec.encode(x)
    ids = codec.codebook.ids_from_colors(codes[..., :3])
    assert torch.equal(codec.codebook.colors(ids), codes[..., :3])
    assert (codes[..., 3] < 16).all()


def test_hierarchy_precision_depth_and_compact_tokenization():
    x = torch.cat((torch.linspace(-1, 1, 16), torch.tensor([.123456, -.987654])))
    coarse = HierarchicalCodec(levels=1, vocabulary_size=16)
    fine = HierarchicalCodec(levels=3, vocabulary_size=16)
    error1 = (coarse.decode(coarse.encode(x)) - x).abs()
    error3 = (fine.decode(fine.encode(x)) - x).abs()
    assert error3.max() < error1.max() / 100
    adaptive = HierarchicalCodec(levels=3, vocabulary_size=16, tolerance=.001)
    codes = adaptive.encode(x)
    assert (codes[:16, -1] == 1).all()
    assert (codes[-2:, -1] > 1).all()
    assert (adaptive.decode(codes) - x).abs().max() <= .001
    tokenizer = ChromaTokenizer(vocabulary_size=16, levels=3, tolerance=.001)
    sequence = tokenizer.encode(x)
    assert sequence.tokens_per_value < 2
    assert sequence.storage_nbytes < codes.numel() * codes.element_size()
    assert torch.allclose(tokenizer.decode(sequence), adaptive.decode(codes).float())


def test_hierarchy_rejects_binary_multistage_precision_but_keeps_one_stage():
    with pytest.raises(ValueError, match="at least three"):
        HierarchicalCodec(levels=2, vocabulary_size=2)
    with pytest.raises(ValueError, match="at least three"):
        ChromaTokenizer(levels=32, vocabulary_size=2)
    codec = HierarchicalCodec(levels=1, vocabulary_size=2)
    endpoints = torch.tensor([-1., 1.])
    assert torch.equal(codec.decode(codec.encode(endpoints)).float(), endpoints)


@pytest.mark.parametrize("codec_class", [RGBCodec, UniformCodec])
@pytest.mark.parametrize("minimum", [1e-20, 1e20, 1e200])
def test_logarithmic_codec_rejects_indistinguishable_fp64_log_endpoints(codec_class, minimum):
    with pytest.raises(ValueError, match="distinct FP64 logarithms"):
        codec_class(minimum=minimum, maximum=math.nextafter(minimum, math.inf), logarithmic=True)


@pytest.mark.parametrize("size", [16, 64, 256, 1024, 4096, 65536])
def test_vocabulary_size_symbols_and_tokenizer_roundtrip(size):
    vocabulary = ChromaVocabulary(size)
    x = torch.linspace(-1, 1, 2001).reshape(1, 2001)
    sequence = ChromaTokenizer(vocabulary).encode(x)
    reconstructed = ChromaDecoder().decode(sequence)
    assert reconstructed.shape == x.shape
    assert (reconstructed - x).abs().max() <= 1 / (size - 1) + 2e-7
    assert vocabulary.token_id(vocabulary.symbol(size - 1)) == size - 1
    assert vocabulary.palette.shape == (size, 3)
    assert vocabulary.palette.unique(dim=0).shape[0] == size


def test_token_sequence_rejects_malformed_shapes_and_vocab_metadata():
    vocabulary = ChromaVocabulary(16)
    with pytest.raises(ValueError):
        TokenSequence(torch.tensor([1, 2]), (3,), vocabulary.codec.to_config())
    with pytest.raises(ValueError):
        vocabulary.token_id("COLOR_FFFFFE")
    with pytest.raises(ValueError):
        ChromaDecoder().decode(torch.tensor([1, 2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("codec", [RGBCodec(), RGBDCodec(), UniformCodec(16), HueCodec(),
                                    LearnedCodec(256), HierarchicalCodec()])
def test_cpu_cuda_codes_are_identical(codec):
    x = torch.linspace(-.95, .95, 1001, dtype=torch.float64)
    cpu = codec.encode(x)
    gpu = codec.encode(x.cuda())
    assert torch.equal(cpu, gpu.cpu())
    assert torch.allclose(codec.decode(cpu), codec.decode(gpu).cpu(), atol=1e-12, rtol=0)

