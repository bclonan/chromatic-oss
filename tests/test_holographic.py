import hashlib
import json
import math
import struct
import zlib

import pytest
import torch

import chromatensor.holographic as holographic
from chromatensor.holographic import (
    HolographicCube, decode_numeric_polar, encode_numeric_polar,
    load_hcube, numeric_polar_reference, save_hcube,
)


def _seeded(shape, seed=824):
    return torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("shape", [(13,), (5, 7), (3, 4, 5), (2, 2, 3, 3, 2, 2)])
def test_nd_global_transform_uses_orthonormal_fft(shape, monkeypatch):
    values = _seeded(shape)
    observed = []
    original_encoder = holographic.encode_numeric_polar

    def recorder(spectrum):
        observed.append(spectrum.clone())
        return original_encoder(spectrum)

    monkeypatch.setattr(holographic, "encode_numeric_polar", recorder)
    cube = HolographicCube.from_tensor(values)
    expected = torch.fft.fftn(values, norm="ortho")
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(expected.abs().square().sum(), values.square().sum())
    assert cube.shape == values.shape
    assert cube.ndim == len(shape)
    assert cube.codes.shape == (*shape, 3)
    assert cube.dtype == torch.uint8
    assert cube.storage_nbytes == values.numel() * 3
    assert cube.numel() == values.numel()
    assert cube.decode().dtype == torch.float64
    assert cube.coefficients().dtype == torch.complex128
    relative_error = torch.linalg.vector_norm(cube.decode() - values) / torch.linalg.vector_norm(values)
    assert relative_error < 0.014
    assert set(cube.__dict__) == {"codes", "amplitude_scale", "axis_names"}


@pytest.mark.parametrize("value", [0.0, 1.25, -3.5])
@pytest.mark.parametrize("shape", [(), (7,), (4, 6), (2, 3, 2, 2)])
def test_zero_constant_and_scalar_reconstruction(value, shape):
    values = torch.full(shape, value, dtype=torch.float64)
    cube = HolographicCube.from_tensor(values)
    torch.testing.assert_close(cube.decode(), values, rtol=1e-14, atol=1e-14)
    if value == 0:
        assert cube.amplitude_scale == 0
        assert not torch.any(cube.codes)
        assert torch.equal(cube.decode(), values)
        assert not torch.any(cube.rgb)


def test_local_impulse_changes_every_frequency_coefficient():
    values = torch.zeros(4, 5, 6, dtype=torch.float64)
    values[1, 2, 3] = 2
    cube = HolographicCube.from_tensor(values)
    assert torch.all(cube.magnitudes > 0)
    expected_magnitude = 2 / math.sqrt(values.numel())
    torch.testing.assert_close(cube.magnitudes, torch.full_like(values, expected_magnitude), rtol=1e-14, atol=1e-14)
    assert torch.unique(cube.codes[..., 0]).numel() > 8
    assert (cube.decode() - values).square().mean().sqrt() < 0.002


def test_circular_phase_wrap_and_exact_cartesian_axis_codes():
    coefficients = torch.tensor([0, 1, 1j, -1, -1j], dtype=torch.complex128)
    codes, scale = encode_numeric_polar(coefficients)
    assert codes[..., 0].tolist() == [0, 0, 64, 128, 192]
    assert codes[..., 1:].tolist() == [[0, 0]] + [[255, 255]] * 4
    torch.testing.assert_close(decode_numeric_polar(codes, scale), coefficients, rtol=0, atol=1e-15)
    tiny_negative_phase = torch.polar(torch.ones(1, dtype=torch.float64), torch.tensor([-0.001], dtype=torch.float64))
    wrapped, _ = encode_numeric_polar(tiny_negative_phase)
    assert wrapped[0, 0].item() == 0


def test_rgb_is_phase_and_normalized_amplitude_preview():
    coefficients = torch.tensor([0, 1, 1j, -1, -1j, 0.5], dtype=torch.complex128)
    codes, scale = encode_numeric_polar(coefficients)
    cube = HolographicCube(codes, scale, ["frequency"])
    assert cube.rgb.dtype == torch.uint8
    assert cube.rgb.shape == (6, 3)
    assert cube.rgb[0].tolist() == [0, 0, 0]
    assert cube.rgb[1].tolist() == [255, 0, 0]
    assert cube.rgb[-1].tolist() == [128, 0, 0]
    assert cube.phase[3].item() == math.pi
    assert cube.axis_names == ("frequency",)


def test_equal_bit_numeric_control_has_identical_reconstruction():
    values = _seeded((5, 7, 3))
    cube = HolographicCube.from_tensor(values)
    codes, scale = encode_numeric_polar(torch.fft.fftn(values, norm="ortho"))
    assert torch.equal(cube.codes, codes)
    assert cube.amplitude_scale == scale
    assert torch.equal(cube.coefficients(), decode_numeric_polar(codes, scale))
    assert torch.equal(cube.decode(), numeric_polar_reference(values))
    keep = cube.erasure_mask(0.375, seed=49)
    assert torch.equal(cube.decode(keep), numeric_polar_reference(values, mask=keep))


def test_mask_reconstruction_and_noise_information_loss():
    values = _seeded((12, 12, 4), seed=730)
    cube = HolographicCube.from_tensor(values)
    keep = cube.erasure_mask(0.5, seed=103)
    erased = cube.mask(keep)
    assert torch.equal(erased.decode(), cube.decode(mask=keep))
    assert torch.equal(erased.codes[keep], cube.codes[keep])
    assert not torch.any(erased.codes[~keep])
    assert torch.equal(cube.erase(0.5, seed=103).codes, erased.codes)
    assert erased.axis_names == cube.axis_names
    assert erased.amplitude_scale == cube.amplitude_scale
    assert not torch.equal(erased.codes, cube.codes)
    full_error = (cube.decode() - values).square().mean().sqrt()
    erased_error = (erased.decode() - values).square().mean().sqrt()
    assert erased_error > full_error * 20
    assert erased_error > 0.3
    assert torch.equal(cube.erase(0).decode(), cube.decode())
    assert torch.equal(cube.erase(1).decode(), torch.zeros_like(values))


def test_erasure_is_seeded_exact_count_and_preserves_global_random_generator():
    cube = HolographicCube.from_tensor(_seeded((3, 3, 3, 3)))
    state = torch.random.get_rng_state().clone()
    mask = cube.erasure_mask(0.5, seed=47)
    assert mask.dtype == torch.bool
    assert mask.shape == cube.shape
    assert mask.sum().item() == 40
    assert torch.equal(mask, cube.erasure_mask(0.5, seed=47))
    assert not torch.equal(mask, cube.erasure_mask(0.5, seed=48))
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("bad_mask", [torch.ones(3, 4), torch.ones(3, 1, dtype=torch.bool), [True] * 12])
def test_masks_require_boolean_tensor_and_exact_dimensions(bad_mask):
    cube = HolographicCube.from_tensor(_seeded((3, 4)))
    with pytest.raises((TypeError, ValueError), match="mask"):
        cube.decode(bad_mask)
    with pytest.raises((TypeError, ValueError), match="mask"):
        cube.mask(bad_mask)


@pytest.mark.parametrize("bad_fraction", [-0.01, 1.01, float("nan"), float("inf"), True, "0.2"])
def test_invalid_erasure_fraction_is_rejected(bad_fraction):
    cube = HolographicCube.from_tensor(torch.ones(4))
    with pytest.raises(ValueError, match="fraction"):
        cube.erasure_mask(bad_fraction)


@pytest.mark.parametrize("bad_seed", [-1, 2**63, 0.2, True])
def test_invalid_erasure_seed_is_rejected(bad_seed):
    cube = HolographicCube.from_tensor(torch.ones(4))
    with pytest.raises(ValueError, match="seed"):
        cube.erasure_mask(0.5, seed=bad_seed)


@pytest.mark.parametrize("values", [torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                    torch.tensor([-float("inf")]), torch.empty(0, 2),
                                    torch.tensor([1j])])
def test_invalid_input_is_rejected_without_silent_replacement(values):
    with pytest.raises((TypeError, ValueError)):
        HolographicCube.from_tensor(values)


def test_overflow_and_bad_code_metadata_are_rejected():
    with pytest.raises(ValueError, match="range"):
        HolographicCube.from_tensor(torch.full((4, 4), torch.finfo(torch.float64).max, dtype=torch.float64))
    with pytest.raises(TypeError, match="uint8"):
        HolographicCube(torch.zeros(4, 3), 1)
    with pytest.raises(ValueError, match="three bytes"):
        HolographicCube(torch.zeros(4, 2, dtype=torch.uint8), 1)
    for scale in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            HolographicCube(torch.zeros(4, 3, dtype=torch.uint8), scale)
    with pytest.raises(ValueError, match="zero magnitude"):
        HolographicCube(torch.ones(4, 3, dtype=torch.uint8), 0)


@pytest.mark.parametrize("names", [[], ["x"], ["x", "x"], ["x", ""], "xy", ["x", 1]])
def test_axis_names_match_dimensions_and_are_unique(names):
    with pytest.raises(ValueError, match="[Aa]xis|axis"):
        HolographicCube.from_tensor(torch.ones(3, 4), axis_names=names)


def test_copy_and_dtype_interoperation():
    values = _seeded((4, 4)).to(torch.float32).requires_grad_()
    cube = HolographicCube.from_tensor(values, ["height", "width"])
    assert not cube.codes.requires_grad
    assert cube.decode(dtype=torch.float32).dtype == torch.float32
    assert cube.decode(dtype=torch.float16).dtype == torch.float16
    assert cube.to("cpu").axis_names == ("height", "width")
    clone = cube.clone()
    clone.codes.fill_(0)
    assert torch.any(cube.codes)
    assert "payload_bytes=48" in repr(cube)
    with pytest.raises(TypeError, match="real floating"):
        cube.decode(dtype=torch.int64)
    with pytest.raises(TypeError, match="integer storage"):
        cube.to(torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("shape", [(4, 5, 6), (2, 2, 2, 2, 2, 3)])
def test_cuda_float64_transform_codes_and_mask_match_cpu(shape):
    values = _seeded(shape, seed=934)
    cpu_cube = HolographicCube.from_tensor(values)
    cuda_cube = HolographicCube.from_tensor(values.cuda())
    assert cuda_cube.device.type == "cuda"
    assert torch.equal(cuda_cube.codes.cpu(), cpu_cube.codes)
    assert cuda_cube.amplitude_scale == pytest.approx(cpu_cube.amplitude_scale, rel=1e-14)
    torch.testing.assert_close(cuda_cube.decode().cpu(), cpu_cube.decode(), rtol=1e-12, atol=1e-12)
    assert torch.equal(cuda_cube.erasure_mask(0.4, seed=71).cpu(), cpu_cube.erasure_mask(0.4, seed=71))
    mask = cpu_cube.erasure_mask(0.4, seed=71)
    torch.testing.assert_close(cuda_cube.decode(mask).cpu(), cpu_cube.decode(mask), rtol=1e-12, atol=1e-12)
    assert torch.equal(cpu_cube.cuda().cpu().codes, cpu_cube.codes)


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("shape", [(), (3, 4, 5), (2, 2, 2, 3)])
def test_safe_file_roundtrip_preserves_names_codes_and_scale(shape, compress, tmp_path):
    cube = HolographicCube.from_tensor(_seeded(shape), [f"coordinate_{i}" for i in range(len(shape))])
    path = tmp_path / "nested" / "cube.hcube"
    reported = cube.save(path, compress=compress)
    loaded = HolographicCube.load(path)
    assert reported == path.stat().st_size
    assert loaded.storage_nbytes == 3 * cube.numel()
    assert loaded.shape == cube.shape
    assert loaded.axis_names == cube.axis_names
    assert loaded.amplitude_scale == cube.amplitude_scale
    assert torch.equal(loaded.codes, cube.codes)
    assert torch.equal(loaded.decode(), cube.decode())


def _rewrite_file(path, change_metadata=lambda metadata: None, payload=None):
    data = path.read_bytes()
    size = struct.unpack("<I", data[8:12])[0]
    metadata = json.loads(data[12:12 + size])
    change_metadata(metadata)
    header = json.dumps(metadata).encode()
    data_payload = data[12 + size:-32] if payload is None else payload
    body = data[:8] + struct.pack("<I", len(header)) + header + data_payload
    path.write_bytes(body + hashlib.sha256(body).digest())


def test_file_checksum_rejects_tampering(tmp_path):
    path = tmp_path / "cube.hcube"
    save_hcube(HolographicCube.from_tensor(_seeded((4, 4))), path)
    raw = bytearray(path.read_bytes())
    raw[-40] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        load_hcube(path)


@pytest.mark.parametrize("field,value", [("version", 2), ("version", True), ("phase_bins", 128),
                                         ("magnitude_levels", 255), ("shape", [2**40]),
                                         ("shape", [0]), ("axis_names", ["x"]),
                                         ("amplitude_scale", float("nan")),
                                         ("amplitude_scale", -1), ("code_dtype", "float32"),
                                         ("norm", "backward"), ("execution_dtype", "float32")])
def test_checksummed_unsupported_metadata_is_rejected(field, value, tmp_path):
    path = tmp_path / "cube.hcube"
    cube = HolographicCube.from_tensor(_seeded((4, 4)))
    cube.save(path)
    _rewrite_file(path, lambda metadata: metadata.update({field: value}))
    with pytest.raises((TypeError, ValueError)):
        load_hcube(path)


def test_expanded_compressed_payload_is_bounded(tmp_path):
    path = tmp_path / "cube.hcube"
    HolographicCube.from_tensor(torch.ones(4, 4)).save(path)
    payload = zlib.compress(bytes(65536))
    _rewrite_file(path, lambda metadata: metadata.update({"payload_bytes": len(payload)}), payload)
    with pytest.raises(ValueError, match="compressed payload"):
        load_hcube(path)


def test_zero_metadata_cannot_hide_nonzero_magnitude_codes(tmp_path):
    path = tmp_path / "cube.hcube"
    HolographicCube.from_tensor(torch.ones(4, 4)).save(path)
    _rewrite_file(path, lambda metadata: metadata.update({"amplitude_scale": 0}))
    with pytest.raises(ValueError, match="zero magnitude"):
        load_hcube(path)
