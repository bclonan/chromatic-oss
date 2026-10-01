import hashlib
import json
import struct

import pytest
import torch

from chromatensor import ChromaTensor, RGBCodec, RGBDCodec, HueCodec, LearnedCodec, LearnedRGBCodec, HierarchicalCodec, save_ctensor, load_ctensor, save_chroma_checkpoint, load_chroma_checkpoint


@pytest.mark.parametrize("codec", [RGBCodec(), RGBDCodec(), HueCodec(), HierarchicalCodec()])
@pytest.mark.parametrize("compress", [False, True])
def test_file_roundtrip_with_dtype_shape_and_payload(codec, compress, tmp_path):
    x = torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(2, 3, 4)
    cx = ChromaTensor.from_tensor(x, codec)
    path = tmp_path / "weights.ctensor"
    reported = save_ctensor(cx, path, compress=compress)
    recovered = load_ctensor(path)
    assert recovered.shape == cx.shape
    assert recovered.decoded_dtype == torch.float64
    assert torch.equal(recovered.codes, cx.codes)
    assert torch.equal(recovered.decode(), cx.decode())
    assert reported == path.stat().st_size


def test_learned_palette_and_values_survive_safe_load(tmp_path):
    samples = torch.randn(256).clamp(-1, 1)
    codec = LearnedCodec(size=16).fit(samples, iterations=3)
    cx = ChromaTensor.from_tensor(samples, codec)
    path = tmp_path / "learned.ctensor"
    save_ctensor(cx, path)
    recovered = load_ctensor(path)
    assert torch.equal(recovered.rgb, cx.rgb)
    assert torch.equal(recovered.decode(), cx.decode())


@pytest.mark.parametrize("kind", ["centroid", "neural"])
@pytest.mark.parametrize("compress", [False, True])
def test_fp64_learned_codec_parameters_and_decoding_survive_file_load(kind, compress, tmp_path):
    if kind == "centroid":
        codec = LearnedCodec(4)
        codec.codebook.double()
        with torch.no_grad():
            codec.codebook.values.copy_(torch.tensor([-1., -.333333333333, .123456789012, 1.], dtype=torch.float64))
        parameters = codec.codebook
    else:
        codec = LearnedRGBCodec(seed=7).double()
        parameters = codec
    numerical = torch.tensor([-.333333333333, .123456789012], dtype=torch.float64)
    tensor = ChromaTensor.from_tensor(numerical, codec)
    path = tmp_path / "fp64.ctensor"
    save_ctensor(tensor, path, compress=compress)
    saved_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        recovered = load_ctensor(path)
    finally:
        torch.set_default_dtype(saved_default)
    restored_parameters = recovered.codec.codebook if kind == "centroid" else recovered.codec
    for name, parameter in parameters.state_dict().items():
        restored = restored_parameters.state_dict()[name]
        assert restored.dtype == parameter.dtype
        assert torch.equal(restored, parameter)
    assert recovered.decoded_dtype == torch.float64
    assert torch.equal(recovered.codes, tensor.codes)
    assert torch.equal(recovered.decode(), tensor.decode())


def test_corruption_and_checksums_are_rejected(tmp_path):
    path = tmp_path / "test.ctensor"
    save_ctensor(ChromaTensor.from_tensor(torch.arange(8.0)), path)
    data = bytearray(path.read_bytes())
    data[20] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        load_ctensor(path)


def test_checksummed_wrong_shape_is_rejected(tmp_path):
    path = tmp_path / "test.ctensor"
    save_ctensor(ChromaTensor.from_tensor(torch.arange(8.0)), path, compress=False)
    data = path.read_bytes()
    length = struct.unpack("<I", data[8:12])[0]
    metadata = json.loads(data[12:12 + length])
    metadata["shape"] = [4, 2]
    header = json.dumps(metadata).encode()
    body = data[:8] + struct.pack("<I", len(header)) + header + data[12 + length:-32]
    path.write_bytes(body + hashlib.sha256(body).digest())
    with pytest.raises(ValueError, match="shape"):
        load_ctensor(path)


def test_empty_scalar_and_checkpoint_roundtrip(tmp_path):
    for value in (torch.empty(0, 3), torch.tensor(1.25)):
        path = tmp_path / "value.ctensor"
        save_ctensor(ChromaTensor.from_tensor(value), path)
        assert torch.equal(load_ctensor(path).decode(), value)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    path = tmp_path / "model.ctcheckpoint"
    x = torch.randn(16, 4)
    expected = model(x)
    save_chroma_checkpoint(model, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    load_chroma_checkpoint(path, model)
    assert torch.equal(expected, model(x))

