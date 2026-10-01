"""Independent RGBD byte checks and decoding after the input process exits."""
import hashlib
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from chromatensor.codecs import RGBDCodec
from chromatensor.tokenizer import ChromaDecoder, ChromaTokenizer, ChromaVocabulary, TokenSequence


KNOWN = [
    (0x00000000, [0, 0, 0, 0]),
    (0x80000000, [128, 0, 0, 0]),
    (0x3F800000, [0, 0, 0, 127]),
    (0xBF800000, [128, 0, 0, 127]),
    (0x3FC00000, [64, 0, 0, 127]),
    (0x40000000, [0, 0, 0, 128]),
    (0x7F7FFFFF, [127, 255, 255, 254]),
    (0x00000001, [0, 0, 1, 0]),
    (0x80000001, [128, 0, 1, 0]),
]


def fp32_from_bits(bits):
    return torch.from_numpy(np.asarray(bits, dtype=np.uint32).view(np.float32).copy())


def bit_array(values):
    return values.detach().cpu().contiguous().numpy().view(np.uint32)


def independent_rgbd(bits):
    """Use NumPy integers, not any ChromaTensor encoder or decoder."""
    bits = np.asarray(bits, dtype=np.uint32)
    return np.stack((((bits >> 24) & 128) | ((bits >> 16) & 127),
                     (bits >> 8) & 255, bits & 255, (bits >> 23) & 255), axis=-1).astype(np.uint8)


@pytest.mark.parametrize("bits,expected", KNOWN)
def test_default_token_is_the_actual_rgbd_tuple(bits, expected):
    values = fp32_from_bits([bits])
    sequence = ChromaTokenizer().encode(values)
    assert sequence.tokens.dtype == torch.uint8
    assert sequence.tokens.shape == (1, 4)
    assert sequence.tokens.tolist() == [expected]
    assert sequence.rgbd.tolist() == [expected]
    assert sequence.rgb.tolist() == [expected[:3]]
    assert sequence.num_tokens == 1
    assert sequence.tokens_per_value == (1 if values.numel() else 0)
    assert sequence.storage_nbytes == 4
    assert sequence.lengths is None
    assert sequence.codec_config == RGBDCodec().to_config()
    assert bit_array(ChromaDecoder().decode(sequence)).tolist() == [bits]


def test_depth_is_required_when_rgb_bytes_are_identical():
    sequence = ChromaTokenizer().encode(torch.tensor([1., 2.]))
    assert torch.equal(sequence.tokens[0, :3], sequence.tokens[1, :3])
    assert sequence.tokens[:, 3].tolist() == [127, 128]
    assert ChromaDecoder().decode(sequence).tolist() == [1., 2.]


def test_raw_rgbd_bytes_decode_without_a_palette_or_original_floats():
    tokens = torch.tensor([item[1] for item in KNOWN], dtype=torch.uint8)
    assert bit_array(ChromaDecoder().decode(tokens)).tolist() == [item[0] for item in KNOWN]
    assert bit_array(ChromaTokenizer().decode(tokens)).tolist() == [item[0] for item in KNOWN]
    with pytest.raises(ValueError):
        ChromaDecoder().decode(tokens[:, :3])


def test_sequence_keeps_only_tokens_shape_codec_lengths_and_dtype():
    sequence = ChromaTokenizer().encode(torch.tensor([1., 2.]))
    assert set(vars(sequence)) == {"tokens", "shape", "codec_config", "lengths", "decoded_dtype"}
    moved = sequence.to("cpu")
    assert torch.equal(moved.tokens, sequence.tokens)
    assert moved.shape == sequence.shape
    assert moved.codec_config == sequence.codec_config
    sequence.tokens[0, 3] = 255
    with pytest.raises(ValueError):
        ChromaDecoder().decode(sequence)


@pytest.mark.parametrize("channel,expected_bits", [(0, 0x3F810000), (1, 0x3F800100),
                                                   (2, 0x3F800001), (3, 0x3F000000)])
def test_each_saved_channel_changes_the_decoded_value(channel, expected_bits):
    tokens = torch.tensor([[0, 0, 0, 127]], dtype=torch.uint8)
    tokens[0, channel] ^= 1
    sequence = TokenSequence(tokens, (1,), RGBDCodec().to_config())
    assert bit_array(ChromaDecoder().decode(sequence)).tolist() == [expected_bits]
    assert expected_bits != 0x3F800000


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (3,), (2, 3), (2, 3, 4)])
def test_default_rgbd_token_shape_and_empty_tensors(shape):
    values = torch.arange(int(np.prod(shape)) if shape else 1, dtype=torch.float32).reshape(shape)
    sequence = ChromaTokenizer().encode(values)
    assert sequence.tokens.shape == (values.numel(), 4)
    assert sequence.rgbd.shape == (*shape, 4)
    assert sequence.rgb.shape == (*shape, 3)
    assert sequence.num_tokens == values.numel()
    assert sequence.tokens_per_value == (1 if values.numel() else 0)
    decoded = ChromaDecoder().decode(sequence)
    assert decoded.shape == values.shape
    assert np.array_equal(bit_array(decoded), bit_array(values))


def test_noncontiguous_values_encode_in_logical_order():
    values = torch.arange(60, dtype=torch.float32).reshape(5, 12).t()[1::2]
    assert not values.is_contiguous()
    sequence = ChromaTokenizer().encode(values)
    assert np.array_equal(sequence.rgbd.numpy(), independent_rgbd(bit_array(values)))
    assert np.array_equal(bit_array(ChromaDecoder().decode(sequence)), bit_array(values))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_supported_narrow_floats_are_exact_and_keep_output_dtype(dtype):
    values = torch.tensor([-2., -0., 0., .125, 1., 128.], dtype=dtype)
    sequence = ChromaTokenizer().encode(values)
    decoded = ChromaDecoder().decode(sequence)
    assert decoded.dtype == dtype
    assert torch.equal(decoded.view(torch.int16 if dtype != torch.float32 else torch.int32),
                       values.view(torch.int16 if dtype != torch.float32 else torch.int32))


def test_fp64_rounds_to_fp32_and_is_not_mislabeled_lossless_fp64():
    values = torch.tensor([1. + 2. ** -40, -.123456789123456, 9.], dtype=torch.float64)
    sequence = ChromaTokenizer().encode(values)
    decoded = ChromaDecoder().decode(sequence)
    assert decoded.dtype == torch.float64
    assert torch.equal(decoded, values.float().double())
    assert not torch.equal(decoded, values)
    assert np.array_equal(sequence.tokens.numpy(), independent_rgbd(bit_array(values.float())))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e100, 1e-100])
def test_default_rgbd_rejects_nonfinite_overflow_and_underflow(value):
    with pytest.raises(ValueError):
        ChromaTokenizer().encode(torch.tensor([value], dtype=torch.float64))


@pytest.mark.parametrize("tokens,shape,lengths", [
    (torch.zeros((2, 3), dtype=torch.uint8), (2,), None),
    (torch.tensor([[0, 0, 0, 255]], dtype=torch.uint8), (1,), None),
    (torch.zeros((2, 4), dtype=torch.float32), (2,), None),
    (torch.zeros((2, 4), dtype=torch.int64), (2,), None),
    (torch.zeros((2, 4), dtype=torch.uint8), (3,), None),
    (torch.zeros((2, 4), dtype=torch.uint8), (-2,), None),
    (torch.zeros((1, 4), dtype=torch.uint8), (True,), None),
    (torch.zeros((2, 4), dtype=torch.uint8), (2,), torch.ones(2, dtype=torch.uint8)),
])
def test_rgbd_sequence_rejects_incompatible_tokens_shape_and_lengths(tokens, shape, lengths):
    with pytest.raises((TypeError, ValueError)):
        sequence = TokenSequence(tokens, shape, RGBDCodec().to_config(), lengths)
        ChromaDecoder().decode(sequence)


@pytest.mark.parametrize("config", [
    {"name": "absent"}, {"name": "rgbd"},
    {"name": "rgbd", "format": "arbitrary-rgb"},
    {"name": "rgbd", "format": "finite-ieee754-fp32-sign-fraction-exponent", "hidden_values": [1.]},
])
def test_unknown_rgbd_metadata_cannot_smuggle_original_floats(config):
    with pytest.raises((TypeError, ValueError)):
        sequence = TokenSequence(torch.tensor([[0, 0, 0, 127]], dtype=torch.uint8), (1,), config)
        ChromaDecoder().decode(sequence)


@pytest.mark.parametrize("kwargs", [{"vocabulary_size": 16}, {"minimum": -2., "maximum": 2.},
                                     {"levels": 2}, {"vocabulary": ChromaVocabulary(size=16)}])
def test_explicit_palette_configuration_preserves_old_integer_ids(kwargs):
    tokenizer = ChromaTokenizer(**kwargs)
    values = torch.tensor([-.5, 0., .5])
    sequence = tokenizer.encode(values)
    assert sequence.tokens.ndim == 1
    assert sequence.codec_config["name"] in {"learned", "hierarchical"}
    assert ChromaDecoder().decode(sequence).shape == values.shape


def checked_parts(path):
    data = path.read_bytes()
    header_size = struct.unpack("<I", data[8:12])[0]
    return data[:8], json.loads(data[12:12 + header_size]), data[12 + header_size:-32]


def write_checked(path, magic, header, payload):
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    body = magic + struct.pack("<I", len(encoded)) + encoded + payload
    path.write_bytes(body + hashlib.sha256(body).digest())


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (3, 3)])
def test_token_file_persists_actual_rgbd_bytes_and_only_decode_metadata(tmp_path, compressed, shape):
    known = fp32_from_bits([item[0] for item in KNOWN])
    values = known[0].reshape(()) if not shape else known[:int(np.prod(shape))].reshape(shape)
    sequence = ChromaTokenizer().encode(values)
    path = tmp_path / "tokens.ctensor"
    written = sequence.save(path, compress=compressed)
    assert written == path.stat().st_size
    magic, metadata, payload = checked_parts(path)
    assert magic == b"CTENSOR\x01"
    assert metadata["code_dtype"] == "uint8"
    assert metadata["code_shape"] == [*shape, 4]
    assert metadata["shape"] == list(shape)
    assert metadata["raw_bytes"] == values.numel() * 4
    assert metadata["codec"] == RGBDCodec().to_config()
    assert set(metadata) == {"version", "shape", "code_shape", "code_dtype", "tensor_dtype", "codec",
                             "compression", "raw_bytes", "payload_bytes"}
    if not compressed:
        assert payload == sequence.tokens.numpy().tobytes()
    loaded = TokenSequence.load(path, max_file_bytes=path.stat().st_size,
                                max_header_bytes=65536, max_payload_bytes=values.numel() * 4)
    assert torch.equal(loaded.tokens, sequence.tokens)
    assert np.array_equal(bit_array(ChromaDecoder().decode(loaded)), bit_array(values))


@pytest.mark.parametrize("budget", ["max_file_bytes", "max_header_bytes", "max_payload_bytes"])
def test_token_loader_checks_byte_budgets_before_numpy_allocation(tmp_path, budget, monkeypatch):
    path = tmp_path / "tokens.ctensor"
    ChromaTokenizer().encode(torch.arange(8.)).save(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("a rejected file allocated its payload")
    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        TokenSequence.load(path, **{budget: 1})


@pytest.mark.parametrize("kind", ["checksum", "truncated", "three_channels", "float_dtype", "d255",
                                 "extra_codec", "wrong_count"])
def test_token_loader_rejects_corrupt_or_malformed_rgbd_file(tmp_path, kind):
    path = tmp_path / "tokens.ctensor"
    ChromaTokenizer().encode(torch.tensor([1., 2.])).save(path)
    magic, metadata, payload = checked_parts(path)
    if kind == "checksum":
        data = bytearray(path.read_bytes())
        data[-33] ^= 1
        path.write_bytes(data)
    elif kind == "truncated":
        path.write_bytes(path.read_bytes()[:-5])
    else:
        if kind == "three_channels":
            metadata["code_shape"] = [2, 3]
            metadata["raw_bytes"] = metadata["payload_bytes"] = 6
            payload = payload[:6]
        elif kind == "float_dtype":
            metadata["code_dtype"] = "float32"
        elif kind == "d255":
            payload = payload[:3] + bytes([255]) + payload[4:]
        elif kind == "extra_codec":
            metadata["codec"]["hidden_values"] = [1., 2.]
        else:
            metadata["shape"] = [3]
        write_checked(path, magic, metadata, payload)
    with pytest.raises((TypeError, ValueError)):
        TokenSequence.load(path)


def test_cold_process_decodes_only_the_saved_rgbd_file(tmp_path):
    path = tmp_path / "tokens.ctensor"
    values = fp32_from_bits([item[0] for item in KNOWN])
    expected = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    ChromaTokenizer().encode(values).save(path)
    del values
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_rgbd_tokens.py"
    process = subprocess.run([sys.executable, str(script), "--decode-file", str(path),
                              "--expected-sha256", expected], check=True, capture_output=True,
                             text=True, timeout=60)
    receipt = json.loads(process.stdout)
    assert receipt["decoded_sha256"] == expected
    assert receipt["exact_sha256_match"] is True
    assert receipt["num_tokens"] == len(KNOWN)
    assert receipt["token_shape"] == [len(KNOWN), 4]
    assert receipt["payload_bytes"] == 4 * len(KNOWN)
    assert receipt["input_arguments"] == ["saved_file", "expected_decoded_sha256"]


@pytest.fixture(scope="module")
def real_rgbd_evidence(tmp_path_factory):
    from argparse import Namespace
    from scripts.verify_rgbd_tokens import verify
    output = tmp_path_factory.mktemp("rgbd-proof") / "verification.json"
    record = verify(Namespace(samples=32, seed=7331, threads=2, child_timeout=60, output=output))
    return record


def test_rgbd_gate_replays_actual_bytes_and_cold_process_receipt(real_rgbd_evidence):
    from scripts.verify_rgbd_tokens import validate_rgbd_evidence
    result = validate_rgbd_evidence(real_rgbd_evidence, require_release=False)
    assert result["verified"] is True
    assert result["samples"] == 32
    assert result["bit_mismatches"] == 0
    assert result["channels_used"] == ["R", "G", "B", "D"]
    assert result["payload_bytes"] == 128


def test_release_rgbd_gate_rejects_smoke_sample_count(real_rgbd_evidence):
    from scripts.verify_rgbd_tokens import validate_rgbd_evidence
    with pytest.raises(ValueError, match="sample count"):
        validate_rgbd_evidence(real_rgbd_evidence, require_release=True)


@pytest.mark.parametrize("tamper", ["incomplete", "missing_source", "stale_source", "missing_artifact",
                                   "vacuous_checks", "count", "file_bytes", "payload_hash", "missing_channel",
                                   "channel_result", "cold_hash", "same_process", "missing_known", "bit_mismatches"])
def test_rgbd_gate_rejects_missing_stale_vacuous_or_tampered_records(real_rgbd_evidence, tamper):
    from scripts.verify_rgbd_tokens import validate_rgbd_evidence
    record = copy.deepcopy(real_rgbd_evidence)
    if tamper == "incomplete":
        record["status"] = "running"
    elif tamper == "missing_source":
        del record["provenance"]
    elif tamper == "stale_source":
        record["provenance"]["source"]["sha256"] = "0" * 64
    elif tamper == "missing_artifact":
        record["provenance"]["artifacts"] = []
    elif tamper == "vacuous_checks":
        record["checks"] = {}
    elif tamper == "count":
        record["config"]["sample_count"] = 0
    elif tamper == "file_bytes":
        record["storage"]["file_bytes"] = 0
    elif tamper == "payload_hash":
        record["storage"]["payload_sha256"] = "0" * 64
    elif tamper == "missing_channel":
        record["decoding"]["channel_mutations"] = []
    elif tamper == "channel_result":
        record["decoding"]["channel_mutations"][0]["after_bits_hex"] = "3f800000"
    elif tamper == "cold_hash":
        record["cold_process"]["decoded_sha256"] = "0" * 64
    elif tamper == "same_process":
        record["encoding_pid"] = record["cold_process"]["pid"]
    elif tamper == "missing_known":
        record["encoding"]["known_patterns"] = []
    else:
        record["decoding"]["bit_mismatches"] = 1
    with pytest.raises(ValueError):
        validate_rgbd_evidence(record, require_release=False)


def test_rgbd_gate_rejects_changed_payload_even_with_repaired_checksum_and_artifact_hash(real_rgbd_evidence):
    from experiments.provenance import artifact_manifest
    from scripts.verify_rgbd_tokens import validate_rgbd_evidence
    root = Path(__file__).resolve().parents[1]
    record = copy.deepcopy(real_rgbd_evidence)
    path = root / record["storage"]["path"]
    original = path.read_bytes()
    magic, metadata, payload = checked_parts(path)
    changed = bytes([payload[0] ^ 1]) + payload[1:]
    try:
        write_checked(path, magic, metadata, changed)
        record["provenance"]["artifacts"] = artifact_manifest(
            [item["path"] for item in record["provenance"]["artifacts"]], root)
        record["storage"]["payload_sha256"] = hashlib.sha256(changed).hexdigest()
        with pytest.raises(ValueError, match="seeded NumPy oracle"):
            validate_rgbd_evidence(record, require_release=False)
    finally:
        path.write_bytes(original)
