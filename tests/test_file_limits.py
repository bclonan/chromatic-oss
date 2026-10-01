"""Exercise public file budgets and forged metadata without large allocations."""
import hashlib
import io
import json
import struct
import zipfile
import zlib

import numpy as np
import pytest
import torch

from chromatensor import (
    ChromaCube, ChromaTensor, HolographicCube, load_chroma_checkpoint,
    load_ctensor, save_chroma_checkpoint, save_ctensor,
)
from chromatensor.holographic import load_hcube
from chromatensor.serialization import format as file_format


def state_model(dtype=torch.float32):
    model = torch.nn.Module()
    model.register_buffer("values", torch.tensor([-.333333333333, .123456789012, 1.], dtype=dtype))
    model.register_buffer("count", torch.tensor(3, dtype=torch.int64))
    return model


@pytest.fixture(params=["ctensor", "checkpoint", "ccube", "hcube"])
def saved_format(request, tmp_path):
    kind = request.param
    path = tmp_path / kind
    if kind == "ctensor":
        save_ctensor(ChromaTensor.from_tensor(torch.arange(8.)), path)
        loader = load_ctensor
    elif kind == "checkpoint":
        save_chroma_checkpoint(state_model(), path)
        loader = load_chroma_checkpoint
    elif kind == "ccube":
        ChromaCube.from_tensor(torch.arange(8.).reshape(2, 2, 2)).save(path)
        loader = ChromaCube.load
    else:
        HolographicCube.from_tensor(torch.arange(8.).reshape(2, 2, 2)).save(path)
        loader = HolographicCube.load
    return path, loader


def checked_parts(path):
    data = path.read_bytes()
    size = struct.unpack("<I", data[8:12])[0]
    return data[:8], json.loads(data[12:12 + size]), data[12 + size:-32]


def write_checked(path, magic, metadata, payload, header=None):
    header = json.dumps(metadata).encode() if header is None else header
    body = magic + struct.pack("<I", len(header)) + header + payload
    path.write_bytes(body + hashlib.sha256(body).digest())


def archive_entries(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def write_archive(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)


def numpy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=True)
    return buffer.getvalue()


def checkpoint_member(path, data, *, kind="integer", version=1):
    manifest = {"version": version, "entries": [{"name": "value", "filename": "value.npy", "kind": kind}]}
    write_archive(path, {"manifest.json": json.dumps(manifest).encode(), "value.npy": data})


def test_all_public_loaders_roundtrip_inside_explicit_budgets(saved_format):
    path, loader = saved_format
    result = loader(path, max_file_bytes=path.stat().st_size, max_header_bytes=65536, max_payload_bytes=1024)
    assert result is not None


@pytest.mark.parametrize("budget", ["max_file_bytes", "max_header_bytes", "max_payload_bytes"])
def test_all_public_loaders_reject_limits_before_numpy_allocation(saved_format, budget, monkeypatch):
    path, loader = saved_format
    def forbidden(*args, **kwargs):
        raise AssertionError("loader allocated a NumPy payload before checking its budget")
    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        loader(path, **{budget: 1})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_all_public_loaders_reject_invalid_budget_types(saved_format, value):
    path, loader = saved_format
    with pytest.raises(ValueError, match="nonnegative integer"):
        loader(path, max_payload_bytes=value)


def test_public_hcube_function_forwards_budgets(tmp_path):
    path = tmp_path / "cube.hcube"
    HolographicCube.from_tensor(torch.ones(2, 2)).save(path)
    with pytest.raises(ValueError, match="allocation"):
        load_hcube(path, max_payload_bytes=1)


@pytest.mark.parametrize("format_name", ["ctensor", "hcube"])
@pytest.mark.parametrize("header_kind", ["duplicate", "invalid_json", "nonobject", "nonfinite"])
def test_checked_files_reject_malformed_json(format_name, header_kind, tmp_path):
    path = tmp_path / format_name
    if format_name == "ctensor":
        save_ctensor(ChromaTensor.from_tensor(torch.ones(4)), path)
        loader = load_ctensor
    else:
        HolographicCube.from_tensor(torch.ones(4)).save(path)
        loader = load_hcube
    magic, metadata, payload = checked_parts(path)
    header = {"duplicate": json.dumps(metadata)[:-1].encode() + b', "version": 1}',
              "invalid_json": b'{"version":', "nonobject": b'[]', "nonfinite": b'{"version": NaN}'}[header_kind]
    write_checked(path, magic, metadata, payload, header)
    with pytest.raises(ValueError):
        loader(path)


@pytest.mark.parametrize("format_name", ["ctensor", "hcube"])
@pytest.mark.parametrize("payload_kind", ["bomb", "invalid", "truncated", "concatenated"])
def test_checked_file_decompression_rejects_bombs_and_bad_streams(format_name, payload_kind, tmp_path, monkeypatch):
    path = tmp_path / format_name
    if format_name == "ctensor":
        save_ctensor(ChromaTensor.from_tensor(torch.ones(4)), path)
        loader = load_ctensor
    else:
        HolographicCube.from_tensor(torch.ones(4)).save(path)
        loader = load_hcube
    magic, metadata, original = checked_parts(path)
    payload = {"bomb": zlib.compress(bytes(2 * 1024 * 1024)), "invalid": b'not-zlib',
               "truncated": original[:-1], "concatenated": original + zlib.compress(b'extra')}[payload_kind]
    metadata["payload_bytes"] = len(payload)
    write_checked(path, magic, metadata, payload)
    def forbidden(*args, **kwargs):
        raise AssertionError("bad compressed data reached NumPy allocation")
    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="compressed payload"):
        loader(path, max_payload_bytes=1024)


@pytest.mark.parametrize("format_name", ["ctensor", "hcube"])
def test_checked_huge_shape_fails_before_decompression(format_name, tmp_path, monkeypatch):
    path = tmp_path / format_name
    if format_name == "ctensor":
        save_ctensor(ChromaTensor.from_tensor(torch.ones(4)), path)
        loader = load_ctensor
    else:
        HolographicCube.from_tensor(torch.ones(4)).save(path)
        loader = load_hcube
    magic, metadata, payload = checked_parts(path)
    metadata["shape"] = [2**40]
    channels = 4 if format_name == "ctensor" else 3
    if format_name == "ctensor":
        metadata["code_shape"] = [2**40, channels]
    else:
        metadata["axis_names"] = ["x"]
    metadata["raw_bytes"] = channels * 2**40
    write_checked(path, magic, metadata, payload)
    def forbidden(*args, **kwargs):
        raise AssertionError("huge shape reached the decompressor")
    monkeypatch.setattr(file_format.zlib, "decompressobj", forbidden)
    with pytest.raises(ValueError, match="allocation"):
        loader(path, max_payload_bytes=1024)


@pytest.mark.parametrize("kind", ["checkpoint", "ccube"])
def test_zip_expansion_budget_precedes_member_reads(kind, tmp_path, monkeypatch):
    path = tmp_path / kind
    if kind == "checkpoint":
        save_chroma_checkpoint(state_model(), path)
        loader = load_chroma_checkpoint
    else:
        ChromaCube.from_tensor(torch.ones(2, 2, 2)).save(path)
        loader = ChromaCube.load
    def forbidden(*args, **kwargs):
        raise AssertionError("archive member opened before expansion budget check")
    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    with pytest.raises(ValueError, match="max_expanded_bytes"):
        loader(path, max_expanded_bytes=1)


@pytest.mark.parametrize("kind", ["checkpoint", "ccube"])
def test_zip_directory_budget_precedes_zipfile_parsing(kind, tmp_path, monkeypatch):
    path = tmp_path / kind
    if kind == "checkpoint":
        save_chroma_checkpoint(state_model(), path)
        loader = load_chroma_checkpoint
    else:
        ChromaCube.from_tensor(torch.ones(2, 2, 2)).save(path)
        loader = ChromaCube.load
    raw = bytearray(path.read_bytes())
    offset = raw.rfind(b"PK\x05\x06")
    struct.pack_into("<I", raw, offset + 12, 2**30)
    path.write_bytes(raw)
    def forbidden(*args, **kwargs):
        raise AssertionError("ZipFile parsed an oversized directory")
    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="directory exceeds"):
        loader(path, max_header_bytes=65536)


@pytest.mark.parametrize("kind", ["checkpoint", "ccube"])
def test_zip_rejects_duplicate_members(kind, tmp_path):
    path = tmp_path / kind
    if kind == "checkpoint":
        save_chroma_checkpoint(state_model(), path)
        loader = load_chroma_checkpoint
    else:
        ChromaCube.from_tensor(torch.ones(2, 2, 2)).save(path)
        loader = ChromaCube.load
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr("manifest.json", b'{}')
    with pytest.raises(ValueError, match="duplicate|entry"):
        loader(path)


def test_checkpoint_rejects_invalid_utf8_zip_member_names(tmp_path):
    path = tmp_path / "bad.zip"
    save_chroma_checkpoint(state_model(), path)
    data = bytearray(path.read_bytes())
    directory = data.index(b"PK\x01\x02")
    flags = struct.unpack_from("<H", data, directory + 8)[0]
    struct.pack_into("<H", data, directory + 8, flags | 2048)
    data[directory + 46] = 255
    path.write_bytes(data)
    with pytest.raises(ValueError, match="archive"):
        load_chroma_checkpoint(path)


@pytest.mark.parametrize("format_name", ["ctensor", "checkpoint", "ccube", "hcube"])
def test_truncated_files_fail_cleanly(format_name, tmp_path):
    path = tmp_path / format_name
    path.write_bytes(b'broken')
    loader = {"ctensor": load_ctensor, "checkpoint": load_chroma_checkpoint,
              "ccube": ChromaCube.load, "hcube": load_hcube}[format_name]
    with pytest.raises(ValueError):
        loader(path)


@pytest.mark.parametrize("manifest", [[], {}, {"version": True, "entries": []},
                                     {"version": 1, "entries": {}}, {"version": 1, "entries": [None]},
                                     {"version": 1, "entries": [{"name": [], "filename": "x", "kind": "integer"}]}])
def test_checkpoint_rejects_malformed_manifests(manifest, tmp_path):
    path = tmp_path / "bad.zip"
    write_archive(path, {"manifest.json": json.dumps(manifest).encode()})
    with pytest.raises(ValueError):
        load_chroma_checkpoint(path)


@pytest.mark.parametrize("attack", ["duplicate_json", "duplicate_name", "duplicate_filename", "unknown_member", "missing_member"])
def test_checkpoint_rejects_ambiguous_manifest_members(attack, tmp_path):
    path = tmp_path / "bad.zip"
    save_chroma_checkpoint(state_model(), path)
    entries = archive_entries(path)
    manifest = json.loads(entries["manifest.json"])
    if attack == "duplicate_json":
        entries["manifest.json"] = json.dumps(manifest)[:-1].encode() + b', "version": 1}'
    elif attack == "unknown_member":
        entries["extra.npy"] = numpy_bytes(np.array([1]))
    elif attack == "missing_member":
        del entries[manifest["entries"][0]["filename"]]
    else:
        second = dict(manifest["entries"][0])
        if attack == "duplicate_name":
            second["filename"] = "other.ctensor"
            entries[second["filename"]] = entries[manifest["entries"][0]["filename"]]
        else:
            second["name"] = "other"
        manifest["entries"].append(second)
        entries["manifest.json"] = json.dumps(manifest).encode()
    write_archive(path, entries)
    with pytest.raises(ValueError):
        load_chroma_checkpoint(path)


@pytest.mark.parametrize("attack", ["object", "float_as_integer", "huge_shape", "truncated", "bad_magic"])
def test_checkpoint_npy_attacks_fail_before_payload_allocation(attack, tmp_path, monkeypatch):
    path = tmp_path / "bad.zip"
    if attack == "object":
        data = numpy_bytes(np.array([{"python": "object"}], dtype=object))
    elif attack == "float_as_integer":
        data = numpy_bytes(np.array([1.], dtype=np.float64))
    elif attack == "huge_shape":
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(buffer, {"descr": "<i8", "fortran_order": False, "shape": (2**40,)})
        data = buffer.getvalue()
    elif attack == "truncated":
        data = numpy_bytes(np.array([1, 2], dtype=np.int64))[:-1]
    else:
        data = b'not an array'
    checkpoint_member(path, data)
    def forbidden(*args, **kwargs):
        raise AssertionError("malicious NPY header reached payload allocation")
    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError):
        load_chroma_checkpoint(path, max_payload_bytes=1024)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_default_checkpoint_roundtrip_is_lossless_for_supported_real_dtypes(dtype, tmp_path):
    model = state_model(dtype)
    path = tmp_path / "state.zip"
    save_chroma_checkpoint(model, path)
    state = load_chroma_checkpoint(path)
    for name, original in model.state_dict().items():
        assert state[name].dtype == original.dtype
        assert torch.equal(state[name], original)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["version"] == (2 if dtype == torch.float64 else 1)


def test_checkpoint_export_rejects_complex_without_replacing_existing_file(tmp_path):
    model = state_model()
    model.register_buffer("complex", torch.tensor([1 + 2j]))
    path = tmp_path / "state.zip"
    path.write_bytes(b'original')
    with pytest.raises(TypeError, match="dense real"):
        save_chroma_checkpoint(model, path)
    assert path.read_bytes() == b'original'


def test_checkpoint_rejects_nonfinite_raw_fp64_state_on_export_and_load(tmp_path):
    path = tmp_path / "state.zip"
    model = state_model(torch.float64)
    model.values[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_chroma_checkpoint(model, path)
    checkpoint_member(path, numpy_bytes(np.array([float("inf")], dtype=np.float64)), kind="numeric", version=2)
    with pytest.raises(ValueError, match="finite"):
        load_chroma_checkpoint(path)


def test_ctensor_dimension_larger_than_platform_index_is_rejected(tmp_path):
    path = tmp_path / "empty.ctensor"
    save_ctensor(ChromaTensor.from_tensor(torch.empty(0)), path)
    magic, metadata, payload = checked_parts(path)
    metadata["shape"] = [0, 2**80]
    metadata["code_shape"] = [0, 2**80, 4]
    write_checked(path, magic, metadata, payload)
    with pytest.raises(ValueError, match="shape"):
        load_ctensor(path)


def test_checkpoint_aggregate_payload_budget_prevents_model_mutation(tmp_path):
    model = torch.nn.Module()
    model.register_buffer("first", torch.arange(4.))
    model.register_buffer("second", torch.arange(4.))
    path = tmp_path / "state.zip"
    save_chroma_checkpoint(model, path)
    model.first.fill_(9)
    model.second.fill_(9)
    with pytest.raises(ValueError):
        load_chroma_checkpoint(path, model, max_payload_bytes=20)
    assert torch.equal(model.first, torch.full_like(model.first, 9))
    assert torch.equal(model.second, torch.full_like(model.second, 9))


def test_legacy_integer_npy_fortran_and_byte_order_roundtrip(tmp_path):
    path = tmp_path / "state.zip"
    array = np.array([[1, 2], [3, 4]], dtype=">i4", order="F")
    checkpoint_member(path, numpy_bytes(array))
    value = load_chroma_checkpoint(path)["value"]
    assert value.dtype == torch.int32
    assert torch.equal(value, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))


def test_ccube_rejects_checksummed_invalid_axis_metadata_before_payload_copy(tmp_path, monkeypatch):
    path = tmp_path / "cube.ccube"
    ChromaCube.from_tensor(torch.ones(2, 2, 2)).save(path)
    entries = archive_entries(path)
    manifest = json.loads(entries["manifest.json"])
    manifest["axis_names"] = ["x", "x", "z"]
    entries["manifest.json"] = json.dumps(manifest).encode()
    entries["manifest.sha256"] = hashlib.sha256(entries["manifest.json"]).hexdigest().encode()
    write_archive(path, entries)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid axes reached NumPy allocation")
    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="axis names"):
        ChromaCube.load(path)


def test_hcube_rejects_overflowing_integer_scale_before_payload_copy(tmp_path, monkeypatch):
    path = tmp_path / "cube.hcube"
    HolographicCube.from_tensor(torch.ones(4)).save(path)
    magic, metadata, payload = checked_parts(path)
    metadata["amplitude_scale"] = 10**500
    write_checked(path, magic, metadata, payload)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid amplitude scale reached NumPy allocation")
    monkeypatch.setattr(np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="amplitude scale"):
        load_hcube(path)
