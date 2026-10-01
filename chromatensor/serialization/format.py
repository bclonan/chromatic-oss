from __future__ import annotations

import hashlib
from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import zipfile
import zlib

import numpy as np
import torch

MAGIC = b"CTENSOR\x01"
MAX_HEADER = 16 * 1024 * 1024
MAX_PAYLOAD = 1024 * 1024 * 1024
MAX_FILE = MAX_PAYLOAD + MAX_HEADER + 1024 * 1024
MAX_ENTRIES = 10000
INTEGER_DTYPES = {"uint8": np.dtype("u1"), "uint16": np.dtype("<u2"), "uint32": np.dtype("<u4"), "uint64": np.dtype("<u8"),
                  "int8": np.dtype("i1"), "int16": np.dtype("<i2"), "int32": np.dtype("<i4"), "int64": np.dtype("<i8")}
FLOAT_DTYPES = {str(t): t for t in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}


def _finite_json(data):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON metadata is forbidden")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON metadata key")
            result[key] = value
        return result
    try:
        return json.loads(data, parse_constant=invalid_constant, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid JSON metadata") from error


def _budget(value, default, name):
    if value is None:
        return default
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer byte limit")
    return min(value, default)


def _read_file(path, limit):
    if Path(path).stat().st_size > limit:
        raise ValueError("file exceeds max_file_bytes")
    with open(path, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file exceeds max_file_bytes")
    return data


def _decompress(payload, expected, label):
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(payload, expected + 1)
    except zlib.error as error:
        raise ValueError("invalid " + label + " compressed payload") from error
    if (len(raw) != expected or not decompressor.eof or decompressor.unused_data
            or decompressor.unconsumed_tail):
        raise ValueError("invalid " + label + " compressed payload")
    return raw


@contextmanager
def _open_archive(path, *, max_file_bytes, max_header_bytes, max_expanded_bytes, max_entries):
    """Bound the ZIP directory before ZipFile allocates its entry objects."""
    file_limit = _budget(max_file_bytes, MAX_FILE, "max_file_bytes")
    header_limit = _budget(max_header_bytes, MAX_HEADER, "max_header_bytes")
    expanded_limit = _budget(max_expanded_bytes, MAX_PAYLOAD + MAX_HEADER, "max_expanded_bytes")
    if type(max_entries) is not int or not 1 <= max_entries <= MAX_ENTRIES:
        raise ValueError("max_entries must be an integer in [1,10000]")
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if size > file_limit:
                raise ValueError("archive exceeds max_file_bytes")
            stream.seek(max(0, size - 65557))
            tail = stream.read(65557)
            position = tail.rfind(b"PK\x05\x06")
            if position < 0 or len(tail) - position < 22:
                raise ValueError("invalid ZIP end record")
            end = struct.unpack("<4s4H2LH", tail[position:position + 22])
            if position + 22 + end[7] != len(tail) or end[1] or end[2] or end[3] != end[4]:
                raise ValueError("invalid or unsupported ZIP directory")
            if end[4] > max_entries or end[5] > header_limit:
                raise ValueError("ZIP directory exceeds the entry or header budget")
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                items = archive.infolist()
                names = [item.filename for item in items]
                if len(items) > max_entries or len(names) != len(set(names)):
                    raise ValueError("archive contains duplicate members or too many entries")
                if sum(item.file_size for item in items) > expanded_limit:
                    raise ValueError("archive exceeds max_expanded_bytes")
                if any(item.flag_bits & 1 or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for item in items):
                    raise ValueError("unsupported archive encryption or compression")
                yield archive
    except (zipfile.BadZipFile, EOFError, zlib.error, struct.error, UnicodeError, RuntimeError) as error:
        raise ValueError("invalid archive") from error


def _read_member(archive, name, limit):
    info = archive.getinfo(name)
    if info.file_size > limit:
        raise ValueError("archive member exceeds its resource budget")
    with archive.open(info) as stream:
        data = stream.read(limit + 1)
    if len(data) != info.file_size or len(data) > limit:
        raise ValueError("truncated or oversized archive member")
    return data


def _dimensions(shape, name):
    if (not isinstance(shape, list) or len(shape) > 64
            or any(type(n) is not int or n < 0 or n > np.iinfo(np.intp).max for n in shape)):
        raise ValueError("invalid " + name)
    return shape


def _atomic_write(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = stream.name
        stream.write(data)
    try:
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode_file(tensor, compress=True):
    if type(compress) is not bool:
        raise TypeError("compress must be boolean")
    if str(tensor.decoded_dtype) not in FLOAT_DTYPES:
        raise ValueError("ctensor output dtype must be a supported real floating dtype")
    if tensor.storage_nbytes > MAX_PAYLOAD:
        raise ValueError("ctensor payload exceeds the format size limit")
    codes = tensor.codes.detach().cpu().contiguous().numpy()
    dtype = codes.dtype.name
    if dtype not in INTEGER_DTYPES:
        raise ValueError("ctensor payload must contain supported integer codes")
    raw = codes.astype(INTEGER_DTYPES[dtype], copy=False).tobytes()
    payload = zlib.compress(raw, 6) if compress else raw
    metadata = {"version": 1, "shape": list(tensor.shape), "code_shape": list(codes.shape), "code_dtype": dtype, "tensor_dtype": str(tensor.decoded_dtype), "codec": tensor.codec.to_config(), "compression": "zlib" if compress else "none", "raw_bytes": len(raw), "payload_bytes": len(payload)}
    header = json.dumps(metadata, separators=(",", ":"), allow_nan=False, sort_keys=True).encode()
    if len(header) > MAX_HEADER:
        raise ValueError("ctensor metadata exceeds the format size limit")
    body = MAGIC + struct.pack("<I", len(header)) + header + payload
    return body + hashlib.sha256(body).digest()


def save_ctensor(tensor, path, compress=True):
    """Atomically write codes, codec metadata, original shape, and SHA-256."""
    data = _encode_file(tensor, compress=compress)
    _atomic_write(path, data)
    return len(data)


def _decode_file(data, device="cpu", *, max_header_bytes=None, max_payload_bytes=None):
    from ..codecs import codec_from_config
    from ..tensor import ChromaTensor

    header_limit = _budget(max_header_bytes, MAX_HEADER, "max_header_bytes")
    payload_limit = _budget(max_payload_bytes, MAX_PAYLOAD, "max_payload_bytes")
    if len(data) < 44 or data[:8] != MAGIC:
        raise ValueError("invalid ctensor magic or version")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("ctensor checksum mismatch")
    header_size = struct.unpack("<I", data[8:12])[0]
    if header_size > header_limit or header_size > len(data) - 44:
        raise ValueError("invalid ctensor header size")
    metadata = _finite_json(data[12:12 + header_size])
    required = {"version", "shape", "code_shape", "code_dtype", "tensor_dtype", "codec", "compression", "raw_bytes", "payload_bytes"}
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError("missing ctensor metadata fields")
    if type(metadata["version"]) is not int or metadata["version"] != 1:
        raise ValueError("unsupported ctensor version")
    shape = _dimensions(metadata["code_shape"], "code shape")
    logical_shape = _dimensions(metadata["shape"], "tensor shape")
    if not shape:
        raise ValueError("code shape requires a channel dimension")
    if not isinstance(metadata["code_dtype"], str) or not isinstance(metadata["tensor_dtype"], str):
        raise ValueError("invalid dtype metadata")
    if type(metadata["raw_bytes"]) is not int or type(metadata["payload_bytes"]) is not int or metadata["payload_bytes"] < 0:
        raise ValueError("invalid payload byte count")
    dtype = INTEGER_DTYPES.get(metadata["code_dtype"])
    if dtype is None:
        raise ValueError("unsupported integer dtype")
    expected = math.prod(shape) * dtype.itemsize
    if expected != metadata["raw_bytes"] or expected > payload_limit:
        raise ValueError("invalid payload allocation")
    payload = data[12 + header_size:-32]
    if len(payload) != metadata["payload_bytes"]:
        raise ValueError("payload size mismatch")
    if metadata["compression"] == "zlib":
        raw = _decompress(payload, expected, "ctensor")
    elif metadata["compression"] == "none":
        raw = payload
    else:
        raise ValueError("unsupported compression")
    if len(raw) != expected:
        raise ValueError("decoded payload size mismatch")
    original_dtype = FLOAT_DTYPES.get(metadata["tensor_dtype"])
    if original_dtype is None:
        raise ValueError("unsupported tensor dtype")
    try:
        codec = codec_from_config(metadata["codec"])
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("invalid codec metadata") from error
    if tuple(shape[:-1]) != tuple(logical_shape) or shape[-1] != codec.channels:
        raise ValueError("codec channels or shape mismatch")
    values = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    codes = torch.from_numpy(values).to(device)
    result = ChromaTensor(codes, codec, dtype=original_dtype)
    return result


def load_ctensor(path, device="cpu", *, max_file_bytes=None, max_header_bytes=None, max_payload_bytes=None):
    header_limit = _budget(max_header_bytes, MAX_HEADER, "max_header_bytes")
    payload_limit = _budget(max_payload_bytes, MAX_PAYLOAD, "max_payload_bytes")
    file_limit = _budget(max_file_bytes, payload_limit + header_limit + 1024 * 1024, "max_file_bytes")
    data = _read_file(path, file_limit)
    return _decode_file(data, device=device, max_header_bytes=header_limit, max_payload_bytes=payload_limit)


def save_chroma_checkpoint(model, path, codec=None):
    """Store real state tensors, retaining FP64 as safe numeric NPY members."""
    from ..codecs import RGBDCodec
    from ..tensor import ChromaTensor

    state = model.state_dict()
    if len(state) + 1 > MAX_ENTRIES:
        raise ValueError("checkpoint has too many state entries")
    for value in state.values():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_complex():
            raise TypeError("checkpoint state must contain dense real tensors")
        if value.is_floating_point() and str(value.dtype) not in FLOAT_DTYPES:
            raise ValueError("unsupported checkpoint floating dtype")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("checkpoint floating state must contain only finite values")
    if sum(value.numel() * value.element_size() for value in state.values()) > MAX_PAYLOAD:
        raise ValueError("checkpoint state exceeds the payload size limit")
    default_codec = codec is None
    codec = RGBDCodec() if default_codec else codec
    output = io.BytesIO()
    raw_fp64 = default_codec and any(value.dtype == torch.float64 for value in state.values())
    manifest = {"version": 2 if raw_fp64 else 1, "entries": []}
    expanded_bytes = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (name, value) in enumerate(state.items()):
            filename = f"tensor_{index}"
            if value.is_floating_point() and not (default_codec and value.dtype == torch.float64):
                filename += ".ctensor"
                data = _encode_file(ChromaTensor.from_tensor(value, codec), compress=False)
                kind = "chroma"
            else:
                filename += ".npy"
                buffer = io.BytesIO()
                np.save(buffer, value.detach().cpu().numpy(), allow_pickle=False)
                data = buffer.getvalue()
                kind = "numeric" if value.is_floating_point() else "integer"
            expanded_bytes += len(data)
            if expanded_bytes > MAX_PAYLOAD + MAX_HEADER:
                raise ValueError("checkpoint expanded archive exceeds the format size limit")
            archive.writestr(filename, data)
            manifest["entries"].append({"name": name, "filename": filename, "kind": kind})
        metadata = json.dumps(manifest, allow_nan=False).encode()
        if len(metadata) > MAX_HEADER or expanded_bytes + len(metadata) > MAX_PAYLOAD + MAX_HEADER:
            raise ValueError("checkpoint manifest exceeds the format size limit")
        archive.writestr("manifest.json", metadata)
    _atomic_write(path, output.getvalue())
    return len(output.getvalue())


def _decode_npy(data, *, kind, max_header_bytes, max_payload_bytes):
    buffer = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(buffer)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(buffer, max_header_size=max_header_bytes)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(buffer, max_header_size=max_header_bytes)
        else:
            raise ValueError("unsupported checkpoint array version")
    except (EOFError, SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid checkpoint array header") from error
    _dimensions(list(shape), "checkpoint array shape")
    expected = math.prod(shape) * dtype.itemsize
    allowed = dtype.kind in "uib" and dtype.itemsize in (1, 2, 4, 8) if kind == "integer" else dtype.kind == "f" and dtype.itemsize == 8
    if dtype.hasobject or not allowed or expected > max_payload_bytes or expected != len(data) - buffer.tell():
        raise ValueError("invalid checkpoint array dtype or payload allocation")
    values = np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=buffer.tell()).reshape(shape, order="F" if fortran else "C")
    if kind == "numeric" and not np.isfinite(values).all():
        raise ValueError("checkpoint numeric state must contain only finite values")
    values = values.astype(dtype.newbyteorder("="), copy=True)
    return torch.from_numpy(values), expected


def load_chroma_checkpoint(path, model=None, device="cpu", *, max_file_bytes=None, max_header_bytes=None,
                           max_payload_bytes=None, max_expanded_bytes=None, max_entries=MAX_ENTRIES):
    state = {}
    header_limit = _budget(max_header_bytes, MAX_HEADER, "max_header_bytes")
    payload_limit = _budget(max_payload_bytes, MAX_PAYLOAD, "max_payload_bytes")
    expanded_limit = _budget(max_expanded_bytes, payload_limit + header_limit, "max_expanded_bytes")
    with _open_archive(path, max_file_bytes=max_file_bytes, max_header_bytes=header_limit,
                       max_expanded_bytes=expanded_limit, max_entries=max_entries) as archive:
        names = set(archive.namelist())
        if "manifest.json" not in names:
            raise ValueError("missing checkpoint manifest")
        manifest = _finite_json(_read_member(archive, "manifest.json", header_limit))
        if not isinstance(manifest, dict) or set(manifest) != {"version", "entries"}:
            raise ValueError("invalid checkpoint manifest")
        if type(manifest["version"]) is not int or manifest["version"] not in (1, 2):
            raise ValueError("unsupported checkpoint version")
        if not isinstance(manifest["entries"], list) or len(manifest["entries"]) + 1 > max_entries:
            raise ValueError("invalid checkpoint entries")
        entry_names, filenames = set(), {"manifest.json"}
        for entry in manifest["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"name", "filename", "kind"}:
                raise ValueError("invalid checkpoint entry")
            if (not isinstance(entry["name"], str) or not entry["name"]
                    or not isinstance(entry["filename"], str) or not entry["filename"]
                    or "/" in entry["filename"] or "\\" in entry["filename"]):
                raise ValueError("invalid checkpoint tensor name or filename")
            if entry["name"] in entry_names or entry["filename"] in filenames:
                raise ValueError("duplicate checkpoint tensor name or filename")
            entry_names.add(entry["name"])
            filenames.add(entry["filename"])
            if entry["kind"] not in ("chroma", "integer", "numeric") or entry["kind"] == "numeric" and manifest["version"] != 2:
                raise ValueError("unsupported checkpoint entry")
        if names != filenames:
            raise ValueError("checkpoint contains missing or unknown archive members")
        allocated = 0
        for entry in manifest["entries"]:
            data = _read_member(archive, entry["filename"], expanded_limit)
            remaining = payload_limit - allocated
            if entry["kind"] == "chroma":
                tensor = _decode_file(data, "cpu", max_header_bytes=header_limit, max_payload_bytes=remaining)
                expected = tensor.numel() * torch.empty((), dtype=tensor.decoded_dtype).element_size()
                if expected > remaining:
                    raise ValueError("checkpoint decoded state exceeds max_payload_bytes")
                state[entry["name"]] = tensor.decode().to(device)
            else:
                value, expected = _decode_npy(data, kind=entry["kind"], max_header_bytes=header_limit, max_payload_bytes=remaining)
                state[entry["name"]] = value.to(device)
            allocated += expected
    if model is not None:
        model.load_state_dict(state)
        return model
    return state
