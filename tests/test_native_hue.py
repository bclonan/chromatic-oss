"""Independent integer, byte mutation and persistence tests for native hue tensors."""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from chromatensor import ColorTensor, HueDecoder, HueEncoder, HueTensor
from chromatensor.hue import native_info
from scripts import verify_native_hue as proof

M = 391680


def assert_phases(tensor, expected):
    expected = np.asarray(expected, dtype=np.uint32)
    assert tensor.shape == expected.shape
    assert tensor.numel == expected.size
    assert tensor.storage_nbytes == expected.size * 4
    assert tensor.phases() == tuple(int(value) for value in expected.reshape(-1))
    assert tensor.rgbd_bytes == proof.reference_rgbd(expected)


def test_exact_color_corners_fine_channel_and_public_aliases():
    values = [0, 255, 256, 65280, 130560, 195840, 261120, 326400, M - 1]
    expected = bytes([255, 0, 0, 0, 255, 0, 0, 255, 255, 1, 0, 0,
                      255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0,
                      0, 0, 255, 0, 255, 0, 255, 0, 255, 0, 1, 255])
    assert ColorTensor is HueTensor
    with HueTensor.from_phases(values) as tensor:
        assert tensor.rgbd_bytes == expected
        assert HueDecoder.decode_phases(tensor) == tuple(values)
        assert HueDecoder.decode_rgb(tensor) == bytes(channel for i in range(0, len(expected), 4) for channel in expected[i:i + 3])
        with HueEncoder.encode_rgbd(expected, (len(values),)) as restored:
            assert restored.phases() == tuple(values)
        with HueEncoder.encode_phases(values) as restored:
            assert restored.rgbd_bytes == expected


def test_rgb_projection_uses_exact_integer_hue_and_discards_saturation_value():
    colors = [(240, 80, 40), (10, 90, 50), (30, 40, 220), (128, 40, 40), (255, 0, 0)]
    expected = []
    for red, green, blue in colors:
        high, low = max(red, green, blue), min(red, green, blue)
        span = high - low
        numerator = green - blue if high == red else 2 * span + blue - red if high == green else 4 * span + red - green
        phase = Fraction(numerator % (6 * span), 6 * span) * M
        expected.append(((2 * phase.numerator + phase.denominator) // (2 * phase.denominator)) % M)
    raw = bytes(channel for color in colors for channel in color)
    with HueEncoder.encode_rgb(raw, (len(colors),)) as tensor:
        assert_phases(tensor, expected)
        assert tensor.phases()[-2:] == (0, 0)
        assert tensor.rgbd_bytes != bytes(channel for color in colors for channel in (*color, 0))
    with pytest.raises(ValueError, match="achromatic"):
        HueTensor.from_rgb(bytes([19, 19, 19]), (1,))


@pytest.mark.parametrize("seed", [7, 23, 7331])
def test_random_modular_algebra_and_immutability(seed):
    generator = np.random.default_rng(seed)
    first = generator.integers(0, M, size=(9, 13), dtype=np.uint32)
    second = generator.integers(0, M, size=(9, 13), dtype=np.uint32)
    first[0, :4], second[0, :4] = [0, M - 1, M // 2, 1], [M - 1, 1, M // 2, M - 1]
    with HueTensor.from_phases(first) as a, HueTensor.from_phases(second) as b:
        original = (a.rgbd_bytes, b.rgbd_bytes)
        with a + b as result:
            assert_phases(result, (first.astype(np.uint64) + second.astype(np.uint64)) % M)
        with a - b as result:
            assert_phases(result, (first.astype(np.int64) - second.astype(np.int64)) % M)
        with a * b as result:
            assert_phases(result, (first.astype(np.uint64) * second.astype(np.uint64)) % M)
        for ticks in (-2**63, -M - 1, -1, 0, 1, M, 2**63 - 1):
            with a.rotate(ticks) as result:
                assert_phases(result, (first.astype(np.int64) + ticks % M) % M)
        with a.sum() as result:
            assert_phases(result, np.asarray(int(first.sum(dtype=np.uint64)) % M, dtype=np.uint32))
        with a.dot(b) as result:
            assert_phases(result, np.asarray(int((first.astype(np.uint64) * second.astype(np.uint64)).sum()) % M, dtype=np.uint32))
        assert (a.rgbd_bytes, b.rgbd_bytes) == original


@pytest.mark.parametrize("shape_a,shape_b", [((3, 1), (1, 5)), ((2, 1, 4), (3, 4)), ((), (2, 3)), ((0, 3), (1, 3))])
def test_native_broadcasting_matches_numpy_integer_shapes(shape_a, shape_b):
    first = np.arange(np.prod(shape_a, dtype=np.int64), dtype=np.uint32).reshape(shape_a)
    second = (np.arange(np.prod(shape_b, dtype=np.int64), dtype=np.uint32) + M - 30).reshape(shape_b)
    with HueTensor.from_phases(first) as a, HueTensor.from_phases(second) as b:
        with a + b as result:
            assert_phases(result, (first.astype(np.uint64) + second.astype(np.uint64)) % M)
        with a - b as result:
            assert_phases(result, (first.astype(np.int64) - second.astype(np.int64)) % M)
        with a * b as result:
            assert_phases(result, (first.astype(np.uint64) * second.astype(np.uint64)) % M)


@pytest.mark.parametrize("weight", [0, 1, 127, 128, 254, 255])
def test_shortest_arc_blend_rounding_wrap_and_positive_halfturn_tie(weight):
    first = np.array([0, M - 10, 10, 0, M // 2, 100], dtype=np.uint32)
    second = np.array([M - 1, 10, M - 10, M // 2, 0, 103], dtype=np.uint32)
    expected = []
    for start, stop in zip(first.tolist(), second.tolist()):
        forward = (stop - start) % M
        direction, distance = (1, forward) if forward <= M // 2 else (-1, M - forward)
        expected.append((start + direction * ((distance * weight + 127) // 255)) % M)
    with HueTensor.from_phases(first) as a, HueTensor.from_phases(second) as b:
        with a.blend(b, weight) as result:
            assert_phases(result, expected)


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 0, 4), (0,), ()])
def test_shape_permutations_slicing_reductions_and_empty_contracts(shape):
    values = np.arange(np.prod(shape, dtype=np.int64), dtype=np.uint32).reshape(shape)
    with HueTensor.from_phases(values) as tensor:
        assert_phases(tensor, values)
        with tensor.reshape(values.size) as reshaped:
            assert_phases(reshaped, values.reshape(-1))
        with tensor.permute(*reversed(range(len(shape)))) as permuted:
            assert_phases(permuted, values.transpose())
        with tensor.sum() as total:
            assert_phases(total, np.asarray(int(values.sum(dtype=np.uint64)) % M, dtype=np.uint32))
        for axis in range(len(shape)):
            with tensor.sum(axis - len(shape)) as total:
                assert_phases(total, values.sum(axis=axis, dtype=np.uint64) % np.uint64(M))
            with tensor.slice(axis, 0, shape[axis], 2) as sliced:
                selection = [slice(None)] * len(shape)
                selection[axis] = slice(0, shape[axis], 2)
                assert_phases(sliced, values[tuple(selection)])


@pytest.mark.parametrize("dimensions", [(3, 5, 2), (2, 0, 3), (0, 4, 3), (3, 4, 0)])
def test_matrix_products_include_empty_inner_and_outer_dimensions(dimensions):
    rows, inner, columns = dimensions
    generator = np.random.default_rng(7331)
    first = generator.integers(0, M, size=(rows, inner), dtype=np.uint32)
    second = generator.integers(0, M, size=(inner, columns), dtype=np.uint32)
    with HueTensor.from_phases(first) as a, HueTensor.from_phases(second) as b:
        with a @ b as result:
            assert_phases(result, (first.astype(np.uint64) @ second.astype(np.uint64)) % M)


def test_actual_rgb_and_D_byte_mutations_change_loaded_phase_and_algebra():
    first = np.array([[0, 255], [65280, M - 1]], dtype=np.uint32)
    second = np.array([[2, 3], [5, 7]], dtype=np.uint32)
    with HueTensor.from_phases(first) as original, HueTensor.from_phases(second) as b:
        original_bytes = original.rgbd_bytes
        D_mutation = bytearray(original_bytes)
        D_mutation[3] = 1
        expected = first.copy()
        expected[0, 0] = 1
        with HueTensor.from_rgbd(D_mutation, (2, 2)) as changed:
            assert_phases(changed, expected)
            with changed @ b as product:
                assert_phases(product, (expected.astype(np.uint64) @ second.astype(np.uint64)) % M)
        RGB_mutation = bytearray(original_bytes)
        RGB_mutation[1] = 1
        expected[0, 0] = 256
        with HueTensor.from_rgbd(RGB_mutation, (2, 2)) as changed:
            assert_phases(changed, expected)
        assert original.rgbd_bytes == original_bytes


@pytest.mark.parametrize("phases", [[-1], [M], [2**64], [1.5], [True], [[1], [2, 3]]])
def test_invalid_phase_inputs_fail_instead_of_rounding_wrapping_or_padding(phases):
    with pytest.raises((TypeError, ValueError)):
        HueTensor.from_phases(phases)


def test_invalid_shapes_operations_and_closed_handles_fail():
    for shape in [(-1,), (True,), (2,), (1,) * 33, (2**64,), (2**63, 3, 0)]:
        with pytest.raises((TypeError, ValueError)):
            HueTensor.from_phases([1], shape=shape)
    with HueTensor.from_phases([[1, 2], [3, 4]]) as a, HueTensor.from_phases([1, 2, 3]) as b:
        for function in [lambda: a + b, lambda: a.dot(b), lambda: a @ b, lambda: a.reshape(-1),
                         lambda: a.reshape(3), lambda: a.permute(0, 0), lambda: a.permute(2, 0),
                         lambda: a.sum(2), lambda: a.slice(0, -1, 2), lambda: a.slice(0, 0, 3),
                         lambda: a.slice(0, 0, 2, 0), lambda: a.slice(0, 0, 2, -1),
                         lambda: a.slice(0, 0, 2, 2**(8 * struct.calcsize("P")) + 1),
                         lambda: a.rotate(2**63), lambda: a.rotate(True), lambda: a.blend(a, 256),
                         lambda: a.blend(a, -1), lambda: a.blend(a, True), lambda: a + 1]:
            with pytest.raises((TypeError, ValueError)):
                function()
    with HueTensor.from_phases(1) as scalar:
        with pytest.raises(ValueError):
            scalar.sum(0)
    for raw in [b"\x00\x00\x00\x00", b"\xff\xff\xff\x00", b"\x80\x80\x01\x00"]:
        with pytest.raises(ValueError):
            HueTensor.from_rgbd(raw, (1,))
    with pytest.raises(ValueError):
        HueTensor.from_rgbd(b"\xff\x00\x00", (1,))
    with pytest.raises(TypeError):
        HueTensor.from_rgbd([255, 0, 0, 0], (1,))
    tensor = HueTensor.from_phases([0])
    tensor.close()
    tensor.close()
    with pytest.raises(RuntimeError, match="closed"):
        tensor.phases()


@pytest.mark.parametrize("values", [np.empty((2, 0, 4), dtype=np.float32), np.array([True, False]),
                                    np.array([1, 2], dtype=np.float64), np.empty((0,), dtype=object)])
def test_numpy_boundary_rejects_floating_boolean_and_object_dtypes_even_when_empty(values):
    with pytest.raises(TypeError):
        HueTensor.from_phases(values)


def test_NUL_path_is_rejected_before_native_IO_and_does_not_overwrite_target(tmp_path):
    path = tmp_path / "preserved.huetensor"
    path.write_bytes(b"existing file must remain")
    unsafe_path = str(path) + "\0ignored-suffix"
    with HueTensor.from_phases([0]) as tensor:
        with pytest.raises(ValueError, match="NUL"):
            tensor.save(unsafe_path)
    assert path.read_bytes() == b"existing file must remain"
    with pytest.raises(ValueError, match="NUL"):
        HueTensor.load(unsafe_path)


def test_native_file_contains_only_header_actual_color_bytes_and_checksum(tmp_path):
    path = tmp_path / "actual-colors.huetensor"
    values = np.array([[0, 255, 256], [65280, 130560, M - 1]], dtype=np.uint32)
    with HueTensor.from_phases(values) as tensor:
        assert tensor.save(path) == path.stat().st_size
        raw = path.read_bytes()
        assert raw[:8] == b"HUECT\0\r\n"
        version, flags, rank, count = struct.unpack("<HHIQ", raw[8:24])
        assert (version, flags, rank, count) == (1, 0, 2, 6)
        assert struct.unpack("<QQ", raw[24:40]) == (2, 3)
        assert raw[40:-32] == tensor.rgbd_bytes
        assert raw[-32:] == hashlib.sha256(raw[:-32]).digest()
    with HueTensor.load(path, max_bytes=path.stat().st_size) as loaded:
        assert_phases(loaded, values)
    with pytest.raises((OSError, ValueError)):
        HueTensor.load(path, max_bytes=path.stat().st_size - 1)
    with pytest.raises(OSError):
        HueTensor.load(tmp_path / "missing.huetensor")


@pytest.mark.parametrize("mutation", ["truncate", "trailing", "checksum", "noncanonical", "shape", "count", "rank", "flags", "version"])
def test_corrupt_or_adversarial_files_fail_even_after_rehashing(tmp_path, mutation):
    path = tmp_path / "bad.huetensor"
    with HueTensor.from_phases([0, 256, M - 1]) as tensor:
        tensor.save(path)
    raw = bytearray(path.read_bytes())
    if mutation == "truncate":
        raw = raw[:-1]
    elif mutation == "trailing":
        raw += b"x"
    elif mutation == "checksum":
        raw[35] ^= 1
    else:
        if mutation == "noncanonical":
            raw[32:36] = b"\x00\x00\x00\x00"
        elif mutation == "shape":
            raw[24:32] = struct.pack("<Q", 2**63)
        elif mutation == "count":
            raw[16:24] = struct.pack("<Q", 2**63)
        elif mutation == "rank":
            raw[12:16] = struct.pack("<I", 33)
        elif mutation == "flags":
            raw[10:12] = struct.pack("<H", 1)
        elif mutation == "version":
            raw[8:10] = struct.pack("<H", 2)
        raw[-32:] = hashlib.sha256(raw[:-32]).digest()
    path.write_bytes(raw)
    with pytest.raises((OSError, ValueError)):
        HueTensor.load(path)


def test_native_binary_receipt_matches_actual_packaged_file():
    info = native_info()
    assert info["abi_version"] == 1 and info["phase_modulus"] == M
    assert info["bytes"] == Path(info["path"]).stat().st_size
    assert info["sha256"] == hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest()


def test_parallel_close_read_and_reversed_operand_calls_do_not_crash_or_deadlock(tmp_path):
    child = tmp_path / "native-concurrency.py"
    child.write_text(
        "import sys, json\n"
        + "sys.path.insert(0, " + repr(str(proof.ROOT)) + ")\n"
        + "from concurrent.futures import ThreadPoolExecutor\n"
        + "from threading import Barrier\n"
        + "from chromatensor import HueTensor\n"
        + "def worker(index):\n"
        + "    barrier.wait()\n"
        + "    try:\n"
        + "        kind = index % 8\n"
        + "        if kind == 0: a.close(); return 'closed'\n"
        + "        if kind == 1: return ('read', len(a.rgbd_bytes))\n"
        + "        if kind == 2: result = a.rotate(index)\n"
        + "        elif kind == 3: result = a + b\n"
        + "        elif kind == 4: result = b + a\n"
        + "        elif kind == 5: result = a @ b\n"
        + "        elif kind == 6: result = b @ a\n"
        + "        else: b.close(); return 'closed'\n"
        + "        with result: return ('valid', result.numel)\n"
        + "    except RuntimeError as error:\n"
        + "        if 'closed' not in str(error): raise\n"
        + "        return 'closed'\n"
        + "with ThreadPoolExecutor(max_workers=8) as pool:\n"
        + "    for turn in range(40):\n"
        + "        a = HueTensor.from_phases(list(range(4096)), (64, 64))\n"
        + "        b = HueTensor.from_phases(list(range(4096, 8192)), (64, 64))\n"
        + "        barrier = Barrier(8)\n"
        + "        rows = list(pool.map(worker, range(8)))\n"
        + "        assert len(rows) == 8\n"
        + "        a.close(); b.close()\n"
        + "c = HueTensor.from_phases(list(range(64)), (8, 8))\n"
        + "d = HueTensor.from_phases(list(range(64, 128)), (8, 8))\n"
        + "def reverse(index):\n"
        + "    with (c @ d if index % 2 else d @ c) as value: return value.numel\n"
        + "with ThreadPoolExecutor(max_workers=8) as pool:\n"
        + "    assert list(pool.map(reverse, range(500))) == [64] * 500\n"
        + "c.close(); d.close()\n"
        + "assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)\n"
        + "print(json.dumps({'calls': 820, 'native_only': True}))\n",
        encoding="utf-8",
    )
    process = subprocess.run([sys.executable, "-I", str(child)], capture_output=True, encoding="utf-8", timeout=15)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {"calls": 820, "native_only": True}


@pytest.fixture
def receipt(tmp_path):
    # Explicitly reduced native evidence cannot pass the million-value gate.
    return proof.run_verification(tmp_path / "native-smoke.json", tmp_path / "native-artifacts",
                                  samples=256, seed=7331, repeats=1)


def test_native_smoke_receipt_runs_file_only_cold_process_without_torch_or_numpy(receipt):
    assert proof.validate_native_hue_receipt(receipt, require_million=False)
    with pytest.raises(ValueError, match="reduced"):
        proof.validate_native_hue_receipt(receipt)
    cold = receipt["cold_process"]["receipt"]
    assert cold["input_scope"] == "saved HueTensor file and reference hashes only"
    assert cold["isolated_python"] is True
    assert cold["torch_imported"] is False and cold["numpy_imported"] is False
    arguments = receipt["cold_process"]["arguments"]
    assert arguments[0] == "scripts/verify_native_hue.py"
    assert not Path(arguments[2]).is_absolute()
    assert not Path(cold["native_binary"]["path"]).is_absolute()
    assert receipt["integer_control"]["payload_bytes"] == receipt["tensor_file"]["payload_bytes"] == 256 * 4
    assert receipt["matrix_and_mutations"]["D_byte_mutation_changes_matmul"] is True
    assert receipt["synthetic_retrieval"]["correct"] == 256


def test_native_receipt_rejects_false_operator_hash_binary_identity_and_missing_operation(receipt):
    changed = copy.deepcopy(receipt)
    changed["operations"]["multiply"]["phase_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="operation disagrees"):
        proof.validate_native_hue_receipt(changed, require_million=False)
    changed = copy.deepcopy(receipt)
    changed["native_binary"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binary identity"):
        proof.validate_native_hue_receipt(changed, require_million=False)
    changed = copy.deepcopy(receipt)
    del changed["operations"]["sum"]
    with pytest.raises(ValueError, match="operation evidence is missing"):
        proof.validate_native_hue_receipt(changed, require_million=False)


def test_rehashed_integer_control_tampering_and_false_execution_claims_are_rejected(receipt):
    from experiments.provenance import artifact_manifest
    changed = copy.deepcopy(receipt)
    changed["execution"]["floating_tensor_core"] = True
    with pytest.raises(ValueError, match="execution claims"):
        proof.validate_native_hue_receipt(changed, require_million=False)
    path = proof.resolve(receipt["integer_control"]["path"])
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    path.write_bytes(raw)
    changed = copy.deepcopy(receipt)
    changed["integer_control"]["sha256"] = hashlib.sha256(raw).hexdigest()
    changed["provenance"]["artifacts"] = artifact_manifest([item["path"] for item in changed["provenance"]["artifacts"]])
    with pytest.raises(ValueError, match="seeded phase source"):
        proof.validate_native_hue_receipt(changed, require_million=False)


def test_reference_encoder_and_blend_oracles_cover_every_wheel_state():
    values = np.arange(M, dtype=np.uint32)
    expected = proof.reference_rgbd(values)
    with HueTensor.from_phases(values) as tensor:
        assert tensor.rgbd_bytes == expected
        with HueTensor.from_rgbd(expected, (M,)) as restored:
            assert restored.phases() == tuple(range(M))
    assert proof.reference_blend(np.array([0], np.uint32), np.array([M // 2], np.uint32), 255)[0] == M // 2


def test_million_value_reference_reductions_do_not_round_large_integer_sums():
    count = 1000001
    values = np.full(count, M - 1, dtype=np.uint32)
    references = proof.reference_operations(values, values)
    dot, total = references["dot"](), references["sum"]()
    assert dot.dtype == total.dtype == np.uint32
    assert int(dot) == count % M
    assert int(total) == (-count) % M
