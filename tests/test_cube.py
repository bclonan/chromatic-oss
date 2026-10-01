import hashlib
import json
import zipfile

import pytest
import torch

from chromatensor.codecs import LearnedCodec, RGBDCodec, UniformCodec
from chromatensor.cube import ChromaCube
from chromatensor.tensor import ChromaTensor


def make_cube(shape=(3, 3, 3, 3, 3, 3), codec=None, device="cpu"):
    values = torch.linspace(-1, 1, torch.tensor(shape).prod().item(), device=device).reshape(shape)
    return ChromaCube.from_tensor(values, codec, axis_names=tuple(f"d{axis}" for axis in range(len(shape))))


def assert_identity_tracks_codes(original, moved):
    original_codes = original.tensor.codes.reshape(-1, original.tensor.codec.channels).to(torch.int64)
    moved_codes = moved.tensor.codes.reshape(-1, moved.tensor.codec.channels).to(torch.int64)
    assert torch.equal(moved_codes, original_codes[moved.cell_ids.flatten()])
    assert torch.equal(torch.sort(moved.cell_ids.flatten()).values,
                       torch.arange(original.tensor.numel(), device=moved.device))


def codes_equal(first, second):
    return first.dtype == second.dtype and torch.equal(first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8))


@pytest.mark.parametrize("codec", [RGBDCodec(), UniformCodec(8), UniformCodec(16), LearnedCodec(1024)])
def test_quarter_turn_moves_known_cells_and_preserves_integer_symbols(codec):
    cube = make_cube((3, 3, 3), codec)
    before_codes, before_ids = cube.tensor.codes.clone(), cube.cell_ids.clone()
    moved = cube.twist(axes=(0, 1), layer_axis=2, layer_index=1)
    assert moved.shape == cube.shape
    assert moved.tensor.codes.dtype == cube.tensor.codes.dtype
    assert moved.cell_ids.dtype == torch.int64
    # Counterclockwise means new[x,y,z] receives old[y,2-x,z].
    for x in range(3):
        for y in range(3):
            assert moved.cell_ids[x, y, 1] == cube.cell_ids[y, 2 - x, 1]
    assert torch.equal(moved.cell_ids[:, :, 0], cube.cell_ids[:, :, 0])
    assert torch.equal(moved.cell_ids[:, :, 2], cube.cell_ids[:, :, 2])
    assert_identity_tracks_codes(cube, moved)
    assert codes_equal(cube.tensor.codes, before_codes)
    assert torch.equal(cube.cell_ids, before_ids)


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_unsigned_word_twists_preserve_every_code_bit_without_decoding(dtype, monkeypatch):
    bits = 16 if dtype == torch.uint16 else 32
    codec = UniformCodec(bits)
    values = ((torch.arange(27, dtype=torch.int64) * 200000000 + 1000000) % 2**bits).reshape(3, 3, 3, 1)
    cube = ChromaCube(ChromaTensor(values.to(dtype), codec))
    def forbidden(*args, **kwargs):
        raise AssertionError("unsigned rotation decoded scalar values")
    monkeypatch.setattr(codec, "decode", forbidden)
    moved = cube.twist()
    assert_identity_tracks_codes(cube, moved)
    assert moved.tensor.codes.dtype == dtype
    restored = moved.inverse_twist()
    assert codes_equal(restored.tensor.codes, cube.tensor.codes)


@pytest.mark.parametrize("axes,layer_axis", [((0, 1), 2), ((5, 1), 0), ((3, 4), None)])
def test_six_dimensions_four_turns_and_inverse_restore_every_cell(axes, layer_axis):
    cube = make_cube()
    moved = cube
    for _ in range(4):
        moved = moved.twist(axes, layer_axis, 2)
        assert_identity_tracks_codes(cube, moved)
    assert codes_equal(moved.tensor.codes, cube.tensor.codes)
    assert torch.equal(moved.cell_ids, cube.cell_ids)
    recovered = cube.twist(axes, layer_axis, 2, turns=3).inverse_twist(axes, layer_axis, 2, turns=3)
    assert codes_equal(recovered.tensor.codes, cube.tensor.codes)
    assert torch.equal(recovered.cell_ids, cube.cell_ids)


def test_named_axes_negative_coordinates_and_move_sequence_are_reversible():
    cube = make_cube()
    moves = [(("d0", "d1"), "d5", -1, 1), ((-2, -3), "d0", 1, -3),
             (("d2", "d5"), None, 0, 2), (("d3", "d1"), "d4", 2, 7)]
    moved = cube
    for move in moves:
        moved = moved.twist(*move)
    assert not torch.equal(moved.cell_ids, cube.cell_ids)
    assert_identity_tracks_codes(cube, moved)
    for move in reversed(moves):
        moved = moved.inverse_twist(*move)
    assert torch.equal(moved.cell_ids, cube.cell_ids)
    assert codes_equal(moved.tensor.codes, cube.tensor.codes)
    assert moved.axis_names == cube.axis_names


def test_half_turn_accepts_rectangular_plane_but_quarter_turn_keeps_shape():
    cube = make_cube((2, 3, 4))
    moved = cube.twist(turns=2)
    assert moved.shape == (2, 3, 4)
    assert moved.cell_ids[0, 0, 0] == cube.cell_ids[1, 2, 0]
    assert moved.cell_ids[1, 2, 0] == cube.cell_ids[0, 0, 0]
    assert_identity_tracks_codes(cube, moved)
    with pytest.raises(ValueError, match="equal"):
        cube.twist(turns=1)


def test_slice_three_dimensions_respects_requested_axis_order_and_fixed_coordinates():
    cube = make_cube((2, 3, 4, 5, 6, 7))
    result = cube.slice3d(axes=("d5", "d2", "d0"), fixed={"d1": 2, "d3": -1, "d4": 3})
    assert result["axes"] == (5, 2, 0)
    assert result["axis_names"] == ("d5", "d2", "d0")
    assert result["fixed"] == {1: 2, 3: 4, 4: 3}
    assert result["tensor"].shape == (7, 4, 2)
    for a, b, c in [(0, 0, 0), (6, 3, 1), (4, 2, 0)]:
        assert result["cell_ids"][a, b, c] == cube.cell_ids[c, 2, b, 4, 3, a]
        assert codes_equal(result["tensor"].codes[a, b, c], cube.tensor.codes[c, 2, b, 4, 3, a])
    default = cube.slice3d()
    assert default["fixed"] == {3: 0, 4: 0, 5: 0}
    assert torch.equal(default["cell_ids"], cube.cell_ids[:, :, :, 0, 0, 0])


def test_twists_slices_device_moves_and_safe_files_never_decode(monkeypatch, tmp_path):
    codec = UniformCodec(16)
    cube = make_cube(codec=codec)

    def rejected_decode(*args, **kwargs):
        raise AssertionError("Cube geometry decoded numerical values")

    monkeypatch.setattr(codec, "decode", rejected_decode)
    moved = cube.twist((4, 2), 0, 1).inverse_twist((4, 2), 0, 1)
    assert codes_equal(moved.tensor.codes, cube.tensor.codes)
    assert cube.slice3d((5, 3, 0))["tensor"].shape == (3, 3, 3)
    assert cube.to("cpu").tensor.dtype == torch.uint16
    path = tmp_path / "symbols.ccube"
    cube.save(path)
    loaded = ChromaCube.load(path)
    assert codes_equal(loaded.tensor.codes, cube.tensor.codes)
    assert torch.equal(loaded.cell_ids, cube.cell_ids)


@pytest.mark.parametrize("compress", [False, True])
def test_cube_safe_file_roundtrip_restores_geometry_codec_and_ids(compress, tmp_path):
    cube = make_cube(codec=UniformCodec(16)).twist((3, 5), 1, 2)
    path = tmp_path / "six_dimensions.ccube"
    file_bytes = cube.save(path, compress=compress)
    loaded = ChromaCube.load(path)
    assert file_bytes == path.stat().st_size
    assert loaded.axis_names == cube.axis_names
    assert loaded.tensor.codec.equivalent(cube.tensor.codec)
    assert loaded.tensor.decoded_dtype == cube.tensor.decoded_dtype
    assert codes_equal(loaded.tensor.codes, cube.tensor.codes)
    assert torch.equal(loaded.cell_ids, cube.cell_ids)
    assert torch.equal(loaded.decode(), cube.decode())
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"tensor.ctensor", "cell_ids.bin", "manifest.json", "manifest.sha256"}


def rewrite_archive(path, change):
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    change(entries)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


@pytest.mark.parametrize("entry", ["tensor.ctensor", "cell_ids.bin", "manifest.json"])
def test_cube_rejects_archive_tampering_with_valid_zip_crc(entry, tmp_path):
    path = tmp_path / "bad.ccube"
    make_cube().save(path)

    def corrupt(entries):
        data = bytearray(entries[entry])
        data[0] ^= 1
        entries[entry] = bytes(data)

    rewrite_archive(path, corrupt)
    with pytest.raises(ValueError, match="checksum"):
        ChromaCube.load(path)


def test_cube_rejects_checksummed_duplicate_identity_and_wrong_shape(tmp_path):
    path = tmp_path / "bad.ccube"
    make_cube().save(path)

    def duplicate_identity(entries):
        # Keep length unchanged while assigning the first identity to the second cell.
        entries["cell_ids.bin"] = entries["cell_ids.bin"][:8] * 2 + entries["cell_ids.bin"][16:]
        manifest = json.loads(entries["manifest.json"])
        manifest["ids_sha256"] = hashlib.sha256(entries["cell_ids.bin"]).hexdigest()
        entries["manifest.json"] = json.dumps(manifest).encode()
        entries["manifest.sha256"] = hashlib.sha256(entries["manifest.json"]).hexdigest().encode()

    rewrite_archive(path, duplicate_identity)
    with pytest.raises(ValueError, match="permutation"):
        ChromaCube.load(path)
    make_cube().save(path)

    def huge_shape(entries):
        manifest = json.loads(entries["manifest.json"])
        manifest["shape"] = [2 ** 60, 3, 3]
        entries["manifest.json"] = json.dumps(manifest).encode()
        entries["manifest.sha256"] = hashlib.sha256(entries["manifest.json"]).hexdigest().encode()

    rewrite_archive(path, huge_shape)
    with pytest.raises(ValueError, match="shape"):
        ChromaCube.load(path)


@pytest.mark.parametrize("kwargs,error", [
    ({"axes": (0, 0)}, ValueError), ({"axes": (0, 9)}, IndexError),
    ({"axes": ("missing", 1)}, ValueError), ({"axes": (True, 1)}, TypeError),
    ({"layer_axis": 1}, ValueError), ({"layer_index": 3}, IndexError),
    ({"turns": 0.5}, TypeError), ({"turns": True}, TypeError),
])
def test_cube_rejects_invalid_moves(kwargs, error):
    with pytest.raises(error):
        make_cube().twist(**kwargs)


def test_cube_validates_initial_shape_names_ids_and_slice_arguments():
    with pytest.raises(ValueError, match="three"):
        ChromaCube.from_tensor(torch.ones(3, 3))
    with pytest.raises(ValueError, match="nonempty"):
        ChromaCube.from_tensor(torch.ones(3, 3, 0))
    for names in [("x", "x", "z"), ("x", "y"), ("x", "", "z"), "xyz"]:
        with pytest.raises(ValueError, match="Axis names"):
            ChromaCube.from_tensor(torch.ones(3, 3, 3), axis_names=names)
    cube = make_cube()
    with pytest.raises(TypeError, match="integer"):
        ChromaCube(cube.tensor, cube.cell_ids.float())
    with pytest.raises(ValueError, match="permutation"):
        ChromaCube(cube.tensor, torch.zeros(cube.shape, dtype=torch.int64))
    with pytest.raises(ValueError, match="displayed"):
        cube.slice3d(fixed={0: 1})
    with pytest.raises(ValueError, match="more than once"):
        cube.slice3d(fixed={3: 1, "d3": 2})
    with pytest.raises(IndexError):
        cube.slice3d(fixed={3: 3})
    with pytest.raises(ValueError, match="distinct"):
        cube.slice3d((0, 1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("codec", [RGBDCodec(), UniformCodec(16)])
def test_cube_real_cuda_twist_slice_recovery_and_cpu_file(codec, tmp_path):
    original = make_cube(codec=codec)
    cube = original.to("cuda")
    assert cube.device.type == "cuda"
    moved = cube.twist((5, 2), 0, 2)
    assert_identity_tracks_codes(cube, moved)
    assert moved.slice3d((5, 2, 1), {0: 2, 3: 1, 4: 0})["tensor"].device.type == "cuda"
    recovered = moved.inverse_twist((5, 2), 0, 2).to("cpu")
    assert torch.equal(recovered.cell_ids, original.cell_ids)
    assert codes_equal(recovered.tensor.codes, original.tensor.codes)
    path = tmp_path / "cuda.ccube"
    moved.save(path)
    loaded = ChromaCube.load(path, device="cuda")
    assert loaded.device.type == "cuda"
    assert torch.equal(loaded.cell_ids, moved.cell_ids)
    assert codes_equal(loaded.tensor.codes, moved.tensor.codes)
