"""Exercise a clean native hue wheel under isolated Python without tensor runtimes."""
import argparse
import hashlib
from importlib import metadata
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import site
import sys
import tempfile


def environment_path(path, prefix):
    """Record package placement without disclosing a local account path."""
    return "<environment>/" + Path(path).resolve().relative_to(Path(prefix).resolve()).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import chromatic_hue as hue
    prefix = Path(sys.prefix).resolve()
    module = Path(hue.__file__).resolve()
    binary = Path(hue.native_info()["path"])
    assert module.is_relative_to(prefix) and binary.is_relative_to(prefix)
    assert sys.flags.isolated and not site.ENABLE_USER_SITE
    assert metadata.version("chromatic-hue") == "0.1.0"
    dependencies = metadata.requires("chromatic-hue") or []
    assert dependencies == []
    assert not {"torch", "numpy"}.intersection(sys.modules)
    assert find_spec("torch") is None and find_spec("numpy") is None
    a = hue.HueEncoder.encode_phases([0, 65280, 130560, 391679], (2, 2))
    assert a.rgbd_bytes == bytes([255, 0, 0, 0, 255, 255, 0, 0, 0, 255, 0, 0, 255, 0, 1, 255])
    assert hue.HueDecoder.decode_phases(a) == (0, 65280, 130560, 391679)
    assert hue.HueDecoder.decode_rgb(a) == bytes([255, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 1])
    b = hue.Tensor.from_phases([1, 2, 3, 4], (2, 2))
    expected = tuple(sum(x * y for x, y in zip(row, column)) % hue.PHASE_MODULUS
                     for row in ((0, 65280), (130560, 391679)) for column in ((1, 3), (2, 4)))
    assert (a @ b).phases() == expected
    assert a.rotate(1).phases() == (1, 65281, 130561, 0)
    changed = bytearray(a.rgbd_bytes)
    changed[3] = 1
    assert hue.Tensor.from_rgbd(changed, (2, 2)).phases()[0] == 1
    assert a.T.phases() == (0, 130560, 65280, 391679)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "roundtrip.hue"
        file_bytes = a.save(path)
        restored = hue.Tensor.load(path, max_bytes=file_bytes)
        assert restored.rgbd_bytes == a.rgbd_bytes and restored.shape == a.shape
    native = hue.native_info()
    native["path"] = environment_path(native["path"], prefix)
    record = {"schema_version": 1, "python": platform.python_version(), "isolated": bool(sys.flags.isolated),
              "dependencies": dependencies, "module": environment_path(module, prefix), "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
              "native": native, "checks": {"isolated_package": True, "no_tensor_runtime": True,
              "native_color_encode_decode": True, "actual_D_refinement": True, "native_matmul": True,
              "native_shapes": True, "native_file_roundtrip": True}, "tokens": list(a.rgbd_bytes),
              "matmul_phases": list(expected), "payload_bytes": a.storage_nbytes, "file_bytes": file_bytes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
