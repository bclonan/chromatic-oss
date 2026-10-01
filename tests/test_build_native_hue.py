"""Check clean-checkout packaging and portable native build evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from scripts import build_native_hue as builder
from scripts.verify_native_hue_release import validate_native_commands


def source_workspace(tmp_path, monkeypatch):
    original = builder.ROOT
    for relative in ("chromatensor/hue.py", "native/hue_tensor/Cargo.toml", "native/hue_tensor/src/lib.rs", "LICENSE", "THIRD_PARTY_NOTICES.md", "COPYRIGHT"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / relative, target)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "CRATE", tmp_path / "native/hue_tensor/Cargo.toml")
    shutil.copytree(original / "native/rust-notices", tmp_path / "native/rust-notices")


def test_builder_creates_clean_receipt_and_binds_exact_wheel(tmp_path, monkeypatch):
    source_workspace(tmp_path, monkeypatch)
    actual_run = subprocess.run

    def compile_stub(args, **kwargs):
        if args[0] != "cargo":
            return actual_run(args, **kwargs)
        assert Path(kwargs["cwd"]) == tmp_path
        assert not Path(args[-1]).is_absolute()
        flags = kwargs["env"]["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        assert any(flag.startswith("--remap-path-prefix=") for flag in flags)
        directory = Path(kwargs["env"]["CARGO_TARGET_DIR"]) / "release"
        directory.mkdir(parents=True, exist_ok=True)
        windows = builder.os.name == "nt"
        (directory / ("hue_tensor.dll" if windows else "libhue_tensor.so")).write_bytes(b"test-library")
        (directory / ("hue-tensor-cli.exe" if windows else "hue-tensor-cli")).write_bytes(b"test-cli")
        stdout = "test result: ok. 22 passed; 0 failed; 0 ignored\n" if args[1] == "test" else ""
        return subprocess.CompletedProcess(args, 0, stdout, "Compiling source at " + str(tmp_path))

    monkeypatch.setattr(builder.subprocess, "run", compile_stub)
    assert not (tmp_path / "results").exists()
    assert builder.main(["--test", "--wheel-dir", str(tmp_path / "wheels")]) == 0
    receipt = next((tmp_path / "results").glob("native-hue-build-*.json"))
    evidence = json.loads(receipt.read_text(encoding="utf-8"))
    validate_native_commands(evidence, tmp_path)
    assert str(tmp_path) not in json.dumps(evidence["commands"])
    assert all(command["returncode"] == 0 for command in evidence["commands"])
    with zipfile.ZipFile(tmp_path / evidence["wheel"]["path"]) as archive:
        binary = next(name for name in archive.namelist() if name.endswith((".dll", ".so")))
        assert archive.read(binary) == b"test-library"
        assert archive.read("chromatic_hue/__init__.py") == (tmp_path / "chromatensor/hue.py").read_bytes()
        metadata = archive.read("chromatic_hue-0.1.0.dist-info/METADATA").decode("utf-8")
        assert "License-Expression: AGPL-3.0-only" in metadata
        assert "Author: Brad Clonan" in metadata
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "COPYRIGHT"):
            assert archive.read("chromatic_hue-0.1.0.dist-info/licenses/" + name) == (tmp_path / name).read_bytes()
        _, notices = builder.rust_notices(tmp_path)
        for path in notices:
            name = path.relative_to(tmp_path / "native/rust-notices").as_posix()
            assert archive.read("chromatic_hue-0.1.0.dist-info/licenses/rust/" + name) == path.read_bytes()


def test_builder_rejects_attribution_for_a_different_rust_compiler():
    with pytest.raises(ValueError, match="compiler differs"):
        builder.validate_toolchain("rustc 1.97.0", {"rustc": "rustc 1.96.0"})


def test_builder_rejects_outside_output_before_running_cargo(tmp_path, monkeypatch):
    source_workspace(tmp_path / "workspace", monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("an invalid destination started a build")
    monkeypatch.setattr(builder, "command", forbidden)
    with pytest.raises(SystemExit) as error:
        builder.main(["--wheel-dir", str(tmp_path / "outside")])
    assert error.value.code == 2
    assert not (tmp_path / "outside").exists()


def test_build_receipt_redacts_workspace_and_account_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path / "workspace")
    text = builder.public_text(str(builder.ROOT / "native") + " " + str(Path.home() / "toolchain"))
    assert "<workspace>" in text and "<user-home>" in text
    assert str(builder.ROOT) not in text and str(Path.home()) not in text


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_binary_privacy_gate_rejects_embedded_local_build_paths(tmp_path, monkeypatch, encoding):
    monkeypatch.setattr(builder, "ROOT", tmp_path / "workspace")
    binary = tmp_path / "library.bin"
    binary.write_bytes(b"prefix" + str(builder.ROOT).encode(encoding) + b"suffix")
    with pytest.raises(ValueError, match="private build path"):
        builder.check_binary_privacy(binary)


def test_windows_build_uses_encoded_remapping_and_relative_pdb_name():
    flags = builder.rust_flags(windows=True)
    assert "link-arg=/PDBALTPATH:hue_tensor.pdb" in flags
    assert builder.rust_flags(container=True) == ["--remap-path-prefix=/workspace=workspace"]
