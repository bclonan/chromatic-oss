"""Dependency jobs copy public inputs and distinguish optional research replay."""
from pathlib import Path

import pytest

from scripts import verify_dependencies as matrix


def test_matrix_copy_includes_native_licenses_and_explorer_without_caches(tmp_path):
    root, destination = tmp_path / "source", tmp_path / "container"
    required = ("LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
                "native/hue_tensor/src/lib.rs", "native/hue_tensor/Cargo.toml",
                "docs/architecture/explorer.html", "tests/test_native_hue.py",
                "chromatensor/_native/linux-x86_64/libhue_tensor.so")
    excluded = ("native/hue_tensor/target/debug/cache.rs", "chromatensor/__pycache__/cache.py",
                ".release-env/private.txt", "private.txt")
    for name in required + excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    copied = matrix.copy_source(destination, root)
    assert set(required) <= set(copied)
    assert not set(excluded) & set(copied)
    for name in required:
        assert (destination / name).read_bytes() == (root / name).read_bytes()
    with pytest.raises(ValueError, match="outside"):
        matrix.copy_source(root / "copy", root)


def test_toolkit_install_profile_does_not_require_missing_research_models(tmp_path):
    command = matrix.install_command("python", None, tmp_path / "receipt.json")
    assert "--results" not in command
    research = matrix.install_command("python", tmp_path / "research.json", tmp_path / "receipt.json")
    assert research[research.index("--results") + 1] == tmp_path / "research.json"


def test_subset_selection_cannot_imply_the_full_supported_matrix():
    windows, linux = matrix.selected_environments(["python312", "linux312"])
    assert windows == {"python312": "current"}
    assert linux == {"linux312": "3.12"}
    assert (windows, linux) != matrix.selected_environments(None)
    for labels in ([], ["unknown"]):
        with pytest.raises(ValueError, match="nonempty"):
            matrix.selected_environments(labels)


def test_release_packages_do_not_change_recorded_dependency_lock():
    frozen = "numpy==2.4.6\nchromatensor==0.2.0\nchromatic-hue==0.1.0\n"
    assert matrix.freeze_without_release(frozen) == "numpy==2.4.6\n"


def test_windows_setup_resolves_missing_interpreters_and_syncs_public_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(matrix, "run", calls.append)
    matrix.prepare_windows({"minimum": "minimum", "python314": "py314"})
    assert calls[0] == ["uv", "venv", "--python", "3.11", tmp_path / ".release-env/minimum"]
    assert calls[2] == ["uv", "venv", "--python", "3.14", tmp_path / ".release-env/py314"]
    assert calls[1][0:3] == ["uv", "pip", "sync"]
    assert tmp_path / "requirements/minimum-cpu.lock" in calls[1]
