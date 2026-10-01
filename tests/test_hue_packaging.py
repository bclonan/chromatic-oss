"""Native binary discovery keeps the research wheel platform independent."""
from types import SimpleNamespace

import pytest

from chromatensor import hue
from scripts import run_native_hue_package_checks as matrix
from scripts.verify_installed_hue import environment_path


def package_tree(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    return root


def fake_binary(root):
    directory, filename = hue._platform_directory()
    binary = root / "_native" / directory / filename
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary-location-fixture")
    return binary


def test_library_discovery_prefers_built_checkout(tmp_path, monkeypatch):
    checkout = package_tree(tmp_path, "chromatensor")
    native = fake_binary(checkout)
    monkeypatch.setattr(hue, "__file__", str(checkout / "__init__.py"))
    def forbidden(name):
        raise AssertionError("a built checkout searched another package")
    monkeypatch.setattr(hue, "find_spec", forbidden)
    assert hue._library_path() == native


def test_research_install_uses_standalone_platform_library(tmp_path, monkeypatch):
    research = package_tree(tmp_path, "chromatensor")
    standalone = package_tree(tmp_path, "chromatic_hue")
    native = fake_binary(standalone)
    monkeypatch.setattr(hue, "__file__", str(research / "__init__.py"))
    monkeypatch.setattr(hue, "find_spec", lambda name: SimpleNamespace(origin=str(standalone / "__init__.py")))
    assert hue._library_path() == native


@pytest.mark.parametrize("spec", [None, SimpleNamespace(origin=None)])
def test_missing_platform_library_has_install_instruction(tmp_path, monkeypatch, spec):
    research = package_tree(tmp_path, "chromatensor")
    monkeypatch.setattr(hue, "__file__", str(research / "__init__.py"))
    monkeypatch.setattr(hue, "find_spec", lambda name: spec)
    with pytest.raises(RuntimeError, match="chromatic-hue platform wheel"):
        hue._library_path()


def test_native_matrix_resolves_python_without_private_prepared_environments(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "ROOT", tmp_path)
    assert matrix.python_source("python311", "py311") == "3.11"
    assert matrix.python_source("python314", "py314") == "3.14"
    prepared = tmp_path / ".release-env/current/Scripts/python.exe"
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(b"interpreter-fixture")
    assert matrix.python_source("python312", "current") == prepared


def test_installed_receipt_paths_are_relative_and_require_environment_ownership(tmp_path):
    prefix = tmp_path / "environment"
    path = prefix / "Lib/site-packages/chromatic_hue/__init__.py"
    assert environment_path(path, prefix) == "<environment>/Lib/site-packages/chromatic_hue/__init__.py"
    with pytest.raises(ValueError):
        environment_path(tmp_path / "outside/__init__.py", prefix)
