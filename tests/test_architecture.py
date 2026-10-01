"""Checks for source drift, graph semantics and offline explorer wiring."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("architecture", ROOT / "scripts" / "architecture.py")
architecture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(architecture)


@pytest.fixture(scope="module")
def graph():
    return architecture.build_graph(ROOT)


def test_source_graph_covers_both_execution_paths_and_required_absences(graph):
    architecture.validate_graph(graph, ROOT)
    ids = {node["id"] for node in graph["nodes"]}
    assert {"tokens", "tensor", "operators", "ffi", "python", "header", "cli", "storage",
            "legacy-tensor", "codecs", "nn", "cubes", "browser", "http", "datasets", "provenance", "acceptance"} <= ids
    absent = {item["name"] for item in graph["integrations"] if item["status"] == "absent"}
    assert absent == {"Accounts and authentication", "Chat and messaging", "Payments and billing",
                      "Convex or cloud database", "Hosted inference providers"}
    assert graph["semantic_review"]["automatic_discovery_creates_edges"] is False
    assert graph["inventory"] and all("imports" in file for file in graph["inventory"])
    assert all(node["status"] == "implemented" for node in graph["nodes"] if node["group"] in {"native", "entry"})
    assert all(node["status"] == "experimental" for node in graph["nodes"] if node["group"] in {"legacy", "research"})
    assert {edge["type"] for edge in graph["edges"]} >= {"read", "write", "call", "http"}


def test_reference_resolver_rejects_removed_symbol_and_path_escape(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def original():\n    return 1\n", encoding="utf-8")
    reference = architecture.resolve_reference(tmp_path, {"path": "sample.py", "symbol": "original"})
    assert reference["line"] == 1 and reference["end_line"] == 2
    source.write_text("def renamed():\n    return 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected one Python symbol"):
        architecture.resolve_reference(tmp_path, {"path": "sample.py", "symbol": "original"})
    with pytest.raises(ValueError, match="escapes repository"):
        architecture.resolve_reference(tmp_path, {"path": "../sample.py"})


def test_rust_qualified_reference_distinguishes_overloaded_methods(graph):
    reference = architecture.resolve_reference(ROOT, {"path": "native/hue_tensor/src/lib.rs", "symbol": "Token::from_rgb"})
    assert "pub fn from_rgb(r: u8" in (ROOT / reference["path"]).read_text().splitlines()[reference["line"] - 1]
    with pytest.raises(ValueError, match="found 2"):
        architecture.resolve_reference(ROOT, {"path": "native/hue_tensor/src/lib.rs", "symbol": "from_rgb"})


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("line", 999999), ("url", "https://invalid.example/")])
def test_stored_source_reference_drift_is_rejected(graph, field, value):
    changed = copy.deepcopy(graph)
    changed["nodes"][0]["sources"][0][field] = value
    with pytest.raises(ValueError, match="Source reference drift"):
        architecture.validate_graph(changed, ROOT)


@pytest.mark.parametrize("field,value", [("target", "operators"), ("type", "data"), ("boundary", False), ("label", "Hosted inference")])
def test_existing_source_anchor_does_not_authorize_changed_semantic_relationship(graph, field, value):
    changed = copy.deepcopy(graph)
    relationship = next(item for item in changed["edges"] if item["id"] == "python-ffi")
    relationship[field] = value
    with pytest.raises(ValueError, match="Semantic relationship differs from review"):
        architecture.validate_graph(changed, ROOT)


def test_unknown_endpoint_and_duplicate_components_are_rejected(graph):
    changed = copy.deepcopy(graph)
    changed["edges"][0]["target"] = "absent-service"
    with pytest.raises(ValueError, match="Invalid semantic relationship"):
        architecture.validate_graph(changed, ROOT)
    changed = copy.deepcopy(graph)
    changed["nodes"].append(changed["nodes"][0])
    with pytest.raises(ValueError, match="Duplicate component"):
        architecture.validate_graph(changed, ROOT)


def test_discovered_import_does_not_become_a_semantic_edge(tmp_path):
    (tmp_path / "chromatensor").mkdir()
    (tmp_path / "chromatensor" / "sample.py").write_text("import imaginary_cloud_provider\n", encoding="utf-8")
    inventory = architecture.discovery(tmp_path)
    assert inventory[0]["imports"] == [{"module": "imaginary_cloud_provider", "line": 1}]
    assert not any(edge["target"] == "imaginary_cloud_provider" for edge in architecture.EDGES)


def test_generated_snapshot_is_exact_and_json_is_complete(graph):
    generated = architecture.generated_files(ROOT)
    assert json.loads(generated["graph.json"])["source_identity"] == graph["source_identity"]
    assert generated["graph.js"].startswith("window.CHROMATIC_ARCHITECTURE = ")
    snapshots = json.loads(generated["sources.js"].removeprefix("window.CHROMATIC_SOURCES = ").removesuffix(";\n"))
    for reference in architecture.all_references(graph):
        assert snapshots[reference["path"]]["text"] == (ROOT / reference["path"]).read_text(encoding="utf-8-sig")
        assert snapshots[reference["path"]]["sha256"] == reference["sha256"]


def test_offline_explorer_has_local_assets_text_equivalent_and_keyboard_controls():
    directory = ROOT / "docs" / "architecture"
    html = (directory / "index.html").read_text(encoding="utf-8")
    app = (directory / "app.js").read_text(encoding="utf-8")
    for asset in re.findall(r'(?:src|href)="([^"#]+)"', html):
        assert not asset.startswith(("https://", "http://", "//"))
        if not asset.startswith("../../"):
            assert (directory / asset).is_file()
    assert '<table>' in html and 'id="component-list"' in html and 'id="text-map"' in html
    assert 'aria-live="polite"' in html and 'tabindex="0"' in html
    assert all(key in app for key in ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "keydown", "pointermove"])
    assert "fetch(" not in app and "innerHTML" not in (directory / "source.js").read_text()


def test_build_check_detects_generated_data_tampering(tmp_path, monkeypatch):
    expected = {"graph.json": "{\"nodes\": []}\n", "graph.js": "fixture\n", "sources.js": "fixture\n"}
    output = tmp_path / architecture.OUTPUT
    output.mkdir(parents=True)
    for name, text in expected.items():
        (output / name).write_text(text, encoding="utf-8")
    (output / "graph.js").write_text("tampered\n", encoding="utf-8")
    monkeypatch.setattr(architecture, "generated_files", lambda root: expected)
    with pytest.raises(ValueError, match="Stale architecture output"):
        architecture.main(["--root", str(tmp_path), "--check"])
