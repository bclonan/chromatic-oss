"""Reject detached native build evidence and incomplete installed-wheel checks."""
import copy

import pytest

from experiments.provenance import capture_provenance, source_manifest, verify_provenance
from scripts.verify_native_hue_release import PACKAGE_CHECKS, validate_package_row, validate_full_collection


@pytest.mark.parametrize("filename", ["src/lib.rs", "src/ffi.rs", "Cargo.toml", "Cargo.lock", "include/hue_tensor.h"])
def test_rust_and_c_contract_mutations_invalidate_evidence(tmp_path, filename):
    source = tmp_path / "native/hue_tensor" / filename
    source.parent.mkdir(parents=True)
    source.write_text("original\n", encoding="utf-8")
    evidence = capture_provenance({"profile": "native_hue"}, tmp_path)
    source.write_text("modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source differs"):
        verify_provenance(evidence, {"profile": "native_hue"}, tmp_path)


@pytest.mark.parametrize("relative", ["LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md",
                                    "native/rust-notices/manifest.json", "native/rust-notices/licenses/LICENSE-MIT.txt"])
def test_license_and_toolchain_notice_mutations_invalidate_evidence(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original attribution", encoding="utf-8")
    config = {"profile": "native_hue"}
    evidence = capture_provenance(config, tmp_path)
    path.write_text("modified attribution", encoding="utf-8")
    with pytest.raises(ValueError, match="source differs"):
        verify_provenance(evidence, config, tmp_path)


def test_rust_target_cache_is_not_source_identity(tmp_path):
    source = tmp_path / "native/hue_tensor/src/lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("actual source\n", encoding="utf-8")
    expected = source_manifest(tmp_path)
    cache = tmp_path / "native/hue_tensor/target/debug/build/generated.rs"
    cache.parent.mkdir(parents=True)
    cache.write_text("compiler cache\n", encoding="utf-8")
    assert source_manifest(tmp_path) == expected


def package_row():
    row = {"environment": "python311", "python": "3.11.15", "dependencies": [], "isolated": True,
           "checks": dict.fromkeys(PACKAGE_CHECKS, True), "wheel_sha256": "wheel",
           "module_sha256": "binding", "native": {"sha256": "binary", "abi_version": 1, "phase_modulus": 391680}}
    actual = {key: copy.deepcopy(value) for key, value in row.items() if key != "environment" and key != "wheel_sha256"}
    index = {"wheel": ("binary", "binding", "windows-x86_64")}
    return row, actual, index


def test_native_package_row_requires_matching_executed_receipt():
    row, actual, index = package_row()
    validate_package_row(row, actual, index)
    actual["native"]["sha256"] = "substitution"
    with pytest.raises(ValueError, match="receipt disagrees"):
        validate_package_row(row, actual, index)


@pytest.mark.parametrize("actual", [{}, {"dependencies": []}])
def test_native_package_row_rejects_partial_raw_receipt(actual):
    row, _, index = package_row()
    with pytest.raises(ValueError, match="receipt disagrees"):
        validate_package_row(row, actual, index)


def test_complete_collection_rejects_reduced_test_report(tmp_path):
    xml = tmp_path / "suite.xml"
    xml.write_text('<testsuites><testsuite tests="1"><testcase classname="tests.test_native_hue" name="test_colors[red]"/></testsuite></testsuites>', encoding="utf-8")
    command = ["<interpreter>", "-m", "pytest", "tests", "-q", "--junitxml=suite.xml"]
    evidence = {"config": {"command": command, "xml": "suite.xml"}}
    collection = {"command": ["<interpreter>", "-m", "pytest", "tests", "--collect-only", "-q"],
                  "nodeids": ["tests/test_native_hue.py::test_colors[red]"]}
    validate_full_collection(evidence, collection, xml)
    collection["nodeids"].append("tests/test_native_hue.py::test_colors[blue]")
    with pytest.raises(ValueError, match="Executed tests differ"):
        validate_full_collection(evidence, collection, xml)


@pytest.mark.parametrize("change", ["empty_checks", "missing_check", "truthy_check", "python", "platform",
                                  "dependencies", "isolation", "binding", "binary", "abi", "modulus"])
def test_native_package_row_rejects_incomplete_or_substituted_contract(change):
    row, actual, index = package_row()
    if change == "empty_checks":
        row["checks"] = {}
    elif change == "missing_check":
        row["checks"].pop("native_matmul")
    elif change == "truthy_check":
        row["checks"]["native_matmul"] = 1
    elif change == "python":
        row["python"] = "3.12.0"
    elif change == "platform":
        index["wheel"] = ("binary", "binding", "linux-x86_64")
    elif change == "dependencies":
        row["dependencies"] = ["torch"]
    elif change == "isolation":
        row["isolated"] = False
    elif change in {"binding", "binary"}:
        if change == "binding":
            row["module_sha256"] = "substitution"
        else:
            row["native"]["sha256"] = "substitution"
    elif change == "abi":
        row["native"]["abi_version"] = 2
    else:
        row["native"]["phase_modulus"] = 65536
    with pytest.raises(ValueError, match="receipt disagrees"):
        validate_package_row(row, actual, index)
