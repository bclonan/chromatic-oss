"""Acceptance must fail when its execution and provenance evidence are incomplete."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from experiments.provenance import (artifact_manifest, bind_result_evidence, capture_provenance,
                                    file_hash, git_identity, normalize_result_paths, source_manifest, verify_provenance)
from experiments.reporting import paired_comparisons
from experiments.verification import (PRECISION_CODECS, REPRESENTATIONS, TASKS, expected_training_tuples,
    verify_comparisons, verify_dependency_matrix, verify_precision, verify_release, verify_test_report, verify_training_coverage)


@pytest.fixture
def evidence_root(tmp_path):
    for name in ("scripts", "tests", "requirements", ".github/workflows"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "scripts/source.py").write_text("value = 1\n")
    (tmp_path / "tests/test_required.py").write_text("def test_required(): pass\n")
    (tmp_path / "requirements/minimum.txt").write_text("torch==2.14.1\n")
    (tmp_path / ".github/workflows/test.yml").write_text("name: test\n")
    return tmp_path


def test_source_identity_includes_untracked_scripts_locks_and_workflows(evidence_root):
    files = source_manifest(evidence_root)["files"]
    assert {"scripts/source.py", "requirements/minimum.txt", ".github/workflows/test.yml"}.issubset(files)
    assert len(source_manifest(evidence_root)["sha256"]) == 64


def test_receipt_commit_and_dirty_change_preserve_exact_source_evidence(evidence_root, monkeypatch):
    def git(*args):
        return subprocess.run(["git", "-C", str(evidence_root), "-c", "user.name=Evidence test",
                               "-c", "user.email=evidence@example.invalid", *args],
                              check=True, capture_output=True, text=True).stdout.strip()

    git_available = shutil.which("git") is not None
    git_context = {"available": True, "commit": "a" * 40, "dirty": False, "status": []}
    if git_available:
        git("init")
    else:
        # Minimal CPU images omit Git. Exercise the same metadata transitions
        # while checking real source and artifact bytes without skipping.
        monkeypatch.setattr("experiments.provenance.git_identity", lambda root: copy.deepcopy(git_context))

    def current_git():
        return git_identity(evidence_root) if git_available else copy.deepcopy(git_context)

    artifact = evidence_root / "checkpoint.bin"
    artifact.write_bytes(b"measured bytes")
    if git_available:
        git("add", ".")
        git("commit", "-m", "Source used for measurement")
    config = {"seed": 7}
    evidence = capture_provenance(config, evidence_root, [artifact])
    original = copy.deepcopy(evidence)
    assert evidence["git"]["dirty"] is False
    receipt = evidence_root / "results/receipt.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps(evidence))
    if not git_available:
        git_context.update(dirty=True, status=["?? results/receipt.json"])
    assert current_git()["dirty"] is True
    assert verify_provenance(evidence, config, evidence_root, required_artifacts=[artifact])
    if git_available:
        git("add", "results/receipt.json")
        git("commit", "-m", "Store the measured receipt")
    else:
        git_context.update(commit="b" * 40, dirty=False, status=[])
    assert current_git()["commit"] != evidence["git"]["commit"]
    assert verify_provenance(evidence, config, evidence_root, required_artifacts=[artifact])
    assert evidence == original
    assert json.loads(receipt.read_text()) == original
    (evidence_root / "scripts/source.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="source differs"):
        verify_provenance(evidence, config, evidence_root, required_artifacts=[artifact])


@pytest.mark.parametrize("commit_length", [40, 64])
def test_historical_git_context_does_not_require_current_commit_object(evidence_root, commit_length):
    evidence = capture_provenance({"seed": 7}, evidence_root)
    evidence["git"] = {"available": True, "commit": "a" * commit_length,
                       "dirty": True, "status": [" M README.md"]}
    original = copy.deepcopy(evidence)
    assert verify_provenance(evidence, {"seed": 7}, evidence_root)
    assert evidence == original


@pytest.mark.parametrize("context", [
    None, {}, {"available": False},
    {"available": 1, "commit": "a" * 40, "dirty": False, "status": []},
    {"available": True, "commit": "a" * 39, "dirty": False, "status": []},
    {"available": True, "commit": "z" * 40, "dirty": False, "status": []},
    {"available": True, "commit": None, "dirty": False, "status": []},
    {"available": True, "commit": "a" * 40, "dirty": 0, "status": []},
    {"available": True, "commit": "a" * 40, "dirty": False, "status": ""},
    {"available": True, "commit": "a" * 40, "dirty": True, "status": [None]},
    {"available": True, "commit": "a" * 40, "dirty": False, "status": [" M README.md"]},
    {"available": True, "commit": "a" * 40, "dirty": True, "status": []},
    {"available": False, "commit": "a" * 40, "dirty": None, "status": []},
    {"available": False, "commit": None, "dirty": False, "status": []},
    {"available": False, "commit": None, "dirty": None, "status": [" M README.md"]},
    {"available": False, "commit": None, "dirty": None, "status": (), "extra": True},
])
def test_provenance_rejects_missing_or_malformed_historical_git_context(evidence_root, context):
    evidence = capture_provenance({"seed": 7}, evidence_root)
    evidence["git"] = context
    with pytest.raises(ValueError, match="Git"):
        verify_provenance(evidence, {"seed": 7}, evidence_root)


def test_unavailable_historical_git_context_remains_valid(evidence_root):
    evidence = capture_provenance({"seed": 7}, evidence_root)
    evidence["git"] = {"available": False, "commit": None, "dirty": None, "status": []}
    assert verify_provenance(evidence, {"seed": 7}, evidence_root)


@pytest.mark.parametrize("alter", ["source", "config", "artifact", "byte_count", "missing_artifact", "old_flag"])
def test_provenance_rejects_stale_missing_or_altered_evidence(evidence_root, alter):
    artifact = evidence_root / "checkpoint.bin"
    artifact.write_bytes(b"measured bytes")
    config = {"seed": 7}
    evidence = capture_provenance(config, evidence_root, [artifact])
    assert verify_provenance(evidence, config, evidence_root, required_artifacts=[artifact])
    evidence["git"] = {"available": True, "commit": "b" * 40,
                       "dirty": False, "status": []}
    assert verify_provenance(evidence, config, evidence_root, required_artifacts=[artifact])
    if alter == "source":
        (evidence_root / "scripts/source.py").write_text("value = 2\n")
    elif alter == "config":
        config["seed"] = 23
    elif alter == "artifact":
        artifact.write_bytes(b"different bytes")
    elif alter == "byte_count":
        evidence["artifacts"][0]["bytes"] += 1
    elif alter == "missing_artifact":
        evidence["artifacts"] = []
    else:
        evidence = {"verified": True}
    with pytest.raises(ValueError):
        verify_provenance(evidence, config, evidence_root, required_artifacts=[artifact])


def test_existing_complete_evidence_cannot_silently_rehash_changed_artifacts(evidence_root):
    artifact = evidence_root / "model.ctcheckpoint"
    artifact.write_bytes(b"original")
    config = {"seed": 7}
    result = {"config": config, "status": "complete", "experiments": [{"artifact_path": str(artifact)}],
              "provenance": capture_provenance(config, evidence_root)}
    bind_result_evidence(result, evidence_root)
    artifact.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="checksum"):
        bind_result_evidence(result, evidence_root)


def test_old_measured_records_cannot_receive_invented_current_source_identity(evidence_root):
    with pytest.raises(ValueError, match="lack source provenance"):
        bind_result_evidence({"config": {"seed": 7}, "status": "complete", "experiments": [{"id": "old"}]}, evidence_root)


def test_published_artifact_paths_are_portable_without_rewriting_config(evidence_root):
    checkpoint = evidence_root / "artifacts/model.ctcheckpoint"
    probe = checkpoint.with_suffix(".npz")
    result = {"config": {"original_path": str(checkpoint)}, "report_path": str(evidence_root / "report.md"),
              "experiments": [{"artifact_path": str(checkpoint), "checkpoint_probe": {"path": str(probe)}}],
              "file_storage": [{"artifact_path": str(evidence_root / "artifacts/storage.bin")}],
              "environment": {"original_path": str(checkpoint)}}
    normalize_result_paths(result, evidence_root)
    assert result["experiments"][0]["artifact_path"] == "artifacts/model.ctcheckpoint"
    assert result["experiments"][0]["checkpoint_probe"]["path"] == "artifacts/model.npz"
    assert result["file_storage"][0]["artifact_path"] == "artifacts/storage.bin"
    assert result["report_path"] == "report.md"
    assert result["config"]["original_path"] == str(checkpoint)


@pytest.mark.parametrize("kind", ["skipped", "failure", "error", "empty", "missing_module", "wrong_count"])
def test_required_test_gate_rejects_skips_and_incomplete_reports(evidence_root, kind):
    child = f'<{kind}/>' if kind in {"skipped", "failure", "error"} else ""
    case = '<testcase classname="tests.test_required" name="test_required">' + child + '</testcase>'
    if kind == "empty":
        case = ""
    if kind == "missing_module":
        case = case.replace("test_required", "test_other")
    path = evidence_root / "tests.xml"
    path.write_text('<testsuites><testsuite tests="' + ("2" if kind == "wrong_count" else "1") + '">' + case + '</testsuite></testsuites>')
    with pytest.raises(ValueError):
        verify_test_report(path, evidence_root)


def test_required_test_gate_checks_exact_xml_artifact_hash(evidence_root):
    path = evidence_root / "tests.xml"
    path.write_text('<testsuites><testsuite tests="1"><testcase classname="tests.test_required" name="test_required"/></testsuite></testsuites>')
    config = {"command": ["python", "-m", "pytest"], "xml": "tests.xml"}
    evidence = {"returncode": 0, "config": config, "provenance": capture_provenance(config, evidence_root, [path])}
    assert verify_test_report(path, evidence_root, evidence)["tests"] == 1
    path.write_text(path.read_text().replace("test_required\"/>", "test_altered\"/>"))
    with pytest.raises(ValueError, match="checksum"):
        verify_test_report(path, evidence_root, evidence)


@pytest.fixture
def canonical_record():
    config = {"profile": "pilot", "seeds": [7, 23], "tasks": sorted((TASKS - {"decoder"}) | {"precision", "operators"}),
              "representations": sorted(REPRESENTATIONS), "precision_samples": 1_000_000,
              "epochs": 6, "scalar_epochs": 24, "train_images": 12000, "validation_images": 2000,
              "language_windows": 2048, "language_validation_windows": 512}
    rows = []
    for task, representation, seed in sorted(expected_training_tuples(config)):
        metrics = {"validation_loss": .5, "initial_validation_loss": 1., "training_seconds": 1.,
                   "accuracy": .5, "perplexity": .5, "angular_mae_radians": .5}
        counts = {"scalar": (4096, 1024), "mnist": (12000, 2000), "cifar": (12000, 2000),
                  "language": (2048, 512), "periodic": (4096, 1024), "attention": (512, 128), "decoder": (8192, 4096)}
        data = {"sha256": "a" * 64, "train_examples": counts[task][0], "validation_examples": counts[task][1],
                "source": "torchvision.datasets.MNIST" if task == "mnist" else "torchvision.datasets.CIFAR10" if task == "cifar"
                else "Tiny Shakespeare, char-rnn repository" if task == "language" else "analytic",
                "corpus_sha256": "b" * 64}
        rows.append({"id": f"{task}-{representation}-{seed}", "task": task, "representation": representation,
                     "seed": seed, "metrics": metrics, "dataset": data,
                     "curves": [{"epoch": 0, "train_loss": None, "val_loss": 1.}, {"epoch": 1, "train_loss": .4, "val_loss": .5}],
                     "artifact_path": "model.ctcheckpoint"})
    precision = [{"distribution": distribution, "codec": codec, "samples": 1_000_000,
                  "mae": .1, "mse": .02, "max_error": .2, "storage_bytes": 1, "metadata_bytes": 1}
                 for distribution in ("uniform", "normal", "laplace", "bimodal") for codec in PRECISION_CODECS]
    return {"status": "complete", "config": config, "experiments": rows, "comparisons": paired_comparisons(rows),
            "precision": precision, "operators": [{"method": name} for name in ("decode_compute_encode", "lookup", "direct_integer_grid", "learned")]}


def test_canonical_coverage_is_explicit_conditions_not_a_magic_count(canonical_record):
    assert verify_training_coverage(canonical_record)["training_tuples"] == 95
    assert verify_comparisons(canonical_record)["comparisons"] > 0
    assert verify_precision(canonical_record)["precision_conditions"] == 92


@pytest.mark.parametrize("alter", ["missing_model", "duplicate_model", "no_training", "no_curves", "no_checkpoint", "smoke"])
def test_training_gate_rejects_missing_or_inference_only_training(canonical_record, alter):
    if alter == "missing_model":
        canonical_record["experiments"].pop()
    elif alter == "duplicate_model":
        canonical_record["experiments"][-1] = copy.deepcopy(canonical_record["experiments"][0])
    elif alter == "no_training":
        canonical_record["experiments"][0]["metrics"]["training_seconds"] = 0
    elif alter == "no_curves":
        canonical_record["experiments"][0]["curves"] = []
    elif alter == "no_checkpoint":
        canonical_record["experiments"][0]["artifact_path"] = None
    else:
        canonical_record["config"]["profile"] = "smoke"
    with pytest.raises(ValueError):
        verify_training_coverage(canonical_record)


@pytest.mark.parametrize("alter", ["no_comparisons", "empty_pairs", "wrong_delta", "wrong_dataset"])
def test_paired_gate_rejects_vacuous_and_inconsistent_controls(canonical_record, alter):
    if alter == "no_comparisons":
        canonical_record["comparisons"] = []
    elif alter == "empty_pairs":
        canonical_record["comparisons"][0]["pairs"] = []
    elif alter == "wrong_delta":
        canonical_record["comparisons"][0]["pairs"][0]["delta"] = 1
    else:
        canonical_record["comparisons"][0]["pairs"][0]["identical_dataset"] = False
    with pytest.raises(ValueError):
        verify_comparisons(canonical_record)


@pytest.mark.parametrize("alter", ["missing_codec", "missing_distribution", "small_count", "missing_operator"])
def test_precision_gate_rejects_missing_conditions(canonical_record, alter):
    if alter == "missing_codec":
        canonical_record["precision"].pop()
    elif alter == "missing_distribution":
        canonical_record["precision"] = [row for row in canonical_record["precision"] if row["distribution"] != "uniform"]
    elif alter == "small_count":
        canonical_record["precision"][0]["samples"] = 4096
    else:
        canonical_record["operators"].pop()
    with pytest.raises(ValueError):
        verify_precision(canonical_record)


def test_old_package_verified_flag_cannot_pass_release_gate(evidence_root):
    with pytest.raises(ValueError, match="provenance"):
        verify_release({"verified": True}, evidence_root)


@pytest.fixture
def release_record(evidence_root):
    package = evidence_root / "chromatensor"
    package.mkdir()
    (package / "cube.py").write_text("class ChromaCube: pass\n")
    wheel = evidence_root / "chromatensor-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("chromatensor/cube.py", (package / "cube.py").read_bytes())
        archive.writestr("chromatensor-0.2.0.dist-info/METADATA", "Name: chromatensor\nVersion: 0.2.0\n")
    package_assets = {name: digest for name, digest in source_manifest(evidence_root)["files"].items()
                      if name.startswith("chromatensor/")}
    dependency = evidence_root / "dependency.json"
    config = {"matrix": "tested minimum and supported operating systems"}
    labels = {"minimum": "3.11.9", "python311": "3.11.9", "python312": "3.12.9",
              "python313": "3.13.5", "python314": "3.14.0", "linux311": "3.11.9",
              "linux312": "3.12.9", "linux313": "3.13.5", "linux314": "3.14.0"}
    rows, artifacts = [], []
    directory = evidence_root / "artifacts/verification"
    directory.mkdir(parents=True)
    for label, python in labels.items():
        xml = directory / (label + "-tests.xml")
        xml.write_text('<testsuites><testsuite tests="1"><testcase classname="tests.test_required" name="test_required"/></testsuite></testsuites>')
        installed_path = directory / (label + "-installed.json")
        prefix = "<environment>"
        versions = {"chromatensor": "0.2.0", "torch": "2.14.1+cpu", "torchvision": "0.29.1+cpu", "numpy": "2.4.6"}
        replay = [{"record_id": "mnist-" + rep + "-7", "device": "cpu", "examples": 32, "prediction_close": True,
                   "repeats": 10, "batch_ms": 1., "maximum_delta": 0., "relative_tolerance": 1e-4,
                   "absolute_tolerance": 1e-5, "checkpoint_sha256": "a" * 64} for rep in ("fp32", "chroma8")]
        installed = {"python": python, "dependencies": versions, "isolated": True, "user_site_enabled": False,
                     "installed_assets": package_assets, "prefix": prefix,
                     "modules": {name: prefix + "/site-packages/" + name + "/__init__.py" for name in ("chromatensor", "experiments", "dashboard")},
                     "checks": {name: True for name in ("isolated_imports", "cube_inverse", "holographic_decode", "bundled_assets", "rgbd_tokens")},
                     "rgbd_encoding": {"tokens": [[0, 0, 0, 127], [128, 0, 0, 127], [64, 0, 0, 127], [0, 0, 0, 128]],
                        "decoded": [1., -1., 1.5, 2.], "dtype": "torch.uint8", "payload_bytes": 16,
                        "mutated_D_decode": 2., "persisted_tokens": [[0, 0, 0, 127], [128, 0, 0, 127], [64, 0, 0, 127], [0, 0, 0, 128]],
                        "neural_tokens": [[0, 0, 0, 127], [128, 0, 0, 127], [64, 0, 0, 127], [0, 0, 0, 128]], "neural_range_metadata_bytes": 0},
                     "deployed_inference": replay + [{"source_record_id": "mnist-fp32-7", "device": "cpu", "examples": 32, "finite": True}]}
        installed_path.write_text(json.dumps(installed))
        lock = evidence_root / "requirements" / (label + "-cpu.lock")
        lock.write_text("torch==2.14.1+cpu\ntorchvision==0.29.1+cpu\nnumpy==2.4.6\n")
        rows.append({"environment": label, "python": python, "dependencies": versions,
                     "tests": {"passed": 1, "cuda_only_skipped": 0, "failures": 0},
                     "installed_report": installed_path.relative_to(evidence_root).as_posix(),
                     "lock": lock.relative_to(evidence_root).as_posix(), "dependency_check": "All installed packages are compatible"})
        artifacts.extend((xml, installed_path, lock))
    dependency.write_text(json.dumps({"config": config, "checks": {name: True for name in
                         ("pip_check", "minimum_supported", "all_supported_python_minors", "clean_installed_wheel_inference")},
                         "environments": rows, "provenance": capture_provenance(config, evidence_root, artifacts)}))
    config = {"wheel": wheel.name}
    return {"config": config, "provenance": capture_provenance(config, evidence_root, [wheel, dependency]),
            "wheel": artifact_manifest([wheel], evidence_root)[0], "installed_assets": package_assets,
            "checks": {name: True for name in ("isolated_imports", "cube_inverse", "holographic_decode", "bundled_assets", "pip_check")},
            "dependency_evidence": dependency.name}


def test_release_gate_binds_exact_wheel_contents_and_installed_assets(evidence_root, release_record):
    assert verify_release(release_record, evidence_root)["package_assets"] == 1


def test_dependency_gate_rejects_historical_vulnerable_torch_floor(evidence_root, release_record):
    dependency_path = evidence_root / release_record["dependency_evidence"]
    report = json.loads(dependency_path.read_text())
    row = next(row for row in report["environments"] if row["environment"] == "minimum")
    row["dependencies"].update(torch="2.3.0+cpu", torchvision="0.18.0+cpu")
    installed_path = evidence_root / row["installed_report"]
    installed = json.loads(installed_path.read_text())
    installed["dependencies"] = row["dependencies"]
    installed_path.write_text(json.dumps(installed))
    (evidence_root / row["lock"]).write_text("torch==2.3.0+cpu\ntorchvision==0.18.0+cpu\nnumpy==2.4.6\n")
    paths = [item["path"] for item in report["provenance"]["artifacts"]]
    report["provenance"] = capture_provenance(report["config"], evidence_root, paths)
    with pytest.raises(ValueError, match="Minimum Torch versions"):
        verify_dependency_matrix(report, evidence_root, release_record["installed_assets"])


def rewrite_wheel_metadata(root, record, content):
    wheel = root / record["wheel"]["path"]
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    metadata = next(name for name in members if name.endswith(".dist-info/METADATA"))
    members[metadata] = content
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    record["wheel"] = artifact_manifest([wheel], root)[0]
    record["provenance"]["artifacts"] = artifact_manifest(
        [wheel, root / record["dependency_evidence"]], root)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_release_gate_accepts_standard_lf_and_crlf_metadata(evidence_root, release_record, newline):
    content = newline.join([b"Metadata-Version: 2.4", b"Name: chromatensor", b"Version: 0.2.0", b"", b"Description"])
    rewrite_wheel_metadata(evidence_root, release_record, content)
    assert verify_release(release_record, evidence_root)["package_assets"] == 1


@pytest.mark.parametrize("headers", [
    [b"Name: chromatensor", b"Version: 0.1.0"],
    [b"Name: chromatensor", b"Version: 0.2.0.dev1"],
    [b"Name: chromatensor", b"Version: 0.2.0", b"Version: 0.2.0"],
    [b"Name: chromatensor", b"Version: 0.2.0", b"Version: 0.1.0"],
    [b"Name: chromatensor"],
    [b"Name: other-package", b"Version: 0.2.0"],
    [b"Name: chromatensor", b"Name: chromatensor", b"Version: 0.2.0"],
])
def test_release_gate_rejects_wrong_missing_and_duplicate_metadata_headers(evidence_root, release_record, headers):
    # A valid-looking version in the description cannot replace a header.
    content = b"\r\n".join([b"Metadata-Version: 2.4", *headers, b"", b"Version: 0.2.0"])
    rewrite_wheel_metadata(evidence_root, release_record, content)
    with pytest.raises(ValueError, match="version|package name"):
        verify_release(release_record, evidence_root)


@pytest.mark.parametrize("alter", ["wheel_bytes", "wheel_code", "installed_assets", "missing_execution"])
def test_release_gate_rejects_tampered_wheel_and_incomplete_install(evidence_root, release_record, alter):
    wheel = evidence_root / release_record["wheel"]["path"]
    if alter == "wheel_bytes":
        wheel.write_bytes(wheel.read_bytes() + b"changed")
    elif alter == "wheel_code":
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("chromatensor/cube.py", "class WrongCube: pass\n")
            archive.writestr("chromatensor-0.2.0.dist-info/METADATA", "Version: 0.2.0\n")
        release_record["wheel"] = artifact_manifest([wheel], evidence_root)[0]
        release_record["provenance"]["artifacts"] = artifact_manifest([wheel, evidence_root / "dependency.json"], evidence_root)
    elif alter == "installed_assets":
        release_record["installed_assets"] = {}
    else:
        release_record["checks"]["cube_inverse"] = False
    with pytest.raises(ValueError):
        verify_release(release_record, evidence_root)


@pytest.mark.parametrize("alter", ["missing_environment", "empty_xml", "cpu_skip", "test_summary", "external_modules",
                                  "empty_inference", "lock_versions", "missing_artifact", "wrong_python",
                                  "missing_rgbd", "rgbd_palette_ids", "rgbd_wrong_D", "rgbd_cached_decode"])
def test_dependency_gate_rechecks_execution_artifacts(evidence_root, release_record, alter):
    dependency_path = evidence_root / release_record["dependency_evidence"]
    report = json.loads(dependency_path.read_text())
    row = report["environments"][0]
    installed_path = evidence_root / row["installed_report"]
    installed = json.loads(installed_path.read_text())
    xml = evidence_root / "artifacts/verification/minimum-tests.xml"
    if alter == "missing_environment":
        report["environments"].pop()
    elif alter == "empty_xml":
        xml.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    elif alter == "cpu_skip":
        xml.write_text('<testsuites><testsuite tests="1" skipped="1"><testcase classname="tests.test_required"><skipped message="Dataset not installed"/></testcase></testsuite></testsuites>')
        row["tests"] = {"passed": 0, "cuda_only_skipped": 1, "failures": 0}
    elif alter == "test_summary":
        row["tests"]["passed"] = 2
    elif alter == "external_modules":
        installed["modules"]["chromatensor"] = "C:/user-checkout/chromatensor/__init__.py"
    elif alter == "empty_inference":
        installed["deployed_inference"] = []
    elif alter == "lock_versions":
        (evidence_root / row["lock"]).write_text("torch==2.2.0+cpu\n")
    elif alter == "wrong_python":
        row["python"] = "3.10.9"
        installed["python"] = "3.10.9"
    elif alter == "missing_rgbd":
        del installed["rgbd_encoding"]
    elif alter == "rgbd_palette_ids":
        installed["rgbd_encoding"]["tokens"] = [0, 1, 2, 3]
    elif alter == "rgbd_wrong_D":
        installed["rgbd_encoding"]["tokens"][0][3] = 128
    elif alter == "rgbd_cached_decode":
        installed["rgbd_encoding"]["mutated_D_decode"] = 1.
    installed_path.write_text(json.dumps(installed))
    paths = [item["path"] for item in report["provenance"]["artifacts"]]
    report["provenance"] = capture_provenance(report["config"], evidence_root, paths)
    if alter == "missing_artifact":
        report["provenance"]["artifacts"] = []
    with pytest.raises(ValueError):
        verify_dependency_matrix(report, evidence_root, release_record["installed_assets"])
