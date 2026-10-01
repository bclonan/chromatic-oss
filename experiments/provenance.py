"""Bind measured runs to source files, configuration and stored artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("chromatensor", "experiments", "dashboard", "scripts", "tests", "requirements", ".github", "native")
SOURCE_SUFFIXES = {".py", ".js", ".css", ".html", ".txt", ".lock", ".yml", ".yaml", ".rs", ".toml", ".h"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(root=ROOT):
    root = Path(root).resolve()
    paths = [root / name for name in ("pyproject.toml", "AGENTS.md", "LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md")]
    paths.extend(path for path in (root / "native/rust-notices").rglob("*") if path.is_file())
    for name in SOURCE_DIRECTORIES:
        paths.extend(path for path in (root / name).rglob("*")
                     if path.is_file() and path.suffix in SOURCE_SUFFIXES
                     and "__pycache__" not in path.parts and "target" not in path.parts)
    files = {path.relative_to(root).as_posix(): file_hash(path)
             for path in sorted(set(paths)) if path.is_file()}
    if not files:
        raise ValueError("No executable source files were found for provenance")
    return {"files": files, "sha256": canonical_hash(files)}


def git_identity(root=ROOT):
    def command(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        commit = command("rev-parse", "HEAD")
        status = command("status", "--porcelain", "--untracked-files=normal").splitlines()
        return {"available": True, "commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None, "status": []}


def dependency_versions():
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "numpy", "matplotlib", "psutil", "pytest"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def artifact_manifest(paths, root=ROOT):
    root = Path(root).resolve()
    result = []
    resolved = set()
    for path in paths:
        target = Path(path)
        target = (target if target.is_absolute() else root / target).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Evidence artifact is outside the workspace")
        if target in resolved:
            continue
        if not target.is_file():
            raise ValueError("Missing evidence artifact: " + str(target))
        resolved.add(target)
        result.append({"path": target.relative_to(root).as_posix(),
                       "sha256": file_hash(target), "bytes": target.stat().st_size})
    return sorted(result, key=lambda item: item["path"])


def capture_provenance(config, root=ROOT, artifact_paths=()):
    return {"schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": source_manifest(root), "git": git_identity(root),
            "dependencies": dependency_versions(), "config_sha256": canonical_hash(config),
            "artifacts": artifact_manifest(artifact_paths, root)}


def _validate_git_context(context):
    """Keep the original Git context without treating its commit as source identity."""
    fields = {"available", "commit", "dirty", "status"}
    if not isinstance(context, dict) or set(context) != fields:
        raise ValueError("Evidence Git context is missing or malformed")
    if type(context["available"]) is not bool:
        raise ValueError("Evidence Git availability is invalid")
    if not context["available"]:
        if context["commit"] is not None or context["dirty"] is not None or context["status"] != []:
            raise ValueError("Unavailable evidence Git context must contain null identity")
        return
    commit, dirty, status = context["commit"], context["dirty"], context["status"]
    if (not isinstance(commit, str)
            or re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", commit) is None):
        raise ValueError("Evidence Git commit must be a full hexadecimal identity")
    if (type(dirty) is not bool or not isinstance(status, list)
            or any(not isinstance(row, str) for row in status) or dirty != bool(status)):
        raise ValueError("Evidence Git dirty status is malformed or inconsistent")


def verify_provenance(evidence, config, root=ROOT, *, required_artifacts=()):
    """Verify exact file identities while retaining historical Git context unchanged."""
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        raise ValueError("Missing current-source provenance")
    if not isinstance(config, dict):
        raise ValueError("Evidence configuration is missing")
    if evidence.get("source") != source_manifest(root):
        raise ValueError("Evidence source differs from current source")
    if evidence.get("config_sha256") != canonical_hash(config):
        raise ValueError("Evidence configuration hash differs")
    _validate_git_context(evidence.get("git"))
    if not isinstance(evidence.get("dependencies"), dict) or not evidence["dependencies"].get("python"):
        raise ValueError("Evidence dependency versions are missing")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Evidence artifact manifest is missing")
    names = [item.get("path") for item in artifacts if isinstance(item, dict)]
    if len(names) != len(artifacts) or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("Evidence artifact paths are invalid or duplicated")
    actual = artifact_manifest(names, root)
    if artifacts != actual:
        raise ValueError("Evidence artifact checksum or byte count differs")
    required = {item["path"] for item in artifact_manifest(required_artifacts, root)}
    if not required.issubset(names):
        raise ValueError("Required artifacts are absent from evidence")
    return True


def unique_run_id(prefix="run"):
    return prefix + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12]


def result_artifact_paths(result, root=ROOT):
    paths = [result[name] for name in ("report_path", "figure_path") if result.get(name)]
    for record in result.get("experiments", []):
        location = record.get("artifact_path")
        if location:
            paths.append(location)
            path = Path(location)
            if path.suffix == ".npz":
                paths.append(path.with_suffix(".json"))
        probe = record.get("checkpoint_probe", {})
        if isinstance(probe, dict) and probe.get("path"):
            paths.append(probe["path"])
    # Include every auxiliary checkpoint sharing a measured model directory.
    for parent in {str(Path(path).parent) for path in paths}:
        directory = Path(parent)
        directory = directory if directory.is_absolute() else Path(root) / directory
        if directory.is_dir():
            paths.extend(path for path in directory.iterdir() if path.is_file()
                         and path.suffix in {".ctcheckpoint", ".npz", ".json", ".pt", ".png", ".jpg", ".jpeg", ".md"})
    storage = result.get("file_storage", [])
    storage_root = Path(root) / "artifacts" / "storage"
    for item in storage:
        location = item.get("artifact_path")
        if location:
            paths.append(location)
        elif (storage_root / item.get("format", "")).is_file():
            paths.append(storage_root / item["format"])
        else:
            matches = list((Path(root) / "artifacts").glob("*/storage/" + item.get("format", "")))
            if len(matches) != 1:
                raise ValueError("Storage artifact location is absent or ambiguous")
            paths.append(matches[0])
    for path in tuple(paths):
        target = Path(path)
        target = target if target.is_absolute() else Path(root) / target
        if target.suffix == ".bin" and target.with_suffix(".json").is_file():
            paths.append(target.with_suffix(".json"))
    return paths


def normalize_result_paths(result, root=ROOT):
    """Keep saved references portable without changing source or config identity."""
    root = Path(root).resolve()
    fields = {"artifact_path", "report_path", "figure_path", "index_artifact", "prediction_path",
              "payload_path", "metadata_path", "installed_report", "dependency_evidence"}
    def visit(value, parent=None):
        if isinstance(value, list):
            for item in value:
                visit(item, parent)
        elif isinstance(value, dict):
            for key, item in tuple(value.items()):
                if key in {"config", "provenance", "environment"}:
                    continue
                if isinstance(item, str) and (key in fields or key == "path" and parent == "checkpoint_probe"):
                    path = Path(item)
                    if path.is_absolute():
                        target = path.resolve()
                        if not target.is_relative_to(root):
                            raise ValueError("Result artifact is outside the workspace")
                        value[key] = target.relative_to(root).as_posix()
                else:
                    visit(item, key)
    visit(result)
    return result


def bind_result_evidence(result, root=ROOT, output_path=None):
    """Called by the atomic writer for run records, including incomplete snapshots."""
    evidence = result.get("provenance")
    if evidence is None:
        if result.get("experiments"):
            raise ValueError("Measured records lack source provenance; start a new run")
        evidence = capture_provenance(result["config"], root)
        result["provenance"] = evidence
    else:
        verify_provenance(evidence, result["config"], root)
    if result.get("status") == "complete":
        normalize_result_paths(result, root)
        if output_path is not None:
            output = Path(output_path).resolve()
            canonical = (Path(root) / "results" / "latest.json").resolve()
            storage = Path(root) / "artifacts" / ("storage" if output == canonical else output.stem + "/storage")
            for row in result.get("file_storage", []):
                if not row.get("artifact_path"):
                    location = storage / row["format"]
                    if not location.is_file():
                        raise ValueError("Missing storage artifact for this run")
                    row["artifact_path"] = location.relative_to(Path(root).resolve()).as_posix()
        evidence["artifacts"] = artifact_manifest(result_artifact_paths(result, root), root)
    return result
