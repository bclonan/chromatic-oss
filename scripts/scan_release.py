"""Check reviewed files and ZIP members without displaying sensitive values."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

VERSION = "1.0"
MAX_FILE = 256 * 1024 * 1024
MAX_EXPANDED = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 20000
MAX_DEPTH = 3
EXCLUDED_DIRS = {".git", ".venv", ".pytest-work", ".pytest_cache", "__pycache__", "node_modules", "target", "build", "dist", "data", "native-build-host", "native-build-linux"}
PRIVATE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]+(?:Users|home)[\\/]+[^\s\\/\"<>]+|/home/[\w.-]+|/Users/[\w.-]+)")
PATTERNS = {
    "private_user_path": PRIVATE_PATH,
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential_url": re.compile(r"(?i)(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?):\/\/[^/\s:@]+:[^/\s@]+@"),
    "hosted_token": re.compile(r"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})"),
}
SENSITIVE_NAMES = {".env", ".npmrc", ".pypirc", ".netrc", "credentials", "cookies.sqlite", "login data"}


def safe_location(location):
    return PRIVATE_PATH.sub("<private-path>", location)


def scan_bytes(data, location, *, depth=0, budget=None):
    """Inspect raw bytes and archive members without executing file contents."""
    findings, limits = [], []
    budget = {"expanded": 0, "members": 0} if budget is None else budget
    text = data.decode("utf-8", "replace")
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            # Rust's unchanged copyright inventory links the public home crate.
            # Preserve that URL without allowing general home-directory paths.
            prefix = text[max(0, match.start() - 24):match.start()]
            if kind == "private_user_path" and prefix.endswith("https://crates.io/crates") and re.fullmatch(r"/home/[0-9.]+", match.group()):
                continue
            findings.append({"kind": kind, "path": safe_location(location), "line": text.count("\n", 0, match.start()) + 1})
    stream = io.BytesIO(data)
    if zipfile.is_zipfile(stream):
        if depth >= MAX_DEPTH:
            return findings, [{"path": safe_location(location), "reason": "archive depth limit"}]
        try:
            with zipfile.ZipFile(stream) as archive:
                infos = archive.infolist()
                budget["members"] += len(infos)
                if budget["members"] > MAX_MEMBERS:
                    return findings, [{"path": safe_location(location), "reason": "archive member limit"}]
                for info in infos:
                    if info.is_dir():
                        continue
                    member_location = location + "!" + info.filename
                    parts = PurePosixPath(info.filename.replace("\\", "/")).parts
                    if PurePosixPath(info.filename).is_absolute() or ".." in parts or ".git" in parts:
                        findings.append({"kind": "unsafe_archive_member", "path": safe_location(member_location), "line": 0})
                    if any(part.lower() in SENSITIVE_NAMES for part in parts) or any(part.startswith(".env.") and part != ".env.example" for part in parts):
                        findings.append({"kind": "private_configuration_file", "path": safe_location(member_location), "line": 0})
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        findings.append({"kind": "symlink_requires_review", "path": safe_location(member_location), "line": 0})
                    budget["expanded"] += info.file_size
                    if info.file_size > MAX_FILE or budget["expanded"] > MAX_EXPANDED:
                        limits.append({"path": safe_location(member_location), "reason": "archive expansion limit"})
                        continue
                    if info.flag_bits & 1:
                        limits.append({"path": safe_location(member_location), "reason": "encrypted archive member"})
                        continue
                    nested_findings, nested_limits = scan_bytes(archive.read(info), member_location, depth=depth + 1, budget=budget)
                    findings.extend(nested_findings)
                    limits.extend(nested_limits)
        except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError):
            limits.append({"path": safe_location(location), "reason": "unreadable archive"})
    return findings, limits


def selected_paths(root, manifest=None):
    if manifest:
        document = json.loads(Path(manifest).read_text(encoding="utf8"))
        items = document["files"] if isinstance(document, dict) else document
        paths = [item["path"] if isinstance(item, dict) else item for item in items]
        mode = "explicit manifest"
    elif (root / ".git").is_dir():
        process = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
        paths = process.stdout.decode("utf8").strip("\0").split("\0")
        mode = "tracked Git files"
    else:
        paths = []
        for directory, dirs, filenames in __import__("os").walk(root):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.endswith(".egg-info") and not d.startswith(".venv")]
            paths.extend((Path(directory) / name).relative_to(root).as_posix() for name in filenames)
        mode = "working files with explicit local-directory exclusions"
    return sorted({p for p in paths if p}), mode


def scan_tree(root, manifest=None):
    root = Path(root).resolve()
    paths, mode = selected_paths(root, manifest)
    findings, limits, scanned, total = [], [], 0, 0
    for relative in paths:
        parts = PurePosixPath(relative.replace("\\", "/")).parts
        path = root / relative
        if PurePosixPath(relative).is_absolute() or ".." in parts or not path.resolve().is_relative_to(root):
            findings.append({"kind": "invalid_manifest_path", "path": safe_location(relative), "line": 0})
            continue
        if ".git" in parts or any(part.lower() in SENSITIVE_NAMES for part in parts) or path.name.startswith(".env.") and path.name != ".env.example":
            findings.append({"kind": "private_configuration_file", "path": safe_location(relative), "line": 0})
        if path.is_symlink():
            findings.append({"kind": "symlink_requires_review", "path": safe_location(relative), "line": 0})
            continue
        if not path.is_file():
            limits.append({"path": safe_location(relative), "reason": "missing selected file"})
            continue
        size = path.stat().st_size
        if size > MAX_FILE:
            limits.append({"path": safe_location(relative), "reason": "file size limit"})
            continue
        nested_findings, nested_limits = scan_bytes(path.read_bytes(), relative)
        findings.extend(nested_findings)
        limits.extend(nested_limits)
        scanned += 1
        total += size
    return {"scanner": "ChromaTensor release privacy scan", "version": VERSION, "selection": mode,
            "status": "FAIL" if findings or limits else "PASS", "files_scanned": scanned, "bytes_scanned": total,
            "finding_counts": dict(Counter(f["kind"] for f in findings)), "findings": findings, "limits": limits,
            "working_scan_excluded_directories": sorted(EXCLUDED_DIRS) if mode.startswith("working") else [],
            "limitations": ["Pattern checks cannot prove that no sensitive information exists.", "No image OCR, malware analysis, provider credential validation, or legal determination.", "ZIP members are inspected as bytes, never loaded as pickle, executed, or extracted.", "Other archive formats require a separate scanner such as Gitleaks."]}


def license_metadata(root, lock=None):
    names = set()
    files = [Path(lock)] if lock else sorted((Path(root) / "requirements").glob("*"))
    for path in files:
        for line in path.read_text(encoding="utf8").splitlines():
            match = re.match(r"([A-Za-z0-9_.-]+)==", line)
            if match:
                names.add(match[1].lower().replace("_", "-"))
    names.update({"torch", "torchvision", "numpy", "matplotlib", "psutil", "pytest", "setuptools"})
    rows, missing = [], []
    for name in sorted(names):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        meta = dist.metadata
        rows.append({"name": meta.get("Name", name), "version": dist.version,
                     "license_expression": meta.get("License-Expression"),
                     "license": meta.get("License"),
                     "license_classifiers": [c for c in meta.get_all("Classifier", []) if c.startswith("License ::")],
                     "license_files": meta.get_all("License-File", []),
                     "installed_license_files": [str(p) for p in dist.files or [] if "license" in str(p).lower() or "copying" in str(p).lower() or "notice" in str(p).lower()]})
    return {"scanner": "installed distribution license metadata", "python": sys.version.split()[0],
            "scope": "installed packages named in selected requirements and direct dependencies",
            "requirement_files": [p.name for p in files],
            "status": "BLOCKED" if missing else "PASS", "packages": rows, "missing_packages": missing,
            "limitations": ["Metadata inventory is not a legal compatibility opinion.", "Only the active environment's installed distributions are observed.", "Other platform or Python lockfiles are not asserted to have identical bundled license contents."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive", type=Path, help="Also inspect a release ZIP or wheel")
    parser.add_argument("--licenses", action="store_true", help="Inventory active environment dependency license metadata")
    parser.add_argument("--lock", type=Path, help="Requirement file for license inventory, default all committed locks")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = license_metadata(args.root, args.lock) if args.licenses else scan_tree(args.root, args.manifest)
    if args.archive:
        if args.archive.stat().st_size > MAX_EXPANDED:
            findings, limits = [], [{"path": args.archive.name, "reason": "archive file size limit"}]
        else:
            findings, limits = scan_bytes(args.archive.read_bytes(), args.archive.name)
        result["archive"] = {"file": args.archive.name, "findings": findings, "limits": limits,
                             "status": "FAIL" if findings or limits else "PASS"}
        if findings or limits:
            result["status"] = "FAIL"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"], "files_scanned": result.get("files_scanned"),
                      "finding_counts": result.get("finding_counts", {}), "limit_count": len(result.get("limits", [])),
                      "packages": len(result.get("packages", [])), "missing_packages": result.get("missing_packages", []),
                      "archive_status": result.get("archive", {}).get("status")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
