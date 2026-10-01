"""Validate documentation and select public contents without copying local caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"chromatensor", "dashboard", "experiments", "native", "scripts", "tests", "requirements", ".github"}
SOURCE_EXTENSIONS = {".py", ".rs", ".toml", ".lock", ".h", ".c", ".html", ".js", ".css", ".txt", ".yml", ".yaml"}
ROOT_FILES = {"netlify.toml", ".gitattributes", ".gitignore", ".gitleaks.toml", "pyproject.toml", "LICENSE", "COPYRIGHT", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md", "RELEASE_CHECKLIST.md", "AGENTS.md", "ARCHITECTURE.md", "README.md", "BENCHMARKS.md", "EXPERIMENTS.md", "EXPERIMENT_MATRIX.md", "LIMITATIONS.md", "RESEARCH_LOG.md", "RESULTS.md", "THEORY.md", "TODO.md"}
SKIP_PARTS = {"target", "__pycache__", "node_modules", ".git", ".pytest-work", ".pytest_cache", "build", "dist", "_native"}
MANIFEST = "docs/release/manifest.json"
APPROVED_SITE_MEDIA = {"docs/site/media/hardware-build.mp4", "docs/site/media/hardware-build-poster.png", "docs/site/media/hardware-build-captions.vtt"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_files(root=ROOT):
    """Apply an explicit path/type policy; never use .gitignore as the policy."""
    root = Path(root).resolve()
    selected = []
    for name in sorted(ROOT_FILES):
        if (root / name).is_file():
            selected.append(root / name)
    for directory in sorted(SOURCE_DIRS | {"docs"}):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(part in SKIP_PARTS for part in relative.parts):
                continue
            if relative.as_posix().startswith("docs/site/media/"):
                continue  # Media enters the release only through the exact-path allowlist.
            suffixes = SOURCE_EXTENSIONS | {".md", ".json", ".svg", ".png"} if directory == "docs" else SOURCE_EXTENSIONS | {".md"}
            notice = directory == "native" and "rust-notices" in relative.parts
            if path.suffix not in suffixes and path.name != ".gitignore" and not notice:
                continue
            if path.is_symlink():
                raise ValueError("Symlink needs manual review: " + relative.as_posix())
            selected.append(path)
    for pattern in ("artifacts/figures/**/*.png", "artifacts/native/native-color-tensor-diagram.*", "artifacts/dashboard/*.jpg"):
        selected.extend(path for path in root.glob(pattern) if path.is_file() and path.suffix in {".png", ".svg", ".jpg"})
    # Current numerical receipts and generated synthetic proof artifacts are data,
    # not reusable private binaries or downloaded dataset snapshots.
    for pattern in ("results/*.json", "artifacts/native/package-checks/*.json", "artifacts/native-hue/**/*.json", "artifacts/native-hue/**/*.huetensor", "artifacts/native-hue/**/*.u32"):
        selected.extend(path for path in root.glob(pattern) if path.is_file())
    for name in APPROVED_SITE_MEDIA:
        path = root / name
        if path.is_symlink():
            raise ValueError("Symlink needs manual review: " + name)
        if path.is_file():
            selected.append(path)
    return sorted(set(selected))


def build_manifest(root=ROOT):
    root = Path(root).resolve()
    files = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
              "sha256": digest(path), "reason": "selected implementation, documentation, verified measurement or reviewed illustration"}
             for path in select_files(root) if path.relative_to(root).as_posix() != MANIFEST]
    return {"schema_version": 1, "selection_policy": "scripts/release_audit.py select_files; reviewed tracked source plus explicitly selected evidence",
            "self_exclusion": "The manifest excludes its own bytes to avoid a recursive hash. It is included in commits and archives.",
            "excluded_categories": ["original Git history and remotes", "local environments and caches", "downloaded datasets and data-derived probe arrays", "unreviewed logs and test scratch files", "historical executables, wheels and library binaries", "private security reports"],
            "files": files, "total_bytes": sum(row["bytes"] for row in files)}


def check_documentation(root=ROOT):
    root = Path(root).resolve()
    errors, checked = [], 0
    # Markdown destinations, including image links. Fenced examples are not links.
    for path in select_files(root):
        if path.suffix != ".md":
            continue
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for match in re.finditer(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text):
            destination = match.group(1).strip().split(' "', 1)[0].strip("<>")
            if not destination or destination.startswith("#") or urlsplit(destination).scheme:
                continue
            checked += 1
            target = (path.parent / unquote(destination.split("#", 1)[0].split("?", 1)[0])).resolve()
            if not target.is_relative_to(root) or not target.exists():
                errors.append({"path": path.relative_to(root).as_posix(), "target": destination,
                               "line": text.count("\n", 0, match.start()) + 1})
    return {"status": "FAIL" if errors else "PASS", "local_links_checked": checked, "errors": errors,
            "limitations": "Checks local destination existence, not remote URL availability or Markdown anchor rendering."}


def create_archive(destination, root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    rows = manifest["files"]
    for row in rows:
        target = (root / row["path"]).resolve()
        if not target.is_relative_to(root) or digest(target) != row["sha256"]:
            raise ValueError("Selected content changed: " + row["path"])
    destination = Path(destination).resolve()
    if destination.is_relative_to(root):
        raise ValueError("Write release archive outside the repository")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in rows:
            archive.write(root / row["path"], "chromatic-oss/" + row["path"])
        archive.write(root / MANIFEST, "chromatic-oss/" + MANIFEST)
    with zipfile.ZipFile(destination) as archive:
        expected = {"chromatic-oss/" + row["path"] for row in rows} | {"chromatic-oss/" + MANIFEST}
        if set(archive.namelist()) != expected or archive.testzip() is not None:
            raise ValueError("Release archive content mismatch")
        for row in rows:
            if hashlib.sha256(archive.read("chromatic-oss/" + row["path"])).hexdigest() != row["sha256"]:
                raise ValueError("Archive member changed: " + row["path"])
    return {"name": destination.name, "sha256": digest(destination), "bytes": destination.stat().st_size, "members": len(rows) + 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-docs", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    if args.check_docs:
        value = check_documentation()
        print(json.dumps(value, indent=2))
        if value["status"] != "PASS":
            return 1
    if args.manifest:
        value = build_manifest()
        (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / MANIFEST).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": MANIFEST, "files": len(value["files"]), "bytes": value["total_bytes"]}))
    if args.archive:
        print(json.dumps(create_archive(args.archive)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
