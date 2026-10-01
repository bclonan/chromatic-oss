"""Build the dependency-free Rust hue library and a platform-specific Python wheel."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "native/hue_tensor/Cargo.toml"
VERSION = "0.1.0"
RUST_IMAGE = "rust:1.96.0-slim-bookworm@sha256:4732ca96fd086cb9be682050c3f0176288eebaac2b80aa2bcefccfaf198e1950"
COMMANDS = []


def public_text(value):
    """Keep build receipts portable without publishing local account paths."""
    text = str(value)
    for path, replacement in ((ROOT.resolve(), "<workspace>"), (Path.home(), "<user-home>")):
        for spelling in (str(path), path.as_posix()):
            text = text.replace(spelling, replacement)
    return text


def private_path_spellings():
    return {spelling for path in (ROOT.resolve(), Path.home())
            for spelling in (str(path), path.as_posix())}


def rust_flags(*, container=False, windows=False):
    """Remap compiler paths and prevent an absolute Windows PDB reference."""
    flags = ["--remap-path-prefix=/workspace=workspace"] if container else [
        "--remap-path-prefix=" + str(Path.home()) + "=user-home",
        "--remap-path-prefix=" + str(ROOT.resolve()) + "=workspace",
    ]
    if windows:
        flags.extend(["-C", "link-arg=/PDBALTPATH:hue_tensor.pdb"])
    return flags


def check_binary_privacy(path):
    data = Path(path).read_bytes().lower()
    for spelling in private_path_spellings():
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            if spelling.encode(encoding).lower() in data:
                raise ValueError("Native binary contains a private build path; rebuild with path remapping")


def command(args, **kwargs):
    print("Running", subprocess.list2cmdline([str(arg) for arg in args]), flush=True)
    process = subprocess.run([str(arg) for arg in args], capture_output=True, text=True, **kwargs)
    COMMANDS.append({"argv": [public_text(arg) for arg in args], "returncode": process.returncode,
                     "stdout": public_text(process.stdout), "stderr": public_text(process.stderr)})
    print(process.stdout, end="", flush=True)
    print(process.stderr, end="", file=sys.stderr, flush=True)
    process.check_returncode()
    return process


def rust_notices(root=ROOT):
    directory = Path(root) / "native/rust-notices"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = (directory / item["path"]).resolve()
        if (not path.is_relative_to(directory.resolve()) or not path.is_file()
                or path.stat().st_size != item["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]):
            raise ValueError("Rust attribution differs from its reviewed manifest; update notices before building")
    return manifest, sorted(path for path in directory.rglob("*") if path.is_file())


def validate_toolchain(version, notices):
    if version.strip() != notices["rustc"]:
        raise ValueError("Rust compiler differs from bundled attribution; update native/rust-notices for this toolchain")


def make_wheel(binary, directory, tag, destination):
    info = f"chromatic_hue-{VERSION}.dist-info"
    _, notices = rust_notices(ROOT)
    notice_names = ["rust/" + path.relative_to(ROOT / "native/rust-notices").as_posix() for path in notices]
    notice_metadata = "".join("License-File: " + name + "\n" for name in notice_names)
    files = {
        "chromatic_hue/__init__.py": (ROOT / "chromatensor/hue.py").read_bytes(),
        f"chromatic_hue/_native/{directory}/{binary.name}": binary.read_bytes(),
        f"{info}/METADATA": (f"Metadata-Version: 2.4\nName: chromatic-hue\nVersion: {VERSION}\n"
            "Summary: Native RGBD hue tensor algebra in Rust\nRequires-Python: >=3.11\n"
            "Author: Brad Clonan\n"
            "License-Expression: AGPL-3.0-only\nLicense-File: LICENSE\n"
            "License-File: THIRD_PARTY_NOTICES.md\nLicense-File: COPYRIGHT\n" + notice_metadata + "\n"
            "Integer color tensor operations. No PyTorch or NumPy runtime dependency.\n").encode(),
        f"{info}/WHEEL": (f"Wheel-Version: 1.0\nGenerator: chromatic native builder\n"
            f"Root-Is-Purelib: false\nTag: py3-none-{tag}\n").encode(),
    }
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "COPYRIGHT"):
        files[f"{info}/licenses/{name}"] = (ROOT / name).read_bytes()
    for name, path in zip(notice_names, notices):
        files[f"{info}/licenses/{name}"] = path.read_bytes()
    # Include the exact low-level sources and header for inspection and rebuilds.
    for path in CRATE.parent.rglob("*"):
        if path.is_file() and "target" not in path.parts and path.suffix in {".rs", ".toml", ".lock", ".h"}:
            files["chromatic_hue/_source/" + path.relative_to(CRATE.parent).as_posix()] = path.read_bytes()
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    files[f"{info}/RECORD"] = manifest.getvalue().encode()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"chromatic_hue-{VERSION}-py3-none-{tag}.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, data)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("host", "linux"), default="host")
    parser.add_argument("--wheel-dir", type=Path, default=ROOT / "artifacts/packages/native-hue")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args(argv)
    args.wheel_dir = args.wheel_dir.resolve()
    if not args.wheel_dir.is_relative_to(ROOT.resolve()):
        parser.error("--wheel-dir must stay inside the workspace for source-bound artifact evidence")
    COMMANDS.clear()
    sys.path.insert(0, str(ROOT))
    from experiments.provenance import capture_provenance, verify_provenance, file_hash
    notices, _ = rust_notices(ROOT)
    container_build = args.platform == "linux" and platform.system() != "Linux"
    flags = rust_flags(container=container_build, windows=os.name == "nt" and not container_build)
    config = {"platform": args.platform, "test": args.test, "rust_image": RUST_IMAGE,
              "wheel_dir": args.wheel_dir.resolve().relative_to(ROOT).as_posix(),
              "rust_flags": [public_text(flag) for flag in flags], "rustc": notices["rustc"]}
    before = capture_provenance(config, ROOT)
    if container_build:
        directory, tag, filename = "linux-x86_64", "linux_x86_64", "libhue_tensor.so"
        build_dir = ROOT / "artifacts/native-build-linux"
        build_dir.mkdir(parents=True, exist_ok=True)
        cargo_args = "rustc --version && "
        cargo_args += "cargo test --offline --manifest-path /workspace/native/hue_tensor/Cargo.toml && " if args.test else ""
        cargo_args += "cargo build --release --offline --manifest-path /workspace/native/hue_tensor/Cargo.toml"
        process = command(["docker", "run", "--rm", "--network", "none", "--mount",
            f"type=bind,source={ROOT},target=/workspace,readonly", "--mount",
            f"type=bind,source={build_dir},target=/build", "-e", "CARGO_TARGET_DIR=/build",
            "-e", "CARGO_ENCODED_RUSTFLAGS=" + "\x1f".join(flags), "--platform", "linux/amd64",
            RUST_IMAGE, "sh", "-c", cargo_args])
        validate_toolchain(process.stdout.splitlines()[0], notices)
        binary = build_dir / "release" / filename
    else:
        windows = os.name == "nt"
        directory = "windows-x86_64" if windows else "linux-x86_64"
        tag = "win_amd64" if windows else "linux_x86_64"
        filename = "hue_tensor.dll" if windows else "libhue_tensor.so"
        if platform.machine().lower() not in {"amd64", "x86_64"}:
            raise RuntimeError("This build script currently packages x86-64 targets")
        build_dir = ROOT / "artifacts/native-build-host"
        env = dict(os.environ, CARGO_TARGET_DIR=str(build_dir), CARGO_ENCODED_RUSTFLAGS="\x1f".join(flags))
        version = command(["rustc", "--version"], env=env, cwd=ROOT).stdout
        validate_toolchain(version, notices)
        if args.test:
            command(["cargo", "test", "--offline", "--manifest-path", CRATE.relative_to(ROOT)], env=env, cwd=ROOT)
        command(["cargo", "build", "--release", "--offline", "--manifest-path", CRATE.relative_to(ROOT)], env=env, cwd=ROOT)
        binary = build_dir / "release" / filename
    check_binary_privacy(binary)
    installed = ROOT / "chromatensor/_native" / directory / filename
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, installed)
    wheel = make_wheel(binary, directory, tag, args.wheel_dir)
    executable = build_dir / "release" / ("hue-tensor-cli.exe" if directory.startswith("windows") else "hue-tensor-cli")
    check_binary_privacy(executable)
    evidence = {"schema_version": 1, "config": config, "commands": COMMANDS,
                "binary": {"path": installed.relative_to(ROOT).as_posix(), "sha256": file_hash(installed), "bytes": installed.stat().st_size},
                "cli": {"path": executable.relative_to(ROOT).as_posix(), "sha256": file_hash(executable), "bytes": executable.stat().st_size},
                "wheel": {"path": wheel.relative_to(ROOT).as_posix(), "sha256": file_hash(wheel), "bytes": wheel.stat().st_size}}
    evidence["provenance"] = capture_provenance(config, ROOT, [installed, executable, wheel])
    assert evidence["provenance"]["source"] == before["source"], "Source changed during native build"
    verify_provenance(evidence["provenance"], config, ROOT)
    receipt = ROOT / "results" / ("native-hue-build-" + directory + ".json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"binary": str(installed), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                      "wheel": str(wheel), "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      "wheel_bytes": wheel.stat().st_size, "platform": tag}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
