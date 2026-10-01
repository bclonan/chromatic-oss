"""Verify native wheels in eight clean Python environments without ML dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.provenance import capture_provenance, verify_provenance, file_hash

WINDOWS = {"python311": "py311", "python312": "current", "python313": "py313", "python314": "py314"}
LINUX = {"linux311": "3.11", "linux312": "3.12", "linux313": "3.13", "linux314": "3.14"}


def run(args):
    print("Running", subprocess.list2cmdline([str(arg) for arg in args]), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def python_source(label, directory):
    """Reuse a prepared interpreter or let uv resolve the requested version."""
    existing = ROOT / ".release-env" / directory / "Scripts/python.exe"
    return existing if existing.is_file() else "3." + str(int(label[-2:]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/native-hue-package-verification.json")
    args = parser.parse_args(argv)
    windows = ROOT / "artifacts/packages/native-hue/chromatic_hue-0.1.0-py3-none-win_amd64.whl"
    linux = ROOT / "artifacts/packages/native-hue/chromatic_hue-0.1.0-py3-none-linux_x86_64.whl"
    config = {"windows": list(WINDOWS), "linux": list(LINUX), "runtime_dependencies": [],
              "wheels": [str(path.relative_to(ROOT).as_posix()) for path in (windows, linux)]}
    provenance = capture_provenance(config, ROOT)
    output_dir = ROOT / "artifacts/native/package-checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, artifacts = [], [windows, linux]
    for label, directory in WINDOWS.items():
        environment = ROOT / ".release-env" / ("native-hue-" + label)
        interpreter = environment / "Scripts/python.exe"
        if not interpreter.is_file():
            run(["uv", "venv", "--python", python_source(label, directory), environment])
        run(["uv", "pip", "install", "--python", interpreter, "--no-deps", "--reinstall", windows])
        output = output_dir / (label + ".json")
        run([interpreter, "-I", ROOT / "scripts/verify_installed_hue.py", "--output", output])
        row = json.loads(output.read_text(encoding="utf-8"))
        rows.append({"environment": label, "wheel_sha256": file_hash(windows), "receipt": output.relative_to(ROOT).as_posix(), **row})
        artifacts.append(output)
    for label, version in LINUX.items():
        image = "chromatensor-review-ci:python-" + version
        wheel_path = linux.relative_to(ROOT).as_posix()
        shell = ("set -eu\npython -m venv /tmp/hue-native\n"
                 f"/tmp/hue-native/bin/python -m pip install --no-index --no-deps /workspace/{wheel_path}\n"
                 f"/tmp/hue-native/bin/python -I /workspace/scripts/verify_installed_hue.py --output /evidence/{label}.json\n")
        run(["docker", "run", "--rm", "--network", "none", "--mount",
             f"type=bind,source={ROOT},target=/workspace,readonly", "--mount",
             f"type=bind,source={output_dir},target=/evidence", image, "sh", "-c", shell])
        output = output_dir / (label + ".json")
        row = json.loads(output.read_text(encoding="utf-8"))
        image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
        rows.append({"environment": label, "wheel_sha256": file_hash(linux), "receipt": output.relative_to(ROOT).as_posix(),
                     "image_id": image_id, "network_disabled": True, **row})
        artifacts.append(output)
    evidence = {"schema_version": 1, "config": config, "environments": rows,
                "wheels": [{"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}
                           for path in (windows, linux)]}
    evidence["provenance"] = capture_provenance(config, ROOT, artifacts)
    assert evidence["provenance"]["source"] == provenance["source"], "Source changed during installed-wheel verification"
    verify_provenance(evidence["provenance"], config, ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print("Verified eight dependency-free installed native hue wheels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
