"""Run the full dependency matrix and isolated installed-wheel checks."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.provenance import capture_provenance, verify_provenance
from experiments.common import write_json

ENVIRONMENTS = {"minimum": "minimum", "python311": "py311", "python312": "current", "python313": "py313", "python314": "py314"}
LINUX = {"linux311": "3.11", "linux312": "3.12", "linux313": "3.13", "linux314": "3.14"}


def selected_environments(labels):
    supported = set(ENVIRONMENTS) | set(LINUX)
    selected = supported if labels is None else set(labels)
    if not selected or not selected <= supported:
        raise ValueError("Select supported, nonempty environment labels")
    return ({label: value for label, value in ENVIRONMENTS.items() if label in selected},
            {label: value for label, value in LINUX.items() if label in selected})


def copy_source(destination, root=ROOT):
    """Copy the reviewed public source and built libraries needed by source tests."""
    from scripts.release_audit import select_files
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Matrix source copy must be outside the checkout")
    files = list(select_files(root))
    for path in (root / "chromatensor/_native").rglob("*"):
        if path.is_file() and path.name in {"hue_tensor.dll", "libhue_tensor.so"}:
            files.append(path)
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Matrix source copy contains an external file")
        target = destination / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return [path.relative_to(root).as_posix() for path in files]


def prepare_windows(environments):
    for label, directory in environments.items():
        environment = ROOT / ".release-env" / directory
        interpreter = environment / "Scripts/python.exe"
        version = "3.11" if label == "minimum" else "3." + str(int(label[-2:]))
        if not interpreter.is_file():
            run(["uv", "venv", "--python", version, environment])
        run(["uv", "pip", "sync", "--python", interpreter,
             ROOT / "requirements" / (label + "-cpu.lock"),
             "--index", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"])


def install_command(interpreter, results, output):
    args = [interpreter, "-I", ROOT / "scripts/verify_release_install.py", "--output", output]
    if results is not None:
        args.extend(["--results", results])
    return args


def freeze_without_release(value):
    return "\n".join(line for line in value.splitlines()
                     if not line.lower().startswith(("chromatensor", "chromatic-hue", "chromatic_hue"))) + "\n"


def matrix_report(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases or any(list(case.iter(tag)) for case in cases for tag in ("failure", "error")):
        raise ValueError("Dependency matrix tests failed or did not execute")
    skipped = [case for case in cases if list(case.iter("skipped"))]
    if any("CUDA device unavailable" not in next(case.iter("skipped")).attrib.get("message", "") for case in skipped):
        raise ValueError("Dependency matrix skipped a required CPU test")
    modules = {case.attrib.get("classname", "").rsplit(".", 1)[-1] for case in cases}
    if not {path.stem for path in (ROOT / "tests").glob("test_*.py")}.issubset(modules):
        raise ValueError("Dependency matrix omitted a test module")
    return {"passed": len(cases) - len(skipped), "cuda_only_skipped": len(skipped), "failures": 0}


def run(command):
    print("Running " + " ".join(str(part) for part in command), flush=True)
    process = subprocess.run([str(part) for part in command], cwd=ROOT, check=False, text=True, capture_output=True)
    if process.returncode:
        print(process.stdout, flush=True)
        print(process.stderr, flush=True)
        raise subprocess.CalledProcessError(process.returncode, command, process.stdout, process.stderr)
    return process


def linux_image(version):
    return "chromatensor-review-ci:python-" + version


def prepare_linux(environments=LINUX):
    for label, version in environments.items():
        lock = ROOT / "requirements" / (label + "-cpu.lock")
        encoded = base64.b64encode(lock.read_bytes()).decode("ascii")
        dockerfile = f"""FROM python:{version}-slim-bookworm
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg
RUN python -c "import base64;open('/tmp/dependencies.lock','wb').write(base64.b64decode('{encoded}'))" && python -m pip install -r /tmp/dependencies.lock --extra-index-url https://download.pytorch.org/whl/cpu
"""
        print("Preparing independent Linux dependencies for Python " + version, flush=True)
        subprocess.run(["docker", "build", "-t", linux_image(version), "-"], input=dockerfile, text=True, check=True)
        frozen = run(["docker", "run", "--rm", "--network", "none", linux_image(version), "python", "-m", "pip", "freeze", "--all"]).stdout
        if frozen != lock.read_text(encoding="utf-8"):
            raise ValueError("Prepared Linux image differs from its committed CPU lock: " + label)


def linux_check(label, version, wheel, results):
    report_dir = ROOT / "artifacts/verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = wheel.relative_to(ROOT).as_posix()
    native_name = "artifacts/packages/native-hue/chromatic_hue-0.1.0-py3-none-linux_x86_64.whl"
    replay = "" if results is None else " --results " + shlex.quote("/workspace/" + results.relative_to(ROOT).as_posix())
    # The container gets read-only source and checkpoints. Only receipts are writable.
    command = f"""set -eu
python /workspace/scripts/verify_dependencies.py --copy-source /tmp/project
cd /tmp/project
python -m pip install --no-index --no-deps {shlex.quote('/workspace/' + wheel_name)} /workspace/{native_name}
python -m pip check > /evidence/{label}-pip-check.txt
python -m pip freeze --all | sed '/^chromatensor[ =@]/Id; /^chromatic.hue[ =@]/Id' > /evidence/{label}-dependencies.txt
python -m pytest tests -q --junitxml=/evidence/{label}-tests.xml
python -I /workspace/scripts/verify_release_install.py --output /evidence/{label}-installed.json{replay}
"""
    process = run(["docker", "run", "--rm", "--network", "none",
                   "--mount", f"type=bind,source={ROOT},target=/workspace,readonly",
                   "--mount", f"type=bind,source={report_dir},target=/evidence",
                   linux_image(version), "sh", "-c", command])
    print(process.stdout, flush=True)
    installed_path = report_dir / (label + "-installed.json")
    installed = json.loads(installed_path.read_text(encoding="utf-8"))
    lock = ROOT / "requirements" / (label + "-cpu.lock")
    if lock.read_text(encoding="utf-8") != (report_dir / (label + "-dependencies.txt")).read_text(encoding="utf-8"):
        raise ValueError("Linux dependency image differs from recorded lock")
    check = report_dir / (label + "-pip-check.txt")
    xml = report_dir / (label + "-tests.xml")
    row = {"environment": label, "python": installed["python"], "dependencies": installed["dependencies"],
           "tests": matrix_report(xml), "installed_report": installed_path.relative_to(ROOT).as_posix(),
           "lock": lock.relative_to(ROOT).as_posix(), "dependency_check": check.read_text(encoding="utf-8"),
           "container_image": linux_image(version), "container_image_id": run(["docker", "image", "inspect", "--format", "{{.Id}}", linux_image(version)]).stdout.strip(),
           "network_disabled_during_checks": True}
    return row, [xml, installed_path, lock, check, report_dir / (label + "-dependencies.txt")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=ROOT / "artifacts/packages/chromatensor-0.2.0-py3-none-any.whl")
    parser.add_argument("--results", type=Path, help="Replay canonical saved MNIST probes in addition to toolkit checks")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dependency-verification.json")
    parser.add_argument("--write-locks", action="store_true", help="Write dependency freezes before the final source snapshot")
    parser.add_argument("--prepare-linux", action="store_true", help="Build independent Linux dependency images and capture their locks before source freeze")
    parser.add_argument("--prepare-windows", action="store_true", help="Create requested uv environments and install their committed CPU locks")
    parser.add_argument("--environments", nargs="+", choices=list(ENVIRONMENTS) + list(LINUX), help="Run or prepare a subset; default is the full nine-job matrix")
    parser.add_argument("--copy-source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.copy_source is not None:
        copy_source(args.copy_source)
        return 0
    windows, linux = selected_environments(args.environments)
    if args.prepare_windows:
        prepare_windows(windows)
    if args.prepare_linux:
        prepare_linux(linux)
    if args.prepare_windows or args.prepare_linux:
        return 0
    args.wheel = args.wheel.resolve()
    if args.results is not None:
        args.results = args.results.resolve()
    config = {"environments": windows, "linux": linux, "wheel": args.wheel.relative_to(ROOT).as_posix(),
              "results": None if args.results is None else args.results.relative_to(ROOT).as_posix(), "test_scope": "all CPU tests; CUDA-only tests require separate GPU evidence"}
    provenance = capture_provenance(config, ROOT)
    artifacts, rows = [], []
    for label, directory in windows.items():
        interpreter = ROOT / ".release-env" / directory / "Scripts/python.exe"
        if not interpreter.is_file():
            raise ValueError("Environment missing; run --prepare-windows --environments " + label)
        lock = ROOT / "requirements" / (label + "-cpu.lock")
        frozen = run(["uv", "pip", "freeze", "--python", interpreter]).stdout
        frozen = freeze_without_release(frozen)
        if args.write_locks:
            lock.write_text(frozen, encoding="utf-8")
            continue
        if not lock.is_file() or lock.read_text(encoding="utf-8") != frozen:
            raise ValueError("Installed dependency set differs from committed lock: " + label)
        native = ROOT / "artifacts/packages/native-hue/chromatic_hue-0.1.0-py3-none-win_amd64.whl"
        run(["uv", "pip", "install", "--python", interpreter, "--no-deps", "--reinstall", args.wheel, native])
        check = run(["uv", "pip", "check", "--python", interpreter])
        xml = ROOT / "artifacts" / "verification" / (label + "-tests.xml")
        xml.parent.mkdir(parents=True, exist_ok=True)
        tests = run([interpreter, "-m", "pytest", "tests", "-q", "--junitxml=" + str(xml)])
        print(tests.stdout, flush=True)
        installed_report = xml.with_name(label + "-installed.json")
        run(install_command(interpreter, args.results, installed_report))
        installed = json.loads(installed_report.read_text(encoding="utf-8"))
        if label == "minimum" and (installed["dependencies"]["torch"] != "2.14.1+cpu" or installed["dependencies"]["torchvision"] != "0.29.1+cpu"):
            raise ValueError("Minimum Torch/torchvision versions were not tested")
        rows.append({"environment": label, "python": installed["python"], "dependencies": installed["dependencies"],
                     "tests": matrix_report(xml), "installed_report": installed_report.relative_to(ROOT).as_posix(),
                     "dependency_check": check.stdout + check.stderr, "lock": lock.relative_to(ROOT).as_posix()})
        artifacts.extend((xml, installed_report, lock))
    if args.write_locks:
        return 0
    for label, version in linux.items():
        row, files = linux_check(label, version, args.wheel, args.results)
        rows.append(row)
        artifacts.extend(files)
    provenance["artifacts"] = capture_provenance(config, ROOT, artifacts)["artifacts"]
    verify_provenance(provenance, config, ROOT)
    write_json(args.output, {"schema_version": 1, "config": config, "provenance": provenance,
                            "environments": rows, "checks": {"pip_check": True, "minimum_supported": "minimum" in windows,
                            "all_supported_python_minors": windows == ENVIRONMENTS and linux == LINUX,
                            "clean_installed_wheel_inference": args.results is not None},
                            "scope": f"Executed {len(rows)} selected dependency jobs with all CPU tests and isolated installed toolkit checks. Saved-model replay requested: {args.results is not None}. GitHub-hosted execution requires separate evidence."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
