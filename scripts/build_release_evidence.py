"""Build the exact release wheel and verify it in nine clean environments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import write_json
from experiments.provenance import capture_provenance, artifact_manifest, file_hash, verify_provenance
from experiments.verification import verify_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/latest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/release-verification.json")
    args = parser.parse_args(argv)
    wheel = ROOT / "artifacts/packages/chromatensor-0.2.0-py3-none-any.whl"
    dependency_report = ROOT / "results/dependency-verification.json"
    installation_report = ROOT / "artifacts/verification/python312-installed.json"
    config = {"wheel": wheel.relative_to(ROOT).as_posix(), "results": args.results.relative_to(ROOT).as_posix(),
              "dependency_evidence": dependency_report.relative_to(ROOT).as_posix(),
              "installation_evidence": installation_report.relative_to(ROOT).as_posix()}
    provenance = capture_provenance(config, ROOT)
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "--no-build-isolation",
                    "--wheel-dir", str(wheel.parent)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_dependencies.py"), "--wheel", str(wheel),
                    "--results", str(args.results), "--output", str(dependency_report)], cwd=ROOT, check=True)
    installed = json.loads(installation_report.read_text(encoding="utf-8"))
    provenance["artifacts"] = artifact_manifest([wheel, dependency_report, installation_report], ROOT)
    evidence = {"schema_version": 1, "config": config, "provenance": provenance,
                "wheel": {"path": wheel.relative_to(ROOT).as_posix(), "sha256": file_hash(wheel), "bytes": wheel.stat().st_size},
                "installed_assets": installed["installed_assets"],
                "checks": {**installed["checks"], "pip_check": True},
                "environment": {"python": installed["python"], "dependencies": installed["dependencies"],
                                "isolated": installed["isolated"], "user_site_enabled": installed["user_site_enabled"]},
                "dependency_evidence": dependency_report.relative_to(ROOT).as_posix(),
                "installation_evidence": installation_report.relative_to(ROOT).as_posix(),
                "scope": "Exact source wheel with independently installed CPU dependencies, separate from host CUDA verification"}
    verify_provenance(provenance, config, ROOT)
    verify_release(evidence, ROOT)
    write_json(args.output, evidence)
    print(json.dumps({"wheel": evidence["wheel"], "verified": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
