"""Run the complete required suite and bind its JUnit report to current source."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import write_json
from experiments.provenance import capture_provenance, artifact_manifest, verify_provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=ROOT / "artifacts/verification-tests.xml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/test-verification.json")
    args = parser.parse_args(argv)
    args.xml = args.xml.resolve()
    if not args.xml.is_relative_to(ROOT):
        parser.error("--xml must stay inside the workspace for portable evidence")
    args.xml.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "tests", "-q", "--junitxml=" + str(args.xml)]
    relative_xml = args.xml.relative_to(ROOT).as_posix()
    config = {"command": ["<interpreter>", *command[1:-1], "--junitxml=" + relative_xml], "xml": relative_xml}
    provenance = capture_provenance(config, ROOT)
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    collection_command = [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"]
    collected = subprocess.run(collection_command, cwd=ROOT, env=environment, check=True, capture_output=True, text=True)
    identities = [line for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("Full test collection is empty or duplicated")
    collection = args.xml.with_suffix(".collection.json")
    collection.write_text(json.dumps({"command": ["<interpreter>", *collection_command[1:]], "nodeids": identities}, indent=2) + "\n", encoding="utf-8")
    process = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if args.xml.is_file():
        provenance["artifacts"] = artifact_manifest([args.xml, collection], ROOT)
    verify_provenance(provenance, config, ROOT)
    write_json(args.output, {"schema_version": 1, "config": config,
                           "returncode": process.returncode, "collection": collection.relative_to(ROOT).as_posix(),
                           "provenance": provenance})
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
