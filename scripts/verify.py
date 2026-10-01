"""Recheck live source, tests, model checkpoints and the exact release wheel."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.common import write_json
from experiments.provenance import capture_provenance, verify_provenance
from experiments.verification import (read_record, verify_test_report, verify_training_coverage,
    verify_comparisons, verify_precision, verify_model_checkpoints, verify_profiler,
    verify_cube_result, verify_release)


def verify(*, root=ROOT, results_path=None, cube_results_path=None, tests_path=None,
           test_evidence_path=None, release_evidence_path=None, browser_evidence_path=None,
           research_results_path=None, output_path=None):
    root = Path(root).resolve()
    paths = {"results": Path(results_path or root / "results/latest.json"),
             "cube": Path(cube_results_path or root / "results/cube.json"),
             "tests": Path(tests_path or root / "artifacts/verification-tests.xml"),
             "test_evidence": Path(test_evidence_path or root / "results/test-verification.json"),
             "release": Path(release_evidence_path or root / "results/release-verification.json"),
             "browser": Path(browser_evidence_path or root / "artifacts/dashboard/cube-verification.json"),
             "research": Path(research_results_path or root / "results/release-research.json"),
             "rgbd": root / "results/rgbd-token-verification.json"}
    config = {name: path.resolve().relative_to(root).as_posix() for name, path in paths.items()}
    gates, artifacts = [], []
    torch.set_num_threads(4)

    def gate(name, check):
        try:
            evidence = check()
            gates.append({"gate": name, "verified": True, "evidence": evidence})
        except Exception as error:
            gates.append({"gate": name, "verified": False, "evidence": str(error)})

    def run_record(name):
        record = read_record(paths[name])
        artifacts.append(paths[name])
        if name in {"results", "cube", "research"}:
            verify_provenance(record.get("provenance"), record.get("config"), root)
        return record

    def tests():
        evidence = run_record("test_evidence")
        result = verify_test_report(paths["tests"], root, evidence)
        artifacts.append(paths["tests"])
        return result

    def training():
        result = run_record("results")
        verify_provenance(result.get("provenance"), result.get("config"), root)
        return verify_training_coverage(result)

    def browser():
        evidence = run_record("browser")
        verify_provenance(evidence.get("provenance"), evidence.get("config"), root)
        required = {"create", "turn", "undo", "six_dimensions", "hidden_slice", "spectral_erasure", "recorded_results"}
        checks = evidence.get("checks", {})
        if not required.issubset(checks) or not all(checks[name] is True for name in required):
            raise ValueError("Current-source browser interaction evidence is incomplete")
        if not evidence["provenance"]["artifacts"]:
            raise ValueError("Browser evidence has no saved receipt or image")
        return {"interaction_checks": sorted(required)}

    def research():
        from scripts.run_release_research import validate_release_research
        result = run_record("research")
        if result.get("status") != "complete":
            raise ValueError("Sustained release research has not completed")
        validate_release_research(result, root, require_release=True)
        return {"trained_models": len(result["experiments"]), "seeds": result["config"]["seeds"],
                "epochs": result["config"]["epochs"], "final_test_examples": result["final_test"]["examples"],
                "large_tensor_values": result["large_tensor"]["samples"]}

    def rgbd():
        from scripts.verify_rgbd_tokens import validate_rgbd_evidence
        return validate_rgbd_evidence(run_record("rgbd"), root, require_release=True)

    gate("Executed required tests with current source identity", tests)
    gate("Canonical trained task, representation and seed coverage", training)
    gate("Paired equal-bit controls agree with model records", lambda: verify_comparisons(run_record("results")))
    gate("Precision distributions and operator control coverage", lambda: verify_precision(run_record("results")))
    gate("Actual expected-model checkpoint and held-out probe replay", lambda: verify_model_checkpoints(run_record("results"), root))
    gate("Live checkpoint-backed profiler replay", lambda: verify_profiler(run_record("results"), root))
    gate("Immutable cube checkpoints and reconstructed measurement replay", lambda: verify_cube_result(run_record("cube"), root))
    gate("Exact current-source release wheel, installed assets and dependencies", lambda: verify_release(run_record("release"), root))
    gate("Current-source browser interaction evidence", browser)
    gate("Sustained training, untouched final test and large-tensor execution", research)
    gate("Actual RGBD token bytes, independent oracle and cold-process decoding", rgbd)
    report = {"schema_version": 2, "verified_at": datetime.now(timezone.utc).isoformat(),
              "config": config, "gates": gates, "complete": all(row["verified"] for row in gates)}
    report["provenance"] = capture_provenance(config, root, [path for path in artifacts if path.is_file()])
    write_json(output_path or root / "results/acceptance.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="Recheck the earlier floating-point comparison profile")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--cube-results", type=Path)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--test-evidence", type=Path)
    parser.add_argument("--release-evidence", type=Path)
    parser.add_argument("--browser-evidence", type=Path)
    parser.add_argument("--research-results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.legacy:
        from scripts.verify_native_hue_release import verify as verify_native
        return verify_native(output_path=args.output)
    return verify(results_path=args.results, cube_results_path=args.cube_results,
                  tests_path=args.tests, test_evidence_path=args.test_evidence,
                  release_evidence_path=args.release_evidence, browser_evidence_path=args.browser_evidence,
                  research_results_path=args.research_results, output_path=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
