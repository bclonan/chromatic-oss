"""Run the multidimensional cube extension without replacing the original pilot."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.common import environment, write_json
from experiments.provenance import capture_provenance, unique_run_id, artifact_manifest, verify_provenance
from experiments.cube import cube_benchmarks, train_cube_reconstructor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cube.json")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32)")
    torch.set_num_threads(4)
    run_id = unique_run_id("cube")
    root = ROOT / "artifacts" / "checkpoints" / "cube" / run_id
    root.mkdir(parents=True, exist_ok=False)
    config = {"seeds": args.seeds, "epochs": args.epochs, "dimensions": [3, 4, 5, 6], "size": 3,
              "training_shape": [3, 3, 3, 3], "training_device": "cpu"}
    provenance = capture_provenance(config, ROOT)
    precision, geometry = cube_benchmarks(args.seeds)
    records = []
    for seed in args.seeds:
        for representation in ("direct_rgb24", "holographic24", "numeric_polar24"):
            print(f"Training cube {representation} seed {seed}", flush=True)
            record = train_cube_reconstructor(representation, seed, root, epochs=args.epochs)
            record["artifact_path"] = Path(record["artifact_path"]).relative_to(ROOT).as_posix()
            records.append(record)
    comparisons = []
    for seed in args.seeds:
        paired = [record for record in records if record["seed"] == seed]
        chromatic = next(record for record in paired if record["representation"] == "holographic24")
        numeric = next(record for record in paired if record["representation"] == "numeric_polar24")
        comparison = {
            "seed": seed,
            "identical_dataset": len({record["dataset"]["sha256"] for record in paired}) == 1,
            "identical_mask": len({record["dataset"]["mask_sha256"] for record in paired}) == 1,
            "identical_initial_parameters": len({record["dataset"]["initial_parameter_sha256"] for record in paired}) == 1,
            "polar_controls_identical_curves": chromatic["curves"] == numeric["curves"],
            "polar_controls_validation_delta": chromatic["metrics"]["validation_loss"] - numeric["metrics"]["validation_loss"],
        }
        if not all(comparison[key] for key in ("identical_dataset", "identical_mask", "identical_initial_parameters", "polar_controls_identical_curves")):
            raise AssertionError("Cube comparisons lost their paired conditions")
        comparisons.append(comparison)
    mean_mse = {representation: statistics.mean(record["metrics"]["validation_loss"] for record in records if record["representation"] == representation) for representation in ("direct_rgb24", "holographic24", "numeric_polar24")}
    findings = ["The cube supports integer-only layer rotations in three through six dimensions, with exact four-turn and inverse restoration of cell codes and identities.", "The distributed field stores a phase8/magnitude16 full complex FFT. It uses floating point transforms and carries Hermitian redundancy. It is an implemented spectral representation, not a literal optical hologram.", "Every numeric polar control uses exactly the same 24-bit quantization as its chromatic field. Their inputs, training curves, and reconstruction outputs are identical. Color naming supplies no reconstruction advantage.", f"At approximately 50% retained coordinates, mean held-out periodic-field MSE was {mean_mse['direct_rgb24']:.7g} for direct RGB and {mean_mse['holographic24']:.7g} for the distributed field. These masks erase coordinates in different bases.", "Partial field erasure loses information. The trained reconstruction models evaluate held-out low-rank periodic fields from the same generator and a separate random-noise negative control. Their results do not establish recovery of arbitrary data from fragments."]
    result = {"schema_version": 1, "run_id": run_id, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment(), "config": config, "provenance": provenance, "geometry": geometry, "reconstruction": precision, "experiments": records, "comparisons": comparisons, "conclusions": findings}
    lines = ["# Multidimensional cube measurements", "", "These measurements extend the original ChromaTensor pilot. They use distinct records and do not replace its findings.", "", f"The run measured {len(geometry)} geometry cases and {len(precision)} erasure cases. Reconstruction models used {args.epochs} epochs on 1,024 training fields and 256 held-out fields with shape 3 × 3 × 3 × 3. The same seeds, masks, data, initial weights, architecture, and optimizer apply across each paired comparison.", "", "| Representation | Seed | Initial held-out MSE | Final held-out MSE | Random-noise MSE | Training seconds |", "| --- | --- | --- | --- | --- | --- |"]
    for record in records:
        m = record["metrics"]
        lines.append(f"| {record['representation']} | {record['seed']} | {m['initial_validation_loss']:.7g} | {m['validation_loss']:.7g} | {m['out_of_distribution_noise_mse']:.7g} | {m['training_seconds']:.4g} |")
    lines += ["", "Training seconds measure optimization batches only. Encoding, validation, checkpoint serialization, and final evaluation are excluded. Process peak memory includes optimization and per-epoch validation. Packed field payload is 243 bytes per 81-cell sample. Spectral encodings add eight bytes of magnitude-scale metadata. The dense neural input is 648 bytes per sample after numerical expansion and the mask. The model uses FP32 master weights and matrix operations with chroma8 fake quantization.", "", "## Findings", ""] + ["- " + finding for finding in findings]
    lines += ["", "![Measured validation loss](cube-training.png)", "", f"All erasure measurements and execution times are in [the JSON record](../../../../results/runs/{run_id}.json). Geometry and CUDA correctness come from independent unit tests. Six-dimensional visual projection does not display six physical spatial axes."]
    report_path = root / "CUBE_RESULTS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for record in records:
        axes[0].plot([point["epoch"] for point in record["curves"]], [point["val_loss"] for point in record["curves"]], label=f"{record['representation']} seed {record['seed']}", linestyle="--" if record["representation"] == "numeric_polar24" else "-")
    axes[0].set(xlabel="Epoch", ylabel="Held-out MSE", title="Periodic-field reconstruction")
    axes[0].legend(fontsize=7)
    representations = list(mean_mse)
    axes[1].bar(representations, [mean_mse[key] for key in representations], color=["#23997d", "#5264bf", "#be7a35"])
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].set(ylabel="Mean held-out MSE", title="Matched two-seed controls")
    figure_path = root / "cube-training.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    artifact_paths = [ROOT / record["artifact_path"] for record in records] + [report_path, figure_path]
    provenance["artifacts"] = artifact_manifest(artifact_paths, ROOT)
    verify_provenance(provenance, config, ROOT, required_artifacts=artifact_paths)
    result["report_path"] = report_path.relative_to(ROOT).as_posix()
    result["figure_path"] = figure_path.relative_to(ROOT).as_posix()
    # Publish the immutable run first. The canonical pointer changes only after
    # every checkpoint, report, figure and source identity has passed checks.
    write_json(ROOT / "results" / "runs" / (run_id + ".json"), result)
    write_json(args.output, result)
    print(f"Saved {len(geometry)} exact geometry cases, {len(precision)} erasure cases, and {len(records)} trained cube models", flush=True)
    return result


if __name__ == "__main__":
    main()
