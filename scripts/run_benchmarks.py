"""Train paired models and publish measured numerical and operator reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from experiments.common import environment, write_json
from experiments.provenance import unique_run_id, verify_provenance
from experiments.datasets import scalar_data, image_data, language_data
from experiments.training import REPRESENTATIONS, train_one, periodic_data
from experiments.precision import numerical_precision, operator_benchmarks
from experiments.reporting import paired_comparisons, conclusions, write_results_document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["pilot", "full", "smoke"], default="pilot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23])
    parser.add_argument("--tasks", nargs="+", choices=["scalar", "mnist", "cifar", "language", "periodic", "precision", "operators", "attention"], default=["scalar", "mnist", "cifar", "language", "periodic", "precision", "operators", "attention"])
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS + ["latent", "chroma_weights", "chroma_activations"], default=REPRESENTATIONS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--precision-samples", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "latest.json")
    parser.add_argument("--resume", action="store_true", help="Resume a saved run only when its complete configuration matches")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    config = {"profile": args.profile, "device": args.device, "seeds": args.seeds, "tasks": args.tasks, "representations": args.representations, "epochs": args.epochs if args.epochs is not None else (20 if args.profile == "full" else 6 if args.profile == "pilot" else 1), "scalar_epochs": 24 if args.profile != "smoke" else 2, "precision_samples": args.precision_samples if args.precision_samples is not None else (1_000_000 if args.profile != "smoke" else 4096), "train_images": 60000 if args.profile == "full" else 12000 if args.profile == "pilot" else 256, "validation_images": 10000 if args.profile == "full" else 2000 if args.profile == "pilot" else 128, "language_windows": 16384 if args.profile == "full" else 2048 if args.profile == "pilot" else 128, "language_validation_windows": 2048 if args.profile == "full" else 512 if args.profile == "pilot" else 64}
    if config["precision_samples"] < 2 or config["epochs"] < 1:
        parser.error("sample and epoch counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.tasks)) != len(args.tasks) or len(set(args.representations)) != len(args.representations):
        parser.error("seeds, tasks and representations must be unique")
    if "latent" in args.representations and any(task != "scalar" for task in args.tasks):
        parser.error("the latent representation supports the scalar task only")
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result["config"] != config:
            parser.error("resume configuration differs from saved run")
        verify_provenance(result.get("provenance"), config, ROOT)
    else:
        run_id = unique_run_id("bench")
        result = {"schema_version": 1, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "environment": environment(), "config": config, "experiments": [], "precision": [], "operators": [], "tensor_samples": [], "codebooks": [], "conclusions": [], "comparisons": []}
    artifact_dir = ROOT / "artifacts" / "checkpoints" / result["run_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    completed = {record["id"] for record in result["experiments"]}
    write_json(args.output, result)
    for seed in args.seeds:
        for task in (task for task in args.tasks if task in ("scalar", "mnist", "cifar", "language")):
            if task == "scalar":
                dataset = scalar_data(seed, args.device, count=512 if args.profile == "smoke" else 4096)
            elif task in ("mnist", "cifar"):
                dataset = image_data(task, ROOT / "data", config["train_images"], config["validation_images"], seed, args.device)
            else:
                dataset = language_data(seed, ROOT / "data", config["language_windows"], config["language_validation_windows"], args.device)
            representations = list(args.representations)
            if args.profile != "smoke":
                representations += ["latent"] if task == "scalar" else ["chroma_weights", "chroma_activations"] if task in ("mnist", "cifar") else []
            representations = list(dict.fromkeys(representations))
            for representation in representations:
                identifier = f"{task}-{representation}-s{seed}"
                if identifier in completed:
                    continue
                print(f"Training {identifier}", flush=True)
                record, samples, extra = train_one(task, representation, seed, dataset, config, artifact_dir)
                result["experiments"].append(record)
                result["tensor_samples"].extend(samples)
                if extra:
                    result["experiments"].append(extra)
                write_json(args.output, result)
                print(f"Finished {identifier}: validation_loss={record['metrics']['validation_loss']:.6g}", flush=True)
            del dataset
        if "periodic" in args.tasks:
            for representation in ("scalar8", "hue8", "numeric_circular8", "raw_sincos64"):
                identifier = f"periodic-{representation}-s{seed}"
                if identifier in completed:
                    continue
                print(f"Training {identifier}", flush=True)
                dataset = periodic_data(seed, representation, args.device, count=512 if args.profile == "smoke" else 4096)
                record, samples, _ = train_one("periodic", representation, seed, dataset, config, artifact_dir)
                record["metrics"]["input_bits_per_phase"] = dataset[-1]["representation_bits"]
                result["experiments"].append(record)
                result["tensor_samples"].extend(samples)
                write_json(args.output, result)
    if "precision" in args.tasks and not result["precision"]:
        storage_root = artifact_dir
        result["precision"], result["codebooks"], samples, result["file_storage"] = numerical_precision(config, storage_root)
        result["tensor_samples"].extend(samples)
        write_json(args.output, result)
    if "operators" in args.tasks and not result["operators"]:
        result["operators"], result["optimization"] = operator_benchmarks(config)
        write_json(args.output, result)
    if "attention" in args.tasks:
        from experiments.attention import run_attention_ablation
        if not any(record["task"] == "attention" for record in result["experiments"]):
            records, samples = run_attention_ablation(config, artifact_dir)
            result["experiments"].extend(records)
            result["tensor_samples"].extend(samples)
    if "precision" in args.tasks:
        from experiments.decoder import run_decoder_experiment
        if not any(record["task"] == "decoder" for record in result["experiments"]):
            record, samples, palette = run_decoder_experiment(config, artifact_dir)
            result["experiments"].append(record)
            result["tensor_samples"].extend(samples)
            result["codebooks"].append(palette)
        from experiments.ablations import run_robustness
        result["robustness"] = run_robustness(config)
    from experiments.instrumentation import enrich_result
    enrich_result(result, ROOT)
    result["comparisons"] = paired_comparisons(result["experiments"])
    result["conclusions"] = conclusions(result)
    from chromatensor.visualization import training_curves, learned_manifold, result_overview
    plot_dir = artifact_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for task in ("scalar", "mnist", "cifar", "language", "periodic"):
        training_curves(result["experiments"], plot_dir / (task + "_curves.png"), task)
    if len(result["codebooks"]) >= 3:
        learned_manifold(result["codebooks"][2], plot_dir / "learned_palette.png")
    result_overview(result["experiments"], plot_dir / "task_results.png")
    result["figure_path"] = str(plot_dir / "task_results.png")
    result["report_path"] = str(artifact_dir / "results.md")
    result["status"] = "complete"
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_results_document(result, Path(result["report_path"]))
    write_json(ROOT / "results" / "runs" / (result["run_id"] + ".json"), result)
    write_json(args.output, result)
    write_results_document(result, ROOT / "RESULTS.md" if args.output.resolve() == (ROOT / "results" / "latest.json").resolve() else args.output.with_suffix(".md"))
    print(f"Saved {len(result['experiments'])} measured model records to {args.output}", flush=True)
    return result


if __name__ == "__main__":
    main()
