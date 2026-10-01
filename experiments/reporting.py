from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics


def paired_comparisons(records):
    index = {(r["task"], r["representation"], r["seed"]): r for r in records}
    comparisons = []
    for task in sorted(set(r["task"] for r in records)):
        for first, second in [("chroma8", "int8"), ("rgb24", "fixed24"), ("hue8", "numeric_circular8"), ("chroma8", "fp32")]:
            pairs = []
            for seed in sorted(set(r["seed"] for r in records)):
                a, b = index.get((task, first, seed)), index.get((task, second, seed))
                if a and b:
                    key = "accuracy" if task in ("mnist", "cifar") else "perplexity" if task == "language" else "angular_mae_radians" if task == "periodic" else "validation_loss"
                    pairs.append({"seed": seed, "metric": key, "first": a["metrics"][key], "second": b["metrics"][key], "delta": a["metrics"][key] - b["metrics"][key], "identical_dataset": a["dataset"]["sha256"] == b["dataset"]["sha256"]})
            if pairs:
                deltas = [p["delta"] for p in pairs]
                comparisons.append({"task": task, "first": first, "second": second, "pairs": pairs, "mean_delta": statistics.mean(deltas), "sample_standard_deviation": statistics.stdev(deltas) if len(deltas) > 1 else None, "sample_count": len(pairs), "inference_scope": "paired pilot observations, insufficient for population significance claim"})
    return comparisons


def conclusions(result):
    findings = []
    for pair in result.get("comparisons", []):
        if pair["second"] in ("int8", "fixed24", "numeric_circular8"):
            findings.append(f"{pair['task']}: {pair['first']} minus {pair['second']} mean paired {pair['pairs'][0]['metric']} difference {pair['mean_delta']:.8g} across {pair['sample_count']} seeds. These controls share quantization levels, so color labels alone do not establish an advantage.")
    if result.get("precision"):
        findings.append("RGB24 is a fixed-point quantizer with color byte labels. RGBD32 preserves finite FP32 values by exposing sign, fraction, and exponent bytes. Neither removes binary arithmetic.")
        learned = next(r for r in result["precision"] if r["distribution"] == "normal" and r["codec"] == "learned_vocab256")
        uniform = next(r for r in result["precision"] if r["distribution"] == "normal" and r["codec"] == "int8")
        findings.append(f"On clipped normal values, the separately fitted 256-centroid learned palette has MAE {learned['mae']:.6g}, versus {uniform['mae']:.6g} for uniform INT8. This is distribution-aware scalar vector quantization; learned tables have additional storage cost.")
    decoder = next((record for record in result["experiments"] if record["task"] == "decoder"), None)
    if decoder:
        controls = decoder["equal_data_controls"]
        findings.append(f"The jointly trained neural RGB codec reduced held-out MSE from {decoder['metrics']['initial_validation_loss']:.6g} to {decoder['metrics']['validation_loss']:.6g}. Its held-out MAE {decoder['metrics']['mae']:.6g} exceeds uniform8 MAE {controls['uniform8']['mae']:.6g} and uniform24 MAE {controls['uniform24']['mae']:.6g} on the identical values. Learned color coordinates worked, but these measurements do not justify their extra model/storage cost.")
    attention = [record for record in result["experiments"] if record["task"] == "attention"]
    if attention:
        modes = sorted(set(record["representation"] for record in attention))
        means = {mode: statistics.mean(record["metrics"]["validation_loss"] for record in attention if record["representation"] == mode) for mode in modes}
        findings.append("The causal phase attention ablation produced mean held-out MSE " + ", ".join(f"{mode}={value:.6g}" for mode, value in means.items()) + ". Codebook inference still uses floating point nearest-code assignment, similarity-table construction, softmax, and value aggregation.")
    corruption = next((row for row in result.get("robustness", []) if row["representation"] == "chroma8" and row["probe"] == "payload LSB flip"), None)
    if corruption:
        findings.append(f"Flipping a palette payload bit produced an invalid chroma8 symbol for {corruption['invalid_symbol_fraction']:.1%} of the tested values. The palette supplies no inherent bit-corruption robustness; checksums and vocabulary validation remain necessary.")
    findings.append("Quantization-aware networks use floating point master parameters, optimizer state, matrix products, normalization, and softmax. The operator lookup and grid-add paths perform integer operations after their tables or scale metadata have been prepared.")
    findings.append("The executed profile is a seeded research pilot. Limited training, small models, and two seeds do not establish state-of-the-art accuracy, novel mathematics, end-to-end integer training, or a general speed advantage.")
    return findings


def write_results_document(result, path):
    def number(value, specification):
        return "n/a" if value is None else format(value, specification)
    lines = ["# Measured results", "", f"Run `{result['run_id']}` used {result['environment']['gpu'] or 'CPU'}, PyTorch {result['environment']['torch']}, profile `{result['config']['profile']}`.", "", f"The record contains {len(result['experiments'])} model experiments and {len(result.get('precision', []))} numerical representation/distribution comparisons.", "", "Every value below comes from the saved experiment records. Accuracy is a fraction. Training time excludes validation and export; GPU timings synchronize the device. Storage is a diagnostic quantized export estimate unless a row explicitly names an actual checkpoint or integer inference model. Inference-only records have no training time.", "", "| Task | Representation | Seed | Validation loss | Task metric | Training seconds | Inference samples/s |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for record in result["experiments"]:
        metrics = record["metrics"]
        key = "accuracy" if record["task"] in ("mnist", "cifar") else "perplexity" if record["task"] == "language" else "angular_mae_radians" if record["task"] == "periodic" else "validation_loss"
        metric = metrics.get(key, metrics.get("validation_loss", 0))
        lines.append(f"| {record['task']} | {record['representation']} | {record['seed']} | {number(metrics.get('validation_loss'), '.6g')} | {number(metric, '.6g')} | {number(metrics.get('training_seconds'), '.3f')} | {number(metrics.get('inference_throughput'), '.1f')} |")
    lines += ["", "## Paired controls", "", "| Task | Comparison | Mean metric difference | Seeds |", "| --- | --- | --- | --- |"]
    for pair in result.get("comparisons", []):
        lines.append(f"| {pair['task']} | {pair['first']} minus {pair['second']} | {pair['mean_delta']:.9g} | {pair['sample_count']} |")
    lines += ["", f"## Precision on {result['config']['precision_samples']:,} bounded values per distribution", "", "The suite records unclipped out-of-domain sample counts, but evaluates quantization on the same clipped values for each bounded codec. Hue uses circular error, so its MAE does not compare directly with linear error.", "", "| Distribution | Codec | MAE | Max error | Actual code bits/value | Metadata bytes |", "| --- | --- | --- | --- | --- | --- |"]
    for row in result.get("precision", []):
        lines.append(f"| {row['distribution']} | {row['codec']} | {row['mae']:.7g} | {row['max_error']:.7g} | {row['bits_per_value']:.1f} | {row.get('metadata_bytes', 0)} |")
    lines += ["", "## Operator measurements", "", "| Method | MAE | Batch latency ms | Values/s |", "| --- | --- | --- | --- |"]
    for row in result.get("operators", []):
        lines.append(f"| {row['method']} | {row['mae']:.7g} | {row['latency_ms']:.6g} | {row['throughput']:.1f} |")
    lines += ["", "## Actual file storage", "", "| Format | File bytes | Separate metadata bytes | Bits/value including metadata | Reconstruction MAE |", "| --- | --- | --- | --- | --- |"]
    for row in result.get("file_storage", []):
        lines.append(f"| {row['format']} | {row['bytes']} | {row.get('metadata_bytes', 0)} | {row['bits_per_value_with_metadata']:.5g} | {row['reconstruction_mae']:.7g} |")
    lines += ["", "## Instrumentation", "", "The JSON includes per-model matrix operation estimates, live master bytes, runtime RGB buffers, diagnostic exports, and partial PyTorch CPU profiler observations from trained scalar checkpoints. The estimates exclude uncounted operations and are not hardware instruction counts. See docs/MEASUREMENT.md for the complete scope."]
    lines += ["", "## Interpretation", ""] + ["- " + finding for finding in result["conclusions"]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

