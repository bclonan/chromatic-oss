"""Add optional measured probes and verify the completed training record."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from experiments.common import write_json
from experiments.provenance import verify_provenance, unique_run_id
from experiments.ablations import run_robustness
from experiments.instrumentation import enrich_result
from experiments.reporting import paired_comparisons, conclusions, write_results_document


def main():
    torch.set_num_threads(4)
    path = ROOT / "results" / "latest.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    verify_provenance(result.get("provenance"), result.get("config"), ROOT)
    result["parent_run_id"] = result["run_id"]
    result["run_id"] = unique_run_id("finalized")
    artifact_dir = ROOT / "artifacts" / "checkpoints" / result["run_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not any(record["task"] == "decoder" for record in result["experiments"]):
        from experiments.decoder import run_decoder_experiment
        record, samples, palette = run_decoder_experiment(result["config"], artifact_dir)
        result["experiments"].append(record)
        result["tensor_samples"].extend(samples)
        result["codebooks"].append(palette)
    # Recover recorded vector-attention fields from the separate verified pilot
    # when the original runner called the records-only API.
    if not any("latent" in sample["name"] and "attention" in sample["name"] for sample in result["tensor_samples"]):
        from experiments.attention import run_attention_ablation
        records, samples = run_attention_ablation(result["config"], artifact_dir)
        result["experiments"] = [record for record in result["experiments"] if record["task"] != "attention"] + records
        result["tensor_samples"].extend(samples)
    result["robustness"] = run_robustness(result["config"])
    from experiments.precision import file_storage_comparison
    generator = torch.Generator().manual_seed(result["config"]["seeds"][0])
    result["file_storage"] = file_storage_comparison(torch.randn(16384, generator=generator) * 0.2, artifact_dir)
    enrich_result(result, ROOT)
    result["comparisons"] = paired_comparisons(result["experiments"])
    result["conclusions"] = conclusions(result)
    # Relative artifact links survive source exports and a different checkout.
    for record in result["experiments"]:
        if record.get("artifact_path"):
            artifact = Path(record["artifact_path"])
            if artifact.is_absolute() and artifact.is_relative_to(ROOT):
                record["artifact_path"] = artifact.relative_to(ROOT).as_posix()
    from chromatensor.visualization import tensor_triptych, learned_manifold, result_overview
    from chromatensor import ChromaTensor
    from chromatensor.codecs import codec_from_config
    figures = artifact_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for sample in result["tensor_samples"][:4]:
        values = torch.tensor(sample["values"])
        tensor = ChromaTensor.from_tensor(values, codec_from_config(sample["codec"]))
        filename = sample["name"].replace(" ", "_") + ".png"
        tensor_triptych(values, tensor, figures / filename, sample["name"])
    learned_manifold(result["codebooks"][-1], figures / "learned_rgb_manifold.png")
    result_overview(result["experiments"], figures / "task_results.png")
    result["figure_path"] = str(figures / "task_results.png")
    result["report_path"] = str(artifact_dir / "results.md")
    write_results_document(result, Path(result["report_path"]))
    write_json(ROOT / "results" / "runs" / (result["run_id"] + ".json"), result)
    write_json(path, result)
    write_results_document(result, ROOT / "RESULTS.md")
    print(f"Final record contains {len(result['experiments'])} model experiments and {len(result['precision'])} precision comparisons")


if __name__ == "__main__":
    main()
