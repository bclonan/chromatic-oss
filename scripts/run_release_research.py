"""Run four-seed MNIST training with a final test withheld from this run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from chromatensor import RGBCodec, RGBDCodec, UniformCodec
from experiments.checkpoint_probe import build_checkpoint_model, load_numeric_npz, sha256_file, verify_checkpoint_probe
from experiments.common import autocast_context, dataset_hash, environment, error_metrics, PeakMemory, write_json
from experiments.datasets import stratified_indices
from experiments.provenance import artifact_manifest, capture_provenance, normalize_result_paths, unique_run_id, verify_provenance
from experiments.training import evaluate, train_one


REPRESENTATIONS = ("fp32", "int8", "chroma8")


def now():
    return datetime.now(timezone.utc).isoformat()


def stage_clock_metadata():
    clock = time.get_clock_info("perf_counter")
    if not clock.monotonic or clock.adjustable:
        raise RuntimeError("Release stage timing requires a monotonic, nonadjustable performance clock")
    return {"api": "time.perf_counter_ns", "implementation": clock.implementation,
            "monotonic": clock.monotonic, "adjustable": clock.adjustable,
            "resolution_seconds": clock.resolution, "unit": "nanoseconds", "scope": "same process"}


def location(path, root=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def release_config(device="cpu", smoke=False):
    return {"profile": "release_smoke" if smoke else "release_research", "device": device,
            "smoke": smoke, "seeds": [7] if smoke else [7, 23, 41, 59],
            "representations": list(REPRESENTATIONS), "epochs": 2 if smoke else 20,
            "scalar_epochs": 2 if smoke else 20, "train_images": 256 if smoke else 48000,
            "development_images": 128 if smoke else 12000, "final_test_images": 64 if smoke else 10000,
            "large_tensor_values": 4096 if smoke else 10000000,
            "selection_policy": "fixed final epoch; development losses recorded without early stopping or tuning"}


def load_mnist_split(data_root, training):
    from torchvision.datasets import MNIST
    data = MNIST(str(data_root), train=training, download=True)
    metadata = {"source": "torchvision.datasets.MNIST", "namespace": "official_train" if training else "official_test",
                "source_url": "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html",
                "source_examples": len(data), "source_sha256": dataset_hash(data.data, data.targets)}
    return data.data, data.targets, metadata


def partition_indices(labels, train_count, development_count, seed):
    if min(train_count, development_count) < 1 or train_count + development_count > len(labels):
        raise ValueError("Training and development counts must fit the official training split")
    eligible = stratified_indices(labels, train_count + development_count, seed)
    training = eligible[stratified_indices(labels[eligible], train_count, seed + 1000)]
    used = torch.zeros(len(labels), dtype=torch.bool)
    used[training] = True
    development = eligible[~used[eligible]]
    if len(training) != train_count or len(development) != development_count:
        raise ValueError("Partition counts are wrong")
    return training, development


def prepare_training_dataset(source, config, seed, artifacts):
    images, labels, metadata = source
    training, development = partition_indices(labels, config["train_images"], config["development_images"], seed)
    path = Path(artifacts) / f"mnist-s{seed}-partition.npz"
    np.savez_compressed(path, train_ids=training.numpy(), development_ids=development.numpy())
    tx, ty = images[training].flatten(1).float() / 255, labels[training].long()
    vx, vy = images[development].flatten(1).float() / 255, labels[development].long()
    info = {**metadata, "partition": "disjoint training and development subsets of the official training split",
            "train_examples": len(training), "validation_examples": len(development), "selection_seed": seed,
            "sha256": dataset_hash(tx, ty, vx, vy), "preprocessing": "pixels / 255, flattened; no augmentation",
            "index_artifact": str(path), "index_sha256": sha256_file(path)}
    return tuple(value.to(config["device"]) for value in (tx, ty, vx, vy)) + (info,)


def large_tensor_experiment(count, artifacts):
    if type(count) is not int or count < 2:
        raise ValueError("Large tensor sample count must be an integer of at least two")
    directory = Path(artifacts) / "large-tensor"
    directory.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(7331)
    rows = []
    with PeakMemory("cpu") as memory:
        values = torch.rand(count, generator=generator, dtype=torch.float32) * 2 - 1
        rgb_codec, numeric_codec = RGBCodec(), UniformCodec(bits=24)
        rgb = rgb_codec.encode(values)
        numeric = numeric_codec.encode(values)[..., 0].long()
        packed = torch.stack(((numeric >> 16) & 255, (numeric >> 8) & 255, numeric & 255), -1).to(torch.uint8)
        rgb_decoded, numeric_decoded = rgb_codec.decode(rgb), numeric_codec.decode(numeric.unsqueeze(-1))
        metadata = {"minimum": -1, "maximum": 1, "shape": [count],
                    "format": "uniform scalar index, three bytes big-endian", "input_dtype": "torch.float32"}
        metadata_text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        for name, codes, decoded in (("rgb24", rgb, rgb_decoded), ("fixed24_packed", packed, numeric_decoded)):
            path = directory / (name + ".bin")
            path.write_bytes(codes.numpy().tobytes())
            sidecar = path.with_suffix(".json")
            sidecar.write_text(metadata_text, encoding="utf-8")
            rows.append({"representation": name, "payload_path": str(path), "metadata_path": str(sidecar),
                         "payload_bytes": path.stat().st_size, "metadata_bytes": sidecar.stat().st_size,
                         "total_bytes": path.stat().st_size + sidecar.stat().st_size,
                         "physical_payload_bits_per_value": codes.numel() * codes.element_size() * 8 / count,
                         "sha256": sha256_file(path), **error_metrics(values, decoded)})
        rgbd = RGBDCodec()
        rgbd_codes = rgbd.encode(values)
        restored = rgbd.decode(rgbd_codes)
        exact = torch.equal(values.view(torch.int32), restored.view(torch.int32))
        rgbd_path = directory / "rgbd32.bin"
        rgbd_path.write_bytes(rgbd_codes.numpy().tobytes())
        rgbd_metadata = rgbd_path.with_suffix(".json")
        rgbd_metadata.write_text(json.dumps({"shape": [count], "format": "finite FP32 sign/fraction/exponent",
                                             "input_dtype": "torch.float32"}, sort_keys=True), encoding="utf-8")
        source_sha256 = dataset_hash(values)
    return {"samples": count, "seed": 7331, "source": "independent CPU FP32 uniform values in [-1,1]",
            "input_sha256": source_sha256, "device": "cpu", "source_dtype": "torch.float32",
            "rgb24_fixed24_codes_exact": torch.equal(rgb, packed),
            "rgb24_fixed24_decoded_exact": torch.equal(rgb_decoded, numeric_decoded),
            "equal_payload_and_metadata_bytes": rows[0]["total_bytes"] == rows[1]["total_bytes"],
            "rgbd_fp32_bits_exact": exact, "rgbd_payload_bytes": rgbd_codes.numel() * rgbd_codes.element_size(),
            "rgbd": {"payload_path": str(rgbd_path), "metadata_path": str(rgbd_metadata),
                     "payload_bytes": rgbd_path.stat().st_size, "metadata_bytes": rgbd_metadata.stat().st_size,
                     "sha256": sha256_file(rgbd_path)},
            "representations": rows, "memory": memory.report(),
            "scope": "CPU representation and actual file-size test; no integer-kernel, neural-memory or speed claim"}


def final_test_stage(records, source, config, artifacts, official_train_size):
    images, labels, metadata = source
    ids = stratified_indices(labels, config["final_test_images"], 9173)
    global_ids = ids + official_train_size
    x, y = images[ids].flatten(1).float() / 255, labels[ids].long()
    test_info = {**metadata, "examples": len(ids), "sha256": dataset_hash(x, y),
                 "global_id_offset": official_train_size, "evaluation_only": True,
                 "withheld_from_this_run_training": True,
                 "prior_repository_use": metadata.get("namespace") == "official_test"}
    for record in records:
        model = build_checkpoint_model(record).to(config["device"]).eval()
        dx, dy = x.to(config["device"]), y.to(config["device"])
        measured = evaluate(model, dx, dy, "mnist", record["representation"], config["device"])
        logits = []
        with torch.no_grad(), autocast_context(record["representation"], config["device"]):
            for start in range(0, len(dx), 256):
                logits.append(model(dx[start:start + 256]).float().cpu())
        predictions = torch.cat(logits)
        if float((predictions.argmax(-1) == y).float().mean()) != float(torch.tensor(measured["accuracy"]).float()):
            raise ValueError("Final-test predictions disagree with the evaluation accuracy")
        path = Path(artifacts) / (record["id"] + "-final-test.npz")
        np.savez_compressed(path, logits=predictions.numpy(), labels=y.numpy(), test_ids=global_ids.numpy())
        record["final_test_metrics"] = {"loss": measured["validation_loss"], "accuracy": measured["accuracy"],
                                        "examples": len(ids), "dataset_sha256": test_info["sha256"],
                                        "prediction_path": str(path), "prediction_sha256": sha256_file(path),
                                        "checkpoint_sha256": sha256_file(location(record["artifact_path"])),
                                        "evaluated_at": now(), "device": config["device"],
                                        "stored_logit_dtype": "torch.float32", "evaluation_only": True}
        del model, dx, dy
    return test_info


def validate_release_research(result, root=ROOT, *, require_release=True):
    if result.get("schema_version") != 2:
        raise ValueError("Release research requires schema version 2")
    config, records = result["config"], result["experiments"]
    if require_release and (config.get("smoke") or config.get("seeds") != [7, 23, 41, 59]
            or config.get("epochs", 0) < 20 or config.get("train_images") != 48000
            or config.get("development_images") != 12000 or config.get("final_test_images") != 10000
            or config.get("large_tensor_values", 0) < 10000000):
        raise ValueError("Reduced runs do not satisfy the sustained release-research gate")
    expected = {(seed, representation) for seed in config["seeds"] for representation in REPRESENTATIONS}
    actual = [(record["seed"], record["representation"]) for record in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Release research is missing paired models or contains duplicates")
    try:
        training_wall = datetime.fromisoformat(result.get("training_completed_at"))
        test_wall = datetime.fromisoformat(result.get("official_test_opened_at"))
    except (TypeError, ValueError) as error:
        raise ValueError("Release stage wall timestamps must be timezone-aware ISO dates") from error
    if training_wall.tzinfo is None or test_wall.tzinfo is None:
        raise ValueError("Release stage wall timestamps must be timezone-aware ISO dates")
    if training_wall > test_wall:
        raise ValueError("Official test data was opened before training completed")
    clock = result.get("stage_clock", {})
    if (not isinstance(clock, dict) or clock.get("api") != "time.perf_counter_ns"
            or clock.get("monotonic") is not True or clock.get("adjustable") is not False
            or clock.get("unit") != "nanoseconds" or clock.get("scope") != "same process"
            or not isinstance(clock.get("implementation"), str) or not clock["implementation"]
            or type(clock.get("resolution_seconds")) not in (int, float)
            or not math.isfinite(clock["resolution_seconds"]) or clock["resolution_seconds"] <= 0):
        raise ValueError("Release stage monotonic clock metadata is missing or invalid")
    training_monotonic = result.get("training_completed_monotonic_ns")
    test_monotonic = result.get("official_test_opened_monotonic_ns")
    if any(type(value) is not int or value <= 0 for value in (training_monotonic, test_monotonic)):
        raise ValueError("Release stage monotonic timestamps must be positive integer nanoseconds")
    if training_monotonic >= test_monotonic:
        raise ValueError("Official test data was opened before training completed according to monotonic timestamps")
    official_training = official_test = None
    if require_release:
        from torchvision.datasets import MNIST
        official_training = MNIST(str(Path(root) / "data"), train=True, download=False)
        official_test = MNIST(str(Path(root) / "data"), train=False, download=False)
        if len(official_training) != 60000 or len(official_test) != 10000:
            raise ValueError("Official MNIST split sizes are wrong")
        training_source_hash = dataset_hash(official_training.data, official_training.targets)
        test_source_hash = dataset_hash(official_test.data, official_test.targets)
    checked_partitions = {}
    for record in records:
        info = record["dataset"]
        if require_release and (info.get("source") != "torchvision.datasets.MNIST"
                or info.get("namespace") != "official_train" or info.get("source_examples") != 60000):
            raise ValueError("Training provenance is not the full official MNIST training split")
        index_path = location(info["index_artifact"], root)
        arrays = load_numeric_npz(index_path, expected_keys={"train_ids", "development_ids"})
        train_ids, development_ids = arrays["train_ids"], arrays["development_ids"]
        if (len(train_ids) != config["train_images"] or len(development_ids) != config["development_images"]
                or train_ids.dtype != np.int64 or development_ids.dtype != np.int64
                or len(np.unique(np.concatenate((train_ids, development_ids)))) != len(train_ids) + len(development_ids)
                or min(train_ids.min(), development_ids.min()) < 0
                or max(train_ids.max(), development_ids.max()) >= info["source_examples"]
                or sha256_file(index_path) != info["index_sha256"]):
            raise ValueError("Training and development indices are not disjoint valid partitions")
        if info["train_examples"] != config["train_images"] or info["validation_examples"] != config["development_images"]:
            raise ValueError("Recorded training or development sample count is wrong")
        if require_release:
            if info.get("source_sha256") != training_source_hash:
                raise ValueError("Official training data source hash is wrong")
            if index_path not in checked_partitions:
                tx = official_training.data[train_ids].flatten(1).float() / 255
                vx = official_training.data[development_ids].flatten(1).float() / 255
                checked_partitions[index_path] = dataset_hash(tx, official_training.targets[train_ids],
                                                             vx, official_training.targets[development_ids])
                del tx, vx
            if checked_partitions[index_path] != info["sha256"]:
                raise ValueError("Training partition pixels or labels do not match its recorded hash")
        if record["metrics"]["epochs"] != config["epochs"] or len(record["curves"]) != config["epochs"] + 1:
            raise ValueError("Sustained training curve or epoch count is wrong")
        if (not all(math.isfinite(row["val_loss"]) and (row["train_loss"] is None or math.isfinite(row["train_loss"]))
                    for row in record["curves"])
                or [row["epoch"] for row in record["curves"]] != list(range(config["epochs"] + 1))
                or require_release and record["metrics"]["validation_loss"] >= record["metrics"]["initial_validation_loss"]):
            raise ValueError("Sustained model training did not produce finite learning evidence")
        verify_checkpoint_probe(record, root)
        final = record["final_test_metrics"]
        if (final["examples"] != config["final_test_images"]
                or final["checkpoint_sha256"] != sha256_file(location(record["artifact_path"], root))
                or not math.isfinite(final["loss"]) or not math.isfinite(final["accuracy"]) or not 0 <= final["accuracy"] <= 1):
            raise ValueError("Final-test count or frozen checkpoint identity is wrong")
        prediction_path = location(final["prediction_path"], root)
        arrays = load_numeric_npz(prediction_path, expected_keys={"logits", "labels", "test_ids"})
        if sha256_file(prediction_path) != final["prediction_sha256"]:
            raise ValueError("Final-test predictions changed")
        test_ids = arrays["test_ids"]
        if (len(test_ids) != config["final_test_images"] or len(np.unique(test_ids)) != len(test_ids)
                or test_ids.min() < info["source_examples"]
                or test_ids.max() >= info["source_examples"] + result["final_test"]["source_examples"]):
            raise ValueError("Final-test IDs overlap the training namespace")
        if (arrays["logits"].shape != (len(test_ids), 10) or arrays["labels"].shape != (len(test_ids),)
                or arrays["logits"].dtype != np.float32 or arrays["labels"].dtype != np.int64 or test_ids.dtype != np.int64
                or final["dataset_sha256"] != result["final_test"]["sha256"]):
            raise ValueError("Final-test arrays or dataset identity are wrong")
        if require_release:
            local_ids = test_ids - info["source_examples"]
            if (result["final_test"].get("source_sha256") != test_source_hash
                    or not np.array_equal(arrays["labels"], official_test.targets[local_ids].numpy())):
                raise ValueError("Final-test labels do not match the official test split")
            test_x = official_test.data[local_ids].flatten(1).float() / 255
            if dataset_hash(test_x, official_test.targets[local_ids]) != final["dataset_sha256"]:
                raise ValueError("Official final-test input data hash is wrong")
            del test_x
        accuracy = float((arrays["logits"].argmax(-1) == arrays["labels"]).mean())
        if accuracy != final["accuracy"]:
            raise ValueError("Saved final-test predictions do not match reported accuracy")
        total_loss = 0.
        logits, labels = torch.from_numpy(arrays["logits"]), torch.from_numpy(arrays["labels"])
        for start in range(0, len(test_ids), 256):
            batch = logits[start:start + 256]
            total_loss += float(torch.nn.functional.cross_entropy(batch, labels[start:start + 256])) * len(batch)
        if not math.isclose(total_loss / len(test_ids), final["loss"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Saved final-test predictions do not match reported loss")
    test = result["final_test"]
    if require_release and (test.get("namespace") != "official_test" or test.get("source_examples") != 10000):
        raise ValueError("Final evaluation did not use the full official test split")
    for seed in config["seeds"]:
        numeric = next(row for row in records if row["seed"] == seed and row["representation"] == "int8")
        chromatic = next(row for row in records if row["seed"] == seed and row["representation"] == "chroma8")
        if (numeric["dataset"]["sha256"] != chromatic["dataset"]["sha256"] or numeric["curves"] != chromatic["curves"]
                or numeric["checkpoint_probe"]["prediction_sha256"] != chromatic["checkpoint_probe"]["prediction_sha256"]
                or numeric["final_test_metrics"]["accuracy"] != chromatic["final_test_metrics"]["accuracy"]):
            raise ValueError("Equal-level INT8 and chroma8 controls diverged")
    large = result["large_tensor"]
    if large["samples"] != config["large_tensor_values"] or not all(large[key] for key in
            ("rgb24_fixed24_codes_exact", "rgb24_fixed24_decoded_exact", "equal_payload_and_metadata_bytes", "rgbd_fp32_bits_exact")):
        raise ValueError("Large-tensor equal-budget or exact-roundtrip control failed")
    for row in large["representations"]:
        if (location(row["payload_path"], root).stat().st_size != row["payload_bytes"]
                or location(row["metadata_path"], root).stat().st_size != row["metadata_bytes"]
                or row["payload_bytes"] != large["samples"] * 3
                or row["total_bytes"] != row["payload_bytes"] + row["metadata_bytes"]
                or sha256_file(location(row["payload_path"], root)) != row["sha256"]):
            raise ValueError("Large-tensor stored bytes changed")
    first, second = large["representations"]
    if (first["sha256"] != second["sha256"] or first["total_bytes"] != second["total_bytes"]
            or location(first["metadata_path"], root).read_bytes() != location(second["metadata_path"], root).read_bytes()):
        raise ValueError("Large-tensor payload and metadata budgets do not match")
    metadata = json.loads(location(first["metadata_path"], root).read_text(encoding="utf-8"))
    if metadata != {"minimum": -1, "maximum": 1, "shape": [large["samples"]],
                    "format": "uniform scalar index, three bytes big-endian", "input_dtype": "torch.float32"}:
        raise ValueError("Large-tensor scalar domain metadata is wrong")
    rgbd = large["rgbd"]
    rgbd_path = location(rgbd["payload_path"], root)
    if (rgbd_path.stat().st_size != large["samples"] * 4 or rgbd_path.stat().st_size != rgbd["payload_bytes"]
            or sha256_file(rgbd_path) != rgbd["sha256"]
            or location(rgbd["metadata_path"], root).stat().st_size != rgbd["metadata_bytes"]
            or json.loads(location(rgbd["metadata_path"], root).read_text(encoding="utf-8")) !=
                {"shape": [large["samples"]], "format": "finite FP32 sign/fraction/exponent", "input_dtype": "torch.float32"}):
        raise ValueError("Large RGBD payload bytes or checksum are wrong")
    generator = torch.Generator().manual_seed(large["seed"])
    values = torch.rand(large["samples"], generator=generator, dtype=torch.float32) * 2 - 1
    if dataset_hash(values) != large["input_sha256"]:
        raise ValueError("Large-tensor input seed or hash is wrong")
    encoded_rgb = np.memmap(location(first["payload_path"], root), dtype=np.uint8, mode="r", shape=(len(values), 3))
    encoded_rgbd = np.memmap(rgbd_path, dtype=np.uint8, mode="r", shape=(len(values), 4))
    for start in range(0, len(values), 262144):
        batch = values[start:start + 262144]
        rgb_codes = torch.from_numpy(np.array(encoded_rgb[start:start + len(batch)]))
        rgbd_codes = torch.from_numpy(np.array(encoded_rgbd[start:start + len(batch)]))
        if (not torch.equal(RGBCodec().encode(batch), rgb_codes)
                or not torch.equal(RGBDCodec().decode(rgbd_codes).view(torch.int32), batch.view(torch.int32))):
            raise ValueError("Large-tensor numerical payload does not reconstruct its source")
    del encoded_rgb, encoded_rgbd
    return True


def run_release_research(config, output, artifacts, *, training_source=None, final_test_loader=None):
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 2, "run_id": unique_run_id("release-research"), "created_at": now(),
              "config": config, "environment": environment(), "status": "running", "experiments": [],
              "provenance": capture_provenance(config), "stage_clock": stage_clock_metadata(),
              "non_claims": ["This fixed MNIST MLP experiment does not establish general accuracy or speed gains.",
                             "Chroma8 QAT retains FP32 masters and kernels, plus RGB symbols and FP64 range metadata.",
                             "Earlier repository experiments used the official MNIST test split. This procedure withholds it until all new training checkpoints are fixed."]}
    source = training_source if training_source is not None else load_mnist_split(ROOT / "data", True)
    for seed in config["seeds"]:
        dataset = prepare_training_dataset(source, config, seed, artifacts)
        for representation in REPRESENTATIONS:
            print(f"Training release MNIST {representation} seed {seed} for {config['epochs']} epochs", flush=True)
            record, _, _ = train_one("mnist", representation, seed, dataset, config, artifacts)
            record["id"] = result["run_id"] + "-" + record["id"]
            result["experiments"].append(record)
            verify_provenance(result["provenance"], config)
            write_json(output, result)
        del dataset
    result["training_completed_at"] = now()
    # Python 3.11 on Windows uses a coarse tick clock for time.monotonic_ns.
    # The performance counter is monotonic and records the actual stage order.
    result["training_completed_monotonic_ns"] = time.perf_counter_ns()
    test_source = final_test_loader() if final_test_loader is not None else load_mnist_split(ROOT / "data", False)
    result["official_test_opened_at"] = now()
    result["official_test_opened_monotonic_ns"] = time.perf_counter_ns()
    result["final_test"] = final_test_stage(result["experiments"], test_source, config, artifacts, len(source[1]))
    result["large_tensor"] = large_tensor_experiment(config["large_tensor_values"], artifacts)
    validate_release_research(result, require_release=not config["smoke"])
    result["summary"] = {representation: {
        "final_test_accuracy_mean": statistics.mean(row["final_test_metrics"]["accuracy"] for row in result["experiments"] if row["representation"] == representation),
        "seeds": len(config["seeds"])} for representation in REPRESENTATIONS}
    result["status"], result["completed_at"] = "complete", now()
    verify_provenance(result["provenance"], config)
    result["provenance"]["artifacts"] = artifact_manifest([path for path in artifacts.rglob("*") if path.is_file()])
    normalize_result_paths(result)
    write_json(Path(output).parent / "runs" / (result["run_id"] + ".json"), result)
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Reduced checks; cannot satisfy the release-research gate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "release-research.json")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    config = release_config(args.device, args.smoke)
    artifacts = ROOT / "artifacts" / "release-research" / unique_run_id("artifacts")
    result = run_release_research(config, args.output, artifacts)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
