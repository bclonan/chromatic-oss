"""Executable acceptance checks for source-bound research and release evidence."""
from __future__ import annotations

import json
import hashlib
import math
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath, PureWindowsPath
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import torch

from chromatensor import load_chroma_checkpoint
from .common import dataset_hash, error_metrics
from .provenance import artifact_manifest, file_hash, result_artifact_paths, source_manifest, verify_provenance

TASKS = {"scalar", "mnist", "cifar", "language", "periodic", "attention", "decoder"}
REPRESENTATIONS = {"fp32", "fp16", "bf16", "int8", "int4", "fixed24", "chroma8", "rgb24"}
PRECISION_CODECS = {"adaptive_residual", "bfloat16", "fixed24", "float16", "float32", "hsv24_effective8",
                    "hue8", "int4", "int8", "learned_rgbd", "learned_vocab1024", "learned_vocab16",
                    "learned_vocab256", "learned_vocab4096", "learned_vocab64", "learned_vocab65536",
                    "residual16_depth3", "rgb24", "rgbd32", "uniform_vocab1024", "uniform_vocab4096",
                    "uniform_vocab64", "uniform_vocab65536"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def read_record(path):
    def invalid(value):
        raise ValueError("Nonfinite evidence JSON")
    data = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)
    require(isinstance(data, dict), "Evidence must be a JSON object")
    return data


def bound_path(location, root):
    require(isinstance(location, str) and bool(location), "Artifact path is absent")
    root = Path(root).resolve()
    path = Path(location)
    path = (path if path.is_absolute() else root / path).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Artifact is missing or outside the workspace")
    return path


def verify_test_report(xml_path, root, evidence=None):
    tree = ET.parse(xml_path).getroot()
    cases = list(tree.iter("testcase"))
    require(bool(cases), "No executed tests in the supplied report")
    require(not any(list(case.iter(tag)) for case in cases for tag in ("failure", "error", "skipped")),
            "Required test report contains failed, errored or skipped tests")
    suites = list(tree.iter("testsuite"))
    require(bool(suites) and sum(int(s.attrib.get("tests", 0)) for s in suites) == len(cases),
            "Test count differs from executed test cases")
    require(all(int(s.attrib.get(key, 0)) == 0 for s in suites for key in ("failures", "errors", "skipped")),
            "Test suite reports failures, errors or skipped tests")
    modules = {case.attrib.get("classname", "").rsplit(".", 1)[-1] for case in cases}
    required = {path.stem for path in (Path(root) / "tests").glob("test_*.py")}
    require(bool(required) and required.issubset(modules), "Required test modules are missing from the report")
    if evidence is not None:
        require(evidence.get("returncode") == 0, "Test process did not succeed")
        verify_provenance(evidence.get("provenance"), evidence.get("config"), root,
                          required_artifacts=[xml_path])
    return {"tests": len(cases), "required_modules": sorted(required), "skipped": 0}


def expected_training_tuples(config):
    seeds = config.get("seeds")
    require(isinstance(seeds, list) and set(seeds) == {7, 23} and len(seeds) == 2,
            "Canonical acceptance requires seeds 7 and 23")
    require(config.get("profile") in {"pilot", "full"}, "Smoke runs do not establish canonical acceptance")
    require(set(config.get("tasks", [])) == (TASKS - {"decoder"}) | {"precision", "operators"},
            "Canonical task coverage is incomplete")
    require(set(config.get("representations", [])) == REPRESENTATIONS, "Canonical representation coverage is incomplete")
    minimum_epochs = 20 if config["profile"] == "full" else 6
    require(type(config.get("epochs")) is int and config["epochs"] >= minimum_epochs and
            type(config.get("scalar_epochs")) is int and config["scalar_epochs"] >= 24,
            "Canonical optimization budget is incomplete")
    require(config.get("train_images") == (60_000 if config["profile"] == "full" else 12_000) and
            config.get("validation_images") == (10_000 if config["profile"] == "full" else 2_000) and
            config.get("language_windows") == (16_384 if config["profile"] == "full" else 2_048) and
            config.get("language_validation_windows") == (2_048 if config["profile"] == "full" else 512),
            "Canonical image or language sample configuration differs")
    tuples = {(task, rep, seed) for task in ("scalar", "mnist", "cifar", "language")
              for rep in REPRESENTATIONS for seed in seeds}
    tuples |= {(task, "int8_integer_post_training", seed) for task in ("scalar", "mnist", "cifar") for seed in seeds}
    tuples |= {("scalar", "latent", seed) for seed in seeds}
    tuples |= {(task, rep, seed) for task in ("mnist", "cifar")
               for rep in ("chroma_weights", "chroma_activations") for seed in seeds}
    tuples |= {("periodic", rep, seed) for rep in ("scalar8", "hue8", "numeric_circular8", "raw_sincos64") for seed in seeds}
    tuples |= {("attention", rep, seed) for rep in ("fp32_attention_dot", "chroma8_attention_dot", "chroma8_attention_codebook") for seed in seeds}
    tuples.add(("decoder", "learned_rgb", seeds[0]))
    return tuples


def verify_training_coverage(result):
    require(result.get("status") == "complete", "Run has not completed")
    expected = expected_training_tuples(result.get("config", {}))
    rows = result.get("experiments", [])
    require(isinstance(rows, list) and bool(rows), "Training evidence is absent")
    keys = [(row["task"], row["representation"], row["seed"]) for row in rows]
    require(len(set(keys)) == len(keys) and set(keys) == expected, "Training task, representation or seed coverage differs")
    require(len({row.get("id") for row in rows}) == len(rows), "Training record identifiers are duplicated")
    for row in rows:
        metrics, data = row.get("metrics", {}), row.get("dataset", {})
        require(finite(metrics.get("validation_loss")), "Invalid validation loss for " + row["id"])
        require(isinstance(data.get("sha256"), str) and len(data["sha256"]) == 64, "Dataset identity is missing")
        if row["representation"] == "int8_integer_post_training":
            continue
        train_count = data.get("train_examples", data.get("train_samples", data.get("training_samples")))
        val_count = data.get("validation_examples", data.get("validation_samples"))
        require(type(train_count) is int and train_count > 1 and type(val_count) is int and val_count > 1,
                "Training and held-out sample counts are missing")
        config = result["config"]
        counts = {"scalar": (4096, 1024), "periodic": (4096, 1024),
                  "mnist": (min(config["train_images"], 60_000), config["validation_images"]),
                  "cifar": (min(config["train_images"], 50_000), config["validation_images"]),
                  "language": (config["language_windows"], config["language_validation_windows"]),
                  "attention": (config.get("attention_train_samples", 512), config.get("attention_validation_samples", 128)),
                  "decoder": (config.get("decoder_train_samples", 8192), config.get("decoder_validation_samples", 4096))}
        require((train_count, val_count) == counts[row["task"]], "Executed sample counts differ from canonical configuration")
        if row["task"] in {"mnist", "cifar"}:
            require(data.get("source") == "torchvision.datasets." + ("MNIST" if row["task"] == "mnist" else "CIFAR10"),
                    "Image evidence does not name the official dataset")
        if row["task"] == "language":
            require(data.get("source") == "Tiny Shakespeare, char-rnn repository" and
                    isinstance(data.get("corpus_sha256"), str) and len(data["corpus_sha256"]) == 64,
                    "Text corpus provenance is missing")
        curves = row.get("curves", [])
        require(isinstance(curves, list) and len(curves) >= 2 and
                any(point.get("epoch", 0) > 0 and finite(point.get("train_loss")) for point in curves),
                "Executed optimization history is absent")
        require(finite(metrics.get("training_seconds"), positive=True) and finite(metrics.get("initial_validation_loss")),
                "Training execution measurements are absent")
        require(math.isclose(curves[-1]["val_loss"], metrics["validation_loss"], rel_tol=2e-5, abs_tol=1e-10),
                "Final validation metric differs from the recorded curve")
        require(bool(row.get("artifact_path")), "Trained checkpoint is absent")
    return {"training_tuples": len(keys)}


def verify_comparisons(result):
    from .reporting import paired_comparisons
    actual = result.get("comparisons")
    require(isinstance(actual, list) and bool(actual), "Paired comparisons are absent")
    require(all(isinstance(pair.get("pairs"), list) and len(pair["pairs"]) == 2
                and pair.get("sample_count") == 2 for pair in actual), "A paired comparison is empty or incomplete")
    expected = paired_comparisons(result.get("experiments", []))
    require(actual == expected, "Paired comparison values differ from their model records")
    required = {(task, first, second) for task in ("scalar", "mnist", "cifar", "language")
                for first, second in (("chroma8", "int8"), ("rgb24", "fixed24"))}
    required.add(("periodic", "hue8", "numeric_circular8"))
    require(required.issubset({(p["task"], p["first"], p["second"]) for p in actual}), "Equal-bit control coverage is incomplete")
    require(all(p["identical_dataset"] is True for pair in actual for p in pair["pairs"]), "Paired datasets differ")
    return {"comparisons": len(actual)}


def verify_precision(result):
    count = result.get("config", {}).get("precision_samples")
    require(type(count) is int and count >= 1_000_000, "Canonical precision requires one million values per distribution")
    rows = result.get("precision", [])
    required = {(distribution, codec) for distribution in ("uniform", "normal", "laplace", "bimodal") for codec in PRECISION_CODECS}
    keys = [(row.get("distribution"), row.get("codec")) for row in rows]
    require(set(keys) == required and len(keys) == len(required), "Precision codec or distribution coverage differs")
    for row in rows:
        require(row.get("samples") == count, "Precision sample count differs from configuration")
        require(all(finite(row.get(name)) for name in ("mae", "mse", "max_error", "storage_bytes", "metadata_bytes")),
                "Invalid precision or storage measurement")
        require(row["max_error"] >= row["mae"], "Precision maximum error is below mean error")
    require({row.get("method") for row in result.get("operators", [])} ==
            {"decode_compute_encode", "lookup", "direct_integer_grid", "learned"}, "Operator control coverage is incomplete")
    return {"precision_conditions": len(keys), "samples_per_distribution": count}


def reload_model(record, root, config):
    from .checkpoint_probe import build_checkpoint_model
    path = bound_path(record.get("artifact_path"), root)
    return build_checkpoint_model(record, root), path


def verify_checkpoint_probe(record, model, checkpoint, root):
    from .checkpoint_probe import verify_checkpoint_probe as replay
    path = bound_path(record.get("checkpoint_probe", {}).get("path"), root)
    replay(record, root)
    return path


def verify_model_checkpoints(result, root):
    from .models import convert_integer_cpu
    validated, models = [], {}
    for record in result["experiments"]:
        if record["representation"] == "int8_integer_post_training":
            continue
        model, path = reload_model(record, root, result["config"])
        probe = verify_checkpoint_probe(record, model, path, root)
        require(record["metrics"].get("checkpoint_bytes") == path.stat().st_size, "Checkpoint byte count differs")
        validated.extend((path, probe))
        models[(record["task"], record["representation"], record["seed"])] = (model, record)
    for record in result["experiments"]:
        if record["representation"] != "int8_integer_post_training":
            continue
        source, source_record = models[(record["task"], "fp32", record["seed"])]
        from .instrumentation import _load_numpy_state
        inputs = _load_numpy_state(bound_path(source_record["checkpoint_probe"]["path"], root))["inputs"]
        integer = convert_integer_cpu(source)
        with torch.no_grad():
            output = integer(inputs)
        require(torch.isfinite(output).all() and any(name.endswith("weight_codes") for name in integer.state_dict()),
                "Post-training integer model failed execution")
    verify_provenance(result.get("provenance"), result["config"], root,
                      required_artifacts=result_artifact_paths(result, root) + validated)
    return {"reloaded_models": len(models)}


def verify_profiler(result, root):
    from .instrumentation import _profile_scalar
    instrumentation = result.get("instrumentation", {})
    profiles = instrumentation.get("profiler_runs", [])
    require(instrumentation.get("failures") == [] and len(profiles) == 2, "Profiler runs are incomplete or failed")
    records = {row["id"]: row for row in result["experiments"]}
    representations = set()
    for profile in profiles:
        record = records.get(profile.get("record_id"))
        require(record is not None and record["task"] == "scalar", "Profiler references an absent scalar record")
        representations.add(record["representation"])
        path = bound_path(record["artifact_path"], root)
        fresh = _profile_scalar(record, root, result["config"], load_chroma_checkpoint(path))
        for field in ("input_shape", "parameter_dtype", "observed_fp32_dense_kernel_calls", "annotated_float_matmul_flops"):
            require(profile.get(field) == fresh[field], "Profiler operation evidence differs from live checkpoint execution")
        require(profile["observed_fp32_dense_kernel_calls"] > 0 and profile["annotated_float_matmul_flops"] > 0,
                "Profiler lacks observed matrix products")
        require(profile.get("runtime_storage_on_this_batch") == fresh["runtime_storage_on_this_batch"],
                "Profiler storage evidence differs")
        fields = ("operation", "calls", "input_shapes", "profiler_annotated_flops")
        recorded_ops = [{key: row.get(key) for key in fields} for row in profile.get("selected_operations", [])]
        actual_ops = [{key: row.get(key) for key in fields} for row in fresh["selected_operations"]]
        require(recorded_ops == actual_ops, "Profiler recorded operations differ from live execution")
    require(representations == {"fp32", "chroma8"}, "Profiler control coverage differs")
    return {"live_profiles": 2}


def verify_cube_result(result, root):
    from chromatensor.nn import convert_to_chroma
    from .cube import cube_benchmarks, encoded_inputs, smooth_cubes
    from .models import StandardMLP
    config = result.get("config", {})
    require(result.get("status") == "complete" and config.get("seeds") == [7, 23] and config.get("epochs") == 24,
            "Canonical cube configuration is incomplete")
    rows = result.get("experiments", [])
    expected = {(rep, seed) for rep in ("direct_rgb24", "holographic24", "numeric_polar24") for seed in (7, 23)}
    keys = [(row.get("representation"), row.get("seed")) for row in rows]
    require(len(keys) == len(expected) and set(keys) == expected, "Cube model coverage differs")
    require(config.get("training_shape") == [3, 3, 3, 3], "Unsupported canonical cube shape")
    precision, geometry = cube_benchmarks(config["seeds"])
    require(result.get("geometry") == geometry, "Cube geometry evidence differs from execution")
    recorded = result.get("reconstruction", [])
    require(len(recorded) == len(precision), "Cube erasure coverage differs")
    for row, fresh in zip(recorded, precision):
        require({k: v for k, v in row.items() if k != "decode_ms"} ==
                {k: v for k, v in fresh.items() if k != "decode_ms"}, "Cube reconstruction evidence differs from execution")
    for row in rows:
        seed, rep = row["seed"], row["representation"]
        path = bound_path(row.get("artifact_path"), root)
        require(result.get("run_id") in path.parts, "Cube checkpoint is outside its immutable run directory")
        require(row.get("model_config") == {"inputs": 162, "hidden": 128, "outputs": 81, "bits": 8, "representation": "chroma8"},
                "Cube model configuration differs")
        model = convert_to_chroma(StandardMLP(162, 128, 81), bits=8, representation="chroma8")
        load_chroma_checkpoint(path, model)
        model.eval()
        train, validation = smooth_cubes(1024, seed=seed), smooth_cubes(256, seed=seed + 1)
        inputs, masks = encoded_inputs(validation, rep, seed=seed + 3)
        _, train_masks = encoded_inputs(train, rep, seed=seed + 2)
        require(row["dataset"].get("sha256") == dataset_hash(train, validation) and
                row["dataset"].get("mask_sha256") == dataset_hash(train_masks, masks), "Cube dataset or masks differ")
        with torch.no_grad():
            prediction = model(inputs)
        metrics = error_metrics(validation.flatten(1), prediction)
        require(row.get("validation_prediction_sha256") == dataset_hash(prediction), "Reloaded cube prediction hash differs")
        require(row["metrics"].get("validation_loss") == metrics["mse"] and
                row["metrics"].get("checkpoint_bytes") == path.stat().st_size, "Cube checkpoint measurements differ")
        require(len(row.get("curves", [])) == 25 and row["metrics"]["validation_loss"] < row["metrics"]["initial_validation_loss"],
                "Cube training history or improvement is absent")
    comparisons = result.get("comparisons", [])
    require(len(comparisons) == 2 and {p.get("seed") for p in comparisons} == {7, 23}, "Cube paired controls are incomplete")
    for seed in (7, 23):
        colored = next(row for row in rows if row["seed"] == seed and row["representation"] == "holographic24")
        numeric = next(row for row in rows if row["seed"] == seed and row["representation"] == "numeric_polar24")
        require(colored["curves"] == numeric["curves"] and
                colored["validation_prediction_sha256"] == numeric["validation_prediction_sha256"], "Cube polar controls differ")
        comparison = next(p for p in comparisons if p["seed"] == seed)
        require(all(comparison.get(name) is True for name in ("identical_dataset", "identical_mask", "identical_initial_parameters", "polar_controls_identical_curves"))
                and comparison.get("polar_controls_validation_delta") == 0, "Cube comparison statements differ")
    verify_provenance(result.get("provenance"), config, root, required_artifacts=result_artifact_paths(result, root))
    return {"cube_models": len(rows), "geometry_cases": len(geometry), "erasure_cases": len(precision)}


def verify_release(evidence, root):
    verify_provenance(evidence.get("provenance"), evidence.get("config"), root)
    wheel = evidence.get("wheel", {})
    path = bound_path(wheel.get("path"), root)
    require(wheel.get("sha256") == file_hash(path) and wheel.get("bytes") == path.stat().st_size, "Release wheel checksum differs")
    checks = evidence.get("checks", {})
    require(all(checks.get(name) is True for name in ("isolated_imports", "cube_inverse", "holographic_decode", "bundled_assets", "pip_check")),
            "Release installation or execution checks are incomplete")
    source = source_manifest(root)["files"]
    expected = {name: digest for name, digest in source.items() if name.split("/")[0] in {"chromatensor", "experiments", "dashboard"}}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "Release wheel has duplicate files")
        actual = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                  if name.split("/")[0] in {"chromatensor", "experiments", "dashboard"} and Path(name).suffix in {".py", ".js", ".css", ".html"}}
        require(actual == expected, "Release wheel code or browser assets differ from current source")
        metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        require(len(metadata_paths) == 1, "Release wheel metadata is absent or duplicated")
        metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_paths[0]))
        require(not metadata.defects and metadata.get_all("Version", []) == ["0.2.0"],
                "Release wheel version differs or is duplicated")
        require(metadata.get_all("Name", []) == ["chromatensor"], "Release wheel package name differs or is duplicated")
    require(evidence.get("installed_assets") == expected, "Installed package asset hashes differ")
    dependencies = read_record(bound_path(evidence.get("dependency_evidence"), root))
    verify_dependency_matrix(dependencies, root, expected)
    verify_provenance(evidence["provenance"], evidence["config"], root, required_artifacts=[path, bound_path(evidence["dependency_evidence"], root)])
    return {"wheel_sha256": file_hash(path), "package_assets": len(expected)}


def verify_dependency_matrix(evidence, root, expected_assets):
    """Recheck saved execution reports rather than trusting matrix summary flags."""
    checks = evidence.get("checks", {})
    require(all(checks.get(name) is True for name in ("pip_check", "minimum_supported",
                "all_supported_python_minors", "clean_installed_wheel_inference")),
            "Supported dependency checks are incomplete")
    rows = evidence.get("environments", [])
    required = {"minimum": "3.11.", "python311": "3.11.", "python312": "3.12.",
                "python313": "3.13.", "python314": "3.14.", "linux311": "3.11.",
                "linux312": "3.12.", "linux313": "3.13.", "linux314": "3.14."}
    require(isinstance(rows, list) and len(rows) == len(required) and
            {row.get("environment") for row in rows} == set(required), "Dependency environment coverage differs")
    artifacts = []
    for row in rows:
        label = row["environment"]
        require(isinstance(row.get("python"), str) and row["python"].startswith(required[label]),
                "Dependency Python minor differs")
        xml = bound_path("artifacts/verification/" + label + "-tests.xml", root)
        tree = ET.parse(xml).getroot()
        cases, suites = list(tree.iter("testcase")), list(tree.iter("testsuite"))
        require(bool(cases) and bool(suites) and sum(int(s.attrib.get("tests", 0)) for s in suites) == len(cases),
                "Dependency test count differs from executed test cases")
        require(not any(list(case.iter(tag)) for case in cases for tag in ("failure", "error")) and
                all(int(s.attrib.get(key, 0)) == 0 for s in suites for key in ("failures", "errors")),
                "Dependency matrix tests failed")
        skipped = [case for case in cases if list(case.iter("skipped"))]
        require(sum(int(s.attrib.get("skipped", 0)) for s in suites) == len(skipped),
                "Dependency skip count differs")
        require(all("CUDA device unavailable" in next(case.iter("skipped")).attrib.get("message", "") for case in skipped),
                "Dependency matrix skipped a required CPU test")
        modules = {case.attrib.get("classname", "").rsplit(".", 1)[-1] for case in cases}
        required_modules = {path.stem for path in (Path(root) / "tests").glob("test_*.py")}
        require(bool(required_modules) and required_modules.issubset(modules), "Dependency matrix omitted a test module")
        require(row.get("tests") == {"passed": len(cases) - len(skipped), "cuda_only_skipped": len(skipped), "failures": 0},
                "Dependency test summary differs from executed report")
        installed_path = bound_path(row.get("installed_report"), root)
        installed = read_record(installed_path)
        require(installed.get("python") == row["python"] and installed.get("dependencies") == row.get("dependencies"),
                "Installed dependency versions differ from matrix row")
        require(installed.get("isolated") is True and installed.get("user_site_enabled") is False,
                "Installed import execution was not isolated")
        require(installed.get("installed_assets") == expected_assets and
                all(installed.get("checks", {}).get(name) is True for name in
                    ("isolated_imports", "cube_inverse", "holographic_decode", "bundled_assets", "rgbd_tokens")),
                "Installed package execution or asset hashes differ")
        expected_colors = [[0, 0, 0, 127], [128, 0, 0, 127], [64, 0, 0, 127], [0, 0, 0, 128]]
        require(installed.get("rgbd_encoding") == {
                    "tokens": expected_colors, "decoded": [1., -1., 1.5, 2.],
                    "dtype": "torch.uint8", "payload_bytes": 16, "mutated_D_decode": 2.,
                    "persisted_tokens": expected_colors, "neural_tokens": expected_colors,
                    "neural_range_metadata_bytes": 0},
                "Installed RGBD token bytes or reconstruction differ")
        prefix = installed.get("prefix", "")
        modules = installed.get("modules", {})
        require(prefix == "<environment>" and set(modules) == {"chromatensor", "experiments", "dashboard"} and
                all(isinstance(location, str) and location.startswith(prefix + "/") and
                    "\\" not in location and ".." not in PurePosixPath(location).parts
                    for location in modules.values()),
                "Installed module paths reused an external checkout")
        inference = installed.get("deployed_inference", [])
        replay = [item for item in inference if item.get("prediction_close") is True]
        integer = [item for item in inference if item.get("finite") is True]
        require(len(inference) == 3 and len(replay) == 2 and len(integer) == 1,
                "Installed inference control coverage is incomplete")
        require(all(item.get("device") == "cpu" and type(item.get("examples")) is int and item["examples"] > 0 for item in inference),
                "Installed inference measurements are absent")
        require(all(item.get("repeats") == 10 and finite(item.get("batch_ms"), positive=True) and
                    finite(item.get("maximum_delta")) and item.get("relative_tolerance") == 1e-4 and
                    item.get("absolute_tolerance") == 1e-5 and isinstance(item.get("checkpoint_sha256"), str) and
                    len(item["checkpoint_sha256"]) == 64 for item in replay),
                "Installed inference replay measurements differ")
        require(len({item.get("record_id") for item in replay}) == 2 and
                integer[0].get("source_record_id") in {item.get("record_id") for item in replay},
                "Installed inference references differ")
        if label == "minimum":
            require(row["dependencies"].get("torch") == "2.14.1+cpu" and
                    row["dependencies"].get("torchvision") == "0.29.1+cpu", "Minimum Torch versions were not tested")
        lock = bound_path(row.get("lock"), root)
        locked = {}
        for line in lock.read_text(encoding="utf-8").splitlines():
            if "==" in line:
                name, version = line.split("==", 1)
                locked[name.lower().replace("_", "-")] = version
        versions = row.get("dependencies", {})
        require(versions.get("chromatensor") == "0.2.0" and all(
                    locked.get(name.lower().replace("_", "-")) == version
                    for name, version in versions.items() if name != "chromatensor"),
                "Installed dependency versions differ from their lock")
        require(isinstance(row.get("dependency_check"), str) and bool(row["dependency_check"].strip()),
                "Dependency checker output is missing")
        artifacts.extend((xml, installed_path, lock))
    verify_provenance(evidence.get("provenance"), evidence.get("config"), root, required_artifacts=artifacts)
    return {"environments": len(rows)}
